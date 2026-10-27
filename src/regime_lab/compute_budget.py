"""Bound an owned POSIX process group's *aggregate* CPU use.

The caller must own a child started in a new session (PGID == PID), retain its
normal stdout/heartbeat supervision, and call ``tick`` at the returned interval.
Each CONT also arms an independent, at-most-50-ms STOP timer, so a blocked caller
cannot accidentally leave the child running without a budget.  No signal here
terminates a process.  A failed controller leaves the group stopped and reports
an error; it never falls back to unrestricted execution.

Process CPU time includes every thread.  The default sampler discovers only the
owned process group and retains the largest CPU time observed for each PID after
it exits.  It never inventories unrelated processes.  A worker that is created
and reaped entirely between samples, or its work after its final sample, can be
unobserved.  The short run quantum bounds that observational window; this is not
a hard CPU guarantee for arbitrary short-lived subprocess workloads.  Scheduled
model workers/BLAS threads are limited separately and charged in the main PID.
"""

from __future__ import annotations

import math
import os
import resource
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Protocol


class ComputeBudgetError(RuntimeError):
    """CPU supervision failed or was cancelled; do not resume without review."""


class _Timer(Protocol):
    daemon: bool

    def start(self) -> None: ...

    def cancel(self) -> None: ...


class ActiveClock:
    """Monotonic clock excluding only bounded, confirmed budget STOP periods.

    A bounded pause expires even if the supervisor stops ticking.  Safety STOPs
    with no measured CPU debt do not pause this clock, preserving idle detection.
    """

    def __init__(self, monotonic: Callable[[], float] = time.monotonic) -> None:
        self._monotonic = monotonic
        self._excluded = 0.0
        self._pause_start: float | None = None
        self._pause_until: float | None = None
        self._lock = threading.RLock()

    def _current_pause(self, now: float) -> float:
        if self._pause_start is None:
            return 0.0
        assert self._pause_until is not None
        return max(0.0, min(now, self._pause_until) - self._pause_start)

    @property
    def paused_seconds(self) -> float:
        with self._lock:
            return self._excluded + self._current_pause(self._monotonic())

    def now(self) -> float:
        with self._lock:
            now = self._monotonic()
            return now - self._excluded - self._current_pause(now)

    def pause(self, *, until: float) -> None:
        """Exclude a confirmed STOP from now until a known budget deadline."""
        with self._lock:
            now = self._monotonic()
            if not math.isfinite(until) or until < now:
                raise ValueError("budget pause deadline must be finite and in the future")
            self._excluded += self._current_pause(now)
            self._pause_start = now
            self._pause_until = until

    def resume(self) -> None:
        with self._lock:
            self._excluded += self._current_pause(self._monotonic())
            self._pause_start = None
            self._pause_until = None


def parse_cpu_time(value: str) -> float:
    """Parse ps TIME on macOS/Linux: [[days-]hours:]minutes:seconds[.ff]."""
    try:
        days = 0
        remainder = value.strip()
        if "-" in remainder:
            day_text, remainder = remainder.split("-", 1)
            days = int(day_text)
        parts = remainder.split(":")
        if len(parts) == 2:
            hours = 0
            minutes = int(parts[0])
            seconds = float(parts[1])
        elif len(parts) == 3:
            hours, minutes = int(parts[0]), int(parts[1])
            seconds = float(parts[2])
            if minutes >= 60:
                raise ValueError("invalid minutes")
        else:
            raise ValueError("invalid TIME fields")
        result = days * 86400 + hours * 3600 + minutes * 60 + seconds
        if (
            days < 0
            or hours < 0
            or minutes < 0
            or not 0 <= seconds < 60
            or not math.isfinite(result)
        ):
            raise ValueError("invalid TIME value")
        return result
    except (ValueError, OverflowError) as exc:
        raise ComputeBudgetError(f"invalid ps CPU time: {value!r}") from exc


class ProcessCpuSampler:
    """Sample the owned group, returning aggregate CPU under its leader PID.

    Retain each PID's maximum observed CPU when it disappears.  In particular,
    Darwin does not reliably add reaped children's CPU to their parent's ps TIME;
    this accounting does not depend on that behavior or on ps -S.  A decreasing
    per-PID counter (including PID reuse) fails closed rather than losing usage.
    """

    def __init__(self, pgid: int, *, timeout: float = 1.0) -> None:
        self.pgid = pgid
        self.timeout = timeout
        self._observed_cpu: dict[int, float] = {}

    def __call__(self, pids: tuple[int, ...]) -> dict[int, float]:
        if pids != (self.pgid,):
            raise ComputeBudgetError("group sampler accepts only its owned leader PID")
        try:
            inventory = subprocess.run(
                ["pgrep", "-g", str(self.pgid)],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            if inventory.returncode != 0:
                raise ComputeBudgetError("owned-child process-group inventory failed")
            try:
                group_pids = tuple(sorted({int(value) for value in inventory.stdout.split()}))
            except ValueError as exc:
                raise ComputeBudgetError("owned-child group inventory is invalid") from exc
            if self.pgid not in group_pids or any(pid <= 1 for pid in group_pids):
                raise ComputeBudgetError("owned-child group leader is missing")
            completed = subprocess.run(
                [
                    "ps",
                    "-p",
                    ",".join(str(pid) for pid in group_pids),
                    "-o",
                    "pid=,pgid=,time=",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ComputeBudgetError("owned-child CPU sampler failed") from exc
        if completed.returncode != 0:
            raise ComputeBudgetError("owned-child CPU sampler returned an error")
        samples: dict[int, float] = {}
        try:
            for line in completed.stdout.splitlines():
                pid_text, pgid_text, cpu_text = line.split()
                pid, pgid = int(pid_text), int(pgid_text)
                if pid not in group_pids or pid in samples or pgid != self.pgid:
                    raise ValueError("unexpected PID or process group")
                samples[pid] = parse_cpu_time(cpu_text)
        except (ValueError, ComputeBudgetError) as exc:
            raise ComputeBudgetError("owned-child CPU sampler returned invalid data") from exc
        if set(samples) != set(group_pids):
            raise ComputeBudgetError("owned-child CPU sampler missed a tracked process")
        for pid, value in samples.items():
            if value < self._observed_cpu.get(pid, 0.0):
                raise ComputeBudgetError("owned-child per-PID CPU counter decreased")
        self._observed_cpu.update(samples)
        return {self.pgid: sum(self._observed_cpu.values())}


def _controller_cpu_seconds() -> float:
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


class CpuBudget:
    """Measured CPU token budget, expressed as percent of *one* logical core.

    ``30`` means at most about 0.30 CPU seconds per wall second over sustained
    work, including supervisor/ps overhead by default.  A single <=50-ms quantum
    is permitted, then any multi-core burst debt is repaid while stopped.  ps
    resolution and OS scheduling introduce a small finite measurement margin.

    The default sampler aggregates observed CPU for the owned group, retaining
    observed exited-worker CPU.  Short-lived helpers can escape observation;
    arbitrary subprocess workloads therefore have no exact aggregate guarantee.
    ``tracked_pids`` is for an injected sampler that provides per-PID accounting;
    it may be an iterable or callback returning worker PIDs.  Such an inventory
    may not silently drop workers and must provide their final cumulative CPU.

    ``deadline`` passed to tick uses the real monotonic clock.  Cancellation or
    deadline expiry stops the group and raises, leaving graceful exit policy to
    the caller.  ``close(child_exited=True)`` is for a caller-confirmed exit only.
    """

    def __init__(
        self,
        pid: int,
        *,
        percent: float = 30.0,
        active_clock: ActiveClock | None = None,
        sampler: Callable[[tuple[int, ...]], Mapping[int, float]] | None = None,
        signal_group: Callable[[int, int], None] = os.killpg,
        monotonic: Callable[[], float] = time.monotonic,
        tracked_pids: Iterable[int] | Callable[[], Iterable[int]] = (),
        initially_stopped: bool = False,
        max_active_quantum: float = 0.05,
        overhead_sampler: Callable[[], float] | None = _controller_cpu_seconds,
        timer_factory: Callable[[float, Callable[[], None]], _Timer] = threading.Timer,
    ) -> None:
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
            raise ValueError("CPU budget requires an owned child PID greater than one")
        if not math.isfinite(percent) or not 0 < percent <= 100:
            raise ValueError("CPU budget percent must be in (0, 100]")
        if not math.isfinite(max_active_quantum) or not 0 < max_active_quantum <= 0.05:
            raise ValueError("active quantum must be in (0, 0.05] seconds")
        self.pid = pid
        self.percent = float(percent)
        self.active_clock = active_clock or ActiveClock(monotonic)
        self.max_active_quantum = max_active_quantum
        self._rate = self.percent / 100
        self._capacity = self._rate * max_active_quantum
        self._credit = self._capacity
        self._child_credit = self._capacity
        self._sampler = sampler or ProcessCpuSampler(pid)
        self._signal_group = signal_group
        self._monotonic = monotonic
        self._tracked_pids = tracked_pids
        self._overhead_sampler = overhead_sampler
        self._timer_factory = timer_factory
        self._timer: _Timer | None = None
        self._timer_generation = 0
        self._lock = threading.RLock()
        self._stopped = initially_stopped
        self._closed = False
        self._error: ComputeBudgetError | None = None
        self._samples: dict[int, float] = {}
        self._last_overhead: float | None = None
        self._last_wall = monotonic()
        self._cpu_seconds = 0.0
        self._rest_until = self._last_wall
        with self._lock:
            try:
                if not initially_stopped:
                    self._stop()
                self._account(initial=True)
            except Exception as exc:
                self._fail("CPU budget initialization failed", exc)

    @property
    def stopped(self) -> bool:
        with self._lock:
            return self._stopped

    @property
    def cpu_seconds(self) -> float:
        with self._lock:
            return self._cpu_seconds

    @property
    def debt_seconds(self) -> float:
        """Measured aggregate CPU seconds owed, not wall seconds of rest."""
        with self._lock:
            return max(0.0, -self._credit)

    def _pids(self) -> tuple[int, ...]:
        workers = self._tracked_pids() if callable(self._tracked_pids) else self._tracked_pids
        pids = {self.pid, *workers}
        if any(not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1 for pid in pids):
            raise ComputeBudgetError("tracked worker inventory contains an invalid PID")
        if self._samples and not set(self._samples).issubset(pids):
            raise ComputeBudgetError("tracked worker disappeared without final CPU accounting")
        return tuple(sorted(pids))

    def _account(self, *, initial: bool = False, sample_child: bool = True) -> None:
        if sample_child:
            pids = self._pids()
            samples = dict(self._sampler(pids))
            if set(samples) != set(pids):
                raise ComputeBudgetError("CPU sampler did not account for every tracked PID")
            if any(not math.isfinite(value) or value < 0 for value in samples.values()):
                raise ComputeBudgetError("CPU sampler returned a non-finite or negative value")
        else:
            # Every group member is stopped, so its observed CPU cannot advance.
            # Re-running ps here can consume more than the available tokens and
            # starve the child forever.  Only the cheap controller clock changes.
            samples = self._samples
        overhead = self._overhead_sampler() if self._overhead_sampler else 0.0
        now = self._monotonic()
        if not math.isfinite(overhead) or overhead < 0 or now < self._last_wall:
            raise ComputeBudgetError("CPU accounting clock or overhead is invalid")
        if initial:
            delta = 0.0
            child_delta = 0.0
        else:
            if self._last_overhead is None or overhead < self._last_overhead:
                raise ComputeBudgetError("controller CPU time decreased")
            delta = overhead - self._last_overhead
            child_delta = 0.0
            for pid, sample in samples.items():
                # A newly observed worker is charged for all of its CPU time;
                # prior work cannot be hidden by joining the inventory late.
                previous = self._samples.get(pid, 0.0)
                if sample < previous:
                    raise ComputeBudgetError("tracked process CPU time decreased")
                child_delta += sample - previous
            delta += child_delta
        self._credit = min(self._capacity, self._credit + self._rate * (now - self._last_wall) - delta)
        self._child_credit = min(
            self._capacity,
            self._child_credit + self._rate * (now - self._last_wall) - child_delta,
        )
        self._cpu_seconds += delta
        self._samples = samples
        self._last_overhead = overhead
        self._last_wall = now

    def _cancel_timer(self) -> None:
        self._timer_generation += 1
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _stop(self) -> None:
        if not self._stopped:
            self._signal_group(self.pid, signal.SIGSTOP)
            self._stopped = True

    def _rest(self) -> None:
        if self._credit < 0:
            now = self._monotonic()
            # Refill one quantum's reserve as well as repaying the measured
            # debt.  Resuming at exactly zero credit lets even tiny controller
            # accounting costs create an endless stop/re-account cycle.
            self._rest_until = now + (self._capacity - self._credit) / self._rate
            if self._child_credit < 0:
                self.active_clock.pause(until=self._rest_until)
            else:
                # Supervisor overhead can require rest, but an idle child must
                # still age toward the real no-progress watchdog deadline.
                self.active_clock.resume()
        else:
            self._rest_until = self._monotonic()
            self.active_clock.resume()

    def _fail(self, message: str, cause: BaseException) -> None:
        self._cancel_timer()
        stop_error: BaseException | None = None
        try:
            # Always retry STOP after an error, even if our state said stopped.
            self._signal_group(self.pid, signal.SIGSTOP)
            self._stopped = True
        except Exception as exc:
            stop_error = exc
        self.active_clock.resume()
        detail = "; emergency SIGSTOP also failed" if stop_error is not None else "; child group stopped"
        self._error = ComputeBudgetError(message + detail)
        raise self._error from cause

    def _expire_quantum(self, generation: int) -> None:
        with self._lock:
            if self._closed or generation != self._timer_generation:
                return
            try:
                self._stop()
                self._timer = None
                self._account()
                self._rest()
            except Exception as exc:
                try:
                    self._fail("CPU quantum supervision failed", exc)
                except ComputeBudgetError:
                    # tick/close surfaces this error on the supervising thread.
                    pass

    def tick(self, *, deadline: float | None = None, cancelled: bool = False) -> float:
        """Account CPU, run at most one quantum, and return the next poll delay."""
        with self._lock:
            if self._error is not None:
                raise self._error
            if self._closed:
                raise ComputeBudgetError("CPU budget is closed; child must not be resumed")
            try:
                now = self._monotonic()
                if cancelled:
                    raise ComputeBudgetError("CPU budget cancelled")
                if deadline is not None:
                    if not math.isfinite(deadline):
                        raise ComputeBudgetError("CPU budget deadline is invalid")
                    if now >= deadline:
                        raise ComputeBudgetError("CPU budget deadline expired")
                if not self._stopped:
                    # Sampling invokes external commands.  Never hold the lock
                    # doing that while the child runs: the independent STOP
                    # timer must be able to end its quantum on time.
                    return self.max_active_quantum
                if self._stopped and now < self._rest_until:
                    return min(self.max_active_quantum, self._rest_until - now)
                self._account(sample_child=False)
                if self._credit < -1e-12:
                    self._cancel_timer()
                    self._stop()
                    self._rest()
                    return min(self.max_active_quantum, self._rest_until - self._monotonic())
                if self._stopped:
                    self.active_clock.resume()
                    self._cancel_timer()
                    generation = self._timer_generation
                    timer = self._timer_factory(
                        self.max_active_quantum,
                        lambda: self._expire_quantum(generation),
                    )
                    timer.daemon = True
                    self._timer = timer
                    # Arm before CONT.  The lock keeps an early timer callback
                    # from observing an intermediate state.
                    timer.start()
                    self._signal_group(self.pid, signal.SIGCONT)
                    self._stopped = False
                return self.max_active_quantum
            except Exception as exc:
                self._fail("CPU budget tick failed", exc)

    def close(self, *, child_exited: bool = False) -> None:
        """Disarm supervision, leaving a live group stopped (never terminated)."""
        with self._lock:
            self._cancel_timer()
            self._closed = True
            self.active_clock.resume()
            if not child_exited:
                try:
                    self._stop()
                except Exception as exc:
                    self._fail("CPU budget close failed", exc)
            if self._error is not None:
                raise self._error
