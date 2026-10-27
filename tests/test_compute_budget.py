from __future__ import annotations

import signal
import subprocess
from pathlib import Path

import pytest

from regime_lab import compute_budget
from regime_lab.compute_budget import ActiveClock, ComputeBudgetError, CpuBudget, ProcessCpuSampler, parse_cpu_time


class FakeTimer:
    def __init__(self, world, delay, callback):
        self.world = world
        self.delay = delay
        self.callback = callback
        self.daemon = False
        self.cancelled = False

    def start(self):
        self.world.timers.append((self.world.now + self.delay, self))

    def cancel(self):
        self.cancelled = True


class World:
    def __init__(self, *, cores=1.0):
        self.now = 0.0
        self.cpu = 0.0
        self.overhead = 0.0
        self.cores = cores
        self.running = True
        self.timers = []
        self.signals = []
        self.sample_error = None
        self.signal_error = None

    def monotonic(self):
        return self.now

    def sample(self, pids):
        if self.sample_error:
            raise self.sample_error
        return {pids[0]: self.cpu}

    def send(self, pid, sig):
        self.signals.append((pid, sig))
        if self.signal_error == sig:
            raise PermissionError("test signal denied")
        self.running = sig == signal.SIGCONT

    def timer(self, delay, callback):
        return FakeTimer(self, delay, callback)

    def advance(self, seconds):
        end = self.now + seconds
        while True:
            pending = [(at, timer) for at, timer in self.timers if not timer.cancelled and at <= end]
            if not pending:
                break
            at, timer = min(pending, key=lambda item: item[0])
            self.timers.remove((at, timer))
            self.cpu += (at - self.now) * self.cores if self.running else 0
            self.now = at
            timer.callback()
        self.cpu += (end - self.now) * self.cores if self.running else 0
        self.now = end

    def budget(self, **kwargs):
        return CpuBudget(
            4321,
            monotonic=self.monotonic,
            sampler=self.sample,
            signal_group=self.send,
            overhead_sampler=lambda: self.overhead,
            timer_factory=self.timer,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("text", "expected"),
    [("00:00.25", 0.25), ("125:02.50", 7502.5), ("01:02:03", 3723), ("2-01:02:03.75", 176523.75)],
)
def test_parse_macos_and_linux_cpu_times(text, expected):
    assert parse_cpu_time(text) == expected


@pytest.mark.parametrize("text", ["NaN", "1:60", "-1:00", "1:61:00", "1:2:3:4", "inf:01", "1:nan"])
def test_cpu_time_rejects_invalid_samples(text):
    with pytest.raises(ComputeBudgetError):
        parse_cpu_time(text)


def test_active_clock_excludes_only_bounded_rest():
    world = World()
    clock = ActiveClock(world.monotonic)
    world.advance(10)
    clock.pause(until=20)
    world.advance(4)
    assert clock.now() == 10
    assert clock.paused_seconds == 4
    world.advance(20)
    assert clock.now() == 24
    assert clock.paused_seconds == 10
    clock.resume()
    world.advance(1)
    assert clock.now() == 25


@pytest.mark.parametrize("cores", [1.0, 4.0, 12.0])
def test_aggregate_cpu_budget_repays_multicore_bursts(cores):
    world = World(cores=cores)
    budget = world.budget(percent=30)
    end = 20.0
    while world.now < end:
        delay = budget.tick()
        world.advance(min(delay, end - world.now))
    # A finite in-flight quantum is allowed, including a multi-core burst.  It
    # becomes measured debt rather than multiplied sustainable CPU allowance.
    assert world.cpu <= 0.30 * world.now + cores * 0.05 + 0.015 + 1e-9
    assert world.cpu >= 0.30 * world.now - cores * 0.05 - 0.02
    budget.close()
    assert not world.running


def test_independent_timer_stops_child_when_parent_does_not_tick():
    world = World(cores=4)
    budget = world.budget()
    budget.tick()
    world.advance(100)
    assert world.cpu == pytest.approx(0.2)
    assert budget.stopped
    assert not world.running
    # The safety STOP cannot erase 100 seconds of a stalled parent's watchdog.
    assert budget.active_clock.now() > 99


def test_running_tick_never_blocks_stop_timer_with_an_external_sample():
    world = World()
    budget = world.budget()
    budget.tick()
    world.sample_error = AssertionError("a running child must not be sampled")
    # The external sample would hold the lock needed by the safety STOP timer.
    # Active ticks must return before calling it.
    assert budget.tick() == 0.05
    world.sample_error = None
    world.advance(0.05)
    assert budget.stopped


def test_timer_start_failure_cannot_send_cont():
    world = World()

    class BrokenTimer(FakeTimer):
        def start(self):
            raise RuntimeError("timer unavailable")

    budget = world.budget()
    budget._timer_factory = lambda delay, callback: BrokenTimer(world, delay, callback)
    with pytest.raises(ComputeBudgetError):
        budget.tick()
    assert signal.SIGCONT not in [sig for _, sig in world.signals]
    assert not world.running


def test_budget_does_not_bank_unlimited_credit_during_idle():
    world = World(cores=0)
    budget = world.budget()
    budget.tick()
    world.advance(1000)
    world.cores = 4
    budget.tick()
    world.advance(0.05)
    assert budget.stopped
    assert budget.debt_seconds > 0.15


def test_idle_child_does_not_pause_no_progress_clock():
    world = World(cores=0)
    budget = world.budget()
    for _ in range(100):
        budget.tick()
        world.advance(0.05)
    assert budget.active_clock.paused_seconds == 0
    assert budget.active_clock.now() == pytest.approx(world.now)


def test_controller_overhead_is_charged_but_does_not_hide_idle_child():
    world = World(cores=0)
    budget = world.budget()
    budget.tick()
    world.overhead = 0.2
    world.advance(0.05)
    assert budget.cpu_seconds == pytest.approx(0.2)
    assert budget.debt_seconds > 0.16
    world.advance(0.1)
    assert budget.active_clock.paused_seconds == 0
    assert budget.active_clock.now() == pytest.approx(world.now)


def test_expensive_sampler_cannot_starve_work_after_debt_is_repaid():
    world = World()
    sample_count = 0

    def expensive_sample(pids):
        nonlocal sample_count
        sample_count += 1
        # More CPU than the 0.015-second token capacity.  This reproduced an
        # endless sampling/debt loop when every stopped tick invoked ps again.
        world.overhead += 0.03
        return world.sample(pids)

    budget = world.budget()
    budget._sampler = expensive_sample
    end = 10.0
    while world.now < end:
        delay = budget.tick()
        world.advance(min(delay, end - world.now))
    assert world.cpu > 1.0
    assert sample_count > 20
    assert world.cpu + world.overhead <= 0.30 * world.now + 0.08 + 1e-9
    assert sum(sig == signal.SIGCONT for _, sig in world.signals) > 20
    budget.close()


def test_actual_budget_rest_is_excluded_from_active_clock():
    world = World()
    budget = world.budget()
    budget.tick()
    world.advance(0.05)
    assert budget.debt_seconds > 0
    world.advance(0.04)
    assert budget.active_clock.paused_seconds == pytest.approx(0.04)
    assert budget.active_clock.now() == pytest.approx(0.05)


def test_sampler_failure_stops_group_and_never_continues_again():
    world = World()
    budget = world.budget()
    budget.tick()
    world.sample_error = PermissionError("test sample denied")
    world.advance(0.05)
    assert not world.running
    with pytest.raises(ComputeBudgetError, match="child group stopped"):
        budget.tick()
    assert world.signals[-1][1] == signal.SIGSTOP


def test_cont_failure_cancels_timer_and_stops_group():
    world = World()
    budget = world.budget()
    world.signal_error = signal.SIGCONT
    with pytest.raises(ComputeBudgetError, match="child group stopped"):
        budget.tick()
    assert not world.running
    assert world.signals[-1][1] == signal.SIGSTOP
    world.advance(1)
    assert world.cpu == 0


def test_stop_failure_is_explicit_and_no_unbounded_cont_fallback():
    world = World()
    budget = world.budget()
    budget.tick()
    world.signal_error = signal.SIGSTOP
    world.advance(0.05)
    with pytest.raises(ComputeBudgetError, match="emergency SIGSTOP also failed"):
        budget.tick()
    assert sum(sig == signal.SIGCONT for _, sig in world.signals) == 1


@pytest.mark.parametrize("kwargs", [{"cancelled": True}, {"deadline": 0.0}])
def test_cancel_and_wall_deadline_stop_without_terminating(kwargs):
    world = World()
    budget = world.budget()
    budget.tick()
    with pytest.raises(ComputeBudgetError):
        budget.tick(**kwargs)
    assert not world.running
    assert {sig for _, sig in world.signals} <= {signal.SIGSTOP, signal.SIGCONT}


def test_close_leaves_group_stopped_and_invalidates_stale_timer():
    world = World()
    budget = world.budget()
    budget.tick()
    budget.close()
    count = len(world.signals)
    world.advance(1)
    assert len(world.signals) == count
    assert not world.running
    with pytest.raises(ComputeBudgetError, match="closed"):
        budget.tick()


def test_confirmed_child_exit_does_not_signal_reused_pid():
    world = World()
    budget = world.budget()
    budget.tick()
    count = len(world.signals)
    budget.close(child_exited=True)
    world.advance(1)
    assert len(world.signals) == count


def test_cpu_counter_decrease_fails_closed():
    world = World()
    world.cpu = 1.0
    budget = world.budget()
    world.cpu = 0.1
    budget.tick()
    world.advance(0.05)
    with pytest.raises(ComputeBudgetError):
        budget.tick()
    assert not world.running


def test_sampler_discovers_only_owned_group_and_counts_observed_live_workers(monkeypatch):
    monkeypatch.setattr(compute_budget.sys, "platform", "darwin")
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        assert kwargs["timeout"] == 1.0
        if args[0] == "pgrep":
            return subprocess.CompletedProcess(args, 0, "4321\n4322\n", "")
        # Parent TIME excludes the worker.  Each observed process is counted.
        return subprocess.CompletedProcess(args, 0, "4321 4321 00:01.20\n4322 4321 00:00.30\n", "")

    monkeypatch.setattr(compute_budget.subprocess, "run", run)
    assert ProcessCpuSampler(4321)((4321,)) == {4321: 1.5}
    assert calls == [
        ["pgrep", "-g", "4321"],
        ["ps", "-p", "4321,4322", "-o", "pid=,pgid=,time="],
    ]


def test_darwin_reaped_worker_cpu_is_retained_when_parent_time_does_not_increase(monkeypatch):
    monkeypatch.setattr(compute_budget.sys, "platform", "darwin")
    snapshots = iter(
        [
            ("4321\n4322\n", "4321 4321 00:00.10\n4322 4321 00:00.30\n"),
            # Darwin's parent TIME does not absorb the reaped worker's TIME.
            ("4321\n", "4321 4321 00:00.10\n"),
            ("4321\n4323\n", "4321 4321 00:00.20\n4323 4321 00:00.15\n"),
            ("4321\n", "4321 4321 00:00.25\n"),
        ]
    )
    current = None

    def run(args, **kwargs):
        nonlocal current
        assert "-S" not in args
        if args[0] == "pgrep":
            current = next(snapshots)
            output = current[0]
        else:
            output = current[1]
        return subprocess.CompletedProcess(args, 0, output, "")

    monkeypatch.setattr(compute_budget.subprocess, "run", run)
    sampler = ProcessCpuSampler(4321)
    assert sampler((4321,))[4321] == pytest.approx(0.40)
    assert sampler((4321,))[4321] == pytest.approx(0.40)
    assert sampler((4321,))[4321] == pytest.approx(0.65)
    assert sampler((4321,))[4321] == pytest.approx(0.70)


def test_group_sampler_rejects_per_pid_counter_reset_instead_of_hiding_usage(monkeypatch):
    monkeypatch.setattr(compute_budget.sys, "platform", "darwin")
    cpu = iter(["00:00.30", "00:00.10"])

    def run(args, **kwargs):
        output = "4321\n" if args[0] == "pgrep" else f"4321 4321 {next(cpu)}\n"
        return subprocess.CompletedProcess(args, 0, output, "")

    monkeypatch.setattr(compute_budget.subprocess, "run", run)
    sampler = ProcessCpuSampler(4321)
    assert sampler((4321,)) == {4321: 0.3}
    with pytest.raises(ComputeBudgetError, match="per-PID CPU counter decreased"):
        sampler((4321,))


@pytest.mark.parametrize(
    "output", ["4321 9999 00:01.00\n", "4321 4321 00:01.00\n", "4321 4321 nan\n4322 4321 00:01.00\n"]
)
def test_group_sampler_rejects_group_change_missing_member_or_bad_cpu(monkeypatch, output):
    monkeypatch.setattr(compute_budget.sys, "platform", "darwin")
    def run(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, "4321\n4322\n" if args[0] == "pgrep" else output, "")

    monkeypatch.setattr(compute_budget.subprocess, "run", run)
    with pytest.raises(ComputeBudgetError):
        ProcessCpuSampler(4321)((4321,))


def test_group_sampler_error_is_not_an_empty_success(monkeypatch):
    monkeypatch.setattr(
        compute_budget.subprocess, "run", lambda args, **kwargs: subprocess.CompletedProcess(args, 1, "", "denied")
    )
    with pytest.raises(ComputeBudgetError):
        ProcessCpuSampler(4321)((4321,))


@pytest.mark.parametrize("kwargs", [{"percent": 0}, {"percent": float("nan")}, {"max_active_quantum": 0.1}])
def test_invalid_configuration_cannot_start_a_child(kwargs):
    world = World()
    with pytest.raises(ValueError):
        world.budget(**kwargs)
    assert world.signals == []


def _linux_stat(*, pid=4321, pgid=4321, comm="cpu worker (child)", user=7, system=4):
    # Fields 3..15: state, ppid, pgrp, session, tty, tpgid, flags,
    # minflt, cminflt, majflt, cmajflt, utime, stime.
    return f"{pid} ({comm}) S 100 {pgid} 4321 0 -1 0 0 0 0 0 {user} {system}"


@pytest.mark.parametrize("comm", ["worker", "cpu worker", "cpu (nested) worker)"])
def test_linux_stat_cpu_retains_subsecond_precision_with_comm_spaces(comm):
    assert compute_budget.parse_linux_cpu_stat(
        _linux_stat(comm=comm), pid=4321, pgid=4321, clock_ticks=100
    ) == pytest.approx(0.11)


@pytest.mark.parametrize(
    "text,clock_ticks",
    [
        (_linux_stat(pgid=9999), 100),
        (_linux_stat(pid=9999), 100),
        ("4321 (short worker) S 100 4321", 100),
        ("4321 missing comm S 100 4321", 100),
        (_linux_stat(user=-1), 100),
        (_linux_stat(system="bad"), 100),
        (_linux_stat(), 0),
    ],
)
def test_linux_stat_rejects_wrong_group_pid_short_or_invalid_counters(text, clock_ticks):
    with pytest.raises(ComputeBudgetError):
        compute_budget.parse_linux_cpu_stat(text, pid=4321, pgid=4321, clock_ticks=clock_ticks)


def test_linux_group_sampler_reads_only_owned_proc_stat_and_preserves_reaped_cpu(monkeypatch):
    monkeypatch.setattr(compute_budget.sys, "platform", "linux")
    monkeypatch.setattr(compute_budget.os, "sysconf", lambda name: 100)
    inventories = iter(["4321\n4322\n", "4321\n"])
    proc_paths = []

    def run(args, **kwargs):
        assert args == ["pgrep", "-g", "4321"]
        return subprocess.CompletedProcess(args, 0, next(inventories), "")

    def read_text(path, **kwargs):
        proc_paths.append(str(path))
        pid = int(path.parent.name)
        return _linux_stat(pid=pid, user=5 if pid == 4321 else 7, system=1)

    monkeypatch.setattr(compute_budget.subprocess, "run", run)
    monkeypatch.setattr(Path, "read_text", read_text)
    sampler = ProcessCpuSampler(4321)
    assert sampler((4321,)) == {4321: pytest.approx(0.14)}
    assert sampler((4321,)) == {4321: pytest.approx(0.14)}
    assert proc_paths == ["/proc/4321/stat", "/proc/4322/stat", "/proc/4321/stat"]


def test_linux_proc_access_failure_does_not_fall_back_to_coarse_ps(monkeypatch):
    monkeypatch.setattr(compute_budget.sys, "platform", "linux")
    monkeypatch.setattr(
        compute_budget.subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 0, "4321\n", ""),
    )

    def denied(path, **kwargs):
        raise PermissionError("owned proc stat denied")

    monkeypatch.setattr(Path, "read_text", denied)
    with pytest.raises(ComputeBudgetError, match="Linux CPU sampler failed"):
        ProcessCpuSampler(4321)((4321,))


def test_controller_rest_and_reserve_do_not_hide_idle_after_tiny_child_debt():
    world = World(cores=0.7)
    budget = world.budget()
    budget.tick()
    world.overhead = 3.0
    world.advance(0.05)
    # The child used 0.035 CPU seconds: only 0.005 beyond its accrued
    # 0.030 allowance.  Controller debt must not erase the next ten seconds.
    world.advance(5.0)
    assert budget.stopped
    assert budget.active_clock.paused_seconds == pytest.approx(0.005 / 0.30)
    assert budget.active_clock.now() > 5.0
