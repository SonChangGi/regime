from __future__ import annotations

from dataclasses import replace
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from regime_lab import automation
from regime_lab.compute_budget import ActiveClock, ComputeBudgetError


@pytest.mark.parametrize("mismatched_bytes", [False, True])
def test_package_and_verify_share_cpu_budget_and_retain_byte_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mismatched_bytes: bool,
) -> None:
    from test_weekly_automation import _settings

    settings = _settings(tmp_path)
    candidate = tmp_path / "candidate.json"
    candidate.write_bytes(b"strictly-validated-candidate")
    calls = []

    def run(command, **kwargs):
        assert kwargs["cpu_percent"] == 30
        assert callable(kwargs["heartbeat"])
        assert callable(kwargs["paused_heartbeat"])
        assert kwargs["heartbeat_interval"] == settings.heartbeat_interval.total_seconds()
        calls.append(command)
        if "scripts/package_public_demo.py" in command:
            output = Path(command[command.index("--output") + 1])
            payload = output / automation.PUBLIC_PAYLOAD_PATH
            payload.parent.mkdir(parents=True)
            payload.write_bytes(b"changed" if mismatched_bytes else candidate.read_bytes())
        return b""

    monkeypatch.setattr(automation, "_run", run)
    if mismatched_bytes:
        with pytest.raises(automation.AutomationError, match="payload bytes differ"):
            automation._verify_candidate_package(settings, candidate, None, None)
    else:
        automation._verify_candidate_package(settings, candidate, None, None)
    assert [command[1] for command in calls] == [
        "scripts/package_public_demo.py", "scripts/verify_public_package.py",
    ]


class _FakeProcess:
    pid = 987650

    def __init__(self, *, output: bytes = b"captured", complete_after: int = 2):
        self.returncode = None
        self.stdout = io.BytesIO(output)
        self.output = output
        self.calls = 0
        self.complete_after = complete_after

    def poll(self):
        return self.returncode

    def communicate(self, timeout=None):
        self.calls += 1
        if self.calls < self.complete_after:
            raise subprocess.TimeoutExpired("fake", timeout)
        self.returncode = 0
        return self.output, b"stderr"

    def wait(self, timeout=None):
        self.calls += 1
        if self.calls < self.complete_after:
            raise subprocess.TimeoutExpired("fake", timeout)
        self.returncode = 0
        return self.returncode


def test_fake_low_load_capture_keeps_heartbeat_and_limits_native_threads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess(output=b"a" * 200_000)
    captured = {}
    closures = []
    heartbeats = []

    def popen(args, **kwargs):
        captured.update(kwargs)
        assert args == ["fake", "build"]
        return process

    class Budget:
        def __init__(self, pid, **kwargs):
            assert pid == process.pid
            assert kwargs["percent"] == 30

        def tick(self, **_kwargs):
            return .001

        def close(self, *, child_exited=False):
            closures.append(child_exited)

    monkeypatch.setattr(automation.subprocess, "Popen", popen)
    monkeypatch.setattr(automation, "CpuBudget", Budget)
    monkeypatch.setattr(automation, "_compute_group_alive", lambda _pid: process.poll() is None, raising=False)
    result = automation._run(
        ["fake", "build"], cwd=tmp_path, capture=True, cpu_percent=30,
        heartbeat=lambda: heartbeats.append(True), heartbeat_interval=0,
        env={"TEST_VALUE": "preserved", "OMP_NUM_THREADS": "8"},
    )
    assert result == b"a" * 200_000
    assert heartbeats
    assert closures[-1] is True
    assert captured["start_new_session"] is True
    assert captured["env"]["TEST_VALUE"] == "preserved"
    assert all(captured["env"][name] == "1" for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    ))


def test_fake_low_load_controller_failure_retains_lease_until_child_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess(complete_after=999)
    original = ComputeBudgetError("sampler denied")
    constructors = []
    signals = []
    lease_checks = []
    lock_path = tmp_path / "automation.lock"

    class Budget:
        def __init__(self, _pid, **kwargs):
            constructors.append(kwargs)

        def tick(self, **_kwargs):
            raise original

        def close(self, **_kwargs):
            pass

    def paused_heartbeat():
        assert process.poll() is None
        with pytest.raises(automation.AlreadyRunning):
            with automation.automation_lock(lock_path):
                pytest.fail("live child must retain its automation lease")
        lease_checks.append(True)
        process.returncode = 130

    def denied_cont(pid, sig):
        signals.append((pid, sig))
        if sig == signal.SIGCONT:
            raise PermissionError("continuation denied")

    monkeypatch.setattr(automation.subprocess, "Popen", lambda *_a, **_kw: process)
    monkeypatch.setattr(automation, "CpuBudget", Budget)
    monkeypatch.setattr(automation.os, "killpg", denied_cont)
    monkeypatch.setattr(automation, "_compute_group_alive", lambda _pid: process.poll() is None, raising=False)
    monkeypatch.setattr(automation.time, "sleep", lambda _seconds: None)
    with automation.automation_lock(lock_path):
        with pytest.raises(ComputeBudgetError) as caught:
            automation._run(
                ["fake"], cwd=tmp_path, cpu_percent=30,
                paused_heartbeat=paused_heartbeat, heartbeat_interval=.01,
            )
        assert caught.value is original
    assert lease_checks == [True]
    assert len(constructors) == 1
    assert (process.pid, signal.SIGINT) in signals
    assert signals.count((process.pid, signal.SIGCONT)) == 1
    assert all(sig not in (signal.SIGKILL, signal.SIGTERM) for _, sig in signals)


def test_fake_low_load_leader_exit_does_not_release_live_worker_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess()
    process.returncode = 0
    worker_alive = [True]
    signals = []
    lock_path = tmp_path / "automation.lock"

    def signal_group(pid, sig):
        assert worker_alive[0] or sig == signal.SIGSTOP
        if sig == signal.SIGINT:
            with pytest.raises(automation.AlreadyRunning):
                with automation.automation_lock(lock_path):
                    pytest.fail("orphaned worker group must retain the lease")
        if sig == signal.SIGCONT:
            worker_alive[0] = False
        signals.append((pid, sig))

    monkeypatch.setattr(automation.subprocess, "Popen", lambda *_a, **_kw: process)
    monkeypatch.setattr(automation, "_compute_group_alive", lambda _pid: worker_alive[0], raising=False)
    monkeypatch.setattr(automation.os, "killpg", signal_group)
    with automation.automation_lock(lock_path):
        with pytest.raises(ComputeBudgetError, match="leader exited"):
            automation._run(["fake"], cwd=tmp_path, cpu_percent=30)
    assert worker_alive[0] is False
    assert (process.pid, signal.SIGINT) in signals
    assert (process.pid, signal.SIGCONT) in signals
    assert all(sig not in (signal.SIGKILL, signal.SIGTERM) for _, sig in signals)


def test_fake_exited_leader_gives_streamed_worker_full_cancellation_quantum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess(complete_after=1)
    process.returncode = 0
    clock = [0.0]
    worker_alive = [True]
    running_since = [None]
    signals = []
    lock_path = tmp_path / "automation.lock"

    class Thread:
        ident = 1

        def __init__(self, **_kwargs):
            pass

        def start(self):
            pass

        def join(self, **_kwargs):
            assert not worker_alive[0]

    class Timer:
        def __init__(self, *_args):
            pass

        def start(self):
            pass

        def cancel(self):
            pass

    def signal_group(pid, sig):
        signals.append((sig, clock[0]))
        if sig == signal.SIGCONT:
            running_since[0] = clock[0]
        if sig == signal.SIGSTOP:
            assert running_since[0] is not None
            assert clock[0] - running_since[0] >= .03
            worker_alive[0] = False

    def sleep(seconds):
        assert worker_alive[0]
        with pytest.raises(automation.AlreadyRunning):
            with automation.automation_lock(lock_path):
                pytest.fail("lease must remain held for the worker's entire grace slice")
        clock[0] += seconds

    monkeypatch.setattr(automation.subprocess, "Popen", lambda *_a, **_kw: process)
    monkeypatch.setattr(automation.threading, "Thread", Thread)
    monkeypatch.setattr(automation.threading, "Timer", Timer)
    monkeypatch.setattr(automation.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(automation.time, "sleep", sleep)
    monkeypatch.setattr(automation, "_compute_group_alive", lambda _pid: worker_alive[0], raising=False)
    monkeypatch.setattr(automation.os, "killpg", signal_group)
    with automation.automation_lock(lock_path):
        with pytest.raises(ComputeBudgetError, match="leader exited"):
            automation._run(
                ["fake"], cwd=tmp_path, cpu_percent=30,
                output_activity=lambda: None,
            )
    assert not worker_alive[0]
    assert signals == [
        (signal.SIGINT, 0.0), (signal.SIGCONT, 0.0), (signal.SIGSTOP, .03),
    ]


def test_fake_low_load_reader_start_failure_still_cancels_owned_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess()
    signals = []
    closures = []

    class Thread:
        ident = None

        def __init__(self, **_kwargs):
            pass

        def start(self):
            raise RuntimeError("reader could not start")

        def join(self, **_kwargs):
            pytest.fail("an unstarted reader cannot be joined")

    class Budget:
        def __init__(self, _pid, **_kwargs):
            pass

        def close(self, *, child_exited=False):
            closures.append(child_exited)

    def signal_group(pid, sig):
        signals.append((pid, sig))
        if sig == signal.SIGINT:
            process.returncode = 130

    monkeypatch.setattr(automation.subprocess, "Popen", lambda *_a, **_kw: process)
    monkeypatch.setattr(automation.threading, "Thread", Thread)
    monkeypatch.setattr(automation, "CpuBudget", Budget)
    monkeypatch.setattr(automation, "_compute_group_alive", lambda _pid: process.poll() is None, raising=False)
    monkeypatch.setattr(automation.os, "killpg", signal_group)
    with pytest.raises(RuntimeError, match="reader could not start"):
        automation._run(
            ["fake"], cwd=tmp_path, cpu_percent=30,
            output_activity=lambda: None,
        )
    assert process.returncode == 130
    assert closures == [False, True]
    assert signals == [(process.pid, signal.SIGINT)]


def test_compute_pause_health_is_blocked_and_preserves_success_evidence(
    tmp_path: Path,
) -> None:
    from test_weekly_automation import _settings

    settings = _settings(tmp_path)
    previous = {
        "status": "running", "stage": "train_models", "run_id": "run-1",
        "heartbeat_at": "2026-08-21T01:00:00+00:00",
        "stage_started_at": "2026-08-21T00:00:00+00:00",
        "last_stage_successes": {
            "collect_live_data": {"completed_at": "2026-08-21T00:00:00+00:00"},
        },
        "last_full_success_at": "2026-08-14T01:00:00+00:00",
        "last_full_success_target": "2026-08-07T20:00:00+00:00",
        "last_success_at": "2026-08-14T01:00:00+00:00",
        "last_success_target": "2026-08-07T20:00:00+00:00",
        "next_retry_at": "2026-08-22T00:00:00+00:00",
        "retry_class": "transient",
    }
    automation.write_json_atomic(settings.status_path, previous)
    for _ in range(2):
        automation._compute_pause_heartbeat(settings)
        updated = json.loads(settings.status_path.read_text())
        assert updated["status"] == "blocked"
        assert updated["stage"] == "compute_paused"
        assert updated["error_code"] == "compute_control_paused"
        assert updated["next_retry_at"] is None
        assert updated["retry_class"] == "blocked"
        assert updated["heartbeat_at"] != previous["heartbeat_at"]
        for field in (
            "last_stage_successes", "last_full_success_at", "last_full_success_target",
            "last_success_at", "last_success_target", "stage_started_at", "run_id",
        ):
            assert updated[field] == previous[field]
        assert "train_models" not in updated["last_stage_successes"]


@pytest.mark.parametrize("budget_rest", [0.0, 100.0])
def test_fake_build_watchdog_excludes_confirmed_rest_but_rejects_genuine_idle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, budget_rest: float,
) -> None:
    from datetime import timedelta
    from test_weekly_automation import CANDIDATE_CONTEXT, LIVE_TARGET, _settings

    settings = replace(_settings(tmp_path), no_progress_timeout=timedelta(seconds=1))
    clock = [0.0]
    monkeypatch.setattr(automation.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(automation, "_load_collection_report", lambda *_a, **_kw: {"ready_for_training": True})
    monkeypatch.setattr(automation, "_walkforward_checkpoint_progress", lambda *_a: None)
    monkeypatch.setattr(automation, "_heartbeat_status", lambda *_a, **_kw: None)

    def controlled_run(args, *, heartbeat, active_clock, cpu_percent, **_kwargs):
        assert cpu_percent == 30
        assert args[args.index("--model-workers") + 1] == "1"
        if budget_rest:
            active_clock.pause(until=budget_rest)
            clock[0] = budget_rest
            active_clock.resume()
        clock[0] = budget_rest + .9
        heartbeat()
        clock[0] = budget_rest + 1.1
        heartbeat()
        pytest.fail("genuine idle must still fail the watchdog")

    monkeypatch.setattr(automation, "_run", controlled_run)
    with pytest.raises(automation.AutomationError, match="no checkpoint progress"):
        automation._build_candidate_generation(
            settings, target=LIVE_TARGET, context=CANDIDATE_CONTEXT,
            started_at=LIVE_TARGET,
        )


def test_real_low_load_cpu_budget_drains_large_captured_output(
    tmp_path: Path,
) -> None:
    # A fraction of one CPU second is enough to measure throttling; no model fit.
    script = """
import json, sys, time
print('x' * 131072, flush=True)
print('y' * 131072, file=sys.stderr, flush=True)
started = time.process_time()
while time.process_time() - started < .25:
    pass
print(json.dumps({'cpu_seconds': time.process_time() - started}), flush=True)
"""
    heartbeats = []
    active_clock = ActiveClock()
    started = time.monotonic()
    output = automation._run(
        [sys.executable, "-c", script], cwd=tmp_path, capture=True,
        timeout=8, heartbeat=lambda: heartbeats.append(time.monotonic()),
        heartbeat_interval=.03, cpu_percent=30, active_clock=active_clock,
    )
    elapsed = time.monotonic() - started
    lines = output.splitlines()
    assert lines[0] == b"x" * 131072
    cpu_seconds = json.loads(lines[-1])["cpu_seconds"]
    assert cpu_seconds / elapsed < .45
    assert heartbeats
    assert active_clock.paused_seconds > 0


def test_real_low_load_streams_progress_and_forces_native_threads_to_one(
    tmp_path: Path, capfd: pytest.CaptureFixture[str],
) -> None:
    script = """
import json, os, time
names = ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
         'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']
print(json.dumps({name: os.environ[name] for name in names}), flush=True)
started = time.process_time()
while time.process_time() - started < .12:
    pass
print('completed', flush=True)
"""
    activity = []
    heartbeats = []
    automation._run(
        [sys.executable, "-c", script], cwd=tmp_path,
        timeout=8, output_activity=lambda: activity.append(True),
        heartbeat=lambda: heartbeats.append(True), heartbeat_interval=.03,
        env={"OMP_NUM_THREADS": "8"}, cpu_percent=30,
    )
    lines = capfd.readouterr().out.splitlines()
    assert set(json.loads(lines[0]).values()) == {"1"}
    assert lines[-1] == "completed"
    assert len(activity) == 2
    assert heartbeats


@pytest.mark.parametrize("cancel_by", ["heartbeat", "deadline"])
def test_real_low_load_normal_cancellation_waits_for_child_finally(
    tmp_path: Path, cancel_by: str,
) -> None:
    ready = tmp_path / "ready"
    finalized = tmp_path / "finalized"
    script = """
from pathlib import Path
import sys
try:
    Path(sys.argv[1]).write_text('ready')
    while True:
        pass
finally:
    Path(sys.argv[2]).write_text('finalized')
"""
    class CancelRequested(RuntimeError):
        pass

    def heartbeat():
        if cancel_by == "heartbeat" and ready.exists():
            raise CancelRequested("normal cancellation")

    expected = CancelRequested if cancel_by == "heartbeat" else ComputeBudgetError
    with pytest.raises(expected):
        automation._run(
            [sys.executable, "-c", script, str(ready), str(finalized)],
            cwd=tmp_path, capture=True, cpu_percent=30,
            heartbeat=heartbeat, heartbeat_interval=.03,
            timeout=4 if cancel_by == "heartbeat" else .4,
        )
    assert ready.read_text() == "ready"
    assert finalized.read_text() == "finalized"
