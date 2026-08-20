from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import plistlib
import subprocess
import sys
import time

import pytest

from regime_lab import automation


UTC = timezone.utc


def _settings(tmp_path: Path) -> automation.AutomationSettings:
    root = tmp_path / "repo"
    root.mkdir()
    python = root / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    state = root / "build/weekly-automation"
    return automation.AutomationSettings(
        config_path=root / "config/automation.json",
        root=root,
        automation_id="weekly-regime-release-v1",
        schedule_hour=21,
        schedule_minute=17,
        minimum_cutoff_age=timedelta(hours=24),
        require_ac_power=True,
        profile="standard",
        database=root / "data/regime.sqlite3",
        payload=state / "generation/regime-results.json",
        artifacts=state / "generation/artifacts",
        state_directory=state,
        authorization=root / "data/automation/authorization.json",
        repository="SonChangGi/regime",
        remote="origin",
        branch="main",
        workflow="pages.yml",
        public_root="https://sonchanggi.github.io/regime/",
        workflow_timeout=timedelta(hours=1),
        public_readback_timeout=timedelta(minutes=10),
        retry_hours=(3, 9, 15, 21),
        transient_retry_delay=timedelta(hours=6),
        heartbeat_interval=timedelta(seconds=30),
        stale_heartbeat_after=timedelta(minutes=15),
        notification_dedupe=timedelta(hours=24),
    )


def _write_json(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _write_health(
    settings: automation.AutomationSettings,
    *,
    status: str,
    heartbeat_at: datetime,
    consecutive_failures: int,
) -> None:
    _write_json(
        settings.status_path,
        {
            "schema_version": automation.HEALTH_SCHEMA_VERSION,
            "automation_id": settings.automation_id,
            "status": status,
            "stage": "train_models" if status == "running" else "failed",
            "started_at": (heartbeat_at - timedelta(minutes=1)).isoformat(),
            "heartbeat_at": heartbeat_at.isoformat(),
            "updated_at": heartbeat_at.isoformat(),
            "consecutive_failures": consecutive_failures,
        },
    )


def _install_plist(
    settings: automation.AutomationSettings,
    path: Path,
    *,
    document: dict[str, object] | None = None,
) -> dict[str, object]:
    expected = automation.launch_agent_document(settings)
    selected = document or expected
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        plistlib.dumps(selected, fmt=plistlib.FMT_XML, sort_keys=False)
    )
    return expected


def _forbid_external_work(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        pytest.fail("retry guard must stop before git, credentials, provider, or build work")

    for name in (
        "_git_preflight",
        "_validate_local_authorization",
        "_ensure_ac_power",
        "_candidate_context",
        "_load_cached_candidate",
        "_alpha_quota_preflight",
        "_build_candidate",
        "_publish_candidate",
    ):
        monkeypatch.setattr(automation, name, forbidden)


def test_launch_agent_document_uses_retry_schedule_and_long_run_process_policy(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)

    document = automation.launch_agent_document(settings)

    assert document["StartCalendarInterval"] == [
        {"Hour": 3, "Minute": 17},
        {"Hour": 9, "Minute": 17},
        {"Hour": 15, "Minute": 17},
        {"Hour": 21, "Minute": 17},
    ]
    assert document["ProgramArguments"][:2] == ["/usr/bin/caffeinate", "-s"]
    assert document["ProcessType"] == "Standard"
    assert document["ExitTimeOut"] == 120
    assert "LowPriorityIO" not in document


def test_launch_agent_status_compares_plists_semantically_and_reports_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    plist = tmp_path / "LaunchAgents/regime.plist"
    expected = automation.launch_agent_document(settings)
    reordered = {key: expected[key] for key in reversed(tuple(expected))}
    _install_plist(settings, plist, document=reordered)
    _write_health(
        settings,
        status="succeeded",
        heartbeat_at=datetime.now(UTC),
        consecutive_failures=0,
    )
    monkeypatch.setattr(automation, "_launch_agent_path", lambda: plist)
    monkeypatch.setattr(automation, "_launch_agent_loaded", lambda _settings: True)

    matching = automation.launch_agent_status(settings)
    assert matching["configuration_matches"] is True
    assert matching["operational"] is True
    assert matching["ok"] is True

    drifted = dict(expected)
    drifted["ProgramArguments"] = list(expected["ProgramArguments"])[2:]
    _install_plist(settings, plist, document=drifted)

    status = automation.launch_agent_status(settings)
    assert status["installed"] is True
    assert status["loaded"] is True
    assert status["configuration_matches"] is False
    assert status["drift_keys"] == ["ProgramArguments"]
    assert status["operational"] is False
    assert status["ok"] is False


@pytest.mark.parametrize(
    ("health_status", "age", "failures", "expected_stale"),
    [
        ("failed", timedelta(seconds=0), 1, False),
        ("running", timedelta(hours=1), 0, True),
    ],
)
def test_failed_or_stale_heartbeat_is_not_operational(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    health_status: str,
    age: timedelta,
    failures: int,
    expected_stale: bool,
) -> None:
    settings = _settings(tmp_path)
    plist = tmp_path / "LaunchAgents/regime.plist"
    _install_plist(settings, plist)
    _write_health(
        settings,
        status=health_status,
        heartbeat_at=datetime.now(UTC) - age,
        consecutive_failures=failures,
    )
    monkeypatch.setattr(automation, "_launch_agent_path", lambda: plist)
    monkeypatch.setattr(automation, "_launch_agent_loaded", lambda _settings: True)

    status = automation.launch_agent_status(settings)

    assert status["heartbeat_stale"] is expected_stale
    assert status["operational"] is False
    assert status["ok"] is False


def test_old_terminal_health_is_not_reported_as_operational(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    plist = tmp_path / "LaunchAgents/regime.plist"
    _install_plist(settings, plist)
    _write_health(
        settings,
        status="succeeded",
        heartbeat_at=datetime.now(UTC) - timedelta(hours=12),
        consecutive_failures=0,
    )
    monkeypatch.setattr(automation, "_launch_agent_path", lambda: plist)
    monkeypatch.setattr(automation, "_launch_agent_loaded", lambda _settings: True)

    status = automation.launch_agent_status(settings)

    assert status["health_stale"] is True
    assert status["heartbeat_stale"] is False
    assert status["operational"] is False


def test_retry_backoff_skips_external_provider_and_build_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    now = datetime.fromisoformat("2026-08-18T00:00:00+00:00")
    target, ready = automation.target_cutoff(
        now, minimum_age=settings.minimum_cutoff_age
    )
    assert ready is True
    _write_json(
        settings.status_path,
        {
            "schema_version": automation.HEALTH_SCHEMA_VERSION,
            "automation_id": settings.automation_id,
            "status": "failed",
            "stage": "collect_sources",
            "started_at": (now - timedelta(hours=1)).isoformat(),
            "heartbeat_at": (now - timedelta(hours=1)).isoformat(),
            "updated_at": (now - timedelta(hours=1)).isoformat(),
            "target_data_as_of": target.isoformat(),
            "consecutive_failures": 1,
            "error_code": "provider_degraded",
            "retry_class": "transient",
            "next_retry_at": (now + timedelta(hours=5)).isoformat(),
            "recovery_fingerprint": "unchanged",
        },
    )
    _forbid_external_work(monkeypatch)

    result = automation.run_weekly_release(settings, now=now)

    assert result["status"] == "skipped"
    assert result["stage"] == "retry_backoff"
    assert result["next_retry_at"] == (now + timedelta(hours=5)).isoformat()


def test_same_blocked_fingerprint_skips_external_provider_and_build_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    now = datetime.fromisoformat("2026-08-18T00:00:00+00:00")
    target, ready = automation.target_cutoff(
        now, minimum_age=settings.minimum_cutoff_age
    )
    assert ready is True
    _write_json(
        settings.status_path,
        {
            "schema_version": automation.HEALTH_SCHEMA_VERSION,
            "automation_id": settings.automation_id,
            "status": "failed",
            "stage": "audit_candidate",
            "started_at": (now - timedelta(hours=1)).isoformat(),
            "heartbeat_at": (now - timedelta(hours=1)).isoformat(),
            "updated_at": (now - timedelta(hours=1)).isoformat(),
            "target_data_as_of": target.isoformat(),
            "consecutive_failures": 1,
            "error_code": "audit_candidate_blocked",
            "retry_class": "blocked",
            "next_retry_at": None,
            "recovery_fingerprint": "same-fingerprint",
        },
    )
    monkeypatch.setattr(
        automation, "_recovery_fingerprint", lambda _settings: "same-fingerprint"
    )
    _forbid_external_work(monkeypatch)

    result = automation.run_weekly_release(settings, now=now)

    assert result["status"] == "blocked"
    assert result["stage"] == "retry_blocked"
    assert result["recovery_fingerprint"] == "same-fingerprint"


def test_notification_deduplicates_same_failure_but_sends_changed_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    target = datetime.fromisoformat("2026-08-14T20:00:00+00:00")
    first_at = datetime.fromisoformat("2026-08-18T00:00:00+00:00")
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(automation, "_notifications_enabled", lambda _settings: True)
    monkeypatch.setattr(
        automation,
        "_send_macos_notification",
        lambda title, message: sent.append((title, message)),
    )

    first = automation._record_notification(
        settings,
        kind="failure",
        target=target,
        error_code="alpha_timeout",
        retry_class="transient",
        now=first_at,
    )
    duplicate = automation._record_notification(
        settings,
        kind="failure",
        target=target,
        error_code="alpha_timeout",
        retry_class="transient",
        now=first_at + timedelta(hours=1),
    )
    changed = automation._record_notification(
        settings,
        kind="failure",
        target=target,
        error_code="alfred_http_500",
        retry_class="transient",
        now=first_at + timedelta(hours=2),
    )

    assert first["delivery"] == "delivered"
    assert duplicate["delivery"] == "deduplicated"
    assert changed["delivery"] == "delivered"
    assert len(sent) == 2


def test_recovery_notification_is_sent_once_after_an_active_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    target = datetime.fromisoformat("2026-08-14T20:00:00+00:00")
    first_at = datetime.fromisoformat("2026-08-18T00:00:00+00:00")
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(automation, "_notifications_enabled", lambda _settings: True)
    monkeypatch.setattr(
        automation,
        "_send_macos_notification",
        lambda title, message: sent.append((title, message)),
    )
    automation._record_notification(
        settings,
        kind="failure",
        target=target,
        error_code="provider_degraded",
        retry_class="transient",
        now=first_at,
    )

    recovered = automation._record_notification(
        settings,
        kind="recovery",
        target=target,
        now=first_at + timedelta(hours=2),
    )
    duplicate = automation._record_notification(
        settings,
        kind="recovery",
        target=target,
        now=first_at + timedelta(hours=3),
    )

    assert recovered["delivery"] == "delivered"
    assert duplicate["delivery"] == "deduplicated"
    assert len(sent) == 2


def test_notification_delivery_failure_is_nonfatal_and_still_deduplicated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    target = datetime.fromisoformat("2026-08-14T20:00:00+00:00")
    first_at = datetime.fromisoformat("2026-08-18T00:00:00+00:00")
    attempts = 0

    def fail_delivery(_title: str, _message: str) -> None:
        nonlocal attempts
        attempts += 1
        raise automation.AutomationError("notification unavailable")

    monkeypatch.setattr(automation, "_notifications_enabled", lambda _settings: True)
    monkeypatch.setattr(automation, "_send_macos_notification", fail_delivery)

    failed = automation._record_notification(
        settings,
        kind="failure",
        target=target,
        error_code="provider_degraded",
        retry_class="transient",
        now=first_at,
    )
    duplicate = automation._record_notification(
        settings,
        kind="failure",
        target=target,
        error_code="provider_degraded",
        retry_class="transient",
        now=first_at + timedelta(hours=1),
    )

    assert failed["delivery"] == "failed_nonfatal"
    assert duplicate["delivery"] == "deduplicated"
    assert attempts == 1


def test_run_emits_heartbeats_while_subprocess_is_alive(tmp_path: Path) -> None:
    beats: list[float] = []

    output = automation._run(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(0.18); print('finished')",
        ],
        cwd=tmp_path,
        capture=True,
        timeout=2,
        heartbeat=lambda: beats.append(time.monotonic()),
        heartbeat_interval=0.02,
    )

    assert output.strip() == b"finished"
    assert len(beats) >= 2


def test_database_lock_receipt_is_retryable_instead_of_permanently_blocked(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    target = datetime.fromisoformat("2026-08-14T20:00:00+00:00")
    before = datetime.now(UTC)

    failure = automation._collection_failure(
        settings,
        target=target,
        report={"error_code": "database_build_lock_busy"},
    )

    assert failure.error_code == "database_build_lock_busy"
    assert failure.retry_class == "transient"
    assert failure.next_retry_at is not None
    assert before + timedelta(minutes=29) < failure.next_retry_at


def test_pages_wait_refreshes_heartbeat_on_every_poll(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    attempts = 0
    beats: list[int] = []

    def verify(*_args: object, **_kwargs: object) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise automation.AutomationError("Pages is still deploying")

    monkeypatch.setattr(automation, "verify_public_readback", verify)
    automation._wait_for_public_readback(
        settings,
        expected_payload=b"candidate",
        sleep=lambda _seconds: None,
        heartbeat=lambda: beats.append(attempts),
    )

    assert attempts == 3
    assert len(beats) == 3


def test_blocked_recovery_fingerprint_tracks_branch_and_remote_url(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=settings.root,
        check=True,
        capture_output=True,
    )
    tracked = settings.root / "tracked.txt"
    tracked.write_text("stable\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=settings.root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "seed",
        ],
        cwd=settings.root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "remote", "add", "origin", "git@example.invalid:one/repo.git"],
        cwd=settings.root,
        check=True,
    )
    main_fingerprint = automation._recovery_fingerprint(settings)

    subprocess.run(
        ["git", "switch", "-c", "feature"],
        cwd=settings.root,
        check=True,
        capture_output=True,
    )
    branch_fingerprint = automation._recovery_fingerprint(settings)
    subprocess.run(
        ["git", "switch", "main"],
        cwd=settings.root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "remote",
            "set-url",
            "origin",
            "git@example.invalid:two/repo.git",
        ],
        cwd=settings.root,
        check=True,
    )
    remote_fingerprint = automation._recovery_fingerprint(settings)

    assert branch_fingerprint != main_fingerprint
    assert remote_fingerprint != main_fingerprint


def test_run_redacts_network_command_details_on_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timeout(*_args: object, **_kwargs: object) -> object:
        raise subprocess.TimeoutExpired(
            ["git", "clone", "https://credential@example.invalid/repo.git"],
            5,
        )

    monkeypatch.setattr(automation.subprocess, "run", timeout)

    with pytest.raises(automation.AutomationError, match="git timed out") as caught:
        automation._run(
            ["git", "fetch", "origin", "main"],
            cwd=tmp_path,
            timeout=5,
            env=automation.GIT_NONINTERACTIVE_ENV,
        )

    assert "credential" not in str(caught.value)


def test_child_failure_before_collection_receipt_is_transient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    target = datetime.fromisoformat("2026-08-14T20:00:00+00:00")
    monkeypatch.setattr(
        automation,
        "_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            automation.AutomationError("python failed with exit code 1")
        ),
    )

    with pytest.raises(automation.ScheduledRetry) as caught:
        automation._build_candidate(
            settings,
            target=target,
            context={},
        )

    assert caught.value.error_code == "child_precollection_failed"
    assert caught.value.retry_class == "transient"
