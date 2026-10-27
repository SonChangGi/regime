from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from regime_lab import automation


UTC = timezone.utc
NOW = datetime(2026, 10, 27, 2, 0, tzinfo=UTC)
CURRENT_CUTOFF = "2026-10-23T20:00:00+00:00"
PREVIOUS_CUTOFF = "2026-10-16T20:00:00+00:00"


@pytest.fixture
def status_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[automation.AutomationSettings, dict[str, object]]:
    root = tmp_path / "operating-source"
    state_root = tmp_path / "private-state"
    state = state_root / "build/weekly-automation"
    settings = automation.AutomationSettings(
        config_path=root / "config/automation.json",
        root=root,
        state_root=state_root,
        automation_id="weekly-regime-release-v1",
        schedule_hour=21,
        schedule_minute=17,
        minimum_cutoff_age=timedelta(hours=24),
        require_ac_power=True,
        profile="standard",
        database=state_root / "data/regime.sqlite3",
        payload=state / "generation/regime-results.json",
        artifacts=state / "generation/artifacts",
        state_directory=state,
        authorization=state_root / "data/automation/authorization.json",
        repository="SonChangGi/regime",
        remote="origin",
        branch="main",
        workflow="pages.yml",
        public_root="https://sonchanggi.github.io/regime/",
        workflow_timeout=timedelta(hours=1),
        public_readback_timeout=timedelta(minutes=10),
        retry_hours=(3, 9, 15, 21),
    )
    plist = tmp_path / "LaunchAgents/regime.plist"
    plist.parent.mkdir()
    plist.write_text("installed test plist", encoding="utf-8")

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz: object = None) -> datetime:
            return NOW.astimezone(tz) if tz is not None else NOW.replace(tzinfo=None)

    monkeypatch.setattr(automation, "datetime", FrozenDateTime)
    monkeypatch.setattr(automation, "_launch_agent_path", lambda: plist)
    monkeypatch.setattr(automation, "_launch_agent_loaded", lambda _settings: True)
    monkeypatch.setattr(
        automation,
        "_launch_agent_configuration",
        lambda _settings, target=None: {
            "configuration_matches": True,
            "expected_plist_sha256": "a" * 64,
            "installed_plist_sha256": "a" * 64,
            "drift_keys": [],
        },
    )
    monkeypatch.setattr(
        automation, "_validate_provider_rights_policy", lambda _settings, now: None
    )
    monkeypatch.setattr(
        automation, "_validate_local_authorization_document", lambda _settings, now: None
    )
    health: dict[str, object] = {
        "schema_version": automation.HEALTH_SCHEMA_VERSION,
        "status": "succeeded",
        "stage": "already_current",
        "updated_at": (NOW - timedelta(minutes=1)).isoformat(),
        "heartbeat_at": (NOW - timedelta(minutes=1)).isoformat(),
        "consecutive_failures": 0,
        "publication_current": True,
        "public_data_as_of": CURRENT_CUTOFF,
        "last_public_verification_at": (NOW - timedelta(minutes=1)).isoformat(),
    }
    return settings, health


def _status(
    environment: tuple[automation.AutomationSettings, dict[str, object]],
) -> dict[str, object]:
    settings, health = environment
    settings.status_path.parent.mkdir(parents=True, exist_ok=True)
    settings.status_path.write_text(json.dumps(health), encoding="utf-8")
    return automation.launch_agent_status(settings)


def test_status_accepts_current_verified_publication_and_reports_both_roots(
    status_environment: tuple[automation.AutomationSettings, dict[str, object]],
) -> None:
    status = _status(status_environment)
    settings, _ = status_environment

    assert status["ok"] is status["operational"] is True
    assert status["scheduler_healthy"] is True
    assert status["publication_current"] is True
    assert status["expected_public_data_as_of"] == CURRENT_CUTOFF
    assert status["publication_lagging"] is False
    assert status["diagnostic_issues"] == []
    assert status["operating_source_root"] == str(settings.root)
    assert status["state_root"] == str(settings.storage_root)
    assert status["state_root"] != status["operating_source_root"]


def test_new_cutoff_is_not_required_before_the_provider_lag_has_elapsed(
    status_environment: tuple[automation.AutomationSettings, dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    within_lag = datetime(2026, 10, 24, 18, 17, tzinfo=UTC)

    class ProviderLagDateTime(datetime):
        @classmethod
        def now(cls, tz: object = None) -> datetime:
            return within_lag.astimezone(tz) if tz is not None else within_lag.replace(tzinfo=None)

    monkeypatch.setattr(automation, "datetime", ProviderLagDateTime)
    _, health = status_environment
    health.update(
        status="skipped",
        stage="not_due",
        updated_at=within_lag.isoformat(),
        heartbeat_at=within_lag.isoformat(),
        public_data_as_of=PREVIOUS_CUTOFF,
        last_public_verification_at=(within_lag - timedelta(hours=1)).isoformat(),
    )

    status = _status(status_environment)

    assert status["expected_public_data_as_of"] == PREVIOUS_CUTOFF
    assert status["publication_current"] is True
    assert status["publication_lagging"] is False
    assert status["ok"] is status["operational"] is True


@pytest.mark.parametrize(
    "health_status,stage",
    [("succeeded", "already_current"), ("skipped", "not_due")],
)
def test_week_rollover_rejects_cached_current_flag_even_with_recent_health(
    status_environment: tuple[automation.AutomationSettings, dict[str, object]],
    health_status: str,
    stage: str,
) -> None:
    _, health = status_environment
    health.update(status=health_status, stage=stage, public_data_as_of=PREVIOUS_CUTOFF)

    status = _status(status_environment)

    assert status["health_stale"] is False
    assert status["scheduler_healthy"] is True
    assert status["publication_current"] is False
    assert status["publication_lagging"] is True
    assert status["expected_public_data_as_of"] == CURRENT_CUTOFF
    assert "publication_lagging" in status["diagnostic_issues"]
    assert status["ok"] is status["operational"] is False


@pytest.mark.parametrize("installed", [True, False])
def test_missing_launch_agent_registration_is_not_operational(
    status_environment: tuple[automation.AutomationSettings, dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
    installed: bool,
) -> None:
    monkeypatch.setattr(automation, "_launch_agent_loaded", lambda _settings: False)
    if not installed:
        automation._launch_agent_path().unlink()

    status = _status(status_environment)

    assert status["installed"] is installed
    assert status["loaded"] is False
    assert status["publication_current"] is True
    assert status["scheduler_healthy"] is False
    assert status["ok"] is status["operational"] is False
    assert "launch_agent_not_loaded" in status["diagnostic_issues"]
    assert ("launch_agent_not_installed" in status["diagnostic_issues"]) is (not installed)


@pytest.mark.parametrize(
    "verification", [None, "invalid", (NOW + timedelta(minutes=1)).isoformat()]
)
def test_missing_invalid_or_future_verification_cannot_establish_freshness(
    status_environment: tuple[automation.AutomationSettings, dict[str, object]],
    verification: str | None,
) -> None:
    _, health = status_environment
    health["last_public_verification_at"] = verification

    status = _status(status_environment)

    assert status["scheduler_healthy"] is True
    assert status["publication_lagging"] is False
    assert status["publication_current"] is False
    assert status["ok"] is status["operational"] is False
    diagnostic = (
        "publication_verification_in_future"
        if verification == (NOW + timedelta(minutes=1)).isoformat()
        else "publication_verification_missing_or_invalid"
    )
    assert diagnostic in status["diagnostic_issues"]


def test_stale_terminal_health_cannot_keep_current_publication_flag(
    status_environment: tuple[automation.AutomationSettings, dict[str, object]],
) -> None:
    _, health = status_environment
    health["updated_at"] = (NOW - timedelta(hours=12)).isoformat()

    status = _status(status_environment)

    assert status["health_stale"] is True
    assert status["scheduler_healthy"] is False
    assert status["publication_current"] is False
    assert "health_stale" in status["diagnostic_issues"]
    assert status["ok"] is status["operational"] is False


def test_configuration_drift_keeps_current_publication_non_operational(
    status_environment: tuple[automation.AutomationSettings, dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        automation,
        "_launch_agent_configuration",
        lambda _settings, target=None: {
            "configuration_matches": False,
            "expected_plist_sha256": "a" * 64,
            "installed_plist_sha256": "b" * 64,
            "drift_keys": ["ProgramArguments"],
        },
    )

    status = _status(status_environment)

    assert status["publication_current"] is True
    assert status["scheduler_healthy"] is False
    assert "launch_agent_configuration_drift" in status["diagnostic_issues"]
    assert status["ok"] is status["operational"] is False
