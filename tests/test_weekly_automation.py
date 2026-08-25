from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from regime_lab import automation
from regime_lab import cli


ROOT = Path(__file__).resolve().parents[1]
UTC = timezone.utc
LIVE_TARGET = datetime.fromisoformat("2026-08-07T20:00:00+00:00")
LIVE_PAYLOAD = (
    ROOT
    / "publication/baselines/v4-20260821/regime-results.json"
).read_bytes()
CANDIDATE_CONTEXT = {
    "source_tree_sha256": "1" * 64,
    "series_config_sha256": "2" * 64,
    "automation_config_sha256": "3" * 64,
}


def _settings(tmp_path: Path, *, root: Path | None = None) -> automation.AutomationSettings:
    selected_root = root or tmp_path
    if root is None:
        config_directory = selected_root / "config"
        config_directory.mkdir(parents=True, exist_ok=True)
        (config_directory / "series.json").write_text(
            json.dumps({"provider_rights_providers": []}),
            encoding="utf-8",
        )
        (config_directory / "provider_rights.json").write_text(
            json.dumps({"schema_version": 1, "providers": {}}),
            encoding="utf-8",
        )
    return automation.AutomationSettings(
        config_path=selected_root / "config/automation.json",
        root=selected_root,
        automation_id="weekly-regime-release-v1",
        schedule_hour=21,
        schedule_minute=17,
        minimum_cutoff_age=timedelta(hours=24),
        require_ac_power=True,
        profile="standard",
        database=tmp_path / "data/regime.sqlite3",
        payload=tmp_path / "web/data/regime-results.json",
        artifacts=tmp_path / "artifacts/latest",
        state_directory=tmp_path / "build/weekly-automation",
        authorization=tmp_path / "data/automation/authorization.json",
        repository="SonChangGi/regime",
        remote="origin",
        branch="main",
        workflow="pages.yml",
        public_root="https://sonchanggi.github.io/regime/",
        workflow_timeout=timedelta(minutes=1),
        public_readback_timeout=timedelta(minutes=1),
        contract="v4",
    )


def _payload_mutation(path: tuple[object, ...], value: object) -> bytes:
    payload = json.loads(LIVE_PAYLOAD)
    current: object = payload
    for part in path[:-1]:
        current = current[part]  # type: ignore[index]
    current[path[-1]] = value  # type: ignore[index]
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode()


def _remote(payload: bytes = LIVE_PAYLOAD) -> automation.RemotePublication:
    parsed = json.loads(payload)
    return automation.RemotePublication(
        head_sha="a" * 40,
        payload_bytes=payload,
        data_as_of=datetime.fromisoformat(parsed["meta"]["data_as_of"]),
    )


def test_target_cutoff_uses_eastern_friday_and_minimum_age_across_dst() -> None:
    before = datetime.fromisoformat("2026-08-07T19:59:00+00:00")
    cutoff, ready = automation.target_cutoff(
        before, minimum_age=timedelta(hours=24)
    )
    assert cutoff == datetime.fromisoformat("2026-07-31T20:00:00+00:00")
    assert ready is True


def test_status_separates_health_check_from_full_pipeline_success(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    started = datetime.fromisoformat("2026-08-24T00:00:00+00:00")
    target = datetime.fromisoformat("2026-08-21T20:00:00+00:00")

    checked = automation._status_document(
        settings,
        status="succeeded",
        stage="already_current",
        started_at=started,
        target=target,
    )
    assert checked["last_check_at"] is not None
    assert checked["last_public_verification_at"] is not None
    assert checked["last_full_success_at"] is None
    assert checked["last_success_at"] is None

    automation.write_json_atomic(settings.status_path, checked)
    full = automation._status_document(
        settings,
        status="succeeded",
        stage="public_readback_verified",
        started_at=started,
        target=target,
    )
    assert full["last_full_success_at"] is not None
    assert full["last_full_success_target"] == target.isoformat()
    assert full["last_success_at"] == full["last_full_success_at"]


def test_status_records_each_completed_pipeline_stage(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    started = datetime.fromisoformat("2026-08-24T00:00:00+00:00")
    target = datetime.fromisoformat("2026-08-21T20:00:00+00:00")
    first = automation._status_document(
        settings,
        status="running",
        stage="collect_train_audit",
        started_at=started,
        target=target,
        run_id="run-1",
    )
    automation.write_json_atomic(settings.status_path, first)

    second = automation._status_document(
        settings,
        status="running",
        stage="publish_snapshot",
        started_at=started,
        target=target,
        run_id="run-1",
    )
    automation.write_json_atomic(settings.status_path, second)
    final = automation._status_document(
        settings,
        status="succeeded",
        stage="public_readback_verified",
        started_at=started,
        target=target,
        run_id="run-1",
    )

    assert set(final["last_stage_successes"]) == {
        "collect_train_audit",
        "publish_snapshot",
        "public_readback_verified",
    }
    assert final["last_stage_successes"]["collect_train_audit"][
        "target_data_as_of"
    ] == target.isoformat()


def test_heartbeat_stage_transition_records_completed_stage(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    started = datetime.fromisoformat("2026-08-24T00:00:00+00:00")
    target = datetime.fromisoformat("2026-08-21T20:00:00+00:00")
    automation.write_json_atomic(
        settings.status_path,
        automation._status_document(
            settings,
            status="running",
            stage="collect_live_data",
            started_at=started,
            target=target,
            run_id="run-1",
        ),
    )

    automation._heartbeat_status(settings, stage="train_models")
    updated = json.loads(settings.status_path.read_text(encoding="utf-8"))

    assert updated["stage"] == "train_models"
    assert updated["last_stage_successes"]["collect_live_data"] == {
        "completed_at": updated["stage_started_at"],
        "target_data_as_of": target.isoformat(),
        "run_id": "run-1",
    }


def test_failure_status_preserves_exact_failed_stage(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    started = datetime.fromisoformat("2026-08-24T00:00:00+00:00")
    target = datetime.fromisoformat("2026-08-21T20:00:00+00:00")

    failed = automation._status_document(
        settings,
        status="failed",
        stage="failed",
        started_at=started,
        target=target,
        failed_stage="audit_candidate",
    )
    automation.write_json_atomic(settings.status_path, failed)
    blocked = automation._status_document(
        settings,
        status="blocked",
        stage="retry_blocked",
        started_at=started,
        target=target,
    )

    assert failed["failed_stage"] == "audit_candidate"
    assert blocked["failed_stage"] == "audit_candidate"

    just_after = datetime.fromisoformat("2026-08-07T20:01:00+00:00")
    cutoff, ready = automation.target_cutoff(
        just_after, minimum_age=timedelta(hours=24)
    )
    assert cutoff == datetime.fromisoformat("2026-08-07T20:00:00+00:00")
    assert ready is False

    winter = datetime.fromisoformat("2026-01-11T22:00:00+00:00")
    cutoff, ready = automation.target_cutoff(
        winter, minimum_age=timedelta(hours=24)
    )
    assert cutoff == datetime.fromisoformat("2026-01-09T21:00:00+00:00")
    assert ready is True


def test_project_automation_config_and_launch_agent_are_secret_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "checkout"
    config_path = checkout / "config/automation.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_bytes((ROOT / "config/automation.json").read_bytes())
    python = checkout / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("fixture\n", encoding="utf-8")
    monkeypatch.setattr(automation, "project_root", lambda: checkout)

    settings = automation.AutomationSettings.load(config_path)
    document = automation.launch_agent_document(settings)
    rendered = json.dumps(document)

    assert settings.profile == "standard"
    assert settings.contract == "v5"
    assert settings.payload.is_relative_to(settings.state_directory)
    assert settings.artifacts.is_relative_to(settings.state_directory)
    assert settings.payload != checkout / "web/data/regime-results.json"
    assert settings.artifacts != checkout / "artifacts/latest"
    assert settings.schedule_hour == 21
    assert settings.schedule_minute == 17
    assert document["RunAtLoad"] is True
    assert document["StartCalendarInterval"] == [
        {"Hour": 3, "Minute": 17},
        {"Hour": 9, "Minute": 17},
        {"Hour": 15, "Minute": 17},
        {"Hour": 21, "Minute": 17},
    ]
    assert document["Umask"] == 0o077
    assert document["ProcessType"] == "Standard"
    assert document["ExitTimeOut"] == 120
    assert "LowPriorityIO" not in document
    assert document["ProgramArguments"][:2] == ["/usr/bin/caffeinate", "-s"]
    assert document["ProgramArguments"][2] == str(python)
    assert "regime_lab" in document["ProgramArguments"]
    assert "automation" in document["ProgramArguments"]
    assert "ALPHA_VANTAGE_API_KEY" not in rendered
    assert "FRED_API_KEY" not in rendered
    assert "secrets" not in rendered.lower()


def test_launch_agent_document_requires_project_virtualenv(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    with pytest.raises(automation.AutomationError, match="virtualenv Python is missing"):
        automation.launch_agent_document(settings)


def test_launch_agent_installation_remains_macos_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr(automation.sys, "platform", "linux")

    with pytest.raises(automation.AutomationError, match="requires macOS"):
        automation.install_launch_agent(
            settings,
            alfred_rights_confirmed=True,
            personal_noncommercial_publication_acknowledged=True,
        )

    assert not settings.authorization.exists()
    assert not settings.install_lock_path.exists()


def test_process_lock_rejects_overlapping_run(tmp_path: Path) -> None:
    lock = tmp_path / "weekly.lock"
    with automation.automation_lock(lock):
        with pytest.raises(automation.AlreadyRunning):
            with automation.automation_lock(lock):
                pass


def test_candidate_context_rejects_nonignored_untracked_python_shadow(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    tracked = root / "tracked.txt"
    tracked.write_text("safe\n", encoding="utf-8")
    _git(["add", "tracked.txt"], root)
    _git(
        [
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "seed",
        ],
        root,
    )
    (root / "sitecustomize.py").write_text("raise RuntimeError('shadow')\n", encoding="utf-8")
    settings = _settings(tmp_path / "state", root=root)

    with pytest.raises(automation.AutomationError, match="tracked source or config"):
        automation._candidate_context(settings)


def test_candidate_context_is_content_based_not_inode_or_mtime(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    tracked = root / "tracked.txt"
    tracked.write_text("safe\n", encoding="utf-8")
    (root / "config/series.json").write_text("{}\n", encoding="utf-8")
    (root / "config/automation.json").write_text("{}\n", encoding="utf-8")
    _git(["add", "tracked.txt", "config/series.json", "config/automation.json"], root)
    _git(
        [
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "seed",
        ],
        root,
    )
    settings = _settings(tmp_path / "state", root=root)

    before = automation._candidate_context(settings)
    original = tracked.stat()
    os.utime(tracked, ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000))
    after = automation._candidate_context(settings)

    assert after == before


def test_current_weak_generalization_live_payload_is_publishable() -> None:
    payload = automation.validate_automation_candidate(
        LIVE_PAYLOAD, target=LIVE_TARGET
    )
    assert payload["meta"]["status"] == "degraded"
    assert payload["model"]["holdout_diagnostic"]["status"] == "weak_generalization"


def test_candidate_gate_accepts_current_week_holiday_thursday_alpha_period() -> None:
    payload = json.loads(LIVE_PAYLOAD)
    payload["sources"][0]["available_at"] = "2026-08-06T20:00:00+00:00"
    payload["sources"][0]["coverage"] = "2006-01-06–2026-08-06"
    raw = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")

    validated = automation.validate_automation_candidate(raw, target=LIVE_TARGET)

    assert validated["sources"][0]["coverage"].endswith("2026-08-06")


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("sources", 0, "status"), "degraded", "provider-degraded"),
        (("sources", 0, "issues"), ["stale"], "issues"),
        (("sources", 0, "available_at"), "2026-07-31T20:00:00+00:00", "due cutoff"),
        (("weekly", -1, "health", "status"), "degraded", "week health"),
        (("model", "latest_forecast_fallback"), True, "fallback"),
        (("model", "holdout_diagnostic", "status"), "provider_degraded", "allowed diagnostic"),
    ],
)
def test_candidate_gate_rejects_unpublishable_state(
    path: tuple[object, ...], value: object, message: str
) -> None:
    with pytest.raises(automation.AutomationError, match=message):
        automation.validate_automation_candidate(
            _payload_mutation(path, value), target=LIVE_TARGET
        )


def test_candidate_gate_rejects_stale_cutoff() -> None:
    with pytest.raises(automation.AutomationError, match="due cutoff"):
        automation.validate_automation_candidate(
            LIVE_PAYLOAD,
            target=datetime.fromisoformat("2026-08-14T20:00:00+00:00"),
        )


def test_public_readback_requires_exact_payload_manifest_and_consumer(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    digest = hashlib.sha256(LIVE_PAYLOAD).hexdigest()
    assets = {
        "index.html": b"<title>US Market Regime Lab</title><script src='./app.js'></script>",
        "styles.css": b"body { color: black; }\n",
        "app.js": b"console.log('regime');\n",
    }
    manifest = json.dumps(
        {
            "payload_data_as_of": "2026-08-07T20:00:00+00:00",
            "files": {
                "data/regime-results.json": {
                    "sha256": digest,
                    "bytes": len(LIVE_PAYLOAD),
                },
                **{
                    path: {
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "bytes": len(raw),
                    }
                    for path, raw in assets.items()
                },
            },
        }
    ).encode()

    def fetch(url: str) -> bytes:
        if url.endswith("data/regime-results.json"):
            return LIVE_PAYLOAD
        if url.endswith("publication-manifest.json"):
            return manifest
        for path, raw in assets.items():
            if url.endswith(path):
                return raw
        raise AssertionError(url)

    automation.verify_public_readback(
        settings,
        expected_payload=LIVE_PAYLOAD,
        expected_assets=assets,
        fetch=fetch,
    )

    with pytest.raises(automation.AutomationError, match="SHA-256"):
        automation.verify_public_readback(
            settings,
            expected_payload=LIVE_PAYLOAD + b" ",
            expected_assets=assets,
            fetch=fetch,
        )

    def tampered_fetch(url: str) -> bytes:
        if url.endswith("app.js"):
            return assets["app.js"] + b" "
        return fetch(url)

    with pytest.raises(automation.AutomationError, match="app.js"):
        automation.verify_public_readback(
            settings,
            expected_payload=LIVE_PAYLOAD,
            expected_assets=assets,
            fetch=tampered_fetch,
        )

    old_assets = {**assets, "app.js": b"console.log('old release');\n"}
    old_manifest = json.dumps(
        {
            "payload_data_as_of": "2026-08-07T20:00:00+00:00",
            "files": {
                "data/regime-results.json": {
                    "sha256": digest,
                    "bytes": len(LIVE_PAYLOAD),
                },
                **{
                    path: {
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "bytes": len(raw),
                    }
                    for path, raw in old_assets.items()
                },
            },
        }
    ).encode()

    def stale_but_self_consistent(url: str) -> bytes:
        if url.endswith("data/regime-results.json"):
            return LIVE_PAYLOAD
        if url.endswith("publication-manifest.json"):
            return old_manifest
        for path, raw in old_assets.items():
            if url.endswith(path):
                return raw
        raise AssertionError(url)

    with pytest.raises(automation.AutomationError, match="expected checkout"):
        automation.verify_public_readback(
            settings,
            expected_payload=LIVE_PAYLOAD,
            expected_assets=assets,
            fetch=stale_but_self_consistent,
        )


def test_public_readback_rejects_invalid_v5_pair_despite_matching_manifest(
    tmp_path: Path,
) -> None:
    settings = replace(_settings(tmp_path), contract="v5")
    payload = (
        json.dumps(
            {
                "meta": {
                    "result_version": "weekly-regime-result-v5",
                    "data_as_of": "2026-08-21T20:00:00+00:00",
                }
            }
        )
        + "\n"
    ).encode()
    invalid_comparison = b'{"invalid": true}\n'
    manifest = json.dumps(
        {
            "payload_data_as_of": "2026-08-21T20:00:00+00:00",
            "files": {
                automation.PUBLIC_PAYLOAD_PATH: {
                    "sha256": hashlib.sha256(payload).hexdigest()
                },
                automation.PUBLIC_COMPARISON_PATH: {
                    "sha256": hashlib.sha256(invalid_comparison).hexdigest()
                },
            },
        }
    ).encode()

    def fetch(url: str) -> bytes:
        if url.endswith(automation.PUBLIC_PAYLOAD_PATH):
            return payload
        if url.endswith(automation.PUBLIC_MANIFEST_PATH):
            return manifest
        if url.endswith(automation.PUBLIC_COMPARISON_PATH):
            return invalid_comparison
        return b"<title>US Market Regime Lab</title><script src='./app.js'></script>"

    with pytest.raises(
        automation.AutomationError,
        match="expected publication comparison contract failed",
    ):
        automation.verify_public_readback(
            settings,
            expected_payload=payload,
            expected_comparison=invalid_comparison,
            fetch=fetch,
        )


def test_public_readback_rejects_missing_v5_comparison_before_fetch(
    tmp_path: Path,
) -> None:
    settings = replace(_settings(tmp_path), contract="v5")
    payload = b'{"meta":{"result_version":"weekly-regime-result-v5"}}\n'

    with pytest.raises(
        automation.AutomationError,
        match="V5 public readback requires an expected comparison",
    ):
        automation.verify_public_readback(
            settings,
            expected_payload=payload,
            fetch=lambda _url: pytest.fail("public fetch must not start"),
        )


def test_git_preflight_rejects_invalid_remote_v5_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = replace(_settings(tmp_path), contract="v5")
    payload = (
        json.dumps(
            {
                "meta": {
                    "result_version": "weekly-regime-result-v5",
                    "data_as_of": "2026-08-21T20:00:00+00:00",
                }
            }
        )
        + "\n"
    ).encode()
    invalid_comparison = b'{"invalid": true}\n'

    def run(args, **_kwargs) -> bytes:
        command = tuple(args)
        if command == ("git", "branch", "--show-current"):
            return b"main\n"
        if command == ("git", "status", "--porcelain", "--untracked-files=all"):
            return b""
        if command == ("git", "remote", "get-url", "origin"):
            return b"git@github.com:SonChangGi/regime.git\n"
        if command == ("git", "fetch", "--quiet", "origin", "main"):
            return b""
        if command == (
            "git",
            "diff",
            "--name-only",
            "HEAD",
            "origin/main",
            "--",
        ):
            return b""
        if command == ("git", "rev-parse", "origin/main"):
            return b"a" * 40 + b"\n"
        if command == (
            "git",
            "show",
            f"origin/main:{automation.PUBLICATION_PATH}",
        ):
            return payload
        if command == (
            "git",
            "show",
            f"origin/main:{automation.PUBLICATION_COMPARISON_PATH}",
        ):
            return invalid_comparison
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(automation, "_run", run)
    monkeypatch.setattr(
        automation,
        "validate_automation_candidate",
        lambda *_args, **_kwargs: {
            "meta": {"result_version": "weekly-regime-result-v5"}
        },
    )

    with pytest.raises(
        automation.AutomationError,
        match="remote publication comparison contract failed",
    ):
        automation._git_preflight(settings)


def test_installed_preflight_validates_v5_pair_outside_checkout(
    tmp_path: Path,
) -> None:
    payload = ROOT / automation.PUBLICATION_PATH
    comparison = ROOT / automation.PUBLICATION_COMPARISON_PATH
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text(
        f"""#!{sys.executable}
from pathlib import Path
import sys

args = sys.argv[1:]
if args == ["branch", "--show-current"]:
    sys.stdout.write("main\\n")
elif args == ["status", "--porcelain", "--untracked-files=all"]:
    pass
elif args == ["remote", "get-url", "origin"]:
    sys.stdout.write("https://github.com/SonChangGi/regime.git\\n")
elif args == ["fetch", "--quiet", "origin", "main"]:
    pass
elif args == ["diff", "--name-only", "HEAD", "origin/main", "--"]:
    pass
elif args == ["rev-parse", "origin/main"]:
    sys.stdout.write("{'a' * 40}\\n")
elif args == ["show", "origin/main:{automation.PUBLICATION_PATH}"]:
    sys.stdout.buffer.write(Path({str(payload)!r}).read_bytes())
elif args == ["show", "origin/main:{automation.PUBLICATION_COMPARISON_PATH}"]:
    sys.stdout.buffer.write(Path({str(comparison)!r}).read_bytes())
else:
    raise SystemExit(f"unexpected fake git command: {{args!r}}")
""",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    entrypoint = Path(sys.executable).with_name("regime-lab")
    assert entrypoint.is_file()
    database = tmp_path / "regime.sqlite3"
    with sqlite3.connect(database):
        pass
    authorization = tmp_path / "authorization.json"
    reviewed_at = datetime.now(UTC)
    authorization.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "automation_id": "weekly-regime-release-v1",
                "confirmed": True,
                "scopes": [
                    "alfred_local_storage_ml",
                    "personal_noncommercial_derived_publication",
                ],
                "reviewed_at": reviewed_at.isoformat(),
                "review_after": (reviewed_at + timedelta(days=180)).isoformat(),
                "contains_credentials": False,
            }
        ),
        encoding="utf-8",
    )
    config = json.loads((ROOT / "config/automation.json").read_text(encoding="utf-8"))
    config["build"].update(
        {
            "database": str(database),
            "payload": str(tmp_path / "regime-results.json"),
            "artifacts": str(tmp_path / "artifacts"),
            "state_directory": str(tmp_path / "automation-state"),
            "authorization": str(authorization),
        }
    )
    automation_config = tmp_path / "automation.json"
    automation_config.write_text(json.dumps(config), encoding="utf-8")
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '')}"

    completed = subprocess.run(
        [
            str(entrypoint),
            "automation",
            "preflight",
            "--config",
            str(automation_config),
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["ok"] is True
    assert result["authorization_ok"] is True
    assert result["database_ok"] is True
    assert result["remote_head_sha"] == "a" * 40


def test_expired_local_authorization_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr(
        automation,
        "_validate_provider_rights_policy",
        lambda *_args, **_kwargs: None,
    )
    automation._write_local_authorization(
        settings,
        alfred_rights_confirmed=True,
        personal_noncommercial_publication_acknowledged=True,
        now=datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
    )
    with pytest.raises(automation.AutomationError, match="renewal"):
        automation._validate_local_authorization(
            settings, now=datetime.fromisoformat("2026-08-01T00:00:00+00:00")
        )
    assert settings.authorization.stat().st_mode & 0o777 == 0o600


def test_current_automation_accepts_provider_scope_before_local_authorization(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path, root=ROOT)
    checked_at = datetime.fromisoformat("2026-08-25T00:00:00+00:00")
    automation._validate_provider_rights_policy(settings, now=checked_at)
    with pytest.raises(automation.AutomationError, match="authorization is missing"):
        automation._validate_local_authorization(
            settings,
            now=checked_at,
        )


def test_automation_provider_gate_rejects_missing_series_config(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    (settings.root / "config/series.json").unlink()

    with pytest.raises(automation.AutomationError, match="series config is missing"):
        automation._validate_provider_rights_policy(
            settings,
            now=datetime.fromisoformat("2026-08-24T06:00:00+00:00"),
        )


def test_missing_local_authorization_keeps_installed_status_non_operational(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path, root=ROOT)
    health = {
        "schema_version": automation.HEALTH_SCHEMA_VERSION,
        "status": "succeeded",
        "updated_at": datetime.now(UTC).isoformat(),
        "consecutive_failures": 0,
    }
    settings.status_path.parent.mkdir(parents=True, exist_ok=True)
    settings.status_path.write_text(json.dumps(health), encoding="utf-8")
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
    fake_plist = tmp_path / "regime.plist"
    fake_plist.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(automation, "_launch_agent_path", lambda: fake_plist)

    status = automation.launch_agent_status(settings)

    assert status["provider_rights_ok"] is True
    assert status["provider_rights_error"] is None
    assert status["authorization_ok"] is False
    assert "missing" in str(status["authorization_error"])
    assert status["operational"] is False


def test_local_authorization_requires_both_explicit_acknowledgements(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    with pytest.raises(automation.AutomationError, match="ALFRED"):
        automation._write_local_authorization(
            settings,
            alfred_rights_confirmed=False,
            personal_noncommercial_publication_acknowledged=True,
        )
    with pytest.raises(automation.AutomationError, match="noncommercial"):
        automation._write_local_authorization(
            settings,
            alfred_rights_confirmed=True,
            personal_noncommercial_publication_acknowledged=False,
        )
    assert not settings.authorization.exists()


def test_install_requires_tracked_runtime_and_clean_remote_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    for relative in (
        "config/automation.json",
        "config/series.json",
        "src/regime_lab/automation.py",
        "src/regime_lab/cli.py",
        "src/regime_lab/collection.py",
        ".github/workflows/pages.yml",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("tracked\n", encoding="utf-8")
    _git(["add", "."], root)
    _git(
        [
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "seed",
        ],
        root,
    )
    settings = _settings(tmp_path / "state", root=root)
    calls = {"preflight": 0}
    monkeypatch.setattr(
        automation,
        "_git_preflight",
        lambda _settings: calls.__setitem__("preflight", calls["preflight"] + 1),
    )

    automation._ensure_installable_checkout(settings)
    assert calls["preflight"] == 1

    _git(["rm", "--cached", "config/automation.json"], root)
    with pytest.raises(automation.AutomationError, match="git failed"):
        automation._ensure_installable_checkout(settings)


def test_alpha_quota_preflight_blocks_before_full_batch(tmp_path: Path) -> None:
    settings = _settings(tmp_path, root=ROOT)
    budget = automation.DailyRequestBudget(
        limit=25, database_path=settings.database
    )
    assert budget.reserve(3) is not None
    before = settings.database.read_bytes()
    before_mtime = settings.database.stat().st_mtime_ns
    with pytest.raises(automation.AutomationError, match="full batch"):
        automation._alpha_quota_preflight(settings, target=LIVE_TARGET)
    assert settings.database.read_bytes() == before
    assert settings.database.stat().st_mtime_ns == before_mtime


def test_already_current_run_never_builds_or_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    automation._write_local_authorization(
        settings,
        alfred_rights_confirmed=True,
        personal_noncommercial_publication_acknowledged=True,
        now=LIVE_TARGET,
    )
    monkeypatch.setattr(automation, "_ensure_ac_power", lambda _settings: None)
    monkeypatch.setattr(automation, "_git_preflight", lambda _settings: _remote())
    monkeypatch.setattr(
        automation,
        "verify_public_readback",
        lambda _settings, *, expected_payload, expected_comparison=None: None,
    )
    monkeypatch.setattr(
        automation,
        "_build_candidate",
        lambda *_args, **_kwargs: pytest.fail("build must not run"),
    )
    monkeypatch.setattr(
        automation,
        "_publish_candidate",
        lambda *_args, **_kwargs: pytest.fail("publish must not run"),
    )

    result = automation.run_weekly_release(
        settings,
        now=datetime.fromisoformat("2026-08-10T00:00:00+00:00"),
    )

    assert result["status"] == "succeeded"
    assert result["stage"] == "already_current"
    persisted = json.loads(settings.status_path.read_text(encoding="utf-8"))
    assert persisted["commit_sha"] == "a" * 40


def test_push_failure_retry_reuses_cached_candidate_without_provider_or_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    automation._write_local_authorization(
        settings,
        alfred_rights_confirmed=True,
        personal_noncommercial_publication_acknowledged=True,
        now=LIVE_TARGET,
    )
    older = automation.RemotePublication(
        head_sha="b" * 40,
        payload_bytes=b'{"older": true}\n',
        data_as_of=datetime.fromisoformat("2026-07-31T20:00:00+00:00"),
    )
    calls = {"quota": 0, "build": 0, "publish": 0}
    monkeypatch.setattr(automation, "_ensure_ac_power", lambda _settings: None)
    monkeypatch.setattr(automation, "_git_preflight", lambda _settings: older)
    monkeypatch.setattr(
        automation, "_candidate_context", lambda _settings: CANDIDATE_CONTEXT
    )
    monkeypatch.setattr(automation, "_verify_candidate_package", lambda *_args: None)

    def quota(*_args, **_kwargs) -> None:
        calls["quota"] += 1

    def build(*_args, **_kwargs) -> bytes:
        calls["build"] += 1
        automation._cache_candidate(
            settings,
            LIVE_PAYLOAD,
            target=LIVE_TARGET,
            context=CANDIDATE_CONTEXT,
        )
        return LIVE_PAYLOAD

    def fail_publish(*_args, **_kwargs) -> str:
        calls["publish"] += 1
        raise automation.AutomationError("push rejected")

    monkeypatch.setattr(automation, "_alpha_quota_preflight", quota)
    monkeypatch.setattr(automation, "_build_candidate", build)
    monkeypatch.setattr(automation, "_publish_candidate", fail_publish)

    with pytest.raises(automation.AutomationError, match="push rejected"):
        automation.run_weekly_release(
            settings,
            now=datetime.fromisoformat("2026-08-10T00:00:00+00:00"),
        )
    assert calls == {"quota": 1, "build": 1, "publish": 1}

    monkeypatch.setattr(
        automation,
        "_publish_candidate",
        lambda *_args, **_kwargs: calls.__setitem__(
            "publish", calls["publish"] + 1
        )
        or "c" * 40,
    )
    monkeypatch.setattr(
        automation,
        "_wait_for_public_readback",
        lambda *_args, **_kwargs: None,
    )

    result = automation.run_weekly_release(
        settings,
        now=datetime.fromisoformat("2026-08-10T00:00:00+00:00"),
    )

    assert result["status"] == "succeeded"
    assert calls == {"quota": 1, "build": 1, "publish": 2}


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_publication_uses_isolated_checkout_and_commits_only_snapshot(
    tmp_path: Path,
) -> None:
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(bare)], check=True, capture_output=True)
    seed = tmp_path / "seed"
    subprocess.run(["git", "clone", str(bare), str(seed)], check=True, capture_output=True)
    _git(["checkout", "-b", "main"], seed)
    target = seed / automation.PUBLICATION_PATH
    target.parent.mkdir(parents=True)
    target.write_text("old\n", encoding="utf-8")
    (seed / "README.md").write_text("unchanged\n", encoding="utf-8")
    _git(["add", "."], seed)
    _git(
        [
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "seed",
        ],
        seed,
    )
    _git(["push", "-u", "origin", "main"], seed)
    subprocess.run(
        ["git", "--git-dir", str(bare), "symbolic-ref", "HEAD", "refs/heads/main"],
        check=True,
    )

    workspace = tmp_path / "workspace"
    subprocess.run(["git", "clone", str(bare), str(workspace)], check=True, capture_output=True)
    settings = _settings(tmp_path / "state", root=workspace)
    before_head = _git(["rev-parse", "HEAD"], workspace)
    candidate = b'{"derived": true}\n'

    commit_sha = automation._publish_candidate(
        settings,
        candidate=candidate,
        target=LIVE_TARGET,
        expected_head_sha=before_head,
    )

    assert _git(["rev-parse", "HEAD"], workspace) == before_head
    assert _git(["status", "--porcelain"], workspace) == ""
    changed = _git(
        ["--git-dir", str(bare), "diff-tree", "--no-commit-id", "--name-only", "-r", commit_sha],
        tmp_path,
    )
    assert changed == automation.PUBLICATION_PATH
    published = subprocess.run(
        ["git", "--git-dir", str(bare), "show", f"{commit_sha}:{automation.PUBLICATION_PATH}"],
        check=True,
        capture_output=True,
    ).stdout
    assert published == candidate


def test_pages_recovery_uses_an_empty_commit_when_snapshot_is_unchanged(
    tmp_path: Path,
) -> None:
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(bare)], check=True, capture_output=True)
    seed = tmp_path / "seed"
    subprocess.run(["git", "clone", str(bare), str(seed)], check=True, capture_output=True)
    _git(["checkout", "-b", "main"], seed)
    target = seed / automation.PUBLICATION_PATH
    target.parent.mkdir(parents=True)
    target.write_bytes(LIVE_PAYLOAD)
    _git(["add", "."], seed)
    _git(
        [
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "seed",
        ],
        seed,
    )
    before = _git(["rev-parse", "HEAD"], seed)
    _git(["push", "-u", "origin", "main"], seed)
    subprocess.run(
        ["git", "--git-dir", str(bare), "symbolic-ref", "HEAD", "refs/heads/main"],
        check=True,
    )
    settings = _settings(tmp_path / "state", root=seed)

    commit_sha = automation._publish_candidate(
        settings,
        candidate=LIVE_PAYLOAD,
        target=LIVE_TARGET,
        expected_head_sha=before,
        force_pages_rebuild=True,
    )

    assert commit_sha != before
    assert _git(
        ["--git-dir", str(bare), "diff-tree", "--no-commit-id", "--name-only", "-r", commit_sha],
        tmp_path,
    ) == ""
    assert subprocess.run(
        ["git", "--git-dir", str(bare), "show", f"{commit_sha}:{automation.PUBLICATION_PATH}"],
        check=True,
        capture_output=True,
    ).stdout == LIVE_PAYLOAD


def test_publication_refuses_remote_head_race_before_writing(
    tmp_path: Path,
) -> None:
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(bare)], check=True, capture_output=True)
    seed = tmp_path / "seed"
    subprocess.run(["git", "clone", str(bare), str(seed)], check=True, capture_output=True)
    _git(["checkout", "-b", "main"], seed)
    target = seed / automation.PUBLICATION_PATH
    target.parent.mkdir(parents=True)
    target.write_text("old\n", encoding="utf-8")
    _git(["add", "."], seed)
    _git(
        [
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "seed",
        ],
        seed,
    )
    _git(["push", "-u", "origin", "main"], seed)
    subprocess.run(
        ["git", "--git-dir", str(bare), "symbolic-ref", "HEAD", "refs/heads/main"],
        check=True,
    )
    settings = _settings(tmp_path / "state", root=seed)

    with pytest.raises(automation.AutomationError, match="changed after preflight"):
        automation._publish_candidate(
            settings,
            candidate=b"new\n",
            target=LIVE_TARGET,
            expected_head_sha="f" * 40,
        )
    assert _git(["status", "--porcelain"], seed) == ""
    assert target.read_text(encoding="utf-8") == "old\n"


def test_cached_candidate_is_bound_to_source_configs_and_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr(automation, "_verify_candidate_package", lambda *_args: None)
    automation._cache_candidate(
        settings,
        LIVE_PAYLOAD,
        target=LIVE_TARGET,
        context=CANDIDATE_CONTEXT,
    )

    assert (
        automation._load_cached_candidate(
            settings, target=LIVE_TARGET, context=CANDIDATE_CONTEXT
        )
        == LIVE_PAYLOAD
    )
    changed = {**CANDIDATE_CONTEXT, "series_config_sha256": "9" * 64}
    assert (
        automation._load_cached_candidate(
            settings, target=LIVE_TARGET, context=changed
        )
        is None
    )


def test_build_refuses_source_or_config_change_before_audit_and_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    changed = {**CANDIDATE_CONTEXT, "workspace_state_sha256": "4" * 64}
    monkeypatch.setattr(automation, "_run", lambda *_args, **_kwargs: b"")
    monkeypatch.setattr(automation, "_candidate_context", lambda _settings: changed)

    with pytest.raises(automation.AutomationError, match="changed during live build"):
        automation._build_candidate(
            settings,
            target=LIVE_TARGET,
            context=CANDIDATE_CONTEXT,
        )


def test_build_refuses_change_during_audit_before_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    settings.payload.parent.mkdir(parents=True)
    settings.payload.write_bytes(LIVE_PAYLOAD)
    changed = {**CANDIDATE_CONTEXT, "workspace_state_sha256": "4" * 64}
    contexts = iter((CANDIDATE_CONTEXT, changed))
    monkeypatch.setattr(automation, "_run", lambda *_args, **_kwargs: b"")
    monkeypatch.setattr(automation, "_sqlite_quick_check", lambda _path: None)
    monkeypatch.setattr(automation, "_verify_candidate_package", lambda *_args: None)
    monkeypatch.setattr(automation, "_candidate_context", lambda _settings: next(contexts))
    monkeypatch.setattr(
        automation,
        "_cache_candidate",
        lambda *_args, **_kwargs: pytest.fail("changed candidate must not be cached"),
    )

    with pytest.raises(automation.AutomationError, match="changed during live audit"):
        automation._build_candidate(
            settings,
            target=LIVE_TARGET,
            context=CANDIDATE_CONTEXT,
        )


def test_launch_agent_bootstrap_happens_after_weekly_lock_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr(automation.sys, "platform", "darwin")
    plist = tmp_path / "LaunchAgents" / "regime.plist"
    observed = {"bootstrap_with_weekly_lock_free": False}
    loaded = {"value": False}
    monkeypatch.setattr(automation, "_ensure_installable_checkout", lambda _settings: None)
    monkeypatch.setattr(
        automation,
        "launch_agent_document",
        lambda _settings: {"Label": automation.AUTOMATION_LABEL},
    )
    monkeypatch.setattr(automation, "_launch_agent_path", lambda: plist)
    monkeypatch.setattr(
        automation, "_launch_agent_loaded", lambda _settings: loaded["value"]
    )

    def command(args, **_kwargs):
        if "bootstrap" in args:
            with automation.automation_lock(settings.lock_path):
                observed["bootstrap_with_weekly_lock_free"] = True
            loaded["value"] = True
        return b""

    monkeypatch.setattr(automation, "_run", command)
    automation.install_launch_agent(
        settings,
        alfred_rights_confirmed=True,
        personal_noncommercial_publication_acknowledged=True,
    )

    assert observed["bootstrap_with_weekly_lock_free"] is True


def test_uninstall_keeps_plist_and_fails_when_loaded_service_cannot_bootout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    plist = tmp_path / "LaunchAgents" / "regime.plist"
    plist.parent.mkdir(parents=True)
    plist.write_text("loaded\n", encoding="utf-8")
    monkeypatch.setattr(automation, "_launch_agent_path", lambda: plist)
    monkeypatch.setattr(automation, "_launch_agent_loaded", lambda _settings: True)
    monkeypatch.setattr(
        automation,
        "_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            automation.AutomationError("bootout failed")
        ),
    )

    with pytest.raises(automation.AutomationError, match="bootout failed"):
        automation.uninstall_launch_agent(settings)
    assert plist.is_file()


def test_install_or_uninstall_lock_collision_is_not_reported_as_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _settings(tmp_path)
    args = argparse.Namespace(
        action="uninstall",
        config=settings.config_path,
        alfred_rights_confirmed=False,
        acknowledge_personal_noncommercial_publication=False,
    )
    monkeypatch.setattr(
        automation.AutomationSettings,
        "load",
        classmethod(lambda _cls, _path: settings),
    )
    monkeypatch.setattr(
        automation,
        "uninstall_launch_agent",
        lambda _settings: (_ for _ in ()).throw(
            automation.AlreadyRunning("weekly run active")
        ),
    )

    assert automation.command_automation(args) == 1
    error = json.loads(capsys.readouterr().err)
    assert error["ok"] is False
    assert error["status"] == "failed"


def test_force_retry_cli_is_explicit_run_only_and_forwarded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _settings(tmp_path)
    parser = cli.build_parser()
    args = parser.parse_args(["automation", "run", "--force-retry"])
    assert args.force_retry is True
    seen: list[bool] = []
    monkeypatch.setattr(
        automation.AutomationSettings,
        "load",
        classmethod(lambda _cls, _path: settings),
    )
    monkeypatch.setattr(
        automation,
        "run_weekly_release",
        lambda _settings, *, force_transient_retry=False,
        force_blocked_recovery=False: (
            seen.append(force_transient_retry or force_blocked_recovery)
            or {"ok": True, "status": "succeeded"}
        ),
    )

    assert automation.command_automation(args) == 0
    assert seen == [True]
    capsys.readouterr()

    invalid = parser.parse_args(["automation", "status", "--force-retry"])
    assert automation.command_automation(invalid) == 1
    error = json.loads(capsys.readouterr().err)
    assert "valid only" in error["error"]

    blocked_args = parser.parse_args(
        ["automation", "run", "--force-blocked-recovery"]
    )
    assert automation.command_automation(blocked_args) == 0
    assert seen == [True, True]
    capsys.readouterr()

    invalid_blocked = parser.parse_args(
        ["automation", "status", "--force-blocked-recovery"]
    )
    assert automation.command_automation(invalid_blocked) == 1
    blocked_error = json.loads(capsys.readouterr().err)
    assert "valid only" in blocked_error["error"]


def test_public_readback_recovery_pushes_a_credential_independent_empty_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    automation._write_local_authorization(
        settings,
        alfred_rights_confirmed=True,
        personal_noncommercial_publication_acknowledged=True,
        now=datetime.fromisoformat("2026-08-01T00:00:00+00:00"),
    )
    calls: dict[str, object] = {"publish": 0, "wait": 0}
    monkeypatch.setattr(automation, "_git_preflight", lambda _settings: _remote())
    monkeypatch.setattr(
        automation,
        "verify_public_readback",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            automation.AutomationError("stale public page")
        ),
    )

    def publish(*_args, **kwargs) -> str:
        calls["publish"] = int(calls["publish"]) + 1
        calls["force"] = kwargs.get("force_pages_rebuild")
        calls["expected"] = kwargs.get("expected_head_sha")
        return "c" * 40

    monkeypatch.setattr(automation, "_publish_candidate", publish)
    monkeypatch.setattr(
        automation,
        "_wait_for_public_readback",
        lambda *_args, **_kwargs: calls.__setitem__("wait", int(calls["wait"]) + 1),
    )
    result = automation.run_weekly_release(
        settings,
        now=datetime.fromisoformat("2026-08-10T00:00:00+00:00"),
    )

    assert result["status"] == "succeeded"
    assert result["commit_sha"] == "c" * 40
    assert calls == {
        "publish": 1,
        "wait": 1,
        "force": True,
        "expected": "a" * 40,
    }


def test_remote_publication_ahead_of_due_cutoff_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    ahead = automation.RemotePublication(
        head_sha="d" * 40,
        payload_bytes=LIVE_PAYLOAD,
        data_as_of=datetime.fromisoformat("2026-08-14T20:00:00+00:00"),
    )
    monkeypatch.setattr(automation, "_git_preflight", lambda _settings: ahead)

    with pytest.raises(automation.AutomationError, match="ahead of the due cutoff"):
        automation.run_weekly_release(
            settings,
            now=datetime.fromisoformat("2026-08-10T00:00:00+00:00"),
        )


def test_manual_and_scheduled_live_build_share_database_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "regime.sqlite3"
    lock = database.with_name(f"{database.name}.live-build.lock")
    args = argparse.Namespace(
        alfred_rights_confirmed=True,
        profile="standard",
        config=tmp_path / "series.json",
        database=database,
        output=tmp_path / "result.json",
        artifacts=tmp_path / "artifacts",
        from_env=True,
    )
    monkeypatch.setattr(cli, "load_config", lambda _path: {})
    monkeypatch.setattr(cli, "_mutable_path", lambda value, **_kwargs: Path(value))
    monkeypatch.setattr(
        cli,
        "collect_live_data",
        lambda *_args, **_kwargs: pytest.fail("provider must not be called"),
    )

    with automation.automation_lock(lock):
        with pytest.raises(SystemExit, match="another build owns"):
            cli.command_build(args)


def test_pages_workflow_remains_provider_free_and_unscheduled() -> None:
    workflow = (ROOT / ".github/workflows/pages.yml").read_text(encoding="utf-8")
    assert "schedule:" not in workflow
    assert "regime-lab build" not in workflow
    assert "ALPHA_VANTAGE" not in workflow
    assert "FRED_API_KEY" not in workflow
    assert "secrets." not in workflow
    assert "publication/live/regime-results.json" in workflow
    assert "Refuse a stale main deployment" in workflow
    assert 'git/ref/heads/main' in workflow
    assert 'test "$current_sha" = "$EXPECTED_SHA"' in workflow
    assert (
        "actions/deploy-pages@cd2ce8fcbc39b97be8ca5fce6e763baed58fa128 # v5.0.0"
        in workflow
    )
