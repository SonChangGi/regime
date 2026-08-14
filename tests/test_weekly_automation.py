from __future__ import annotations

from datetime import datetime, timedelta, timezone
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from regime_lab import automation
from regime_lab import cli


ROOT = Path(__file__).resolve().parents[1]
UTC = timezone.utc
LIVE_TARGET = datetime.fromisoformat("2026-08-07T20:00:00+00:00")
LIVE_PAYLOAD = (ROOT / "publication/live/regime-results.json").read_bytes()
CANDIDATE_CONTEXT = {
    "source_tree_sha256": "1" * 64,
    "series_config_sha256": "2" * 64,
    "automation_config_sha256": "3" * 64,
}


def _settings(tmp_path: Path, *, root: Path | None = None) -> automation.AutomationSettings:
    selected_root = root or tmp_path
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


def test_project_automation_config_and_launch_agent_are_secret_free() -> None:
    settings = automation.AutomationSettings.load(ROOT / "config/automation.json")
    document = automation.launch_agent_document(settings)
    rendered = json.dumps(document)

    assert settings.profile == "standard"
    assert settings.payload.is_relative_to(settings.state_directory)
    assert settings.artifacts.is_relative_to(settings.state_directory)
    assert settings.payload != ROOT / "web/data/regime-results.json"
    assert settings.artifacts != ROOT / "artifacts/latest"
    assert settings.schedule_hour == 21
    assert settings.schedule_minute == 17
    assert document["RunAtLoad"] is True
    assert document["StartCalendarInterval"] == {"Hour": 21, "Minute": 17}
    assert document["Umask"] == 0o077
    assert document["ProgramArguments"][:2] == ["/usr/bin/caffeinate", "-s"]
    assert "regime_lab" in document["ProgramArguments"]
    assert "automation" in document["ProgramArguments"]
    assert "ALPHA_VANTAGE_API_KEY" not in rendered
    assert "FRED_API_KEY" not in rendered
    assert "secrets" not in rendered.lower()


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


def test_current_weak_generalization_live_payload_is_publishable() -> None:
    payload = automation.validate_automation_candidate(
        LIVE_PAYLOAD, target=LIVE_TARGET
    )
    assert payload["meta"]["status"] == "degraded"
    assert payload["model"]["holdout_diagnostic"]["status"] == "weak_generalization"


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
    manifest = json.dumps(
        {
            "payload_data_as_of": "2026-08-07T20:00:00+00:00",
            "files": {"data/regime-results.json": {"sha256": digest}},
        }
    ).encode()

    def fetch(url: str) -> bytes:
        if url.endswith("data/regime-results.json"):
            return LIVE_PAYLOAD
        if url.endswith("publication-manifest.json"):
            return manifest
        return b"<title>US Market Regime Lab</title><script src='./app.js'></script>"

    automation.verify_public_readback(
        settings, expected_payload=LIVE_PAYLOAD, fetch=fetch
    )

    with pytest.raises(automation.AutomationError, match="SHA-256"):
        automation.verify_public_readback(
            settings,
            expected_payload=LIVE_PAYLOAD + b" ",
            fetch=fetch,
        )


def test_expired_local_authorization_fails_closed(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
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
        "src/regime_lab/automation.py",
        "src/regime_lab/cli.py",
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
    with pytest.raises(automation.AutomationError, match="full batch"):
        automation._alpha_quota_preflight(settings, target=LIVE_TARGET)


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
        lambda _settings, *, expected_payload: None,
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


def test_public_readback_recovery_pushes_a_credential_independent_empty_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
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
    assert "actions/deploy-pages@v4" in workflow
