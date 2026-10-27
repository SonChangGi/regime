from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import plistlib

import pytest

from regime_lab import automation
from regime_lab import path_safety


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> automation.AutomationSettings:
    source, state = tmp_path / "scheduled-source", tmp_path / "existing-private-state"
    source.mkdir()
    state.mkdir()
    config = source / "config/automation.json"
    config.parent.mkdir()
    config.write_bytes((ROOT / "config/automation.json").read_bytes())
    launcher = source / "scripts/run_scheduled_automation.py"
    launcher.parent.mkdir()
    launcher.write_text("# isolated test launcher\n", encoding="utf-8")
    base_python = tmp_path / "base-python"
    base_python.write_text("preserved interpreter fixture\n", encoding="utf-8")
    python = state / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(base_python)
    monkeypatch.setattr(automation, "project_root", lambda: source)
    return automation.AutomationSettings.load(state_root=state)


def test_state_root_routes_mutable_files_without_redirecting_source_config(
    runtime: automation.AutomationSettings,
) -> None:
    source, state = runtime.root, runtime.storage_root

    assert source != state
    assert runtime.config_path == source / "config/automation.json"
    assert runtime.database == state / "data/regime.sqlite3"
    assert runtime.state_directory == state / "build/weekly-automation"
    assert runtime.payload == runtime.state_directory / "generation-v5/regime-results.json"
    assert runtime.artifacts == runtime.state_directory / "generation-v5/artifacts"
    assert runtime.authorization == state / "data/automation/authorization.json"
    assert runtime.status_path == runtime.state_directory / "automation-health.json"
    assert runtime.lock_path == runtime.state_directory / "weekly-release.lock"
    assert runtime.runtime_python == state / ".venv/bin/python"
    assert not (source / "data").exists()
    assert not (source / "build").exists()


def test_omitted_state_root_preserves_existing_single_checkout_behavior(
    runtime: automation.AutomationSettings,
) -> None:
    settings = automation.AutomationSettings.load()

    assert settings.state_root is None
    assert settings.storage_root == runtime.root
    assert settings.database == runtime.root / "data/regime.sqlite3"
    assert settings.runtime_python == runtime.root / ".venv/bin/python"


def test_state_root_must_exist_before_any_mutable_path_is_created(
    runtime: automation.AutomationSettings,
) -> None:
    missing = runtime.storage_root.parent / "missing-state"

    with pytest.raises(automation.AutomationError, match="existing directory"):
        automation.AutomationSettings.load(state_root=missing)

    assert not missing.exists()


@pytest.mark.parametrize("escape", ["parent", "source", "symlink"])
def test_mutable_paths_cannot_escape_or_alias_the_selected_state_root(
    runtime: automation.AutomationSettings,
    monkeypatch: pytest.MonkeyPatch,
    escape: str,
) -> None:
    # Keep the independent system-temp allowance distinct from the two roots.
    allowed_temp = runtime.storage_root.parent / "allowed-system-temp"
    allowed_temp.mkdir()
    monkeypatch.setattr(path_safety.tempfile, "gettempdir", lambda: str(allowed_temp))
    document = json.loads(runtime.config_path.read_text(encoding="utf-8"))
    if escape == "parent":
        document["build"]["database"] = "../outside.sqlite3"
    elif escape == "source":
        document["build"]["database"] = str(runtime.root / "source.sqlite3")
    else:
        actual_data = runtime.storage_root / "actual-data"
        actual_data.mkdir()
        (runtime.storage_root / "data").symlink_to(actual_data, target_is_directory=True)
    runtime.config_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(path_safety.UnsafeMutablePath):
        automation.AutomationSettings.load(state_root=runtime.storage_root)

    assert not (runtime.root / "source.sqlite3").exists()
    assert not (runtime.storage_root.parent / "outside.sqlite3").exists()


def test_isolated_launch_agent_uses_new_source_and_preserves_venv_entry_point(
    runtime: automation.AutomationSettings,
) -> None:
    document = automation.launch_agent_document(runtime)
    arguments = document["ProgramArguments"]

    assert arguments == [
        "/usr/bin/caffeinate", "-s", str(runtime.runtime_python),
        str(runtime.root / "scripts/run_scheduled_automation.py"),
        "--source-root", str(runtime.root),
        "--state-root", str(runtime.storage_root),
        "--python", str(runtime.runtime_python),
    ]
    assert runtime.runtime_python.is_symlink()
    assert arguments[2] != str(runtime.runtime_python.resolve())
    assert document["WorkingDirectory"] == str(runtime.root)
    assert document["EnvironmentVariables"]["PYTHONPATH"] == str(runtime.root / "src")
    assert document["EnvironmentVariables"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert document["StartCalendarInterval"] == [
        {"Hour": hour, "Minute": 17} for hour in (3, 9, 15, 21)
    ]
    assert document["StandardOutPath"] == str(runtime.state_directory / "launchd.stdout.log")
    assert document["StandardErrorPath"] == str(runtime.state_directory / "launchd.stderr.log")
    assert document["RunAtLoad"] is True
    assert not (runtime.root / ".venv").exists()
    assert not (runtime.root / "build").exists()


@dataclass
class RestoreEnvironment:
    settings: automation.AutomationSettings
    plist: Path
    marker: Path
    original_plist: bytes
    authorization: bytes
    expected_document: dict[str, object]
    commands: list[tuple[str, ...]] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    loaded: bool = False


@pytest.fixture
def restoration(
    runtime: automation.AutomationSettings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> RestoreEnvironment:
    runtime.authorization.parent.mkdir(parents=True)
    authorization = b'{ "existing_authorization": "preserve these exact bytes" }\n'
    runtime.authorization.write_bytes(authorization)
    runtime.authorization.chmod(0o600)
    document = automation.launch_agent_document(runtime)
    marker = runtime.state_directory / "runtime-checkout.json"
    marker.write_text(json.dumps({
        "schema_version": 1,
        "repository": runtime.repository,
        "branch": runtime.branch,
        "source_root": str(runtime.root),
        "state_root": str(runtime.storage_root),
        "python_executable": str(runtime.runtime_python),
    }), encoding="utf-8")
    plist = tmp_path / "LaunchAgents/regime.plist"
    plist.parent.mkdir()
    original_plist = plistlib.dumps({
        "Label": automation.AUTOMATION_LABEL,
        "WorkingDirectory": str(runtime.storage_root),
        "ProgramArguments": [str(runtime.runtime_python), "old-launcher"],
    })
    plist.write_bytes(original_plist)
    plist.chmod(0o600)
    environment = RestoreEnvironment(
        runtime, plist, marker, original_plist, authorization, document
    )
    monkeypatch.setattr(automation.sys, "platform", "darwin")
    monkeypatch.setattr(automation, "_launch_agent_path", lambda: plist)
    monkeypatch.setattr(
        automation, "_launch_agent_loaded", lambda _settings: environment.loaded
    )

    def validate(_settings: automation.AutomationSettings, *, now: object) -> None:
        assert _settings is runtime
        assert runtime.authorization.read_bytes() == authorization

    monkeypatch.setattr(automation, "_validate_local_authorization", validate)
    monkeypatch.setattr(automation, "_ensure_installable_checkout", lambda _settings: None)
    preflight_config = {"source_config_fixture": True}

    def prepare(*, source_root: Path, state_root: Path, contract: str) -> dict[str, object]:
        assert source_root == runtime.root
        assert state_root == runtime.storage_root
        assert contract == "v5"
        assert plist.read_bytes() == original_plist or environment.loaded
        environment.events.append("prepare_runtime_inputs")
        return {"ok": True, "baseline": "prepared fixture"}

    def config(path: Path) -> dict[str, object]:
        assert path == runtime.root / "config/series.json"
        environment.events.append("load_config")
        return preflight_config

    def preflight(**kwargs: object) -> dict[str, object]:
        assert kwargs == {
            "profile": runtime.profile,
            "database_path": runtime.database,
            "project_directory": runtime.root,
            "config": preflight_config,
        }
        assert environment.events[-2:] == ["prepare_runtime_inputs", "load_config"]
        environment.events.append("verify_v5_preflight")
        return {"ok": True}

    monkeypatch.setattr(automation, "prepare_runtime_inputs", prepare)
    monkeypatch.setattr(automation, "load_config", config)
    monkeypatch.setattr(automation, "verify_v5_preflight", preflight)

    def forbidden_authorization_write(*_args: object, **_kwargs: object) -> None:
        pytest.fail("restore must validate existing authorization without renewing it")

    monkeypatch.setattr(automation, "_write_local_authorization", forbidden_authorization_write)
    real_replace = os.replace

    def replace(source: Path, target: Path) -> None:
        assert target == plist
        assert source.parent == plist.parent
        assert source != plist
        restoring_previous = source.read_bytes() == original_plist
        if restoring_previous:
            assert plistlib.loads(plist.read_bytes()) == document
            with automation.automation_lock(runtime.lock_path):
                pass  # bootstrap has failed after releasing the weekly lock
        else:
            assert plist.read_bytes() == original_plist
            assert plistlib.loads(source.read_bytes()) == document
        temporary_inode = source.stat().st_ino
        if not restoring_previous:
            with pytest.raises(automation.AlreadyRunning):
                with automation.automation_lock(runtime.lock_path):
                    pytest.fail("weekly lock must remain held while replacing its plist")
        real_replace(source, target)
        assert not source.exists()
        assert target.stat().st_ino == temporary_inode
        environment.events.append("atomic_rollback" if restoring_previous else "atomic_replace")

    monkeypatch.setattr(automation.os, "replace", replace)

    def command(args: list[str], **kwargs: object) -> bytes:
        selected = tuple(str(value) for value in args)
        environment.commands.append(selected)
        assert kwargs["cwd"] == runtime.root
        if selected[:2] == ("/usr/bin/plutil", "-lint"):
            temporary = Path(selected[2])
            assert plistlib.loads(temporary.read_bytes()) == document
            assert plist.read_bytes() == original_plist
            environment.events.append("lint")
        elif selected[:2] == ("/bin/launchctl", "bootout"):
            assert environment.loaded is True
            environment.loaded = False
            environment.events.append("bootout")
        elif selected[:2] == ("/bin/launchctl", "bootstrap"):
            assert selected == (
                "/bin/launchctl", "bootstrap", f"gui/{os.getuid()}", str(plist)
            )
            # This is a real flock acquisition, after the real replacement.
            with automation.automation_lock(runtime.lock_path):
                assert plistlib.loads(plist.read_bytes()) == document
                assert runtime.authorization.read_bytes() == authorization
            environment.loaded = True
            environment.events.append("bootstrap_with_weekly_lock_free")
        else:
            pytest.fail(f"restore must not enable, kickstart, or execute other commands: {selected}")
        return b""

    monkeypatch.setattr(automation, "_run", command)
    return environment


@pytest.mark.parametrize("previously_loaded", [False, True])
def test_restore_atomically_backs_up_existing_plist_and_releases_lock_before_bootstrap(
    restoration: RestoreEnvironment,
    previously_loaded: bool,
) -> None:
    environment = restoration
    settings = environment.settings
    environment.loaded = previously_loaded
    original_auth_stat = settings.authorization.stat()

    result = automation.restore_launch_agent(settings)

    assert result["ok"] is result["restored"] is True
    assert result["configuration_matches"] is True
    assert Path(result["previous_plist"]).read_bytes() == environment.original_plist
    assert Path(result["previous_plist"]).parent == settings.state_directory
    assert plistlib.loads(environment.plist.read_bytes()) == environment.expected_document
    assert environment.plist.stat().st_mode & 0o777 == 0o600
    assert not list(environment.plist.parent.glob(f".{environment.plist.name}.*"))
    assert settings.authorization.read_bytes() == environment.authorization
    current_auth_stat = settings.authorization.stat()
    assert current_auth_stat.st_ino == original_auth_stat.st_ino
    assert current_auth_stat.st_mtime_ns == original_auth_stat.st_mtime_ns
    assert current_auth_stat.st_mode == original_auth_stat.st_mode
    expected_events = [
        "prepare_runtime_inputs", "load_config", "verify_v5_preflight", "lint"
    ] + (["bootout"] if previously_loaded else [])
    assert environment.events == expected_events + [
        "atomic_replace", "bootstrap_with_weekly_lock_free"
    ]
    assert all("enable" not in command and "kickstart" not in command
               for command in environment.commands)


def test_restore_prepares_and_validates_private_inputs_before_replacing_the_plist(
    restoration: RestoreEnvironment,
) -> None:
    environment = restoration

    result = automation.restore_launch_agent(environment.settings)

    assert environment.events.index("prepare_runtime_inputs") < environment.events.index(
        "verify_v5_preflight"
    ) < environment.events.index("atomic_replace")
    assert result["runtime_inputs"] == {"ok": True, "baseline": "prepared fixture"}
    assert environment.settings.authorization.read_bytes() == environment.authorization
    assert Path(result["previous_plist"]).read_bytes() == environment.original_plist


def test_restore_is_idempotent_when_the_matching_service_is_already_loaded(
    restoration: RestoreEnvironment,
) -> None:
    environment = restoration
    settings = environment.settings
    automation.restore_launch_agent(settings)
    saved_plist = environment.plist.read_bytes()
    saved_inode = environment.plist.stat().st_ino
    backups = list(settings.state_directory.glob("launch-agent-before-restore-*.plist"))
    commands = list(environment.commands)

    result = automation.restore_launch_agent(settings)

    assert result["ok"] is True
    assert result["restored"] is False
    assert environment.commands == commands
    assert environment.plist.read_bytes() == saved_plist
    assert environment.plist.stat().st_ino == saved_inode
    assert list(settings.state_directory.glob("launch-agent-before-restore-*.plist")) == backups
    assert settings.authorization.read_bytes() == environment.authorization


@pytest.mark.parametrize("invalid_marker", ["source_root", "state_root", "python_executable", "missing", "invalid"])
def test_restore_refuses_missing_invalid_or_mismatched_runtime_receipt_before_replacement(
    restoration: RestoreEnvironment,
    invalid_marker: str,
) -> None:
    environment = restoration
    if invalid_marker == "missing":
        environment.marker.unlink()
    elif invalid_marker == "invalid":
        environment.marker.write_text("not-json", encoding="utf-8")
    else:
        document = json.loads(environment.marker.read_text(encoding="utf-8"))
        document[invalid_marker] += "-different"
        environment.marker.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(automation.AutomationError, match="ownership receipt"):
        automation.restore_launch_agent(environment.settings)

    assert environment.plist.read_bytes() == environment.original_plist
    assert environment.settings.authorization.read_bytes() == environment.authorization
    assert environment.commands == []
    assert environment.events == []
    assert not list(environment.settings.state_directory.glob("launch-agent-before-restore-*.plist"))


def test_restore_does_not_create_a_schedule_when_the_existing_plist_is_missing(
    restoration: RestoreEnvironment,
) -> None:
    environment = restoration
    environment.plist.unlink()

    with pytest.raises(automation.AutomationError, match="existing LaunchAgent plist"):
        automation.restore_launch_agent(environment.settings)

    assert not environment.plist.exists()
    assert environment.commands == []
    assert environment.settings.authorization.read_bytes() == environment.authorization


def test_restore_never_enables_a_service_if_bootstrap_is_refused(
    restoration: RestoreEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = restoration
    normal_command = automation._run

    def command(args: list[str], **kwargs: object) -> bytes:
        if args[:2] == ["/bin/launchctl", "bootstrap"]:
            environment.commands.append(tuple(args))
            raise automation.AutomationError("bootstrap refused by existing disabled setting")
        return normal_command(args, **kwargs)

    monkeypatch.setattr(automation, "_run", command)

    with pytest.raises(automation.AutomationError, match="bootstrap refused"):
        automation.restore_launch_agent(environment.settings)

    assert environment.loaded is False
    assert environment.plist.read_bytes() == environment.original_plist
    assert environment.commands[-1][1] == "bootstrap"
    assert all("enable" not in selected for selected in environment.commands)
    assert environment.settings.authorization.read_bytes() == environment.authorization
    backups = list(environment.settings.state_directory.glob("launch-agent-before-restore-*.plist"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == environment.original_plist
