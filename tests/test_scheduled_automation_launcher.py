"""Exercise the pre-import launcher against real, isolated Git repositories."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_scheduled_automation.py"
GITHUB_URL = "https://github.com/SonChangGi/regime.git"
BOOTSTRAP = """
import importlib.util, pathlib, sys
sys.dont_write_bytecode = True
source, state, python, origin = sys.argv[1:]
spec = importlib.util.spec_from_file_location('scheduled_launcher', pathlib.Path(source) / 'scripts/run_scheduled_automation.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
# Only the test's bare remote is accepted in addition to the production identity.
# All Git operations and the final exec still run in real subprocesses.
identity = launcher._repository_identity
launcher._repository_identity = lambda url: launcher.REPOSITORY if url == origin else identity(url)
sys.argv = ['run_scheduled_automation.py', '--source-root', source, '--state-root', state, '--python', python]
raise SystemExit(launcher.main())
"""
APPLICATION = """
import fcntl, json, os, pathlib, sys
TOKEN = {token!r}
state = pathlib.Path(sys.argv[sys.argv.index('--state-root') + 1])
with (state / 'build/weekly-automation/scheduled-launch.lock').open('a+') as lock:
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        held = False
    except BlockingIOError:
        held = True
print(json.dumps(dict(token=TOKEN, argv=sys.argv[1:], python=sys.executable,
    pythonpath=os.environ.get('PYTHONPATH'), module=__file__, cwd=os.getcwd(), lock_held=held,
    git_context={{key: os.environ[key] for key in ['GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE'] if key in os.environ}})))
"""


def git(directory: Path, *arguments: str) -> str:
    environment = {
        **os.environ, "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull, "GIT_OPTIONAL_LOCKS": "0",
    }
    result = subprocess.run(
        ["git", *arguments], cwd=directory, env=environment, check=True,
        text=True, capture_output=True,
    )
    return result.stdout.strip()


def commit(directory: Path, message: str) -> None:
    git(directory, "add", ".")
    git(directory, "-c", "user.name=Launcher Tests", "-c",
        "user.email=launcher@example.invalid", "commit", "-m", message)


@pytest.fixture
def checkout(tmp_path: Path) -> dict[str, Path]:
    seed, origin, source, state = (tmp_path / name for name in ("seed", "origin.git", "runtime", "editor"))
    git(tmp_path, "init", "-b", "main", str(seed))
    (seed / "scripts").mkdir()
    (seed / "scripts/run_scheduled_automation.py").write_bytes(SCRIPT.read_bytes())
    (seed / "src/regime_lab").mkdir(parents=True)
    (seed / "src/regime_lab/__init__.py").write_text("")
    (seed / "src/regime_lab/__main__.py").write_text(APPLICATION.format(token="old"))
    (seed / "config").mkdir()
    (seed / "config/automation.json").write_text('{"schema_version": 1}\n')
    (seed / ".gitignore").write_text("__pycache__/\n")
    commit(seed, "initial source")
    git(tmp_path, "clone", "--bare", str(seed), str(origin))
    git(tmp_path, "clone", str(origin), str(source))

    git(tmp_path, "init", "-b", "main", str(state))
    (state / ".gitignore").write_text("/build/\n/data/\n/.venv/\n")
    (state / "editor.py").write_text("saved = True\n")
    commit(state, "user working repository")
    (state / "editor.py").write_text("saved = False  # unsaved user work\n")
    (state / "data").mkdir()
    (state / "data/regime.sqlite3").write_bytes(b"existing-database-must-not-change")
    python = state / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    (python.parent.parent / "pyvenv.cfg").write_text(
        f"home = {Path(sys._base_executable).parent}\n"
        "include-system-site-packages = false\n"
    )
    directory = state / "build/weekly-automation"
    directory.mkdir(parents=True)
    marker = directory / "runtime-checkout.json"
    marker.write_text(json.dumps({
        "schema_version": 1, "repository": "SonChangGi/regime", "branch": "main",
        "source_root": str(source.resolve()), "state_root": str(state.resolve()),
        "python_executable": str(python),
    }))
    return {"seed": seed, "origin": origin, "source": source, "state": state,
            "python": python, "marker": marker}


def run_launcher(checkout: dict[str, Path], **environment: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", BOOTSTRAP, str(checkout["source"]),
         str(checkout["state"]), str(checkout["python"]), str(checkout["origin"])],
        cwd=checkout["state"], text=True, capture_output=True, check=False,
        env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull,
             "GIT_CONFIG_SYSTEM": os.devnull, "PYTHONDONTWRITEBYTECODE": "1",
             **environment}, timeout=20,
    )


def advance_origin(checkout: dict[str, Path]) -> str:
    seed = checkout["seed"]
    (seed / "src/regime_lab/__main__.py").write_text(APPLICATION.format(token="fresh remote"))
    commit(seed, "new automation source")
    git(seed, "push", str(checkout["origin"]), "main")
    return git(seed, "rev-parse", "HEAD")


def git_inventory(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (root / ".git").rglob("*") if path.is_file()}


@pytest.mark.parametrize("inherited_git_context", [False, True])
def test_clean_fast_forward_executes_fresh_source_and_preserves_dirty_editor(
    checkout: dict[str, Path], inherited_git_context: bool,
) -> None:
    state, source = checkout["state"], checkout["source"]
    original = {"git": git_inventory(state), "editor": (state / "editor.py").read_bytes(),
                "database": (state / "data/regime.sqlite3").read_bytes()}
    expected_head = advance_origin(checkout)
    context = {"GIT_DIR": str(state / ".git"), "GIT_WORK_TREE": str(state),
               "GIT_INDEX_FILE": str(state / ".git/index")} if inherited_git_context else {}
    result = run_launcher(checkout, PYTHONPATH=str(state / "unrelated-src"), **context)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output == {
        "token": "fresh remote", "argv": ["automation", "run", "--config",
            str(source / "config/automation.json"), "--state-root", str(state)],
        "python": str(checkout["python"]), "pythonpath": str(source / "src"),
        "module": str(source / "src/regime_lab/__main__.py"), "cwd": str(source),
        "lock_held": True, "git_context": {},
    }
    assert git(source, "rev-parse", "HEAD") == expected_head
    assert git(source, "status", "--porcelain", "--untracked-files=all") == ""
    assert git_inventory(state) == original["git"]
    assert (state / "editor.py").read_bytes() == original["editor"]
    assert (state / "data/regime.sqlite3").read_bytes() == original["database"]
    assert git(state, "status", "--porcelain") == "M editor.py"


@pytest.mark.parametrize("untracked", [False, True])
def test_dirty_runtime_is_preserved_and_never_executes(checkout: dict[str, Path], untracked: bool) -> None:
    source = checkout["source"]
    edited = source / ("untracked.py" if untracked else "config/automation.json")
    edited.write_text("user runtime edit\n")
    before = git(source, "rev-parse", "HEAD")
    advance_origin(checkout)
    result = run_launcher(checkout)
    assert result.returncode == 1 and "dirty" in result.stderr
    assert result.stdout == ""
    assert edited.read_text() == "user runtime edit\n"
    assert git(source, "rev-parse", "HEAD") == before


@pytest.mark.parametrize("remote_advanced", [False, True])
def test_unpublished_or_diverged_runtime_commit_is_preserved(checkout: dict[str, Path], remote_advanced: bool) -> None:
    source = checkout["source"]
    (source / "local.txt").write_text("unpublished local work\n")
    commit(source, "local main commit")
    before = git(source, "rev-parse", "HEAD")
    if remote_advanced:
        advance_origin(checkout)
    result = run_launcher(checkout)
    assert result.returncode == 1 and "merge-base --is-ancestor failed" in result.stderr
    assert result.stdout == ""
    assert git(source, "rev-parse", "HEAD") == before
    assert (source / "local.txt").read_text() == "unpublished local work\n"


def test_wrong_runtime_branch_is_rejected_before_fetch(checkout: dict[str, Path]) -> None:
    git(checkout["source"], "checkout", "-b", "user-work")
    before = git(checkout["source"], "rev-parse", "origin/main")
    advance_origin(checkout)
    result = run_launcher(checkout)
    assert result.returncode == 1 and "must be on main" in result.stderr
    assert result.stdout == ""
    assert git(checkout["source"], "rev-parse", "origin/main") == before


def test_fetch_failure_preserves_source_and_does_not_execute(checkout: dict[str, Path]) -> None:
    source = checkout["source"]
    before = git(source, "rev-parse", "HEAD")
    checkout["origin"].rename(checkout["origin"].with_name("unavailable-origin.git"))
    result = run_launcher(checkout)
    assert result.returncode == 1 and "git fetch origin failed" in result.stderr
    assert result.stdout == ""
    assert git(source, "rev-parse", "HEAD") == before
    assert git(source, "status", "--porcelain", "--untracked-files=all") == ""


def test_shared_git_directory_is_rejected(checkout: dict[str, Path]) -> None:
    source = checkout["source"]
    owned = source / ".git-owned"
    (source / ".git").rename(owned)
    (source / ".git").write_text(f"gitdir: {owned}\n")
    before = git_inventory(checkout["state"])
    result = run_launcher(checkout)
    assert result.returncode == 1 and "own .git directory" in result.stderr
    assert result.stdout == ""
    assert git_inventory(checkout["state"]) == before


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("repository", "Other/regime"), ("branch", "other"),
    ("source_root", "/wrong/source"), ("state_root", "/wrong/state"),
    ("python_executable", "/wrong/python"), ("extra", "unreviewed"),
])
def test_marker_identity_mismatch_refuses_without_updating_source(checkout: dict[str, Path], field: str, value: object) -> None:
    marker = checkout["marker"]
    document = json.loads(marker.read_text())
    document[field] = value
    marker.write_text(json.dumps(document))
    before = git(checkout["source"], "rev-parse", "HEAD")
    advance_origin(checkout)
    result = run_launcher(checkout)
    assert result.returncode == 1 and "marker identity mismatch" in result.stderr
    assert result.stdout == ""
    assert git(checkout["source"], "rev-parse", "HEAD") == before


@pytest.mark.parametrize("contents", [None, "not JSON", "[]"])
def test_missing_or_malformed_marker_fails_closed(checkout: dict[str, Path], contents: str | None) -> None:
    marker = checkout["marker"]
    if contents is None:
        marker.unlink()
    else:
        marker.write_text(contents)
    result = run_launcher(checkout)
    assert result.returncode == 1 and "marker" in result.stderr
    assert result.stdout == ""


def test_wrong_origin_and_rewritten_effective_origin_are_rejected(checkout: dict[str, Path]) -> None:
    source = checkout["source"]
    git(source, "remote", "set-url", "origin", "https://github.com/Other/regime.git")
    result = run_launcher(checkout)
    assert result.returncode == 1 and "source origin" in result.stderr
    git(source, "remote", "set-url", "origin", GITHUB_URL)
    git(source, "config", "url.https://example.invalid/.insteadOf", "https://github.com/")
    result = run_launcher(checkout)
    assert result.returncode == 1 and "effective source origin" in result.stderr
    assert result.stdout == ""


def test_concurrent_launch_skips_successfully_without_source_or_editor_changes(checkout: dict[str, Path]) -> None:
    lock_path = checkout["marker"].with_name("scheduled-launch.lock")
    before = git(checkout["source"], "rev-parse", "HEAD")
    remote_before = git(checkout["source"], "rev-parse", "origin/main")
    editor_before = (checkout["state"] / "editor.py").read_bytes()
    state_git_before = git_inventory(checkout["state"])
    advance_origin(checkout)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run_launcher(checkout)
    assert result.returncode == 0 and result.stderr == ""
    assert json.loads(result.stdout) == {"ok": True, "status": "already_running"}
    assert git(checkout["source"], "rev-parse", "HEAD") == before
    assert git(checkout["source"], "rev-parse", "origin/main") == remote_before
    assert (checkout["state"] / "editor.py").read_bytes() == editor_before
    assert git_inventory(checkout["state"]) == state_git_before


def test_source_root_must_match_script_and_differ_from_state(checkout: dict[str, Path]) -> None:
    spec = importlib.util.spec_from_file_location("scheduled_launcher_root_check", SCRIPT)
    launcher = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(launcher)
    with pytest.raises(launcher.LaunchError, match="match this launcher's repository"):
        launcher.launch(checkout["source"], checkout["state"], checkout["python"])
    with pytest.raises(launcher.LaunchError, match="must be different"):
        launcher.launch(ROOT, ROOT, checkout["python"])
