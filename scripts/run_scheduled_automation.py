#!/usr/bin/env python3
"""Refresh the dedicated scheduled checkout before importing Regime automation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit


REPOSITORY = "SonChangGi/regime"
BRANCH = "main"
ROOT = Path(__file__).resolve().parents[1]
GIT_TIMEOUT_SECONDS = 300.0


class LaunchError(RuntimeError):
    """The scheduled checkout is not safe to refresh or execute."""


class LaunchAlreadyRunning(LaunchError):
    """An earlier scheduled execution still owns this runtime."""


def _environment() -> dict[str, str]:
    # A caller's editor/terminal Git context must never redirect operations to
    # the state checkout or its index, including after the final Python exec.
    local = {
        "GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_IMPLICIT_WORK_TREE", "GIT_PREFIX", "GIT_SUPER_PREFIX",
        "GIT_GRAFT_FILE", "GIT_SHALLOW_FILE", "GIT_REPLACE_REF_BASE",
        "GIT_QUARANTINE_PATH", "GIT_CONFIG", "GIT_CONFIG_PARAMETERS", "GIT_CONFIG_COUNT",
    }
    return {key: value for key, value in os.environ.items()
            if key not in local and not key.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_"))}


def _git(source: Path, *arguments: str) -> str:
    environment = _environment()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    try:
        completed = subprocess.run(
            ["git", *arguments], cwd=source, text=True, capture_output=True,
            check=False, env=environment, timeout=GIT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise LaunchError(f"git {' '.join(arguments[:2])} timed out") from exc
    if completed.returncode:
        raise LaunchError(f"git {' '.join(arguments[:2])} failed (exit {completed.returncode})")
    return completed.stdout.strip()


def _repository_identity(remote: str) -> str | None:
    if remote.startswith("git@github.com:"):
        repository = remote.removeprefix("git@github.com:")
    else:
        parsed = urlsplit(remote)
        if (parsed.scheme not in {"https", "ssh"} or parsed.hostname != "github.com"
                or parsed.password is not None or parsed.query or parsed.fragment
                or parsed.port is not None
                or parsed.username not in ({None} if parsed.scheme == "https" else {"git"})):
            return None
        repository = parsed.path.removeprefix("/")
    return repository.removesuffix(".git")


def _check_checkout(source: Path) -> None:
    git_directory = source / ".git"
    if not git_directory.is_dir() or git_directory.is_symlink():
        raise LaunchError("source must have its own .git directory")
    if Path(_git(source, "rev-parse", "--show-toplevel")).resolve() != source:
        raise LaunchError("source is not the checkout root")
    for option in ("--git-dir", "--git-common-dir"):
        directory = Path(_git(source, "rev-parse", option))
        if not directory.is_absolute():
            directory = source / directory
        if directory.resolve() != git_directory:
            raise LaunchError("source must have an independent Git directory")
    if _git(source, "branch", "--show-current") != BRANCH:
        raise LaunchError("source must be on main")
    if _git(source, "status", "--porcelain", "--untracked-files=all"):
        raise LaunchError("source checkout is dirty; refusing to update")
    remotes = _git(source, "config", "--get-all", "remote.origin.url").splitlines()
    if len(remotes) != 1 or _repository_identity(remotes[0]) != REPOSITORY:
        raise LaunchError("source origin must identify SonChangGi/regime on GitHub")
    if _repository_identity(_git(source, "remote", "get-url", "origin")) != REPOSITORY:
        raise LaunchError("effective source origin must identify SonChangGi/regime on GitHub")


def launch(source_root: Path, state_root: Path, python: Path, *, force_retry: bool = False) -> None:
    source, state = source_root.resolve(), state_root.resolve()
    # Preserve the venv entry point: resolving its symlink would select the base
    # interpreter and lose the existing virtual environment.
    executable = Path(os.path.abspath(python))
    if source != ROOT:
        raise LaunchError("source root must match this launcher's repository")
    if source == state:
        raise LaunchError("scheduled source and state roots must be different")
    if not state.is_dir():
        raise LaunchError("state root does not exist")
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise LaunchError("the configured Python executable is unavailable")

    directory = state / "build/weekly-automation"
    marker = directory / "runtime-checkout.json"
    expected = {
        "schema_version": 1,
        "repository": REPOSITORY,
        "branch": BRANCH,
        "source_root": str(source),
        "state_root": str(state),
        "python_executable": str(executable),
    }
    if not marker.is_file():
        raise LaunchError("scheduled runtime checkout marker is missing")
    with (directory / "scheduled-launch.lock").open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LaunchAlreadyRunning("another scheduled launch is running") from exc
        try:
            document = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise LaunchError("scheduled runtime checkout marker is unreadable") from exc
        if document != expected or type(document.get("schema_version")) is not int:
            raise LaunchError("scheduled runtime checkout marker identity mismatch")

        _check_checkout(source)
        _git(source, "fetch", "origin", BRANCH)
        remote_head = _git(source, "rev-parse", f"refs/remotes/origin/{BRANCH}")
        if _git(source, "rev-parse", "FETCH_HEAD") != remote_head:
            raise LaunchError("fetched main does not match origin/main")
        # --ff-only alone accepts a locally-ahead branch. Such a branch must
        # also fail closed instead of running code absent from remote main.
        _git(source, "merge-base", "--is-ancestor", "HEAD", remote_head)
        _git(source, "merge", "--ff-only", f"origin/{BRANCH}")
        _check_checkout(source)
        if _git(source, "rev-parse", "HEAD") != remote_head:
            raise LaunchError("scheduled source does not match remote main")

        environment = _environment()
        environment["PYTHONPATH"] = str(source / "src")
        arguments = [
            str(executable), "-m", "regime_lab", "automation", "run",
            "--config", str(source / "config/automation.json"),
            "--state-root", str(state),
        ]
        if force_retry:
            arguments.append("--force-retry")
        # flock must survive exec and cover the automation, not only git fetch.
        os.set_inheritable(lock.fileno(), True)
        os.chdir(source)
        os.execve(str(executable), arguments, environment)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--force-retry", action="store_true",
                        help="explicit operator recovery of a repaired transient failure; normal guards remain")
    arguments = parser.parse_args()
    try:
        launch(arguments.source_root, arguments.state_root, arguments.python,
               force_retry=arguments.force_retry)
    except LaunchAlreadyRunning:
        print(json.dumps({"ok": True, "status": "already_running",
                          "checked_at": datetime.now(timezone.utc).isoformat()}))
        return 0
    except (LaunchError, OSError, ValueError) as exc:
        print(f"scheduled automation refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
