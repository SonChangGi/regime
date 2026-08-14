"""Fail-closed local scheduling and publication for weekly Regime releases.

Collection and model fitting stay on the user's Mac because the authoritative
SQLite revision store and provider Keychain items are local-only.  A validated
derived-result candidate is published from an isolated temporary checkout, so
the scheduled job never stages, commits, or rebases the user's working tree.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Iterator, Mapping, Sequence
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

from regime_lab.collection import EASTERN, last_completed_week_cutoff
from regime_lab.config import project_root
from regime_lab.io import write_json_atomic
from regime_lab.path_safety import confined_mutable_path
from regime_lab.schema import ContractError, validate_dashboard_payload
from regime_lab.data import DailyRequestBudget, SQLiteSnapshotStore


UTC = timezone.utc
PUBLICATION_PATH = "publication/live/regime-results.json"
PUBLIC_PAYLOAD_PATH = "data/regime-results.json"
PUBLIC_MANIFEST_PATH = "publication-manifest.json"
AUTOMATION_LABEL = "com.sonchanggi.regime.weekly-release"
AUTOMATION_TRAILER = "Regime-Automation: weekly-release-v1"
ALLOWED_REMOTE_DRIFT = frozenset({PUBLICATION_PATH})


class AutomationError(RuntimeError):
    """Raised when a weekly run must stop without changing the public result."""


class AlreadyRunning(AutomationError):
    """Raised when another local weekly run owns the process lock."""


@dataclass(frozen=True)
class AutomationSettings:
    config_path: Path
    root: Path
    automation_id: str
    schedule_hour: int
    schedule_minute: int
    minimum_cutoff_age: timedelta
    require_ac_power: bool
    profile: str
    database: Path
    payload: Path
    artifacts: Path
    state_directory: Path
    authorization: Path
    repository: str
    remote: str
    branch: str
    workflow: str
    public_root: str
    workflow_timeout: timedelta
    public_readback_timeout: timedelta

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AutomationSettings":
        root = project_root().resolve()
        raw_path = Path(path or "config/automation.json")
        config_path = raw_path if raw_path.is_absolute() else root / raw_path
        try:
            document = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AutomationError(f"automation config is unavailable: {config_path}") from exc
        if not isinstance(document, dict) or document.get("schema_version") != 1:
            raise AutomationError("automation config schema_version must be 1")
        try:
            schedule = document["schedule"]
            build = document["build"]
            github = document["github"]
            hour = int(schedule["hour"])
            minute = int(schedule["minute"])
            minimum_age = int(schedule["minimum_cutoff_age_hours"])
            workflow_minutes = int(github["workflow_timeout_minutes"])
            readback_minutes = int(github["public_readback_timeout_minutes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AutomationError("automation config fields are invalid") from exc
        if not 0 <= hour <= 23 or not 0 <= minute <= 59:
            raise AutomationError("automation schedule time is invalid")
        if minimum_age < 1 or workflow_minutes < 1 or readback_minutes < 1:
            raise AutomationError("automation timing values must be positive")
        if type(schedule.get("require_ac_power", True)) is not bool:
            raise AutomationError("schedule.require_ac_power must be boolean")

        def mutable(value: object, label: str) -> Path:
            return confined_mutable_path(
                str(value), project_directory=root, label=label
            )

        repository = str(github.get("repository", ""))
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise AutomationError("github.repository must be owner/name")
        public_root = str(github.get("public_root", ""))
        if not public_root.startswith("https://") or not public_root.endswith("/"):
            raise AutomationError("github.public_root must be an HTTPS directory URL")
        profile = str(build.get("profile", ""))
        if profile != "standard":
            raise AutomationError("weekly live automation must use the standard profile")
        automation_id = str(document.get("automation_id", ""))
        if automation_id != "weekly-regime-release-v1":
            raise AutomationError("automation_id is unsupported")

        return cls(
            config_path=config_path.resolve(),
            root=root,
            automation_id=automation_id,
            schedule_hour=hour,
            schedule_minute=minute,
            minimum_cutoff_age=timedelta(hours=minimum_age),
            require_ac_power=schedule.get("require_ac_power", True),
            profile=profile,
            database=mutable(build["database"], "automation database"),
            payload=mutable(build["payload"], "automation payload"),
            artifacts=mutable(build["artifacts"], "automation artifacts"),
            state_directory=mutable(
                build["state_directory"], "automation state directory"
            ),
            authorization=mutable(
                build["authorization"], "automation authorization"
            ),
            repository=repository,
            remote=str(github.get("remote", "origin")),
            branch=str(github.get("branch", "main")),
            workflow=str(github.get("workflow", "pages.yml")),
            public_root=public_root,
            workflow_timeout=timedelta(minutes=workflow_minutes),
            public_readback_timeout=timedelta(minutes=readback_minutes),
        )

    @property
    def status_path(self) -> Path:
        return self.state_directory / "automation-health.json"

    @property
    def lock_path(self) -> Path:
        return self.state_directory / "weekly-release.lock"

    @property
    def install_lock_path(self) -> Path:
        return self.state_directory / "launch-agent-install.lock"

    @property
    def candidate_path(self) -> Path:
        return self.state_directory / "candidate" / "regime-results.json"

    @property
    def candidate_metadata_path(self) -> Path:
        return self.state_directory / "candidate" / "metadata.json"


@dataclass(frozen=True)
class RemotePublication:
    head_sha: str
    payload_bytes: bytes
    data_as_of: datetime

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.payload_bytes).hexdigest()


def _safe_error(exc: BaseException) -> str:
    message = f"{type(exc).__name__}: {exc}"
    message = re.sub(
        r"(?i)(api[_-]?key|apikey|authorization|password|token)\s*[=:]\s*\S+",
        r"\1=[redacted]",
        message,
    )
    return message[:1000]


def _parse_aware_datetime(value: object, *, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise AutomationError(f"{label} must be an ISO-8601 datetime") from exc
    if parsed.tzinfo is None:
        raise AutomationError(f"{label} must include a timezone")
    return parsed.astimezone(UTC)


def _json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AutomationError(f"{label} must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise AutomationError(f"{label} root must be an object")
    return value


def _run(
    args: Sequence[str],
    *,
    cwd: Path,
    capture: bool = False,
    timeout: float | None = None,
) -> bytes:
    completed = subprocess.run(
        list(args),
        cwd=cwd,
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        timeout=timeout,
    )
    if completed.returncode != 0:
        label = Path(args[0]).name if args else "command"
        raise AutomationError(f"{label} failed with exit code {completed.returncode}")
    return completed.stdout or b""


@contextmanager
def automation_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AlreadyRunning("another weekly Regime release is running") from exc
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(f"pid={os.getpid()}\n")
            handle.flush()
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def target_cutoff(
    now: datetime,
    *,
    minimum_age: timedelta,
) -> tuple[datetime, bool]:
    current = now.astimezone(UTC)
    cutoff = last_completed_week_cutoff(current)
    return cutoff, current - cutoff >= minimum_age


def _status_document(
    settings: AutomationSettings,
    *,
    status: str,
    stage: str,
    started_at: datetime,
    target: datetime | None = None,
    detail: str | None = None,
    commit_sha: str | None = None,
    workflow_url: str | None = None,
) -> dict[str, Any]:
    now = datetime.now(UTC)
    return {
        "schema_version": 1,
        "automation_id": settings.automation_id,
        "status": status,
        "stage": stage,
        "started_at": started_at.isoformat(),
        "updated_at": now.isoformat(),
        "target_data_as_of": target.isoformat() if target else None,
        "detail": detail,
        "commit_sha": commit_sha,
        "workflow_url": workflow_url,
        "public_url": settings.public_root,
    }


def _write_status(settings: AutomationSettings, **values: Any) -> dict[str, Any]:
    document = _status_document(settings, **values)
    write_json_atomic(settings.status_path, document)
    return document


def _origin_slug(url: str) -> str | None:
    match = re.search(r"github\.com(?::|/)([^/]+/[^/]+?)(?:\.git)?$", url.strip())
    return match.group(1) if match else None


def _ensure_ac_power(settings: AutomationSettings) -> None:
    if not settings.require_ac_power or sys.platform != "darwin":
        return
    output = _run(["/usr/bin/pmset", "-g", "batt"], cwd=settings.root, capture=True)
    if b"AC Power" not in output:
        raise AutomationError("weekly automation waits for AC power")


def _validate_local_authorization(
    settings: AutomationSettings,
    *,
    now: datetime,
) -> None:
    try:
        document = json.loads(settings.authorization.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AutomationError(
            "local automation authorization is missing; reinstall or renew it"
        ) from exc
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise AutomationError("local automation authorization is invalid")
    if document.get("automation_id") != settings.automation_id:
        raise AutomationError("local automation authorization has the wrong identity")
    expected_scopes = {
        "alfred_local_storage_ml",
        "personal_noncommercial_derived_publication",
    }
    if document.get("confirmed") is not True or set(document.get("scopes", [])) != expected_scopes:
        raise AutomationError("local automation authorization scopes are incomplete")
    reviewed_at = _parse_aware_datetime(
        document.get("reviewed_at"), label="authorization reviewed_at"
    )
    review_after = _parse_aware_datetime(
        document.get("review_after"), label="authorization review_after"
    )
    current = now.astimezone(UTC)
    if reviewed_at > current or review_after <= current:
        raise AutomationError("local automation authorization requires renewal")


def _write_local_authorization(
    settings: AutomationSettings,
    *,
    alfred_rights_confirmed: bool,
    personal_noncommercial_publication_acknowledged: bool,
    now: datetime | None = None,
) -> None:
    if not alfred_rights_confirmed:
        raise AutomationError(
            "install requires explicit ALFRED local-storage and ML-training permission"
        )
    if not personal_noncommercial_publication_acknowledged:
        raise AutomationError(
            "install requires explicit personal noncommercial derived-publication permission"
        )
    reviewed_at = (now or datetime.now(UTC)).astimezone(UTC)
    write_json_atomic(
        settings.authorization,
        {
            "schema_version": 1,
            "automation_id": settings.automation_id,
            "confirmed": True,
            "scopes": [
                "alfred_local_storage_ml",
                "personal_noncommercial_derived_publication",
            ],
            "reviewed_at": reviewed_at.isoformat(),
            "review_after": (reviewed_at + timedelta(days=180)).isoformat(),
            "contains_credentials": False,
        },
    )
    os.chmod(settings.authorization, 0o600)


def _alpha_quota_preflight(
    settings: AutomationSettings,
    *,
    target: datetime,
) -> None:
    series_config = json.loads(
        (settings.root / "config" / "series.json").read_text(encoding="utf-8")
    )
    alpha = series_config.get("alpha_vantage", {})
    symbols = tuple(str(item).strip().upper() for item in alpha.get("symbols", []))
    if not symbols or len(symbols) != len(set(symbols)):
        raise AutomationError("Alpha Vantage automation symbols are invalid")
    raw_limit = alpha.get("daily_request_cap")
    if type(raw_limit) is not int or raw_limit != 25:
        raise AutomationError("Alpha Vantage automation requires the literal free cap 25")
    requests_needed = len(symbols)
    if settings.database.is_file():
        with SQLiteSnapshotStore(settings.database) as store:
            last_good = store.get_last_good_provenance(
                source="alpha_vantage", dataset="weekly_adjusted_etf"
            )
        if last_good is not None and last_good.cutoff == target.astimezone(UTC):
            configured = {
                str(item).strip().upper()
                for item in last_good.request_params.get("symbols", [])
            }
            requests_needed = len(set(symbols) - configured)
    if requests_needed == 0:
        return
    budget = DailyRequestBudget(
        limit=raw_limit,
        database_path=settings.database,
    )
    remaining = budget.remaining
    if remaining < requests_needed:
        next_available = budget.next_available_at(requests_needed)
        raise AutomationError(
            "Alpha Vantage rolling-24h budget cannot reserve the full batch; "
            f"needed={requests_needed}, remaining={remaining}, "
            f"next_available_at={next_available.isoformat()}"
        )


def _sqlite_quick_check(path: Path) -> None:
    if not path.is_file():
        raise AutomationError("automation snapshot database is missing")
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        rows = [str(row[0]) for row in connection.execute("PRAGMA quick_check")]
    if rows != ["ok"]:
        raise AutomationError("automation snapshot database quick_check failed")


def _git_preflight(settings: AutomationSettings) -> RemotePublication:
    branch = _run(
        ["git", "branch", "--show-current"], cwd=settings.root, capture=True
    ).decode().strip()
    if branch != settings.branch:
        raise AutomationError(f"automation requires branch {settings.branch}")
    dirty = _run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=settings.root,
        capture=True,
    ).decode().strip()
    if dirty:
        raise AutomationError("tracked working tree changes block weekly automation")
    origin_url = _run(
        ["git", "remote", "get-url", settings.remote],
        cwd=settings.root,
        capture=True,
    ).decode().strip()
    if _origin_slug(origin_url) != settings.repository:
        raise AutomationError("git remote does not match configured repository")
    _run(
        ["git", "fetch", "--quiet", settings.remote, settings.branch],
        cwd=settings.root,
    )
    remote_ref = f"{settings.remote}/{settings.branch}"
    changed = set(
        _run(
            ["git", "diff", "--name-only", "HEAD", remote_ref, "--"],
            cwd=settings.root,
            capture=True,
        ).decode().splitlines()
    )
    if changed - ALLOWED_REMOTE_DRIFT:
        raise AutomationError(
            "remote source changed outside the publication snapshot; update the local checkout"
        )
    head_sha = _run(
        ["git", "rev-parse", remote_ref], cwd=settings.root, capture=True
    ).decode().strip()
    payload_bytes = _run(
        ["git", "show", f"{remote_ref}:{PUBLICATION_PATH}"],
        cwd=settings.root,
        capture=True,
    )
    payload = _json_object(payload_bytes, label="remote publication")
    data_as_of = _parse_aware_datetime(
        payload.get("meta", {}).get("data_as_of"), label="remote data_as_of"
    )
    validate_automation_candidate(payload_bytes, target=data_as_of)
    return RemotePublication(head_sha, payload_bytes, data_as_of)


def validate_automation_candidate(raw: bytes, *, target: datetime) -> dict[str, Any]:
    payload = _json_object(raw, label="automation candidate")
    try:
        validate_dashboard_payload(payload)
    except (ContractError, KeyError, TypeError, ValueError) as exc:
        raise AutomationError(f"candidate dashboard contract failed: {exc}") from exc
    meta = payload.get("meta", {})
    if meta.get("mode") != "live" or meta.get("result_version") != "weekly-regime-result-v4":
        raise AutomationError("candidate must be a live v4 result")
    candidate_cutoff = _parse_aware_datetime(
        meta.get("data_as_of"), label="candidate data_as_of"
    )
    if candidate_cutoff != target.astimezone(UTC):
        raise AutomationError("candidate data_as_of does not match the due cutoff")
    sources = payload.get("sources")
    if not isinstance(sources, list):
        raise AutomationError("candidate sources are missing")
    by_id = {
        str(source.get("id")): source
        for source in sources
        if isinstance(source, dict) and isinstance(source.get("id"), str)
    }
    if set(by_id) != {"alpha_vantage", "alfred"}:
        raise AutomationError("candidate sources must be exactly alpha_vantage and alfred")
    if any(source.get("status") != "ok" for source in by_id.values()):
        raise AutomationError("provider-degraded candidate may not be promoted")
    if any(source.get("issues") for source in by_id.values()):
        raise AutomationError("candidate provider issues must be empty")
    target_date = target.astimezone(EASTERN).date().isoformat()
    alpha = by_id["alpha_vantage"]
    alpha_available = _parse_aware_datetime(
        alpha.get("available_at"), label="alpha_vantage available_at"
    )
    if alpha_available != target.astimezone(UTC):
        raise AutomationError("Alpha Vantage has not reached the due cutoff")
    if not str(alpha.get("coverage", "")).endswith(target_date):
        raise AutomationError("Alpha Vantage coverage has not reached the due cutoff")
    weekly = payload.get("weekly")
    if not isinstance(weekly, list) or not weekly or not isinstance(weekly[-1], dict):
        raise AutomationError("candidate weekly results are missing")
    latest = weekly[-1]
    if latest.get("date") != target_date:
        raise AutomationError("candidate latest week does not match the due cutoff")
    if latest.get("health", {}).get("status") != "ok":
        raise AutomationError("candidate latest week health is not ok")
    if bool(payload.get("model", {}).get("latest_forecast_fallback")):
        raise AutomationError("candidate latest forecast fallback may not be promoted")
    meta_status = meta.get("status")
    if meta_status == "degraded":
        diagnostic = payload.get("model", {}).get("holdout_diagnostic", {})
        if diagnostic.get("status") != "weak_generalization":
            raise AutomationError("candidate degradation is not the allowed diagnostic warning")
    elif meta_status != "ok":
        raise AutomationError("candidate meta.status is not publishable")
    return payload


def _verify_candidate_package(settings: AutomationSettings, candidate_path: Path) -> None:
    settings.state_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".package-check-", dir=settings.state_directory
    ) as temporary:
        output = Path(temporary) / "public-dashboard"
        _run(
            [
                sys.executable,
                "scripts/package_public_demo.py",
                "--payload",
                str(candidate_path),
                "--publication-mode",
                "live-derived",
                "--acknowledge-personal-noncommercial-publication",
                "--output",
                str(output),
            ],
            cwd=settings.root,
        )
        _run(
            [sys.executable, "scripts/verify_public_package.py", str(output)],
            cwd=settings.root,
        )
        if (output / PUBLIC_PAYLOAD_PATH).read_bytes() != candidate_path.read_bytes():
            raise AutomationError("verified package payload bytes differ from the candidate")


def _candidate_context(settings: AutomationSettings) -> dict[str, str]:
    """Bind a resumable candidate to its tracked source and effective configs."""

    dirty = _run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=settings.root,
        capture=True,
    ).decode().strip()
    if dirty:
        raise AutomationError("tracked source or config changed during live build")
    tree = _run(
        ["git", "ls-tree", "-r", "--full-tree", "HEAD"],
        cwd=settings.root,
        capture=True,
    ).decode("utf-8")
    retained = [
        line
        for line in tree.splitlines()
        if not line.endswith(f"\t{PUBLICATION_PATH}")
    ]
    workspace = hashlib.sha256()
    for line in retained:
        try:
            relative = line.split("\t", 1)[1]
            path = settings.root / relative
            stat = path.lstat()
        except (IndexError, OSError) as exc:
            raise AutomationError("tracked workspace state is unavailable") from exc
        workspace.update(relative.encode("utf-8"))
        workspace.update(
            f"\0{stat.st_mode}\0{stat.st_size}\0{stat.st_mtime_ns}\0{stat.st_ino}\0".encode(
                "ascii"
            )
        )
        if path.is_symlink():
            workspace.update(os.readlink(path).encode("utf-8"))
        elif path.is_file():
            workspace.update(hashlib.sha256(path.read_bytes()).digest())
        else:
            raise AutomationError("tracked workspace entry must be a file")

    def file_sha256(path: Path, label: str) -> str:
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise AutomationError(f"{label} is unavailable") from exc

    return {
        "source_tree_sha256": hashlib.sha256(
            ("\n".join(retained) + "\n").encode("utf-8")
        ).hexdigest(),
        "workspace_state_sha256": workspace.hexdigest(),
        "series_config_sha256": file_sha256(
            settings.root / "config" / "series.json", "series config"
        ),
        "automation_config_sha256": file_sha256(
            settings.config_path, "automation config"
        ),
    }


def _cache_candidate(
    settings: AutomationSettings,
    raw: bytes,
    *,
    target: datetime,
    context: Mapping[str, str],
) -> None:
    payload = validate_automation_candidate(raw, target=target)
    generation_id = str(payload.get("meta", {}).get("generation_id", ""))
    if not generation_id:
        raise AutomationError("candidate generation_id is missing")
    settings.candidate_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = settings.candidate_path.with_name(".regime-results.json.tmp")
    temporary.write_bytes(raw)
    os.chmod(temporary, 0o600)
    os.replace(temporary, settings.candidate_path)
    write_json_atomic(
        settings.candidate_metadata_path,
        {
            "schema_version": 2,
            "automation_id": settings.automation_id,
            "data_as_of": target.astimezone(UTC).isoformat(),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "generation_id": generation_id,
            **context,
            "cached_at": datetime.now(UTC).isoformat(),
        },
    )


def _load_cached_candidate(
    settings: AutomationSettings,
    *,
    target: datetime,
    context: Mapping[str, str],
) -> bytes | None:
    try:
        metadata = json.loads(
            settings.candidate_metadata_path.read_text(encoding="utf-8")
        )
        raw = settings.candidate_path.read_bytes()
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 2:
        return None
    if metadata.get("automation_id") != settings.automation_id:
        return None
    if metadata.get("data_as_of") != target.astimezone(UTC).isoformat():
        return None
    if metadata.get("sha256") != hashlib.sha256(raw).hexdigest():
        return None
    if any(metadata.get(key) != value for key, value in context.items()):
        return None
    payload = validate_automation_candidate(raw, target=target)
    if metadata.get("generation_id") != payload.get("meta", {}).get("generation_id"):
        return None
    _verify_candidate_package(settings, settings.candidate_path)
    return raw


def _build_candidate(
    settings: AutomationSettings,
    *,
    target: datetime,
    context: Mapping[str, str],
) -> bytes:
    _run(
        [
            sys.executable,
            "-m",
            "regime_lab",
            "build",
            "--config",
            "config/series.json",
            "--database",
            str(settings.database),
            "--output",
            str(settings.payload),
            "--artifacts",
            str(settings.artifacts),
            "--profile",
            settings.profile,
            "--alfred-rights-confirmed",
        ],
        cwd=settings.root,
    )
    if _candidate_context(settings) != dict(context):
        raise AutomationError("tracked source or config changed during live build")
    _sqlite_quick_check(settings.database)
    _run(
        [sys.executable, "-m", "regime_lab", "validate", str(settings.payload)],
        cwd=settings.root,
    )
    _run(
        [
            sys.executable,
            "scripts/audit_outputs.py",
            "--payload",
            str(settings.payload),
            "--artifacts",
            str(settings.artifacts),
            "--mode",
            "live",
        ],
        cwd=settings.root,
    )
    raw = settings.payload.read_bytes()
    validate_automation_candidate(raw, target=target)
    _verify_candidate_package(settings, settings.payload)
    if _candidate_context(settings) != dict(context):
        raise AutomationError("tracked source or config changed during live audit")
    _cache_candidate(settings, raw, target=target, context=context)
    return raw


def _publish_candidate(
    settings: AutomationSettings,
    *,
    candidate: bytes,
    target: datetime,
    expected_head_sha: str,
    force_pages_rebuild: bool = False,
) -> str:
    origin_url = _run(
        ["git", "remote", "get-url", settings.remote],
        cwd=settings.root,
        capture=True,
    ).decode().strip()
    settings.state_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".release-checkout-", dir=settings.state_directory
    ) as temporary:
        checkout = Path(temporary) / "repo"
        _run(
            [
                "git",
                "clone",
                "--quiet",
                "--depth",
                "1",
                "--branch",
                settings.branch,
                origin_url,
                str(checkout),
            ],
            cwd=settings.root,
        )
        cloned_head = _run(
            ["git", "rev-parse", "HEAD"], cwd=checkout, capture=True
        ).decode().strip()
        if cloned_head != expected_head_sha:
            raise AutomationError(
                "remote main changed after preflight; retry from a fresh preflight"
            )
        target_path = checkout / PUBLICATION_PATH
        current = checkout
        for part in Path(PUBLICATION_PATH).parent.parts:
            current = current / part
            if current.is_symlink() or not current.is_dir():
                raise AutomationError("publication parent must be a regular directory")
        if target_path.is_symlink() or not target_path.is_file():
            raise AutomationError("publication target must be an existing regular file")
        target_path.write_bytes(candidate)
        os.chmod(target_path, 0o644)
        changed = set(
            _run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=checkout,
                capture=True,
            ).decode().splitlines()
        )
        expected = {f" M {PUBLICATION_PATH}"}
        if changed != expected:
            if not changed and target_path.read_bytes() == candidate:
                if not force_pages_rebuild:
                    return cloned_head
                cutoff_date = target.astimezone(EASTERN).date().isoformat()
                _run(
                    [
                        "git",
                        "-c",
                        "user.name=Regime Automation",
                        "-c",
                        "user.email=regime-automation@users.noreply.github.com",
                        "commit",
                        "--allow-empty",
                        "--quiet",
                        "-m",
                        f"Retry Regime Pages through {cutoff_date}",
                        "-m",
                        AUTOMATION_TRAILER,
                    ],
                    cwd=checkout,
                )
                commit_sha = _run(
                    ["git", "rev-parse", "HEAD"], cwd=checkout, capture=True
                ).decode().strip()
                _run(
                    ["git", "push", "--quiet", "origin", f"HEAD:{settings.branch}"],
                    cwd=checkout,
                )
                return commit_sha
            raise AutomationError("release checkout changed outside the publication snapshot")
        _run(["git", "add", "--", PUBLICATION_PATH], cwd=checkout)
        staged = set(
            _run(
                ["git", "diff", "--cached", "--name-only"],
                cwd=checkout,
                capture=True,
            ).decode().splitlines()
        )
        if staged != {PUBLICATION_PATH}:
            raise AutomationError("release commit contains a non-publication path")
        cutoff_date = target.astimezone(EASTERN).date().isoformat()
        _run(
            [
                "git",
                "-c",
                "user.name=Regime Automation",
                "-c",
                "user.email=regime-automation@users.noreply.github.com",
                "commit",
                "--quiet",
                "-m",
                f"Refresh Regime snapshot through {cutoff_date}",
                "-m",
                AUTOMATION_TRAILER,
            ],
            cwd=checkout,
        )
        commit_sha = _run(
            ["git", "rev-parse", "HEAD"], cwd=checkout, capture=True
        ).decode().strip()
        _run(
            ["git", "push", "--quiet", "origin", f"HEAD:{settings.branch}"],
            cwd=checkout,
        )
        return commit_sha


def _workflow_url(settings: AutomationSettings) -> str:
    return (
        f"https://github.com/{settings.repository}/actions/workflows/"
        f"{settings.workflow}"
    )


def _fetch_url(url: str) -> bytes:
    cache_buster = urlencode({"automation_check": str(time.time_ns())})
    separator = "&" if "?" in url else "?"
    request = Request(
        f"{url}{separator}{cache_buster}",
        headers={"Cache-Control": "no-cache", "User-Agent": "regime-weekly-automation/1"},
    )
    with urlopen(request, timeout=30) as response:
        if getattr(response, "status", 200) != 200:
            raise AutomationError(f"public readback returned HTTP {response.status}")
        return response.read()


def verify_public_readback(
    settings: AutomationSettings,
    *,
    expected_payload: bytes,
    fetch: Callable[[str], bytes] = _fetch_url,
) -> None:
    public_payload = fetch(urljoin(settings.public_root, PUBLIC_PAYLOAD_PATH))
    expected_hash = hashlib.sha256(expected_payload).hexdigest()
    if hashlib.sha256(public_payload).hexdigest() != expected_hash:
        raise AutomationError("public payload SHA-256 does not match the promoted result")
    manifest = _json_object(
        fetch(urljoin(settings.public_root, PUBLIC_MANIFEST_PATH)),
        label="public publication manifest",
    )
    record = manifest.get("files", {}).get(PUBLIC_PAYLOAD_PATH, {})
    if record.get("sha256") != expected_hash:
        raise AutomationError("public manifest does not identify the promoted payload")
    expected = _json_object(expected_payload, label="expected public payload")
    if manifest.get("payload_data_as_of") != expected.get("meta", {}).get("data_as_of"):
        raise AutomationError("public manifest data_as_of does not match the payload")
    html = fetch(settings.public_root)
    if b"US Market Regime Lab" not in html or b"app.js" not in html:
        raise AutomationError("public dashboard HTML is not the Regime consumer")


def _wait_for_public_readback(
    settings: AutomationSettings,
    *,
    expected_payload: bytes,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    deadline = time.monotonic() + (
        settings.workflow_timeout + settings.public_readback_timeout
    ).total_seconds()
    last_error: AutomationError | None = None
    while time.monotonic() < deadline:
        try:
            verify_public_readback(settings, expected_payload=expected_payload)
            return
        except (AutomationError, OSError) as exc:
            last_error = exc if isinstance(exc, AutomationError) else AutomationError(str(exc))
            sleep(15)
    raise AutomationError(
        f"public Pages readback timed out: {last_error or 'no response'}"
    )


def preflight_summary(
    settings: AutomationSettings,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = (now or datetime.now(UTC)).astimezone(UTC)
    target, old_enough = target_cutoff(
        current, minimum_age=settings.minimum_cutoff_age
    )
    remote = _git_preflight(settings)
    due = old_enough and remote.data_as_of < target
    return {
        "ok": True,
        "automation_id": settings.automation_id,
        "now": current.isoformat(),
        "target_data_as_of": target.isoformat(),
        "remote_data_as_of": remote.data_as_of.isoformat(),
        "cutoff_age_ready": old_enough,
        "due": due,
        "remote_head_sha": remote.head_sha,
    }


def run_weekly_release(
    settings: AutomationSettings,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    started = (now or datetime.now(UTC)).astimezone(UTC)
    target, old_enough = target_cutoff(
        started, minimum_age=settings.minimum_cutoff_age
    )
    settings.state_directory.mkdir(parents=True, exist_ok=True)
    with automation_lock(settings.lock_path):
        _write_status(
            settings,
            status="running",
            stage="preflight",
            started_at=started,
            target=target,
        )
        try:
            if not old_enough:
                return _write_status(
                    settings,
                    status="skipped",
                    stage="not_due",
                    started_at=started,
                    target=target,
                    detail="completed cutoff has not reached the configured provider lag",
                )
            remote = _git_preflight(settings)
            if remote.data_as_of > target:
                raise AutomationError("remote publication is ahead of the due cutoff")
            if remote.data_as_of == target:
                _write_status(
                    settings,
                    status="running",
                    stage="deployment_recovery",
                    started_at=started,
                    target=target,
                    commit_sha=remote.head_sha,
                )
                try:
                    verify_public_readback(
                        settings, expected_payload=remote.payload_bytes
                    )
                    workflow_url = None
                except (AutomationError, OSError):
                    recovery_sha = _publish_candidate(
                        settings,
                        candidate=remote.payload_bytes,
                        target=target,
                        expected_head_sha=remote.head_sha,
                        force_pages_rebuild=True,
                    )
                    workflow_url = _workflow_url(settings)
                    _wait_for_public_readback(
                        settings, expected_payload=remote.payload_bytes
                    )
                    remote = RemotePublication(
                        recovery_sha,
                        remote.payload_bytes,
                        remote.data_as_of,
                    )
                return _write_status(
                    settings,
                    status="succeeded",
                    stage="already_current",
                    started_at=started,
                    target=target,
                    detail="remote publication and public Pages are current",
                    commit_sha=remote.head_sha,
                    workflow_url=workflow_url,
                )

            _validate_local_authorization(settings, now=started)
            _ensure_ac_power(settings)
            context = _candidate_context(settings)
            candidate = _load_cached_candidate(
                settings, target=target, context=context
            )
            if candidate is None:
                _alpha_quota_preflight(settings, target=target)
                _write_status(
                    settings,
                    status="running",
                    stage="collect_train_audit",
                    started_at=started,
                    target=target,
                )
                candidate = _build_candidate(
                    settings, target=target, context=context
                )
            else:
                _write_status(
                    settings,
                    status="running",
                    stage="resume_validated_candidate",
                    started_at=started,
                    target=target,
                    detail="reusing the validated candidate without provider access",
                )

            remote = _git_preflight(settings)
            if remote.data_as_of > target:
                raise AutomationError("remote publication is ahead of the due cutoff")
            if remote.data_as_of == target:
                if remote.payload_bytes != candidate:
                    candidate = remote.payload_bytes
                commit_sha = remote.head_sha
            else:
                _write_status(
                    settings,
                    status="running",
                    stage="publish_snapshot",
                    started_at=started,
                    target=target,
                )
                commit_sha = _publish_candidate(
                    settings,
                    candidate=candidate,
                    target=target,
                    expected_head_sha=remote.head_sha,
                )

            _write_status(
                settings,
                status="running",
                stage="wait_for_pages",
                started_at=started,
                target=target,
                commit_sha=commit_sha,
            )
            workflow_url = _workflow_url(settings)
            _wait_for_public_readback(settings, expected_payload=candidate)
            return _write_status(
                settings,
                status="succeeded",
                stage="public_readback_verified",
                started_at=started,
                target=target,
                detail="collection, training, audit, promotion, Pages, and public readback passed",
                commit_sha=commit_sha,
                workflow_url=workflow_url,
            )
        except BaseException as exc:
            _write_status(
                settings,
                status="failed",
                stage="failed",
                started_at=started,
                target=target,
                detail=_safe_error(exc),
            )
            raise


def launch_agent_document(settings: AutomationSettings) -> dict[str, Any]:
    python = settings.root / ".venv" / "bin" / "python"
    if not python.is_file():
        raise AutomationError(f"project virtualenv Python is missing: {python}")
    settings.state_directory.mkdir(parents=True, exist_ok=True)
    return {
        "Label": AUTOMATION_LABEL,
        "ProgramArguments": [
            "/usr/bin/caffeinate",
            "-s",
            str(python),
            "-m",
            "regime_lab",
            "automation",
            "run",
            "--config",
            str(settings.config_path),
        ],
        "WorkingDirectory": str(settings.root),
        "EnvironmentVariables": {
            "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
            "PYTHONUNBUFFERED": "1",
        },
        "StartCalendarInterval": {
            "Hour": settings.schedule_hour,
            "Minute": settings.schedule_minute,
        },
        "RunAtLoad": True,
        "ProcessType": "Background",
        "LowPriorityIO": True,
        "Umask": 0o077,
        "ThrottleInterval": 300,
        "StandardOutPath": str(settings.state_directory / "launchd.stdout.log"),
        "StandardErrorPath": str(settings.state_directory / "launchd.stderr.log"),
    }


def _launch_agent_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{AUTOMATION_LABEL}.plist"


def _launch_agent_loaded(settings: AutomationSettings) -> bool:
    service = f"gui/{os.getuid()}/{AUTOMATION_LABEL}"
    completed = subprocess.run(
        ["/bin/launchctl", "print", service],
        cwd=settings.root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0


def _bootout_launch_agent_if_loaded(settings: AutomationSettings) -> None:
    if not _launch_agent_loaded(settings):
        return
    service = f"gui/{os.getuid()}/{AUTOMATION_LABEL}"
    _run(["/bin/launchctl", "bootout", service], cwd=settings.root)
    if _launch_agent_loaded(settings):
        raise AutomationError("LaunchAgent remained loaded after bootout")


def _ensure_installable_checkout(settings: AutomationSettings) -> None:
    try:
        config_relative = settings.config_path.relative_to(settings.root).as_posix()
    except ValueError as exc:
        raise AutomationError("LaunchAgent config must be tracked inside the project") from exc
    required = (
        config_relative,
        "src/regime_lab/automation.py",
        "src/regime_lab/cli.py",
        ".github/workflows/pages.yml",
    )
    _run(
        ["git", "ls-files", "--error-unmatch", "--", *required],
        cwd=settings.root,
        capture=True,
    )
    _git_preflight(settings)


def install_launch_agent(
    settings: AutomationSettings,
    *,
    alfred_rights_confirmed: bool,
    personal_noncommercial_publication_acknowledged: bool,
) -> dict[str, Any]:
    if sys.platform != "darwin":
        raise AutomationError("LaunchAgent installation requires macOS")
    with automation_lock(settings.install_lock_path):
        with automation_lock(settings.lock_path):
            _ensure_installable_checkout(settings)
            _write_local_authorization(
                settings,
                alfred_rights_confirmed=alfred_rights_confirmed,
                personal_noncommercial_publication_acknowledged=(
                    personal_noncommercial_publication_acknowledged
                ),
            )
            document = launch_agent_document(settings)
            target = _launch_agent_path()
            target.parent.mkdir(parents=True, exist_ok=True)
            raw = plistlib.dumps(document, fmt=plistlib.FMT_XML, sort_keys=False)
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=target.parent, prefix=f".{target.name}.", delete=False
            ) as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
                temporary = Path(handle.name)
            os.chmod(temporary, 0o644)
            service = f"gui/{os.getuid()}/{AUTOMATION_LABEL}"
            try:
                _run(["/usr/bin/plutil", "-lint", str(temporary)], cwd=settings.root)
                _bootout_launch_agent_if_loaded(settings)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        # RunAtLoad must start after the weekly lock is released, otherwise the
        # catch-up process exits as already_running and waits until the next day.
        _run(["/bin/launchctl", "enable", service], cwd=settings.root)
        _run(
            [
                "/bin/launchctl",
                "bootstrap",
                f"gui/{os.getuid()}",
                str(target),
            ],
            cwd=settings.root,
        )
        _run(["/bin/launchctl", "kickstart", service], cwd=settings.root)
        if not _launch_agent_loaded(settings):
            raise AutomationError("LaunchAgent did not remain loaded after installation")
    return {
        "ok": True,
        "installed": True,
        "label": AUTOMATION_LABEL,
        "path": str(target),
        "schedule": (
            f"daily {settings.schedule_hour:02d}:"
            f"{settings.schedule_minute:02d} local time"
        ),
        "runs_immediately": True,
    }


def uninstall_launch_agent(settings: AutomationSettings) -> dict[str, Any]:
    with automation_lock(settings.install_lock_path):
        with automation_lock(settings.lock_path):
            target = _launch_agent_path()
            _bootout_launch_agent_if_loaded(settings)
            if target.exists():
                target.unlink()
            if _launch_agent_loaded(settings):
                raise AutomationError("LaunchAgent uninstall could not verify unload")
    return {"ok": True, "installed": False, "label": AUTOMATION_LABEL}


def launch_agent_status(settings: AutomationSettings) -> dict[str, Any]:
    target = _launch_agent_path()
    health: Mapping[str, Any] | None = None
    try:
        loaded = json.loads(settings.status_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            health = loaded
    except (OSError, json.JSONDecodeError):
        pass
    return {
        "ok": True,
        "installed": target.is_file(),
        "loaded": _launch_agent_loaded(settings),
        "label": AUTOMATION_LABEL,
        "path": str(target),
        "health": health,
    }


def command_automation(args: argparse.Namespace) -> int:
    try:
        settings = AutomationSettings.load(args.config)
        if args.action == "preflight":
            result = preflight_summary(settings)
        elif args.action == "run":
            result = run_weekly_release(settings)
        elif args.action == "install":
            result = install_launch_agent(
                settings,
                alfred_rights_confirmed=args.alfred_rights_confirmed,
                personal_noncommercial_publication_acknowledged=(
                    args.acknowledge_personal_noncommercial_publication
                ),
            )
        elif args.action == "uninstall":
            result = uninstall_launch_agent(settings)
        elif args.action == "status":
            result = launch_agent_status(settings)
        else:  # pragma: no cover - argparse owns this boundary.
            raise AutomationError(f"unsupported automation action: {args.action}")
    except AlreadyRunning as exc:
        if args.action == "run":
            result = {
                "ok": True,
                "status": "already_running",
                "detail": str(exc),
            }
        else:
            print(
                json.dumps(
                    {"ok": False, "status": "failed", "error": _safe_error(exc)},
                    ensure_ascii=False,
                    indent=2,
                ),
                file=sys.stderr,
            )
            return 1
    except Exception as exc:
        print(
            json.dumps(
                {"ok": False, "status": "failed", "error": _safe_error(exc)},
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


__all__ = [
    "AUTOMATION_LABEL",
    "AlreadyRunning",
    "AutomationError",
    "AutomationSettings",
    "RemotePublication",
    "automation_lock",
    "command_automation",
    "launch_agent_document",
    "preflight_summary",
    "run_weekly_release",
    "target_cutoff",
    "validate_automation_candidate",
    "verify_public_readback",
]
