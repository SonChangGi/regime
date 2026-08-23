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
from datetime import date, datetime, timedelta, timezone
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
import uuid
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

from regime_lab.collection import (
    EASTERN,
    alpha_market_week_is_current,
    last_completed_week_cutoff,
)
from regime_lab.config import project_root
from regime_lab.io import write_json_atomic
from regime_lab.keychain import KEYCHAIN_SERVICES
from regime_lab.path_safety import confined_mutable_path
from regime_lab.schema import ContractError, validate_dashboard_payload
from regime_lab.data import DailyRequestBudget, SQLiteSnapshotStore


UTC = timezone.utc
PUBLICATION_PATH = "publication/live/regime-results.json"
PUBLICATION_COMPARISON_PATH = "publication/live/v5-vs-v4-comparison.json"
PUBLIC_PAYLOAD_PATH = "data/regime-results.json"
PUBLIC_COMPARISON_PATH = "data/v5-vs-v4-comparison.json"
PUBLIC_MANIFEST_PATH = "publication-manifest.json"
AUTOMATION_LABEL = "com.sonchanggi.regime.weekly-release"
AUTOMATION_TRAILER = "Regime-Automation: weekly-release-v1"
ALLOWED_REMOTE_DRIFT = frozenset(
    {PUBLICATION_PATH, PUBLICATION_COMPARISON_PATH}
)
DEFAULT_RETRY_HOURS = (3, 9, 15, 21)
HEALTH_SCHEMA_VERSION = 2
GIT_NETWORK_TIMEOUT_SECONDS = 300.0
GIT_NONINTERACTIVE_ENV = {
    "GIT_TERMINAL_PROMPT": "0",
    "GCM_INTERACTIVE": "Never",
}


class AutomationError(RuntimeError):
    """Raised when a weekly run must stop without changing the public result."""


class AlreadyRunning(AutomationError):
    """Raised when another local weekly run owns the process lock."""


class ScheduledRetry(AutomationError):
    """An operational failure with a bounded, provider-safe retry time."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str,
        retry_class: str,
        next_retry_at: datetime | None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.retry_class = retry_class
        self.next_retry_at = (
            next_retry_at.astimezone(UTC) if next_retry_at is not None else None
        )


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
    contract: str = "v5"
    retry_hours: tuple[int, ...] = ()
    transient_retry_delay: timedelta = timedelta(hours=6)
    heartbeat_interval: timedelta = timedelta(minutes=2)
    stale_heartbeat_after: timedelta = timedelta(minutes=15)
    notification_dedupe: timedelta = timedelta(hours=24)

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
            retry_hours_value = schedule.get("retry_hours", DEFAULT_RETRY_HOURS)
            if not isinstance(retry_hours_value, list):
                raise TypeError("retry_hours")
            retry_hours = tuple(sorted({int(value) for value in retry_hours_value}))
            transient_retry_hours = int(schedule.get("transient_retry_hours", 6))
            heartbeat_seconds = int(schedule.get("heartbeat_interval_seconds", 120))
            stale_heartbeat_minutes = int(schedule.get("stale_heartbeat_minutes", 15))
            notification_dedupe_hours = int(
                schedule.get("notification_dedupe_hours", 24)
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AutomationError("automation config fields are invalid") from exc
        if not 0 <= hour <= 23 or not 0 <= minute <= 59:
            raise AutomationError("automation schedule time is invalid")
        if not retry_hours or any(value < 0 or value > 23 for value in retry_hours):
            raise AutomationError("automation retry hours are invalid")
        if minimum_age < 1 or workflow_minutes < 1 or readback_minutes < 1:
            raise AutomationError("automation timing values must be positive")
        if (
            transient_retry_hours < 1
            or heartbeat_seconds < 30
            or stale_heartbeat_minutes < 2
            or notification_dedupe_hours < 1
        ):
            raise AutomationError("automation retry and health timing values are invalid")
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
        contract = str(build.get("contract", ""))
        if contract != "v5":
            raise AutomationError("weekly live automation must use the v5 contract")
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
            contract=contract,
            retry_hours=retry_hours,
            transient_retry_delay=timedelta(hours=transient_retry_hours),
            heartbeat_interval=timedelta(seconds=heartbeat_seconds),
            stale_heartbeat_after=timedelta(minutes=stale_heartbeat_minutes),
            notification_dedupe=timedelta(hours=notification_dedupe_hours),
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

    @property
    def candidate_comparison_path(self) -> Path:
        return self.state_directory / "candidate" / "v5-vs-v4-comparison.json"

    @property
    def reviewed_payload_path(self) -> Path:
        return self.payload.with_name("reviewed-regime-results.json")

    @property
    def comparison_path(self) -> Path:
        return self.payload.with_name("v5-vs-v4-comparison.json")

    @property
    def collection_report_path(self) -> Path:
        return self.state_directory / "collection-report.json"

    @property
    def notification_state_path(self) -> Path:
        return self.state_directory / "notification-state.json"


@dataclass(frozen=True)
class RemotePublication:
    head_sha: str
    payload_bytes: bytes
    data_as_of: datetime
    comparison_bytes: bytes | None = None

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


def _validate_v5_comparison_bytes(
    payload_raw: bytes,
    comparison_raw: bytes,
    *,
    label: str,
) -> None:
    """Validate a derived V5 comparison and its exact payload-byte binding."""

    payload = _json_object(payload_raw, label=f"{label} payload")
    comparison = _json_object(comparison_raw, label=f"{label} comparison")
    try:
        from scripts.package_public_demo import (
            PackagingError,
            validate_v5_comparison_sidecar,
        )
    except ImportError as exc:
        raise AutomationError(
            f"{label} comparison validator is unavailable: {exc}"
        ) from exc
    try:
        validate_v5_comparison_sidecar(
            comparison,
            payload=payload,
            payload_raw=payload_raw,
        )
    except (PackagingError, KeyError, TypeError, ValueError) as exc:
        raise AutomationError(f"{label} comparison contract failed: {exc}") from exc


def _run(
    args: Sequence[str],
    *,
    cwd: Path,
    capture: bool = False,
    timeout: float | None = None,
    heartbeat: Callable[[], None] | None = None,
    heartbeat_interval: float = 120.0,
    env: Mapping[str, str] | None = None,
) -> bytes:
    process_env = None
    if env is not None:
        process_env = os.environ.copy()
        process_env.update(env)
    if heartbeat is None:
        try:
            completed = subprocess.run(
                list(args),
                cwd=cwd,
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE if capture else None,
                stderr=subprocess.PIPE if capture else None,
                timeout=timeout,
                env=process_env,
            )
        except subprocess.TimeoutExpired as exc:
            label = Path(args[0]).name if args else "command"
            raise AutomationError(
                f"{label} timed out after {timeout:g} seconds"
            ) from exc
    else:
        process = subprocess.Popen(
            list(args),
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
            env=process_env,
        )
        deadline = time.monotonic() + timeout if timeout is not None else None
        while True:
            wait_for = heartbeat_interval
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    process.kill()
                    process.wait()
                    label = Path(args[0]).name if args else "command"
                    raise AutomationError(
                        f"{label} timed out after {timeout:g} seconds"
                    )
                wait_for = min(wait_for, remaining)
            try:
                stdout, stderr = process.communicate(timeout=wait_for)
                break
            except subprocess.TimeoutExpired:
                heartbeat()
        completed = subprocess.CompletedProcess(
            list(args), process.returncode, stdout or b"", stderr or b""
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
    run_id: str | None = None,
    error_code: str | None = None,
    retry_class: str | None = None,
    next_retry_at: datetime | None = None,
    recovery_fingerprint: str | None = None,
    notification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    now = datetime.now(UTC)
    previous = _read_status(settings)
    previous_run_id = previous.get("run_id")
    selected_run_id = run_id or (
        str(previous_run_id)
        if previous.get("started_at") == started_at.isoformat() and previous_run_id
        else uuid.uuid4().hex
    )
    stage_started_at = (
        previous.get("stage_started_at")
        if previous.get("run_id") == selected_run_id and previous.get("stage") == stage
        else now.isoformat()
    )
    previous_failures = previous.get("consecutive_failures", 0)
    if type(previous_failures) is not int or previous_failures < 0:
        previous_failures = 0
    consecutive_failures = previous_failures
    last_success_at = previous.get("last_success_at")
    last_success_target = previous.get("last_success_target")
    last_failure_at = previous.get("last_failure_at")
    public_data_as_of = previous.get("public_data_as_of") or _local_public_data_as_of(
        settings
    )
    if status == "succeeded":
        consecutive_failures = 0
        last_success_at = now.isoformat()
        last_success_target = target.isoformat() if target else None
        public_data_as_of = target.isoformat() if target else public_data_as_of
    elif status == "failed":
        consecutive_failures += 1
        last_failure_at = now.isoformat()
    return {
        "schema_version": HEALTH_SCHEMA_VERSION,
        "automation_id": settings.automation_id,
        "status": status,
        "stage": stage,
        "run_id": selected_run_id,
        "pid": os.getpid(),
        "started_at": started_at.isoformat(),
        "stage_started_at": stage_started_at,
        "heartbeat_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "target_data_as_of": target.isoformat() if target else None,
        "detail": detail,
        "error_code": error_code,
        "retry_class": retry_class,
        "next_retry_at": next_retry_at.astimezone(UTC).isoformat()
        if next_retry_at
        else None,
        "recovery_fingerprint": recovery_fingerprint,
        "last_success_at": last_success_at,
        "last_success_target": last_success_target,
        "last_failure_at": last_failure_at,
        "consecutive_failures": consecutive_failures,
        "commit_sha": commit_sha,
        "workflow_url": workflow_url,
        "public_url": settings.public_root,
        "public_data_as_of": public_data_as_of,
        "notification": dict(notification) if notification is not None else None,
    }


def _write_status(settings: AutomationSettings, **values: Any) -> dict[str, Any]:
    document = _status_document(settings, **values)
    write_json_atomic(settings.status_path, document)
    return document


def _read_status(settings: AutomationSettings) -> dict[str, Any]:
    try:
        value = json.loads(settings.status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _local_public_data_as_of(settings: AutomationSettings) -> str | None:
    try:
        payload = json.loads(
            (settings.root / PUBLICATION_PATH).read_text(encoding="utf-8")
        )
        value = payload.get("meta", {}).get("data_as_of")
        return _parse_aware_datetime(value, label="public data_as_of").isoformat()
    except (OSError, json.JSONDecodeError, AutomationError, AttributeError):
        return None


def _heartbeat_status(
    settings: AutomationSettings,
    *,
    stage: str | None = None,
) -> None:
    document = _read_status(settings)
    if not document:
        return
    now = datetime.now(UTC).isoformat()
    if stage is not None and document.get("stage") != stage:
        document["stage"] = stage
        document["stage_started_at"] = now
    document["heartbeat_at"] = now
    document["updated_at"] = now
    write_json_atomic(settings.status_path, document)


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


def _recovery_fingerprint(settings: AutomationSettings) -> str:
    """Fingerprint local state whose change may legitimately clear a block."""

    digest = hashlib.sha256()
    for path in (settings.config_path, settings.authorization):
        digest.update(str(path).encode("utf-8"))
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"<missing>")
    for args in (
        ["git", "rev-parse", "HEAD"],
        ["git", "branch", "--show-current"],
        ["git", "remote", "get-url", settings.remote],
        ["git", "status", "--porcelain", "--untracked-files=all"],
    ):
        try:
            digest.update(_run(args, cwd=settings.root, capture=True))
        except AutomationError:
            digest.update(b"<git-unavailable>")
    for path in (
        settings.database,
        settings.payload,
        settings.artifacts,
        settings.artifacts / "build-generation.json",
        settings.candidate_path,
        settings.candidate_metadata_path,
    ):
        digest.update(str(path).encode("utf-8"))
        try:
            stat = path.lstat()
            digest.update(
                f"{stat.st_mode}:{stat.st_size}:{stat.st_mtime_ns}:{stat.st_ino}".encode(
                    "ascii"
                )
            )
        except OSError:
            digest.update(b"<missing>")
    if sys.platform == "darwin" and settings.root == project_root().resolve():
        for service in sorted(KEYCHAIN_SERVICES.values()):
            try:
                completed = subprocess.run(
                    ["/usr/bin/security", "find-generic-password", "-s", service],
                    cwd=settings.root,
                    check=False,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                )
                returncode: int | str = completed.returncode
            except (OSError, subprocess.TimeoutExpired):
                returncode = "unavailable"
            digest.update(f"{service}:{returncode}".encode("utf-8"))
    return digest.hexdigest()


def _failure_policy(
    exc: BaseException,
    *,
    stage: str,
    now: datetime,
    settings: AutomationSettings,
) -> tuple[str, str, datetime | None]:
    if isinstance(exc, ScheduledRetry):
        return exc.error_code, exc.retry_class, exc.next_retry_at
    message = str(exc).lower()
    blocked_markers = (
        "authorization",
        "working tree",
        "requires branch",
        "remote does not match",
        "source changed",
        "tracked source or config",
        "candidate dashboard contract",
        "audit",
        "package",
        "schema",
        "permission",
        "keychain",
    )
    if "ac power" in message:
        return "ac_power_unavailable", "transient", now + settings.transient_retry_delay
    if any(marker in message for marker in blocked_markers):
        return f"{stage}_blocked", "blocked", None
    if stage in {"collect_train_audit", "train_models", "audit_candidate"}:
        return "analysis_build_failed", "blocked", None
    if stage in {"publish_snapshot", "wait_for_pages", "deployment_recovery"}:
        return f"{stage}_failed", "resume", None
    return f"{stage}_failed", "transient", now + settings.transient_retry_delay


def _retry_guard(
    settings: AutomationSettings,
    *,
    target: datetime,
    now: datetime,
    force_transient_retry: bool = False,
) -> dict[str, Any] | None:
    previous = _read_status(settings)
    if previous.get("schema_version") != HEALTH_SCHEMA_VERSION:
        return None
    if previous.get("target_data_as_of") != target.astimezone(UTC).isoformat():
        return None
    failures = previous.get("consecutive_failures", 0)
    if type(failures) is not int or failures < 1:
        return None
    retry_class = previous.get("retry_class")
    error_code = previous.get("error_code")
    fingerprint = previous.get("recovery_fingerprint")
    if retry_class == "blocked" and fingerprint == _recovery_fingerprint(settings):
        return _write_status(
            settings,
            status="blocked",
            stage="retry_blocked",
            started_at=now,
            target=target,
            detail="automatic retry remains blocked until local code, config, or authorization changes",
            error_code=str(error_code or "blocked"),
            retry_class="blocked",
            recovery_fingerprint=str(fingerprint),
        )
    if retry_class == "transient" and force_transient_retry:
        # Explicit operator recovery may bypass only a transient delay. Quota
        # and blocked failures retain their normal fail-closed guards, and the
        # run still performs every provider, AC, Git, and publication preflight.
        return None
    raw_next = previous.get("next_retry_at")
    if retry_class in {"transient", "quota"} and raw_next:
        try:
            next_retry = _parse_aware_datetime(raw_next, label="next_retry_at")
        except AutomationError:
            return None
        if now < next_retry:
            return _write_status(
                settings,
                status="skipped",
                stage="retry_backoff",
                started_at=now,
                target=target,
                detail="provider and model execution skipped until the bounded retry time",
                error_code=str(error_code or "retry_backoff"),
                retry_class=str(retry_class),
                next_retry_at=next_retry,
                recovery_fingerprint=str(fingerprint) if fingerprint else None,
            )
    return None


def _notifications_enabled(settings: AutomationSettings) -> bool:
    return (
        sys.platform == "darwin"
        and settings.root == project_root().resolve()
        and _launch_agent_path().is_file()
    )


def _send_macos_notification(title: str, message: str) -> None:
    script = (
        f"display notification {json.dumps(message)} "
        f"with title {json.dumps(title)}"
    )
    completed = subprocess.run(
        ["/usr/bin/osascript", "-e", script],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=10,
    )
    if completed.returncode != 0:
        raise AutomationError("macOS notification delivery failed")


def _notification_state(settings: AutomationSettings) -> dict[str, Any]:
    try:
        value = json.loads(settings.notification_state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": 1}
    return value if isinstance(value, dict) else {"schema_version": 1}


def _record_notification(
    settings: AutomationSettings,
    *,
    kind: str,
    target: datetime,
    error_code: str | None = None,
    retry_class: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = (now or datetime.now(UTC)).astimezone(UTC)
    state = _notification_state(settings)
    active = state.get("active_failure_signature")
    if kind == "failure":
        signature = hashlib.sha256(
            f"{target.astimezone(UTC).isoformat()}|{error_code}|{retry_class}".encode(
                "utf-8"
            )
        ).hexdigest()
        raw_notified = state.get("last_failure_notified_at")
        last_notified: datetime | None = None
        if raw_notified:
            try:
                last_notified = _parse_aware_datetime(
                    raw_notified, label="last_failure_notified_at"
                )
            except AutomationError:
                pass
        should_send = active != signature or (
            last_notified is None or current - last_notified >= settings.notification_dedupe
        )
        state["active_failure_signature"] = signature
        title = "Regime weekly automation needs attention"
        message = (
            f"target={target.date().isoformat()}, code={error_code or 'failed'}, "
            f"retry={retry_class or 'blocked'}"
        )
    else:
        signature = None
        should_send = bool(active)
        state["active_failure_signature"] = None
        title = "Regime weekly automation recovered"
        message = f"target={target.date().isoformat()}, public snapshot verified"
    delivery = "deduplicated"
    if should_send and _notifications_enabled(settings):
        try:
            _send_macos_notification(title, message)
            delivery = "delivered"
        except Exception:
            delivery = "failed_nonfatal"
    elif should_send:
        delivery = "not_available"
    if kind == "failure" and should_send:
        state["last_failure_notified_at"] = current.isoformat()
    if kind == "recovery" and should_send:
        state["last_recovery_notified_at"] = current.isoformat()
    state.update(
        {
            "schema_version": 1,
            "updated_at": current.isoformat(),
            "last_delivery": delivery,
            "last_kind": kind,
        }
    )
    write_json_atomic(settings.notification_state_path, state)
    return {"kind": kind, "delivery": delivery, "signature": signature}


def _attach_notification(
    settings: AutomationSettings, notification: Mapping[str, Any]
) -> None:
    document = _read_status(settings)
    if not document:
        return
    document["notification"] = dict(notification)
    document["updated_at"] = datetime.now(UTC).isoformat()
    write_json_atomic(settings.status_path, document)


def _notify_status_best_effort(
    settings: AutomationSettings,
    *,
    kind: str,
    target: datetime,
    error_code: str | None = None,
    retry_class: str | None = None,
) -> None:
    try:
        notification = _record_notification(
            settings,
            kind=kind,
            target=target,
            error_code=error_code,
            retry_class=retry_class,
        )
    except Exception:
        notification = {
            "kind": kind,
            "delivery": "failed_nonfatal",
            "signature": None,
        }
    try:
        _attach_notification(settings, notification)
    except Exception:
        pass


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
    retry_config = alpha.get("retry", {})
    retry_reserve = retry_config.get("reserve_calls", 0)
    if type(retry_reserve) is not int or retry_reserve < 0:
        raise AutomationError("Alpha Vantage retry reserve is invalid")
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
    requests_needed += retry_reserve
    if requests_needed > raw_limit:
        raise AutomationError("Alpha Vantage full batch plus retry reserve exceeds cap")
    budget = DailyRequestBudget(
        limit=raw_limit,
        database_path=settings.database,
    )
    remaining = budget.remaining
    if remaining < requests_needed:
        next_available = budget.next_available_at(requests_needed)
        raise ScheduledRetry(
            "Alpha Vantage rolling-24h budget cannot reserve the full batch; "
            f"needed={requests_needed}, remaining={remaining}, "
            f"next_available_at={next_available.isoformat()}",
            error_code="alpha_quota_unavailable",
            retry_class="quota",
            next_retry_at=next_available,
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
        timeout=GIT_NETWORK_TIMEOUT_SECONDS,
        env=GIT_NONINTERACTIVE_ENV,
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
    validated = validate_automation_candidate(
        payload_bytes,
        target=data_as_of,
        expected_contract=settings.contract,
    )
    comparison_bytes = None
    if validated.get("meta", {}).get("result_version") == "weekly-regime-result-v5":
        comparison_bytes = _run(
            [
                "git",
                "show",
                f"{remote_ref}:{PUBLICATION_COMPARISON_PATH}",
            ],
            cwd=settings.root,
            capture=True,
        )
        _validate_v5_comparison_bytes(
            payload_bytes,
            comparison_bytes,
            label="remote publication",
        )
    return RemotePublication(
        head_sha,
        payload_bytes,
        data_as_of,
        comparison_bytes,
    )


def validate_automation_candidate(
    raw: bytes,
    *,
    target: datetime,
    expected_contract: str | None = None,
) -> dict[str, Any]:
    payload = _json_object(raw, label="automation candidate")
    try:
        validate_dashboard_payload(payload)
    except (ContractError, KeyError, TypeError, ValueError) as exc:
        raise AutomationError(f"candidate dashboard contract failed: {exc}") from exc
    meta = payload.get("meta", {})
    result_version = meta.get("result_version")
    contract_by_result = {
        "weekly-regime-result-v4": "v4",
        "weekly-regime-result-v5": "v5",
    }
    actual_contract = contract_by_result.get(result_version)
    if meta.get("mode") != "live" or actual_contract is None:
        raise AutomationError("candidate must be a supported live result")
    if expected_contract is not None and actual_contract != expected_contract:
        raise AutomationError(
            f"candidate must be a live {expected_contract} result"
        )
    is_v5 = actual_contract == "v5"
    if is_v5 and meta.get("publication_status") != "reviewed_publication":
        raise AutomationError("live v5 candidate must have reviewed publication status")
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
    expected_sources = (
        {"alpha_vantage", "alfred", "frb_h10"}
        if is_v5
        else {"alpha_vantage", "alfred"}
    )
    if set(by_id) != expected_sources:
        raise AutomationError("candidate source identities do not match its contract")
    if any(source.get("status") != "ok" for source in by_id.values()):
        raise AutomationError("provider-degraded candidate may not be promoted")
    if any(source.get("issues") for source in by_id.values()):
        raise AutomationError("candidate provider issues must be empty")
    alpha = by_id["alpha_vantage"]
    alpha_available = _parse_aware_datetime(
        alpha.get("available_at"), label="alpha_vantage available_at"
    )
    coverage_start, separator, coverage_end = str(
        alpha.get("coverage", "")
    ).partition("–")
    try:
        coverage_end_date = date.fromisoformat(coverage_end)
    except ValueError as exc:
        raise AutomationError("Alpha Vantage coverage is invalid") from exc
    if not coverage_start or not separator or not alpha_market_week_is_current(
        available_at=alpha_available,
        coverage_end=coverage_end_date,
        cutoff=target,
    ):
        raise AutomationError(
            "Alpha Vantage has not reached the due cutoff market week"
        )
    target_date = target.astimezone(EASTERN).date().isoformat()
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
        if is_v5:
            health = payload.get("model", {}).get("model_health", {})
            reasons = health.get("reasons") if isinstance(health, dict) else None
            allowed_reasons = {"weak_generalization", "calibration_drift"}
            if (
                health.get("status") != "review_due"
                or not isinstance(reasons, list)
                or not reasons
                or not set(reasons).issubset(allowed_reasons)
            ):
                raise AutomationError(
                    "candidate degradation is not an allowed V5 model review warning"
                )
        else:
            diagnostic = payload.get("model", {}).get("holdout_diagnostic", {})
            if diagnostic.get("status") != "weak_generalization":
                raise AutomationError(
                    "candidate degradation is not the allowed diagnostic warning"
                )
    elif meta_status != "ok":
        raise AutomationError("candidate meta.status is not publishable")
    if is_v5:
        model = payload.get("model", {})
        fx = model.get("fx_ablation", {}) if isinstance(model, dict) else {}
        fx_gate = fx.get("gate", {}) if isinstance(fx, dict) else {}
        if (
            model.get("champion") != "markov"
            or fx.get("promotion_allowed") is not False
            or fx.get("core_champion_promoted") is not False
            or fx_gate.get("passed_variants") != []
        ):
            raise AutomationError(
                "V5 publication must preserve the Markov champion and FX non-promotion"
            )
    return payload


def _verify_candidate_package(
    settings: AutomationSettings,
    candidate_path: Path,
    comparison_path: Path | None,
) -> None:
    settings.state_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".package-check-", dir=settings.state_directory
    ) as temporary:
        output = Path(temporary) / "public-dashboard"
        command = [
            sys.executable,
            "scripts/package_public_demo.py",
            "--payload",
            str(candidate_path),
            "--publication-mode",
            "live-derived",
            "--acknowledge-personal-noncommercial-publication",
            "--output",
            str(output),
        ]
        if comparison_path is not None:
            command.extend(["--comparison", str(comparison_path)])
        _run(
            command,
            cwd=settings.root,
        )
        _run(
            [sys.executable, "scripts/verify_public_package.py", str(output)],
            cwd=settings.root,
        )
        if (output / PUBLIC_PAYLOAD_PATH).read_bytes() != candidate_path.read_bytes():
            raise AutomationError("verified package payload bytes differ from the candidate")
        if comparison_path is not None and (
            output / PUBLIC_COMPARISON_PATH
        ).read_bytes() != comparison_path.read_bytes():
            raise AutomationError(
                "verified package comparison bytes differ from the candidate"
            )


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
        if not any(line.endswith(f"\t{path}") for path in ALLOWED_REMOTE_DRIFT)
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
    comparison_raw: bytes | None = None,
    *,
    target: datetime,
    context: Mapping[str, str],
) -> None:
    payload = validate_automation_candidate(
        raw,
        target=target,
        expected_contract=settings.contract,
    )
    generation_id = str(payload.get("meta", {}).get("generation_id", ""))
    if not generation_id:
        raise AutomationError("candidate generation_id is missing")
    settings.candidate_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = settings.candidate_path.with_name(".regime-results.json.tmp")
    temporary.write_bytes(raw)
    os.chmod(temporary, 0o600)
    os.replace(temporary, settings.candidate_path)
    if settings.contract == "v5" and comparison_raw is None:
        raise AutomationError("V5 candidate comparison is missing")
    if comparison_raw is not None:
        comparison_temporary = settings.candidate_comparison_path.with_name(
            ".v5-vs-v4-comparison.json.tmp"
        )
        comparison_temporary.write_bytes(comparison_raw)
        os.chmod(comparison_temporary, 0o600)
        os.replace(comparison_temporary, settings.candidate_comparison_path)
    write_json_atomic(
        settings.candidate_metadata_path,
        {
            "schema_version": 2,
            "automation_id": settings.automation_id,
            "data_as_of": target.astimezone(UTC).isoformat(),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "comparison_sha256": (
                hashlib.sha256(comparison_raw).hexdigest()
                if comparison_raw is not None
                else None
            ),
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
        comparison_raw = (
            settings.candidate_comparison_path.read_bytes()
            if metadata.get("comparison_sha256") is not None
            else None
        )
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
    if comparison_raw is not None and metadata.get(
        "comparison_sha256"
    ) != hashlib.sha256(comparison_raw).hexdigest():
        return None
    if settings.contract == "v5" and comparison_raw is None:
        return None
    if any(metadata.get(key) != value for key, value in context.items()):
        return None
    payload = validate_automation_candidate(
        raw,
        target=target,
        expected_contract=settings.contract,
    )
    if metadata.get("generation_id") != payload.get("meta", {}).get("generation_id"):
        return None
    _verify_candidate_package(
        settings,
        settings.candidate_path,
        settings.candidate_comparison_path if comparison_raw is not None else None,
    )
    return raw


def _load_collection_report(
    settings: AutomationSettings,
    *,
    target: datetime,
) -> dict[str, Any] | None:
    try:
        value = json.loads(settings.collection_report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        return None
    if value.get("expected_cutoff") != target.astimezone(UTC).isoformat():
        return None
    return value


def _collection_failure(
    settings: AutomationSettings,
    *,
    target: datetime,
    report: Mapping[str, Any],
) -> ScheduledRetry:
    if report.get("error_code") == "database_build_lock_busy":
        return ScheduledRetry(
            "another live build temporarily owns the snapshot database lock",
            error_code="database_build_lock_busy",
            retry_class="transient",
            next_retry_at=datetime.now(UTC) + timedelta(minutes=30),
        )
    issues = [str(item) for item in report.get("issues", [])]
    if report.get("gate_error"):
        issues.append(str(report.get("gate_error")))
    sources = report.get("sources", [])
    if isinstance(sources, list):
        for source in sources:
            if isinstance(source, Mapping):
                issues.extend(str(item) for item in source.get("issues", []))
    normalized = " ".join(issues).lower()
    if "ac power" in normalized:
        return ScheduledRetry(
            "AC power was disconnected before model analysis",
            error_code="ac_power_unavailable",
            retry_class="transient",
            next_retry_at=datetime.now(UTC) + settings.transient_retry_delay,
        )
    if any(
        marker in normalized
        for marker in ("not configured", "rights", "api key", "keychain")
    ):
        return ScheduledRetry(
            "provider collection credentials or rights acknowledgement require attention",
            error_code="provider_credentials_blocked",
            retry_class="blocked",
            next_retry_at=None,
        )
    alpha_status = None
    if isinstance(sources, list):
        for source in sources:
            if isinstance(source, Mapping) and source.get("id") == "alpha_vantage":
                alpha_status = source.get("status")
                break
    if alpha_status == "quota_exhausted" or "rolling-24h" in normalized:
        budget = DailyRequestBudget(limit=25, database_path=settings.database)
        next_available = budget.next_available_at(25)
        return ScheduledRetry(
            "Alpha Vantage quota prevented a complete provider snapshot",
            error_code="alpha_quota_exhausted",
            retry_class="quota",
            next_retry_at=next_available,
        )
    return ScheduledRetry(
        "provider collection did not pass the strict training gate",
        error_code="provider_collection_degraded",
        retry_class="transient",
        next_retry_at=datetime.now(UTC) + settings.transient_retry_delay,
    )


def _build_candidate(
    settings: AutomationSettings,
    *,
    target: datetime,
    context: Mapping[str, str],
    started_at: datetime | None = None,
    run_id: str | None = None,
) -> bytes:
    settings.collection_report_path.unlink(missing_ok=True)
    settings.reviewed_payload_path.unlink(missing_ok=True)
    settings.comparison_path.unlink(missing_ok=True)

    def heartbeat() -> None:
        report = _load_collection_report(settings, target=target)
        stage = "train_models" if report and report.get("ready_for_training") else "collect_live_data"
        _heartbeat_status(settings, stage=stage)

    command = [
        sys.executable,
        "-m",
        "regime_lab",
        "build",
        "--contract",
        settings.contract,
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
        "--expected-cutoff",
        target.astimezone(UTC).isoformat(),
        "--collection-report",
        str(settings.collection_report_path),
        "--alfred-rights-confirmed",
    ]
    if settings.require_ac_power:
        command.append("--require-ac-power")
    try:
        _run(
            command,
            cwd=settings.root,
            heartbeat=heartbeat if started_at is not None else None,
            heartbeat_interval=settings.heartbeat_interval.total_seconds(),
        )
    except AutomationError as exc:
        report = _load_collection_report(settings, target=target)
        if report is not None and report.get("ready_for_training") is not True:
            raise _collection_failure(settings, target=target, report=report) from exc
        if report is None:
            raise ScheduledRetry(
                "live build failed before a validated collection receipt was written",
                error_code="child_precollection_failed",
                retry_class="transient",
                next_retry_at=datetime.now(UTC) + settings.transient_retry_delay,
            ) from exc
        raise
    if _candidate_context(settings) != dict(context):
        raise AutomationError("tracked source or config changed during live build")
    _heartbeat_status(settings, stage="audit_candidate")
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
    if settings.contract == "v5":
        _run(
            [
                sys.executable,
                "scripts/compare_v5_to_frozen_v4.py",
                "--v5-artifacts",
                str(settings.artifacts),
                "--v5-payload",
                str(settings.payload),
                "--output",
                str(settings.comparison_path),
            ],
            cwd=settings.root,
        )
        _run(
            [
                sys.executable,
                "scripts/promote_v5_publication.py",
                "--candidate",
                str(settings.payload),
                "--comparison",
                str(settings.comparison_path),
                "--output",
                str(settings.reviewed_payload_path),
            ],
            cwd=settings.root,
        )
        settings.comparison_path.unlink(missing_ok=True)
        _run(
            [
                sys.executable,
                "scripts/compare_v5_to_frozen_v4.py",
                "--v5-artifacts",
                str(settings.artifacts),
                "--v5-payload",
                str(settings.reviewed_payload_path),
                "--output",
                str(settings.comparison_path),
            ],
            cwd=settings.root,
        )
        candidate_path = settings.reviewed_payload_path
        comparison_path: Path | None = settings.comparison_path
        _run(
            [sys.executable, "-m", "regime_lab", "validate", str(candidate_path)],
            cwd=settings.root,
        )
        _run(
            [
                sys.executable,
                "scripts/audit_outputs.py",
                "--payload",
                str(candidate_path),
                "--artifacts",
                str(settings.artifacts),
                "--mode",
                "live",
            ],
            cwd=settings.root,
        )
    else:
        candidate_path = settings.payload
        comparison_path = None
    raw = candidate_path.read_bytes()
    validate_automation_candidate(
        raw,
        target=target,
        expected_contract=settings.contract,
    )
    _verify_candidate_package(settings, candidate_path, comparison_path)
    if _candidate_context(settings) != dict(context):
        raise AutomationError("tracked source or config changed during live audit")
    comparison_raw = comparison_path.read_bytes() if comparison_path is not None else None
    _cache_candidate(
        settings,
        raw,
        comparison_raw,
        target=target,
        context=context,
    )
    return raw


def _publish_candidate(
    settings: AutomationSettings,
    *,
    candidate: bytes,
    comparison: bytes | None = None,
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
            timeout=GIT_NETWORK_TIMEOUT_SECONDS,
            env=GIT_NONINTERACTIVE_ENV,
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
        comparison_target = checkout / PUBLICATION_COMPARISON_PATH
        if comparison is not None:
            if comparison_target.is_symlink() or not comparison_target.is_file():
                raise AutomationError(
                    "publication comparison target must be an existing regular file"
                )
            comparison_target.write_bytes(comparison)
            os.chmod(comparison_target, 0o644)
        changed = set(
            _run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=checkout,
                capture=True,
            ).decode().splitlines()
        )
        expected = {f" M {PUBLICATION_PATH}"}
        if comparison is not None:
            expected.add(f" M {PUBLICATION_COMPARISON_PATH}")
        if changed != expected:
            if (
                not changed
                and target_path.read_bytes() == candidate
                and (
                    comparison is None
                    or comparison_target.read_bytes() == comparison
                )
            ):
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
                    timeout=GIT_NETWORK_TIMEOUT_SECONDS,
                    env=GIT_NONINTERACTIVE_ENV,
                )
                return commit_sha
            raise AutomationError("release checkout changed outside the publication snapshot")
        staged_paths = [PUBLICATION_PATH]
        if comparison is not None:
            staged_paths.append(PUBLICATION_COMPARISON_PATH)
        _run(["git", "add", "--", *staged_paths], cwd=checkout)
        staged = set(
            _run(
                ["git", "diff", "--cached", "--name-only"],
                cwd=checkout,
                capture=True,
            ).decode().splitlines()
        )
        if staged != set(staged_paths):
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
            timeout=GIT_NETWORK_TIMEOUT_SECONDS,
            env=GIT_NONINTERACTIVE_ENV,
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
    expected_comparison: bytes | None = None,
    fetch: Callable[[str], bytes] = _fetch_url,
) -> None:
    expected = _json_object(expected_payload, label="expected public payload")
    if (
        expected.get("meta", {}).get("result_version")
        == "weekly-regime-result-v5"
        and expected_comparison is None
    ):
        raise AutomationError("V5 public readback requires an expected comparison")
    if expected_comparison is not None:
        _validate_v5_comparison_bytes(
            expected_payload,
            expected_comparison,
            label="expected publication",
        )
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
    if expected_comparison is not None:
        public_comparison = fetch(
            urljoin(settings.public_root, PUBLIC_COMPARISON_PATH)
        )
        comparison_hash = hashlib.sha256(expected_comparison).hexdigest()
        if hashlib.sha256(public_comparison).hexdigest() != comparison_hash:
            raise AutomationError(
                "public comparison SHA-256 does not match the promoted result"
            )
        comparison_record = manifest.get("files", {}).get(
            PUBLIC_COMPARISON_PATH,
            {},
        )
        if comparison_record.get("sha256") != comparison_hash:
            raise AutomationError(
                "public manifest does not identify the promoted comparison"
            )
        _validate_v5_comparison_bytes(
            public_payload,
            public_comparison,
            label="public publication",
        )
    if manifest.get("payload_data_as_of") != expected.get("meta", {}).get("data_as_of"):
        raise AutomationError("public manifest data_as_of does not match the payload")
    html = fetch(settings.public_root)
    if b"US Market Regime Lab" not in html or b"app.js" not in html:
        raise AutomationError("public dashboard HTML is not the Regime consumer")


def _wait_for_public_readback(
    settings: AutomationSettings,
    *,
    expected_payload: bytes,
    expected_comparison: bytes | None = None,
    sleep: Callable[[float], None] = time.sleep,
    heartbeat: Callable[[], None] | None = None,
) -> None:
    deadline = time.monotonic() + (
        settings.workflow_timeout + settings.public_readback_timeout
    ).total_seconds()
    last_error: AutomationError | None = None
    while time.monotonic() < deadline:
        if heartbeat is not None:
            heartbeat()
        try:
            verify_public_readback(
                settings,
                expected_payload=expected_payload,
                expected_comparison=expected_comparison,
            )
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
    force_transient_retry: bool = False,
) -> dict[str, Any]:
    started = (now or datetime.now(UTC)).astimezone(UTC)
    run_id = uuid.uuid4().hex
    target, old_enough = target_cutoff(
        started, minimum_age=settings.minimum_cutoff_age
    )
    settings.state_directory.mkdir(parents=True, exist_ok=True)
    with automation_lock(settings.lock_path):
        if old_enough:
            guarded = _retry_guard(
                settings,
                target=target,
                now=started,
                force_transient_retry=force_transient_retry,
            )
            if guarded is not None:
                return guarded
        _write_status(
            settings,
            status="running",
            stage="preflight",
            started_at=started,
            target=target,
            run_id=run_id,
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
                    run_id=run_id,
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
                    run_id=run_id,
                )
                try:
                    verify_public_readback(
                        settings,
                        expected_payload=remote.payload_bytes,
                        expected_comparison=remote.comparison_bytes,
                    )
                    workflow_url = None
                except (AutomationError, OSError):
                    recovery_sha = _publish_candidate(
                        settings,
                        candidate=remote.payload_bytes,
                        comparison=remote.comparison_bytes,
                        target=target,
                        expected_head_sha=remote.head_sha,
                        force_pages_rebuild=True,
                    )
                    workflow_url = _workflow_url(settings)
                    _wait_for_public_readback(
                        settings,
                        expected_payload=remote.payload_bytes,
                        expected_comparison=remote.comparison_bytes,
                        heartbeat=lambda: _heartbeat_status(
                            settings, stage="deployment_recovery"
                        ),
                    )
                    remote = RemotePublication(
                        recovery_sha,
                        remote.payload_bytes,
                        remote.data_as_of,
                        remote.comparison_bytes,
                    )
                result = _write_status(
                    settings,
                    status="succeeded",
                    stage="already_current",
                    started_at=started,
                    target=target,
                    detail="remote publication and public Pages are current",
                    commit_sha=remote.head_sha,
                    workflow_url=workflow_url,
                    run_id=run_id,
                )
                _notify_status_best_effort(
                    settings, kind="recovery", target=target
                )
                return _read_status(settings) or result

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
                    run_id=run_id,
                )
                candidate = _build_candidate(
                    settings,
                    target=target,
                    context=context,
                    started_at=started,
                    run_id=run_id,
                )
            else:
                _write_status(
                    settings,
                    status="running",
                    stage="resume_validated_candidate",
                    started_at=started,
                    target=target,
                    detail="reusing the validated candidate without provider access",
                    run_id=run_id,
                )

            candidate_comparison = (
                settings.candidate_comparison_path.read_bytes()
                if settings.contract == "v5"
                else None
            )

            remote = _git_preflight(settings)
            if remote.data_as_of > target:
                raise AutomationError("remote publication is ahead of the due cutoff")
            if remote.data_as_of == target:
                if remote.payload_bytes != candidate:
                    candidate = remote.payload_bytes
                candidate_comparison = remote.comparison_bytes
                commit_sha = remote.head_sha
            else:
                _write_status(
                    settings,
                    status="running",
                    stage="publish_snapshot",
                    started_at=started,
                    target=target,
                    run_id=run_id,
                )
                commit_sha = _publish_candidate(
                    settings,
                    candidate=candidate,
                    comparison=candidate_comparison,
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
                run_id=run_id,
            )
            workflow_url = _workflow_url(settings)
            _wait_for_public_readback(
                settings,
                expected_payload=candidate,
                expected_comparison=candidate_comparison,
                heartbeat=lambda: _heartbeat_status(
                    settings, stage="wait_for_pages"
                ),
            )
            result = _write_status(
                settings,
                status="succeeded",
                stage="public_readback_verified",
                started_at=started,
                target=target,
                detail="collection, training, audit, promotion, Pages, and public readback passed",
                commit_sha=commit_sha,
                workflow_url=workflow_url,
                run_id=run_id,
            )
            _notify_status_best_effort(settings, kind="recovery", target=target)
            return _read_status(settings) or result
        except BaseException as exc:
            current_stage = str(_read_status(settings).get("stage") or "failed")
            error_code, retry_class, next_retry_at = _failure_policy(
                exc,
                stage=current_stage,
                now=datetime.now(UTC),
                settings=settings,
            )
            fingerprint = _recovery_fingerprint(settings)
            _write_status(
                settings,
                status="failed",
                stage="failed",
                started_at=started,
                target=target,
                detail=_safe_error(exc),
                run_id=run_id,
                error_code=error_code,
                retry_class=retry_class,
                next_retry_at=next_retry_at,
                recovery_fingerprint=fingerprint,
            )
            _notify_status_best_effort(
                settings,
                kind="failure",
                target=target,
                error_code=error_code,
                retry_class=retry_class,
            )
            raise


def launch_agent_document(settings: AutomationSettings) -> dict[str, Any]:
    python = settings.root / ".venv" / "bin" / "python"
    if not python.is_file():
        raise AutomationError(f"project virtualenv Python is missing: {python}")
    settings.state_directory.mkdir(parents=True, exist_ok=True)
    schedule_hours = settings.retry_hours or (settings.schedule_hour,)
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
        "StartCalendarInterval": [
            {"Hour": hour, "Minute": settings.schedule_minute}
            for hour in schedule_hours
        ],
        "RunAtLoad": True,
        "ProcessType": "Standard",
        "ExitTimeOut": 120,
        "Umask": 0o077,
        "ThrottleInterval": 300,
        "StandardOutPath": str(settings.state_directory / "launchd.stdout.log"),
        "StandardErrorPath": str(settings.state_directory / "launchd.stderr.log"),
    }


def _launch_agent_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{AUTOMATION_LABEL}.plist"


def _canonical_plist_bytes(document: Mapping[str, Any]) -> bytes:
    return plistlib.dumps(dict(document), fmt=plistlib.FMT_BINARY, sort_keys=True)


def _launch_agent_configuration(
    settings: AutomationSettings,
    *,
    target: Path | None = None,
) -> dict[str, Any]:
    selected = target or _launch_agent_path()
    expected = launch_agent_document(settings)
    expected_sha = hashlib.sha256(_canonical_plist_bytes(expected)).hexdigest()
    if not selected.is_file():
        return {
            "configuration_matches": False,
            "expected_plist_sha256": expected_sha,
            "installed_plist_sha256": None,
            "drift_keys": ["missing"],
        }
    try:
        installed = plistlib.loads(selected.read_bytes())
    except (OSError, plistlib.InvalidFileException, ValueError, TypeError):
        return {
            "configuration_matches": False,
            "expected_plist_sha256": expected_sha,
            "installed_plist_sha256": None,
            "drift_keys": ["invalid"],
        }
    if not isinstance(installed, dict):
        return {
            "configuration_matches": False,
            "expected_plist_sha256": expected_sha,
            "installed_plist_sha256": None,
            "drift_keys": ["invalid"],
        }
    installed_sha = hashlib.sha256(_canonical_plist_bytes(installed)).hexdigest()
    keys = sorted(set(expected) | set(installed))
    drift = [key for key in keys if expected.get(key) != installed.get(key)]
    return {
        "configuration_matches": not drift,
        "expected_plist_sha256": expected_sha,
        "installed_plist_sha256": installed_sha,
        "drift_keys": drift,
    }


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
        "config/series.json",
        "src/regime_lab/automation.py",
        "src/regime_lab/cli.py",
        "src/regime_lab/collection.py",
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
            configuration = _launch_agent_configuration(settings, target=target)
            if not configuration["configuration_matches"]:
                raise AutomationError("installed LaunchAgent plist failed exact verification")
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
        configuration = _launch_agent_configuration(settings, target=target)
        if not configuration["configuration_matches"]:
            raise AutomationError("loaded LaunchAgent configuration drifted after installation")
    hours = settings.retry_hours or (settings.schedule_hour,)
    return {
        "ok": True,
        "installed": True,
        "label": AUTOMATION_LABEL,
        "path": str(target),
        "schedule": [f"{hour:02d}:{settings.schedule_minute:02d}" for hour in hours],
        "runs_immediately": True,
        **configuration,
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
    health = _read_status(settings) or None
    installed = target.is_file()
    loaded = _launch_agent_loaded(settings)
    configuration = _launch_agent_configuration(settings, target=target)
    health_ok = bool(health)
    heartbeat_stale = False
    health_stale = False
    if health:
        if health.get("schema_version") != HEALTH_SCHEMA_VERSION:
            health_ok = False
        if health.get("status") in {"failed", "blocked"}:
            health_ok = False
        failures = health.get("consecutive_failures", 0)
        if type(failures) is int and failures > 0:
            health_ok = False
        if health.get("status") == "running":
            try:
                heartbeat = _parse_aware_datetime(
                    health.get("heartbeat_at"), label="health heartbeat_at"
                )
                heartbeat_stale = (
                    datetime.now(UTC) - heartbeat > settings.stale_heartbeat_after
                )
            except AutomationError:
                heartbeat_stale = True
            if heartbeat_stale:
                health_ok = False
        else:
            schedule_hours = settings.retry_hours or (settings.schedule_hour,)
            schedule_minutes = sorted(
                hour * 60 + settings.schedule_minute for hour in schedule_hours
            )
            cyclic = schedule_minutes + [schedule_minutes[0] + 24 * 60]
            max_gap_minutes = max(
                later - earlier
                for earlier, later in zip(schedule_minutes, cyclic[1:])
            )
            terminal_stale_after = timedelta(minutes=max_gap_minutes + 120)
            try:
                updated = _parse_aware_datetime(
                    health.get("updated_at"), label="health updated_at"
                )
                health_stale = datetime.now(UTC) - updated > terminal_stale_after
            except AutomationError:
                health_stale = True
            if health_stale:
                health_ok = False
    operational = bool(
        installed
        and loaded
        and configuration["configuration_matches"]
        and health_ok
    )
    return {
        "ok": operational,
        "operational": operational,
        "installed": installed,
        "loaded": loaded,
        "label": AUTOMATION_LABEL,
        "path": str(target),
        "heartbeat_stale": heartbeat_stale,
        "health_stale": health_stale,
        **configuration,
        "health": health,
    }


def command_automation(args: argparse.Namespace) -> int:
    try:
        settings = AutomationSettings.load(args.config)
        force_retry = bool(getattr(args, "force_retry", False))
        if force_retry and args.action != "run":
            raise AutomationError("--force-retry is valid only with automation run")
        if args.action == "preflight":
            result = preflight_summary(settings)
        elif args.action == "run":
            result = run_weekly_release(
                settings,
                force_transient_retry=force_retry,
            )
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
    return 0 if result.get("ok", True) else 1


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
