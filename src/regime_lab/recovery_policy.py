"""Explicit, non-destructive storage recovery policy and backup preflight."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import stat
from typing import Any

from regime_lab.config import project_root


RECOVERY_POLICY_SCHEMA_VERSION = "regime-recovery-policy/1"


class RecoveryPolicyError(RuntimeError):
    """Raised when backup capacity or recovery policy cannot be proven safe."""


@dataclass(frozen=True)
class RecoveryPolicy:
    automatic_deletion: bool
    retention_valid_generations: int
    max_source_bytes: int
    max_total_backup_bytes: int
    minimum_free_bytes_after_backup: int
    max_backup_seconds: int
    minimum_assumed_bytes_per_second: int
    restore_drill_max_seconds: int
    restore_drill_max_age_hours: int

    def as_document(self) -> dict[str, Any]:
        return {
            "schema_version": RECOVERY_POLICY_SCHEMA_VERSION,
            **asdict(self),
        }


def load_recovery_policy(path: str | Path | None = None) -> RecoveryPolicy:
    target = (
        Path(path)
        if path is not None
        else project_root() / "config/recovery-policy.json"
    )
    if target.is_symlink() or not target.is_file():
        raise RecoveryPolicyError(f"recovery policy is missing: {target}")
    try:
        document = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RecoveryPolicyError("recovery policy must be valid UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise RecoveryPolicyError("recovery policy root must be an object")
    expected = {
        "schema_version",
        "automatic_deletion",
        "retention_valid_generations",
        "max_source_bytes",
        "max_total_backup_bytes",
        "minimum_free_bytes_after_backup",
        "max_backup_seconds",
        "minimum_assumed_bytes_per_second",
        "restore_drill_max_seconds",
        "restore_drill_max_age_hours",
    }
    if set(document) != expected:
        raise RecoveryPolicyError("recovery policy fields are not exact")
    if document.get("schema_version") != RECOVERY_POLICY_SCHEMA_VERSION:
        raise RecoveryPolicyError("recovery policy schema_version is invalid")
    if document.get("automatic_deletion") is not False:
        raise RecoveryPolicyError("automatic backup deletion must remain disabled")
    values: dict[str, int] = {}
    for field in expected - {"schema_version", "automatic_deletion"}:
        value = document.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise RecoveryPolicyError(f"recovery policy {field} must be positive")
        values[field] = value
    if not 1 <= values["retention_valid_generations"] <= 32:
        raise RecoveryPolicyError(
            "retention_valid_generations must be between 1 and 32"
        )
    return RecoveryPolicy(automatic_deletion=False, **values)


def backup_capacity_preflight(
    source_database: str | Path,
    backup_directory: str | Path,
    *,
    policy: RecoveryPolicy,
) -> dict[str, Any]:
    """Fail before SQLite access when the byte/time budget cannot fit."""

    source = Path(source_database)
    backup_root = Path(backup_directory)
    projected_bytes = _projected_sqlite_bytes(source)
    current_backup_bytes = _tree_bytes(backup_root) if backup_root.exists() else 0
    filesystem_anchor = _nearest_existing_parent(backup_root)
    free_bytes = shutil.disk_usage(filesystem_anchor).free
    estimated_seconds = projected_bytes / policy.minimum_assumed_bytes_per_second
    if projected_bytes > policy.max_source_bytes:
        raise RecoveryPolicyError("backup source exceeds max_source_bytes")
    if current_backup_bytes + projected_bytes > policy.max_total_backup_bytes:
        raise RecoveryPolicyError("projected backup inventory exceeds byte ceiling")
    if free_bytes - projected_bytes < policy.minimum_free_bytes_after_backup:
        raise RecoveryPolicyError("projected backup violates minimum free-space ceiling")
    if estimated_seconds > policy.max_backup_seconds:
        raise RecoveryPolicyError("projected backup exceeds time ceiling")
    return {
        "projected_bytes": projected_bytes,
        "current_backup_bytes": current_backup_bytes,
        "projected_total_backup_bytes": current_backup_bytes + projected_bytes,
        "filesystem_free_bytes_before": free_bytes,
        "estimated_seconds": estimated_seconds,
        "max_backup_seconds": policy.max_backup_seconds,
        "automatic_deletion": False,
    }


def _projected_sqlite_bytes(source: Path) -> int:
    total = 0
    for path in (source, Path(f"{source}-wal"), Path(f"{source}-shm")):
        if not path.exists():
            continue
        if path.is_symlink():
            raise RecoveryPolicyError("SQLite backup source must not use symlinks")
        info = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode):
            raise RecoveryPolicyError("SQLite backup source members must be files")
        total += info.st_size
    if total <= 0:
        raise RecoveryPolicyError("SQLite backup source is empty or missing")
    return total


def _tree_bytes(root: Path) -> int:
    if root.is_symlink():
        raise RecoveryPolicyError("backup inventory root must not be a symlink")
    if root.is_file():
        return root.stat(follow_symlinks=False).st_size
    total = 0
    for directory, names, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in (*names, *files):
            path = base / name
            if path.is_symlink():
                raise RecoveryPolicyError(
                    "backup inventory must not contain symbolic links"
                )
            if path.is_file():
                total += path.stat(follow_symlinks=False).st_size
    return total


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path.absolute()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise RecoveryPolicyError("backup filesystem could not be resolved")
        candidate = parent
    return candidate


__all__ = [
    "RECOVERY_POLICY_SCHEMA_VERSION",
    "RecoveryPolicy",
    "RecoveryPolicyError",
    "backup_capacity_preflight",
    "load_recovery_policy",
]
