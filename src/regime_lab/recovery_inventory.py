"""Read-only recovery inventory and bounded restore-drill evidence."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import shutil
import stat
import time
from typing import Any, Iterable

from regime_lab.database_backup import (
    classify_database_backup,
    verify_database_restore,
)
from regime_lab.recovery_policy import RecoveryPolicy, load_recovery_policy


RECOVERY_INVENTORY_SCHEMA_VERSION = "regime-recovery-inventory/1"
RESTORE_DRILL_SCHEMA_VERSION = "regime-restore-drill/1"
_GENERATION_RE = re.compile(r"^regime-snapshot-\d{8}T\d{12}Z-[0-9a-f]{12}$")


class RecoveryInventoryError(RuntimeError):
    """Raised when a read-only recovery inventory cannot be audited safely."""


def build_recovery_inventory(
    backup_directory: str | Path,
    *,
    checkpoint_paths: Iterable[str | Path] = (),
    preview_paths: Iterable[str | Path] = (),
    policy: RecoveryPolicy | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    selected_policy = policy or load_recovery_policy()
    moment = _utc_moment(generated_at)
    entries: list[dict[str, Any]] = []
    backup_root = Path(backup_directory).absolute()
    if backup_root.is_symlink():
        raise RecoveryInventoryError("backup directory must not be a symlink")
    if backup_root.exists() and not backup_root.is_dir():
        raise RecoveryInventoryError("backup directory must be a directory")
    if backup_root.exists():
        for path in sorted(backup_root.iterdir(), key=lambda item: item.name):
            if not (
                _GENERATION_RE.fullmatch(path.name)
                or path.name.startswith(".regime-snapshot-")
            ):
                continue
            category, detail = (
                classify_database_backup(path)
                if _GENERATION_RE.fullmatch(path.name)
                else ("corrupt", "incomplete_generation")
            )
            entries.append(_inventory_entry(path, category=category, detail=detail))
    for category, paths in (
        ("checkpoint", checkpoint_paths),
        ("preview", preview_paths),
    ):
        for raw_path in paths:
            path = Path(raw_path).absolute()
            entries.append(
                _inventory_entry(
                    path,
                    category=category,
                    detail=None if path.exists() else "missing",
                )
            )

    categories = ("valid-current", "legacy", "corrupt", "checkpoint", "preview")
    counts = {
        category: sum(1 for entry in entries if entry["category"] == category)
        for category in categories
    }
    bytes_by_category = {
        category: sum(
            int(entry["bytes"])
            for entry in entries
            if entry["category"] == category
        )
        for category in categories
    }
    valid_count = counts["valid-current"]
    retention_target = selected_policy.retention_valid_generations
    return {
        "schema_version": RECOVERY_INVENTORY_SCHEMA_VERSION,
        "generated_at": _iso(moment),
        "policy": selected_policy.as_document(),
        "retention": {
            "automatic_deletion": False,
            "target_valid_generations": retention_target,
            "valid_generations": valid_count,
            "over_target_generations": max(0, valid_count - retention_target),
            "action": "review_only_no_automatic_deletion",
        },
        "summary": {
            "counts": counts,
            "bytes": bytes_by_category,
            "total_bytes": sum(bytes_by_category.values()),
        },
        "entries": sorted(
            entries,
            key=lambda item: (item["category"], item["path"]),
        ),
    }


def run_restore_drill(
    backup_directory: str | Path,
    *,
    generation_path: str | Path | None = None,
    policy: RecoveryPolicy | None = None,
    performed_at: datetime | None = None,
) -> dict[str, Any]:
    selected_policy = policy or load_recovery_policy()
    moment = _utc_moment(performed_at)
    if generation_path is None:
        root = Path(backup_directory).absolute()
        candidates = [
            path
            for path in root.iterdir()
            if _GENERATION_RE.fullmatch(path.name)
            and classify_database_backup(path)[0] == "valid-current"
        ]
        if not candidates:
            raise RecoveryInventoryError("no valid-current backup is available")
        target = max(candidates, key=lambda path: path.name)
    else:
        target = Path(generation_path).absolute()
        category, _detail = classify_database_backup(target)
        if category != "valid-current":
            raise RecoveryInventoryError(
                "restore drill requires a valid-current backup"
            )

    try:
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        projected_bytes = int(manifest["backup"]["bytes"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RecoveryInventoryError(
            "restore drill backup byte evidence is unavailable"
        ) from exc
    free_bytes = shutil.disk_usage(target.parent).free
    estimated_seconds = (
        projected_bytes / selected_policy.minimum_assumed_bytes_per_second
    )
    if free_bytes - projected_bytes < selected_policy.minimum_free_bytes_after_backup:
        raise RecoveryInventoryError(
            "restore drill violates minimum free-space ceiling"
        )
    if estimated_seconds > selected_policy.restore_drill_max_seconds:
        raise RecoveryInventoryError("restore drill exceeds estimated time ceiling")

    started = time.monotonic()
    verification = verify_database_restore(
        target,
        max_duration_seconds=selected_policy.restore_drill_max_seconds,
    )
    duration = time.monotonic() - started
    valid_until = moment + timedelta(
        hours=selected_policy.restore_drill_max_age_hours
    )
    return {
        "schema_version": RESTORE_DRILL_SCHEMA_VERSION,
        "performed_at": _iso(moment),
        "valid_until": _iso(valid_until),
        "generation_id": verification.generation_id,
        "duration_seconds": duration,
        "time_ceiling_seconds": selected_policy.restore_drill_max_seconds,
        "within_time_ceiling": duration <= selected_policy.restore_drill_max_seconds,
        "capacity_preflight": {
            "projected_bytes": projected_bytes,
            "filesystem_free_bytes_before": free_bytes,
            "estimated_seconds": estimated_seconds,
        },
        "verification": {
            "sha256": verification.sha256,
            "bytes": verification.bytes,
            "quick_check": verification.quick_check,
            "integrity_check": verification.integrity_check,
            "foreign_key_check": verification.foreign_key_check,
            "core_table_counts": dict(verification.core_table_counts),
        },
    }


def _inventory_entry(
    path: Path,
    *,
    category: str,
    detail: str | None,
) -> dict[str, Any]:
    if path.is_symlink():
        return {
            "category": "corrupt" if category not in {"checkpoint", "preview"} else category,
            "path": str(path),
            "present": True,
            "bytes": 0,
            "modified_at": None,
            "detail": "symbolic_link",
        }
    if not path.exists():
        return {
            "category": category,
            "path": str(path),
            "present": False,
            "bytes": 0,
            "modified_at": None,
            "detail": detail or "missing",
        }
    try:
        size, latest_mtime = _tree_state(path)
    except (OSError, RecoveryInventoryError) as exc:
        return {
            "category": "corrupt" if category not in {"checkpoint", "preview"} else category,
            "path": str(path),
            "present": True,
            "bytes": 0,
            "modified_at": None,
            "detail": f"unreadable:{type(exc).__name__}",
        }
    return {
        "category": category,
        "path": str(path),
        "present": True,
        "bytes": size,
        "modified_at": _iso(datetime.fromtimestamp(latest_mtime, tz=timezone.utc)),
        "detail": detail,
    }


def _tree_state(path: Path) -> tuple[int, float]:
    info = path.stat(follow_symlinks=False)
    if stat.S_ISREG(info.st_mode):
        return info.st_size, info.st_mtime
    if not stat.S_ISDIR(info.st_mode):
        raise RecoveryInventoryError("recovery inventory member is not file/directory")
    total = 0
    latest = info.st_mtime
    for directory, names, files in os.walk(path, followlinks=False):
        base = Path(directory)
        for name in (*names, *files):
            member = base / name
            if member.is_symlink():
                raise RecoveryInventoryError("recovery inventory contains symlink")
            member_info = member.stat(follow_symlinks=False)
            latest = max(latest, member_info.st_mtime)
            if stat.S_ISREG(member_info.st_mode):
                total += member_info.st_size
    return total, latest


def _utc_moment(value: datetime | None) -> datetime:
    moment = value or datetime.now(timezone.utc)
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("recovery evidence timestamp must be timezone-aware")
    return moment.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


__all__ = [
    "RECOVERY_INVENTORY_SCHEMA_VERSION",
    "RESTORE_DRILL_SCHEMA_VERSION",
    "RecoveryInventoryError",
    "build_recovery_inventory",
    "run_restore_drill",
]
