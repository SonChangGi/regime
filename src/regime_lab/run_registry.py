"""Append-only lifecycle registry for local collection, analysis, and publication runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping


RUN_REGISTRY_SCHEMA = "regime-run-registry-event/1"
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{7,127}")
ALLOWED_TRANSITIONS = {
    None: {"started"},
    "started": {"collecting", "interrupted"},
    "collecting": {"analyzing", "interrupted"},
    "analyzing": {"completed", "interrupted"},
    "completed": {"publication_reviewed", "published"},
    "publication_reviewed": {"published"},
    "published": set(),
    "interrupted": set(),
}


class RunRegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunEvent:
    run_id: str
    status: str
    occurred_at: str
    generation_id: str | None = None
    detail: Mapping[str, Any] | None = None
    schema_version: str = RUN_REGISTRY_SCHEMA


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                raise RunRegistryError(f"blank run-registry row at line {line_number}")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise RunRegistryError(f"invalid run-registry row at line {line_number}")
            rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise RunRegistryError(f"run registry cannot be read: {path}") from exc
    return rows


def append_run_event(
    path: str | Path,
    *,
    run_id: str,
    status: str,
    occurred_at: datetime | None = None,
    generation_id: str | None = None,
    detail: Mapping[str, Any] | None = None,
) -> RunEvent:
    """Append one valid state transition and fsync it before returning."""

    registry = Path(path)
    if RUN_ID.fullmatch(str(run_id)) is None:
        raise RunRegistryError("run_id is invalid")
    if status not in {value for targets in ALLOWED_TRANSITIONS.values() for value in targets}:
        raise RunRegistryError(f"unsupported run status: {status}")
    previous = None
    for row in _read_events(registry):
        if row.get("schema_version") != RUN_REGISTRY_SCHEMA:
            raise RunRegistryError("run registry schema drift detected")
        if row.get("run_id") == run_id:
            previous = str(row.get("status"))
    if status not in ALLOWED_TRANSITIONS.get(previous, set()):
        raise RunRegistryError(f"invalid run transition for {run_id}: {previous!r} -> {status!r}")
    timestamp = occurred_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise RunRegistryError("occurred_at must be timezone-aware")
    event = RunEvent(
        run_id=str(run_id),
        status=status,
        occurred_at=timestamp.astimezone(timezone.utc).isoformat(),
        generation_id=None if generation_id is None else str(generation_id),
        detail=None if detail is None else dict(detail),
    )
    registry.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(asdict(event), ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
    descriptor = os.open(registry, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, encoded.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return event


def current_run_status(path: str | Path, run_id: str) -> str | None:
    status = None
    for row in _read_events(Path(path)):
        if row.get("run_id") == run_id:
            status = str(row.get("status"))
    return status


__all__ = ["RUN_REGISTRY_SCHEMA", "RunEvent", "RunRegistryError", "append_run_event", "current_run_status"]
