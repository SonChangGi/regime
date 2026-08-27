"""Append-only lifecycle registry for local collection, analysis, and publication runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping
from uuid import uuid4


RUN_REGISTRY_SCHEMA = "regime-run-registry-event/1"
LINE_CHECKSUM_FIELD = "line_checksum_sha256"
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{7,127}")
ALLOWED_TRANSITIONS = {
    None: {"started"},
    "started": {"collecting", "interrupted"},
    "collecting": {"analyzing", "interrupted"},
    "analyzing": {"completed", "interrupted"},
    "completed": {"publication_reviewed"},
    "publication_reviewed": {"published"},
    "published": set(),
    "interrupted": set(),
}
ALL_STATUSES = frozenset(
    {status for targets in ALLOWED_TRANSITIONS.values() for status in targets}
)


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
    line_checksum_sha256: str | None = None


def _canonical_event_bytes(row: Mapping[str, Any]) -> bytes:
    payload = dict(row)
    payload.pop(LINE_CHECKSUM_FIELD, None)
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _validate_line_checksum(row: Mapping[str, Any], *, line_number: int) -> None:
    checksum = row.get(LINE_CHECKSUM_FIELD)
    # Historical event/1 rows predate line checksums and remain readable.
    if checksum is None:
        return
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise RunRegistryError(
            f"invalid run-registry checksum at line {line_number}"
        )
    expected = hashlib.sha256(_canonical_event_bytes(row)).hexdigest()
    if checksum != expected:
        raise RunRegistryError(
            f"run-registry checksum mismatch at line {line_number}"
        )


def _validate_event_row(row: Mapping[str, Any], *, line_number: int) -> None:
    legacy_fields = {
        "schema_version",
        "run_id",
        "status",
        "occurred_at",
        "generation_id",
        "detail",
    }
    if frozenset(row) not in {
        frozenset(legacy_fields),
        frozenset({*legacy_fields, LINE_CHECKSUM_FIELD}),
    }:
        raise RunRegistryError(
            f"invalid run-registry fields at line {line_number}"
        )
    if row.get("schema_version") != RUN_REGISTRY_SCHEMA:
        raise RunRegistryError(f"run-registry schema drift at line {line_number}")
    if not isinstance(row.get("run_id"), str) or RUN_ID.fullmatch(row["run_id"]) is None:
        raise RunRegistryError(f"invalid run_id at line {line_number}")
    if row.get("status") not in ALL_STATUSES:
        raise RunRegistryError(f"invalid run status at line {line_number}")
    try:
        occurred_at = datetime.fromisoformat(str(row.get("occurred_at")))
    except ValueError as exc:
        raise RunRegistryError(f"invalid occurred_at at line {line_number}") from exc
    if occurred_at.tzinfo is None or occurred_at.utcoffset() is None:
        raise RunRegistryError(f"invalid occurred_at at line {line_number}")
    generation_id = row.get("generation_id")
    if generation_id is not None and (
        not isinstance(generation_id, str) or not generation_id
    ):
        raise RunRegistryError(f"invalid generation_id at line {line_number}")
    detail = row.get("detail")
    if detail is not None and not isinstance(detail, Mapping):
        raise RunRegistryError(f"invalid detail at line {line_number}")


def _decode_events(
    raw: bytes,
    *,
    tolerate_partial_tail: bool,
) -> tuple[list[dict[str, Any]], int | None]:
    """Decode complete events and identify one interrupted trailing fragment.

    A process can be killed between ``write`` and ``fsync``.  Only an invalid
    final fragment without a newline is recoverable; malformed complete rows
    remain a hard audit failure.
    """

    rows: list[dict[str, Any]] = []
    offset = 0
    lines = raw.splitlines(keepends=True)
    for line_number, encoded in enumerate(lines, start=1):
        complete = encoded.endswith(b"\n")
        content = encoded[:-1] if complete else encoded
        if content.endswith(b"\r"):
            content = content[:-1]
        try:
            text = content.decode("utf-8")
            if not text.strip():
                raise RunRegistryError(f"blank run-registry row at line {line_number}")
            value = json.loads(text)
            if not isinstance(value, dict):
                raise RunRegistryError(f"invalid run-registry row at line {line_number}")
            _validate_line_checksum(value, line_number=line_number)
            _validate_event_row(value, line_number=line_number)
            rows.append(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if tolerate_partial_tail and not complete and line_number == len(lines):
                return rows, offset
            raise RunRegistryError(
                f"invalid run-registry row at line {line_number}"
            ) from exc
        offset += len(encoded)
    return rows, None


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError as exc:
        raise RunRegistryError(f"run registry cannot be read: {path}") from exc
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH)
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            raw = handle.read()
        rows, _partial_offset = _decode_events(
            raw,
            tolerate_partial_tail=True,
        )
        return rows
    except OSError as exc:
        raise RunRegistryError(f"run registry cannot be read: {path}") from exc
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _preserve_partial_fragment(registry: Path, fragment: bytes) -> Path:
    quarantine = registry.with_name(
        f"{registry.name}.partial-{uuid4().hex}.corrupt"
    )
    descriptor = os.open(
        quarantine,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o600,
    )
    try:
        os.write(descriptor, fragment)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return quarantine


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
    if status not in ALL_STATUSES:
        raise RunRegistryError(f"unsupported run status: {status}")
    timestamp = occurred_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise RunRegistryError("occurred_at must be timezone-aware")
    base_event = RunEvent(
        run_id=str(run_id),
        status=status,
        occurred_at=timestamp.astimezone(timezone.utc).isoformat(),
        generation_id=None if generation_id is None else str(generation_id),
        detail=None if detail is None else dict(detail),
    )
    registry.parent.mkdir(parents=True, exist_ok=True)
    event_document = asdict(base_event)
    event_document.pop(LINE_CHECKSUM_FIELD)
    checksum = hashlib.sha256(_canonical_event_bytes(event_document)).hexdigest()
    event = RunEvent(**event_document, line_checksum_sha256=checksum)
    encoded = (
        json.dumps(
            asdict(event),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(registry, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        os.lseek(descriptor, 0, os.SEEK_SET)
        raw = b""
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            raw += chunk
        rows, partial_offset = _decode_events(
            raw,
            tolerate_partial_tail=True,
        )
        if partial_offset is not None:
            _preserve_partial_fragment(registry, raw[partial_offset:])
            os.ftruncate(descriptor, partial_offset)
            raw = raw[:partial_offset]
        previous = None
        for row in rows:
            if row.get("schema_version") != RUN_REGISTRY_SCHEMA:
                raise RunRegistryError("run registry schema drift detected")
            if row.get("run_id") == run_id:
                previous = str(row.get("status"))
        if status not in ALLOWED_TRANSITIONS.get(previous, set()):
            raise RunRegistryError(
                f"invalid run transition for {run_id}: {previous!r} -> {status!r}"
            )
        os.lseek(descriptor, 0, os.SEEK_END)
        if raw and not raw.endswith(b"\n"):
            os.write(descriptor, b"\n")
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    return event


def current_run_status(path: str | Path, run_id: str) -> str | None:
    status = None
    for row in _read_events(Path(path)):
        if row.get("run_id") == run_id:
            status = str(row.get("status"))
    return status


def completed_run_for_generation(
    path: str | Path,
    generation_id: str,
) -> str | None:
    """Return the unique completed publication run for one generation."""

    matches = {
        str(row.get("run_id"))
        for row in _read_events(Path(path))
        if row.get("generation_id") == generation_id
        and row.get("status") in {"completed", "publication_reviewed", "published"}
    }
    if not matches:
        return None
    if len(matches) != 1:
        raise RunRegistryError(
            f"generation_id is associated with multiple runs: {generation_id}"
        )
    return next(iter(matches))


__all__ = [
    "RUN_REGISTRY_SCHEMA",
    "RunEvent",
    "RunRegistryError",
    "append_run_event",
    "completed_run_for_generation",
    "current_run_status",
]
