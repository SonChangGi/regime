"""Canonical whole-generation inventories for private analysis artifacts."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import tempfile
from typing import Any


ARTIFACT_INVENTORY_FILENAME = "SHA256SUMS"
_INVENTORY_LINE = re.compile(
    r"(?P<sha256>[0-9a-f]{64})  (?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
)


class ArtifactInventoryError(RuntimeError):
    """Raised when an artifact generation is incomplete or has changed."""


def _require_directory(directory: Path) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise ArtifactInventoryError(
            f"artifact generation must be a real directory: {directory}"
        )


def _sha256_regular_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ArtifactInventoryError(
            f"artifact inventory accepts regular files only: {path.name}"
        )
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity_fields):
        raise ArtifactInventoryError(
            f"artifact changed while hashing: {path.name}"
        )
    if path.is_symlink() or not path.is_file():
        raise ArtifactInventoryError(
            f"artifact changed type while hashing: {path.name}"
        )
    return digest.hexdigest()


def _generation_files(directory: Path) -> dict[str, Path]:
    _require_directory(directory)
    files: dict[str, Path] = {}
    for path in directory.iterdir():
        if path.name == ARTIFACT_INVENTORY_FILENAME:
            continue
        if _INVENTORY_LINE.fullmatch(f"{'0' * 64}  {path.name}") is None:
            raise ArtifactInventoryError(
                f"artifact filename cannot be represented canonically: {path.name}"
            )
        if path.is_symlink() or not path.is_file():
            raise ArtifactInventoryError(
                f"artifact generation contains a non-regular entry: {path.name}"
            )
        files[path.name] = path
    if not files:
        raise ArtifactInventoryError("artifact generation contains no files")
    return files


def canonical_artifact_inventory_bytes(directory: str | Path) -> bytes:
    """Hash every final top-level file except ``SHA256SUMS`` itself."""

    files = _generation_files(Path(directory))
    return "".join(
        f"{_sha256_regular_file(files[name])}  {name}\n"
        for name in sorted(files)
    ).encode("ascii")


def write_artifact_inventory(directory: str | Path) -> Path:
    """Atomically write and immediately verify a canonical generation inventory."""

    root = Path(directory)
    inventory = canonical_artifact_inventory_bytes(root)
    target = root / ARTIFACT_INVENTORY_FILENAME
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=root,
            prefix=f".{ARTIFACT_INVENTORY_FILENAME}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(inventory)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, target)
        temporary = None
        verify_artifact_inventory(root)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return target


def _parse_inventory(raw: bytes) -> dict[str, str]:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ArtifactInventoryError("artifact SHA256SUMS must be ASCII") from exc
    if not text.endswith("\n"):
        raise ArtifactInventoryError(
            "artifact SHA256SUMS must end with a newline"
        )
    entries: dict[str, str] = {}
    for line in text.splitlines():
        match = _INVENTORY_LINE.fullmatch(line)
        if match is None:
            raise ArtifactInventoryError(
                "artifact SHA256SUMS contains an invalid row"
            )
        name = match.group("name")
        if name in entries or name == ARTIFACT_INVENTORY_FILENAME:
            raise ArtifactInventoryError(
                "artifact SHA256SUMS contains a duplicate/reserved entry"
            )
        entries[name] = match.group("sha256")
    if not entries:
        raise ArtifactInventoryError("artifact SHA256SUMS contains no entries")
    canonical = "".join(
        f"{entries[name]}  {name}\n" for name in sorted(entries)
    ).encode("ascii")
    if canonical != raw:
        raise ArtifactInventoryError("artifact SHA256SUMS is not canonical")
    return entries


def verify_artifact_inventory(directory: str | Path) -> dict[str, Any]:
    """Fail closed on inventory syntax, file-set drift, or byte-hash drift."""

    root = Path(directory)
    _require_directory(root)
    inventory_path = root / ARTIFACT_INVENTORY_FILENAME
    if inventory_path.is_symlink() or not inventory_path.is_file():
        raise ArtifactInventoryError("artifact SHA256SUMS is missing/non-regular")
    raw = inventory_path.read_bytes()
    entries = _parse_inventory(raw)
    files = _generation_files(root)
    if set(files) != set(entries):
        missing = sorted(set(entries) - set(files))
        extra = sorted(set(files) - set(entries))
        raise ArtifactInventoryError(
            f"artifact inventory file set differs (missing={missing}, extra={extra})"
        )
    for name in sorted(entries):
        if _sha256_regular_file(files[name]) != entries[name]:
            raise ArtifactInventoryError(
                f"artifact hash does not match SHA256SUMS: {name}"
            )
    return {
        "path": ARTIFACT_INVENTORY_FILENAME,
        "file_count": len(entries),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


__all__ = [
    "ARTIFACT_INVENTORY_FILENAME",
    "ArtifactInventoryError",
    "canonical_artifact_inventory_bytes",
    "verify_artifact_inventory",
    "write_artifact_inventory",
]
