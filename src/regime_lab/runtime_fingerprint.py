"""Deterministic runtime identity for reproducible local model execution."""

from __future__ import annotations

import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import platform
import re
from typing import Any, Mapping


RUNTIME_FINGERPRINT_SCHEMA = "regime-runtime-fingerprint/1"
LOCKFILE_NAME = "requirements-ci.lock"
MODEL_RUNTIME_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "xgboost",
    "hmmlearn",
    "joblib",
    "threadpoolctl",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


class RuntimeFingerprintError(RuntimeError):
    """Raised when the locked model runtime cannot be identified exactly."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def build_runtime_fingerprint(
    root: str | Path,
    *,
    require_lock: bool = True,
) -> dict[str, Any]:
    project = Path(root).resolve()
    lockfile = project / LOCKFILE_NAME
    if lockfile.is_symlink() or (require_lock and not lockfile.is_file()):
        raise RuntimeFingerprintError(
            f"locked runtime file is missing or unsafe: {lockfile}"
        )
    lock_sha256 = (
        hashlib.sha256(lockfile.read_bytes()).hexdigest()
        if lockfile.is_file()
        else None
    )
    packages: dict[str, str | None] = {}
    for package in MODEL_RUNTIME_PACKAGES:
        try:
            packages[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            packages[package] = None
    identity: dict[str, Any] = {
        "schema_version": RUNTIME_FINGERPRINT_SCHEMA,
        "python": platform.python_version(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "lockfile": {
            "path": LOCKFILE_NAME,
            "sha256": lock_sha256,
        },
        "packages": packages,
    }
    return {
        **identity,
        "sha256": hashlib.sha256(_canonical_bytes(identity)).hexdigest(),
    }


def validate_runtime_fingerprint(value: Mapping[str, Any]) -> str:
    expected_fields = {
        "schema_version",
        "python",
        "platform",
        "lockfile",
        "packages",
        "sha256",
    }
    if set(value) != expected_fields:
        raise RuntimeFingerprintError("runtime fingerprint fields are invalid")
    if value.get("schema_version") != RUNTIME_FINGERPRINT_SCHEMA:
        raise RuntimeFingerprintError("runtime fingerprint schema is invalid")
    checksum = value.get("sha256")
    if not isinstance(checksum, str) or _SHA256.fullmatch(checksum) is None:
        raise RuntimeFingerprintError("runtime fingerprint SHA-256 is invalid")
    identity = dict(value)
    identity.pop("sha256")
    actual = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
    if actual != checksum:
        raise RuntimeFingerprintError("runtime fingerprint checksum mismatch")
    lockfile = value.get("lockfile")
    if not isinstance(lockfile, Mapping):
        raise RuntimeFingerprintError("runtime lockfile identity is invalid")
    lock_checksum = lockfile.get("sha256")
    if not isinstance(lock_checksum, str) or _SHA256.fullmatch(lock_checksum) is None:
        raise RuntimeFingerprintError("runtime lockfile SHA-256 is invalid")
    return checksum


__all__ = [
    "LOCKFILE_NAME",
    "MODEL_RUNTIME_PACKAGES",
    "RUNTIME_FINGERPRINT_SCHEMA",
    "RuntimeFingerprintError",
    "build_runtime_fingerprint",
    "validate_runtime_fingerprint",
]
