"""Fail-fast checks for an expensive local v5 research build."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import importlib.util
from pathlib import Path
import sqlite3
from typing import Any

from regime_lab.config import project_root
from regime_lab.frozen_v4 import verify_frozen_v4_baseline


STRUCTURAL_V5_PREREGISTRATION_SHA256 = (
    "a8c72cb0dd90ee6de7c7300521f476a7e82ea4ead1ef666014714ccc8808ca24"
)


class V5PreflightError(RuntimeError):
    """The local v5 build is not safe to start or publish."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_structural_v5_preregistration(
    *,
    project_directory: Path | None = None,
) -> str:
    """Verify the preregistration bytes before any provider is contacted."""

    root = (project_directory or project_root()).resolve()
    path = root / "config" / "structural_v5.json"
    if path.is_symlink() or not path.is_file():
        raise V5PreflightError("v5 structural preregistration is missing")
    actual = _sha256(path)
    if actual != STRUCTURAL_V5_PREREGISTRATION_SHA256:
        raise V5PreflightError(
            "v5 structural preregistration SHA-256 does not match the code contract"
        )
    return actual


def _fingerprint_paths(root: Path) -> tuple[Path, ...]:
    required = (
        root / "pyproject.toml",
        root / "config" / "series.json",
        root / "config" / "structural_v4.json",
        root / "config" / "structural_v5.json",
    )
    source_root = root / "src" / "regime_lab"
    if source_root.is_symlink() or not source_root.is_dir():
        raise V5PreflightError("v5 analysis source directory is missing")
    paths = (*required, *source_root.rglob("*.py"))
    checked: list[Path] = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            try:
                label = path.relative_to(root).as_posix()
            except ValueError:
                label = path.name
            raise V5PreflightError(f"v5 analysis input is missing or unsafe: {label}")
        checked.append(path)
    return tuple(sorted(set(checked), key=lambda item: item.relative_to(root).as_posix()))


def v5_analysis_source_fingerprint(
    *,
    project_directory: Path | None = None,
) -> tuple[str, int]:
    """Hash every source/config byte that defines a v5 model generation."""

    root = (project_directory or project_root()).resolve()
    digest = hashlib.sha256()
    paths = _fingerprint_paths(root)
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        body = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(body).to_bytes(8, "big"))
        digest.update(body)
    return digest.hexdigest(), len(paths)


def require_v5_analysis_source_unchanged(
    expected_sha256: str,
    *,
    project_directory: Path | None = None,
) -> None:
    """Block candidate publication if code/config changed during training."""

    actual, _ = v5_analysis_source_fingerprint(
        project_directory=project_directory,
    )
    if actual != expected_sha256:
        raise V5PreflightError(
            "v5 analysis source/config changed during the build; candidate retained unpublished"
        )


def _database_quick_check(path: Path) -> str:
    if not path.exists():
        return "not_present"
    if path.is_symlink() or not path.is_file():
        raise V5PreflightError("v5 snapshot database is not a regular file")
    uri = f"{path.resolve().as_uri()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True, timeout=30) as connection:
            rows = tuple(str(row[0]) for row in connection.execute("PRAGMA quick_check"))
    except sqlite3.Error as exc:
        raise V5PreflightError("v5 snapshot database quick_check failed") from exc
    if rows != ("ok",):
        raise V5PreflightError("v5 snapshot database quick_check failed")
    return "ok"


def _require_profile_dependencies(profile: str) -> None:
    required = ["xgboost"]
    if profile == "full":
        required.append("hmmlearn")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise V5PreflightError(
            "v5 profile dependency is unavailable: " + ", ".join(missing)
        )


def verify_v5_preflight(
    *,
    profile: str,
    database_path: Path,
    project_directory: Path | None = None,
) -> dict[str, Any]:
    """Return a secret-free receipt for the checks required before collection."""

    if profile not in {"standard", "full"}:
        raise V5PreflightError("v5 live build profile must be standard or full")
    root = (project_directory or project_root()).resolve()
    preregistration_sha256 = verify_structural_v5_preregistration(
        project_directory=root,
    )
    baseline: Mapping[str, Any] = verify_frozen_v4_baseline(
        project_directory=root,
    )
    _require_profile_dependencies(profile)
    database_status = _database_quick_check(database_path)
    source_sha256, source_file_count = v5_analysis_source_fingerprint(
        project_directory=root,
    )
    return {
        "profile": profile,
        "structural_v5_sha256": preregistration_sha256,
        "frozen_v4_payload_sha256": str(baseline["payload_sha256"]),
        "source_fingerprint_sha256": source_sha256,
        "source_file_count": source_file_count,
        "database_quick_check": database_status,
    }


__all__ = [
    "STRUCTURAL_V5_PREREGISTRATION_SHA256",
    "V5PreflightError",
    "require_v5_analysis_source_unchanged",
    "v5_analysis_source_fingerprint",
    "verify_structural_v5_preregistration",
    "verify_v5_preflight",
]
