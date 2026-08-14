#!/usr/bin/env python3
"""Verify the exact allowlisted package before public upload."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.package_public_demo import (  # noqa: E402
    MANIFEST_DESTINATION,
    PAYLOAD_DESTINATION,
    PUBLICATION_MODE_DEMO,
    PUBLICATION_MODE_LIVE_DERIVED,
    STATIC_ALLOWLIST,
    PackagingError,
    validate_public_payload,
)


EXPECTED_FILES = frozenset(
    (*STATIC_ALLOWLIST, PAYLOAD_DESTINATION, MANIFEST_DESTINATION)
)
EXPECTED_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "package_kind",
        "payload_mode",
        "publication_scope",
        "contains_raw_observations",
        "source_ids",
        "payload_data_as_of",
        "files",
    }
)
SECRET_PATTERNS = (
    re.compile(rb"(?i)FRED_API_KEY\s*="),
    re.compile(rb"(?i)ALPHA_VANTAGE_API_KEY\s*="),
    re.compile(rb"(?i)api[_-]?key\s*[=:]\s*[^\s\"']{8,}"),
    re.compile(rb"(?i)apikey=[^&\s\"']{8,}"),
    re.compile(rb"(?i)authorization\s*[:=]\s*(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}"),
)
PACKAGE_MODE_BY_KIND = {
    "synthetic_demo_only": PUBLICATION_MODE_DEMO,
    "personal_noncommercial_live_derived": PUBLICATION_MODE_LIVE_DERIVED,
}


class VerificationError(RuntimeError):
    """Raised when a package is not safe to upload."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"{label} must be valid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"{label} root must be a JSON object")
    return value


def verify_public_package(directory: str | Path) -> dict[str, Any]:
    package_root = Path(directory)
    if package_root.is_symlink() or not package_root.is_dir():
        raise VerificationError(f"package root must be a real directory: {package_root}")

    actual_files: set[str] = set()
    for path in package_root.rglob("*"):
        if path.is_symlink():
            raise VerificationError(f"package must not contain symbolic links: {path}")
        if path.is_file():
            actual_files.add(path.relative_to(package_root).as_posix())
    if actual_files != EXPECTED_FILES:
        missing = sorted(EXPECTED_FILES - actual_files)
        extra = sorted(actual_files - EXPECTED_FILES)
        raise VerificationError(
            f"package inventory mismatch (missing={missing}, extra={extra})"
        )

    manifest = _load_json(
        package_root / MANIFEST_DESTINATION,
        label="publication manifest",
    )
    if set(manifest) != EXPECTED_MANIFEST_KEYS:
        raise VerificationError("publication manifest keys are not exact")
    manifest_raw = (package_root / MANIFEST_DESTINATION).read_bytes()
    if any(pattern.search(manifest_raw) for pattern in SECRET_PATTERNS):
        raise VerificationError("credential-like material found in publication manifest")
    if manifest.get("schema_version") != "1.0":
        raise VerificationError("publication manifest schema_version must be 1.0")
    package_kind = manifest.get("package_kind")
    publication_mode = PACKAGE_MODE_BY_KIND.get(package_kind)
    if publication_mode is None:
        raise VerificationError("publication manifest package_kind is unsupported")
    expected_payload_mode = (
        "live" if publication_mode == PUBLICATION_MODE_LIVE_DERIVED else "demo"
    )
    if manifest.get("payload_mode") != expected_payload_mode:
        raise VerificationError(
            f"payload_mode must be {expected_payload_mode} for {package_kind}"
        )
    expected_scope = (
        "personal_noncommercial_derived_results"
        if publication_mode == PUBLICATION_MODE_LIVE_DERIVED
        else "synthetic_fixture"
    )
    if manifest.get("publication_scope") != expected_scope:
        raise VerificationError("publication_scope does not match package_kind")
    if manifest.get("contains_raw_observations") is not False:
        raise VerificationError("contains_raw_observations must be false")

    manifest_files = manifest.get("files")
    expected_manifest_files = EXPECTED_FILES - {MANIFEST_DESTINATION}
    if not isinstance(manifest_files, dict) or set(manifest_files) != expected_manifest_files:
        raise VerificationError("publication manifest file inventory is not exact")

    for relative_path in sorted(expected_manifest_files):
        raw = (package_root / relative_path).read_bytes()
        record = manifest_files.get(relative_path)
        if not isinstance(record, dict) or set(record) != {"bytes", "sha256"}:
            raise VerificationError(f"manifest record is invalid: {relative_path}")
        if record.get("bytes") != len(raw):
            raise VerificationError(f"byte count mismatch: {relative_path}")
        if record.get("sha256") != hashlib.sha256(raw).hexdigest():
            raise VerificationError(f"SHA-256 mismatch: {relative_path}")
        if any(pattern.search(raw) for pattern in SECRET_PATTERNS):
            raise VerificationError(f"credential-like material found: {relative_path}")

    payload = _load_json(
        package_root / PAYLOAD_DESTINATION,
        label="dashboard payload",
    )
    try:
        validate_public_payload(
            payload,
            publication_mode=publication_mode,
            rights_acknowledged=publication_mode == PUBLICATION_MODE_LIVE_DERIVED,
        )
    except PackagingError as exc:
        raise VerificationError(str(exc)) from exc
    payload_source_ids = sorted(
        source["id"]
        for source in payload.get("sources", [])
        if isinstance(source, dict) and isinstance(source.get("id"), str)
    )
    if manifest.get("source_ids") != payload_source_ids:
        raise VerificationError("publication manifest source_ids mismatch")
    if manifest.get("payload_data_as_of") != payload.get("meta", {}).get("data_as_of"):
        raise VerificationError("publication manifest payload_data_as_of mismatch")

    return {
        "ok": True,
        "package_kind": package_kind,
        "payload_mode": manifest["payload_mode"],
        "payload_data_as_of": manifest.get("payload_data_as_of"),
        "files": sorted(actual_files),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify an allowlisted Regime package before Pages upload"
    )
    parser.add_argument("directory", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = verify_public_package(args.directory)
    except VerificationError as exc:
        print(f"public package verification refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
