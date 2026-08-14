#!/usr/bin/env python3
"""Create a fail-closed static dashboard package.

Only the three static application assets and one explicitly selected dashboard
payload are copied.  Live publication is limited to the user's personal,
non-commercial *derived result* snapshot: raw observations, databases, model
artifacts, credentials, and arbitrary files are never copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any

from regime_lab.schema import ContractError, validate_dashboard_payload
from regime_lab.config import project_root
from regime_lab.path_safety import UnsafeMutablePath, confined_mutable_path


STATIC_ALLOWLIST = ("index.html", "styles.css", "app.js")
PAYLOAD_DESTINATION = "data/regime-results.json"
MANIFEST_DESTINATION = "publication-manifest.json"
PUBLICATION_MODE_DEMO = "demo"
PUBLICATION_MODE_LIVE_DERIVED = "live-derived"
LIVE_SOURCE_LICENSES = {
    "alpha_vantage": "private_noncommercial",
    "alfred": "user_confirmed_ml_storage_derived",
}
FORBIDDEN_LIVE_DERIVED_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "credential",
        "credentials",
        "password",
        "provider_response",
        "provider_responses",
        "raw_observation",
        "raw_observations",
        "request_params",
        "revision_seq",
        "secret",
        "snapshot_id",
        "token",
        "vintage_date",
    }
)


class PackagingError(RuntimeError):
    """Raised when an input is not safe to package for public demo use."""


def _read_regular_file(path: Path, *, label: str) -> bytes:
    if path.is_symlink():
        raise PackagingError(f"{label} must not be a symbolic link: {path}")
    if not path.is_file():
        raise PackagingError(f"{label} is missing or is not a regular file: {path}")
    return path.read_bytes()


def _decode_payload(raw: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackagingError("payload must be valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise PackagingError("payload root must be a JSON object")
    return payload


def validate_public_demo_payload(payload: dict[str, Any]) -> None:
    """Fail closed unless the payload is an explicitly synthetic demo."""

    # Rights metadata is necessary but not sufficient for publication.  The
    # packaged document must also satisfy the exact dashboard contract; this
    # prevents a synthetic-looking but malformed payload from becoming a
    # public page that only fails after browser load.
    try:
        validate_dashboard_payload(payload)
    except ContractError as exc:
        raise PackagingError(f"dashboard payload contract is invalid: {exc}") from exc

    meta = payload.get("meta")
    if not isinstance(meta, dict) or meta.get("mode") != "demo":
        mode = meta.get("mode") if isinstance(meta, dict) else None
        raise PackagingError(f"only meta.mode=demo may be packaged (got {mode!r})")

    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise PackagingError("demo payload must contain a non-empty sources array")
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise PackagingError(f"sources[{index}] must be an object")
        source_id = source.get("id")
        license_class = source.get("license_class")
        if not isinstance(source_id, str) or not source_id.startswith("synthetic_"):
            raise PackagingError(
                f"sources[{index}].id must identify a synthetic fixture"
            )
        if license_class != "synthetic_fixture":
            raise PackagingError(
                f"sources[{index}].license_class must be synthetic_fixture"
            )


def _normalized_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _reject_raw_provider_material(value: object, *, path: str = "payload") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = _normalized_key(key)
            is_provider_observation = (
                path.startswith("payload.sources[")
                and normalized in {"observation", "observations"}
            )
            if normalized in FORBIDDEN_LIVE_DERIVED_KEYS or is_provider_observation:
                raise PackagingError(
                    f"live-derived payload contains forbidden provider material at {path}.{key}"
                )
            _reject_raw_provider_material(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_raw_provider_material(child, path=f"{path}[{index}]")
    elif isinstance(value, str) and (
        value.startswith(("/Users/", "/private/", "file://")) or "\\Users\\" in value
    ):
        raise PackagingError(
            f"live-derived payload contains a machine-local path at {path}"
        )


def validate_public_live_derived_payload(payload: dict[str, Any]) -> None:
    """Validate a personal, non-commercial snapshot containing derived results only."""

    try:
        validate_dashboard_payload(payload)
    except ContractError as exc:
        raise PackagingError(f"dashboard payload contract is invalid: {exc}") from exc

    meta = payload.get("meta")
    if not isinstance(meta, dict) or meta.get("mode") != "live":
        mode = meta.get("mode") if isinstance(meta, dict) else None
        raise PackagingError(f"live-derived publication requires meta.mode=live (got {mode!r})")
    if meta.get("result_version") != "weekly-regime-result-v4":
        raise PackagingError("live-derived publication requires weekly-regime-result-v4")

    weekly = payload.get("weekly")
    if not isinstance(weekly, list) or len(weekly) < 52:
        count = len(weekly) if isinstance(weekly, list) else 0
        raise PackagingError(
            f"live-derived publication requires at least 52 weekly results (got {count})"
        )

    sources = payload.get("sources")
    if not isinstance(sources, list):
        raise PackagingError("live-derived payload must contain sources")
    by_id: dict[str, dict[str, Any]] = {}
    for index, source in enumerate(sources):
        if not isinstance(source, dict) or not isinstance(source.get("id"), str):
            raise PackagingError(f"sources[{index}] must contain an id")
        source_id = source["id"]
        if source_id in by_id:
            raise PackagingError(f"duplicate live-derived source: {source_id}")
        by_id[source_id] = source
    if set(by_id) != set(LIVE_SOURCE_LICENSES):
        raise PackagingError(
            "live-derived sources must be exactly alpha_vantage and alfred"
        )
    for source_id, expected_license in LIVE_SOURCE_LICENSES.items():
        actual = by_id[source_id].get("license_class")
        if actual != expected_license:
            raise PackagingError(
                f"{source_id}.license_class must be {expected_license}"
            )

    _reject_raw_provider_material(payload)


def validate_public_payload(
    payload: dict[str, Any],
    *,
    publication_mode: str,
    rights_acknowledged: bool,
) -> None:
    if publication_mode == PUBLICATION_MODE_DEMO:
        validate_public_demo_payload(payload)
        return
    if publication_mode == PUBLICATION_MODE_LIVE_DERIVED:
        if rights_acknowledged is not True:
            raise PackagingError(
                "live-derived publication requires explicit personal/non-commercial rights acknowledgement"
            )
        validate_public_live_derived_payload(payload)
        return
    raise PackagingError(f"unsupported publication mode: {publication_mode!r}")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_public_file(root: Path, relative_path: str, value: bytes) -> None:
    target = root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(value)
    target.chmod(0o644)


def package_public_dashboard(
    *,
    web_root: str | Path,
    payload_path: str | Path,
    output_directory: str | Path,
    publication_mode: str = PUBLICATION_MODE_DEMO,
    rights_acknowledged: bool = False,
) -> dict[str, Any]:
    """Build a new static directory from an explicit, minimal allowlist."""

    web_root = Path(web_root)
    payload_path = Path(payload_path)
    output_directory = confined_mutable_path(
        output_directory,
        project_directory=project_root(),
        label="public dashboard output",
    )

    if output_directory.exists() or output_directory.is_symlink():
        raise PackagingError(
            f"output already exists; refusing to overwrite it: {output_directory}"
        )

    files: dict[str, bytes] = {}
    for relative_path in STATIC_ALLOWLIST:
        files[relative_path] = _read_regular_file(
            web_root / relative_path,
            label=f"allowlisted web asset {relative_path}",
        )

    payload_raw = _read_regular_file(payload_path, label="dashboard payload")
    payload = _decode_payload(payload_raw)
    validate_public_payload(
        payload,
        publication_mode=publication_mode,
        rights_acknowledged=rights_acknowledged,
    )
    files[PAYLOAD_DESTINATION] = payload_raw

    is_live_derived = publication_mode == PUBLICATION_MODE_LIVE_DERIVED
    source_ids = sorted(
        source["id"]
        for source in payload.get("sources", [])
        if isinstance(source, dict) and isinstance(source.get("id"), str)
    )

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "package_kind": (
            "personal_noncommercial_live_derived"
            if is_live_derived
            else "synthetic_demo_only"
        ),
        "payload_mode": "live" if is_live_derived else "demo",
        "publication_scope": (
            "personal_noncommercial_derived_results"
            if is_live_derived
            else "synthetic_fixture"
        ),
        "contains_raw_observations": False,
        "source_ids": source_ids,
        "payload_data_as_of": payload["meta"].get("data_as_of"),
        "files": {
            path: {"bytes": len(value), "sha256": _sha256(value)}
            for path, value in sorted(files.items())
        },
    }
    manifest_raw = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.",
            dir=output_directory.parent,
        )
    )
    staging.chmod(0o755)
    try:
        for relative_path, value in files.items():
            _write_public_file(staging, relative_path, value)
        _write_public_file(staging, MANIFEST_DESTINATION, manifest_raw)
        if output_directory.exists() or output_directory.is_symlink():
            raise PackagingError(
                f"output appeared during packaging; refusing overwrite: {output_directory}"
            )
        os.replace(staging, output_directory)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def package_public_demo(
    *,
    web_root: str | Path,
    payload_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Backward-compatible synthetic demo wrapper."""

    return package_public_dashboard(
        web_root=web_root,
        payload_path=payload_path,
        output_directory=output_directory,
        publication_mode=PUBLICATION_MODE_DEMO,
        rights_acknowledged=False,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package an allowlisted Regime dashboard for static publication"
    )
    parser.add_argument("--web-root", type=Path, default=Path("web"))
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("dist/public-dashboard")
    )
    parser.add_argument(
        "--publication-mode",
        choices=(PUBLICATION_MODE_DEMO, PUBLICATION_MODE_LIVE_DERIVED),
        default=PUBLICATION_MODE_DEMO,
    )
    parser.add_argument(
        "--acknowledge-personal-noncommercial-publication",
        action="store_true",
        help="Required for live-derived publication; never expands provider rights",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = package_public_dashboard(
            web_root=args.web_root,
            payload_path=args.payload,
            output_directory=args.output,
            publication_mode=args.publication_mode,
            rights_acknowledged=args.acknowledge_personal_noncommercial_publication,
        )
    except (PackagingError, UnsafeMutablePath, OSError) as exc:
        print(f"public dashboard package refused: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output),
                "package_kind": manifest["package_kind"],
                "files": sorted(
                    [*manifest["files"], MANIFEST_DESTINATION]
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
