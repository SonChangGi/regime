#!/usr/bin/env python3
"""Create a fail-closed, synthetic-only static dashboard package.

The live dashboard payload can contain provider-derived observations.  This
script deliberately copies only the three static application assets and a
payload that identifies every source as a synthetic fixture.  It never copies
the repository, data directory, model artifacts, or arbitrary files from the
web root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
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


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_public_file(root: Path, relative_path: str, value: bytes) -> None:
    target = root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(value)
    target.chmod(0o644)


def package_public_demo(
    *,
    web_root: str | Path,
    payload_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Build a new static directory from an explicit, minimal allowlist."""

    web_root = Path(web_root)
    payload_path = Path(payload_path)
    output_directory = confined_mutable_path(
        output_directory,
        project_directory=project_root(),
        label="public demo output",
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
    validate_public_demo_payload(payload)
    files[PAYLOAD_DESTINATION] = payload_raw

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "package_kind": "synthetic_demo_only",
        "payload_mode": "demo",
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package an explicitly synthetic dashboard for static preview"
    )
    parser.add_argument("--web-root", type=Path, default=Path("web"))
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("dist/public-demo")
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = package_public_demo(
            web_root=args.web_root,
            payload_path=args.payload,
            output_directory=args.output,
        )
    except (PackagingError, UnsafeMutablePath, OSError) as exc:
        print(f"public demo package refused: {exc}", file=sys.stderr)
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
