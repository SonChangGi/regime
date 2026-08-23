#!/usr/bin/env python3
"""Freeze the audited v4 payload and artifact bundle without rewriting bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from regime_lab.config import project_root


PAYLOAD_SHA256 = "e58eda3f5519e1c3c340c671e6c6c1c69279dae068f9c21f9bedfde22e03b96b"
INVENTORY_SHA256 = "3b0ffe79dea816b2a47c22ecba7eebb9b8fa8f4e9e2bb4ccba30f982d69c7613"
GENERATION_ID = "20260813T190841.471317Z"
BASELINE_NAME = "v4-20260821"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_files(root: Path) -> dict[str, Path]:
    payload = (
        root
        / "publication"
        / "baselines"
        / BASELINE_NAME
        / "regime-results.json"
    )
    artifacts = root / "artifacts" / "latest"
    files = {"regime-results.json": payload}
    files.update(
        {
            path.name: path
            for path in sorted(artifacts.iterdir())
            if path.is_file()
        }
    )
    if len(files) != 23:
        raise RuntimeError(f"expected 23 frozen files, found {len(files)}")
    if any(not path.is_file() for path in files.values()):
        raise RuntimeError("a frozen v4 source file is missing")
    return files


def _inventory(files: dict[str, Path]) -> bytes:
    return "".join(
        f"{_sha256(files[name])}  {name}\n" for name in sorted(files)
    ).encode("utf-8")


def _verify_sources(root: Path) -> tuple[dict[str, Path], bytes]:
    files = _source_files(root)
    if _sha256(files["regime-results.json"]) != PAYLOAD_SHA256:
        raise RuntimeError("audited v4 payload hash changed")
    payload = json.loads(files["regime-results.json"].read_text(encoding="utf-8"))
    generation = json.loads(files["build-generation.json"].read_text(encoding="utf-8"))
    if payload.get("meta", {}).get("generation_id") != GENERATION_ID:
        raise RuntimeError("audited v4 payload generation changed")
    if generation.get("generation_id") != GENERATION_ID:
        raise RuntimeError("audited v4 artifact generation changed")
    inventory = _inventory(files)
    if hashlib.sha256(inventory).hexdigest() != INVENTORY_SHA256:
        raise RuntimeError("audited v4 artifact inventory changed")
    return files, inventory


def freeze(*, write: bool) -> Path:
    root = project_root()
    destination = root / "artifacts" / "baselines" / BASELINE_NAME
    if destination.is_symlink():
        raise RuntimeError("existing v4 baseline must not be a symbolic link")
    if destination.exists():
        if not destination.is_dir():
            raise RuntimeError("existing v4 baseline must be a directory")
        frozen = {
            path.name: path
            for path in destination.iterdir()
            if path.is_file() and path.name != "SHA256SUMS"
        }
        if len(frozen) != 23:
            raise RuntimeError("existing v4 baseline file count differs")
        inventory = _inventory(frozen)
        if hashlib.sha256(inventory).hexdigest() != INVENTORY_SHA256:
            raise RuntimeError("existing v4 baseline differs from audited bytes")
        if (destination / "SHA256SUMS").read_bytes() != inventory:
            raise RuntimeError("existing v4 baseline inventory file differs")
        if _sha256(frozen["regime-results.json"]) != PAYLOAD_SHA256:
            raise RuntimeError("existing v4 payload differs from audited bytes")
        return destination

    files, inventory = _verify_sources(root)
    if not write:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{BASELINE_NAME}-",
            dir=destination.parent,
        )
    )
    try:
        for name, source in files.items():
            shutil.copy2(source, staging / name)
        (staging / "SHA256SUMS").write_bytes(inventory)
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    destination = freeze(write=args.write)
    print(
        json.dumps(
            {
                "valid": True,
                "written": bool(args.write),
                "destination": str(destination),
                "payload_sha256": PAYLOAD_SHA256,
                "artifacts_inventory_sha256": INVENTORY_SHA256,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
