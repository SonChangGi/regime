#!/usr/bin/env python3
"""Rebuild one selection-family audit into a new local research output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from regime_lab.config import project_root  # noqa: E402
from regime_lab.io import write_json_atomic  # noqa: E402
from regime_lab.path_safety import confined_mutable_path  # noqa: E402
from regime_lab.selection_family_audit import (  # noqa: E402
    build_selection_family_audit_from_artifacts,
    validate_selection_family_audit,
)


class ReplayError(RuntimeError):
    """Raised when a replay input/output is unsafe or inconsistent."""


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReplayError(f"{label} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayError(f"{label} must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ReplayError(f"{label} must be a JSON object")
    return value


def rebuild_selection_family_audit(
    *,
    payload_path: str | Path,
    artifact_directory: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    payload_file = Path(payload_path)
    artifacts = Path(artifact_directory)
    output = confined_mutable_path(
        output_path,
        project_directory=project_root(),
        label="selection-family replay output",
    )
    if output.exists() or output.is_symlink():
        raise ReplayError(f"output already exists; refusing overwrite: {output}")
    payload = _read_json(payload_file, label="selection-family replay payload")
    try:
        document = build_selection_family_audit_from_artifacts(payload, artifacts)
        validate_selection_family_audit(
            document,
            expected_generation_id=str(payload["meta"]["generation_id"]),
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ReplayError(f"selection-family replay failed: {exc}") from exc
    write_json_atomic(output, document)
    persisted = _read_json(output, label="persisted selection-family replay")
    if persisted != document:
        output.unlink(missing_ok=True)
        raise ReplayError("persisted selection-family replay differs from memory")
    return {
        "ok": True,
        "output": str(output),
        "generation_id": document["generation_id"],
        "evidence_status": document["evidence_status"],
        "candidate_count": document["candidate_count"],
        "selection_origin_count": document["common_origin_contract"][
            "origin_count"
        ],
        "sha256": document["sha256"],
        "source_artifacts": document["source_artifacts"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild a selection-family-audit/v2 from immutable V5 selection "
            "sources without modifying the source artifact generation"
        )
    )
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = rebuild_selection_family_audit(
            payload_path=args.payload,
            artifact_directory=args.artifacts,
            output_path=args.output,
        )
    except (OSError, ReplayError, ValueError) as exc:
        print(f"selection-family replay refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
