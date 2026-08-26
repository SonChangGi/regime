#!/usr/bin/env python3
"""Run the private, derived-only five-track mechanism ablation."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from regime_lab.config import load_config, project_root
from regime_lab.mechanism_ablation_run import (
    run_real_mechanism_ablation,
    write_mechanism_ablation_generation,
)
from regime_lab.path_safety import confined_mutable_path


def _aware_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--as-of must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("--as-of must include a timezone")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run matched-origin state/label/market/macro/full diagnostics. "
            "The result cannot select or promote a production model."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("config/series.json"))
    parser.add_argument("--database", type=Path, default=Path("data/regime.sqlite3"))
    parser.add_argument("--as-of", type=_aware_datetime, required=True)
    parser.add_argument(
        "--profile",
        choices=("quick", "standard", "full"),
        default="quick",
    )
    parser.add_argument(
        "--role-manifest",
        type=Path,
        default=Path("config/feature-role-manifest-v2.json"),
    )
    parser.add_argument(
        "--specification",
        type=Path,
        default=Path("config/mechanism-ablation-v2.json"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("build/mechanism-ablation"),
    )
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=Path("build/private-checkpoints/mechanism-ablation"),
        help=(
            "Private identity-bound origin checkpoints. Reusing this path resumes "
            "a compatible standard/full run and creates a new namespace otherwise."
        ),
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable origin checkpointing (intended only for isolated smoke tests).",
    )
    args = parser.parse_args()

    root = project_root()
    database = args.database if args.database.is_absolute() else root / args.database
    role_manifest = (
        args.role_manifest
        if args.role_manifest.is_absolute()
        else root / args.role_manifest
    )
    specification = (
        args.specification
        if args.specification.is_absolute()
        else root / args.specification
    )
    output = confined_mutable_path(
        args.output_directory,
        project_directory=root,
        label="mechanism ablation output",
    )
    checkpoint = None
    if not args.no_checkpoint:
        checkpoint = confined_mutable_path(
            args.checkpoint_directory,
            project_directory=root,
            label="mechanism ablation checkpoint",
        )

    config = load_config(args.config)
    report, frames = run_real_mechanism_ablation(
        config,
        database=database,
        as_of=args.as_of,
        profile_name=args.profile,
        role_manifest_path=role_manifest,
        specification_path=specification,
        checkpoint_directory=checkpoint,
        progress=lambda message: print(message, flush=True),
    )
    generation = write_mechanism_ablation_generation(
        output,
        report,
        frames,
        expected_source_fingerprint_sha256=report["input"][
            "analysis_source_fingerprint_sha256"
        ],
        source_config=config,
        role_manifest_path=role_manifest,
        specification_path=specification,
    )
    print(
        json.dumps(
            {
                "generation": str(generation),
                "data_as_of": report["data_as_of"],
                "profile": report["profile"],
                "evidence_status": report["evidence_status"],
                "common_origins": report["result_counts"]["common_origins"],
                "automatic_promotion_eligible": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
