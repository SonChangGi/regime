#!/usr/bin/env python3
"""Run the private matched-OOS opt-in model comparison."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from regime_lab.config import load_config, project_root
from regime_lab.path_safety import confined_mutable_path
from regime_lab.research_comparison import (
    run_research_comparison,
    write_research_generation,
)


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("--as-of must include a timezone")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/series.json"))
    parser.add_argument("--database", type=Path, default=Path("data/regime.sqlite3"))
    parser.add_argument("--as-of", type=_timestamp, required=True)
    parser.add_argument(
        "--profile",
        choices=("quick", "standard", "full"),
        default="standard",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("build/research-model-comparison"),
    )
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=Path("build/private-checkpoints/research-model-comparison"),
    )
    args = parser.parse_args()
    root = project_root()
    database = args.database if args.database.is_absolute() else root / args.database
    output = confined_mutable_path(
        args.output_directory,
        project_directory=root,
        label="research comparison output",
    )
    checkpoint = confined_mutable_path(
        args.checkpoint_directory,
        project_directory=root,
        label="research comparison checkpoint",
    )
    config = load_config(args.config)
    report, frames, quality = run_research_comparison(
        config,
        database=database,
        as_of=args.as_of,
        profile_name=args.profile,
        checkpoint_directory=checkpoint,
        progress=lambda message: print(message, flush=True),
    )
    generation = write_research_generation(
        output,
        report,
        frames,
        quality,
        expected_source_fingerprint_sha256=report["input"][
            "analysis_source_fingerprint_sha256"
        ],
        source_config=config,
    )
    print(
        json.dumps(
            {
                "generation": str(generation),
                "data_as_of": report["data_as_of"],
                "profile": report["profile"],
                "champion": report["restricted_suite_champion"],
                "models": report["models"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
