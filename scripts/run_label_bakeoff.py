#!/usr/bin/env python3
"""Run the private, reconstructed-OOS regime-label bake-off."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Sequence

from regime_lab.config import project_root
from regime_lab.label_bakeoff import (
    DEFAULT_DATASET,
    DEFAULT_SOURCE,
    run_label_bakeoff,
    write_label_bakeoff_generation,
)
from regime_lab.path_safety import confined_mutable_path


_CLI_TO_POLICY = {
    "exact-split-series": "exact_split_series",
    "adjusted-close-composite": "adjusted_close_composite",
}


def _timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--as-of must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("--as-of must include a timezone")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare frozen v1 with PIT-total-return and broad-equity label "
            "challengers on one reconstructed, matched weekly history."
        )
    )
    parser.add_argument("--database", type=Path, default=Path("data/regime.sqlite3"))
    parser.add_argument("--as-of", type=_timestamp, required=True)
    parser.add_argument(
        "--corporate-action-mode",
        choices=tuple(_CLI_TO_POLICY),
        required=True,
        help=(
            "exact-split-series requires dated split rows; "
            "adjusted-close-composite explicitly infers split coefficients from "
            "provider-current adjusted close and can never be operational OOS"
        ),
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("build/label-bakeoff"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = project_root()
    database = args.database if args.database.is_absolute() else root / args.database
    output = confined_mutable_path(
        args.output_directory,
        project_directory=root,
        label="label bake-off output",
    )
    result = run_label_bakeoff(
        database,
        as_of=args.as_of,
        split_policy=_CLI_TO_POLICY[args.corporate_action_mode],
        source=str(args.source),
        dataset=str(args.dataset),
    )
    generation = write_label_bakeoff_generation(output, result)
    issues = result.pit_replay_report.get("input_issues", [])
    print(
        json.dumps(
            {
                "status": result.status,
                "generation": str(generation),
                "evidence_track": result.pit_replay_report["evidence_track"],
                "corporate_action_mode": args.corporate_action_mode,
                "matched_origin_comparison_completed": (
                    result.label_audit_report[
                        "matched_origin_comparison_completed"
                    ]
                ),
                "input_issues": issues,
                "automatic_promotion_eligible": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0 if result.complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
