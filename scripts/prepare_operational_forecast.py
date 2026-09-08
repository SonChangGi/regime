"""Prepare one local locked-model forecast. Never issue, publish or schedule."""

from __future__ import annotations
import argparse
from datetime import datetime
import json
from pathlib import Path
import pandas as pd
from regime_lab.operational_forecast import (
    frame_sha256,
    prepare_operational_forecast,
    write_prepared_forecast,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--oos-predictions", type=Path, required=True)
    parser.add_argument("--transition-predictions", type=Path, required=True)
    parser.add_argument("--locked-payload", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--recipe-lock", type=Path)
    parser.add_argument("--issue-deadline-at")
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--research-replay", action="store_true")
    parser.add_argument(
        "--decision-at", help="Past clock, allowed only with --research-replay"
    )
    args = parser.parse_args()
    if not args.research_replay and not (args.input_manifest and args.recipe_lock and args.issue_deadline_at):
        parser.error("local preparation requires --input-manifest, --recipe-lock and --issue-deadline-at")
    features = pd.read_pickle(args.features)
    states = pd.read_pickle(args.states)
    oos = pd.read_csv(args.oos_predictions)
    transition = pd.read_csv(args.transition_predictions)
    input_manifest = json.loads(args.input_manifest.read_text()) if args.input_manifest else None
    result = prepare_operational_forecast(
        features,
        states,
        oos,
        transition,
        locked_payload=json.loads(args.locked_payload.read_text()),
        feature_manifest=json.loads(args.feature_manifest.read_text()),
        expected_input_hashes=input_manifest["frames"] if input_manifest is not None else {
            "features": frame_sha256(features),
            "states": frame_sha256(states),
            "oos_predictions": frame_sha256(oos),
            "transition_predictions": frame_sha256(transition),
        },
        research_replay=args.research_replay,
        input_manifest=input_manifest,
        recipe_lock=json.loads(args.recipe_lock.read_text()) if args.recipe_lock else None,
        issue_deadline_at=datetime.fromisoformat(args.issue_deadline_at) if args.issue_deadline_at else None,
        decision_at=(
            datetime.fromisoformat(args.decision_at) if args.decision_at else None
        ),
    )
    path = write_prepared_forecast(result, args.output_directory)
    print(
        json.dumps(
            {
                "path": str(path.resolve()),
                "status": result["status"],
                "issued": False,
                "blocked_reasons": result["blocked_reasons"],
                "reference_parity": result.get("reference_parity"),
                "elapsed_seconds": result["elapsed_seconds"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
