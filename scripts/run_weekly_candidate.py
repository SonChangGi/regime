#!/usr/bin/env python3
"""Update isolated weekly candidate records from a complete matching generation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from regime_lab.candidate_weekly import run_weekly_candidate_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("workspace", "payload", "enhancements", "states", "input-manifest", "artifact-manifest"):
        parser.add_argument("--" + field, type=Path, required=True)
    args = parser.parse_args()
    result = run_weekly_candidate_files(args.workspace, payload_path=args.payload,
        enhancement_path=args.enhancements, states_path=args.states,
        input_manifest_path=args.input_manifest, artifact_manifest_path=args.artifact_manifest)
    destination = args.workspace.resolve() / "candidate-summary.json"
    print(json.dumps({"status": result["status"], "issued_packets": result["issued_packets"],
                      "pending_predictions": result["pending_predictions"], "matured_predictions": result["matured_predictions"],
                      "summary": str(destination)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
