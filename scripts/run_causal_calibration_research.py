#!/usr/bin/env python3
"""Run the registered completed-OOS calibration study without publishing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import pandas as pd

from regime_lab.analysis.causal_calibration import (
    CANDIDATE_MODELS, EXPERT_MODELS, build_causal_calibration,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry_path = args.output / "candidate-registry.json"
    registry = json.loads(registry_path.read_text())
    input_sha = hashlib.sha256(args.predictions.read_bytes()).hexdigest()
    if registry.get("source_sha256") != input_sha:
        raise ValueError("Registered input SHA does not match predictions")
    if [candidate["model"] for candidate in registry["candidates"]] != list(CANDIDATE_MODELS):
        raise ValueError("Candidate registration does not match implemented family")
    if registry["experts"] != list(EXPERT_MODELS):
        raise ValueError("Registered experts do not match implementation")
    registry_sha = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    started = time.monotonic()
    source = pd.read_csv(args.predictions)
    result = build_causal_calibration(source, progress=lambda done, total: print(f"{done}/{total} origins", flush=True))
    result.predictions.to_csv(args.output / "oos-predictions.csv", index=False)
    result.metrics.to_csv(args.output / "metrics.csv", index=False)
    record = {
        "registered_input_sha256": input_sha,
        "candidate_registry_sha256": registry_sha,
        "selected_on_pre_2023_only": result.selection,
        "elapsed_seconds": time.monotonic() - started,
        "n_predictions": len(result.predictions),
        "n_origins": int(result.predictions.origin_date.nunique()),
        "n_fallback_origins_per_model": result.predictions.groupby("model").fallback.sum().to_dict(),
        "last_train_target_strictly_before_origin": bool((result.predictions.last_train_target.dropna() < result.predictions.loc[result.predictions.last_train_target.notna(), "origin_date"]).all()),
        "frozen_source_unchanged": hashlib.sha256(args.predictions.read_bytes()).hexdigest() == input_sha,
    }
    (args.output / "run-manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)
    print(result.metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
