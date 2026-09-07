#!/usr/bin/env python3
"""Evaluate fixed mechanistic next-week models against immutable source OOS."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import pandas as pd

from regime_lab.analysis.boundary_forecast import MODELS, build_boundary_inputs, forecast_boundary_latest, run_boundary_walk_forward, summarize_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-oos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    args = parser.parse_args()
    if args.stride < 1:
        raise ValueError("stride must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    source_hashes = {name: hashlib.sha256((args.input/name).read_bytes()).hexdigest() for name in ("canonical.pkl", "states.pkl", "input-manifest.json")}
    started = time.monotonic()
    canonical = pd.read_pickle(args.input/"canonical.pkl")
    states = pd.read_pickle(args.input/"states.pkl")
    inputs = build_boundary_inputs(canonical, states)
    baseline = pd.read_csv(args.source_oos)
    source_origins = sorted(pd.to_datetime(baseline.origin_date, utc=True).unique())
    positions = [int(states.index.get_loc(origin)) for origin in source_origins][::args.stride]
    prediction = run_boundary_walk_forward(inputs, origin_positions=positions, models=tuple(args.models), progress=lambda message: print(message, flush=True))
    prediction.to_csv(args.output/"oos-predictions.csv", index=False)
    origins = pd.to_datetime(prediction.origin_date, utc=True)
    baseline = baseline[pd.to_datetime(baseline.origin_date, utc=True).isin(origins)]
    baseline = baseline[baseline.model.isin(["causal_dynamic_ensemble", "xgboost", "recency_weighted_xgboost_208w", "persistence", "discounted_markov_208w"])]
    summary = summarize_predictions(pd.concat([prediction, baseline], ignore_index=True))
    summary.to_csv(args.output/"model-summary.csv", index=False)
    selection = summary[(summary.evaluation_split == "selection") & summary.model.isin(args.models)].sort_values(["log_loss", "model"])
    winner = str(selection.iloc[0].model)
    latest = forecast_boundary_latest(inputs, prediction, model=winner)
    (args.output/"latest-forecast.json").write_text(json.dumps(latest, indent=2))
    report = {"selection_rule": "lowest 2016-2022 purged walk-forward multiclass log loss; held-out 2023+ never selects", "selected_model": winner,
              "stride": args.stride, "complete_weekly_evaluation": args.stride == 1, "rows": len(prediction), "models": args.models,
              "elapsed_seconds": time.monotonic()-started, "source_hashes": source_hashes,
              "fit_period": [str(canonical.index[0]), str(canonical.index[519])],
              "official_label_reconstructed_exactly": True, "pending_confirmation_in_official_label": False,
              "selection_boundary_embargo_origin": "2022-12-30T21:00:00+00:00",
              "source_oos_sha256": hashlib.sha256(args.source_oos.read_bytes()).hexdigest(),
              "transition_capture_definition": "argmax(next-state probabilities) differs from origin state; exact-origin event",
              "last_train_target_strictly_before_origin": bool((pd.to_datetime(prediction.last_train_target, utc=True) < pd.to_datetime(prediction.origin_date, utc=True)).all()),
              "calibration": "scalar temperature on up to 156 previous uncalibrated OOS probabilities, target strictly before origin; at least 52 rows",
              "holdout_used_to_select_or_retune_candidates": False,
              "candidate_definition_timing": "all five candidate definitions were fixed before inspecting their first OOS results; earlier baseline holdout diagnostics were already known",
              "standalone_timestamped_preregistration": False}
    for name, digest in source_hashes.items():
        if hashlib.sha256((args.input/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("input changed during research")
    (args.output/"research-manifest.json").write_text(json.dumps(report, indent=2))
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
