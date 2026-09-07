#!/usr/bin/env python3
"""Reproduce aligned model quality and paired uncertainty without publishing."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from regime_lab.analysis.forecast_research_evaluation import (
    match_forecasts, paired_comparisons, per_week_scores, summarize_scores,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.baseline, args.states, *args.candidates]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    baseline = pd.read_csv(args.baseline)
    candidates = pd.concat([pd.read_csv(path) for path in args.candidates], ignore_index=True)
    candidate_names = sorted(candidates.model.unique())
    matched = match_forecasts(baseline, candidates, pd.read_pickle(args.states))
    scored = per_week_scores(matched)
    summary = summarize_scores(scored)
    selection = summary.loc[summary.period.eq("selection_2016_2022")].sort_values(["log_loss", "model"])
    chosen = selection.loc[selection.model.isin(candidate_names)].iloc[0].model
    existing_best = selection.loc[~selection.model.isin(candidate_names)].iloc[0].model
    comparisons = paired_comparisons(scored, candidate_names, list(dict.fromkeys([
        "causal_dynamic_ensemble", "recency_weighted_xgboost_208w", existing_best,
    ])))
    args.output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output / "model-metrics.csv", index=False)
    comparisons.to_csv(args.output / "paired-comparisons.csv", index=False)
    # This small row ledger allows other checks to reproduce every headline.
    scored.to_csv(args.output / "matched-forecast-rows.csv", index=False)
    report = {
        "schema": "regime-forecast-research-evaluation/1", "matched_origins": int(matched.origin_date.nunique()),
        "candidate_models": candidate_names, "selected_model": chosen,
        "selection_rule": "lowest multiclass log loss on original selection origins only",
        "strongest_existing_selection_model": existing_best,
        "post_2022_role": "previously observed retrospective diagnostic; not an untouched holdout",
        "bootstrap": {"block_weeks": 13, "resamples": 4999, "seed": 20260907, "multiplicity": "Holm across all ten new candidates for each baseline"},
        "input_sha256": hashes,
    }
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path)] for path in paths):
        raise RuntimeError("input changed during evaluation")
    (args.output / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(summary.loc[summary.model.isin([chosen, "causal_dynamic_ensemble", "recency_weighted_xgboost_208w"]),
                      ["model", "period", "weeks", "log_loss", "brier", "departure_hits", "departure_events", "departure_false_alarms", "worsening_hits", "worsening_events"]].to_string(index=False))


if __name__ == "__main__":
    main()
