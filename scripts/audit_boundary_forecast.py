#!/usr/bin/env python3
"""Independent frozen-input audit; does not edit models or publish results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss

from regime_lab.analysis.boundary_forecast import (
    MODELS, PROBABILITY_COLUMNS, build_boundary_inputs, next_scores,
    next_states, prequential_temperature, run_boundary_walk_forward,
)
from regime_lab.analysis.labels import CausalRegimeLabeler, RegimeLabelConfig
from regime_lab.schema import STATE_ORDER


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for field in ("origin_date", "target_date"):
        frame[field] = pd.to_datetime(frame[field], utc=True)
    return frame


def scores(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, split), group in frame.groupby(["model", "evaluation_split"]):
        actual = np.array([STATE_ORDER.index(value) for value in group.actual])
        current = np.array([STATE_ORDER.index(value) for value in group.current_state])
        probability = group[list(PROBABILITY_COLUMNS)].to_numpy()
        departure = actual != current
        p_change = 1 - probability[np.arange(len(group)), current]
        years = (group.target_date.max() - group.origin_date.min()).total_seconds() / (365.25 * 86400)
        for policy in ("argmax", "p_departure_at_least_half"):
            alert = probability.argmax(axis=1) != current if policy == "argmax" else p_change >= .5
            rows.append({
                "model": model, "evaluation_split": split, "policy": policy,
                "origins": len(group), "events": int(departure.sum()),
                "log_loss": float(log_loss(actual, probability, labels=range(3))),
                "brier": float(np.mean(np.sum((probability - np.eye(3)[actual]) ** 2, axis=1))),
                "departure_log_loss": float(log_loss(departure, p_change, labels=[False, True])),
                "departure_brier": float(np.mean((p_change - departure) ** 2)),
                "departure_ap": float(average_precision_score(departure, p_change)),
                "hits": int((alert & departure).sum()),
                "false_alerts": int((alert & ~departure).sum()),
                "false_alerts_per_year": float((alert & ~departure).sum() / years),
                "deterioration_events": int((actual > current).sum()),
                "deterioration_direction_hits": int(((actual > current) & (probability.argmax(axis=1) > current)).sum()),
                "recovery_events": int((actual < current).sum()),
                "recovery_direction_hits": int(((actual < current) & (probability.argmax(axis=1) < current)).sum()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-oos", type=Path, required=True)
    parser.add_argument("--candidate-oos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    tracked_paths = [args.input / name for name in ("canonical.pkl", "states.pkl", "input-manifest.json")]
    tracked_paths += [args.source_oos, args.candidate_oos, Path("src/regime_lab/analysis/boundary_forecast.py")]
    hashes_before = {str(path): digest(path) for path in tracked_paths}
    canonical = pd.read_pickle(args.input / "canonical.pkl")
    states = pd.read_pickle(args.input / "states.pkl")
    source = normalize(pd.read_csv(args.source_oos))
    candidate = normalize(pd.read_csv(args.candidate_oos))
    assert not candidate.duplicated(["model", "origin_date", "target_date"]).any()
    inputs = build_boundary_inputs(canonical, states)
    official = CausalRegimeLabeler(RegimeLabelConfig(minimum_fit_observations=260)).fit(canonical.iloc[:520])
    pd.testing.assert_series_equal(official.transform(canonical), states.rename("regime"))
    assert inputs.labeler.component_stats_ == official.component_stats_
    assert inputs.labeler.composite_stats_ == official.composite_stats_
    assert inputs.labeler.lower_threshold_ == official.lower_threshold_
    assert inputs.labeler.upper_threshold_ == official.upper_threshold_
    all_scores = official.score_frame(canonical).risk_score
    positions = list(range(521, len(canonical) - 1))
    max_actual_error = 0.0
    for position in positions:
        future_return = np.log(canonical.spy_close.iloc[position + 1] / canonical.spy_close.iloc[position])
        predicted_score = next_scores(canonical.spy_close.iloc[:position + 1].to_numpy(), np.array([future_return]), official)[0]
        max_actual_error = max(max_actual_error, abs(predicted_score - all_scores.iloc[position + 1]))
        assert next_states(np.array([predicted_score]), str(states.iloc[position]), official)[0] == states.iloc[position + 1]
    assert max_actual_error < 1e-10
    print(f"Official next-score and next-state replay passed on {len(positions)} origins", flush=True)

    grid_positions = np.unique(np.linspace(521, len(canonical) - 2, 20, dtype=int))
    shocks = np.array([-.3, -.1, -.03, -.01, 0, .01, .03, .1, .3])
    max_grid_error = 0.0
    for position in grid_positions:
        vectorized = next_scores(canonical.spy_close.iloc[:position + 1].to_numpy(), shocks, official)
        for shock, predicted_score in zip(shocks, vectorized, strict=True):
            scenario = canonical.iloc[:position + 2].copy()
            scenario.loc[scenario.index[-1], "spy_close"] = canonical.spy_close.iloc[position] * np.exp(shock)
            independent_score = official.score_frame(scenario).risk_score.iloc[-1]
            max_grid_error = max(max_grid_error, abs(predicted_score - independent_score))
    assert max_grid_error < 1e-10

    lower, upper = official.lower_threshold_, official.upper_threshold_
    margin = (upper - lower) * official.config.hysteresis_fraction
    boundary_scores = np.unique(np.array([lower, upper, lower - margin, lower + margin, upper - margin, upper + margin])[:, None] + np.array([-1e-12, 0, 1e-12]))
    for state in STATE_ORDER:
        for score, predicted in zip(boundary_scores, next_states(boundary_scores, state, official), strict=True):
            independent = CausalRegimeLabeler(official.config)
            independent.component_stats_ = official.component_stats_
            independent.composite_stats_ = official.composite_stats_
            independent.lower_threshold_ = lower
            independent.upper_threshold_ = upper
            independent.score_frame = lambda frame, score=score: pd.DataFrame({"risk_score": [score]}, index=frame.index)
            actual = independent.transform(canonical.iloc[[-1]], initial_state=state).iloc[0]
            assert predicted == actual

    reference_replay = normalize(run_boundary_walk_forward(inputs, models=MODELS[:2]))
    stored_mechanical = candidate.loc[candidate.model.isin(MODELS[:2])].sort_values(["origin_date", "model"]).reset_index(drop=True)
    ordered_replay = reference_replay.sort_values(["origin_date", "model"]).reset_index(drop=True)
    assert stored_mechanical[["origin_date", "target_date", "model"]].equals(ordered_replay[["origin_date", "target_date", "model"]])
    replay_error = float(np.max(np.abs(stored_mechanical[list(PROBABILITY_COLUMNS)].to_numpy() - ordered_replay[list(PROBABILITY_COLUMNS)].to_numpy())))
    assert replay_error < 1e-12

    prefix_results = []
    for cutoff in (730, 1000):
        changed = canonical.copy()
        future_positions = np.arange(len(changed) - cutoff - 1)
        for number, column in enumerate(changed.select_dtypes(include="number").columns):
            # Every post-origin numeric series changes; positive prices remain positive.
            changed.loc[changed.index[cutoff + 1:], column] *= np.exp(.4 * np.cos(future_positions / 3 + number))
        changed_states = official.transform(changed)
        rebuilt = build_boundary_inputs(changed, changed_states)
        pd.testing.assert_frame_equal(inputs.features.iloc[:cutoff + 1], rebuilt.features.iloc[:cutoff + 1], check_exact=True)
        for model in MODELS[:2]:
            pd.testing.assert_frame_equal(inputs.mechanistic[model].iloc[:cutoff + 1], rebuilt.mechanistic[model].iloc[:cutoff + 1], check_exact=True)
        altered = normalize(run_boundary_walk_forward(rebuilt, models=MODELS[:2]))
        prefix = reference_replay.origin_date <= canonical.index[cutoff]
        pd.testing.assert_frame_equal(reference_replay.loc[prefix, list(PROBABILITY_COLUMNS)], altered.loc[prefix, list(PROBABILITY_COLUMNS)], check_exact=True)
        classifier_positions = list(range(cutoff - 2, cutoff + 1))
        original_classifier = run_boundary_walk_forward(inputs, origin_positions=classifier_positions, models=MODELS[2:])
        altered_classifier = run_boundary_walk_forward(rebuilt, origin_positions=classifier_positions, models=MODELS[2:])
        pd.testing.assert_frame_equal(original_classifier[list(PROBABILITY_COLUMNS)], altered_classifier[list(PROBABILITY_COLUMNS)], check_exact=True)
        prefix_results.append({"cutoff_position": cutoff, "origin": str(canonical.index[cutoff]), "mechanistic_forecast_rows_unchanged": int(prefix.sum()), "classifier_forecast_rows_unchanged": len(original_classifier), "feature_prefix_exact": True})
        print(f"Future canonical mutation at {canonical.index[cutoff]} preserved all earlier probabilities", flush=True)

    history = [(i, np.array([.7, .2, .1]) if i % 2 else np.array([.2, .6, .2]), i % 3) for i in range(300)]
    now = np.array([.6, .3, .1])
    before = prequential_temperature(now, history, 200)
    future_changed = history[:200] + [(i, np.array([.001, .001, .998]), 2) for i in range(200, 500)]
    after = prequential_temperature(now, future_changed, 200)
    np.testing.assert_array_equal(before[0], after[0])
    assert before[1:] == after[1:] and before[2] == 156
    assert (pd.to_datetime(candidate.last_train_target, utc=True) < candidate.origin_date).all()
    assert official.train_end_ < candidate.origin_date.min()
    for row in candidate.itertuples():
        assert row.actual == states.loc[row.target_date]
        assert row.current_state == states.loc[row.origin_date]
    assert (candidate[list(PROBABILITY_COLUMNS)] > 0).all().all()
    assert np.allclose(candidate[list(PROBABILITY_COLUMNS)].sum(axis=1), 1, atol=1e-12)

    baseline = source.loc[source.model.isin(["causal_dynamic_ensemble", "recency_weighted_xgboost_208w"])]
    source_keys = source.loc[source.model == "causal_dynamic_ensemble", ["origin_date", "target_date", "evaluation_split", "actual", "current_state"]]
    common = candidate.merge(source_keys, on=["origin_date", "target_date"], suffixes=("", "_source"), validate="many_to_one")
    for field in ("actual", "current_state", "evaluation_split"):
        assert common[field].equals(common[f"{field}_source"])
    common = common.drop(columns=[f"{field}_source" for field in ("actual", "current_state", "evaluation_split")])
    comparison = scores(pd.concat([common, baseline], ignore_index=True))
    comparison.to_csv(args.output / "common-origin-metrics.csv", index=False)
    extra_origins = sorted(set(candidate.origin_date) - set(source.origin_date))
    common_positions = [position for position in positions if canonical.index[position] in set(source.origin_date)]
    strict_common_replay = normalize(run_boundary_walk_forward(inputs, models=MODELS[:2], origin_positions=common_positions))
    sensitivity = scores(strict_common_replay)
    sensitivity.to_csv(args.output / "common-calibration-history-sensitivity.csv", index=False)
    selection_scores = comparison.loc[(comparison.evaluation_split == "selection") & (comparison.policy == "argmax") & comparison.model.isin(MODELS)]
    selected = selection_scores.sort_values(["log_loss", "model"]).iloc[0].model
    hashes_after = {str(path): digest(path) for path in tracked_paths}
    assert hashes_after == hashes_before, "Source or model changed during independent audit"
    report = {
        "ok": True, "elapsed_seconds": time.monotonic() - started,
        "source_hashes": hashes_before, "sources_and_model_unchanged": True,
        "label_fit_start": str(canonical.index[0]), "label_fit_end": str(official.train_end_),
        "label_fit_weeks": 520, "official_states_exact": True,
        "all_actual_next_scores_max_abs_error": max_actual_error,
        "all_actual_next_states_matched": len(positions),
        "hypothetical_grid_next_scores_max_abs_error": max_grid_error,
        "hypothetical_grid_checks": len(grid_positions) * len(shocks),
        "boundary_inequality_checks": len(boundary_scores) * len(STATE_ORDER),
        "saved_mechanistic_probabilities_replay_max_abs_error": replay_error,
        "prefix_mutations": prefix_results,
        "temperature_excludes_equal_and_future_targets": True,
        "temperature_max_completed_history": 156,
        "all_declared_train_targets_strictly_before_origin": True,
        "common_origin_count_per_candidate": int(common.loc[common.model == MODELS[0]].shape[0]),
        "common_holdout_origins_per_candidate": int(common.loc[(common.model == MODELS[0]) & (common.evaluation_split == "holdout")].shape[0]),
        "candidate_only_origins": [str(value) for value in extra_origins],
        "selected_on_common_selection_log_loss": str(selected),
        "scope": "Implementation and frozen OOS audit; no proof of unseen future or economic returns; no publication",
    }
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print(comparison.loc[(comparison.evaluation_split == "holdout") & (comparison.policy == "argmax")].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
