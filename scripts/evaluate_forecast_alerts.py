#!/usr/bin/env python3
"""Compare candidates using thresholds frozen on official selection origins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from diagnose_forecast_transitions import (
    PROBABILITIES, STATES, RECENT, binary_metrics, identity, threshold_for_budget,
)

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (4., 8., 12.)
FOCAL_MODELS = ("causal_dynamic_ensemble", "boundary_filtered_history", "causal_probability_stack_shrunk")


def align_candidates(paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.read_csv(paths["official"])
    base["origin_date"] = pd.to_datetime(base.origin_date, utc=True)
    base["target_date"] = pd.to_datetime(base.target_date, utc=True)
    truth = base.loc[base.model.eq(FOCAL_MODELS[0])].set_index("origin_date").sort_index()
    if len(truth) != 556 or int(truth.evaluation_split.eq("holdout").sum()) != 191:
        raise ValueError("expected the frozen 556 official origins and 191 diagnostic origins")
    frames = []
    discarded = []
    seen = set()
    for source, path in paths.items():
        source_rows = pd.read_csv(path)
        source_rows["origin_date"] = pd.to_datetime(source_rows.origin_date, utc=True)
        source_rows["target_date"] = pd.to_datetime(source_rows.target_date, utc=True)
        for model, rows in source_rows.groupby("model"):
            if model in seen:
                raise ValueError(f"duplicate model across inputs: {model}")
            seen.add(model)
            if rows.origin_date.duplicated().any():
                raise ValueError(f"duplicate origins for {model}")
            indexed = rows.set_index("origin_date")
            if len(truth.index.difference(indexed.index)):
                raise ValueError(f"missing official origins for {model}")
            extra = indexed.loc[indexed.index.difference(truth.index)].reset_index()
            if len(extra):
                extra["input_source"] = source
                extra["exclusion_reason"] = "not_in_frozen_official_origin_set"
                discarded.append(extra)
            indexed = indexed.reindex(truth.index)
            for column in ["target_date", "current_state", "actual", "evaluation_split"]:
                if not np.array_equal(indexed[column].to_numpy(), truth[column].to_numpy()):
                    raise ValueError(f"{model} disagrees with official {column}")
            if "last_train_target" in indexed:
                completed = pd.to_datetime(indexed.last_train_target, utc=True)
                if (completed.notna() & (completed >= indexed.index)).any():
                    raise ValueError(f"incomplete outcomes in {model} training")
                if (completed.isna() & indexed.train_size.gt(0)).any():
                    raise ValueError(f"missing completed-outcome cutoff: {model}")
            probability = indexed[PROBABILITIES].to_numpy(float)
            if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any() or not np.allclose(probability.sum(axis=1), 1, atol=1e-12, rtol=0):
                raise ValueError(f"invalid probabilities: {model}")
            argmax = np.asarray(STATES)[probability.argmax(axis=1)]
            if "predicted" in indexed and not np.array_equal(indexed.predicted.to_numpy(), argmax):
                raise ValueError(f"stored argmax mismatch: {model}")
            indexed["predicted"] = argmax
            indexed["input_source"] = source
            indexed["focal_model"] = model in FOCAL_MODELS
            current = indexed.current_state.map(dict(zip(STATES, range(3)))).to_numpy(int)
            actual = indexed.actual.map(dict(zip(STATES, range(3)))).to_numpy(int)
            predicted = probability.argmax(axis=1)
            for event, mask in [
                ("departure", np.arange(3)[None, :] != current[:, None]),
                ("worsening", np.arange(3)[None, :] > current[:, None]),
                ("recovery", np.arange(3)[None, :] < current[:, None]),
            ]:
                indexed[f"p_{event}"] = (probability * mask).sum(axis=1)
                actual_event = {"departure": actual != current, "worsening": actual > current, "recovery": actual < current}[event]
                predicted_event = {"departure": predicted != current, "worsening": predicted > current, "recovery": predicted < current}[event]
                indexed[f"actual_{event}"] = actual_event
                indexed[f"argmax_{event}"] = predicted_event
                destination = np.where(mask.any(axis=1), np.where(mask, probability, -1).argmax(axis=1), current)
                indexed[f"conditional_destination_{event}"] = np.asarray(STATES)[destination]
            indexed["loss"] = -np.log(np.clip(probability[np.arange(len(indexed)), actual], 1e-9, 1))
            indexed["brier"] = ((probability - np.eye(3)[actual]) ** 2).sum(axis=1)
            frames.append(indexed.reset_index())
    excluded = pd.concat(discarded, ignore_index=True) if discarded else pd.DataFrame(columns=["model", "origin_date", "target_date", "input_source", "exclusion_reason"])
    return pd.concat(frames, ignore_index=True), excluded


def evaluate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    policy_rows, threshold_rows, row_alerts = [], [], []
    for model, rows in frame.groupby("model"):
        selection = rows.loc[rows.evaluation_split.eq("selection")]
        periods = {
            "selection_2016_2022": selection,
            "retrospective_2023_2026": rows.loc[rows.evaluation_split.eq("holdout")],
            "recent_audit_2025_2026": rows.loc[rows.origin_date >= RECENT],
        }
        for event in ("departure", "worsening", "recovery"):
            thresholds = [("argmax", None, None)]
            for budget in BUDGETS:
                threshold = threshold_for_budget(selection, event, budget)
                thresholds.append(("frozen_selection_probability", budget, threshold))
                threshold_rows.append({"model": model, "event": event, "budget_per_year": budget, "threshold": threshold, "selected_on": "official_selection_origins_only", "selection_weeks": len(selection), "selection_origin_end": str(selection.origin_date.max())})
            for period, group in periods.items():
                for rule, budget, threshold in thresholds:
                    probability = group[f"p_{event}"].to_numpy()
                    alert = group[f"argmax_{event}"].to_numpy() if rule == "argmax" else probability >= threshold
                    implied = group.predicted.to_numpy() if rule == "argmax" else group[f"conditional_destination_{event}"].to_numpy()
                    actual = group[f"actual_{event}"].to_numpy()
                    metrics = binary_metrics(actual, probability, alert)
                    destination_hits = int((actual & alert & (implied == group.actual.to_numpy())).sum())
                    record = {"model": model, "focal_model": model in FOCAL_MODELS, "event": event, "period": period, "rule": rule, "budget_per_year": budget, "threshold": threshold, "budget_met": metrics["false_alarms_per_year"] <= budget + 1e-12 if budget is not None else None, "destination_correct_hits": destination_hits, "departure_hit_wrong_destination": metrics["hits"] - destination_hits, "multiclass_log_loss": float(group.loss.mean()), "multiclass_brier": float(group.brier.mean())} | metrics
                    policy_rows.append(record)
                    if period == "retrospective_2023_2026":
                        detail = group[["origin_date", "target_date", "model", "current_state", "actual"]].copy()
                        detail["event"] = event
                        detail["rule"] = rule
                        detail["budget_per_year"] = np.nan if budget is None else budget
                        detail["threshold"] = np.nan if threshold is None else threshold
                        detail["probability"] = probability
                        detail["event_observed"] = actual
                        detail["alert"] = alert
                        detail["hit"] = actual & alert
                        detail["false_alarm"] = ~actual & alert
                        detail["implied_destination"] = implied
                        row_alerts.append(detail)
    return pd.DataFrame(policy_rows), pd.DataFrame(threshold_rows), pd.concat(row_alerts, ignore_index=True)


def run(official: Path, boundary: Path, calibration: Path, output: Path) -> dict:
    paths = {"official": official, "boundary": boundary, "calibration": calibration}
    if any(output.resolve() == path.parent.resolve() for path in paths.values()):
        raise ValueError("output must be separate from source directories")
    inputs = {name: identity(path) for name, path in paths.items()}
    frame, excluded = align_candidates(paths)
    policy, thresholds, alerts = evaluate(frame)
    if not policy.loc[policy.period.eq("selection_2016_2022") & policy.rule.ne("argmax"), "budget_met"].all():
        raise ValueError("selected thresholds exceed their selection budget")
    # Guard against accidental diagnostic label use: changing all diagnostic
    # outcomes must leave every selection threshold exactly unchanged.
    changed = frame.copy()
    for event in ("departure", "worsening", "recovery"):
        mask = changed.evaluation_split.eq("holdout")
        changed.loc[mask, f"actual_{event}"] = ~changed.loc[mask, f"actual_{event}"]
    for (model, event, budget), row in thresholds.set_index(["model", "event", "budget_per_year"]).iterrows():
        selection = changed.loc[changed.model.eq(model) & changed.evaluation_split.eq("selection")]
        if threshold_for_budget(selection, event, budget) != row.threshold:
            raise ValueError("diagnostic targets influenced threshold")
    output.mkdir(parents=True, exist_ok=True)
    tables = {"aligned-forecasts": frame, "excluded-origins": excluded, "policy-metrics": policy, "frozen-thresholds": thresholds, "diagnostic-alert-rows": alerts}
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False, float_format="%.17g")
    result = {"ok": True, "model_count": int(frame.model.nunique()), "selection_origins": 365, "diagnostic_origins": 191, "recent_audit_origins": 87, "excluded_rows": len(excluded), "budgets_per_year": BUDGETS, "focal_models": FOCAL_MODELS, "focal_model_role": "families selected by pre-2023 log loss; outcomes after 2022 are already inspected diagnostics", "threshold_rule": "selection-only negative-score order statistic; tied boundary excluded; no diagnostic optimization", "diagnostic_label_mutation_invariance": True, "inputs": inputs, "artifacts": {name: identity(output / f"{name}.csv") for name in tables}}
    if inputs != {name: identity(path) for name, path in paths.items()}:
        raise ValueError("input changed during alert validation")
    result["inputs_preserved"] = True
    (output / "alert-validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, default=ROOT / "build/forecast-performance/diagnosis/forecast-rows.csv")
    parser.add_argument("--boundary", type=Path, default=ROOT / "build/forecast-performance/boundary/aligned/oos-predictions.csv")
    parser.add_argument("--calibration", type=Path, default=ROOT / "build/forecast-performance/calibration/oos-predictions.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "build/forecast-performance/alerts")
    args = parser.parse_args()
    result = run(args.official, args.boundary, args.calibration, args.output)
    print(json.dumps({key: result[key] for key in ("ok", "model_count", "diagnostic_origins", "excluded_rows", "diagnostic_label_mutation_invariance", "inputs_preserved")}))


if __name__ == "__main__":
    main()
