#!/usr/bin/env python3
"""Independently audit forecast events from frozen, read-only evidence.

No training, download, model selection, or publication occurs. The post-2022
period has already been inspected and is always named retrospective diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
STATES = ("risk_on", "transition", "risk_off")
PROBABILITIES = [f"p_{state}" for state in STATES]
YEARS_IN_WEEKS = 52.1775
CUTOFF = pd.Timestamp("2023-01-01", tz="UTC")
RECENT = pd.Timestamp("2025-01-01", tz="UTC")


def identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def binary_metrics(actual: np.ndarray, probability: np.ndarray, alert: np.ndarray) -> dict:
    actual, alert = np.asarray(actual, dtype=bool), np.asarray(alert, dtype=bool)
    n = len(actual)
    true_positives = int((actual & alert).sum())
    false_positives = int((~actual & alert).sum())
    events = int(actual.sum())
    clipped = np.clip(probability, 1e-9, 1 - 1e-9)
    return {
        "n": n,
        "events": events,
        "event_rate": float(actual.mean()),
        "mean_probability": float(np.mean(probability)),
        "average_precision": float(average_precision_score(actual, probability)) if events else None,
        "binary_log_loss": float(-np.mean(actual * np.log(clipped) + ~actual * np.log(1 - clipped))),
        "binary_brier": float(np.mean((actual.astype(float) - probability) ** 2)),
        "hits": true_positives,
        "false_alarms": false_positives,
        "recall": true_positives / events if events else None,
        "precision": true_positives / int(alert.sum()) if alert.any() else None,
        "false_alarms_per_year": false_positives / (n / YEARS_IN_WEEKS),
    }


def enrich(frame: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in ("origin_date", "target_date"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    if frame.duplicated(["model", "origin_date"]).any():
        raise ValueError("duplicate model/origin")
    if not (frame.target_date > frame.origin_date).all():
        raise ValueError("invalid forecast timing")
    probability = frame[PROBABILITIES].to_numpy(float)
    if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
        raise ValueError("invalid probability")
    if not np.allclose(probability.sum(axis=1), 1, atol=1e-12, rtol=0):
        raise ValueError("probabilities must sum to one")
    lookup = {state: index for index, state in enumerate(STATES)}
    current = frame.current_state.map(lookup).to_numpy()
    actual = frame.actual.map(lookup).to_numpy()
    if frame.current_state.map(lookup).isna().any() or frame.actual.map(lookup).isna().any():
        raise ValueError("invalid state")
    predicted = probability.argmax(axis=1)
    if not (frame.predicted.to_numpy() == np.asarray(STATES)[predicted]).all():
        raise ValueError("stored predictions disagree with argmax")
    observed = labels.set_index("date").state
    if not (observed.reindex(frame.origin_date).to_numpy() == frame.current_state.to_numpy()).all():
        raise ValueError("current labels disagree with frozen history")
    if not (observed.reindex(frame.target_date).to_numpy() == frame.actual.to_numpy()).all():
        raise ValueError("targets disagree with frozen history")
    frame["period"] = np.where(frame.origin_date < CUTOFF, "selection_2016_2022", "retrospective_2023_2026")
    frame["year"] = frame.origin_date.dt.year
    frame["actual_departure"] = actual != current
    frame["actual_worsening"] = actual > current
    frame["actual_recovery"] = actual < current
    frame["argmax_departure"] = predicted != current
    frame["argmax_worsening"] = predicted > current
    frame["argmax_recovery"] = predicted < current
    frame["destination_correct"] = predicted == actual
    frame["departure_hit"] = frame.actual_departure & frame.argmax_departure
    frame["destination_hit"] = frame.actual_departure & frame.destination_correct
    frame["false_departure"] = ~frame.actual_departure & frame.argmax_departure
    frame["p_departure"] = 1 - probability[np.arange(len(frame)), current]
    frame["p_worsening"] = np.sum(probability * (np.arange(3)[None, :] > current[:, None]), axis=1)
    frame["p_recovery"] = np.sum(probability * (np.arange(3)[None, :] < current[:, None]), axis=1)
    frame["loss"] = -np.log(np.clip(probability[np.arange(len(frame)), actual], 1e-9, 1))
    onehot = np.eye(3)[actual]
    frame["brier"] = np.sum((probability - onehot) ** 2, axis=1)
    frame["event_type"] = frame.current_state + "->" + frame.actual
    frame["p_departure_bin"] = pd.cut(frame.p_departure, [0, .1, .2, .3, .4, .5, .6, 1], include_lowest=True).astype(str)
    # Boundary distance is contemporaneous. Target episode length below is
    # explicitly retrospective and never supplied to any model or threshold.
    history = labels.sort_values("date").copy()
    episode = history.state.ne(history.state.shift()).cumsum()
    history["causal_duration"] = history.groupby(episode).cumcount() + 1
    history["retrospective_episode_length"] = history.groupby(episode).state.transform("size")
    risk, lo, hi, margin = (history[x] for x in ["risk_score", "lower_threshold", "upper_threshold", "hysteresis_margin"])
    history["distance_to_worsening"] = np.where(history.state.eq("risk_on"), risk - (hi - margin), np.where(history.state.eq("transition"), risk - lo, np.nan))
    history["distance_to_recovery"] = np.where(history.state.eq("risk_off"), (lo + margin) - risk, np.where(history.state.eq("transition"), hi - risk, np.nan))
    history["nearest_exit_distance"] = history[["distance_to_worsening", "distance_to_recovery"]].min(axis=1) / (hi - lo)
    history = history.set_index("date")
    for column in ["causal_duration", "distance_to_worsening", "distance_to_recovery", "nearest_exit_distance"]:
        frame[column] = history[column].reindex(frame.origin_date).to_numpy()
    frame["retrospective_target_episode_length"] = history.retrospective_episode_length.reindex(frame.target_date).to_numpy()
    frame["duration_band"] = pd.cut(frame.causal_duration, [0, 1, 2, 4, 8, 16, 9999], include_lowest=True).astype(str)
    frame["exit_distance_band"] = pd.cut(frame.nearest_exit_distance, [-np.inf, .125, .25, .5, 1., np.inf]).astype(str)
    frame["target_episode_band"] = pd.cut(frame.retrospective_target_episode_length, [0, 1, 2, 4, 8, 16, 9999], include_lowest=True).astype(str)
    return frame.sort_values(["model", "origin_date"]).reset_index(drop=True)


def summarize(group: pd.DataFrame, *, include_recognition: bool = False) -> dict:
    result = {
        "weeks": len(group), "log_loss": float(group.loss.mean()), "brier": float(group.brier.mean()),
        "accuracy": float(group.destination_correct.mean()),
        "destination_hits": int(group.destination_hit.sum()),
        "destination_misses": int((group.actual_departure & ~group.destination_correct).sum()),
        "departure_wrong_destination": int((group.departure_hit & ~group.destination_correct).sum()),
    }
    for event in ("departure", "worsening", "recovery"):
        metrics = binary_metrics(group[f"actual_{event}"].to_numpy(), group[f"p_{event}"].to_numpy(), group[f"argmax_{event}"].to_numpy())
        result.update({f"{event}_{key}": value for key, value in metrics.items()})
    # Risk-off origins cannot worsen under this label order. Report ranking
    # again on eligible origins, so structurally impossible negatives cannot
    # make a worsening model appear more discriminative than it is.
    eligible = group.loc[group.current_state.ne("risk_off")]
    result["worsening_eligible_weeks"] = len(eligible)
    result["worsening_eligible_event_rate"] = float(eligible.actual_worsening.mean()) if len(eligible) else None
    result["worsening_eligible_average_precision"] = float(average_precision_score(eligible.actual_worsening, eligible.p_worsening)) if eligible.actual_worsening.any() else None
    if not include_recognition:
        return result
    # Only contiguous origin groups support delay. A filtered bin or event
    # subtype would omit intervening weeks and produce a misleading delay.
    # Each transition has equal status; recognizing destination at t+1 is not
    # counted as advance warning for the transition at t.
    transitions = np.flatnonzero(group.actual_departure.to_numpy())
    predicted = group.predicted.to_numpy()
    actual = group.actual.to_numpy()
    delays = []
    for number, start in enumerate(transitions):
        stop = transitions[number + 1] if number + 1 < len(transitions) else len(group)
        hits = np.flatnonzero(predicted[start:stop] == actual[start])
        if len(hits):
            delays.append(int(hits[0]))
    result["destination_eventually_recognized"] = len(delays)
    result["mean_recognition_delay_weeks"] = float(np.mean(delays)) if delays else None
    return result


def grouped(frame: pd.DataFrame, keys: list[str], *, include_recognition: bool = False) -> pd.DataFrame:
    records = []
    for values, group in frame.groupby(keys, observed=True, sort=True):
        if not isinstance(values, tuple):
            values = (values,)
        records.append(dict(zip(keys, values)) | summarize(group.sort_values("origin_date"), include_recognition=include_recognition))
    return pd.DataFrame(records)


def threshold_for_budget(group: pd.DataFrame, event: str, budget: float) -> float:
    """Conservative threshold whose tied negative scores stay within budget.

    Threshold selection sees only negative scores, not positive recall. An
    all-alert or tied boundary cannot bypass the false-alarm cap.
    """
    negative = group.loc[~group[f"actual_{event}"], f"p_{event}"].to_numpy()
    allowed = int(np.floor(budget * len(group) / YEARS_IN_WEEKS))
    if allowed >= len(negative):
        return 0.0
    descending = np.sort(negative)[::-1]
    return float(np.nextafter(descending[allowed], np.inf))


def budget_tables(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for model, model_rows in frame.groupby("model"):
        selection = model_rows.loc[model_rows.origin_date < CUTOFF]
        for event in ("departure", "worsening"):
            for budget in (0.5, 1., 2., 4., 8., 12.):
                frozen_threshold = threshold_for_budget(selection, event, budget)
                periods = {
                    "selection_2016_2022": selection,
                    "retrospective_2023_2026": model_rows.loc[model_rows.origin_date >= CUTOFF],
                    "recent_audit_2025_2026": model_rows.loc[model_rows.origin_date >= RECENT],
                }
                for period, group in periods.items():
                    # Oracle curve diagnoses ranking only. It cannot select a
                    # deployed threshold on this already inspected period.
                    oracle_threshold = threshold_for_budget(group, event, budget)
                    for rule, threshold in [("frozen_selection_threshold", frozen_threshold), ("retrospective_ranking_only", oracle_threshold)]:
                        metrics = binary_metrics(group[f"actual_{event}"].to_numpy(), group[f"p_{event}"].to_numpy(), group[f"p_{event}"].to_numpy() >= threshold)
                        records.append({"model": model, "event": event, "period": period, "rule": rule, "budget_per_year": budget, "threshold": threshold} | metrics)
    return pd.DataFrame(records)


def run(source: Path, output: Path) -> dict:
    paths = {
        "predictions": source / "source/artifacts/oos-predictions.csv",
        "hazards": source / "source/artifacts/transition-oos-predictions.csv",
        "label_history": source / "source/artifacts/state-label-history.csv",
        "leaderboard": source / "source/artifacts/model-leaderboard.csv",
        "payload": source / "source/publication/regime-results.json",
        "states": source / "input/states.pkl",
    }
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError("output must not overlap frozen source")
    inputs = {key: identity(path) for key, path in paths.items()}
    payload = json.loads(paths["payload"].read_text())
    champion = payload["model"]["champion"]
    labels = pd.read_csv(paths["label_history"])
    labels["date"] = pd.to_datetime(labels.date, utc=True)
    states = pd.read_pickle(paths["states"])
    if not (states.to_numpy() == labels.state.to_numpy()).all() or not states.index.equals(pd.DatetimeIndex(labels.date)):
        raise ValueError("frozen state sources disagree")
    frame = enrich(pd.read_csv(paths["predictions"]), labels)
    first_origins = None
    for _, rows in frame.groupby("model"):
        origins = rows.origin_date.reset_index(drop=True)
        if first_origins is None:
            first_origins = origins
        elif not origins.equals(first_origins):
            raise ValueError("models must share exactly the same forecast origins")
    period_summary = grouped(frame, ["model", "period"], include_recognition=True)
    recent_summary = grouped(frame.loc[frame.origin_date >= RECENT], ["model"], include_recognition=True)
    recent_summary.insert(1, "period", "recent_audit_2025_2026")
    period_summary = pd.concat([period_summary, recent_summary], ignore_index=True)
    stored = pd.read_csv(paths["leaderboard"]).set_index("model")
    from regime_lab.analysis.validation import evaluate_predictions
    production = evaluate_predictions(frame.loc[frame.origin_date >= CUTOFF]).set_index("model")
    metric_mapping = {"log_loss": "log_loss", "brier": "brier", "departure_events": "transition_event_count", "departure_hits": "on_time_departure_count", "departure_false_alarms": "false_alarm_count", "departure_recall": "transition_recall", "destination_eventually_recognized": "detected_event_count", "mean_recognition_delay_weeks": "mean_detection_delay_forecast_weeks"}
    for row in period_summary.loc[period_summary.period.eq("retrospective_2023_2026")].to_dict("records"):
        for independent, original in metric_mapping.items():
            for reference in (stored, production):
                if not np.isclose(row[independent], reference.loc[row["model"], original], atol=1e-12, rtol=0, equal_nan=True):
                    raise ValueError(f"independent result disagrees: {row['model']}/{independent}")
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "forecast-rows": frame,
        "period-summary": period_summary,
        "transition-types": grouped(frame, ["model", "period", "event_type"]),
        "yearly-summary": grouped(frame, ["model", "year"]),
        "probability-bins": grouped(frame, ["model", "period", "p_departure_bin"]),
        "duration-bins": grouped(frame, ["model", "period", "duration_band"]),
        "boundary-distance-bins": grouped(frame, ["model", "period", "exit_distance_band"]),
        "target-episode-bins": grouped(frame.loc[frame.actual_departure], ["model", "period", "target_episode_band"]),
        "false-alarm-budgets": budget_tables(frame),
    }
    hazard_rows = pd.read_csv(paths["hazards"])
    hazard_rows = hazard_rows.loc[hazard_rows.horizon.eq(1)].copy()
    hazard_rows["origin_date"] = pd.to_datetime(hazard_rows.origin_date, utc=True)
    hazard_rows["period"] = np.where(hazard_rows.origin_date < CUTOFF, "selection_2016_2022", "retrospective_2023_2026")
    hazard_records = []
    for (model, period), group in hazard_rows.groupby(["model", "period"]):
        truth = frame.loc[frame.model.eq(champion)].set_index("origin_date").actual_departure.reindex(group.origin_date).to_numpy()
        if not np.array_equal(truth, group.actual_change.to_numpy()):
            raise ValueError("hazard targets disagree with one-week departure events")
        hazard_records.append({"model": model, "period": period, "threshold_min": float(group.threshold.min()), "threshold_max": float(group.threshold.max())} | binary_metrics(group.actual_change.to_numpy(), group.p_change.to_numpy(), group.predicted_change.to_numpy()))
    tables["hazard-summary"] = pd.DataFrame(hazard_records)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False, float_format="%.17g")
    champion_rows = period_summary.loc[period_summary.model.eq(champion)].replace({np.nan: None}).to_dict("records")
    summary = {
        "ok": True,
        "champion": champion,
        "source_data_as_of": str(states.index.max()),
        "source_payload_sha256": inputs["payload"]["sha256"],
        "model_count": int(frame.model.nunique()),
        "forecast_rows": len(frame),
        "selection_origin_start": str(frame.origin_date.min()),
        "selection_origin_end": str(frame.loc[frame.origin_date < CUTOFF, "origin_date"].max()),
        "diagnostic_origin_start": str(frame.loc[frame.origin_date >= CUTOFF, "origin_date"].min()),
        "diagnostic_origin_end": str(frame.origin_date.max()),
        "post_2022_role": "already_inspected_retrospective_diagnostic_not_new_holdout",
        "new_predictions_or_training": False,
        "independent_production_metric_match": True,
        "champion_periods": champion_rows,
        "definitions": {
            "departure": "actual next state differs from origin state",
            "worsening": "risk_on -> transition/risk_off or transition -> risk_off",
            "destination_hit": "argmax exactly matches actual destination on a departure week",
            "argmax_departure": "most likely state differs from origin state; not the same as a low-threshold probability alarm",
            "average_precision": "non-interpolated PR summary; not trapezoidal PR area",
            "frozen_selection_threshold": "negative-score quantile from 2016-2022; reused unchanged after cutoff",
            "retrospective_ranking_only": "threshold uses evaluated-period negatives solely to compare ranking at the same realized false-alarm budget; not deployable evidence",
            "retrospective_target_episode_length": "future-known label episode diagnostic only; never a model feature",
            "exposure_years": "all comparable weekly origins / 52.1775; includes current risk_off for worsening metrics",
        },
        "inputs": inputs,
        "artifacts": {name: identity(output / f"{name}.csv") for name in tables},
    }
    if inputs != {key: identity(path) for key, path in paths.items()}:
        raise ValueError("read-only source changed during diagnosis")
    summary["source_preserved"] = True
    (output / "diagnosis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Frozen release directory")
    parser.add_argument("--output", type=Path, default=ROOT / "build/forecast-performance/diagnosis")
    args = parser.parse_args()
    result = run(args.source, args.output)
    print(json.dumps({key: result[key] for key in ("ok", "champion", "model_count", "forecast_rows", "independent_production_metric_match", "source_preserved")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
