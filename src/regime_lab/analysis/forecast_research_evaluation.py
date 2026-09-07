"""Matched-origin validation of new forecasting models against frozen evidence.

Selection uses the original selection rows only. Post-2022 observations have
already been inspected and remain retrospective diagnostics, not a fresh test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from regime_lab.schema import STATE_ORDER
from regime_lab.analysis.causal_calibration import validate_frozen_splits
from regime_lab.analysis.validation import (
    _holm_adjusted_pvalues, _moving_block_bootstrap_pvalues,
)

PROBABILITIES = [f"p_{state}" for state in STATE_ORDER]
KEYS = ["origin_date", "target_date", "current_state", "actual", "evaluation_split"]


def validate_forecasts(frame: pd.DataFrame, *, require_training_boundary: bool) -> pd.DataFrame:
    missing = set(KEYS + PROBABILITIES + ["model"]).difference(frame.columns)
    if missing:
        raise ValueError(f"missing forecast columns: {sorted(missing)}")
    result = frame.copy()
    for key in ["origin_date", "target_date"]:
        result[key] = pd.to_datetime(result[key], utc=True, errors="raise", format="ISO8601")
    if result[KEYS + ["model"]].isna().any().any():
        raise ValueError("forecast keys and resolved outcomes cannot be missing")
    if result.duplicated(["model", "origin_date"]).any():
        raise ValueError("duplicate model/origin")
    if not (result.target_date > result.origin_date).all():
        raise ValueError("forecast target must follow its origin")
    # Market-close timestamps move by one hour across daylight-saving changes.
    elapsed = (result.target_date - result.origin_date).dt.total_seconds() / 3600
    if not elapsed.between(167, 169).all():
        raise ValueError("only adjacent one-week predictions are comparable")
    if not result.current_state.isin(STATE_ORDER).all() or not result.actual.isin(STATE_ORDER).all():
        raise ValueError("unknown current or target state")
    if not result.evaluation_split.isin(["selection", "holdout"]).all():
        raise ValueError("unknown evaluation split")
    validate_frozen_splits(result)
    probability = result[PROBABILITIES].to_numpy(float)
    if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
        raise ValueError("non-finite or out-of-range probabilities")
    if not np.allclose(probability.sum(axis=1), 1, rtol=0, atol=1e-8):
        raise ValueError("probabilities must sum to one")
    if require_training_boundary:
        if "last_train_target" not in result or "fallback" not in result:
            raise ValueError("new models must record training boundary and warmup")
        fallback = result.fallback
        if not fallback.isin([True, False]).all():
            raise ValueError("fallback must be boolean")
        trained = pd.to_datetime(result.last_train_target, utc=True, errors="raise", format="ISO8601")
        if (trained.notna() & (trained >= result.origin_date)).any():
            raise ValueError("training target reaches or follows forecast origin")
        if ((~fallback) & trained.isna()).any():
            raise ValueError("trained forecasts need a completed training boundary")
        warmup_reason = result.get("fallback_reason", pd.Series("", index=result.index))
        if (fallback & warmup_reason.fillna("").eq("")).any():
            raise ValueError("warmup must have an explicit reason")
    return result.sort_values(["model", "origin_date"]).reset_index(drop=True)


def match_forecasts(baseline: pd.DataFrame, candidates: pd.DataFrame,
                    states: pd.Series, *, anchor: str = "causal_dynamic_ensemble") -> pd.DataFrame:
    baseline = validate_forecasts(baseline, require_training_boundary=False)
    candidates = validate_forecasts(candidates, require_training_boundary=True)
    if set(baseline.model) & set(candidates.model):
        raise ValueError("candidate names must not overwrite existing models")
    expected = baseline.loc[baseline.model.eq(anchor), KEYS].set_index("origin_date").sort_index()
    if expected.empty:
        raise ValueError("authoritative anchor is absent")
    states = states.copy()
    states.index = pd.to_datetime(states.index, utc=True)
    if states.index.has_duplicates:
        raise ValueError("duplicate authoritative state dates")
    all_rows = pd.concat([baseline, candidates], ignore_index=True)
    for model, group in all_rows.groupby("model"):
        indexed = group.set_index("origin_date").sort_index()
        if not indexed.index.equals(expected.index):
            raise ValueError(f"{model}: missing or additional origins; no silent intersection allowed")
        if not indexed[KEYS[1:]].equals(expected):
            raise ValueError(f"{model}: target, label, or split differs from authoritative comparison")
        current = states.reindex(indexed.index).to_numpy()
        actual = states.reindex(indexed.target_date).to_numpy()
        if not np.array_equal(current, indexed.current_state.to_numpy()) or not np.array_equal(actual, indexed.actual.to_numpy()):
            raise ValueError(f"{model}: outcomes disagree with authoritative frozen states")
    return all_rows.sort_values(["model", "origin_date"]).reset_index(drop=True)


def per_week_scores(frame: pd.DataFrame) -> pd.DataFrame:
    validate_frozen_splits(frame)
    result = frame.copy()
    p = result[PROBABILITIES].to_numpy(float)
    actual = np.array([STATE_ORDER.index(state) for state in result.actual])
    current = np.array([STATE_ORDER.index(state) for state in result.current_state])
    predicted = p.argmax(axis=1)
    result["loss"] = -np.log(np.clip(p[np.arange(len(p)), actual], 1e-9, 1))
    result["brier"] = ((p - np.eye(3)[actual]) ** 2).sum(axis=1)
    result["correct"] = predicted == actual
    for name, truth, alert, probability in [
        ("departure", actual != current, predicted != current, 1 - p[np.arange(len(p)), current]),
        ("worsening", actual > current, predicted > current, (p * (np.arange(3) > current[:, None])).sum(axis=1)),
        ("recovery", actual < current, predicted < current, (p * (np.arange(3) < current[:, None])).sum(axis=1)),
    ]:
        result[f"{name}_event"] = truth
        result[f"{name}_hit"] = truth & alert
        result[f"{name}_false_alarm"] = ~truth & alert
        result[f"{name}_probability"] = probability
        clipped = np.clip(probability, 1e-9, 1 - 1e-9)
        result[f"{name}_loss"] = -(truth * np.log(clipped) + ~truth * np.log1p(-clipped))
        result[f"{name}_brier"] = (probability - truth.astype(float)) ** 2
    result["destination_hit"] = (actual != current) & (predicted == actual)
    return result


def summarize_scores(scored: pd.DataFrame) -> pd.DataFrame:
    validate_frozen_splits(scored)
    rows = []
    masks = {
        "selection_2016_2022": scored.evaluation_split.eq("selection"),
        "retrospective_2023_2026": scored.evaluation_split.eq("holdout"),
        "recent_2025_2026": scored.evaluation_split.eq("holdout") & scored.origin_date.ge(pd.Timestamp("2025-01-01", tz="UTC")),
    }
    for period, mask in masks.items():
        for model, group in scored.loc[mask].groupby("model"):
            row = {"model": model, "period": period, "weeks": len(group),
                   "log_loss": float(group.loss.mean()), "brier": float(group.brier.mean()),
                   "accuracy": float(group.correct.mean()), "destination_hits": int(group.destination_hit.sum()),
                   "warmup_rows": int(group.fallback.eq(True).sum()) if "fallback" in group else 0}
            for event in ["departure", "worsening", "recovery"]:
                truth = group[f"{event}_event"]
                count = int(truth.sum())
                hits = int(group[f"{event}_hit"].sum())
                false = int(group[f"{event}_false_alarm"].sum())
                row.update({f"{event}_events": count, f"{event}_hits": hits,
                            f"{event}_recall": hits / count if count else None,
                            f"{event}_false_alarms": false,
                            f"{event}_false_alarms_per_year": false / len(group) * 52.1775,
                            f"{event}_average_precision": float(average_precision_score(truth, group[f"{event}_probability"])) if count else None,
                            f"{event}_log_loss": float(group[f"{event}_loss"].mean()),
                            f"{event}_brier": float(group[f"{event}_brier"].mean())})
            rows.append(row)
    return pd.DataFrame(rows)


def paired_comparisons(scored: pd.DataFrame, candidates: list[str], baselines: list[str],
                       *, resamples: int = 4999, seed: int = 20260907) -> pd.DataFrame:
    """Paired circular 13-week bootstrap; Holm covers all new candidates."""
    validate_frozen_splits(scored)
    rows = []
    for split, group in scored.groupby("evaluation_split"):
        tables = {model: part.set_index("origin_date").sort_index() for model, part in group.groupby("model")}
        count = len(next(iter(tables.values())))
        block = min(13, max(1, count // 2))
        starts = np.random.default_rng(seed).integers(0, count, (resamples, int(np.ceil(count / block))))
        indices = ((starts[..., None] + np.arange(block)) % count).reshape(resamples, -1)[:, :count]
        for baseline in baselines:
            differentials = {model: tables[baseline].loss.to_numpy() - tables[model].loss.to_numpy() for model in candidates}
            pvalues, _ = _moving_block_bootstrap_pvalues(differentials, block_length=13, resamples=resamples, random_state=seed)
            adjusted = _holm_adjusted_pvalues(pvalues)
            for model in candidates:
                row = {"model": model, "baseline": baseline, "evaluation_split": split,
                       "weeks": count, "bootstrap_resamples": resamples, "block_weeks": block,
                       "log_loss_holm_p_value": adjusted[model]}
                for metric in ["loss", "brier", "departure_loss", "departure_brier", "worsening_loss", "worsening_brier", "departure_hit", "departure_false_alarm"]:
                    delta = tables[model][metric].to_numpy(float) - tables[baseline][metric].to_numpy(float)
                    low, high = np.quantile(delta[indices].mean(axis=1), [.025, .975])
                    row.update({f"{metric}_delta": float(delta.mean()), f"{metric}_ci_low": float(low), f"{metric}_ci_high": float(high)})
                rows.append(row)
    return pd.DataFrame(rows)
