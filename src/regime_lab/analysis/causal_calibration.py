"""Prequential calibration of completed out-of-sample regime forecasts.

The small candidate family is fixed before diagnostic evaluation. Each refit
uses targets strictly earlier than the forecast origin; diagnostic outcomes
may update later fits, but may not select candidates or hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss

from .labels import STATE_ORDER

PROBABILITY_COLUMNS = tuple(f"p_{state}" for state in STATE_ORDER)
BASELINE_MODEL = "recency_weighted_xgboost_208w"
EXPERT_MODELS = (
    "discounted_markov_208w",
    BASELINE_MODEL,
    "recency_weighted_ridge_logistic_208w",
    "xgb_hazard_destination",
)
CANDIDATE_MODELS = (
    "causal_dirichlet_xgb",
    "causal_probability_stack",
    "causal_probability_stack_shrunk",
    "causal_departure_calibration",
    "causal_departure_destination_stack",
)
EPSILON = 1e-5
MINIMUM_TRAINING_ROWS = 104
HALF_LIFE_WEEKS = 104.0


@dataclass(frozen=True)
class CalibrationResult:
    predictions: pd.DataFrame
    selection: dict[str, str]
    metrics: pd.DataFrame


def _simplex(probability: np.ndarray) -> np.ndarray:
    probability = np.maximum(np.asarray(probability, dtype=float), EPSILON)
    return probability / probability.sum(axis=-1, keepdims=True)


def _prepare(predictions: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    required = {
        "origin_date", "target_date", "model", "evaluation_split",
        "current_state", "actual", *PROBABILITY_COLUMNS,
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing OOS columns: {sorted(missing)}")
    data = predictions.loc[predictions.model.isin(EXPERT_MODELS)].copy()
    if set(data.model) != set(EXPERT_MODELS):
        raise ValueError("Every registered expert is required")
    for column in ("origin_date", "target_date"):
        data[column] = pd.to_datetime(data[column], utc=True)
    if data[["origin_date", "target_date"]].isna().any().any():
        raise ValueError("OOS dates cannot be missing")
    if not (data.target_date > data.origin_date).all():
        raise ValueError("Every target must follow its origin")
    if data.duplicated(["model", "origin_date"]).any():
        raise ValueError("Duplicate expert origin")
    reference = data.loc[data.model == BASELINE_MODEL].sort_values("origin_date")
    reference = reference.reset_index(drop=True)
    probability_blocks = []
    for expert in EXPERT_MODELS:
        part = data.loc[data.model == expert].sort_values("origin_date").reset_index(drop=True)
        for column in ("origin_date", "target_date", "evaluation_split", "current_state", "actual"):
            if not part[column].equals(reference[column]):
                raise ValueError(f"Expert origins or labels disagree: {expert}/{column}")
        probability = part[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
        if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
            raise ValueError("OOS probabilities must be finite and within [0, 1]")
        if not np.allclose(probability.sum(axis=1), 1, atol=1e-8, rtol=0):
            raise ValueError("OOS probabilities must sum to one")
        # Failed expert estimates are unavailable, not legitimate uniform votes.
        if "fallback" in part:
            flags = part.fallback
            if not flags.map(lambda x: isinstance(x, (bool, np.bool_))).all():
                raise ValueError("Fallback flags must be boolean")
            if expert == BASELINE_MODEL and flags.any():
                raise ValueError("The calibration fallback baseline must be available")
            probability[flags.to_numpy()] = reference.loc[flags, list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
        probability_blocks.append(probability)
    if not reference.current_state.isin(STATE_ORDER).all() or not reference.actual.isin(STATE_ORDER).all():
        raise ValueError("Unsupported regime label")
    expected_split = np.where(reference.origin_date.dt.year < 2023, "selection", "holdout")
    if not np.array_equal(reference.evaluation_split, expected_split):
        raise ValueError("Evaluation split must follow the fixed 2023 boundary")
    return reference, np.stack(probability_blocks, axis=1)


def _features(probability: np.ndarray, current: np.ndarray) -> dict[str, np.ndarray]:
    logs = np.log(_simplex(probability))
    centered = logs - logs.mean(axis=2, keepdims=True)
    onehot = np.eye(len(STATE_ORDER))[current]
    baseline = centered[:, EXPERT_MODELS.index(BASELINE_MODEL), :]
    stack = np.column_stack([centered.reshape(len(current), -1), onehot])
    dirichlet = np.column_stack([
        baseline, onehot,
        (onehot[:, :, None] * baseline[:, None, :]).reshape(len(current), -1),
    ])
    staying = np.take_along_axis(probability, current[:, None, None], axis=2).squeeze(2)
    departure = np.clip(1 - staying, EPSILON, 1 - EPSILON)
    logits = np.log(departure / (1 - departure))
    hazard = np.column_stack([
        logits, onehot,
        (onehot[:, :, None] * logits[:, None, :]).reshape(len(current), -1),
    ])
    return {"dirichlet": dirichlet, "stack": stack, "hazard": hazard}


def _fit_probability(
    features: np.ndarray, labels: np.ndarray, mask: np.ndarray,
    weights: np.ndarray, current_features: np.ndarray, *, C: float,
    class_count: int,
) -> np.ndarray:
    estimator = LogisticRegression(C=C, max_iter=1000, solver="lbfgs", tol=1e-7)
    estimator.fit(features[mask], labels[mask], sample_weight=weights / weights.mean())
    result = np.zeros(class_count, dtype=float)
    result[estimator.classes_.astype(int)] = estimator.predict_proba(current_features[None, :])[0]
    return _simplex(result)


def _route_departure(
    departure: float, destination: np.ndarray, current: int,
) -> np.ndarray:
    destination = np.asarray(destination, dtype=float).copy()
    destination[current] = 0
    if destination.sum() <= 0:
        destination[:] = 1
        destination[current] = 0
    result = destination / destination.sum() * float(departure)
    result[current] = 1 - float(departure)
    return _simplex(result)


def build_causal_calibration(
    predictions: pd.DataFrame,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> CalibrationResult:
    """Fit five fixed candidates on the complete, purged OOS origin sequence."""
    reference, probabilities = _prepare(predictions)
    state_to_id = {state: i for i, state in enumerate(STATE_ORDER)}
    current = reference.current_state.map(state_to_id).to_numpy(dtype=int)
    actual = reference.actual.map(state_to_id).to_numpy(dtype=int)
    departure_actual = (actual != current).astype(int)
    design = _features(probabilities, current)
    baseline = probabilities[:, EXPERT_MODELS.index(BASELINE_MODEL), :]
    rows = []
    for i, reference_row in reference.iterrows():
        origin = reference_row.origin_date
        mask = (reference.target_date < origin).to_numpy()
        train_size = int(mask.sum())
        warmup = train_size < MINIMUM_TRAINING_ROWS
        inadequate_classes = len(np.unique(actual[mask])) < 2 or len(np.unique(departure_actual[mask])) < 2
        fallback = warmup or inadequate_classes
        last_target = reference.loc[mask, "target_date"].max() if train_size else pd.NaT
        forecasts = {name: baseline[i].copy() for name in CANDIDATE_MODELS}
        destination_fallback = False
        if not fallback:
            age = (origin - reference.loc[mask, "target_date"]).dt.total_seconds().to_numpy() / (7 * 86400)
            weights = np.exp2(-age / HALF_LIFE_WEEKS)
            calibrated = _fit_probability(
                design["dirichlet"], actual, mask, weights, design["dirichlet"][i],
                C=1.0, class_count=len(STATE_ORDER),
            )
            stacked = _fit_probability(
                design["stack"], actual, mask, weights, design["stack"][i],
                C=0.3, class_count=len(STATE_ORDER),
            )
            hazard = _fit_probability(
                design["hazard"], departure_actual, mask, weights, design["hazard"][i],
                C=1.0, class_count=2,
            )[1]
            destination_mask = mask & (departure_actual == 1)
            destination_fallback = int(destination_mask.sum()) < 30 or len(np.unique(actual[destination_mask])) < 2
            destination = baseline[i].copy()
            destination[current[i]] = 0
            destination /= destination.sum()
            if not destination_fallback:
                destination_weights = weights[departure_actual[mask] == 1]
                learned_destination = _fit_probability(
                    design["stack"], actual, destination_mask, destination_weights, design["stack"][i],
                    C=0.3, class_count=len(STATE_ORDER),
                )
                learned_destination[current[i]] = 0
                learned_destination /= learned_destination.sum()
                destination = 0.5 * destination + 0.5 * learned_destination
            forecasts = {
                "causal_dirichlet_xgb": calibrated,
                "causal_probability_stack": stacked,
                "causal_probability_stack_shrunk": _simplex(0.5 * stacked + 0.5 * baseline[i]),
                "causal_departure_calibration": _route_departure(hazard, baseline[i], current[i]),
                "causal_departure_destination_stack": _route_departure(hazard, destination, current[i]),
            }
        for name, probability in forecasts.items():
            rows.append({
                "origin_date": origin, "target_date": reference_row.target_date,
                "model": name, "evaluation_split": reference_row.evaluation_split,
                "current_state": reference_row.current_state, "actual": reference_row.actual,
                "predicted": STATE_ORDER[int(np.argmax(probability))],
                **dict(zip(PROBABILITY_COLUMNS, probability, strict=True)),
                "train_size": train_size, "last_train_target": last_target,
                "fallback": fallback,
                "fallback_reason": "insufficient_completed_oos" if warmup else "insufficient_classes" if inadequate_classes else "",
                "destination_fallback": bool(destination_fallback and name == "causal_departure_destination_stack"),
            })
        if progress is not None and (i % 50 == 0 or i == len(reference) - 1):
            progress(i + 1, len(reference))
    result = pd.DataFrame(rows)
    comparison = pd.concat([result, reference.assign(fallback=False)], ignore_index=True)
    metrics = calibration_metrics(comparison, origins=result.loc[~result.fallback, "origin_date"].unique())
    selection_metrics = metrics.loc[(metrics.evaluation_split == "selection") & metrics.model.isin(CANDIDATE_MODELS)] if not metrics.empty else metrics
    choices = {
        "probability": str(selection_metrics.sort_values(["multiclass_log_loss", "model"]).iloc[0].model),
        "departure": str(selection_metrics.sort_values(["departure_log_loss", "model"]).iloc[0].model),
    } if not selection_metrics.empty else {}
    return CalibrationResult(result, choices, metrics)


def calibration_metrics(predictions: pd.DataFrame, *, origins=None) -> pd.DataFrame:
    """Common-origin proper scores, distinct from threshold-based alarm metrics."""
    data = predictions if origins is None else predictions.loc[predictions.origin_date.isin(origins)]
    state_to_id = {state: i for i, state in enumerate(STATE_ORDER)}
    rows = []
    for (model, split), part in data.groupby(["model", "evaluation_split"], sort=True):
        actual = part.actual.map(state_to_id).to_numpy(dtype=int)
        current = part.current_state.map(state_to_id).to_numpy(dtype=int)
        probability = part[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
        departure_probability = 1 - probability[np.arange(len(part)), current]
        departure = (actual != current).astype(int)
        rows.append({
            "model": model, "evaluation_split": split, "n_origins": len(part),
            "n_departures": int(departure.sum()),
            "multiclass_log_loss": float(log_loss(actual, probability, labels=range(len(STATE_ORDER)))),
            "multiclass_brier": float(np.mean(np.sum((probability - np.eye(len(STATE_ORDER))[actual]) ** 2, axis=1))),
            "departure_log_loss": float(log_loss(departure, departure_probability, labels=[0, 1])),
            "departure_brier": float(np.mean((departure_probability - departure) ** 2)),
            "departure_average_precision": float(average_precision_score(departure, departure_probability)) if departure.any() else None,
        })
    return pd.DataFrame(rows)
