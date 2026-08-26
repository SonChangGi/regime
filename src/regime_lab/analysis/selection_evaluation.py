"""Selection-only supplemental diagnostics for matched multiclass forecasts.

The selection gate remains owned by :mod:`regime_lab.analysis.validation`.
This module independently recomputes the frozen primary scores, then adds
sharpness, state-level recall, transition tracking, and a Model Confidence Set
diagnostic.  It never accepts holdout rows and never returns a replacement
champion.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import numpy as np
import pandas as pd

from regime_lab.integrity import canonical_json_sha256_v1

from .labels import STATE_ORDER
from .model_confidence_set import model_confidence_set
from .validation import PROBABILITY_COLUMNS


SELECTION_EVALUATION_SCHEMA_VERSION = "regime-selection-evaluation/1"
SELECTION_EVALUATION_ROLE = "supplemental_not_selection_gate"
SELECTION_EVIDENCE_STATUSES: tuple[str, ...] = (
    "historical_reconstructed_oos",
    "operational_oos",
    "synthetic_fixture",
)
_MATCH_COLUMNS: tuple[str, ...] = (
    "origin_date",
    "target_date",
    "evaluation_split",
    "current_state",
    "actual",
    "train_size",
    "gap",
)
_REQUIRED_PREDICTION_COLUMNS = frozenset(
    {
        "model",
        "predicted",
        "fallback",
        *_MATCH_COLUMNS,
        *PROBABILITY_COLUMNS,
    }
)
_REQUIRED_DIAGNOSTIC_COLUMNS = frozenset(
    {
        "model",
        "selected",
        "log_loss",
        "brier",
        "n_predictions",
        "fallback_count",
    }
)
_WEEKS_PER_YEAR = 52.1775


def _json_number(value: float | int | np.number | None) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _validated_status(value: str) -> str:
    status = str(value)
    if status not in SELECTION_EVIDENCE_STATUSES:
        raise ValueError(
            "evidence_status must be historical_reconstructed_oos, "
            "operational_oos, or synthetic_fixture"
        )
    return status


def _validate_inputs(
    predictions: pd.DataFrame,
    selection_diagnostics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...], pd.DataFrame]:
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a DataFrame")
    if not isinstance(selection_diagnostics, pd.DataFrame):
        raise TypeError("selection_diagnostics must be a DataFrame")
    missing = sorted(_REQUIRED_PREDICTION_COLUMNS.difference(predictions.columns))
    if missing:
        raise ValueError(f"selection predictions missing columns: {missing}")
    missing = sorted(
        _REQUIRED_DIAGNOSTIC_COLUMNS.difference(selection_diagnostics.columns)
    )
    if missing:
        raise ValueError(f"selection diagnostics missing columns: {missing}")
    if predictions.empty or selection_diagnostics.empty:
        raise ValueError("selection evaluation inputs must not be empty")

    frame = predictions.copy()
    frame["model"] = frame["model"].astype(str)
    split_values = set(frame["evaluation_split"].astype(str))
    if split_values != {"selection"}:
        raise ValueError(
            "selection evaluation accepts selection rows only; holdout or other "
            f"splits are prohibited: {sorted(split_values)}"
        )
    frame["origin_date"] = pd.to_datetime(frame["origin_date"], errors="raise", utc=True)
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="raise", utc=True)
    if not (frame["origin_date"] < frame["target_date"]).all():
        raise ValueError("every selection origin must precede its target")
    if frame.duplicated(["model", "origin_date", "target_date"]).any():
        raise ValueError("selection predictions contain duplicate model-origin rows")
    if set(frame["actual"].astype(str)).difference(STATE_ORDER):
        raise ValueError("selection predictions contain invalid actual states")
    if set(frame["current_state"].astype(str)).difference(STATE_ORDER):
        raise ValueError("selection predictions contain invalid current states")
    gap = pd.to_numeric(frame["gap"], errors="coerce")
    if gap.isna().any() or not gap.eq(1).all():
        raise ValueError("selection evaluation requires the official gap=1 contract")
    train_size = pd.to_numeric(frame["train_size"], errors="coerce")
    if (
        train_size.isna().any()
        or (train_size < 1).any()
        or not np.equal(train_size, np.floor(train_size)).all()
    ):
        raise ValueError("selection train_size must contain positive integers")
    frame["gap"] = gap.astype(int)
    frame["train_size"] = train_size.astype(int)
    if not frame["fallback"].map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).all():
        raise ValueError("selection fallback values must be booleans")
    frame["fallback"] = frame["fallback"].astype(bool)

    raw_probability = frame[list(PROBABILITY_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    )
    values = raw_probability.to_numpy(dtype=float)
    if (
        not np.isfinite(values).all()
        or (values < 0.0).any()
        or (values > 1.0).any()
    ):
        raise ValueError("selection probabilities must be finite and in [0, 1]")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("selection probability rows must sum to one")
    frame.loc[:, list(PROBABILITY_COLUMNS)] = raw_probability
    normalized = np.clip(values, 1e-9, 1.0)
    normalized /= normalized.sum(axis=1, keepdims=True)
    predicted = np.asarray(STATE_ORDER, dtype=object)[normalized.argmax(axis=1)]
    if not np.array_equal(predicted, frame["predicted"].astype(str).to_numpy()):
        raise ValueError("selection predicted state differs from probability argmax")

    diagnostics = selection_diagnostics.copy()
    diagnostics["model"] = diagnostics["model"].astype(str)
    if diagnostics["model"].duplicated().any():
        raise ValueError("selection diagnostics contain duplicate models")
    if not diagnostics["selected"].map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).all():
        raise ValueError("selection diagnostics selected values must be booleans")
    diagnostics["selected"] = diagnostics["selected"].astype(bool)
    candidates = tuple(diagnostics["model"].tolist())
    if len(candidates) < 2 or set(candidates) != set(frame["model"]):
        raise ValueError(
            "selection diagnostics and predictions must contain the same model family"
        )
    selected = diagnostics.loc[diagnostics["selected"], "model"].tolist()
    if len(selected) != 1:
        raise ValueError("selection diagnostics must retain exactly one champion")

    reference = (
        frame.loc[frame["model"].eq(candidates[0]), list(_MATCH_COLUMNS)]
        .sort_values(["origin_date", "target_date"], ignore_index=True)
    )
    if len(reference) < 3:
        raise ValueError("selection evaluation needs at least three matched origins")
    if reference["origin_date"].duplicated().any():
        raise ValueError("selection evaluation requires unique weekly origins")
    for model in candidates[1:]:
        candidate = (
            frame.loc[frame["model"].eq(model), list(_MATCH_COLUMNS)]
            .sort_values(["origin_date", "target_date"], ignore_index=True)
        )
        try:
            pd.testing.assert_frame_equal(reference, candidate, check_dtype=False)
        except AssertionError as exc:
            raise ValueError(
                f"model {model} does not share exact selection origins and targets"
            ) from exc
    return frame, diagnostics, candidates, reference


def _model_rows(
    frame: pd.DataFrame,
    diagnostics: pd.DataFrame,
    candidates: tuple[str, ...],
    *,
    metric_tolerance: float,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    pd.DataFrame,
]:
    if not math.isfinite(metric_tolerance) or metric_tolerance < 0.0:
        raise ValueError("metric_tolerance must be finite and non-negative")
    positions = {state: index for index, state in enumerate(STATE_ORDER)}
    summaries: list[dict[str, Any]] = []
    recalls: list[dict[str, Any]] = []
    transition_summaries: list[dict[str, Any]] = []
    transition_events: list[dict[str, Any]] = []
    loss_columns: dict[str, np.ndarray] = {}

    diagnostic_by_model = diagnostics.set_index("model", drop=False)
    for model in candidates:
        group = (
            frame.loc[frame["model"].eq(model)]
            .sort_values(["origin_date", "target_date"], ignore_index=True)
        )
        probability = group[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
        probability = np.clip(probability, 1e-9, 1.0)
        probability /= probability.sum(axis=1, keepdims=True)
        actual = group["actual"].astype(str).to_numpy()
        current = group["current_state"].astype(str).to_numpy()
        predicted = np.asarray(STATE_ORDER, dtype=object)[probability.argmax(axis=1)]
        actual_probability = np.asarray(
            [probability[row, positions[state]] for row, state in enumerate(actual)],
            dtype=float,
        )
        losses = -np.log(actual_probability)
        loss_columns[model] = losses
        one_hot = np.zeros_like(probability)
        one_hot[
            np.arange(len(actual)),
            np.asarray([positions[state] for state in actual], dtype=int),
        ] = 1.0
        log_loss = float(losses.mean())
        brier = float(np.mean(np.sum((probability - one_hot) ** 2, axis=1)))
        entropy_terms = np.zeros_like(probability)
        positive = probability > 0.0
        entropy_terms[positive] = -probability[positive] * np.log(
            probability[positive]
        )
        normalized_entropy = entropy_terms.sum(axis=1) / math.log(len(STATE_ORDER))
        sharpness = float(np.clip(1.0 - normalized_entropy.mean(), 0.0, 1.0))

        diagnostic = diagnostic_by_model.loc[model]
        recorded_log_loss = float(diagnostic["log_loss"])
        recorded_brier = float(diagnostic["brier"])
        recorded_count = int(diagnostic["n_predictions"])
        recorded_fallback = int(diagnostic["fallback_count"])
        if not math.isclose(
            log_loss, recorded_log_loss, rel_tol=0.0, abs_tol=metric_tolerance
        ):
            raise ValueError(f"{model} independently recomputed log_loss differs")
        if not math.isclose(
            brier, recorded_brier, rel_tol=0.0, abs_tol=metric_tolerance
        ):
            raise ValueError(f"{model} independently recomputed brier differs")
        if recorded_count != len(group):
            raise ValueError(f"{model} n_predictions differs from matched rows")
        if recorded_fallback != int(group["fallback"].sum()):
            raise ValueError(f"{model} fallback_count differs from matched rows")
        summaries.append(
            {
                "model": model,
                "n_predictions": int(len(group)),
                "log_loss": log_loss,
                "brier": brier,
                "normalized_entropy_sharpness": sharpness,
                "mean_max_probability": float(probability.max(axis=1).mean()),
                "fallback_count": recorded_fallback,
                "primary_metric_crosscheck": "matched",
            }
        )

        for state in STATE_ORDER:
            state_actual = actual == state
            support = int(state_actual.sum())
            true_positive = int(np.count_nonzero(state_actual & (predicted == state)))
            recalls.append(
                {
                    "model": model,
                    "state": state,
                    "support": support,
                    "true_positive": true_positive,
                    "recall": None if support == 0 else float(true_positive / support),
                    "status": "insufficient_support" if support == 0 else "computed",
                }
            )

        actual_transition = actual != current
        predicted_transition = predicted != current
        event_positions = np.flatnonzero(actual_transition)
        detected_delays: list[int] = []
        on_time_departure = 0
        on_time_destination = 0
        for event_number, event_position_raw in enumerate(event_positions, start=1):
            event_position = int(event_position_raw)
            next_event_position = (
                int(event_positions[event_number])
                if event_number < len(event_positions)
                else len(group)
            )
            destination = str(actual[event_position])
            if predicted_transition[event_position]:
                on_time_departure += 1
            if str(predicted[event_position]) == destination:
                on_time_destination += 1
            detection_position: int | None = None
            for candidate_position in range(event_position, next_event_position):
                if str(predicted[candidate_position]) == destination:
                    detection_position = candidate_position
                    break
            delay = (
                None
                if detection_position is None
                else int(detection_position - event_position)
            )
            if delay is not None:
                detected_delays.append(delay)
            transition_events.append(
                {
                    "model": model,
                    "event_number": event_number,
                    "target_at": pd.Timestamp(
                        group.iloc[event_position]["target_date"]
                    ).isoformat(),
                    "source_state": str(current[event_position]),
                    "destination_state": destination,
                    "status": "missed_before_next_transition"
                    if delay is None
                    else "detected",
                    "detected_target_at": None
                    if detection_position is None
                    else pd.Timestamp(
                        group.iloc[detection_position]["target_date"]
                    ).isoformat(),
                    "detection_delay_forecast_weeks": delay,
                }
            )
        false_alarm_count = int(
            np.count_nonzero(predicted_transition & ~actual_transition)
        )
        exposure_years = float(len(group) / _WEEKS_PER_YEAR)
        transition_summaries.append(
            {
                "model": model,
                "transition_event_count": int(len(event_positions)),
                "detected_event_count": int(len(detected_delays)),
                "missed_event_count": int(len(event_positions) - len(detected_delays)),
                "on_time_departure_count": int(on_time_departure),
                "on_time_destination_count": int(on_time_destination),
                "destination_detection_rate": None
                if len(event_positions) == 0
                else float(len(detected_delays) / len(event_positions)),
                "mean_detection_delay_forecast_weeks": None
                if not detected_delays
                else float(np.mean(detected_delays)),
                "median_detection_delay_forecast_weeks": None
                if not detected_delays
                else float(np.median(detected_delays)),
                "maximum_detection_delay_forecast_weeks": None
                if not detected_delays
                else int(max(detected_delays)),
                "false_alarm_count": false_alarm_count,
                "exposure_years": exposure_years,
                "false_alarms_per_year": float(false_alarm_count / exposure_years),
            }
        )

    losses = pd.DataFrame(loss_columns)
    return summaries, recalls, transition_summaries, transition_events, losses


def _mcs_document(
    losses: pd.DataFrame,
    *,
    alpha: float,
    block_length: int,
    resamples: int,
    random_state: int,
) -> dict[str, Any]:
    result = model_confidence_set(
        losses,
        alpha=alpha,
        block_length=block_length,
        resamples=resamples,
        random_state=random_state,
    )
    return {
        "role": SELECTION_EVALUATION_ROLE,
        "changes_champion": False,
        "method": result.method,
        "alpha": result.alpha,
        "observation_count": result.observation_count,
        "nominal_block_length": result.nominal_block_length,
        "effective_block_length": result.effective_block_length,
        "bootstrap_resamples": result.bootstrap_resamples,
        "bootstrap_seed": result.bootstrap_seed,
        "retained_models": list(result.retained_models),
        "eliminated_models": list(result.eliminated_models),
        "termination_reason": result.termination_reason,
        "elimination_path": [
            {
                "step": step.step,
                "active_models": list(step.active_models),
                "test_statistic": _json_number(step.test_statistic),
                "test_statistic_nonfinite": not math.isfinite(step.test_statistic),
                "bootstrap_p_value": step.bootstrap_p_value,
                "rejected": step.rejected,
                "eliminated_model": step.eliminated_model,
                "elimination_score": _json_number(step.elimination_score),
                "remaining_models": list(step.remaining_models),
            }
            for step in result.elimination_path
        ],
    }


def build_selection_evaluation(
    predictions: pd.DataFrame,
    selection_diagnostics: pd.DataFrame,
    *,
    evidence_status: str,
    metric_tolerance: float = 1e-12,
    mcs_alpha: float = 0.10,
    mcs_block_length: int = 13,
    mcs_resamples: int = 1_999,
    mcs_random_state: int = 17,
) -> dict[str, Any]:
    """Build a self-hashed selection-only diagnostic document.

    ``selection_diagnostics`` supplies the already frozen champion and primary
    scores.  The scores are independently recomputed and must match, but this
    function has no code path that can change ``selected`` or the champion.
    """

    status = _validated_status(evidence_status)
    frame, diagnostics, candidates, origins = _validate_inputs(
        predictions, selection_diagnostics
    )
    summaries, recalls, transitions, events, losses = _model_rows(
        frame,
        diagnostics,
        candidates,
        metric_tolerance=metric_tolerance,
    )
    losses.index = pd.DatetimeIndex(origins["origin_date"], name="origin_date")
    selected_champion = str(
        diagnostics.loc[diagnostics["selected"], "model"].iloc[0]
    )
    origin_records = [
        {
            "origin_at": pd.Timestamp(row.origin_date).isoformat(),
            "target_at": pd.Timestamp(row.target_date).isoformat(),
            "current_state": str(row.current_state),
            "actual": str(row.actual),
            "train_size": int(row.train_size),
            "gap": int(row.gap),
        }
        for row in origins.itertuples(index=False)
    ]
    body: dict[str, Any] = {
        "schema_version": SELECTION_EVALUATION_SCHEMA_VERSION,
        "status": "completed",
        "evidence_status": status,
        "role": SELECTION_EVALUATION_ROLE,
        "evaluation_split": "selection",
        "holdout_rows_used": 0,
        "selection_effect": "none",
        "selected_champion_unchanged": selected_champion,
        "candidate_set": list(candidates),
        "common_origin_contract": {
            "status": "matched",
            "columns": list(_MATCH_COLUMNS),
            "origin_count": len(origin_records),
            "first_origin_at": origin_records[0]["origin_at"],
            "last_origin_at": origin_records[-1]["origin_at"],
            "origins_sha256": canonical_json_sha256_v1(origin_records),
        },
        "metric_definitions": {
            "log_loss": (
                "mean negative log actual-state probability after 1e-9 "
                "clipping and row renormalization"
            ),
            "brier": "mean three-state sum squared probability error",
            "normalized_entropy_sharpness": (
                "one minus mean forecast entropy divided by log(3); zero is "
                "uniform and one is degenerate"
            ),
            "state_recall": (
                "argmax true positives divided by actual support for each state"
            ),
            "transition_event": (
                "actual next state differs from current state at the forecast origin"
            ),
            "transition_detection_delay": (
                "matched weekly forecast steps from an actual transition target "
                "until the first argmax prediction of its destination, censored "
                "at the next actual transition"
            ),
            "false_alarm": (
                "argmax predicts departure from current state while the actual "
                "next state remains current"
            ),
            "false_alarms_per_year": (
                f"false alarm count divided by n_predictions/{_WEEKS_PER_YEAR}"
            ),
        },
        "primary_metric_crosscheck": {
            "status": "matched",
            "metric_tolerance": float(metric_tolerance),
            "metrics": ["multiclass_log_loss", "multiclass_brier"],
            "changes_holm_gate": False,
            "changes_champion": False,
        },
        "model_metrics": summaries,
        "state_recall": recalls,
        "transition_diagnostics": transitions,
        "transition_events": events,
        "model_confidence_set": _mcs_document(
            losses,
            alpha=mcs_alpha,
            block_length=mcs_block_length,
            resamples=mcs_resamples,
            random_state=mcs_random_state,
        ),
    }
    document = {**body, "sha256": canonical_json_sha256_v1(body)}
    validate_selection_evaluation(document)
    return document


def validate_selection_evaluation(document: Mapping[str, Any]) -> None:
    """Fail closed on semantic hash, evidence status, split, and row coverage."""

    if not isinstance(document, Mapping):
        raise TypeError("selection evaluation must be an object")
    body = dict(document)
    digest = str(body.pop("sha256", ""))
    if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
        raise ValueError("selection evaluation sha256 must be lowercase hexadecimal")
    if canonical_json_sha256_v1(body) != digest:
        raise ValueError("selection evaluation canonical hash mismatch")
    if body.get("schema_version") != SELECTION_EVALUATION_SCHEMA_VERSION:
        raise ValueError("unsupported selection evaluation schema")
    if body.get("status") != "completed":
        raise ValueError("selection evaluation status must be completed")
    _validated_status(str(body.get("evidence_status")))
    if (
        body.get("role") != SELECTION_EVALUATION_ROLE
        or body.get("evaluation_split") != "selection"
        or body.get("holdout_rows_used") != 0
        or body.get("selection_effect") != "none"
    ):
        raise ValueError("selection evaluation role or split contract is invalid")
    candidates = body.get("candidate_set")
    if (
        not isinstance(candidates, list)
        or len(candidates) < 2
        or len(candidates) != len(set(candidates))
    ):
        raise ValueError("selection evaluation candidate_set is invalid")
    if body.get("selected_champion_unchanged") not in candidates:
        raise ValueError("selection evaluation champion is absent from candidates")
    common = body.get("common_origin_contract")
    if not isinstance(common, Mapping) or common.get("status") != "matched":
        raise ValueError("selection evaluation common origins are not matched")
    if not isinstance(common.get("origin_count"), int) or common["origin_count"] < 3:
        raise ValueError("selection evaluation origin count is invalid")
    crosscheck = body.get("primary_metric_crosscheck")
    if (
        not isinstance(crosscheck, Mapping)
        or crosscheck.get("status") != "matched"
        or crosscheck.get("changes_holm_gate") is not False
        or crosscheck.get("changes_champion") is not False
    ):
        raise ValueError("selection evaluation primary score crosscheck is invalid")

    metrics = body.get("model_metrics")
    if not isinstance(metrics, list) or [row.get("model") for row in metrics] != candidates:
        raise ValueError("selection evaluation model metrics are incomplete")
    for row in metrics:
        if row.get("primary_metric_crosscheck") != "matched":
            raise ValueError("selection evaluation model score is not crosschecked")
        sharpness = row.get("normalized_entropy_sharpness")
        if not isinstance(sharpness, (int, float)) or not 0.0 <= sharpness <= 1.0:
            raise ValueError("selection evaluation sharpness is invalid")

    recalls = body.get("state_recall")
    expected_recall_keys = {
        (model, state) for model in candidates for state in STATE_ORDER
    }
    if not isinstance(recalls, list) or {
        (row.get("model"), row.get("state")) for row in recalls
    } != expected_recall_keys:
        raise ValueError("selection evaluation state recall table is incomplete")
    for row in recalls:
        recall = row.get("recall")
        if recall is not None and (
            not isinstance(recall, (int, float)) or not 0.0 <= recall <= 1.0
        ):
            raise ValueError("selection evaluation state recall is invalid")

    transitions = body.get("transition_diagnostics")
    if not isinstance(transitions, list) or [
        row.get("model") for row in transitions
    ] != candidates:
        raise ValueError("selection evaluation transition diagnostics are incomplete")
    for row in transitions:
        if (
            not isinstance(row.get("false_alarms_per_year"), (int, float))
            or row["false_alarms_per_year"] < 0.0
        ):
            raise ValueError("selection evaluation false alarm rate is invalid")

    mcs = body.get("model_confidence_set")
    if (
        not isinstance(mcs, Mapping)
        or mcs.get("role") != SELECTION_EVALUATION_ROLE
        or mcs.get("changes_champion") is not False
        or mcs.get("observation_count") != common.get("origin_count")
    ):
        raise ValueError("selection evaluation MCS contract is invalid")
    retained = mcs.get("retained_models")
    eliminated = mcs.get("eliminated_models")
    if (
        not isinstance(retained, list)
        or not retained
        or not isinstance(eliminated, list)
        or set(retained).intersection(eliminated)
        or set(retained).union(eliminated) != set(candidates)
    ):
        raise ValueError("selection evaluation MCS model partition is invalid")


__all__ = [
    "SELECTION_EVALUATION_ROLE",
    "SELECTION_EVALUATION_SCHEMA_VERSION",
    "SELECTION_EVIDENCE_STATUSES",
    "build_selection_evaluation",
    "validate_selection_evaluation",
]
