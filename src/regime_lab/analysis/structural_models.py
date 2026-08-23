"""Causal structural forecast composition for the preregistered v4 suite.

The functions in this module compose already-produced out-of-sample (OOS)
forecasts.  They do not refit the v3 estimators or change their walk-forward
paths.  This separation makes the structural candidates auditable: every
joint probability is tied to a common origin and every ensemble weight uses
only labels whose target date is strictly earlier than the forecast origin.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .labels import STATE_ORDER

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance for type checkers
    from .validation import BenchmarkResult, TransitionBenchmarkResult


PROBABILITY_COLUMNS: tuple[str, ...] = tuple(
    f"p_{state}" for state in STATE_ORDER
)
DEFAULT_ENSEMBLE_EXPERTS: tuple[str, ...] = (
    "markov",
    "xgboost",
    "xgb_hazard_destination",
)
JOINT_MODEL_NAME = "xgb_hazard_destination"
ENSEMBLE_MODEL_NAME = "causal_dynamic_ensemble"
MULTISCALE_ENSEMBLE_MODEL_NAME = "causal_multiscale_ensemble"
DIRECT_JUMP_FLOOR = 1e-6
ENSEMBLE_HALF_LIFE_WEEKS = 52.0
ENSEMBLE_MINIMUM_HISTORY_ROWS = 26
MULTISCALE_ENSEMBLE_HALF_LIVES_WEEKS: tuple[int, ...] = (26, 52, 104)
MULTISCALE_ENSEMBLE_AGGREGATION = "fixed_equal_probability_average"
MULTISCALE_INNER_POOL_METHOD = "causal_discounted_completed_oos_log_score"
ELIGIBLE_LOSS_RULE = "target_date_strictly_before_origin"
MULTISCALE_SCALE_FORECAST_COLUMNS: tuple[str, ...] = (
    "row_role",
    "origin_date",
    "target_date",
    "evaluation_split",
    "ensemble_model",
    "scale_half_life_weeks",
    "outer_scale_weight",
    "minimum_history_rows",
    "eligible_loss_rule",
    "inner_pool_method",
    "expert_models",
    "expert_forecast_artifact",
    "expert_forecast_key",
    *PROBABILITY_COLUMNS,
    "fallback",
    "fallback_reason",
)
_PROBABILITY_EPSILON = 1e-12


@dataclass(frozen=True)
class DynamicEnsembleResult:
    """Long-form OOS ensemble predictions and their causal expert weights."""

    predictions: pd.DataFrame
    weights: pd.DataFrame


@dataclass(frozen=True)
class MultiscaleEnsembleResult:
    """Fixed average of causal discounted-score pools at frozen time scales."""

    predictions: pd.DataFrame
    weights: pd.DataFrame
    scale_predictions: pd.DataFrame


@dataclass(frozen=True)
class StructuralForecastResult:
    """Latest joint/ensemble probabilities plus causal weighting evidence."""

    probabilities: pd.DataFrame
    stacking_weights: pd.DataFrame
    survival_probabilities: pd.DataFrame
    multiscale_scale_predictions: pd.DataFrame | None = None


def _validate_state(value: object, *, field: str = "current_state") -> str:
    state = str(value)
    if state not in STATE_ORDER:
        raise ValueError(f"{field} must be one of {STATE_ORDER}; got {state!r}")
    return state


def _validate_probability_vector(
    values: Iterable[float],
    *,
    field: str,
) -> np.ndarray:
    probability = np.asarray(list(values), dtype=float).reshape(-1)
    if probability.shape != (len(STATE_ORDER),):
        raise ValueError(
            f"{field} must contain exactly {len(STATE_ORDER)} probabilities"
        )
    if not np.isfinite(probability).all():
        raise ValueError(f"{field} contains non-finite probabilities")
    if ((probability < 0.0) | (probability > 1.0)).any():
        raise ValueError(f"{field} probabilities must be within [0, 1]")
    if not np.isclose(probability.sum(), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError(f"{field} probabilities must sum to one")
    return probability / probability.sum()


def _validate_probability_frame(frame: pd.DataFrame, *, context: str) -> None:
    missing = sorted(set(PROBABILITY_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"{context} missing probability columns: {missing}")
    probability = frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(probability).all():
        raise ValueError(f"{context} contains non-finite probabilities")
    if ((probability < 0.0) | (probability > 1.0)).any():
        raise ValueError(f"{context} probabilities must be within [0, 1]")
    if not np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError(f"{context} probabilities must sum to one")


def _validate_boolean_series(series: pd.Series, *, field: str) -> pd.Series:
    valid = series.map(
        lambda value: isinstance(value, (bool, np.bool_))
        or (isinstance(value, (int, np.integer)) and int(value) in (0, 1))
    )
    if not valid.all():
        raise ValueError(f"{field} flags must be boolean")
    return series.astype(bool)


def _prediction_keys(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frame[["origin_date", "target_date"]])


def _assert_common_origins(
    frame: pd.DataFrame,
    model_names: Sequence[str],
    *,
    context: str,
) -> None:
    if not model_names:
        raise ValueError(f"{context} must contain at least one model")
    reference: pd.MultiIndex | None = None
    reference_name = ""
    for name in model_names:
        rows = frame.loc[frame["model"].astype(str).eq(name)].sort_values(
            ["origin_date", "target_date"]
        )
        if rows.empty:
            raise ValueError(f"{context} is missing model {name!r}")
        if rows.duplicated(["origin_date", "target_date"]).any():
            raise ValueError(f"{context} has duplicate origin rows for {name!r}")
        keys = _prediction_keys(rows)
        if reference is None:
            reference = keys
            reference_name = name
        elif not keys.equals(reference):
            raise ValueError(
                f"{context} model {name!r} does not share common origins with "
                f"{reference_name!r}"
            )


def xgb_hazard_destination_probability(
    xgboost_probability: Iterable[float],
    hazard_probability: float,
    current_state: str,
    *,
    direct_jump_floor: float = DIRECT_JUMP_FLOOR,
) -> np.ndarray:
    """Combine a one-week departure hazard with conditional XGBoost routing.

    The current-state mass is exactly ``1 - hazard_probability``.  The hazard
    mass is distributed across the two non-current states in proportion to the
    multiclass XGBoost forecast after applying the preregistered floor.  Thus a
    direct jump remains representable without altering the canonical labels.
    """

    state = _validate_state(current_state)
    destination = _validate_probability_vector(
        xgboost_probability, field="xgboost_probability"
    )
    hazard = float(hazard_probability)
    if not np.isfinite(hazard) or not 0.0 < hazard < 1.0:
        raise ValueError("hazard_probability must be finite and strictly in (0, 1)")
    floor = float(direct_jump_floor)
    if not np.isfinite(floor) or floor <= 0.0 or floor >= 0.5:
        raise ValueError("direct_jump_floor must be finite and in (0, 0.5)")

    current_position = STATE_ORDER.index(state)
    leave_positions = [
        position for position in range(len(STATE_ORDER)) if position != current_position
    ]
    conditional = np.maximum(destination[leave_positions], floor)
    conditional /= conditional.sum()
    output = np.zeros(len(STATE_ORDER), dtype=float)
    output[current_position] = 1.0 - hazard
    output[leave_positions] = hazard * conditional
    # This is an assertion of the composition identity, not a silent repair.
    if not np.isclose(output.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("joint hazard/destination probabilities do not sum to one")
    return output


def build_xgb_hazard_destination_oos(
    benchmark_predictions: pd.DataFrame,
    transition_predictions: pd.DataFrame,
    *,
    direct_jump_floor: float = DIRECT_JUMP_FLOOR,
) -> pd.DataFrame:
    """Create the joint structural candidate on strict common one-week origins."""

    required_base = {
        "origin_date",
        "target_date",
        "model",
        "evaluation_split",
        "current_state",
        "actual",
        "fallback",
        *PROBABILITY_COLUMNS,
    }
    missing_base = sorted(required_base.difference(benchmark_predictions.columns))
    if missing_base:
        raise ValueError(f"benchmark predictions missing columns: {missing_base}")
    required_transition = {
        "origin_date",
        "horizon",
        "model",
        "current_state",
        "actual_change",
        "p_change",
        "fallback",
    }
    missing_transition = sorted(
        required_transition.difference(transition_predictions.columns)
    )
    if missing_transition:
        raise ValueError(
            f"transition predictions missing columns: {missing_transition}"
        )
    transition_target_column = (
        "target_end"
        if "target_end" in transition_predictions.columns
        else "target_date"
        if "target_date" in transition_predictions.columns
        else None
    )
    if transition_target_column is None:
        raise ValueError("transition predictions require target_end or target_date")

    base = benchmark_predictions.loc[
        benchmark_predictions["model"].astype(str).eq("xgboost")
    ].copy()
    transition = transition_predictions.loc[
        transition_predictions["model"].astype(str).eq("binary_xgboost")
        & transition_predictions["horizon"].eq(1)
    ].copy()
    if base.empty:
        raise ValueError("benchmark predictions do not contain xgboost")
    if transition.empty:
        raise ValueError(
            "transition predictions do not contain horizon-1 binary_xgboost"
        )
    base["origin_date"] = pd.to_datetime(base["origin_date"], errors="raise")
    base["target_date"] = pd.to_datetime(base["target_date"], errors="raise")
    transition["origin_date"] = pd.to_datetime(
        transition["origin_date"], errors="raise"
    )
    transition[transition_target_column] = pd.to_datetime(
        transition[transition_target_column], errors="raise"
    )
    if base.duplicated(["origin_date", "target_date"]).any():
        raise ValueError("xgboost predictions contain duplicate origins")
    if transition.duplicated(["origin_date", transition_target_column]).any():
        raise ValueError("binary_xgboost predictions contain duplicate origins")
    _validate_probability_frame(base, context="xgboost predictions")
    base["fallback"] = _validate_boolean_series(
        base["fallback"], field="xgboost fallback"
    )
    transition["fallback"] = _validate_boolean_series(
        transition["fallback"], field="binary_xgboost fallback"
    )
    if "calibration_fallback" in transition:
        transition["calibration_fallback"] = _validate_boolean_series(
            transition["calibration_fallback"],
            field="binary_xgboost calibration_fallback",
        )
    transition["p_change"] = pd.to_numeric(transition["p_change"], errors="raise")
    if (
        not np.isfinite(transition["p_change"].to_numpy(dtype=float)).all()
        or (transition["p_change"] <= 0.0).any()
        or (transition["p_change"] >= 1.0).any()
    ):
        raise ValueError("binary_xgboost p_change must be strictly in (0, 1)")

    transition_column_names = [
        "origin_date",
        transition_target_column,
        "current_state",
        "actual_change",
        "p_change",
        "fallback",
    ]
    transition_column_names.extend(
        column
        for column in (
            "fallback_reason",
            "calibration_fallback",
            "calibration_fallback_reason",
        )
        if column in transition
    )
    transition_columns = transition[transition_column_names].rename(
        columns={
            transition_target_column: "transition_target_date",
            "current_state": "transition_current_state",
            "fallback": "transition_fallback",
            "fallback_reason": "transition_fallback_reason",
        }
    )
    merged = base.merge(
        transition_columns,
        on="origin_date",
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("xgboost and binary_xgboost have no common one-week origins")
    if not pd.to_datetime(merged["target_date"]).equals(
        pd.to_datetime(merged["transition_target_date"])
    ):
        raise ValueError("joint sources disagree on target_date at a common origin")
    if not merged["current_state"].astype(str).equals(
        merged["transition_current_state"].astype(str)
    ):
        raise ValueError("joint sources disagree on current_state at a common origin")
    expected_change = merged["actual"].astype(str).ne(
        merged["current_state"].astype(str)
    )
    if not expected_change.equals(merged["actual_change"].astype(bool)):
        raise ValueError("binary departure actual is inconsistent with next-state actual")

    rows: list[dict[str, object]] = []
    for row in merged.to_dict(orient="records"):
        current_state = _validate_state(row["current_state"])
        xgboost_probability = [row[column] for column in PROBABILITY_COLUMNS]
        probability = xgb_hazard_destination_probability(
            xgboost_probability,
            float(row["p_change"]),
            current_state,
            direct_jump_floor=direct_jump_floor,
        )
        output = {
            column: row[column]
            for column in base.columns
            if column in row
        }
        source_reasons = []
        if bool(row["fallback"]):
            source_reasons.append(
                f"xgboost:{str(row.get('fallback_reason', '')).strip()}"
            )
        if bool(row["transition_fallback"]):
            source_reasons.append(
                "binary_xgboost:"
                f"{str(row.get('transition_fallback_reason', '')).strip()}"
            )
        if bool(row.get("calibration_fallback", False)):
            source_reasons.append(
                "binary_xgboost_calibration:"
                f"{str(row.get('calibration_fallback_reason', '')).strip()}"
            )
        output.update(
            {
                "model": JOINT_MODEL_NAME,
                "predicted": STATE_ORDER[int(np.argmax(probability))],
                **{
                    column: float(probability[position])
                    for position, column in enumerate(PROBABILITY_COLUMNS)
                },
                "fallback": bool(row["fallback"])
                or bool(row["transition_fallback"])
                or bool(row.get("calibration_fallback", False)),
                "fallback_reason": ";".join(source_reasons),
                "hazard_model": "binary_xgboost",
                "destination_model": "xgboost",
                "p_change": float(row["p_change"]),
                "direct_jump_floor": float(direct_jump_floor),
                "calibration_fallback": bool(
                    row.get("calibration_fallback", False)
                ),
                "calibration_fallback_reason": str(
                    row.get("calibration_fallback_reason", "")
                ),
            }
        )
        rows.append(output)
    result = pd.DataFrame(rows).sort_values(
        ["origin_date", "target_date"], ignore_index=True
    )
    _validate_probability_frame(result, context=JOINT_MODEL_NAME)
    return result


def _prepare_expert_predictions(
    expert_predictions: pd.DataFrame,
    expert_names: Sequence[str],
) -> pd.DataFrame:
    required = {
        "origin_date",
        "target_date",
        "model",
        "evaluation_split",
        "current_state",
        "actual",
        "fallback",
        *PROBABILITY_COLUMNS,
    }
    missing = sorted(required.difference(expert_predictions.columns))
    if missing:
        raise ValueError(f"expert predictions missing columns: {missing}")
    if len(expert_names) < 2 or len(set(expert_names)) != len(expert_names):
        raise ValueError("expert_names must contain at least two unique models")
    frame = expert_predictions.loc[
        expert_predictions["model"].astype(str).isin(expert_names)
    ].copy()
    frame["model"] = frame["model"].astype(str)
    frame["origin_date"] = pd.to_datetime(frame["origin_date"], errors="raise")
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="raise")
    if (frame["target_date"] <= frame["origin_date"]).any():
        raise ValueError("expert target_date must be strictly after origin_date")
    frame["fallback"] = _validate_boolean_series(
        frame["fallback"], field="expert fallback"
    )
    _validate_probability_frame(frame, context="expert predictions")
    _assert_common_origins(frame, expert_names, context="expert predictions")
    if not frame["actual"].astype(str).isin(STATE_ORDER).all():
        raise ValueError("expert predictions contain unsupported actual states")
    if not frame["current_state"].astype(str).isin(STATE_ORDER).all():
        raise ValueError("expert predictions contain unsupported current states")

    reference = frame.loc[frame["model"].eq(expert_names[0])].set_index(
        ["origin_date", "target_date"]
    )
    comparison_columns = ["evaluation_split", "current_state", "actual"]
    for name in expert_names[1:]:
        candidate = frame.loc[frame["model"].eq(name)].set_index(
            ["origin_date", "target_date"]
        )
        if not candidate[comparison_columns].equals(reference[comparison_columns]):
            raise ValueError(
                f"expert {name!r} disagrees on split, current_state, or actual"
            )
    return frame.sort_values(
        ["origin_date", "target_date", "model"], ignore_index=True
    )


def _discounted_expert_weights(
    history: pd.DataFrame,
    *,
    origin_date: pd.Timestamp,
    expert_names: Sequence[str],
    current_fallbacks: Mapping[str, bool],
    half_life_weeks: float,
    minimum_history_rows: int,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    if not np.isfinite(half_life_weeks) or half_life_weeks <= 0.0:
        raise ValueError("half_life_weeks must be positive and finite")
    if isinstance(minimum_history_rows, bool) or minimum_history_rows < 1:
        raise ValueError("minimum_history_rows must be a positive integer")
    eligible_history = history.loc[history["target_date"] < origin_date].copy()
    key_columns = ["origin_date", "target_date"]
    if eligible_history.empty:
        scored_history = eligible_history.copy()
        common_history_rows = 0
    else:
        nonfallback = eligible_history.assign(
            _eligible=~eligible_history["fallback"].astype(bool)
        ).pivot(index=key_columns, columns="model", values="_eligible")
        common_keys = nonfallback.loc[
            nonfallback.reindex(columns=expert_names, fill_value=False).all(axis=1)
        ].index
        common_key_frame = pd.DataFrame(
            list(common_keys), columns=key_columns
        )
        scored_history = eligible_history.merge(
            common_key_frame,
            on=key_columns,
            how="inner",
            validate="many_to_one",
        )
        common_history_rows = int(len(common_keys))
    latest_eligible_target = (
        pd.Timestamp(scored_history["target_date"].max())
        if not scored_history.empty
        else pd.NaT
    )
    eligible_experts = [
        name for name in expert_names if not bool(current_fallbacks.get(name, False))
    ]
    if not eligible_experts:
        return (
            {name: 0.0 for name in expert_names},
            [
                {
                    "expert": name,
                    "weight": 0.0,
                    "eligible": False,
                    "current_fallback": True,
                    "history_rows": common_history_rows,
                    "common_history_rows": common_history_rows,
                    "latest_eligible_target_date": latest_eligible_target,
                    "discounted_log_loss": np.nan,
                    "warmup": common_history_rows < minimum_history_rows,
                }
                for name in expert_names
            ],
        )

    decay = float(np.exp(np.log(0.5) / float(half_life_weeks)))
    warmup = common_history_rows < int(minimum_history_rows)
    discounted_losses: dict[str, float] = {}
    history_counts: dict[str, int] = {}
    for name in expert_names:
        expert_history = scored_history.loc[
            scored_history["model"].eq(name)
        ].copy()
        history_counts[name] = int(len(expert_history))
        if expert_history.empty:
            discounted_losses[name] = 0.0
            continue
        positions = {state: position for position, state in enumerate(STATE_ORDER)}
        actual_positions = np.asarray(
            [positions[str(value)] for value in expert_history["actual"]], dtype=int
        )
        probability = expert_history[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
        realised = np.clip(
            probability[np.arange(len(expert_history)), actual_positions],
            _PROBABILITY_EPSILON,
            1.0,
        )
        age_weeks = (
            (origin_date - expert_history["target_date"]).dt.total_seconds()
            / (7.0 * 24.0 * 60.0 * 60.0)
        ).to_numpy(dtype=float)
        if (age_weeks <= 0.0).any():
            raise RuntimeError("eligible ensemble loss is not strictly pre-origin")
        discounted_losses[name] = float(
            np.sum((decay**age_weeks) * (-np.log(realised)))
        )

    if warmup:
        weights = {
            name: (1.0 / len(eligible_experts) if name in eligible_experts else 0.0)
            for name in expert_names
        }
    else:
        scores = np.asarray(
            [-discounted_losses[name] for name in eligible_experts], dtype=float
        )
        scores -= scores.max()
        raw = np.exp(scores)
        raw /= raw.sum()
        weights = {name: 0.0 for name in expert_names}
        weights.update(
            {name: float(raw[index]) for index, name in enumerate(eligible_experts)}
        )
    evidence = [
        {
            "expert": name,
            "weight": float(weights[name]),
            "eligible": name in eligible_experts,
            "current_fallback": bool(current_fallbacks.get(name, False)),
            "history_rows": history_counts.get(name, 0),
            "common_history_rows": common_history_rows,
            "latest_eligible_target_date": latest_eligible_target,
            "discounted_log_loss": float(discounted_losses.get(name, np.nan)),
            "warmup": warmup,
        }
        for name in expert_names
    ]
    return weights, evidence


def causal_dynamic_ensemble(
    expert_predictions: pd.DataFrame,
    *,
    expert_names: Sequence[str] = DEFAULT_ENSEMBLE_EXPERTS,
    half_life_weeks: float = ENSEMBLE_HALF_LIFE_WEEKS,
    minimum_history_rows: int = ENSEMBLE_MINIMUM_HISTORY_ROWS,
) -> DynamicEnsembleResult:
    """Create causal discounted-log-loss ensemble forecasts.

    At origin ``t`` the weight calculation sees only OOS outcomes with
    ``target_date < t``.  It deliberately does not restrict history to the
    pre-2023 selection split: after the family and half-life are locked, a
    realised diagnostic-period outcome may update a later diagnostic weight.
    An expert whose current forecast is a fallback receives zero weight and the
    remaining weights are renormalised.
    """

    names = tuple(str(name) for name in expert_names)
    frame = _prepare_expert_predictions(expert_predictions, names)
    prediction_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    for (origin_date, target_date), current in frame.groupby(
        ["origin_date", "target_date"], sort=True
    ):
        current = current.set_index("model", drop=False).loc[list(names)]
        current_fallbacks = {
            name: bool(current.loc[name, "fallback"]) for name in names
        }
        weights, evidence = _discounted_expert_weights(
            frame,
            origin_date=pd.Timestamp(origin_date),
            expert_names=names,
            current_fallbacks=current_fallbacks,
            half_life_weeks=float(half_life_weeks),
            minimum_history_rows=int(minimum_history_rows),
        )
        eligible = [name for name in names if weights[name] > 0.0]
        fallback = not eligible
        if fallback:
            probability = np.full(len(STATE_ORDER), 1.0 / len(STATE_ORDER))
            fallback_reason = "all_structural_experts_fallback"
        else:
            probability = sum(
                weights[name]
                * current.loc[name, list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
                for name in eligible
            )
            probability = np.asarray(probability, dtype=float)
            probability /= probability.sum()
            fallback_reason = ""
        reference = current.iloc[0].to_dict()
        output = {
            column: reference[column]
            for column in frame.columns
            if column in reference
        }
        output.update(
            {
                "model": ENSEMBLE_MODEL_NAME,
                "predicted": STATE_ORDER[int(np.argmax(probability))],
                **{
                    column: float(probability[position])
                    for position, column in enumerate(PROBABILITY_COLUMNS)
                },
                "fallback": fallback,
                "fallback_reason": fallback_reason,
                "excluded_experts": ";".join(
                    name for name in names if current_fallbacks[name]
                ),
                "ensemble_half_life_weeks": float(half_life_weeks),
                "ensemble_minimum_history_rows": int(minimum_history_rows),
            }
        )
        prediction_rows.append(output)
        for row in evidence:
            row.update(
                {
                    "origin_date": pd.Timestamp(origin_date),
                    "target_date": pd.Timestamp(target_date),
                    "evaluation_split": str(reference["evaluation_split"]),
                    "ensemble_model": ENSEMBLE_MODEL_NAME,
                    "half_life_weeks": float(half_life_weeks),
                    "minimum_history_rows": int(minimum_history_rows),
                    "eligible_loss_rule": "target_date_strictly_before_origin",
                }
            )
            weight_rows.append(row)
    predictions = pd.DataFrame(prediction_rows).sort_values(
        ["origin_date", "target_date"], ignore_index=True
    )
    weights_frame = pd.DataFrame(weight_rows).sort_values(
        ["origin_date", "target_date", "expert"], ignore_index=True
    )
    _validate_probability_frame(predictions, context=ENSEMBLE_MODEL_NAME)
    weight_sums = weights_frame.groupby(["origin_date", "target_date"])[
        "weight"
    ].sum()
    nonfallback_keys = predictions.loc[~predictions["fallback"], [
        "origin_date",
        "target_date",
    ]]
    nonfallback_index = pd.MultiIndex.from_frame(nonfallback_keys)
    if not np.allclose(
        weight_sums.loc[nonfallback_index].to_numpy(dtype=float),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("eligible causal ensemble weights do not sum to one")
    return DynamicEnsembleResult(predictions=predictions, weights=weights_frame)


def build_causal_dynamic_ensemble_oos(
    expert_predictions: pd.DataFrame,
    *,
    expert_names: Sequence[str] = DEFAULT_ENSEMBLE_EXPERTS,
    half_life_weeks: float = ENSEMBLE_HALF_LIFE_WEEKS,
    minimum_history_rows: int = ENSEMBLE_MINIMUM_HISTORY_ROWS,
) -> DynamicEnsembleResult:
    """Named OOS alias matching the structural artifact vocabulary."""

    return causal_dynamic_ensemble(
        expert_predictions,
        expert_names=expert_names,
        half_life_weeks=half_life_weeks,
        minimum_history_rows=minimum_history_rows,
    )


def _validate_multiscale_contract(
    expert_names: Sequence[str],
    scale_half_lives_weeks: Sequence[int],
    minimum_history_rows: int,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    names = tuple(str(name) for name in expert_names)
    if names != DEFAULT_ENSEMBLE_EXPERTS:
        raise ValueError(
            "causal_multiscale_ensemble experts must match the frozen V5 order"
        )
    scales = tuple(scale_half_lives_weeks)
    if scales != MULTISCALE_ENSEMBLE_HALF_LIVES_WEEKS or any(
        isinstance(value, (bool, np.bool_)) or int(value) != value
        for value in scales
    ):
        raise ValueError(
            "causal_multiscale_ensemble half-lives must be exactly (26, 52, 104)"
        )
    if (
        isinstance(minimum_history_rows, (bool, np.bool_))
        or int(minimum_history_rows) != ENSEMBLE_MINIMUM_HISTORY_ROWS
    ):
        raise ValueError(
            "causal_multiscale_ensemble minimum_history_rows must be exactly 26"
        )
    return names, tuple(int(value) for value in scales)


def _expert_forecast_key(
    *,
    artifact: str,
    origin_date: pd.Timestamp,
    target_date: pd.Timestamp,
    expert_names: Sequence[str],
) -> str:
    return (
        f"{artifact}|origin={pd.Timestamp(origin_date).date().isoformat()}"
        f"|target={pd.Timestamp(target_date).date().isoformat()}"
        f"|models={';'.join(expert_names)}"
    )


def causal_multiscale_ensemble(
    expert_predictions: pd.DataFrame,
    *,
    expert_names: Sequence[str] = DEFAULT_ENSEMBLE_EXPERTS,
    scale_half_lives_weeks: Sequence[int] = (
        MULTISCALE_ENSEMBLE_HALF_LIVES_WEEKS
    ),
    minimum_history_rows: int = ENSEMBLE_MINIMUM_HISTORY_ROWS,
) -> MultiscaleEnsembleResult:
    """Compose the frozen V5 multiscale causal discounted-score candidate."""

    names, scales = _validate_multiscale_contract(
        expert_names,
        scale_half_lives_weeks,
        minimum_history_rows,
    )
    outer_weight = 1.0 / float(len(scales))
    scale_results = [
        (
            scale,
            causal_dynamic_ensemble(
                expert_predictions,
                expert_names=names,
                half_life_weeks=float(scale),
                minimum_history_rows=int(minimum_history_rows),
            ),
        )
        for scale in scales
    ]
    reference = scale_results[0][1].predictions.sort_values(
        ["origin_date", "target_date"], ignore_index=True
    )
    reference_keys = _prediction_keys(reference)
    scale_probability_arrays: list[np.ndarray] = []
    scale_rows: list[dict[str, object]] = []
    multiscale_weight_frames: list[pd.DataFrame] = []
    expert_models = ";".join(names)
    for scale, result in scale_results:
        predictions = result.predictions.sort_values(
            ["origin_date", "target_date"], ignore_index=True
        )
        if not _prediction_keys(predictions).equals(reference_keys):
            raise RuntimeError("multiscale inner pools do not share common origins")
        comparison_columns = [
            "evaluation_split",
            "current_state",
            "actual",
            "fallback",
            "fallback_reason",
        ]
        if not predictions[comparison_columns].equals(reference[comparison_columns]):
            raise RuntimeError("multiscale inner pools disagree on forecast metadata")
        probability_array = predictions[list(PROBABILITY_COLUMNS)].to_numpy(
            dtype=float
        )
        scale_probability_arrays.append(probability_array)
        for row_index, row in predictions.iterrows():
            origin = pd.Timestamp(row["origin_date"])
            target = pd.Timestamp(row["target_date"])
            scale_rows.append(
                {
                    "row_role": "oos",
                    "origin_date": origin,
                    "target_date": target,
                    "evaluation_split": str(row["evaluation_split"]),
                    "ensemble_model": MULTISCALE_ENSEMBLE_MODEL_NAME,
                    "scale_half_life_weeks": int(scale),
                    "outer_scale_weight": outer_weight,
                    "minimum_history_rows": int(minimum_history_rows),
                    "eligible_loss_rule": ELIGIBLE_LOSS_RULE,
                    "inner_pool_method": MULTISCALE_INNER_POOL_METHOD,
                    "expert_models": expert_models,
                    "expert_forecast_artifact": "oos-predictions.csv",
                    "expert_forecast_key": _expert_forecast_key(
                        artifact="oos-predictions.csv",
                        origin_date=origin,
                        target_date=target,
                        expert_names=names,
                    ),
                    **{
                        column: float(probability_array[row_index, position])
                        for position, column in enumerate(PROBABILITY_COLUMNS)
                    },
                    "fallback": bool(row["fallback"]),
                    "fallback_reason": str(row["fallback_reason"]),
                }
            )
        scale_weights = result.weights.copy()
        scale_weights["ensemble_model"] = MULTISCALE_ENSEMBLE_MODEL_NAME
        scale_weights["half_life_weeks"] = int(scale)
        scale_weights["outer_scale_weight"] = outer_weight
        scale_weights["inner_pool_method"] = MULTISCALE_INNER_POOL_METHOD
        multiscale_weight_frames.append(scale_weights)

    # This is the preregistered arithmetic mean; no fitted outer weights or
    # post-pooling repair is applied.
    aggregate_probability = np.mean(np.stack(scale_probability_arrays), axis=0)
    predictions = reference.copy()
    predictions["model"] = MULTISCALE_ENSEMBLE_MODEL_NAME
    predictions[list(PROBABILITY_COLUMNS)] = aggregate_probability
    predictions["predicted"] = [
        STATE_ORDER[int(position)] for position in aggregate_probability.argmax(axis=1)
    ]
    predictions["multiscale_half_lives_weeks"] = ";".join(
        str(value) for value in scales
    )
    predictions["multiscale_outer_weights"] = ";".join(
        format(outer_weight, ".17g") for _ in scales
    )
    predictions["multiscale_aggregation"] = MULTISCALE_ENSEMBLE_AGGREGATION
    predictions["inner_pool_method"] = MULTISCALE_INNER_POOL_METHOD
    predictions["eligible_loss_rule"] = ELIGIBLE_LOSS_RULE
    predictions["ensemble_minimum_history_rows"] = int(minimum_history_rows)
    predictions = predictions.drop(columns=["ensemble_half_life_weeks"], errors="ignore")
    _validate_probability_frame(
        predictions, context=MULTISCALE_ENSEMBLE_MODEL_NAME
    )
    scale_predictions = pd.DataFrame(
        scale_rows, columns=MULTISCALE_SCALE_FORECAST_COLUMNS
    ).sort_values(
        ["origin_date", "target_date", "scale_half_life_weeks"],
        ignore_index=True,
    )
    _validate_probability_frame(
        scale_predictions, context="multiscale ensemble scale forecasts"
    )
    weights = pd.concat(
        multiscale_weight_frames, ignore_index=True, sort=False
    ).sort_values(
        ["origin_date", "target_date", "half_life_weeks", "expert"],
        ignore_index=True,
    )
    return MultiscaleEnsembleResult(
        predictions=predictions,
        weights=weights,
        scale_predictions=scale_predictions,
    )


def _common_key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[["origin_date", "target_date"]].drop_duplicates().reset_index(
        drop=True
    )


def augment_benchmark_with_structural_models(
    benchmark: BenchmarkResult,
    transition_benchmark: TransitionBenchmarkResult | pd.DataFrame,
    *,
    direct_jump_floor: float = DIRECT_JUMP_FLOOR,
    half_life_weeks: float = ENSEMBLE_HALF_LIFE_WEEKS,
    minimum_history_rows: int = ENSEMBLE_MINIMUM_HISTORY_ROWS,
    include_multiscale: bool = False,
    random_state: int = 17,
) -> BenchmarkResult:
    """Return a common-origin BenchmarkResult with v4 candidates recomputed.

    Selection diagnostics and the champion are recomputed with the existing
    0.05 log-loss, Holm, Brier, and zero-fallback gate.  Holdout rows remain
    retrospective diagnostics and cannot change the selected champion.
    """

    # Local import keeps validation's public re-export free of an import cycle.
    from .validation import evaluate_predictions, select_champion
    from .validation import select_champion_with_diagnostics

    if not hasattr(benchmark, "predictions"):
        raise TypeError("benchmark must be a BenchmarkResult-like object")
    transition_predictions = (
        transition_benchmark
        if isinstance(transition_benchmark, pd.DataFrame)
        else transition_benchmark.predictions
    )
    base = benchmark.predictions.copy()
    if base.empty:
        raise ValueError("benchmark predictions must not be empty")
    base["model"] = base["model"].astype(str)
    base["origin_date"] = pd.to_datetime(base["origin_date"], errors="raise")
    base["target_date"] = pd.to_datetime(base["target_date"], errors="raise")
    base_models = tuple(base["model"].drop_duplicates())
    _assert_common_origins(base, base_models, context="benchmark predictions")

    joint = build_xgb_hazard_destination_oos(
        base,
        transition_predictions,
        direct_jump_floor=direct_jump_floor,
    )
    common_keys = _common_key_frame(joint)
    base_common = base.merge(
        common_keys,
        on=["origin_date", "target_date"],
        how="inner",
        validate="many_to_one",
    )
    _assert_common_origins(
        base_common, base_models, context="common-origin benchmark predictions"
    )
    expert_frame = pd.concat(
        [
            base_common.loc[
                base_common["model"].isin(("markov", "xgboost"))
            ],
            joint,
        ],
        ignore_index=True,
        sort=False,
    )
    ensemble = causal_dynamic_ensemble(
        expert_frame,
        half_life_weeks=half_life_weeks,
        minimum_history_rows=minimum_history_rows,
    )
    multiscale = (
        causal_multiscale_ensemble(expert_frame)
        if include_multiscale
        else None
    )
    structural_predictions = [joint, ensemble.predictions]
    stacking_weights = ensemble.weights
    multiscale_scale_forecasts = None
    if multiscale is not None:
        structural_predictions.append(multiscale.predictions)
        stacking_weights = pd.concat(
            [stacking_weights, multiscale.weights], ignore_index=True, sort=False
        ).sort_values(
            ["origin_date", "target_date", "ensemble_model", "half_life_weeks", "expert"],
            ignore_index=True,
        )
        multiscale_scale_forecasts = multiscale.scale_predictions
    predictions = pd.concat(
        [base_common, *structural_predictions],
        ignore_index=True,
        sort=False,
    ).sort_values(["origin_date", "target_date", "model"], ignore_index=True)
    all_models = tuple(predictions["model"].drop_duplicates())
    _assert_common_origins(
        predictions, all_models, context="augmented benchmark predictions"
    )
    _validate_probability_frame(predictions, context="augmented benchmark")

    selection_leaderboard: pd.DataFrame | None = None
    holdout_leaderboard: pd.DataFrame | None = None
    selection_diagnostics: pd.DataFrame | None = None
    if benchmark.selection_end is None:
        leaderboard = evaluate_predictions(predictions)
        champion = select_champion(leaderboard)
        leaderboard.insert(1, "selected", leaderboard["model"].eq(champion))
    else:
        selection_predictions = predictions.loc[
            predictions["evaluation_split"].eq("selection")
        ]
        holdout_predictions = predictions.loc[
            predictions["evaluation_split"].eq("holdout")
        ]
        if selection_predictions.empty or holdout_predictions.empty:
            raise ValueError(
                "structural augmentation requires common selection and holdout origins"
            )
        selection_leaderboard = evaluate_predictions(selection_predictions)
        champion, selection_diagnostics = select_champion_with_diagnostics(
            selection_leaderboard,
            selection_predictions,
            random_state=random_state,
        )
        selection_leaderboard.insert(
            1, "selected", selection_leaderboard["model"].eq(champion)
        )
        holdout_leaderboard = evaluate_predictions(holdout_predictions)
        holdout_leaderboard.insert(
            1, "selected", holdout_leaderboard["model"].eq(champion)
        )
        selection_metrics = selection_leaderboard.drop(columns=["selected"]).rename(
            columns={
                column: f"selection_{column}"
                for column in selection_leaderboard.columns
                if column not in {"model", "selected"}
            }
        )
        leaderboard = holdout_leaderboard.merge(
            selection_metrics, on="model", how="left", validate="one_to_one"
        )
        leaderboard.insert(2, "evaluation_split", "holdout")

    split_audit = benchmark.split_audit.copy()
    if not split_audit.empty and {"origin_date", "target_date"}.issubset(
        split_audit.columns
    ):
        split_audit["origin_date"] = pd.to_datetime(
            split_audit["origin_date"], errors="raise"
        )
        split_audit["target_date"] = pd.to_datetime(
            split_audit["target_date"], errors="raise"
        )
        split_audit = split_audit.merge(
            common_keys,
            on=["origin_date", "target_date"],
            how="inner",
            validate="one_to_one",
        ).sort_values(["origin_date", "target_date"], ignore_index=True)
    return replace(
        benchmark,
        leaderboard=leaderboard,
        champion=champion,
        predictions=predictions,
        split_audit=split_audit,
        selection_leaderboard=selection_leaderboard,
        holdout_leaderboard=holdout_leaderboard,
        selection_diagnostics=selection_diagnostics,
        stacking_weights=stacking_weights,
        multiscale_scale_forecasts=multiscale_scale_forecasts,
    )


def project_joint_survival_hazard(
    one_week_hazard: float | Callable[[int], float] | Sequence[float],
    *,
    current_duration_weeks: int,
    horizons: Sequence[int] = (1, 4, 13),
) -> pd.DataFrame:
    """Project cumulative departure risk from one causal weekly hazard model.

    A callable receives the incremented state duration for each future week;
    callers can close over the frozen origin covariates.  A scalar freezes the
    entire one-week hazard, while a sequence supplies an explicit causal hazard
    path.  Cumulative risk is ``1 - product(1 - weekly_hazard)``.
    """

    if isinstance(current_duration_weeks, bool) or current_duration_weeks < 1:
        raise ValueError("current_duration_weeks must be a positive integer")
    if not horizons or any(
        isinstance(value, (bool, np.bool_)) or int(value) != value or int(value) < 1
        for value in horizons
    ):
        raise ValueError("horizons must contain positive integers")
    resolved_horizons = tuple(dict.fromkeys(int(value) for value in horizons))
    maximum_horizon = max(resolved_horizons)
    if callable(one_week_hazard):
        hazards = np.asarray(
            [
                float(one_week_hazard(current_duration_weeks + step))
                for step in range(1, maximum_horizon + 1)
            ],
            dtype=float,
        )
    elif np.isscalar(one_week_hazard):
        hazards = np.full(maximum_horizon, float(one_week_hazard), dtype=float)
    else:
        hazards = np.asarray(list(one_week_hazard), dtype=float).reshape(-1)
        if len(hazards) < maximum_horizon:
            raise ValueError("one_week_hazard path is shorter than the maximum horizon")
        hazards = hazards[:maximum_horizon]
    if (
        not np.isfinite(hazards).all()
        or (hazards <= 0.0).any()
        or (hazards >= 1.0).any()
    ):
        raise ValueError("weekly hazards must be finite and strictly in (0, 1)")
    survival_path = np.cumprod(1.0 - hazards)
    rows = [
        {
            "horizon": horizon,
            "p_change": float(1.0 - survival_path[horizon - 1]),
            "survival_probability": float(survival_path[horizon - 1]),
            "current_duration_weeks": int(current_duration_weeks),
            "projected_duration_weeks": int(current_duration_weeks + horizon),
            "future_covariates": "origin_values_frozen",
            "duration_policy": "increment_one_week_per_step",
        }
        for horizon in resolved_horizons
    ]
    result = pd.DataFrame(rows)
    ordered = result.sort_values("horizon")
    if (ordered["p_change"].diff().dropna() < -1e-12).any():
        raise RuntimeError("joint survival probabilities are not monotone")
    return result


def forecast_structural_probabilities(
    *,
    origin_date: str | pd.Timestamp,
    current_state: str,
    markov_probability: Iterable[float],
    xgboost_probability: Iterable[float],
    binary_xgboost_p_change: float,
    historical_oos_predictions: pd.DataFrame,
    expert_fallbacks: Mapping[str, bool] | None = None,
    current_duration_weeks: int = 1,
    direct_jump_floor: float = DIRECT_JUMP_FLOOR,
    half_life_weeks: float = ENSEMBLE_HALF_LIFE_WEEKS,
    minimum_history_rows: int = ENSEMBLE_MINIMUM_HISTORY_ROWS,
    include_multiscale: bool = False,
) -> StructuralForecastResult:
    """Compose latest joint and ensemble forecasts without an unknown target."""

    origin = pd.Timestamp(origin_date)
    state = _validate_state(current_state)
    markov = _validate_probability_vector(markov_probability, field="markov_probability")
    xgboost = _validate_probability_vector(
        xgboost_probability, field="xgboost_probability"
    )
    joint = xgb_hazard_destination_probability(
        xgboost,
        binary_xgboost_p_change,
        state,
        direct_jump_floor=direct_jump_floor,
    )
    names = DEFAULT_ENSEMBLE_EXPERTS
    current_probabilities = {
        "markov": markov,
        "xgboost": xgboost,
        JOINT_MODEL_NAME: joint,
    }
    fallbacks = {name: False for name in names}
    if expert_fallbacks is not None:
        unknown = sorted(set(expert_fallbacks).difference(names))
        if unknown:
            raise ValueError(f"expert_fallbacks contains unknown experts: {unknown}")
        fallbacks.update({name: bool(value) for name, value in expert_fallbacks.items()})
    # The joint expert cannot be eligible when its multiclass destination
    # source is a fallback, even if a caller did not redundantly flag the joint.
    fallbacks[JOINT_MODEL_NAME] = (
        fallbacks[JOINT_MODEL_NAME] or fallbacks["xgboost"]
    )

    history = _prepare_expert_predictions(historical_oos_predictions, names)
    weights, evidence = _discounted_expert_weights(
        history,
        origin_date=origin,
        expert_names=names,
        current_fallbacks=fallbacks,
        half_life_weeks=float(half_life_weeks),
        minimum_history_rows=int(minimum_history_rows),
    )
    eligible = [name for name in names if weights[name] > 0.0]
    ensemble_fallback = not eligible
    if ensemble_fallback:
        ensemble = np.full(len(STATE_ORDER), 1.0 / len(STATE_ORDER))
    else:
        ensemble = sum(weights[name] * current_probabilities[name] for name in eligible)
        ensemble = np.asarray(ensemble, dtype=float)
        ensemble /= ensemble.sum()
    multiscale_probability: np.ndarray | None = None
    multiscale_fallback = False
    multiscale_weight_frames: list[pd.DataFrame] = []
    multiscale_scale_rows: list[dict[str, object]] = []
    if include_multiscale:
        names, scales = _validate_multiscale_contract(
            names,
            MULTISCALE_ENSEMBLE_HALF_LIVES_WEEKS,
            minimum_history_rows,
        )
        outer_weight = 1.0 / float(len(scales))
        target = origin + pd.offsets.Week(1)
        scale_probabilities: list[np.ndarray] = []
        scale_fallbacks: list[bool] = []
        for scale in scales:
            scale_weights, scale_evidence = _discounted_expert_weights(
                history,
                origin_date=origin,
                expert_names=names,
                current_fallbacks=fallbacks,
                half_life_weeks=float(scale),
                minimum_history_rows=int(minimum_history_rows),
            )
            scale_eligible = [
                name for name in names if scale_weights[name] > 0.0
            ]
            scale_fallback = not scale_eligible
            if scale_fallback:
                scale_probability = np.full(
                    len(STATE_ORDER), 1.0 / len(STATE_ORDER)
                )
                scale_fallback_reason = "all_structural_experts_fallback"
            else:
                scale_probability = sum(
                    scale_weights[name] * current_probabilities[name]
                    for name in scale_eligible
                )
                scale_probability = np.asarray(scale_probability, dtype=float)
                scale_probability /= scale_probability.sum()
                scale_fallback_reason = ""
            scale_probabilities.append(scale_probability)
            scale_fallbacks.append(scale_fallback)
            multiscale_scale_rows.append(
                {
                    "row_role": "latest_forecast",
                    "origin_date": origin,
                    "target_date": target,
                    "evaluation_split": "prospective",
                    "ensemble_model": MULTISCALE_ENSEMBLE_MODEL_NAME,
                    "scale_half_life_weeks": int(scale),
                    "outer_scale_weight": outer_weight,
                    "minimum_history_rows": int(minimum_history_rows),
                    "eligible_loss_rule": ELIGIBLE_LOSS_RULE,
                    "inner_pool_method": MULTISCALE_INNER_POOL_METHOD,
                    "expert_models": ";".join(names),
                    "expert_forecast_artifact": "structural-forecasts.csv",
                    "expert_forecast_key": _expert_forecast_key(
                        artifact="structural-forecasts.csv",
                        origin_date=origin,
                        target_date=target,
                        expert_names=names,
                    ),
                    **{
                        column: float(scale_probability[position])
                        for position, column in enumerate(PROBABILITY_COLUMNS)
                    },
                    "fallback": scale_fallback,
                    "fallback_reason": scale_fallback_reason,
                }
            )
            scale_weight_frame = pd.DataFrame(scale_evidence)
            scale_weight_frame.insert(0, "origin_date", origin)
            scale_weight_frame["target_date"] = target
            scale_weight_frame["evaluation_split"] = "prospective"
            scale_weight_frame["ensemble_model"] = (
                MULTISCALE_ENSEMBLE_MODEL_NAME
            )
            scale_weight_frame["half_life_weeks"] = int(scale)
            scale_weight_frame["outer_scale_weight"] = outer_weight
            scale_weight_frame["minimum_history_rows"] = int(
                minimum_history_rows
            )
            scale_weight_frame["eligible_loss_rule"] = ELIGIBLE_LOSS_RULE
            scale_weight_frame["inner_pool_method"] = MULTISCALE_INNER_POOL_METHOD
            multiscale_weight_frames.append(scale_weight_frame)
        if len(set(scale_fallbacks)) != 1:
            raise RuntimeError("multiscale latest pools disagree on fallback status")
        multiscale_fallback = scale_fallbacks[0]
        multiscale_probability = np.mean(
            np.stack(scale_probabilities), axis=0
        )
    forecast_entries: list[tuple[str, np.ndarray]] = [
        ("markov", markov),
        ("xgboost", xgboost),
        (JOINT_MODEL_NAME, joint),
        (ENSEMBLE_MODEL_NAME, ensemble),
    ]
    if multiscale_probability is not None:
        forecast_entries.append(
            (MULTISCALE_ENSEMBLE_MODEL_NAME, multiscale_probability)
        )
    rows = []
    for name, probability in forecast_entries:
        source_fallback = (
            ensemble_fallback
            if name == ENSEMBLE_MODEL_NAME
            else multiscale_fallback
            if name == MULTISCALE_ENSEMBLE_MODEL_NAME
            else bool(fallbacks.get(name, False))
        )
        rows.append(
            {
                "origin_date": origin,
                "model": name,
                "current_state": state,
                **{
                    column: float(probability[position])
                    for position, column in enumerate(PROBABILITY_COLUMNS)
                },
                "predicted": STATE_ORDER[int(np.argmax(probability))],
                "fallback": source_fallback,
                "fallback_reason": (
                    "all_structural_experts_fallback"
                    if name
                    in (ENSEMBLE_MODEL_NAME, MULTISCALE_ENSEMBLE_MODEL_NAME)
                    and source_fallback
                    else ""
                ),
                "source_role": (
                    "causal_ensemble"
                    if name == ENSEMBLE_MODEL_NAME
                    else "causal_discounted_log_score_multiscale_pool"
                    if name == MULTISCALE_ENSEMBLE_MODEL_NAME
                    else "hazard_destination_joint"
                    if name == JOINT_MODEL_NAME
                    else "base_expert"
                ),
                "ensemble_weight": (
                    1.0
                    if name
                    in (ENSEMBLE_MODEL_NAME, MULTISCALE_ENSEMBLE_MODEL_NAME)
                    else float(weights[name])
                ),
                "binary_xgboost_p_change": float(binary_xgboost_p_change),
            }
        )
    weights_frame = pd.DataFrame(evidence)
    weights_frame.insert(0, "origin_date", origin)
    weights_frame["ensemble_model"] = ENSEMBLE_MODEL_NAME
    weights_frame["half_life_weeks"] = float(half_life_weeks)
    weights_frame["minimum_history_rows"] = int(minimum_history_rows)
    weights_frame["eligible_loss_rule"] = ELIGIBLE_LOSS_RULE
    if multiscale_weight_frames:
        weights_frame = pd.concat(
            [weights_frame, *multiscale_weight_frames],
            ignore_index=True,
            sort=False,
        ).sort_values(
            ["origin_date", "ensemble_model", "half_life_weeks", "expert"],
            ignore_index=True,
        )
    survival = project_joint_survival_hazard(
        float(binary_xgboost_p_change),
        current_duration_weeks=current_duration_weeks,
    )
    probabilities = pd.DataFrame(rows)
    _validate_probability_frame(probabilities, context="latest structural forecast")
    multiscale_scale_predictions = (
        pd.DataFrame(
            multiscale_scale_rows,
            columns=MULTISCALE_SCALE_FORECAST_COLUMNS,
        ).sort_values("scale_half_life_weeks", ignore_index=True)
        if multiscale_scale_rows
        else None
    )
    if multiscale_scale_predictions is not None:
        _validate_probability_frame(
            multiscale_scale_predictions,
            context="latest multiscale ensemble scale forecasts",
        )
    return StructuralForecastResult(
        probabilities=probabilities,
        stacking_weights=weights_frame,
        survival_probabilities=survival,
        multiscale_scale_predictions=multiscale_scale_predictions,
    )


__all__ = [
    "DEFAULT_ENSEMBLE_EXPERTS",
    "DIRECT_JUMP_FLOOR",
    "ELIGIBLE_LOSS_RULE",
    "ENSEMBLE_HALF_LIFE_WEEKS",
    "ENSEMBLE_MINIMUM_HISTORY_ROWS",
    "ENSEMBLE_MODEL_NAME",
    "JOINT_MODEL_NAME",
    "MULTISCALE_ENSEMBLE_AGGREGATION",
    "MULTISCALE_ENSEMBLE_HALF_LIVES_WEEKS",
    "MULTISCALE_ENSEMBLE_MODEL_NAME",
    "MULTISCALE_INNER_POOL_METHOD",
    "MULTISCALE_SCALE_FORECAST_COLUMNS",
    "DynamicEnsembleResult",
    "MultiscaleEnsembleResult",
    "StructuralForecastResult",
    "augment_benchmark_with_structural_models",
    "build_causal_dynamic_ensemble_oos",
    "build_xgb_hazard_destination_oos",
    "causal_dynamic_ensemble",
    "causal_multiscale_ensemble",
    "forecast_structural_probabilities",
    "project_joint_survival_hazard",
    "xgb_hazard_destination_probability",
]
