from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from regime_lab.analysis.labels import STATE_ORDER
from regime_lab.analysis.models import MODEL_NAMES, MODEL_REGISTRY, BenchmarkProfile
from regime_lab.analysis.structural_models import (
    DEFAULT_ENSEMBLE_EXPERTS,
    ENSEMBLE_MODEL_NAME,
    JOINT_MODEL_NAME,
    augment_benchmark_with_structural_models,
    build_xgb_hazard_destination_oos,
    causal_dynamic_ensemble,
    forecast_structural_probabilities,
    project_joint_survival_hazard,
    xgb_hazard_destination_probability,
)
from regime_lab.analysis.validation import BenchmarkResult, evaluate_predictions


PROBABILITY_COLUMNS = tuple(f"p_{state}" for state in STATE_ORDER)


def _probability_for(actual: str, confidence: float) -> np.ndarray:
    probability = np.full(3, (1.0 - confidence) / 2.0, dtype=float)
    probability[STATE_ORDER.index(actual)] = confidence
    return probability


def _expert_oos(rows: int = 38) -> pd.DataFrame:
    origins = pd.date_range("2021-01-01", periods=rows, freq="W-FRI")
    state_pattern = np.asarray(
        [
            "risk_on",
            "risk_on",
            "transition",
            "risk_off",
            "risk_off",
            "transition",
        ],
        dtype=object,
    )
    current = np.resize(state_pattern, rows)
    actual = np.roll(current, -1)
    confidence = {
        "markov": 0.62,
        "xgboost": 0.71,
        JOINT_MODEL_NAME: 0.76,
    }
    output: list[dict[str, object]] = []
    for position, origin in enumerate(origins):
        origin = pd.Timestamp(origin)
        for model in DEFAULT_ENSEMBLE_EXPERTS:
            probability = _probability_for(str(actual[position]), confidence[model])
            output.append(
                {
                    "origin_date": origin,
                    "target_date": origin + pd.offsets.Week(1),
                    "model": model,
                    "evaluation_split": (
                        "selection" if position < 30 else "holdout"
                    ),
                    "current_state": str(current[position]),
                    "actual": str(actual[position]),
                    **{
                        column: float(probability[index])
                        for index, column in enumerate(PROBABILITY_COLUMNS)
                    },
                    "fallback": False,
                    "fallback_reason": "",
                }
            )
    return pd.DataFrame(output)


def _base_and_transition(
    rows: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    origins = pd.date_range("2021-01-01", periods=rows, freq="W-FRI")
    pattern = np.asarray(
        [
            "risk_on",
            "risk_on",
            "transition",
            "risk_off",
            "risk_off",
            "transition",
        ],
        dtype=object,
    )
    current = np.resize(pattern, rows)
    actual = np.roll(current, -1)
    model_confidence = {
        "majority": 0.36,
        "persistence": 0.42,
        "markov": 0.60,
        "xgboost": 0.70,
    }
    base_rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    for position, origin in enumerate(origins):
        origin = pd.Timestamp(origin)
        target = origin + pd.offsets.Week(1)
        split = "selection" if position < 30 else "holdout"
        for model, confidence in model_confidence.items():
            probability = _probability_for(str(actual[position]), confidence)
            base_rows.append(
                {
                    "origin_date": origin,
                    "target_date": target,
                    "model": model,
                    "evaluation_split": split,
                    "current_state": str(current[position]),
                    "actual": str(actual[position]),
                    "predicted": str(actual[position]),
                    **{
                        column: float(probability[index])
                        for index, column in enumerate(PROBABILITY_COLUMNS)
                    },
                    "train_size": 520 + position,
                    "gap": 1,
                    "fallback": False,
                    "fallback_reason": "",
                }
            )
        changed = bool(current[position] != actual[position])
        transition_rows.append(
            {
                "origin_date": origin,
                "target_start": target,
                "target_end": target,
                "horizon": 1,
                "model": "binary_xgboost",
                "evaluation_split": (
                    "selection" if position < 30 else "retrospective_diagnostic"
                ),
                "current_state": str(current[position]),
                "actual_change": changed,
                "raw_p_change": 0.78 if changed else 0.12,
                "p_change": 0.78 if changed else 0.12,
                "fallback": False,
                "fallback_reason": "",
                "calibration_fallback": False,
                "calibration_fallback_reason": "",
            }
        )
        split_rows.append(
            {
                "origin_date": origin,
                "target_date": target,
                "train_size": 520 + position,
                "evaluation_split": split,
            }
        )
    return pd.DataFrame(base_rows), pd.DataFrame(transition_rows), pd.DataFrame(split_rows)


def test_joint_probability_preserves_hazard_and_floors_direct_jump() -> None:
    probability = xgb_hazard_destination_probability(
        [1.0, 0.0, 0.0],
        0.2,
        "risk_on",
        direct_jump_floor=1e-6,
    )

    assert probability[STATE_ORDER.index("risk_on")] == pytest.approx(0.8)
    assert probability[STATE_ORDER.index("transition")] == pytest.approx(0.1)
    assert probability[STATE_ORDER.index("risk_off")] == pytest.approx(0.1)
    assert (probability > 0.0).all()
    assert probability.sum() == pytest.approx(1.0)

    with pytest.raises(ValueError, match="strictly in"):
        xgb_hazard_destination_probability([0.8, 0.1, 0.1], 1.0, "risk_on")
    with pytest.raises(ValueError, match="sum to one"):
        xgb_hazard_destination_probability([0.8, 0.1, 0.2], 0.2, "risk_on")


def test_joint_oos_uses_only_common_consistent_one_week_origins() -> None:
    base, transition, _ = _base_and_transition(rows=6)
    # An extra transition origin is harmless; the result is an inner common-origin
    # composition rather than an unpaired comparison.
    extra = transition.iloc[[-1]].copy()
    extra["origin_date"] += pd.offsets.Week(5)
    extra["target_start"] += pd.offsets.Week(5)
    extra["target_end"] += pd.offsets.Week(5)
    result = build_xgb_hazard_destination_oos(
        base,
        pd.concat([transition, extra], ignore_index=True),
    )

    assert len(result) == 6
    assert set(result["model"]) == {JOINT_MODEL_NAME}
    np.testing.assert_allclose(result[list(PROBABILITY_COLUMNS)].sum(axis=1), 1.0)
    expected_stay = 1.0 - transition["p_change"].to_numpy(dtype=float)
    actual_stay = np.asarray(
        [
            result.iloc[row][f"p_{result.iloc[row]['current_state']}"]
            for row in range(len(result))
        ],
        dtype=float,
    )
    np.testing.assert_allclose(actual_stay, expected_stay, atol=1e-12)

    inconsistent = transition.copy()
    inconsistent.loc[0, "actual_change"] = not bool(
        inconsistent.loc[0, "actual_change"]
    )
    with pytest.raises(ValueError, match="inconsistent"):
        build_xgb_hazard_destination_oos(base, inconsistent)

    degraded = transition.copy()
    degraded.loc[0, "calibration_fallback"] = True
    degraded.loc[0, "calibration_fallback_reason"] = "insufficient_prequential_rows"
    degraded_joint = build_xgb_hazard_destination_oos(base, degraded)
    assert bool(degraded_joint.loc[0, "fallback"]) is True
    assert "binary_xgboost_calibration" in degraded_joint.loc[0, "fallback_reason"]
    degraded_experts = pd.concat(
        [base.loc[base["model"].isin(("markov", "xgboost"))], degraded_joint],
        ignore_index=True,
        sort=False,
    )
    degraded_ensemble = causal_dynamic_ensemble(degraded_experts)
    first_origin_weights = degraded_ensemble.weights.loc[
        degraded_ensemble.weights["origin_date"].eq(base["origin_date"].min())
    ].set_index("expert")["weight"]
    assert first_origin_weights[JOINT_MODEL_NAME] == 0.0
    assert first_origin_weights["markov"] == pytest.approx(0.5)
    assert first_origin_weights["xgboost"] == pytest.approx(0.5)


def test_dynamic_ensemble_is_common_origin_causal_and_append_invariant() -> None:
    full = _expert_oos(rows=38)
    prefix = full.loc[full["origin_date"] < full["origin_date"].unique()[32]].copy()
    prefix_result = causal_dynamic_ensemble(prefix)
    full_result = causal_dynamic_ensemble(full)

    common_predictions = full_result.predictions.loc[
        full_result.predictions["origin_date"].isin(prefix["origin_date"].unique())
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(prefix_result.predictions, common_predictions)
    common_weights = full_result.weights.loc[
        full_result.weights["origin_date"].isin(prefix["origin_date"].unique())
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(prefix_result.weights, common_weights)

    weight_sums = full_result.weights.groupby("origin_date")["weight"].sum()
    np.testing.assert_allclose(weight_sums, 1.0, atol=1e-12)
    assert (
        full_result.weights["latest_eligible_target_date"].dropna()
        < full_result.weights.loc[
            full_result.weights["latest_eligible_target_date"].notna(), "origin_date"
        ]
    ).all()
    first_origin = full_result.weights["origin_date"].min()
    first_weights = full_result.weights.loc[
        full_result.weights["origin_date"].eq(first_origin), "weight"
    ]
    np.testing.assert_allclose(first_weights, 1.0 / 3.0, atol=1e-12)

    missing = full.loc[
        ~(
            full["model"].eq("markov")
            & full["origin_date"].eq(full["origin_date"].max())
        )
    ]
    with pytest.raises(ValueError, match="common origins"):
        causal_dynamic_ensemble(missing)


def test_dynamic_ensemble_excludes_current_fallback_and_never_uses_current_actual() -> None:
    source = _expert_oos(rows=35)
    origin = source["origin_date"].unique()[30]
    fallback_source = source.copy()
    fallback_source.loc[
        fallback_source["origin_date"].eq(origin)
        & fallback_source["model"].eq("xgboost"),
        "fallback",
    ] = True
    result = causal_dynamic_ensemble(fallback_source)
    origin_weights = result.weights.loc[result.weights["origin_date"].eq(origin)]
    xgboost_weight = origin_weights.loc[
        origin_weights["expert"].eq("xgboost"), "weight"
    ].iloc[0]
    assert float(xgboost_weight) == 0.0
    assert origin_weights["weight"].sum() == pytest.approx(1.0)

    changed_actual = fallback_source.copy()
    replacement = {
        "risk_on": "risk_off",
        "transition": "risk_on",
        "risk_off": "transition",
    }
    mask = changed_actual["origin_date"].eq(origin)
    changed_actual.loc[mask, "actual"] = changed_actual.loc[mask, "actual"].map(
        replacement
    )
    changed_result = causal_dynamic_ensemble(changed_actual)
    original_row = result.predictions.loc[result.predictions["origin_date"].eq(origin)]
    changed_row = changed_result.predictions.loc[
        changed_result.predictions["origin_date"].eq(origin)
    ]
    np.testing.assert_allclose(
        original_row[list(PROBABILITY_COLUMNS)],
        changed_row[list(PROBABILITY_COLUMNS)],
        atol=1e-12,
    )
    original_weights = result.weights.loc[result.weights["origin_date"].eq(origin)]
    changed_weights = changed_result.weights.loc[
        changed_result.weights["origin_date"].eq(origin)
    ]
    np.testing.assert_allclose(
        original_weights["weight"], changed_weights["weight"], atol=1e-12
    )


def test_dynamic_ensemble_scores_every_expert_on_identical_nonfallback_history() -> None:
    source = _expert_oos(rows=35)
    excluded_origin = source["origin_date"].unique()[15]
    latest_origin = source["origin_date"].max()
    degraded = source.copy()
    degraded.loc[
        degraded["origin_date"].eq(excluded_origin)
        & degraded["model"].eq("xgboost"),
        "fallback",
    ] = True

    result = causal_dynamic_ensemble(degraded)
    latest_weights = result.weights.loc[
        result.weights["origin_date"].eq(latest_origin)
    ].sort_values("expert")
    assert latest_weights["history_rows"].nunique() == 1
    assert latest_weights["common_history_rows"].nunique() == 1
    assert latest_weights["history_rows"].iloc[0] == latest_weights[
        "common_history_rows"
    ].iloc[0]

    without_contaminated_origin = source.loc[
        ~source["origin_date"].eq(excluded_origin)
    ]
    reference = causal_dynamic_ensemble(without_contaminated_origin)
    reference_weights = reference.weights.loc[
        reference.weights["origin_date"].eq(latest_origin)
    ].sort_values("expert")
    np.testing.assert_allclose(
        latest_weights["weight"], reference_weights["weight"], atol=1e-12
    )
    np.testing.assert_allclose(
        latest_weights["discounted_log_loss"],
        reference_weights["discounted_log_loss"],
        atol=1e-12,
    )


def test_augment_recomputes_common_origin_gate_and_stores_weight_artifact() -> None:
    base, transition, split_audit = _base_and_transition()
    selection = base.loc[base["evaluation_split"].eq("selection")]
    holdout = base.loc[base["evaluation_split"].eq("holdout")]
    selection_leaderboard = evaluate_predictions(selection)
    holdout_leaderboard = evaluate_predictions(holdout)
    benchmark = BenchmarkResult(
        leaderboard=holdout_leaderboard,
        champion="markov",
        predictions=base,
        split_audit=split_audit,
        profile=BenchmarkProfile.quick(),
        selection_end=pd.Timestamp("2021-08-01"),
        selection_leaderboard=selection_leaderboard,
        holdout_leaderboard=holdout_leaderboard,
    )

    result = augment_benchmark_with_structural_models(
        benchmark,
        SimpleNamespace(predictions=transition),
    )

    assert "stacking_weights" in {field.name for field in fields(BenchmarkResult)}
    assert result.stacking_weights is not None
    assert len(result.stacking_weights) == 40 * len(DEFAULT_ENSEMBLE_EXPERTS)
    assert {JOINT_MODEL_NAME, ENSEMBLE_MODEL_NAME} < set(result.leaderboard["model"])
    assert result.selection_diagnostics is not None
    assert result.selection_diagnostics["selected"].sum() == 1
    assert result.selection_diagnostics.loc[
        result.selection_diagnostics["selected"], "model"
    ].iloc[0] == result.champion
    assert set(result.selection_diagnostics["n_predictions"]) == {30}
    origin_counts = result.predictions.groupby("model")["origin_date"].nunique()
    assert origin_counts.nunique() == 1
    assert int(origin_counts.iloc[0]) == 40
    assert set(result.predictions["evaluation_split"]) == {"selection", "holdout"}
    assert MODEL_NAMES == tuple(
        name for name in MODEL_NAMES
    )  # structural candidates remain opt-in
    for name in (JOINT_MODEL_NAME, ENSEMBLE_MODEL_NAME, "joint_survival_hazard"):
        assert MODEL_REGISTRY[name].default is False
        assert MODEL_REGISTRY[name].kind == "synthetic"


def test_latest_forecast_uses_only_strictly_eligible_oos_losses() -> None:
    history = _expert_oos(rows=34)
    latest_target = pd.Timestamp(history["target_date"].max())
    origin = latest_target
    first = forecast_structural_probabilities(
        origin_date=origin,
        current_state="transition",
        markov_probability=[0.2, 0.6, 0.2],
        xgboost_probability=[0.25, 0.5, 0.25],
        binary_xgboost_p_change=0.3,
        historical_oos_predictions=history,
        expert_fallbacks={"xgboost": True},
        current_duration_weeks=7,
    )
    changed = history.copy()
    mask = changed["target_date"].eq(origin)
    replacement = {
        "risk_on": "risk_off",
        "transition": "risk_on",
        "risk_off": "transition",
    }
    changed.loc[mask, "actual"] = changed.loc[mask, "actual"].map(replacement)
    second = forecast_structural_probabilities(
        origin_date=origin,
        current_state="transition",
        markov_probability=[0.2, 0.6, 0.2],
        xgboost_probability=[0.25, 0.5, 0.25],
        binary_xgboost_p_change=0.3,
        historical_oos_predictions=changed,
        expert_fallbacks={"xgboost": True},
        current_duration_weeks=7,
    )

    pd.testing.assert_frame_equal(first.probabilities, second.probabilities)
    pd.testing.assert_frame_equal(first.stacking_weights, second.stacking_weights)
    weights = first.stacking_weights.set_index("expert")["weight"]
    assert weights["xgboost"] == 0.0
    assert weights[JOINT_MODEL_NAME] == 0.0
    assert weights["markov"] == pytest.approx(1.0)
    eligible_targets = first.stacking_weights["latest_eligible_target_date"].dropna()
    assert (eligible_targets < origin).all()
    np.testing.assert_allclose(
        first.probabilities[list(PROBABILITY_COLUMNS)].sum(axis=1), 1.0
    )
    assert set(first.probabilities["model"]) == {
        "markov",
        "xgboost",
        JOINT_MODEL_NAME,
        ENSEMBLE_MODEL_NAME,
    }
    audited = first.probabilities.set_index("model")
    assert audited.loc["markov", "ensemble_weight"] == pytest.approx(1.0)
    assert audited.loc["xgboost", "ensemble_weight"] == 0.0
    assert audited.loc[JOINT_MODEL_NAME, "ensemble_weight"] == 0.0
    assert set(first.probabilities["source_role"]) == {
        "base_expert",
        "hazard_destination_joint",
        "causal_ensemble",
    }


def test_joint_survival_projection_is_monotone_and_increments_duration() -> None:
    seen_durations: list[int] = []

    def duration_hazard(duration: int) -> float:
        seen_durations.append(duration)
        return min(0.05 + 0.002 * duration, 0.4)

    result = project_joint_survival_hazard(
        duration_hazard,
        current_duration_weeks=5,
        horizons=(1, 4, 13),
    )

    assert seen_durations == list(range(6, 19))
    assert result["horizon"].tolist() == [1, 4, 13]
    assert (result["p_change"].diff().dropna() >= 0.0).all()
    assert result.loc[0, "p_change"] <= result.loc[1, "p_change"]
    assert result.loc[1, "p_change"] <= result.loc[2, "p_change"]
    np.testing.assert_allclose(
        result["p_change"] + result["survival_probability"], 1.0
    )
