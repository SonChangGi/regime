from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from regime_lab.analysis.labels import STATE_ORDER
from regime_lab.analysis.models import MODEL_REGISTRY, BenchmarkProfile
from regime_lab.analysis.structural_models import (
    DEFAULT_ENSEMBLE_EXPERTS,
    ENSEMBLE_MODEL_NAME,
    JOINT_MODEL_NAME,
    MULTISCALE_ENSEMBLE_AGGREGATION,
    MULTISCALE_ENSEMBLE_HALF_LIVES_WEEKS,
    MULTISCALE_ENSEMBLE_MODEL_NAME,
    MULTISCALE_INNER_POOL_METHOD,
    MULTISCALE_SCALE_FORECAST_COLUMNS,
    augment_benchmark_with_structural_models,
    causal_dynamic_ensemble,
    causal_multiscale_ensemble,
    forecast_structural_probabilities,
)
from regime_lab.analysis.validation import BenchmarkResult, evaluate_predictions


PROBABILITY_COLUMNS = tuple(f"p_{state}" for state in STATE_ORDER)


def _probability_for(actual: str, confidence: float) -> np.ndarray:
    probability = np.full(3, (1.0 - confidence) / 2.0, dtype=float)
    probability[STATE_ORDER.index(actual)] = confidence
    return probability


def _expert_oos(rows: int = 40) -> pd.DataFrame:
    origins = pd.date_range(
        "2020-01-03 21:00:00", periods=rows, freq="W-FRI", tz="UTC"
    )
    pattern = np.asarray(
        ["risk_on", "risk_on", "transition", "risk_off", "transition"],
        dtype=object,
    )
    current = np.resize(pattern, rows)
    actual = np.roll(current, -1)
    confidence = {
        "markov": 0.58,
        "xgboost": 0.67,
        JOINT_MODEL_NAME: 0.72,
    }
    records: list[dict[str, object]] = []
    for position, origin in enumerate(origins):
        origin = pd.Timestamp(origin)
        for model in DEFAULT_ENSEMBLE_EXPERTS:
            probability = _probability_for(
                str(actual[position]), confidence[model]
            )
            records.append(
                {
                    "origin_date": origin,
                    "target_date": origin + pd.offsets.Week(1),
                    "model": model,
                    "evaluation_split": (
                        "selection" if position < 30 else "holdout"
                    ),
                    "current_state": str(current[position]),
                    "actual": str(actual[position]),
                    "predicted": str(actual[position]),
                    **{
                        column: float(probability[index])
                        for index, column in enumerate(PROBABILITY_COLUMNS)
                    },
                    "fallback": False,
                    "fallback_reason": "",
                }
            )
    return pd.DataFrame(records)


def _base_and_transition(
    rows: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    experts = _expert_oos(rows)
    origins = experts["origin_date"].drop_duplicates().reset_index(drop=True)
    base_records: list[dict[str, object]] = []
    transition_records: list[dict[str, object]] = []
    split_records: list[dict[str, object]] = []
    for position, origin in enumerate(origins):
        expert_origin = experts.loc[experts["origin_date"].eq(origin)]
        reference = expert_origin.iloc[0]
        split = str(reference["evaluation_split"])
        for model, confidence in (("majority", 0.36), ("persistence", 0.42)):
            probability = _probability_for(str(reference["actual"]), confidence)
            base_records.append(
                {
                    **reference.to_dict(),
                    "model": model,
                    **{
                        column: float(probability[index])
                        for index, column in enumerate(PROBABILITY_COLUMNS)
                    },
                    "train_size": 520 + position,
                    "gap": 1,
                }
            )
        for model in ("markov", "xgboost"):
            row = expert_origin.loc[expert_origin["model"].eq(model)].iloc[0]
            base_records.append(
                {**row.to_dict(), "train_size": 520 + position, "gap": 1}
            )
        changed = str(reference["actual"]) != str(reference["current_state"])
        transition_records.append(
            {
                "origin_date": origin,
                "target_start": reference["target_date"],
                "target_end": reference["target_date"],
                "horizon": 1,
                "model": "binary_xgboost",
                "evaluation_split": (
                    split if split == "selection" else "retrospective_diagnostic"
                ),
                "current_state": reference["current_state"],
                "actual_change": changed,
                "raw_p_change": 0.75 if changed else 0.15,
                "p_change": 0.75 if changed else 0.15,
                "fallback": False,
                "fallback_reason": "",
                "calibration_fallback": False,
                "calibration_fallback_reason": "",
            }
        )
        split_records.append(
            {
                "origin_date": origin,
                "target_date": reference["target_date"],
                "train_size": 520 + position,
                "evaluation_split": split,
            }
        )
    return (
        pd.DataFrame(base_records),
        pd.DataFrame(transition_records),
        pd.DataFrame(split_records),
    )


def test_multiscale_oos_is_exact_fixed_average_and_append_invariant() -> None:
    full = _expert_oos(40)
    prefix_origin = full["origin_date"].drop_duplicates().iloc[34]
    prefix = full.loc[full["origin_date"] < prefix_origin]
    result = causal_multiscale_ensemble(full)
    prefix_result = causal_multiscale_ensemble(prefix)

    assert tuple(result.scale_predictions.columns) == (
        MULTISCALE_SCALE_FORECAST_COLUMNS
    )
    assert set(result.scale_predictions["scale_half_life_weeks"]) == set(
        MULTISCALE_ENSEMBLE_HALF_LIVES_WEEKS
    )
    assert result.scale_predictions["outer_scale_weight"].eq(1.0 / 3.0).all()
    assert result.weights.groupby("origin_date").size().eq(9).all()
    assert result.scale_predictions.groupby("origin_date").size().eq(3).all()

    scale_arrays: list[np.ndarray] = []
    for scale in MULTISCALE_ENSEMBLE_HALF_LIVES_WEEKS:
        inner = causal_dynamic_ensemble(full, half_life_weeks=float(scale))
        audited = result.scale_predictions.loc[
            result.scale_predictions["scale_half_life_weeks"].eq(scale)
        ].sort_values(["origin_date", "target_date"])
        np.testing.assert_array_equal(
            audited[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float),
            inner.predictions[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float),
        )
        scale_arrays.append(
            audited[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
        )
    np.testing.assert_array_equal(
        result.predictions[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float),
        np.mean(np.stack(scale_arrays), axis=0),
    )

    prefix_origins = prefix["origin_date"].unique()
    pd.testing.assert_frame_equal(
        prefix_result.predictions,
        result.predictions.loc[
            result.predictions["origin_date"].isin(prefix_origins)
        ].reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        prefix_result.weights,
        result.weights.loc[result.weights["origin_date"].isin(prefix_origins)].reset_index(
            drop=True
        ),
    )
    pd.testing.assert_frame_equal(
        prefix_result.scale_predictions,
        result.scale_predictions.loc[
            result.scale_predictions["origin_date"].isin(prefix_origins)
        ].reset_index(drop=True),
    )
    eligible = result.weights["latest_eligible_target_date"].notna()
    assert (
        result.weights.loc[eligible, "latest_eligible_target_date"]
        < result.weights.loc[eligible, "origin_date"]
    ).all()


def test_multiscale_zeroes_current_fallbacks_and_has_uniform_all_fallback() -> None:
    source = _expert_oos(38)
    origin = source["origin_date"].drop_duplicates().iloc[34]
    degraded = source.copy()
    degraded.loc[
        degraded["origin_date"].eq(origin)
        & degraded["model"].eq("xgboost"),
        "fallback",
    ] = True
    result = causal_multiscale_ensemble(degraded)
    current_weights = result.weights.loc[result.weights["origin_date"].eq(origin)]
    assert current_weights.loc[
        current_weights["expert"].eq("xgboost"), "weight"
    ].eq(0.0).all()
    np.testing.assert_allclose(
        current_weights.groupby("half_life_weeks")["weight"].sum(), 1.0
    )

    all_fallback = source.copy()
    all_fallback.loc[all_fallback["origin_date"].eq(origin), "fallback"] = True
    fallback_result = causal_multiscale_ensemble(all_fallback)
    final = fallback_result.predictions.loc[
        fallback_result.predictions["origin_date"].eq(origin)
    ].iloc[0]
    assert bool(final["fallback"]) is True
    np.testing.assert_array_equal(
        final[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float),
        np.full(3, 1.0 / 3.0),
    )
    scale_rows = fallback_result.scale_predictions.loc[
        fallback_result.scale_predictions["origin_date"].eq(origin)
    ]
    assert scale_rows["fallback"].all()
    np.testing.assert_array_equal(
        scale_rows[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float),
        np.full((3, 3), 1.0 / 3.0),
    )


def test_latest_multiscale_emits_three_bound_scales_and_nine_weights() -> None:
    history = _expert_oos(38)
    origin = pd.Timestamp(history["target_date"].max())
    base_arguments = {
        "origin_date": origin,
        "current_state": "transition",
        "markov_probability": [0.2, 0.6, 0.2],
        "xgboost_probability": [0.25, 0.5, 0.25],
        "binary_xgboost_p_change": 0.3,
        "historical_oos_predictions": history,
        "current_duration_weeks": 7,
    }
    default = forecast_structural_probabilities(**base_arguments)
    result = forecast_structural_probabilities(
        **base_arguments, include_multiscale=True
    )

    assert MULTISCALE_ENSEMBLE_MODEL_NAME not in set(default.probabilities["model"])
    assert default.multiscale_scale_predictions is None
    assert MULTISCALE_ENSEMBLE_MODEL_NAME in set(result.probabilities["model"])
    scales = result.multiscale_scale_predictions
    assert scales is not None
    assert len(scales) == 3
    assert scales["row_role"].eq("latest_forecast").all()
    assert scales["evaluation_split"].eq("prospective").all()
    assert scales["target_date"].eq(origin + pd.offsets.Week(1)).all()
    assert scales["expert_forecast_artifact"].eq(
        "structural-forecasts.csv"
    ).all()
    assert scales["expert_forecast_key"].str.contains(
        f"origin={origin.date().isoformat()}", regex=False
    ).all()
    multiscale_weights = result.stacking_weights.loc[
        result.stacking_weights["ensemble_model"].eq(
            MULTISCALE_ENSEMBLE_MODEL_NAME
        )
    ]
    assert len(multiscale_weights) == 9
    np.testing.assert_allclose(
        multiscale_weights.groupby("half_life_weeks")["weight"].sum(), 1.0
    )
    eligible = multiscale_weights["latest_eligible_target_date"].notna()
    assert (
        multiscale_weights.loc[eligible, "latest_eligible_target_date"] < origin
    ).all()
    aggregate = result.probabilities.loc[
        result.probabilities["model"].eq(MULTISCALE_ENSEMBLE_MODEL_NAME),
        list(PROBABILITY_COLUMNS),
    ].to_numpy(dtype=float)
    np.testing.assert_array_equal(
        aggregate,
        scales.groupby("origin_date", sort=True)[list(PROBABILITY_COLUMNS)]
        .mean()
        .to_numpy(dtype=float),
    )


def test_v5_opt_in_augmentation_uses_existing_selection_gate() -> None:
    base, transition, split_audit = _base_and_transition()
    selection = base.loc[base["evaluation_split"].eq("selection")]
    holdout = base.loc[base["evaluation_split"].eq("holdout")]
    benchmark = BenchmarkResult(
        leaderboard=evaluate_predictions(holdout),
        champion="markov",
        predictions=base,
        split_audit=split_audit,
        profile=BenchmarkProfile.quick(),
        selection_end=pd.Timestamp("2020-08-01", tz="UTC"),
        selection_leaderboard=evaluate_predictions(selection),
        holdout_leaderboard=evaluate_predictions(holdout),
    )

    default = augment_benchmark_with_structural_models(
        benchmark, SimpleNamespace(predictions=transition)
    )
    v5 = augment_benchmark_with_structural_models(
        benchmark,
        SimpleNamespace(predictions=transition),
        include_multiscale=True,
    )

    assert MULTISCALE_ENSEMBLE_MODEL_NAME not in set(default.predictions["model"])
    assert default.multiscale_scale_forecasts is None
    assert MULTISCALE_ENSEMBLE_MODEL_NAME in set(v5.predictions["model"])
    assert v5.multiscale_scale_forecasts is not None
    assert len(v5.multiscale_scale_forecasts) == 40 * 3
    assert v5.stacking_weights is not None
    assert len(
        v5.stacking_weights.loc[
            v5.stacking_weights["ensemble_model"].eq(
                MULTISCALE_ENSEMBLE_MODEL_NAME
            )
        ]
    ) == 40 * 9
    assert v5.selection_diagnostics is not None
    diagnostic = v5.selection_diagnostics.loc[
        v5.selection_diagnostics["model"].eq(MULTISCALE_ENSEMBLE_MODEL_NAME)
    ]
    assert len(diagnostic) == 1
    assert {
        "absolute_log_loss_improvement",
        "holm_adjusted_p_value",
        "brier_difference",
        "fallback_count",
        "gate_passed",
    }.issubset(diagnostic.columns)
    assert v5.selection_diagnostics["selected"].sum() == 1


def test_multiscale_registry_and_frozen_method_contract() -> None:
    spec = MODEL_REGISTRY[MULTISCALE_ENSEMBLE_MODEL_NAME]
    assert spec.default is False
    assert spec.kind == "synthetic"
    assert spec.complexity_rank == 16
    assert spec.search_space["scale_half_lives_weeks"] == ((26, 52, 104),)
    assert spec.search_space["outer_scale_weights"] == ((1.0 / 3.0,) * 3,)
    assert spec.search_space["aggregation"] == (
        MULTISCALE_ENSEMBLE_AGGREGATION,
    )
    assert spec.search_space["inner_pool_method"] == (
        MULTISCALE_INNER_POOL_METHOD,
    )

    source = _expert_oos(30)
    with pytest.raises(ValueError, match=r"exactly \(26, 52, 104\)"):
        causal_multiscale_ensemble(
            source, scale_half_lives_weeks=(13, 52, 104)
        )
    with pytest.raises(ValueError, match="exactly 26"):
        causal_multiscale_ensemble(source, minimum_history_rows=25)
