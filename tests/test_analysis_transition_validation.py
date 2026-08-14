from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from regime_lab.analysis.labels import STATE_ORDER
from regime_lab.analysis.models import BenchmarkProfile
from regime_lab.analysis.validation import TransitionBenchmarkResult
from regime_lab.analysis.validation import _transition_targets
from regime_lab.analysis.validation import evaluate_transition_predictions
from regime_lab.analysis.validation import run_transition_benchmark


def _inputs(periods: int = 270) -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2018-01-05", periods=periods, freq="W-FRI")
    position = np.arange(periods, dtype=float)
    features = pd.DataFrame(
        {
            "risk_score": np.sin(position / 9.0) + 0.15 * np.cos(position / 3.0),
            "credit": np.cos(position / 13.0),
            "trend": position / periods + np.sin(position / 17.0),
        },
        index=index,
    )
    episode = (np.arange(periods) // 11) % len(STATE_ORDER)
    states = pd.Series(
        [STATE_ORDER[int(value)] for value in episode],
        index=index,
        name="regime",
        dtype="object",
    )
    return features, states


def _profile(*, max_origins: int | None = 8) -> BenchmarkProfile:
    return BenchmarkProfile(
        name="test",
        max_origins=max_origins,
        minimum_train_weeks=52,
        random_forest_trees=8,
        extra_trees=8,
        hist_gradient_iterations=12,
        svm_calibration_splits=2,
        hmm_iterations=10,
        xgboost_trees=8,
        spline_pca_components=3,
    )


@pytest.fixture(scope="module")
def benchmark() -> TransitionBenchmarkResult:
    features, states = _inputs()
    return run_transition_benchmark(
        features,
        states,
        profile=_profile(),
        selection_end="2022-01-01",
        selection_max_origins=8,
        minimum_selection_predictions=5,
        minimum_diagnostic_predictions=5,
        minimum_inner_predictions=5,
    )


def test_event_target_detects_leave_and_return_without_terminal_state_bias() -> None:
    index = pd.date_range("2020-01-03", periods=5, freq="W-FRI")
    states = pd.Series(
        ["risk_on", "transition", "risk_on", "risk_on", "risk_off"],
        index=index,
    )
    event, destination = _transition_targets(states, 2)

    assert bool(event.iloc[0]) is True
    assert destination.iloc[0] == "transition"
    assert bool(event.iloc[1]) is True
    assert destination.iloc[1] == "risk_on"


def test_horizon_specific_purge_keeps_every_training_target_pre_origin(
    benchmark: TransitionBenchmarkResult,
) -> None:
    audit = benchmark.split_audit
    assert set(audit["horizon"]) == {1, 4, 13}
    assert (audit["last_train_target_end"] < audit["origin_date"]).all()
    assert (audit["gap"] == audit["horizon"]).all()
    assert (audit["purged_origin_count"] == audit["horizon"]).all()
    assert (
        (audit["target_end"] - audit["origin_date"]).dt.days
        == audit["horizon"] * 7
    ).all()


def test_probabilities_are_strict_and_latest_forecasts_are_not_scored(
    benchmark: TransitionBenchmarkResult,
) -> None:
    predictions = benchmark.predictions
    assert predictions["p_change"].between(0.0, 1.0, inclusive="neither").all()
    assert predictions["raw_p_change"].between(
        0.0, 1.0, inclusive="neither"
    ).all()
    assert predictions["threshold"].between(0.0, 1.0).all()
    assert set(predictions["model"]) == {
        "empirical_hazard",
        "markov_hazard",
        "duration_tvtp_hurdle",
        "regularized_logistic",
    }

    latest = benchmark.latest_forecasts()
    assert set(latest["horizon"]) == {1, 4, 13}
    assert latest.groupby("horizon").size().to_dict() == {1: 1, 4: 4, 13: 13}
    assert latest["evaluation_split"].eq("prospective").all()
    assert latest["actual_change"].isna().all()
    assert (latest["last_train_target_end"] < latest["origin_date"]).all()
    assert (latest["gap"] == latest["horizon"]).all()
    assert (latest["target_start"] - latest["origin_date"]).dt.days.eq(7).all()
    assert (
        (latest["target_end"] - latest["origin_date"]).dt.days
        == latest["horizon"] * 7
    ).all()
    evaluated_keys = set(zip(predictions["horizon"], predictions["origin_date"]))
    prospective_keys = set(zip(latest["horizon"], latest["origin_date"]))
    assert evaluated_keys.isdisjoint(prospective_keys)
    assert len(benchmark.latest_forecasts(4)) == 4
    candidates = benchmark.latest_candidate_forecasts(horizon=1)
    expected_candidates = set(benchmark.candidate_status.loc[
        benchmark.candidate_status["available"], "model"
    ].astype(str))
    assert set(candidates["model"].astype(str)) == expected_candidates
    assert len(candidates) == len(expected_candidates)
    for model in expected_candidates:
        row = benchmark.latest_candidate_forecasts(horizon=1, model=model)
        assert len(row) == 1
        assert row.iloc[0]["last_train_target_end"] < row.iloc[0]["origin_date"]
    for horizon, group in latest.groupby("horizon"):
        assert group["model"].eq(benchmark.champions_by_horizon[horizon]).all()
        assert group["threshold"].nunique() == 1
        assert group["selection_scope"].eq("selection_oos_only").all()


def test_selection_and_retrospective_diagnostics_are_separated(
    benchmark: TransitionBenchmarkResult,
) -> None:
    cutoff = benchmark.selection_end
    selection = benchmark.predictions.loc[
        benchmark.predictions["evaluation_split"].eq("selection")
    ]
    diagnostic = benchmark.predictions.loc[
        benchmark.predictions["evaluation_split"].eq("retrospective_diagnostic")
    ]
    assert (selection["target_end"] < cutoff).all()
    assert (diagnostic["origin_date"] >= cutoff).all()
    for horizon in (1, 4, 13):
        horizon_selection = selection.loc[selection["horizon"].eq(horizon)]
        horizon_diagnostic = diagnostic.loc[diagnostic["horizon"].eq(horizon)]
        assert horizon_selection["target_end"].max() < horizon_diagnostic[
            "target_start"
        ].min()
    assert set(benchmark.leaderboard["evaluation_split"]) == {
        "selection",
        "retrospective_diagnostic",
    }
    nested_diagnostic = benchmark.nested_selection.loc[
        benchmark.nested_selection["evaluation_split"].eq(
            "retrospective_diagnostic"
        )
    ]
    assert nested_diagnostic["selection_locked"].all()
    assert nested_diagnostic["selection_scope"].eq(
        "earlier_selection_oos_only"
    ).all()


def test_appending_future_rows_cannot_change_existing_oos_predictions() -> None:
    features, states = _inputs(285)
    common = {
        "horizons": (4,),
        "profile": _profile(max_origins=None),
        "models": ("empirical_hazard", "markov_hazard"),
        "selection_end": "2022-01-01",
        "minimum_selection_predictions": 5,
        "minimum_diagnostic_predictions": 5,
        "minimum_inner_predictions": 5,
    }
    prefix = run_transition_benchmark(features.iloc[:260], states.iloc[:260], **common)
    extended = run_transition_benchmark(features, states, **common)
    columns = [
        "origin_date",
        "target_start",
        "target_end",
        "horizon",
        "model",
        "evaluation_split",
        "actual_change",
        "raw_p_change",
        "p_change",
        "threshold",
        "predicted_change",
        "train_size",
        "gap",
        "fallback",
        "fallback_reason",
        "calibration_method",
        "calibration_fallback",
        "calibration_fallback_reason",
        "threshold_method",
    ]
    extended_common = extended.predictions.loc[
        extended.predictions["target_end"] <= prefix.predictions["target_end"].max()
    ]
    assert_frame_equal(
        prefix.predictions[columns].reset_index(drop=True),
        extended_common[columns].reset_index(drop=True),
    )

    # Once future weeks arrive, a formerly unresolved prospective forecast
    # becomes an evaluation row.  Its probability and frozen decision layer
    # must remain exactly what was knowable at the original origin.
    former_prospective = prefix.latest_forecasts(4)
    realised = extended.predictions.merge(
        former_prospective[["origin_date", "horizon", "model"]],
        on=["origin_date", "horizon", "model"],
        how="inner",
        validate="one_to_one",
    )
    assert len(realised) == 4
    forecast_comparison = former_prospective.sort_values("origin_date").reset_index(
        drop=True
    )
    realised = realised.sort_values("origin_date").reset_index(drop=True)
    for column in (
        "target_start",
        "target_end",
        "raw_p_change",
        "p_change",
        "threshold",
        "predicted_change",
        "train_size",
        "gap",
        "fallback",
        "fallback_reason",
        "calibration_method",
        "calibration_fallback",
        "calibration_fallback_reason",
        "threshold_method",
    ):
        pd.testing.assert_series_equal(
            forecast_comparison[column],
            realised[column],
            check_names=False,
        )


def test_average_precision_is_null_when_no_positive_event_exists() -> None:
    frame = pd.DataFrame(
        {
            "horizon": [1, 1, 1],
            "model": ["constant"] * 3,
            "evaluation_split": ["selection"] * 3,
            "actual_change": [False, False, False],
            "p_change": [0.1, 0.2, 0.3],
            "predicted_change": [False, False, False],
            "fallback": [False, False, False],
        }
    )
    result = evaluate_transition_predictions(frame)
    assert pd.isna(result.loc[0, "average_precision"])
    assert result.loc[0, "event_count"] == 0


def test_joint_survival_is_opt_in_and_uses_one_week_hazard_for_each_horizon() -> None:
    features, states = _inputs(235)
    result = run_transition_benchmark(
        features,
        states,
        horizons=(1, 4),
        profile=_profile(max_origins=3),
        include_xgboost=True,
        include_joint_survival=True,
        selection_end="2022-01-01",
        selection_max_origins=3,
        minimum_selection_predictions=3,
        minimum_diagnostic_predictions=3,
        minimum_inner_predictions=3,
    )

    joint = result.predictions.loc[
        result.predictions["model"].eq("joint_survival_hazard")
    ]
    assert set(joint["horizon"]) == {1, 4}
    assert joint["raw_p_change"].between(0.0, 1.0, inclusive="neither").all()
    assert result.candidate_status.set_index("model").loc[
        "joint_survival_hazard", "available"
    ]
    status = result.candidate_status.set_index("model").loc[
        "joint_survival_hazard"
    ]
    assert not bool(status["selection_eligible"])
    assert status["role"] == "shadow_coherence_benchmark"
    assert "joint_survival_hazard" not in set(result.champions_by_horizon.values())
    latest = result.latest_candidate_forecasts(
        horizon=4, model="joint_survival_hazard"
    )
    assert len(latest) == 4
    assert (latest["last_train_target_end"] < latest["origin_date"]).all()
