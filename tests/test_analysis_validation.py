from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from regime_lab.analysis import BenchmarkProfile, GaussianHMMChallenger
from regime_lab.analysis import forecast_next_regime, run_benchmark
from regime_lab.analysis.models import DIRECT_NEXT_STATE_MODEL_NAMES, MODEL_NAMES
from regime_lab.analysis.validation import PROBABILITY_COLUMNS, evaluate_predictions
from regime_lab.analysis.validation import select_champion_with_diagnostics
from regime_lab.schema import STATE_ORDER


def _model_inputs(rows: int = 170) -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2017-01-06", periods=rows, freq="W-FRI")
    position = np.arange(rows)
    slow_cycle = np.sin(position / 7.0)
    fast_cycle = np.cos(position / 3.5)
    features = pd.DataFrame(
        {
            "trend": slow_cycle + 0.1 * fast_cycle,
            "stress": -slow_cycle + 0.15 * np.sin(position / 2.0),
            "macro": np.sin((position - 2) / 11.0),
            "credit": np.cos((position - 1) / 13.0),
        },
        index=index,
    )
    # Add missing values that must be handled by a fold-local imputer.
    features.loc[index[::17], "macro"] = np.nan
    states = pd.Series(
        np.where(
            slow_cycle > 0.35,
            "risk_on",
            np.where(slow_cycle < -0.35, "risk_off", "transition"),
        ),
        index=index,
        name="regime",
    )
    return features, states


def _paired_selection_predictions() -> pd.DataFrame:
    index = pd.date_range("2021-01-08", periods=78, freq="W-FRI")
    actual = np.resize(np.asarray(STATE_ORDER, dtype=object), len(index))
    specifications = {
        "markov": (0.55, "balanced", False),
        "good_model": (0.75, "balanced", False),
        "marginal_model": (0.57, "balanced", False),
        "brier_bad_model": (0.59, "concentrated", False),
        "fallback_model": (0.80, "balanced", True),
    }
    rows: list[dict[str, object]] = []
    for model, (actual_probability, allocation, has_fallback) in specifications.items():
        for position, (target_date, actual_state) in enumerate(
            zip(index, actual, strict=True)
        ):
            probabilities = {state: 0.0 for state in STATE_ORDER}
            probabilities[str(actual_state)] = actual_probability
            alternatives = [state for state in STATE_ORDER if state != actual_state]
            if allocation == "concentrated":
                probabilities[alternatives[0]] = 1.0 - actual_probability
            else:
                for state in alternatives:
                    probabilities[state] = (1.0 - actual_probability) / 2.0
            rows.append(
                {
                    "origin_date": pd.Timestamp(target_date)
                    - pd.Timedelta(7, unit="D"),
                    "target_date": target_date,
                    "model": model,
                    "evaluation_split": "selection",
                    "current_state": str(actual_state),
                    "actual": str(actual_state),
                    "predicted": str(actual_state),
                    **{f"p_{state}": probabilities[state] for state in STATE_ORDER},
                    "fallback": bool(has_fallback and position == 0),
                }
            )
    return pd.DataFrame(rows)


def test_one_week_gap_is_visible_in_split_audit() -> None:
    features, states = _model_inputs()
    result = run_benchmark(
        features,
        states,
        profile="quick",
        models=("persistence",),
        gap=1,
        minimum_train_weeks=52,
    )

    audit = result.split_audit
    assert (audit["purged_origin_count"] == 1).all()
    assert (audit["gap"] == 1).all()
    assert (
        audit["origin_date"] - audit["last_train_origin"]
        == pd.to_timedelta(14, unit="D")
    ).all()
    assert (
        audit["origin_date"] - audit["first_purged_origin"]
        == pd.to_timedelta(7, unit="D")
    ).all()
    assert (audit["last_train_target"] < audit["origin_date"]).all()


def test_persistence_baseline_copies_current_state_and_probabilities_sum() -> None:
    features, states = _model_inputs()
    result = run_benchmark(
        features,
        states,
        profile="quick",
        models=("persistence",),
        gap=1,
        minimum_train_weeks=52,
    )
    predictions = result.predictions

    expected = predictions["origin_date"].map(states)
    assert predictions["predicted"].tolist() == expected.tolist()
    np.testing.assert_allclose(
        predictions[list(PROBABILITY_COLUMNS)].sum(axis=1), 1.0, atol=1e-12
    )


def test_quick_benchmark_is_deterministic_and_limited_to_ten_origins() -> None:
    features, states = _model_inputs()
    models = (
        "majority",
        "persistence",
        "markov",
        "elastic_net_logistic",
        "calibrated_linear_svm",
    )
    first = run_benchmark(
        features,
        states,
        profile="quick",
        models=models,
        minimum_train_weeks=52,
        random_state=29,
    )
    second = run_benchmark(
        features,
        states,
        profile="quick",
        models=models,
        minimum_train_weeks=52,
        random_state=29,
    )

    assert first.champion == second.champion
    assert first.predictions["origin_date"].nunique() == 10
    columns = ["model", "origin_date", "actual", *PROBABILITY_COLUMNS, "fallback"]
    assert_frame_equal(
        first.predictions[columns].reset_index(drop=True),
        second.predictions[columns].reset_index(drop=True),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_parallel_model_workers_preserve_serial_predictions() -> None:
    features, states = _model_inputs()
    models = (
        "majority",
        "markov",
        "ridge_logistic",
        "shrinkage_lda",
        "spline_logistic",
    )
    arguments = {
        "profile": "quick",
        "models": models,
        "minimum_train_weeks": 52,
        "random_state": 41,
    }
    serial = run_benchmark(features, states, model_workers=1, **arguments)
    parallel = run_benchmark(features, states, model_workers=3, **arguments)

    columns = ["model", "origin_date", "actual", *PROBABILITY_COLUMNS, "fallback"]
    assert_frame_equal(
        serial.predictions[columns].reset_index(drop=True),
        parallel.predictions[columns].reset_index(drop=True),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    assert serial.champion == parallel.champion


def test_full_model_suite_emits_aligned_probabilities() -> None:
    features, states = _model_inputs()
    result = run_benchmark(
        features,
        states,
        profile=BenchmarkProfile.quick(),
        minimum_train_weeks=52,
    )

    assert set(result.leaderboard["model"]) == set(MODEL_NAMES)
    np.testing.assert_allclose(
        result.predictions[list(PROBABILITY_COLUMNS)].sum(axis=1), 1.0, atol=1e-12
    )


def test_next_week_forecast_returns_exact_shared_state_order() -> None:
    features, states = _model_inputs()
    probabilities = forecast_next_regime(
        features,
        states,
        champion_name="markov",
        as_of=features.index[-1] + pd.DateOffset(days=2),
        gap=1,
        minimum_train_weeks=52,
    )

    assert tuple(probabilities.index) == STATE_ORDER
    assert np.isclose(probabilities.sum(), 1.0)
    assert probabilities.attrs["as_of"] == features.index[-1].isoformat()


def test_explicit_direct_models_run_without_changing_default_suite() -> None:
    features, states = _model_inputs()
    requested = (
        "markov",
        "pca_ridge_logistic",
        "discounted_markov_208w",
    )
    result = run_benchmark(
        features,
        states,
        profile="quick",
        models=requested,
        gap=1,
        minimum_train_weeks=52,
    )

    assert set(result.predictions["model"]) == set(requested)
    assert all(name not in MODEL_NAMES for name in DIRECT_NEXT_STATE_MODEL_NAMES)
    np.testing.assert_allclose(
        result.predictions[list(PROBABILITY_COLUMNS)].sum(axis=1),
        1.0,
        atol=1e-12,
    )


def test_explicit_models_continue_to_reject_synthetic_structural_candidates() -> None:
    features, states = _model_inputs()

    with np.testing.assert_raises_regex(
        ValueError,
        "unknown benchmark models.*xgb_hazard_destination",
    ):
        run_benchmark(
            features,
            states,
            profile="quick",
            models=("markov", "xgb_hazard_destination"),
            minimum_train_weeks=52,
        )


def test_pca_ridge_forecast_does_not_read_rows_after_as_of() -> None:
    features, states = _model_inputs()
    as_of = features.index[-12]
    altered_features = features.copy()
    altered_states = states.copy()
    future = features.index > as_of
    altered_features.loc[future, :] = 1_000_000.0
    altered_states.loc[future] = np.resize(
        np.asarray(STATE_ORDER, dtype=object),
        int(future.sum()),
    )
    arguments = {
        "champion_name": "pca_ridge_logistic",
        "as_of": as_of,
        "gap": 1,
        "minimum_train_weeks": 52,
        "random_state": 37,
    }

    original = forecast_next_regime(features, states, **arguments)
    changed = forecast_next_regime(altered_features, altered_states, **arguments)

    np.testing.assert_allclose(original.to_numpy(), changed.to_numpy(), rtol=0, atol=0)
    assert original.attrs == changed.attrs


@pytest.mark.parametrize(
    "champion_name",
    ("pca_ridge_logistic", "discounted_markov_208w"),
)
def test_optional_direct_models_support_next_regime_forecast(champion_name: str) -> None:
    features, states = _model_inputs()
    probability = forecast_next_regime(
        features,
        states,
        champion_name=champion_name,
        gap=1,
        minimum_train_weeks=52,
        random_state=43,
    )

    assert tuple(probability.index) == STATE_ORDER
    assert np.isclose(probability.sum(), 1.0)
    assert probability.attrs["champion"] == champion_name
    assert probability.attrs["fallback"] is False


def test_recency_weighted_xgboost_supports_next_regime_forecast() -> None:
    pytest.importorskip("xgboost")
    features, states = _model_inputs()
    probability = forecast_next_regime(
        features,
        states,
        champion_name="recency_weighted_xgboost_208w",
        profile=replace(BenchmarkProfile.quick(), xgboost_trees=8),
        gap=1,
        minimum_train_weeks=52,
        random_state=43,
    )

    assert tuple(probability.index) == STATE_ORDER
    assert np.isclose(probability.sum(), 1.0)
    assert probability.attrs["champion"] == "recency_weighted_xgboost_208w"
    assert probability.attrs["fallback"] is False


def test_hmm_mapping_keeps_exactly_one_week_supervised_horizon() -> None:
    class IdentityTransformer:
        def transform(self, values):
            return np.asarray(values)

    class FakeHMM:
        # A transition would reverse the hidden state, making the test fail.
        transmat_ = np.array([[0.0, 1.0], [1.0, 0.0]])

        def predict_proba(self, values):
            return np.array([[0.8, 0.2]])

    challenger = GaussianHMMChallenger.__new__(GaussianHMMChallenger)
    challenger.imputer_ = IdentityTransformer()
    challenger.scaler_ = IdentityTransformer()
    challenger.model_ = FakeHMM()
    challenger.hidden_to_state_ = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.3, 0.6],
        ]
    )

    result = challenger.predict_proba(pd.DataFrame([[1.0]]))
    expected = np.array([[0.8, 0.2]]) @ challenger.hidden_to_state_
    np.testing.assert_allclose(result, expected / expected.sum(axis=1, keepdims=True))


def test_time_split_selects_champion_without_reading_holdout_actuals() -> None:
    features, states = _model_inputs()
    cutoff = features.index[105]
    altered_holdout = states.copy()
    tail_length = len(altered_holdout.iloc[105:])
    altered_holdout.iloc[105:] = np.resize(
        np.array(["risk_on", "risk_off", "transition"], dtype=object), tail_length
    )
    profile = BenchmarkProfile.quick().with_overrides(
        max_origins=40, minimum_train_weeks=52
    )
    arguments = {
        "profile": profile,
        "models": ("majority", "persistence", "markov"),
        "selection_end": cutoff,
        "minimum_selection_predictions": 5,
        "minimum_holdout_predictions": 5,
    }

    original = run_benchmark(features, states, **arguments)
    changed = run_benchmark(features, altered_holdout, **arguments)

    assert original.champion == changed.champion
    assert_frame_equal(original.selection_leaderboard, changed.selection_leaderboard)
    assert_frame_equal(original.selection_diagnostics, changed.selection_diagnostics)
    assert not np.allclose(
        original.holdout_leaderboard["log_loss"],
        changed.holdout_leaderboard["log_loss"],
    )


def test_selection_budget_is_independent_from_holdout_profile_budget() -> None:
    features, states = _model_inputs()
    cutoff = features.index[105]
    base_profile = BenchmarkProfile.quick().with_overrides(
        max_origins=10,
        minimum_train_weeks=52,
    )
    standard_profile = replace(base_profile, name="standard")
    common = {
        "features": features,
        "states": states,
        "models": ("persistence",),
        "selection_end": cutoff,
        "minimum_selection_predictions": 3,
        "minimum_holdout_predictions": 3,
    }

    standard = run_benchmark(profile=standard_profile, **common)
    quick = run_benchmark(profile=base_profile, **common)
    explicit = run_benchmark(
        profile=standard_profile,
        selection_max_origins=7,
        **common,
    )

    assert standard.predictions_for_split("selection")["target_date"].nunique() == 51
    assert quick.predictions_for_split("selection")["target_date"].nunique() == 3
    assert explicit.predictions_for_split("selection")["target_date"].nunique() == 7
    assert standard.predictions_for_split("holdout")["target_date"].nunique() == 10


def test_conservative_selection_gate_is_paired_adjusted_and_deterministic() -> None:
    predictions = _paired_selection_predictions()
    leaderboard = evaluate_predictions(predictions)

    first_champion, first = select_champion_with_diagnostics(
        leaderboard,
        predictions,
    )
    second_champion, second = select_champion_with_diagnostics(
        leaderboard,
        predictions,
    )

    assert first_champion == second_champion == "good_model"
    assert_frame_equal(first, second)
    indexed = first.set_index("model")
    assert bool(indexed.loc["good_model", "gate_passed"])
    assert indexed.loc["good_model", "absolute_log_loss_improvement"] >= 0.05
    assert indexed.loc["good_model", "holm_adjusted_p_value"] <= 0.05
    assert "insufficient_log_loss_improvement" in indexed.loc[
        "marginal_model", "gate_reason"
    ]
    assert "brier_degradation" in indexed.loc["brier_bad_model", "gate_reason"]
    assert "fallback_present" in indexed.loc["fallback_model", "gate_reason"]
    assert indexed.loc["good_model", "bootstrap_block_weeks"] == 13
    assert indexed.loc["good_model", "bootstrap_seed"] == 17


def test_conservative_selection_gate_rejects_mismatched_origins() -> None:
    predictions = _paired_selection_predictions()
    mismatched = predictions.drop(
        predictions.loc[predictions["model"] == "good_model"].index[-1]
    )
    leaderboard = evaluate_predictions(mismatched)

    with np.testing.assert_raises_regex(ValueError, "identical target origins"):
        select_champion_with_diagnostics(leaderboard, mismatched)


def test_conservative_selection_gate_rejects_invalid_probabilities() -> None:
    predictions = _paired_selection_predictions()
    leaderboard = evaluate_predictions(predictions)
    corrupted = predictions.copy()
    corrupted.loc[corrupted.index[0], "p_risk_on"] = np.nan

    with np.testing.assert_raises_regex(ValueError, "non-finite probabilities"):
        select_champion_with_diagnostics(leaderboard, corrupted)


def test_conservative_selection_gate_rejects_holdout_rows() -> None:
    predictions = _paired_selection_predictions()
    leaderboard = evaluate_predictions(predictions)
    holdout = predictions.assign(evaluation_split="holdout")

    with np.testing.assert_raises_regex(ValueError, "selection predictions only"):
        select_champion_with_diagnostics(leaderboard, holdout)


def test_time_split_leaderboard_is_holdout_and_default_track_is_holdout_only() -> None:
    features, states = _model_inputs()
    cutoff = features.index[105]
    profile = BenchmarkProfile.quick().with_overrides(
        max_origins=40, minimum_train_weeks=52
    )
    result = run_benchmark(
        features,
        states,
        profile=profile,
        models=("majority", "persistence", "markov"),
        selection_end=cutoff,
        minimum_selection_predictions=5,
        minimum_holdout_predictions=5,
    )

    assert set(result.predictions["evaluation_split"]) == {"selection", "holdout"}
    assert (result.champion_predictions()["target_date"] >= cutoff).all()
    assert (result.champion_predictions(split="selection")["target_date"] < cutoff).all()
    assert len(result.champion_predictions(split="holdout")) == 40
    assert len(result.champion_predictions(split="all")) > 40
    assert len(result.predictions_for_split("selection")["origin_date"].unique()) >= 5
    assert len(result.predictions_for_split("holdout")["origin_date"].unique()) >= 5

    holdout = result.holdout_leaderboard.set_index("model")
    selection = result.selection_leaderboard.set_index("model")
    combined = result.leaderboard.set_index("model")
    np.testing.assert_allclose(
        combined["log_loss"], holdout.reindex(combined.index)["log_loss"]
    )
    np.testing.assert_allclose(
        combined["selection_log_loss"],
        selection.reindex(combined.index)["log_loss"],
    )
    assert (result.leaderboard["evaluation_split"] == "holdout").all()


def test_time_split_fails_loudly_when_holdout_is_too_short() -> None:
    features, states = _model_inputs()
    cutoff = features.index[-2]
    profile = BenchmarkProfile.quick().with_overrides(
        max_origins=20, minimum_train_weeks=52
    )

    with np.testing.assert_raises_regex(ValueError, "insufficient holdout"):
        run_benchmark(
            features,
            states,
            profile=profile,
            models=("majority", "persistence", "markov"),
            selection_end=cutoff,
            minimum_selection_predictions=5,
            minimum_holdout_predictions=5,
        )


def test_live_quick_split_succeeds_with_three_origins_per_side() -> None:
    features, states = _model_inputs()
    cutoff = features.index[105]
    result = run_benchmark(
        features,
        states,
        profile=BenchmarkProfile.quick().with_overrides(minimum_train_weeks=52),
        models=("majority", "persistence", "markov"),
        selection_end=cutoff,
        minimum_selection_predictions=3,
        minimum_holdout_predictions=3,
    )

    assert result.predictions_for_split("selection")["origin_date"].nunique() == 3
    assert result.predictions_for_split("holdout")["origin_date"].nunique() == 10
    assert not result.champion_holdout_predictions().empty


def test_walk_forward_progress_reports_first_last_and_five_percent_steps() -> None:
    features, states = _model_inputs()
    messages: list[str] = []
    profile = BenchmarkProfile.quick().with_overrides(
        max_origins=40, minimum_train_weeks=52
    )

    run_benchmark(
        features,
        states,
        profile=profile,
        models=("persistence",),
        progress=messages.append,
    )

    assert 20 <= len(messages) <= 21
    assert messages[0].startswith("walk-forward 1/40: ")
    assert messages[-1].startswith("walk-forward 40/40: ")
    assert messages[0].endswith(
        f"{features.index[-41].date().isoformat()} → "
        f"{features.index[-40].date().isoformat()}"
    )
    assert messages[-1].endswith(
        f"{features.index[-2].date().isoformat()} → "
        f"{features.index[-1].date().isoformat()}"
    )


def test_transition_metrics_mean_any_change_from_current_state() -> None:
    current = ["risk_on", "risk_on", "risk_off", "transition"]
    actual = ["transition", "risk_on", "risk_on", "risk_off"]
    predicted = ["transition", "risk_off", "risk_on", "risk_off"]
    rows = []
    for current_state, actual_state, predicted_state in zip(
        current, actual, predicted, strict=True
    ):
        probability = {state: 1e-9 for state in STATE_ORDER}
        probability[predicted_state] = 1.0 - 2e-9
        rows.append(
            {
                "model": "markov",
                "current_state": current_state,
                "actual": actual_state,
                "predicted": predicted_state,
                **{f"p_{state}": probability[state] for state in STATE_ORDER},
                "fallback": False,
            }
        )

    metrics = evaluate_predictions(pd.DataFrame(rows)).iloc[0]
    assert np.isclose(metrics["transition_precision"], 0.75)
    assert np.isclose(metrics["transition_recall"], 1.0)
    assert np.isclose(metrics["transition_state_precision"], 1.0)
    assert np.isclose(metrics["transition_state_recall"], 1.0)
