"""Mechanistic probabilities must implement the official causal target exactly."""
import numpy as np
import pandas as pd
import pytest

from regime_lab.analysis.boundary_forecast import (
    MODELS, build_boundary_inputs, next_scores, next_states,
    forecast_boundary_latest, prequential_temperature, run_boundary_walk_forward,
)
from regime_lab.analysis.labels import CausalRegimeLabeler, RegimeLabelConfig
from regime_lab.schema import STATE_ORDER


@pytest.fixture
def market():
    rng = np.random.default_rng(23)
    returns = rng.standard_t(5, 550) * .015 + .001
    index = pd.date_range("2006-01-06", periods=len(returns), freq="W-FRI", tz="UTC")
    frame = pd.DataFrame({"spy_close": 100 * np.exp(returns.cumsum())}, index=index)
    labeler = CausalRegimeLabeler(RegimeLabelConfig(minimum_fit_observations=260)).fit(frame.iloc[:520])
    return frame, labeler


def test_hypothetical_score_matches_full_official_recalculation(market):
    frame, labeler = market
    for position in (521, 529, 548):
        shocks = np.asarray([-.20, -.05, -.01, 0., .01, .05, .20])
        calculated = next_scores(frame.spy_close.iloc[:position+1].to_numpy(), shocks, labeler)
        expected = []
        for shock in shocks:
            future = frame.iloc[:position+2].copy()
            future.iloc[-1, 0] = frame.spy_close.iloc[position] * np.exp(shock)
            expected.append(labeler.score_frame(future).risk_score.iloc[-1])
        np.testing.assert_allclose(calculated, expected, rtol=1e-11, atol=1e-11)


def test_hysteresis_inequalities_match_official_boundaries(market):
    _, labeler = market
    lower, upper = labeler.lower_threshold_, labeler.upper_threshold_
    margin = (upper - lower) * labeler.config.hysteresis_fraction
    assert next_states(np.asarray([lower, upper]), "transition", labeler).tolist() == ["risk_off", "risk_on"]
    assert next_states(np.asarray([lower-margin, upper-margin]), "risk_on", labeler).tolist() == ["risk_off", "risk_on"]
    assert next_states(np.asarray([lower+margin, upper+margin]), "risk_off", labeler).tolist() == ["risk_off", "risk_on"]


def test_inputs_cannot_change_with_future_market_observations(market):
    frame, labeler = market
    states = labeler.transform(frame)
    original = build_boundary_inputs(frame, states)
    changed = frame.copy()
    changed.iloc[535:, 0] *= 1.5
    rebuilt = build_boundary_inputs(changed, labeler.transform(changed))
    pd.testing.assert_frame_equal(original.features.iloc[:535], rebuilt.features.iloc[:535])
    for model in MODELS[:2]:
        pd.testing.assert_frame_equal(original.mechanistic[model].iloc[:535], rebuilt.mechanistic[model].iloc[:535])


def test_refuse_changed_authoritative_states(market):
    frame, labeler = market
    states = labeler.transform(frame)
    states.iloc[-1] = next(state for state in STATE_ORDER if state != states.iloc[-1])
    with pytest.raises(ValueError, match="authoritative"):
        build_boundary_inputs(frame, states)


def test_calibration_excludes_origin_and_future_targets():
    raw = np.asarray([.8, .15, .05])
    history = [(i, raw, i % 3) for i in range(100)]
    p, temperature, count = prequential_temperature(raw, history, 75)
    altered = history[:75] + [(i, np.asarray([.01, .01, .98]), 2) for i in range(75, 150)]
    other, other_temperature, other_count = prequential_temperature(raw, altered, 75)
    np.testing.assert_array_equal(p, other)
    assert (temperature, count) == (other_temperature, other_count) == (temperature, 75)


def test_oos_targets_purged_and_state_aligned(market):
    frame, labeler = market
    states = labeler.transform(frame)
    inputs = build_boundary_inputs(frame, states)
    result = run_boundary_walk_forward(inputs, origin_positions=[521, 522, 523], models=MODELS[:2])
    assert len(result) == 6
    assert (pd.to_datetime(result.last_train_target) < pd.to_datetime(result.origin_date)).all()
    for row in result.itertuples():
        assert row.current_state == states.loc[pd.Timestamp(row.origin_date)]
        assert row.actual == states.loc[pd.Timestamp(row.target_date)]
        assert sum(getattr(row, f"p_{state}") for state in STATE_ORDER) == pytest.approx(1)
    assert not result.fallback.any()


def test_classifier_forecast_ignores_target_and_future_changes(market):
    frame, labeler = market
    original = build_boundary_inputs(frame, labeler.transform(frame))
    changed = frame.copy()
    changed.iloc[535:, 0] *= 1.5
    rebuilt = build_boundary_inputs(changed, labeler.transform(changed))
    first = run_boundary_walk_forward(original, origin_positions=[534], models=MODELS[2:])
    second = run_boundary_walk_forward(rebuilt, origin_positions=[534], models=MODELS[2:])
    for state in STATE_ORDER:
        np.testing.assert_array_equal(first[f"p_{state}"], second[f"p_{state}"])


def test_latest_forecast_has_no_fabricated_actual_and_preserves_history(market):
    frame, labeler = market
    inputs = build_boundary_inputs(frame, labeler.transform(frame))
    history = run_boundary_walk_forward(inputs, origin_positions=[521, 522, 523], models=MODELS[:2])
    before = history.copy(deep=True)
    forecast = forecast_boundary_latest(inputs, history)
    pd.testing.assert_frame_equal(history, before)
    assert forecast["actual"] is None
    assert forecast["evaluation_split"] == "unobserved"
    assert pd.Timestamp(forecast["origin_date"]) == frame.index[-1]
    assert pd.Timestamp(forecast["target_date"]) > frame.index[-1]
    assert pd.Timestamp(forecast["last_train_target"]) < frame.index[-1]
    assert sum(forecast[f"p_{state}"] for state in STATE_ORDER) == pytest.approx(1)


def test_latest_calibration_excludes_target_completed_at_origin(market):
    frame, labeler = market
    inputs = build_boundary_inputs(frame, labeler.transform(frame))
    history = run_boundary_walk_forward(inputs, origin_positions=list(range(521, 549)), models=MODELS[:2])
    first = forecast_boundary_latest(inputs, history)
    modified = history.copy()
    modified.loc[pd.to_datetime(modified.target_date).eq(frame.index[-1]), "actual"] = "tampered_future"
    second = forecast_boundary_latest(inputs, modified)
    assert first == second


def test_zero_volatility_hypothetical_scores_follow_missing_rule(market):
    frame, labeler = market
    history = frame.iloc[:530].copy()
    history.iloc[-53:, 0] = history.iloc[-54, 0]
    actual = next_scores(history.spy_close.to_numpy(), np.asarray([0.]), labeler)
    assert np.isnan(actual[0])
    assert next_states(actual, "risk_on", labeler)[0] == "risk_on"


def test_latest_refuses_unpurged_calibration_evidence(market):
    frame, labeler = market
    inputs = build_boundary_inputs(frame, labeler.transform(frame))
    history = run_boundary_walk_forward(inputs, origin_positions=[521], models=("boundary_filtered_history",))
    history.loc[0, "last_train_target"] = history.loc[0, "origin_date"]
    with pytest.raises(ValueError, match="strictly precede"):
        forecast_boundary_latest(inputs, history)
