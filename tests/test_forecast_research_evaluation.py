"""Reject false improvement caused by missing weeks, changed labels, or leakage."""
import numpy as np
import pandas as pd
import pytest
from datetime import timedelta

from regime_lab.analysis.forecast_research_evaluation import (
    match_forecasts, paired_comparisons, per_week_scores, summarize_scores,
)


@pytest.fixture
def evidence():
    dates = pd.date_range("2022-09-02", periods=31, freq="7D", tz="UTC")
    states = pd.Series(["risk_on", "transition", "risk_off", "transition", "risk_on"] * 6 + ["risk_on"], index=dates)
    names = ["risk_on", "transition", "risk_off"]
    rows = []
    for index, date in enumerate(dates[:-1]):
        actual, current = states.iloc[index + 1], states.iloc[index]
        probability = np.full(3, .15)
        probability[names.index(actual)] = .7
        rows.append({"origin_date": date, "target_date": dates[index+1], "model": "causal_dynamic_ensemble",
                     "current_state": current, "actual": actual,
                     "evaluation_split": "selection" if index < 15 else "holdout",
                     **dict(zip([f"p_{name}" for name in names], probability))})
    baseline = pd.DataFrame(rows)
    candidate = baseline.copy()
    candidate["model"] = "new_model"
    candidate["last_train_target"] = [date.to_pydatetime() - timedelta(days=7) for date in candidate.origin_date]
    candidate["fallback"] = False
    for index, row in candidate.iterrows():
        for name in names:
            candidate.loc[index, f"p_{name}"] = .9 if name == row.actual else .05
    return baseline, candidate, states


def test_matched_scores_and_paired_uncertainty(evidence):
    scored = per_week_scores(match_forecasts(*evidence))
    summary = summarize_scores(scored)
    assert set(summary.weeks) == {15}
    comparisons = paired_comparisons(scored, ["new_model"], ["causal_dynamic_ensemble"], resamples=499)
    assert (comparisons.loss_ci_high < 0).all()
    assert (comparisons.brier_ci_high < 0).all()
    assert (comparisons.log_loss_holm_p_value < .05).all()


@pytest.mark.parametrize("change", ["drop", "duplicate", "actual", "split", "same_origin_training", "nan", "probability_sum", "missing_training"])
def test_comparison_rejects_invalid_evidence(evidence, change):
    baseline, candidate, states = evidence
    if change == "drop":
        candidate = candidate.iloc[1:]
    elif change == "duplicate":
        candidate = pd.concat([candidate, candidate.iloc[[0]]])
    elif change == "actual":
        candidate.loc[0, "actual"] = "risk_on"
    elif change == "split":
        candidate.loc[0, "evaluation_split"] = "holdout"
    elif change == "same_origin_training":
        candidate.loc[0, "last_train_target"] = candidate.loc[0, "origin_date"]
    elif change == "nan":
        candidate.loc[0, "p_risk_on"] = np.nan
    elif change == "probability_sum":
        candidate.loc[0, "p_risk_on"] += .1
    else:
        candidate.loc[0, "last_train_target"] = pd.NaT
    with pytest.raises(ValueError):
        match_forecasts(baseline, candidate, states)


def test_argmax_change_does_not_use_half_departure_threshold(evidence):
    baseline, candidate, states = evidence
    # Current risk-on, actual transition. Departure mass exceeds half, yet
    # risk-on remains the most likely state: the established argmax metric misses.
    candidate.loc[0, ["p_risk_on", "p_transition", "p_risk_off"]] = [.4, .35, .25]
    scored = per_week_scores(match_forecasts(baseline, candidate, states))
    row = scored.loc[scored.model.eq("new_model")].iloc[0]
    assert row.departure_probability == pytest.approx(.6)
    assert not row.departure_hit


def test_warmup_keeps_identical_week_coverage(evidence):
    baseline, candidate, states = evidence
    candidate.loc[0, "fallback"] = True
    candidate.loc[0, "fallback_reason"] = "insufficient_completed_oos"
    candidate.loc[0, "last_train_target"] = pd.NaT
    summary = summarize_scores(per_week_scores(match_forecasts(baseline, candidate, states)))
    row = summary.loc[summary.model.eq("new_model") & summary.period.eq("selection_2016_2022")].iloc[0]
    assert row.weeks == 15
    assert row.warmup_rows == 1
