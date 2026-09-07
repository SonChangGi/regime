"""Same-origin, frozen-policy and economic evidence contracts."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from regime_lab.analysis.forecast_audit_research import (
    AuditResearchProtocol, budget_threshold, economic_outcomes, episode_validation,
    evaluate_alert_policies, forecast_research_extension, persistence_validation,
    residual_sign_size_diagnostics, run_audit_research, split_for_dates,
    validate_forecast_research_extension,
)
from regime_lab.analysis.forecast_paths import project_multistate_paths
from regime_lab.analysis.labels import CausalRegimeLabeler, RegimeLabelConfig
from regime_lab.schema import STATE_ORDER


def test_calendar_split_purges_crossing_horizon():
    assert split_for_dates(pd.Timestamp("2022-12-30", tz="UTC"), pd.Timestamp("2023-01-06", tz="UTC")) is None
    assert split_for_dates(pd.Timestamp("2023-01-06", tz="UTC"), pd.Timestamp("2023-04-07", tz="UTC")) == "retrospective_diagnostic"


def test_budget_rule_handles_ties_and_budget_exposure():
    p = np.array([.7, .7, .3, .2])
    y = np.array([False, False, True, False])
    threshold = budget_threshold(p, y, annual_budget=0)
    assert threshold > .7
    assert not (p >= threshold).any()
    threshold = budget_threshold(p, y, annual_budget=27)
    assert ((p >= threshold) & ~y).sum() <= 27 * len(p) / 52.1775


def test_alert_policy_does_not_use_current_or_future_outcomes():
    dates = pd.date_range("2021-01-01", periods=140, freq="W-FRI", tz="UTC")
    frame = pd.DataFrame({"model": "test", "origin_date": dates, "target_date": dates + pd.Timedelta(weeks=1),
                          "worsening_probability": np.linspace(.01, .99, len(dates)),
                          "worsening_event": np.arange(len(dates)) % 4 == 0})
    protocol = AuditResearchProtocol()
    first, _ = evaluate_alert_policies(frame, protocol)
    origin = dates[120]
    modified = frame.copy()
    modified.loc[modified.target_date >= origin, "worsening_event"] = ~modified.loc[modified.target_date >= origin, "worsening_event"]
    second, _ = evaluate_alert_policies(modified, protocol)
    left = first.loc[first.origin_date.eq(origin)].drop(columns="event").reset_index(drop=True)
    right = second.loc[second.origin_date.eq(origin)].drop(columns="event").reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)
    assert (first.last_policy_target < first.origin_date).all()


def test_economic_outcomes_are_future_only_and_not_label_dependent():
    dates = pd.date_range("2024-01-05", periods=6, freq="W-FRI", tz="UTC")
    frame = pd.DataFrame({"spy_close": [100, 110, 90, 100, 120, 125]}, index=dates)
    rows = economic_outcomes(frame, horizons=(4,))
    row = rows.iloc[0]
    assert row.forward_return == pytest.approx(.2)
    assert row.minimum_cumulative_return == pytest.approx(-.1)
    assert row.downside_event
    assert row.annualized_realized_volatility == pytest.approx(np.diff(np.log(frame.spy_close.to_numpy()[:5])).std() * np.sqrt(52))
    assert len(rows) == 2  # no fabricated outcome for unfinished horizon


def test_episode_denominator_counts_spells_once_and_marks_censoring():
    dates = pd.date_range("2024-01-05", periods=12, freq="W-FRI", tz="UTC")
    states = pd.Series(["risk_on"] * 4 + ["risk_off"] * 3 + ["transition"] * 3 + ["risk_off"] * 2, index=dates)
    predictions = [{"model": "test", "origin_date": origin.isoformat(), "horizon_weeks": h,
                    "any_risk_off_entry": .6} for origin in dates for h in (1, 4, 13)]
    details, metrics = episode_validation(predictions, states, AuditResearchProtocol())
    assert len(details) == 2
    assert metrics[0]["eligible_episodes"] == metrics[0]["detected_episodes"] == 2
    assert metrics[0]["right_censored_episodes"] == 1
    assert metrics[0]["completed_episodes"] == 1


def test_residual_diagnostic_uses_lagged_scale_and_reports_sample_support():
    dates = pd.date_range("2020-01-03", periods=40, freq="W-FRI", tz="UTC")
    returns = pd.Series(np.sin(np.arange(40)) * .02, index=dates)
    result = residual_sign_size_diagnostics(returns, pd.Series(.02, index=dates))
    assert result["rows"] == 38
    assert sum(group["rows"] for group in result["groups"]) == 38
    assert result["used_to_select_or_tune_candidate"] is False


def extension_fixture():
    origin = pd.Timestamp("2026-03-06T21:00:00Z")
    matrix = np.array([[.8, .15, .05], [.1, .8, .1], [.03, .12, .85]])
    paths = project_multistate_paths(lambda state, age: matrix[STATE_ORDER.index(state)], "risk_off", 3)
    for path in paths:
        path["target_date"] = (origin.tz_convert("America/New_York") + pd.DateOffset(weeks=path["horizon_weeks"])).tz_convert("UTC").isoformat()
    row = {"origin_date": origin.isoformat(), "current_state": "risk_off", "next_state": paths[0]["endpoint"],
           "horizons": {f"{p['horizon_weeks']}w": p for p in paths}}
    return {"schema_version": "regime-forecast-research/1", "data_as_of": origin.isoformat(),
            "selected_model": "test", "automatic_promotion": False,
            "models": [{"id": "test", "label": "Test", "history": [row], "latest": deepcopy(row)}]}


def test_extension_dst_and_distinct_entry_occupancy_are_valid():
    validate_forecast_research_extension(extension_fixture())


@pytest.mark.parametrize("mutation", ["occupancy_alias", "negative_probability", "late_target", "missing_horizon", "future_origin", "duplicate_origin", "nan"])
def test_extension_rejects_semantic_corruption(mutation):
    document = extension_fixture()
    row = document["models"][0]["history"][0]
    path = row["horizons"]["1w"]
    if mutation == "occupancy_alias": path["any_risk_off_entry"] = path["any_risk_off_occupancy"]
    if mutation == "negative_probability": path["endpoint"]["risk_off"] = -.1
    if mutation == "late_target": path["target_date"] = "2026-03-21T20:00:00Z"
    if mutation == "missing_horizon": del row["horizons"]["4w"]
    if mutation == "future_origin": row["origin_date"] = "2027-01-01T21:00:00Z"
    if mutation == "duplicate_origin": document["models"][0]["history"].append(deepcopy(row))
    if mutation == "nan": path["any_risk_off_entry"] = float("nan")
    with pytest.raises(ValueError): validate_forecast_research_extension(document)


@pytest.fixture(scope="module")
def experiment():
    rng = np.random.default_rng(101)
    dates = pd.date_range("2012-01-06", periods=565, freq="W-FRI", tz="UTC")
    canonical = pd.DataFrame({"spy_close": 100 * np.exp(np.cumsum(rng.standard_t(5, len(dates)) * .025))}, index=dates)
    labeler = CausalRegimeLabeler(RegimeLabelConfig(minimum_fit_observations=260)).fit(canonical.iloc[:520])
    states = labeler.transform(canonical)
    rows = []
    for position in range(521, len(states) - 1):
        split = split_for_dates(dates[position], dates[position + 1])
        if split is None: continue
        for model in ("causal_dynamic_ensemble", "recency_weighted_xgboost_208w"):
            rows.append({"model": model, "origin_date": dates[position], "target_date": dates[position + 1],
                         "current_state": states.iloc[position], "actual": states.iloc[position + 1],
                         "evaluation_split": "selection" if split == "selection" else "holdout",
                         "p_risk_on": .3, "p_transition": .4, "p_risk_off": .3})
    baseline = pd.DataFrame(rows)
    result = run_audit_research(canonical, states, baseline, protocol=AuditResearchProtocol(bootstrap_resamples=19))
    return canonical, states, baseline, result


def test_complete_research_produces_matched_scores_and_finite_ui_extension(experiment):
    _, _, baseline, result = experiment
    counts = result.predictions.groupby("model").size()
    assert counts.nunique() == 1 and counts.iloc[0] == len(baseline) // 2
    extension = forecast_research_extension(result.document)
    validate_forecast_research_extension(extension)
    assert extension["automatic_promotion"] is False
    assert len(extension["latest"]) == 4
    assert result.document["protocol"]["diagnostic_hyperparameter_tuning"] is False
    assert result.document["economic_validation"] and result.document["persistence"]
    for _, group in result.path_scores.groupby(["split", "horizon_weeks", "target"]):
        assert group.groupby("model").size().nunique() == 1


def test_shared_missing_origin_is_rejected(experiment):
    canonical, states, baseline, _ = experiment
    removed = baseline.loc[baseline.origin_date.ne(baseline.origin_date.iloc[3])]
    with pytest.raises(ValueError, match="every eligible weekly origin"):
        run_audit_research(canonical, states, removed)
