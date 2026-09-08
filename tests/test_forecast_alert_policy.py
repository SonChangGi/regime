"""Behavioral checks for bounded matching and causally enforced alarm budgets."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from regime_lab.research.forecast_alert_policy import (
    AlertPolicy, POLICY, YEAR_WEEKS, alert_policies, bounded_alert_history,
    bounded_metrics, replay_candidate, select_policy,
)
from regime_lab.research.forecast_enhancement_diagnostics import alert_policies as legacy_alert_policies


def source(periods=180):
    index = np.arange(periods)
    dates = pd.date_range("2020-01-03", periods=periods, freq="W-FRI", tz="UTC")
    events = index % 8 == 0
    return pd.DataFrame({"model": "test", "origin_date": dates,
                         "target_date": dates + pd.Timedelta(weeks=1), "current_state": "risk_on",
                         "actual": np.where(events, "transition", "risk_on"),
                         "worsening_probability": np.where(events, .9, .1),
                         "evaluation_split": "retrospective_diagnostic"})


def test_packet_expires_and_later_event_cannot_redeem_continuous_alarm():
    frame = source(100)
    frame.loc[:79, "evaluation_split"] = "selection"
    frame.loc[80:, "worsening_probability"] = .9
    frame.loc[80:98, "actual"] = "risk_on"
    frame.loc[99, "actual"] = "transition"
    config = AlertPolicy(annual_false_week_budget=100)
    history = bounded_alert_history(frame, config)
    tail = history.iloc[80:]
    assert tail.alert.sum() == 1
    assert tail.iloc[0].alert
    assert tail.iloc[0].valid_until == frame.target_date.iloc[80]
    assert tail.iloc[0].matched_event_id is None
    assert tail.iloc[-1].actual and not tail.iloc[-1].alert
    metric = next(r for r in bounded_metrics(history, config) if r["evaluation_split"] == "retrospective_diagnostic")
    assert metric["hit_count"] == 0
    assert metric["false_alert_episodes"] == metric["false_alert_weeks"] == 1
    assert metric["mean_lead_weeks_detected_events"] is None


def test_packet_matching_is_one_to_one_and_has_one_week_lead():
    history = bounded_alert_history(source())
    hits = history.loc[history.alert & history.actual.astype(bool)]
    assert len(hits) > 4
    assert hits.matched_event_id.notna().all() and hits.matched_event_id.is_unique
    assert hits.matched_packet_id.notna().all() and hits.matched_packet_id.is_unique
    assert set(hits.lead_weeks) == {1.}
    assert (hits.valid_until == hits.target_date).all()
    metric = bounded_metrics(history)[0]
    assert metric["hit_count"] == len(hits)
    assert metric["evaluation_start"] == history.origin_date.min()
    assert metric["evaluation_target_end"] == history.target_date.max()


def test_unmatured_truth_cannot_change_policy_budget_or_state():
    frame = source()
    cutoff = frame.origin_date.iloc[130]
    changed = frame.copy()
    changed.loc[changed.target_date >= cutoff, "actual"] = "risk_off"
    first, second = bounded_alert_history(frame), bounded_alert_history(changed)
    derived_truth = ["actual", "event_id", "matched_event_id", "matched_packet_id", "lead_weeks"]
    pd.testing.assert_frame_equal(first.loc[first.origin_date <= cutoff].drop(columns=derived_truth),
                                  second.loc[second.origin_date <= cutoff].drop(columns=derived_truth))
    point = first.loc[first.origin_date == cutoff].iloc[0]
    assert point.last_policy_target < cutoff


def test_full_policy_selection_counts_expiry_cooldown_and_reset():
    # A persistent signal emits one packet, never 156 active warning weeks.
    probabilities = np.full(156, .9)
    truth = np.zeros(156, bool)
    truth[0] = True
    selected = select_policy(probabilities, truth, AlertPolicy())
    assert selected["alert_packets"] == selected["hit_count"] == 1
    assert selected["false_alert_weeks"] == 0
    # The later event does not convert an earlier false packet into a hit.
    truth[0], truth[-1] = False, True
    replay = replay_candidate(probabilities, truth, .5, AlertPolicy())
    assert replay["hit_count"] == 0 and replay["false_alert_weeks"] == 1


def test_budget_reserves_unresolved_packets_and_never_issues_without_capacity():
    frame = source(260)
    # Preserve training support, then deliberately create many false signals.
    frame.loc[90:, "actual"] = "risk_on"
    frame.loc[90:, "worsening_probability"] = np.tile([.95, .1, .1, .1, .95], 34)
    history = bounded_alert_history(frame)
    issued = history.loc[history.alert]
    assert len(issued) > 3
    assert (issued.policy_validation_false_alerts_per_year <= 4 + 1e-12).all()
    assert (issued.budget_settled_false_weeks + issued.budget_pending_reservations + 1 <= issued.budget_allowed_false_weeks).all()
    assert (issued.budget_capacity_remaining >= 0).all()
    assert history.budget_pending_reservations.max() > 0
    zero = bounded_alert_history(frame, AlertPolicy(annual_false_week_budget=0))
    assert not zero.alert.any()


def test_missing_past_packet_outcome_keeps_budget_reserved():
    frame = source(110)
    initial = bounded_alert_history(frame)
    issued_position = int(np.flatnonzero(initial.alert)[0])
    frame.loc[issued_position, "actual"] = None
    after = bounded_alert_history(frame)
    row = after.iloc[issued_position + 5]
    assert row.budget_pending_reservations >= 1


def test_legacy_baseline_numbers_remain_exactly_reproducible():
    frame = source()
    before, after = legacy_alert_policies(frame), alert_policies(frame)
    assert before["history"] == after["history"][:len(before["history"])]
    assert before["latest"] == after["latest"][:len(before["latest"])]
    for old, new in zip(before["metrics"], after["metrics"]):
        assert old == {key: new[key] for key in old}
    assert after["policy"]["schema_version"] == "forecast-alert-policy/2"
    assert after["policy"]["budget_unit"] == "false_alert_weeks_per_year"
    assert after["policy"]["validity_weeks"] == 1
    assert POLICY in {r["policy"] for r in after["metrics"]}


def test_observed_budget_excess_is_not_hidden_in_metrics():
    history = bounded_alert_history(source())
    history["alert"] = True
    history["actual"] = False
    history["matched_event_id"] = None
    metric = bounded_metrics(history)[0]
    assert metric["false_alerts_per_year"] == pytest.approx(YEAR_WEEKS)
    assert metric["false_week_budget_exceeded"] is True
    assert metric["false_episode_budget_exceeded"] is None  # no invented episode budget


@pytest.mark.parametrize("change", ["duplicate", "horizon", "probability"])
def test_rejects_ambiguous_matching_and_invalid_signal(change):
    frame = source(60)
    if change == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif change == "horizon":
        frame.loc[0, "target_date"] += pd.Timedelta(weeks=1)
    else:
        frame.loc[0, "worsening_probability"] = float("nan")
    with pytest.raises(ValueError):
        bounded_alert_history(frame)


def test_policy_changes_do_not_alter_numerical_model_protocol():
    from regime_lab.research.forecast_enhancement_models import EnhancementProtocol
    before = EnhancementProtocol().record()
    assert replace(AlertPolicy(), cooldown_weeks=3).record()["cooldown_weeks"] == 3
    assert EnhancementProtocol().record() == before
