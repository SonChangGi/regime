from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import regime_lab.analysis.directional as directional


def test_first_departure_and_reconciliation_contract() -> None:
    index = pd.date_range("2020-01-03", periods=7, freq="W-FRI")
    states = pd.Series(
        ["risk_on", "risk_on", "transition", "risk_off", "risk_off", "risk_on", "risk_on"],
        index=index,
    )
    targets = directional.first_departure_targets(states, 3)
    assert targets.loc[0, "outcome"] == "transition"
    assert targets.loc[3, "outcome"] == "risk_on"

    probability = directional.markov_first_passage_probabilities(
        states.iloc[:6], "risk_on", 3
    )
    assert probability["risk_on"] == 0.0
    assert sum(probability.values()) == pytest.approx(1.0)

    result = directional.reconcile_directional_risk(0.37, "risk_on", probability)
    assert result["first_destination"]["risk_on"] == 0.0
    assert sum(result["first_destination"].values()) == pytest.approx(0.37)
    assert result["no_departure"] == pytest.approx(0.63)


def test_benchmark_purges_horizon_and_returns_probability_simplex() -> None:
    index = pd.date_range("2020-01-03", periods=120, freq="W-FRI")
    values = np.where(
        np.arange(len(index)) % 29 < 15,
        "risk_on",
        np.where(np.arange(len(index)) % 29 < 20, "transition", "risk_off"),
    )
    states = pd.Series(values, index=index)
    position = np.arange(len(index), dtype=float)
    features = pd.DataFrame(
        {
            "trend": np.sin(position / 6.0),
            "stress": np.cos(position / 9.0),
        },
        index=index,
    )
    result = directional.run_directional_transition_benchmark(
        features,
        states,
        horizons=(1, 4),
        models=(
            "empirical_first_passage",
            "markov_first_passage",
            "regularized_multinomial",
        ),
        minimum_train_weeks=20,
        selection_end=index[70],
        minimum_selection_predictions=8,
        minimum_diagnostic_predictions=8,
        maximum_diagnostic_origins=12,
        selection_max_origins=20,
    )
    probability_columns = [
        f"p_{name}" for name in directional.OUTCOME_ORDER
    ]
    np.testing.assert_allclose(
        result.predictions[probability_columns].sum(axis=1), 1.0, atol=1e-10
    )
    for _, row in result.predictions.iterrows():
        assert row[f"p_{row['current_state']}"] == 0.0
    assert (
        result.split_audit["last_train_target_end"]
        < result.split_audit["origin_date"]
    ).all()
    assert set(result.champions_by_horizon) == {1, 4}


def test_directional_score_matches_deployed_conditional_destination() -> None:
    rows = []
    for model, no_departure, transition, risk_off in (
        ("low_hazard", 0.9, 0.08, 0.02),
        ("high_hazard", 0.1, 0.72, 0.18),
    ):
        rows.append(
            {
                "horizon_weeks": 4,
                "evaluation_split": "selection",
                "model": model,
                "origin_date": pd.Timestamp("2020-01-03"),
                "current_state": "risk_on",
                "actual_outcome": "transition",
                "actual_change": True,
                "p_no_departure": no_departure,
                "p_risk_on": 0.0,
                "p_transition": transition,
                "p_risk_off": risk_off,
                "fallback": False,
            }
        )

    leaderboard = directional._evaluate(pd.DataFrame(rows))

    assert set(leaderboard["score_target"]) == {
        "first_destination_given_departure"
    }
    assert leaderboard["log_loss"].to_numpy() == pytest.approx(
        [leaderboard.iloc[0]["log_loss"]] * 2
    )
    assert leaderboard["brier"].to_numpy() == pytest.approx(
        [leaderboard.iloc[0]["brier"]] * 2
    )


def test_zero_event_selection_fails_closed_to_empirical_baseline() -> None:
    rows = []
    models = (
        "empirical_first_passage",
        "markov_first_passage",
        "regularized_multinomial",
    )
    for position, date in enumerate(pd.date_range("2020-01-03", periods=12, freq="W-FRI")):
        for model in models:
            rows.append(
                {
                    "horizon_weeks": 4,
                    "evaluation_split": "selection",
                    "model": model,
                    "origin_date": date,
                    "current_state": "risk_on",
                    "actual_outcome": "no_departure",
                    "actual_change": False,
                    "p_no_departure": 0.8,
                    "p_risk_on": 0.0,
                    "p_transition": 0.15,
                    "p_risk_off": 0.05,
                    "fallback": False,
                }
            )

    champion, diagnostics = directional._select_horizon(
        pd.DataFrame(rows),
        4,
        minimum_selection_events=8,
        minimum_destination_classes=2,
        minimum_event_blocks=3,
    )

    assert champion == "empirical_first_passage"
    assert all(not row["gate_passed"] for row in diagnostics)
    assert all("insufficient_departure_events" in row["gate_reason"] for row in diagnostics)
    assert all(row["holm_adjusted_p_value"] is None for row in diagnostics)


def test_champion_is_selected_for_deployed_direction_not_raw_departure_mass() -> None:
    event_positions = {0, 4, 8, 13, 17, 21, 26, 30, 34, 39, 43, 47}
    rows = []
    for position, date in enumerate(pd.date_range("2020-01-03", periods=52, freq="W-FRI")):
        departed = position in event_positions
        destination = "transition" if position % 2 == 0 else "risk_off"
        for model in ("empirical_first_passage", "regularized_multinomial"):
            if model == "empirical_first_passage":
                no_departure = 0.99 if not departed else 0.80
                actual_destination = 0.11
                other_destination = 0.09
            else:
                no_departure = 0.01
                actual_destination = 0.9405
                other_destination = 0.0495
            transition = (
                actual_destination if destination == "transition" else other_destination
            )
            risk_off = (
                actual_destination if destination == "risk_off" else other_destination
            )
            rows.append(
                {
                    "horizon_weeks": 4,
                    "evaluation_split": "selection",
                    "model": model,
                    "origin_date": date,
                    "current_state": "risk_on",
                    "actual_outcome": destination if departed else "no_departure",
                    "actual_change": departed,
                    "p_no_departure": no_departure,
                    "p_risk_on": 0.0,
                    "p_transition": transition,
                    "p_risk_off": risk_off,
                    "fallback": False,
                }
            )
    frame = pd.DataFrame(rows)
    raw_losses = {}
    for model, group in frame.groupby("model"):
        actual_probability = np.where(
            group["actual_change"].to_numpy(),
            np.where(
                group["actual_outcome"].eq("transition").to_numpy(),
                group["p_transition"].to_numpy(),
                group["p_risk_off"].to_numpy(),
            ),
            group["p_no_departure"].to_numpy(),
        )
        raw_losses[model] = float(-np.log(actual_probability).mean())

    champion, diagnostics = directional._select_horizon(
        frame,
        4,
        minimum_selection_events=8,
        minimum_destination_classes=2,
        minimum_event_blocks=3,
    )

    assert raw_losses["empirical_first_passage"] < raw_losses["regularized_multinomial"]
    assert champion == "regularized_multinomial"
    selected = next(row for row in diagnostics if row["selected"])
    assert selected["score_target"] == "first_destination_given_departure"
    assert selected["gate_passed"] is True
