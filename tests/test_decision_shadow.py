from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from regime_lab.analysis.decision_shadow import (
    _current_signal_contract,
    _metrics,
    _run_self_financing_strategy,
    build_decision_shadow,
    split_safe_price_only_return_frames,
)


FORECAST_MODEL = "causal_dynamic_ensemble"


def _model_forecast(
    origin: pd.Timestamp,
    probabilities: dict[str, float] | None = None,
) -> dict[str, object]:
    return {
        "model": FORECAST_MODEL,
        "date": (origin + timedelta(days=7)).date().isoformat(),
        "probabilities": probabilities
        or {"risk_on": 0.2, "transition": 0.7, "risk_off": 0.1},
    }


def test_decision_shadow_has_matched_costed_benchmarks_and_separate_evidence_tracks() -> None:
    index = pd.date_range("2020-01-03", periods=120, freq="W-FRI", tz="UTC")
    position = np.arange(len(index), dtype=float)
    prices = pd.DataFrame(
        {
            "spy_close": 100.0
            * np.exp(np.cumsum(0.002 + 0.01 * np.sin(position / 9))),
            "tlt_close": 100.0
            * np.exp(np.cumsum(0.001 - 0.006 * np.sin(position / 9))),
        },
        index=index,
    )
    prices["spy_raw_open"] = prices["spy_close"] / np.exp(
        0.001 + 0.004 * np.sin(position / 7)
    )
    prices["tlt_raw_open"] = prices["tlt_close"] / np.exp(
        0.0005 - 0.003 * np.sin(position / 7)
    )
    for asset in ("spy", "tlt"):
        prices[f"{asset}_raw_close"] = prices[f"{asset}_close"]
        prices[f"{asset}_dividend_amount"] = 0.0
    weekly = []
    for offset, origin in enumerate(index[20:-2]):
        if offset % 2:
            probabilities = {"risk_on": 0.7, "transition": 0.2, "risk_off": 0.1}
        else:
            probabilities = {"risk_on": 0.1, "transition": 0.2, "risk_off": 0.7}
        weekly.append(
            {
                "date": origin.date().isoformat(),
                "next_week": {
                    "date": (origin + timedelta(days=7)).date().isoformat(),
                    "probabilities": probabilities,
                },
                "model_forecasts": [_model_forecast(origin)],
            }
        )
    # Public forecast probabilities are rounded to 8 decimals.  Their displayed
    # values can differ from exactly one by one final decimal place while still
    # satisfying the v5 public contract.
    weekly[0]["model_forecasts"][0]["probabilities"] = {
        "risk_on": 0.33333334,
        "transition": 0.33333334,
        "risk_off": 0.33333333,
    }

    result = build_decision_shadow(
        weekly,
        prices,
        forecast_model=FORECAST_MODEL,
    )

    historical = result["historical_reconstructed_shadow"]
    prospective = result["prospective_ledger"]
    strategies = historical["strategies"]
    assert result["schema_version"] == "regime-prospective-decision-shadow/2"
    assert historical["evidence_track"] == "reconstructed_oos"
    assert prospective["evidence_track"] == "operational_oos"
    assert prospective["affects_official_forecast"] is False
    assert prospective["affects_champion_selection"] is False
    assert set(strategies) == {
        "probability_shadow",
        "spy_buy_and_hold",
        "static_60_40",
        "vol_target_60_40",
    }
    assert {row["weeks"] for row in strategies.values()} == {
        strategies["probability_shadow"]["weeks"]
    }
    assert result["execution_contract"] == {
        "signal_origin": "completed_weekly_close",
        "first_tradable_point": "next_week_adjusted_open",
        "target_return_window": "next_week_open_to_close",
        "rebalance_frequency": "weekly",
        "late_signal_policy": "no_trade",
        "holding_period_weeks": 1,
    }
    assert historical["first_tradable_week"] == index[21].date().isoformat()
    assert historical["evaluation_start_week"] == index[21].date().isoformat()
    assert historical["evaluation_end_week"] == index[-2].date().isoformat()
    assert result["current_signal"] == {
        "origin_date": index[-3].date().isoformat(),
        "target_week": index[-2].date().isoformat(),
        "scheduled_entry_at": "2022-04-04T09:30:00-04:00",
        "decision_at": "2022-04-01T20:00:00+00:00",
        "forecast_model": FORECAST_MODEL,
        "status": "scheduled",
        "action": "trade_at_scheduled_open",
    }
    allocation = historical["allocation_policy"]
    latest_probabilities = weekly[-1]["model_forecasts"][0]["probabilities"]
    expected_spy = (
        0.8 * latest_probabilities["risk_on"]
        + 0.5 * latest_probabilities["transition"]
        + 0.2 * latest_probabilities["risk_off"]
    )
    assert allocation["forecast_model"] == FORECAST_MODEL
    assert allocation["latest_signal_origin"] == index[-3].date().isoformat()
    np.testing.assert_allclose(
        [
            allocation["latest_target_weights"]["SPY"],
            allocation["latest_target_weights"]["TLT"],
        ],
        [expected_spy, 1.0 - expected_spy],
    )
    assert historical["latest_target_weights"] == allocation["latest_target_weights"]
    shadow = strategies["probability_shadow"]
    assert shadow["annualized_turnover"] > 0.0
    assert shadow["transaction_cost_rate_sum"] > 0.0
    assert shadow["cumulative_return"] <= shadow["gross_cumulative_return"]
    initial_only_annualized_turnover = 52.1775 / shadow["weeks"]
    assert (
        strategies["static_60_40"]["annualized_turnover"]
        > initial_only_annualized_turnover
    )
    for field in (
        "sharpe",
        "certainty_equivalent_return",
        "maximum_drawdown",
    ):
        assert shadow[field] is not None


def test_current_signal_uses_strict_scheduled_open_boundary() -> None:
    origin = pd.Timestamp("2026-03-06")
    latest = {
        "date": "2026-03-06",
        "next_week": {"date": "2026-03-13"},
        "model_forecasts": [_model_forecast(origin)],
    }

    before = _current_signal_contract(
        latest,
        decision_at="2026-03-09T09:29:00-04:00",
        forecast_model=FORECAST_MODEL,
    )
    after = _current_signal_contract(
        latest,
        decision_at="2026-03-09T09:31:00-04:00",
        forecast_model=FORECAST_MODEL,
    )
    equal = _current_signal_contract(
        latest,
        decision_at="2026-03-09T09:30:00-04:00",
        forecast_model=FORECAST_MODEL,
    )

    assert before["scheduled_entry_at"] == "2026-03-09T09:30:00-04:00"
    assert before["status"] == "scheduled"
    assert before["action"] == "trade_at_scheduled_open"
    assert equal["status"] == "missed_entry"
    assert equal["action"] == "no_trade"
    assert after["status"] == "missed_entry"
    assert after["action"] == "no_trade"


def test_current_signal_moves_monday_holiday_entry_to_tuesday_open() -> None:
    origin = pd.Timestamp("2025-01-17")
    latest = {
        "date": "2025-01-17",
        "next_week": {"date": "2025-01-24"},
        "model_forecasts": [_model_forecast(origin)],
    }

    signal = _current_signal_contract(
        latest,
        decision_at="2025-01-21T09:29:00-05:00",
        forecast_model=FORECAST_MODEL,
    )

    assert signal["scheduled_entry_at"] == "2025-01-21T09:30:00-05:00"
    assert signal["status"] == "scheduled"


def test_metrics_include_initial_wealth_in_first_week_drawdown() -> None:
    index = pd.date_range("2024-01-05", periods=2, freq="W-FRI", tz="UTC")
    metrics = _metrics(
        pd.Series([-0.2, 0.1], index=index),
        pd.Series([0.0, 0.0], index=index),
        annualization=52.1775,
        risk_aversion=3.0,
    )

    assert np.isclose(metrics["maximum_drawdown"], -0.2)


def test_initial_turnover_is_l1_target_from_zero_asset_weights() -> None:
    index = pd.date_range("2024-01-05", periods=1, freq="W-FRI", tz="UTC")
    gap_relatives = pd.DataFrame({"SPY": [np.nan], "TLT": [np.nan]}, index=index)
    open_to_close = pd.DataFrame({"SPY": [1.0], "TLT": [1.0]}, index=index)
    targets = pd.DataFrame({"SPY": [0.3], "TLT": [0.1]}, index=index)

    path = _run_self_financing_strategy(
        targets,
        gap_relatives,
        open_to_close,
        index,
        cost_rate=0.001,
        initial_allocation_costed_from_cash=True,
        late_signal_policy="no_trade",
    )

    np.testing.assert_allclose(path["turnover"], [0.4])
    np.testing.assert_allclose(path["transaction_cost"], [0.0004])


def test_target_week_dividend_is_excluded_from_both_weight_legs() -> None:
    index = pd.date_range("2024-01-05", periods=2, freq="W-FRI", tz="UTC")
    prices = pd.DataFrame(
        {
            "spy_close": [100.0, 110.0],
            "spy_raw_open": [100.0, 100.0],
            "spy_raw_close": [100.0, 100.0],
            "spy_dividend_amount": [0.0, 10.0],
            "tlt_close": [100.0, 100.0],
            "tlt_raw_open": [100.0, 100.0],
            "tlt_raw_close": [100.0, 100.0],
            "tlt_dividend_amount": [0.0, 0.0],
        },
        index=index,
    )
    gap_relatives, open_to_close = split_safe_price_only_return_frames(prices)
    targets = pd.DataFrame(
        {"SPY": [1.0, 0.0], "TLT": [0.0, 1.0]},
        index=index,
    )

    path = _run_self_financing_strategy(
        targets,
        gap_relatives,
        open_to_close,
        index,
        cost_rate=0.0,
        initial_allocation_costed_from_cash=True,
        late_signal_policy="no_trade",
    )

    np.testing.assert_allclose(gap_relatives.iloc[1], [1.0, 1.0])
    np.testing.assert_allclose(open_to_close.iloc[1], [1.0, 1.0])
    np.testing.assert_allclose(path["gross_returns"], [0.0, 0.0], atol=1e-12)


def test_self_financing_engine_uses_gap_drift_for_turnover_and_cost() -> None:
    index = pd.date_range("2024-01-05", periods=2, freq="W-FRI", tz="UTC")
    gap_relatives = pd.DataFrame(
        {"SPY": [np.nan, 1.1], "TLT": [np.nan, 1.0]}, index=index
    )
    open_to_close = pd.DataFrame(
        {"SPY": [1.1, 1.0], "TLT": [1.0, 1.1]}, index=index
    )
    targets = pd.DataFrame(
        {"SPY": [0.5, 0.25], "TLT": [0.5, 0.75]}, index=index
    )

    path = _run_self_financing_strategy(
        targets,
        gap_relatives,
        open_to_close,
        index,
        cost_rate=0.001,
        initial_allocation_costed_from_cash=True,
        late_signal_policy="no_trade",
    )

    first_intraday_factor = 0.5 * 1.1 + 0.5
    first_close_weights = np.array([0.5 * 1.1, 0.5]) / first_intraday_factor
    second_gap_relatives = np.array([1.1, 1.0])
    gap_factor = float(np.dot(first_close_weights, second_gap_relatives))
    second_pretrade = first_close_weights * second_gap_relatives / gap_factor
    second_turnover = float(np.abs(np.array([0.25, 0.75]) - second_pretrade).sum())
    second_intraday_factor = 0.25 + 0.75 * 1.1

    np.testing.assert_allclose(path["turnover"], [1.0, second_turnover])
    np.testing.assert_allclose(
        path["pretrade_weights"].iloc[1].to_numpy(), second_pretrade
    )
    np.testing.assert_allclose(
        path["gross_returns"],
        [first_intraday_factor - 1.0, gap_factor * second_intraday_factor - 1.0],
    )
    np.testing.assert_allclose(
        path["net_returns"],
        [
            (1.0 - 0.001) * first_intraday_factor - 1.0,
            gap_factor
            * (1.0 - 0.001 * second_turnover)
            * second_intraday_factor
            - 1.0,
        ],
    )


def test_self_financing_engine_holds_drifted_weights_when_signal_is_missing() -> None:
    index = pd.date_range("2024-01-05", periods=2, freq="W-FRI", tz="UTC")
    gap_relatives = pd.DataFrame(
        {"SPY": [np.nan, 1.1], "TLT": [np.nan, 1.05]}, index=index
    )
    open_to_close = pd.DataFrame(
        {"SPY": [1.05, 115.0 / 110.0], "TLT": [1.0, 110.0 / 105.0]},
        index=index,
    )
    targets = pd.DataFrame(
        {"SPY": [0.6, np.nan], "TLT": [0.4, np.nan]}, index=index
    )

    path = _run_self_financing_strategy(
        targets,
        gap_relatives,
        open_to_close,
        index,
        cost_rate=0.001,
        initial_allocation_costed_from_cash=True,
        late_signal_policy="no_trade",
    )

    assert path["turnover"].iloc[1] == 0.0
    np.testing.assert_allclose(
        path["target_weights"].iloc[1], path["pretrade_weights"].iloc[1]
    )
