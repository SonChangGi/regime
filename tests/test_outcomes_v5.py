from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lab.analysis.outcomes import (
    ASSETS,
    HORIZONS,
    OUTCOME_COLUMNS,
    POINT_METRICS,
    _episode_bounded_circular_blocks,
    build_conditional_asset_statistics,
    build_forward_outcomes,
    summarize_conditional_outcomes,
)


def _weekly_inputs(periods: int = 40) -> tuple[pd.DataFrame, pd.Series]:
    dates = pd.date_range("2020-01-03", periods=periods, freq="7D")
    trend = 100.0 * np.power(1.01, np.arange(periods))
    prices = pd.DataFrame(
        {f"{asset.lower()}_close": trend * (1.0 + index / 10.0)
         for index, asset in enumerate(ASSETS)},
        index=dates,
    )
    states = pd.Series(
        (["risk_on"] * 5 + ["transition"] * 3 + ["risk_off"] * 4)
        * (periods // 12)
        + ["risk_on"] * (periods % 12),
        index=dates,
        dtype="object",
    )
    return prices, states


def _manual_outcomes(
    returns: list[float],
    *,
    episode_ids: list[int],
) -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=len(returns), freq="7D")
    rows = []
    for position, (value, episode_id) in enumerate(
        zip(returns, episode_ids, strict=True)
    ):
        rows.append(
            {
                "origin_position": position,
                "origin_date": dates[position],
                "entry_date": dates[position] + pd.DateOffset(weeks=1),
                "exit_date": dates[position] + pd.DateOffset(weeks=2),
                "state": "risk_on",
                "episode_id": episode_id,
                "asset": "SPY",
                "horizon_weeks": 1,
                "execution_lag_weeks": 1,
                "return_currency": "USD",
                "forward_return": value,
                "max_drawdown": min(value, 0.0),
            }
        )
    return pd.DataFrame(rows, columns=OUTCOME_COLUMNS)


def test_forward_outcome_uses_t_plus_one_entry_and_h_week_exit() -> None:
    prices, states = _weekly_inputs(20)

    outcomes = build_forward_outcomes(prices, states, horizons=(1,))
    first = outcomes.loc[
        outcomes["origin_position"].eq(0) & outcomes["asset"].eq("SPY")
    ].iloc[0]

    assert first["entry_date"] == states.index[1]
    assert first["exit_date"] == states.index[2]
    assert first["forward_return"] == pytest.approx(
        prices["spy_close"].iloc[2] / prices["spy_close"].iloc[1] - 1.0
    )
    assert outcomes["origin_position"].max() == 17


def test_outcome_requires_a_complete_positive_price_path() -> None:
    prices, states = _weekly_inputs(20)
    prices.loc[states.index[1], "spy_close"] = np.nan

    outcomes = build_forward_outcomes(prices, states, horizons=(1,))

    missing = outcomes.loc[
        outcomes["origin_position"].eq(0) & outcomes["asset"].eq("SPY")
    ]
    control = outcomes.loc[
        outcomes["origin_position"].eq(0) & outcomes["asset"].eq("QQQ")
    ]
    assert missing.empty
    assert len(control) == 1


def test_within_window_max_drawdown_uses_the_full_holding_path() -> None:
    prices, states = _weekly_inputs(20)
    prices.loc[states.index[1:6], "spy_close"] = [100.0, 120.0, 90.0, 95.0, 110.0]

    outcomes = build_forward_outcomes(prices, states, horizons=(4,))
    first = outcomes.loc[
        outcomes["origin_position"].eq(0) & outcomes["asset"].eq("SPY")
    ].iloc[0]

    assert first["max_drawdown"] == pytest.approx(-0.25)
    assert first["forward_return"] == pytest.approx(0.10)


def test_conditional_point_statistics_and_unique_episodes() -> None:
    outcomes = _manual_outcomes(
        [-0.20, -0.10, 0.10, 0.20], episode_ids=[0, 0, 1, 1]
    )

    statistics = summarize_conditional_outcomes(
        outcomes,
        min_observations=1,
        min_unique_episodes=1,
        bootstrap_resamples=0,
    )
    row = statistics.loc[
        statistics["state"].eq("risk_on")
        & statistics["asset"].eq("SPY")
        & statistics["horizon_weeks"].eq(1)
    ].iloc[0]

    assert row["n"] == 4
    assert row["unique_episodes"] == 2
    assert row["mean_return"] == pytest.approx(0.0)
    assert row["median_return"] == pytest.approx(0.0)
    assert row["positive_rate"] == pytest.approx(0.5)
    assert row["cvar_5"] == pytest.approx(-0.20)
    assert row["downside_volatility"] == pytest.approx(
        np.sqrt((0.20**2 + 0.10**2) / 4) * np.sqrt(52.0)
    )
    # Weekly-origin and episode-equal estimands are both explicit.  With equal
    # episode lengths they coincide; the benchmark covers all states/origins.
    assert row["episode_equal_mean_return"] == pytest.approx(0.0)
    assert row["unconditional_benchmark_mean_return"] == pytest.approx(0.0)
    assert row["excess_mean_return"] == pytest.approx(0.0)
    assert row["episode_bootstrap_method"] == "whole_episode_resampling"


def test_episode_equal_statistic_does_not_overweight_a_long_episode() -> None:
    outcomes = _manual_outcomes(
        [0.10, 0.10, 0.10, -0.50], episode_ids=[0, 0, 0, 1]
    )

    statistics = summarize_conditional_outcomes(
        outcomes,
        min_observations=1,
        min_unique_episodes=1,
        bootstrap_resamples=99,
    )
    row = statistics.loc[
        statistics["state"].eq("risk_on")
        & statistics["asset"].eq("SPY")
        & statistics["horizon_weeks"].eq(1)
    ].iloc[0]

    assert row["mean_return"] == pytest.approx(-0.05)
    assert row["episode_equal_mean_return"] == pytest.approx(-0.20)
    assert np.isfinite(row["episode_equal_mean_return_ci95_lower"])


def test_episode_bounded_block_bootstrap_is_deterministic() -> None:
    returns = np.linspace(-0.15, 0.20, 30).tolist()
    episodes = np.repeat(np.arange(6), 5).tolist()
    outcomes = _manual_outcomes(returns, episode_ids=episodes)
    kwargs = {
        "min_observations": 20,
        "min_unique_episodes": 5,
        "bootstrap_block_weeks": 3,
        "bootstrap_resamples": 99,
        "bootstrap_seed": 17,
    }

    first = summarize_conditional_outcomes(outcomes, **kwargs)
    second = summarize_conditional_outcomes(outcomes, **kwargs)
    columns = [
        column
        for column in first.columns
        if "_ci95_" in column or column == "bootstrap_seed"
    ]

    pd.testing.assert_frame_equal(first[columns], second[columns])
    row = first.loc[
        first["state"].eq("risk_on")
        & first["asset"].eq("SPY")
        & first["horizon_weeks"].eq(1)
    ].iloc[0]
    assert row["status"] == "ok"
    assert row["bootstrap_method"] == "episode_bounded_circular_block"
    assert np.isfinite(row["mean_return_ci95_lower"])
    assert row["mean_return_ci95_lower"] <= row["mean_return_ci95_upper"]


def test_episode_bounded_blocks_give_every_origin_equal_weight() -> None:
    outcomes = _manual_outcomes(
        [0.10, 0.20, 0.30, -0.50],
        episode_ids=[0, 0, 0, 1],
    )

    blocks = _episode_bounded_circular_blocks(outcomes, block_length=4)

    assert len(blocks) == len(outcomes)
    assert {len(block) for block in blocks} == {4}
    assert all(block["episode_id"].nunique() == 1 for block in blocks)
    pooled = pd.concat(blocks, ignore_index=True)
    assert pooled["origin_position"].value_counts().to_dict() == {
        position: 4 for position in range(4)
    }


def test_bootstrap_keeps_weekly_origin_weighting_across_unequal_episodes() -> None:
    outcomes = _manual_outcomes(
        [0.10] * 30 + [-0.50] * 5,
        episode_ids=[0] * 30 + [1, 2, 3, 4, 5],
    )

    statistics = summarize_conditional_outcomes(
        outcomes,
        min_observations=20,
        min_unique_episodes=5,
        bootstrap_block_weeks=13,
        bootstrap_resamples=1_999,
        bootstrap_seed=17,
    )
    row = statistics.loc[
        statistics["state"].eq("risk_on")
        & statistics["asset"].eq("SPY")
        & statistics["horizon_weeks"].eq(1)
    ].iloc[0]

    assert row["mean_return"] == pytest.approx((30 * 0.10 - 5 * 0.50) / 35)
    for metric in POINT_METRICS:
        assert row[f"{metric}_ci95_lower"] <= row[metric]
        assert row[metric] <= row[f"{metric}_ci95_upper"]


def test_generated_statistics_keep_supported_points_inside_ci() -> None:
    prices, states = _weekly_inputs(80)

    statistics = build_conditional_asset_statistics(
        prices,
        states,
        min_observations=1,
        min_unique_episodes=1,
        bootstrap_resamples=199,
    ).statistics

    supported = statistics.loc[statistics["status"].eq("ok")]
    assert len(supported) == len(ASSETS) * len(HORIZONS) * 3
    for metric in POINT_METRICS:
        assert (
            supported[f"{metric}_ci95_lower"] <= supported[metric]
        ).all()
        assert (
            supported[metric] <= supported[f"{metric}_ci95_upper"]
        ).all()


def test_full_builder_has_fixed_grid_and_no_allocation_fields() -> None:
    prices, states = _weekly_inputs(80)

    result = build_conditional_asset_statistics(
        prices,
        states,
        min_observations=1,
        min_unique_episodes=1,
        bootstrap_resamples=0,
    )

    assert len(result.statistics) == 3 * len(ASSETS) * len(HORIZONS)
    assert set(result.statistics["asset"]) == set(ASSETS)
    assert set(result.statistics["horizon_weeks"]) == set(HORIZONS)
    prohibited = {
        column
        for column in (*result.outcomes.columns, *result.statistics.columns)
        if "allocation" in column.lower() or "weight" in column.lower()
    }
    assert prohibited == set()
    assert set(POINT_METRICS).issubset(result.statistics.columns)


def test_support_gate_keeps_point_estimates_but_withholds_ci() -> None:
    outcomes = _manual_outcomes([0.1, -0.1], episode_ids=[0, 0])

    statistics = summarize_conditional_outcomes(
        outcomes,
        min_observations=3,
        min_unique_episodes=2,
        bootstrap_resamples=99,
    )
    row = statistics.loc[
        statistics["state"].eq("risk_on")
        & statistics["asset"].eq("SPY")
        & statistics["horizon_weeks"].eq(1)
    ].iloc[0]

    assert row["status"] == "insufficient_support"
    assert row["mean_return"] == pytest.approx(0.0)
    assert np.isnan(row["mean_return_ci95_lower"])


def test_forward_outcomes_accept_utc_dst_hour_shift() -> None:
    index = pd.date_range(
        "2026-02-27 16:00",
        periods=20,
        freq="W-FRI",
        tz="America/New_York",
    ).tz_convert("UTC")
    prices, states = _weekly_inputs(20)
    prices.index = index
    states.index = index

    outcomes = build_forward_outcomes(prices, states, horizons=(1,))

    assert not outcomes.empty
    assert outcomes.iloc[0]["entry_date"] == index[1]
