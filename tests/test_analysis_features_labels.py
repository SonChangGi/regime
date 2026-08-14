from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from regime_lab.analysis import CausalRegimeLabeler, FeatureConfig
from regime_lab.analysis import build_weekly_features
from regime_lab.schema import STATE_ORDER


def _canonical_market_frame(rows: int = 180) -> pd.DataFrame:
    index = pd.date_range("2019-01-04", periods=rows, freq="W-FRI")
    rng = np.random.default_rng(20260811)
    cycle = 0.012 * np.sin(np.arange(rows) / 9.0)
    shocks = rng.normal(0.001, 0.018, rows) + cycle
    spy = 100.0 * np.exp(np.cumsum(shocks))
    qqq = 80.0 * np.exp(np.cumsum(shocks * 1.1 + rng.normal(0, 0.007, rows)))
    macro = 50 + np.cumsum(rng.normal(0.02, 0.15, rows))
    return pd.DataFrame(
        {
            "spy_close": spy,
            "qqq_close": qqq,
            "spy_volume": rng.integers(60_000_000, 130_000_000, rows),
            "qqq_volume": rng.integers(40_000_000, 100_000_000, rows),
            "weekly_economic_index": macro,
        },
        index=index,
    )


def test_features_never_change_when_only_future_values_change() -> None:
    original = _canonical_market_frame()
    revised_future = original.copy()
    cutoff_position = 130
    revised_future.iloc[cutoff_position + 1 :, revised_future.columns.get_loc("spy_close")] *= 4
    revised_future.iloc[
        cutoff_position + 1 :,
        revised_future.columns.get_loc("weekly_economic_index"),
    ] -= 1_000
    revised_future.iloc[
        cutoff_position + 1 :,
        revised_future.columns.get_loc("qqq_volume"),
    ] *= 20

    config = FeatureConfig(
        price_columns=("spy_close", "qqq_close"),
        volume_columns=("spy_volume", "qqq_volume"),
    )
    first = build_weekly_features(original, config)
    second = build_weekly_features(revised_future, config)
    prefix_only = build_weekly_features(original.iloc[: cutoff_position + 1], config)
    assert_frame_equal(first.iloc[: cutoff_position + 1], second.iloc[: cutoff_position + 1])
    assert_frame_equal(first.iloc[: cutoff_position + 1], prefix_only)


def test_market_internal_and_spread_features_have_exact_definitions() -> None:
    index = pd.date_range("2026-01-02", periods=5, freq="W-FRI")
    frame = pd.DataFrame(
        {
            "spy_close": [100.0, 110.0, 99.0, 108.0, 120.0],
            "qqq_close": [100.0, 90.0, 99.0, 90.0, 81.0],
            "xly_close": [100.0, 105.0, 110.0, 115.0, 121.0],
            "xlp_close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "hyg_close": [100.0, 100.0, 101.0, 102.0, 104.0],
            "ief_close": [100.0, 99.0, 98.0, 97.0, 96.0],
            "spy_volume": [100.0, 90.0, 100.0, 110.0, 120.0],
            "qqq_volume": [100.0, 110.0, 100.0, 90.0, 80.0],
        },
        index=index,
    )
    config = FeatureConfig(
        price_columns=(
            "spy_close",
            "qqq_close",
            "xly_close",
            "xlp_close",
            "hyg_close",
            "ief_close",
        ),
        volume_columns=("spy_volume", "qqq_volume"),
        breadth_trend_windows=(4,),
        correlation_windows=(4,),
    )

    features = build_weekly_features(frame, config)
    at = index[-1]
    one_week_equity_returns = np.log(
        np.array([120.0 / 108.0, 81.0 / 90.0, 121.0 / 115.0, 104.0 / 103.0])
    )

    assert features.loc[at, "market_internal__positive_return_share_1w"] == 0.75
    assert features.loc[at, "market_internal__positive_return_share_4w"] == 0.75
    assert features.loc[at, "market_internal__above_trailing_mean_share_4w"] == 0.75
    np.testing.assert_allclose(
        features.loc[at, "market_internal__log_return_dispersion_1w"],
        one_week_equity_returns.std(ddof=0),
    )
    assert features.loc[at, "market_internal__directional_synchronization_1w"] == 0.5
    equity_columns = ("spy_close", "qqq_close", "xly_close", "xlp_close")
    trailing_returns = np.log(frame.loc[:, equity_columns]).diff().iloc[-4:]
    expected_average_correlation = np.mean(
        [
            trailing_returns[left].corr(trailing_returns[right])
            for left, right in combinations(equity_columns, 2)
        ]
    )
    np.testing.assert_allclose(
        features.loc[at, "market_internal__average_pairwise_correlation_4w"],
        expected_average_correlation,
    )
    np.testing.assert_allclose(
        features.loc[at, "market_spread__cyclical_defensive__relative_return_4w"],
        np.log(121.0 / 104.0),
    )
    np.testing.assert_allclose(
        features.loc[at, "market_spread__credit_treasury__relative_return_4w"],
        np.log(104.0 / 96.0),
    )
    assert features.loc[at, "volume_internal__rising_volume_share_1w"] == 0.5
    assert (
        features.loc[at, "volume_internal__net_price_volume_confirmation_1w"]
        == 0.5
    )


def test_new_feature_groups_are_compact_and_optional_pairs_are_skipped() -> None:
    frame = _canonical_market_frame()
    features = build_weekly_features(
        frame,
        FeatureConfig(
            price_columns=("spy_close", "qqq_close"),
            volume_columns=("spy_volume", "qqq_volume"),
        ),
    )

    expected = {
        "market_internal__positive_return_share_1w",
        "market_internal__above_trailing_mean_share_13w",
        "market_internal__log_return_dispersion_1w",
        "market_internal__directional_synchronization_1w",
        "market_internal__average_pairwise_correlation_13w",
        "volume_internal__rising_volume_share_1w",
        "volume_internal__net_price_volume_confirmation_1w",
        "weekly_economic_index__change_4w_z_52w",
    }
    assert expected.issubset(features.columns)
    assert not any(column.startswith("market_spread__") for column in features.columns)
    assert sum(column.startswith("market_internal__") for column in features.columns) == 9
    assert sum(column.startswith("volume_internal__") for column in features.columns) == 2

    macro_change = frame["weekly_economic_index"].diff(4)
    expected_macro = (
        macro_change
        - macro_change.rolling(52, min_periods=26).mean()
    ) / macro_change.rolling(52, min_periods=26).std(ddof=0)
    assert_series_equal(
        features["weekly_economic_index__change_4w_z_52w"],
        expected_macro,
        check_names=False,
    )


def test_feature_builder_preserves_missingness_in_new_feature_groups() -> None:
    frame = _canonical_market_frame()
    missing_at = frame.index[80]
    next_at = frame.index[81]
    frame.loc[missing_at, ["spy_close", "qqq_close"]] = np.nan
    frame.loc[missing_at, ["spy_volume", "qqq_volume"]] = np.nan
    frame.loc[missing_at, "weekly_economic_index"] = np.nan

    features = build_weekly_features(
        frame,
        FeatureConfig(
            price_columns=("spy_close", "qqq_close"),
            volume_columns=("spy_volume", "qqq_volume"),
        ),
    )

    assert np.isnan(features.loc[missing_at, "market_internal__positive_return_share_1w"])
    assert np.isnan(features.loc[next_at, "market_internal__positive_return_share_1w"])
    assert np.isnan(features.loc[missing_at, "volume_internal__rising_volume_share_1w"])
    assert np.isnan(features.loc[next_at, "volume_internal__rising_volume_share_1w"])
    assert np.isnan(features.loc[missing_at, "weekly_economic_index__level"])
    assert np.isnan(features.loc[missing_at, "weekly_economic_index__change_1w"])
    assert np.isnan(features.loc[next_at, "weekly_economic_index__change_1w"])


def test_train_only_label_thresholds_and_past_labels_ignore_future() -> None:
    original = _canonical_market_frame()
    train = original.iloc[:120]
    first_labeler = CausalRegimeLabeler().fit(train)

    changed = original.copy()
    changed.iloc[150:, changed.columns.get_loc("spy_close")] *= np.linspace(1, 10, 30)
    second_labeler = CausalRegimeLabeler().fit(changed.iloc[:120])

    assert first_labeler.lower_threshold_ == second_labeler.lower_threshold_
    assert first_labeler.upper_threshold_ == second_labeler.upper_threshold_
    first_labels = first_labeler.transform(original)
    second_labels = second_labeler.transform(changed)
    assert_series_equal(first_labels.iloc[:150], second_labels.iloc[:150])

    first_scores = first_labeler.score_frame(original)
    second_scores = second_labeler.score_frame(changed)
    assert_frame_equal(first_scores.iloc[:150], second_scores.iloc[:150])


def test_current_state_probabilities_follow_shared_order_and_sum_to_one() -> None:
    frame = _canonical_market_frame()
    labeler = CausalRegimeLabeler().fit(frame.iloc[:120])
    probabilities = labeler.state_probabilities(frame)

    assert tuple(probabilities.columns) == STATE_ORDER
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all().all()


def test_labeler_does_not_backdate_transition_after_a_later_shock() -> None:
    frame = _canonical_market_frame()
    labeler = CausalRegimeLabeler().fit(frame.iloc[:120])
    before = labeler.transform(frame.iloc[:150])
    with_later_crash = frame.copy()
    with_later_crash.iloc[150:, with_later_crash.columns.get_loc("spy_close")] *= 0.2
    after = labeler.transform(with_later_crash)

    assert_series_equal(before, after.iloc[:150])
