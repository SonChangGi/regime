from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from regime_lab.analysis import FeatureConfig, build_weekly_features
from regime_lab.analysis.structural_features import (
    build_bank_credit_features,
    build_market_group_features,
    build_nelson_siegel_features,
    build_release_innovation_features,
    build_structural_feature_manifest,
)


def _market_frame() -> pd.DataFrame:
    index = pd.date_range("2026-01-02", periods=9, freq="W-FRI")
    return pd.DataFrame(
        {
            "xlb_close": [100, 110, 121, 118, 124, 130, 128, 136, 140],
            "xlc_close": [100, 90, 99, 101, 98, 100, 105, 103, 108],
            "xle_close": [100, 105, 100, 108, 112, 110, 115, 120, 117],
            "spy_close": [100, 102, 104, 103, 106, 108, 109, 111, 114],
            "iwm_close": [100, 98, 101, 102, 100, 104, 106, 105, 109],
            "hyg_close": [100, 101, 102, 101, 103, 104, 105, 106, 107],
            "tlt_close": [100, 99, 98, 100, 101, 100, 99, 101, 102],
        },
        index=index,
        dtype=float,
    )


def _price_groups() -> dict[str, tuple[str, ...]]:
    return {
        "gics_sector": ("XLB", "XLC", "XLE"),
        "broad_size_style": ("SPY", "IWM"),
        "cross_asset": ("HYG", "TLT"),
    }


def test_market_groups_are_separate_robust_and_prefix_causal() -> None:
    frame = _market_frame()
    features = build_market_group_features(frame, _price_groups())
    at = frame.index[1]
    sector_returns = np.log(np.array([110 / 100, 90 / 100, 105 / 100]))
    positive = sector_returns.clip(min=0.0)
    weights = positive / positive.sum()

    assert features.loc[
        at, "market_group__gics_sector__positive_return_share_1w"
    ] == 2 / 3
    assert features.loc[
        at, "market_group__gics_sector__downside_share_1w"
    ] == 1 / 3
    np.testing.assert_allclose(
        features.loc[at, "market_group__gics_sector__median_log_return_1w"],
        np.median(sector_returns),
    )
    np.testing.assert_allclose(
        features.loc[at, "market_group__gics_sector__mad_dispersion_1w"],
        np.median(np.abs(sector_returns - np.median(sector_returns))),
    )
    np.testing.assert_allclose(
        features.loc[
            at, "market_group__gics_sector__leadership_concentration_1w"
        ],
        np.square(weights).sum(),
    )
    assert "market_group__broad_size_style__positive_return_share_1w" in features
    assert "market_group__cross_asset__positive_return_share_1w" in features

    missing = frame.copy()
    missing.loc[missing.index[-1], "xle_close"] = np.nan
    missing_features = build_market_group_features(missing, _price_groups())
    assert missing_features.loc[
        missing.index[-1], "market_group__gics_sector__coverage_1w"
    ] == 2 / 3

    prefix = build_market_group_features(frame.iloc[:6], _price_groups())
    assert_frame_equal(features.iloc[:6], prefix)
    changed_future = frame.copy()
    changed_future.iloc[6:] *= 10.0
    changed = build_market_group_features(changed_future, _price_groups())
    assert_frame_equal(features.iloc[:6], changed.iloc[:6])


def test_shared_feature_builder_adds_groups_without_replacing_v3_internals() -> None:
    frame = _market_frame()
    features = build_weekly_features(
        frame,
        FeatureConfig(
            price_columns=tuple(frame.columns),
            volume_columns=(),
            price_groups=_price_groups(),
        ),
    )

    assert "market_internal__positive_return_share_1w" in features
    assert "market_group__gics_sector__positive_return_share_1w" in features
    assert "market_group__broad_size_style__positive_return_share_1w" in features
    assert "market_group__cross_asset__positive_return_share_1w" in features


def test_fixed_nelson_siegel_recovers_factors_and_requires_four_maturities() -> None:
    index = pd.date_range("2026-01-02", periods=6, freq="W-FRI")
    maturities = {
        "DGS3MO": 3,
        "DGS1": 12,
        "DGS2": 24,
        "DGS5": 60,
        "DGS7": 84,
        "DGS10": 120,
        "DGS20": 240,
        "DGS30": 360,
    }
    tau = np.array(list(maturities.values()), dtype=float)
    scaled = 0.0609 * tau
    slope_loading = -np.expm1(-scaled) / scaled
    curvature_loading = slope_loading - np.exp(-scaled)
    design = np.column_stack([np.ones_like(tau), slope_loading, curvature_loading])
    coefficients = np.column_stack(
        [np.linspace(3.0, 4.0, len(index)), np.full(len(index), -1.5), np.full(len(index), 0.8)]
    )
    yields = coefficients @ design.T
    frame = pd.DataFrame(
        yields,
        index=index,
        columns=[item.lower() for item in maturities],
    )
    frame.iloc[2, 3:] = np.nan

    features = build_nelson_siegel_features(frame, maturities)
    np.testing.assert_allclose(
        features.loc[index[[0, 1, 3, 4, 5]], [
            "treasury_curve__nelson_siegel_level",
            "treasury_curve__nelson_siegel_slope",
            "treasury_curve__nelson_siegel_curvature",
        ]],
        coefficients[[0, 1, 3, 4, 5]],
        atol=1e-11,
    )
    assert features.loc[index[2], "treasury_curve__coverage"] == 3 / 8
    assert features.loc[index[2], :].iloc[:3].isna().all()


def test_bank_credit_features_apply_growth_and_unit_contract() -> None:
    index = pd.date_range("2025-01-03", periods=20, freq="W-FRI")
    total = 10_000.0 * np.exp(np.arange(len(index)) * 0.01)
    frame = pd.DataFrame(
        {
            "totbkcr": total,
            "totci": total * 0.25,
            "dpsacbw027sbog": total * 0.8,
            # Source series is millions; other three series are billions.
            "h8b3094ncba": total * 0.02 * 1000.0,
        },
        index=index,
    )

    features = build_bank_credit_features(frame)
    at = index[-1]
    np.testing.assert_allclose(features.loc[at, "bank_credit__log_growth_4w"], 0.04)
    np.testing.assert_allclose(features.loc[at, "bank_credit__log_growth_13w"], 0.13)
    np.testing.assert_allclose(features.loc[at, "bank_credit__ci_log_growth_4w"], 0.04)
    np.testing.assert_allclose(features.loc[at, "bank_credit__ci_share"], 0.25)
    np.testing.assert_allclose(
        features.loc[at, "bank_credit__deposit_funding_ratio"], 0.8
    )
    np.testing.assert_allclose(features.loc[at, "bank_credit__borrowing_ratio"], 0.02)
    assert features.loc[at, "bank_credit__coverage"] == 1.0


def test_bank_borrowing_ratio_requires_both_eligible_legs() -> None:
    index = pd.date_range("2025-01-03", periods=3, freq="W-FRI")
    frame = pd.DataFrame(
        {
            "totbkcr": [10_000.0, np.nan, 10_000.0],
            "totci": [2_500.0, 2_500.0, 2_500.0],
            "dpsacbw027sbog": [8_000.0, 8_000.0, 8_000.0],
            # H8 is millions while total credit is billions.
            "h8b3094ncba": [np.nan, 200_000.0, 200_000.0],
        },
        index=index,
    )

    features = build_bank_credit_features(frame)

    assert features["bank_credit__borrowing_ratio"].iloc[:2].isna().all()
    np.testing.assert_allclose(
        features.loc[index[2], "bank_credit__borrowing_ratio"],
        (200_000.0 / 1_000.0) / 10_000.0,
    )
    np.testing.assert_allclose(
        features["bank_credit__coverage"],
        [0.75, 0.75, 1.0],
    )


def test_release_innovation_uses_only_prior_events_and_zeroes_non_events() -> None:
    index = pd.date_range("2025-01-03", periods=12, freq="W-FRI")
    periods = [
        date(2024, 12, 1),
        date(2024, 12, 1),
        date(2025, 1, 1),
        date(2025, 1, 1),
        date(2025, 2, 1),
        date(2025, 2, 1),
        date(2025, 3, 1),
        date(2025, 3, 1),
        date(2025, 4, 1),
        date(2025, 4, 1),
        date(2025, 4, 1),
        date(2025, 5, 1),
    ]
    # Event deltas through row 9 are 1, 2, 3, 4, then a same-period revision -1.
    values = [100, 100, 101, 101, 103, 103, 106, 106, 110, 109, 109, 112]
    value_frame = pd.DataFrame({"unrate": values}, index=index, dtype=float)
    period_frame = pd.DataFrame({"unrate": periods}, index=index)
    revision_frame = pd.DataFrame(
        {"unrate": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]}, index=index
    )

    features = build_release_innovation_features(
        value_frame,
        period_frame,
        revision_sequences=revision_frame,
        series=("UNRATE",),
    )
    prefix = "release_innovation__unrate"
    assert features.loc[index[2], f"{prefix}__period_change"] == 1.0
    assert features.loc[index[2], f"{prefix}__revision_event"] == 0.0
    assert np.isnan(features.loc[index[8], f"{prefix}__expected_delta"])
    assert features.loc[index[9], f"{prefix}__revision_event"] == 1.0
    assert features.loc[index[9], f"{prefix}__delta"] == -1.0
    assert features.loc[index[9], f"{prefix}__expected_delta"] == 2.5
    np.testing.assert_allclose(
        features.loc[index[9], f"{prefix}__mad_scale"], 1.4826
    )
    np.testing.assert_allclose(
        features.loc[index[9], f"{prefix}__standardized"],
        (-1.0 - 2.5) / 1.4826,
    )
    assert features.loc[index[9], f"{prefix}__coverage"] == 1.0
    for suffix in ("event", "period_change", "revision_event", "delta", "expected_delta", "mad_scale", "standardized"):
        assert features.loc[index[10], f"{prefix}__{suffix}"] == 0.0

    prefix_features = build_release_innovation_features(
        value_frame.iloc[:10],
        period_frame.iloc[:10],
        revision_sequences=revision_frame.iloc[:10],
        series=("UNRATE",),
    )
    assert_frame_equal(features.iloc[:10], prefix_features)
    changed_values = value_frame.copy()
    changed_values.iloc[10:] += 10_000.0
    changed = build_release_innovation_features(
        changed_values,
        period_frame,
        revision_sequences=revision_frame,
        series=("UNRATE",),
    )
    assert_frame_equal(features.iloc[:10], changed.iloc[:10])


def test_feature_manifest_assigns_every_feature_once() -> None:
    columns = (
        "market_group__gics_sector__positive_return_share_1w",
        "market_group__broad_size_style__positive_return_share_1w",
        "market_group__cross_asset__positive_return_share_1w",
        "treasury_curve__nelson_siegel_level",
        "bank_credit__ci_share",
        "anfci__z_52w",
        "release_innovation__unrate__event",
        "spy_close__log_return_1w",
    )
    manifest = build_structural_feature_manifest(columns)
    flattened = [feature for group in manifest for feature in group["features"]]

    assert len(flattened) == len(set(flattened)) == len(columns)
    assert set(flattened) == set(columns)
    assert {group["id"] for group in manifest} == {
        "sector_breadth",
        "broad_size_style_breadth",
        "cross_asset_breadth",
        "treasury_curve",
        "bank_credit",
        "financial_conditions",
        "release_innovation",
        "legacy_v3",
    }
