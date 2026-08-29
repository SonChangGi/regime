from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from regime_lab.data import Observation
from regime_lab.dataset import build_weekly_dataset


SYMBOLS = (
    "SPY",
    "IWM",
    "RSP",
    "HYG",
    "TLT",
    "XLY",
    "XLP",
    "LQD",
    "IEF",
    "SHY",
)
FIELDS = ("open", "high", "low", "close", "adjusted_close", "volume")
CORE_OHLC = ("SPY", "IWM", "RSP", "HYG", "TLT")


def _config() -> dict[str, object]:
    return {
        "alpha_vantage": {
            "symbols": list(SYMBOLS),
            "fields": list(FIELDS),
            "ohlc_feature_symbols": list(CORE_OHLC),
        },
        "alfred": {
            "series": [
                {
                    "id": "NFCIRISK",
                    "domain": "financial_conditions",
                    "frequency": "weekly",
                },
                {
                    "id": "ANFCI",
                    "domain": "financial_conditions",
                    "frequency": "weekly",
                }
            ]
        },
    }


def _history(rows: int = 70) -> tuple[tuple[datetime, ...], tuple[Observation, ...]]:
    cutoffs = tuple(
        item.to_pydatetime()
        for item in pd.date_range(
            "2024-01-05 21:00:00+00:00",
            periods=rows,
            freq="W-FRI",
        )
    )
    retrieved_at = cutoffs[-1] + timedelta(days=2)
    records: list[Observation] = []
    for row_number, cutoff in enumerate(cutoffs):
        split_factor = 1.0 if row_number < 35 else 2.0
        for symbol_number, symbol in enumerate(SYMBOLS):
            adjusted_close = 100.0 + symbol_number * 8.0 + row_number * (
                1.0 + symbol_number / 40.0
            )
            raw_close = adjusted_close / split_factor
            values = {
                "open": raw_close * 0.99,
                "high": raw_close * 1.02,
                "low": raw_close * 0.97,
                "close": raw_close,
                "adjusted_close": adjusted_close,
                "volume": 1_000_000.0 + symbol_number * 10_000 + row_number * 1_000,
            }
            for field, value in values.items():
                records.append(
                    Observation(
                        source="alpha_vantage",
                        series_id=f"{symbol}.{field}",
                        observed_period_end=cutoff.date(),
                        value=value,
                        released_at=cutoff,
                        available_at=cutoff,
                        vintage_date=cutoff.date(),
                        retrieved_at=retrieved_at,
                        raw_sha256=f"{symbol}-{field}-{row_number}",
                    )
                )
        records.append(
            Observation(
                source="alfred",
                series_id="NFCIRISK",
                observed_period_end=cutoff.date(),
                value=np.sin(row_number / 7.0) + row_number / 100.0,
                released_at=cutoff,
                available_at=cutoff,
                vintage_date=cutoff.date(),
                retrieved_at=retrieved_at,
                raw_sha256=f"nfcirisk-{row_number}",
            )
        )
        records.append(
            Observation(
                source="alfred",
                series_id="ANFCI",
                observed_period_end=cutoff.date(),
                value=np.cos(row_number / 9.0) - row_number / 120.0,
                released_at=cutoff,
                available_at=cutoff,
                vintage_date=cutoff.date(),
                retrieved_at=retrieved_at,
                raw_sha256=f"anfci-{row_number}",
            )
        )
    return cutoffs, tuple(records)


def test_v3_dataset_uses_all_volumes_and_split_adjusted_ohlc_without_expanding_features() -> None:
    cutoffs, observations = _history()
    dataset = build_weekly_dataset(_config(), cutoffs, observations)

    assert {f"{symbol.lower()}_volume" for symbol in SYMBOLS}.issubset(
        dataset.canonical.columns
    )
    assert "spy_raw_close" in dataset.canonical
    assert "spy_close" in dataset.canonical
    assert dataset.canonical.columns.is_unique
    expected_adjusted_columns = {
        f"{symbol.lower()}_{suffix}"
        for symbol in SYMBOLS
        for suffix in (
            "adjusted_open",
            "adjusted_high",
            "adjusted_low",
            "adjustment_factor",
        )
    }
    assert expected_adjusted_columns.issubset(dataset.canonical.columns)

    before_split = dataset.canonical.index[34]
    after_split = dataset.canonical.index[35]
    assert dataset.canonical.loc[before_split, "spy_adjustment_factor"] == 1.0
    assert dataset.canonical.loc[after_split, "spy_adjustment_factor"] == 2.0
    np.testing.assert_allclose(
        dataset.canonical["spy_adjusted_open"],
        dataset.canonical["spy_close"] * 0.99,
    )
    np.testing.assert_allclose(
        dataset.canonical["xly_adjusted_open"],
        dataset.canonical["xly_close"] * 0.99,
    )
    np.testing.assert_allclose(
        dataset.canonical["xly_adjusted_high"],
        dataset.canonical["xly_close"] * 1.02,
    )
    np.testing.assert_allclose(
        dataset.canonical["xly_adjusted_low"],
        dataset.canonical["xly_close"] * 0.97,
    )
    assert dataset.canonical.loc[after_split, "xly_adjustment_factor"] == 2.0

    expected_range = np.log(1.02 / 0.97)
    np.testing.assert_allclose(
        dataset.features["market_ohlc__spy__log_high_low_range_1w"],
        expected_range,
    )
    expected_gap = np.log(
        dataset.canonical.loc[after_split, "spy_adjusted_open"]
        / dataset.canonical.loc[before_split, "spy_close"]
    )
    np.testing.assert_allclose(
        dataset.features.loc[after_split, "market_ohlc__spy__log_gap_1w"],
        expected_gap,
    )

    assert sum(
        column.startswith("market_ohlc__") for column in dataset.features
    ) == len(CORE_OHLC) * 3
    assert {
        column.split("__")[1]
        for column in dataset.features
        if column.startswith("market_ohlc__")
    } == {symbol.lower() for symbol in CORE_OHLC}
    assert not any(
        column.startswith("market_ohlc__xly__") for column in dataset.features
    )
    assert any(column.startswith("market_internal__") for column in dataset.features)
    assert any(column.startswith("market_spread__") for column in dataset.features)
    assert any(column.startswith("volume_internal__") for column in dataset.features)
    assert "nfcirisk__change_4w_z_52w" in dataset.features
    assert {
        column
        for column in dataset.features
        if column.startswith("anfci__")
        and column not in {
            "anfci__age_days",
            "anfci__release_lag_days",
            "anfci__is_filled",
        }
    } == {
        "anfci__level",
        "anfci__change_1w",
        "anfci__change_4w",
        "anfci__z_52w",
    }
    financial_group = next(
        group
        for group in dataset.feature_group_manifest
        if group["id"] == "financial_conditions"
    )
    assert set(financial_group["features"]) == {
        "anfci__level",
        "anfci__change_1w",
        "anfci__change_4w",
        "anfci__z_52w",
    }
    assert {
        "alpha_weekly_volume_internals",
        "alpha_adjusted_ohlc_internals",
    }.issubset({item["id"] for item in dataset.feature_catalog})

    # Preserve the v2 inputs while avoiding a 20x expansion of individual
    # volume features; non-SPY volumes feed only the compact internals.
    assert "spy_volume__log_change_1w" in dataset.features
    assert "iwm_volume__log_change_1w" not in dataset.features
    assert "iwm_close_vs_spy_close__relative_return_13w" in dataset.features
    assert "iwm_close_vs_spy_close__relative_return_26w" in dataset.features
    assert not any("_raw_" in column for column in dataset.features)


def test_v3_dataset_is_prefix_causal_and_rejects_incomplete_ohlc_config() -> None:
    cutoffs, observations = _history()
    full = build_weekly_dataset(_config(), cutoffs, observations)
    prefix_cutoffs = cutoffs[:55]
    prefix = build_weekly_dataset(_config(), prefix_cutoffs, observations)
    assert_frame_equal(full.features.iloc[:55], prefix.features)

    incomplete = _config()
    alpha = incomplete["alpha_vantage"]
    assert isinstance(alpha, dict)
    alpha["fields"] = ["adjusted_close", "volume"]
    with pytest.raises(ValueError, match="require open, high, low, and close"):
        build_weekly_dataset(incomplete, cutoffs, observations)


def test_research_dividend_rows_are_available_to_analysis_but_not_model_features() -> None:
    cutoffs, observations = _history(rows=24)
    config = _config()
    alpha = config["alpha_vantage"]
    assert isinstance(alpha, dict)
    alpha["research_fields"] = ["dividend_amount"]

    dividends = tuple(
        Observation(
            source="alpha_vantage",
            series_id=f"{symbol}.dividend_amount",
            observed_period_end=cutoff.date(),
            value=0.25 if row_number % 13 == 0 else 0.0,
            released_at=cutoff,
            available_at=cutoff,
            vintage_date=cutoff.date(),
            retrieved_at=cutoffs[-1] + timedelta(days=2),
            raw_sha256=f"{symbol}-dividend-{row_number}",
        )
        for row_number, cutoff in enumerate(cutoffs)
        for symbol in SYMBOLS
    )
    with_research_rows = build_weekly_dataset(
        config,
        cutoffs,
        observations + dividends,
    )
    without_research_rows = build_weekly_dataset(
        _config(),
        cutoffs,
        observations,
    )

    shared_columns = without_research_rows.canonical.columns
    assert_frame_equal(
        with_research_rows.canonical.loc[:, shared_columns],
        without_research_rows.canonical,
    )
    for symbol in SYMBOLS:
        column = f"{symbol.lower()}_dividend_amount"
        assert column in with_research_rows.canonical
        assert column not in without_research_rows.canonical
        assert with_research_rows.canonical[column].notna().all()
    assert_frame_equal(with_research_rows.features, without_research_rows.features)
    assert not any("dividend" in column for column in with_research_rows.features)


def test_late_h8_vintage_uses_missingness_and_coverage_without_prefix_leakage() -> None:
    cutoffs, base_observations = _history(rows=24)
    config = _config()
    alfred = config["alfred"]
    assert isinstance(alfred, dict)
    series = alfred["series"]
    assert isinstance(series, list)
    series.extend(
        {
            "id": series_id,
            "domain": "bank_credit",
            "frequency": "weekly",
        }
        for series_id in (
            "TOTBKCR",
            "TOTCI",
            "DPSACBW027SBOG",
            "H8B3094NCBA",
        )
    )
    config["feature_engineering"] = {
        "bank_credit": {
            "total_credit": "TOTBKCR",
            "commercial_industrial": "TOTCI",
            "deposits": "DPSACBW027SBOG",
            "borrowings_millions": "H8B3094NCBA",
        }
    }

    retrieved_at = cutoffs[-1] + timedelta(days=2)
    observations = list(base_observations)
    for row_number, cutoff in enumerate(cutoffs):
        for series_id, value in (
            ("TOTBKCR", 10_000.0 + row_number),
            ("TOTCI", 2_500.0 + row_number),
            ("DPSACBW027SBOG", 8_000.0 + row_number),
        ):
            observations.append(
                Observation(
                    source="alfred",
                    series_id=series_id,
                    observed_period_end=cutoff.date(),
                    value=value,
                    released_at=cutoff,
                    available_at=cutoff,
                    vintage_date=cutoff.date(),
                    retrieved_at=retrieved_at,
                    raw_sha256=f"{series_id}-{row_number}",
                )
            )

    release_position = 12
    h8_release = cutoffs[release_position] + timedelta(hours=2)
    observations.append(
        Observation(
            source="alfred",
            series_id="H8B3094NCBA",
            observed_period_end=cutoffs[release_position].date(),
            value=200_000.0,
            released_at=h8_release,
            available_at=h8_release,
            vintage_date=h8_release.date(),
            retrieved_at=retrieved_at,
            raw_sha256="h8-late-vintage",
        )
    )

    dataset = build_weekly_dataset(config, cutoffs, tuple(observations))
    first_eligible_position = release_position + 1
    early = dataset.features.iloc[:first_eligible_position]

    assert early["h8b3094ncba__level"].isna().all()
    assert early["bank_credit__borrowing_ratio"].isna().all()
    assert early["h8b3094ncba__missing"].eq(1.0).all()
    assert early["bank_credit__coverage"].eq(0.75).all()
    first_eligible = dataset.features.index[first_eligible_position]
    assert dataset.features.loc[first_eligible, "h8b3094ncba__missing"] == 0.0
    assert dataset.features.loc[first_eligible, "bank_credit__coverage"] == 1.0
    np.testing.assert_allclose(
        dataset.features.loc[first_eligible, "bank_credit__borrowing_ratio"],
        (200_000.0 / 1_000.0) / (10_000.0 + first_eligible_position),
    )

    revised_observations = tuple(
        replace(item, value=900_000.0)
        if item.series_id == "H8B3094NCBA"
        else item
        for item in observations
    )
    revised = build_weekly_dataset(config, cutoffs, revised_observations)
    assert_frame_equal(
        dataset.features.iloc[:first_eligible_position],
        revised.features.iloc[:first_eligible_position],
    )


def test_ohlc_features_fail_closed_on_cross_field_period_mismatch() -> None:
    cutoffs, observations = _history(rows=12)
    missing_period = cutoffs[-1].date()
    mismatched = tuple(
        item
        for item in observations
        if not (
            item.series_id == "SPY.open"
            and item.observed_period_end == missing_period
        )
    )
    dataset = build_weekly_dataset(_config(), cutoffs, mismatched)

    at = dataset.features.index[-1]
    assert np.isnan(dataset.canonical.loc[at, "spy_adjusted_open"])
    assert np.isnan(dataset.features.loc[at, "market_ohlc__spy__log_gap_1w"])
    assert np.isnan(
        dataset.features.loc[at, "market_ohlc__spy__log_high_low_range_1w"]
    )


def test_dataset_exposes_separate_market_groups_and_feature_manifest() -> None:
    cutoffs, observations = _history()
    config = _config()
    alpha = config["alpha_vantage"]
    assert isinstance(alpha, dict)
    alpha["symbol_groups"] = {
        "gics_sector": ["XLY", "XLP"],
        "broad_size_style": ["SPY", "IWM", "RSP"],
        "cross_asset": ["HYG", "TLT", "LQD", "IEF", "SHY"],
    }

    dataset = build_weekly_dataset(config, cutoffs, observations)

    assert "market_group__gics_sector__downside_share_1w" in dataset.features
    assert (
        "market_group__broad_size_style__positive_return_share_1w"
        in dataset.features
    )
    assert (
        "market_group__cross_asset__positive_return_share_1w"
        in dataset.features
    )
    manifest_ids = {item["id"] for item in dataset.feature_group_manifest}
    assert {
        "sector_breadth",
        "broad_size_style_breadth",
        "cross_asset_breadth",
        "legacy_v3",
    }.issubset(manifest_ids)
    flattened = [
        feature
        for group in dataset.feature_group_manifest
        for feature in group["features"]
    ]
    assert len(flattened) == len(dataset.features.columns)
    assert set(flattened) == set(dataset.features.columns)
