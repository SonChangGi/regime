from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from regime_lab.analysis.dynamic_factor_tvtp import (
    DynamicFactorTVTPConfig,
    run_dynamic_factor_tvtp_shadow,
)


def _inputs(rows: int = 34) -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2021-01-01", periods=rows, freq="W-FRI")
    position = np.arange(rows, dtype=float)
    features = pd.DataFrame(
        {
            "market": np.sin(position / 3.0) + position / 30.0,
            "breadth": np.cos(position / 5.0),
            "credit": np.sin(position / 7.0) - position / 50.0,
            "macro": np.cos(position / 9.0) + position / 80.0,
        },
        index=index,
    )
    pattern = (
        ["risk_on"] * 5
        + ["risk_off"] * 3
        + ["transition"] * 2
        + ["risk_on"] * 4
        + ["transition"] * 2
        + ["risk_off"] * 4
    )
    states = pd.Series(
        [pattern[item % len(pattern)] for item in range(rows)],
        index=index,
        name="state",
    )
    return features, states


def _config() -> DynamicFactorTVTPConfig:
    return DynamicFactorTVTPConfig(
        n_factors=2,
        factor_difference_lags=(1, 4),
        min_train_size=14,
    )


def test_dynamic_factor_tvtp_is_prefix_stable_and_purged() -> None:
    features, states = _inputs()
    prefix = run_dynamic_factor_tvtp_shadow(
        features.iloc[:27],
        states.iloc[:27],
        config=_config(),
    )
    revised = features.copy()
    revised.iloc[27:] = 10_000.0
    full = run_dynamic_factor_tvtp_shadow(revised, states, config=_config())

    common = full.predictions.loc[
        full.predictions["origin_date"].isin(prefix.predictions["origin_date"])
    ].reset_index(drop=True)
    assert_frame_equal(prefix.predictions.reset_index(drop=True), common)
    assert (
        pd.to_datetime(full.predictions["last_train_target_date"])
        < pd.to_datetime(full.predictions["origin_date"])
    ).all()
    assert full.predictions["gap"].eq(1).all()
    assert not full.predictions["target_passed_to_prediction"].any()
    assert full.canonical_target is False
    assert full.automatic_promotion_eligible is False
    assert full.causality_scope == "structural_row_prefix_only"
    assert full.operational_oos_eligible is False


def test_dynamic_factor_tvtp_allows_direct_destinations_and_normalizes() -> None:
    features, states = _inputs()
    result = run_dynamic_factor_tvtp_shadow(features, states, config=_config())
    probability = result.predictions[["p_risk_on", "p_transition", "p_risk_off"]]

    np.testing.assert_allclose(probability.sum(axis=1), 1.0)
    assert result.predictions["direct_jump_allowed"].all()
    risk_on = result.predictions["current_state"].eq("risk_on")
    risk_off = result.predictions["current_state"].eq("risk_off")
    assert (result.predictions.loc[risk_on, "p_risk_off"] > 0.0).all()
    assert (result.predictions.loc[risk_off, "p_risk_on"] > 0.0).all()
    assert len(result.configuration_sha256) == 64


def test_dynamic_factor_tvtp_can_bound_a_late_origin_shadow_without_shortening_history() -> None:
    features, states = _inputs()
    unbounded = run_dynamic_factor_tvtp_shadow(features, states, config=_config())
    bounded = run_dynamic_factor_tvtp_shadow(
        features,
        states,
        config=DynamicFactorTVTPConfig(
            n_factors=2,
            factor_difference_lags=(1, 4),
            min_train_size=14,
            max_origins=3,
        ),
    )

    assert len(bounded.predictions) == 3
    assert_frame_equal(
        bounded.predictions.reset_index(drop=True),
        unbounded.predictions.tail(3).reset_index(drop=True),
    )
    assert bounded.predictions.iloc[0]["train_size"] > 14


def test_dynamic_factor_tvtp_rejects_index_and_gap_contract_drift() -> None:
    features, states = _inputs()
    with pytest.raises(ValueError, match="exact feature index"):
        run_dynamic_factor_tvtp_shadow(
            features,
            states.iloc[::-1],
            config=_config(),
        )
    with pytest.raises(ValueError, match="official gap=1"):
        DynamicFactorTVTPConfig(gap_weeks=0)
    with pytest.raises(ValueError, match="max_origins"):
        DynamicFactorTVTPConfig(max_origins=0)


def test_dynamic_factor_configuration_hashes_effective_semantics() -> None:
    features, states = _inputs()
    integer = run_dynamic_factor_tvtp_shadow(
        features,
        states,
        config=DynamicFactorTVTPConfig(
            n_factors=2,
            factor_difference_lags=(1, 4),
            min_train_size=14,
        ),
    )
    equivalent = run_dynamic_factor_tvtp_shadow(
        features,
        states,
        config=DynamicFactorTVTPConfig(
            n_factors=2.0,
            factor_difference_lags=[1.0, 4.0],
            min_train_size=14.0,
            gap_weeks=1.0,
            random_state=17.0,
        ),
    )
    assert integer.configuration_sha256 == equivalent.configuration_sha256
    assert_frame_equal(integer.predictions, equivalent.predictions)
