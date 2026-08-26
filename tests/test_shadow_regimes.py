from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lab.analysis.labels import STATE_ORDER
from regime_lab.analysis.shadow_regimes import (
    BOCPDConfig,
    DirectJumpHSMMConfig,
    bayesian_online_changepoint_shadow,
    filter_direct_jump_hsmm_shadow,
    shadow_model_registry_document,
)


def _emissions(periods: int = 30) -> pd.DataFrame:
    index = pd.date_range("2020-01-03", periods=periods, freq="W-FRI")
    values = np.tile([0.80, 0.15, 0.05], (periods, 1))
    values[periods // 2 :] = [0.05, 0.10, 0.85]
    return pd.DataFrame(values, index=index, columns=STATE_ORDER)


def test_hsmm_is_prefix_stable_and_never_claims_canonical_target() -> None:
    emissions = _emissions()
    prefix = filter_direct_jump_hsmm_shadow(emissions.iloc[:18])
    full = filter_direct_jump_hsmm_shadow(emissions)

    pd.testing.assert_frame_equal(prefix.probabilities, full.probabilities.iloc[:18])
    pd.testing.assert_series_equal(prefix.states, full.states.iloc[:18])
    pd.testing.assert_frame_equal(prefix.diagnostics, full.diagnostics.iloc[:18])
    assert full.canonical_target is False
    assert not full.diagnostics["uses_backward_smoothing"].any()
    assert not full.diagnostics["uses_supervised_target"].any()
    assert np.allclose(full.probabilities.sum(axis=1), 1.0)


def test_hsmm_allows_a_direct_risk_jump_without_invented_transition() -> None:
    index = pd.date_range("2020-01-03", periods=2, freq="W-FRI")
    emissions = pd.DataFrame(
        [[0.999, 0.0005, 0.0005], [0.0005, 0.0005, 0.999]],
        index=index,
        columns=STATE_ORDER,
    )
    config = DirectJumpHSMMConfig(
        minimum_duration_weeks=(1, 1, 1),
        base_exit_hazards=(0.95, 0.95, 0.95),
        duration_hazard_slopes=(0.0, 0.0, 0.0),
        maximum_exit_hazard=0.99,
        destination_weights=(
            (0.0, 0.01, 0.99),
            (0.5, 0.0, 0.5),
            (0.99, 0.01, 0.0),
        ),
    )
    result = filter_direct_jump_hsmm_shadow(
        emissions,
        config=config,
        initial_prior={
            "risk_on": 1.0,
            "transition": 0.0,
            "risk_off": 0.0,
        },
    )
    assert result.states.tolist() == ["risk_on", "risk_off"]
    assert bool(result.diagnostics.iloc[-1]["map_direct_jump"])
    assert result.diagnostics.iloc[-1]["filtered_direct_jump_probability"] > 0.9


def test_bocpd_is_prefix_stable_and_flags_a_large_mean_shift() -> None:
    index = pd.date_range("2020-01-03", periods=40, freq="W-FRI")
    signal = pd.Series(np.r_[np.zeros(20), np.full(20, 4.0)], index=index)
    config = BOCPDConfig(
        constant_hazard=1.0 / 26.0,
        prior_mean_variance=2.0,
        observation_variance=0.2,
    )
    prefix = bayesian_online_changepoint_shadow(signal.iloc[:25], config=config)
    full = bayesian_online_changepoint_shadow(signal, config=config)

    pd.testing.assert_frame_equal(prefix.diagnostics, full.diagnostics.iloc[:25])
    assert full.diagnostics.loc[index[20], "changepoint_probability"] > 0.95
    assert int(full.diagnostics.loc[index[20], "map_run_length"]) == 0
    assert full.canonical_target is False
    assert not full.diagnostics["uses_future_observation"].any()
    assert not full.diagnostics["uses_supervised_target"].any()


def test_bocpd_default_520_run_support_survives_real_scale_long_history() -> None:
    rows = 1_064
    position = np.arange(rows, dtype=float)
    values = (
        1.4 * np.sin(position / 11.0)
        + 0.8 * np.cos(position / 37.0)
        + 0.002 * position
    )
    # Match the scale and long-history shocks observed in the preserved V5
    # risk-score artifact that exposed dense-branch underflow.
    values[178] = -11.213
    values[519] = 4.190
    values[815] = -8.75
    index = pd.date_range("2006-01-06", periods=rows, freq="W-FRI")
    signal = pd.Series(values, index=index)

    prefix = bayesian_online_changepoint_shadow(signal.iloc[:600])
    full = bayesian_online_changepoint_shadow(signal)

    pd.testing.assert_frame_equal(prefix.diagnostics, full.diagnostics.iloc[:600])
    assert len(full.final_run_length_posterior) == 521
    assert np.isfinite(full.diagnostics.to_numpy(dtype=float)).all()
    assert np.isclose(full.final_run_length_posterior.sum(), 1.0)


def test_shadow_registry_cannot_masquerade_as_completed_results() -> None:
    document = shadow_model_registry_document()
    assert document["automatic_promotion_eligible"] is False
    assert document["canonical_target"] is False
    assert {row["status"] for row in document["models"]} == {"implemented_unrun"}
    assert all(row["result"] is None for row in document["models"])
    hsmm = next(row for row in document["models"] if row["id"] == "filtered_hsmm")
    assert hsmm["aliases"] == ["hsmm_explicit_duration"]
    factor_tvtp = next(
        row for row in document["models"] if row["id"] == "dynamic_factor_tvtp"
    )
    assert factor_tvtp["gap_weeks"] == 1
    assert factor_tvtp["causality_scope"] == "structural_row_prefix_only"
    assert factor_tvtp["operational_oos_eligible"] is False
    assert factor_tvtp["vintage_safety"].startswith("not_established")


def test_shadow_inputs_fail_closed() -> None:
    emissions = _emissions()
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        filter_direct_jump_hsmm_shadow(emissions.assign(risk_on=2.0))
    signal = pd.Series([0.0, np.nan], index=emissions.index[:2])
    with pytest.raises(ValueError, match="finite and complete"):
        bayesian_online_changepoint_shadow(signal)
