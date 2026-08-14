from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lab.analysis.nowcast import (
    ShadowNowcastConfig,
    filter_shadow_nowcast,
)
from regime_lab.schema import STATE_ORDER


def _emissions(rows: int = 24) -> pd.DataFrame:
    index = pd.date_range("2020-01-03", periods=rows, freq="W-FRI")
    angles = np.linspace(0.0, 3.0 * np.pi, rows)
    values = np.column_stack(
        [
            0.2 + 0.6 * (np.sin(angles) + 1.0) / 2.0,
            np.full(rows, 0.25),
            0.2 + 0.6 * (np.cos(angles) + 1.0) / 2.0,
        ]
    )
    values /= values.sum(axis=1, keepdims=True)
    return pd.DataFrame(values, index=index, columns=STATE_ORDER)


def test_shadow_nowcast_preserves_state_order_and_probability_contract() -> None:
    result = filter_shadow_nowcast(_emissions())

    assert tuple(result.probabilities.columns) == STATE_ORDER
    assert result.probabilities.index.equals(result.states.index)
    assert result.diagnostics.index.equals(result.states.index)
    np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0, atol=1e-12)
    assert set(result.states.unique()).issubset(STATE_ORDER)
    assert (result.diagnostics["duration_weeks"] >= 1).all()
    assert result.diagnostics["confidence"].between(0.0, 1.0).all()
    assert result.diagnostics["posterior_switch_probability"].between(0.0, 1.0).all()


def test_shadow_nowcast_is_prefix_invariant_when_future_evidence_changes() -> None:
    full = _emissions(36)
    changed = full.copy()
    changed.iloc[22:] = np.asarray([0.001, 0.001, 0.998])

    original = filter_shadow_nowcast(full)
    mutated = filter_shadow_nowcast(changed)
    prefix = full.index[:22]

    pd.testing.assert_frame_equal(
        original.probabilities.loc[prefix], mutated.probabilities.loc[prefix]
    )
    pd.testing.assert_series_equal(original.states.loc[prefix], mutated.states.loc[prefix])
    pd.testing.assert_frame_equal(
        original.diagnostics.loc[prefix], mutated.diagnostics.loc[prefix]
    )


def test_running_a_prefix_matches_the_same_rows_from_the_full_filter() -> None:
    emissions = _emissions(30)
    full = filter_shadow_nowcast(emissions)
    prefix = filter_shadow_nowcast(emissions.iloc[:17])

    pd.testing.assert_frame_equal(full.probabilities.iloc[:17], prefix.probabilities)
    pd.testing.assert_series_equal(full.states.iloc[:17], prefix.states)
    pd.testing.assert_frame_equal(full.diagnostics.iloc[:17], prefix.diagnostics)


def test_display_path_routes_apparent_direct_risk_state_jump_through_transition() -> None:
    index = pd.date_range("2022-01-07", periods=12, freq="W-FRI")
    values = np.asarray(
        [[0.998, 0.001, 0.001]] * 4
        + [[0.001, 0.001, 0.998]] * 4
        + [[0.998, 0.001, 0.001]] * 4
    )
    result = filter_shadow_nowcast(
        pd.DataFrame(values, index=index, columns=STATE_ORDER),
        config=ShadowNowcastConfig(minimum_duration_weeks=1),
    )

    adjacent = list(zip(result.states.iloc[:-1], result.states.iloc[1:]))
    assert ("risk_on", "risk_off") not in adjacent
    assert ("risk_off", "risk_on") not in adjacent
    assert result.diagnostics["transition_routed"].any()


def test_duration_and_flip_summary_are_consistent_with_the_routed_path() -> None:
    result = filter_shadow_nowcast(_emissions(40))
    states = result.states.to_numpy(dtype=object)
    expected_duration: list[int] = []
    duration = 0
    previous = None
    for state in states:
        duration = duration + 1 if state == previous else 1
        expected_duration.append(duration)
        previous = state

    assert result.diagnostics["duration_weeks"].tolist() == expected_duration
    summary = result.summary()
    assert summary["observations"] == len(result.states)
    assert summary["state_changes"] == int(
        result.diagnostics["state_changed"].sum()
    )
    assert summary["latest_duration_weeks"] == expected_duration[-1]


def test_shadow_filter_rejects_ambiguous_or_invalid_probability_inputs() -> None:
    emissions = _emissions(6)
    with pytest.raises(ValueError, match="exactly ordered"):
        filter_shadow_nowcast(emissions.loc[:, list(reversed(STATE_ORDER))])
    with pytest.raises(ValueError, match="sum to one"):
        filter_shadow_nowcast(emissions * 0.8)
    invalid = emissions.copy()
    invalid.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        filter_shadow_nowcast(invalid)


def test_configuration_rejects_incoherent_duration_and_hazard_values() -> None:
    with pytest.raises(ValueError, match="maximum_duration"):
        ShadowNowcastConfig(minimum_duration_weeks=4, maximum_duration_weeks=3)
    with pytest.raises(ValueError, match="maximum_exit_hazard"):
        ShadowNowcastConfig(
            transition_base_hazard=0.5,
            maximum_exit_hazard=0.4,
        )
