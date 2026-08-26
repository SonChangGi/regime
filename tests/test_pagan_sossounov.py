from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lab.analysis.pagan_sossounov import (
    PaganSossounovConfig,
    pagan_sossounov_chronology,
)
from regime_lab.analysis.label_spec import load_label_spec_registry


def _cycle_prices() -> pd.Series:
    anchors = {
        0: 100.0,
        10: 150.0,
        20: 90.0,
        30: 160.0,
        40: 80.0,
        51: 145.0,
        61: 85.0,
        71: 155.0,
        82: 100.0,
    }
    values = np.empty(83, dtype=float)
    points = sorted(anchors)
    for left, right in zip(points[:-1], points[1:], strict=True):
        values[left : right + 1] = np.linspace(
            anchors[left], anchors[right], right - left + 1
        )
    return pd.Series(
        values,
        index=pd.period_range("2010-01", periods=len(values), freq="M"),
        name="price",
    )


def test_pagan_sossounov_is_ex_post_monthly_and_alternating() -> None:
    result = pagan_sossounov_chronology(_cycle_prices())

    assert result.turning_points["kind"].tolist()[:4] == [
        "peak",
        "trough",
        "peak",
        "trough",
    ]
    assert set(result.states) == {"bull", "bear"}
    assert result.uses_future_observations is True
    assert result.canonical_target is False
    assert result.automatic_promotion_eligible is False
    assert result.role == "retrospective_ex_post_sensitivity_only"
    assert result.turning_points["future_confirmation_months"].eq(8).all()
    assert (
        result.turning_points["confirmed_at"] > result.turning_points["at"]
    ).all()
    assert len(result.configuration_sha256) == 64
    method = load_label_spec_registry().shadow_methods[
        "pagan_sossounov_bull_bear"
    ]
    assert dict(method.configuration) == {
        "window_months": 8,
        "censor_margin_months": 6,
        "minimum_phase_months": 4,
        "minimum_cycle_months": 16,
        "large_move_threshold": 0.20,
    }
    assert result.label_method_spec_sha256 == method.spec_sha256
    assert result.configuration_origin == "label_spec_default"


def test_large_move_can_preserve_a_short_phase_but_small_move_cannot() -> None:
    config = PaganSossounovConfig(
        window_months=1,
        censor_margin_months=1,
        minimum_phase_months=4,
        minimum_cycle_months=8,
        large_move_threshold=0.20,
    )
    anchors = {
        0: 100.0,
        8: 150.0,
        12: 100.0,
        15: 170.0,
        16: 110.0,
        24: 160.0,
        32: 90.0,
        39: 130.0,
    }
    values = np.empty(40, dtype=float)
    points = sorted(anchors)
    for left, right in zip(points[:-1], points[1:], strict=True):
        values[left : right + 1] = np.linspace(
            anchors[left], anchors[right], right - left + 1
        )
    large = pd.Series(
        values,
        index=pd.period_range("2010-01", periods=len(values), freq="M"),
    )
    result = pagan_sossounov_chronology(large, config=config)
    points = result.turning_points
    positions = [large.index.get_loc(pd.Timestamp(at).to_period("M")) for at in points["at"]]
    assert any(
        right - left < config.minimum_phase_months
        for left, right in zip(positions[:-1], positions[1:], strict=True)
    )

    small = large.copy()
    small.iloc[15:17] = [125.0, 110.0]
    small_result = pagan_sossounov_chronology(small, config=config)
    small_points = small_result.turning_points
    small_positions = [
        small.index.get_loc(pd.Timestamp(at).to_period("M"))
        for at in small_points["at"]
    ]
    for left, right in zip(small_positions[:-1], small_positions[1:], strict=True):
        amplitude = abs(float(small.iloc[right] / small.iloc[left] - 1.0))
        assert right - left >= config.minimum_phase_months or amplitude >= 0.20


def test_pagan_sossounov_rejects_weekly_gaps_and_incomplete_chronology() -> None:
    weekly = pd.Series(
        np.arange(60, dtype=float) + 100.0,
        index=pd.date_range("2020-01-03", periods=60, freq="W-FRI"),
    )
    with pytest.raises(ValueError, match="one observation per month"):
        pagan_sossounov_chronology(weekly)

    monotonic = pd.Series(
        np.arange(60, dtype=float) + 100.0,
        index=pd.period_range("2020-01", periods=60, freq="M"),
    )
    with pytest.raises(ValueError, match="two completed ex-post turns"):
        pagan_sossounov_chronology(monotonic)


def test_pagan_sossounov_configuration_hashes_effective_semantics() -> None:
    integer = pagan_sossounov_chronology(
        _cycle_prices(),
        config=PaganSossounovConfig(8, 6, 4, 16, 0.20),
    )
    equivalent = pagan_sossounov_chronology(
        _cycle_prices(),
        config=PaganSossounovConfig(8.0, 6.0, 4.0, 16.0, 0.20),
    )
    assert integer.configuration_sha256 == equivalent.configuration_sha256
    assert integer.turning_points.equals(equivalent.turning_points)
