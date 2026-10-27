"""V5 forecast serialization conserves mass before cumulative reconciliation."""

from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
from itertools import permutations

import numpy as np
import pandas as pd
import pytest

from regime_lab import pipeline
from regime_lab.analysis.directional_coherence import reconcile_directional_path
from regime_lab.contract_v5 import _validate_directional_risk
from regime_lab.payload import estimate_from_probabilities, normalized_probabilities


STATES = ("risk_on", "transition", "risk_off")
SCALE = 100_000_000
ACTUAL_RAW = (0.016716306458297827, 0.33975542633813505, 0.643528267203567)
ACTUAL_CANONICAL = {"risk_on": 0.01671631, "transition": 0.33975542, "risk_off": 0.64352827}


def units(probabilities):
    return {state: round(probabilities[state] * SCALE) for state in STATES}


def test_actual_raw_forecast_is_jointly_quantized_without_changing_current_or_departure_anchor():
    raw = dict(zip(STATES, ACTUAL_RAW, strict=True))
    before = deepcopy(raw)
    result = pipeline._next_week_estimate(raw, pd.Timestamp("2023-03-31"), conserve_mass=True)
    assert result["probabilities"] == ACTUAL_CANONICAL
    assert result["confidence"] == 0.64352827
    assert result["state"] == "risk_off"
    assert sum(units(result["probabilities"]).values()) == SCALE
    assert SCALE - units(result["probabilities"])["risk_off"] == 35_647_173
    assert raw == before
    # Every model's comparison surface uses the same unrounded model vector.
    comparison = pipeline._comparison_forecast_record({
        **{f"p_{state}": raw[state] for state in STATES},
        "target_date": "2023-03-31", "model": "causal_dynamic_ensemble", "fallback": False,
    }, conserve_mass=True)
    for field in ("probabilities", "confidence", "state", "entropy"):
        assert comparison[field] == result[field]
    origin = date(2023, 3, 24)
    totals = (0.35647173, 0.35647173, 0.42044338)
    rows = {
        f"{horizon}w": {
            "probability": total, "no_departure": 1 - total,
            "first_destination": {"risk_on": total / 2, "transition": total / 2, "risk_off": 0.0},
            "target_end": (origin + timedelta(weeks=horizon)).isoformat(),
            "model": "synthetic_directional", "method": "first_departure_state_within_h_or_no_departure",
        }
        for horizon, total in zip((1, 4, 13), totals, strict=True)
    }
    reconciled = reconcile_directional_path("risk_off", result["probabilities"], rows)
    assert reconciled["4w"]["first_destination"] == {
        "risk_on": 0.01671631, "transition": 0.33975542, "risk_off": 0.0,
    }
    _validate_directional_risk(
        reconciled, "actual.directional_risk", current_state="risk_off",
        departure=dict(zip((1, 4, 13), totals, strict=True)), origin=origin,
        canonical_probabilities=result["probabilities"],
    )


@pytest.mark.parametrize("values", [
    ACTUAL_RAW, (0.333333333, 0.333333333, 0.333333334),
    (0.60319863, 0.07252331, 0.32427806),
    (0.50158228, 0.25525824, 0.24315948),
    (0.0, 0.0, 1.0), (0.5, 0.5, 0.0), (1 / 3, 1 / 3, 1 / 3),
    (0.250000005, 0.249999995, 0.5),
    (np.nextafter(0.250000005, 0.0), 0.249999995, 0.5),
    (np.nextafter(0.250000005, 1.0), 0.249999995, 0.5),
])
def test_joint_quantization_conserves_integer_mass_and_is_idempotent(values):
    original = np.array(values)
    before = original.copy()
    result = normalized_probabilities(original, conserve_mass=True)
    assert list(result) == list(STATES)
    assert sum(units(result).values()) == SCALE
    assert all(0 <= value <= 1 for value in result.values())
    np.testing.assert_array_equal(original, before)
    normalized = original / original.sum()
    assert max(abs(result[state] - normalized[index]) for index, state in enumerate(STATES)) <= 1 / SCALE
    assert normalized_probabilities(result, conserve_mass=True) == result


def test_largest_remainder_ties_use_state_order_even_with_reordered_mapping():
    values = dict(zip(reversed(STATES), [1 / 3] * 3, strict=True))
    assert normalized_probabilities(values, conserve_mass=True) == {
        "risk_on": 0.33333334, "transition": 0.33333333, "risk_off": 0.33333333,
    }


def test_joint_quantization_preserves_integer_simplex_grid_including_normalization_boundaries():
    rng = np.random.default_rng(315)
    for _ in range(500):
        cutpoints = sorted(rng.integers(0, SCALE + 1, size=2).tolist())
        expected_units = (cutpoints[0], cutpoints[1] - cutpoints[0], SCALE - cutpoints[1])
        values = dict(zip(STATES, [value / SCALE for value in expected_units], strict=True))
        assert normalized_probabilities(values, conserve_mass=True) == values


@pytest.mark.parametrize("current_state", STATES)
@pytest.mark.parametrize("raw", list(permutations(ACTUAL_RAW)) + [
    (0.333333333, 0.333333333, 0.333333334),
    (0.333333336, 0.333333336, 0.333333328),
    (0.0, 0.0, 1.0), (1.0, 0.0, 0.0),
])
@pytest.mark.parametrize("longer_increment", [0, 1, 6_397_165])
def test_quantized_forecast_and_joint_projection_satisfy_unchanged_strict_cdf_contract(
    current_state, raw, longer_increment,
):
    canonical = pipeline._next_week_estimate(
        dict(zip(STATES, raw, strict=True)), pd.Timestamp("2023-03-31"), conserve_mass=True,
    )["probabilities"]
    canonical_units = units(canonical)
    p1 = SCALE - canonical_units[current_state]
    totals_units = (p1, min(SCALE, p1 + longer_increment), min(SCALE, p1 + 2 * longer_increment))
    alternatives = [state for state in STATES if state != current_state]
    origin = date(2023, 3, 24)
    rows = {}
    for horizon, total, left_fraction in zip((1, 4, 13), totals_units, (0.8, 0.1, 0.95), strict=True):
        probability = total / SCALE
        rows[f"{horizon}w"] = {
            "probability": probability, "no_departure": 1 - probability,
            "first_destination": {current_state: 0.0, alternatives[0]: probability * left_fraction,
                                  alternatives[1]: probability * (1 - left_fraction)},
            "target_end": (origin + timedelta(weeks=horizon)).isoformat(),
            "model": "synthetic_directional", "method": "first_departure_state_within_h_or_no_departure",
        }
    result = reconcile_directional_path(current_state, canonical, rows)
    departure = dict(zip((1, 4, 13), [value / SCALE for value in totals_units], strict=True))
    _validate_directional_risk(result, "synthetic.directional_risk", current_state=current_state,
                               departure=departure, origin=origin, canonical_probabilities=canonical)
    for state in STATES:
        actual_units = [units(result[f"{horizon}w"]["first_destination"])[state] for horizon in (1, 4, 13)]
        assert actual_units == sorted(actual_units)
        assert result["1w"]["first_destination"][state] == (0.0 if state == current_state else canonical[state])
    for horizon, expected_total in zip((1, 4, 13), totals_units, strict=True):
        assert result[f"{horizon}w"]["first_destination"][current_state] == 0.0
        assert sum(units(result[f"{horizon}w"]["first_destination"]).values()) == expected_total


@pytest.mark.parametrize("values", [
    [0, 0, 0], [np.nan, 1, 0], [np.inf, 0, 0], [-0.1, 0.6, 0.5], [2, 1, 1],
    [0.5, 0.5, 0.5], [True, 0.0, 0.0], [np.bool_(True), 0.0, 0.0], ["1", 0, 0],
    [0.5, 0.5], {"risk_on": 1}, {"risk_on": 1, "transition": 0, "risk_off": 0, "extra": 0},
])
def test_joint_quantization_never_relaxes_model_input_validation(values):
    with pytest.raises(ValueError):
        estimate_from_probabilities(values, conserve_mass=True)


@pytest.mark.parametrize("invalid", [True, np.bool_(True), "1", np.nan, np.inf, -1])
def test_v5_comparison_serialization_does_not_cast_invalid_model_values_into_valid_probabilities(invalid):
    with pytest.raises(ValueError):
        pipeline._comparison_forecast_record({
            "p_risk_on": invalid, "p_transition": 0.0, "p_risk_off": 0.0,
            "target_date": "2023-03-31", "model": "synthetic", "fallback": False,
        }, conserve_mass=True)


def test_default_forecast_and_current_membership_serialization_keep_existing_rounding():
    raw = dict(zip(STATES, ACTUAL_RAW, strict=True))
    expected = {"risk_on": 0.01671631, "transition": 0.33975543, "risk_off": 0.64352827}
    assert normalized_probabilities(raw) == expected
    assert estimate_from_probabilities(raw)["probabilities"] == expected
    assert pipeline._next_week_estimate(raw, pd.Timestamp("2023-03-31"))["probabilities"] == expected
    assert pipeline._current_estimate(pd.Series(raw), "risk_off")["probabilities"] == expected
