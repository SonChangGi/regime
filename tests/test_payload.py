from __future__ import annotations

import json

from regime_lab.payload import estimate_from_probabilities, normalized_probabilities


def test_probability_normalization_is_state_ordered() -> None:
    probabilities = normalized_probabilities({"risk_off": 2, "risk_on": 1, "transition": 1})
    assert list(probabilities) == ["risk_on", "transition", "risk_off"]
    assert sum(probabilities.values()) == 1.0


def test_zero_vector_becomes_honest_uniform_estimate() -> None:
    estimate = estimate_from_probabilities([0, 0, 0])
    assert estimate["state"] == "risk_on"
    assert estimate["confidence"] == 0.33333333
    assert estimate["entropy"] == 1.0
    json.dumps(estimate, allow_nan=False)
