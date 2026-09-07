"""Helpers for producing the dashboard result contract."""

from __future__ import annotations

import math
from numbers import Real
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from regime_lab.io import write_json_atomic
from regime_lab.operating_contract import load_operating_contract
from regime_lab.schema import SCHEMA_VERSION, STATE_ORDER, validate_dashboard_payload

STATE_DEFINITIONS = [
    dict(value) for value in load_operating_contract().state_definitions
]


def normalized_probabilities(values: Mapping[str, float] | list[float] | np.ndarray) -> dict[str, float]:
    """Validate probabilities before correcting only numerical round-off.

    Invalid model output must fail at the publication boundary; silently
    replacing a broken vector would invent confidence without a fallback.
    """
    if isinstance(values, Mapping):
        if set(values) != set(STATE_ORDER):
            raise ValueError("probabilities must contain exactly the three state keys")
        supplied = [values[state] for state in STATE_ORDER]
    else:
        supplied = np.asarray(values, dtype=object).reshape(-1).tolist()
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) for value in supplied):
        raise ValueError("probabilities must be numeric, not boolean or text")
    raw = np.asarray(supplied, dtype=float)
    if raw.shape != (len(STATE_ORDER),):
        raise ValueError(f"expected {len(STATE_ORDER)} probabilities, got {raw.shape}")
    if not np.isfinite(raw).all() or (raw < 0.0).any() or (raw > 1.0).any():
        raise ValueError("probabilities must be finite and between zero and one")
    total = float(raw.sum())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("probabilities must sum to one within rounding tolerance")
    raw /= total
    return {state: round(float(raw[index]), 8) for index, state in enumerate(STATE_ORDER)}


def estimate_from_probabilities(values: Mapping[str, float] | list[float] | np.ndarray) -> dict[str, Any]:
    probabilities = normalized_probabilities(values)
    winner = max(STATE_ORDER, key=lambda state: probabilities[state])
    nonzero = [prob for prob in probabilities.values() if prob > 0]
    entropy = -sum(prob * math.log(prob) for prob in nonzero) / math.log(len(STATE_ORDER))
    return {
        "state": winner,
        "probabilities": probabilities,
        "confidence": round(probabilities[winner], 8),
        "entropy": round(float(entropy), 8),
    }


def write_dashboard_payload(payload: dict[str, Any], path: str | Path) -> Path:
    payload.setdefault("meta", {}).setdefault("schema_version", SCHEMA_VERSION)
    validate_dashboard_payload(payload)
    return write_json_atomic(path, payload)
