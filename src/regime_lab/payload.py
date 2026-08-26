"""Helpers for producing the dashboard result contract."""

from __future__ import annotations

import math
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
    if isinstance(values, Mapping):
        raw = np.asarray([float(values.get(state, 0.0)) for state in STATE_ORDER], dtype=float)
    else:
        raw = np.asarray(values, dtype=float).reshape(-1)
    if raw.shape != (len(STATE_ORDER),):
        raise ValueError(f"expected {len(STATE_ORDER)} probabilities, got {raw.shape}")
    raw = np.where(np.isfinite(raw), raw, 0.0)
    raw = np.clip(raw, 0.0, None)
    total = float(raw.sum())
    if total <= 0:
        raw = np.full(len(STATE_ORDER), 1.0 / len(STATE_ORDER))
    else:
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
