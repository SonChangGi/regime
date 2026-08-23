"""Canonical, hash-linked evidence tables for published v4 results."""

from __future__ import annotations

import hashlib
from io import StringIO
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from regime_lab.schema import STATE_ORDER


STATE_LABEL_HISTORY_COLUMNS: tuple[str, ...] = (
    "date",
    "state",
    "p_risk_on",
    "p_transition",
    "p_risk_off",
    "risk_score",
    "lower_threshold",
    "upper_threshold",
    "hysteresis_margin",
    "previous_state",
    "probability_temperature",
)

WEEKLY_STATE_FORECAST_COLUMNS: tuple[str, ...] = (
    "origin_date",
    "current_state",
    "current_p_risk_on",
    "current_p_transition",
    "current_p_risk_off",
    "target_date",
    "model",
    "next_p_risk_on",
    "next_p_transition",
    "next_p_risk_off",
    "fallback",
    "fallback_reason",
)

STATE_MEMBERSHIP_HISTORY_COLUMNS: tuple[str, ...] = (
    "date",
    "state",
    "m_risk_on",
    "m_transition",
    "m_risk_off",
    "risk_score",
    "lower_threshold",
    "upper_threshold",
    "hysteresis_margin",
    "previous_state",
    "membership_temperature",
)

WEEKLY_STATE_FORECAST_V5_COLUMNS: tuple[str, ...] = (
    "origin_date",
    "current_state",
    "current_m_risk_on",
    "current_m_transition",
    "current_m_risk_off",
    "target_date",
    "model",
    "next_p_risk_on",
    "next_p_transition",
    "next_p_risk_off",
    "fallback",
    "fallback_reason",
)


def canonical_evidence_csv_bytes(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> bytes:
    """Serialize one evidence table with a frozen byte-level CSV contract."""

    expected = tuple(str(column) for column in columns)
    actual = tuple(str(column) for column in frame.columns)
    if actual != expected:
        raise ValueError(
            "evidence columns must exactly match the frozen contract: "
            f"expected {expected}, got {actual}"
        )
    stream = StringIO(newline="")
    frame.to_csv(
        stream,
        index=False,
        lineterminator="\n",
        na_rep="",
        float_format="%.17g",
    )
    return stream.getvalue().encode("utf-8")


def evidence_csv_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    return hashlib.sha256(canonical_evidence_csv_bytes(frame, columns)).hexdigest()


def state_label_history(
    states: pd.Series,
    probabilities: pd.DataFrame,
    risk_scores: pd.Series,
    *,
    lower_threshold: float,
    upper_threshold: float,
    hysteresis_fraction: float,
    probability_temperature: float,
) -> pd.DataFrame:
    """Build the complete sequential label surface used by published weeks."""

    if not isinstance(states.index, pd.DatetimeIndex):
        raise TypeError("states must use a DatetimeIndex")
    if not states.index.equals(probabilities.index) or not states.index.equals(
        risk_scores.index
    ):
        raise ValueError("label evidence inputs must have identical indexes")
    if tuple(probabilities.columns) != STATE_ORDER:
        raise ValueError(f"probabilities must be ordered as {STATE_ORDER}")
    lower = float(lower_threshold)
    upper = float(upper_threshold)
    fraction = float(hysteresis_fraction)
    temperature = float(probability_temperature)
    constants = np.asarray([lower, upper, fraction, temperature], dtype=float)
    if not np.isfinite(constants).all() or lower >= upper:
        raise ValueError("label thresholds and configuration must be finite")
    margin = (upper - lower) * fraction

    rows: list[dict[str, Any]] = []
    previous_state: str | None = None
    for at in states.index:
        state = str(states.loc[at])
        if state not in STATE_ORDER:
            raise ValueError(f"unsupported state at {at}: {state}")
        rows.append(
            {
                "date": pd.Timestamp(at).isoformat(),
                "state": state,
                **{
                    f"p_{label}": float(probabilities.loc[at, label])
                    for label in STATE_ORDER
                },
                "risk_score": float(risk_scores.loc[at]),
                "lower_threshold": lower,
                "upper_threshold": upper,
                "hysteresis_margin": margin,
                "previous_state": previous_state,
                "probability_temperature": temperature,
            }
        )
        previous_state = state
    return pd.DataFrame(rows, columns=STATE_LABEL_HISTORY_COLUMNS)


def weekly_state_forecasts(
    weekly: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Mirror every public weekly current/next estimate without model assumptions."""

    rows: list[dict[str, Any]] = []
    for position, week in enumerate(weekly):
        current = week.get("current")
        next_week = week.get("next_week")
        if not isinstance(current, Mapping) or not isinstance(next_week, Mapping):
            raise ValueError(f"weekly[{position}] lacks current/next estimates")
        current_probabilities = current.get("probabilities")
        next_probabilities = next_week.get("probabilities")
        if not isinstance(current_probabilities, Mapping) or not isinstance(
            next_probabilities, Mapping
        ):
            raise ValueError(f"weekly[{position}] lacks probability mappings")
        rows.append(
            {
                "origin_date": str(week["data_as_of"]),
                "current_state": str(current["state"]),
                **{
                    f"current_p_{state}": float(current_probabilities[state])
                    for state in STATE_ORDER
                },
                "target_date": str(next_week["date"]),
                "model": str(next_week["model"]),
                **{
                    f"next_p_{state}": float(next_probabilities[state])
                    for state in STATE_ORDER
                },
                "fallback": bool(next_week["fallback"]),
                "fallback_reason": str(next_week["fallback_reason"]),
            }
        )
    return pd.DataFrame(rows, columns=WEEKLY_STATE_FORECAST_COLUMNS)


def state_membership_history(label_history: pd.DataFrame) -> pd.DataFrame:
    """Relabel v4 anchor evidence as memberships without changing its values."""

    if tuple(label_history.columns) != STATE_LABEL_HISTORY_COLUMNS:
        raise ValueError("label_history does not match the v4 evidence contract")
    return label_history.rename(
        columns={
            "p_risk_on": "m_risk_on",
            "p_transition": "m_transition",
            "p_risk_off": "m_risk_off",
            "probability_temperature": "membership_temperature",
        }
    ).loc[:, STATE_MEMBERSHIP_HISTORY_COLUMNS]


def weekly_state_forecasts_v5(
    weekly: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Serialize v5 current memberships and next-week forecast probabilities."""

    rows: list[dict[str, Any]] = []
    for position, week in enumerate(weekly):
        current = week.get("current")
        next_week = week.get("next_week")
        if not isinstance(current, Mapping) or not isinstance(next_week, Mapping):
            raise ValueError(f"weekly[{position}] lacks current/next estimates")
        memberships = current.get("memberships")
        probabilities = next_week.get("probabilities")
        if not isinstance(memberships, Mapping) or not isinstance(
            probabilities, Mapping
        ):
            raise ValueError(f"weekly[{position}] lacks membership/forecast mappings")
        rows.append(
            {
                "origin_date": str(week["data_as_of"]),
                "current_state": str(current["state"]),
                **{
                    f"current_m_{state}": float(memberships[state])
                    for state in STATE_ORDER
                },
                "target_date": str(next_week["date"]),
                "model": str(next_week["model"]),
                **{
                    f"next_p_{state}": float(probabilities[state])
                    for state in STATE_ORDER
                },
                "fallback": bool(next_week["fallback"]),
                "fallback_reason": str(next_week["fallback_reason"]),
            }
        )
    return pd.DataFrame(rows, columns=WEEKLY_STATE_FORECAST_V5_COLUMNS)


__all__ = [
    "STATE_LABEL_HISTORY_COLUMNS",
    "STATE_MEMBERSHIP_HISTORY_COLUMNS",
    "WEEKLY_STATE_FORECAST_COLUMNS",
    "WEEKLY_STATE_FORECAST_V5_COLUMNS",
    "canonical_evidence_csv_bytes",
    "evidence_csv_sha256",
    "state_label_history",
    "state_membership_history",
    "weekly_state_forecasts",
    "weekly_state_forecasts_v5",
]
