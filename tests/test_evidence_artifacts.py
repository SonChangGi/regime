from __future__ import annotations

from dataclasses import fields
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from regime_lab import cli
from regime_lab.analysis import BenchmarkProfile, BenchmarkResult
from regime_lab.evidence import (
    STATE_LABEL_HISTORY_COLUMNS,
    STATE_MEMBERSHIP_HISTORY_COLUMNS,
    WEEKLY_STATE_FORECAST_COLUMNS,
    WEEKLY_STATE_FORECAST_V5_COLUMNS,
    canonical_evidence_csv_bytes,
    state_label_history,
    state_membership_history,
    weekly_state_forecasts,
    weekly_state_forecasts_v5,
)


def _weekly_rows() -> list[dict[str, object]]:
    return [
        {
            "date": "2026-08-07",
            "data_as_of": "2026-08-07T20:00:00+00:00",
            "current": {
                "state": "risk_on",
                "probabilities": {
                    "risk_on": 0.72,
                    "transition": 0.2,
                    "risk_off": 0.08,
                },
            },
            "next_week": {
                "date": "2026-08-14",
                "model": "ridge_logistic",
                "probabilities": {
                    "risk_on": 0.51,
                    "transition": 0.31,
                    "risk_off": 0.18,
                },
                "fallback": True,
                "fallback_reason": "arbitrary champion fit fallback",
            },
        },
        {
            "date": "2026-08-14",
            "data_as_of": "2026-08-14T20:00:00+00:00",
            "current": {
                "state": "transition",
                "probabilities": {
                    "risk_on": 0.25,
                    "transition": 0.6,
                    "risk_off": 0.15,
                },
            },
            "next_week": {
                "date": "2026-08-21",
                "model": "causal_dynamic_ensemble",
                "probabilities": {
                    "risk_on": 0.35,
                    "transition": 0.45,
                    "risk_off": 0.2,
                },
                "fallback": False,
                "fallback_reason": "",
            },
        },
    ]


def _label_frame() -> pd.DataFrame:
    index = pd.date_range("2026-07-31", periods=3, freq="W-FRI", tz="UTC")
    states = pd.Series(
        ["transition", "risk_on", "transition"], index=index, dtype="object"
    )
    probabilities = pd.DataFrame(
        [
            [0.2, 0.6, 0.2],
            [0.8000000000000123, 0.15, 0.0499999999999877],
            [0.3, 0.55, 0.15],
        ],
        index=index,
        columns=("risk_on", "transition", "risk_off"),
    )
    scores = pd.Series([0.0, 1.5, 0.5], index=index)
    return state_label_history(
        states,
        probabilities,
        scores,
        lower_threshold=-1.0,
        upper_threshold=1.0,
        hysteresis_fraction=0.15,
        probability_temperature=0.75,
    )


def test_weekly_evidence_has_exact_published_week_parity_and_arbitrary_model() -> None:
    weekly = _weekly_rows()
    frame = weekly_state_forecasts(weekly)

    assert tuple(frame.columns) == WEEKLY_STATE_FORECAST_COLUMNS
    assert len(frame) == len(weekly)
    for position, source in enumerate(weekly):
        current = source["current"]
        next_week = source["next_week"]
        row = frame.iloc[position]
        assert row["origin_date"] == source["data_as_of"]
        assert row["current_state"] == current["state"]
        assert row["target_date"] == next_week["date"]
        assert row["model"] == next_week["model"]
        assert bool(row["fallback"]) is next_week["fallback"]
        assert row["fallback_reason"] == next_week["fallback_reason"]
        for state in ("risk_on", "transition", "risk_off"):
            assert row[f"current_p_{state}"] == current["probabilities"][state]
            assert row[f"next_p_{state}"] == next_week["probabilities"][state]

    # The producer must serialize the model actually selected for the latest
    # row, not assume that only markov/xgboost structural names are possible.
    single = weekly_state_forecasts([weekly[0]])
    assert single.iloc[-1]["model"] == "ridge_logistic"


def test_state_label_history_preserves_full_precision_and_prior_chain() -> None:
    frame = _label_frame()

    assert tuple(frame.columns) == STATE_LABEL_HISTORY_COLUMNS
    assert pd.isna(frame.iloc[0]["previous_state"])
    assert frame.iloc[1]["previous_state"] == frame.iloc[0]["state"]
    assert frame.iloc[2]["previous_state"] == frame.iloc[1]["state"]
    assert frame.iloc[1]["p_risk_on"] == 0.8000000000000123
    assert frame["hysteresis_margin"].unique().tolist() == [0.3]


def test_v5_evidence_renames_anchor_values_without_changing_them() -> None:
    labels = _label_frame()
    memberships = state_membership_history(labels)

    assert tuple(memberships.columns) == STATE_MEMBERSHIP_HISTORY_COLUMNS
    assert memberships["m_risk_on"].tolist() == labels["p_risk_on"].tolist()
    assert memberships["m_transition"].tolist() == labels["p_transition"].tolist()
    assert memberships["m_risk_off"].tolist() == labels["p_risk_off"].tolist()
    assert "probability_temperature" not in memberships
    assert memberships["membership_temperature"].tolist() == labels[
        "probability_temperature"
    ].tolist()


def test_v5_forecast_evidence_separates_membership_from_forecast_probability() -> None:
    weekly = _weekly_rows()
    for row in weekly:
        current = row["current"]
        current["memberships"] = current.pop("probabilities")
    frame = weekly_state_forecasts_v5(weekly)

    assert tuple(frame.columns) == WEEKLY_STATE_FORECAST_V5_COLUMNS
    assert frame.iloc[0]["current_m_risk_on"] == 0.72
    assert frame.iloc[0]["next_p_risk_on"] == 0.51
    assert not any(column.startswith("current_p_") for column in frame.columns)


def test_cli_writes_the_same_canonical_bytes_used_for_payload_hash(tmp_path: Path) -> None:
    label_frame = _label_frame()
    forecast_frame = weekly_state_forecasts(_weekly_rows())
    benchmark = SimpleNamespace(
        leaderboard=pd.DataFrame({"model": ["markov"]}),
        predictions=pd.DataFrame({"model": ["markov"]}),
        split_audit=pd.DataFrame({"origin_date": []}),
        profile=BenchmarkProfile.quick(),
        state_label_history=label_frame,
        weekly_state_forecasts=forecast_frame,
    )
    output = tmp_path / "latest"

    cli._write_supporting_results(benchmark, output)

    expected = {
        "state-label-history.csv": (
            label_frame,
            STATE_LABEL_HISTORY_COLUMNS,
        ),
        "weekly-state-forecasts.csv": (
            forecast_frame,
            WEEKLY_STATE_FORECAST_COLUMNS,
        ),
    }
    for filename, (frame, columns) in expected.items():
        raw = (output / filename).read_bytes()
        canonical = canonical_evidence_csv_bytes(frame, columns)
        assert raw == canonical
        assert hashlib.sha256(raw).hexdigest() == hashlib.sha256(
            canonical
        ).hexdigest()


def test_benchmark_result_exposes_both_canonical_evidence_fields() -> None:
    names = {field.name for field in fields(BenchmarkResult)}
    assert {"state_label_history", "weekly_state_forecasts"} <= names
