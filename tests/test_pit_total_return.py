from __future__ import annotations

from datetime import timedelta

import pandas as pd
from pandas.testing import assert_series_equal
import pytest

from regime_lab.analysis.pit_total_return import (
    CORPORATE_ACTION_CONTRACT,
    PITTotalReturnResult,
    build_pit_total_return_panel,
    reconstruct_pit_total_return,
)
from regime_lab.operating_contract import load_operating_contract


def _inputs(rows: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2024-01-05", periods=rows, freq="W-FRI")
    first_seen = pd.date_range(
        "2024-01-05 21:30:00+00:00", periods=rows, freq="W-FRI"
    )
    decision = pd.Series(
        pd.date_range("2024-01-06 12:00:00+00:00", periods=rows, freq="W-FRI"),
        index=index,
    )
    frame = pd.DataFrame(
        {
            "raw_close": [100.0, 102.0, 51.0, 52.0, 53.0][:rows],
            "dividend_amount": [0.0, 1.0, 0.0, 0.5, 0.0][:rows],
            "split_coefficient": [1.0, 1.0, 2.0, 1.0, 1.0][:rows],
            "corporate_action_contract": [CORPORATE_ACTION_CONTRACT] * rows,
            "source_released_at": first_seen - timedelta(minutes=15),
            "provider_first_seen_at": first_seen,
            "system_retrieved_at": first_seen + timedelta(minutes=1),
            "revision_seq": [0] * rows,
            "raw_sha256": [f"{position + 1:064x}" for position in range(rows)],
        },
        index=index,
    )
    return frame, decision


def test_pit_total_return_handles_dividend_and_split_without_price_jump() -> None:
    frame, decisions = _inputs()
    result = reconstruct_pit_total_return(
        frame,
        decision_at=decisions,
        evidence_track="operational_oos",
    )

    expected = pd.Series(
        [100.0, 103.0, 103.0, 106.02941176470588, 108.06843891402715],
        index=frame.index,
        name="pit_total_return",
    )
    assert_series_equal(result.total_return_index, expected)
    assert result.audit["operational_eligible"].all()
    assert len(result.input_snapshot_sha256) == 64


def test_reconstruction_relaxes_first_seen_but_not_source_release() -> None:
    frame, decisions = _inputs()
    frame.loc[frame.index[2], "provider_first_seen_at"] = decisions.iloc[2] + timedelta(
        days=2
    )
    frame.loc[frame.index[2], "system_retrieved_at"] = decisions.iloc[2] + timedelta(
        days=2, minutes=1
    )

    with pytest.raises(ValueError, match="not available by decision_at"):
        reconstruct_pit_total_return(
            frame,
            decision_at=decisions,
            evidence_track="operational_oos",
        )

    replay = reconstruct_pit_total_return(
        frame,
        decision_at=decisions,
        evidence_track="reconstructed_oos",
    )
    assert replay.audit["operational_eligible"].tolist() == [
        True,
        True,
        False,
        True,
        True,
    ]
    assert replay.audit["reconstructed_eligible"].all()

    source_late = frame.copy()
    source_late.loc[source_late.index[1], "source_released_at"] = (
        decisions.iloc[1] + timedelta(minutes=1)
    )
    source_late.loc[source_late.index[1], "system_retrieved_at"] = (
        decisions.iloc[1] + timedelta(minutes=2)
    )
    with pytest.raises(ValueError, match="source was not released"):
        reconstruct_pit_total_return(
            source_late,
            decision_at=decisions,
            evidence_track="reconstructed_oos",
        )


def test_future_rows_cannot_change_an_unchanged_pit_prefix() -> None:
    frame, decisions = _inputs(4)
    prefix = reconstruct_pit_total_return(
        frame.iloc[:3],
        decision_at=decisions.iloc[:3],
        evidence_track="operational_oos",
    )
    frame.loc[frame.index[-1], "raw_close"] = 5000.0
    extended = reconstruct_pit_total_return(
        frame,
        decision_at=decisions,
        evidence_track="operational_oos",
    )
    assert_series_equal(
        prefix.total_return_index,
        extended.total_return_index.iloc[:3],
    )


def test_pit_evidence_vocabulary_and_panel_lineage_are_bound() -> None:
    contract_tracks = tuple(
        load_operating_contract().document["forecast"]["evidence_tracks"]
    )
    assert contract_tracks == ("operational_oos", "reconstructed_oos")

    frame, decisions = _inputs()
    operational = reconstruct_pit_total_return(
        frame,
        decision_at=decisions,
        evidence_track="operational_oos",
    )
    reconstructed = reconstruct_pit_total_return(
        frame,
        decision_at=decisions,
        evidence_track="reconstructed_oos",
    )
    with pytest.raises(ValueError, match="mix evidence tracks"):
        build_pit_total_return_panel(
            {"SPY": operational, "RSP": reconstructed}
        )

    tampered = PITTotalReturnResult(
        total_return_index=operational.total_return_index,
        audit=operational.audit,
        input_snapshot_sha256="0" * 64,
        evidence_track="operational_oos",
    )
    with pytest.raises(ValueError, match="snapshot hash"):
        build_pit_total_return_panel({"SPY": tampered})


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("derived_index", "corporate-action inputs"),
        ("period_return", "period return"),
        ("eligibility", "eligibility"),
    ],
)
def test_pit_panel_recomputes_derived_economics_and_eligibility(
    mutation: str,
    message: str,
) -> None:
    frame, decisions = _inputs()
    result = reconstruct_pit_total_return(
        frame,
        decision_at=decisions,
        evidence_track="operational_oos",
    )
    if mutation == "derived_index":
        result.total_return_index.iloc[-1] = 999.0
        result.audit.loc[result.audit.index[-1], "total_return_index"] = 999.0
    elif mutation == "period_return":
        result.audit.loc[result.audit.index[-1], "period_return"] = 9.0
    else:
        result.audit.loc[result.audit.index[-1], "operational_eligible"] = False

    with pytest.raises(ValueError, match=message):
        build_pit_total_return_panel({"SPY": result})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("raw_close", 0.0, "raw closes"),
        ("dividend_amount", -0.1, "dividends"),
        ("split_coefficient", 0.0, "splits"),
        ("corporate_action_contract", "ambiguous", "normalization contract"),
        ("raw_sha256", "not-a-hash", "raw_sha256"),
    ],
)
def test_pit_total_return_fails_closed_on_invalid_event_contract(
    field: str,
    value: object,
    message: str,
) -> None:
    frame, decisions = _inputs()
    frame.loc[frame.index[1], field] = value
    with pytest.raises((ValueError, KeyError), match=message):
        reconstruct_pit_total_return(
            frame,
            decision_at=decisions,
            evidence_track="operational_oos",
        )
