from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
import sqlite3
import stat

import pytest

from regime_lab.data import AsOfValue, HealthStatus
from regime_lab.forecast_ledger import (
    ConflictingForecastError,
    DuplicateForecastError,
    ForecastLedger,
    ForecastLedgerEntry,
    OperationalInput,
    operational_input_manifest_sha256,
)


UTC = timezone.utc
SOURCE_FINAL = datetime(2026, 8, 14, 20, 15, tzinfo=UTC)
FIRST_SEEN = datetime(2026, 8, 20, 1, 33, 35, tzinfo=UTC)
DECISION = datetime(2026, 8, 21, 20, 0, tzinfo=UTC)
TARGET = datetime(2026, 8, 28, 20, 0, tzinfo=UTC)


def _operational_input(
    *,
    source: str = "alpha_vantage",
    series_id: str = "SPY.adjusted_close",
    first_seen: datetime = FIRST_SEEN,
    raw_sha256: str = "d" * 64,
) -> OperationalInput:
    return OperationalInput(
        source=source,
        series_id=series_id,
        observed_period_end=date(2026, 8, 14),
        source_released_at=SOURCE_FINAL,
        provider_first_seen_at=first_seen,
        system_retrieved_at=first_seen,
        revision_seq=0,
        raw_sha256=raw_sha256,
    )


def _entry(
    *,
    decision_at: datetime = DECISION,
    target_at: datetime = TARGET,
    operational_inputs: tuple[OperationalInput, ...] | None = None,
) -> ForecastLedgerEntry:
    inputs = operational_inputs or (_operational_input(),)
    return ForecastLedgerEntry(
        origin_week=date(2026, 8, 14),
        decision_at=decision_at,
        target_at=target_at,
        label_spec_sha256="a" * 64,
        model_manifest_sha256="b" * 64,
        input_snapshot_sha256="c" * 64,
        operational_inputs=inputs,
        forecast={
            "predicted_state": "risk_on",
            "probabilities": {
                "risk_off": 0.2,
                "risk_on": 0.5,
                "transition": 0.3,
            },
        },
    )


def test_operational_input_preserves_the_exact_asof_revision_identity() -> None:
    value = AsOfValue(
        cutoff=DECISION,
        source="alpha_vantage",
        series_id="SPY.adjusted_close",
        value=643.0,
        observed_period_end=date(2026, 8, 14),
        released_at=SOURCE_FINAL,
        source_released_at=SOURCE_FINAL,
        available_at=FIRST_SEEN,
        provider_first_seen_at=FIRST_SEEN,
        system_retrieved_at=FIRST_SEEN,
        vintage_date=FIRST_SEEN.date(),
        revision_seq=3,
        raw_sha256="9" * 64,
        age_days=7,
        release_lag_days=6,
        is_filled=True,
        quality_status=HealthStatus.OK,
    )

    operational = OperationalInput.from_asof_value(value)

    assert operational.observed_period_end == value.observed_period_end
    assert operational.source_released_at == value.source_released_at
    assert operational.provider_first_seen_at == value.provider_first_seen_at
    assert operational.system_retrieved_at == value.system_retrieved_at
    assert operational.revision_seq == 3
    assert operational.raw_sha256 == "9" * 64


def test_forecast_ledger_roundtrip_is_append_only_and_private(tmp_path: Path) -> None:
    path = tmp_path / "ledger.sqlite3"
    entry = _entry()

    with ForecastLedger(path, clock=lambda: DECISION) as ledger:
        ledger.append(entry)
        assert ledger.read(entry.key) == entry
        assert ledger.list_entries() == (entry,)

        with pytest.raises(DuplicateForecastError, match="already exists"):
            ledger.append(entry)
        with pytest.raises(ConflictingForecastError, match="different content"):
            ledger.append(
                replace(
                    entry,
                    forecast={
                        "predicted_state": "risk_off",
                        "probabilities": {
                            "risk_off": 0.6,
                            "risk_on": 0.2,
                            "transition": 0.2,
                        },
                    },
                )
            )

        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            ledger._connection.execute(
                "UPDATE forecast_ledger SET forecast_json = '{}'"
            )
        ledger._connection.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            ledger._connection.execute("DELETE FROM forecast_ledger")
        ledger._connection.rollback()
        assert ledger.list_entries() == (entry,)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_public_summary_hashes_only_ordered_primary_keys_and_can_bind_pending() -> None:
    first = _entry()
    second = replace(
        _entry(
            decision_at=datetime(2026, 8, 28, 20, 0, tzinfo=UTC),
            target_at=datetime(2026, 9, 4, 20, 0, tzinfo=UTC),
        ),
        origin_week=date(2026, 8, 21),
    )
    with ForecastLedger(":memory:", clock=lambda: DECISION) as ledger:
        empty = ledger.public_summary()
        pending = ledger.public_summary(pending_key=first.key)
        ledger.append(first)
        recorded = ledger.public_summary()
        two = ledger.public_summary(pending_key=second.key)

    assert empty["entry_count"] == 0
    assert pending == recorded
    assert recorded["entry_count"] == 1
    assert recorded["hash_scope"] == "ordered_ledger_primary_keys_only"
    assert two["entry_count"] == 2
    assert two["key_manifest_sha256"] != recorded["key_manifest_sha256"]


def test_2026_08_14_input_cannot_be_claimed_before_its_first_seen_clock() -> None:
    with pytest.raises(
        ValueError,
        match="provider_first_seen_at exceeds decision_at",
    ):
        _entry(
            decision_at=datetime(2026, 8, 14, 20, 0, tzinfo=UTC),
            target_at=datetime(2026, 8, 21, 20, 0, tzinfo=UTC),
        )

    accepted = _entry()
    assert accepted.operational_inputs[0].operating_available_at == FIRST_SEEN
    assert accepted.operational_inputs[0].provider_first_seen_at <= accepted.decision_at


def test_forecast_ledger_rejects_non_forward_target_and_future_system_retrieval() -> None:
    with pytest.raises(ValueError, match="strictly before"):
        _entry(target_at=DECISION)

    future_retrieval = _operational_input(
        first_seen=datetime(2026, 8, 22, 0, 0, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match="provider_first_seen_at exceeds decision_at"):
        _entry(operational_inputs=(future_retrieval,))


def test_operational_input_manifest_hash_is_order_independent_and_clock_bound() -> None:
    first = _operational_input()
    second = _operational_input(
        source="alfred",
        series_id="UNRATE",
        raw_sha256="e" * 64,
    )

    forward = operational_input_manifest_sha256((first, second))
    reversed_order = operational_input_manifest_sha256((second, first))
    later_first_seen = operational_input_manifest_sha256(
        (
            replace(
                first,
                provider_first_seen_at=FIRST_SEEN.replace(second=36),
                system_retrieved_at=FIRST_SEEN.replace(second=36),
            ),
            second,
        )
    )

    assert forward == reversed_order
    assert forward != later_first_seen
    assert len(forward) == 64


def test_forecast_ledger_validates_all_hashes_and_key_timestamps() -> None:
    with pytest.raises(ValueError, match="raw_sha256"):
        _operational_input(raw_sha256="not-a-hash")
    with pytest.raises(ValueError, match="label_spec_sha256"):
        replace(_entry(), label_spec_sha256="not-a-hash")
    with pytest.raises(ValueError, match="origin_week"):
        replace(_entry(), origin_week=date(2026, 8, 22))
