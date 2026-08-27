from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
import sqlite3

import pytest

from regime_lab.data import (
    HealthStatus,
    Observation,
    REDACTED,
    SnapshotMode,
    SQLiteSnapshotStore,
    SnapshotProvenance,
    plan_incremental_realtime_window,
)


UTC = timezone.utc


def test_read_only_snapshot_store_does_not_initialize_or_mutate_database(
    tmp_path: Path,
) -> None:
    path = tmp_path / "data.sqlite3"
    with SQLiteSnapshotStore(path):
        pass
    before = (path.stat().st_size, path.stat().st_mtime_ns, path.read_bytes())

    with SQLiteSnapshotStore(path, read_only=True) as store:
        assert store.list_provenance() == ()
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            store._connection.execute("CREATE TABLE forbidden(value INTEGER)")

    after = (path.stat().st_size, path.stat().st_mtime_ns, path.read_bytes())
    assert after == before


def test_schema_migration_runs_once_and_records_user_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "data.sqlite3"
    with SQLiteSnapshotStore(path) as store:
        assert store._connection.execute("PRAGMA user_version").fetchone()[0] == 1

    def forbidden(_self: SQLiteSnapshotStore) -> None:
        pytest.fail("completed bitemporal migration must not run again")

    monkeypatch.setattr(
        SQLiteSnapshotStore,
        "_ensure_bitemporal_observation_columns",
        forbidden,
    )
    with SQLiteSnapshotStore(path) as store:
        assert store._connection.execute("PRAGMA user_version").fetchone()[0] == 1


def test_snapshot_store_roundtrip_preserves_pit_fields_and_redacts_secrets(
    tmp_path: Path,
) -> None:
    path = tmp_path / "data.sqlite3"
    available = datetime(2024, 1, 5, 18, tzinfo=UTC)
    retrieved = datetime(2024, 2, 1, 12, tzinfo=UTC)
    record = Observation(
        source="alfred",
        series_id="UNRATE",
        observed_period_end=date(2023, 12, 31),
        released_at=available,
        available_at=available,
        vintage_date=date(2024, 1, 5),
        retrieved_at=retrieved,
        revision_seq=2,
        value=3.7,
        units="percent",
        adjustment="seasonally_adjusted",
        license_class="permission_required",
        quality_status=HealthStatus.OK,
        raw_sha256="abc123",
        metadata={"password": "metadata-secret", "domain": "labor"},
    )
    provenance = SnapshotProvenance(
        source="alfred",
        dataset="series_observations",
        cutoff=datetime(2024, 1, 26, 21, tzinfo=UTC),
        requested_at=retrieved,
        retrieved_at=retrieved,
        quality_status=HealthStatus.OK,
        license_class="permission_required",
        request_params={
            "series_id": "UNRATE",
            "api_key": "database-secret",
            "url": "https://example.test?token=url-secret&x=1",
        },
        response_sha256="response-hash",
        issues=("failed url api_key=issue-secret",),
    )

    with SQLiteSnapshotStore(path) as store:
        snapshot_id = store.write_snapshot([record], provenance)
        loaded = store.read_observations(snapshot_id=snapshot_id)
        loaded_provenance = store.get_provenance(snapshot_id)

    assert len(loaded) == 1
    roundtrip = loaded[0]
    assert roundtrip.source == record.source
    assert roundtrip.series_id == record.series_id
    assert roundtrip.observed_period_end == record.observed_period_end
    assert roundtrip.released_at == record.released_at
    assert roundtrip.available_at == record.available_at
    assert roundtrip.vintage_date == record.vintage_date
    assert roundtrip.retrieved_at == record.retrieved_at
    assert roundtrip.revision_seq == record.revision_seq
    assert roundtrip.value == record.value
    assert roundtrip.units == record.units
    assert roundtrip.quality_status == record.quality_status
    assert roundtrip.raw_sha256 == record.raw_sha256
    assert roundtrip.metadata["password"] == REDACTED

    assert loaded_provenance is not None
    assert loaded_provenance.request_params["api_key"] == REDACTED
    assert "url-secret" not in loaded_provenance.request_params["url"]
    database_bytes = path.read_bytes()
    assert b"database-secret" not in database_bytes
    assert b"metadata-secret" not in database_bytes
    assert b"url-secret" not in database_bytes
    assert b"issue-secret" not in database_bytes


def test_last_good_series_filter_is_pushed_into_sql_with_exact_parity(
    tmp_path: Path,
) -> None:
    retrieved = datetime(2024, 2, 1, tzinfo=UTC)
    base = Observation(
        source="alfred",
        series_id="SERIES_A",
        observed_period_end=date(2023, 12, 31),
        released_at=retrieved,
        available_at=retrieved,
        vintage_date=retrieved.date(),
        retrieved_at=retrieved,
        revision_seq=0,
        value=1.0,
        raw_sha256="a",
    )
    provenance = SnapshotProvenance(
        source="alfred",
        dataset="panel",
        cutoff=retrieved,
        requested_at=retrieved,
        retrieved_at=retrieved,
        quality_status=HealthStatus.OK,
        request_params={"snapshot_mode": SnapshotMode.FULL.value},
        response_sha256="response",
    )
    path = tmp_path / "data.sqlite3"
    with SQLiteSnapshotStore(path) as store:
        store.write_snapshot(
            [
                base,
                replace(base, series_id="SERIES_B", value=2.0, raw_sha256="b"),
            ],
            provenance,
        )
        statements: list[str] = []
        store._connection.set_trace_callback(statements.append)
        filtered = store.read_last_good_observations(
            source="alfred",
            series_ids=("SERIES_A",),
        )
        store._connection.set_trace_callback(None)
        complete = store.read_last_good_observations()

    assert filtered == tuple(row for row in complete if row.series_id == "SERIES_A")
    assert any("source = 'alfred'" in statement for statement in statements)
    assert any("o.series_id IN" in statement for statement in statements)


def test_store_available_as_of_filter_is_inclusive_and_leakage_safe(tmp_path: Path) -> None:
    retrieved = datetime(2024, 2, 1, tzinfo=UTC)
    first_time = datetime(2024, 1, 5, tzinfo=UTC)
    second_time = datetime(2024, 1, 12, tzinfo=UTC)
    records = tuple(
        Observation(
            source="alfred",
            series_id="SERIES",
            observed_period_end=date(2023, 12, 31),
            value=value,
            released_at=available,
            available_at=available,
            vintage_date=available.date(),
            retrieved_at=retrieved,
            revision_seq=index,
            raw_sha256=f"hash-{index}",
        )
        for index, (value, available) in enumerate(((1.0, first_time), (2.0, second_time)))
    )
    provenance = SnapshotProvenance(
        source="alfred",
        dataset="fixture",
        cutoff=second_time,
        requested_at=retrieved,
        retrieved_at=retrieved,
        quality_status=HealthStatus.OK,
    )

    with SQLiteSnapshotStore(tmp_path / "data.sqlite3") as store:
        store.write_snapshot(records, provenance)
        at_first = store.read_observations(available_as_of=first_time)

    assert [record.value for record in at_first] == [1.0]


def test_last_good_reader_ignores_newer_failed_snapshot_and_advances_on_success(
    tmp_path: Path,
) -> None:
    path = tmp_path / "last-good.sqlite3"
    available = datetime(2024, 1, 5, tzinfo=UTC)

    def record(value: float, retrieved: datetime) -> Observation:
        return Observation(
            source="alfred",
            series_id="SERIES",
            observed_period_end=date(2023, 12, 31),
            value=value,
            released_at=available,
            available_at=available,
            vintage_date=available.date(),
            retrieved_at=retrieved,
            raw_sha256=f"hash-{value}",
        )

    def provenance(
        status: HealthStatus,
        retrieved: datetime,
    ) -> SnapshotProvenance:
        return SnapshotProvenance(
            source="alfred",
            dataset="SERIES",
            cutoff=available,
            requested_at=retrieved,
            retrieved_at=retrieved,
            quality_status=status,
        )

    first_time = datetime(2024, 2, 1, tzinfo=UTC)
    failed_time = datetime(2024, 2, 2, tzinfo=UTC)
    second_time = datetime(2024, 2, 3, tzinfo=UTC)
    with SQLiteSnapshotStore(path) as store:
        first_id = store.write_snapshot(
            [record(1.0, first_time)],
            provenance(HealthStatus.OK, first_time),
        )
        store.write_snapshot(
            [record(999.0, failed_time)],
            provenance(HealthStatus.SCHEMA_CHANGED, failed_time),
        )

        last_good = store.read_last_good_observations()
        assert [item.value for item in last_good] == [1.0]
        assert store.get_last_good_provenance(
            source="alfred", dataset="SERIES"
        ).snapshot_id == first_id

        second_id = store.write_snapshot(
            [record(2.0, second_time)],
            provenance(HealthStatus.OK, second_time),
        )
        advanced = store.read_last_good_observations(
            source="alfred",
            dataset="SERIES",
        )
        latest_provenance = store.get_last_good_provenance(
            source="alfred", dataset="SERIES"
        )

    assert [item.value for item in advanced] == [2.0]
    assert latest_provenance is not None
    assert latest_provenance.snapshot_id == second_id


def test_last_good_reader_merges_full_and_deltas_with_overlap_and_failure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "incremental.sqlite3"
    cutoff = datetime(2024, 1, 26, 21, tzinfo=UTC)
    first_time = datetime(2024, 2, 1, tzinfo=UTC)
    second_time = datetime(2024, 2, 2, tzinfo=UTC)
    failed_time = datetime(2024, 2, 3, tzinfo=UTC)
    empty_time = datetime(2024, 2, 4, tzinfo=UTC)

    def observation(
        *,
        period_end: date,
        vintage: date,
        value: float,
        retrieved: datetime,
        raw_hash: str,
        local_revision: int = 0,
    ) -> Observation:
        available = datetime(
            vintage.year,
            vintage.month,
            vintage.day,
            23,
            tzinfo=UTC,
        )
        return Observation(
            source="alfred",
            series_id="SERIES",
            observed_period_end=period_end,
            value=value,
            released_at=available,
            available_at=available,
            vintage_date=vintage,
            retrieved_at=retrieved,
            revision_seq=local_revision,
            raw_sha256=raw_hash,
        )

    def provenance(
        *,
        retrieved: datetime,
        status: HealthStatus,
        mode: SnapshotMode,
        start: str,
        end: str,
    ) -> SnapshotProvenance:
        return SnapshotProvenance(
            source="alfred",
            dataset="SERIES",
            cutoff=cutoff,
            requested_at=retrieved,
            retrieved_at=retrieved,
            quality_status=status,
            request_params={
                "snapshot_mode": mode.value,
                "realtime_start": start,
                "realtime_end": end,
                "observation_start": "2006-01-01",
            },
        )

    base_records = (
        observation(
            period_end=date(2023, 12, 31),
            vintage=date(2024, 1, 5),
            value=1.0,
            retrieved=first_time,
            raw_hash="base-initial",
        ),
        observation(
            period_end=date(2024, 1, 7),
            vintage=date(2024, 1, 12),
            value=2.0,
            retrieved=first_time,
            raw_hash="base-overlap",
        ),
    )
    delta_records = (
        # Inclusive overlap: later successful snapshot wins for this event.
        observation(
            period_end=date(2024, 1, 7),
            vintage=date(2024, 1, 12),
            value=2.1,
            retrieved=second_time,
            raw_hash="delta-overlap-corrected",
        ),
        # A current realtime event revising an older observation period must be
        # retained even though its observed period predates the delta window.
        observation(
            period_end=date(2023, 12, 31),
            vintage=date(2024, 1, 19),
            value=1.1,
            retrieved=second_time,
            raw_hash="historical-revision",
        ),
        observation(
            period_end=date(2024, 1, 14),
            vintage=date(2024, 1, 19),
            value=3.0,
            retrieved=second_time,
            raw_hash="new-period",
        ),
    )
    failed_record = observation(
        period_end=date(2024, 1, 14),
        vintage=date(2024, 1, 26),
        value=999.0,
        retrieved=failed_time,
        raw_hash="failed-partial",
    )

    with SQLiteSnapshotStore(path) as store:
        store.write_snapshot(
            base_records,
            provenance(
                retrieved=first_time,
                status=HealthStatus.OK,
                mode=SnapshotMode.FULL,
                start="2006-01-01",
                end="2024-01-12",
            ),
        )
        store.write_snapshot(
            delta_records,
            provenance(
                retrieved=second_time,
                status=HealthStatus.OK,
                mode=SnapshotMode.DELTA,
                start="2024-01-12",
                end="2024-01-19",
            ),
        )
        store.write_snapshot(
            (failed_record,),
            provenance(
                retrieved=failed_time,
                status=HealthStatus.SCHEMA_CHANGED,
                mode=SnapshotMode.DELTA,
                start="2024-01-19",
                end="2024-01-26",
            ),
        )
        empty_id = store.write_snapshot(
            (),
            provenance(
                retrieved=empty_time,
                status=HealthStatus.OK,
                mode=SnapshotMode.DELTA,
                start="2024-01-19",
                end="2024-01-26",
            ),
        )

        assembled = store.read_last_good_observations(
            source="alfred",
            dataset="SERIES",
        )
        before_revision = store.read_last_good_observations(
            source="alfred",
            dataset="SERIES",
            available_as_of=datetime(2024, 1, 13, tzinfo=UTC),
        )
        latest = store.get_last_good_provenance(
            source="alfred",
            dataset="SERIES",
        )

    assert len(assembled) == 4
    assert 999.0 not in {item.value for item in assembled}
    overlap = [
        item
        for item in assembled
        if item.observed_period_end == date(2024, 1, 7)
        and item.vintage_date == date(2024, 1, 12)
    ]
    assert len(overlap) == 1
    assert overlap[0].value == 2.1
    old_period = [
        item for item in assembled if item.observed_period_end == date(2023, 12, 31)
    ]
    assert [item.value for item in old_period] == [1.0, 1.1]
    assert [item.revision_seq for item in old_period] == [0, 1]
    assert {item.value for item in before_revision} == {1.0, 2.1}
    assert latest is not None and latest.snapshot_id == empty_id


def test_last_good_reader_preserves_prospective_revisions_on_same_vintage_date(
    tmp_path: Path,
) -> None:
    path = tmp_path / "alpha-revisions.sqlite3"
    period_end = date(2024, 1, 5)
    first_available = datetime(2024, 1, 5, 21, tzinfo=UTC)
    first_retrieved = datetime(2024, 2, 1, tzinfo=UTC)
    first_discovery = datetime(2024, 2, 10, 12, tzinfo=UTC)
    second_discovery = datetime(2024, 2, 10, 18, tzinfo=UTC)

    def record(value: float, available: datetime, raw_hash: str) -> Observation:
        return Observation(
            source="alpha_vantage",
            series_id="SPY.adjusted_close",
            observed_period_end=period_end,
            value=value,
            released_at=available,
            available_at=available,
            vintage_date=available.date(),
            retrieved_at=max(available, first_retrieved),
            raw_sha256=raw_hash,
        )

    def provenance(retrieved: datetime, mode: SnapshotMode) -> SnapshotProvenance:
        return SnapshotProvenance(
            source="alpha_vantage",
            dataset="weekly_adjusted_etf",
            cutoff=retrieved,
            requested_at=retrieved,
            retrieved_at=retrieved,
            quality_status=HealthStatus.OK,
            request_params={"snapshot_mode": mode.value},
        )

    with SQLiteSnapshotStore(path) as store:
        store.write_snapshot(
            (record(100.0, first_available, "initial"),),
            provenance(first_retrieved, SnapshotMode.FULL),
        )
        store.write_snapshot(
            (record(90.0, first_discovery, "revision-1"),),
            provenance(first_discovery, SnapshotMode.DELTA),
        )
        store.write_snapshot(
            (record(95.0, second_discovery, "revision-2"),),
            provenance(second_discovery, SnapshotMode.DELTA),
        )
        assembled = store.read_last_good_observations(
            source="alpha_vantage",
            dataset="weekly_adjusted_etf",
        )
        before_discovery = store.read_last_good_observations(
            source="alpha_vantage",
            dataset="weekly_adjusted_etf",
            available_as_of=datetime(2024, 2, 9, tzinfo=UTC),
        )

    assert [item.value for item in assembled] == [100.0, 90.0, 95.0]
    assert [item.revision_seq for item in assembled] == [0, 1, 2]
    assert [item.value for item in before_discovery] == [100.0]


def test_alpha_reader_permanently_quarantines_future_period_by_own_snapshot_cutoff(
    tmp_path: Path,
) -> None:
    path = tmp_path / "alpha-partial.sqlite3"

    def record(
        period_end: date,
        value: float,
        available: datetime,
        retrieved: datetime,
        raw_hash: str,
    ) -> Observation:
        return Observation(
            source="alpha_vantage",
            series_id="SPY.adjusted_close",
            observed_period_end=period_end,
            value=value,
            released_at=available,
            available_at=available,
            vintage_date=available.date(),
            retrieved_at=retrieved,
            raw_sha256=raw_hash,
        )

    def provenance(
        cutoff: datetime,
        retrieved: datetime,
        mode: SnapshotMode,
    ) -> SnapshotProvenance:
        return SnapshotProvenance(
            source="alpha_vantage",
            dataset="weekly_adjusted_etf",
            cutoff=cutoff,
            requested_at=retrieved,
            retrieved_at=retrieved,
            quality_status=HealthStatus.OK,
            request_params={"snapshot_mode": mode.value},
        )

    first_retrieved = datetime(2024, 1, 10, tzinfo=UTC)
    first_cutoff = datetime(2024, 1, 5, 21, tzinfo=UTC)
    second_retrieved = datetime(2024, 1, 14, tzinfo=UTC)
    second_cutoff = datetime(2024, 1, 12, 21, tzinfo=UTC)
    third_retrieved = datetime(2024, 1, 20, tzinfo=UTC)
    third_cutoff = datetime(2024, 1, 19, 21, tzinfo=UTC)

    with SQLiteSnapshotStore(path) as store:
        store.write_snapshot(
            (
                record(
                    date(2024, 1, 1),
                    100.0,
                    datetime(2024, 1, 1, 21, tzinfo=UTC),
                    first_retrieved,
                    "valid-base",
                ),
                # Legacy full response incorrectly stored a period after its
                # own completed-week cutoff.
                record(
                    date(2024, 1, 8),
                    999.0,
                    datetime(2024, 1, 8, 21, tzinfo=UTC),
                    first_retrieved,
                    "legacy-partial",
                ),
            ),
            provenance(first_cutoff, first_retrieved, SnapshotMode.FULL),
        )
        store.write_snapshot(
            (
                # Its period is old enough for this snapshot, so a revision
                # discovered after the cutoff remains in the chain.
                record(
                    date(2024, 1, 1),
                    101.0,
                    datetime(2024, 1, 13, 12, tzinfo=UTC),
                    second_retrieved,
                    "prospective-old-period",
                ),
            ),
            provenance(second_cutoff, second_retrieved, SnapshotMode.DELTA),
        )
        after_cutoff_advanced = store.read_last_good_observations(
            source="alpha_vantage",
            dataset="weekly_adjusted_etf",
        )
        at_second_cutoff = store.read_last_good_observations(
            source="alpha_vantage",
            dataset="weekly_adjusted_etf",
            available_as_of=second_cutoff,
        )
        store.write_snapshot(
            (
                record(
                    date(2024, 1, 8),
                    110.0,
                    datetime(2024, 1, 8, 21, tzinfo=UTC),
                    third_retrieved,
                    "completed-period",
                ),
            ),
            provenance(third_cutoff, third_retrieved, SnapshotMode.DELTA),
        )
        after_completed_delta = store.read_last_good_observations(
            source="alpha_vantage",
            dataset="weekly_adjusted_etf",
        )

    assert [item.value for item in after_cutoff_advanced] == [100.0, 101.0]
    assert [item.value for item in at_second_cutoff] == [100.0]
    assert [item.value for item in after_completed_delta] == [100.0, 101.0, 110.0]
    assert 999.0 not in {item.value for item in after_completed_delta}


def test_incremental_window_repeats_last_successful_day_and_keeps_full_observation_start() -> None:
    last_good = SnapshotProvenance(
        source="alfred",
        dataset="DGS10",
        cutoff=datetime(2026, 8, 7, 20, tzinfo=UTC),
        requested_at=datetime(2026, 8, 11, tzinfo=UTC),
        retrieved_at=datetime(2026, 8, 11, tzinfo=UTC),
        quality_status=HealthStatus.OK,
        request_params={
            "snapshot_mode": SnapshotMode.FULL.value,
            "realtime_start": "2006-01-01",
            "realtime_end": "2026-08-07",
        },
    )

    window = plan_incremental_realtime_window(
        last_good,
        history_start=date(2014, 1, 1),
        realtime_end=date(2026, 8, 14),
        observation_start=date(2006, 1, 1),
    )
    initial = plan_incremental_realtime_window(
        None,
        history_start=date(2006, 1, 1),
        realtime_end=date(2026, 8, 7),
    )

    assert window.snapshot_mode is SnapshotMode.DELTA
    assert window.realtime_start == date(2026, 8, 7)
    assert window.realtime_end == date(2026, 8, 14)
    assert window.observation_start == date(2006, 1, 1)
    assert initial.snapshot_mode is SnapshotMode.FULL
    assert initial.realtime_start == date(2006, 1, 1)


def test_incremental_window_continues_from_type3_vintage_dates() -> None:
    last_good = SnapshotProvenance(
        source="alfred",
        dataset="UNRATE",
        # Deliberately later than the last requested vintage: the provider
        # query, not the orchestration timestamp, defines the next overlap.
        cutoff=datetime(2026, 8, 7, 20, tzinfo=UTC),
        requested_at=datetime(2026, 8, 8, tzinfo=UTC),
        retrieved_at=datetime(2026, 8, 8, tzinfo=UTC),
        quality_status=HealthStatus.OK,
        request_params={
            "snapshot_mode": SnapshotMode.DELTA.value,
            "output_type": 3,
            "vintage_dates": "2026-07-31,2026-08-01",
        },
    )

    window = plan_incremental_realtime_window(
        last_good,
        history_start=date(2006, 1, 1),
        realtime_end=date(2026, 8, 14),
    )

    assert window.snapshot_mode is SnapshotMode.DELTA
    assert window.realtime_start == date(2026, 8, 1)
    assert window.realtime_end == date(2026, 8, 14)
