from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path

import pytest

from regime_lab.data.h10 import (
    DEFAULT_ALLOWED_FX,
    FIXED_BILATERAL_PANEL,
    H10ParseResult,
    ReleaseMetadata,
    SERIES_CATALOG,
)
from regime_lab.h10_store import (
    H10_ARCHIVE_CACHE_DATASET,
    H10_ARCHIVE_DATASET,
    H10_DATASET,
    H10_SOURCE,
    H10StoreError,
    h10_collection_receipt_document,
    ingest_h10_archive_store,
    refresh_h10_archive_store,
    refresh_h10_store,
)
from regime_lab.data.h10_archive import (
    H10ArchiveCollection,
    H10ArchiveRelease,
    archive_release_available_at,
)
from regime_lab.data import HealthStatus, Observation, SQLiteSnapshotStore, SnapshotMode


UTC = timezone.utc
FIRST_RETRIEVAL = datetime(2026, 8, 17, 20, 20, tzinfo=UTC)
SECOND_RETRIEVAL = datetime(2026, 8, 24, 20, 20, tzinfo=UTC)
THIRD_RETRIEVAL = datetime(2026, 8, 31, 20, 20, tzinfo=UTC)
WEEK_ENDS = tuple(
    date(2025, 12, 12) + timedelta(weeks=offset) for offset in range(36)
)


class _FakeH10Client:
    def __init__(self, *outcomes: H10ParseResult | Exception) -> None:
        self.outcomes = list(outcomes)
        self.first_seen_values: list[datetime | None] = []

    def collect(
        self,
        *,
        first_seen_at: datetime | None = None,
    ) -> H10ParseResult:
        self.first_seen_values.append(first_seen_at)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _raw_value(code: str, week_number: int) -> float:
    slopes = {
        "BRD": 0.0010,
        "BRL": 0.0060,
        "AFE": 0.0005,
        "EME": 0.0015,
        "AUD": 0.0020,
        "CAD": -0.0010,
        "CHF": 0.0030,
        "CNY": -0.0020,
        "EUR": 0.0040,
        "GBP": -0.0030,
        "JPY": 0.0050,
        "MXN": -0.0040,
    }
    spec = SERIES_CATALOG[code]
    usd_log_level = slopes[code] * week_number
    return math.exp(usd_log_level / spec.usd_strength_sign)


def _full_response(
    retrieved_at: datetime,
    *,
    revised_period: date | None = None,
) -> H10ParseResult:
    records: list[Observation] = []
    for week_number, period_end in enumerate(WEEK_ENDS):
        for code in DEFAULT_ALLOWED_FX:
            spec = SERIES_CATALOG[code]
            value = _raw_value(code, week_number)
            if code == "EUR" and period_end == revised_period:
                value *= 1.01
            records.append(
                Observation(
                    source="frb_h10",
                    series_id=f"H10|FX={code}|FREQ=9",
                    observed_period_end=period_end,
                    value=value,
                    released_at=retrieved_at - timedelta(minutes=5),
                    available_at=retrieved_at,
                    vintage_date=retrieved_at.date(),
                    retrieved_at=retrieved_at,
                    units="fixture",
                    license_class=(
                        "federal_reserve_board_public_domain_citation_requested"
                    ),
                    raw_sha256=(
                        f"{retrieved_at.date().isoformat()}-{code}-{period_end}"
                    ),
                    metadata={
                        "fx_code": code,
                        "frequency_code": "9",
                        "quote_convention": spec.quote_convention.value,
                        "usd_strength_sign": spec.usd_strength_sign,
                        "obs_status": "A",
                    },
                )
            )
    release = ReleaseMetadata(
        source_url="https://www.federalreserve.gov/releases/h10/data/FRB_h10_xml.zip",
        message_id="H10",
        message_name="Foreign Exchange Rates",
        sender_id="FRB",
        message_prepared_raw=retrieved_at.replace(tzinfo=None).isoformat(),
        source_object_modified_at=retrieved_at - timedelta(minutes=5),
        release_date_et=retrieved_at.date(),
        first_seen_at=retrieved_at,
        retrieved_at=retrieved_at,
        etag='"fixture"',
        snapshot_sha256=(
            ("b" if revised_period is not None else "a") * 64
        ),
    )
    return H10ParseResult(
        release=release,
        series=(),
        records=tuple(records),
    )


def _request_time(retrieved_at: datetime) -> datetime:
    return retrieved_at - timedelta(minutes=1)


def _archive_record(record: Observation) -> Observation:
    code = str(record.metadata["fx_code"])
    available = datetime.combine(
        record.observed_period_end + timedelta(days=3),
        datetime.min.time(),
        tzinfo=UTC,
    ) + timedelta(hours=20, minutes=15)
    return replace(
        record,
        released_at=available,
        available_at=available,
        vintage_date=available.date(),
        retrieved_at=FIRST_RETRIEVAL,
        raw_sha256=f"archive-{code}-{record.observed_period_end}",
        metadata={
            **dict(record.metadata),
            "official_release_archive_ingest": True,
            "archive_chain_availability_basis": (
                "official_archive_release_schedule"
            ),
            "availability_basis": "archived_release_date_16_15_ET",
            "archive_revision_policy": (
                "later_official_release_preserved_as_new_vintage"
            ),
        },
    )


def _archive_release_and_collection() -> tuple[H10ArchiveRelease, H10ArchiveCollection]:
    records = tuple(_archive_record(row) for row in _full_response(FIRST_RETRIEVAL).records)
    event_date = date(2026, 8, 17)
    available = archive_release_available_at(event_date)
    release = H10ArchiveRelease(
        release_date=event_date,
        last_update_date=event_date,
        available_at=available,
        availability_basis="archived_release_date_16_15_ET",
        retrieved_at=FIRST_RETRIEVAL,
        source_url="https://www.federalreserve.gov/releases/h10/20260817/",
        snapshot_sha256="c" * 64,
        records=records,
    )
    collection = H10ArchiveCollection(
        releases=(release,),
        records=records,
        requested_at=_request_time(FIRST_RETRIEVAL),
        retrieved_at=FIRST_RETRIEVAL,
        index_sha256="d" * 64,
        collection_sha256="e" * 64,
        revision_event_count=0,
        discovered_release_dates=(event_date,),
    )
    return release, collection


@pytest.mark.parametrize(
    ("first_seen_at", "message", "collection_expected"),
    (
        (
            _request_time(FIRST_RETRIEVAL) - timedelta(microseconds=1),
            "first_seen_at must not precede requested_at",
            False,
        ),
        (
            FIRST_RETRIEVAL + timedelta(microseconds=1),
            "first_seen_at must not follow retrieved_at",
            True,
        ),
    ),
)
def test_explicit_first_seen_outside_request_window_is_not_written(
    tmp_path: Path,
    first_seen_at: datetime,
    message: str,
    collection_expected: bool,
) -> None:
    client = _FakeH10Client(_full_response(FIRST_RETRIEVAL))

    with SQLiteSnapshotStore(tmp_path / "invalid-first-seen.sqlite3") as store:
        with pytest.raises(H10StoreError, match=message):
            refresh_h10_store(
                store,
                client,
                requested_at=_request_time(FIRST_RETRIEVAL),
                first_seen_at=first_seen_at,
            )

        assert store.list_provenance(source=H10_SOURCE) == ()
        assert store.read_observations(source=H10_SOURCE) == ()

    assert client.first_seen_values == (
        [first_seen_at] if collection_expected else []
    )


def test_explicit_first_seen_with_naive_retrieval_is_not_written(
    tmp_path: Path,
) -> None:
    response = _full_response(FIRST_RETRIEVAL)
    client = _FakeH10Client(
        replace(
            response,
            release=replace(
                response.release,
                retrieved_at=FIRST_RETRIEVAL.replace(tzinfo=None),
            ),
        )
    )

    with SQLiteSnapshotStore(tmp_path / "naive-retrieval.sqlite3") as store:
        with pytest.raises(
            H10StoreError,
            match="retrieved_at must be timezone-aware",
        ):
            refresh_h10_store(
                store,
                client,
                requested_at=_request_time(FIRST_RETRIEVAL),
                first_seen_at=FIRST_RETRIEVAL,
            )

        assert store.list_provenance(source=H10_SOURCE) == ()
        assert store.read_observations(source=H10_SOURCE) == ()

    assert client.first_seen_values == [FIRST_RETRIEVAL]


def test_full_then_zero_delta_then_prospective_revision(tmp_path: Path) -> None:
    database = tmp_path / "h10.sqlite3"
    revised_period = WEEK_ENDS[-4]
    client = _FakeH10Client(
        _full_response(FIRST_RETRIEVAL),
        _full_response(SECOND_RETRIEVAL),
        _full_response(THIRD_RETRIEVAL, revised_period=revised_period),
    )

    with SQLiteSnapshotStore(database) as store:
        first = refresh_h10_store(
            store,
            client,
            requested_at=_request_time(FIRST_RETRIEVAL),
        )
        second = refresh_h10_store(
            store,
            client,
            requested_at=_request_time(SECOND_RETRIEVAL),
        )
        third = refresh_h10_store(
            store,
            client,
            requested_at=_request_time(THIRD_RETRIEVAL),
        )

        assert first.prepared.snapshot_mode is SnapshotMode.FULL
        expected_records = len(WEEK_ENDS) * len(DEFAULT_ALLOWED_FX)
        assert first.prepared.added_count == expected_records
        assert len(store.read_observations(snapshot_id=first.snapshot_id)) == (
            expected_records
        )
        assert first.fx_context["status"] == "ok"

        assert second.prepared.snapshot_mode is SnapshotMode.DELTA
        assert second.prepared.snapshot_result.records == ()
        assert second.prepared.unchanged_count == expected_records
        assert store.read_observations(snapshot_id=second.snapshot_id) == ()

        assert third.prepared.snapshot_mode is SnapshotMode.DELTA
        assert third.prepared.changed_count == 1
        assert third.prepared.added_count == 0
        assert len(store.read_observations(snapshot_id=third.snapshot_id)) == 1

        effective = store.read_last_good_observations(
            source=H10_SOURCE,
            dataset=H10_DATASET,
        )
        revisions = tuple(
            record
            for record in effective
            if record.series_id == "H10|FX=EUR|FREQ=9"
            and record.observed_period_end == revised_period
        )
        provenance = store.list_provenance(source=H10_SOURCE)

    assert len(revisions) == 2
    assert [record.revision_seq for record in revisions] == [0, 1]
    assert revisions[1].available_at == THIRD_RETRIEVAL
    assert revisions[1].released_at == THIRD_RETRIEVAL
    assert revisions[1].metadata["prospective_revision"] is True
    assert revisions[1].metadata["prospective_revision_reason"] == (
        "provider_value_changed"
    )
    assert [row.request_params["snapshot_mode"] for row in provenance] == [
        "full",
        "delta",
        "delta",
    ]
    assert client.first_seen_values == [None, None, None]


def test_failed_refresh_uses_last_good_and_publishes_only_derived_context(
    tmp_path: Path,
) -> None:
    database = tmp_path / "h10-last-good.sqlite3"
    secret = "api_key=do-not-persist-or-publish"
    failure_time = SECOND_RETRIEVAL
    client = _FakeH10Client(
        _full_response(FIRST_RETRIEVAL),
        RuntimeError(secret),
    )

    with SQLiteSnapshotStore(database) as store:
        baseline = refresh_h10_store(
            store,
            client,
            requested_at=_request_time(FIRST_RETRIEVAL),
        )
        failed = refresh_h10_store(
            store,
            client,
            requested_at=_request_time(failure_time),
            clock=lambda: failure_time,
        )
        failed_rows = store.read_observations(snapshot_id=failed.snapshot_id)
        last_good = store.get_last_good_provenance(
            source=H10_SOURCE,
            dataset=H10_DATASET,
        )

    assert failed.prepared.snapshot_mode is SnapshotMode.DELTA
    assert failed.prepared.snapshot_result.health is HealthStatus.DEGRADED
    assert failed_rows == ()
    assert failed.used_last_good is True
    assert len(failed.effective_records) == len(baseline.effective_records)
    assert failed.source_row["status"] == "degraded"
    assert failed.source_row["last_good_used"] is True
    assert failed.fx_context["status"] == "degraded"
    assert failed.fx_context["source_status"] == "degraded"
    assert failed.fx_context["last_good_used"] is True
    assert failed.fx_context["observation_week"] == WEEK_ENDS[-1].isoformat()
    assert failed.fx_context["feature_available_at"] == (
        FIRST_RETRIEVAL.isoformat()
    )
    assert failed.fx_context["coverage"]["available_pairs"] == len(
        FIXED_BILATERAL_PANEL
    )
    assert last_good is not None
    assert last_good.snapshot_id == baseline.snapshot_id

    public_blob = json.dumps(
        {
            "source": failed.source_row,
            "fx_context": failed.fx_context,
        },
        sort_keys=True,
    )
    assert secret not in public_blob
    for forbidden in (
        "raw_sha256",
        "snapshot_sha256",
        "raw_value_token",
        "source_url",
        "weekly_usd_log_levels",
        '"records"',
    ):
        assert forbidden not in public_blob
    assert secret.encode() not in database.read_bytes()


def test_successful_refresh_marks_old_observation_stale(tmp_path: Path) -> None:
    client = _FakeH10Client(_full_response(FIRST_RETRIEVAL))

    with SQLiteSnapshotStore(tmp_path / "h10-stale.sqlite3") as store:
        stale = refresh_h10_store(
            store,
            client,
            requested_at=_request_time(FIRST_RETRIEVAL),
            as_of=THIRD_RETRIEVAL,
        )

    assert stale.prepared.snapshot_result.health is HealthStatus.OK
    assert stale.used_last_good is False
    assert stale.source_row["status"] == "stale"
    assert stale.source_row["last_good_used"] is False
    assert stale.fx_context["status"] == "stale"
    assert stale.fx_context["source_status"] == "stale"
    assert stale.fx_context["observation_week"] == WEEK_ENDS[-1].isoformat()
    assert stale.fx_context["observation_age_days"] == 17
    assert stale.fx_context["maximum_age_days"] == 10


def test_failed_initial_refresh_is_unavailable(tmp_path: Path) -> None:
    failure_time = FIRST_RETRIEVAL
    client = _FakeH10Client(TimeoutError("private request detail"))

    with SQLiteSnapshotStore(tmp_path / "empty.sqlite3") as store:
        failed = refresh_h10_store(
            store,
            client,
            requested_at=_request_time(failure_time),
            clock=lambda: failure_time,
        )
        stored = store.read_observations(snapshot_id=failed.snapshot_id)

    assert failed.prepared.snapshot_mode is SnapshotMode.FULL
    assert failed.prepared.snapshot_result.health is HealthStatus.UNAVAILABLE
    assert stored == ()
    assert failed.effective_records == ()
    assert failed.used_last_good is False
    assert failed.source_row["status"] == "unavailable"
    assert failed.fx_context["status"] == "unavailable"
    assert failed.fx_context["coverage"]["available_pairs"] == 0


def test_archive_bootstrap_then_zero_delta_stays_separate_and_derived_only(
    tmp_path: Path,
) -> None:
    _release, first_collection = _archive_release_and_collection()
    requested = first_collection.requested_at
    cutoff = datetime(2026, 8, 21, 20, tzinfo=UTC)

    with SQLiteSnapshotStore(tmp_path / "archive.sqlite3") as store:
        first = ingest_h10_archive_store(
            store,
            first_collection,
            requested_at=requested,
            as_of=cutoff,
        )
        zero_delta = H10ArchiveCollection(
            releases=(),
            records=(),
            requested_at=requested,
            retrieved_at=FIRST_RETRIEVAL,
            index_sha256="f" * 64,
            collection_sha256="0" * 64,
            revision_event_count=0,
            discovered_release_dates=(date(2026, 8, 17),),
        )
        second = ingest_h10_archive_store(
            store,
            zero_delta,
            requested_at=requested,
            as_of=cutoff,
        )
        archive_records = store.read_last_good_observations(
            source=H10_SOURCE,
            dataset=H10_ARCHIVE_DATASET,
        )
        current_xml_records = store.read_last_good_observations(
            source=H10_SOURCE,
            dataset=H10_DATASET,
        )

    assert first.prepared.snapshot_mode is SnapshotMode.FULL
    assert second.prepared.snapshot_mode is SnapshotMode.DELTA
    assert second.prepared.added_count == 0
    assert second.prepared.changed_count == 0
    assert second.effective_records == first.effective_records == archive_records
    assert current_xml_records == ()
    assert second.source_row["official_release_archive_ingest"] is True
    assert second.source_row["archive_release_count"] == 1
    receipt = h10_collection_receipt_document(
        second,
        requested_at=requested,
        as_of=cutoff,
    )
    assert receipt["official_release_archive_ingest"] is True
    assert receipt["archive_release_count"] == 1
    assert receipt["archive_correction_count"] == 0
    encoded = json.dumps(receipt, sort_keys=True)
    for forbidden in ("source_url", "raw_sha256", "snapshot_sha256", '"records"'):
        assert forbidden not in encoded


class _InterruptedArchiveClient:
    def __init__(self, release: H10ArchiveRelease) -> None:
        self.release = release
        self.calls = 0
        self.cached_dates: list[date] = []

    def collect(self, **kwargs) -> H10ArchiveCollection:
        self.calls += 1
        cached = dict(kwargs["cached_releases"])
        self.cached_dates = sorted(cached)
        if self.calls == 1:
            kwargs["on_release"](self.release)
            raise KeyboardInterrupt
        assert {
            row.raw_sha256
            for row in cached[self.release.last_update_date].records
        } == {row.raw_sha256 for row in self.release.records}
        return H10ArchiveCollection(
            releases=(cached[self.release.last_update_date],),
            records=cached[self.release.last_update_date].records,
            requested_at=kwargs["requested_at"],
            retrieved_at=FIRST_RETRIEVAL,
            index_sha256="1" * 64,
            collection_sha256="2" * 64,
            revision_event_count=0,
            discovered_release_dates=(self.release.last_update_date,),
        )


def test_interrupted_archive_bootstrap_resumes_from_private_page_cache(
    tmp_path: Path,
) -> None:
    release, _collection = _archive_release_and_collection()
    client = _InterruptedArchiveClient(release)
    requested = _request_time(FIRST_RETRIEVAL)
    cutoff = datetime(2026, 8, 21, 20, tzinfo=UTC)

    with SQLiteSnapshotStore(tmp_path / "archive-cache.sqlite3") as store:
        with pytest.raises(KeyboardInterrupt):
            refresh_h10_archive_store(
                store,
                client,
                requested_at=requested,
                as_of=cutoff,
            )
        cache_provenance = tuple(
            row
            for row in store.list_provenance(source=H10_SOURCE)
            if row.dataset == H10_ARCHIVE_CACHE_DATASET
        )
        assert len(cache_provenance) == 1
        assert store.read_last_good_observations(
            source=H10_SOURCE,
            dataset=H10_ARCHIVE_DATASET,
        ) == ()

        resumed = refresh_h10_archive_store(
            store,
            client,
            requested_at=requested,
            as_of=cutoff,
        )

    assert client.cached_dates == [release.last_update_date]
    assert resumed.prepared.snapshot_result.health is HealthStatus.OK
    assert resumed.archive_release_count == 1
