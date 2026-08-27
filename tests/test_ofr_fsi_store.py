from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from regime_lab.data import SQLiteSnapshotStore
from regime_lab.data.ofr_fsi import (
    OFR_FSI_DATASET,
    OFR_FSI_SOURCE,
    OFRFSIParseResult,
    OFRFSISchemaError,
    load_ofr_fsi_contract,
    parse_ofr_fsi_csv,
)
from regime_lab.ofr_fsi_store import (
    OFRFSIStoreRefresh,
    ofr_fsi_collection_receipt_document,
    refresh_ofr_fsi_store,
)


UTC = timezone.utc
ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = (ROOT / "tests" / "fixtures" / "ofr_fsi.csv").read_bytes()
FIRST_REQUEST = datetime(2026, 3, 16, 19, 59, tzinfo=UTC)
FIRST_SEEN = datetime(2026, 3, 16, 20, 0, tzinfo=UTC)
SECOND_REQUEST = datetime(2026, 3, 17, 19, 59, tzinfo=UTC)
SECOND_SEEN = datetime(2026, 3, 17, 20, 0, tzinfo=UTC)


class StaticCollector:
    def __init__(self, result: OFRFSIParseResult) -> None:
        self.result = result

    def collect(self) -> OFRFSIParseResult:
        return self.result


class FailingCollector:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def collect(self) -> OFRFSIParseResult:
        raise self.error


def _parse(payload: bytes, at: datetime) -> OFRFSIParseResult:
    return parse_ofr_fsi_csv(
        payload,
        load_ofr_fsi_contract(),
        first_seen_at=at,
        retrieved_at=at,
    )


def _first_refresh(store: SQLiteSnapshotStore) -> OFRFSIStoreRefresh:
    return refresh_ofr_fsi_store(
        store,
        StaticCollector(_parse(PAYLOAD, FIRST_SEEN)),
        requested_at=FIRST_REQUEST,
        as_of=FIRST_SEEN,
    )


def test_store_appends_same_period_correction_as_a_new_first_seen_revision(
    tmp_path: Path,
) -> None:
    database = tmp_path / "ofr.sqlite3"
    corrected = PAYLOAD.replace(b"2026-03-12,-0.1000", b"2026-03-12,0.4000")
    with SQLiteSnapshotStore(database) as store:
        first = _first_refresh(store)
        second = refresh_ofr_fsi_store(
            store,
            StaticCollector(_parse(corrected, SECOND_SEEN)),
            requested_at=SECOND_REQUEST,
            as_of=SECOND_SEEN,
        )
        all_events = store.read_last_good_observations(
            source=OFR_FSI_SOURCE,
            dataset=OFR_FSI_DATASET,
        )
        provenance = store.list_provenance(source=OFR_FSI_SOURCE)

    assert first.prepared.snapshot_mode.value == "full"
    assert first.prepared.added_count == 27
    assert second.prepared.snapshot_mode.value == "delta"
    assert second.prepared.changed_count == 1
    assert second.prepared.unchanged_count == 26
    revisions = [
        record
        for record in all_events
        if record.series_id == "OFR_FSI"
        and record.observed_period_end.isoformat() == "2026-03-12"
    ]
    assert [record.revision_seq for record in revisions] == [0, 1]
    assert [record.value for record in revisions] == [-0.1, 0.4]
    assert revisions[0].provider_first_seen_at == FIRST_SEEN
    assert revisions[1].provider_first_seen_at == SECOND_SEEN
    assert all(record.source_released_at is None for record in revisions)
    assert revisions[0].raw_sha256 != revisions[1].raw_sha256
    assert len(provenance) == 2
    assert provenance[0].response_sha256 != provenance[1].response_sha256


def test_schema_failure_preserves_last_good_and_writes_no_bad_values(
    tmp_path: Path,
) -> None:
    database = tmp_path / "ofr.sqlite3"
    failure_at = datetime(2026, 3, 17, 20, 1, tzinfo=UTC)
    with SQLiteSnapshotStore(database) as store:
        first = _first_refresh(store)
        failed = refresh_ofr_fsi_store(
            store,
            FailingCollector(OFRFSISchemaError("unexpected header")),
            requested_at=SECOND_REQUEST,
            as_of=SECOND_SEEN,
            clock=lambda: failure_at,
        )
        effective = store.read_last_good_observations(
            source=OFR_FSI_SOURCE,
            dataset=OFR_FSI_DATASET,
        )
        attempts = store.list_provenance(source=OFR_FSI_SOURCE)

    assert failed.prepared.snapshot_result.health.value == "schema_changed"
    assert failed.used_last_good is True
    assert failed.effective_records == first.effective_records
    assert effective == first.effective_records
    assert len(attempts) == 2
    assert attempts[-1].quality_status.value == "schema_changed"
    assert "last_good_retained" in attempts[-1].issues[0]


def test_removed_period_is_degraded_and_last_good_is_retained(tmp_path: Path) -> None:
    shortened = b"\n".join(PAYLOAD.splitlines()[:-1]) + b"\n"
    database = tmp_path / "ofr.sqlite3"
    with SQLiteSnapshotStore(database) as store:
        first = _first_refresh(store)
        failed = refresh_ofr_fsi_store(
            store,
            StaticCollector(_parse(shortened, SECOND_SEEN)),
            requested_at=SECOND_REQUEST,
            as_of=SECOND_SEEN,
        )

    assert failed.prepared.snapshot_result.health.value == "degraded"
    assert failed.prepared.removed_count == 9
    assert failed.used_last_good is True
    assert failed.effective_records == first.effective_records


def test_late_response_is_stored_but_not_eligible_before_first_seen(
    tmp_path: Path,
) -> None:
    decision = datetime(2026, 3, 16, 19, 59, 30, tzinfo=UTC)
    database = tmp_path / "ofr.sqlite3"
    with SQLiteSnapshotStore(database) as store:
        refresh = refresh_ofr_fsi_store(
            store,
            StaticCollector(_parse(PAYLOAD, FIRST_SEEN)),
            requested_at=FIRST_REQUEST,
            as_of=decision,
        )

    assert len(refresh.effective_records) == 27
    assert refresh.eligible_records == ()
    assert refresh.source_row["status"] == "stale"
    assert refresh.source_row["available_at"] is None


def test_receipt_is_value_free_private_shadow_and_excludes_raw_package_fields(
    tmp_path: Path,
) -> None:
    with SQLiteSnapshotStore(tmp_path / "ofr.sqlite3") as store:
        refresh = _first_refresh(store)
    receipt = ofr_fsi_collection_receipt_document(
        refresh,
        requested_at=FIRST_REQUEST,
        as_of=FIRST_SEEN,
    )

    assert receipt["contract"] == "v6"
    assert receipt["operation"] == "collect_ofr_fsi"
    assert receipt["evidence_track"] == "prospective_shadow"
    assert receipt["published_aggregate_only"] is True
    assert receipt["underlying_proprietary_inputs_included"] is False
    assert receipt["raw_payload_publication"] is False
    assert receipt["public_package_inclusion"] is False
    assert len(receipt["response_sha256"]) == 64
    encoded = json.dumps(receipt, sort_keys=True)
    for forbidden in (
        '"records"',
        '"values"',
        '"raw_body"',
        '"snapshot_id"',
        "-0.1000",
    ):
        assert forbidden not in encoded


def test_no_last_good_network_failure_is_unavailable(tmp_path: Path) -> None:
    requested = datetime(2026, 3, 16, 19, 59, tzinfo=UTC)
    failed_at = datetime(2026, 3, 16, 20, 1, tzinfo=UTC)
    with SQLiteSnapshotStore(tmp_path / "ofr.sqlite3") as store:
        refresh = refresh_ofr_fsi_store(
            store,
            FailingCollector(TimeoutError("offline fixture")),
            requested_at=requested,
            as_of=requested,
            clock=lambda: failed_at,
        )

    assert refresh.prepared.snapshot_result.health.value == "unavailable"
    assert refresh.used_last_good is False
    assert refresh.effective_records == ()
