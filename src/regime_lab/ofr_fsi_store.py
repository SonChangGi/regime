"""Append-only private storage and receipt for the prospective OFR FSI shadow."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from typing import Any, Callable, Mapping, Protocol
from zoneinfo import ZoneInfo

from regime_lab.data.contracts import (
    CollectionResult,
    HealthStatus,
    Observation,
    PreparedSnapshot,
    SnapshotMode,
    SnapshotProvenance,
    ensure_utc,
    prepare_incremental_snapshot,
)
from regime_lab.data.ofr_fsi import (
    OFR_FSI_DATASET,
    OFR_FSI_LICENSE_CLASS,
    OFR_FSI_SOURCE,
    OFRFSIParseResult,
    OFRFSISchemaError,
    OFRFSITimestampError,
)
from regime_lab.data.store import SQLiteSnapshotStore


UTC = timezone.utc
EASTERN = ZoneInfo("America/New_York")


class OFRFSICollector(Protocol):
    def collect(self) -> OFRFSIParseResult: ...


class OFRFSIStoreError(ValueError):
    """A response cannot safely enter the private OFR shadow store."""


@dataclass(frozen=True, slots=True)
class OFRFSIStoreRefresh:
    snapshot_id: str
    prepared: PreparedSnapshot
    effective_records: tuple[Observation, ...]
    eligible_records: tuple[Observation, ...]
    source_row: Mapping[str, Any]
    used_last_good: bool
    response_sha256: str
    contract_sha256: str
    as_of: datetime


def _latest_by_period(
    records: tuple[Observation, ...],
) -> tuple[Observation, ...]:
    latest: dict[tuple[str, str, date], Observation] = {}
    for record in records:
        key = (record.source, record.series_id, record.observed_period_end)
        current = latest.get(key)
        ordering = (
            record.operating_available_at,
            record.revision_seq,
            record.system_retrieved_at,
            record.raw_sha256,
        )
        if current is None or ordering > (
            current.operating_available_at,
            current.revision_seq,
            current.system_retrieved_at,
            current.raw_sha256,
        ):
            latest[key] = record
    return tuple(latest[key] for key in sorted(latest))


def _source_row(
    *,
    result: CollectionResult,
    prepared: PreparedSnapshot,
    effective: tuple[Observation, ...],
    eligible: tuple[Observation, ...],
    as_of: datetime,
    used_last_good: bool,
) -> dict[str, Any]:
    periods = tuple(record.observed_period_end for record in eligible)
    latest_available = max(
        (record.operating_available_at for record in eligible),
        default=None,
    )
    source_status = result.health
    if source_status is HealthStatus.OK and effective and not eligible:
        source_status = HealthStatus.STALE
    return {
        "id": OFR_FSI_SOURCE,
        "name": "OFR Financial Stress Index prospective shadow",
        "status": source_status.value,
        "collection_status": result.health.value,
        "available_at": (
            latest_available.isoformat() if latest_available is not None else None
        ),
        "coverage": (
            f"{min(periods).isoformat()}–{max(periods).isoformat()}"
            if periods
            else None
        ),
        "frequency": "business_daily",
        "license_class": OFR_FSI_LICENSE_CLASS,
        "snapshot_mode": prepared.snapshot_mode.value,
        "stored_delta_records": len(result.records),
        "added_records": prepared.added_count,
        "changed_records": prepared.changed_count,
        "removed_records": prepared.removed_count,
        "effective_record_count": len(effective),
        "eligible_record_count": len(eligible),
        "last_good_used": used_last_good,
        "evidence_track": "prospective_shadow",
        "availability_basis": "collection_first_seen_at",
        "historical_availability_backfill": False,
        "published_aggregate_only": True,
        "underlying_proprietary_inputs_included": False,
        "raw_payload_publication": False,
        "public_package_inclusion": False,
        "as_of": as_of.isoformat(),
        "issues": list(result.issues),
    }


def _preserve_unknown_source_release(prepared: PreparedSnapshot) -> PreparedSnapshot:
    """Undo the generic source-release proxy on OFR discovered revisions."""

    def cleaned(record: Observation) -> Observation:
        if record.metadata.get("source_release_time_known") is not False:
            raise OFRFSIStoreError("OFR FSI record lost its unknown-release contract")
        assert record.provider_first_seen_at is not None
        return replace(
            record,
            released_at=None,
            source_released_at=None,
            available_at=record.provider_first_seen_at,
            vintage_date=record.provider_first_seen_at.astimezone(EASTERN).date(),
        )

    result = prepared.snapshot_result
    cleaned_result = replace(
        result,
        records=tuple(cleaned(record) for record in result.records),
    )
    return replace(
        prepared,
        snapshot_result=cleaned_result,
        effective_records=tuple(
            cleaned(record) for record in prepared.effective_records
        ),
    )


def refresh_ofr_fsi_store(
    store: SQLiteSnapshotStore,
    client: OFRFSICollector,
    *,
    requested_at: datetime,
    as_of: datetime | None = None,
    clock: Callable[[], datetime] | None = None,
) -> OFRFSIStoreRefresh:
    """Collect a full CSV and append only new periods or discovered revisions."""

    requested = ensure_utc(requested_at, field_name="requested_at")
    declared_cutoff = (
        ensure_utc(as_of, field_name="as_of") if as_of is not None else None
    )
    now = clock or (lambda: datetime.now(UTC))
    existing = store.read_last_good_observations(
        source=OFR_FSI_SOURCE,
        dataset=OFR_FSI_DATASET,
    )
    parsed: OFRFSIParseResult | None = None
    response_sha256 = ""
    contract_sha256 = ""
    try:
        parsed = client.collect()
        if not isinstance(parsed, OFRFSIParseResult):
            raise OFRFSIStoreError("OFR FSI collector returned an unexpected result")
        if not parsed.records:
            raise OFRFSISchemaError("OFR FSI full response has no observations")
        retrieved = ensure_utc(
            parsed.retrieved_at,
            field_name="system_retrieved_at",
        )
        if parsed.first_seen_at > retrieved:
            raise OFRFSITimestampError(
                "provider_first_seen_at must not follow system_retrieved_at"
            )
        if retrieved < requested:
            raise OFRFSITimestampError(
                "system_retrieved_at must not precede requested_at"
            )
        response_sha256 = parsed.response_sha256
        contract_sha256 = parsed.contract_sha256
        collected = CollectionResult(
            records=parsed.records,
            health=HealthStatus.OK,
            requests_made=1,
            attempts=1,
            diagnostics={
                "row_count": parsed.row_count,
                "series_count": len(parsed.series_ids),
                "first_period": parsed.first_period.isoformat(),
                "last_period": parsed.last_period.isoformat(),
            },
        )
    except OFRFSITimestampError as exc:
        # Impossible clocks would corrupt PIT evidence, so they are not written
        # even as provider-health snapshots.
        raise OFRFSIStoreError(str(exc)) from exc
    except Exception as exc:  # provider/schema boundary preserves last-good
        retrieved = ensure_utc(now(), field_name="failure_at")
        if retrieved < requested:
            raise OFRFSIStoreError("failure_at must not precede requested_at") from exc
        has_last_good = bool(existing)
        schema_failure = isinstance(exc, OFRFSISchemaError)
        if schema_failure:
            health = HealthStatus.SCHEMA_CHANGED
            issue = "ofr_fsi_schema_changed_last_good_retained" if has_last_good else (
                "ofr_fsi_schema_changed_no_last_good"
            )
        else:
            health = HealthStatus.DEGRADED if has_last_good else HealthStatus.UNAVAILABLE
            issue = "ofr_fsi_collection_failed_last_good_retained" if has_last_good else (
                "ofr_fsi_collection_failed_no_last_good"
            )
        collected = CollectionResult(
            health=health,
            issues=(issue,),
            requests_made=1,
            attempts=1,
            diagnostics={"failure_class": type(exc).__name__},
        )

    cutoff = declared_cutoff or retrieved
    if cutoff > retrieved:
        raise OFRFSIStoreError("OFR FSI as_of must not follow system_retrieved_at")

    prepared = _preserve_unknown_source_release(
        prepare_incremental_snapshot(existing, collected)
    )
    attempt = prepared.snapshot_result
    provenance = SnapshotProvenance(
        source=OFR_FSI_SOURCE,
        dataset=OFR_FSI_DATASET,
        cutoff=cutoff,
        requested_at=requested,
        retrieved_at=retrieved,
        quality_status=attempt.health,
        license_class=OFR_FSI_LICENSE_CLASS,
        request_params={
            "contract": "v6",
            "contract_sha256": contract_sha256,
            "snapshot_mode": prepared.snapshot_mode.value,
            "full_response_contract": "ofr_fsi_published_aggregate_csv_full_history",
            "stored_records": len(attempt.records),
            "added_records": prepared.added_count,
            "changed_records": prepared.changed_count,
            "unchanged_records": prepared.unchanged_count,
            "removed_records": prepared.removed_count,
            "evidence_track": "prospective_shadow",
            "availability_basis": "collection_first_seen_at",
            "historical_availability_backfill": False,
            "published_aggregate_only": True,
            "underlying_proprietary_inputs_included": False,
            "raw_payload_publication": False,
            "public_package_inclusion": False,
        },
        response_sha256=response_sha256,
        issues=attempt.issues,
    )
    stored = attempt.records if attempt.health is HealthStatus.OK else ()
    snapshot_id = store.write_snapshot(stored, provenance)
    effective = store.read_last_good_observations(
        source=OFR_FSI_SOURCE,
        dataset=OFR_FSI_DATASET,
    )
    eligible = _latest_by_period(
        tuple(
            record
            for record in effective
            if record.operating_available_at <= cutoff
            and record.observed_period_end <= cutoff.date()
        )
    )
    used_last_good = attempt.health is not HealthStatus.OK and bool(effective)
    source_row = _source_row(
        result=attempt,
        prepared=prepared,
        effective=effective,
        eligible=eligible,
        as_of=cutoff,
        used_last_good=used_last_good,
    )
    return OFRFSIStoreRefresh(
        snapshot_id=snapshot_id,
        prepared=prepared,
        effective_records=effective,
        eligible_records=eligible,
        source_row=source_row,
        used_last_good=used_last_good,
        response_sha256=response_sha256,
        contract_sha256=contract_sha256,
        as_of=cutoff,
    )


def ofr_fsi_collection_receipt_document(
    refresh: OFRFSIStoreRefresh,
    *,
    requested_at: datetime,
    as_of: datetime,
) -> dict[str, Any]:
    """Return a value-free local receipt; raw rows remain only in private SQLite."""

    if not isinstance(refresh, OFRFSIStoreRefresh):
        raise TypeError("refresh must be an OFRFSIStoreRefresh")
    requested = ensure_utc(requested_at, field_name="requested_at")
    cutoff = ensure_utc(as_of, field_name="as_of")
    source = refresh.source_row
    issues = source.get("issues", ())
    if not isinstance(issues, (list, tuple)) or any(
        not isinstance(item, str) for item in issues
    ):
        raise OFRFSIStoreError("OFR FSI source issues must be fixed string codes")
    return {
        "schema_version": 1,
        "contract": "v6",
        "operation": "collect_ofr_fsi",
        "requested_at": requested.isoformat(),
        "as_of": cutoff.isoformat(),
        "snapshot_mode": refresh.prepared.snapshot_mode.value,
        "collection_status": refresh.prepared.snapshot_result.health.value,
        "source_status": str(source["status"]),
        "last_good_used": bool(refresh.used_last_good),
        "added_records": int(refresh.prepared.added_count),
        "changed_records": int(refresh.prepared.changed_count),
        "removed_records": int(refresh.prepared.removed_count),
        "effective_record_count": len(refresh.effective_records),
        "eligible_record_count": len(refresh.eligible_records),
        "coverage": source.get("coverage"),
        "available_at": source.get("available_at"),
        "response_sha256": refresh.response_sha256 or None,
        "contract_sha256": refresh.contract_sha256 or None,
        "evidence_track": "prospective_shadow",
        "availability_basis": "collection_first_seen_at",
        "historical_availability_backfill": False,
        "publication_role": "private_derived_only_research",
        "published_aggregate_only": True,
        "underlying_proprietary_inputs_included": False,
        "raw_payload_publication": False,
        "public_package_inclusion": False,
        "issues": list(issues),
    }


__all__ = [
    "OFRFSICollector",
    "OFRFSIStoreError",
    "OFRFSIStoreRefresh",
    "ofr_fsi_collection_receipt_document",
    "refresh_ofr_fsi_store",
]
