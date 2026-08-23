"""Incremental private storage and derived-only context for Fed H.10."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import pandas as pd

from regime_lab.analysis.fx import (
    FXFeatureResult,
    FX_ARCHIVE_AVAILABILITY_BASIS,
    FX_ARCHIVE_CORRECTION_AVAILABILITY_BASIS,
    FX_ARCHIVE_CORRECTION_QUARANTINE_WEEKS,
    FX_ARCHIVE_REVISION_POLICY,
    FX_FIRST_SEEN_AVAILABILITY_BASIS,
    FX_MAX_OBSERVATION_AGE_DAYS,
    build_fx_features,
    build_official_archive_fx_features,
    fx_context_at,
    unavailable_fx_context,
)
from regime_lab.analysis.fx_ablation import fx_ablation_readiness
from regime_lab.data.h10 import H10ParseResult, H10TimestampError
from regime_lab.data.h10_archive import (
    H10_ARCHIVE_EVALUATION_START,
    H10_ARCHIVE_EVALUATION_START_RATIONALE,
    H10_ARCHIVE_AVAILABILITY_BASIS,
    H10_ARCHIVE_CORRECTION_AVAILABILITY_BASIS,
    H10_ARCHIVE_CORRECTION_EQUIVALENT_COMPONENTS,
    H10_ARCHIVE_CORRECTION_EQUIVALENT_POLICY,
    H10_ARCHIVE_REVISION_POLICY,
    H10_ARCHIVE_SHORT_GAP_AUXILIARY_DAYS,
    H10_ARCHIVE_NORMAL_AVAILABILITY_BASIS,
    H10ArchiveCollection,
    H10ArchiveCorrectionEquivalentEvent,
    H10ArchiveLineage,
    H10ArchiveRelease,
    archive_release_available_at,
    detect_h10_archive_correction_equivalents,
)
from regime_lab.data import (
    CollectionResult,
    HealthStatus,
    Observation,
    PreparedSnapshot,
    SQLiteSnapshotStore,
    SnapshotMode,
    SnapshotProvenance,
    normalize_revision_sequences,
    observation_natural_key,
    prepare_incremental_snapshot,
)


UTC = timezone.utc
EASTERN = ZoneInfo("America/New_York")
H10_SOURCE = "frb_h10"
H10_DATASET = "h10_business_day_fx"
H10_ARCHIVE_DATASET = "h10_official_release_archive_fx"
H10_ARCHIVE_CACHE_DATASET = "h10_official_release_archive_page_cache"
H10_LICENSE_CLASS = "federal_reserve_board_public_domain_citation_requested"


class H10Collector(Protocol):
    def collect(
        self,
        *,
        first_seen_at: datetime | None = None,
    ) -> H10ParseResult: ...


class H10ArchiveCollector(Protocol):
    def collect(
        self,
        *,
        requested_at: datetime,
        start_date: date | None = None,
        end_date: date | None = None,
        known_release_dates: Sequence[date] = (),
        cached_releases: Mapping[date, H10ArchiveRelease] | None = None,
        on_release: Callable[[H10ArchiveRelease], None] | None = None,
    ) -> H10ArchiveCollection: ...


class H10StoreError(ValueError):
    """The collector returned a value that cannot enter the H.10 store."""


@dataclass(frozen=True, slots=True)
class H10StoreRefresh:
    """Private effective history plus derived-only publication surfaces."""

    snapshot_id: str
    prepared: PreparedSnapshot
    effective_records: tuple[Observation, ...]
    fx_features: FXFeatureResult | None
    source_row: Mapping[str, Any]
    fx_context: Mapping[str, Any]
    used_last_good: bool
    archive_release_count: int = 0
    archive_correction_count: int = 0
    archive_correction_available_at: tuple[datetime, ...] = ()
    archive_quarantine_event_count: int = 0
    archive_quarantine_event_dates: tuple[date, ...] = ()
    archive_quarantine_available_at: tuple[datetime, ...] = ()
    archive_detected_revision_event_count: int = 0
    archive_detected_revision_row_count: int = 0
    archive_complete_republication_event_count: int = 0
    archive_short_gap_auxiliary_event_count: int = 0
    archive_index_sha256: str = ""
    archive_lineage_sha256: str = ""


@dataclass(frozen=True, slots=True)
class _ArchiveContract:
    release_event_dates: tuple[date, ...]
    correction_event_dates: tuple[date, ...]
    correction_available_at: tuple[datetime, ...]
    quarantine_event_dates: tuple[date, ...] = ()
    quarantine_available_at: tuple[datetime, ...] = ()
    lineage_events: tuple[H10ArchiveCorrectionEquivalentEvent, ...] = ()
    archive_index_sha256: str = ""
    archive_lineage_sha256: str = ""
    lineage_complete: bool = False


def _archive_refresh_metadata(contract: _ArchiveContract) -> dict[str, Any]:
    return {
        "archive_release_count": len(contract.release_event_dates),
        "archive_correction_count": len(contract.correction_event_dates),
        "archive_correction_available_at": contract.correction_available_at,
        "archive_quarantine_event_count": len(contract.quarantine_event_dates),
        "archive_quarantine_event_dates": contract.quarantine_event_dates,
        "archive_quarantine_available_at": contract.quarantine_available_at,
        "archive_detected_revision_event_count": sum(
            event.material_revision_rows > 0
            for event in contract.lineage_events
        ),
        "archive_detected_revision_row_count": sum(
            event.material_revision_rows for event in contract.lineage_events
        ),
        "archive_complete_republication_event_count": sum(
            event.complete_republication for event in contract.lineage_events
        ),
        "archive_short_gap_auxiliary_event_count": sum(
            event.short_gap_auxiliary_evidence
            for event in contract.lineage_events
        ),
        "archive_index_sha256": contract.archive_index_sha256,
        "archive_lineage_sha256": contract.archive_lineage_sha256,
    }


def _aware_utc(value: datetime, *, field_name: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def _readiness_cutoffs(
    result: FXFeatureResult | None,
    *,
    as_of: datetime,
) -> pd.DatetimeIndex:
    """Build the weekly Friday grid used by the prospective FX gate."""

    resolved = _aware_utc(as_of, field_name="as_of")
    local_date = resolved.astimezone(EASTERN).date()
    last_friday = local_date - timedelta(days=(local_date.weekday() - 4) % 7)
    end = pd.Timestamp(last_friday)
    if result is None or result.coverage.empty:
        return pd.DatetimeIndex([end])

    start = pd.Timestamp(result.coverage.index.min())
    if start.tzinfo is not None:
        start = start.tz_convert(EASTERN).tz_localize(None)
    start = start.normalize()
    start = start - timedelta(days=int((start.weekday() - 4) % 7))
    if start > end:
        start = end
    return pd.date_range(start=start, end=end, freq="W-FRI")


def _archive_quarantine_ranges(
    result: FXFeatureResult | None,
) -> list[dict[str, Any]]:
    if (
        result is None
        or "archive_correction_quarantined" not in result.coverage
    ):
        return []
    weeks = pd.DatetimeIndex(
        result.coverage.index[
            result.coverage["archive_correction_quarantined"].eq(True)
        ]
    )
    if weeks.empty:
        return []
    ranges: list[dict[str, Any]] = []
    start = previous = pd.Timestamp(weeks[0])
    for raw_week in weeks[1:]:
        week = pd.Timestamp(raw_week)
        if week - previous == timedelta(weeks=1):
            previous = week
            continue
        ranges.append(
            {
                "first_observation_week": start.date().isoformat(),
                "last_observation_week": previous.date().isoformat(),
                "first_model_cutoff": (
                    start + timedelta(weeks=1)
                ).date().isoformat(),
                "last_model_cutoff": (
                    previous + timedelta(weeks=1)
                ).date().isoformat(),
                "origin_count": int((previous - start).days // 7 + 1),
            }
        )
        start = previous = week
    ranges.append(
        {
            "first_observation_week": start.date().isoformat(),
            "last_observation_week": previous.date().isoformat(),
            "first_model_cutoff": (start + timedelta(weeks=1)).date().isoformat(),
            "last_model_cutoff": (
                previous + timedelta(weeks=1)
            ).date().isoformat(),
            "origin_count": int((previous - start).days // 7 + 1),
        }
    )
    return ranges


def h10_collection_receipt_document(
    refresh: H10StoreRefresh,
    *,
    requested_at: datetime,
    as_of: datetime,
) -> dict[str, Any]:
    """Return a derived-only receipt for one isolated H.10 collection."""

    if not isinstance(refresh, H10StoreRefresh):
        raise TypeError("refresh must be an H10StoreRefresh")
    requested = _aware_utc(requested_at, field_name="requested_at")
    cutoff = _aware_utc(as_of, field_name="as_of")
    readiness = fx_ablation_readiness(
        refresh.fx_features,
        _readiness_cutoffs(refresh.fx_features, as_of=cutoff),
    )
    issues = refresh.source_row.get("issues", ())
    if not isinstance(issues, (list, tuple)) or any(
        not isinstance(issue, str) for issue in issues
    ):
        raise H10StoreError("H.10 source issues must be fixed string codes")
    return {
        "schema_version": 1,
        "contract": "v5",
        "operation": "collect_h10",
        "requested_at": requested.isoformat(),
        "as_of": cutoff.isoformat(),
        "snapshot_mode": refresh.prepared.snapshot_mode.value,
        "collection_status": refresh.prepared.snapshot_result.health.value,
        "source_status": str(refresh.source_row["status"]),
        "fx_status": str(refresh.fx_context["status"]),
        "last_good_used": bool(refresh.used_last_good),
        "added_records": int(refresh.prepared.added_count),
        "changed_records": int(refresh.prepared.changed_count),
        "removed_records": int(refresh.prepared.removed_count),
        "effective_record_count": len(refresh.effective_records),
        "eligible_common_weeks": int(readiness["eligible_common_weeks"]),
        "readiness": str(readiness["status"]),
        "minimum_common_weeks": int(readiness["minimum_common_weeks"]),
        "first_eligible_cutoff": readiness["first_eligible_cutoff"],
        "last_eligible_cutoff": readiness["last_eligible_cutoff"],
        "historical_availability_backfill": False,
        "official_release_archive_ingest": bool(
            refresh.source_row["official_release_archive_ingest"]
        ),
        "availability_basis": str(refresh.source_row["availability_basis"]),
        "archive_revision_policy": str(
            refresh.source_row["archive_revision_policy"]
        ),
        "archive_correction_availability_basis": str(
            refresh.source_row["archive_correction_availability_basis"]
        ),
        "archive_release_count": int(refresh.archive_release_count),
        "archive_correction_count": int(refresh.archive_correction_count),
        "archive_declared_correction_count": int(
            refresh.archive_correction_count
        ),
        "archive_correction_available_at": [
            value.isoformat()
            for value in refresh.archive_correction_available_at
        ],
        "archive_correction_equivalent_policy": (
            H10_ARCHIVE_CORRECTION_EQUIVALENT_POLICY
        ),
        "archive_correction_equivalent_components": list(
            H10_ARCHIVE_CORRECTION_EQUIVALENT_COMPONENTS
        ),
        "archive_short_gap_auxiliary_days": (
            H10_ARCHIVE_SHORT_GAP_AUXILIARY_DAYS
        ),
        "archive_quarantine_event_count": int(
            refresh.archive_quarantine_event_count
        ),
        "archive_quarantine_event_dates": [
            value.isoformat()
            for value in refresh.archive_quarantine_event_dates
        ],
        "archive_quarantine_available_at": [
            value.isoformat()
            for value in refresh.archive_quarantine_available_at
        ],
        "archive_detected_revision_event_count": int(
            refresh.archive_detected_revision_event_count
        ),
        "archive_detected_revision_row_count": int(
            refresh.archive_detected_revision_row_count
        ),
        "archive_complete_republication_event_count": int(
            refresh.archive_complete_republication_event_count
        ),
        "archive_short_gap_auxiliary_event_count": int(
            refresh.archive_short_gap_auxiliary_event_count
        ),
        "archive_index_sha256": refresh.archive_index_sha256 or None,
        "archive_lineage_sha256": refresh.archive_lineage_sha256 or None,
        "archive_lineage_complete": bool(refresh.archive_lineage_sha256),
        "archive_correction_quarantine_weeks": (
            FX_ARCHIVE_CORRECTION_QUARANTINE_WEEKS
        ),
        "archive_quarantined_origin_count": int(
            refresh.fx_features.coverage[
                "archive_correction_quarantined"
            ].eq(True).sum()
            if refresh.fx_features is not None
            and "archive_correction_quarantined"
            in refresh.fx_features.coverage
            else 0
        ),
        "archive_quarantine_ranges": _archive_quarantine_ranges(
            refresh.fx_features
        ),
        "archive_evaluation_start": H10_ARCHIVE_EVALUATION_START.isoformat(),
        "archive_evaluation_start_rationale": (
            H10_ARCHIVE_EVALUATION_START_RATIONALE
        ),
        "issues": list(issues),
    }


def _source_row(
    *,
    result: CollectionResult,
    source_status: HealthStatus,
    effective_records: tuple[Observation, ...],
    as_of: datetime,
    prepared: PreparedSnapshot,
    used_last_good: bool,
    official_release_archive_ingest: bool = False,
    archive_release_count: int = 0,
    archive_correction_available_at: Sequence[datetime] = (),
) -> dict[str, Any]:
    eligible = tuple(
        record for record in effective_records if record.available_at <= as_of
    )
    periods = tuple(record.observed_period_end for record in eligible)
    latest_available = max(
        (record.available_at for record in eligible),
        default=None,
    )
    corrections = tuple(
        _aware_utc(value, field_name="archive_correction_available_at")
        for value in archive_correction_available_at
    )
    return {
        "id": H10_SOURCE,
        "name": "Federal Reserve H.10 foreign exchange rates",
        "status": source_status.value,
        "available_at": (
            latest_available.isoformat() if latest_available is not None else None
        ),
        "coverage": (
            f"{min(periods).isoformat()}–{max(periods).isoformat()}"
            if periods
            else None
        ),
        "frequency": "business_day_to_weekly",
        "license_class": H10_LICENSE_CLASS,
        "snapshot_mode": prepared.snapshot_mode.value,
        "stored_delta_records": len(result.records),
        "added_records": prepared.added_count,
        "changed_records": prepared.changed_count,
        "removed_records": prepared.removed_count,
        "last_good_used": used_last_good,
        "official_release_archive_ingest": bool(
            official_release_archive_ingest
        ),
        "availability_basis": (
            FX_ARCHIVE_AVAILABILITY_BASIS
            if official_release_archive_ingest
            else FX_FIRST_SEEN_AVAILABILITY_BASIS
        ),
        "archive_revision_policy": FX_ARCHIVE_REVISION_POLICY,
        "archive_correction_availability_basis": (
            FX_ARCHIVE_CORRECTION_AVAILABILITY_BASIS
        ),
        "archive_release_count": int(archive_release_count),
        "archive_correction_count": len(corrections),
        "archive_correction_available_at": [
            value.isoformat() for value in corrections
        ],
        "archive_correction_quarantine_weeks": (
            FX_ARCHIVE_CORRECTION_QUARANTINE_WEEKS
        ),
        "archive_evaluation_start": H10_ARCHIVE_EVALUATION_START.isoformat(),
        "archive_evaluation_start_rationale": (
            H10_ARCHIVE_EVALUATION_START_RATIONALE
        ),
        # Issues are fixed adapter codes. Provider exception text and request
        # material never cross this publication-facing boundary.
        "issues": list(result.issues),
    }


def _empty_fx_context(
    *,
    status: str,
    source_status: HealthStatus,
    used_last_good: bool,
) -> dict[str, Any]:
    context = unavailable_fx_context()
    context.update(
        {
            "status": status,
            "source_status": source_status.value,
            "last_good_used": used_last_good,
        }
    )
    return context


def _derive_fx_context(
    records: tuple[Observation, ...],
    *,
    as_of: datetime,
    source_status: HealthStatus,
    used_last_good: bool,
    official_release_archive_ingest: bool = False,
    archive_correction_available_at: Sequence[datetime] = (),
) -> tuple[FXFeatureResult | None, dict[str, Any]]:
    if not records:
        return None, _empty_fx_context(
            status="unavailable",
            source_status=source_status,
            used_last_good=used_last_good,
        )

    try:
        result = (
            build_official_archive_fx_features(
                records,
                as_of=as_of,
                correction_available_at=archive_correction_available_at,
            )
            if official_release_archive_ingest
            else build_fx_features(records, as_of=as_of)
        )
    except ValueError:
        status = "degraded" if used_last_good else "unavailable"
        return None, _empty_fx_context(
            status=status,
            source_status=source_status,
            used_last_good=used_last_good,
        )

    context = fx_context_at(result, cutoff=as_of)
    if source_status is HealthStatus.DEGRADED and used_last_good:
        context["status"] = "degraded"
    elif source_status is HealthStatus.STALE:
        context["status"] = "stale"
    elif source_status is not HealthStatus.OK:
        context["status"] = "unavailable"
    context["source_status"] = source_status.value
    context["last_good_used"] = used_last_good
    return result, context


def _analysis_source_status(
    attempt_status: HealthStatus,
    records: tuple[Observation, ...],
    *,
    as_of: datetime,
) -> HealthStatus:
    if attempt_status is not HealthStatus.OK:
        return attempt_status
    eligible = tuple(record for record in records if record.available_at <= as_of)
    if not eligible:
        return HealthStatus.UNAVAILABLE
    latest_period = max(record.observed_period_end for record in eligible)
    age_days = (as_of.astimezone(EASTERN).date() - latest_period).days
    return (
        HealthStatus.STALE
        if age_days > FX_MAX_OBSERVATION_AGE_DAYS
        else HealthStatus.OK
    )


def _archive_inventory_sha256(values: Sequence[date]) -> str:
    encoded = json.dumps(
        [value.isoformat() for value in values],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _lineage_event_document(
    event: H10ArchiveCorrectionEquivalentEvent,
) -> dict[str, Any]:
    return {
        "event_date": event.event_date.isoformat(),
        "available_at": event.available_at.isoformat(),
        "prior_release_date": (
            event.prior_release_date.isoformat()
            if event.prior_release_date is not None
            else None
        ),
        "prior_release_gap_days": event.prior_release_gap_days,
        "short_gap_auxiliary_evidence": (
            event.short_gap_auxiliary_evidence
        ),
        "declared_correction": event.declared_correction,
        "material_revision_rows": event.material_revision_rows,
        "complete_republication": event.complete_republication,
        "current_series_date_rows": event.current_series_date_rows,
        "overlap_series_date_rows": event.overlap_series_date_rows,
        "new_series_date_rows": event.new_series_date_rows,
        "trigger_components": list(event.trigger_components),
    }


def _lineage_event_from_document(
    value: object,
) -> H10ArchiveCorrectionEquivalentEvent:
    if not isinstance(value, Mapping):
        raise H10StoreError("official H.10 archive lineage event is malformed")
    expected_keys = set(
        _lineage_event_document(
            H10ArchiveCorrectionEquivalentEvent(
                event_date=date(2000, 1, 2),
                available_at=datetime(2000, 1, 2, tzinfo=UTC),
                prior_release_date=date(2000, 1, 1),
                prior_release_gap_days=1,
                short_gap_auxiliary_evidence=True,
                declared_correction=True,
                material_revision_rows=0,
                complete_republication=False,
                current_series_date_rows=1,
                overlap_series_date_rows=0,
                new_series_date_rows=1,
                trigger_components=("declared_correction",),
            )
        )
    )
    if set(value) != expected_keys:
        raise H10StoreError("official H.10 archive lineage event schema changed")
    try:
        event_date = date.fromisoformat(str(value["event_date"]))
        available_at = _aware_utc(
            datetime.fromisoformat(str(value["available_at"])),
            field_name="archive_quarantine_available_at",
        )
        prior_release_date = (
            date.fromisoformat(str(value["prior_release_date"]))
            if value["prior_release_date"] is not None
            else None
        )
        prior_release_gap_days = (
            int(value["prior_release_gap_days"])
            if value["prior_release_gap_days"] is not None
            else None
        )
        material_revision_rows = int(value["material_revision_rows"])
        current_rows = int(value["current_series_date_rows"])
        overlap_rows = int(value["overlap_series_date_rows"])
        new_rows = int(value["new_series_date_rows"])
        triggers = tuple(str(item) for item in value["trigger_components"])
    except (TypeError, ValueError) as exc:
        raise H10StoreError(
            "official H.10 archive lineage event is malformed"
        ) from exc
    boolean_fields = (
        "short_gap_auxiliary_evidence",
        "declared_correction",
        "complete_republication",
    )
    if any(not isinstance(value[field], bool) for field in boolean_fields):
        raise H10StoreError("official H.10 archive lineage flags are invalid")
    expected_triggers = tuple(
        component
        for component, matched in (
            ("declared_correction", bool(value["declared_correction"])),
            ("material_revision", material_revision_rows > 0),
            (
                "complete_republication",
                bool(value["complete_republication"]),
            ),
        )
        if matched
    )
    gap_consistent = (
        prior_release_gap_days is None
        and prior_release_date is None
        and not bool(value["short_gap_auxiliary_evidence"])
    ) or (
        prior_release_gap_days is not None
        and prior_release_date is not None
        and prior_release_gap_days == (event_date - prior_release_date).days
        and bool(value["short_gap_auxiliary_evidence"])
        == (prior_release_gap_days <= H10_ARCHIVE_SHORT_GAP_AUXILIARY_DAYS)
    )
    if (
        not expected_triggers
        or triggers != expected_triggers
        or not set(triggers).issubset(H10_ARCHIVE_CORRECTION_EQUIVALENT_COMPONENTS)
        or not gap_consistent
        or current_rows < 1
        or not 0 <= overlap_rows <= current_rows
        or new_rows != current_rows - overlap_rows
        or not 0 <= material_revision_rows <= overlap_rows
        or (
            bool(value["complete_republication"])
            and (new_rows != 0 or overlap_rows != current_rows)
        )
    ):
        raise H10StoreError("official H.10 archive lineage event is inconsistent")
    return H10ArchiveCorrectionEquivalentEvent(
        event_date=event_date,
        available_at=available_at,
        prior_release_date=prior_release_date,
        prior_release_gap_days=prior_release_gap_days,
        short_gap_auxiliary_evidence=bool(
            value["short_gap_auxiliary_evidence"]
        ),
        declared_correction=bool(value["declared_correction"]),
        material_revision_rows=material_revision_rows,
        complete_republication=bool(value["complete_republication"]),
        current_series_date_rows=current_rows,
        overlap_series_date_rows=overlap_rows,
        new_series_date_rows=new_rows,
        trigger_components=triggers,
    )


def _archive_contract_from_lineage(
    release_event_dates: Sequence[date],
    *,
    archive_index_sha256: str,
    lineage: H10ArchiveLineage,
) -> _ArchiveContract:
    events = tuple(release_event_dates)
    if (
        events != tuple(sorted(set(events)))
        or lineage.release_count != len(events)
        or not _is_sha256(archive_index_sha256)
        or not _is_sha256(lineage.lineage_sha256)
    ):
        raise H10StoreError("official H.10 archive lineage inventory is invalid")
    lineage_events = tuple(lineage.events)
    event_dates = tuple(event.event_date for event in lineage_events)
    if (
        event_dates != tuple(sorted(set(event_dates)))
        or set(event_dates).difference(events)
    ):
        raise H10StoreError("official H.10 archive lineage dates are invalid")
    declared = tuple(event for event in lineage_events if event.declared_correction)
    return _ArchiveContract(
        release_event_dates=events,
        correction_event_dates=tuple(event.event_date for event in declared),
        correction_available_at=tuple(event.available_at for event in declared),
        quarantine_event_dates=event_dates,
        quarantine_available_at=tuple(
            event.available_at for event in lineage_events
        ),
        lineage_events=lineage_events,
        archive_index_sha256=archive_index_sha256,
        archive_lineage_sha256=lineage.lineage_sha256,
        lineage_complete=True,
    )


def _archive_contract_from_provenance(
    provenance: SnapshotProvenance | None,
) -> _ArchiveContract:
    if provenance is None:
        return _ArchiveContract((), (), ())
    if (
        provenance.source != H10_SOURCE
        or provenance.dataset != H10_ARCHIVE_DATASET
        or provenance.quality_status is not HealthStatus.OK
        or provenance.license_class != H10_LICENSE_CLASS
    ):
        raise H10StoreError("official H.10 archive provenance identity is invalid")
    params = dict(provenance.request_params)
    expected = {
        "official_release_archive_ingest": True,
        "availability_basis": H10_ARCHIVE_AVAILABILITY_BASIS,
        "archive_normal_availability_basis": (
            H10_ARCHIVE_NORMAL_AVAILABILITY_BASIS
        ),
        "archive_correction_availability_basis": (
            H10_ARCHIVE_CORRECTION_AVAILABILITY_BASIS
        ),
        "archive_revision_policy": H10_ARCHIVE_REVISION_POLICY,
        "archive_correction_quarantine_weeks": (
            FX_ARCHIVE_CORRECTION_QUARANTINE_WEEKS
        ),
        "archive_evaluation_start": H10_ARCHIVE_EVALUATION_START.isoformat(),
        "archive_evaluation_start_rationale": (
            H10_ARCHIVE_EVALUATION_START_RATIONALE
        ),
        "raw_payload_publication": False,
    }
    if any(params.get(key) != value for key, value in expected.items()):
        raise H10StoreError("official H.10 archive provenance contract is invalid")

    raw_events = params.get("archive_release_event_dates")
    raw_corrections = params.get("archive_correction_event_dates")
    raw_available = params.get("archive_correction_available_at")
    if (
        not isinstance(raw_events, list)
        or not isinstance(raw_corrections, list)
        or not isinstance(raw_available, list)
    ):
        raise H10StoreError("official H.10 archive inventory is missing")
    try:
        events = tuple(date.fromisoformat(str(value)) for value in raw_events)
        correction_events = tuple(
            date.fromisoformat(str(value)) for value in raw_corrections
        )
        correction_available = tuple(
            _aware_utc(
                datetime.fromisoformat(str(value)),
                field_name="archive_correction_available_at",
            )
            for value in raw_available
        )
    except (TypeError, ValueError) as exc:
        raise H10StoreError("official H.10 archive inventory is malformed") from exc
    if (
        not events
        or events != tuple(sorted(set(events)))
        or correction_events != tuple(sorted(set(correction_events)))
        or correction_available != tuple(sorted(set(correction_available)))
        or len(correction_events) != len(correction_available)
        or set(correction_events).difference(events)
        or params.get("archive_release_count") != len(events)
        or params.get("archive_correction_count") != len(correction_events)
        or params.get("archive_release_event_dates_sha256")
        != _archive_inventory_sha256(events)
    ):
        raise H10StoreError("official H.10 archive inventory is inconsistent")
    policy = params.get("archive_correction_equivalent_policy")
    if policy is None:
        # Legacy archive provenance only declared correction pages. A
        # successful cache-backed refresh upgrades it without rewriting data.
        return _ArchiveContract(
            events,
            correction_events,
            correction_available,
            quarantine_event_dates=correction_events,
            quarantine_available_at=correction_available,
        )
    if (
        policy != H10_ARCHIVE_CORRECTION_EQUIVALENT_POLICY
        or params.get("archive_correction_equivalent_components")
        != list(H10_ARCHIVE_CORRECTION_EQUIVALENT_COMPONENTS)
        or params.get("archive_short_gap_auxiliary_days")
        != H10_ARCHIVE_SHORT_GAP_AUXILIARY_DAYS
        or params.get("archive_declared_correction_count")
        != len(correction_events)
        or not isinstance(params.get("archive_lineage_complete"), bool)
    ):
        raise H10StoreError("official H.10 archive lineage policy is invalid")
    lineage_complete = bool(params["archive_lineage_complete"])
    raw_quarantine_dates = params.get("archive_quarantine_event_dates")
    raw_quarantine_available = params.get("archive_quarantine_available_at")
    raw_lineage = params.get("archive_correction_equivalent_event_lineage")
    if (
        not isinstance(raw_quarantine_dates, list)
        or not isinstance(raw_quarantine_available, list)
        or not isinstance(raw_lineage, list)
    ):
        raise H10StoreError("official H.10 archive lineage inventory is missing")
    try:
        quarantine_dates = tuple(
            date.fromisoformat(str(value)) for value in raw_quarantine_dates
        )
        quarantine_available = tuple(
            _aware_utc(
                datetime.fromisoformat(str(value)),
                field_name="archive_quarantine_available_at",
            )
            for value in raw_quarantine_available
        )
        lineage_events = tuple(
            _lineage_event_from_document(value) for value in raw_lineage
        )
    except (TypeError, ValueError) as exc:
        raise H10StoreError(
            "official H.10 archive lineage inventory is malformed"
        ) from exc
    index_sha256 = params.get("archive_index_sha256") or ""
    lineage_sha256 = params.get("archive_lineage_sha256") or ""
    detected_events = tuple(
        event for event in lineage_events if event.material_revision_rows > 0
    )
    complete_republications = tuple(
        event for event in lineage_events if event.complete_republication
    )
    short_gap_events = tuple(
        event
        for event in lineage_events
        if event.short_gap_auxiliary_evidence
    )
    declared_lineage = tuple(
        event for event in lineage_events if event.declared_correction
    )
    if (
        quarantine_dates != tuple(sorted(set(quarantine_dates)))
        or quarantine_available != tuple(sorted(set(quarantine_available)))
        or len(quarantine_dates) != len(quarantine_available)
        or set(quarantine_dates).difference(events)
        or params.get("archive_quarantine_event_count")
        != len(quarantine_dates)
        or params.get("archive_detected_revision_event_count")
        != len(detected_events)
        or params.get("archive_detected_revision_row_count")
        != sum(event.material_revision_rows for event in detected_events)
        or params.get("archive_complete_republication_event_count")
        != len(complete_republications)
        or params.get("archive_short_gap_auxiliary_event_count")
        != len(short_gap_events)
    ):
        raise H10StoreError("official H.10 archive lineage inventory is inconsistent")
    if lineage_complete:
        if (
            not _is_sha256(index_sha256)
            or not _is_sha256(lineage_sha256)
            or tuple(event.event_date for event in lineage_events)
            != quarantine_dates
            or tuple(event.available_at for event in lineage_events)
            != quarantine_available
            or tuple(event.event_date for event in declared_lineage)
            != correction_events
            or tuple(event.available_at for event in declared_lineage)
            != correction_available
        ):
            raise H10StoreError(
                "official H.10 complete archive lineage is inconsistent"
            )
    elif (
        lineage_events
        or quarantine_dates != correction_events
        or quarantine_available != correction_available
        or index_sha256
        or lineage_sha256
    ):
        raise H10StoreError("official H.10 partial archive lineage is inconsistent")
    return _ArchiveContract(
        release_event_dates=events,
        correction_event_dates=correction_events,
        correction_available_at=correction_available,
        quarantine_event_dates=quarantine_dates,
        quarantine_available_at=quarantine_available,
        lineage_events=lineage_events,
        archive_index_sha256=index_sha256,
        archive_lineage_sha256=lineage_sha256,
        lineage_complete=lineage_complete,
    )


def _load_archive_release_cache(
    store: SQLiteSnapshotStore,
) -> dict[date, H10ArchiveRelease]:
    cached: dict[date, H10ArchiveRelease] = {}
    for provenance in store.list_provenance(source=H10_SOURCE):
        if (
            provenance.dataset != H10_ARCHIVE_CACHE_DATASET
            or provenance.quality_status is not HealthStatus.OK
        ):
            continue
        params = dict(provenance.request_params)
        try:
            release_date = date.fromisoformat(str(params["archive_release_date"]))
            last_update = date.fromisoformat(
                str(params["archive_last_update_date"])
            )
        except (KeyError, ValueError) as exc:
            raise H10StoreError("H.10 archive page cache metadata is invalid") from exc
        if (
            params.get("snapshot_mode") != SnapshotMode.FULL.value
            or params.get("raw_payload_publication") is not False
            or last_update in cached
        ):
            raise H10StoreError("H.10 archive page cache is inconsistent")
        available_at = archive_release_available_at(
            release_date,
            last_update_date=last_update,
        )
        availability_basis = (
            H10_ARCHIVE_NORMAL_AVAILABILITY_BASIS
            if release_date == last_update
            else H10_ARCHIVE_CORRECTION_AVAILABILITY_BASIS
        )
        if params.get("availability_basis") != availability_basis:
            raise H10StoreError("H.10 archive page cache availability is invalid")
        records = store.read_observations(snapshot_id=provenance.snapshot_id)
        if not records:
            raise H10StoreError("H.10 archive page cache contains no records")
        cached[last_update] = H10ArchiveRelease(
            release_date=release_date,
            last_update_date=last_update,
            available_at=available_at,
            availability_basis=availability_basis,
            retrieved_at=provenance.retrieved_at,
            source_url=(
                "https://www.federalreserve.gov/releases/h10/"
                f"{last_update.strftime('%Y%m%d')}/"
            ),
            snapshot_sha256=provenance.response_sha256,
            records=records,
        )
    return cached


def _write_archive_release_cache(
    store: SQLiteSnapshotStore,
    release: H10ArchiveRelease,
    *,
    requested_at: datetime,
    cutoff: datetime,
) -> None:
    provenance = SnapshotProvenance(
        source=H10_SOURCE,
        dataset=H10_ARCHIVE_CACHE_DATASET,
        cutoff=cutoff,
        requested_at=requested_at,
        retrieved_at=release.retrieved_at,
        quality_status=HealthStatus.OK,
        license_class=H10_LICENSE_CLASS,
        request_params={
            "snapshot_mode": SnapshotMode.FULL.value,
            "archive_release_date": release.release_date.isoformat(),
            "archive_last_update_date": release.last_update_date.isoformat(),
            "availability_basis": release.availability_basis,
            "raw_payload_publication": False,
            "checkpoint_only": True,
        },
        response_sha256=release.snapshot_sha256,
    )
    store.write_snapshot(release.records, provenance)


def _merge_archive_records(
    existing: Sequence[Observation],
    incoming: Sequence[Observation],
) -> tuple[tuple[Observation, ...], tuple[Observation, ...], int, int, int]:
    accepted = list(existing)
    by_identity = {observation_natural_key(record): record for record in existing}
    latest_by_period: dict[tuple[str, str, date], Observation] = {}
    for record in sorted(existing, key=lambda item: item.available_at):
        latest_by_period[
            (record.source, record.series_id, record.observed_period_end)
        ] = record
    delta: list[Observation] = []
    added = 0
    changed = 0
    unchanged = 0
    for record in sorted(
        incoming,
        key=lambda item: (
            item.available_at,
            item.series_id,
            item.observed_period_end,
            item.raw_sha256,
        ),
    ):
        identity = observation_natural_key(record)
        prior_identity = by_identity.get(identity)
        if prior_identity is not None:
            if (
                prior_identity.value,
                prior_identity.quality_status,
                prior_identity.units,
                prior_identity.raw_sha256,
            ) != (
                record.value,
                record.quality_status,
                record.units,
                record.raw_sha256,
            ):
                raise H10StoreError("official H.10 archive event conflicts with storage")
            unchanged += 1
            continue
        period_key = (record.source, record.series_id, record.observed_period_end)
        prior = latest_by_period.get(period_key)
        semantic = (record.value, record.quality_status, record.units)
        prior_semantic = (
            None
            if prior is None
            else (prior.value, prior.quality_status, prior.units)
        )
        if semantic == prior_semantic:
            unchanged += 1
            continue
        if prior is None:
            added += 1
        else:
            changed += 1
        accepted.append(record)
        delta.append(record)
        by_identity[identity] = record
        latest_by_period[period_key] = record
    return (
        normalize_revision_sequences(accepted),
        normalize_revision_sequences(delta),
        added,
        changed,
        unchanged,
    )


def _archive_request_params(
    *,
    snapshot_mode: SnapshotMode,
    contract: _ArchiveContract,
    added_records: int,
    changed_records: int,
    unchanged_records: int,
) -> dict[str, Any]:
    detected_revision_events = tuple(
        event
        for event in contract.lineage_events
        if event.material_revision_rows > 0
    )
    complete_republication_events = tuple(
        event
        for event in contract.lineage_events
        if event.complete_republication
    )
    short_gap_events = tuple(
        event
        for event in contract.lineage_events
        if event.short_gap_auxiliary_evidence
    )
    return {
        "snapshot_mode": snapshot_mode.value,
        "full_response_contract": "fed_h10_official_release_archive_events",
        "official_release_archive_ingest": True,
        "availability_basis": H10_ARCHIVE_AVAILABILITY_BASIS,
        "archive_normal_availability_basis": (
            H10_ARCHIVE_NORMAL_AVAILABILITY_BASIS
        ),
        "archive_correction_availability_basis": (
            H10_ARCHIVE_CORRECTION_AVAILABILITY_BASIS
        ),
        "archive_revision_policy": H10_ARCHIVE_REVISION_POLICY,
        "archive_correction_equivalent_policy": (
            H10_ARCHIVE_CORRECTION_EQUIVALENT_POLICY
        ),
        "archive_correction_equivalent_components": list(
            H10_ARCHIVE_CORRECTION_EQUIVALENT_COMPONENTS
        ),
        "archive_short_gap_auxiliary_days": (
            H10_ARCHIVE_SHORT_GAP_AUXILIARY_DAYS
        ),
        "archive_correction_quarantine_weeks": (
            FX_ARCHIVE_CORRECTION_QUARANTINE_WEEKS
        ),
        "archive_evaluation_start": H10_ARCHIVE_EVALUATION_START.isoformat(),
        "archive_evaluation_start_rationale": (
            H10_ARCHIVE_EVALUATION_START_RATIONALE
        ),
        "archive_release_count": len(contract.release_event_dates),
        "archive_release_event_dates": [
            value.isoformat() for value in contract.release_event_dates
        ],
        "archive_release_event_dates_sha256": _archive_inventory_sha256(
            contract.release_event_dates
        ),
        "archive_correction_count": len(contract.correction_event_dates),
        "archive_declared_correction_count": len(
            contract.correction_event_dates
        ),
        "archive_correction_event_dates": [
            value.isoformat() for value in contract.correction_event_dates
        ],
        "archive_correction_available_at": [
            value.isoformat() for value in contract.correction_available_at
        ],
        "archive_quarantine_event_count": len(
            contract.quarantine_event_dates
        ),
        "archive_quarantine_event_dates": [
            value.isoformat() for value in contract.quarantine_event_dates
        ],
        "archive_quarantine_available_at": [
            value.isoformat() for value in contract.quarantine_available_at
        ],
        "archive_detected_revision_event_count": len(
            detected_revision_events
        ),
        "archive_detected_revision_row_count": sum(
            event.material_revision_rows for event in detected_revision_events
        ),
        "archive_complete_republication_event_count": len(
            complete_republication_events
        ),
        "archive_short_gap_auxiliary_event_count": len(short_gap_events),
        "archive_index_sha256": contract.archive_index_sha256 or None,
        "archive_lineage_sha256": contract.archive_lineage_sha256 or None,
        "archive_lineage_complete": contract.lineage_complete,
        "archive_correction_equivalent_event_lineage": [
            _lineage_event_document(event) for event in contract.lineage_events
        ],
        "added_records": int(added_records),
        "changed_records": int(changed_records),
        "unchanged_records": int(unchanged_records),
        "removed_records": 0,
        "raw_payload_publication": False,
    }


def ingest_h10_archive_store(
    store: SQLiteSnapshotStore,
    collection: H10ArchiveCollection,
    *,
    requested_at: datetime,
    as_of: datetime,
) -> H10StoreRefresh:
    """Atomically append validated official release events to a separate chain."""

    if not isinstance(collection, H10ArchiveCollection):
        raise TypeError("collection must be an H10ArchiveCollection")
    requested = _aware_utc(requested_at, field_name="requested_at")
    cutoff = _aware_utc(as_of, field_name="as_of")
    if collection.requested_at != requested or collection.retrieved_at < requested:
        raise H10StoreError("official H.10 archive collection window is invalid")
    existing = store.read_last_good_observations(
        source=H10_SOURCE,
        dataset=H10_ARCHIVE_DATASET,
    )
    previous = _archive_contract_from_provenance(
        store.get_last_good_provenance(
            source=H10_SOURCE,
            dataset=H10_ARCHIVE_DATASET,
        )
    )
    discovered = (
        tuple(collection.discovered_release_dates)
        if collection.discovered_release_dates
        else tuple(
            sorted(
                set(previous.release_event_dates).union(
                    release.last_update_date for release in collection.releases
                )
            )
        )
    )
    if (
        not discovered
        or discovered != tuple(sorted(set(discovered)))
        or set(previous.release_event_dates).difference(discovered)
        or any(
            release.last_update_date not in discovered
            for release in collection.releases
        )
    ):
        raise H10StoreError("official H.10 archive release inventory is invalid")

    if not _is_sha256(collection.index_sha256):
        raise H10StoreError("official H.10 archive index hash is invalid")
    if collection.lineage is not None:
        contract = _archive_contract_from_lineage(
            discovered,
            archive_index_sha256=collection.index_sha256,
            lineage=collection.lineage,
        )
    elif previous.lineage_complete:
        if collection.releases:
            raise H10StoreError(
                "new H.10 archive releases require complete lineage"
            )
        contract = replace(
            previous,
            release_event_dates=discovered,
            archive_index_sha256=collection.index_sha256,
        )
    else:
        new_corrections = tuple(
            sorted(
                (
                    release.last_update_date,
                    release.available_at,
                )
                for release in collection.releases
                if release.release_date != release.last_update_date
            )
        )
        correction_map = dict(
            zip(
                previous.correction_event_dates,
                previous.correction_available_at,
                strict=True,
            )
        )
        for event_date, available_at in new_corrections:
            prior = correction_map.get(event_date)
            if prior is not None and prior != available_at:
                raise H10StoreError(
                    "official H.10 correction availability changed"
                )
            correction_map[event_date] = available_at
        correction_dates = tuple(sorted(correction_map))
        correction_available = tuple(
            correction_map[key] for key in correction_dates
        )
        contract = _ArchiveContract(
            release_event_dates=discovered,
            correction_event_dates=correction_dates,
            correction_available_at=correction_available,
            quarantine_event_dates=correction_dates,
            quarantine_available_at=correction_available,
        )
    merged, delta, added, changed, unchanged = _merge_archive_records(
        existing,
        collection.records,
    )
    mode = SnapshotMode.DELTA if existing else SnapshotMode.FULL
    result = CollectionResult(
        records=delta if existing else merged,
        health=HealthStatus.OK,
        requests_made=1 + len(collection.releases),
        attempts=1 + len(collection.releases),
        diagnostics={
            "archive_release_count": len(contract.release_event_dates),
            "archive_new_release_count": len(collection.releases),
        },
    )
    prepared = PreparedSnapshot(
        snapshot_result=result,
        effective_records=merged,
        snapshot_mode=mode,
        added_count=added,
        changed_count=changed,
        unchanged_count=unchanged,
        removed_count=0,
    )
    provenance = SnapshotProvenance(
        source=H10_SOURCE,
        dataset=H10_ARCHIVE_DATASET,
        cutoff=cutoff,
        requested_at=requested,
        retrieved_at=collection.retrieved_at,
        quality_status=HealthStatus.OK,
        license_class=H10_LICENSE_CLASS,
        request_params=_archive_request_params(
            snapshot_mode=mode,
            contract=contract,
            added_records=added,
            changed_records=changed,
            unchanged_records=unchanged,
        ),
        response_sha256=collection.collection_sha256,
    )
    snapshot_id = store.write_snapshot(result.records, provenance)
    effective = store.read_last_good_observations(
        source=H10_SOURCE,
        dataset=H10_ARCHIVE_DATASET,
    )
    if effective != merged:
        raise H10StoreError("official H.10 archive storage replay is inconsistent")
    source_status = _analysis_source_status(
        HealthStatus.OK,
        effective,
        as_of=cutoff,
    )
    features, context = _derive_fx_context(
        effective,
        as_of=cutoff,
        source_status=source_status,
        used_last_good=False,
        official_release_archive_ingest=True,
        archive_correction_available_at=contract.quarantine_available_at,
    )
    source_row = _source_row(
        result=result,
        source_status=source_status,
        effective_records=effective,
        as_of=cutoff,
        prepared=prepared,
        used_last_good=False,
        official_release_archive_ingest=True,
        archive_release_count=len(contract.release_event_dates),
        # The publication-facing legacy field denotes every origin-quarantine
        # event, including undeclared correction-equivalent republications.
        archive_correction_available_at=contract.quarantine_available_at,
    )
    return H10StoreRefresh(
        snapshot_id=snapshot_id,
        prepared=prepared,
        effective_records=effective,
        fx_features=features,
        source_row=source_row,
        fx_context=context,
        used_last_good=False,
        **_archive_refresh_metadata(contract),
    )


def refresh_h10_archive_store(
    store: SQLiteSnapshotStore,
    client: H10ArchiveCollector,
    *,
    requested_at: datetime,
    as_of: datetime,
    start_date: date = H10_ARCHIVE_EVALUATION_START,
    end_date: date | None = None,
    clock: Callable[[], datetime] | None = None,
) -> H10StoreRefresh:
    """Refresh only unseen official release dates with private page checkpoints."""

    requested = _aware_utc(requested_at, field_name="requested_at")
    cutoff = _aware_utc(as_of, field_name="as_of")
    if start_date < H10_ARCHIVE_EVALUATION_START:
        raise H10StoreError("official H.10 archive start precedes the frozen segment")
    resolved_end = end_date or cutoff.astimezone(EASTERN).date()
    if resolved_end < start_date:
        raise H10StoreError("official H.10 archive end precedes its start")
    existing = store.read_last_good_observations(
        source=H10_SOURCE,
        dataset=H10_ARCHIVE_DATASET,
    )
    previous = _archive_contract_from_provenance(
        store.get_last_good_provenance(
            source=H10_SOURCE,
            dataset=H10_ARCHIVE_DATASET,
        )
    )
    cached = _load_archive_release_cache(store)

    def cache_release(release: H10ArchiveRelease) -> None:
        _write_archive_release_cache(
            store,
            release,
            requested_at=requested,
            cutoff=cutoff,
        )

    try:
        collection = client.collect(
            requested_at=requested,
            start_date=start_date,
            end_date=resolved_end,
            known_release_dates=previous.release_event_dates,
            cached_releases=cached,
            on_release=cache_release,
        )
        release_inventory = {
            event_date: release
            for event_date, release in cached.items()
            if event_date in set(collection.discovered_release_dates)
        }
        release_inventory.update(
            {
                release.last_update_date: release
                for release in collection.releases
            }
        )
        if set(release_inventory) == set(collection.discovered_release_dates):
            collection = replace(
                collection,
                lineage=detect_h10_archive_correction_equivalents(
                    tuple(
                        release_inventory[event_date]
                        for event_date in collection.discovered_release_dates
                    )
                ),
            )
        return ingest_h10_archive_store(
            store,
            collection,
            requested_at=requested,
            as_of=cutoff,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        failed_at = _aware_utc(
            (clock or (lambda: datetime.now(UTC)))(),
            field_name="failure_at",
        )
        if failed_at < requested:
            raise ValueError("failure_at must not precede requested_at") from exc
        has_last_good = bool(existing)
        status = (
            HealthStatus.DEGRADED if has_last_good else HealthStatus.UNAVAILABLE
        )
        issue = (
            "h10_archive_collection_failed_last_good_retained"
            if has_last_good
            else "h10_archive_collection_failed_no_last_good"
        )
        result = CollectionResult(
            health=status,
            issues=(issue,),
            requests_made=1,
            attempts=1,
            diagnostics={"failure_class": type(exc).__name__},
        )
        mode = SnapshotMode.DELTA if has_last_good else SnapshotMode.FULL
        prepared = PreparedSnapshot(
            snapshot_result=result,
            effective_records=existing,
            snapshot_mode=mode,
        )
        provenance = SnapshotProvenance(
            source=H10_SOURCE,
            dataset=H10_ARCHIVE_DATASET,
            cutoff=cutoff,
            requested_at=requested,
            retrieved_at=failed_at,
            quality_status=status,
            license_class=H10_LICENSE_CLASS,
            request_params=_archive_request_params(
                snapshot_mode=mode,
                contract=previous,
                added_records=0,
                changed_records=0,
                unchanged_records=0,
            ),
            issues=(issue,),
        )
        snapshot_id = store.write_snapshot((), provenance)
        source_status = _analysis_source_status(
            status,
            existing,
            as_of=cutoff,
        )
        features, context = _derive_fx_context(
            existing,
            as_of=cutoff,
            source_status=source_status,
            used_last_good=has_last_good,
            official_release_archive_ingest=has_last_good,
            archive_correction_available_at=previous.quarantine_available_at,
        )
        source_row = _source_row(
            result=result,
            source_status=source_status,
            effective_records=existing,
            as_of=cutoff,
            prepared=prepared,
            used_last_good=has_last_good,
            official_release_archive_ingest=True,
            archive_release_count=len(previous.release_event_dates),
            archive_correction_available_at=previous.correction_available_at,
        )
        return H10StoreRefresh(
            snapshot_id=snapshot_id,
            prepared=prepared,
            effective_records=existing,
            fx_features=features,
            source_row=source_row,
            fx_context=context,
            used_last_good=has_last_good,
            **_archive_refresh_metadata(previous),
        )


def refresh_h10_store(
    store: SQLiteSnapshotStore,
    client: H10Collector,
    *,
    requested_at: datetime,
    as_of: datetime | None = None,
    first_seen_at: datetime | None = None,
    clock: Callable[[], datetime] | None = None,
) -> H10StoreRefresh:
    """Collect a full H.10 ZIP, persist only its incremental observation diff."""

    requested = _aware_utc(requested_at, field_name="requested_at")
    first_seen = (
        _aware_utc(first_seen_at, field_name="first_seen_at")
        if first_seen_at is not None
        else None
    )
    if first_seen is not None and first_seen < requested:
        raise H10StoreError("first_seen_at must not precede requested_at")
    now = clock or (lambda: datetime.now(UTC))
    existing = store.read_last_good_observations(
        source=H10_SOURCE,
        dataset=H10_DATASET,
    )

    parsed: H10ParseResult | None = None
    response_sha256 = ""
    try:
        parsed = client.collect(first_seen_at=first_seen)
        if not isinstance(parsed, H10ParseResult):
            raise H10StoreError("H.10 collector returned an unexpected result")
        if not parsed.records:
            raise H10StoreError("H.10 full response contains no observations")
        try:
            retrieved = _aware_utc(
                parsed.release.retrieved_at,
                field_name="retrieved_at",
            )
        except ValueError as exc:
            raise H10TimestampError(str(exc)) from exc
        if first_seen is not None and first_seen > retrieved:
            raise H10TimestampError(
                "first_seen_at must not follow retrieved_at"
            )
        if retrieved < requested:
            raise H10TimestampError(
                "retrieved_at must not precede requested_at"
            )
        response_sha256 = parsed.release.snapshot_sha256
        collected = CollectionResult(
            records=parsed.records,
            health=HealthStatus.OK,
            requests_made=1,
            attempts=1,
            diagnostics={
                "message_id": parsed.release.message_id,
                "selected_series": len(parsed.series),
            },
        )
    except H10TimestampError as exc:
        # An impossible request/availability window is an integrity error, not
        # provider degradation. Never persist it as either data or health.
        raise H10StoreError(str(exc)) from exc
    except Exception as exc:  # provider boundary must preserve last-good
        retrieved = _aware_utc(now(), field_name="failure_at")
        if retrieved < requested:
            raise ValueError("failure_at must not precede requested_at") from exc
        has_last_good = bool(existing)
        collected = CollectionResult(
            health=(
                HealthStatus.DEGRADED
                if has_last_good
                else HealthStatus.UNAVAILABLE
            ),
            issues=(
                "h10_collection_failed_last_good_retained"
                if has_last_good
                else "h10_collection_failed_no_last_good",
            ),
            requests_made=1,
            attempts=1,
            diagnostics={"failure_class": type(exc).__name__},
        )

    prepared = prepare_incremental_snapshot(existing, collected)
    attempt_result = prepared.snapshot_result
    context_as_of = (
        _aware_utc(as_of, field_name="as_of") if as_of is not None else retrieved
    )
    provenance = SnapshotProvenance(
        source=H10_SOURCE,
        dataset=H10_DATASET,
        cutoff=context_as_of,
        requested_at=requested,
        retrieved_at=retrieved,
        quality_status=attempt_result.health,
        license_class=H10_LICENSE_CLASS,
        request_params={
            "snapshot_mode": prepared.snapshot_mode.value,
            "full_response_contract": "fed_h10_release_xml_full_history",
            "stored_records": len(attempt_result.records),
            "added_records": prepared.added_count,
            "changed_records": prepared.changed_count,
            "unchanged_records": prepared.unchanged_count,
            "removed_records": prepared.removed_count,
            "raw_payload_publication": False,
        },
        response_sha256=response_sha256,
        issues=attempt_result.issues,
    )
    stored_records = (
        attempt_result.records
        if attempt_result.health is HealthStatus.OK
        else ()
    )
    snapshot_id = store.write_snapshot(stored_records, provenance)
    effective = store.read_last_good_observations(
        source=H10_SOURCE,
        dataset=H10_DATASET,
    )
    used_last_good = attempt_result.health is not HealthStatus.OK and bool(effective)
    source_status = _analysis_source_status(
        attempt_result.health,
        effective,
        as_of=context_as_of,
    )
    fx_features, fx_context = _derive_fx_context(
        effective,
        as_of=context_as_of,
        source_status=source_status,
        used_last_good=used_last_good,
    )
    archive_effective = store.read_last_good_observations(
        source=H10_SOURCE,
        dataset=H10_ARCHIVE_DATASET,
    )
    archive_provenance = store.get_last_good_provenance(
        source=H10_SOURCE,
        dataset=H10_ARCHIVE_DATASET,
    )
    if bool(archive_effective) != (archive_provenance is not None):
        raise H10StoreError("official H.10 archive chain is incomplete")
    selected_archive_contract = _ArchiveContract((), (), ())
    if archive_effective:
        selected_archive_contract = _archive_contract_from_provenance(
            archive_provenance
        )
        archive_status = _analysis_source_status(
            HealthStatus.OK,
            archive_effective,
            as_of=context_as_of,
        )
        archive_features, archive_context = _derive_fx_context(
            archive_effective,
            as_of=context_as_of,
            source_status=archive_status,
            used_last_good=False,
            official_release_archive_ingest=True,
            archive_correction_available_at=(
                selected_archive_contract.quarantine_available_at
            ),
        )
        archive_readiness = fx_ablation_readiness(
            archive_features,
            _readiness_cutoffs(archive_features, as_of=context_as_of),
        )
        if (
            archive_status is HealthStatus.OK
            and archive_features is not None
            and archive_readiness["status"] == "ready_for_evaluation"
        ):
            effective = archive_effective
            fx_features = archive_features
            fx_context = archive_context
            if attempt_result.health is not HealthStatus.OK:
                source_status = attempt_result.health
                used_last_good = True
                fx_context["status"] = "degraded"
                fx_context["source_status"] = source_status.value
                fx_context["last_good_used"] = True
            else:
                source_status = archive_status
    reported_archive_contract = (
        selected_archive_contract
        if fx_features is not None
        and fx_features.official_release_archive_ingest
        else _ArchiveContract((), (), ())
    )
    source_row = _source_row(
        result=attempt_result,
        source_status=source_status,
        effective_records=effective,
        as_of=context_as_of,
        prepared=prepared,
        used_last_good=used_last_good,
        official_release_archive_ingest=(
            bool(fx_features)
            and fx_features.official_release_archive_ingest
        ),
        archive_release_count=(
            len(selected_archive_contract.release_event_dates)
            if fx_features is not None
            and fx_features.official_release_archive_ingest
            else 0
        ),
        archive_correction_available_at=(
            selected_archive_contract.quarantine_available_at
            if fx_features is not None
            and fx_features.official_release_archive_ingest
            else ()
        ),
    )
    return H10StoreRefresh(
        snapshot_id=snapshot_id,
        prepared=prepared,
        effective_records=effective,
        fx_features=fx_features,
        source_row=source_row,
        fx_context=fx_context,
        used_last_good=used_last_good,
        **_archive_refresh_metadata(reported_archive_contract),
    )
