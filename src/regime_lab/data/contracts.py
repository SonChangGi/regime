"""Shared, provider-neutral data contracts.

The central invariant in this module is that ``available_at`` means the first
timestamp at which a value may be used by a model.  Point-in-time consumers
must never substitute ``retrieved_at`` or an observation period for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from enum import StrEnum
import math
from typing import Any, Iterable, Mapping, Sequence


UTC = timezone.utc


class HealthStatus(StrEnum):
    """Machine-readable source and record health states."""

    OK = "ok"
    UNAVAILABLE = "unavailable"
    STALE = "stale"
    DEGRADED = "degraded"
    QUOTA_EXHAUSTED = "quota_exhausted"
    SCHEMA_CHANGED = "schema_changed"
    REVISION_GAP = "revision_gap"
    RIGHTS_UNCONFIRMED = "rights_unconfirmed"
    LICENSE_BLOCKED = "license_blocked"


class SnapshotMode(StrEnum):
    """Whether a successful snapshot replaces or extends prior history."""

    FULL = "full"
    DELTA = "delta"


_HEALTH_PRIORITY = {
    HealthStatus.OK: 0,
    HealthStatus.UNAVAILABLE: 15,
    HealthStatus.STALE: 10,
    HealthStatus.DEGRADED: 20,
    HealthStatus.REVISION_GAP: 30,
    HealthStatus.QUOTA_EXHAUSTED: 40,
    HealthStatus.RIGHTS_UNCONFIRMED: 50,
    HealthStatus.SCHEMA_CHANGED: 60,
    HealthStatus.LICENSE_BLOCKED: 70,
}


def combine_health(statuses: Iterable[HealthStatus]) -> HealthStatus:
    """Return the most severe status in a deterministic order."""

    return max(statuses, key=_HEALTH_PRIORITY.get, default=HealthStatus.OK)


def ensure_utc(value: datetime, *, field_name: str = "timestamp") -> datetime:
    """Normalize an aware timestamp to UTC, rejecting ambiguous naive values."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class Observation:
    """A single revision-aware observation suitable for PIT replay."""

    source: str
    series_id: str
    observed_period_end: date
    value: float | None
    released_at: datetime | None
    available_at: datetime
    vintage_date: date
    retrieved_at: datetime
    revision_seq: int = 0
    units: str = ""
    adjustment: str = ""
    license_class: str = ""
    quality_status: HealthStatus = HealthStatus.OK
    raw_sha256: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("source must not be empty")
        if not self.series_id.strip():
            raise ValueError("series_id must not be empty")
        if self.revision_seq < 0:
            raise ValueError("revision_seq must be non-negative")
        if self.value is not None and not math.isfinite(float(self.value)):
            raise ValueError("value must be finite or None")

        available_at = ensure_utc(self.available_at, field_name="available_at")
        retrieved_at = ensure_utc(self.retrieved_at, field_name="retrieved_at")
        released_at = (
            ensure_utc(self.released_at, field_name="released_at")
            if self.released_at is not None
            else None
        )
        if released_at is not None and released_at > available_at:
            raise ValueError("released_at must not be after available_at")
        if retrieved_at < available_at:
            raise ValueError("retrieved_at must not be before available_at")

        status = HealthStatus(self.quality_status)
        object.__setattr__(self, "available_at", available_at)
        object.__setattr__(self, "retrieved_at", retrieved_at)
        object.__setattr__(self, "released_at", released_at)
        object.__setattr__(self, "quality_status", status)
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.value is not None:
            object.__setattr__(self, "value", float(self.value))

    def with_revision_seq(self, revision_seq: int) -> "Observation":
        return replace(self, revision_seq=revision_seq)


@dataclass(frozen=True, slots=True)
class CollectionResult:
    """Non-throwing boundary returned by provider adapters."""

    records: tuple[Observation, ...] = ()
    health: HealthStatus = HealthStatus.OK
    issues: tuple[str, ...] = ()
    requests_made: int = 0
    attempts: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "issues", tuple(self.issues))
        object.__setattr__(self, "health", HealthStatus(self.health))
        if self.requests_made < 0 or self.attempts < 0:
            raise ValueError("request counts must be non-negative")

    @property
    def ok(self) -> bool:
        return self.health is HealthStatus.OK


def normalize_revision_sequences(
    records: Iterable[Observation],
) -> tuple[Observation, ...]:
    """Assign globally stable revision numbers within each observation period.

    Provider requests may split one series across realtime chunks.  Each chunk
    necessarily numbers its local revisions from zero, so callers must run this
    function after concatenating chunks.  Input ``revision_seq`` values are
    deliberately ignored; availability and vintage timestamps define the
    global order.
    """

    grouped: dict[tuple[str, str, date], list[Observation]] = {}
    for record in records:
        grouped.setdefault(
            (record.source, record.series_id, record.observed_period_end),
            [],
        ).append(record)

    normalized: list[Observation] = []
    for group in grouped.values():
        ordered = sorted(
            group,
            key=lambda item: (
                item.available_at,
                item.vintage_date,
                item.raw_sha256,
                "" if item.value is None else repr(item.value),
                item.retrieved_at,
            ),
        )
        normalized.extend(
            item.with_revision_seq(revision_seq)
            for revision_seq, item in enumerate(ordered)
        )
    normalized.sort(
        key=lambda item: (
            item.source,
            item.series_id,
            item.observed_period_end,
            item.available_at,
            item.vintage_date,
            item.revision_seq,
        )
    )
    return tuple(normalized)


def merge_collection_results(
    results: Sequence[CollectionResult],
    *,
    normalize_revisions: bool = False,
) -> CollectionResult:
    """Merge chunked provider results without losing health or request audit."""

    records = tuple(record for result in results for record in result.records)
    if normalize_revisions:
        records = normalize_revision_sequences(records)
    return CollectionResult(
        records=records,
        health=combine_health(result.health for result in results),
        issues=tuple(
            dict.fromkeys(issue for result in results for issue in result.issues)
        ),
        requests_made=sum(result.requests_made for result in results),
        attempts=sum(result.attempts for result in results),
    )


def provenance_safe_result(result: CollectionResult) -> CollectionResult:
    """Keep successful records; make every non-OK snapshot provenance-only."""

    if result.health is HealthStatus.OK:
        return result
    return CollectionResult(
        records=(),
        health=result.health,
        issues=result.issues,
        requests_made=result.requests_made,
        attempts=result.attempts,
    )


ObservationNaturalKey = tuple[str, str, date, date, datetime]
ObservationPeriodKey = tuple[str, str, date]


def observation_natural_key(record: Observation) -> ObservationNaturalKey:
    """Provider event identity used for overlap and full-response diffs."""

    return (
        record.source,
        record.series_id,
        record.observed_period_end,
        record.vintage_date,
        record.available_at,
    )


def _provider_value_key(record: Observation) -> tuple[Any, ...]:
    """Comparable provider value, excluding PIT/discovery bookkeeping."""

    return (
        record.value,
        record.units,
        record.adjustment,
        record.license_class,
        record.quality_status,
        record.metadata.get("symbol"),
        record.metadata.get("field"),
    )


def _index_observations(
    records: Iterable[Observation],
) -> dict[ObservationNaturalKey, Observation]:
    indexed: dict[ObservationNaturalKey, Observation] = {}
    for record in sorted(
        records,
        key=lambda item: (
            item.retrieved_at,
            item.available_at,
            item.revision_seq,
            item.raw_sha256,
        ),
    ):
        indexed[observation_natural_key(record)] = record
    return indexed


def _period_key(record: Observation) -> ObservationPeriodKey:
    return (record.source, record.series_id, record.observed_period_end)


def _event_order(record: Observation) -> tuple[Any, ...]:
    return (
        record.available_at,
        record.vintage_date,
        record.revision_seq,
        record.retrieved_at,
        record.raw_sha256,
    )


def _latest_by_period(
    records: Iterable[Observation],
) -> dict[ObservationPeriodKey, Observation]:
    latest: dict[ObservationPeriodKey, Observation] = {}
    for record in records:
        key = _period_key(record)
        current = latest.get(key)
        if current is None or _event_order(record) > _event_order(current):
            latest[key] = record
    return latest


def _prospective_revision(
    record: Observation,
    *,
    reason: str,
) -> Observation:
    """Move a later-discovered historical provider value to discovery time."""

    discovered_at = record.retrieved_at
    metadata = dict(record.metadata)
    metadata.update(
        {
            "prospective_revision": True,
            "prospective_revision_reason": reason,
            "provider_reported_available_at": record.available_at.isoformat(),
            "provider_reported_vintage_date": record.vintage_date.isoformat(),
            "availability_precision": "collection_discovery_time",
        }
    )
    return replace(
        record,
        released_at=discovered_at,
        available_at=discovered_at,
        vintage_date=discovered_at.date(),
        revision_seq=0,
        metadata=metadata,
    )


@dataclass(frozen=True, slots=True)
class PreparedSnapshot:
    """Storage result and full effective history for one provider response."""

    snapshot_result: CollectionResult
    effective_records: tuple[Observation, ...]
    snapshot_mode: SnapshotMode
    added_count: int = 0
    changed_count: int = 0
    unchanged_count: int = 0
    removed_count: int = 0


def prepare_incremental_snapshot(
    existing_records: Iterable[Observation],
    collected: CollectionResult,
) -> PreparedSnapshot:
    """Prepare an append-only snapshot from a provider's full response.

    Successful full responses store only new periods and prospective revision
    events.  A changed historical value becomes available at its collection
    timestamp rather than being backdated to the observation period.  If a
    prior period disappears, the response is degraded and provenance-only:
    without an explicit provider tombstone or confirmation policy, omission is
    not evidence that previously accepted history should be deleted.
    """

    existing = _index_observations(existing_records)
    if collected.health is not HealthStatus.OK:
        return PreparedSnapshot(
            snapshot_result=provenance_safe_result(collected),
            effective_records=normalize_revision_sequences(existing.values()),
            snapshot_mode=(SnapshotMode.DELTA if existing else SnapshotMode.FULL),
        )

    incoming = _index_observations(collected.records)
    existing_latest = _latest_by_period(existing.values())
    incoming_latest = _latest_by_period(incoming.values())
    existing_keys = set(existing_latest)
    incoming_keys = set(incoming_latest)
    added_keys = incoming_keys - existing_keys
    removed_keys = existing_keys - incoming_keys
    common_keys = existing_keys & incoming_keys
    changed_keys = {
        key
        for key in common_keys
        if _provider_value_key(existing_latest[key])
        != _provider_value_key(incoming_latest[key])
    }
    unchanged_count = len(common_keys - changed_keys)

    if not existing:
        effective = normalize_revision_sequences(incoming.values())
        return PreparedSnapshot(
            snapshot_result=CollectionResult(
                records=effective,
                health=collected.health,
                issues=collected.issues,
                requests_made=collected.requests_made,
                attempts=collected.attempts,
            ),
            effective_records=effective,
            snapshot_mode=SnapshotMode.FULL,
            added_count=len(incoming_keys),
        )

    if removed_keys:
        issue = (
            "provider full response omitted previously accepted observations; "
            "kept last-good history pending an explicit tombstone policy"
        )
        return PreparedSnapshot(
            snapshot_result=CollectionResult(
                records=(),
                health=HealthStatus.DEGRADED,
                issues=tuple(dict.fromkeys((*collected.issues, issue))),
                requests_made=collected.requests_made,
                attempts=collected.attempts,
            ),
            effective_records=normalize_revision_sequences(existing.values()),
            snapshot_mode=SnapshotMode.DELTA,
            added_count=len(added_keys),
            changed_count=len(changed_keys),
            unchanged_count=unchanged_count,
            removed_count=len(removed_keys),
        )

    unsafe_changes = {
        key
        for key in changed_keys
        if incoming_latest[key].retrieved_at <= existing_latest[key].available_at
    }
    if unsafe_changes:
        issue = (
            "provider revision discovery time did not advance beyond last-good; "
            "kept last-good history"
        )
        return PreparedSnapshot(
            snapshot_result=CollectionResult(
                records=(),
                health=HealthStatus.DEGRADED,
                issues=tuple(dict.fromkeys((*collected.issues, issue))),
                requests_made=collected.requests_made,
                attempts=collected.attempts,
            ),
            effective_records=normalize_revision_sequences(existing.values()),
            snapshot_mode=SnapshotMode.DELTA,
            added_count=len(added_keys),
            changed_count=len(changed_keys),
            unchanged_count=unchanged_count,
        )

    latest_period_by_series: dict[tuple[str, str], date] = {}
    for source, series_id, period_end in existing_keys:
        series_key = (source, series_id)
        latest_period_by_series[series_key] = max(
            period_end,
            latest_period_by_series.get(series_key, period_end),
        )

    snapshot_candidates: list[Observation] = []
    for key in sorted(added_keys):
        incoming_record = incoming_latest[key]
        source, series_id, period_end = key
        previous_latest_period = latest_period_by_series.get((source, series_id))
        if previous_latest_period is not None and period_end < previous_latest_period:
            incoming_record = _prospective_revision(
                incoming_record,
                reason="historical_backfill",
            )
        snapshot_candidates.append(incoming_record)
    snapshot_candidates.extend(
        _prospective_revision(
            incoming_latest[key],
            reason="provider_value_changed",
        )
        for key in sorted(changed_keys)
    )

    snapshot_records = normalize_revision_sequences(snapshot_candidates)
    effective_events = dict(existing)
    for record in snapshot_records:
        effective_events[observation_natural_key(record)] = record
    effective = normalize_revision_sequences(effective_events.values())
    return PreparedSnapshot(
        snapshot_result=CollectionResult(
            records=snapshot_records,
            health=collected.health,
            issues=collected.issues,
            requests_made=collected.requests_made,
            attempts=collected.attempts,
        ),
        effective_records=effective,
        snapshot_mode=SnapshotMode.DELTA,
        added_count=len(added_keys),
        changed_count=len(changed_keys),
        unchanged_count=unchanged_count,
        removed_count=len(removed_keys),
    )


@dataclass(frozen=True, slots=True)
class SnapshotProvenance:
    """Audit metadata for one immutable provider snapshot."""

    source: str
    dataset: str
    cutoff: datetime
    requested_at: datetime
    retrieved_at: datetime
    quality_status: HealthStatus
    license_class: str = ""
    request_params: Mapping[str, Any] = field(default_factory=dict)
    response_sha256: str = ""
    issues: tuple[str, ...] = ()
    snapshot_id: str | None = None

    def __post_init__(self) -> None:
        if not self.source.strip() or not self.dataset.strip():
            raise ValueError("source and dataset must not be empty")
        cutoff = ensure_utc(self.cutoff, field_name="cutoff")
        requested_at = ensure_utc(self.requested_at, field_name="requested_at")
        retrieved_at = ensure_utc(self.retrieved_at, field_name="retrieved_at")
        if retrieved_at < requested_at:
            raise ValueError("retrieved_at must not be before requested_at")
        object.__setattr__(self, "cutoff", cutoff)
        object.__setattr__(self, "requested_at", requested_at)
        object.__setattr__(self, "retrieved_at", retrieved_at)
        object.__setattr__(self, "quality_status", HealthStatus(self.quality_status))
        object.__setattr__(self, "request_params", dict(self.request_params))
        object.__setattr__(self, "issues", tuple(self.issues))


@dataclass(frozen=True, slots=True)
class RealtimeCollectionWindow:
    """Inclusive provider window plus the full observation-history boundary."""

    realtime_start: date
    realtime_end: date
    observation_start: date
    snapshot_mode: SnapshotMode

    def __post_init__(self) -> None:
        if self.observation_start > self.realtime_start:
            raise ValueError("observation_start must not be after realtime_start")
        if self.realtime_start > self.realtime_end:
            raise ValueError("realtime_start must not be after realtime_end")
        object.__setattr__(self, "snapshot_mode", SnapshotMode(self.snapshot_mode))


def snapshot_mode_from_provenance(provenance: SnapshotProvenance) -> SnapshotMode:
    """Read mode from provenance; legacy snapshots are complete/full."""

    raw_mode = provenance.request_params.get("snapshot_mode", SnapshotMode.FULL.value)
    return SnapshotMode(str(raw_mode))


def plan_incremental_realtime_window(
    last_good: SnapshotProvenance | None,
    *,
    history_start: date,
    realtime_end: date,
    observation_start: date | None = None,
    overlap_days: int = 1,
) -> RealtimeCollectionWindow:
    """Plan an inclusive ALFRED delta without dropping historical revisions.

    One overlap day means the prior successful ``realtime_end`` is requested
    again.  ``observation_start`` remains at the original observation-history
    boundary (which may predate the realtime history), allowing a new realtime
    event to revise any older observation period.
    """

    if history_start > realtime_end:
        raise ValueError("history_start must not be after realtime_end")
    if overlap_days < 1:
        raise ValueError("overlap_days must be at least 1")
    observation_history_start = observation_start or history_start
    if observation_history_start > history_start:
        raise ValueError("observation_start must not be after history_start")
    if last_good is None:
        return RealtimeCollectionWindow(
            realtime_start=history_start,
            realtime_end=realtime_end,
            observation_start=observation_history_start,
            snapshot_mode=SnapshotMode.FULL,
        )

    raw_previous_end = last_good.request_params.get("realtime_end")
    raw_vintage_dates = last_good.request_params.get("vintage_dates")
    if raw_previous_end is not None:
        try:
            previous_end = date.fromisoformat(str(raw_previous_end))
        except ValueError as exc:
            raise ValueError("last-good realtime_end is invalid") from exc
    elif raw_vintage_dates:
        if isinstance(raw_vintage_dates, str):
            values = tuple(
                item.strip() for item in raw_vintage_dates.split(",") if item.strip()
            )
        elif isinstance(raw_vintage_dates, Sequence):
            values = tuple(str(item) for item in raw_vintage_dates)
        else:
            raise ValueError("last-good vintage_dates is invalid")
        try:
            previous_end = max(date.fromisoformat(item) for item in values)
        except (ValueError, TypeError) as exc:
            raise ValueError("last-good vintage_dates is invalid") from exc
    else:
        previous_end = last_good.cutoff.date()
    if previous_end > realtime_end:
        raise ValueError("realtime_end is before the last successful collection")
    # Inclusive endpoints: subtracting overlap_days - 1 yields exactly the
    # requested number of repeated calendar dates.
    start = previous_end - timedelta(days=overlap_days - 1)
    start = max(history_start, start)
    return RealtimeCollectionWindow(
        realtime_start=start,
        realtime_end=realtime_end,
        observation_start=observation_history_start,
        snapshot_mode=SnapshotMode.DELTA,
    )
