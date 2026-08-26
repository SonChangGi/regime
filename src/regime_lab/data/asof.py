"""Leakage-safe weekly joins over revision-aware observations."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from heapq import heappop, heappush
from typing import Iterable, Literal, Mapping, Sequence
from zoneinfo import ZoneInfo

from .contracts import HealthStatus, Observation, combine_health, ensure_utc


@dataclass(frozen=True, slots=True)
class AsOfValue:
    cutoff: datetime
    source: str
    series_id: str
    value: float | None
    observed_period_end: date | None
    released_at: datetime | None
    source_released_at: datetime | None
    available_at: datetime | None
    provider_first_seen_at: datetime | None
    system_retrieved_at: datetime | None
    vintage_date: date | None
    revision_seq: int | None
    raw_sha256: str | None
    age_days: int | None
    release_lag_days: int | None
    is_filled: bool
    quality_status: HealthStatus

    def __post_init__(self) -> None:
        object.__setattr__(self, "cutoff", ensure_utc(self.cutoff, field_name="cutoff"))
        object.__setattr__(self, "quality_status", HealthStatus(self.quality_status))
        if self.available_at is not None:
            available = ensure_utc(self.available_at, field_name="available_at")
            if available > self.cutoff:
                raise ValueError("as-of value leaks a future available_at")
            object.__setattr__(self, "available_at", available)
        for field_name in (
            "released_at",
            "source_released_at",
            "provider_first_seen_at",
            "system_retrieved_at",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    ensure_utc(value, field_name=field_name),
                )


def weekly_asof_join(
    cutoffs: Sequence[datetime],
    observations: Iterable[Observation],
    *,
    required_series: Sequence[tuple[str, str]] | None = None,
    max_age_by_series: Mapping[tuple[str, str] | str, timedelta] | None = None,
    include_missing_values: bool = False,
    availability_basis: Literal[
        "source", "operational", "reconstructed_market"
    ] = "source",
) -> tuple[AsOfValue, ...]:
    """Select the latest eligible vintage for every series and weekly cutoff.

    Selection order is latest observation period, then latest eligibility
    timestamp, then revision sequence.  ``source`` preserves the legacy
    ``available_at <= cutoff`` research contract.  ``operational`` instead uses
    ``max(source_released_at, provider_first_seen_at) <= cutoff`` so a value
    cannot be replayed before the provider was actually observed.
    ``reconstructed_market`` is the explicitly retrospective research basis:
    Alpha Vantage weekly bars are aligned to their market-period end while all
    other sources retain their source availability clock.  It exists to avoid
    silently shifting the frozen market label by one week when the provider's
    weekly endpoint is first finalized or observed after the Friday close.
    Tombstones participate in per-period revision ordering regardless.  The
    default then selects the latest non-null period (preserving forward-fill
    across holidays); ``include_missing_values`` instead exposes the newest
    period's tombstone metadata for audit.
    """

    if availability_basis not in {
        "source",
        "operational",
        "reconstructed_market",
    }:
        raise ValueError(
            "availability_basis must be source, operational, or reconstructed_market"
        )

    eastern = ZoneInfo("America/New_York")

    def reconstructed_market_time(record: Observation) -> datetime:
        local = datetime.combine(
            record.observed_period_end,
            time(16, 0),
            tzinfo=eastern,
        )
        return local.astimezone(timezone.utc)

    def eligibility_time(record: Observation) -> datetime:
        if availability_basis == "source":
            return record.available_at
        if availability_basis == "operational":
            return record.operating_available_at
        if record.source == "alpha_vantage":
            return reconstructed_market_time(record)
        return record.available_at

    normalized_cutoffs = tuple(sorted({ensure_utc(item, field_name="cutoff") for item in cutoffs}))
    records = tuple(observations)
    grouped: dict[tuple[str, str], list[Observation]] = {}
    for record in records:
        # Missing values are revision events too.  Filtering them before the
        # PIT ordering would let an older non-null value for the same period
        # reappear after the provider explicitly tombstoned it.
        grouped.setdefault((record.source, record.series_id), []).append(record)
    for group in grouped.values():
        group.sort(
            key=lambda item: (
                eligibility_time(item),
                item.observed_period_end,
                item.revision_seq,
                item.retrieved_at,
            )
        )

    keys = tuple(dict.fromkeys(required_series or tuple(sorted(grouped))))
    max_ages = max_age_by_series or {}
    available_times = {
        key: tuple(eligibility_time(record) for record in grouped.get(key, ()))
        for key in keys
    }
    positions = {key: 0 for key in keys}
    latest_by_period: dict[
        tuple[str, str],
        dict[date, Observation],
    ] = {key: {} for key in keys}
    candidates: dict[
        tuple[str, str],
        list[tuple[int, int, Observation]],
    ] = {key: [] for key in keys}
    pending: dict[
        tuple[str, str],
        list[tuple[int, datetime, int, Observation]],
    ] = {key: [] for key in keys}
    serial = 0

    def revision_key(record: Observation) -> tuple[datetime, date, int, datetime, str]:
        return (
            eligibility_time(record),
            record.vintage_date,
            record.revision_seq,
            record.retrieved_at,
            record.raw_sha256,
        )

    def consider(key: tuple[str, str], record: Observation) -> None:
        nonlocal serial
        period_state = latest_by_period[key]
        current = period_state.get(record.observed_period_end)
        if current is None or revision_key(record) > revision_key(current):
            period_state[record.observed_period_end] = record
            # In model mode, a null event invalidates the old candidate for
            # this period but is not itself selectable.  Lazy heap validation
            # removes that stale old revision without rescanning all periods.
            if include_missing_values or record.value is not None:
                serial += 1
                heappush(
                    candidates[key],
                    (-record.observed_period_end.toordinal(), serial, record),
                )

    output: list[AsOfValue] = []
    for cutoff in normalized_cutoffs:
        for source, series_id in keys:
            key = (source, series_id)
            group = grouped.get(key, ())
            left = positions[key]
            right = bisect_right(available_times[key], cutoff, lo=left)
            for record in group[left:right]:
                if record.observed_period_end <= cutoff.date():
                    consider(key, record)
                else:
                    serial += 1
                    heappush(
                        pending[key],
                        (
                            record.observed_period_end.toordinal(),
                            eligibility_time(record),
                            serial,
                            record,
                        ),
                    )
            positions[key] = right
            cutoff_ordinal = cutoff.date().toordinal()
            while pending[key] and pending[key][0][0] <= cutoff_ordinal:
                _, _, _, newly_eligible = heappop(pending[key])
                consider(key, newly_eligible)

            candidate_heap = candidates[key]
            selected: Observation | None = None
            while candidate_heap:
                _, _, candidate = candidate_heap[0]
                current = latest_by_period[key].get(candidate.observed_period_end)
                if current is not candidate or (
                    not include_missing_values and current.value is None
                ):
                    heappop(candidate_heap)
                    continue
                selected = current
                break

            if selected is None:
                output.append(
                    AsOfValue(
                        cutoff=cutoff,
                        source=source,
                        series_id=series_id,
                        value=None,
                        observed_period_end=None,
                        released_at=None,
                        source_released_at=None,
                        available_at=None,
                        provider_first_seen_at=None,
                        system_retrieved_at=None,
                        vintage_date=None,
                        revision_seq=None,
                        raw_sha256=None,
                        age_days=None,
                        release_lag_days=None,
                        is_filled=False,
                        quality_status=HealthStatus.UNAVAILABLE,
                    )
                )
                continue
            age_days = (cutoff.date() - selected.observed_period_end).days
            selected_available_at = eligibility_time(selected)
            release_lag_days = (
                selected_available_at.date() - selected.observed_period_end
            ).days
            status = selected.quality_status
            if selected.value is None:
                status = combine_health((status, HealthStatus.UNAVAILABLE))
            max_age = max_ages.get((source, series_id), max_ages.get(series_id))
            if max_age is not None and cutoff - selected_available_at > max_age:
                status = combine_health((status, HealthStatus.STALE))
            output.append(
                AsOfValue(
                    cutoff=cutoff,
                    source=source,
                    series_id=series_id,
                    value=selected.value,
                    observed_period_end=selected.observed_period_end,
                    released_at=selected.released_at,
                    source_released_at=selected.source_released_at,
                    available_at=selected_available_at,
                    provider_first_seen_at=selected.provider_first_seen_at,
                    system_retrieved_at=selected.system_retrieved_at,
                    vintage_date=selected.vintage_date,
                    revision_seq=selected.revision_seq,
                    raw_sha256=selected.raw_sha256,
                    age_days=age_days,
                    release_lag_days=release_lag_days,
                    is_filled=selected.observed_period_end < cutoff.date(),
                    quality_status=status,
                )
            )
    return tuple(output)


def weekly_asof_frame(*args: object, **kwargs: object):
    """Pandas convenience wrapper imported lazily for a lightweight core."""

    import pandas as pd

    rows = weekly_asof_join(*args, **kwargs)
    return pd.DataFrame(asdict(row) for row in rows)
