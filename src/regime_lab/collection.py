"""Live provider orchestration with bounded cost, provenance, and PIT cutoffs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from regime_lab.data import (
    AlfredClient,
    AlfredConfig,
    AlphaVantageClient,
    AlphaVantageConfig,
    CollectionResult,
    DailyRequestBudget,
    HealthStatus,
    Observation,
    RetryPolicy,
    SQLiteSnapshotStore,
    SnapshotMode,
    SnapshotProvenance,
    combine_health,
    merge_collection_results,
    plan_incremental_realtime_window,
    prepare_incremental_snapshot,
)


EASTERN = ZoneInfo("America/New_York")
UTC = timezone.utc


@dataclass(frozen=True)
class LiveCollection:
    records: tuple[Observation, ...]
    cutoffs: tuple[datetime, ...]
    sources: tuple[dict[str, Any], ...]
    overall_health: HealthStatus
    issues: tuple[str, ...]
    model_cutoff: datetime
    database_path: Path


class CollectionGateError(RuntimeError):
    """The completed provider pass is not safe to use for model analysis."""


def _aware_utc(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)


def alpha_market_week_is_current(
    *,
    available_at: datetime,
    coverage_end: date,
    cutoff: datetime,
) -> bool:
    """Accept the latest trading day in the cutoff's market week.

    Alpha's weekly timestamp is normally Friday, but U.S. market holidays such
    as Good Friday legitimately produce a Thursday period.  The cutoff remains
    Friday 16:00 ET; a prior market week is never accepted.
    """

    available_local = _aware_utc(
        available_at, label="Alpha Vantage available_at"
    ).astimezone(EASTERN)
    cutoff_local = _aware_utc(cutoff, label="cutoff").astimezone(EASTERN)
    week_start = cutoff_local.date() - timedelta(days=cutoff_local.weekday())
    return bool(
        week_start <= coverage_end <= cutoff_local.date()
        and available_local.date() == coverage_end
        and available_local.time().replace(tzinfo=None)
        == cutoff_local.time().replace(tzinfo=None)
        and available_local <= cutoff_local
    )


def _configured_timeout(
    config: Mapping[str, Any],
    *,
    provider: str,
    default: float,
) -> float:
    raw = config.get("timeout_seconds", default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{provider} timeout_seconds must be numeric")
    timeout = float(raw)
    if not math.isfinite(timeout) or not 1 <= timeout <= 300:
        raise ValueError(f"{provider} timeout_seconds must be in [1, 300]")
    return timeout


def _configured_retry(
    config: Mapping[str, Any],
    *,
    provider: str,
    default_attempts: int,
    require_reserve: bool = False,
) -> tuple[RetryPolicy, int]:
    raw_retry = config.get("retry", {})
    if not isinstance(raw_retry, Mapping):
        raise ValueError(f"{provider} retry must be a mapping")

    raw_attempts = raw_retry.get("max_attempts", default_attempts)
    raw_backoff = raw_retry.get("backoff_seconds", 0.25)
    raw_max_backoff = raw_retry.get("max_backoff_seconds", 4.0)
    raw_reserve = raw_retry.get("reserve_calls", 0)
    if isinstance(raw_attempts, bool) or type(raw_attempts) is not int:
        raise ValueError(f"{provider} retry max_attempts must be an integer")
    if not 1 <= raw_attempts <= 5:
        raise ValueError(f"{provider} retry max_attempts must be in [1, 5]")
    for name, raw_value in (
        ("backoff_seconds", raw_backoff),
        ("max_backoff_seconds", raw_max_backoff),
    ):
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(f"{provider} retry {name} must be numeric")
        value = float(raw_value)
        if not math.isfinite(value) or not 0 <= value <= 300:
            raise ValueError(f"{provider} retry {name} must be in [0, 300]")
    if isinstance(raw_reserve, bool) or type(raw_reserve) is not int:
        raise ValueError(f"{provider} retry reserve_calls must be an integer")
    if not 0 <= raw_reserve <= 25:
        raise ValueError(f"{provider} retry reserve_calls must be in [0, 25]")
    if require_reserve and raw_attempts - 1 > raw_reserve and raw_attempts > 1:
        raise ValueError(
            f"{provider} retry reserve_calls must cover max_attempts - 1"
        )
    return (
        RetryPolicy(
            max_attempts=raw_attempts,
            backoff_seconds=float(raw_backoff),
            max_backoff_seconds=float(raw_max_backoff),
        ),
        raw_reserve,
    )


def _with_identified_issues(
    result: CollectionResult,
    *,
    identifier: str,
) -> CollectionResult:
    issues = tuple(
        issue if identifier in issue else f"{identifier}: {issue}"
        for issue in result.issues
    )
    if issues == result.issues:
        return result
    return CollectionResult(
        records=result.records,
        health=result.health,
        issues=issues,
        requests_made=result.requests_made,
        attempts=result.attempts,
        diagnostics=result.diagnostics,
    )


def _identify_alpha_transport_failure(
    result: CollectionResult,
    *,
    requested_symbols: Sequence[str],
) -> CollectionResult:
    transport_markers = (
        "Alpha Vantage request failed:",
        "Alpha Vantage rolling 24-hour request budget exhausted",
        "Alpha Vantage reported a quota or entitlement limit",
    )
    if not any(
        marker in issue for marker in transport_markers for issue in result.issues
    ):
        return result
    if not requested_symbols:
        return result
    index = min(result.requests_made, len(requested_symbols) - 1)
    if any("reported a quota or entitlement limit" in issue for issue in result.issues):
        index = max(0, min(result.requests_made - 1, len(requested_symbols) - 1))
    return _with_identified_issues(
        result,
        identifier=f"Alpha Vantage {requested_symbols[index]}",
    )


def validate_collection_for_training(
    collection: LiveCollection,
    *,
    expected_cutoff: datetime,
) -> None:
    """Fail before analysis unless the provider pass matches the release target."""

    target = _aware_utc(expected_cutoff, label="expected cutoff")
    if collection.model_cutoff.astimezone(UTC) != target:
        raise CollectionGateError("collection cutoff does not match the expected cutoff")
    if collection.overall_health is not HealthStatus.OK:
        raise CollectionGateError(
            f"collection health is {collection.overall_health.value}, not ok"
        )
    if collection.issues:
        raise CollectionGateError("collection issues must be empty")
    by_id = {
        str(source.get("id")): source
        for source in collection.sources
        if isinstance(source, Mapping) and isinstance(source.get("id"), str)
    }
    if len(collection.sources) != 2 or set(by_id) != {"alpha_vantage", "alfred"}:
        raise CollectionGateError(
            "collection sources must be exactly alpha_vantage and alfred"
        )
    for source_id, source in by_id.items():
        if source.get("status") != HealthStatus.OK.value:
            raise CollectionGateError(f"{source_id} source health is not ok")
        if source.get("issues"):
            raise CollectionGateError(f"{source_id} source issues must be empty")

    alpha = by_id["alpha_vantage"]
    try:
        alpha_available = _aware_utc(
            datetime.fromisoformat(str(alpha.get("available_at"))),
            label="Alpha Vantage available_at",
        )
    except (TypeError, ValueError) as exc:
        raise CollectionGateError("Alpha Vantage available_at is invalid") from exc
    coverage_start, separator, coverage_end = str(
        alpha.get("coverage", "")
    ).partition("–")
    try:
        coverage_end_date = date.fromisoformat(coverage_end)
    except ValueError as exc:
        raise CollectionGateError("Alpha Vantage coverage is invalid") from exc
    if (
        not coverage_start
        or not separator
        or not alpha_market_week_is_current(
            available_at=alpha_available,
            coverage_end=coverage_end_date,
            cutoff=target,
        )
    ):
        raise CollectionGateError(
            "Alpha Vantage has not reached the expected cutoff market week"
        )


def collection_report_document(
    collection: LiveCollection,
    *,
    expected_cutoff: datetime,
    gate_error: str | None,
) -> dict[str, Any]:
    """Return a secret-free local receipt suitable for atomic persistence."""

    return {
        "schema_version": 1,
        "expected_cutoff": _aware_utc(
            expected_cutoff, label="expected cutoff"
        ).isoformat(),
        "model_cutoff": _aware_utc(
            collection.model_cutoff, label="collection model cutoff"
        ).isoformat(),
        "ready_for_training": gate_error is None,
        "overall_health": collection.overall_health.value,
        "issues": list(collection.issues),
        "sources": [dict(source) for source in collection.sources],
        "gate_error": gate_error,
    }


def last_completed_week_cutoff(
    now: datetime | None = None,
    *,
    cutoff_time: time = time(16, 0),
) -> datetime:
    current = (now or datetime.now(UTC)).astimezone(EASTERN)
    days_since_friday = (current.weekday() - 4) % 7
    candidate_date = current.date() - timedelta(days=days_since_friday)
    candidate = datetime.combine(candidate_date, cutoff_time, tzinfo=EASTERN)
    if candidate > current:
        candidate -= timedelta(days=7)
    return candidate.astimezone(UTC)


def weekly_cutoffs(start: date, end_cutoff: datetime) -> tuple[datetime, ...]:
    end_date = end_cutoff.astimezone(EASTERN).date()
    fridays = pd.date_range(start=start, end=end_date, freq="W-FRI")
    return tuple(
        datetime.combine(item.date(), time(16, 0), tzinfo=EASTERN).astimezone(UTC)
        for item in fridays
    )


def _realtime_year_chunks(
    start: date,
    end: date,
    *,
    years_per_chunk: int = 4,
) -> tuple[tuple[date, date], ...]:
    """Split long daily-series realtime ranges below ALFRED's 2,000-vintage cap."""

    if years_per_chunk < 1 or start > end:
        raise ValueError("invalid realtime chunk range")
    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(end, date(cursor.year + years_per_chunk - 1, 12, 31))
        chunks.append((cursor, chunk_end))
        cursor = date(chunk_end.year + 1, 1, 1)
    return tuple(chunks)


def _fetch_alfred_series(
    client: AlfredClient,
    *,
    series_id: str,
    frequency: str,
    history_start: date,
    observation_start: date | None = None,
    realtime_end: date,
    cutoff: datetime,
    snapshot_mode: SnapshotMode = SnapshotMode.FULL,
) -> CollectionResult:
    mode = SnapshotMode(snapshot_mode)
    if mode is SnapshotMode.DELTA:
        vintage_dates = tuple(
            history_start + timedelta(days=offset)
            for offset in range((realtime_end - history_start).days + 1)
        )
        return client.fetch_revision_events(
            [series_id],
            vintage_dates=vintage_dates,
            cutoff=cutoff,
            # A new vintage can revise an observation from any prior period.
            observation_start=observation_start or history_start,
            observation_end=realtime_end,
        )

    ranges = (
        _realtime_year_chunks(history_start, realtime_end)
        if frequency == "daily"
        else ((history_start, realtime_end),)
    )
    chunk_results: list[CollectionResult] = []
    for realtime_start, chunk_end in ranges:
        result = client.fetch_realtime_observations(
            [series_id],
            realtime_start=realtime_start,
            realtime_end=chunk_end,
            cutoff=cutoff,
            # Revisions during a later realtime chunk can apply to an older
            # observation, so every chunk retains the full observation period.
            observation_start=observation_start or history_start,
            observation_end=realtime_end,
        )
        chunk_results.append(result)
    return merge_collection_results(
        chunk_results,
        normalize_revisions=True,
    )


def _alfred_request_params(
    *,
    result: CollectionResult,
    series_id: str,
    frequency: str,
    realtime_start: date,
    realtime_end: date,
    observation_start: date,
    snapshot_mode: SnapshotMode,
) -> dict[str, Any]:
    """Return secret-free provenance matching the provider query shape."""

    mode = SnapshotMode(snapshot_mode)
    fallback_used = (
        result.diagnostics.get("vintage_discovery_fallback_used") is True
    )
    raw_fallback_series_count = result.diagnostics.get(
        "vintage_discovery_fallback_series_count",
        0,
    )
    fallback_series_count = (
        raw_fallback_series_count
        if type(raw_fallback_series_count) is int
        and raw_fallback_series_count >= 0
        else 0
    )
    params: dict[str, Any] = {
        "series_id": series_id,
        "output_type": 1 if mode is SnapshotMode.FULL else 3,
        "snapshot_mode": mode.value,
        "observation_start": observation_start.isoformat(),
        "observation_end": realtime_end.isoformat(),
        "provider_requests_made": result.requests_made,
        "provider_attempts": result.attempts,
    }
    if mode is SnapshotMode.FULL:
        params.update(
            {
                "realtime_start": realtime_start.isoformat(),
                "realtime_end": realtime_end.isoformat(),
                "realtime_chunk_years": 4 if frequency == "daily" else None,
            }
        )
    else:
        vintage_dates = (
            realtime_start + timedelta(days=offset)
            for offset in range((realtime_end - realtime_start).days + 1)
        )
        params["vintage_dates"] = ",".join(
            item.isoformat() for item in vintage_dates
        )
        params.update(
            {
                "vintage_discovery_policy": (
                    "narrow_then_observation_start_wide_on_5xx_v1"
                ),
                "vintage_discovery_fallback_realtime_start": (
                    observation_start.isoformat()
                ),
                "vintage_discovery_fallback_realtime_end": (
                    realtime_end.isoformat()
                ),
                "vintage_discovery_fallback_used": (
                    fallback_used
                ),
                "vintage_discovery_mode": (
                    "wide_fallback"
                    if fallback_used
                    else "narrow"
                ),
                "vintage_discovery_fallback_series_count": (
                    fallback_series_count
                ),
            }
        )
    return params


def _write_result_snapshot(
    store: SQLiteSnapshotStore,
    result: CollectionResult,
    *,
    source: str,
    dataset: str,
    cutoff: datetime,
    requested_at: datetime,
    license_class: str,
    request_params: Mapping[str, Any],
) -> None:
    retrieved_at = datetime.now(UTC)
    provenance = SnapshotProvenance(
        source=source,
        dataset=dataset,
        cutoff=cutoff,
        requested_at=requested_at,
        retrieved_at=retrieved_at,
        quality_status=result.health,
        license_class=license_class,
        request_params=request_params,
        issues=result.issues,
    )
    # Failed/partial attempts keep provenance for audit, but their rows never
    # become a competing history chain or consume hundreds of MB on retries.
    stored_records = result.records if result.health is HealthStatus.OK else ()
    store.write_snapshot(stored_records, provenance)


def _source_row(
    *,
    source_id: str,
    name: str,
    result: CollectionResult,
    records: tuple[Observation, ...],
    as_of: datetime,
    frequency: str,
    license_class: str,
) -> dict[str, Any]:
    eligible = tuple(item for item in records if item.available_at <= as_of)
    available = max((item.available_at for item in eligible), default=None)
    periods = [item.observed_period_end for item in eligible]
    coverage_start = min(periods) if periods else None
    coverage_end = max(periods) if periods else None
    return {
        "id": source_id,
        "name": name,
        "status": result.health.value,
        "available_at": available.isoformat() if available else None,
        "coverage": (
            f"{coverage_start.isoformat()}–{coverage_end.isoformat()}"
            if coverage_start and coverage_end
            else "수집 결과 없음"
        ),
        "frequency": frequency,
        "license_class": license_class,
        "records": len(eligible),
        "raw_records": len(records),
        "requests_made": result.requests_made,
        "issues": list(result.issues),
    }


def _validate_initial_alpha_baseline(
    result: CollectionResult,
    *,
    expected_series: set[str],
    history_start: date,
    cutoff: datetime,
    minimum_coverage: float = 0.9,
    max_latest_lag_days: int = 10,
    strict_response: bool = True,
    history_start_by_symbol: Mapping[str, date] | None = None,
) -> tuple[CollectionResult, dict[str, Any]]:
    """Fail closed when an Alpha full-response history is incomplete.

    The provider returns a full weekly history but does not publish a fixed row
    count contract. Coverage is measured against weekly cutoffs in the
    requested history window. When official symbol inception dates are
    configured, each series gets its own coverage window; fields for the same
    symbol must still expose exactly the same periods and every series must be
    current. Without that mapping, the legacy all-series common-period rule is
    preserved unchanged.
    """

    if not expected_series:
        raise ValueError("expected_series must not be empty")
    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be in (0, 1]")
    if max_latest_lag_days < 0:
        raise ValueError("max_latest_lag_days must be non-negative")

    expected_periods = len(weekly_cutoffs(history_start, cutoff))
    if expected_periods < 1:
        raise ValueError("initial Alpha baseline requires at least one weekly cutoff")
    minimum_periods = max(1, math.ceil(expected_periods * minimum_coverage))
    symbol_history_starts = {
        str(symbol).strip().upper(): max(history_start, start)
        for symbol, start in (history_start_by_symbol or {}).items()
    }
    symbol_specific_coverage = bool(symbol_history_starts)
    expected_periods_by_series: dict[str, int] = {}
    minimum_periods_by_series: dict[str, int] = {}
    for series_id in expected_series:
        symbol, separator, _field = series_id.rpartition(".")
        series_start = symbol_history_starts.get(symbol, history_start)
        series_expected = len(weekly_cutoffs(series_start, cutoff))
        if not separator or series_expected < 1:
            raise ValueError(
                f"invalid Alpha series coverage window for {series_id}"
            )
        expected_periods_by_series[series_id] = series_expected
        minimum_periods_by_series[series_id] = max(
            1,
            math.ceil(series_expected * minimum_coverage),
        )
    metrics: dict[str, Any] = {
        "initial_baseline_validation": "not_evaluated_provider_non_ok",
        "initial_baseline_expected_series": len(expected_series),
        "initial_baseline_expected_periods": expected_periods,
        "initial_baseline_minimum_periods": minimum_periods,
        "initial_baseline_minimum_coverage": minimum_coverage,
        "initial_baseline_max_latest_lag_days": max_latest_lag_days,
        "initial_baseline_symbol_specific_coverage": symbol_specific_coverage,
        "initial_baseline_minimum_series_periods": min(
            minimum_periods_by_series.values()
        ),
        "initial_baseline_maximum_series_periods": max(
            minimum_periods_by_series.values()
        ),
    }
    if result.health is not HealthStatus.OK:
        return result, metrics

    cutoff_date = cutoff.astimezone(EASTERN).date()
    periods_by_series: dict[str, set[date]] = {
        series_id: set() for series_id in expected_series
    }
    unexpected_series: set[str] = set()
    invalid_records = 0
    duplicate_records = 0
    for record in result.records:
        if record.series_id not in expected_series:
            unexpected_series.add(record.series_id)
            continue
        invalid_contract = (
            record.source != "alpha_vantage"
            or record.value is None
            or record.quality_status is not HealthStatus.OK
        )
        outside_window = (
            record.observed_period_end < history_start
            or record.observed_period_end > cutoff_date
            or record.available_at > cutoff
        )
        if invalid_contract or (strict_response and outside_window):
            invalid_records += 1
            continue
        if outside_window:
            # Existing append-only chains can legitimately include older rows,
            # future partial periods, and prospective revisions not yet
            # available at the baseline's own cutoff.
            continue
        periods = periods_by_series[record.series_id]
        if record.observed_period_end in periods:
            duplicate_records += 1
        periods.add(record.observed_period_end)

    observed_series = {
        series_id for series_id, periods in periods_by_series.items() if periods
    }
    missing_series = expected_series - observed_series
    insufficient_series = {
        series_id
        for series_id, periods in periods_by_series.items()
        if len(periods) < minimum_periods_by_series[series_id]
    }
    latest_required = cutoff_date - timedelta(days=max_latest_lag_days)
    stale_series = {
        series_id
        for series_id, periods in periods_by_series.items()
        if not periods or max(periods) < latest_required
    }
    common_periods = (
        set.intersection(*(periods_by_series[item] for item in expected_series))
        if not missing_series
        else set()
    )
    expected_fields_by_symbol: dict[str, set[str]] = {}
    for series_id in expected_series:
        symbol, separator, field = series_id.rpartition(".")
        if separator:
            expected_fields_by_symbol.setdefault(symbol, set()).add(field)
    field_mismatch_symbols: set[str] = set()
    for symbol, fields in expected_fields_by_symbol.items():
        field_periods = [periods_by_series[f"{symbol}.{field}"] for field in fields]
        if len(field_periods) > 1 and any(
            periods != field_periods[0] for periods in field_periods[1:]
        ):
            field_mismatch_symbols.add(symbol)
    latest_common_period = max(common_periods) if common_periods else None
    metrics.update(
        {
            "initial_baseline_observed_series": len(observed_series),
            "initial_baseline_common_periods": len(common_periods),
            "initial_baseline_latest_common_period": (
                latest_common_period.isoformat() if latest_common_period else None
            ),
            "initial_baseline_invalid_records": invalid_records,
            "initial_baseline_duplicate_records": duplicate_records,
            # Retain the v2 metric name for downstream provenance readers while
            # extending it from the old close/volume pair to every configured
            # OHLCV field on a symbol.
            "initial_baseline_pair_mismatch_symbols": len(field_mismatch_symbols),
            "initial_baseline_field_mismatch_symbols": len(field_mismatch_symbols),
            "initial_baseline_stale_series": len(stale_series),
        }
    )
    common_window_failed = bool(
        not symbol_specific_coverage
        and (
            len(common_periods) < minimum_periods
            or latest_common_period is None
            or latest_common_period < latest_required
        )
    )
    failed = bool(
        missing_series
        or unexpected_series
        or insufficient_series
        or field_mismatch_symbols
        or stale_series
        or common_window_failed
        or invalid_records
        or (strict_response and duplicate_records)
    )
    if not failed:
        metrics["initial_baseline_validation"] = "passed"
        return result, metrics

    metrics["initial_baseline_validation"] = "failed"
    issue = (
        "initial Alpha baseline failed completeness/coverage validation: "
        f"expected_series={len(expected_series)}, "
        f"observed_series={len(observed_series)}, "
        f"insufficient_series={len(insufficient_series)}, "
        f"unexpected_series={len(unexpected_series)}, "
        f"field_mismatch_symbols={len(field_mismatch_symbols)}, "
        f"stale_series={len(stale_series)}, "
        f"common_periods={len(common_periods)}, "
        f"minimum_periods={minimum_periods}, "
        f"latest_common_period={latest_common_period}, "
        f"latest_required={latest_required}, "
        f"invalid_records={invalid_records}, "
        f"duplicate_records={duplicate_records}"
    )
    return (
        CollectionResult(
            records=(),
            health=HealthStatus.DEGRADED,
            issues=tuple(dict.fromkeys((*result.issues, issue))),
            requests_made=result.requests_made,
            attempts=result.attempts,
        ),
        metrics,
    )


def _same_cutoff_alpha_expansion(
    *,
    records: tuple[Observation, ...],
    last_good: SnapshotProvenance | None,
    configured_symbols: tuple[str, ...],
    configured_fields: tuple[str, ...],
    history_start: date,
    cutoff: datetime,
    minimum_coverage: float,
    max_latest_lag_days: int,
    history_start_by_symbol: Mapping[str, date] | None = None,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Return a safe, additive-only Alpha symbol expansion plan.

    The optimization is intentionally narrow.  It is available only when the
    last-good snapshot was accepted at the current model cutoff and its
    provenance describes an exact, complete symbol/field subset of the new
    configuration.  Anything ambiguous falls back to the ordinary full
    response path, where the existing completeness checks remain fail-closed.
    """

    metrics: dict[str, Any] = {
        "config_expansion_validation": "not_applicable",
        "config_expansion_existing_symbols": 0,
        "config_expansion_added_symbols": 0,
    }
    if last_good is None or last_good.cutoff != cutoff or not records:
        return (), metrics

    raw_symbols = last_good.request_params.get("symbols")
    raw_fields = last_good.request_params.get("fields")
    if (
        not isinstance(raw_symbols, (list, tuple))
        or isinstance(raw_symbols, (str, bytes))
        or not isinstance(raw_fields, (list, tuple))
        or isinstance(raw_fields, (str, bytes))
    ):
        metrics["config_expansion_validation"] = "failed_provenance_contract"
        return (), metrics

    previous_symbols = tuple(
        str(item).strip().upper() for item in raw_symbols if str(item).strip()
    )
    previous_fields = tuple(
        str(item).strip() for item in raw_fields if str(item).strip()
    )
    configured_symbol_set = set(configured_symbols)
    previous_symbol_set = set(previous_symbols)
    if (
        not previous_symbols
        or len(previous_symbols) != len(previous_symbol_set)
        or len(previous_fields) != len(set(previous_fields))
        or set(previous_fields) != set(configured_fields)
        or not previous_symbol_set < configured_symbol_set
    ):
        metrics["config_expansion_validation"] = "failed_non_additive_config"
        return (), metrics

    observed_series = {item.series_id for item in records}
    expected_existing_series = {
        f"{symbol}.{field}"
        for symbol in previous_symbols
        for field in configured_fields
    }
    if observed_series != expected_existing_series:
        metrics["config_expansion_validation"] = "failed_series_coverage"
        return (), metrics

    checked, validation_metrics = _validate_initial_alpha_baseline(
        CollectionResult(records=records),
        expected_series=expected_existing_series,
        history_start=history_start,
        cutoff=cutoff,
        minimum_coverage=minimum_coverage,
        max_latest_lag_days=max_latest_lag_days,
        history_start_by_symbol=history_start_by_symbol,
        # Accepted append-only chains may contain prospective revisions and
        # rows outside their own model window.  The exact series contract above
        # remains strict; only response-window/duplicate handling is relaxed.
        strict_response=False,
    )
    metrics.update(
        {
            "config_expansion_validation": (
                "passed"
                if checked.health is HealthStatus.OK
                else "failed_existing_history"
            ),
            "config_expansion_existing_symbols": len(previous_symbols),
            "config_expansion_added_symbols": (
                len(configured_symbols) - len(previous_symbols)
            ),
            "config_expansion_existing_history_validation": validation_metrics[
                "initial_baseline_validation"
            ],
        }
    )
    if checked.health is not HealthStatus.OK:
        return (), metrics

    missing_symbols = tuple(
        symbol for symbol in configured_symbols if symbol not in previous_symbol_set
    )
    return missing_symbols, metrics


def collect_live_data(
    config: Mapping[str, Any],
    *,
    database_path: str | Path,
    history_start: date = date(2006, 1, 1),
    now: datetime | None = None,
    expected_cutoff: datetime | None = None,
    progress: Callable[[str], None] | None = None,
) -> LiveCollection:
    """Collect Alpha Vantage and ALFRED data without paid fallback."""

    emit = progress or (lambda _message: None)
    current = (now or datetime.now(UTC)).astimezone(UTC)
    latest_completed = last_completed_week_cutoff(current)
    if expected_cutoff is None:
        model_cutoff = latest_completed
    else:
        model_cutoff = _aware_utc(expected_cutoff, label="expected cutoff")
        local_cutoff = model_cutoff.astimezone(EASTERN)
        if (
            local_cutoff.weekday() != 4
            or local_cutoff.time().replace(tzinfo=None) != time(16, 0)
        ):
            raise ValueError("expected cutoff must be Friday 16:00 America/New_York")
        if model_cutoff > latest_completed:
            raise ValueError("expected cutoff is after the last completed market week")
    cutoffs = weekly_cutoffs(history_start, model_cutoff)
    if not cutoffs:
        raise RuntimeError("history_start does not yield a completed weekly cutoff")
    database = Path(database_path)
    database.parent.mkdir(parents=True, exist_ok=True)
    sources: list[dict[str, Any]] = []
    statuses: list[HealthStatus] = []
    issues: list[str] = []

    with SQLiteSnapshotStore(database) as store:
        # Only successful snapshots may feed a later run.  A partial, rejected,
        # schema-changed, or quota-exhausted response is still preserved for
        # audit, but never becomes model input.
        existing_records = store.read_last_good_observations()

        alpha_cfg = config["alpha_vantage"]
        alpha_symbols = tuple(
            str(item).strip().upper()
            for item in alpha_cfg["symbols"]
            if str(item).strip()
        )
        if not alpha_symbols or len(alpha_symbols) != len(set(alpha_symbols)):
            raise ValueError("alpha_vantage symbols must be non-empty and unique")
        alpha_all_existing = tuple(
            item for item in existing_records if item.source == "alpha_vantage"
        )
        alpha_fields = tuple(
            dict.fromkeys(
                str(item).strip()
                for item in alpha_cfg.get(
                    "fields",
                    ("adjusted_close", "volume"),
                )
                if str(item).strip()
            )
        )
        if not alpha_fields:
            raise ValueError("alpha_vantage fields must be non-empty")
        expected_alpha_series = {
            f"{symbol}.{field}"
            for symbol in alpha_symbols
            for field in alpha_fields
        }
        alpha_last_good = store.get_last_good_provenance(
            source="alpha_vantage", dataset="weekly_adjusted_etf"
        )
        alpha_existing_cutoff_date = (
            alpha_last_good.cutoff.astimezone(EASTERN).date()
            if alpha_last_good is not None
            else None
        )
        alpha_existing = tuple(
            item
            for item in alpha_all_existing
            if alpha_existing_cutoff_date is not None
            and item.observed_period_end <= alpha_existing_cutoff_date
        )
        alpha_minimum_coverage = float(
            alpha_cfg.get("initial_baseline_minimum_coverage", 0.9)
        )
        alpha_max_latest_lag_days = int(
            alpha_cfg.get("initial_baseline_max_latest_lag_days", 10)
        )
        raw_alpha_history_starts = alpha_cfg.get("history_start_by_symbol", {})
        if not isinstance(raw_alpha_history_starts, Mapping):
            raise ValueError("alpha_vantage history_start_by_symbol must be a mapping")
        alpha_history_start_by_symbol: dict[str, date] = {}
        for raw_symbol, raw_start in raw_alpha_history_starts.items():
            symbol = str(raw_symbol).strip().upper()
            try:
                parsed_start = date.fromisoformat(str(raw_start))
            except ValueError as exc:
                raise ValueError(
                    f"invalid Alpha history start for {symbol}"
                ) from exc
            if symbol not in alpha_symbols or not symbol:
                raise ValueError(
                    "alpha_vantage history_start_by_symbol contains an "
                    f"unconfigured symbol: {symbol}"
                )
            alpha_history_start_by_symbol[symbol] = parsed_start
        alpha_existing_usable = False
        alpha_existing_validation = "absent"
        if alpha_existing and alpha_last_good is not None:
            existing_check, existing_metrics = _validate_initial_alpha_baseline(
                CollectionResult(records=alpha_existing),
                expected_series=expected_alpha_series,
                history_start=history_start,
                # Judge an accepted chain against the cutoff at which it was
                # stored, not against a later run that may have skipped weeks.
                cutoff=alpha_last_good.cutoff,
                minimum_coverage=alpha_minimum_coverage,
                max_latest_lag_days=alpha_max_latest_lag_days,
                strict_response=False,
                history_start_by_symbol=alpha_history_start_by_symbol,
            )
            alpha_existing_usable = existing_check.health is HealthStatus.OK
            alpha_existing_validation = str(
                existing_metrics["initial_baseline_validation"]
            )
        alpha_reused = bool(
            alpha_last_good
            and alpha_last_good.cutoff == model_cutoff
            and alpha_existing_usable
        )
        alpha_expansion_symbols, alpha_expansion_metrics = (
            _same_cutoff_alpha_expansion(
                records=alpha_existing,
                last_good=alpha_last_good,
                configured_symbols=alpha_symbols,
                configured_fields=alpha_fields,
                history_start=history_start,
                cutoff=model_cutoff,
                minimum_coverage=alpha_minimum_coverage,
                max_latest_lag_days=alpha_max_latest_lag_days,
                history_start_by_symbol=alpha_history_start_by_symbol,
            )
        )
        alpha_additive_expansion = bool(alpha_expansion_symbols)
        if alpha_reused:
            emit("Alpha Vantage: 동일 cutoff의 정상 snapshot 재사용")
            alpha_result = CollectionResult(
                records=alpha_existing,
                health=HealthStatus.OK,
                requests_made=0,
                attempts=0,
            )
            alpha_effective_records = alpha_existing
        else:
            alpha_requested_at = datetime.now(UTC)
            alpha_requested_symbols = (
                alpha_expansion_symbols
                if alpha_additive_expansion
                else alpha_symbols
            )
            if alpha_additive_expansion:
                emit(
                    "Alpha Vantage: 동일 cutoff config 확장 "
                    f"{len(alpha_requested_symbols)}개 ETF 추가 수집 시작"
                )
            else:
                emit(
                    f"Alpha Vantage: {len(alpha_requested_symbols)}개 ETF "
                    "주별 데이터 수집 시작"
                )
            alpha_timeout = _configured_timeout(
                alpha_cfg,
                provider="Alpha Vantage",
                default=30.0,
            )
            configured_alpha_retry, configured_alpha_retry_reserve = (
                _configured_retry(
                    alpha_cfg,
                    provider="Alpha Vantage",
                    default_attempts=1,
                    require_reserve=True,
                )
            )
            alpha_client_config = AlphaVantageConfig.from_env(
                base_url=str(alpha_cfg["base_url"]),
                timeout_seconds=alpha_timeout,
                market_available_time_et=time(16, 0),
                request_spacing_seconds=0.8,
            )
            alpha_budget: DailyRequestBudget | None = None
            alpha_reservation = None
            raw_alpha_limit = alpha_cfg["daily_request_cap"]
            alpha_limit_valid = (
                type(raw_alpha_limit) is int and raw_alpha_limit == 25
            )
            requested_count = len(alpha_requested_symbols)
            effective_retry_reserve = (
                min(
                    configured_alpha_retry_reserve,
                    max(0, raw_alpha_limit - requested_count),
                )
                if alpha_limit_valid
                else 0
            )
            alpha_reservation_units = requested_count + effective_retry_reserve
            alpha_retry = RetryPolicy(
                max_attempts=min(
                    configured_alpha_retry.max_attempts,
                    effective_retry_reserve + 1,
                ),
                backoff_seconds=configured_alpha_retry.backoff_seconds,
                max_backoff_seconds=configured_alpha_retry.max_backoff_seconds,
            )
            if alpha_client_config.api_key and alpha_limit_valid:
                alpha_budget = DailyRequestBudget(
                    limit=raw_alpha_limit,
                    database_path=database,
                )
                alpha_reservation = alpha_budget.reserve(alpha_reservation_units)
            alpha_reserved_requests = (
                alpha_reservation_units if alpha_reservation is not None else 0
            )
            if not alpha_limit_valid:
                alpha_result = CollectionResult(
                    health=HealthStatus.DEGRADED,
                    issues=(
                        "Alpha Vantage standard-free daily_request_cap must be "
                        "the integer 25",
                    ),
                )
            elif not alpha_client_config.api_key:
                alpha_result = CollectionResult(
                    health=HealthStatus.DEGRADED,
                    issues=("ALPHA_VANTAGE_API_KEY is not configured",),
                )
            elif alpha_reservation is None:
                # Reserve every planned symbol in one SQLite transaction.  No
                # transport starts unless the full batch is already charged;
                # unused credits remain charged after any failure or crash.
                assert alpha_budget is not None
                retry_detail = (
                    "requested batch plus retry reserve exceeds the configured "
                    "standard-free cap"
                    if alpha_reservation_units > alpha_budget.limit
                    else (
                        "earliest full-batch retry="
                        f"{alpha_budget.next_available_at(alpha_reservation_units).isoformat()}"
                    )
                )
                alpha_result = CollectionResult(
                    health=HealthStatus.QUOTA_EXHAUSTED,
                    issues=(
                        "Alpha Vantage rolling 24-hour request budget cannot complete "
                        f"the requested batch; {retry_detail}",
                    ),
                )
            else:
                alpha_client = AlphaVantageClient(
                    alpha_client_config,
                    budget=alpha_reservation,
                    # Every retry credit is included in the same persisted,
                    # fail-closed batch reservation.  The bounded policy can
                    # therefore never exceed the literal 25-call rolling cap.
                    retry=alpha_retry,
                )
                alpha_result = alpha_client.fetch_weekly_adjusted(
                    alpha_requested_symbols,
                    cutoff=model_cutoff,
                    fields=alpha_fields,
                    observation_start=history_start,
                )
                alpha_result = _identify_alpha_transport_failure(
                    alpha_result,
                    requested_symbols=alpha_requested_symbols,
                )
            alpha_unused_reserved_requests = (
                alpha_reservation.remaining if alpha_reservation is not None else 0
            )
            # The endpoint is a full response on every run. Validate it against
            # the current cutoff even when a usable base exists: otherwise a
            # provider that silently stops advancing could be accepted as an
            # OK empty delta until ordinary feature staleness catches up.
            if alpha_additive_expansion:
                expected_added_series = {
                    f"{symbol}.{field}"
                    for symbol in alpha_requested_symbols
                    for field in alpha_fields
                }
                fetched_result, fetched_metrics = _validate_initial_alpha_baseline(
                    alpha_result,
                    expected_series=expected_added_series,
                    history_start=history_start,
                    cutoff=model_cutoff,
                    minimum_coverage=alpha_minimum_coverage,
                    max_latest_lag_days=alpha_max_latest_lag_days,
                    history_start_by_symbol=alpha_history_start_by_symbol,
                )
                if fetched_result.health is HealthStatus.OK:
                    combined_result = CollectionResult(
                        records=tuple(alpha_existing) + tuple(fetched_result.records),
                        health=HealthStatus.OK,
                        issues=fetched_result.issues,
                        requests_made=fetched_result.requests_made,
                        attempts=fetched_result.attempts,
                    )
                    alpha_result, alpha_baseline_metrics = (
                        _validate_initial_alpha_baseline(
                            combined_result,
                            expected_series=expected_alpha_series,
                            history_start=history_start,
                            cutoff=model_cutoff,
                            minimum_coverage=alpha_minimum_coverage,
                            max_latest_lag_days=alpha_max_latest_lag_days,
                            strict_response=False,
                            history_start_by_symbol=alpha_history_start_by_symbol,
                        )
                    )
                else:
                    alpha_result = fetched_result
                    alpha_baseline_metrics = {
                        "initial_baseline_validation": (
                            "not_evaluated_config_expansion_fetch_non_ok"
                        ),
                        "initial_baseline_expected_series": len(
                            expected_alpha_series
                        ),
                    }
                alpha_baseline_metrics.update(alpha_expansion_metrics)
                alpha_baseline_metrics.update(
                    {
                        f"config_expansion_fetched_{key}": value
                        for key, value in fetched_metrics.items()
                    }
                )
                alpha_baseline_metrics["existing_history_validation"] = (
                    alpha_expansion_metrics.get(
                        "config_expansion_existing_history_validation",
                        "failed",
                    )
                )
                alpha_existing_for_prepare = alpha_existing
            else:
                alpha_result, alpha_baseline_metrics = (
                    _validate_initial_alpha_baseline(
                        alpha_result,
                        expected_series=expected_alpha_series,
                        history_start=history_start,
                        cutoff=model_cutoff,
                        minimum_coverage=alpha_minimum_coverage,
                        max_latest_lag_days=alpha_max_latest_lag_days,
                        history_start_by_symbol=alpha_history_start_by_symbol,
                    )
                )
                alpha_baseline_metrics.update(alpha_expansion_metrics)
                alpha_baseline_metrics["existing_history_validation"] = (
                    alpha_existing_validation
                )
                alpha_existing_for_prepare = (
                    alpha_existing if alpha_existing_usable else ()
                )
            prepared_alpha = prepare_incremental_snapshot(
                alpha_existing_for_prepare,
                alpha_result,
            )
            # Diff-level safety findings (for example a transient missing
            # historical row) are source health, not merely storage details.
            # Surface that prepared status to the dashboard and overall run.
            alpha_result = prepared_alpha.snapshot_result
            _write_result_snapshot(
                store,
                alpha_result,
                source="alpha_vantage",
                dataset="weekly_adjusted_etf",
                cutoff=model_cutoff,
                requested_at=alpha_requested_at,
                license_class="alpha_vantage_private_research",
                request_params={
                    "symbols": list(alpha_symbols),
                    "requested_symbols": list(alpha_requested_symbols),
                    "fields": list(alpha_fields),
                    "history_start_by_symbol": {
                        symbol: start.isoformat()
                        for symbol, start in alpha_history_start_by_symbol.items()
                    },
                    "observation_start": history_start.isoformat(),
                    "snapshot_mode": prepared_alpha.snapshot_mode.value,
                    "added_records": prepared_alpha.added_count,
                    "changed_records": prepared_alpha.changed_count,
                    "unchanged_records": prepared_alpha.unchanged_count,
                    "removed_records": prepared_alpha.removed_count,
                    "budget_policy": "rolling_24h_utc_atomic_batch_v3",
                    "budget_reserved_requests": alpha_reserved_requests,
                    "budget_unused_reserved_requests": (
                        alpha_unused_reserved_requests
                    ),
                    "budget_retry_reserve_requests": effective_retry_reserve,
                    "request_timeout_seconds": alpha_timeout,
                    "retry_max_attempts": alpha_retry.max_attempts,
                    **alpha_baseline_metrics,
                },
            )
            alpha_effective_records = prepared_alpha.effective_records
        # Model/source contracts never expose a partial provider period.  This
        # period filter intentionally does not use available_at: prospective
        # revisions to older periods must survive for the next eligible week.
        alpha_model_period_end = model_cutoff.astimezone(EASTERN).date()
        alpha_effective_records = tuple(
            item
            for item in alpha_effective_records
            if item.observed_period_end <= alpha_model_period_end
        )
        statuses.append(alpha_result.health)
        issues.extend(alpha_result.issues)
        sources.append(
            _source_row(
                source_id="alpha_vantage",
                name="Alpha Vantage ETF weekly adjusted",
                result=alpha_result,
                records=tuple(alpha_effective_records),
                as_of=model_cutoff,
                frequency="weekly",
                license_class="private_noncommercial",
            )
        )
        emit(
            f"Alpha Vantage: {alpha_result.health.value}, "
            f"{len(alpha_effective_records):,} effective records"
        )

        alfred_cfg = config["alfred"]
        alfred_timeout = _configured_timeout(
            alfred_cfg,
            provider="ALFRED",
            default=30.0,
        )
        alfred_retry, _alfred_retry_reserve = _configured_retry(
            alfred_cfg,
            provider="ALFRED",
            default_attempts=3,
        )
        alfred_client = AlfredClient(
            AlfredConfig.from_env(
                base_url=str(alfred_cfg["base_url"]),
                timeout_seconds=alfred_timeout,
                request_spacing_seconds=0.2,
            ),
            retry=alfred_retry,
        )
        alfred_records: list[Observation] = []
        alfred_statuses: list[HealthStatus] = []
        alfred_issues: list[str] = []
        alfred_requests = 0
        alfred_attempts = 0
        series_config = tuple(alfred_cfg["series"])
        for index, series in enumerate(series_config, start=1):
            series_id = str(series["id"])
            frequency = str(series.get("frequency", ""))
            realtime_history_start = date.fromisoformat(
                str(series.get("realtime_start", history_start.isoformat()))
            )
            emit(f"ALFRED {index}/{len(series_config)}: {series_id}")
            existing_series = tuple(
                item
                for item in existing_records
                if item.source == "alfred" and item.series_id == series_id
            )
            last_good = store.get_last_good_provenance(
                source="alfred",
                dataset=series_id,
            )
            reused = bool(
                last_good
                and last_good.cutoff == model_cutoff
                and existing_series
            )
            if reused:
                emit(f"ALFRED {series_id}: 동일 cutoff의 정상 snapshot 재사용")
                result = CollectionResult(
                    records=existing_series,
                    health=HealthStatus.OK,
                    requests_made=0,
                    attempts=0,
                )
            else:
                requested_at = datetime.now(UTC)
                # A provenance row without an assembled effective history is
                # not a usable delta base (for example after legacy damage or
                # an empty/incomplete initial snapshot).  Recover with a new
                # full collection instead of extending a base-less chain.
                planner_last_good = last_good if existing_series else None
                window = plan_incremental_realtime_window(
                    planner_last_good,
                    history_start=realtime_history_start,
                    observation_start=history_start,
                    realtime_end=model_cutoff.astimezone(EASTERN).date(),
                )
                result = _fetch_alfred_series(
                    alfred_client,
                    series_id=series_id,
                    frequency=frequency,
                    history_start=window.realtime_start,
                    observation_start=window.observation_start,
                    realtime_end=window.realtime_end,
                    cutoff=current,
                    snapshot_mode=window.snapshot_mode,
                )
                result = _with_identified_issues(
                    result,
                    identifier=f"ALFRED {series_id}",
                )
                _write_result_snapshot(
                    store,
                    result,
                    source="alfred",
                    dataset=series_id,
                    cutoff=model_cutoff,
                    requested_at=requested_at,
                    license_class="user_confirmed_ml_storage_derived",
                    request_params=_alfred_request_params(
                        result=result,
                        series_id=series_id,
                        frequency=frequency,
                        realtime_start=window.realtime_start,
                        realtime_end=window.realtime_end,
                        observation_start=window.observation_start,
                        snapshot_mode=window.snapshot_mode,
                    ),
                )

            effective_records = (
                store.read_last_good_observations(
                    source="alfred",
                    dataset=series_id,
                )
                if result.health is HealthStatus.OK
                else existing_series
            )
            alfred_records.extend(effective_records)
            alfred_statuses.append(result.health)
            alfred_issues.extend(result.issues)
            alfred_requests += result.requests_made
            alfred_attempts += result.attempts
            if result.health is not HealthStatus.OK:
                emit(f"ALFRED {series_id}: {result.health.value}")

        alfred_health = combine_health(alfred_statuses)
        statuses.append(alfred_health)
        issues.extend(alfred_issues)
        grouped_result = CollectionResult(
            records=tuple(alfred_records),
            health=alfred_health,
            issues=tuple(dict.fromkeys(alfred_issues)),
            requests_made=alfred_requests,
            attempts=alfred_attempts,
        )
        sources.append(
            _source_row(
                source_id="alfred",
                name="ALFRED real-time revision events",
                result=grouped_result,
                records=tuple(alfred_records),
                as_of=model_cutoff,
                frequency="daily/weekly/monthly/quarterly → weekly as-of",
                license_class="user_confirmed_ml_storage_derived",
            )
        )

        # Return one assembled, deduplicated history per provider.  The store
        # may contain a full base plus overlapping deltas, but model input must
        # not carry those storage-level duplicates forward.
        merged_records = tuple(alpha_effective_records) + tuple(alfred_records)

    return LiveCollection(
        records=merged_records,
        cutoffs=cutoffs,
        sources=tuple(sources),
        overall_health=combine_health(statuses),
        issues=tuple(dict.fromkeys(issues)),
        model_cutoff=model_cutoff,
        database_path=database,
    )
