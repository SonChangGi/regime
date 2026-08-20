"""Point-in-time ALFRED/FRED observation adapter.

Live collection is deliberately fail-closed.  It requires both ``FRED_API_KEY``
and an explicit ``ALFRED_ML_RIGHTS_ACK`` acknowledgement.  The acknowledgement
is an application control, not a representation that the upstream licence
actually grants a particular use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import os
from typing import Any, Callable, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from .contracts import (
    CollectionResult,
    HealthStatus,
    Observation,
    combine_health,
    ensure_utc,
    normalize_revision_sequences,
)
from .transport import (
    JsonTransport,
    ProviderRequestError,
    RetryPolicy,
    UrllibJsonTransport,
    request_json_with_retry,
)


FRED_API_KEY_ENV = "FRED_API_KEY"
ALFRED_RIGHTS_ACK_ENV = "ALFRED_ML_RIGHTS_ACK"
_ACK_VALUES = {"1", "true", "yes", "y", "ack", "acknowledged"}
_EASTERN = ZoneInfo("America/New_York")
_DEFAULT_VINTAGE_FALLBACK_LOOKBACK_DAYS = 5 * 366


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _date(value: str, *, field: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field}") from exc


def _raw_sha256(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _chunks(values: Sequence[date], size: int) -> Iterable[tuple[date, ...]]:
    for start in range(0, len(values), size):
        yield tuple(values[start : start + size])


def _sequence_revisions(records: Iterable[Observation]) -> tuple[Observation, ...]:
    """Deduplicate provider repeats, then apply the shared global sequence."""

    unique: dict[
        tuple[str, str, date, datetime, date, str],
        Observation,
    ] = {}
    for record in records:
        identity = (
            record.source,
            record.series_id,
            record.observed_period_end,
            record.available_at,
            record.vintage_date,
            record.raw_sha256,
        )
        unique[identity] = record

    return normalize_revision_sequences(unique.values())


@dataclass(frozen=True, slots=True)
class AlfredConfig:
    api_key: str | None = field(repr=False)
    rights_acknowledged: bool
    base_url: str = "https://api.stlouisfed.org/fred/series/observations"
    vintage_dates_url: str | None = None
    page_size: int = 100_000
    vintage_page_size: int = 10_000
    vintage_batch_size: int = 20
    timeout_seconds: float = 30.0
    availability_hour_et: int = 18
    request_spacing_seconds: float = 0.0
    license_class: str = "fred_alfred_permission_required"

    def __post_init__(self) -> None:
        if not 1 <= self.page_size <= 100_000:
            raise ValueError("page_size must be in [1, 100000]")
        if not 1 <= self.vintage_page_size <= 10_000:
            raise ValueError("vintage_page_size must be in [1, 10000]")
        if not 1 <= self.vintage_batch_size <= 2_000:
            raise ValueError("vintage_batch_size must be in [1, 2000]")
        if not 0 <= self.availability_hour_et <= 23:
            raise ValueError("availability_hour_et must be in [0, 23]")
        if self.request_spacing_seconds < 0:
            raise ValueError("request_spacing_seconds must be non-negative")
        normalized_url = self.base_url.rstrip("/")
        if normalized_url.endswith("/fred"):
            normalized_url += "/series/observations"
        object.__setattr__(self, "base_url", normalized_url)
        vintage_dates_url = self.vintage_dates_url
        if vintage_dates_url is None:
            observations_suffix = "/series/observations"
            if normalized_url.endswith(observations_suffix):
                vintage_dates_url = (
                    normalized_url[: -len(observations_suffix)]
                    + "/series/vintagedates"
                )
            else:
                vintage_dates_url = normalized_url + "/series/vintagedates"
        object.__setattr__(self, "vintage_dates_url", vintage_dates_url.rstrip("/"))

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None, **kwargs: Any) -> "AlfredConfig":
        env = os.environ if environ is None else environ
        acknowledgement = env.get(ALFRED_RIGHTS_ACK_ENV, "").strip().lower()
        return cls(
            api_key=env.get(FRED_API_KEY_ENV) or None,
            rights_acknowledged=acknowledgement in _ACK_VALUES,
            **kwargs,
        )


@dataclass(frozen=True, slots=True)
class _VintageDateDiscovery:
    dates: tuple[date, ...] = ()
    health: HealthStatus = HealthStatus.OK
    issues: tuple[str, ...] = ()
    requests_made: int = 0
    attempts: int = 0
    failure_status_code: int | None = None
    fallback_used: bool = False


class AlfredClient:
    """Fetch revision-aware observations using explicit vintage boundaries."""

    def __init__(
        self,
        config: AlfredConfig,
        *,
        transport: JsonTransport | None = None,
        retry: RetryPolicy | None = None,
        sleeper: Callable[[float], None] | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.config = config
        self.transport = transport or UrllibJsonTransport()
        self.retry = retry or RetryPolicy()
        self.sleeper = sleeper or __import__("time").sleep
        self.clock = clock

    def _available_at(self, vintage: date) -> datetime:
        # ALFRED exposes a date, not a release timestamp.  A configurable
        # after-session time prevents same-day look-ahead while still allowing
        # a later weekly cutoff to consume Friday releases.  At the configured
        # Friday 16:00 ET cutoff, a date-only Friday release rolls to next week.
        local = datetime.combine(
            vintage,
            time(self.config.availability_hour_et),
            tzinfo=_EASTERN,
        )
        return local.astimezone(timezone.utc)

    def _request_vintage_date_range(
        self,
        series_id: str,
        *,
        realtime_start: date,
        realtime_end: date,
        request_index: int,
        failure_label: str,
    ) -> tuple[_VintageDateDiscovery, int]:
        dates: list[date] = []
        issues: list[str] = []
        statuses: list[HealthStatus] = []
        requests_made = 0
        attempts = 0
        offset = 0

        while True:
            if request_index and self.config.request_spacing_seconds:
                self.sleeper(self.config.request_spacing_seconds)
            params: dict[str, Any] = {
                "series_id": series_id,
                "api_key": self.config.api_key,
                "file_type": "json",
                "realtime_start": realtime_start.isoformat(),
                "realtime_end": realtime_end.isoformat(),
                "limit": self.config.vintage_page_size,
                "offset": offset,
                "sort_order": "asc",
            }
            request_index += 1
            try:
                payload, used_attempts = request_json_with_retry(
                    self.transport,
                    self.config.vintage_dates_url or "",
                    params,
                    timeout=self.config.timeout_seconds,
                    retry=self.retry,
                    sleeper=self.sleeper,
                    secrets=(self.config.api_key or "",),
                )
            except ProviderRequestError as exc:
                attempts += exc.attempts
                status = (
                    HealthStatus.QUOTA_EXHAUSTED
                    if exc.status_code == 429
                    else HealthStatus.DEGRADED
                )
                return (
                    _VintageDateDiscovery(
                        dates=tuple(sorted(set(dates))),
                        health=status,
                        issues=(f"ALFRED {failure_label} request failed: {exc}",),
                        requests_made=requests_made,
                        attempts=attempts,
                        failure_status_code=exc.status_code,
                    ),
                    request_index,
                )

            requests_made += 1
            attempts += used_attempts
            raw_dates = payload.get("vintage_dates")
            try:
                total_count = int(payload["count"])
            except (KeyError, TypeError, ValueError):
                total_count = -1
            if not isinstance(raw_dates, list) or total_count < 0:
                statuses.append(HealthStatus.SCHEMA_CHANGED)
                issues.append(
                    f"ALFRED vintage-date response schema changed for {series_id}"
                )
                break

            page_schema_changed = False
            for raw_date in raw_dates:
                try:
                    parsed = _date(raw_date, field="vintage_date")
                except ValueError:
                    page_schema_changed = True
                    continue
                if not realtime_start <= parsed <= realtime_end:
                    page_schema_changed = True
                    continue
                dates.append(parsed)
            if page_schema_changed:
                statuses.append(HealthStatus.SCHEMA_CHANGED)
                issues.append(
                    f"ALFRED vintage-date rows were rejected for {series_id}"
                )

            next_offset = offset + len(raw_dates)
            if next_offset >= total_count:
                break
            if not raw_dates:
                statuses.append(HealthStatus.REVISION_GAP)
                issues.append(
                    "ALFRED vintage-date pagination ended before count "
                    f"for {series_id} at offset {offset}"
                )
                break
            offset = next_offset

        return (
            _VintageDateDiscovery(
                dates=tuple(sorted(set(dates))),
                health=combine_health(statuses),
                issues=tuple(dict.fromkeys(issues)),
                requests_made=requests_made,
                attempts=attempts,
            ),
            request_index,
        )

    def _discover_vintage_dates(
        self,
        series_id: str,
        *,
        realtime_start: date,
        realtime_end: date,
        fallback_realtime_start: date | None,
        request_index: int,
    ) -> tuple[_VintageDateDiscovery, int]:
        """Resolve a calendar window to provider-recognized vintage dates.

        A live UNRATE check returned HTTP 400 when an ``output_type=3`` request
        included dates that were not actual vintages for that series.  Rather
        than generalizing undocumented rejection behavior, use the dedicated
        ``series/vintagedates`` endpoint as the authoritative input to type 3.

        FRED currently returns HTTP 500 for some valid narrow ranges containing
        no vintage at all.  Only after the configured retries for such a 5xx
        are exhausted, repeat discovery over the caller's bounded observation
        history.  A schema-valid wide response can prove the target interval is
        empty; every other fallback failure remains fail-closed.
        """

        narrow, request_index = self._request_vintage_date_range(
            series_id,
            realtime_start=realtime_start,
            realtime_end=realtime_end,
            request_index=request_index,
            failure_label="vintage-date",
        )
        status_code = narrow.failure_status_code
        can_fallback = bool(
            status_code is not None
            and 500 <= status_code <= 599
            and fallback_realtime_start is not None
            and fallback_realtime_start < realtime_start
        )
        if not can_fallback:
            return narrow, request_index

        assert fallback_realtime_start is not None
        wide, request_index = self._request_vintage_date_range(
            series_id,
            realtime_start=fallback_realtime_start,
            realtime_end=realtime_end,
            request_index=request_index,
            failure_label="vintage-date fallback",
        )
        requests_made = narrow.requests_made + wide.requests_made
        attempts = narrow.attempts + wide.attempts
        if wide.health is not HealthStatus.OK:
            return (
                _VintageDateDiscovery(
                    health=combine_health((narrow.health, wide.health)),
                    issues=tuple(dict.fromkeys((*narrow.issues, *wide.issues))),
                    requests_made=requests_made,
                    attempts=attempts,
                    failure_status_code=wide.failure_status_code,
                    fallback_used=True,
                ),
                request_index,
            )

        return (
            _VintageDateDiscovery(
                dates=tuple(
                    item
                    for item in wide.dates
                    if realtime_start <= item <= realtime_end
                ),
                requests_made=requests_made,
                attempts=attempts,
                fallback_used=True,
            ),
            request_index,
        )

    def fetch_realtime_observations(
        self,
        series_ids: Sequence[str],
        *,
        realtime_start: date,
        realtime_end: date,
        cutoff: datetime,
        observation_start: date | None = None,
        observation_end: date | None = None,
    ) -> CollectionResult:
        """Collect raw-frequency revision events with FRED ``output_type=1``.

        This is the preferred history collector.  It requests one realtime
        range per series and paginates that response, rather than expanding
        every weekly cutoff into a wide vintage column.  Raw daily, weekly,
        monthly, and quarterly observations are retained for the local as-of
        join; no provider-side frequency aggregation is requested.
        """

        cutoff = ensure_utc(cutoff, field_name="cutoff")
        now = ensure_utc(self.clock(), field_name="clock")
        clean_series = tuple(
            dict.fromkeys(item.strip() for item in series_ids if item.strip())
        )
        if not clean_series:
            raise ValueError("series_ids must not be empty")
        if realtime_start > realtime_end:
            raise ValueError("realtime_start must not be after realtime_end")
        cutoff_et_date = cutoff.astimezone(_EASTERN).date()
        if realtime_end > cutoff_et_date:
            raise ValueError("realtime_end must not be after the cutoff date in US/Eastern")
        if observation_start and observation_end and observation_start > observation_end:
            raise ValueError("observation_start must not be after observation_end")

        if not self.config.rights_acknowledged:
            return CollectionResult(
                health=HealthStatus.LICENSE_BLOCKED,
                issues=(
                    f"live ALFRED collection requires explicit {ALFRED_RIGHTS_ACK_ENV} acknowledgement",
                ),
            )
        if not self.config.api_key:
            return CollectionResult(
                health=HealthStatus.DEGRADED,
                issues=(f"{FRED_API_KEY_ENV} is not configured",),
            )

        records: list[Observation] = []
        issues: list[str] = []
        statuses: list[HealthStatus] = []
        requests_made = 0
        attempts = 0
        schema_changed = False
        page_request_index = 0

        for series_id in clean_series:
            offset = 0
            while True:
                if page_request_index and self.config.request_spacing_seconds:
                    self.sleeper(self.config.request_spacing_seconds)
                params: dict[str, Any] = {
                    "series_id": series_id,
                    "api_key": self.config.api_key,
                    "file_type": "json",
                    "output_type": 1,
                    "realtime_start": realtime_start.isoformat(),
                    "realtime_end": realtime_end.isoformat(),
                    "limit": self.config.page_size,
                    "offset": offset,
                    "sort_order": "asc",
                }
                if observation_start is not None:
                    params["observation_start"] = observation_start.isoformat()
                if observation_end is not None:
                    params["observation_end"] = observation_end.isoformat()

                page_request_index += 1
                try:
                    payload, used_attempts = request_json_with_retry(
                        self.transport,
                        self.config.base_url,
                        params,
                        timeout=self.config.timeout_seconds,
                        retry=self.retry,
                        sleeper=self.sleeper,
                        secrets=(self.config.api_key,),
                    )
                except ProviderRequestError as exc:
                    attempts += exc.attempts
                    status = (
                        HealthStatus.QUOTA_EXHAUSTED
                        if exc.status_code == 429
                        else HealthStatus.DEGRADED
                    )
                    return CollectionResult(
                        records=_sequence_revisions(records),
                        health=status,
                        issues=tuple(issues + [f"ALFRED request failed: {exc}"]),
                        requests_made=requests_made,
                        attempts=attempts,
                    )

                requests_made += 1
                attempts += used_attempts
                rows = payload.get("observations")
                try:
                    total_count = int(payload["count"])
                except (KeyError, TypeError, ValueError):
                    total_count = -1
                if not isinstance(rows, list) or total_count < 0:
                    schema_changed = True
                    issues.append(f"ALFRED response schema changed for {series_id}")
                    break

                for row in rows:
                    if not isinstance(row, Mapping):
                        schema_changed = True
                        continue
                    try:
                        period_end = _date(row["date"], field="date")
                        vintage = _date(row["realtime_start"], field="realtime_start")
                        realtime_row_end = _date(
                            row["realtime_end"], field="realtime_end"
                        )
                        raw_value = row.get("value")
                        value = (
                            None
                            if raw_value in (None, ".", "")
                            else float(raw_value)
                        )
                    except (KeyError, TypeError, ValueError):
                        schema_changed = True
                        continue
                    if not realtime_start <= vintage <= realtime_end:
                        schema_changed = True
                        continue
                    available_at = self._available_at(vintage)
                    if available_at > cutoff:
                        continue
                    records.append(
                        Observation(
                            source="alfred",
                            series_id=series_id,
                            observed_period_end=period_end,
                            value=value,
                            released_at=available_at,
                            available_at=available_at,
                            vintage_date=vintage,
                            retrieved_at=now,
                            units="",
                            adjustment="provider_series_definition",
                            license_class=self.config.license_class,
                            quality_status=HealthStatus.OK,
                            raw_sha256=_raw_sha256(row),
                            metadata={
                                "realtime_end": realtime_row_end.isoformat(),
                                "availability_precision": "date_proxy_18_et",
                                "provider_output_type": 1,
                                "provider_frequency": "raw",
                            },
                        )
                    )

                next_offset = offset + len(rows)
                if next_offset >= total_count:
                    break
                if not rows:
                    statuses.append(HealthStatus.REVISION_GAP)
                    issues.append(
                        f"ALFRED pagination ended before count for {series_id} at offset {offset}"
                    )
                    break
                offset = next_offset

        if schema_changed:
            statuses.append(HealthStatus.SCHEMA_CHANGED)
            issues.append("one or more ALFRED rows were rejected by schema validation")
        return CollectionResult(
            records=_sequence_revisions(records),
            health=combine_health(statuses),
            issues=tuple(dict.fromkeys(issues)),
            requests_made=requests_made,
            attempts=attempts,
        )

    collect_realtime_observations = fetch_realtime_observations

    def fetch_revision_events(
        self,
        series_ids: Sequence[str],
        *,
        vintage_dates: Sequence[date],
        cutoff: datetime,
        observation_start: date | None = None,
        observation_end: date | None = None,
    ) -> CollectionResult:
        """Collect only new/revised vintage cells (FRED ``output_type=3``).

        The official JSON representation is a wide cross-tabulation: every
        row has an observation ``date`` and changed cells are named
        ``SERIES_YYYYMMDD``.  A missing cell means there was no change for that
        observation on that requested vintage date.  A present ``\".\"`` cell
        is retained as a real revision-to-missing event.

        ``vintage_dates`` is explicit by design.  Incremental callers should
        pass each calendar date in their inclusive overlap window.  Before the
        type-3 call, the client resolves that window through FRED's
        ``series/vintagedates`` endpoint.  This avoids the HTTP 400 observed in
        a live UNRATE check with non-discovered dates without assuming that the
        behavior is universal.  Only the recognized dates are then batched
        below FRED's 2,000-vintage JSON limit.  No provider-side frequency
        aggregation is requested.
        """

        cutoff = ensure_utc(cutoff, field_name="cutoff")
        now = ensure_utc(self.clock(), field_name="clock")
        clean_series = tuple(
            dict.fromkeys(item.strip() for item in series_ids if item.strip())
        )
        clean_vintages = tuple(sorted(set(vintage_dates)))
        if not clean_series:
            raise ValueError("series_ids must not be empty")
        if not clean_vintages:
            raise ValueError("vintage_dates must not be empty")
        cutoff_et_date = cutoff.astimezone(_EASTERN).date()
        if clean_vintages[-1] > cutoff_et_date:
            raise ValueError(
                "vintage_dates must not be after the cutoff date in US/Eastern"
            )
        if observation_start and observation_end and observation_start > observation_end:
            raise ValueError("observation_start must not be after observation_end")

        if not self.config.rights_acknowledged:
            return CollectionResult(
                health=HealthStatus.LICENSE_BLOCKED,
                issues=(
                    f"live ALFRED collection requires explicit {ALFRED_RIGHTS_ACK_ENV} acknowledgement",
                ),
            )
        if not self.config.api_key:
            return CollectionResult(
                health=HealthStatus.DEGRADED,
                issues=(f"{FRED_API_KEY_ENV} is not configured",),
            )

        records: list[Observation] = []
        issues: list[str] = []
        statuses: list[HealthStatus] = []
        requests_made = 0
        attempts = 0
        schema_changed = False
        page_request_index = 0
        fallback_series_count = 0

        def request_diagnostics() -> dict[str, Any]:
            fallback_used = fallback_series_count > 0
            return {
                "vintage_discovery_fallback_used": fallback_used,
                "vintage_discovery_mode": (
                    "wide_fallback" if fallback_used else "narrow"
                ),
                "vintage_discovery_fallback_series_count": (
                    fallback_series_count
                ),
            }

        candidate_vintages = frozenset(clean_vintages)
        # Production callers supply their explicit observation-history floor.
        # Direct adapter callers still get a finite recovery range rather than
        # an unbounded all-history request; failure of that range remains fatal.
        fallback_realtime_start = observation_start or date.fromordinal(
            max(
                date.min.toordinal(),
                clean_vintages[0].toordinal()
                - _DEFAULT_VINTAGE_FALLBACK_LOOKBACK_DAYS,
            )
        )
        for series_id in clean_series:
            discovery, page_request_index = self._discover_vintage_dates(
                series_id,
                realtime_start=clean_vintages[0],
                realtime_end=clean_vintages[-1],
                fallback_realtime_start=fallback_realtime_start,
                request_index=page_request_index,
            )
            requests_made += discovery.requests_made
            attempts += discovery.attempts
            if discovery.fallback_used:
                fallback_series_count += 1
            issues.extend(discovery.issues)
            if discovery.health is not HealthStatus.OK:
                return CollectionResult(
                    records=_sequence_revisions(records),
                    health=discovery.health,
                    issues=tuple(dict.fromkeys(issues)),
                    requests_made=requests_made,
                    attempts=attempts,
                    diagnostics=request_diagnostics(),
                )

            recognized_vintages = tuple(
                item for item in discovery.dates if item in candidate_vintages
            )
            for vintage_batch in _chunks(
                recognized_vintages,
                self.config.vintage_batch_size,
            ):
                offset = 0
                while True:
                    if page_request_index and self.config.request_spacing_seconds:
                        self.sleeper(self.config.request_spacing_seconds)
                    params: dict[str, Any] = {
                        "series_id": series_id,
                        "api_key": self.config.api_key,
                        "file_type": "json",
                        "output_type": 3,
                        "vintage_dates": ",".join(
                            item.isoformat() for item in vintage_batch
                        ),
                        "limit": self.config.page_size,
                        "offset": offset,
                        "sort_order": "asc",
                    }
                    if observation_start is not None:
                        params["observation_start"] = observation_start.isoformat()
                    if observation_end is not None:
                        params["observation_end"] = observation_end.isoformat()

                    page_request_index += 1
                    try:
                        payload, used_attempts = request_json_with_retry(
                            self.transport,
                            self.config.base_url,
                            params,
                            timeout=self.config.timeout_seconds,
                            retry=self.retry,
                            sleeper=self.sleeper,
                            secrets=(self.config.api_key,),
                        )
                    except ProviderRequestError as exc:
                        attempts += exc.attempts
                        status = (
                            HealthStatus.QUOTA_EXHAUSTED
                            if exc.status_code == 429
                            else HealthStatus.DEGRADED
                        )
                        return CollectionResult(
                            records=_sequence_revisions(records),
                            health=status,
                            issues=tuple(
                                issues + [f"ALFRED request failed: {exc}"]
                            ),
                            requests_made=requests_made,
                            attempts=attempts,
                            diagnostics=request_diagnostics(),
                        )

                    requests_made += 1
                    attempts += used_attempts
                    rows = payload.get("observations")
                    try:
                        total_count = int(payload["count"])
                    except (KeyError, TypeError, ValueError):
                        total_count = -1
                    response_output_type = payload.get("output_type")
                    if (
                        not isinstance(rows, list)
                        or total_count < 0
                        or (
                            response_output_type is not None
                            and response_output_type not in (3, "3")
                        )
                    ):
                        schema_changed = True
                        issues.append(
                            f"ALFRED output_type=3 response schema changed for {series_id}"
                        )
                        break

                    recognized_cells = 0
                    for row in rows:
                        if not isinstance(row, Mapping):
                            schema_changed = True
                            continue
                        try:
                            period_end = _date(row["date"], field="date")
                        except (KeyError, TypeError, ValueError):
                            schema_changed = True
                            continue

                        for vintage in vintage_batch:
                            provider_field = (
                                f"{series_id}_{vintage.strftime('%Y%m%d')}"
                            )
                            if provider_field not in row:
                                # output_type=3 omits unchanged cross-tab cells.
                                continue
                            recognized_cells += 1
                            raw_value = row[provider_field]
                            available_at = self._available_at(vintage)
                            if available_at > cutoff:
                                continue
                            if raw_value in (None, ""):
                                schema_changed = True
                                continue
                            try:
                                value = None if raw_value == "." else float(raw_value)
                            except (TypeError, ValueError):
                                schema_changed = True
                                continue

                            raw_fragment = {
                                "date": period_end.isoformat(),
                                "vintage_date": vintage.isoformat(),
                                "provider_field": provider_field,
                                "value": raw_value,
                            }
                            records.append(
                                Observation(
                                    source="alfred",
                                    series_id=series_id,
                                    observed_period_end=period_end,
                                    value=value,
                                    released_at=available_at,
                                    available_at=available_at,
                                    vintage_date=vintage,
                                    retrieved_at=now,
                                    units="",
                                    adjustment="provider_series_definition",
                                    license_class=self.config.license_class,
                                    quality_status=HealthStatus.OK,
                                    raw_sha256=_raw_sha256(raw_fragment),
                                    metadata={
                                        "provider_field": provider_field,
                                        "availability_precision": "date_proxy_18_et",
                                        "provider_output_type": 3,
                                        "provider_frequency": "raw",
                                    },
                                )
                            )

                    # A positive row count with no requested vintage field is
                    # not a valid type-3 cross-tab.  A count of zero is a valid
                    # successful delta containing no revisions.
                    if rows and recognized_cells == 0:
                        schema_changed = True
                        issues.append(
                            f"ALFRED output_type=3 fields were not recognized for {series_id}"
                        )

                    next_offset = offset + len(rows)
                    if next_offset >= total_count:
                        break
                    if not rows:
                        statuses.append(HealthStatus.REVISION_GAP)
                        issues.append(
                            f"ALFRED pagination ended before count for {series_id} at offset {offset}"
                        )
                        break
                    offset = next_offset

        if schema_changed:
            statuses.append(HealthStatus.SCHEMA_CHANGED)
            issues.append("one or more ALFRED output_type=3 rows were rejected")
        return CollectionResult(
            records=_sequence_revisions(records),
            health=combine_health(statuses),
            issues=tuple(dict.fromkeys(issues)),
            requests_made=requests_made,
            attempts=attempts,
            diagnostics=request_diagnostics(),
        )

    collect_revision_events = fetch_revision_events

    def fetch_observations(
        self,
        series_ids: Sequence[str],
        *,
        vintage_dates: Sequence[date],
        realtime_start: date,
        realtime_end: date,
        cutoff: datetime,
        observation_start: date | None = None,
        observation_end: date | None = None,
        frequency: str | None = None,
        aggregation_method: str = "eop",
    ) -> CollectionResult:
        """Collect full-by-vintage rows (FRED ``output_type=2``).

        Every vintage and realtime boundary is required and validated against
        ``cutoff``.  Records whose conservative ``available_at`` is later than
        the cutoff are omitted even if the provider returns them.
        """

        cutoff = ensure_utc(cutoff, field_name="cutoff")
        now = ensure_utc(self.clock(), field_name="clock")
        clean_series = tuple(dict.fromkeys(item.strip() for item in series_ids if item.strip()))
        clean_vintages = tuple(sorted(set(vintage_dates)))
        if not clean_series:
            raise ValueError("series_ids must not be empty")
        if not clean_vintages:
            raise ValueError("vintage_dates must not be empty")
        if realtime_start > realtime_end:
            raise ValueError("realtime_start must not be after realtime_end")
        if any(not realtime_start <= item <= realtime_end for item in clean_vintages):
            raise ValueError("every vintage_date must fall inside the realtime period")
        cutoff_et_date = cutoff.astimezone(_EASTERN).date()
        if realtime_end > cutoff_et_date:
            raise ValueError("realtime_end must not be after the cutoff date in US/Eastern")
        if observation_start and observation_end and observation_start > observation_end:
            raise ValueError("observation_start must not be after observation_end")

        if not self.config.rights_acknowledged:
            return CollectionResult(
                health=HealthStatus.LICENSE_BLOCKED,
                issues=(
                    f"live ALFRED collection requires explicit {ALFRED_RIGHTS_ACK_ENV} acknowledgement",
                ),
            )
        if not self.config.api_key:
            return CollectionResult(
                health=HealthStatus.DEGRADED,
                issues=(f"{FRED_API_KEY_ENV} is not configured",),
            )

        records: list[Observation] = []
        issues: list[str] = []
        statuses: list[HealthStatus] = []
        requests_made = 0
        attempts = 0
        schema_changed = False

        for series_id in clean_series:
            for vintage_batch in _chunks(clean_vintages, self.config.vintage_batch_size):
                offset = 0
                while True:
                    params: dict[str, Any] = {
                        "series_id": series_id,
                        "api_key": self.config.api_key,
                        "file_type": "json",
                        "output_type": 2,
                        "vintage_dates": ",".join(item.isoformat() for item in vintage_batch),
                        "limit": self.config.page_size,
                        "offset": offset,
                        "sort_order": "asc",
                    }
                    if observation_start is not None:
                        params["observation_start"] = observation_start.isoformat()
                    if observation_end is not None:
                        params["observation_end"] = observation_end.isoformat()
                    if frequency is not None:
                        params["frequency"] = frequency
                        params["aggregation_method"] = aggregation_method

                    try:
                        payload, used_attempts = request_json_with_retry(
                            self.transport,
                            self.config.base_url,
                            params,
                            timeout=self.config.timeout_seconds,
                            retry=self.retry,
                            sleeper=self.sleeper,
                            secrets=(self.config.api_key,),
                        )
                    except ProviderRequestError as exc:
                        attempts += exc.attempts
                        status = (
                            HealthStatus.QUOTA_EXHAUSTED
                            if exc.status_code == 429
                            else HealthStatus.DEGRADED
                        )
                        return CollectionResult(
                            records=tuple(records),
                            health=status,
                            issues=tuple(issues + [f"ALFRED request failed: {exc}"]),
                            requests_made=requests_made,
                            attempts=attempts,
                        )

                    requests_made += 1
                    attempts += used_attempts
                    if self.config.request_spacing_seconds:
                        self.sleeper(self.config.request_spacing_seconds)
                    rows = payload.get("observations")
                    try:
                        total_count = int(payload["count"])
                    except (KeyError, TypeError, ValueError):
                        total_count = -1
                    if not isinstance(rows, list) or total_count < 0:
                        schema_changed = True
                        issues.append(f"ALFRED response schema changed for {series_id}")
                        break

                    for row in rows:
                        if not isinstance(row, Mapping):
                            schema_changed = True
                            continue
                        try:
                            period_end = _date(row["date"], field="date")
                        except (KeyError, TypeError, ValueError):
                            schema_changed = True
                            continue

                        # JSON output_type=2 is a wide table.  Each requested
                        # vintage appears as SERIES_YYYYMMDD rather than as a
                        # realtime_start/value pair on the row.
                        vintage_values: list[tuple[date, Any, str]] = []
                        if "realtime_start" in row and "value" in row:
                            # Retain compatibility with older/mock row-shaped
                            # payloads while treating the wide shape as canonical.
                            try:
                                row_vintage = _date(
                                    row["realtime_start"], field="realtime_start"
                                )
                            except (TypeError, ValueError):
                                schema_changed = True
                                continue
                            vintage_values.append((row_vintage, row.get("value"), "value"))
                        else:
                            for requested_vintage in vintage_batch:
                                provider_field = (
                                    f"{series_id}_{requested_vintage.strftime('%Y%m%d')}"
                                )
                                if provider_field not in row:
                                    schema_changed = True
                                    continue
                                vintage_values.append(
                                    (requested_vintage, row.get(provider_field), provider_field)
                                )

                        for vintage, raw_value, provider_field in vintage_values:
                            available_at = self._available_at(vintage)
                            if available_at > cutoff:
                                continue
                            try:
                                value = (
                                    None
                                    if raw_value in (None, ".", "")
                                    else float(raw_value)
                                )
                            except (TypeError, ValueError):
                                schema_changed = True
                                continue
                            raw_fragment = {
                                "date": period_end.isoformat(),
                                "vintage_date": vintage.isoformat(),
                                "provider_field": provider_field,
                                "value": raw_value,
                            }
                            records.append(
                                Observation(
                                    source="alfred",
                                    series_id=series_id,
                                    observed_period_end=period_end,
                                    value=value,
                                    released_at=available_at,
                                    available_at=available_at,
                                    vintage_date=vintage,
                                    retrieved_at=now,
                                    units="",
                                    adjustment="provider_series_definition",
                                    license_class=self.config.license_class,
                                    quality_status=HealthStatus.OK,
                                    raw_sha256=_raw_sha256(raw_fragment),
                                    metadata={
                                        "provider_field": provider_field,
                                        "availability_precision": "date_proxy_18_et",
                                        "provider_output_type": 2,
                                    },
                                )
                            )

                    next_offset = offset + len(rows)
                    if next_offset >= total_count:
                        break
                    if not rows:
                        statuses.append(HealthStatus.REVISION_GAP)
                        issues.append(
                            f"ALFRED pagination ended before count for {series_id} at offset {offset}"
                        )
                        break
                    offset = next_offset

        if schema_changed:
            statuses.append(HealthStatus.SCHEMA_CHANGED)
            issues.append("one or more ALFRED rows were rejected by schema validation")
        health = combine_health(statuses)
        return CollectionResult(
            records=_sequence_revisions(records),
            health=health,
            issues=tuple(dict.fromkeys(issues)),
            requests_made=requests_made,
            attempts=attempts,
        )

    collect_observations = fetch_observations
