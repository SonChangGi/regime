"""Strict, prospective-only collector for the published OFR FSI CSV.

The OFR file is a current-history aggregate, not an official vintage archive.
This adapter therefore never reconstructs the proprietary underlying inputs or
backdates availability.  Every accepted file is first-seen at collection time
and later corrections are handled by the append-only snapshot layer.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import io
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from regime_lab.provider_rights import load_provider_rights

from .contracts import HealthStatus, Observation, ensure_utc
from .h10 import ByteTransport, UrllibByteTransport
from .release_archive import (
    DEFAULT_RELEASE_SOURCE_CATALOG,
    PublicationRole,
    TimestampSemantics,
    UnknownTimePolicy,
    VintagePolicy,
    load_release_source_catalog,
)


UTC = timezone.utc
EASTERN = ZoneInfo("America/New_York")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STRUCTURAL_V6_CONFIG = PROJECT_ROOT / "config" / "structural_v6_research.json"
DEFAULT_PROVIDER_RIGHTS = PROJECT_ROOT / "config" / "provider_rights.json"
OFR_FSI_SOURCE = "ofr_fsi"
OFR_FSI_DATASET = "ofr_fsi_published_aggregate"
OFR_FSI_URL = (
    "https://www.financialresearch.gov/financial-stress-index/data/fsi.csv"
)
OFR_FSI_PAGE_URL = "https://www.financialresearch.gov/financial-stress-index/"
OFR_FSI_LICENSE_CLASS = "us_federal_public_domain_published_aggregate_credit_requested"
MAX_OFR_FSI_CSV_BYTES = 8 * 1024 * 1024


# This is a data-minimization boundary, not merely a schema convenience.  A
# config edit cannot make the collector retain an underlying proprietary input.
_PUBLIC_SERIES_ALLOWLIST: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "OFR_FSI": ("OFR FSI", "published_aggregate"),
        "OFR_FSI_CREDIT": ("Credit", "published_category_contribution"),
        "OFR_FSI_EQUITY": (
            "Equity valuation",
            "published_category_contribution",
        ),
        "OFR_FSI_SAFE_ASSETS": (
            "Safe assets",
            "published_category_contribution",
        ),
        "OFR_FSI_FUNDING": ("Funding", "published_category_contribution"),
        "OFR_FSI_VOLATILITY": (
            "Volatility",
            "published_category_contribution",
        ),
        "OFR_FSI_US": ("United States", "published_region_contribution"),
        "OFR_FSI_ADVANCED_ECONOMIES": (
            "Other advanced economies",
            "published_region_contribution",
        ),
        "OFR_FSI_EMERGING_MARKETS": (
            "Emerging markets",
            "published_region_contribution",
        ),
    }
)


class OFRFSIError(ValueError):
    """Base error for an OFR FSI contract or payload violation."""


class OFRFSISchemaError(OFRFSIError):
    """The official file no longer matches the declared public schema."""


class OFRFSITimestampError(OFRFSIError):
    """A collection clock or observation lag is impossible."""


@dataclass(frozen=True, slots=True)
class OFRFSISeriesSpec:
    output_id: str
    csv_column: str
    measurement_role: str
    required: bool


@dataclass(frozen=True, slots=True)
class OFRFSIContract:
    source_url: str
    official_page_url: str
    timeout_seconds: float
    max_csv_bytes: int
    observation_lag_business_days: int
    business_day_calendar: str
    series: tuple[OFRFSISeriesSpec, ...]
    contract_sha256: str
    release_catalog_sha256: str

    def __post_init__(self) -> None:
        parsed = urlparse(self.source_url)
        if (
            self.source_url != OFR_FSI_URL
            or parsed.scheme != "https"
            or parsed.hostname != "www.financialresearch.gov"
        ):
            raise OFRFSIError("OFR FSI source must be the declared official HTTPS file")
        if self.official_page_url != OFR_FSI_PAGE_URL:
            raise OFRFSIError("OFR FSI official page contract is inconsistent")
        if self.timeout_seconds <= 0 or self.max_csv_bytes <= 0:
            raise OFRFSIError("OFR FSI transport limits must be positive")
        if self.observation_lag_business_days != 2:
            raise OFRFSIError("OFR FSI observation lag contract must remain two business days")
        if self.business_day_calendar != "USFederalHolidayCalendar":
            raise OFRFSIError("OFR FSI business-day calendar contract is inconsistent")
        if not self.series or not any(
            item.output_id == "OFR_FSI" and item.required for item in self.series
        ):
            raise OFRFSIError("OFR FSI aggregate series must be required")
        output_ids = [item.output_id for item in self.series]
        columns = [item.csv_column for item in self.series]
        if len(output_ids) != len(set(output_ids)) or len(columns) != len(set(columns)):
            raise OFRFSIError("OFR FSI series ids and columns must be unique")

    @property
    def expected_header(self) -> tuple[str, ...]:
        return ("Date", *(item.csv_column for item in self.series))


@dataclass(frozen=True, slots=True)
class OFRFSIParseResult:
    records: tuple[Observation, ...]
    response_sha256: str
    first_seen_at: datetime
    retrieved_at: datetime
    first_period: date
    last_period: date
    row_count: int
    series_ids: tuple[str, ...]
    contract_sha256: str


def _load_json_object(path: str | Path, *, label: str) -> Mapping[str, Any]:
    selected = Path(path)
    try:
        value = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OFRFSIError(f"{label} is unavailable: {selected}") from exc
    if not isinstance(value, Mapping):
        raise OFRFSIError(f"{label} root must be an object")
    return value


def _canonical_sha256(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def load_ofr_fsi_contract(
    structural_path: str | Path = DEFAULT_STRUCTURAL_V6_CONFIG,
    *,
    catalog_path: str | Path = DEFAULT_RELEASE_SOURCE_CATALOG,
    rights_path: str | Path = DEFAULT_PROVIDER_RIGHTS,
) -> OFRFSIContract:
    """Load and cross-check the V6, release-catalog, and rights contracts."""

    structural = _load_json_object(structural_path, label="structural V6 contract")
    if structural.get("research_version") != "weekly-regime-research-v6":
        raise OFRFSIError("OFR FSI collection requires the structural V6 contract")
    source_policy = structural.get("source_policy")
    sources = structural.get("core_sources")
    raw = sources.get(OFR_FSI_SOURCE)
    if not isinstance(raw, Mapping):
        raise OFRFSIError("structural V6 contract has no ofr_fsi source")
    if not isinstance(source_policy, Mapping) or (
        source_policy.get("official_primary_sources_only") is not True
        or source_policy.get("private_raw_inputs") is not True
        or source_policy.get("public_output") != "derived_results_only"
        or source_policy.get("raw_input_publication") is not False
    ):
        raise OFRFSIError("structural V6 source policy is not derived-only")
    if (
        raw.get("rights_profile") != "ofr_published_aggregate"
        or raw.get("official_page_url") != OFR_FSI_PAGE_URL
        or raw.get("download_url") != OFR_FSI_URL
        or raw.get("format") != "csv"
        or raw.get("role") != "retrospective_sensitivity_then_prospective_shadow"
    ):
        raise OFRFSIError("structural V6 OFR aggregate contract is inconsistent")
    schedule = raw.get("release_schedule")
    cutoff = raw.get("cutoff_rule")
    pit = raw.get("pit")
    if not isinstance(schedule, Mapping) or (
        schedule.get("frequency") != "daily"
        or schedule.get("observation_lag_business_days") != 2
        or schedule.get("business_day_calendar") != "USFederalHolidayCalendar"
        or schedule.get("time") is not None
    ):
        raise OFRFSIError("structural V6 OFR release schedule is inconsistent")
    if not isinstance(cutoff, Mapping) or (
        cutoff.get("live_availability") != "collection_first_seen_at"
        or cutoff.get("same_day_unknown_time") != "next_week"
    ):
        raise OFRFSIError("structural V6 OFR cutoff rule is inconsistent")
    if not isinstance(pit, Mapping) or (
        pit.get("historical_vintage_archive") != "not_available"
        or pit.get("live_input")
        != "freeze_each_csv_snapshot_with_retrieved_at_and_sha256"
    ):
        raise OFRFSIError("structural V6 OFR PIT contract is inconsistent")

    series_raw = raw.get("series")
    if not isinstance(series_raw, list) or not series_raw:
        raise OFRFSIError("structural V6 OFR series must be non-empty")
    parsed_series: list[OFRFSISeriesSpec] = []
    for item in series_raw:
        if not isinstance(item, Mapping) or set(item) != {
            "output_id",
            "csv_column",
            "domain",
            "frequency",
            "required",
        }:
            raise OFRFSIError("each OFR FSI series must use the exact V6 schema")
        output_id = str(item["output_id"])
        try:
            allowed_column, measurement_role = _PUBLIC_SERIES_ALLOWLIST[output_id]
        except KeyError as exc:
            raise OFRFSIError(
                f"OFR FSI series is not a published aggregate/contribution: {output_id}"
            ) from exc
        if (
            item["csv_column"] != allowed_column
            or item["domain"] != "financial_conditions"
            or item["frequency"] != "daily"
            or not isinstance(item["required"], bool)
        ):
            raise OFRFSIError(f"OFR FSI series contract is invalid: {output_id}")
        parsed_series.append(
            OFRFSISeriesSpec(
                output_id=output_id,
                csv_column=allowed_column,
                measurement_role=measurement_role,
                required=bool(item["required"]),
            )
        )

    catalog = load_release_source_catalog(catalog_path)
    catalog_source = catalog.source(OFR_FSI_SOURCE)
    if (
        catalog_source.official_primary_url != OFR_FSI_PAGE_URL
        or catalog_source.expected_publication.timestamp_semantics
        is not TimestampSemantics.FIRST_SEEN_ONLY
        or catalog_source.expected_publication.unknown_time_policy
        is not UnknownTimePolicy.FIRST_SEEN
        or catalog_source.vintage_policy is not VintagePolicy.APPEND_ONLY_FIRST_SEEN
        or catalog_source.rights_profile != "ofr_published_aggregate"
        or catalog_source.publication_role is not PublicationRole.DERIVED_ONLY_RESEARCH
        or catalog_source.enabled
        or catalog_source.ingested
        or "excluding reconstructable proprietary inputs"
        not in catalog_source.measurement_contract
    ):
        raise OFRFSIError("release-source catalog OFR contract is inconsistent")

    rights = load_provider_rights(rights_path)
    rights_entry = rights.get("providers", {}).get(OFR_FSI_SOURCE)
    if not isinstance(rights_entry, Mapping):
        raise OFRFSIError("provider-rights policy has no OFR FSI entry")
    capabilities = rights_entry.get("capabilities")
    conditions = rights_entry.get("conditions")
    if (
        rights_entry.get("status") != "allowed"
        or not isinstance(capabilities, Mapping)
        or capabilities.get("collection") is not True
        or capabilities.get("local_storage") is not True
        or not isinstance(conditions, Mapping)
        or conditions.get("published_aggregate_only") is not True
        or conditions.get("underlying_proprietary_inputs_excluded") is not True
        or conditions.get("credit_requested") is not True
    ):
        raise OFRFSIError("provider-rights OFR aggregate conditions are incomplete")

    contract_material = {
        "research_version": structural["research_version"],
        "source_policy": source_policy,
        "ofr_fsi": raw,
        "release_catalog_sha256": catalog.sha256,
        "rights": rights_entry,
    }
    return OFRFSIContract(
        source_url=OFR_FSI_URL,
        official_page_url=OFR_FSI_PAGE_URL,
        timeout_seconds=30.0,
        max_csv_bytes=MAX_OFR_FSI_CSV_BYTES,
        observation_lag_business_days=2,
        business_day_calendar="USFederalHolidayCalendar",
        series=tuple(parsed_series),
        contract_sha256=_canonical_sha256(contract_material),
        release_catalog_sha256=catalog.sha256,
    )


def _business_day_on_or_before(value: date, lag: int) -> date:
    calendar = USFederalHolidayCalendar()
    holiday_start = value - timedelta(days=max(14, lag * 4 + 7))
    federal_holidays = {
        timestamp.date()
        for timestamp in calendar.holidays(
            start=pd.Timestamp(holiday_start),
            end=pd.Timestamp(value),
        )
    }
    result = value
    remaining = lag
    while remaining:
        result -= timedelta(days=1)
        if result.weekday() < 5 and result not in federal_holidays:
            remaining -= 1
    return result


def _finite_float(token: str, *, row_number: int, column: str) -> float:
    selected = token.strip()
    if not selected:
        raise OFRFSISchemaError(f"row {row_number} {column} is empty")
    try:
        value = float(selected)
    except ValueError as exc:
        raise OFRFSISchemaError(
            f"row {row_number} {column} is not numeric"
        ) from exc
    if not math.isfinite(value):
        raise OFRFSISchemaError(f"row {row_number} {column} must be finite")
    return value


def parse_ofr_fsi_csv(
    payload: bytes,
    contract: OFRFSIContract,
    *,
    first_seen_at: datetime,
    retrieved_at: datetime,
) -> OFRFSIParseResult:
    """Parse one complete official CSV snapshot under strict PIT semantics."""

    if not isinstance(payload, bytes):
        raise TypeError("OFR FSI payload must be bytes")
    if not payload or len(payload) > contract.max_csv_bytes:
        raise OFRFSISchemaError("OFR FSI CSV is empty or exceeds the size limit")
    first_seen = ensure_utc(first_seen_at, field_name="provider_first_seen_at")
    retrieved = ensure_utc(retrieved_at, field_name="system_retrieved_at")
    if first_seen > retrieved:
        raise OFRFSITimestampError(
            "provider_first_seen_at must not follow system_retrieved_at"
        )
    if b"\x00" in payload:
        raise OFRFSISchemaError("OFR FSI CSV contains NUL bytes")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise OFRFSISchemaError("OFR FSI CSV must be UTF-8") from exc
    try:
        rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as exc:
        raise OFRFSISchemaError("OFR FSI CSV is malformed") from exc
    if len(rows) < 2:
        raise OFRFSISchemaError("OFR FSI CSV has no data rows")
    header = tuple(rows[0])
    if header != contract.expected_header:
        raise OFRFSISchemaError(
            "OFR FSI CSV header changed; expected only the declared public "
            "aggregate/category/region columns"
        )

    digest = sha256(payload).hexdigest()
    periods: list[date] = []
    records: list[Observation] = []
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            raise OFRFSISchemaError(f"row {row_number} has the wrong column count")
        raw_date = row[0]
        try:
            period = date.fromisoformat(raw_date)
        except ValueError as exc:
            raise OFRFSISchemaError(f"row {row_number} Date must be ISO-8601") from exc
        if period.isoformat() != raw_date or period.weekday() >= 5:
            raise OFRFSISchemaError(
                f"row {row_number} Date must be a canonical business-day date"
            )
        if periods and period <= periods[-1]:
            raise OFRFSISchemaError("OFR FSI dates must be unique and strictly increasing")
        periods.append(period)
        for column_number, spec in enumerate(contract.series, start=1):
            value = _finite_float(
                row[column_number],
                row_number=row_number,
                column=spec.csv_column,
            )
            records.append(
                Observation(
                    source=OFR_FSI_SOURCE,
                    series_id=spec.output_id,
                    observed_period_end=period,
                    value=value,
                    released_at=None,
                    source_released_at=None,
                    available_at=first_seen,
                    provider_first_seen_at=first_seen,
                    vintage_date=first_seen.astimezone(EASTERN).date(),
                    retrieved_at=retrieved,
                    system_retrieved_at=retrieved,
                    revision_seq=0,
                    units="index_points",
                    adjustment="published_aggregate_snapshot",
                    license_class=OFR_FSI_LICENSE_CLASS,
                    quality_status=HealthStatus.OK,
                    raw_sha256=digest,
                    metadata={
                        "contract": "v6",
                        "contract_sha256": contract.contract_sha256,
                        "source_column": spec.csv_column,
                        "measurement_role": spec.measurement_role,
                        "availability_basis": "collection_first_seen_at",
                        "observation_lag_calendar": contract.business_day_calendar,
                        "source_release_time_known": False,
                        "historical_vintage_certified": False,
                        "underlying_proprietary_inputs_included": False,
                        "raw_payload_publication": False,
                    },
                )
            )

    latest_allowed = _business_day_on_or_before(
        first_seen.astimezone(EASTERN).date(),
        contract.observation_lag_business_days,
    )
    if periods[-1] > latest_allowed:
        raise OFRFSITimestampError(
            "OFR FSI latest observation violates the declared two-business-day lag"
        )
    return OFRFSIParseResult(
        records=tuple(records),
        response_sha256=digest,
        first_seen_at=first_seen,
        retrieved_at=retrieved,
        first_period=periods[0],
        last_period=periods[-1],
        row_count=len(periods),
        series_ids=tuple(item.output_id for item in contract.series),
        contract_sha256=contract.contract_sha256,
    )


@dataclass(frozen=True, slots=True)
class OFRFSIConfig:
    contract: OFRFSIContract


class OFRFSIClient:
    """Fetch exactly one official OFR CSV; transport is injectable in tests."""

    def __init__(
        self,
        config: OFRFSIConfig,
        *,
        transport: ByteTransport | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self.clock = clock or (lambda: datetime.now(UTC))
        self.transport = transport or UrllibByteTransport(clock=self.clock)

    def collect(self) -> OFRFSIParseResult:
        contract = self.config.contract
        response = self.transport.get_bytes(
            contract.source_url,
            timeout=contract.timeout_seconds,
        )
        content_type = next(
            (
                value
                for key, value in response.headers.items()
                if key.lower() == "content-type"
            ),
            "",
        ).split(";", 1)[0].strip().lower()
        if content_type and content_type not in {
            "text/csv",
            "application/csv",
            "application/octet-stream",
            "text/plain",
        }:
            raise OFRFSISchemaError("OFR FSI response content type is not CSV")
        if len(response.body) > contract.max_csv_bytes:
            raise OFRFSISchemaError("OFR FSI response exceeds the size limit")
        # With no certified source publication time, leave source_released_at
        # unknown; first successful receipt is the only operational clock.
        return parse_ofr_fsi_csv(
            response.body,
            contract,
            first_seen_at=response.retrieved_at,
            retrieved_at=response.retrieved_at,
        )


__all__ = [
    "DEFAULT_PROVIDER_RIGHTS",
    "DEFAULT_STRUCTURAL_V6_CONFIG",
    "MAX_OFR_FSI_CSV_BYTES",
    "OFR_FSI_DATASET",
    "OFR_FSI_LICENSE_CLASS",
    "OFR_FSI_PAGE_URL",
    "OFR_FSI_SOURCE",
    "OFR_FSI_URL",
    "OFRFSIClient",
    "OFRFSIConfig",
    "OFRFSIContract",
    "OFRFSIError",
    "OFRFSIParseResult",
    "OFRFSISchemaError",
    "OFRFSISeriesSpec",
    "OFRFSITimestampError",
    "load_ofr_fsi_contract",
    "parse_ofr_fsi_csv",
]
