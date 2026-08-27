"""Release-aware source catalog and point-in-time record primitives.

This module is deliberately a collection *foundation*, not a collector.  A
catalog entry describes how an official release would become eligible and a
``ReleaseRecord`` preserves the clocks needed to replay that decision.  Merely
appearing in the catalog never means that a source has been downloaded,
parsed, certified, or admitted to the operating feature set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from enum import StrEnum
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .contracts import HealthStatus, Observation, ensure_utc


UTC = timezone.utc
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RELEASE_SOURCE_CATALOG = PROJECT_ROOT / "config" / "release-source-catalog.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ReleaseCatalogError(ValueError):
    """Raised when the source catalog or a release record is inconsistent."""


class SourceFrequency(StrEnum):
    BUSINESS_DAILY = "business_daily"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    MIXED_FREQUENCY = "mixed_frequency"
    RELEASE_DRIVEN = "release_driven"


class TimestampSemantics(StrEnum):
    """How the source timestamp is established before PIT alignment."""

    OFFICIAL_SCHEDULE = "official_schedule"
    OFFICIAL_ARCHIVE_TIMESTAMP = "official_archive_timestamp"
    SOURCE_FILE_VINTAGE_TIMESTAMP = "source_file_vintage_timestamp"
    DATE_ONLY_RELEASE = "date_only_release"
    EXACT_OBSERVATION_TIMESTAMP = "exact_observation_timestamp"
    FIRST_SEEN_ONLY = "first_seen_only"


class UnknownTimePolicy(StrEnum):
    ERROR = "error"
    NEXT_CALENDAR_DAY = "next_calendar_day"
    NEXT_WEEKLY_DECISION = "next_weekly_decision"
    FIRST_SEEN = "first_seen"


class VintagePolicy(StrEnum):
    OFFICIAL_ALL_VINTAGES = "official_all_vintages"
    DATED_RELEASE_SNAPSHOTS = "dated_release_snapshots"
    APPEND_ONLY_FIRST_SEEN = "append_only_first_seen"
    CURRENT_HISTORY_RETROSPECTIVE_ONLY = "current_history_retrospective_only"
    REVISION_PRONE_SHADOW = "revision_prone_shadow"


class PublicationRole(StrEnum):
    DERIVED_ONLY_RESEARCH = "derived_only_research"
    PRIVATE_SHADOW_ONLY = "private_shadow_only"
    RECONCILIATION_SHADOW = "reconciliation_shadow"


class SourceStatus(StrEnum):
    PLANNED = "planned"
    RIGHTS_REVIEW_REQUIRED = "rights_review_required"
    BLOCKED_PENDING_WRITTEN_LICENSE = "blocked_pending_written_license"
    PARSER_IMPLEMENTED = "parser_implemented"
    INGESTING = "ingesting"
    INGESTED = "ingested"
    CERTIFIED = "certified"
    SUSPENDED = "suspended"


class EvidenceTrack(StrEnum):
    OPERATIONAL_OOS = "operational_oos"
    RECONSTRUCTED_OOS = "reconstructed_oos"


@dataclass(frozen=True, slots=True)
class ExpectedPublication:
    """Declared release-clock semantics for one official source."""

    lag_semantics: str
    timezone_name: str
    nominal_local_time: time | None
    timestamp_semantics: TimestampSemantics
    unknown_time_policy: UnknownTimePolicy
    decision_weekday: int = 4
    decision_local_time: time = time(16, 0)

    def __post_init__(self) -> None:
        if not self.lag_semantics.strip():
            raise ReleaseCatalogError("publication lag_semantics must not be empty")
        try:
            ZoneInfo(self.timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise ReleaseCatalogError(
                f"unknown release timezone: {self.timezone_name}"
            ) from exc
        if self.decision_weekday not in range(7):
            raise ReleaseCatalogError("decision_weekday must be between 0 and 6")
        if (
            self.timestamp_semantics is TimestampSemantics.OFFICIAL_SCHEDULE
            and self.nominal_local_time is None
        ):
            raise ReleaseCatalogError(
                "official_schedule sources require nominal_local_time"
            )


@dataclass(frozen=True, slots=True)
class ReleaseSource:
    """Typed, non-claiming metadata for a planned or admitted source."""

    source_id: str
    display_name: str
    institution: str
    official_primary_url: str
    official_archive_url: str | None
    frequency: SourceFrequency
    expected_publication: ExpectedPublication
    vintage_policy: VintagePolicy
    revision_policy: str
    historical_certification: str
    rights_profile: str
    publication_role: PublicationRole
    measurement_contract: str
    enabled: bool
    ingested: bool
    status: SourceStatus

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.display_name.strip():
            raise ReleaseCatalogError("source id and display name must not be empty")
        if not self.institution.strip():
            raise ReleaseCatalogError(f"{self.source_id}: institution is required")
        _validate_official_url(
            self.official_primary_url,
            field_name=f"{self.source_id}.official_primary_url",
        )
        if self.official_archive_url is not None:
            _validate_official_url(
                self.official_archive_url,
                field_name=f"{self.source_id}.official_archive_url",
            )
        for field_name, value in (
            ("revision_policy", self.revision_policy),
            ("historical_certification", self.historical_certification),
            ("rights_profile", self.rights_profile),
            ("measurement_contract", self.measurement_contract),
        ):
            if not value.strip():
                raise ReleaseCatalogError(f"{self.source_id}.{field_name} is required")
        if self.ingested and not self.enabled:
            raise ReleaseCatalogError(
                f"{self.source_id}: an ingested source must also be enabled"
            )
        if self.status in {
            SourceStatus.RIGHTS_REVIEW_REQUIRED,
            SourceStatus.BLOCKED_PENDING_WRITTEN_LICENSE,
        } and (self.enabled or self.ingested):
            raise ReleaseCatalogError(
                f"{self.source_id}: a rights-blocked source cannot be enabled or ingested"
            )
        if self.ingested and self.status not in {
            SourceStatus.INGESTED,
            SourceStatus.CERTIFIED,
        }:
            raise ReleaseCatalogError(
                f"{self.source_id}: ingested=true requires ingested or certified status"
            )
        if not self.ingested and self.status in {
            SourceStatus.INGESTED,
            SourceStatus.CERTIFIED,
        }:
            raise ReleaseCatalogError(
                f"{self.source_id}: status cannot claim ingestion when ingested=false"
            )


@dataclass(frozen=True, slots=True)
class ReleaseSourceCatalog:
    schema_version: int
    catalog_version: str
    as_of: date
    sources: tuple[ReleaseSource, ...]
    sha256: str
    _by_id: Mapping[str, ReleaseSource] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ReleaseCatalogError("release source catalog schema_version must be 1")
        if not self.catalog_version.strip():
            raise ReleaseCatalogError("catalog_version must not be empty")
        ids = [source.source_id for source in self.sources]
        if len(ids) != len(set(ids)):
            raise ReleaseCatalogError("release source ids must be unique")
        if not _SHA256_RE.fullmatch(self.sha256):
            raise ReleaseCatalogError("catalog sha256 must be lowercase hexadecimal")
        object.__setattr__(
            self,
            "_by_id",
            MappingProxyType({source.source_id: source for source in self.sources}),
        )

    def source(self, source_id: str) -> ReleaseSource:
        try:
            return self._by_id[source_id]
        except KeyError as exc:
            raise KeyError(f"unknown release source: {source_id}") from exc

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(source.source_id for source in self.sources)

    @property
    def admitted_source_ids(self) -> tuple[str, ...]:
        """Return only sources that are both enabled and actually ingested."""

        return tuple(
            source.source_id
            for source in self.sources
            if source.enabled and source.ingested
        )


@dataclass(frozen=True, slots=True)
class ReleaseRecord:
    """One revision with explicit economic, release, receipt, and system clocks."""

    source_id: str
    series_id: str
    observed_period_end: date
    value: float | None
    source_released_at: datetime
    provider_first_seen_at: datetime
    system_retrieved_at: datetime
    revision_seq: int
    raw_sha256: str
    units: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.series_id.strip():
            raise ReleaseCatalogError("release record source and series are required")
        if self.revision_seq < 0:
            raise ReleaseCatalogError("revision_seq must be non-negative")
        if not _SHA256_RE.fullmatch(self.raw_sha256):
            raise ReleaseCatalogError(
                "raw_sha256 must be a 64-character lowercase hexadecimal digest"
            )
        if self.value is not None and not math.isfinite(float(self.value)):
            raise ReleaseCatalogError("release record value must be finite or None")
        source_released_at = ensure_utc(
            self.source_released_at,
            field_name="source_released_at",
        )
        provider_first_seen_at = ensure_utc(
            self.provider_first_seen_at,
            field_name="provider_first_seen_at",
        )
        system_retrieved_at = ensure_utc(
            self.system_retrieved_at,
            field_name="system_retrieved_at",
        )
        if provider_first_seen_at > system_retrieved_at:
            raise ReleaseCatalogError(
                "provider_first_seen_at must not be after system_retrieved_at"
            )
        if source_released_at > system_retrieved_at:
            raise ReleaseCatalogError(
                "source_released_at must not be after system_retrieved_at"
            )
        object.__setattr__(self, "source_released_at", source_released_at)
        object.__setattr__(self, "provider_first_seen_at", provider_first_seen_at)
        object.__setattr__(self, "system_retrieved_at", system_retrieved_at)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        if self.value is not None:
            object.__setattr__(self, "value", float(self.value))

    def eligibility_at(self, track: EvidenceTrack | str) -> datetime:
        """Return the only timestamp allowed for the requested evidence track."""

        selected = EvidenceTrack(track)
        if selected is EvidenceTrack.OPERATIONAL_OOS:
            return max(self.source_released_at, self.provider_first_seen_at)
        return self.source_released_at

    def is_eligible(
        self,
        decision_at: datetime,
        *,
        track: EvidenceTrack | str,
    ) -> bool:
        decision = ensure_utc(decision_at, field_name="decision_at")
        # This independent period check protects against a malformed archive
        # row whose timestamp claims publication before its economic period.
        if self.observed_period_end > decision.date():
            return False
        return self.eligibility_at(track) <= decision

    def to_observation(
        self,
        *,
        track: EvidenceTrack | str,
        license_class: str = "private_research",
    ) -> Observation:
        """Adapt this record to the existing immutable snapshot-store schema."""

        selected = EvidenceTrack(track)
        available_at = self.eligibility_at(selected)
        metadata = dict(self.metadata)
        metadata.update(
            {
                "evidence_track": selected.value,
                "availability_policy": (
                    "max_source_released_provider_first_seen"
                    if selected is EvidenceTrack.OPERATIONAL_OOS
                    else "official_release_reconstruction"
                ),
            }
        )
        return Observation(
            source=self.source_id,
            series_id=self.series_id,
            observed_period_end=self.observed_period_end,
            value=self.value,
            released_at=self.source_released_at,
            source_released_at=self.source_released_at,
            available_at=available_at,
            provider_first_seen_at=self.provider_first_seen_at,
            vintage_date=self.source_released_at.date(),
            retrieved_at=self.system_retrieved_at,
            system_retrieved_at=self.system_retrieved_at,
            revision_seq=self.revision_seq,
            units=self.units,
            adjustment="release_vintage",
            license_class=license_class,
            quality_status=HealthStatus.OK,
            raw_sha256=self.raw_sha256,
            metadata=metadata,
        )


def _validate_official_url(value: str, *, field_name: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ReleaseCatalogError(f"{field_name} must be an absolute https URL")


def _parse_local_time(value: object, *, field_name: str) -> time | None:
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"\d{2}:\d{2}", value):
        raise ReleaseCatalogError(f"{field_name} must use HH:MM or null")
    try:
        parsed = time.fromisoformat(value)
    except ValueError as exc:
        raise ReleaseCatalogError(f"{field_name} is not a valid local time") from exc
    if parsed.second or parsed.microsecond or parsed.tzinfo is not None:
        raise ReleaseCatalogError(f"{field_name} must be an unzoned minute")
    return parsed


def _parse_expected_publication(
    raw: Mapping[str, Any],
    *,
    source_id: str,
) -> ExpectedPublication:
    required = {
        "lag_semantics",
        "timezone",
        "nominal_local_time",
        "timestamp_semantics",
        "unknown_time_policy",
    }
    if set(raw) != required:
        raise ReleaseCatalogError(
            f"{source_id}.expected_publication fields must be exactly {sorted(required)}"
        )
    return ExpectedPublication(
        lag_semantics=str(raw["lag_semantics"]),
        timezone_name=str(raw["timezone"]),
        nominal_local_time=_parse_local_time(
            raw["nominal_local_time"],
            field_name=f"{source_id}.expected_publication.nominal_local_time",
        ),
        timestamp_semantics=TimestampSemantics(str(raw["timestamp_semantics"])),
        unknown_time_policy=UnknownTimePolicy(str(raw["unknown_time_policy"])),
    )


def _parse_source(raw: Mapping[str, Any]) -> ReleaseSource:
    required = {
        "id",
        "display_name",
        "institution",
        "official_primary_url",
        "official_archive_url",
        "frequency",
        "expected_publication",
        "vintage_policy",
        "revision_policy",
        "historical_certification",
        "rights_profile",
        "publication_role",
        "measurement_contract",
        "enabled",
        "ingested",
        "status",
    }
    if set(raw) != required:
        source_id = str(raw.get("id", "<unknown>"))
        raise ReleaseCatalogError(
            f"{source_id} fields must be exactly {sorted(required)}"
        )
    source_id = str(raw["id"])
    expected_raw = raw["expected_publication"]
    if not isinstance(expected_raw, Mapping):
        raise ReleaseCatalogError(f"{source_id}.expected_publication must be an object")
    if not isinstance(raw["enabled"], bool) or not isinstance(raw["ingested"], bool):
        raise ReleaseCatalogError(f"{source_id}: enabled and ingested must be booleans")
    archive_url = raw["official_archive_url"]
    if archive_url is not None and not isinstance(archive_url, str):
        raise ReleaseCatalogError(f"{source_id}.official_archive_url is invalid")
    return ReleaseSource(
        source_id=source_id,
        display_name=str(raw["display_name"]),
        institution=str(raw["institution"]),
        official_primary_url=str(raw["official_primary_url"]),
        official_archive_url=archive_url,
        frequency=SourceFrequency(str(raw["frequency"])),
        expected_publication=_parse_expected_publication(
            expected_raw,
            source_id=source_id,
        ),
        vintage_policy=VintagePolicy(str(raw["vintage_policy"])),
        revision_policy=str(raw["revision_policy"]),
        historical_certification=str(raw["historical_certification"]),
        rights_profile=str(raw["rights_profile"]),
        publication_role=PublicationRole(str(raw["publication_role"])),
        measurement_contract=str(raw["measurement_contract"]),
        enabled=raw["enabled"],
        ingested=raw["ingested"],
        status=SourceStatus(str(raw["status"])),
    )


def load_release_source_catalog(
    path: str | Path = DEFAULT_RELEASE_SOURCE_CATALOG,
) -> ReleaseSourceCatalog:
    selected = Path(path)
    try:
        payload = selected.read_bytes()
        raw = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseCatalogError(f"release source catalog unavailable: {selected}") from exc
    if not isinstance(raw, Mapping):
        raise ReleaseCatalogError("release source catalog root must be an object")
    required = {"schema_version", "catalog_version", "as_of", "sources"}
    if set(raw) != required:
        raise ReleaseCatalogError(
            f"release source catalog fields must be exactly {sorted(required)}"
        )
    sources_raw = raw["sources"]
    if not isinstance(sources_raw, list) or not sources_raw:
        raise ReleaseCatalogError("release source catalog sources must be non-empty")
    if any(not isinstance(item, Mapping) for item in sources_raw):
        raise ReleaseCatalogError("every release source must be an object")
    try:
        as_of = date.fromisoformat(str(raw["as_of"]))
    except ValueError as exc:
        raise ReleaseCatalogError("release source catalog as_of is invalid") from exc
    return ReleaseSourceCatalog(
        schema_version=int(raw["schema_version"]),
        catalog_version=str(raw["catalog_version"]),
        as_of=as_of,
        sources=tuple(_parse_source(item) for item in sources_raw),
        sha256=sha256(
            json.dumps(
                raw,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    )


def weekly_decision_at(
    week_end: date,
    *,
    timezone_name: str = "America/New_York",
    local_time: time = time(16, 0),
    weekday: int = 4,
) -> datetime:
    """Return an exact DST-aware weekly cutoff in UTC."""

    if week_end.weekday() != weekday:
        raise ReleaseCatalogError(
            f"weekly decision date must have weekday={weekday}, got {week_end.weekday()}"
        )
    try:
        zone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ReleaseCatalogError(f"unknown cutoff timezone: {timezone_name}") from exc
    return datetime.combine(week_end, local_time, tzinfo=zone).astimezone(UTC)


def _next_weekly_decision_after_date(
    release_date: date,
    publication: ExpectedPublication,
) -> datetime:
    days_until = (publication.decision_weekday - release_date.weekday()) % 7
    candidate_date = release_date + timedelta(days=days_until)
    # A release with no time is known only after the local date ends.  A Friday
    # date-only release therefore cannot enter that same Friday's 16:00 model.
    if days_until == 0:
        candidate_date += timedelta(days=7)
    return weekly_decision_at(
        candidate_date,
        timezone_name=publication.timezone_name,
        local_time=publication.decision_local_time,
        weekday=publication.decision_weekday,
    )


def resolve_source_released_at(
    source: ReleaseSource,
    release_date: date,
    *,
    exact_timestamp: datetime | None = None,
    provider_first_seen_at: datetime | None = None,
) -> datetime:
    """Resolve one source release without silently inventing an early time.

    Exact timestamps always win and must agree with the declared local release
    date.  When an archive exposes only a date, the source's explicit
    conservative policy is applied.  ``FIRST_SEEN`` never reconstructs a
    timestamp that the system did not observe.
    """

    publication = source.expected_publication
    zone = ZoneInfo(publication.timezone_name)
    if exact_timestamp is not None:
        exact = ensure_utc(exact_timestamp, field_name="exact_timestamp")
        if exact.astimezone(zone).date() != release_date:
            raise ReleaseCatalogError(
                f"{source.source_id}: exact timestamp disagrees with release_date"
            )
        return exact

    if publication.timestamp_semantics is TimestampSemantics.OFFICIAL_SCHEDULE:
        assert publication.nominal_local_time is not None
        return datetime.combine(
            release_date,
            publication.nominal_local_time,
            tzinfo=zone,
        ).astimezone(UTC)

    if publication.timestamp_semantics is TimestampSemantics.FIRST_SEEN_ONLY:
        if provider_first_seen_at is None:
            raise ReleaseCatalogError(
                f"{source.source_id}: provider first-seen timestamp is required"
            )
        return ensure_utc(provider_first_seen_at, field_name="provider_first_seen_at")

    policy = publication.unknown_time_policy
    if policy is UnknownTimePolicy.NEXT_CALENDAR_DAY:
        return datetime.combine(
            release_date + timedelta(days=1),
            time(0, 0),
            tzinfo=zone,
        ).astimezone(UTC)
    if policy is UnknownTimePolicy.NEXT_WEEKLY_DECISION:
        return _next_weekly_decision_after_date(release_date, publication)
    if policy is UnknownTimePolicy.FIRST_SEEN:
        if provider_first_seen_at is None:
            raise ReleaseCatalogError(
                f"{source.source_id}: provider first-seen timestamp is required"
            )
        return ensure_utc(provider_first_seen_at, field_name="provider_first_seen_at")
    raise ReleaseCatalogError(
        f"{source.source_id}: exact source timestamp is required by catalog policy"
    )


def eligible_release_records(
    records: Iterable[ReleaseRecord],
    decision_at: datetime,
    *,
    track: EvidenceTrack | str,
) -> tuple[ReleaseRecord, ...]:
    """Select the latest eligible revision for each source/series/period."""

    decision = ensure_utc(decision_at, field_name="decision_at")
    selected_track = EvidenceTrack(track)
    latest: dict[tuple[str, str, date], ReleaseRecord] = {}
    for record in records:
        if not record.is_eligible(decision, track=selected_track):
            continue
        key = (record.source_id, record.series_id, record.observed_period_end)
        current = latest.get(key)
        ordering = (
            record.eligibility_at(selected_track),
            record.revision_seq,
            record.system_retrieved_at,
            record.raw_sha256,
        )
        if current is None or ordering > (
            current.eligibility_at(selected_track),
            current.revision_seq,
            current.system_retrieved_at,
            current.raw_sha256,
        ):
            latest[key] = record
    return tuple(
        latest[key]
        for key in sorted(latest, key=lambda item: (item[0], item[1], item[2]))
    )


__all__ = [
    "DEFAULT_RELEASE_SOURCE_CATALOG",
    "EvidenceTrack",
    "ExpectedPublication",
    "PublicationRole",
    "ReleaseCatalogError",
    "ReleaseRecord",
    "ReleaseSource",
    "ReleaseSourceCatalog",
    "SourceFrequency",
    "SourceStatus",
    "TimestampSemantics",
    "UnknownTimePolicy",
    "VintagePolicy",
    "eligible_release_records",
    "load_release_source_catalog",
    "resolve_source_released_at",
    "weekly_decision_at",
]
