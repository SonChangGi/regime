"""Strict parser and collector for immutable Federal Reserve H.10 releases."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
import hashlib
from html.parser import HTMLParser
import json
import math
import re
from time import sleep as _sleep
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from .contracts import HealthStatus, Observation, normalize_revision_sequences
from .h10 import (
    BUSINESS_DAY_FREQUENCY,
    ByteTransport,
    DEFAULT_ALLOWED_FX,
    H10SchemaError,
    SERIES_CATALOG,
    UrllibByteTransport,
)


UTC = timezone.utc
EASTERN = ZoneInfo("America/New_York")
H10_ARCHIVE_INDEX_URL = (
    "https://www.federalreserve.gov/releases/h10/releaseDates.json"
)
H10_ARCHIVE_AVAILABILITY_BASIS = "official_archive_release_schedule"
H10_ARCHIVE_NORMAL_AVAILABILITY_BASIS = "archived_release_date_16_15_ET"
H10_ARCHIVE_CORRECTION_AVAILABILITY_BASIS = "date_only_conservative_next_day"
H10_FIRST_SEEN_AVAILABILITY_BASIS = "collection_first_seen_at"
H10_ARCHIVE_REVISION_POLICY = "later_official_release_preserved_as_new_vintage"
H10_ARCHIVE_CORRECTION_EQUIVALENT_POLICY = (
    "declared_or_material_revision_or_complete_republication"
)
H10_ARCHIVE_CORRECTION_EQUIVALENT_COMPONENTS: tuple[str, ...] = (
    "declared_correction",
    "material_revision",
    "complete_republication",
)
H10_ARCHIVE_SHORT_GAP_AUXILIARY_DAYS = 3
H10_ARCHIVE_MAX_RELEASES = 5_000
H10_ARCHIVE_MAX_HTML_BYTES = 2_000_000
H10_ARCHIVE_RELEASE_TIME = time(16, 15)
H10_ARCHIVE_EVALUATION_START = date(2022, 1, 1)
H10_ARCHIVE_EVALUATION_START_RATIONALE = (
    "post_2019_06_24_jan06_index_rebase_common_scale"
)
H10_ARCHIVE_RELEASE_PATTERN = re.compile(
    r"^/releases/h10/(?P<date>\d{8})/?$",
    flags=re.IGNORECASE,
)


class H10ArchiveError(ValueError):
    """An official archive page violated the frozen PIT ingest contract."""


@dataclass(frozen=True, slots=True)
class H10ArchiveReleaseLink:
    release_date: date
    url: str


@dataclass(frozen=True, slots=True)
class H10ArchiveRelease:
    release_date: date
    last_update_date: date
    available_at: datetime
    availability_basis: str
    retrieved_at: datetime
    source_url: str
    snapshot_sha256: str
    records: tuple[Observation, ...]


@dataclass(frozen=True, slots=True)
class H10ArchiveCorrectionEquivalentEvent:
    event_date: date
    available_at: datetime
    prior_release_date: date | None
    prior_release_gap_days: int | None
    short_gap_auxiliary_evidence: bool
    declared_correction: bool
    material_revision_rows: int
    complete_republication: bool
    current_series_date_rows: int
    overlap_series_date_rows: int
    new_series_date_rows: int
    trigger_components: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class H10ArchiveLineage:
    release_count: int
    events: tuple[H10ArchiveCorrectionEquivalentEvent, ...]
    lineage_sha256: str

    @property
    def declared_correction_event_count(self) -> int:
        return sum(event.declared_correction for event in self.events)

    @property
    def detected_revision_event_count(self) -> int:
        return sum(event.material_revision_rows > 0 for event in self.events)

    @property
    def detected_revision_row_count(self) -> int:
        return sum(event.material_revision_rows for event in self.events)

    @property
    def complete_republication_event_count(self) -> int:
        return sum(event.complete_republication for event in self.events)

    @property
    def short_gap_auxiliary_event_count(self) -> int:
        return sum(event.short_gap_auxiliary_evidence for event in self.events)


@dataclass(frozen=True, slots=True)
class H10ArchiveCollection:
    releases: tuple[H10ArchiveRelease, ...]
    records: tuple[Observation, ...]
    requested_at: datetime
    retrieved_at: datetime
    index_sha256: str
    collection_sha256: str
    revision_event_count: int
    discovered_release_dates: tuple[date, ...] = ()
    lineage: H10ArchiveLineage | None = None


@dataclass(slots=True)
class _Cell:
    tag: str
    text: str


class _ArchiveHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.anchors: list[str] = []
        self.date_text: list[str] = []
        self.last_update_text: list[str] = []
        self.tables: list[list[list[_Cell]]] = []
        self._date_depth = 0
        self._last_update_depth = 0
        self._table_depth = 0
        self._table: list[list[_Cell]] | None = None
        self._row: list[_Cell] | None = None
        self._cell_tag: str | None = None
        self._cell_text: list[str] = []
        self._skip_depth = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = {str(key).lower(): value for key, value in attrs}
        lowered = tag.lower()
        if lowered == "a" and attributes.get("href"):
            self.anchors.append(str(attributes["href"]))
        classes = set(str(attributes.get("class") or "").lower().split())
        if lowered == "div":
            if self._date_depth:
                self._date_depth += 1
            elif "dates" in classes:
                self._date_depth = 1
            if self._last_update_depth:
                self._last_update_depth += 1
            elif "lastupdate" in classes:
                self._last_update_depth = 1
        if lowered == "table":
            if self._table_depth:
                raise H10ArchiveError("nested H.10 archive tables are unsupported")
            self._table_depth = 1
            self._table = []
        elif self._table_depth:
            if lowered == "tr":
                if self._row is not None:
                    raise H10ArchiveError("nested H.10 archive rows are unsupported")
                self._row = []
            elif lowered in {"th", "td"} and self._row is not None:
                if self._cell_tag is not None:
                    raise H10ArchiveError("nested H.10 archive cells are unsupported")
                self._cell_tag = lowered
                self._cell_text = []
            elif lowered == "sup" and self._cell_tag is not None:
                self._skip_depth = 1
            elif self._skip_depth:
                self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if self._skip_depth:
            self._skip_depth -= 1
        elif (
            lowered in {"th", "td"}
            and self._cell_tag == lowered
            and self._row is not None
        ):
            text = " ".join("".join(self._cell_text).split())
            self._row.append(_Cell(lowered, text))
            self._cell_tag = None
            self._cell_text = []
        elif lowered == "tr" and self._row is not None:
            if self._table is None:
                raise H10ArchiveError("H.10 archive row is outside a table")
            if self._row:
                self._table.append(self._row)
            self._row = None

        if self._table_depth and lowered == "table":
            if self._table is None:
                raise H10ArchiveError("H.10 archive table parser lost state")
            self.tables.append(self._table)
            self._table = None
            self._table_depth = 0
        if self._date_depth and lowered == "div":
            self._date_depth -= 1
        if self._last_update_depth and lowered == "div":
            self._last_update_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._date_depth:
            self.date_text.append(data)
        if self._last_update_depth:
            self.last_update_text.append(data)
        if self._cell_tag is not None and not self._skip_depth:
            self._cell_text.append(data)


def _aware_utc(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def archive_release_available_at(
    release_date: date,
    *,
    last_update_date: date | None = None,
) -> datetime:
    if not isinstance(release_date, date):
        raise TypeError("release_date must be a date")
    updated = release_date if last_update_date is None else last_update_date
    if not isinstance(updated, date) or updated < release_date:
        raise ValueError("last_update_date must not precede release_date")
    if updated == release_date:
        local = datetime.combine(
            release_date,
            H10_ARCHIVE_RELEASE_TIME,
            tzinfo=EASTERN,
        )
    else:
        # The archive gives correction pages a date but no intraday timestamp.
        # The following midnight is a conservative, non-backdated availability.
        local = datetime.combine(
            updated + timedelta(days=1),
            time(0, 0),
            tzinfo=EASTERN,
        )
    return local.astimezone(UTC)


def _decode_html(payload: bytes) -> str:
    if not isinstance(payload, bytes):
        raise TypeError("H.10 archive payload must be bytes")
    if not payload or len(payload) > H10_ARCHIVE_MAX_HTML_BYTES:
        raise H10ArchiveError("H.10 archive HTML size is outside accepted bounds")
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise H10ArchiveError("H.10 archive HTML must be UTF-8") from exc


def _parse_document(payload: bytes) -> _ArchiveHTMLParser:
    parser = _ArchiveHTMLParser()
    try:
        parser.feed(_decode_html(payload))
        parser.close()
    except H10ArchiveError:
        raise
    except Exception as exc:
        raise H10ArchiveError("H.10 archive HTML is malformed") from exc
    return parser


def discover_h10_archive_releases(
    payload: bytes,
    *,
    index_url: str = H10_ARCHIVE_INDEX_URL,
) -> tuple[H10ArchiveReleaseLink, ...]:
    """Discover official dated releases without assuming a weekly cadence."""

    parsed_index = urlparse(index_url)
    if (
        parsed_index.scheme != "https"
        or parsed_index.hostname != "www.federalreserve.gov"
        or parsed_index.path.lower()
        != "/releases/h10/releasedates.json"
        or parsed_index.params
        or parsed_index.query
        or parsed_index.fragment
    ):
        raise H10ArchiveError(
            "H.10 archive index must use the official releaseDates.json URL"
        )
    if not isinstance(payload, bytes):
        raise TypeError("H.10 archive index payload must be bytes")
    if not payload or len(payload) > H10_ARCHIVE_MAX_HTML_BYTES:
        raise H10ArchiveError("H.10 archive index size is outside accepted bounds")
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise H10ArchiveError("H.10 archive index must be valid UTF-8 JSON") from exc
    if not isinstance(document, list) or not document:
        raise H10ArchiveError("H.10 archive index must be a non-empty list")
    links: dict[date, str] = {}
    for year_row in document:
        if not isinstance(year_row, dict) or set(year_row) != {"yearValue", "Months"}:
            raise H10ArchiveError("H.10 archive year schema changed")
        year_value = year_row["yearValue"]
        months = year_row["Months"]
        if (
            not isinstance(year_value, str)
            or re.fullmatch(r"\d{4}", year_value) is None
            or not isinstance(months, list)
            or not months
        ):
            raise H10ArchiveError("H.10 archive year row is invalid")
        for month_row in months:
            if not isinstance(month_row, dict) or set(month_row) != {
                "MonthName",
                "MonthValue",
                "Dates",
            }:
                raise H10ArchiveError("H.10 archive month schema changed")
            month_name = month_row["MonthName"]
            month_value = month_row["MonthValue"]
            dates = month_row["Dates"]
            if (
                not isinstance(month_name, str)
                or not month_name.strip()
                or not isinstance(month_value, str)
                or re.fullmatch(r"\d{6}", month_value) is None
                or not month_value.startswith(year_value)
                or not isinstance(dates, list)
                or not dates
                or any(
                    not isinstance(raw_date, str)
                    or re.fullmatch(r"\d{8}", raw_date) is None
                    for raw_date in dates
                )
            ):
                raise H10ArchiveError("H.10 archive month row is invalid")
            try:
                parsed_month = datetime.strptime(month_value, "%Y%m")
            except ValueError as exc:
                raise H10ArchiveError("H.10 archive month value is invalid") from exc
            if parsed_month.strftime("%B").casefold() != month_name.strip().casefold():
                raise H10ArchiveError("H.10 archive month name does not match its value")
            for raw_date in dates:
                if not raw_date.startswith(month_value):
                    raise H10ArchiveError("H.10 archive date is outside its month")
                try:
                    release_date = datetime.strptime(raw_date, "%Y%m%d").date()
                except ValueError as exc:
                    raise H10ArchiveError("H.10 archive date is invalid") from exc
                if release_date in links:
                    raise H10ArchiveError("H.10 archive index contains a duplicate date")
                links[release_date] = (
                    "https://www.federalreserve.gov"
                    f"/releases/h10/{release_date.strftime('%Y%m%d')}/"
                )
    if len(links) > H10_ARCHIVE_MAX_RELEASES:
        raise H10ArchiveError("H.10 archive index exceeds the release bound")
    return tuple(
        H10ArchiveReleaseLink(release_date=release_date, url=links[release_date])
        for release_date in sorted(links)
    )


def _document_date(parts: Sequence[str], *, label: str) -> date:
    text = " ".join(" ".join(parts).split())
    match = re.search(
        rf"{re.escape(label)}\s*:\s*([A-Za-z]+\s+\d{{1,2}},\s+\d{{4}})",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise H10ArchiveError(f"H.10 archive {label.lower()} is missing")
    for pattern in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(match.group(1), pattern).date()
        except ValueError:
            continue
    raise H10ArchiveError(f"H.10 archive {label.lower()} is invalid")


_MONTHS = {
    name.lower(): number
    for number, name in enumerate(
        (
            "",
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
        )
    )
    if name
}


def _header_observation_date(value: str, *, release_date: date) -> date | None:
    text = " ".join(value.replace(".", " ").split())
    numeric = re.fullmatch(r"(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?", text)
    if numeric is not None:
        month = int(numeric.group(1))
        day = int(numeric.group(2))
        year_raw = numeric.group(3)
        year = release_date.year if year_raw is None else int(year_raw)
        if year < 100:
            year += 2_000
    else:
        named = re.fullmatch(
            r"([A-Za-z]+)\s+(\d{1,2})(?:,?\s+(\d{4}))?",
            text,
        )
        if named is None:
            return None
        month = _MONTHS.get(named.group(1)[:3].lower(), 0)
        day = int(named.group(2))
        year = int(named.group(3) or release_date.year)
    try:
        candidate = date(year, month, day)
    except ValueError as exc:
        raise H10ArchiveError("H.10 archive table date is invalid") from exc
    if candidate > release_date and candidate.year == release_date.year:
        try:
            candidate = candidate.replace(year=candidate.year - 1)
        except ValueError as exc:
            raise H10ArchiveError("H.10 archive year rollover is invalid") from exc
    return candidate


def _normalized_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _fx_code_from_label(value: str) -> str | None:
    label = _normalized_label(value)
    if "goods only" in label:
        return None
    modern_index = re.fullmatch(
        r"\d+\s+(broad|afe|eme)\s+jan06\s+100",
        label,
    )
    if modern_index is not None:
        return {
            "broad": "BRD",
            "afe": "AFE",
            "eme": "EME",
        }[modern_index.group(1)]
    rules = (
        ("BRD", ("broad dollar index",)),
        (
            "AFE",
            ("advanced foreign economies", "a f e dollar index"),
        ),
        (
            "EME",
            ("emerging market economies", "e m e dollar index"),
        ),
        ("AUD", ("australia", "australian")),
        ("BRL", ("brazil", "brazilian")),
        ("CAD", ("canada", "canadian")),
        ("CNY", ("china", "chinese")),
        ("EUR", ("euro",)),
        ("JPY", ("japan", "japanese")),
        ("MXN", ("mexico", "mexican")),
        ("CHF", ("switzerland", "swiss")),
        ("GBP", ("united kingdom", "u k pound", "british pound")),
    )
    matches = [
        code
        for code, needles in rules
        if any(needle in label for needle in needles)
    ]
    if len(matches) > 1:
        raise H10ArchiveError(f"ambiguous H.10 archive series label: {value!r}")
    return matches[0] if matches else None


def _observation_value(raw: str) -> tuple[float | None, str]:
    token = "".join(raw.replace("\u2212", "-").split()).upper()
    if token in {"", "--", "---", "NA", "N/A", "ND", "NC", "N.A."}:
        return None, "ND"
    token = token.replace(",", "")
    try:
        value = float(token)
    except ValueError as exc:
        raise H10ArchiveError(f"invalid H.10 archive observation: {raw!r}") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise H10ArchiveError("H.10 archive observation must be positive and finite")
    return value, "A"


def _series_records_from_tables(
    tables: Sequence[Sequence[Sequence[_Cell]]],
    *,
    release_date: date,
    last_update_date: date,
    available_at: datetime,
    availability_basis: str,
    retrieved_at: datetime,
    source_url: str,
    snapshot_sha256: str,
    allowed_fx: tuple[str, ...],
) -> tuple[Observation, ...]:
    parsed: dict[tuple[str, date], Observation] = {}
    seen_codes: set[str] = set()
    for table in tables:
        date_positions: dict[int, date] = {}
        header_index: int | None = None
        for index, row in enumerate(table):
            candidates = {
                position: observed
                for position, cell in enumerate(row)
                if (
                    observed := _header_observation_date(
                        cell.text,
                        release_date=release_date,
                    )
                )
                is not None
            }
            if len(candidates) > len(date_positions):
                date_positions = candidates
                header_index = index
        if header_index is None or not date_positions:
            continue
        for row in table[header_index + 1 :]:
            if not row:
                continue
            first_date_position = min(date_positions)
            leading_cells = row[:first_date_position]
            if not leading_cells:
                raise H10ArchiveError("H.10 archive series row has no label cells")
            combined_label = " ".join(
                cell.text for cell in leading_cells if cell.text.strip()
            )
            code = _fx_code_from_label(combined_label)
            if code is None or code not in allowed_fx:
                continue
            if max(date_positions) >= len(row):
                raise H10ArchiveError(
                    f"H.10 archive row length changed for {code}"
                )
            seen_codes.add(code)
            spec = SERIES_CATALOG[code]
            unit_raw = leading_cells[-1].text.strip()
            if not unit_raw:
                raise H10ArchiveError(
                    f"H.10 archive unit/currency cell is empty for {code}"
                )
            for position, observed in date_positions.items():
                if observed >= release_date:
                    raise H10ArchiveError(
                        "H.10 archive observation must precede its release date"
                    )
                value, status = _observation_value(row[position].text)
                raw_identity = "|".join(
                    (
                        snapshot_sha256,
                        code,
                        observed.isoformat(),
                        status,
                        "" if value is None else format(value, ".17g"),
                    )
                )
                raw_sha256 = hashlib.sha256(raw_identity.encode("utf-8")).hexdigest()
                metadata: dict[str, Any] = {
                    "release_id": "H10",
                    "dataset_id": "H10_OFFICIAL_RELEASE_ARCHIVE",
                    "fx_code": code,
                    "frequency_code": BUSINESS_DAY_FREQUENCY,
                    "series_name": spec.series_name,
                    "currency_code": spec.currency_code,
                    "unit_raw": unit_raw,
                    "unit_multiplier_raw": "1",
                    "quote_convention": spec.quote_convention.value,
                    "usd_strength_sign": spec.usd_strength_sign,
                    "obs_status": status,
                    "archive_release_date": release_date.isoformat(),
                    "archive_last_update_date": last_update_date.isoformat(),
                    "availability_basis": availability_basis,
                    "archive_chain_availability_basis": (
                        H10_ARCHIVE_AVAILABILITY_BASIS
                    ),
                    "official_release_archive_ingest": True,
                    "archive_revision_policy": H10_ARCHIVE_REVISION_POLICY,
                    "archive_snapshot_sha256": snapshot_sha256,
                    "source_url": source_url,
                }
                record = Observation(
                    source="frb_h10",
                    series_id=f"H10|FX={code}|FREQ={BUSINESS_DAY_FREQUENCY}",
                    observed_period_end=observed,
                    value=value,
                    released_at=available_at,
                    available_at=available_at,
                    vintage_date=last_update_date,
                    retrieved_at=retrieved_at,
                    units=str(metadata["unit_raw"]),
                    license_class=(
                        "federal_reserve_board_public_domain_citation_requested"
                    ),
                    quality_status=(
                        HealthStatus.OK if status == "A" else HealthStatus.UNAVAILABLE
                    ),
                    raw_sha256=raw_sha256,
                    metadata=metadata,
                )
                key = (code, observed)
                prior = parsed.get(key)
                if prior is not None and (
                    prior.value != record.value
                    or prior.quality_status is not record.quality_status
                ):
                    raise H10ArchiveError(
                        f"conflicting H.10 archive values within one release for {code}"
                    )
                parsed[key] = record
    missing = sorted(set(allowed_fx).difference(seen_codes))
    if missing:
        raise H10ArchiveError(
            "H.10 archive release is missing required series: " + ", ".join(missing)
        )
    if not parsed:
        raise H10ArchiveError("H.10 archive release contains no observations")
    return tuple(
        parsed[key]
        for key in sorted(parsed, key=lambda item: (item[0], item[1]))
    )


def parse_h10_archive_release(
    payload: bytes,
    *,
    source_url: str,
    retrieved_at: datetime,
    expected_release_date: date | None = None,
    allowed_fx: Sequence[str] = DEFAULT_ALLOWED_FX,
) -> H10ArchiveRelease:
    """Parse one immutable release page and stamp its official 16:15 ET time."""

    retrieved = _aware_utc(retrieved_at, field_name="retrieved_at")
    parsed_url = urlparse(source_url)
    if parsed_url.scheme != "https" or parsed_url.hostname != "www.federalreserve.gov":
        raise H10ArchiveError("H.10 archive release must use the official HTTPS host")
    match = H10_ARCHIVE_RELEASE_PATTERN.fullmatch(parsed_url.path)
    if match is None:
        raise H10ArchiveError("H.10 archive release URL is not dated")
    url_date = datetime.strptime(match.group("date"), "%Y%m%d").date()
    document = _parse_document(payload)
    release_date = _document_date(document.date_text, label="Release Date")
    last_update_date = _document_date(
        document.last_update_text,
        label="Last Update",
    )
    if last_update_date != url_date or (
        expected_release_date is not None
        and last_update_date != expected_release_date
    ):
        raise H10ArchiveError("H.10 archive last update does not match its URL")
    if release_date > last_update_date:
        raise H10ArchiveError("H.10 archive release date follows its last update")
    is_correction = last_update_date != release_date
    available_at = archive_release_available_at(
        release_date,
        last_update_date=last_update_date,
    )
    availability_basis = (
        H10_ARCHIVE_CORRECTION_AVAILABILITY_BASIS
        if is_correction
        else H10_ARCHIVE_NORMAL_AVAILABILITY_BASIS
    )
    if retrieved < available_at:
        raise H10ArchiveError("H.10 archive retrieval precedes official availability")
    selected = tuple(str(code).strip().upper() for code in allowed_fx)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or set(selected).difference(SERIES_CATALOG)
    ):
        raise H10ArchiveError("H.10 archive allowed_fx contract is invalid")
    snapshot_sha256 = hashlib.sha256(payload).hexdigest()
    records = _series_records_from_tables(
        document.tables,
        release_date=release_date,
        last_update_date=last_update_date,
        available_at=available_at,
        availability_basis=availability_basis,
        retrieved_at=retrieved,
        source_url=source_url,
        snapshot_sha256=snapshot_sha256,
        allowed_fx=selected,
    )
    return H10ArchiveRelease(
        release_date=release_date,
        last_update_date=last_update_date,
        available_at=available_at,
        availability_basis=availability_basis,
        retrieved_at=retrieved,
        source_url=source_url,
        snapshot_sha256=snapshot_sha256,
        records=records,
    )


def detect_h10_archive_correction_equivalents(
    releases: Sequence[H10ArchiveRelease],
) -> H10ArchiveLineage:
    """Classify correction-equivalent pages from adjacent archive lineage."""

    ordered = tuple(sorted(releases, key=lambda item: item.last_update_date))
    if len({release.last_update_date for release in ordered}) != len(ordered):
        raise H10ArchiveError("H.10 archive lineage release dates must be unique")
    events: list[H10ArchiveCorrectionEquivalentEvent] = []
    lineage_rows: list[dict[str, Any]] = []
    prior_release: H10ArchiveRelease | None = None
    prior_records: dict[tuple[str, str, date], Observation] = {}
    for release in ordered:
        if release.available_at != archive_release_available_at(
            release.release_date,
            last_update_date=release.last_update_date,
        ):
            raise H10ArchiveError("H.10 archive lineage availability is invalid")
        current_records: dict[tuple[str, str, date], Observation] = {}
        for record in release.records:
            key = (record.source, record.series_id, record.observed_period_end)
            if key in current_records:
                raise H10ArchiveError(
                    "H.10 archive lineage contains duplicate series-date rows"
                )
            current_records[key] = record
        if not current_records:
            raise H10ArchiveError("H.10 archive lineage page contains no records")

        current_keys = set(current_records)
        prior_keys = set(prior_records)
        overlap_keys = current_keys & prior_keys
        new_keys = current_keys - prior_keys
        material_revision_rows = sum(
            (
                current_records[key].value,
                current_records[key].quality_status,
                current_records[key].units,
            )
            != (
                prior_records[key].value,
                prior_records[key].quality_status,
                prior_records[key].units,
            )
            for key in overlap_keys
        )
        complete_republication = bool(
            prior_release is not None
            and not new_keys
            and current_keys == prior_keys
        )
        declared_correction = release.release_date != release.last_update_date
        prior_gap_days = (
            (release.last_update_date - prior_release.last_update_date).days
            if prior_release is not None
            else None
        )
        if prior_gap_days is not None and prior_gap_days <= 0:
            raise H10ArchiveError("H.10 archive lineage dates are not increasing")
        short_gap = bool(
            prior_gap_days is not None
            and prior_gap_days <= H10_ARCHIVE_SHORT_GAP_AUXILIARY_DAYS
        )
        triggers = tuple(
            component
            for component, matched in (
                ("declared_correction", declared_correction),
                ("material_revision", material_revision_rows > 0),
                ("complete_republication", complete_republication),
            )
            if matched
        )
        lineage_rows.append(
            {
                "event_date": release.last_update_date.isoformat(),
                "release_date": release.release_date.isoformat(),
                "available_at": release.available_at.isoformat(),
                "snapshot_sha256": release.snapshot_sha256,
                "prior_release_date": (
                    prior_release.last_update_date.isoformat()
                    if prior_release is not None
                    else None
                ),
                "prior_release_gap_days": prior_gap_days,
                "short_gap_auxiliary_evidence": short_gap,
                "current_series_date_rows": len(current_keys),
                "overlap_series_date_rows": len(overlap_keys),
                "new_series_date_rows": len(new_keys),
                "material_revision_rows": material_revision_rows,
                "trigger_components": list(triggers),
            }
        )
        if triggers:
            events.append(
                H10ArchiveCorrectionEquivalentEvent(
                    event_date=release.last_update_date,
                    available_at=release.available_at,
                    prior_release_date=(
                        prior_release.last_update_date
                        if prior_release is not None
                        else None
                    ),
                    prior_release_gap_days=prior_gap_days,
                    short_gap_auxiliary_evidence=short_gap,
                    declared_correction=declared_correction,
                    material_revision_rows=material_revision_rows,
                    complete_republication=complete_republication,
                    current_series_date_rows=len(current_keys),
                    overlap_series_date_rows=len(overlap_keys),
                    new_series_date_rows=len(new_keys),
                    trigger_components=triggers,
                )
            )
        prior_release = release
        prior_records = current_records
    encoded = json.dumps(
        {
            "policy": H10_ARCHIVE_CORRECTION_EQUIVALENT_POLICY,
            "components": list(H10_ARCHIVE_CORRECTION_EQUIVALENT_COMPONENTS),
            "short_gap_auxiliary_days": H10_ARCHIVE_SHORT_GAP_AUXILIARY_DAYS,
            "pages": lineage_rows,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return H10ArchiveLineage(
        release_count=len(ordered),
        events=tuple(events),
        lineage_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def merge_h10_archive_releases(
    releases: Sequence[H10ArchiveRelease],
) -> tuple[tuple[Observation, ...], int]:
    """Keep immutable originals and only material later-release corrections."""

    ordered = tuple(sorted(releases, key=lambda item: item.available_at))
    if not ordered or len({item.last_update_date for item in ordered}) != len(ordered):
        raise H10ArchiveError("H.10 archive releases must be non-empty and unique")
    events: list[Observation] = []
    latest: dict[tuple[str, str, date], Observation] = {}
    revision_count = 0
    for release in ordered:
        if release.available_at != archive_release_available_at(
            release.release_date,
            last_update_date=release.last_update_date,
        ):
            raise H10ArchiveError("H.10 archive availability timestamp is invalid")
        for record in release.records:
            key = (record.source, record.series_id, record.observed_period_end)
            prior = latest.get(key)
            semantic = (record.value, record.quality_status, record.units)
            prior_semantic = (
                None
                if prior is None
                else (prior.value, prior.quality_status, prior.units)
            )
            if semantic == prior_semantic:
                continue
            metadata = dict(record.metadata)
            is_revision = prior is not None
            metadata["archive_revision_event"] = is_revision
            metadata["supersedes_available_at"] = (
                prior.available_at.isoformat() if prior is not None else None
            )
            event = replace(record, metadata=metadata)
            events.append(event)
            latest[key] = event
            revision_count += int(is_revision)
    return normalize_revision_sequences(events), revision_count


class H10ArchiveClient:
    """Fetch the official index and every selected immutable release page."""

    def __init__(
        self,
        *,
        transport: ByteTransport | None = None,
        index_url: str = H10_ARCHIVE_INDEX_URL,
        timeout_seconds: float = 30.0,
        retry_attempts: int = 3,
        retry_delay_seconds: float = 0.5,
        request_interval_seconds: float = 0.05,
        sleeper: Callable[[float], None] = _sleep,
    ) -> None:
        self.transport = transport or UrllibByteTransport()
        self.index_url = index_url
        self.timeout_seconds = float(timeout_seconds)
        self.retry_attempts = int(retry_attempts)
        self.retry_delay_seconds = float(retry_delay_seconds)
        self.request_interval_seconds = float(request_interval_seconds)
        self.sleeper = sleeper
        self._network_request_made = False
        if not 1.0 <= self.timeout_seconds <= 300.0:
            raise ValueError("H.10 archive timeout must be in [1, 300]")
        if self.retry_attempts != retry_attempts or not 1 <= self.retry_attempts <= 5:
            raise ValueError("H.10 archive retry_attempts must be in [1, 5]")
        if not 0.0 <= self.retry_delay_seconds <= 60.0:
            raise ValueError("H.10 archive retry delay is outside accepted bounds")
        if not 0.0 <= self.request_interval_seconds <= 5.0:
            raise ValueError("H.10 archive request interval is outside accepted bounds")

    @staticmethod
    def _retryable(exc: BaseException) -> bool:
        if isinstance(exc, HTTPError):
            return exc.code == 429 or 500 <= exc.code <= 599
        return isinstance(exc, (URLError, TimeoutError))

    def _get_bytes(self, url: str):
        if self._network_request_made and self.request_interval_seconds:
            self.sleeper(self.request_interval_seconds)
        self._network_request_made = True
        for attempt in range(1, self.retry_attempts + 1):
            try:
                return self.transport.get_bytes(
                    url,
                    timeout=self.timeout_seconds,
                )
            except Exception as exc:
                if not self._retryable(exc) or attempt >= self.retry_attempts:
                    raise
                self.sleeper(self.retry_delay_seconds * attempt)
        raise AssertionError("H.10 archive retry loop is unreachable")

    def collect(
        self,
        *,
        requested_at: datetime,
        start_date: date | None = None,
        end_date: date | None = None,
        known_release_dates: Sequence[date] = (),
        cached_releases: Mapping[date, H10ArchiveRelease] | None = None,
        on_release: Callable[[H10ArchiveRelease], None] | None = None,
    ) -> H10ArchiveCollection:
        requested = _aware_utc(requested_at, field_name="requested_at")
        if start_date is not None and end_date is not None and start_date > end_date:
            raise ValueError("H.10 archive start_date must not follow end_date")
        known = tuple(known_release_dates)
        if len(set(known)) != len(known) or any(
            not isinstance(value, date) for value in known
        ):
            raise ValueError("known H.10 archive release dates are invalid")
        cached = dict(cached_releases or {})
        if any(
            not isinstance(key, date)
            or not isinstance(value, H10ArchiveRelease)
            or value.last_update_date != key
            for key, value in cached.items()
        ):
            raise ValueError("cached H.10 archive releases are invalid")
        index_response = self._get_bytes(self.index_url)
        if index_response.retrieved_at < requested:
            raise H10ArchiveError("H.10 archive index retrieval precedes request")
        discovered = tuple(
            link
            for link in discover_h10_archive_releases(
                index_response.body,
                index_url=self.index_url,
            )
            if (start_date is None or link.release_date >= start_date)
            and (end_date is None or link.release_date <= end_date)
        )
        if not discovered:
            raise H10ArchiveError("H.10 archive date range contains no releases")
        discovered_dates = tuple(link.release_date for link in discovered)
        if set(known).difference(discovered_dates):
            raise H10ArchiveError(
                "H.10 archive index omitted a previously accepted release date"
            )
        links = tuple(
            link for link in discovered if link.release_date not in set(known)
        )
        releases: list[H10ArchiveRelease] = []
        for link in links:
            cached_release = cached.get(link.release_date)
            if cached_release is not None:
                if cached_release.source_url != link.url:
                    raise H10ArchiveError(
                        "cached H.10 archive release URL is inconsistent"
                    )
                release = cached_release
            else:
                response = self._get_bytes(link.url)
                release = parse_h10_archive_release(
                    response.body,
                    source_url=link.url,
                    retrieved_at=response.retrieved_at,
                    expected_release_date=link.release_date,
                )
                if on_release is not None:
                    on_release(release)
            releases.append(release)
        if releases:
            records, revision_count = merge_h10_archive_releases(releases)
        else:
            records, revision_count = (), 0
        lineage = (
            detect_h10_archive_correction_equivalents(releases)
            if tuple(release.last_update_date for release in releases)
            == discovered_dates
            else None
        )
        collection_body = "\n".join(
            [
                hashlib.sha256(index_response.body).hexdigest(),
                *(release.snapshot_sha256 for release in releases),
            ]
        ).encode("ascii")
        return H10ArchiveCollection(
            releases=tuple(releases),
            records=records,
            requested_at=requested,
            retrieved_at=max(
                (index_response.retrieved_at, *(release.retrieved_at for release in releases))
            ),
            index_sha256=hashlib.sha256(index_response.body).hexdigest(),
            collection_sha256=hashlib.sha256(collection_body).hexdigest(),
            revision_event_count=revision_count,
            discovered_release_dates=discovered_dates,
            lineage=lineage,
        )


__all__ = [
    "H10_ARCHIVE_AVAILABILITY_BASIS",
    "H10_ARCHIVE_CORRECTION_AVAILABILITY_BASIS",
    "H10_ARCHIVE_CORRECTION_EQUIVALENT_COMPONENTS",
    "H10_ARCHIVE_CORRECTION_EQUIVALENT_POLICY",
    "H10_ARCHIVE_EVALUATION_START",
    "H10_ARCHIVE_EVALUATION_START_RATIONALE",
    "H10_ARCHIVE_INDEX_URL",
    "H10_ARCHIVE_REVISION_POLICY",
    "H10_ARCHIVE_SHORT_GAP_AUXILIARY_DAYS",
    "H10_ARCHIVE_NORMAL_AVAILABILITY_BASIS",
    "H10_FIRST_SEEN_AVAILABILITY_BASIS",
    "H10ArchiveClient",
    "H10ArchiveCollection",
    "H10ArchiveCorrectionEquivalentEvent",
    "H10ArchiveError",
    "H10ArchiveLineage",
    "H10ArchiveRelease",
    "H10ArchiveReleaseLink",
    "archive_release_available_at",
    "discover_h10_archive_releases",
    "detect_h10_archive_correction_equivalents",
    "merge_h10_archive_releases",
    "parse_h10_archive_release",
]
