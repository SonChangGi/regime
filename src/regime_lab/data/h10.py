"""Strict Federal Reserve Board H.10 XML snapshot parser for the v5 draft.

The release-page ZIP is a full-history SDMX document that mixes business-day,
monthly, and annual series.  This draft accepts only predeclared business-day
(``FREQ=9``) series, preserves the provider's raw metadata, and stamps every
observation with the collector's first-seen time.  It never interprets the
timezone-free SDMX ``Prepared`` value as a release timestamp.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from enum import StrEnum
import hashlib
from io import BytesIO
import math
from typing import Protocol
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET
from zipfile import BadZipFile, ZipFile
from zoneinfo import ZoneInfo

from .contracts import HealthStatus, Observation


UTC = timezone.utc
EASTERN = ZoneInfo("America/New_York")
H10_XML_URL = (
    "https://www.federalreserve.gov/releases/h10/data/FRB_h10_xml.zip"
)
BUSINESS_DAY_FREQUENCY = "9"
REQUIRED_ZIP_MEMBERS = frozenset(
    {"H10_data.xml", "H10_H10.xsd", "H10_struct.xml", "frb_common.xsd"}
)
MAX_ZIP_BYTES = 8_000_000
MAX_XML_BYTES = 64_000_000

_MESSAGE_NS = "http://www.SDMX.org/resources/SDMXML/schemas/v1_0/message"
_COMMON_NS = "http://www.SDMX.org/resources/SDMXML/schemas/v1_0/common"
_FRB_NS = "http://www.federalreserve.gov/structure/compact/common"
_KF_NS = "http://www.federalreserve.gov/structure/compact/H10_H10"

_MESSAGE_GROUP = f"{{{_MESSAGE_NS}}}MessageGroup"
_HEADER = f"{{{_MESSAGE_NS}}}Header"
_MESSAGE_ID = f"{{{_MESSAGE_NS}}}ID"
_MESSAGE_TEST = f"{{{_MESSAGE_NS}}}Test"
_MESSAGE_NAME = f"{{{_MESSAGE_NS}}}Name"
_MESSAGE_PREPARED = f"{{{_MESSAGE_NS}}}Prepared"
_MESSAGE_SENDER = f"{{{_MESSAGE_NS}}}Sender"
_DATASET = f"{{{_FRB_NS}}}DataSet"
_SERIES = f"{{{_KF_NS}}}Series"
_OBS = f"{{{_FRB_NS}}}Obs"
_ANNOTATION = f"{{{_COMMON_NS}}}Annotation"
_ANNOTATION_TYPE = f"{{{_COMMON_NS}}}AnnotationType"
_ANNOTATION_TEXT = f"{{{_COMMON_NS}}}AnnotationText"

_VALID_OBSERVATION_STATUSES = frozenset({"A", "NA", "ND", "NC"})


class H10SchemaError(ValueError):
    """The upstream ZIP or XML violated the frozen parser contract."""


class H10TimestampError(ValueError):
    """H.10 collection timestamps form an impossible ordering."""


class QuoteConvention(StrEnum):
    USD_INDEX = "usd_index"
    FOREIGN_PER_USD = "foreign_per_usd"
    USD_PER_FOREIGN = "usd_per_foreign"


@dataclass(frozen=True, slots=True)
class SeriesSpec:
    fx_code: str
    series_name: str
    currency_code: str
    quote_convention: QuoteConvention
    usd_strength_sign: int

    def __post_init__(self) -> None:
        if not self.fx_code or not self.series_name or not self.currency_code:
            raise ValueError("H.10 series identifiers must not be empty")
        quote = QuoteConvention(self.quote_convention)
        expected_sign = (
            -1 if quote is QuoteConvention.USD_PER_FOREIGN else 1
        )
        if self.usd_strength_sign != expected_sign:
            raise ValueError("usd_strength_sign conflicts with quote convention")
        object.__setattr__(self, "quote_convention", quote)


FIXED_BILATERAL_PANEL: tuple[str, ...] = (
    "EUR",
    "JPY",
    "GBP",
    "CHF",
    "CAD",
    "AUD",
    "CNY",
    "MXN",
    "BRL",
)

SERIES_CATALOG: Mapping[str, SeriesSpec] = {
    "BRD": SeriesSpec(
        "BRD", "JRXWTFB_N.B", "NA", QuoteConvention.USD_INDEX, 1
    ),
    "AFE": SeriesSpec(
        "AFE", "JRXWTFN_N.B", "NA", QuoteConvention.USD_INDEX, 1
    ),
    "EME": SeriesSpec(
        "EME", "JRXWTFO_N.B", "NA", QuoteConvention.USD_INDEX, 1
    ),
    "AUD": SeriesSpec(
        "AUD", "RXI$US_N.B.AL", "AUD", QuoteConvention.USD_PER_FOREIGN, -1
    ),
    "CAD": SeriesSpec(
        "CAD", "RXI_N.B.CA", "CAD", QuoteConvention.FOREIGN_PER_USD, 1
    ),
    "CHF": SeriesSpec(
        "CHF", "RXI_N.B.SZ", "CHF", QuoteConvention.FOREIGN_PER_USD, 1
    ),
    "CNY": SeriesSpec(
        "CNY", "RXI_N.B.CH", "CNY", QuoteConvention.FOREIGN_PER_USD, 1
    ),
    "EUR": SeriesSpec(
        "EUR", "RXI$US_N.B.EU", "EUR", QuoteConvention.USD_PER_FOREIGN, -1
    ),
    "GBP": SeriesSpec(
        "GBP", "RXI$US_N.B.UK", "GBP", QuoteConvention.USD_PER_FOREIGN, -1
    ),
    "JPY": SeriesSpec(
        "JPY", "RXI_N.B.JA", "JPY", QuoteConvention.FOREIGN_PER_USD, 1
    ),
    "MXN": SeriesSpec(
        "MXN", "RXI_N.B.MX", "MXN", QuoteConvention.FOREIGN_PER_USD, 1
    ),
    "BRL": SeriesSpec(
        "BRL", "RXI_N.B.BZ", "BRL", QuoteConvention.FOREIGN_PER_USD, 1
    ),
    # Kept for parser/metadata tests.  It is intentionally absent from the
    # fixed bilateral feature panel above.
    "KRW": SeriesSpec(
        "KRW", "RXI_N.B.KO", "KRW", QuoteConvention.FOREIGN_PER_USD, 1
    ),
}

DEFAULT_ALLOWED_FX: tuple[str, ...] = (
    "BRD",
    "AFE",
    "EME",
    *FIXED_BILATERAL_PANEL,
)


def _utc(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def _text(parent: ET.Element, tag: str, *, field_name: str) -> str:
    child = parent.find(tag)
    value = "" if child is None or child.text is None else child.text.strip()
    if not value:
        raise H10SchemaError(f"missing {field_name}")
    return value


def _canonical_series_id(fx_code: str) -> str:
    return f"H10|FX={fx_code}|FREQ={BUSINESS_DAY_FREQUENCY}"


def _header_value(headers: Mapping[str, str], name: str) -> str | None:
    target = name.lower()
    for key, value in headers.items():
        if str(key).lower() == target:
            stripped = str(value).strip()
            return stripped or None
    return None


def _http_datetime(headers: Mapping[str, str], name: str) -> datetime | None:
    raw = _header_value(headers, name)
    if raw is None:
        return None
    try:
        parsed = parsedate_to_datetime(raw)
    except (TypeError, ValueError) as exc:
        raise H10SchemaError(f"invalid HTTP {name} header") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise H10SchemaError(f"HTTP {name} header is timezone-naive")
    return parsed.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class ByteResponse:
    body: bytes
    headers: Mapping[str, str]
    retrieved_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.body, bytes):
            raise TypeError("byte transport body must be bytes")
        object.__setattr__(self, "headers", dict(self.headers))
        object.__setattr__(
            self,
            "retrieved_at",
            _utc(self.retrieved_at, field_name="retrieved_at"),
        )


class ByteTransport(Protocol):
    def get_bytes(self, url: str, *, timeout: float) -> ByteResponse: ...


class UrllibByteTransport:
    """Small injectable binary transport; tests need no network access."""

    def __init__(
        self,
        *,
        user_agent: str = "regime-lab/1",
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.user_agent = user_agent
        self.clock = clock or (lambda: datetime.now(UTC))

    def get_bytes(self, url: str, *, timeout: float) -> ByteResponse:
        request = Request(url, headers={"User-Agent": self.user_agent})
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            body = response.read()
            headers = {key: value for key, value in response.headers.items()}
        return ByteResponse(
            body=body,
            headers=headers,
            retrieved_at=self.clock(),
        )


@dataclass(frozen=True, slots=True)
class H10Config:
    source_url: str = H10_XML_URL
    timeout_seconds: float = 30.0
    allowed_fx: tuple[str, ...] = DEFAULT_ALLOWED_FX
    max_zip_bytes: int = MAX_ZIP_BYTES
    max_xml_bytes: int = MAX_XML_BYTES

    def __post_init__(self) -> None:
        allowed = tuple(str(code).strip().upper() for code in self.allowed_fx)
        if not allowed or any(not code for code in allowed):
            raise ValueError("allowed_fx must not be empty")
        if len(set(allowed)) != len(allowed):
            raise ValueError("allowed_fx must not contain duplicates")
        unknown = sorted(set(allowed).difference(SERIES_CATALOG))
        if unknown:
            raise ValueError(f"unknown H.10 allowlist entries: {unknown}")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_zip_bytes < 1 or self.max_xml_bytes < 1:
            raise ValueError("payload limits must be positive")
        object.__setattr__(self, "allowed_fx", allowed)


@dataclass(frozen=True, slots=True)
class ReleaseMetadata:
    source_url: str
    message_id: str
    message_name: str
    sender_id: str
    message_prepared_raw: str
    source_object_modified_at: datetime | None
    release_date_et: date | None
    first_seen_at: datetime
    retrieved_at: datetime
    etag: str | None
    snapshot_sha256: str


@dataclass(frozen=True, slots=True)
class SeriesMetadata:
    series_id: str
    fx_code: str
    frequency_code: str
    series_name: str
    currency_code: str
    unit_raw: str
    unit_multiplier_raw: str
    quote_convention: QuoteConvention
    usd_strength_sign: int
    short_description: str
    long_description: str


@dataclass(frozen=True, slots=True)
class H10ParseResult:
    release: ReleaseMetadata
    series: tuple[SeriesMetadata, ...]
    records: tuple[Observation, ...]


@dataclass(slots=True)
class _SeriesContext:
    spec: SeriesSpec
    series_name: str
    currency_code: str
    unit_raw: str
    unit_multiplier_raw: str
    short_description: str = ""
    long_description: str = ""
    dates: set[date] = field(default_factory=set)


def _validate_series_attributes(
    attributes: Mapping[str, str],
    spec: SeriesSpec,
) -> _SeriesContext:
    required = ("FX", "FREQ", "SERIES_NAME", "CURRENCY", "UNIT", "UNIT_MULT")
    missing = [name for name in required if not attributes.get(name)]
    if missing:
        raise H10SchemaError(
            f"H.10 {spec.fx_code} series is missing attributes: {missing}"
        )
    if attributes["FREQ"] != BUSINESS_DAY_FREQUENCY:
        raise H10SchemaError("selected H.10 series is not business-day frequency")
    if attributes["FX"] != spec.fx_code:
        raise H10SchemaError("H.10 FX code changed")
    if attributes["SERIES_NAME"] != spec.series_name:
        raise H10SchemaError(
            f"H.10 series name changed for {spec.fx_code}: "
            f"{attributes['SERIES_NAME']}"
        )
    if attributes["CURRENCY"] != spec.currency_code:
        raise H10SchemaError(
            f"H.10 currency metadata changed for {spec.fx_code}: "
            f"{attributes['CURRENCY']}"
        )
    try:
        multiplier = float(attributes["UNIT_MULT"])
    except ValueError as exc:
        raise H10SchemaError("invalid H.10 unit multiplier") from exc
    if not math.isclose(multiplier, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise H10SchemaError(
            f"unsupported daily unit multiplier for {spec.fx_code}"
        )
    return _SeriesContext(
        spec=spec,
        series_name=attributes["SERIES_NAME"],
        currency_code=attributes["CURRENCY"],
        unit_raw=attributes["UNIT"],
        unit_multiplier_raw=attributes["UNIT_MULT"],
    )


def _parse_observation(
    element: ET.Element,
    *,
    context: _SeriesContext,
    source_url: str,
    header: Mapping[str, str],
    snapshot_sha256: str,
    source_object_modified_at: datetime | None,
    first_seen_at: datetime,
    retrieved_at: datetime,
) -> Observation:
    raw_date = element.attrib.get("TIME_PERIOD", "")
    try:
        observed = date.fromisoformat(raw_date)
    except ValueError as exc:
        raise H10SchemaError(
            f"invalid H.10 observation date for {context.spec.fx_code}"
        ) from exc
    if observed in context.dates:
        raise H10SchemaError(
            f"duplicate H.10 observation date for {context.spec.fx_code}: {observed}"
        )
    context.dates.add(observed)

    status = element.attrib.get("OBS_STATUS", "")
    if status not in _VALID_OBSERVATION_STATUSES:
        raise H10SchemaError(
            f"unsupported H.10 observation status for {context.spec.fx_code}"
        )
    raw_value_token = element.attrib.get("OBS_VALUE")
    value: float | None = None
    if status == "A":
        if raw_value_token is None:
            raise H10SchemaError("normal H.10 observation has no value")
        try:
            value = float(raw_value_token)
        except ValueError as exc:
            raise H10SchemaError("invalid normal H.10 observation value") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise H10SchemaError("normal H.10 observation must be positive and finite")

    series_id = _canonical_series_id(context.spec.fx_code)
    raw_identity = "|".join(
        (
            snapshot_sha256,
            series_id,
            observed.isoformat(),
            status,
            "" if raw_value_token is None else raw_value_token,
        )
    )
    raw_sha256 = hashlib.sha256(raw_identity.encode("utf-8")).hexdigest()
    metadata = {
        "release_id": "H10",
        "dataset_id": "H10",
        "fx_code": context.spec.fx_code,
        "frequency_code": BUSINESS_DAY_FREQUENCY,
        "series_name": context.series_name,
        "currency_code": context.currency_code,
        "unit_raw": context.unit_raw,
        "unit_multiplier_raw": context.unit_multiplier_raw,
        "quote_convention": context.spec.quote_convention.value,
        "usd_strength_sign": context.spec.usd_strength_sign,
        "obs_status": status,
        "raw_value_token": raw_value_token,
        "short_description": context.short_description,
        "long_description": context.long_description,
        "message_prepared_raw": header["prepared"],
        "snapshot_sha256": snapshot_sha256,
        "first_seen_at_utc": first_seen_at.isoformat(),
        "source_object_modified_at_utc": (
            source_object_modified_at.isoformat()
            if source_object_modified_at is not None
            else None
        ),
        "source_url": source_url,
    }
    return Observation(
        source="frb_h10",
        series_id=series_id,
        observed_period_end=observed,
        value=value,
        released_at=source_object_modified_at,
        available_at=first_seen_at,
        vintage_date=first_seen_at.date(),
        retrieved_at=retrieved_at,
        units=context.unit_raw,
        license_class="federal_reserve_board_public_domain_citation_requested",
        quality_status=(
            HealthStatus.OK if status == "A" else HealthStatus.UNAVAILABLE
        ),
        raw_sha256=raw_sha256,
        metadata=metadata,
    )


def parse_h10_zip(
    payload: bytes,
    *,
    source_url: str = H10_XML_URL,
    headers: Mapping[str, str] | None = None,
    first_seen_at: datetime,
    retrieved_at: datetime,
    allowed_fx: Sequence[str] = DEFAULT_ALLOWED_FX,
    max_zip_bytes: int = MAX_ZIP_BYTES,
    max_xml_bytes: int = MAX_XML_BYTES,
) -> H10ParseResult:
    """Parse one release-page ZIP without assigning retrospective availability."""

    if not isinstance(payload, bytes):
        raise TypeError("H.10 payload must be bytes")
    if not payload or len(payload) > max_zip_bytes:
        raise H10SchemaError("H.10 ZIP size is outside the accepted bounds")
    first_seen = _utc(first_seen_at, field_name="first_seen_at")
    retrieved = _utc(retrieved_at, field_name="retrieved_at")
    if retrieved < first_seen:
        raise H10TimestampError(
            "retrieved_at must not precede first_seen_at"
        )

    selected_codes = tuple(str(code).strip().upper() for code in allowed_fx)
    if not selected_codes or len(set(selected_codes)) != len(selected_codes):
        raise ValueError("allowed_fx must be non-empty and unique")
    unknown = sorted(set(selected_codes).difference(SERIES_CATALOG))
    if unknown:
        raise ValueError(f"unknown H.10 allowlist entries: {unknown}")
    selected_specs = {code: SERIES_CATALOG[code] for code in selected_codes}

    normalized_headers = dict(headers or {})
    source_modified = _http_datetime(normalized_headers, "Last-Modified")
    if source_modified is not None and source_modified > first_seen:
        raise H10SchemaError("HTTP Last-Modified is after first_seen_at")
    etag = _header_value(normalized_headers, "ETag")
    snapshot_sha256 = hashlib.sha256(payload).hexdigest()

    try:
        archive = ZipFile(BytesIO(payload))
    except BadZipFile as exc:
        raise H10SchemaError("invalid H.10 ZIP") from exc
    with archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise H10SchemaError("H.10 ZIP contains duplicate member names")
        missing_members = sorted(REQUIRED_ZIP_MEMBERS.difference(names))
        if missing_members:
            raise H10SchemaError(
                f"H.10 ZIP is missing required members: {missing_members}"
            )
        data_info = archive.getinfo("H10_data.xml")
        if data_info.file_size < 1 or data_info.file_size > max_xml_bytes:
            raise H10SchemaError("H10_data.xml size is outside accepted bounds")

        records: list[Observation] = []
        series_metadata: list[SeriesMetadata] = []
        found: set[str] = set()
        current: _SeriesContext | None = None
        header: dict[str, str] | None = None
        dataset_seen = False
        root_seen = False

        try:
            with archive.open("H10_data.xml") as xml_file:
                for event, element in ET.iterparse(
                    xml_file,
                    events=("start", "end"),
                ):
                    if event == "start":
                        if not root_seen:
                            if element.tag != _MESSAGE_GROUP:
                                raise H10SchemaError("unexpected H.10 XML root")
                            root_seen = True
                        if element.tag == _DATASET:
                            if element.attrib.get("id") != "H10":
                                raise H10SchemaError("unexpected H.10 dataset id")
                            dataset_seen = True
                        elif element.tag == _SERIES:
                            frequency = element.attrib.get("FREQ")
                            fx_code = element.attrib.get("FX", "")
                            if frequency == BUSINESS_DAY_FREQUENCY and fx_code in selected_specs:
                                if fx_code in found:
                                    raise H10SchemaError(
                                        f"duplicate selected H.10 series: {fx_code}"
                                    )
                                current = _validate_series_attributes(
                                    element.attrib,
                                    selected_specs[fx_code],
                                )
                            else:
                                current = None
                        continue

                    if element.tag == _HEADER:
                        message_id = _text(element, _MESSAGE_ID, field_name="message ID")
                        test = _text(element, _MESSAGE_TEST, field_name="test flag")
                        message_name = _text(
                            element,
                            _MESSAGE_NAME,
                            field_name="message name",
                        )
                        prepared = _text(
                            element,
                            _MESSAGE_PREPARED,
                            field_name="prepared metadata",
                        )
                        sender = element.find(_MESSAGE_SENDER)
                        sender_id = "" if sender is None else sender.attrib.get("id", "")
                        if message_id != "H10" or test.lower() != "false" or sender_id != "FRB":
                            raise H10SchemaError("unexpected H.10 message header")
                        header = {
                            "id": message_id,
                            "name": message_name,
                            "prepared": prepared,
                            "sender_id": sender_id,
                        }
                        element.clear()
                    elif element.tag == _ANNOTATION and current is not None:
                        kind = element.findtext(_ANNOTATION_TYPE, default="").strip()
                        text = element.findtext(_ANNOTATION_TEXT, default="").strip()
                        if kind == "Short Description":
                            current.short_description = text
                        elif kind == "Long Description":
                            current.long_description = text
                        element.clear()
                    elif element.tag == _OBS:
                        if current is not None:
                            if header is None:
                                raise H10SchemaError("H.10 observations precede header")
                            records.append(
                                _parse_observation(
                                    element,
                                    context=current,
                                    source_url=source_url,
                                    header=header,
                                    snapshot_sha256=snapshot_sha256,
                                    source_object_modified_at=source_modified,
                                    first_seen_at=first_seen,
                                    retrieved_at=retrieved,
                                )
                            )
                        element.clear()
                    elif element.tag == _SERIES:
                        if current is not None:
                            code = current.spec.fx_code
                            if not current.dates:
                                raise H10SchemaError(
                                    f"selected H.10 series has no observations: {code}"
                                )
                            series_metadata.append(
                                SeriesMetadata(
                                    series_id=_canonical_series_id(code),
                                    fx_code=code,
                                    frequency_code=BUSINESS_DAY_FREQUENCY,
                                    series_name=current.series_name,
                                    currency_code=current.currency_code,
                                    unit_raw=current.unit_raw,
                                    unit_multiplier_raw=current.unit_multiplier_raw,
                                    quote_convention=current.spec.quote_convention,
                                    usd_strength_sign=current.spec.usd_strength_sign,
                                    short_description=current.short_description,
                                    long_description=current.long_description,
                                )
                            )
                            found.add(code)
                        current = None
                        element.clear()
        except ET.ParseError as exc:
            raise H10SchemaError("invalid H10_data.xml") from exc

    if header is None or not dataset_seen:
        raise H10SchemaError("H.10 header or dataset is missing")
    missing_series = sorted(set(selected_codes).difference(found))
    if missing_series:
        raise H10SchemaError(
            f"allowlisted H.10 series are missing: {missing_series}"
        )

    release_date_et = (
        source_modified.astimezone(EASTERN).date()
        if source_modified is not None
        else None
    )
    release = ReleaseMetadata(
        source_url=source_url,
        message_id=header["id"],
        message_name=header["name"],
        sender_id=header["sender_id"],
        message_prepared_raw=header["prepared"],
        source_object_modified_at=source_modified,
        release_date_et=release_date_et,
        first_seen_at=first_seen,
        retrieved_at=retrieved,
        etag=etag,
        snapshot_sha256=snapshot_sha256,
    )
    return H10ParseResult(
        release=release,
        series=tuple(series_metadata),
        records=tuple(records),
    )


class H10Client:
    """Fetch and parse one immutable-in-memory H.10 snapshot."""

    def __init__(
        self,
        config: H10Config | None = None,
        *,
        transport: ByteTransport | None = None,
    ) -> None:
        self.config = config or H10Config()
        self.transport = transport or UrllibByteTransport()

    def collect(self, *, first_seen_at: datetime | None = None) -> H10ParseResult:
        response = self.transport.get_bytes(
            self.config.source_url,
            timeout=self.config.timeout_seconds,
        )
        first_seen = response.retrieved_at if first_seen_at is None else first_seen_at
        return parse_h10_zip(
            response.body,
            source_url=self.config.source_url,
            headers=response.headers,
            first_seen_at=first_seen,
            retrieved_at=response.retrieved_at,
            allowed_fx=self.config.allowed_fx,
            max_zip_bytes=self.config.max_zip_bytes,
            max_xml_bytes=self.config.max_xml_bytes,
        )
