from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import pytest

from regime_lab.data import ByteResponse
from regime_lab.data.ofr_fsi import (
    DEFAULT_STRUCTURAL_V6_CONFIG,
    OFR_FSI_SOURCE,
    OFR_FSI_URL,
    OFRFSIClient,
    OFRFSIConfig,
    OFRFSIError,
    OFRFSISchemaError,
    OFRFSITimestampError,
    load_ofr_fsi_contract,
    parse_ofr_fsi_csv,
)


UTC = timezone.utc
ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "ofr_fsi.csv"
FIRST_SEEN = datetime(2026, 3, 16, 20, 0, tzinfo=UTC)
RETRIEVED = datetime(2026, 3, 16, 20, 0, 2, tzinfo=UTC)


class FakeTransport:
    def __init__(self, response: ByteResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, float]] = []

    def get_bytes(self, url: str, *, timeout: float) -> ByteResponse:
        self.calls.append((url, timeout))
        return self.response


def _payload() -> bytes:
    return FIXTURE.read_bytes()


def test_contract_cross_checks_official_aggregate_and_private_boundaries() -> None:
    contract = load_ofr_fsi_contract()

    assert contract.source_url == OFR_FSI_URL
    assert contract.expected_header == (
        "Date",
        "OFR FSI",
        "Credit",
        "Equity valuation",
        "Safe assets",
        "Funding",
        "Volatility",
        "United States",
        "Other advanced economies",
        "Emerging markets",
    )
    assert contract.observation_lag_business_days == 2
    assert contract.business_day_calendar == "USFederalHolidayCalendar"
    assert {item.measurement_role for item in contract.series} == {
        "published_aggregate",
        "published_category_contribution",
        "published_region_contribution",
    }
    assert len(contract.contract_sha256) == 64


def test_contract_rejects_attempt_to_collect_an_underlying_input(tmp_path: Path) -> None:
    document = json.loads(DEFAULT_STRUCTURAL_V6_CONFIG.read_text(encoding="utf-8"))
    document["core_sources"]["ofr_fsi"]["series"].append(
        {
            "output_id": "OFR_FSI_UNDERLYING_SWAP_SPREAD",
            "csv_column": "Underlying swap spread",
            "domain": "financial_conditions",
            "frequency": "daily",
            "required": False,
        }
    )
    path = tmp_path / "unsafe-v6.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(OFRFSIError, match="not a published aggregate/contribution"):
        load_ofr_fsi_contract(path)


def test_strict_parser_preserves_distinct_clocks_hash_and_dst_vintage() -> None:
    payload = _payload()
    parsed = parse_ofr_fsi_csv(
        payload,
        load_ofr_fsi_contract(),
        first_seen_at=FIRST_SEEN,
        retrieved_at=RETRIEVED,
    )

    assert parsed.row_count == 3
    assert len(parsed.records) == 27
    assert parsed.response_sha256 == sha256(payload).hexdigest()
    assert parsed.last_period.isoformat() == "2026-03-12"
    aggregate = next(
        record
        for record in parsed.records
        if record.series_id == "OFR_FSI"
        and record.observed_period_end.isoformat() == "2026-03-12"
    )
    assert aggregate.source == OFR_FSI_SOURCE
    assert aggregate.source_released_at is None
    assert aggregate.released_at is None
    assert aggregate.provider_first_seen_at == FIRST_SEEN
    assert aggregate.system_retrieved_at == RETRIEVED
    assert aggregate.available_at == FIRST_SEEN
    assert aggregate.operating_available_at == FIRST_SEEN
    assert aggregate.vintage_date.isoformat() == "2026-03-16"
    assert aggregate.raw_sha256 == parsed.response_sha256
    assert aggregate.metadata["source_release_time_known"] is False
    assert aggregate.metadata["underlying_proprietary_inputs_included"] is False
    assert aggregate.metadata["raw_payload_publication"] is False


def test_parser_rejects_naive_or_reversed_collection_clocks() -> None:
    contract = load_ofr_fsi_contract()
    with pytest.raises(ValueError, match="timezone-aware"):
        parse_ofr_fsi_csv(
            _payload(),
            contract,
            first_seen_at=datetime(2026, 3, 16, 16, 0),
            retrieved_at=RETRIEVED,
        )
    with pytest.raises(OFRFSITimestampError, match="must not follow"):
        parse_ofr_fsi_csv(
            _payload(),
            contract,
            first_seen_at=RETRIEVED,
            retrieved_at=FIRST_SEEN,
        )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda text: text.replace(
                "Emerging markets\n",
                "Emerging markets,Underlying proprietary input\n",
                1,
            ),
            "header changed",
        ),
        (
            lambda text: text.replace(
                "2026-03-12,-0.1000",
                "2026-03-11,-0.1000",
            ),
            "unique and strictly increasing",
        ),
        (
            lambda text: text.replace("-0.1000", "NaN", 1),
            "must be finite",
        ),
    ],
)
def test_parser_fails_closed_on_schema_drift(mutator, message: str) -> None:
    changed = mutator(_payload().decode("utf-8")).encode("utf-8")
    with pytest.raises(OFRFSISchemaError, match=message):
        parse_ofr_fsi_csv(
            changed,
            load_ofr_fsi_contract(),
            first_seen_at=FIRST_SEEN,
            retrieved_at=RETRIEVED,
        )


def test_parser_rejects_observation_that_is_too_recent_for_declared_lag() -> None:
    changed = _payload().replace(b"2026-03-12", b"2026-03-13")

    with pytest.raises(OFRFSITimestampError, match="two-business-day lag"):
        parse_ofr_fsi_csv(
            changed,
            load_ofr_fsi_contract(),
            first_seen_at=FIRST_SEEN,
            retrieved_at=RETRIEVED,
        )


@pytest.mark.parametrize(
    ("first_seen", "accepted_dates", "too_recent_date"),
    [
        (
            datetime(2026, 1, 20, 21, 0, tzinfo=UTC),
            ("2026-01-13", "2026-01-14", "2026-01-15"),
            "2026-01-16",
        ),
        (
            datetime(2026, 11, 30, 21, 0, tzinfo=UTC),
            ("2026-11-23", "2026-11-24", "2026-11-25"),
            "2026-11-27",
        ),
    ],
)
def test_two_business_day_lag_uses_us_federal_holidays(
    first_seen: datetime,
    accepted_dates: tuple[str, str, str],
    too_recent_date: str,
) -> None:
    text = _payload().decode("utf-8")
    for original, replacement in zip(
        ("2026-03-10", "2026-03-11", "2026-03-12"),
        accepted_dates,
        strict=True,
    ):
        text = text.replace(original, replacement)
    accepted = parse_ofr_fsi_csv(
        text.encode("utf-8"),
        load_ofr_fsi_contract(),
        first_seen_at=first_seen,
        retrieved_at=first_seen,
    )
    assert accepted.last_period.isoformat() == accepted_dates[-1]
    assert all(
        record.metadata["observation_lag_calendar"]
        == "USFederalHolidayCalendar"
        for record in accepted.records
    )

    too_recent = text.replace(accepted_dates[-1], too_recent_date).encode("utf-8")
    with pytest.raises(OFRFSITimestampError, match="two-business-day lag"):
        parse_ofr_fsi_csv(
            too_recent,
            load_ofr_fsi_contract(),
            first_seen_at=first_seen,
            retrieved_at=first_seen,
        )


def test_client_requests_only_official_https_file_and_rejects_html() -> None:
    transport = FakeTransport(
        ByteResponse(
            body=_payload(),
            headers={"Content-Type": "text/csv; charset=utf-8"},
            retrieved_at=RETRIEVED,
        )
    )
    contract = load_ofr_fsi_contract()
    result = OFRFSIClient(OFRFSIConfig(contract), transport=transport).collect()

    assert result.row_count == 3
    assert transport.calls == [(OFR_FSI_URL, contract.timeout_seconds)]

    html = FakeTransport(
        ByteResponse(
            body=b"<html>blocked</html>",
            headers={"Content-Type": "text/html"},
            retrieved_at=RETRIEVED,
        )
    )
    with pytest.raises(OFRFSISchemaError, match="content type"):
        OFRFSIClient(OFRFSIConfig(contract), transport=html).collect()
