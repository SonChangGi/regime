from __future__ import annotations

from datetime import date, datetime, timezone
import json
from urllib.error import URLError

import pytest

from regime_lab.data.h10 import ByteResponse
from regime_lab.data.h10_archive import (
    H10_ARCHIVE_INDEX_URL,
    H10ArchiveClient,
    H10ArchiveError,
    discover_h10_archive_releases,
    merge_h10_archive_releases,
    parse_h10_archive_release,
)


UTC = timezone.utc
RETRIEVED = datetime(2026, 8, 23, 1, tzinfo=UTC)


_BILATERAL_ROWS = (
    ("AUSTRALIA", "DOLLAR", 1.5),
    ("BRAZIL", "REAL", 5.0),
    ("CANADA", "DOLLAR", 1.3),
    ("CHINA", "YUAN", 7.1),
    ("*EMU MEMBERS", "EURO", 1.1),
    ("JAPAN", "YEN", 145.0),
    ("MEXICO", "PESO", 18.0),
    ("SWITZERLAND", "FRANC", 0.9),
    ("UNITED KINGDOM", "POUND", 1.3),
)


def _index_payload(*dates: str) -> bytes:
    by_month: dict[str, list[str]] = {}
    for value in dates:
        by_month.setdefault(value[:6], []).append(value)
    by_year: dict[str, list[dict[str, object]]] = {}
    for month_value, month_dates in by_month.items():
        month_name = datetime.strptime(month_value, "%Y%m").strftime("%B")
        by_year.setdefault(month_value[:4], []).append(
            {
                "MonthName": month_name,
                "MonthValue": month_value,
                "Dates": sorted(month_dates),
            }
        )
    document = [
        {"yearValue": year, "Months": months}
        for year, months in sorted(by_year.items(), reverse=True)
    ]
    return json.dumps(document).encode("utf-8")


def _release_page(
    *,
    release_date: str,
    last_update: str,
    correction_delta: float = 0.0,
    omit_country: str | None = None,
) -> bytes:
    released = datetime.strptime(release_date, "%B %d, %Y").date()
    weekdays = [released.fromordinal(released.toordinal() - offset) for offset in (7, 6, 5, 4, 3)]
    headers = "".join(f"<th>{value.strftime('%b %d')}</th>" for value in weekdays)
    rows: list[str] = []
    for country, currency, base in _BILATERAL_ROWS:
        if country == omit_country:
            continue
        values = [base + position * 0.001 for position in range(5)]
        if country == "*EMU MEMBERS":
            values[-1] += correction_delta
        cells = "".join(f"<td>{value:.6f}</td>" for value in values)
        rows.append(f"<tr><th>{country}</th><td>{currency}</td>{cells}</tr>")
    index_rows = []
    for number, label, base in (
        (1, "BROAD", 101.0),
        (2, "AFE", 102.0),
        (3, "EME", 103.0),
    ):
        cells = "".join(
            f"<td>{base + position * 0.01:.4f}</td>" for position in range(5)
        )
        index_rows.append(
            f"<tr><th>{number} {label}</th><td>JAN06=100</td>{cells}</tr>"
        )
    goods_only = "".join("<td>99.0</td>" for _ in range(5))
    html = f"""
    <html><body>
      <div class="dates">Release Date: {release_date}</div>
      <div class="lastUpdate">Last Update: {last_update}</div>
      <table>
        <tr><th>COUNTRY</th><th>CURRENCY</th>{headers}</tr>
        {''.join(rows)}
      </table>
      <table>
        <tr><th>INDEX</th><th>UNIT</th>{headers}</tr>
        {''.join(index_rows)}
        <tr><th>BROAD - goods only</th><td>JAN06=100</td>{goods_only}</tr>
      </table>
    </body></html>
    """
    return html.encode("utf-8")


def _normal_release():
    return parse_h10_archive_release(
        _release_page(
            release_date="August 05, 2024",
            last_update="August 05, 2024",
        ),
        source_url="https://www.federalreserve.gov/releases/h10/20240805/",
        retrieved_at=RETRIEVED,
        expected_release_date=date(2024, 8, 5),
    )


def _correction_release():
    return parse_h10_archive_release(
        _release_page(
            release_date="August 05, 2024",
            last_update="August 07, 2024",
            correction_delta=0.01,
        ),
        source_url="https://www.federalreserve.gov/releases/h10/20240807/",
        retrieved_at=RETRIEVED,
        expected_release_date=date(2024, 8, 7),
    )


def test_release_dates_json_is_strict_and_keeps_correction_extra_dates() -> None:
    links = discover_h10_archive_releases(
        _index_payload("20240805", "20240807", "20241015")
    )

    assert [link.release_date for link in links] == [
        date(2024, 8, 5),
        date(2024, 8, 7),
        date(2024, 10, 15),
    ]
    assert links[1].url.endswith("/releases/h10/20240807/")

    changed = json.loads(_index_payload("20240805"))
    changed[0]["unexpected"] = True
    with pytest.raises(H10ArchiveError, match="year schema changed"):
        discover_h10_archive_releases(json.dumps(changed).encode())
    with pytest.raises(H10ArchiveError, match="releaseDates.json"):
        discover_h10_archive_releases(
            _index_payload("20240805"),
            index_url="https://www.federalreserve.gov/releases/H10/",
        )


def test_normal_and_correction_pages_preserve_pit_precision_and_units() -> None:
    normal = _normal_release()
    correction = _correction_release()

    assert normal.available_at == datetime(2024, 8, 5, 20, 15, tzinfo=UTC)
    assert normal.availability_basis == "archived_release_date_16_15_ET"
    assert correction.available_at == datetime(2024, 8, 8, 4, 0, tzinfo=UTC)
    assert correction.availability_basis == "date_only_conservative_next_day"
    assert len(normal.records) == 60
    by_code = {str(row.metadata["fx_code"]): row for row in normal.records}
    assert by_code["EUR"].metadata["unit_raw"] == "EURO"
    assert by_code["BRD"].metadata["unit_raw"] == "JAN06=100"
    assert {row.metadata["fx_code"] for row in normal.records} == {
        "BRD", "AFE", "EME", "AUD", "BRL", "CAD", "CNY", "EUR", "JPY", "MXN", "CHF", "GBP"
    }


def test_release_page_date_and_required_series_mismatches_fail_closed() -> None:
    with pytest.raises(H10ArchiveError, match="last update does not match"):
        parse_h10_archive_release(
            _release_page(
                release_date="August 05, 2024",
                last_update="August 07, 2024",
            ),
            source_url="https://www.federalreserve.gov/releases/h10/20240808/",
            retrieved_at=RETRIEVED,
        )
    with pytest.raises(H10ArchiveError, match="missing required series: EUR"):
        parse_h10_archive_release(
            _release_page(
                release_date="August 05, 2024",
                last_update="August 05, 2024",
                omit_country="*EMU MEMBERS",
            ),
            source_url="https://www.federalreserve.gov/releases/h10/20240805/",
            retrieved_at=RETRIEVED,
        )


def test_correction_is_a_later_vintage_only_when_the_value_changes() -> None:
    records, revision_count = merge_h10_archive_releases(
        (_normal_release(), _correction_release())
    )

    assert len(records) == 61
    assert revision_count == 1
    revised = [row for row in records if row.revision_seq == 1]
    assert len(revised) == 1
    assert revised[0].metadata["archive_revision_event"] is True
    assert revised[0].available_at == datetime(2024, 8, 8, 4, tzinfo=UTC)


class _Transport:
    def __init__(self, outcomes: dict[str, list[object]]) -> None:
        self.outcomes = outcomes
        self.calls: list[str] = []

    def get_bytes(self, url: str, *, timeout: float) -> ByteResponse:
        assert timeout == 10.0
        self.calls.append(url)
        outcome = self.outcomes[url].pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return ByteResponse(body=outcome, headers={}, retrieved_at=RETRIEVED)


def test_client_retries_transient_index_and_resumes_from_cached_page() -> None:
    page_url = "https://www.federalreserve.gov/releases/h10/20240807/"
    transport = _Transport(
        {
            H10_ARCHIVE_INDEX_URL: [
                URLError("temporary"),
                _index_payload("20240805", "20240807"),
            ],
        }
    )
    sleeps: list[float] = []
    client = H10ArchiveClient(
        transport=transport,
        timeout_seconds=10,
        request_interval_seconds=0,
        retry_delay_seconds=0.25,
        sleeper=sleeps.append,
    )

    collection = client.collect(
        requested_at=datetime(2026, 8, 23, 0, tzinfo=UTC),
        start_date=date(2024, 8, 1),
        end_date=date(2024, 8, 31),
        known_release_dates=(date(2024, 8, 5),),
        cached_releases={date(2024, 8, 7): _correction_release()},
    )

    assert transport.calls == [H10_ARCHIVE_INDEX_URL, H10_ARCHIVE_INDEX_URL]
    assert sleeps == [0.25]
    assert collection.discovered_release_dates == (
        date(2024, 8, 5),
        date(2024, 8, 7),
    )
    assert [release.source_url for release in collection.releases] == [page_url]
