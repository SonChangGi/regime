from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Mapping

from regime_lab.data import (
    AlfredClient,
    AlfredConfig,
    CollectionResult,
    HealthStatus,
    HttpStatusError,
    Observation,
    RetryPolicy,
    merge_collection_results,
    provenance_safe_result,
)


UTC = timezone.utc
NOW = datetime(2024, 2, 10, 12, tzinfo=UTC)
CUTOFF = datetime(2024, 2, 2, 23, tzinfo=UTC)


class NeverTransport:
    def __init__(self) -> None:
        self.calls = 0

    def get_json(self, *_: object, **__: object) -> Mapping[str, Any]:
        self.calls += 1
        raise AssertionError("transport must not be called")


class PaginatedRetryTransport:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.failed_once = False

    def get_json(
        self,
        _url: str,
        params: Mapping[str, Any],
        *,
        timeout: float,
    ) -> Mapping[str, Any]:
        del timeout
        self.calls.append(dict(params))
        if not self.failed_once:
            self.failed_once = True
            raise HttpStatusError(503, "temporary")
        rows = [
            {
                "date": "2023-11-30",
                "UNRATE_20240131": "100.0",
            },
            {
                "date": "2023-12-31",
                "UNRATE_20240131": "101.0",
            },
            {
                "date": "2024-01-31",
                "UNRATE_20240131": "102.0",
            },
        ]
        offset = int(params["offset"])
        limit = int(params["limit"])
        return {
            "count": len(rows),
            "offset": offset,
            "limit": limit,
            "observations": rows[offset : offset + limit],
        }


def test_alfred_rights_gate_is_fail_closed_before_transport() -> None:
    transport = NeverTransport()
    client = AlfredClient(
        AlfredConfig(api_key="not-used", rights_acknowledged=False),
        transport=transport,
        clock=lambda: NOW,
    )

    result = client.fetch_observations(
        ["UNRATE"],
        vintage_dates=[date(2024, 1, 31)],
        realtime_start=date(2024, 1, 31),
        realtime_end=date(2024, 1, 31),
        cutoff=CUTOFF,
    )

    assert result.health is HealthStatus.LICENSE_BLOCKED
    assert transport.calls == 0
    assert "ALFRED_ML_RIGHTS_ACK" in result.issues[0]


def test_alfred_retries_and_paginates_with_explicit_pit_boundaries() -> None:
    transport = PaginatedRetryTransport()
    client = AlfredClient(
        AlfredConfig(
            api_key="secret-key",
            rights_acknowledged=True,
            page_size=2,
            vintage_batch_size=1,
        ),
        transport=transport,
        retry=RetryPolicy(max_attempts=2, backoff_seconds=0),
        sleeper=lambda _: None,
        clock=lambda: NOW,
    )

    result = client.fetch_observations(
        ["UNRATE"],
        vintage_dates=[date(2024, 1, 31)],
        realtime_start=date(2024, 1, 1),
        realtime_end=date(2024, 1, 31),
        cutoff=CUTOFF,
        observation_start=date(2023, 1, 1),
    )

    assert result.health is HealthStatus.OK
    assert result.requests_made == 2
    assert result.attempts == 3
    assert len(result.records) == 3
    assert [call["offset"] for call in transport.calls] == [0, 0, 2]
    assert {record.vintage_date for record in result.records} == {date(2024, 1, 31)}
    assert result.records[0].metadata["provider_field"] == "UNRATE_20240131"
    for call in transport.calls:
        assert call["output_type"] == 2
        assert call["vintage_dates"] == "2024-01-31"
        # Official FRED API contract: vintage_dates is supplied instead of a
        # realtime_start/realtime_end request period.
        assert "realtime_start" not in call
        assert "realtime_end" not in call
    assert all(record.available_at <= CUTOFF for record in result.records)


class SecretFailureTransport:
    def get_json(
        self,
        _url: str,
        params: Mapping[str, Any],
        *,
        timeout: float,
    ) -> Mapping[str, Any]:
        del timeout
        raise ValueError(f"invalid api_key={params['api_key']}")


def test_provider_error_never_echoes_api_key() -> None:
    key = "a-very-secret-key"
    client = AlfredClient(
        AlfredConfig(api_key=key, rights_acknowledged=True),
        transport=SecretFailureTransport(),
        retry=RetryPolicy(max_attempts=1),
        clock=lambda: NOW,
    )
    result = client.fetch_observations(
        ["UNRATE"],
        vintage_dates=[date(2024, 1, 31)],
        realtime_start=date(2024, 1, 31),
        realtime_end=date(2024, 1, 31),
        cutoff=CUTOFF,
    )

    rendered = " ".join(result.issues)
    assert result.health is HealthStatus.DEGRADED
    assert key not in rendered
    assert "[REDACTED]" in rendered
    assert key not in repr(client.config)


class RealtimeEventTransport:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def get_json(
        self,
        _url: str,
        params: Mapping[str, Any],
        *,
        timeout: float,
    ) -> Mapping[str, Any]:
        del timeout
        self.calls.append(dict(params))
        rows = [
            {
                "realtime_start": "2024-01-05",
                "realtime_end": "2024-01-11",
                "date": "2023-12-31",
                "value": "3.6",
            },
            {
                "realtime_start": "2024-01-12",
                "realtime_end": "9999-12-31",
                "date": "2023-12-31",
                "value": "3.7",
            },
            {
                "realtime_start": "2024-01-20",
                "realtime_end": "9999-12-31",
                "date": "2024-01-19",
                "value": "3.8",
            },
        ]
        offset = int(params["offset"])
        limit = int(params["limit"])
        return {
            "count": len(rows),
            "offset": offset,
            "limit": limit,
            "observations": rows[offset : offset + limit],
        }


def test_realtime_event_api_uses_one_range_per_series_without_frequency_aggregation() -> None:
    transport = RealtimeEventTransport()
    spacing: list[float] = []
    client = AlfredClient(
        AlfredConfig(
            api_key="secret-key",
            rights_acknowledged=True,
            page_size=2,
            request_spacing_seconds=0.25,
        ),
        transport=transport,
        retry=RetryPolicy(max_attempts=1),
        sleeper=spacing.append,
        clock=lambda: NOW,
    )

    result = client.fetch_realtime_observations(
        ["UNRATE"],
        realtime_start=date(2024, 1, 1),
        realtime_end=date(2024, 1, 31),
        cutoff=CUTOFF,
        observation_start=date(2023, 1, 1),
        observation_end=date(2024, 1, 31),
    )

    assert result.health is HealthStatus.OK
    assert result.requests_made == 2
    assert [call["offset"] for call in transport.calls] == [0, 2]
    assert spacing == [0.25]
    for call in transport.calls:
        assert call["output_type"] == 1
        assert call["realtime_start"] == "2024-01-01"
        assert call["realtime_end"] == "2024-01-31"
        assert "vintage_dates" not in call
        assert "frequency" not in call
        assert "aggregation_method" not in call
    first_period = [
        record
        for record in result.records
        if record.observed_period_end == date(2023, 12, 31)
    ]
    assert [record.value for record in first_period] == [3.6, 3.7]
    assert [record.revision_seq for record in first_period] == [0, 1]
    assert all(record.metadata["provider_output_type"] == 1 for record in result.records)
    assert all(record.metadata["provider_frequency"] == "raw" for record in result.records)
    assert all(record.available_at <= CUTOFF for record in result.records)


class RevisionEventTransport:
    """Official output_type=3 JSON cross-tab shape, split over two pages."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def get_json(
        self,
        _url: str,
        params: Mapping[str, Any],
        *,
        timeout: float,
    ) -> Mapping[str, Any]:
        del timeout
        self.calls.append(dict(params))
        if "output_type" not in params:
            return {
                "count": 2,
                "offset": 0,
                "limit": 10_000,
                "vintage_dates": ["2024-01-31", "2024-02-01"],
            }
        rows = [
            {
                "date": "2023-11-30",
                "UNRATE_20240131": "3.6",
                # UNRATE_20240201 is absent: unchanged, not malformed.
            },
            {
                "date": "2023-12-31",
                "UNRATE_20240131": "3.7",
                "UNRATE_20240201": "3.8",
            },
            {
                "date": "2024-01-31",
                # A present period is a real revision-to-missing event.
                "UNRATE_20240201": ".",
            },
        ]
        offset = int(params["offset"])
        limit = int(params["limit"])
        return {
            "realtime_start": "2024-01-31",
            "realtime_end": "2024-02-01",
            "observation_start": "2023-01-01",
            "observation_end": "2024-01-31",
            "units": "lin",
            "output_type": 3,
            "file_type": "json",
            "count": len(rows),
            "offset": offset,
            "limit": limit,
            "observations": rows[offset : offset + limit],
        }


def test_revision_event_api_parses_official_output_type_3_wide_shape() -> None:
    transport = RevisionEventTransport()
    client = AlfredClient(
        AlfredConfig(
            api_key="secret-key",
            rights_acknowledged=True,
            page_size=2,
            vintage_batch_size=2,
        ),
        transport=transport,
        retry=RetryPolicy(max_attempts=1),
        clock=lambda: NOW,
    )

    result = client.fetch_revision_events(
        ["UNRATE"],
        vintage_dates=[
            date(2024, 1, 30),
            date(2024, 1, 31),
            date(2024, 2, 1),
        ],
        cutoff=CUTOFF,
        observation_start=date(2023, 1, 1),
        observation_end=date(2024, 1, 31),
    )

    assert result.health is HealthStatus.OK
    assert result.requests_made == 3
    assert result.attempts == 3
    discovery_calls = [call for call in transport.calls if "output_type" not in call]
    revision_calls = [call for call in transport.calls if call.get("output_type") == 3]
    assert len(discovery_calls) == 1
    assert discovery_calls[0]["realtime_start"] == "2024-01-30"
    assert discovery_calls[0]["realtime_end"] == "2024-02-01"
    assert [call["offset"] for call in revision_calls] == [0, 2]
    assert len(result.records) == 4
    assert {
        (record.observed_period_end, record.vintage_date, record.value)
        for record in result.records
    } == {
        (date(2023, 11, 30), date(2024, 1, 31), 3.6),
        (date(2023, 12, 31), date(2024, 1, 31), 3.7),
        (date(2023, 12, 31), date(2024, 2, 1), 3.8),
        (date(2024, 1, 31), date(2024, 2, 1), None),
    }
    revised_period = [
        record
        for record in result.records
        if record.observed_period_end == date(2023, 12, 31)
    ]
    assert [record.revision_seq for record in revised_period] == [0, 1]
    assert all(
        record.metadata["provider_output_type"] == 3
        for record in result.records
    )
    assert all(
        record.metadata["provider_frequency"] == "raw"
        for record in result.records
    )
    for call in revision_calls:
        assert call["output_type"] == 3
        assert call["vintage_dates"] == "2024-01-31,2024-02-01"
        assert "realtime_start" not in call
        assert "realtime_end" not in call
        assert "frequency" not in call
        assert "aggregation_method" not in call


class VintageDiscoveryRetryTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.failed_once = False

    def get_json(
        self,
        url: str,
        params: Mapping[str, Any],
        *,
        timeout: float,
    ) -> Mapping[str, Any]:
        del timeout
        self.calls.append((url, dict(params)))
        if "output_type" not in params:
            if not self.failed_once:
                self.failed_once = True
                raise HttpStatusError(503, "temporary vintage-date failure")
            valid_dates = ["2024-01-31", "2024-02-01"]
            offset = int(params["offset"])
            limit = int(params["limit"])
            return {
                "count": len(valid_dates),
                "offset": offset,
                "limit": limit,
                "vintage_dates": valid_dates[offset : offset + limit],
            }
        return {
            "output_type": 3,
            "count": 1,
            "offset": 0,
            "limit": 100_000,
            "observations": [
                {
                    "date": "2023-12-31",
                    "UNRATE_20240131": "3.7",
                    "UNRATE_20240201": "3.8",
                }
            ],
        }


def test_revision_event_api_retries_and_paginates_vintage_discovery() -> None:
    transport = VintageDiscoveryRetryTransport()
    client = AlfredClient(
        AlfredConfig(
            api_key="secret-key",
            rights_acknowledged=True,
            vintage_page_size=1,
        ),
        transport=transport,
        retry=RetryPolicy(max_attempts=2, backoff_seconds=0),
        sleeper=lambda _: None,
        clock=lambda: NOW,
    )

    result = client.fetch_revision_events(
        ["UNRATE"],
        vintage_dates=[
            date(2024, 1, 30),
            date(2024, 1, 31),
            date(2024, 2, 1),
        ],
        cutoff=CUTOFF,
    )

    assert result.health is HealthStatus.OK
    assert result.requests_made == 3
    assert result.attempts == 4
    discovery_calls = [call for call in transport.calls if "output_type" not in call[1]]
    revision_calls = [call for call in transport.calls if call[1].get("output_type") == 3]
    assert len(discovery_calls) == 3  # includes the retried first page
    assert [call[1]["offset"] for call in discovery_calls] == [0, 0, 1]
    assert all(call[0].endswith("/series/vintagedates") for call in discovery_calls)
    assert len(revision_calls) == 1
    assert revision_calls[0][0].endswith("/series/observations")
    assert revision_calls[0][1]["vintage_dates"] == "2024-01-31,2024-02-01"
    assert {record.vintage_date for record in result.records} == {
        date(2024, 1, 31),
        date(2024, 2, 1),
    }


def test_vintage_discovery_quota_failure_is_redacted_and_stops_type_3() -> None:
    secret = "vintage-discovery-secret"

    class QuotaTransport:
        def __init__(self) -> None:
            self.calls = 0

        def get_json(
            self,
            _url: str,
            params: Mapping[str, Any],
            *,
            timeout: float,
        ) -> Mapping[str, Any]:
            del timeout
            self.calls += 1
            if "output_type" in params:
                raise AssertionError("type-3 must not run after discovery failure")
            raise HttpStatusError(429, f"quota api_key={params['api_key']}")

    transport = QuotaTransport()
    client = AlfredClient(
        AlfredConfig(api_key=secret, rights_acknowledged=True),
        transport=transport,
        retry=RetryPolicy(max_attempts=1),
        clock=lambda: NOW,
    )

    result = client.fetch_revision_events(
        ["UNRATE"],
        vintage_dates=[date(2024, 2, 1)],
        cutoff=CUTOFF,
    )

    rendered = " ".join(result.issues)
    assert result.health is HealthStatus.QUOTA_EXHAUSTED
    assert result.requests_made == 0
    assert result.attempts == 1
    assert transport.calls == 1
    assert secret not in rendered
    assert "[REDACTED]" in rendered


def test_revision_event_api_accepts_an_empty_successful_delta() -> None:
    class EmptyRevisionTransport:
        def get_json(
            self,
            _url: str,
            params: Mapping[str, Any],
            *,
            timeout: float,
        ) -> Mapping[str, Any]:
            del timeout
            if "output_type" not in params:
                return {
                    "count": 1,
                    "offset": 0,
                    "limit": 10_000,
                    "vintage_dates": ["2024-02-01"],
                }
            return {
                "count": 0,
                "offset": 0,
                "limit": 100_000,
                "observations": [],
            }

    client = AlfredClient(
        AlfredConfig(api_key="secret-key", rights_acknowledged=True),
        transport=EmptyRevisionTransport(),
        retry=RetryPolicy(max_attempts=1),
        clock=lambda: NOW,
    )

    result = client.fetch_revision_events(
        ["UNRATE"],
        vintage_dates=[date(2024, 2, 1)],
        cutoff=CUTOFF,
    )

    assert result.health is HealthStatus.OK
    assert result.records == ()
    assert result.requests_made == 2


def test_revision_event_api_accepts_wide_shape_without_output_type_envelope() -> None:
    class WideTransport:
        def get_json(
            self,
            _url: str,
            params: Mapping[str, Any],
            *,
            timeout: float,
        ) -> Mapping[str, Any]:
            del timeout
            if "output_type" not in params:
                return {
                    "count": 1,
                    "offset": 0,
                    "limit": 10_000,
                    "vintage_dates": ["2024-02-01"],
                }
            return {
                "count": 1,
                "offset": 0,
                "limit": 100_000,
                "observations": [
                    {"date": "2024-01-31", "UNRATE_20240201": "3.8"}
                ],
            }

    client = AlfredClient(
        AlfredConfig(api_key="secret-key", rights_acknowledged=True),
        transport=WideTransport(),
        retry=RetryPolicy(max_attempts=1),
        clock=lambda: NOW,
    )

    result = client.fetch_revision_events(
        ["UNRATE"],
        vintage_dates=[date(2024, 2, 1)],
        cutoff=CUTOFF,
    )

    assert result.health is HealthStatus.OK
    assert len(result.records) == 1
    assert result.records[0].value == 3.8


def test_revision_event_api_rejects_explicit_wrong_output_type() -> None:
    class WrongOutputTransport:
        def get_json(
            self,
            _url: str,
            params: Mapping[str, Any],
            *,
            timeout: float,
        ) -> Mapping[str, Any]:
            del timeout
            if "output_type" not in params:
                return {
                    "count": 1,
                    "offset": 0,
                    "limit": 10_000,
                    "vintage_dates": ["2024-02-01"],
                }
            return {
                "output_type": 2,
                "count": 1,
                "offset": 0,
                "limit": 100_000,
                "observations": [
                    {"date": "2024-01-31", "UNRATE_20240201": "3.8"}
                ],
            }

    client = AlfredClient(
        AlfredConfig(api_key="secret-key", rights_acknowledged=True),
        transport=WrongOutputTransport(),
        retry=RetryPolicy(max_attempts=1),
        clock=lambda: NOW,
    )

    result = client.fetch_revision_events(
        ["UNRATE"],
        vintage_dates=[date(2024, 2, 1)],
        cutoff=CUTOFF,
    )

    assert result.health is HealthStatus.SCHEMA_CHANGED
    assert result.records == ()


def test_revision_event_api_rejects_non_crosstab_rows_as_schema_changed() -> None:
    class RowShapedTransport:
        def get_json(
            self,
            _url: str,
            params: Mapping[str, Any],
            *,
            timeout: float,
        ) -> Mapping[str, Any]:
            del timeout
            if "output_type" not in params:
                return {
                    "count": 1,
                    "offset": 0,
                    "limit": 10_000,
                    "vintage_dates": ["2024-02-01"],
                }
            return {
                "output_type": 3,
                "count": 1,
                "offset": 0,
                "limit": 100_000,
                "observations": [
                    {
                        "date": "2024-01-31",
                        "realtime_start": "2024-02-01",
                        "value": "3.8",
                    }
                ],
            }

    client = AlfredClient(
        AlfredConfig(api_key="secret-key", rights_acknowledged=True),
        transport=RowShapedTransport(),
        retry=RetryPolicy(max_attempts=1),
        clock=lambda: NOW,
    )

    result = client.fetch_revision_events(
        ["UNRATE"],
        vintage_dates=[date(2024, 2, 1)],
        cutoff=CUTOFF,
    )

    assert result.health is HealthStatus.SCHEMA_CHANGED
    assert result.records == ()
    assert any("fields were not recognized" in issue for issue in result.issues)


def test_chunk_merge_reassigns_one_global_deterministic_revision_sequence() -> None:
    period_end = date(2023, 12, 31)

    def revision(vintage: date, value: float, local_seq: int) -> Observation:
        available = datetime(vintage.year, vintage.month, vintage.day, 23, tzinfo=UTC)
        return Observation(
            source="alfred",
            series_id="DGS10",
            observed_period_end=period_end,
            value=value,
            released_at=available,
            available_at=available,
            vintage_date=vintage,
            retrieved_at=NOW,
            revision_seq=local_seq,
            raw_sha256=f"hash-{vintage.isoformat()}",
        )

    first_chunk = CollectionResult(
        records=(
            revision(date(2010, 1, 5), 1.0, 0),
            revision(date(2013, 12, 20), 2.0, 1),
        ),
        requests_made=2,
        attempts=2,
    )
    second_chunk = CollectionResult(
        records=(
            revision(date(2014, 1, 3), 3.0, 0),
            revision(date(2017, 12, 22), 4.0, 1),
        ),
        requests_made=3,
        attempts=4,
    )

    forward = merge_collection_results(
        (first_chunk, second_chunk),
        normalize_revisions=True,
    )
    reversed_chunks = merge_collection_results(
        (second_chunk, first_chunk),
        normalize_revisions=True,
    )

    expected = {
        date(2010, 1, 5): 0,
        date(2013, 12, 20): 1,
        date(2014, 1, 3): 2,
        date(2017, 12, 22): 3,
    }
    assert {item.vintage_date: item.revision_seq for item in forward.records} == expected
    assert {
        item.vintage_date: item.revision_seq for item in reversed_chunks.records
    } == expected
    assert forward.requests_made == 5
    assert forward.attempts == 6
    assert len({item.revision_seq for item in forward.records}) == len(forward.records)


def test_non_ok_alfred_chunk_is_provenance_only_before_snapshot_write() -> None:
    partial = CollectionResult(
        records=(
            Observation(
                source="alfred",
                series_id="DGS10",
                observed_period_end=date(2024, 1, 1),
                value=4.0,
                released_at=datetime(2024, 1, 2, tzinfo=UTC),
                available_at=datetime(2024, 1, 2, tzinfo=UTC),
                vintage_date=date(2024, 1, 2),
                retrieved_at=NOW,
                raw_sha256="partial",
            ),
        ),
        health=HealthStatus.DEGRADED,
        issues=("later page failed",),
        requests_made=1,
        attempts=3,
    )

    safe = provenance_safe_result(partial)

    assert safe.records == ()
    assert safe.health is HealthStatus.DEGRADED
    assert safe.issues == partial.issues
    assert safe.requests_made == 1
    assert safe.attempts == 3
