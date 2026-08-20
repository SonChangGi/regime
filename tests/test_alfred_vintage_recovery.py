from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import sqlite3
from typing import Any, Mapping

import pytest

import regime_lab.collection as collection_module
from regime_lab.collection import validate_collection_for_training
from regime_lab.data import (
    AlfredClient,
    AlfredConfig,
    CollectionResult,
    HealthStatus,
    HttpStatusError,
    Observation,
    RetryPolicy,
    SnapshotMode,
    SnapshotProvenance,
    SQLiteSnapshotStore,
)


UTC = timezone.utc
TARGET = datetime(2024, 1, 12, 21, tzinfo=UTC)
PREVIOUS = datetime(2024, 1, 5, 21, tzinfo=UTC)
RUN_AT = datetime(2024, 1, 13, 12, tzinfo=UTC)
FAILED_SERIES = ("INDPRO", "HOUST", "PCEPI", "FEDFUNDS", "GDPC1")
ALL_SERIES = (
    "DGS3MO",
    "DGS1",
    "DGS2",
    "DGS5",
    "DGS7",
    "DGS10",
    "DGS20",
    "DGS30",
    "DFII10",
    "T10Y2Y",
    "T10YIE",
    "DTWEXBGS",
    "ICSA",
    "CCSA",
    "NFCI",
    "ANFCI",
    "NFCIRISK",
    "NFCICREDIT",
    "NFCILEVERAGE",
    "NFCINONFINLEVERAGE",
    "STLFSI4",
    "WALCL",
    "TOTBKCR",
    "TOTCI",
    "DPSACBW027SBOG",
    "H8B3094NCBA",
    "UNRATE",
    "PAYEMS",
    "RSAFS",
    "CPIAUCSL",
    *FAILED_SERIES,
)
SUCCESSFUL_SERIES = tuple(item for item in ALL_SERIES if item not in FAILED_SERIES)


def _client(transport: object) -> AlfredClient:
    return AlfredClient(
        AlfredConfig(
            api_key="secret-key",
            rights_acknowledged=True,
            request_spacing_seconds=0,
        ),
        transport=transport,  # type: ignore[arg-type]
        retry=RetryPolicy(max_attempts=1, backoff_seconds=0),
        sleeper=lambda _seconds: None,
        clock=lambda: datetime(2024, 2, 10, 12, tzinfo=UTC),
    )


def test_empty_narrow_vintage_window_500_requires_successful_wider_discovery() -> None:
    candidate_start = date(2024, 2, 1)
    candidate_end = date(2024, 2, 8)

    class EmptyWindowTransport:
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
            call = dict(params)
            self.calls.append(call)
            if "output_type" in call:
                raise AssertionError("no type-3 call is valid for an empty candidate window")
            if date.fromisoformat(str(call["realtime_start"])) >= candidate_start:
                raise HttpStatusError(500, "empty narrow vintage window")
            return {
                "count": 2,
                "offset": 0,
                "limit": 10_000,
                "vintage_dates": ["2024-01-05", "2024-01-19"],
            }

    transport = EmptyWindowTransport()
    result = _client(transport).fetch_revision_events(
        ["INDPRO"],
        vintage_dates=tuple(
            candidate_start + timedelta(days=offset)
            for offset in range((candidate_end - candidate_start).days + 1)
        ),
        cutoff=datetime(2024, 2, 9, 23, tzinfo=UTC),
    )

    assert result.health is HealthStatus.OK
    assert result.issues == ()
    assert result.records == ()
    assert result.diagnostics["vintage_discovery_fallback_used"] is True
    assert result.diagnostics["vintage_discovery_mode"] == "wide_fallback"
    assert result.diagnostics["vintage_discovery_fallback_series_count"] == 1
    discovery_calls = [call for call in transport.calls if "output_type" not in call]
    assert any(
        date.fromisoformat(str(call["realtime_start"])) < candidate_start
        for call in discovery_calls
    )
    assert all(call["realtime_end"] == candidate_end.isoformat() for call in discovery_calls)


def test_wider_vintage_discovery_failure_remains_degraded_and_skips_type_3() -> None:
    class FailingTransport:
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
            call = dict(params)
            self.calls.append(call)
            if "output_type" in call:
                raise AssertionError("type-3 must not run without validated discovery")
            raise HttpStatusError(500, "HTTP 500")

    transport = FailingTransport()
    result = _client(transport).fetch_revision_events(
        ["GDPC1"],
        vintage_dates=[date(2024, 2, 1), date(2024, 2, 2)],
        cutoff=datetime(2024, 2, 2, 23, tzinfo=UTC),
    )

    assert result.health is HealthStatus.DEGRADED
    assert result.records == ()
    assert any("HTTP 500" in issue for issue in result.issues)
    assert all("output_type" not in call for call in transport.calls)


def test_wider_discovery_passes_only_candidate_intersection_to_type_3() -> None:
    candidate_start = date(2024, 2, 1)
    candidate_end = date(2024, 2, 8)

    class CandidateIntersectionTransport:
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
            call = dict(params)
            self.calls.append(call)
            if "output_type" not in call:
                if date.fromisoformat(str(call["realtime_start"])) >= candidate_start:
                    raise HttpStatusError(500, "empty narrow vintage window")
                return {
                    "count": 3,
                    "offset": 0,
                    "limit": 10_000,
                    "vintage_dates": [
                        "2024-01-05",
                        "2024-01-19",
                        "2024-02-05",
                    ],
                }
            assert call["vintage_dates"] == "2024-02-05"
            return {
                "output_type": 3,
                "count": 1,
                "offset": 0,
                "limit": 100_000,
                "observations": [
                    {"date": "2024-01-01", "PCEPI_20240205": "121.5"}
                ],
            }

    transport = CandidateIntersectionTransport()
    result = _client(transport).fetch_revision_events(
        ["PCEPI"],
        vintage_dates=tuple(
            candidate_start + timedelta(days=offset)
            for offset in range((candidate_end - candidate_start).days + 1)
        ),
        cutoff=datetime(2024, 2, 9, 23, tzinfo=UTC),
    )

    assert result.health is HealthStatus.OK
    assert len(result.records) == 1
    assert result.records[0].vintage_date == date(2024, 2, 5)
    assert result.records[0].value == 121.5


def _seed_provider_state(database: Any) -> str:
    with SQLiteSnapshotStore(database) as store:
        alpha_records = tuple(
            Observation(
                source="alpha_vantage",
                series_id="SPY.adjusted_close",
                observed_period_end=period,
                value=value,
                released_at=cutoff,
                available_at=cutoff,
                vintage_date=period,
                retrieved_at=RUN_AT,
                raw_sha256=f"spy-{period.isoformat()}",
                metadata={"symbol": "SPY", "field": "adjusted_close"},
            )
            for period, cutoff, value in (
                (PREVIOUS.date(), PREVIOUS, 100.0),
                (TARGET.date(), TARGET, 101.0),
            )
        )
        alpha_snapshot = store.write_snapshot(
            alpha_records,
            SnapshotProvenance(
                source="alpha_vantage",
                dataset="weekly_adjusted_etf",
                cutoff=TARGET,
                requested_at=RUN_AT,
                retrieved_at=RUN_AT,
                quality_status=HealthStatus.OK,
                request_params={
                    "symbols": ["SPY"],
                    "fields": ["adjusted_close"],
                    "snapshot_mode": SnapshotMode.FULL.value,
                },
            ),
        )
        for ordinal, series_id in enumerate(ALL_SERIES):
            retrieved = PREVIOUS + timedelta(minutes=ordinal + 1)
            store.write_snapshot(
                (
                    Observation(
                        source="alfred",
                        series_id=series_id,
                        observed_period_end=PREVIOUS.date(),
                        value=float(ordinal + 1),
                        released_at=PREVIOUS,
                        available_at=PREVIOUS,
                        vintage_date=PREVIOUS.date(),
                        retrieved_at=retrieved,
                        raw_sha256=f"{series_id}-base",
                    ),
                ),
                SnapshotProvenance(
                    source="alfred",
                    dataset=series_id,
                    cutoff=PREVIOUS,
                    requested_at=retrieved,
                    retrieved_at=retrieved,
                    quality_status=HealthStatus.OK,
                    request_params={
                        "series_id": series_id,
                        "output_type": 1,
                        "snapshot_mode": SnapshotMode.FULL.value,
                        "observation_start": PREVIOUS.date().isoformat(),
                        "observation_end": PREVIOUS.date().isoformat(),
                        "realtime_start": PREVIOUS.date().isoformat(),
                        "realtime_end": PREVIOUS.date().isoformat(),
                    },
                ),
            )
            quality = (
                HealthStatus.OK
                if series_id in SUCCESSFUL_SERIES
                else HealthStatus.DEGRADED
            )
            store.write_snapshot(
                (),
                SnapshotProvenance(
                    source="alfred",
                    dataset=series_id,
                    cutoff=TARGET,
                    requested_at=RUN_AT + timedelta(minutes=ordinal + 1),
                    retrieved_at=RUN_AT + timedelta(minutes=ordinal + 1),
                    quality_status=quality,
                    request_params={
                        "series_id": series_id,
                        "output_type": 3,
                        "snapshot_mode": SnapshotMode.DELTA.value,
                        "observation_start": PREVIOUS.date().isoformat(),
                        "observation_end": TARGET.date().isoformat(),
                        "vintage_dates": (
                            f"{PREVIOUS.date().isoformat()},{TARGET.date().isoformat()}"
                        ),
                    },
                    issues=(
                        (f"ALFRED {series_id}: vintage discovery HTTP 500",)
                        if quality is HealthStatus.DEGRADED
                        else ()
                    ),
                ),
            )
    return alpha_snapshot


def test_same_cutoff_reuses_30_successes_retries_only_5_failures_and_passes_gate(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "alfred-five-series-recovery.sqlite3"
    alpha_snapshot = _seed_provider_state(database)
    monkeypatch.setenv("FRED_API_KEY", "test-fred-key")
    monkeypatch.setenv("ALFRED_ML_RIGHTS_ACK", "1")
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-alpha-key")

    class ForbiddenAlphaClient:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("same-cutoff Alpha last-good must consume no quota")

    class RecoveringAlfredClient:
        calls: list[str] = []

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def fetch_revision_events(
            self,
            series_ids: Any,
            **_kwargs: object,
        ) -> CollectionResult:
            series_id = str(series_ids[0])
            type(self).calls.append(series_id)
            return CollectionResult(
                records=(),
                health=HealthStatus.OK,
                requests_made=1,
                attempts=2,
                diagnostics={
                    "vintage_discovery_fallback_used": True,
                    "vintage_discovery_mode": "wide_fallback",
                    "vintage_discovery_fallback_series_count": 1,
                },
            )

        def fetch_realtime_observations(
            self,
            *_args: object,
            **_kwargs: object,
        ) -> CollectionResult:
            raise AssertionError("failed series with a full last-good must retry as delta")

    monkeypatch.setattr(collection_module, "AlphaVantageClient", ForbiddenAlphaClient)
    monkeypatch.setattr(collection_module, "AlfredClient", RecoveringAlfredClient)
    config = {
        "alpha_vantage": {
            "base_url": "https://example.invalid/alpha",
            "daily_request_cap": 25,
            "symbols": ["SPY"],
            "fields": ["adjusted_close"],
        },
        "alfred": {
            "base_url": "https://example.invalid/fred",
            "series": [
                {
                    "id": series_id,
                    "frequency": (
                        "quarterly" if series_id == "GDPC1" else "monthly"
                    ),
                    "realtime_start": PREVIOUS.date().isoformat(),
                }
                for series_id in ALL_SERIES
            ],
        },
    }

    collection = collection_module.collect_live_data(
        config,
        database_path=database,
        history_start=PREVIOUS.date(),
        now=RUN_AT,
    )

    assert len(SUCCESSFUL_SERIES) == 30
    assert RecoveringAlfredClient.calls == list(FAILED_SERIES)
    assert collection.overall_health is HealthStatus.OK
    assert collection.issues == ()
    validate_collection_for_training(collection, expected_cutoff=TARGET)
    alpha_source = next(
        source for source in collection.sources if source["id"] == "alpha_vantage"
    )
    alfred_source = next(
        source for source in collection.sources if source["id"] == "alfred"
    )
    assert alpha_source["requests_made"] == 0
    assert alfred_source["requests_made"] == len(FAILED_SERIES)

    with SQLiteSnapshotStore(database) as store:
        current_alpha = store.get_last_good_provenance(
            source="alpha_vantage", dataset="weekly_adjusted_etf"
        )
        assert current_alpha is not None
        assert current_alpha.snapshot_id == alpha_snapshot
        for series_id in ALL_SERIES:
            last_good = store.get_last_good_provenance(
                source="alfred", dataset=series_id
            )
            assert last_good is not None
            assert last_good.cutoff == TARGET
        for series_id in SUCCESSFUL_SERIES:
            target_rows = [
                item
                for item in store.list_provenance(source="alfred")
                if item.dataset == series_id and item.cutoff == TARGET
            ]
            assert len(target_rows) == 1
        for series_id in FAILED_SERIES:
            target_rows = [
                item
                for item in store.list_provenance(source="alfred")
                if item.dataset == series_id and item.cutoff == TARGET
            ]
            assert [item.quality_status for item in target_rows] == [
                HealthStatus.DEGRADED,
                HealthStatus.OK,
            ]
            recovered_params = target_rows[-1].request_params
            assert recovered_params["provider_requests_made"] == 1
            assert recovered_params["provider_attempts"] == 2
            assert recovered_params["vintage_discovery_fallback_used"] is True
            assert recovered_params["vintage_discovery_mode"] == "wide_fallback"
            assert recovered_params[
                "vintage_discovery_fallback_series_count"
            ] == 1

    with sqlite3.connect(database) as connection:
        has_budget_table = connection.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type='table' AND name='request_budget_events'"
        ).fetchone()[0]
        budget_events = (
            connection.execute(
                "SELECT COUNT(*) FROM request_budget_events"
            ).fetchone()[0]
            if has_budget_table
            else 0
        )
    assert budget_events == 0
