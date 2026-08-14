from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3

import pytest

import regime_lab.collection as collection_module
from regime_lab.collection import (
    _alfred_request_params,
    _fetch_alfred_series,
    _realtime_year_chunks,
    _source_row,
    _validate_initial_alpha_baseline,
    _write_result_snapshot,
    last_completed_week_cutoff,
    weekly_cutoffs,
)
from regime_lab.data import (
    CollectionResult,
    DailyRequestBudget,
    HealthStatus,
    Observation,
    SQLiteSnapshotStore,
    SnapshotMode,
    SnapshotProvenance,
)


ALPHA_TEST_FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
)


@pytest.fixture(autouse=True)
def _configured_test_alpha_key(monkeypatch) -> None:
    """Collection tests use fake clients but still exercise live-key gating."""

    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-alpha-key")


def _alpha_test_record(
    symbol: str,
    field: str,
    *,
    period: date,
    cutoff: datetime,
    retrieved_at: datetime,
) -> Observation:
    return Observation(
        source="alpha_vantage",
        series_id=f"{symbol}.{field}",
        observed_period_end=period,
        value=(1_000_000.0 if field == "volume" else 100.0),
        released_at=cutoff,
        available_at=cutoff,
        vintage_date=period,
        retrieved_at=retrieved_at,
        raw_sha256=f"{symbol}-{field}-{period.isoformat()}",
        metadata={"symbol": symbol, "field": field},
    )


def _seed_alpha_test_snapshot(
    database: Path,
    *,
    cutoff: datetime,
    symbols: tuple[str, ...],
    request_symbols: tuple[str, ...] | None = None,
    omitted_series: set[str] | None = None,
) -> None:
    retrieved_at = cutoff + timedelta(hours=1)
    omitted = omitted_series or set()
    records = tuple(
        _alpha_test_record(
            symbol,
            field,
            period=cutoff.date(),
            cutoff=cutoff,
            retrieved_at=retrieved_at,
        )
        for symbol in symbols
        for field in ALPHA_TEST_FIELDS
        if f"{symbol}.{field}" not in omitted
    )
    with SQLiteSnapshotStore(database) as store:
        store.write_snapshot(
            records,
            SnapshotProvenance(
                source="alpha_vantage",
                dataset="weekly_adjusted_etf",
                cutoff=cutoff,
                requested_at=retrieved_at,
                retrieved_at=retrieved_at,
                quality_status=HealthStatus.OK,
                request_params={
                    "symbols": list(request_symbols or symbols),
                    "fields": list(ALPHA_TEST_FIELDS),
                    "snapshot_mode": SnapshotMode.FULL.value,
                },
            ),
        )


def test_default_alpha_config_declares_unique_symbols_and_configured_series() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config" / "series.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    alpha = config["alpha_vantage"]
    symbols = tuple(alpha["symbols"])
    fields = tuple(alpha["fields"])

    assert len(symbols) == 23
    assert len(set(symbols)) == 23
    assert fields == ("open", "high", "low", "close", "adjusted_close", "volume")
    assert len(
        {
            f"{symbol}.{field}"
            for symbol in symbols
            for field in fields
        }
    ) == len(symbols) * len(fields)
    assert len(symbols) <= alpha["daily_request_cap"] - 2
    groups = alpha["symbol_groups"]
    assert set(groups) == {"gics_sector", "broad_size_style", "cross_asset"}
    assert len(groups["gics_sector"]) == 11
    assert set(groups["gics_sector"]).issubset(symbols)
    assert alpha["initial_baseline_minimum_coverage"] == 0.9
    assert alpha["initial_baseline_max_latest_lag_days"] == 10
    assert alpha["history_start_by_symbol"] == {
        "XLC": "2018-06-18",
        "XLRE": "2015-10-07",
    }

    alfred_ids = {item["id"] for item in config["alfred"]["series"]}
    assert {
        "DGS1",
        "DGS5",
        "DGS7",
        "DGS20",
        "DGS30",
        "ANFCI",
        "TOTBKCR",
        "TOTCI",
        "DPSACBW027SBOG",
        "H8B3094NCBA",
        "NFCIRISK",
        "NFCICREDIT",
        "NFCILEVERAGE",
        "NFCINONFINLEVERAGE",
    }.issubset(alfred_ids)


def test_live_collection_passes_configured_ohlcv_fields_in_one_symbol_batch(
    tmp_path,
    monkeypatch,
) -> None:
    cutoff = datetime(2024, 1, 5, 21, tzinfo=timezone.utc)
    retrieved_at = datetime(2024, 1, 6, 12, tzinfo=timezone.utc)
    configured_fields = ("open", "high", "low", "close", "adjusted_close", "volume")

    class FakeAlphaVantageClient:
        calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
        reservations: list[tuple[int, int]] = []

        def __init__(self, *_args, **kwargs) -> None:
            type(self).reservations.append(
                (kwargs["budget"].remaining, kwargs["retry"].max_attempts)
            )

        def fetch_weekly_adjusted(self, symbols, **kwargs) -> CollectionResult:
            fields = tuple(kwargs["fields"])
            clean_symbols = tuple(symbols)
            type(self).calls.append((clean_symbols, fields))
            return CollectionResult(
                records=tuple(
                    Observation(
                        source="alpha_vantage",
                        series_id=f"{symbol}.{field}",
                        observed_period_end=cutoff.date(),
                        value=(1_000_000.0 if field == "volume" else 100.0),
                        released_at=cutoff,
                        available_at=cutoff,
                        vintage_date=cutoff.date(),
                        retrieved_at=retrieved_at,
                        raw_sha256=f"{symbol}-{field}",
                    )
                    for symbol in clean_symbols
                    for field in fields
                ),
                # The provider adapter spends one request per symbol, not per
                # selected field in the same WEEKLY_ADJUSTED response.
                requests_made=len(clean_symbols),
                attempts=len(clean_symbols),
            )

    monkeypatch.setattr(
        collection_module,
        "AlphaVantageClient",
        FakeAlphaVantageClient,
    )
    config = {
        "alpha_vantage": {
            "base_url": "https://example.invalid/alpha",
            "daily_request_cap": 25,
            "symbols": ["SPY", "IWM"],
            "fields": list(configured_fields),
        },
        "alfred": {"base_url": "https://example.invalid/fred", "series": []},
    }

    result = collection_module.collect_live_data(
        config,
        database_path=tmp_path / "configured-ohlcv.sqlite3",
        history_start=cutoff.date(),
        now=retrieved_at,
    )

    assert FakeAlphaVantageClient.calls == [
        (("SPY", "IWM"), configured_fields)
    ]
    assert FakeAlphaVantageClient.reservations == [(2, 1)]
    with sqlite3.connect(tmp_path / "configured-ohlcv.sqlite3") as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM request_budget_events"
        ).fetchone()[0] == 2
    alpha_source = next(
        item for item in result.sources if item["id"] == "alpha_vantage"
    )
    assert alpha_source["requests_made"] == 2
    assert alpha_source["records"] == 2 * len(configured_fields)
    assert result.overall_health is HealthStatus.OK


def test_missing_alpha_key_does_not_reserve_or_construct_client(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY")

    class ForbiddenAlphaClient:
        def __init__(self, *_args, **_kwargs) -> None:
            raise AssertionError("missing key must stop before client construction")

    monkeypatch.setattr(collection_module, "AlphaVantageClient", ForbiddenAlphaClient)
    cutoff = datetime(2024, 1, 5, 21, tzinfo=timezone.utc)
    database = tmp_path / "missing-key.sqlite3"
    result = collection_module.collect_live_data(
        {
            "alpha_vantage": {
                "base_url": "https://example.invalid/alpha",
                "daily_request_cap": 25,
                "symbols": ["SPY"],
            },
            "alfred": {"base_url": "https://example.invalid/fred", "series": []},
        },
        database_path=database,
        history_start=cutoff.date(),
        now=cutoff + timedelta(hours=15),
    )

    assert result.overall_health is HealthStatus.DEGRADED
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type = 'table' AND name = 'request_budget_events'"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM observations WHERE source = 'alpha_vantage'"
        ).fetchone()[0] == 0
    with SQLiteSnapshotStore(database) as store:
        missing_key = store.list_provenance(source="alpha_vantage")[-1]
    assert missing_key.request_params["budget_reserved_requests"] == 0
    assert missing_key.request_params["budget_unused_reserved_requests"] == 0


@pytest.mark.parametrize("invalid_cap", [True, 25.0, "25", 24, 26])
def test_invalid_standard_free_cap_fails_closed_before_client_or_event(
    tmp_path,
    monkeypatch,
    invalid_cap,
) -> None:
    class ForbiddenAlphaClient:
        def __init__(self, *_args, **_kwargs) -> None:
            raise AssertionError("invalid cap must stop before client construction")

    monkeypatch.setattr(collection_module, "AlphaVantageClient", ForbiddenAlphaClient)
    cutoff = datetime(2024, 1, 5, 21, tzinfo=timezone.utc)
    database = tmp_path / f"invalid-cap-{type(invalid_cap).__name__}.sqlite3"
    result = collection_module.collect_live_data(
        {
            "alpha_vantage": {
                "base_url": "https://example.invalid/alpha",
                "daily_request_cap": invalid_cap,
                "symbols": ["SPY"],
            },
            "alfred": {"base_url": "https://example.invalid/fred", "series": []},
        },
        database_path=database,
        history_start=cutoff.date(),
        now=cutoff + timedelta(hours=15),
    )

    assert result.overall_health is HealthStatus.DEGRADED
    assert any("integer 25" in issue for issue in result.issues)
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type = 'table' AND name = 'request_budget_events'"
        ).fetchone()[0] == 0
    with SQLiteSnapshotStore(database) as store:
        invalid = store.list_provenance(source="alpha_vantage")[-1]
    assert invalid.request_params["budget_reserved_requests"] == 0
    assert invalid.request_params["budget_unused_reserved_requests"] == 0


def test_provider_failure_keeps_entire_atomic_reservation(
    tmp_path,
    monkeypatch,
) -> None:
    class FailingAlphaClient:
        reservation: tuple[int, int] | None = None

        def __init__(self, *_args, **kwargs) -> None:
            type(self).reservation = (
                kwargs["budget"].remaining,
                kwargs["retry"].max_attempts,
            )

        def fetch_weekly_adjusted(self, *_args, **_kwargs) -> CollectionResult:
            return CollectionResult(
                health=HealthStatus.DEGRADED,
                issues=("provider failed before first completed response",),
            )

    monkeypatch.setattr(collection_module, "AlphaVantageClient", FailingAlphaClient)
    cutoff = datetime(2024, 1, 5, 21, tzinfo=timezone.utc)
    database = tmp_path / "provider-failure-reservation.sqlite3"
    result = collection_module.collect_live_data(
        {
            "alpha_vantage": {
                "base_url": "https://example.invalid/alpha",
                "daily_request_cap": 25,
                "symbols": ["SPY", "IWM", "QQQ"],
            },
            "alfred": {"base_url": "https://example.invalid/fred", "series": []},
        },
        database_path=database,
        history_start=cutoff.date(),
        now=cutoff + timedelta(hours=15),
    )

    assert result.overall_health is HealthStatus.DEGRADED
    assert FailingAlphaClient.reservation == (3, 1)
    restarted = DailyRequestBudget(limit=25, database_path=database)
    assert restarted.remaining == 22
    with SQLiteSnapshotStore(database) as store:
        failed = store.list_provenance(source="alpha_vantage")[-1]
    assert failed.request_params["budget_reserved_requests"] == 3
    assert failed.request_params["budget_unused_reserved_requests"] == 3


def test_impossible_full_batch_over_cap_fails_closed_without_reservation(
    tmp_path,
    monkeypatch,
) -> None:
    class ForbiddenAlphaClient:
        def __init__(self, *_args, **_kwargs) -> None:
            raise AssertionError("over-cap batch must not construct a client")

    monkeypatch.setattr(collection_module, "AlphaVantageClient", ForbiddenAlphaClient)
    cutoff = datetime(2024, 1, 5, 21, tzinfo=timezone.utc)
    database = tmp_path / "over-cap.sqlite3"
    symbols = [f"ETF{index:02d}" for index in range(26)]
    result = collection_module.collect_live_data(
        {
            "alpha_vantage": {
                "base_url": "https://example.invalid/alpha",
                "daily_request_cap": 25,
                "symbols": symbols,
            },
            "alfred": {"base_url": "https://example.invalid/fred", "series": []},
        },
        database_path=database,
        history_start=cutoff.date(),
        now=cutoff + timedelta(hours=15),
    )

    assert result.overall_health is HealthStatus.QUOTA_EXHAUSTED
    assert any("exceeds" in issue for issue in result.issues)
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM request_budget_events"
        ).fetchone()[0] == 0


def test_same_cutoff_additive_alpha_config_fetches_only_missing_symbols(
    tmp_path,
    monkeypatch,
) -> None:
    cutoff = datetime(2024, 1, 5, 21, tzinfo=timezone.utc)
    current = cutoff + timedelta(hours=15)
    database = tmp_path / "alpha-additive-expansion.sqlite3"
    configured_symbols = (
        "SPY",
        "QQQ",
        "IWM",
        "DIA",
        "RSP",
        "XLB",
        "XLC",
        "XLE",
        "XLF",
        "XLI",
        "XLK",
        "XLP",
        "XLRE",
        "XLU",
        "XLV",
        "XLY",
        "SHY",
        "IEF",
        "TLT",
        "HYG",
        "LQD",
        "GLD",
        "UUP",
    )
    existing_symbols = configured_symbols[:20]
    added_symbols = configured_symbols[20:]
    _seed_alpha_test_snapshot(
        database,
        cutoff=cutoff,
        symbols=existing_symbols,
    )
    persisted_budget = DailyRequestBudget(limit=25, database_path=database)
    assert persisted_budget.consume(20)

    class FakeAlphaVantageClient:
        calls: list[tuple[str, ...]] = []
        reservations: list[tuple[int, int]] = []

        def __init__(self, *_args, **kwargs) -> None:
            type(self).reservations.append(
                (kwargs["budget"].remaining, kwargs["retry"].max_attempts)
            )

        def fetch_weekly_adjusted(self, symbols, **kwargs) -> CollectionResult:
            requested = tuple(symbols)
            type(self).calls.append(requested)
            return CollectionResult(
                records=tuple(
                    _alpha_test_record(
                        symbol,
                        field,
                        period=cutoff.date(),
                        cutoff=cutoff,
                        retrieved_at=current,
                    )
                    for symbol in requested
                    for field in tuple(kwargs["fields"])
                ),
                requests_made=len(requested),
                attempts=len(requested),
            )

    monkeypatch.setattr(
        collection_module,
        "AlphaVantageClient",
        FakeAlphaVantageClient,
    )
    config = {
        "alpha_vantage": {
            "base_url": "https://example.invalid/alpha",
            "daily_request_cap": 25,
            "symbols": list(configured_symbols),
            "fields": list(ALPHA_TEST_FIELDS),
        },
        "alfred": {"base_url": "https://example.invalid/fred", "series": []},
    }

    result = collection_module.collect_live_data(
        config,
        database_path=database,
        history_start=cutoff.date(),
        now=current,
    )

    assert FakeAlphaVantageClient.calls == [added_symbols]
    assert FakeAlphaVantageClient.reservations == [(len(added_symbols), 1)]
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM request_budget_events"
        ).fetchone()[0] == 23
    assert result.overall_health is HealthStatus.OK
    alpha_records = tuple(
        item for item in result.records if item.source == "alpha_vantage"
    )
    assert {item.series_id for item in alpha_records} == {
        f"{symbol}.{field}"
        for symbol in configured_symbols
        for field in ALPHA_TEST_FIELDS
    }
    with SQLiteSnapshotStore(database) as store:
        provenance = store.list_provenance(source="alpha_vantage")
        stored_delta = store.read_observations(
            snapshot_id=provenance[-1].snapshot_id
        )
        assembled = store.read_last_good_observations(
            source="alpha_vantage",
            dataset="weekly_adjusted_etf",
        )

    assert len(provenance) == 2
    assert provenance[-1].quality_status is HealthStatus.OK
    assert provenance[-1].request_params["symbols"] == list(configured_symbols)
    assert provenance[-1].request_params["requested_symbols"] == list(added_symbols)
    assert provenance[-1].request_params["snapshot_mode"] == SnapshotMode.DELTA.value
    assert provenance[-1].request_params["added_records"] == (
        len(added_symbols) * len(ALPHA_TEST_FIELDS)
    )
    assert provenance[-1].request_params["changed_records"] == 0
    assert provenance[-1].request_params["config_expansion_validation"] == "passed"
    assert provenance[-1].request_params["initial_baseline_validation"] == "passed"
    assert {item.series_id for item in stored_delta} == {
        f"{symbol}.{field}"
        for symbol in added_symbols
        for field in ALPHA_TEST_FIELDS
    }
    assert len(assembled) == len(configured_symbols) * len(ALPHA_TEST_FIELDS)


def test_impossible_additive_batch_over_cap_fails_closed_without_reservation(
    tmp_path,
    monkeypatch,
) -> None:
    cutoff = datetime(2024, 1, 5, 21, tzinfo=timezone.utc)
    database = tmp_path / "over-cap-additive.sqlite3"
    _seed_alpha_test_snapshot(database, cutoff=cutoff, symbols=("SPY",))
    symbols = ["SPY", *(f"ETF{index:02d}" for index in range(26))]

    class ForbiddenAlphaClient:
        def __init__(self, *_args, **_kwargs) -> None:
            raise AssertionError("over-cap additive batch must not construct client")

    monkeypatch.setattr(collection_module, "AlphaVantageClient", ForbiddenAlphaClient)
    result = collection_module.collect_live_data(
        {
            "alpha_vantage": {
                "base_url": "https://example.invalid/alpha",
                "daily_request_cap": 25,
                "symbols": symbols,
                "fields": list(ALPHA_TEST_FIELDS),
            },
            "alfred": {"base_url": "https://example.invalid/fred", "series": []},
        },
        database_path=database,
        history_start=cutoff.date(),
        now=cutoff + timedelta(hours=15),
    )

    assert result.overall_health is HealthStatus.QUOTA_EXHAUSTED
    assert any("exceeds" in issue for issue in result.issues)
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM request_budget_events"
        ).fetchone()[0] == 0


def test_same_cutoff_complete_alpha_reuse_adds_no_budget_event(
    tmp_path,
    monkeypatch,
) -> None:
    cutoff = datetime(2024, 1, 5, 21, tzinfo=timezone.utc)
    database = tmp_path / "complete-reuse.sqlite3"
    _seed_alpha_test_snapshot(database, cutoff=cutoff, symbols=("SPY", "IWM"))

    class ForbiddenAlphaClient:
        def __init__(self, *_args, **_kwargs) -> None:
            raise AssertionError("complete same-cutoff history must be reused")

    monkeypatch.setattr(collection_module, "AlphaVantageClient", ForbiddenAlphaClient)
    result = collection_module.collect_live_data(
        {
            "alpha_vantage": {
                "base_url": "https://example.invalid/alpha",
                "daily_request_cap": 25,
                "symbols": ["SPY", "IWM"],
                "fields": list(ALPHA_TEST_FIELDS),
            },
            "alfred": {"base_url": "https://example.invalid/fred", "series": []},
        },
        database_path=database,
        history_start=cutoff.date(),
        now=cutoff + timedelta(hours=15),
    )

    assert result.overall_health is HealthStatus.OK
    with SQLiteSnapshotStore(database) as store:
        assert len(store.list_provenance(source="alpha_vantage")) == 1
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type = 'table' AND name = 'request_budget_events'"
        ).fetchone()[0] == 0


@pytest.mark.parametrize(
    ("seed_cutoff_delta", "request_symbols", "omitted_series"),
    [
        (timedelta(0), ("SPY", "IWM"), {"IWM.volume"}),
        (timedelta(0), ("SPY", "IWM", "IWM"), set()),
        (timedelta(days=-7), ("SPY", "IWM"), set()),
    ],
    ids=("partial-field", "duplicate-provenance-symbol", "different-cutoff"),
)
def test_unsafe_alpha_config_expansion_falls_back_to_full_fetch(
    tmp_path,
    monkeypatch,
    seed_cutoff_delta: timedelta,
    request_symbols: tuple[str, ...],
    omitted_series: set[str],
) -> None:
    cutoff = datetime(2024, 1, 12, 21, tzinfo=timezone.utc)
    seed_cutoff = cutoff + seed_cutoff_delta
    current = cutoff + timedelta(hours=15)
    database = tmp_path / f"alpha-unsafe-expansion-{seed_cutoff.date()}.sqlite3"
    _seed_alpha_test_snapshot(
        database,
        cutoff=seed_cutoff,
        symbols=("SPY", "IWM"),
        request_symbols=request_symbols,
        omitted_series=omitted_series,
    )

    class FakeAlphaVantageClient:
        calls: list[tuple[str, ...]] = []

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def fetch_weekly_adjusted(self, symbols, **kwargs) -> CollectionResult:
            requested = tuple(symbols)
            type(self).calls.append(requested)
            provider_periods = tuple(
                seed_cutoff.date() + timedelta(weeks=index)
                for index in range(
                    ((cutoff.date() - seed_cutoff.date()).days // 7) + 1
                )
            )
            return CollectionResult(
                records=tuple(
                    _alpha_test_record(
                        symbol,
                        field,
                        period=period,
                        cutoff=datetime.combine(period, cutoff.timetz()),
                        retrieved_at=current,
                    )
                    for symbol in requested
                    for field in tuple(kwargs["fields"])
                    for period in provider_periods
                ),
                requests_made=len(requested),
                attempts=len(requested),
            )

    monkeypatch.setattr(
        collection_module,
        "AlphaVantageClient",
        FakeAlphaVantageClient,
    )
    config = {
        "alpha_vantage": {
            "base_url": "https://example.invalid/alpha",
            "daily_request_cap": 25,
            "symbols": ["SPY", "IWM", "QQQ"],
            "fields": list(ALPHA_TEST_FIELDS),
        },
        "alfred": {"base_url": "https://example.invalid/fred", "series": []},
    }

    result = collection_module.collect_live_data(
        config,
        database_path=database,
        history_start=seed_cutoff.date(),
        now=current,
    )

    assert FakeAlphaVantageClient.calls == [("SPY", "IWM", "QQQ")]
    assert result.overall_health is HealthStatus.OK


def test_alpha_config_expansion_preflights_persisted_daily_budget(
    tmp_path,
    monkeypatch,
) -> None:
    cutoff = datetime(2024, 1, 5, 21, tzinfo=timezone.utc)
    current = cutoff + timedelta(hours=15)
    database = tmp_path / "alpha-expansion-budget.sqlite3"
    _seed_alpha_test_snapshot(
        database,
        cutoff=cutoff,
        symbols=("SPY", "IWM"),
    )
    budget = DailyRequestBudget(limit=25, database_path=database)
    assert budget.consume(25)

    class FakeAlphaVantageClient:
        calls = 0

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def fetch_weekly_adjusted(self, *_args, **_kwargs) -> CollectionResult:
            type(self).calls += 1
            raise AssertionError("preflight must not spend a partial expansion batch")

    monkeypatch.setattr(
        collection_module,
        "AlphaVantageClient",
        FakeAlphaVantageClient,
    )
    config = {
        "alpha_vantage": {
            "base_url": "https://example.invalid/alpha",
            "daily_request_cap": 25,
            "symbols": ["SPY", "IWM", "QQQ"],
            "fields": list(ALPHA_TEST_FIELDS),
        },
        "alfred": {"base_url": "https://example.invalid/fred", "series": []},
    }

    result = collection_module.collect_live_data(
        config,
        database_path=database,
        history_start=cutoff.date(),
        now=current,
    )

    assert FakeAlphaVantageClient.calls == 0
    assert result.overall_health is HealthStatus.QUOTA_EXHAUSTED
    alpha_source = next(
        item for item in result.sources if item["id"] == "alpha_vantage"
    )
    assert alpha_source["status"] == HealthStatus.QUOTA_EXHAUSTED.value
    with SQLiteSnapshotStore(database) as store:
        provenance = store.list_provenance(source="alpha_vantage")
        last_good = store.get_last_good_provenance(
            source="alpha_vantage",
            dataset="weekly_adjusted_etf",
        )
        failed_rows = store.read_observations(
            snapshot_id=provenance[-1].snapshot_id
        )
    assert provenance[-1].quality_status is HealthStatus.QUOTA_EXHAUSTED
    assert last_good is not None and last_good.snapshot_id == provenance[0].snapshot_id
    assert failed_rows == ()


def test_realtime_year_chunks_are_contiguous_and_bounded() -> None:
    chunks = _realtime_year_chunks(date(2006, 1, 1), date(2026, 8, 7))
    assert chunks[0] == (date(2006, 1, 1), date(2009, 12, 31))
    assert chunks[-1] == (date(2026, 1, 1), date(2026, 8, 7))
    assert len(chunks) == 6
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert (current[0] - previous[1]).days == 1


def test_daily_alfred_fetch_keeps_full_observation_history_per_chunk() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def fetch_realtime_observations(self, series_ids, **kwargs):
            self.calls.append({"series_ids": series_ids, **kwargs})
            return CollectionResult(
                health=HealthStatus.OK,
                requests_made=1,
                attempts=1,
            )

        def fetch_revision_events(self, *_args, **_kwargs):
            raise AssertionError("FULL collection must not use output_type=3")

    client = FakeClient()
    result = _fetch_alfred_series(
        client,  # type: ignore[arg-type]
        series_id="DGS10",
        frequency="daily",
        history_start=date(2006, 1, 1),
        realtime_end=date(2026, 8, 7),
        cutoff=datetime(2026, 8, 11, tzinfo=timezone.utc),
        snapshot_mode=SnapshotMode.FULL,
    )

    assert len(client.calls) == 6
    assert result.requests_made == 6
    assert result.health is HealthStatus.OK
    assert {call["observation_start"] for call in client.calls} == {
        date(2006, 1, 1)
    }
    assert client.calls[0]["realtime_end"] == date(2009, 12, 31)
    assert client.calls[-1]["realtime_start"] == date(2026, 1, 1)
    request_params = _alfred_request_params(
        series_id="DGS10",
        frequency="daily",
        realtime_start=date(2006, 1, 1),
        realtime_end=date(2026, 8, 7),
        observation_start=date(2006, 1, 1),
        snapshot_mode=SnapshotMode.FULL,
    )
    assert request_params["output_type"] == 1
    assert request_params["realtime_start"] == "2006-01-01"
    assert "vintage_dates" not in request_params


def test_delta_alfred_fetch_uses_type3_once_with_inclusive_calendar_vintages() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.realtime_calls = 0
            self.revision_calls: list[dict[str, object]] = []

        def fetch_realtime_observations(self, *_args, **_kwargs):
            self.realtime_calls += 1
            raise AssertionError("DELTA collection must not use output_type=1")

        def fetch_revision_events(self, series_ids, **kwargs):
            self.revision_calls.append({"series_ids": series_ids, **kwargs})
            return CollectionResult(
                health=HealthStatus.OK,
                requests_made=1,
                attempts=1,
            )

    client = FakeClient()
    result = _fetch_alfred_series(
        client,  # type: ignore[arg-type]
        series_id="DGS10",
        frequency="daily",
        history_start=date(2026, 7, 31),
        observation_start=date(2006, 1, 1),
        realtime_end=date(2026, 8, 7),
        cutoff=datetime(2026, 8, 11, tzinfo=timezone.utc),
        snapshot_mode=SnapshotMode.DELTA,
    )

    assert result.health is HealthStatus.OK
    assert client.realtime_calls == 0
    assert len(client.revision_calls) == 1
    call = client.revision_calls[0]
    assert call["series_ids"] == ["DGS10"]
    vintages = call["vintage_dates"]
    assert isinstance(vintages, tuple)
    assert vintages == tuple(
        date(2026, 7, 31) + timedelta(days=offset) for offset in range(8)
    )
    assert call["observation_start"] == date(2006, 1, 1)
    assert call["observation_end"] == date(2026, 8, 7)

    request_params = _alfred_request_params(
        series_id="DGS10",
        frequency="daily",
        realtime_start=date(2026, 7, 31),
        realtime_end=date(2026, 8, 7),
        observation_start=date(2006, 1, 1),
        snapshot_mode=SnapshotMode.DELTA,
    )
    assert request_params["output_type"] == 3
    assert request_params["vintage_dates"] == (
        "2026-07-31,2026-08-01,2026-08-02,2026-08-03,"
        "2026-08-04,2026-08-05,2026-08-06,2026-08-07"
    )
    assert "realtime_start" not in request_params
    assert "realtime_end" not in request_params


def test_daily_chunk_merge_assigns_one_global_revision_sequence() -> None:
    class FakeClient:
        def fetch_realtime_observations(
            self,
            series_ids,
            **kwargs,
        ) -> CollectionResult:
            vintage = kwargs["realtime_start"]
            available = datetime(
                vintage.year,
                vintage.month,
                vintage.day,
                23,
                tzinfo=timezone.utc,
            )
            return CollectionResult(
                records=(
                    Observation(
                        source="alfred",
                        series_id=series_ids[0],
                        observed_period_end=date(2006, 1, 1),
                        value=float(vintage.year),
                        released_at=available,
                        available_at=available,
                        vintage_date=vintage,
                        retrieved_at=datetime(2026, 8, 12, tzinfo=timezone.utc),
                        # Every provider chunk starts with its own local zero.
                        revision_seq=0,
                        raw_sha256=f"chunk-{vintage.isoformat()}",
                    ),
                ),
                health=HealthStatus.OK,
                requests_made=1,
                attempts=1,
            )

    result = _fetch_alfred_series(
        FakeClient(),  # type: ignore[arg-type]
        series_id="DGS10",
        frequency="daily",
        history_start=date(2006, 1, 1),
        realtime_end=date(2026, 8, 7),
        cutoff=datetime(2026, 8, 11, tzinfo=timezone.utc),
    )

    assert [item.vintage_date.year for item in result.records] == [
        2006,
        2010,
        2014,
        2018,
        2022,
        2026,
    ]
    assert [item.revision_seq for item in result.records] == list(range(6))
    assert result.requests_made == 6
    assert result.attempts == 6


def test_full_plus_delta_assembly_ignores_degraded_partial_snapshot(tmp_path) -> None:
    retrieved_at = datetime(2026, 8, 11, 23, tzinfo=timezone.utc)

    def revision(vintage: date, value: float, raw_hash: str) -> Observation:
        available_at = datetime(
            vintage.year,
            vintage.month,
            vintage.day,
            19,
            tzinfo=timezone.utc,
        )
        return Observation(
            source="alfred",
            series_id="UNRATE",
            observed_period_end=date(2026, 7, 1),
            value=value,
            released_at=available_at,
            available_at=available_at,
            vintage_date=vintage,
            retrieved_at=retrieved_at,
            raw_sha256=raw_hash,
        )

    database = tmp_path / "snapshots.sqlite3"
    with SQLiteSnapshotStore(database) as store:
        _write_result_snapshot(
            store,
            CollectionResult(records=(revision(date(2026, 7, 31), 4.2, "base"),)),
            source="alfred",
            dataset="UNRATE",
            cutoff=datetime(2026, 7, 31, 20, tzinfo=timezone.utc),
            requested_at=datetime.now(timezone.utc),
            license_class="test",
            request_params=_alfred_request_params(
                series_id="UNRATE",
                frequency="monthly",
                realtime_start=date(2006, 1, 1),
                realtime_end=date(2026, 7, 31),
                observation_start=date(2006, 1, 1),
                snapshot_mode=SnapshotMode.FULL,
            ),
        )
        _write_result_snapshot(
            store,
            CollectionResult(records=(revision(date(2026, 8, 7), 4.1, "delta"),)),
            source="alfred",
            dataset="UNRATE",
            cutoff=datetime(2026, 8, 7, 20, tzinfo=timezone.utc),
            requested_at=datetime.now(timezone.utc),
            license_class="test",
            request_params=_alfred_request_params(
                series_id="UNRATE",
                frequency="monthly",
                realtime_start=date(2026, 7, 31),
                realtime_end=date(2026, 8, 7),
                observation_start=date(2006, 1, 1),
                snapshot_mode=SnapshotMode.DELTA,
            ),
        )

        assembled_before_failure = store.read_last_good_observations(
            source="alfred",
            dataset="UNRATE",
        )
        assert [item.value for item in assembled_before_failure] == [4.2, 4.1]
        assert [item.revision_seq for item in assembled_before_failure] == [0, 1]

        _write_result_snapshot(
            store,
            CollectionResult(
                records=(revision(date(2026, 8, 7), 99.0, "partial"),),
                health=HealthStatus.DEGRADED,
                issues=("later page failed",),
                requests_made=1,
                attempts=3,
            ),
            source="alfred",
            dataset="UNRATE",
            cutoff=datetime(2026, 8, 7, 20, tzinfo=timezone.utc),
            requested_at=datetime.now(timezone.utc),
            license_class="test",
            request_params=_alfred_request_params(
                series_id="UNRATE",
                frequency="monthly",
                realtime_start=date(2026, 8, 7),
                realtime_end=date(2026, 8, 7),
                observation_start=date(2006, 1, 1),
                snapshot_mode=SnapshotMode.DELTA,
            ),
        )

        assembled_after_failure = store.read_last_good_observations(
            source="alfred",
            dataset="UNRATE",
        )
        provenance = store.list_provenance(source="alfred")
        last_good = store.get_last_good_provenance(
            source="alfred",
            dataset="UNRATE",
        )

    assert [item.raw_sha256 for item in assembled_after_failure] == ["base", "delta"]
    assert [item.quality_status for item in provenance] == [
        HealthStatus.OK,
        HealthStatus.OK,
        HealthStatus.DEGRADED,
    ]
    assert last_good is not None
    assert last_good.request_params["output_type"] == 3


def test_alpha_baseline_coverage_uses_common_periods_and_dedupes_revisions() -> None:
    cutoff = datetime(2024, 1, 26, 21, tzinfo=timezone.utc)
    retrieved_at = datetime(2024, 1, 30, 12, tzinfo=timezone.utc)
    periods = tuple(date(2024, 1, day) for day in (5, 12, 19, 26))
    expected = {"SPY.adjusted_close", "SPY.volume"}

    def record(
        field: str,
        period: date,
        *,
        available_at: datetime | None = None,
        value: float = 1.0,
    ) -> Observation:
        available = available_at or datetime(
            period.year,
            period.month,
            period.day,
            21,
            tzinfo=timezone.utc,
        )
        return Observation(
            source="alpha_vantage",
            series_id=f"SPY.{field}",
            observed_period_end=period,
            value=value,
            released_at=available,
            available_at=available,
            vintage_date=available.date(),
            retrieved_at=retrieved_at,
            raw_sha256=f"{field}-{period.isoformat()}-{value}",
            metadata={"symbol": "SPY", "field": field},
        )

    complete = tuple(
        record(field, period)
        for period in periods
        for field in ("adjusted_close", "volume")
    )
    accepted, metrics = _validate_initial_alpha_baseline(
        CollectionResult(records=complete, requests_made=1, attempts=1),
        expected_series=expected,
        history_start=date(2024, 1, 1),
        cutoff=cutoff,
        minimum_coverage=0.75,
    )
    assert accepted.health is HealthStatus.OK
    assert metrics["initial_baseline_validation"] == "passed"
    assert metrics["initial_baseline_common_periods"] == 4

    # An assembled append-only chain can contain an eligible prospective
    # revision for a period already present. Coverage counts that period once.
    revision = record(
        "adjusted_close",
        periods[0],
        available_at=datetime(2024, 1, 20, 21, tzinfo=timezone.utc),
        value=2.0,
    )
    existing_check, existing_metrics = _validate_initial_alpha_baseline(
        CollectionResult(records=(*complete, revision)),
        expected_series=expected,
        history_start=date(2024, 1, 1),
        cutoff=cutoff,
        minimum_coverage=0.75,
        strict_response=False,
    )
    assert existing_check.health is HealthStatus.OK
    assert existing_metrics["initial_baseline_duplicate_records"] == 1

    # Three common weeks satisfy the numeric 75% threshold, but a field-pair
    # mismatch is still rejected rather than silently accepting an incomplete
    # symbol response.
    mismatched = tuple(
        item
        for item in complete
        if not (
            item.series_id == "SPY.volume"
            and item.observed_period_end == periods[0]
        )
    )
    rejected, rejected_metrics = _validate_initial_alpha_baseline(
        CollectionResult(records=mismatched, requests_made=1, attempts=1),
        expected_series=expected,
        history_start=date(2024, 1, 1),
        cutoff=cutoff,
        minimum_coverage=0.75,
    )
    assert rejected.health is HealthStatus.DEGRADED
    assert rejected.records == ()
    assert rejected.requests_made == 1
    assert rejected.attempts == 1
    assert rejected_metrics["initial_baseline_pair_mismatch_symbols"] == 1


def test_alpha_symbol_specific_inception_accepts_complete_short_history_and_rejects_truncation() -> None:
    history_start = date(2018, 1, 1)
    xlc_start = date(2018, 6, 18)
    cutoff = datetime(2018, 8, 31, 20, tzinfo=timezone.utc)
    retrieved_at = cutoff + timedelta(days=1)
    global_periods = tuple(
        item.date() for item in weekly_cutoffs(history_start, cutoff)
    )
    xlc_periods = tuple(
        item.date() for item in weekly_cutoffs(xlc_start, cutoff)
    )
    fields = ("adjusted_close", "volume")
    expected = {
        f"{symbol}.{field}"
        for symbol in ("SPY", "XLC")
        for field in fields
    }

    def records(symbol: str, periods: tuple[date, ...]) -> tuple[Observation, ...]:
        return tuple(
            Observation(
                source="alpha_vantage",
                series_id=f"{symbol}.{field}",
                observed_period_end=period,
                value=(1_000_000.0 if field == "volume" else 100.0),
                released_at=datetime.combine(
                    period,
                    datetime.min.time(),
                    tzinfo=timezone.utc,
                ),
                available_at=datetime.combine(
                    period,
                    datetime.min.time(),
                    tzinfo=timezone.utc,
                ),
                vintage_date=period,
                retrieved_at=retrieved_at,
                raw_sha256=f"{symbol}-{field}-{period.isoformat()}",
                metadata={"symbol": symbol, "field": field},
            )
            for period in periods
            for field in fields
        )

    complete = (*records("SPY", global_periods), *records("XLC", xlc_periods))
    accepted, metrics = _validate_initial_alpha_baseline(
        CollectionResult(records=complete),
        expected_series=expected,
        history_start=history_start,
        cutoff=cutoff,
        history_start_by_symbol={"XLC": xlc_start},
    )

    assert accepted.health is HealthStatus.OK
    assert metrics["initial_baseline_validation"] == "passed"
    assert metrics["initial_baseline_symbol_specific_coverage"] is True
    assert metrics["initial_baseline_common_periods"] == len(xlc_periods)
    assert len(xlc_periods) < metrics["initial_baseline_minimum_periods"]

    truncated_xlc = xlc_periods[: max(1, len(xlc_periods) // 2)]
    rejected, rejected_metrics = _validate_initial_alpha_baseline(
        CollectionResult(
            records=(*records("SPY", global_periods), *records("XLC", truncated_xlc))
        ),
        expected_series=expected,
        history_start=history_start,
        cutoff=cutoff,
        history_start_by_symbol={"XLC": xlc_start},
    )

    assert rejected.health is HealthStatus.DEGRADED
    assert rejected.records == ()
    assert rejected_metrics["initial_baseline_validation"] == "failed"
    assert rejected_metrics["initial_baseline_stale_series"] == len(fields)


@pytest.mark.parametrize("failure_mode", ["missing_series", "truncated_history"])
def test_invalid_initial_alpha_baseline_is_provenance_only(
    tmp_path,
    monkeypatch,
    failure_mode: str,
) -> None:
    cutoff = datetime(2024, 1, 26, 21, tzinfo=timezone.utc)
    retrieved_at = datetime(2024, 1, 30, 12, tzinfo=timezone.utc)
    full_periods = tuple(date(2024, 1, day) for day in (5, 12, 19, 26))

    class FakeAlphaVantageClient:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def fetch_weekly_adjusted(self, *_args, **_kwargs) -> CollectionResult:
            periods = (
                (full_periods[-1],)
                if failure_mode == "truncated_history"
                else full_periods
            )
            fields = (
                ("adjusted_close",)
                if failure_mode == "missing_series"
                else ("adjusted_close", "volume")
            )
            records = tuple(
                Observation(
                    source="alpha_vantage",
                    series_id=f"SPY.{field}",
                    observed_period_end=period,
                    value=1.0,
                    released_at=datetime(
                        period.year,
                        period.month,
                        period.day,
                        21,
                        tzinfo=timezone.utc,
                    ),
                    available_at=datetime(
                        period.year,
                        period.month,
                        period.day,
                        21,
                        tzinfo=timezone.utc,
                    ),
                    vintage_date=period,
                    retrieved_at=retrieved_at,
                    raw_sha256=f"{field}-{period.isoformat()}",
                    metadata={"symbol": "SPY", "field": field},
                )
                for period in periods
                for field in fields
            )
            return CollectionResult(records=records, requests_made=1, attempts=1)

    monkeypatch.setattr(
        collection_module,
        "AlphaVantageClient",
        FakeAlphaVantageClient,
    )
    config = {
        "alpha_vantage": {
            "base_url": "https://example.invalid/alpha",
            "daily_request_cap": 25,
            "symbols": ["SPY"],
        },
        "alfred": {"base_url": "https://example.invalid/fred", "series": []},
    }
    database = tmp_path / f"alpha-{failure_mode}.sqlite3"

    result = collection_module.collect_live_data(
        config,
        database_path=database,
        history_start=date(2024, 1, 1),
        now=datetime(2024, 1, 30, 12, tzinfo=timezone.utc),
    )

    assert result.model_cutoff == cutoff
    assert result.overall_health is HealthStatus.DEGRADED
    assert not any(item.source == "alpha_vantage" for item in result.records)
    alpha_source = next(
        item for item in result.sources if item["id"] == "alpha_vantage"
    )
    assert alpha_source["status"] == HealthStatus.DEGRADED.value
    assert any("completeness/coverage" in issue for issue in alpha_source["issues"])

    with SQLiteSnapshotStore(database) as store:
        provenance = store.list_provenance(source="alpha_vantage")
        stored_rows = store.read_observations(
            snapshot_id=provenance[-1].snapshot_id
        )
        last_good = store.get_last_good_provenance(
            source="alpha_vantage",
            dataset="weekly_adjusted_etf",
        )

    assert len(provenance) == 1
    assert provenance[0].quality_status is HealthStatus.DEGRADED
    assert provenance[0].request_params["initial_baseline_validation"] == "failed"
    assert stored_rows == ()
    assert last_good is None


@pytest.mark.parametrize("provider_recovers", [False, True])
def test_invalid_legacy_alpha_baseline_is_never_reused_or_exposed(
    tmp_path,
    monkeypatch,
    provider_recovers: bool,
) -> None:
    cutoff = datetime(2024, 1, 26, 21, tzinfo=timezone.utc)
    legacy_retrieved = datetime(2024, 1, 27, 12, tzinfo=timezone.utc)
    current_retrieved = datetime(2024, 1, 30, 12, tzinfo=timezone.utc)
    periods = tuple(date(2024, 1, day) for day in (5, 12, 19, 26))

    def alpha_record(field: str, period: date, retrieved: datetime) -> Observation:
        available = datetime(
            period.year,
            period.month,
            period.day,
            21,
            tzinfo=timezone.utc,
        )
        return Observation(
            source="alpha_vantage",
            series_id=f"SPY.{field}",
            observed_period_end=period,
            value=1.0,
            released_at=available,
            available_at=available,
            vintage_date=period,
            retrieved_at=retrieved,
            raw_sha256=f"{field}-{period.isoformat()}-{retrieved.isoformat()}",
            metadata={"symbol": "SPY", "field": field},
        )

    database = tmp_path / f"legacy-recovery-{provider_recovers}.sqlite3"
    with SQLiteSnapshotStore(database) as store:
        store.write_snapshot(
            tuple(
                alpha_record(field, periods[-1], legacy_retrieved)
                for field in ("adjusted_close", "volume")
            ),
            SnapshotProvenance(
                source="alpha_vantage",
                dataset="weekly_adjusted_etf",
                cutoff=cutoff,
                requested_at=legacy_retrieved,
                retrieved_at=legacy_retrieved,
                quality_status=HealthStatus.OK,
                request_params={"snapshot_mode": SnapshotMode.FULL.value},
            ),
        )

    class FakeAlphaVantageClient:
        calls = 0

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def fetch_weekly_adjusted(self, *_args, **_kwargs) -> CollectionResult:
            type(self).calls += 1
            if not provider_recovers:
                return CollectionResult(
                    health=HealthStatus.DEGRADED,
                    issues=("provider unavailable",),
                    requests_made=1,
                    attempts=1,
                )
            return CollectionResult(
                records=tuple(
                    alpha_record(field, period, current_retrieved)
                    for period in periods
                    for field in ("adjusted_close", "volume")
                ),
                requests_made=1,
                attempts=1,
            )

    monkeypatch.setattr(
        collection_module,
        "AlphaVantageClient",
        FakeAlphaVantageClient,
    )
    config = {
        "alpha_vantage": {
            "base_url": "https://example.invalid/alpha",
            "daily_request_cap": 25,
            "symbols": ["SPY"],
        },
        "alfred": {"base_url": "https://example.invalid/fred", "series": []},
    }

    result = collection_module.collect_live_data(
        config,
        database_path=database,
        history_start=date(2024, 1, 1),
        now=datetime(2024, 1, 30, 12, tzinfo=timezone.utc),
    )

    assert FakeAlphaVantageClient.calls == 1
    alpha_records = tuple(
        item for item in result.records if item.source == "alpha_vantage"
    )
    with SQLiteSnapshotStore(database) as store:
        provenance = store.list_provenance(source="alpha_vantage")
        assembled = store.read_last_good_observations(
            source="alpha_vantage",
            dataset="weekly_adjusted_etf",
        )

    if provider_recovers:
        assert len(alpha_records) == 8
        assert len(assembled) == 8
        assert provenance[-1].quality_status is HealthStatus.OK
        assert provenance[-1].request_params["snapshot_mode"] == SnapshotMode.FULL.value
        assert provenance[-1].request_params["existing_history_validation"] == "failed"
    else:
        assert alpha_records == ()
        assert result.overall_health is HealthStatus.DEGRADED
        assert [item.quality_status for item in provenance] == [
            HealthStatus.OK,
            HealthStatus.DEGRADED,
        ]
        # The audit store preserves the legacy last-good snapshot, but the
        # current collection result deliberately excludes it from model input.
        assert len(assembled) == 2


def test_alpha_full_response_that_stops_advancing_is_degraded(tmp_path, monkeypatch) -> None:
    history_start = date(2023, 9, 15)
    periods = tuple(history_start + timedelta(weeks=index) for index in range(20))
    base_cutoff = datetime(2024, 1, 26, 21, tzinfo=timezone.utc)
    current_cutoff = datetime(2024, 2, 9, 21, tzinfo=timezone.utc)
    base_retrieved = datetime(2024, 1, 30, 12, tzinfo=timezone.utc)
    current_retrieved = datetime(2024, 2, 13, 12, tzinfo=timezone.utc)

    def alpha_record(field: str, period: date, retrieved: datetime) -> Observation:
        available = datetime(
            period.year,
            period.month,
            period.day,
            21,
            tzinfo=timezone.utc,
        )
        return Observation(
            source="alpha_vantage",
            series_id=f"SPY.{field}",
            observed_period_end=period,
            value=1.0,
            released_at=available,
            available_at=available,
            vintage_date=period,
            retrieved_at=retrieved,
            raw_sha256=f"{field}-{period.isoformat()}-{retrieved.isoformat()}",
            metadata={"symbol": "SPY", "field": field},
        )

    database = tmp_path / "alpha-stopped.sqlite3"
    with SQLiteSnapshotStore(database) as store:
        store.write_snapshot(
            tuple(
                alpha_record(field, period, base_retrieved)
                for period in periods
                for field in ("adjusted_close", "volume")
            ),
            SnapshotProvenance(
                source="alpha_vantage",
                dataset="weekly_adjusted_etf",
                cutoff=base_cutoff,
                requested_at=base_retrieved,
                retrieved_at=base_retrieved,
                quality_status=HealthStatus.OK,
                request_params={"snapshot_mode": SnapshotMode.FULL.value},
            ),
        )

    class FakeAlphaVantageClient:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def fetch_weekly_adjusted(self, *_args, **_kwargs) -> CollectionResult:
            # Provider says OK but repeats the old full response at T+14.
            return CollectionResult(
                records=tuple(
                    alpha_record(field, period, current_retrieved)
                    for period in periods
                    for field in ("adjusted_close", "volume")
                ),
                requests_made=1,
                attempts=1,
            )

    monkeypatch.setattr(
        collection_module,
        "AlphaVantageClient",
        FakeAlphaVantageClient,
    )
    config = {
        "alpha_vantage": {
            "base_url": "https://example.invalid/alpha",
            "daily_request_cap": 25,
            "symbols": ["SPY"],
        },
        "alfred": {"base_url": "https://example.invalid/fred", "series": []},
    }

    result = collection_module.collect_live_data(
        config,
        database_path=database,
        history_start=history_start,
        now=datetime(2024, 2, 13, 12, tzinfo=timezone.utc),
    )

    assert result.model_cutoff == current_cutoff
    assert result.overall_health is HealthStatus.DEGRADED
    assert len(
        [item for item in result.records if item.source == "alpha_vantage"]
    ) == 40
    assert next(
        item for item in result.sources if item["id"] == "alpha_vantage"
    )["status"] == HealthStatus.DEGRADED.value

    with SQLiteSnapshotStore(database) as store:
        provenance = store.list_provenance(source="alpha_vantage")
        last_good = store.get_last_good_provenance(
            source="alpha_vantage",
            dataset="weekly_adjusted_etf",
        )
        failed_rows = store.read_observations(
            snapshot_id=provenance[-1].snapshot_id
        )

    assert [item.quality_status for item in provenance] == [
        HealthStatus.OK,
        HealthStatus.DEGRADED,
    ]
    assert provenance[-1].request_params["initial_baseline_common_periods"] == 20
    assert provenance[-1].request_params["initial_baseline_minimum_periods"] == 20
    assert failed_rows == ()
    assert last_good is not None
    assert last_good.cutoff == base_cutoff


def test_live_collection_runs_full_then_delta_and_keeps_last_good_on_failure(
    tmp_path,
    monkeypatch,
) -> None:
    retrieved_at = datetime(2026, 8, 12, tzinfo=timezone.utc)

    class FakeAlphaVantageClient:
        calls = 0
        cutoffs: list[datetime] = []

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def fetch_weekly_adjusted(self, *_args, **kwargs) -> CollectionResult:
            type(self).calls += 1
            self.cutoffs.append(kwargs["cutoff"])
            discovery_times = (
                datetime(2026, 7, 28, 12, tzinfo=timezone.utc),
                datetime(2026, 8, 4, 12, tzinfo=timezone.utc),
                datetime(2026, 8, 11, 12, tzinfo=timezone.utc),
            )
            discovery_time = discovery_times[self.calls - 1]
            periods = tuple(
                date(2026, 7, 1) + timedelta(weeks=index)
                for index in range(self.calls + 3)
            )
            fields = (
                ("adjusted_close",)
                if self.calls == 3
                else ("adjusted_close", "volume")
            )
            records = tuple(
                Observation(
                    source="alpha_vantage",
                    series_id=f"SPY.{field}",
                    observed_period_end=period,
                    value=(
                        601.0
                        if field == "adjusted_close"
                        and self.calls > 1
                        and index == 0
                        else (
                            600.0 + index
                            if field == "adjusted_close"
                            else 1_000.0 + index * 100
                        )
                    ),
                    released_at=available_at,
                    available_at=available_at,
                    vintage_date=period,
                    retrieved_at=discovery_time,
                    raw_sha256=f"alpha-{field}-{period.isoformat()}",
                    metadata={"symbol": "SPY", "field": field},
                )
                for index, period in enumerate(periods)
                for field in fields
                for available_at in (
                    datetime(
                        period.year,
                        period.month,
                        period.day,
                        19,
                        tzinfo=timezone.utc,
                    ),
                )
            )
            return CollectionResult(records=records, requests_made=1, attempts=1)

    class FakeAlfredClient:
        calls: list[tuple[str, dict[str, object]]] = []

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        @staticmethod
        def _revision(vintage: date, value: float, raw_hash: str) -> Observation:
            available_at = datetime(
                vintage.year,
                vintage.month,
                vintage.day,
                19,
                tzinfo=timezone.utc,
            )
            return Observation(
                source="alfred",
                series_id="UNRATE",
                observed_period_end=date(2026, 7, 1),
                value=value,
                released_at=available_at,
                available_at=available_at,
                vintage_date=vintage,
                retrieved_at=retrieved_at,
                raw_sha256=raw_hash,
            )

        def fetch_realtime_observations(
            self,
            _series_ids,
            **kwargs,
        ) -> CollectionResult:
            self.calls.append(("full", dict(kwargs)))
            return CollectionResult(
                records=(self._revision(date(2026, 7, 24), 4.2, "base"),),
                requests_made=1,
                attempts=1,
            )

        def fetch_revision_events(
            self,
            _series_ids,
            **kwargs,
        ) -> CollectionResult:
            self.calls.append(("delta", dict(kwargs)))
            delta_number = sum(mode == "delta" for mode, _params in self.calls)
            if delta_number == 1:
                return CollectionResult(
                    records=(
                        self._revision(date(2026, 7, 31), 4.1, "delta"),
                    ),
                    requests_made=1,
                    attempts=1,
                )
            return CollectionResult(
                records=(
                    self._revision(date(2026, 8, 7), 99.0, "partial"),
                ),
                health=HealthStatus.DEGRADED,
                issues=("later page failed",),
                requests_made=1,
                attempts=3,
            )

    monkeypatch.setattr(
        collection_module,
        "AlphaVantageClient",
        FakeAlphaVantageClient,
    )
    monkeypatch.setattr(collection_module, "AlfredClient", FakeAlfredClient)
    config = {
        "alpha_vantage": {
            "base_url": "https://example.invalid/alpha",
            "daily_request_cap": 25,
            "symbols": ["SPY"],
        },
        "alfred": {
            "base_url": "https://example.invalid/fred",
            "series": [
                {
                    "id": "UNRATE",
                    "frequency": "monthly",
                    "realtime_start": "2026-07-01",
                }
            ],
        },
    }
    database = tmp_path / "live.sqlite3"

    full = collection_module.collect_live_data(
        config,
        database_path=database,
        history_start=date(2026, 7, 1),
        now=datetime(2026, 7, 28, 12, tzinfo=timezone.utc),
    )
    delta = collection_module.collect_live_data(
        config,
        database_path=database,
        history_start=date(2026, 7, 1),
        now=datetime(2026, 8, 4, 12, tzinfo=timezone.utc),
    )
    degraded = collection_module.collect_live_data(
        config,
        database_path=database,
        history_start=date(2026, 7, 1),
        now=datetime(2026, 8, 11, 12, tzinfo=timezone.utc),
    )

    assert [mode for mode, _params in FakeAlfredClient.calls] == [
        "full",
        "delta",
        "delta",
    ]
    first_delta_vintages = FakeAlfredClient.calls[1][1]["vintage_dates"]
    second_delta_vintages = FakeAlfredClient.calls[2][1]["vintage_dates"]
    assert first_delta_vintages[0] == date(2026, 7, 24)
    assert first_delta_vintages[-1] == date(2026, 7, 31)
    assert second_delta_vintages[0] == date(2026, 7, 31)
    assert second_delta_vintages[-1] == date(2026, 8, 7)
    assert FakeAlfredClient.calls[1][1]["observation_start"] == date(2026, 7, 1)
    assert FakeAlphaVantageClient.cutoffs == [
        datetime(2026, 7, 24, 20, tzinfo=timezone.utc),
        datetime(2026, 7, 31, 20, tzinfo=timezone.utc),
        datetime(2026, 8, 7, 20, tzinfo=timezone.utc),
    ]

    assert [
        item.value for item in full.records if item.source == "alfred"
    ] == [4.2]
    assert [
        item.value for item in delta.records if item.source == "alfred"
    ] == [4.2, 4.1]
    assert [
        item.value for item in degraded.records if item.source == "alfred"
    ] == [4.2, 4.1]
    assert degraded.overall_health is HealthStatus.DEGRADED
    degraded_alpha = tuple(
        item for item in degraded.records if item.source == "alpha_vantage"
    )
    assert len(degraded_alpha) == 11
    assert {item.series_id for item in degraded_alpha} == {
        "SPY.adjusted_close",
        "SPY.volume",
    }
    assert next(
        source for source in degraded.sources if source["id"] == "alpha_vantage"
    )["status"] == HealthStatus.DEGRADED.value

    with SQLiteSnapshotStore(database) as store:
        provenance = tuple(
            item
            for item in store.list_provenance(source="alfred")
            if item.dataset == "UNRATE"
        )
        assembled = store.read_last_good_observations(
            source="alfred",
            dataset="UNRATE",
        )
        last_good = store.get_last_good_provenance(
            source="alfred",
            dataset="UNRATE",
        )
        failed_rows = store.read_observations(
            snapshot_id=provenance[-1].snapshot_id,
        )
        alpha_provenance = store.list_provenance(source="alpha_vantage")
        alpha_failed_rows = store.read_observations(
            snapshot_id=alpha_provenance[-1].snapshot_id,
        )
        alpha_assembled = store.read_last_good_observations(
            source="alpha_vantage",
            dataset="weekly_adjusted_etf",
        )
        alpha_before_revision = store.read_last_good_observations(
            source="alpha_vantage",
            dataset="weekly_adjusted_etf",
            available_as_of=datetime(2026, 8, 1, tzinfo=timezone.utc),
        )

    assert [item.request_params["output_type"] for item in provenance] == [1, 3, 3]
    assert [item.request_params["snapshot_mode"] for item in provenance] == [
        SnapshotMode.FULL.value,
        SnapshotMode.DELTA.value,
        SnapshotMode.DELTA.value,
    ]
    assert [item.quality_status for item in provenance] == [
        HealthStatus.OK,
        HealthStatus.OK,
        HealthStatus.DEGRADED,
    ]
    assert [item.raw_sha256 for item in assembled] == ["base", "delta"]
    assert failed_rows == ()
    assert last_good is not None
    assert last_good.snapshot_id == provenance[1].snapshot_id
    assert [item.quality_status for item in alpha_provenance] == [
        HealthStatus.OK,
        HealthStatus.OK,
        HealthStatus.DEGRADED,
    ]
    assert alpha_failed_rows == ()
    adjusted_chain = [
        item
        for item in alpha_assembled
        if item.series_id == "SPY.adjusted_close"
        and item.observed_period_end == date(2026, 7, 1)
    ]
    assert [item.value for item in adjusted_chain] == [600.0, 601.0]
    assert [item.revision_seq for item in adjusted_chain] == [0, 1]
    assert adjusted_chain[1].available_at == datetime(
        2026,
        8,
        4,
        12,
        tzinfo=timezone.utc,
    )
    assert adjusted_chain[1].metadata["prospective_revision"] is True
    assert [
        item.value
        for item in alpha_before_revision
        if item.series_id == "SPY.adjusted_close"
        and item.observed_period_end == date(2026, 7, 1)
    ] == [600.0]


def test_alpha_legacy_partial_is_quarantined_before_next_completed_delta(
    tmp_path,
    monkeypatch,
) -> None:
    database = tmp_path / "legacy-alpha.sqlite3"
    prior_cutoff = datetime(2024, 1, 5, 21, tzinfo=timezone.utc)
    base_retrieved = datetime(2024, 1, 10, 12, tzinfo=timezone.utc)
    revision_retrieved = datetime(2024, 1, 10, 13, tzinfo=timezone.utc)

    def alpha_record(
        field: str,
        period_end: date,
        value: float,
        available_at: datetime,
        retrieved_at: datetime,
        raw_hash: str,
    ) -> Observation:
        return Observation(
            source="alpha_vantage",
            series_id=f"SPY.{field}",
            observed_period_end=period_end,
            value=value,
            released_at=available_at,
            available_at=available_at,
            vintage_date=available_at.date(),
            retrieved_at=retrieved_at,
            raw_sha256=raw_hash,
            metadata={"symbol": "SPY", "field": field},
        )

    def alpha_provenance(
        retrieved_at: datetime,
        mode: SnapshotMode,
    ) -> SnapshotProvenance:
        return SnapshotProvenance(
            source="alpha_vantage",
            dataset="weekly_adjusted_etf",
            cutoff=prior_cutoff,
            requested_at=retrieved_at,
            retrieved_at=retrieved_at,
            quality_status=HealthStatus.OK,
            request_params={"snapshot_mode": mode.value},
        )

    period_one = date(2024, 1, 1)
    partial_period = date(2024, 1, 8)
    period_one_available = datetime(2024, 1, 1, 21, tzinfo=timezone.utc)
    partial_available = datetime(2024, 1, 8, 21, tzinfo=timezone.utc)
    prospective_available = revision_retrieved
    with SQLiteSnapshotStore(database) as store:
        store.write_snapshot(
            (
                alpha_record(
                    "adjusted_close",
                    period_one,
                    100.0,
                    period_one_available,
                    base_retrieved,
                    "base-adjusted",
                ),
                alpha_record(
                    "volume",
                    period_one,
                    1_000.0,
                    period_one_available,
                    base_retrieved,
                    "base-volume",
                ),
                alpha_record(
                    "adjusted_close",
                    partial_period,
                    999.0,
                    partial_available,
                    base_retrieved,
                    "legacy-partial-adjusted",
                ),
                alpha_record(
                    "volume",
                    partial_period,
                    9_999.0,
                    partial_available,
                    base_retrieved,
                    "legacy-partial-volume",
                ),
            ),
            alpha_provenance(base_retrieved, SnapshotMode.FULL),
        )
        store.write_snapshot(
            (
                alpha_record(
                    "adjusted_close",
                    period_one,
                    101.0,
                    prospective_available,
                    revision_retrieved,
                    "prospective-adjusted",
                ),
            ),
            alpha_provenance(revision_retrieved, SnapshotMode.DELTA),
        )

    class FakeAlphaVantageClient:
        cutoffs: list[datetime] = []

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def fetch_weekly_adjusted(
            self,
            *_args,
            **kwargs,
        ) -> CollectionResult:
            self.cutoffs.append(kwargs["cutoff"])
            retrieved = datetime(2024, 1, 16, 12, tzinfo=timezone.utc)
            return CollectionResult(
                records=(
                    alpha_record(
                        "adjusted_close",
                        period_one,
                        101.0,
                        period_one_available,
                        retrieved,
                        "current-adjusted",
                    ),
                    alpha_record(
                        "volume",
                        period_one,
                        1_000.0,
                        period_one_available,
                        retrieved,
                        "current-volume",
                    ),
                    alpha_record(
                        "adjusted_close",
                        partial_period,
                        110.0,
                        partial_available,
                        retrieved,
                        "completed-adjusted",
                    ),
                    alpha_record(
                        "volume",
                        partial_period,
                        1_100.0,
                        partial_available,
                        retrieved,
                        "completed-volume",
                    ),
                ),
                health=HealthStatus.OK,
                requests_made=1,
                attempts=1,
            )

    monkeypatch.setattr(
        collection_module,
        "AlphaVantageClient",
        FakeAlphaVantageClient,
    )
    config = {
        "alpha_vantage": {
            "base_url": "https://example.invalid/alpha",
            "daily_request_cap": 25,
            "symbols": ["SPY"],
        },
        "alfred": {
            "base_url": "https://example.invalid/fred",
            "series": [],
        },
    }

    result = collection_module.collect_live_data(
        config,
        database_path=database,
        history_start=date(2024, 1, 1),
        now=datetime(2024, 1, 16, 12, tzinfo=timezone.utc),
    )

    expected_cutoff = datetime(2024, 1, 12, 21, tzinfo=timezone.utc)
    assert FakeAlphaVantageClient.cutoffs == [expected_cutoff]
    alpha_values = [
        item.value for item in result.records if item.source == "alpha_vantage"
    ]
    assert alpha_values == [100.0, 101.0, 110.0, 1_000.0, 1_100.0]
    assert 999.0 not in alpha_values
    assert 9_999.0 not in alpha_values
    assert next(
        source for source in result.sources if source["id"] == "alpha_vantage"
    )["status"] == HealthStatus.OK.value

    with SQLiteSnapshotStore(database) as store:
        assembled = store.read_last_good_observations(
            source="alpha_vantage",
            dataset="weekly_adjusted_etf",
        )
        latest = store.get_last_good_provenance(
            source="alpha_vantage",
            dataset="weekly_adjusted_etf",
        )

    assert 999.0 not in {item.value for item in assembled}
    assert 9_999.0 not in {item.value for item in assembled}
    assert any(item.raw_sha256 == "prospective-adjusted" for item in assembled)
    assert latest is not None
    assert latest.cutoff == expected_cutoff
    assert latest.quality_status is HealthStatus.OK
    assert latest.request_params["removed_records"] == 0


def test_last_completed_week_before_friday_close_uses_prior_friday() -> None:
    # 2026-08-07 19:00 UTC is 15:00 ET, one hour before the weekly cutoff.
    result = last_completed_week_cutoff(
        datetime(2026, 8, 7, 19, 0, tzinfo=timezone.utc)
    )
    assert result == datetime(2026, 7, 31, 20, 0, tzinfo=timezone.utc)


def test_source_row_reports_only_records_eligible_at_model_cutoff() -> None:
    cutoff = datetime(2026, 8, 7, 20, 0, tzinfo=timezone.utc)

    def observation(period: date, available: datetime) -> Observation:
        return Observation(
            source="alfred",
            series_id="DGS10",
            observed_period_end=period,
            value=4.0,
            released_at=available,
            available_at=available,
            vintage_date=available.date(),
            retrieved_at=datetime(2026, 8, 11, tzinfo=timezone.utc),
        )

    eligible = observation(
        date(2026, 8, 6),
        datetime(2026, 8, 7, 19, 0, tzinfo=timezone.utc),
    )
    after_cutoff = observation(
        date(2026, 8, 7),
        datetime(2026, 8, 7, 22, 0, tzinfo=timezone.utc),
    )
    row = _source_row(
        source_id="alfred",
        name="ALFRED",
        result=CollectionResult(health=HealthStatus.OK),
        records=(eligible, after_cutoff),
        as_of=cutoff,
        frequency="daily",
        license_class="private",
    )

    assert row["available_at"] == eligible.available_at.isoformat()
    assert row["coverage"] == "2026-08-06–2026-08-06"
    assert row["records"] == 1
    assert row["raw_records"] == 2
