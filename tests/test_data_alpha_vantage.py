from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
from threading import Barrier
from typing import Any, Mapping

import pytest

from regime_lab.data import (
    AlphaVantageClient,
    AlphaVantageConfig,
    CollectionResult,
    DailyRequestBudget,
    HealthStatus,
    Observation,
    RetryPolicy,
    SnapshotMode,
    prepare_incremental_snapshot,
    weekly_asof_join,
)


UTC = timezone.utc
NOW = datetime(2024, 2, 10, 12, tzinfo=UTC)


class AlphaFixtureTransport:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get_json(
        self,
        _url: str,
        params: Mapping[str, Any],
        *,
        timeout: float,
    ) -> Mapping[str, Any]:
        del timeout
        self.calls.append(str(params["symbol"]))
        return {
            "Meta Data": {"2. Symbol": params["symbol"]},
            "Weekly Adjusted Time Series": {
                "2024-02-02": {
                    "1. open": "100",
                    "2. high": "103",
                    "3. low": "99",
                    "4. close": "102",
                    "5. adjusted close": "101.5",
                    "6. volume": "123456",
                    "7. dividend amount": "0.5",
                }
            },
        }


def test_alpha_budget_counts_actual_calls_and_fails_closed_at_cap() -> None:
    transport = AlphaFixtureTransport()
    budget = DailyRequestBudget(limit=25, clock=lambda: NOW)
    assert budget.consume(24)

    def discovery_clock() -> datetime:
        assert transport.calls, "discovery timestamp must be captured after the response"
        return NOW

    client = AlphaVantageClient(
        AlphaVantageConfig(api_key="secret"),
        transport=transport,
        budget=budget,
        retry=RetryPolicy(max_attempts=1),
        clock=discovery_clock,
    )

    result = client.fetch_weekly_adjusted(
        ["SPY", "QQQ"],
        cutoff=NOW,
        fields=("adjusted_close", "volume", "dividend_amount"),
    )

    assert result.health is HealthStatus.QUOTA_EXHAUSTED
    assert result.requests_made == 1
    assert transport.calls == ["SPY"]
    assert budget.remaining == 0
    assert {record.series_id for record in result.records} == {
        "SPY.adjusted_close",
        "SPY.volume",
        "SPY.dividend_amount",
    }
    assert all(
        record.metadata["pit_revision_policy"] == "prospective_on_later_diff"
        for record in result.records
    )
    by_series = {record.series_id: record for record in result.records}
    assert (
        by_series["SPY.dividend_amount"].metadata["research_role"]
        == "pit_corporate_action_input"
    )
    assert (
        by_series["SPY.adjusted_close"].metadata["research_role"]
        == "operating_feature_input"
    )
    assert "secret" not in repr(client.config)


def test_daily_budget_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "snapshots.sqlite3"
    first = DailyRequestBudget(limit=25, database_path=path, clock=lambda: NOW)
    assert first.consume(23)
    assert first.consume()
    assert first.remaining == 1

    restarted = DailyRequestBudget(limit=25, database_path=path, clock=lambda: NOW)
    assert restarted.consume()
    assert restarted.remaining == 0
    assert not restarted.consume()


def test_in_memory_budget_does_not_reset_at_a_calendar_midnight() -> None:
    current = datetime(2024, 7, 2, 3, 59, tzinfo=UTC)
    budget = DailyRequestBudget(limit=25, clock=lambda: current)
    assert budget.consume(25)

    current = datetime(2024, 7, 2, 4, 0, tzinfo=UTC)  # New York midnight
    assert budget.remaining == 0
    current = datetime(2024, 7, 3, 3, 58, 59, tzinfo=UTC)
    assert budget.remaining == 0
    current = datetime(2024, 7, 3, 3, 59, tzinfo=UTC)
    assert budget.remaining == 25


def test_daily_budget_persistence_resets_only_after_rolling_24_hours(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rolling-budget.sqlite3"
    first_call = datetime(2024, 7, 2, 3, 59, tzinfo=UTC)

    first = DailyRequestBudget(
        limit=25,
        database_path=path,
        clock=lambda: first_call,
    )
    assert first.consume(25)
    restarted_after_midnight = DailyRequestBudget(
        limit=25,
        database_path=path,
        clock=lambda: datetime(2024, 7, 2, 4, 0, tzinfo=UTC),
    )
    assert restarted_after_midnight.remaining == 0

    restarted_just_before_24h = DailyRequestBudget(
        limit=25,
        database_path=path,
        clock=lambda: datetime(2024, 7, 3, 3, 58, 59, tzinfo=UTC),
    )
    assert restarted_just_before_24h.remaining == 0
    restarted_at_24h = DailyRequestBudget(
        limit=25,
        database_path=path,
        clock=lambda: datetime(2024, 7, 3, 3, 59, tzinfo=UTC),
    )
    assert restarted_at_24h.remaining == 25


def test_next_available_at_requires_capacity_for_the_entire_batch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "next-capacity.sqlite3"
    current = datetime(2024, 7, 2, 12, 0, tzinfo=UTC)
    budget = DailyRequestBudget(limit=25, database_path=path, clock=lambda: current)
    assert budget.consume(15)

    current = datetime(2024, 7, 2, 13, 0, tzinfo=UTC)
    assert budget.consume(10)
    assert budget.next_available_at(1) == datetime(2024, 7, 3, 12, 0, tzinfo=UTC)
    assert budget.next_available_at(15) == datetime(2024, 7, 3, 12, 0, tzinfo=UTC)
    assert budget.next_available_at(16) == datetime(2024, 7, 3, 13, 0, tzinfo=UTC)

    current = datetime(2024, 7, 3, 12, 0, tzinfo=UTC)
    assert budget.next_available_at(16) == datetime(2024, 7, 3, 13, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="must not exceed"):
        budget.next_available_at(26)


def test_atomic_batch_reservation_survives_restart_and_unused_credits_are_charged(
    tmp_path: Path,
) -> None:
    path = tmp_path / "reserved-crash.sqlite3"
    budget = DailyRequestBudget(limit=25, database_path=path, clock=lambda: NOW)

    prepaid = budget.reserve(23)
    assert prepaid is not None and prepaid.remaining == 23
    # Simulate a crash before any prepaid transport credit is consumed.
    restarted = DailyRequestBudget(limit=25, database_path=path, clock=lambda: NOW)
    assert restarted.remaining == 2
    assert restarted.reserve(3) is None
    assert restarted.next_available_at(3) == NOW + timedelta(hours=24)


def test_spent_reserved_credit_expires_from_actual_attempt_time(
    tmp_path: Path,
) -> None:
    path = tmp_path / "attempt-time-reservation.sqlite3"
    current = datetime(2024, 7, 2, 12, tzinfo=UTC)
    budget = DailyRequestBudget(limit=25, database_path=path, clock=lambda: current)
    prepaid = budget.reserve(25)
    assert prepaid is not None
    assert prepaid.consume(24)
    current += timedelta(hours=1)
    assert prepaid.consume()

    current = datetime(2024, 7, 3, 12, tzinfo=UTC)
    restarted = DailyRequestBudget(
        limit=25,
        database_path=path,
        clock=lambda: current,
    )
    assert restarted.remaining == 24
    assert restarted.next_available_at(25) == datetime(
        2024, 7, 3, 13, tzinfo=UTC
    )


def test_expired_prepaid_credit_cannot_reenter_after_capacity_is_reused(
    tmp_path: Path,
) -> None:
    path = tmp_path / "expired-token.sqlite3"
    current = datetime(2024, 7, 2, 12, tzinfo=UTC)
    original = DailyRequestBudget(limit=25, database_path=path, clock=lambda: current)
    stale_token = original.reserve(25)
    assert stale_token is not None

    current += timedelta(hours=24)
    replacement = DailyRequestBudget(
        limit=25,
        database_path=path,
        clock=lambda: current,
    ).reserve(25)
    assert replacement is not None
    assert not stale_token.consume()
    assert replacement.consume()


def test_concurrent_batch_reservations_are_all_or_nothing(tmp_path: Path) -> None:
    path = tmp_path / "concurrent-reservations.sqlite3"
    first = DailyRequestBudget(limit=25, database_path=path, clock=lambda: NOW)
    second = DailyRequestBudget(limit=25, database_path=path, clock=lambda: NOW)
    barrier = Barrier(2)

    def reserve(budget: DailyRequestBudget) -> bool:
        barrier.wait()
        return budget.reserve(13) is not None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(reserve, (first, second)))

    assert sorted(results) == [False, True]
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM request_budget_events"
        ).fetchone()[0] == 13


def test_reservation_reconciles_legacy_write_after_budget_construction(
    tmp_path: Path,
) -> None:
    path = tmp_path / "concurrent-legacy-writer.sqlite3"
    budget = DailyRequestBudget(limit=25, database_path=path, clock=lambda: NOW)
    # Represents an already-running v1 process writing after v3 initialized.
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO daily_request_budget VALUES (?, ?, ?, ?)",
            ("alpha_vantage", NOW.date().isoformat(), 2, 25),
        )

    assert budget.reserve(24) is None
    # The failed reservation still commits reconciliation, so its retry advice
    # includes the legacy calls instead of incorrectly reporting capacity now.
    assert budget.remaining == 23
    assert budget.next_available_at(24) == NOW + timedelta(hours=24)
    prepaid = budget.reserve(23)
    assert prepaid is not None and prepaid.remaining == 23
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM daily_request_budget"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM request_budget_events"
        ).fetchone()[0] == 25


def test_v2_policy_migrates_once_and_rejects_configured_limit_drift(
    tmp_path: Path,
) -> None:
    path = tmp_path / "v2-policy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE request_budget_policy (
                provider TEXT PRIMARY KEY,
                policy_version INTEGER NOT NULL,
                policy_kind TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO request_budget_policy VALUES (?, ?, ?)",
            ("alpha_vantage", 2, "rolling_24h_utc"),
        )
        connection.execute(
            """
            CREATE TABLE request_budget_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL,
                consumed_at TEXT NOT NULL
            )
            """
        )
        connection.executemany(
            "INSERT INTO request_budget_events(provider, consumed_at) VALUES (?, ?)",
            (("alpha_vantage", NOW.isoformat()),) * 2,
        )
        connection.execute(
            """
            CREATE TABLE daily_request_budget (
                provider TEXT NOT NULL,
                usage_day TEXT NOT NULL,
                used INTEGER NOT NULL,
                limit_value INTEGER NOT NULL,
                PRIMARY KEY (provider, usage_day)
            )
            """
        )
        connection.execute(
            "INSERT INTO daily_request_budget VALUES (?, ?, ?, ?)",
            ("alpha_vantage", NOW.date().isoformat(), 1, 25),
        )

    migrated = DailyRequestBudget(limit=25, database_path=path, clock=lambda: NOW)
    assert migrated.remaining == 22
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT policy_version, policy_kind, limit_value "
            "FROM request_budget_policy"
        ).fetchone() == (3, "rolling_24h_utc", 25)
        assert connection.execute(
            "SELECT COUNT(*) FROM request_budget_events"
        ).fetchone()[0] == 3

    restarted = DailyRequestBudget(limit=25, database_path=path, clock=lambda: NOW)
    assert restarted.remaining == 22
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM request_budget_events"
        ).fetchone()[0] == 3

    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE request_budget_policy SET limit_value = 24 "
            "WHERE provider = 'alpha_vantage'"
        )
    with pytest.raises(RuntimeError, match="does not match"):
        DailyRequestBudget(limit=25, database_path=path, clock=lambda: NOW)
    with pytest.raises(ValueError, match="exactly 25"):
        DailyRequestBudget(limit=26)
    with pytest.raises(ValueError, match="exactly 25"):
        DailyRequestBudget(limit=5.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exactly 25"):
        DailyRequestBudget(limit=True)  # type: ignore[arg-type]


def test_v3_policy_rejects_non_integer_persisted_limit(tmp_path: Path) -> None:
    path = tmp_path / "invalid-v3-limit.sqlite3"
    budget = DailyRequestBudget(limit=25, database_path=path, clock=lambda: NOW)
    assert budget.remaining == 25
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE request_budget_policy SET limit_value = 5.5 "
            "WHERE provider = 'alpha_vantage'"
        )
    with pytest.raises(RuntimeError, match="request limit is invalid"):
        DailyRequestBudget(limit=25, database_path=path, clock=lambda: NOW)


def test_budget_event_order_uses_parsed_instants_not_iso_lexical_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "offset-timestamps.sqlite3"
    budget = DailyRequestBudget(limit=25, database_path=path, clock=lambda: NOW)
    with sqlite3.connect(path) as connection:
        connection.executemany(
            "INSERT INTO request_budget_events(provider, consumed_at) VALUES (?, ?)",
            (
                ("alpha_vantage", "2024-02-09T08:00:00-05:00"),  # 13:00 UTC
                ("alpha_vantage", "2024-02-09T12:30:00+00:00"),
            ),
        )

    assert budget.next_available_at(24) == datetime(2024, 2, 10, 12, 30, tzinfo=UTC)
    assert budget.next_available_at(25) == datetime(2024, 2, 10, 13, 0, tzinfo=UTC)


def _create_legacy_budget_database(
    path: Path,
    rows: list[tuple[str, int]],
) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE daily_request_budget (
                provider TEXT NOT NULL,
                usage_day TEXT NOT NULL,
                used INTEGER NOT NULL,
                limit_value INTEGER NOT NULL,
                PRIMARY KEY (provider, usage_day)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE snapshots (
                snapshot_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                requested_at TEXT NOT NULL,
                retrieved_at TEXT NOT NULL,
                quality_status TEXT NOT NULL,
                request_params_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE observations (
                snapshot_id TEXT NOT NULL,
                source TEXT NOT NULL,
                series_id TEXT NOT NULL,
                retrieved_at TEXT NOT NULL
            )
            """
        )
        connection.executemany(
            "INSERT INTO daily_request_budget VALUES (?, ?, ?, ?)",
            (("alpha_vantage", usage_day, used, 25) for usage_day, used in rows),
        )


def _insert_successful_alpha_batch(
    path: Path,
    *,
    snapshot_id: str,
    symbols: tuple[str, ...],
    requested_at: datetime,
    retrieved_at: datetime,
    response_times: tuple[datetime, ...] = (),
    use_requested_symbols: bool = True,
) -> None:
    key = "requested_symbols" if use_requested_symbols else "symbols"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO snapshots VALUES (?, ?, ?, ?, ?, ?)",
            (
                snapshot_id,
                "alpha_vantage",
                requested_at.isoformat(),
                retrieved_at.isoformat(),
                "ok",
                json.dumps({key: list(symbols)}),
            ),
        )
        if response_times:
            if len(response_times) != len(symbols):
                raise ValueError("response_times must cover every symbol")
            connection.executemany(
                "INSERT INTO observations VALUES (?, ?, ?, ?)",
                (
                    (
                        snapshot_id,
                        "alpha_vantage",
                        f"{symbol}.adjusted_close",
                        response_at.isoformat(),
                    )
                    for symbol, response_at in zip(
                        symbols,
                        response_times,
                        strict=True,
                    )
                ),
            )


def _insert_completed_failed_expansion(
    path: Path,
    *,
    snapshot_id: str,
    symbols: tuple[str, ...],
    fields: tuple[str, ...],
    requested_at: datetime,
    retrieved_at: datetime,
) -> None:
    expected = len(symbols) * len(fields)
    params = {
        "requested_symbols": list(symbols),
        "fields": list(fields),
        "config_expansion_fetched_initial_baseline_expected_series": expected,
        "config_expansion_fetched_initial_baseline_observed_series": expected,
        "config_expansion_fetched_initial_baseline_invalid_records": 0,
        "config_expansion_fetched_initial_baseline_duplicate_records": 0,
        "config_expansion_fetched_initial_baseline_validation": "failed",
    }
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO snapshots VALUES (?, ?, ?, ?, ?, ?)",
            (
                snapshot_id,
                "alpha_vantage",
                requested_at.isoformat(),
                retrieved_at.isoformat(),
                "degraded",
                json.dumps(params),
            ),
        )


def test_legacy_utc_budget_migration_uses_safe_success_batch_upper_bound(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy-utc-budget.sqlite3"
    _create_legacy_budget_database(path, [("2024-07-02", 20)])
    symbols = tuple(f"ETF{index:02d}" for index in range(20))
    anchor = datetime(2024, 7, 2, 3, 30, tzinfo=UTC)
    _insert_successful_alpha_batch(
        path,
        snapshot_id="successful-batch",
        symbols=symbols,
        requested_at=anchor - timedelta(minutes=1),
        retrieved_at=anchor,
    )

    migrated = DailyRequestBudget(
        limit=25,
        database_path=path,
        clock=lambda: datetime(2024, 7, 2, 12, 0, tzinfo=UTC),
    )
    assert migrated.remaining == 5
    assert migrated.consume(5)
    assert not migrated.consume()

    just_before_anchor_expiry = DailyRequestBudget(
        limit=25,
        database_path=path,
        clock=lambda: anchor + timedelta(hours=24) - timedelta(seconds=1),
    )
    assert just_before_anchor_expiry.remaining == 0
    at_anchor_expiry = DailyRequestBudget(
        limit=25,
        database_path=path,
        clock=lambda: anchor + timedelta(hours=24),
    )
    assert at_anchor_expiry.remaining == 20
    with sqlite3.connect(path) as connection:
        policy = connection.execute(
            """
            SELECT policy_version, policy_kind, limit_value
            FROM request_budget_policy
            WHERE provider = 'alpha_vantage'
            """
        ).fetchone()
        legacy_rows = connection.execute(
            "SELECT COUNT(*) FROM daily_request_budget WHERE provider = 'alpha_vantage'"
        ).fetchone()[0]
    assert policy == (3, "rolling_24h_utc", 25)
    assert legacy_rows == 0


def test_legacy_migration_reconstructs_overlapping_buckets_and_current_20_plus_3(
    tmp_path: Path,
) -> None:
    path = tmp_path / "evidence-like-budget.sqlite3"
    _create_legacy_budget_database(
        path,
        [("2026-08-11", 20), ("2026-08-12", 23)],
    )
    old_symbols = tuple(f"OLD{index:02d}" for index in range(20))
    old_response = datetime(2026, 8, 11, 13, 46, 39, tzinfo=UTC)
    _insert_successful_alpha_batch(
        path,
        snapshot_id="expired-20",
        symbols=old_symbols,
        requested_at=old_response - timedelta(microseconds=381),
        retrieved_at=datetime(2026, 8, 11, 13, 47, 17, tzinfo=UTC),
        response_times=(old_response,) * 20,
        use_requested_symbols=False,
    )
    current_symbols = (
        "SPY",
        "QQQ",
        "IWM",
        "DIA",
        "RSP",
        "XLY",
        "XLP",
        "XLF",
        "XLK",
        "XLI",
        "XLE",
        "XLV",
        "XLU",
        "SHY",
        "IEF",
        "TLT",
        "HYG",
        "LQD",
        "GLD",
        "UUP",
    )
    first_response = datetime(
        2026,
        8,
        12,
        6,
        55,
        11,
        133186,
        tzinfo=UTC,
    )
    response_times = tuple(
        first_response + timedelta(seconds=2 * index)
        for index in range(len(current_symbols))
    )
    _insert_successful_alpha_batch(
        path,
        snapshot_id="current-20",
        symbols=current_symbols,
        requested_at=datetime(2026, 8, 12, 6, 55, 8, 985504, tzinfo=UTC),
        retrieved_at=datetime(2026, 8, 12, 6, 55, 51, 578602, tzinfo=UTC),
        response_times=response_times,
        use_requested_symbols=False,
    )
    failed_anchor = datetime(2026, 8, 12, 16, 56, 25, 620513, tzinfo=UTC)
    _insert_completed_failed_expansion(
        path,
        snapshot_id="failed-3",
        symbols=("XLB", "XLC", "XLRE"),
        fields=("open", "high", "low", "close", "adjusted_close", "volume"),
        requested_at=datetime(2026, 8, 12, 16, 56, 18, 934790, tzinfo=UTC),
        retrieved_at=failed_anchor,
    )

    now = datetime(2026, 8, 12, 23, 5, tzinfo=UTC)
    migrated = DailyRequestBudget(
        limit=25,
        database_path=path,
        clock=lambda: now,
    )
    assert migrated.remaining == 2
    with sqlite3.connect(path) as connection:
        events = connection.execute(
            """
            SELECT consumed_at, COUNT(*)
            FROM request_budget_events
            WHERE provider = 'alpha_vantage'
            GROUP BY consumed_at
            ORDER BY consumed_at
            """
        ).fetchall()
        assert sum(count for _timestamp, count in events) == 23
        assert events[0] == (first_response.isoformat(), 1)
        assert events[-1] == (failed_anchor.isoformat(), 3)
        assert connection.execute(
            "SELECT COUNT(*) FROM daily_request_budget"
        ).fetchone()[0] == 0

    just_before_first_expiry = DailyRequestBudget(
        limit=25,
        database_path=path,
        clock=lambda: first_response + timedelta(hours=24, microseconds=-1),
    )
    assert just_before_first_expiry.remaining == 2
    at_first_expiry = DailyRequestBudget(
        limit=25,
        database_path=path,
        clock=lambda: first_response + timedelta(hours=24),
    )
    assert at_first_expiry.remaining == 3
    assert at_first_expiry.consume(3)
    assert at_first_expiry.remaining == 0


def test_legacy_migration_never_drops_newer_unmatched_calls_when_older_proof_expired(
    tmp_path: Path,
) -> None:
    path = tmp_path / "newer-unmatched.sqlite3"
    _create_legacy_budget_database(
        path,
        [("2024-07-01", 20), ("2024-07-02", 3)],
    )
    symbols = tuple(f"OLD{index:02d}" for index in range(20))
    old_anchor = datetime(2024, 7, 1, 11, tzinfo=UTC)
    _insert_successful_alpha_batch(
        path,
        snapshot_id="expired",
        symbols=symbols,
        requested_at=old_anchor - timedelta(minutes=1),
        retrieved_at=old_anchor,
    )
    now = datetime(2024, 7, 2, 12, tzinfo=UTC)

    migrated = DailyRequestBudget(
        limit=25,
        database_path=path,
        clock=lambda: now,
    )

    assert migrated.remaining == 22
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            """
            SELECT consumed_at, COUNT(*)
            FROM request_budget_events
            GROUP BY consumed_at
            """
        ).fetchall() == [(now.isoformat(), 3)]


def test_legacy_migration_anchors_crash_without_snapshot_at_now(
    tmp_path: Path,
) -> None:
    path = tmp_path / "crash-before-snapshot.sqlite3"
    _create_legacy_budget_database(path, [("2024-07-02", 4)])
    now = datetime(2024, 7, 2, 12, tzinfo=UTC)

    migrated = DailyRequestBudget(
        limit=25,
        database_path=path,
        clock=lambda: now,
    )
    assert migrated.remaining == 21
    just_before_expiry = DailyRequestBudget(
        limit=25,
        database_path=path,
        clock=lambda: now + timedelta(hours=24, microseconds=-1),
    )
    assert just_before_expiry.remaining == 21
    at_expiry = DailyRequestBudget(
        limit=25,
        database_path=path,
        clock=lambda: now + timedelta(hours=24),
    )
    assert at_expiry.remaining == 25


def test_legacy_migration_rolls_back_atomically_when_proof_exceeds_counter(
    tmp_path: Path,
) -> None:
    path = tmp_path / "contradictory-provenance.sqlite3"
    _create_legacy_budget_database(path, [("2024-07-02", 1)])
    anchor = datetime(2024, 7, 2, 11, tzinfo=UTC)
    _insert_successful_alpha_batch(
        path,
        snapshot_id="two-proven-calls",
        symbols=("SPY", "QQQ"),
        requested_at=anchor - timedelta(minutes=1),
        retrieved_at=anchor,
    )

    with pytest.raises(
        RuntimeError,
        match="snapshot calls exceed budget counter",
    ):
        DailyRequestBudget(
            limit=25,
            database_path=path,
            clock=lambda: datetime(2024, 7, 2, 12, tzinfo=UTC),
        )

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT usage_day, used FROM daily_request_budget"
        ).fetchall() == [("2024-07-02", 1)]
        assert connection.execute(
            """
            SELECT COUNT(*) FROM sqlite_master
            WHERE type = 'table'
              AND name IN ('request_budget_events', 'request_budget_policy')
            """
        ).fetchone()[0] == 0


def _alpha_record(
    period_end: date,
    value: float,
    *,
    retrieved_at: datetime,
    raw_hash: str,
) -> Observation:
    available = datetime(
        period_end.year,
        period_end.month,
        period_end.day,
        21,
        tzinfo=UTC,
    )
    return Observation(
        source="alpha_vantage",
        series_id="SPY.adjusted_close",
        observed_period_end=period_end,
        value=value,
        released_at=available,
        available_at=available,
        vintage_date=period_end,
        retrieved_at=retrieved_at,
        units="USD",
        adjustment="weekly_adjusted",
        raw_sha256=raw_hash,
        metadata={"symbol": "SPY", "field": "adjusted_close"},
    )


def test_full_response_diff_stores_only_new_and_changed_but_returns_full_history() -> None:
    old_retrieval = datetime(2024, 2, 10, tzinfo=UTC)
    new_retrieval = datetime(2024, 2, 12, tzinfo=UTC)
    existing = (
        _alpha_record(
            date(2024, 1, 26),
            100.0,
            retrieved_at=old_retrieval,
            raw_hash="unchanged",
        ),
        _alpha_record(
            date(2024, 2, 2),
            101.0,
            retrieved_at=old_retrieval,
            raw_hash="will-change",
        ),
    )
    incoming = CollectionResult(
        records=(
            # Retrieval time is ignored when content is otherwise identical.
            _alpha_record(
                date(2024, 1, 26),
                100.0,
                retrieved_at=new_retrieval,
                raw_hash="unchanged",
            ),
            _alpha_record(
                date(2024, 2, 2),
                102.0,
                retrieved_at=new_retrieval,
                raw_hash="changed",
            ),
            _alpha_record(
                date(2024, 2, 9),
                103.0,
                retrieved_at=new_retrieval,
                raw_hash="new",
            ),
        ),
        health=HealthStatus.OK,
        requests_made=20,
        attempts=20,
    )

    prepared = prepare_incremental_snapshot(existing, incoming)

    assert prepared.snapshot_mode is SnapshotMode.DELTA
    assert prepared.added_count == 1
    assert prepared.changed_count == 1
    assert prepared.unchanged_count == 1
    assert prepared.removed_count == 0
    assert {
        item.observed_period_end for item in prepared.snapshot_result.records
    } == {date(2024, 2, 2), date(2024, 2, 9)}
    assert len(prepared.effective_records) == 4
    assert prepared.snapshot_result.requests_made == 20

    changed_chain = [
        item
        for item in prepared.effective_records
        if item.observed_period_end == date(2024, 2, 2)
    ]
    assert [item.value for item in changed_chain] == [101.0, 102.0]
    assert [item.revision_seq for item in changed_chain] == [0, 1]
    assert changed_chain[1].available_at == new_retrieval
    assert changed_chain[1].released_at == new_retrieval
    assert changed_chain[1].vintage_date == new_retrieval.date()
    assert changed_chain[1].metadata["prospective_revision"] is True

    joined = weekly_asof_join(
        [
            datetime(2024, 2, 10, tzinfo=UTC),
            datetime(2024, 2, 13, tzinfo=UTC),
        ],
        changed_chain,
    )
    assert [item.value for item in joined] == [101.0, 102.0]

    # A later provider full response with the same current value must compare
    # against the latest prospective event, not generate another revision.
    repeated = prepare_incremental_snapshot(
        prepared.effective_records,
        CollectionResult(
            records=tuple(
                _alpha_record(
                    item.observed_period_end,
                    102.0 if item.observed_period_end == date(2024, 2, 2) else item.value,
                    retrieved_at=datetime(2024, 2, 14, tzinfo=UTC),
                    raw_hash=(
                        "changed-again-same-value"
                        if item.observed_period_end == date(2024, 2, 2)
                        else item.raw_sha256
                    ),
                )
                for item in incoming.records
            ),
            health=HealthStatus.OK,
        ),
    )
    assert repeated.snapshot_result.records == ()
    assert repeated.changed_count == 0
    assert len(repeated.effective_records) == 4


def test_non_ok_full_response_is_provenance_only_and_reuses_existing_history() -> None:
    retrieved = datetime(2024, 2, 12, tzinfo=UTC)
    existing = (
        _alpha_record(
            date(2024, 2, 2),
            101.0,
            retrieved_at=retrieved,
            raw_hash="existing",
        ),
    )
    partial = CollectionResult(
        records=(
            _alpha_record(
                date(2024, 2, 9),
                999.0,
                retrieved_at=retrieved,
                raw_hash="partial",
            ),
        ),
        health=HealthStatus.QUOTA_EXHAUSTED,
        issues=("quota",),
        requests_made=1,
        attempts=1,
    )

    prepared = prepare_incremental_snapshot(existing, partial)

    assert prepared.snapshot_mode is SnapshotMode.DELTA
    assert prepared.snapshot_result.records == ()
    assert prepared.snapshot_result.health is HealthStatus.QUOTA_EXHAUSTED
    assert [item.value for item in prepared.effective_records] == [101.0]


def test_removed_period_degrades_and_keeps_last_good_without_tombstone_policy() -> None:
    retrieved = datetime(2024, 2, 12, tzinfo=UTC)
    first = _alpha_record(
        date(2024, 1, 26),
        100.0,
        retrieved_at=retrieved,
        raw_hash="first",
    )
    second = _alpha_record(
        date(2024, 2, 2),
        101.0,
        retrieved_at=retrieved,
        raw_hash="second",
    )

    prepared = prepare_incremental_snapshot(
        (first, second),
        CollectionResult(records=(second,), health=HealthStatus.OK),
    )

    assert prepared.snapshot_mode is SnapshotMode.DELTA
    assert prepared.removed_count == 1
    assert prepared.snapshot_result.health is HealthStatus.DEGRADED
    assert prepared.snapshot_result.records == ()
    assert [item.observed_period_end for item in prepared.effective_records] == [
        date(2024, 1, 26),
        date(2024, 2, 2),
    ]
    assert any("omitted previously accepted" in issue for issue in prepared.snapshot_result.issues)

    recovered = prepare_incremental_snapshot(
        prepared.effective_records,
        CollectionResult(
            records=(
                _alpha_record(
                    date(2024, 1, 26),
                    100.0,
                    retrieved_at=datetime(2024, 2, 14, tzinfo=UTC),
                    raw_hash="first-returned",
                ),
                _alpha_record(
                    date(2024, 2, 2),
                    101.0,
                    retrieved_at=datetime(2024, 2, 14, tzinfo=UTC),
                    raw_hash="second-returned",
                ),
            ),
            health=HealthStatus.OK,
        ),
    )
    assert recovered.snapshot_result.health is HealthStatus.OK
    assert recovered.snapshot_result.records == ()
    assert [item.value for item in recovered.effective_records] == [100.0, 101.0]


def test_changed_row_with_non_advancing_discovery_time_is_degraded() -> None:
    period_end = date(2024, 2, 2)
    provider_available = datetime(2024, 2, 2, 21, tzinfo=UTC)
    existing = _alpha_record(
        period_end,
        101.0,
        retrieved_at=datetime(2024, 2, 10, tzinfo=UTC),
        raw_hash="existing",
    )
    unsafe_incoming = _alpha_record(
        period_end,
        999.0,
        retrieved_at=provider_available,
        raw_hash="clock-regressed",
    )

    prepared = prepare_incremental_snapshot(
        (existing,),
        CollectionResult(
            records=(unsafe_incoming,),
            health=HealthStatus.OK,
            requests_made=1,
            attempts=1,
        ),
    )

    assert prepared.snapshot_result.health is HealthStatus.DEGRADED
    assert prepared.snapshot_result.records == ()
    assert [item.value for item in prepared.effective_records] == [101.0]
    assert prepared.snapshot_result.requests_made == 1
    assert prepared.snapshot_result.attempts == 1
    assert any(
        "discovery time did not advance" in issue
        for issue in prepared.snapshot_result.issues
    )
