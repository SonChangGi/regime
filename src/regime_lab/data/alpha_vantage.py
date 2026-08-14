"""Alpha Vantage adapter with a persisted rolling-24-hour request guard."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Any, Callable, Mapping, Protocol, Sequence
from zoneinfo import ZoneInfo

from .contracts import CollectionResult, HealthStatus, Observation, ensure_utc
from .transport import (
    JsonTransport,
    ProviderRequestError,
    RetryPolicy,
    UrllibJsonTransport,
    request_json_with_retry,
)


ALPHA_VANTAGE_API_KEY_ENV = "ALPHA_VANTAGE_API_KEY"
_EASTERN = ZoneInfo("America/New_York")
_BUDGET_POLICY_VERSION = 3
_BUDGET_POLICY_KIND = "rolling_24h_utc"
_BUDGET_WINDOW = timedelta(hours=24)
_STANDARD_FREE_LIMIT = 25
_FIELD_MAP = {
    "open": "1. open",
    "high": "2. high",
    "low": "3. low",
    "close": "4. close",
    "adjusted_close": "5. adjusted close",
    "volume": "6. volume",
    "dividend_amount": "7. dividend amount",
}
_FIELD_UNITS = {
    "open": "USD",
    "high": "USD",
    "low": "USD",
    "close": "USD",
    "adjusted_close": "USD",
    "volume": "shares",
    "dividend_amount": "USD/share",
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RequestBudget(Protocol):
    @property
    def remaining(self) -> int: ...

    def consume(self, units: int = 1) -> bool: ...


class ReservedRequestBudget:
    """In-process credits backed by an already-persisted batch reservation.

    Reservation events are deliberately not refundable.  If the process dies,
    validation fails, or the provider stops the batch early, the unused credits
    remain charged until their rolling-window expiry.
    """

    def __init__(
        self,
        credit_ids: Sequence[int],
        *,
        touch: Callable[[tuple[int, ...]], bool] | None = None,
    ) -> None:
        if not credit_ids:
            raise ValueError("units must be positive")
        if len(set(credit_ids)) != len(credit_ids):
            raise ValueError("reserved credit ids must be unique")
        self._credit_ids = tuple(credit_ids)
        self._consumed = 0
        self._touch = touch
        self._lock = RLock()

    @property
    def remaining(self) -> int:
        with self._lock:
            return len(self._credit_ids) - self._consumed

    def consume(self, units: int = 1) -> bool:
        if units < 1:
            raise ValueError("units must be positive")
        with self._lock:
            if units > self.remaining:
                return False
            selected = self._credit_ids[self._consumed : self._consumed + units]
            # Move a spent credit's expiry no earlier than the actual transport
            # attempt. If the original reservation has already expired and its
            # capacity was reused, the callback rejects re-entry instead of
            # permitting a call outside the rolling cap.
            if self._touch is not None and not self._touch(selected):
                return False
            self._consumed += units
            return True


class DailyRequestBudget:
    """Thread-safe rolling-24-hour guard, optionally persisted in SQLite.

    Pass the application's snapshot database path in production so restarting a
    process cannot reset the provider budget.  Alpha Vantage documents a daily
    allowance but not its reset timezone, so a rolling UTC event ledger is the
    conservative local enforcement policy.
    """

    def __init__(
        self,
        *,
        limit: int = 25,
        provider: str = "alpha_vantage",
        database_path: str | Path | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if type(limit) is not int or limit != _STANDARD_FREE_LIMIT:
            raise ValueError(
                "standard-free Alpha Vantage request limit must be exactly 25"
            )
        self.limit = limit
        self.provider = provider
        self.database_path = str(database_path) if database_path is not None else None
        self.clock = clock
        self._usage: dict[int, datetime] = {}
        self._next_memory_credit_id = 1
        self._lock = RLock()
        if self.database_path is not None:
            Path(self.database_path).expanduser().resolve().parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            self._initialize_database()

    def _now(self) -> datetime:
        return ensure_utc(self.clock(), field_name="budget clock")

    def _connect(self) -> sqlite3.Connection:
        assert self.database_path is not None
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_request_budget (
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
                CREATE TABLE IF NOT EXISTS daily_request_budget_policy (
                    provider TEXT PRIMARY KEY,
                    calendar_version INTEGER NOT NULL,
                    calendar_timezone TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS request_budget_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL,
                    consumed_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS request_budget_events_window_idx
                ON request_budget_events(provider, consumed_at)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS request_budget_policy (
                    provider TEXT PRIMARY KEY,
                    policy_version INTEGER NOT NULL,
                    policy_kind TEXT NOT NULL,
                    limit_value INTEGER NOT NULL
                )
                """
            )
            policy_columns = self._table_columns(
                connection,
                "request_budget_policy",
            )
            if "limit_value" not in policy_columns:
                # SQLite cannot add a NOT NULL column without a default to an
                # existing v2 table.  The v2 row is bound to the configured
                # limit below in the same transaction.
                connection.execute(
                    "ALTER TABLE request_budget_policy ADD COLUMN limit_value INTEGER"
                )
            policy = connection.execute(
                """
                SELECT policy_version, policy_kind, limit_value
                FROM request_budget_policy
                WHERE provider = ?
                """,
                (self.provider,),
            ).fetchone()
            if policy is None:
                legacy_policy = connection.execute(
                    """
                    SELECT calendar_version, calendar_timezone
                    FROM daily_request_budget_policy
                    WHERE provider = ?
                    """,
                    (self.provider,),
                ).fetchone()
                legacy_basis = (
                    "new_york_calendar"
                    if legacy_policy == (1, "America/New_York")
                    else "utc_calendar"
                )
                if legacy_policy is not None and legacy_basis == "utc_calendar":
                    raise RuntimeError(
                        "persisted Alpha Vantage budget calendar policy is unsupported"
                    )
                self._migrate_legacy_buckets(
                    connection,
                    basis=legacy_basis,
                    now=self._now(),
                )
                connection.execute(
                    """
                    INSERT INTO request_budget_policy(
                        provider, policy_version, policy_kind, limit_value
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        self.provider,
                        _BUDGET_POLICY_VERSION,
                        _BUDGET_POLICY_KIND,
                        self.limit,
                    ),
                )
            elif tuple(policy[:2]) == (2, _BUDGET_POLICY_KIND):
                # v2 already has timestamped events but did not persist the
                # configured cap.  Bind it once, while conservatively charging
                # any calendar-counter writes from a concurrently running old
                # process.
                self._migrate_legacy_buckets(
                    connection,
                    basis="unknown_recent",
                    now=self._now(),
                )
                connection.execute(
                    """
                    UPDATE request_budget_policy
                    SET policy_version = ?, limit_value = ?
                    WHERE provider = ?
                    """,
                    (_BUDGET_POLICY_VERSION, self.limit, self.provider),
                )
            elif tuple(policy[:2]) != (
                _BUDGET_POLICY_VERSION,
                _BUDGET_POLICY_KIND,
            ):
                raise RuntimeError(
                    "persisted Alpha Vantage rolling budget policy is unsupported"
                )
            else:
                try:
                    raw_persisted_limit = policy[2]
                    if type(raw_persisted_limit) is not int:
                        raise TypeError
                    persisted_limit = raw_persisted_limit
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        "persisted Alpha Vantage request limit is invalid"
                    ) from exc
                if (
                    persisted_limit < 1
                    or persisted_limit > _STANDARD_FREE_LIMIT
                    or persisted_limit != self.limit
                ):
                    raise RuntimeError(
                        "persisted Alpha Vantage request limit does not match "
                        "the configured standard-free limit"
                    )
                # A legacy process writing calendar buckets after migration is
                # an ambiguous concurrent downgrade. Charge every such call at
                # the present instant before clearing those rows.
                self._migrate_legacy_buckets(
                    connection,
                    basis="unknown_recent",
                    now=self._now(),
                )
            connection.commit()

    @staticmethod
    def _legacy_bucket_interval(
        usage_day: date,
        *,
        basis: str,
    ) -> tuple[datetime, datetime]:
        if basis == "new_york_calendar":
            start = datetime.combine(usage_day, time.min, tzinfo=_EASTERN)
            end = datetime.combine(
                usage_day + timedelta(days=1),
                time.min,
                tzinfo=_EASTERN,
            )
            return start.astimezone(timezone.utc), end.astimezone(timezone.utc)
        start = datetime.combine(usage_day, time.min, tzinfo=timezone.utc)
        return start, start + timedelta(days=1)

    @staticmethod
    def _table_columns(
        connection: sqlite3.Connection,
        table: str,
    ) -> set[str]:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        if exists is None:
            return set()
        return {
            str(row[1])
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }

    @staticmethod
    def _legacy_timestamp(raw: object, *, field_name: str) -> datetime:
        try:
            return ensure_utc(
                datetime.fromisoformat(str(raw)),
                field_name=field_name,
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "persisted Alpha Vantage snapshot timestamp is invalid"
            ) from exc

    @staticmethod
    def _snapshot_symbols(raw: object) -> tuple[str, ...]:
        if not isinstance(raw, list):
            return ()
        symbols = tuple(
            dict.fromkeys(
                str(item).strip().upper()
                for item in raw
                if isinstance(item, str) and item.strip()
            )
        )
        return symbols if len(symbols) == len(raw) else ()

    def _exact_snapshot_call_times(
        self,
        connection: sqlite3.Connection,
        *,
        snapshot_id: str | None,
        requested_symbols: tuple[str, ...],
        requested_at: datetime,
        retrieved_at: datetime,
    ) -> tuple[datetime, ...]:
        """Return one post-response timestamp per provable symbol call.

        Alpha records created from one successful symbol response all share a
        retrieval timestamp captured after that response.  Such timestamps are
        conservative upper bounds on the corresponding budget consumption.
        Delta snapshots may not retain every returned row, so partial evidence
        is deliberately rejected rather than extrapolated.
        """

        if snapshot_id is None:
            return ()
        columns = self._table_columns(connection, "observations")
        required = {"snapshot_id", "source", "series_id", "retrieved_at"}
        if not required.issubset(columns):
            return ()
        rows = connection.execute(
            """
            SELECT series_id, retrieved_at
            FROM observations
            WHERE snapshot_id = ? AND source = ?
            GROUP BY series_id, retrieved_at
            """,
            (snapshot_id, self.provider),
        ).fetchall()
        by_symbol: dict[str, set[datetime]] = {}
        for raw_series, raw_retrieved in rows:
            series_id = str(raw_series)
            symbol, separator, _field = series_id.rpartition(".")
            if not separator or not symbol:
                return ()
            timestamp = self._legacy_timestamp(
                raw_retrieved,
                field_name="legacy Alpha observation retrieval timestamp",
            )
            if not requested_at <= timestamp <= retrieved_at:
                return ()
            by_symbol.setdefault(symbol.upper(), set()).add(timestamp)

        expected = set(requested_symbols) if requested_symbols else set(by_symbol)
        if not expected or set(by_symbol) != expected:
            return ()
        if any(len(by_symbol[symbol]) != 1 for symbol in expected):
            return ()
        order = requested_symbols or tuple(sorted(expected))
        call_times = tuple(next(iter(by_symbol[symbol])) for symbol in order)
        # The production client is sequential. A timestamp copied once across
        # multiple symbols is legacy batch metadata, not per-response evidence;
        # the caller will use the later snapshot retrieval upper bound instead.
        if len(call_times) > 1 and len(set(call_times)) != len(call_times):
            return ()
        return call_times

    @staticmethod
    def _completed_failed_expansion_count(
        params: Mapping[str, Any],
        *,
        requested_symbols: tuple[str, ...],
    ) -> int:
        """Prove a rejected config-expansion batch still completed its calls."""

        fields = DailyRequestBudget._snapshot_symbols(params.get("fields"))
        # Fields are not ticker symbols, but the normalization and uniqueness
        # rules are identical and avoid accepting duplicate provenance entries.
        if not requested_symbols or not fields:
            return 0
        expected = len(requested_symbols) * len(fields)
        try:
            recorded_expected = int(
                params[
                    "config_expansion_fetched_initial_baseline_expected_series"
                ]
            )
            observed = int(
                params[
                    "config_expansion_fetched_initial_baseline_observed_series"
                ]
            )
            invalid = int(
                params[
                    "config_expansion_fetched_initial_baseline_invalid_records"
                ]
            )
            duplicates = int(
                params[
                    "config_expansion_fetched_initial_baseline_duplicate_records"
                ]
            )
        except (KeyError, TypeError, ValueError):
            return 0
        validation = params.get(
            "config_expansion_fetched_initial_baseline_validation"
        )
        if (
            validation not in {"passed", "failed"}
            or recorded_expected != expected
            or observed != expected
            or invalid != 0
            or duplicates != 0
        ):
            return 0
        return len(requested_symbols)

    def _legacy_bucket_evidence(
        self,
        connection: sqlite3.Connection,
        *,
        start: datetime,
        end: datetime,
        now: datetime,
    ) -> tuple[datetime, ...]:
        """Reconstruct only calls with auditable post-call upper bounds."""

        columns = self._table_columns(connection, "snapshots")
        required = {
            "source",
            "requested_at",
            "retrieved_at",
            "quality_status",
            "request_params_json",
        }
        if not required.issubset(columns):
            return ()
        has_snapshot_id = "snapshot_id" in columns
        selected = "snapshot_id" if has_snapshot_id else "NULL"
        rows = connection.execute(
            f"""
            SELECT {selected}, requested_at, retrieved_at,
                   quality_status, request_params_json
            FROM snapshots
            WHERE source = ?
            ORDER BY requested_at, retrieved_at
            """,
            (self.provider,),
        ).fetchall()
        evidence: list[datetime] = []
        for (
            raw_snapshot_id,
            raw_requested,
            raw_retrieved,
            raw_quality,
            raw_params,
        ) in rows:
            requested_at = self._legacy_timestamp(
                raw_requested,
                field_name="legacy Alpha snapshot request timestamp",
            )
            retrieved_at = self._legacy_timestamp(
                raw_retrieved,
                field_name="legacy Alpha snapshot retrieval timestamp",
            )
            if retrieved_at < requested_at or retrieved_at > now:
                raise RuntimeError(
                    "persisted Alpha Vantage snapshot timestamp is invalid"
                )
            # A batch crossing a calendar boundary cannot be assigned safely to
            # one timestamp-free bucket. Its count remains an unmatched residual.
            if not (start <= requested_at <= retrieved_at < end):
                continue
            try:
                params = json.loads(str(raw_params))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "persisted Alpha Vantage snapshot provenance is invalid"
                ) from exc
            if not isinstance(params, Mapping):
                raise RuntimeError(
                    "persisted Alpha Vantage snapshot provenance is invalid"
                )
            requested_symbols = self._snapshot_symbols(
                params.get("requested_symbols")
            )
            if not requested_symbols:
                requested_symbols = self._snapshot_symbols(params.get("symbols"))
            snapshot_id = (
                str(raw_snapshot_id) if raw_snapshot_id is not None else None
            )
            if str(raw_quality) == HealthStatus.OK.value:
                exact = self._exact_snapshot_call_times(
                    connection,
                    snapshot_id=snapshot_id,
                    requested_symbols=requested_symbols,
                    requested_at=requested_at,
                    retrieved_at=retrieved_at,
                )
                if exact:
                    evidence.extend(exact)
                elif requested_symbols:
                    # An OK batch proves every requested symbol completed. The
                    # immutable snapshot time is a safe upper bound for all calls.
                    evidence.extend([retrieved_at] * len(requested_symbols))
                continue
            failed_count = self._completed_failed_expansion_count(
                params,
                requested_symbols=requested_symbols,
            )
            if failed_count:
                evidence.extend([retrieved_at] * failed_count)
        return tuple(evidence)

    def _migrate_legacy_buckets(
        self,
        connection: sqlite3.Connection,
        *,
        basis: str,
        now: datetime,
    ) -> None:
        """Convert timestamp-free counters into conservative rolling events.

        For the unversioned UTC and interim New York calendar counters, each
        overlapping bucket is reconciled independently against immutable Alpha
        snapshot provenance. Successful calls use post-response observation
        timestamps when complete, or the enclosing snapshot retrieval time as
        a safe batch upper bound. A rejected config expansion is reconstructible
        only when its exact expected/observed series contract proves the full
        requested batch completed. Every unmatched call is charged at migration
        time. An ambiguous post-migration calendar write is also charged at now.
        """

        rows = connection.execute(
            """
            SELECT usage_day, used, limit_value
            FROM daily_request_budget
            WHERE provider = ?
            ORDER BY usage_day
            """,
            (self.provider,),
        ).fetchall()
        window_start = now - _BUDGET_WINDOW
        migrated_events: list[datetime] = []
        for raw_day, raw_used, raw_limit in rows:
            try:
                usage_day = date.fromisoformat(str(raw_day))
                used = int(raw_used)
                legacy_limit = int(raw_limit)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "persisted Alpha Vantage budget row is invalid"
                ) from exc
            if (
                used < 0
                or legacy_limit < 1
                or legacy_limit > _STANDARD_FREE_LIMIT
                or used > legacy_limit
                or legacy_limit != self.limit
            ):
                raise RuntimeError("persisted Alpha Vantage budget row is invalid")
            if used == 0:
                continue
            if basis == "unknown_recent":
                migrated_events.extend([now] * used)
                continue
            start, end = self._legacy_bucket_interval(usage_day, basis=basis)
            if start > now:
                raise RuntimeError("persisted Alpha Vantage budget day is in the future")
            if end <= window_start:
                # Even a call at the very end of this calendar bucket is no
                # longer inside the strict ``consumed_at > window_start`` window.
                continue
            evidence = self._legacy_bucket_evidence(
                connection,
                start=start,
                end=end,
                now=now,
            )
            if len(evidence) > used:
                raise RuntimeError(
                    "persisted Alpha Vantage snapshot calls exceed budget counter"
                )
            migrated_events.extend(
                timestamp for timestamp in evidence if timestamp > window_start
            )
            # Proven expired calls still account for this bucket's counter. Only
            # the genuinely unaccounted residual is conservatively placed at now.
            migrated_events.extend([now] * (used - len(evidence)))

        if rows:
            connection.execute(
                "DELETE FROM daily_request_budget WHERE provider = ?",
                (self.provider,),
            )
        if migrated_events:
            connection.executemany(
                """
                INSERT INTO request_budget_events(provider, consumed_at)
                VALUES (?, ?)
                """,
                (
                    (self.provider, timestamp.isoformat())
                    for timestamp in migrated_events
                ),
            )

    def _database_active_events(
        self,
        connection: sqlite3.Connection,
        *,
        now: datetime,
    ) -> list[datetime]:
        """Read active events chronologically without ISO-text comparisons."""

        window_start = now - _BUDGET_WINDOW
        rows = connection.execute(
            """
            SELECT consumed_at, event_id
            FROM request_budget_events
            WHERE provider = ?
            """,
            (self.provider,),
        ).fetchall()
        parsed = [
            (
                self._legacy_timestamp(
                    raw_timestamp,
                    field_name="request budget event timestamp",
                ),
                int(event_id),
            )
            for raw_timestamp, event_id in rows
        ]
        parsed.sort(key=lambda item: (item[0], item[1]))
        return [timestamp for timestamp, _event_id in parsed if timestamp > window_start]

    def _touch_memory_credits(self, credit_ids: tuple[int, ...]) -> bool:
        now = self._now()
        window_start = now - _BUDGET_WINDOW
        with self._lock:
            self._usage = {
                credit_id: timestamp
                for credit_id, timestamp in self._usage.items()
                if timestamp > window_start
            }
            missing = tuple(
                credit_id for credit_id in credit_ids if credit_id not in self._usage
            )
            if missing:
                # An expired reservation may re-enter only if its calls still
                # fit after other processes/credits reused that capacity.
                if len(self._usage) + len(missing) > self.limit:
                    return False
            for credit_id in credit_ids:
                previous = self._usage.get(credit_id)
                self._usage[credit_id] = max(previous, now) if previous else now
            return len(self._usage) <= self.limit

    def _touch_database_credits(self, credit_ids: tuple[int, ...]) -> bool:
        now = self._now()
        placeholders = ",".join("?" for _item in credit_ids)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._migrate_legacy_buckets(
                connection,
                basis="unknown_recent",
                now=now,
            )
            rows = connection.execute(
                f"""
                SELECT event_id, consumed_at
                FROM request_budget_events
                WHERE provider = ? AND event_id IN ({placeholders})
                """,
                (self.provider, *credit_ids),
            ).fetchall()
            by_id = {
                int(event_id): self._legacy_timestamp(
                    consumed_at,
                    field_name="reserved request budget event timestamp",
                )
                for event_id, consumed_at in rows
            }
            if set(by_id) != set(credit_ids):
                connection.rollback()
                raise RuntimeError("reserved Alpha Vantage budget credit is missing")
            active = self._database_active_events(connection, now=now)
            expired_selected = sum(
                timestamp <= now - _BUDGET_WINDOW for timestamp in by_id.values()
            )
            if len(active) + expired_selected > self.limit:
                connection.commit()
                return False
            connection.executemany(
                """
                UPDATE request_budget_events
                SET consumed_at = ?
                WHERE provider = ? AND event_id = ?
                """,
                (
                    (max(by_id[event_id], now).isoformat(), self.provider, event_id)
                    for event_id in credit_ids
                ),
            )
            connection.commit()
            return True

    @property
    def remaining(self) -> int:
        now = self._now()
        window_start = now - _BUDGET_WINDOW
        with self._lock:
            if self.database_path is not None:
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    self._migrate_legacy_buckets(
                        connection,
                        basis="unknown_recent",
                        now=now,
                    )
                    used = len(self._database_active_events(connection, now=now))
                    connection.commit()
            else:
                self._usage = {
                    credit_id: timestamp
                    for credit_id, timestamp in self._usage.items()
                    if timestamp > window_start
                }
                used = len(self._usage)
            return max(0, self.limit - used)

    def next_available_at(self, units: int = 1) -> datetime:
        """Return the earliest instant when an entire batch can be reserved.

        This is an advisory calculation over the same strict rolling-window
        predicate used by :meth:`reserve`; it does not reserve calls. The
        caller must still reserve atomically, and each prepaid credit is touched
        immediately before transport, so this advisory timestamp can never
        bypass the hard cap.
        """

        if units < 1:
            raise ValueError("units must be positive")
        if units > self.limit:
            raise ValueError("units must not exceed the request limit")
        now = self._now()
        window_start = now - _BUDGET_WINDOW
        with self._lock:
            if self.database_path is None:
                self._usage = {
                    credit_id: timestamp
                    for credit_id, timestamp in self._usage.items()
                    if timestamp > window_start
                }
                active = sorted(self._usage.values())
            else:
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    self._migrate_legacy_buckets(
                        connection,
                        basis="unknown_recent",
                        now=now,
                    )
                    active = self._database_active_events(connection, now=now)
                    connection.commit()
            excess = len(active) + units - self.limit
            if excess <= 0:
                return now
            return active[excess - 1] + _BUDGET_WINDOW

    def reserve(self, units: int = 1) -> ReservedRequestBudget | None:
        """Atomically charge an entire batch before any provider transport."""

        if units < 1:
            raise ValueError("units must be positive")
        if units > self.limit:
            return None
        now = self._now()
        window_start = now - _BUDGET_WINDOW
        with self._lock:
            if self.database_path is None:
                self._usage = {
                    credit_id: timestamp
                    for credit_id, timestamp in self._usage.items()
                    if timestamp > window_start
                }
                if len(self._usage) + units > self.limit:
                    return None
                credit_ids = tuple(
                    range(
                        self._next_memory_credit_id,
                        self._next_memory_credit_id + units,
                    )
                )
                self._next_memory_credit_id += units
                self._usage.update({credit_id: now for credit_id in credit_ids})
                return ReservedRequestBudget(
                    credit_ids,
                    touch=self._touch_memory_credits,
                )

            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                # During a controlled upgrade, fold every legacy counter write
                # already present into this locked transaction before checking
                # capacity. Operators must stop old binaries first: a legacy
                # write made only after this commit cannot share this lock.
                self._migrate_legacy_buckets(
                    connection,
                    basis="unknown_recent",
                    now=now,
                )
                used = len(self._database_active_events(connection, now=now))
                if used + units > self.limit:
                    # Persist any reconciled legacy writes even though this
                    # reservation failed. That keeps ``next_available_at`` and
                    # later processes on the same authoritative event ledger.
                    connection.commit()
                    return None
                credit_ids: list[int] = []
                for _item in range(units):
                    cursor = connection.execute(
                        """
                        INSERT INTO request_budget_events(provider, consumed_at)
                        VALUES (?, ?)
                        """,
                        (self.provider, now.isoformat()),
                    )
                    if cursor.lastrowid is None:
                        raise RuntimeError(
                            "failed to persist Alpha Vantage budget reservation"
                        )
                    credit_ids.append(int(cursor.lastrowid))
                connection.commit()
                return ReservedRequestBudget(
                    credit_ids,
                    touch=self._touch_database_credits,
                )

    def consume(self, units: int = 1) -> bool:
        """Atomically charge calls without retaining prepaid credits."""

        return self.reserve(units) is not None


class _BudgetExhausted(RuntimeError):
    pass


class _BudgetedTransport:
    def __init__(self, delegate: JsonTransport, budget: RequestBudget) -> None:
        self.delegate = delegate
        self.budget = budget

    def get_json(
        self,
        url: str,
        params: Mapping[str, Any],
        *,
        timeout: float,
    ) -> Mapping[str, Any]:
        if not self.budget.consume():
            raise _BudgetExhausted(
                "Alpha Vantage rolling 24-hour request budget exhausted"
            )
        return self.delegate.get_json(url, params, timeout=timeout)


@dataclass(frozen=True, slots=True)
class AlphaVantageConfig:
    api_key: str | None = field(repr=False)
    base_url: str = "https://www.alphavantage.co/query"
    timeout_seconds: float = 30.0
    license_class: str = "alpha_vantage_private_research"
    market_available_time_et: time = time(16, 15)
    request_spacing_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.request_spacing_seconds < 0:
            raise ValueError("request_spacing_seconds must be non-negative")

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> "AlphaVantageConfig":
        env = os.environ if environ is None else environ
        return cls(api_key=env.get(ALPHA_VANTAGE_API_KEY_ENV) or None, **kwargs)


class AlphaVantageClient:
    def __init__(
        self,
        config: AlphaVantageConfig,
        *,
        transport: JsonTransport | None = None,
        budget: RequestBudget | None = None,
        retry: RetryPolicy | None = None,
        sleeper: Callable[[float], None] | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.config = config
        self.budget = budget or DailyRequestBudget()
        self.transport = _BudgetedTransport(transport or UrllibJsonTransport(), self.budget)
        self.retry = retry or RetryPolicy()
        self.sleeper = sleeper or __import__("time").sleep
        self.clock = clock

    def fetch_weekly_adjusted(
        self,
        symbols: Sequence[str],
        *,
        cutoff: datetime,
        fields: Sequence[str] = ("adjusted_close",),
        observation_start: date | None = None,
    ) -> CollectionResult:
        cutoff = ensure_utc(cutoff, field_name="cutoff")
        clean_symbols = tuple(dict.fromkeys(item.strip().upper() for item in symbols if item.strip()))
        clean_fields = tuple(dict.fromkeys(fields))
        if not clean_symbols:
            raise ValueError("symbols must not be empty")
        unknown_fields = set(clean_fields) - set(_FIELD_MAP)
        if unknown_fields or not clean_fields:
            raise ValueError(f"unsupported weekly fields: {sorted(unknown_fields)}")
        if not self.config.api_key:
            return CollectionResult(
                health=HealthStatus.DEGRADED,
                issues=(f"{ALPHA_VANTAGE_API_KEY_ENV} is not configured",),
            )

        records: list[Observation] = []
        issues: list[str] = []
        attempts = 0
        requests_made = 0
        schema_changed = False

        for symbol_index, symbol in enumerate(clean_symbols):
            params = {
                "function": "TIME_SERIES_WEEKLY_ADJUSTED",
                "symbol": symbol,
                "apikey": self.config.api_key,
                "datatype": "json",
            }
            try:
                payload, used_attempts = request_json_with_retry(
                    self.transport,
                    self.config.base_url,
                    params,
                    timeout=self.config.timeout_seconds,
                    retry=self.retry,
                    sleeper=self.sleeper,
                    secrets=(self.config.api_key,),
                )
            except _BudgetExhausted:
                return CollectionResult(
                    records=tuple(records),
                    health=HealthStatus.QUOTA_EXHAUSTED,
                    issues=tuple(
                        issues
                        + ["Alpha Vantage rolling 24-hour request budget exhausted"]
                    ),
                    requests_made=requests_made,
                    attempts=attempts,
                )
            except ProviderRequestError as exc:
                attempts += exc.attempts
                status = (
                    HealthStatus.QUOTA_EXHAUSTED
                    if exc.status_code == 429
                    else HealthStatus.DEGRADED
                )
                return CollectionResult(
                    records=tuple(records),
                    health=status,
                    issues=tuple(issues + [f"Alpha Vantage request failed: {exc}"]),
                    requests_made=requests_made,
                    attempts=attempts,
                )

            attempts += used_attempts
            requests_made += 1
            if (
                self.config.request_spacing_seconds
                and symbol_index < len(clean_symbols) - 1
            ):
                self.sleeper(self.config.request_spacing_seconds)
            if "Note" in payload or "Information" in payload:
                return CollectionResult(
                    records=tuple(records),
                    health=HealthStatus.QUOTA_EXHAUSTED,
                    issues=tuple(issues + ["Alpha Vantage reported a quota or entitlement limit"]),
                    requests_made=requests_made,
                    attempts=attempts,
                )
            if "Error Message" in payload:
                issues.append(f"Alpha Vantage rejected symbol {symbol}")
                continue
            weekly = payload.get("Weekly Adjusted Time Series")
            if not isinstance(weekly, Mapping):
                schema_changed = True
                issues.append(f"Alpha Vantage weekly schema changed for {symbol}")
                continue
            # Timestamp after the response is received: later full-response
            # diffs use this as the earliest defensible discovery time.
            retrieved_at = ensure_utc(self.clock(), field_name="clock")

            for raw_period, raw_row in weekly.items():
                if not isinstance(raw_row, Mapping):
                    schema_changed = True
                    continue
                try:
                    period_end = date.fromisoformat(str(raw_period))
                except ValueError:
                    schema_changed = True
                    continue
                if observation_start is not None and period_end < observation_start:
                    continue
                available_at = datetime.combine(
                    period_end,
                    self.config.market_available_time_et,
                    tzinfo=_EASTERN,
                ).astimezone(timezone.utc)
                if available_at > cutoff:
                    continue
                row_hash = hashlib.sha256(
                    json.dumps(raw_row, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                for field_name in clean_fields:
                    provider_field = _FIELD_MAP[field_name]
                    try:
                        value = float(raw_row[provider_field])
                    except (KeyError, TypeError, ValueError):
                        schema_changed = True
                        continue
                    records.append(
                        Observation(
                            source="alpha_vantage",
                            series_id=f"{symbol}.{field_name}",
                            observed_period_end=period_end,
                            value=value,
                            released_at=available_at,
                            available_at=available_at,
                            vintage_date=period_end,
                            retrieved_at=retrieved_at,
                            units=_FIELD_UNITS[field_name],
                            adjustment="weekly_adjusted" if field_name == "adjusted_close" else "weekly",
                            license_class=self.config.license_class,
                            quality_status=HealthStatus.OK,
                            raw_sha256=row_hash,
                            metadata={
                                "symbol": symbol,
                                "field": field_name,
                                "provider_full_response": True,
                                "snapshot_retrieved_at": retrieved_at.isoformat(),
                                "pit_revision_policy": "prospective_on_later_diff",
                                "historical_vintage_note": (
                                    "provider supplies current adjusted history; later changes are not backdated"
                                ),
                            },
                        )
                    )

        records.sort(key=lambda item: (item.series_id, item.observed_period_end))
        health = HealthStatus.SCHEMA_CHANGED if schema_changed else (
            HealthStatus.DEGRADED if issues else HealthStatus.OK
        )
        return CollectionResult(
            records=tuple(records),
            health=health,
            issues=tuple(dict.fromkeys(issues)),
            requests_made=requests_made,
            attempts=attempts,
        )

    collect_weekly_adjusted = fetch_weekly_adjusted
