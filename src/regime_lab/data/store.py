"""Immutable SQLite snapshots for provider payload provenance and PIT replay."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Any, Iterable, Sequence
from uuid import uuid4
from zoneinfo import ZoneInfo

from .contracts import (
    HealthStatus,
    Observation,
    SnapshotMode,
    SnapshotProvenance,
    ensure_utc,
    normalize_revision_sequences,
)
from .security import sanitize_mapping


_EASTERN = ZoneInfo("America/New_York")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    dataset TEXT NOT NULL,
    cutoff TEXT NOT NULL,
    requested_at TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    quality_status TEXT NOT NULL,
    license_class TEXT NOT NULL,
    request_params_json TEXT NOT NULL,
    response_sha256 TEXT NOT NULL,
    issues_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS observations (
    snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id),
    source TEXT NOT NULL,
    series_id TEXT NOT NULL,
    observed_period_end TEXT NOT NULL,
    released_at TEXT,
    available_at TEXT NOT NULL,
    vintage_date TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    revision_seq INTEGER NOT NULL,
    value REAL,
    units TEXT NOT NULL,
    adjustment TEXT NOT NULL,
    license_class TEXT NOT NULL,
    quality_status TEXT NOT NULL,
    raw_sha256 TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    PRIMARY KEY (
        snapshot_id,
        source,
        series_id,
        observed_period_end,
        vintage_date,
        revision_seq,
        retrieved_at
    )
);

CREATE INDEX IF NOT EXISTS observations_asof_idx
ON observations(source, series_id, available_at, observed_period_end);

CREATE TABLE IF NOT EXISTS daily_request_budget (
    provider TEXT NOT NULL,
    usage_day TEXT NOT NULL,
    used INTEGER NOT NULL,
    limit_value INTEGER NOT NULL,
    PRIMARY KEY (provider, usage_day)
);
"""


class SQLiteSnapshotStore:
    """Transactional append-only snapshot store.

    The store owns one connection so ``:memory:`` behaves predictably and all
    writes are serialized.  Provider request parameters are recursively
    sanitized before they cross the persistence boundary.
    """

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self.path = str(path)
        self.read_only = bool(read_only)
        if self.read_only and self.path == ":memory:":
            raise ValueError("an in-memory snapshot store cannot be read-only")
        connect_target = self.path
        connect_uri = False
        if self.path != ":memory:":
            selected = Path(self.path).expanduser()
            if self.read_only:
                if selected.is_symlink() or not selected.is_file():
                    raise FileNotFoundError(
                        "read-only snapshot database must be an existing regular file"
                    )
                resolved = selected.resolve(strict=True)
                connect_target = f"{resolved.as_uri()}?mode=ro"
                connect_uri = True
            else:
                selected.resolve().parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            connect_target,
            timeout=30,
            check_same_thread=False,
            uri=connect_uri,
        )
        if self.path != ":memory:" and not self.read_only:
            os.chmod(self.path, 0o600)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA busy_timeout = 30000")
        if self.read_only:
            self._connection.execute("PRAGMA query_only = ON")
        elif self.path != ":memory:":
            self._connection.execute("PRAGMA journal_mode = WAL")
        self._lock = RLock()
        if not self.read_only:
            with self._connection:
                self._connection.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "SQLiteSnapshotStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def write_snapshot(
        self,
        records: Iterable[Observation],
        provenance: SnapshotProvenance,
    ) -> str:
        records = tuple(records)
        snapshot_id = provenance.snapshot_id or str(uuid4())
        safe_params = sanitize_mapping(provenance.request_params)
        safe_issues = sanitize_mapping(provenance.issues)
        snapshot_values = (
            snapshot_id,
            provenance.source,
            provenance.dataset,
            provenance.cutoff.isoformat(),
            provenance.requested_at.isoformat(),
            provenance.retrieved_at.isoformat(),
            provenance.quality_status.value,
            provenance.license_class,
            json.dumps(safe_params, sort_keys=True, separators=(",", ":"), default=str),
            provenance.response_sha256,
            json.dumps(safe_issues, ensure_ascii=False),
        )
        observation_values = [
            (
                snapshot_id,
                record.source,
                record.series_id,
                record.observed_period_end.isoformat(),
                record.released_at.isoformat() if record.released_at else None,
                record.available_at.isoformat(),
                record.vintage_date.isoformat(),
                record.retrieved_at.isoformat(),
                record.revision_seq,
                record.value,
                record.units,
                record.adjustment,
                record.license_class,
                record.quality_status.value,
                record.raw_sha256,
                json.dumps(
                    sanitize_mapping(record.metadata),
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ),
            )
            for record in records
        ]
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO snapshots(
                    snapshot_id, source, dataset, cutoff, requested_at,
                    retrieved_at, quality_status, license_class,
                    request_params_json, response_sha256, issues_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                snapshot_values,
            )
            self._connection.executemany(
                """
                INSERT INTO observations(
                    snapshot_id, source, series_id, observed_period_end,
                    released_at, available_at, vintage_date, retrieved_at,
                    revision_seq, value, units, adjustment, license_class,
                    quality_status, raw_sha256, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                observation_values,
            )
        return snapshot_id

    append_snapshot = write_snapshot

    def get_provenance(self, snapshot_id: str) -> SnapshotProvenance | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM snapshots WHERE snapshot_id = ?",
                (snapshot_id,),
            ).fetchone()
        if row is None:
            return None
        return SnapshotProvenance(
            snapshot_id=row["snapshot_id"],
            source=row["source"],
            dataset=row["dataset"],
            cutoff=datetime.fromisoformat(row["cutoff"]),
            requested_at=datetime.fromisoformat(row["requested_at"]),
            retrieved_at=datetime.fromisoformat(row["retrieved_at"]),
            quality_status=HealthStatus(row["quality_status"]),
            license_class=row["license_class"],
            request_params=json.loads(row["request_params_json"]),
            response_sha256=row["response_sha256"],
            issues=tuple(json.loads(row["issues_json"])),
        )

    def read_observations(
        self,
        *,
        snapshot_id: str | None = None,
        source: str | None = None,
        series_ids: Sequence[str] | None = None,
        available_as_of: datetime | None = None,
    ) -> tuple[Observation, ...]:
        clauses: list[str] = []
        values: list[Any] = []
        if snapshot_id is not None:
            clauses.append("snapshot_id = ?")
            values.append(snapshot_id)
        if source is not None:
            clauses.append("source = ?")
            values.append(source)
        if series_ids:
            unique = tuple(dict.fromkeys(series_ids))
            clauses.append(f"series_id IN ({','.join('?' for _ in unique)})")
            values.extend(unique)
        if available_as_of is not None:
            available_as_of = ensure_utc(available_as_of, field_name="available_as_of")
            clauses.append("available_at <= ?")
            values.append(available_as_of.isoformat())
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        query = (
            "SELECT * FROM observations"
            + where
            + " ORDER BY source, series_id, observed_period_end, available_at, revision_seq"
        )
        with self._lock:
            rows = self._connection.execute(query, values).fetchall()
        return tuple(self._row_to_observation(row) for row in rows)

    def read_last_good_observations(
        self,
        *,
        source: str | None = None,
        dataset: str | None = None,
        series_ids: Sequence[str] | None = None,
        available_as_of: datetime | None = None,
    ) -> tuple[Observation, ...]:
        """Assemble the latest successful full snapshot and subsequent deltas.

        Legacy snapshots have no ``snapshot_mode`` and are treated as full.
        Failed attempts never enter the chain.  A later successful full snapshot
        compacts/replaces the older chain.  Alpha rows whose observation period
        was beyond the cutoff of the snapshot that stored them are permanently
        quarantined; later cutoffs cannot make a legacy partial week reappear.
        Inclusive-overlap events are then deduplicated, with the later
        successful snapshot winning, and revision sequences normalized globally.
        """

        clauses = ["quality_status = ?"]
        values: list[Any] = [HealthStatus.OK.value]
        if source is not None:
            clauses.append("source = ?")
            values.append(source)
        if dataset is not None:
            clauses.append("dataset = ?")
            values.append(dataset)
        with self._lock:
            snapshot_rows = self._connection.execute(
                f"""
                SELECT * FROM snapshots
                WHERE {' AND '.join(clauses)}
                ORDER BY source, dataset, retrieved_at, snapshot_id
                """,
                values,
            ).fetchall()

        grouped: dict[tuple[str, str], list[sqlite3.Row]] = {}
        for row in snapshot_rows:
            grouped.setdefault((row["source"], row["dataset"]), []).append(row)

        selected_snapshots: list[sqlite3.Row] = []
        for group in grouped.values():
            last_full_index: int | None = None
            for index, row in enumerate(group):
                params = json.loads(row["request_params_json"])
                raw_mode = params.get("snapshot_mode", SnapshotMode.FULL.value)
                try:
                    mode = SnapshotMode(str(raw_mode))
                except ValueError as exc:
                    raise ValueError(
                        f"unsupported snapshot_mode for {row['source']}/{row['dataset']}"
                    ) from exc
                if mode is SnapshotMode.FULL:
                    last_full_index = index
            # A delta without a successful base is incomplete and therefore
            # deliberately unavailable rather than silently treated as full.
            if last_full_index is not None:
                selected_snapshots.extend(group[last_full_index:])

        selected_snapshots.sort(
            key=lambda row: (
                row["source"],
                row["dataset"],
                row["retrieved_at"],
                row["snapshot_id"],
            )
        )
        if not selected_snapshots:
            return ()

        allowed_series = set(series_ids) if series_ids else None
        as_of_iso: str | None = None
        if available_as_of is not None:
            as_of_iso = ensure_utc(
                available_as_of,
                field_name="available_as_of",
            ).isoformat()

        # Later successful deltas replace their overlap copy.  Vintage date and
        # availability timestamp identify a provider event: ALFRED overlap
        # replays share both, while Alpha can have multiple prospectively
        # discovered revisions on the same calendar date.  Raw hash is
        # intentionally excluded so a corrected replay supersedes its copy.
        deduplicated: dict[
            tuple[str, str, str, str, str],
            Observation,
        ] = {}
        batch_size = 200
        for batch_start in range(0, len(selected_snapshots), batch_size):
            batch = selected_snapshots[batch_start : batch_start + batch_size]
            value_rows = ",".join("(?, ?, ?)" for _ in batch)
            query_values: list[Any] = []
            for global_rank, snapshot in enumerate(
                batch,
                start=batch_start,
            ):
                query_values.extend(
                    (
                        snapshot["snapshot_id"],
                        global_rank,
                        snapshot["cutoff"],
                    )
                )
            as_of_clause = ""
            if as_of_iso is not None:
                as_of_clause = "WHERE o.available_at <= ?"
                query_values.append(as_of_iso)
            with self._lock:
                observation_rows = self._connection.execute(
                    f"""
                    WITH selected(snapshot_id, snapshot_rank, snapshot_cutoff) AS (
                        VALUES {value_rows}
                    )
                    SELECT o.*, selected.snapshot_rank, selected.snapshot_cutoff
                    FROM observations AS o
                    JOIN selected ON selected.snapshot_id = o.snapshot_id
                    {as_of_clause}
                    ORDER BY selected.snapshot_rank, o.source, o.series_id,
                             o.observed_period_end, o.vintage_date,
                             o.available_at, o.revision_seq
                    """,
                    query_values,
                ).fetchall()
            for row in observation_rows:
                if allowed_series is not None and row["series_id"] not in allowed_series:
                    continue
                record = self._row_to_observation(row)
                if record.source == "alpha_vantage":
                    snapshot_cutoff_date = datetime.fromisoformat(
                        row["snapshot_cutoff"]
                    ).astimezone(_EASTERN).date()
                    if record.observed_period_end > snapshot_cutoff_date:
                        continue
                identity = (
                    record.source,
                    record.series_id,
                    record.observed_period_end.isoformat(),
                    record.vintage_date.isoformat(),
                    record.available_at.isoformat(),
                )
                deduplicated[identity] = record

        return normalize_revision_sequences(deduplicated.values())

    def get_last_good_provenance(
        self,
        *,
        source: str,
        dataset: str,
    ) -> SnapshotProvenance | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT snapshot_id
                FROM snapshots
                WHERE source = ? AND dataset = ? AND quality_status = ?
                ORDER BY retrieved_at DESC, snapshot_id DESC
                LIMIT 1
                """,
                (source, dataset, HealthStatus.OK.value),
            ).fetchone()
        return self.get_provenance(row["snapshot_id"]) if row is not None else None

    def list_provenance(self, *, source: str | None = None) -> tuple[SnapshotProvenance, ...]:
        query = "SELECT snapshot_id FROM snapshots"
        values: tuple[str, ...] = ()
        if source is not None:
            query += " WHERE source = ?"
            values = (source,)
        query += " ORDER BY retrieved_at, snapshot_id"
        with self._lock:
            ids = [row[0] for row in self._connection.execute(query, values).fetchall()]
        return tuple(
            item
            for item in (self.get_provenance(snapshot_id) for snapshot_id in ids)
            if item is not None
        )

    @staticmethod
    def _row_to_observation(row: sqlite3.Row) -> Observation:
        return Observation(
            source=row["source"],
            series_id=row["series_id"],
            observed_period_end=datetime.fromisoformat(row["observed_period_end"]).date(),
            released_at=(
                datetime.fromisoformat(row["released_at"])
                if row["released_at"] is not None
                else None
            ),
            available_at=datetime.fromisoformat(row["available_at"]),
            vintage_date=datetime.fromisoformat(row["vintage_date"]).date(),
            retrieved_at=datetime.fromisoformat(row["retrieved_at"]),
            revision_seq=int(row["revision_seq"]),
            value=float(row["value"]) if row["value"] is not None else None,
            units=row["units"],
            adjustment=row["adjustment"],
            license_class=row["license_class"],
            quality_status=HealthStatus(row["quality_status"]),
            raw_sha256=row["raw_sha256"],
            metadata=json.loads(row["metadata_json"]),
        )
