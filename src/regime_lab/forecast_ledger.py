"""Append-only operational forecast ledger.

The research artifacts can be regenerated as data vintages improve.  This
ledger records what an operating process actually knew and emitted at one
decision instant, so later reconstruction cannot overwrite prospective OOS
evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Any

from regime_lab.data.asof import AsOfValue
from regime_lab.data.contracts import Observation, ensure_utc
from regime_lab.integrity import canonical_json_sha256_v1


UTC = timezone.utc
_SHA256_LENGTH = 64


class ForecastLedgerError(RuntimeError):
    """Base error for immutable forecast-ledger violations."""


class DuplicateForecastError(ForecastLedgerError):
    """Raised when an identical ledger key and content already exist."""


class ConflictingForecastError(ForecastLedgerError):
    """Raised when an existing ledger key is presented with different content."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _sha256(value: str, *, field_name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return normalized


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("forecast ledger content must be finite JSON data") from exc


def _json_sha256(value: Any) -> str:
    return canonical_json_sha256_v1(value)


@dataclass(frozen=True, slots=True)
class OperationalInput:
    """One exact provider revision used by an operational forecast."""

    source: str
    series_id: str
    observed_period_end: date
    source_released_at: datetime | None
    provider_first_seen_at: datetime
    system_retrieved_at: datetime
    revision_seq: int
    raw_sha256: str

    def __post_init__(self) -> None:
        if not self.source.strip() or not self.series_id.strip():
            raise ValueError("operational input source and series_id must be non-empty")
        if isinstance(self.observed_period_end, datetime) or not isinstance(
            self.observed_period_end, date
        ):
            raise TypeError("observed_period_end must be a date")
        source_released_at = (
            ensure_utc(self.source_released_at, field_name="source_released_at")
            if self.source_released_at is not None
            else None
        )
        provider_first_seen_at = ensure_utc(
            self.provider_first_seen_at,
            field_name="provider_first_seen_at",
        )
        system_retrieved_at = ensure_utc(
            self.system_retrieved_at,
            field_name="system_retrieved_at",
        )
        if provider_first_seen_at > system_retrieved_at:
            raise ValueError(
                "provider_first_seen_at must not be after system_retrieved_at"
            )
        if self.revision_seq < 0:
            raise ValueError("revision_seq must be non-negative")
        object.__setattr__(self, "source", self.source.strip())
        object.__setattr__(self, "series_id", self.series_id.strip())
        object.__setattr__(self, "source_released_at", source_released_at)
        object.__setattr__(self, "provider_first_seen_at", provider_first_seen_at)
        object.__setattr__(self, "system_retrieved_at", system_retrieved_at)
        object.__setattr__(
            self,
            "raw_sha256",
            _sha256(self.raw_sha256, field_name="raw_sha256"),
        )

    @property
    def operating_available_at(self) -> datetime:
        candidates = [self.provider_first_seen_at]
        if self.source_released_at is not None:
            candidates.append(self.source_released_at)
        return max(candidates)

    @classmethod
    def from_observation(cls, observation: Observation) -> "OperationalInput":
        assert observation.provider_first_seen_at is not None
        assert observation.system_retrieved_at is not None
        return cls(
            source=observation.source,
            series_id=observation.series_id,
            observed_period_end=observation.observed_period_end,
            source_released_at=observation.source_released_at,
            provider_first_seen_at=observation.provider_first_seen_at,
            system_retrieved_at=observation.system_retrieved_at,
            revision_seq=observation.revision_seq,
            raw_sha256=observation.raw_sha256,
        )

    @classmethod
    def from_asof_value(cls, value: AsOfValue) -> "OperationalInput":
        """Preserve the exact revision selected by an operational as-of join."""

        required = {
            "observed_period_end": value.observed_period_end,
            "provider_first_seen_at": value.provider_first_seen_at,
            "system_retrieved_at": value.system_retrieved_at,
            "revision_seq": value.revision_seq,
            "raw_sha256": value.raw_sha256,
        }
        missing = sorted(name for name, item in required.items() if item is None)
        if missing:
            raise ValueError(
                "as-of value lacks an operational vintage: " + ", ".join(missing)
            )
        return cls(
            source=value.source,
            series_id=value.series_id,
            observed_period_end=value.observed_period_end,
            source_released_at=value.source_released_at,
            provider_first_seen_at=value.provider_first_seen_at,
            system_retrieved_at=value.system_retrieved_at,
            revision_seq=value.revision_seq,
            raw_sha256=value.raw_sha256,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "series_id": self.series_id,
            "observed_period_end": self.observed_period_end.isoformat(),
            "source_released_at": (
                self.source_released_at.isoformat()
                if self.source_released_at is not None
                else None
            ),
            "provider_first_seen_at": self.provider_first_seen_at.isoformat(),
            "system_retrieved_at": self.system_retrieved_at.isoformat(),
            "revision_seq": self.revision_seq,
            "raw_sha256": self.raw_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OperationalInput":
        raw_source_release = value.get("source_released_at")
        return cls(
            source=str(value["source"]),
            series_id=str(value["series_id"]),
            observed_period_end=date.fromisoformat(str(value["observed_period_end"])),
            source_released_at=(
                datetime.fromisoformat(str(raw_source_release))
                if raw_source_release is not None
                else None
            ),
            provider_first_seen_at=datetime.fromisoformat(
                str(value["provider_first_seen_at"])
            ),
            system_retrieved_at=datetime.fromisoformat(
                str(value["system_retrieved_at"])
            ),
            revision_seq=int(value["revision_seq"]),
            raw_sha256=str(value["raw_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class ForecastLedgerKey:
    origin_week: date
    decision_at: datetime
    target_at: datetime
    label_spec_sha256: str
    model_manifest_sha256: str
    input_snapshot_sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.origin_week, datetime) or not isinstance(
            self.origin_week, date
        ):
            raise TypeError("origin_week must be a date")
        decision_at = ensure_utc(self.decision_at, field_name="decision_at")
        target_at = ensure_utc(self.target_at, field_name="target_at")
        if decision_at >= target_at:
            raise ValueError("decision_at must be strictly before target_at")
        if self.origin_week > decision_at.date():
            raise ValueError("origin_week must not be after decision_at")
        object.__setattr__(self, "decision_at", decision_at)
        object.__setattr__(self, "target_at", target_at)
        object.__setattr__(
            self,
            "label_spec_sha256",
            _sha256(self.label_spec_sha256, field_name="label_spec_sha256"),
        )
        object.__setattr__(
            self,
            "model_manifest_sha256",
            _sha256(
                self.model_manifest_sha256,
                field_name="model_manifest_sha256",
            ),
        )
        object.__setattr__(
            self,
            "input_snapshot_sha256",
            _sha256(
                self.input_snapshot_sha256,
                field_name="input_snapshot_sha256",
            ),
        )

    def as_sql_tuple(self) -> tuple[str, ...]:
        return (
            self.origin_week.isoformat(),
            self.decision_at.isoformat(),
            self.target_at.isoformat(),
            self.label_spec_sha256,
            self.model_manifest_sha256,
            self.input_snapshot_sha256,
        )


@dataclass(frozen=True, slots=True)
class ForecastLedgerEntry:
    """Immutable forecast plus the exact operational input revisions it used."""

    origin_week: date
    decision_at: datetime
    target_at: datetime
    label_spec_sha256: str
    model_manifest_sha256: str
    input_snapshot_sha256: str
    operational_inputs: tuple[OperationalInput, ...]
    forecast: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.origin_week, datetime) or not isinstance(
            self.origin_week, date
        ):
            raise TypeError("origin_week must be a date")
        decision_at = ensure_utc(self.decision_at, field_name="decision_at")
        target_at = ensure_utc(self.target_at, field_name="target_at")
        if decision_at >= target_at:
            raise ValueError("decision_at must be strictly before target_at")
        if self.origin_week > decision_at.date():
            raise ValueError("origin_week must not be after decision_at")

        inputs = tuple(
            sorted(
                self.operational_inputs,
                key=lambda item: (
                    item.source,
                    item.series_id,
                    item.observed_period_end,
                    item.revision_seq,
                    item.raw_sha256,
                ),
            )
        )
        if not inputs:
            raise ValueError("operational_inputs must not be empty")
        identities = {
            (
                item.source,
                item.series_id,
                item.observed_period_end,
                item.revision_seq,
                item.raw_sha256,
            )
            for item in inputs
        }
        if len(identities) != len(inputs):
            raise ValueError("operational_inputs contain duplicate revisions")
        for item in inputs:
            if item.observed_period_end > self.origin_week:
                raise ValueError("operational input period must not exceed origin_week")
            if item.provider_first_seen_at > decision_at:
                raise ValueError(
                    "operational input provider_first_seen_at exceeds decision_at"
                )
            if item.operating_available_at > decision_at:
                raise ValueError(
                    "operational input source finalization exceeds decision_at"
                )
            if item.system_retrieved_at > decision_at:
                raise ValueError(
                    "operational input system_retrieved_at exceeds decision_at"
                )

        if not isinstance(self.forecast, Mapping) or not self.forecast:
            raise ValueError("forecast must be a non-empty mapping")
        normalized_forecast = json.loads(_canonical_json(dict(self.forecast)))
        object.__setattr__(self, "decision_at", decision_at)
        object.__setattr__(self, "target_at", target_at)
        object.__setattr__(
            self,
            "label_spec_sha256",
            _sha256(self.label_spec_sha256, field_name="label_spec_sha256"),
        )
        object.__setattr__(
            self,
            "model_manifest_sha256",
            _sha256(
                self.model_manifest_sha256,
                field_name="model_manifest_sha256",
            ),
        )
        object.__setattr__(
            self,
            "input_snapshot_sha256",
            _sha256(
                self.input_snapshot_sha256,
                field_name="input_snapshot_sha256",
            ),
        )
        object.__setattr__(self, "operational_inputs", inputs)
        object.__setattr__(self, "forecast", normalized_forecast)

    @property
    def key(self) -> ForecastLedgerKey:
        return ForecastLedgerKey(
            origin_week=self.origin_week,
            decision_at=self.decision_at,
            target_at=self.target_at,
            label_spec_sha256=self.label_spec_sha256,
            model_manifest_sha256=self.model_manifest_sha256,
            input_snapshot_sha256=self.input_snapshot_sha256,
        )

    @property
    def forecast_sha256(self) -> str:
        return _json_sha256(self.forecast)

    @property
    def operational_inputs_sha256(self) -> str:
        return _json_sha256([item.as_dict() for item in self.operational_inputs])


_LEDGER_SCHEMA = """
CREATE TABLE IF NOT EXISTS forecast_ledger (
    origin_week TEXT NOT NULL,
    decision_at TEXT NOT NULL,
    target_at TEXT NOT NULL,
    label_spec_sha256 TEXT NOT NULL,
    model_manifest_sha256 TEXT NOT NULL,
    input_snapshot_sha256 TEXT NOT NULL,
    operational_inputs_json TEXT NOT NULL,
    operational_inputs_sha256 TEXT NOT NULL,
    forecast_json TEXT NOT NULL,
    forecast_sha256 TEXT NOT NULL,
    inserted_at TEXT NOT NULL,
    PRIMARY KEY (
        origin_week,
        decision_at,
        target_at,
        label_spec_sha256,
        model_manifest_sha256,
        input_snapshot_sha256
    )
);

CREATE TRIGGER IF NOT EXISTS forecast_ledger_no_update
BEFORE UPDATE ON forecast_ledger
BEGIN
    SELECT RAISE(ABORT, 'forecast ledger is append-only');
END;

CREATE TRIGGER IF NOT EXISTS forecast_ledger_no_delete
BEFORE DELETE ON forecast_ledger
BEGIN
    SELECT RAISE(ABORT, 'forecast ledger is append-only');
END;
"""


class ForecastLedger:
    """Small SQLite-backed append-only registry for prospective forecasts."""

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        raw_path = str(path)
        selected = Path(raw_path).expanduser()
        self.path = raw_path if raw_path == ":memory:" else str(selected)
        if self.path != ":memory:":
            selected.resolve().parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            self.path,
            timeout=30,
            check_same_thread=False,
        )
        if self.path != ":memory:":
            os.chmod(self.path, 0o600)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA busy_timeout = 30000")
        self._lock = RLock()
        self._clock = clock
        with self._connection:
            self._connection.executescript(_LEDGER_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "ForecastLedger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def append(self, entry: ForecastLedgerEntry) -> None:
        """Append exactly once; identical and conflicting duplicates both fail."""

        if not isinstance(entry, ForecastLedgerEntry):
            raise TypeError("entry must be a ForecastLedgerEntry")
        input_document = [item.as_dict() for item in entry.operational_inputs]
        inputs_json = _canonical_json(input_document)
        forecast_json = _canonical_json(entry.forecast)
        key = entry.key.as_sql_tuple()
        inserted_at = ensure_utc(self._clock(), field_name="ledger clock").isoformat()
        with self._lock:
            try:
                # Serialize the existence check across processes as well as
                # threads.  Without an immediate transaction, two schedulers
                # could both observe an empty key and expose a raw UNIQUE error
                # instead of the ledger's duplicate/conflict contract.
                self._connection.execute("BEGIN IMMEDIATE")
                existing = self._connection.execute(
                    """
                    SELECT operational_inputs_sha256, forecast_sha256
                    FROM forecast_ledger
                    WHERE origin_week = ? AND decision_at = ? AND target_at = ?
                      AND label_spec_sha256 = ? AND model_manifest_sha256 = ?
                      AND input_snapshot_sha256 = ?
                    """,
                    key,
                ).fetchone()
                if existing is not None:
                    if (
                        existing["operational_inputs_sha256"]
                        == entry.operational_inputs_sha256
                        and existing["forecast_sha256"] == entry.forecast_sha256
                    ):
                        raise DuplicateForecastError(
                            "forecast ledger entry already exists"
                        )
                    raise ConflictingForecastError(
                        "forecast ledger key already exists with different content"
                    )
                self._connection.execute(
                    """
                    INSERT INTO forecast_ledger(
                        origin_week, decision_at, target_at, label_spec_sha256,
                        model_manifest_sha256, input_snapshot_sha256,
                        operational_inputs_json, operational_inputs_sha256,
                        forecast_json, forecast_sha256, inserted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        *key,
                        inputs_json,
                        entry.operational_inputs_sha256,
                        forecast_json,
                        entry.forecast_sha256,
                        inserted_at,
                    ),
                )
            except BaseException:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def read(self, key: ForecastLedgerKey) -> ForecastLedgerEntry | None:
        if not isinstance(key, ForecastLedgerKey):
            raise TypeError("key must be a ForecastLedgerKey")
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM forecast_ledger
                WHERE origin_week = ? AND decision_at = ? AND target_at = ?
                  AND label_spec_sha256 = ? AND model_manifest_sha256 = ?
                  AND input_snapshot_sha256 = ?
                """,
                key.as_sql_tuple(),
            ).fetchone()
        return self._row_to_entry(row) if row is not None else None

    def list_entries(self) -> tuple[ForecastLedgerEntry, ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM forecast_ledger
                ORDER BY decision_at, origin_week, target_at,
                         label_spec_sha256, model_manifest_sha256,
                         input_snapshot_sha256
                """
            ).fetchall()
        return tuple(self._row_to_entry(row) for row in rows)

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> ForecastLedgerEntry:
        inputs = tuple(
            OperationalInput.from_dict(item)
            for item in json.loads(row["operational_inputs_json"])
        )
        entry = ForecastLedgerEntry(
            origin_week=date.fromisoformat(row["origin_week"]),
            decision_at=datetime.fromisoformat(row["decision_at"]),
            target_at=datetime.fromisoformat(row["target_at"]),
            label_spec_sha256=row["label_spec_sha256"],
            model_manifest_sha256=row["model_manifest_sha256"],
            input_snapshot_sha256=row["input_snapshot_sha256"],
            operational_inputs=inputs,
            forecast=json.loads(row["forecast_json"]),
        )
        if entry.operational_inputs_sha256 != row["operational_inputs_sha256"]:
            raise ForecastLedgerError("stored operational input hash is invalid")
        if entry.forecast_sha256 != row["forecast_sha256"]:
            raise ForecastLedgerError("stored forecast hash is invalid")
        return entry


def operational_input_manifest_sha256(
    inputs: Iterable[OperationalInput],
) -> str:
    """Hash an exact, order-independent operational input manifest."""

    ordered = sorted(
        inputs,
        key=lambda item: (
            item.source,
            item.series_id,
            item.observed_period_end,
            item.revision_seq,
            item.raw_sha256,
        ),
    )
    return _json_sha256([item.as_dict() for item in ordered])


__all__ = [
    "ConflictingForecastError",
    "DuplicateForecastError",
    "ForecastLedger",
    "ForecastLedgerEntry",
    "ForecastLedgerError",
    "ForecastLedgerKey",
    "OperationalInput",
    "operational_input_manifest_sha256",
]
