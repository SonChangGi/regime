"""Append-only operational forecast ledger.

The research artifacts can be regenerated as data vintages improve.  This
ledger records what an operating process actually knew and emitted at one
decision instant, so later reconstruction cannot overwrite prospective OOS
evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

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


class DuplicateEvaluationError(ForecastLedgerError):
    """Raised when an identical forecast evaluation already exists."""


class ConflictingEvaluationError(ForecastLedgerError):
    """Raised when a forecast key already has a different evaluation."""


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


def operational_inputs_for_generation(
    dataset: object,
    *,
    additional_records: tuple[object, ...] = (),
    origin_at: datetime,
    decision_at: datetime,
) -> tuple[OperationalInput, ...]:
    """Bind the exact provider revisions available to one forecast decision."""

    if getattr(dataset, "availability_basis", None) not in {
        "source",
        "operational",
        "reconstructed_market",
    }:
        raise ValueError("forecast ledger dataset has an invalid availability basis")
    if origin_at.tzinfo is None or origin_at.utcoffset() is None:
        raise ValueError("forecast origin must include a timezone")
    if decision_at.tzinfo is None or decision_at.utcoffset() is None:
        raise ValueError("forecast decision must include a timezone")
    origin = origin_at.astimezone(timezone.utc)
    decision = decision_at.astimezone(timezone.utc)
    if origin > decision:
        raise ValueError("forecast origin must not follow the decision")
    values = tuple(getattr(dataset, "input_vintages", ()))
    inputs = [OperationalInput.from_asof_value(value) for value in values]
    for item in inputs:
        if item.observed_period_end > origin.date():
            raise ValueError("forecast input period exceeds the forecast origin")
        if item.operating_available_at > decision:
            raise ValueError("forecast input was first seen after the decision")
        if item.system_retrieved_at > decision:
            raise ValueError("forecast input was retrieved after the decision")
    for record in additional_records:
        if (
            record.observed_period_end > origin.date()
            or record.operating_available_at > decision
            or record.system_retrieved_at > decision
        ):
            continue
        inputs.append(OperationalInput.from_observation(record))
    unique = {
        (
            item.source,
            item.series_id,
            item.observed_period_end,
            item.revision_seq,
            item.raw_sha256,
        ): item
        for item in inputs
    }
    if not unique:
        raise ValueError("operational generation has no bound input vintages")
    return tuple(unique.values())


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

    def as_dict(self) -> dict[str, str]:
        return {
            "origin_week": self.origin_week.isoformat(),
            "decision_at": self.decision_at.isoformat(),
            "target_at": self.target_at.isoformat(),
            "label_spec_sha256": self.label_spec_sha256,
            "model_manifest_sha256": self.model_manifest_sha256,
            "input_snapshot_sha256": self.input_snapshot_sha256,
        }


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


@dataclass(frozen=True, slots=True)
class ForecastEvaluationEntry:
    """One terminal, immutable evaluation of an operational forecast key.

    A due forecast is evaluated exactly once.  ``partial`` is deliberately a
    terminal evidence state reserved for permanent contract failures such as
    an ambiguous forecast identity or an unsupported legacy execution
    contract.  Temporarily missing target data never enters this table.
    """

    forecast_key: ForecastLedgerKey
    evaluated_at: datetime
    status: str
    evaluation: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.forecast_key, ForecastLedgerKey):
            raise TypeError("forecast_key must be a ForecastLedgerKey")
        evaluated_at = ensure_utc(self.evaluated_at, field_name="evaluated_at")
        if evaluated_at < self.forecast_key.target_at:
            raise ValueError("evaluated_at must not precede forecast target_at")
        status = str(self.status)
        if status not in {"completed", "partial"}:
            raise ValueError("forecast evaluation status must be completed or partial")
        if not isinstance(self.evaluation, Mapping) or not self.evaluation:
            raise ValueError("evaluation must be a non-empty mapping")
        normalized = json.loads(_canonical_json(dict(self.evaluation)))
        if normalized.get("schema_version") != "regime-operational-forecast-evaluation/1":
            raise ValueError("forecast evaluation schema is invalid")
        if normalized.get("status") != status:
            raise ValueError("forecast evaluation document status is inconsistent")
        if normalized.get("forecast_key") != self.forecast_key.as_dict():
            raise ValueError("forecast evaluation document key is inconsistent")
        _sha256(
            normalized.get("forecast_sha256", ""),
            field_name="forecast_sha256",
        )
        if status == "partial" and not str(normalized.get("reason", "")).strip():
            raise ValueError("partial forecast evaluation requires a reason")
        object.__setattr__(self, "evaluated_at", evaluated_at)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "evaluation", normalized)

    @property
    def evaluation_sha256(self) -> str:
        return _json_sha256(self.evaluation)


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

CREATE TABLE IF NOT EXISTS forecast_evaluation_ledger (
    origin_week TEXT NOT NULL,
    decision_at TEXT NOT NULL,
    target_at TEXT NOT NULL,
    label_spec_sha256 TEXT NOT NULL,
    model_manifest_sha256 TEXT NOT NULL,
    input_snapshot_sha256 TEXT NOT NULL,
    evaluated_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('completed', 'partial')),
    evaluation_json TEXT NOT NULL,
    evaluation_sha256 TEXT NOT NULL,
    inserted_at TEXT NOT NULL,
    PRIMARY KEY (
        origin_week,
        decision_at,
        target_at,
        label_spec_sha256,
        model_manifest_sha256,
        input_snapshot_sha256
    ),
    FOREIGN KEY (
        origin_week,
        decision_at,
        target_at,
        label_spec_sha256,
        model_manifest_sha256,
        input_snapshot_sha256
    ) REFERENCES forecast_ledger (
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

CREATE TRIGGER IF NOT EXISTS forecast_evaluation_ledger_no_update
BEFORE UPDATE ON forecast_evaluation_ledger
BEGIN
    SELECT RAISE(ABORT, 'forecast evaluation ledger is append-only');
END;

CREATE TRIGGER IF NOT EXISTS forecast_evaluation_ledger_no_delete
BEFORE DELETE ON forecast_evaluation_ledger
BEGIN
    SELECT RAISE(ABORT, 'forecast evaluation ledger is append-only');
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
        self._connection.execute("PRAGMA foreign_keys = ON")
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

    def append_evaluation(self, entry: ForecastEvaluationEntry) -> None:
        """Append one terminal evaluation for an existing forecast key."""

        if not isinstance(entry, ForecastEvaluationEntry):
            raise TypeError("entry must be a ForecastEvaluationEntry")
        key = entry.forecast_key.as_sql_tuple()
        evaluation_json = _canonical_json(entry.evaluation)
        inserted_at = ensure_utc(
            self._clock(), field_name="ledger clock"
        ).isoformat()
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                forecast = self._connection.execute(
                    """
                    SELECT 1 FROM forecast_ledger
                    WHERE origin_week = ? AND decision_at = ? AND target_at = ?
                      AND label_spec_sha256 = ? AND model_manifest_sha256 = ?
                      AND input_snapshot_sha256 = ?
                    """,
                    key,
                ).fetchone()
                if forecast is None:
                    raise ForecastLedgerError(
                        "forecast evaluation requires an existing forecast entry"
                    )
                existing = self._connection.execute(
                    """
                    SELECT evaluation_sha256
                    FROM forecast_evaluation_ledger
                    WHERE origin_week = ? AND decision_at = ? AND target_at = ?
                      AND label_spec_sha256 = ? AND model_manifest_sha256 = ?
                      AND input_snapshot_sha256 = ?
                    """,
                    key,
                ).fetchone()
                if existing is not None:
                    if existing["evaluation_sha256"] == entry.evaluation_sha256:
                        raise DuplicateEvaluationError(
                            "forecast evaluation already exists"
                        )
                    raise ConflictingEvaluationError(
                        "forecast key already has a different evaluation"
                    )
                self._connection.execute(
                    """
                    INSERT INTO forecast_evaluation_ledger(
                        origin_week, decision_at, target_at, label_spec_sha256,
                        model_manifest_sha256, input_snapshot_sha256,
                        evaluated_at, status, evaluation_json,
                        evaluation_sha256, inserted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        *key,
                        entry.evaluated_at.isoformat(),
                        entry.status,
                        evaluation_json,
                        entry.evaluation_sha256,
                        inserted_at,
                    ),
                )
            except BaseException:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def read_evaluation(
        self,
        key: ForecastLedgerKey,
    ) -> ForecastEvaluationEntry | None:
        if not isinstance(key, ForecastLedgerKey):
            raise TypeError("key must be a ForecastLedgerKey")
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM forecast_evaluation_ledger
                WHERE origin_week = ? AND decision_at = ? AND target_at = ?
                  AND label_spec_sha256 = ? AND model_manifest_sha256 = ?
                  AND input_snapshot_sha256 = ?
                """,
                key.as_sql_tuple(),
            ).fetchone()
        return self._row_to_evaluation(row) if row is not None else None

    def list_evaluations(self) -> tuple[ForecastEvaluationEntry, ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM forecast_evaluation_ledger
                ORDER BY target_at, decision_at, origin_week,
                         label_spec_sha256, model_manifest_sha256,
                         input_snapshot_sha256
                """
            ).fetchall()
        return tuple(self._row_to_evaluation(row) for row in rows)

    def public_summary(
        self,
        *,
        pending_key: ForecastLedgerKey | None = None,
        unresolved_due: Mapping[ForecastLedgerKey, str] | None = None,
    ) -> dict[str, Any]:
        """Return a raw-data-free count/hash over immutable primary keys.

        ``pending_key`` lets publication bind the exact key that will be
        appended by the rollback-protected finalization callback.  Forecast
        values and provider input identities are deliberately excluded.
        """

        if pending_key is not None and not isinstance(pending_key, ForecastLedgerKey):
            raise TypeError("pending_key must be a ForecastLedgerKey or None")
        unresolved = dict(unresolved_due or {})
        if any(
            not isinstance(key, ForecastLedgerKey) or not str(reason).strip()
            for key, reason in unresolved.items()
        ):
            raise TypeError(
                "unresolved_due must map ForecastLedgerKey values to reasons"
            )
        keys = [entry.key for entry in self.list_entries()]
        if pending_key is not None and pending_key.as_sql_tuple() not in {
            key.as_sql_tuple() for key in keys
        }:
            keys.append(pending_key)
        keys.sort(key=lambda key: key.as_sql_tuple())
        manifest = [key.as_dict() for key in keys]
        evaluations = sorted(
            self.list_evaluations(),
            key=lambda item: item.forecast_key.as_sql_tuple(),
        )
        evaluation_manifest = [
            {
                "forecast_key": item.forecast_key.as_dict(),
                "status": item.status,
                "evaluation_sha256": item.evaluation_sha256,
            }
            for item in evaluations
        ]
        completed = [item for item in evaluations if item.status == "completed"]
        partial_count = sum(item.status == "partial" for item in evaluations)
        key_tuples = {key.as_sql_tuple() for key in keys}
        evaluation_key_tuples = {
            item.forecast_key.as_sql_tuple() for item in evaluations
        }
        unresolved_key_tuples = {key.as_sql_tuple() for key in unresolved}
        if not unresolved_key_tuples <= key_tuples:
            raise ForecastLedgerError(
                "unresolved due evaluations must belong to the forecast manifest"
            )
        if unresolved_key_tuples & evaluation_key_tuples:
            raise ForecastLedgerError(
                "a forecast cannot be both evaluated and unresolved"
            )
        pending_count = (
            len(manifest) - len(evaluations) - len(unresolved_key_tuples)
        )
        if not manifest:
            status = "empty"
        elif len(completed) == len(manifest):
            status = "completed"
        elif not completed and not partial_count and not unresolved_key_tuples:
            status = "pending"
        else:
            status = "partial"
        return {
            "schema_version": "regime-prospective-ledger-summary/2",
            "status": status,
            "entry_count": len(manifest),
            "pending_evaluation_count": pending_count,
            "unresolved_due_evaluation_count": len(unresolved_key_tuples),
            "realized_evaluation_count": len(completed),
            "partial_evaluation_count": partial_count,
            "key_manifest_sha256": _json_sha256(manifest),
            "evaluation_manifest_sha256": _json_sha256(evaluation_manifest),
            "hash_scope": "ordered_ledger_primary_keys_only",
            "evaluation_hash_scope": (
                "ordered_forecast_primary_keys_status_and_evaluation_sha256"
            ),
            "performance": _public_evaluation_performance(completed, status=status),
        }

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

    @staticmethod
    def _row_to_evaluation(row: sqlite3.Row) -> ForecastEvaluationEntry:
        key = ForecastLedgerKey(
            origin_week=date.fromisoformat(row["origin_week"]),
            decision_at=datetime.fromisoformat(row["decision_at"]),
            target_at=datetime.fromisoformat(row["target_at"]),
            label_spec_sha256=row["label_spec_sha256"],
            model_manifest_sha256=row["model_manifest_sha256"],
            input_snapshot_sha256=row["input_snapshot_sha256"],
        )
        entry = ForecastEvaluationEntry(
            forecast_key=key,
            evaluated_at=datetime.fromisoformat(row["evaluated_at"]),
            status=row["status"],
            evaluation=json.loads(row["evaluation_json"]),
        )
        if entry.evaluation_sha256 != row["evaluation_sha256"]:
            raise ForecastLedgerError("stored forecast evaluation hash is invalid")
        return entry


def _completed_metric(
    evaluation: Mapping[str, Any],
    section: str,
    field_name: str,
) -> float:
    raw_section = evaluation.get(section)
    if not isinstance(raw_section, Mapping):
        raise ForecastLedgerError(
            f"completed forecast evaluation lacks {section}"
        )
    value = raw_section.get(field_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ForecastLedgerError(
            f"completed forecast evaluation lacks {section}.{field_name}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ForecastLedgerError(
            f"completed forecast evaluation has non-finite {section}.{field_name}"
        )
    return number


def _public_evaluation_performance(
    completed: Sequence[ForecastEvaluationEntry],
    *,
    status: str,
) -> dict[str, Any]:
    """Aggregate only completed immutable evaluations into derived public data."""

    weeks = len(completed)
    if weeks == 0:
        return {
            "status": "pending" if status in {"empty", "pending"} else "partial",
            "weeks": 0,
            "gross_cumulative_return": None,
            "net_cumulative_return": None,
            "turnover_sum": None,
            "transaction_cost_rate_sum": None,
            "transaction_cost_bps": None,
            "forecast_hit_count": None,
            "forecast_accuracy": None,
            "actual_state_counts": None,
        }

    gross_returns: list[float] = []
    net_returns: list[float] = []
    turnovers: list[float] = []
    costs: list[float] = []
    cost_bps: set[float] = set()
    hit_count = 0
    actual_state_counts = {state: 0 for state in ("risk_on", "transition", "risk_off")}
    for entry in completed:
        evaluation = entry.evaluation
        gross_returns.append(_completed_metric(evaluation, "returns", "gross_return"))
        net_returns.append(_completed_metric(evaluation, "returns", "net_return"))
        turnovers.append(_completed_metric(evaluation, "execution", "turnover"))
        costs.append(
            _completed_metric(evaluation, "execution", "transaction_cost_rate")
        )
        cost_bps.add(
            _completed_metric(evaluation, "execution", "one_way_turnover_bps")
        )
        actual = str(evaluation.get("actual_next_state", ""))
        if actual not in actual_state_counts:
            raise ForecastLedgerError(
                "completed forecast evaluation has invalid actual_next_state"
            )
        forecast = evaluation.get("forecast")
        if not isinstance(forecast, Mapping):
            raise ForecastLedgerError("completed forecast evaluation lacks forecast")
        predicted = str(forecast.get("predicted_state", ""))
        if predicted not in actual_state_counts:
            raise ForecastLedgerError(
                "completed forecast evaluation has invalid predicted_state"
            )
        actual_state_counts[actual] += 1
        hit_count += int(predicted == actual)
    if len(cost_bps) != 1:
        raise ForecastLedgerError(
            "completed forecast evaluations use inconsistent transaction costs"
        )
    gross_wealth = float(np.prod(1.0 + np.asarray(gross_returns, dtype=float)))
    net_wealth = float(np.prod(1.0 + np.asarray(net_returns, dtype=float)))
    return {
        "status": "completed" if status == "completed" else "partial",
        "weeks": weeks,
        "gross_cumulative_return": gross_wealth - 1.0,
        "net_cumulative_return": net_wealth - 1.0,
        "turnover_sum": float(sum(turnovers)),
        "transaction_cost_rate_sum": float(sum(costs)),
        "transaction_cost_bps": next(iter(cost_bps)),
        "forecast_hit_count": hit_count,
        "forecast_accuracy": float(hit_count / weeks),
        "actual_state_counts": actual_state_counts,
    }


@dataclass(frozen=True, slots=True)
class ForecastMaturityReport:
    """Result of one idempotent maturity pass over the private ledger."""

    appended: tuple[ForecastEvaluationEntry, ...]
    pending: tuple[ForecastLedgerKey, ...]
    unresolved_due: Mapping[ForecastLedgerKey, str]


def _entry_target_week(entry: ForecastLedgerEntry) -> date:
    shadow = entry.forecast.get("decision_shadow")
    if isinstance(shadow, Mapping):
        signal = shadow.get("current_signal")
        if isinstance(signal, Mapping) and signal.get("target_week") is not None:
            try:
                return date.fromisoformat(str(signal["target_week"]))
            except ValueError:
                pass
    official = entry.forecast.get("official")
    if isinstance(official, Mapping) and official.get("date") is not None:
        try:
            return date.fromisoformat(str(official["date"]))
        except ValueError:
            pass
    return entry.target_at.astimezone(ZoneInfo("America/New_York")).date()


def _date_index(index: pd.Index, *, context: str) -> dict[date, Any]:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"{context} must use a DatetimeIndex")
    resolved: dict[date, Any] = {}
    for raw in index:
        key = pd.Timestamp(raw).date()
        if key in resolved:
            raise ValueError(f"{context} contains duplicate market dates")
        resolved[key] = raw
    return resolved


def _partial_evaluation(
    entry: ForecastLedgerEntry,
    *,
    evaluated_at: datetime,
    target_week: date,
    reason: str,
) -> ForecastEvaluationEntry:
    document = {
        "schema_version": "regime-operational-forecast-evaluation/1",
        "status": "partial",
        "forecast_key": entry.key.as_dict(),
        "forecast_sha256": entry.forecast_sha256,
        "evaluated_at": ensure_utc(
            evaluated_at, field_name="evaluated_at"
        ).isoformat(),
        "target_week": target_week.isoformat(),
        "reason": str(reason),
    }
    return ForecastEvaluationEntry(
        forecast_key=entry.key,
        evaluated_at=evaluated_at,
        status="partial",
        evaluation=document,
    )


def _investment_shadow_contract(
    entry: ForecastLedgerEntry,
) -> Mapping[str, Any] | None:
    shadow = entry.forecast.get("decision_shadow")
    if not isinstance(shadow, Mapping):
        return None
    spec = shadow.get("spec")
    signal = shadow.get("current_signal")
    if (
        shadow.get("schema_version") != "regime-prospective-decision-shadow/2"
        or not isinstance(spec, Mapping)
        or spec.get("spec_id") != "spy-tlt-probability-shadow-v2"
        or not isinstance(signal, Mapping)
    ):
        return None
    return shadow


def _completed_prior_portfolio(
    evaluation: ForecastEvaluationEntry,
) -> tuple[np.ndarray, float]:
    document = evaluation.evaluation
    portfolio = document.get("portfolio")
    if evaluation.status != "completed" or not isinstance(portfolio, Mapping):
        raise ForecastLedgerError("prior forecast evaluation is not completed")
    raw_weights = portfolio.get("close_weights")
    raw_cash = portfolio.get("close_cash")
    if not isinstance(raw_weights, Mapping):
        raise ForecastLedgerError("prior forecast evaluation lacks closing state")
    try:
        weights = np.asarray(
            [float(raw_weights[asset]) for asset in ("SPY", "TLT")],
            dtype=float,
        )
        cash = float(raw_cash)
    except (KeyError, TypeError, ValueError) as exc:
        raise ForecastLedgerError(
            "prior forecast evaluation closing state is invalid"
        ) from exc
    if (
        not np.isfinite(weights).all()
        or not math.isfinite(cash)
        or (weights < 0.0).any()
        or cash < 0.0
        or not math.isclose(float(weights.sum()) + cash, 1.0, abs_tol=1e-8)
    ):
        raise ForecastLedgerError(
            "prior forecast evaluation closing state is invalid"
        )
    return weights, cash


def _selected_forecast_contract(
    entry: ForecastLedgerEntry,
    shadow: Mapping[str, Any],
    *,
    target_week: date,
) -> tuple[str, dict[str, float], str, Mapping[str, Any]]:
    selection = entry.forecast.get("selection")
    if not isinstance(selection, Mapping):
        raise ForecastLedgerError("forecast lacks selection contract")
    forecast_model = str(selection.get("operating_champion", ""))
    if not forecast_model:
        raise ForecastLedgerError("forecast lacks operating champion")
    signal = shadow.get("current_signal")
    if not isinstance(signal, Mapping):
        raise ForecastLedgerError("forecast lacks current signal")
    required_signal_fields = {
        "origin_date",
        "target_week",
        "scheduled_entry_at",
        "decision_at",
        "forecast_model",
        "status",
        "action",
    }
    if set(signal) != required_signal_fields:
        raise ForecastLedgerError("forecast current signal fields are invalid")
    if (
        str(signal.get("origin_date")) != entry.origin_week.isoformat()
        or str(signal.get("target_week")) != target_week.isoformat()
        or str(signal.get("forecast_model")) != forecast_model
    ):
        raise ForecastLedgerError("forecast current signal identity is invalid")
    try:
        decision_at = ensure_utc(
            datetime.fromisoformat(str(signal["decision_at"])),
            field_name="current_signal.decision_at",
        )
        scheduled_entry_at = ensure_utc(
            datetime.fromisoformat(str(signal["scheduled_entry_at"])),
            field_name="current_signal.scheduled_entry_at",
        )
    except (TypeError, ValueError) as exc:
        raise ForecastLedgerError("forecast current signal clocks are invalid") from exc
    if decision_at != entry.decision_at:
        raise ForecastLedgerError("forecast current signal decision clock differs")
    expected = (
        ("scheduled", "trade_at_scheduled_open")
        if decision_at < scheduled_entry_at
        else ("missed_entry", "no_trade")
    )
    if (signal.get("status"), signal.get("action")) != expected:
        raise ForecastLedgerError("forecast current signal action is invalid")

    rows = entry.forecast.get("model_forecasts")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ForecastLedgerError("forecast model suite is missing")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("model") == forecast_model
    ]
    if len(matches) != 1:
        raise ForecastLedgerError("operating forecast model is ambiguous")
    selected = matches[0]
    if str(selected.get("date")) != target_week.isoformat():
        raise ForecastLedgerError("operating forecast targets the wrong week")
    raw_probabilities = selected.get("probabilities")
    if not isinstance(raw_probabilities, Mapping) or set(raw_probabilities) != {
        "risk_on",
        "transition",
        "risk_off",
    }:
        raise ForecastLedgerError("operating forecast probabilities are invalid")
    probabilities = {
        state: float(raw_probabilities[state])
        for state in ("risk_on", "transition", "risk_off")
    }
    values = np.asarray(list(probabilities.values()), dtype=float)
    if (
        not np.isfinite(values).all()
        or (values < 0.0).any()
        or not math.isclose(
            float(values.sum()),
            1.0,
            abs_tol=1e-6,
            rel_tol=0.0,
        )
    ):
        raise ForecastLedgerError("operating forecast probabilities are invalid")
    predicted_state = str(selected.get("state", ""))
    if predicted_state not in probabilities:
        raise ForecastLedgerError("operating forecast predicted state is invalid")
    expected_state = ("risk_on", "transition", "risk_off")[int(np.argmax(values))]
    if predicted_state != expected_state:
        raise ForecastLedgerError("operating forecast state differs from argmax")
    return forecast_model, probabilities, predicted_state, signal


def _evaluate_completed_week(
    entry: ForecastLedgerEntry,
    shadow: Mapping[str, Any],
    *,
    target_week: date,
    actual_next_state: str,
    gap_relatives: Mapping[str, float],
    open_to_close_relatives: Mapping[str, float],
    price_observations: Mapping[str, Mapping[str, float]],
    evaluated_at: datetime,
    prior_evaluation: ForecastEvaluationEntry | None,
    genesis_source: str,
) -> ForecastEvaluationEntry:
    # Import lazily so the private ledger remains usable without importing the
    # dashboard composer and to avoid a module cycle.
    from regime_lab.analysis.decision_shadow import load_decision_shadow_spec

    spec_document = shadow.get("spec")
    if not isinstance(spec_document, Mapping):
        raise ForecastLedgerError("decision shadow spec identity is missing")
    spec_path = str(spec_document.get("path", ""))
    if spec_path != "config/decision-shadow-v2.json":
        raise ForecastLedgerError("decision shadow spec path is invalid")
    raw_snapshot = shadow.get("spec_snapshot")
    if isinstance(raw_snapshot, Mapping):
        spec = json.loads(_canonical_json(dict(raw_snapshot)))
    else:
        # Compatibility for the short-lived v2 forecast records emitted before
        # the private ledger began freezing the full execution specification.
        spec = load_decision_shadow_spec()
    if (
        spec.get("schema_version") != "regime-decision-shadow-spec/1"
        or
        spec_document.get("spec_id") != spec.get("spec_id")
        or spec_document.get("sha256") != canonical_json_sha256_v1(spec)
    ):
        raise ForecastLedgerError("decision shadow spec hash is inconsistent")
    forecast_model, probabilities, predicted_state, signal = (
        _selected_forecast_contract(entry, shadow, target_week=target_week)
    )
    mapping = spec["probability_weight_mapping"]
    probability_total = float(sum(probabilities.values()))
    requested_weights = np.asarray(
        [
            sum(
                probabilities[state]
                / probability_total
                * float(mapping[state][asset])
                for state in ("risk_on", "transition", "risk_off")
            )
            for asset in ("SPY", "TLT")
        ],
        dtype=float,
    )
    if (
        not np.isfinite(requested_weights).all()
        or (requested_weights < -1e-12).any()
        or float(requested_weights.sum()) > 1.0 + 1e-12
    ):
        raise ForecastLedgerError("forecast target weights are invalid")
    requested_weights = np.maximum(requested_weights, 0.0)
    intraday_relatives = np.asarray(
        [open_to_close_relatives[asset] for asset in ("SPY", "TLT")],
        dtype=float,
    )
    if (
        not np.isfinite(intraday_relatives).all()
        or (intraday_relatives <= 0.0).any()
    ):
        raise ForecastLedgerError(
            "portfolio open-to-close relatives are invalid"
        )
    if prior_evaluation is None:
        prior_weights = np.zeros(2, dtype=float)
        prior_cash = 1.0
        applied_gap_relatives = np.ones(2, dtype=float)
        gap_factor = 1.0
        pretrade_weights = prior_weights.copy()
        pretrade_cash = prior_cash
        prior_source = genesis_source
    else:
        prior_weights, prior_cash = _completed_prior_portfolio(prior_evaluation)
        applied_gap_relatives = np.asarray(
            [gap_relatives[asset] for asset in ("SPY", "TLT")],
            dtype=float,
        )
        if (
            not np.isfinite(applied_gap_relatives).all()
            or (applied_gap_relatives <= 0.0).any()
        ):
            raise ForecastLedgerError("portfolio gap relatives are invalid")
        gap_factor = float(
            prior_cash + np.dot(prior_weights, applied_gap_relatives)
        )
        if not math.isfinite(gap_factor) or gap_factor <= 0.0:
            raise ForecastLedgerError("portfolio close-to-open wealth is invalid")
        pretrade_weights = prior_weights * applied_gap_relatives / gap_factor
        pretrade_cash = prior_cash / gap_factor
        prior_source = "prior_completed_evaluation"

    action = str(signal["action"])
    if action == "trade_at_scheduled_open":
        applied_weights = requested_weights.copy()
        applied_cash = max(0.0, 1.0 - float(applied_weights.sum()))
        turnover = (
            float(np.abs(applied_weights - pretrade_weights).sum())
            if prior_evaluation is not None
            or bool(spec["cost"]["initial_allocation_costed_from_cash"])
            else 0.0
        )
    elif action == "no_trade":
        applied_weights = pretrade_weights.copy()
        applied_cash = float(pretrade_cash)
        turnover = 0.0
    else:
        raise ForecastLedgerError("unsupported current signal action")
    cost_bps = float(spec["cost"]["one_way_turnover_bps"])
    transaction_cost_rate = turnover * cost_bps / 10_000.0
    if transaction_cost_rate >= 1.0:
        raise ForecastLedgerError("transaction cost exhausts portfolio wealth")
    intraday_factor = float(
        applied_cash + np.dot(applied_weights, intraday_relatives)
    )
    if not math.isfinite(intraday_factor) or intraday_factor <= 0.0:
        raise ForecastLedgerError("portfolio open-to-close wealth is invalid")
    gross_factor = gap_factor * intraday_factor
    net_factor = gap_factor * (1.0 - transaction_cost_rate) * intraday_factor
    close_weights = applied_weights * intraday_relatives / intraday_factor
    close_cash = applied_cash / intraday_factor
    assets = ("SPY", "TLT")
    document = {
        "schema_version": "regime-operational-forecast-evaluation/1",
        "status": "completed",
        "forecast_key": entry.key.as_dict(),
        "forecast_sha256": entry.forecast_sha256,
        "evaluated_at": ensure_utc(
            evaluated_at, field_name="evaluated_at"
        ).isoformat(),
        "target_week": target_week.isoformat(),
        "spec": dict(spec_document),
        "allocation_contract": {
            "assets": list(spec["assets"]),
            "probability_weight_mapping": spec["probability_weight_mapping"],
        },
        "return_accounting": dict(spec["return_accounting"]),
        "forecast": {
            "model": forecast_model,
            "probabilities": probabilities,
            "predicted_state": predicted_state,
            "requested_target_weights": {
                asset: float(requested_weights[index])
                for index, asset in enumerate(assets)
            },
        },
        "current_signal": dict(signal),
        "actual_next_state": actual_next_state,
        "prices": {
            "session_date": target_week.isoformat(),
            "scheduled_entry_at": str(signal["scheduled_entry_at"]),
            "assets": {
                asset: {
                    field: float(value)
                    for field, value in price_observations[asset].items()
                }
                for asset in assets
            },
        },
        "portfolio": {
            "prior_source": prior_source,
            "prior_forecast_key": (
                None
                if prior_evaluation is None
                else prior_evaluation.forecast_key.as_dict()
            ),
            "prior_close_weights": {
                asset: float(prior_weights[index])
                for index, asset in enumerate(assets)
            },
            "prior_close_cash": float(prior_cash),
            "gap_relatives": {
                asset: float(applied_gap_relatives[index])
                for index, asset in enumerate(assets)
            },
            "gap_factor": gap_factor,
            "pretrade_weights": {
                asset: float(pretrade_weights[index])
                for index, asset in enumerate(assets)
            },
            "pretrade_cash": float(pretrade_cash),
            "applied_target_weights": {
                asset: float(applied_weights[index])
                for index, asset in enumerate(assets)
            },
            "applied_cash": float(applied_cash),
            "close_weights": {
                asset: float(close_weights[index])
                for index, asset in enumerate(assets)
            },
            "close_cash": float(close_cash),
        },
        "execution": {
            "action": action,
            "turnover_definition": spec["cost"]["turnover_definition"],
            "turnover": turnover,
            "one_way_turnover_bps": cost_bps,
            "transaction_cost_rate": transaction_cost_rate,
        },
        "returns": {
            "gap_return": gap_factor - 1.0,
            "open_to_close_asset_returns": {
                asset: float(intraday_relatives[index] - 1.0)
                for index, asset in enumerate(assets)
            },
            "gross_return": gross_factor - 1.0,
            "net_return": net_factor - 1.0,
        },
    }
    return ForecastEvaluationEntry(
        forecast_key=entry.key,
        evaluated_at=evaluated_at,
        status="completed",
        evaluation=document,
    )


def mature_forecast_evaluations(
    ledger: ForecastLedger,
    *,
    canonical: pd.DataFrame,
    states: pd.Series,
    evaluated_at: datetime,
) -> ForecastMaturityReport:
    """Mature due v2 forecasts once, preserving transient gaps for retry.

    Completed target rows and structurally impossible legacy/ambiguous records
    are appended exactly once.  A due row with temporarily missing canonical
    data stays out of the immutable evaluation table and is returned through
    ``unresolved_due`` so a later live collection can safely retry it.
    """

    if not isinstance(ledger, ForecastLedger):
        raise TypeError("ledger must be a ForecastLedger")
    if not isinstance(canonical, pd.DataFrame):
        raise TypeError("canonical must be a DataFrame")
    if not isinstance(states, pd.Series):
        raise TypeError("states must be a Series")
    evaluation_clock = ensure_utc(evaluated_at, field_name="evaluated_at")
    canonical_dates = _date_index(canonical.index, context="canonical")
    state_dates = _date_index(states.index, context="states")
    # Use the exact return decomposition shared by reconstructed and benchmark
    # strategies.  Failure to derive it from the current canonical snapshot is
    # retriable data unavailability, not immutable evidence about a forecast.
    from regime_lab.analysis.decision_shadow import (
        split_safe_price_only_return_frames,
    )

    try:
        gap_frame, intraday_frame = split_safe_price_only_return_frames(canonical)
        return_frame_error: str | None = None
    except (KeyError, TypeError, ValueError) as exc:
        gap_frame = pd.DataFrame(index=canonical.index)
        intraday_frame = pd.DataFrame(index=canonical.index)
        return_frame_error = str(exc)
    entries = list(ledger.list_entries())
    existing = {
        item.forecast_key.as_sql_tuple(): item for item in ledger.list_evaluations()
    }
    pending: list[ForecastLedgerKey] = []
    unresolved: dict[ForecastLedgerKey, str] = {}
    appended: list[ForecastEvaluationEntry] = []

    target_by_key = {
        entry.key.as_sql_tuple(): _entry_target_week(entry) for entry in entries
    }
    v2_entries = [entry for entry in entries if _investment_shadow_contract(entry)]
    duplicate_targets = {
        target
        for target in {target_by_key[entry.key.as_sql_tuple()] for entry in v2_entries}
        if sum(
            target_by_key[item.key.as_sql_tuple()] == target for item in v2_entries
        )
        > 1
    }
    v2_entries.sort(
        key=lambda item: (
            target_by_key[item.key.as_sql_tuple()],
            item.decision_at,
            item.key.as_sql_tuple(),
        )
    )
    previous_v2: ForecastLedgerEntry | None = None
    for entry in sorted(
        entries,
        key=lambda item: (
            target_by_key[item.key.as_sql_tuple()],
            item.decision_at,
            item.key.as_sql_tuple(),
        ),
    ):
        key_tuple = entry.key.as_sql_tuple()
        target_week = target_by_key[key_tuple]
        shadow = _investment_shadow_contract(entry)
        is_due = evaluation_clock >= entry.target_at
        if key_tuple in existing:
            if shadow is not None:
                previous_v2 = entry
            continue
        if not is_due:
            pending.append(entry.key)
            continue
        if shadow is None:
            evaluation = _partial_evaluation(
                entry,
                evaluated_at=evaluation_clock,
                target_week=target_week,
                reason="legacy_or_missing_investment_execution_contract",
            )
            ledger.append_evaluation(evaluation)
            existing[key_tuple] = evaluation
            appended.append(evaluation)
            continue
        if target_week in duplicate_targets:
            evaluation = _partial_evaluation(
                entry,
                evaluated_at=evaluation_clock,
                target_week=target_week,
                reason="ambiguous_multiple_forecasts_for_target_week",
            )
            ledger.append_evaluation(evaluation)
            existing[key_tuple] = evaluation
            appended.append(evaluation)
            previous_v2 = entry
            continue
        if previous_v2 is not None:
            previous_target = target_by_key[previous_v2.key.as_sql_tuple()]
            if (
                entry.origin_week != previous_target
                or target_week != previous_target + timedelta(days=7)
            ):
                evaluation = _partial_evaluation(
                    entry,
                    evaluated_at=evaluation_clock,
                    target_week=target_week,
                    reason="forecast_sequence_gap",
                )
                ledger.append_evaluation(evaluation)
                existing[key_tuple] = evaluation
                appended.append(evaluation)
                previous_v2 = entry
                continue
            prior_evaluation = existing.get(previous_v2.key.as_sql_tuple())
            if prior_evaluation is None:
                unresolved[entry.key] = "prior_evaluation_unresolved"
                previous_v2 = entry
                continue
            if prior_evaluation.status != "completed":
                # A permanent failure closes only the affected segment.  The
                # next consecutive forecast can be evaluated from cash instead
                # of poisoning every later operational forecast.
                prior_evaluation = None
                genesis_source = "cash_segment_restart_after_terminal_partial"
            else:
                genesis_source = "not_applicable"
        else:
            prior_evaluation = None
            genesis_source = "cash_genesis"

        canonical_at = canonical_dates.get(target_week)
        state_at = state_dates.get(target_week)
        if canonical_at is None:
            unresolved[entry.key] = "target_week_missing_from_canonical"
            previous_v2 = entry
            continue
        if state_at is None or pd.isna(states.loc[state_at]):
            unresolved[entry.key] = "actual_next_state_unavailable"
            previous_v2 = entry
            continue
        actual_next_state = str(states.loc[state_at])
        if actual_next_state not in {"risk_on", "transition", "risk_off"}:
            unresolved[entry.key] = "actual_next_state_unavailable"
            previous_v2 = entry
            continue
        price_fields = {
            "SPY": {
                "adjusted_close": "spy_close",
                "raw_open": "spy_raw_open",
                "raw_close": "spy_raw_close",
                "dividend_amount": "spy_dividend_amount",
            },
            "TLT": {
                "adjusted_close": "tlt_close",
                "raw_open": "tlt_raw_open",
                "raw_close": "tlt_raw_close",
                "dividend_amount": "tlt_dividend_amount",
            },
        }
        if any(
            column not in canonical
            for fields in price_fields.values()
            for column in fields.values()
        ):
            unresolved[entry.key] = "target_prices_unavailable"
            previous_v2 = entry
            continue
        price_observations = {
            asset: {
                field: float(canonical.loc[canonical_at, column])
                for field, column in fields.items()
            }
            for asset, fields in price_fields.items()
        }
        price_values = np.asarray(
            [
                value
                for observations in price_observations.values()
                for value in observations.values()
            ],
            dtype=float,
        )
        if (
            not np.isfinite(price_values).all()
            or any(
                observations["adjusted_close"] <= 0.0
                or observations["raw_open"] <= 0.0
                or observations["raw_close"] <= 0.0
                or observations["dividend_amount"] < 0.0
                for observations in price_observations.values()
            )
        ):
            unresolved[entry.key] = "target_prices_unavailable"
            previous_v2 = entry
            continue
        if return_frame_error is not None:
            unresolved[entry.key] = "price_return_contract_unavailable"
            previous_v2 = entry
            continue
        gap_relatives = {
            asset: float(gap_frame.loc[canonical_at, asset])
            for asset in ("SPY", "TLT")
        }
        open_to_close_relatives = {
            asset: float(intraday_frame.loc[canonical_at, asset])
            for asset in ("SPY", "TLT")
        }
        relative_values = np.asarray(
            list(open_to_close_relatives.values()), dtype=float
        )
        if prior_evaluation is not None:
            relative_values = np.concatenate(
                [
                    np.asarray(list(gap_relatives.values()), dtype=float),
                    relative_values,
                ]
            )
        if (
            not np.isfinite(relative_values).all()
            or (relative_values <= 0.0).any()
        ):
            unresolved[entry.key] = "target_price_relatives_unavailable"
            previous_v2 = entry
            continue
        try:
            evaluation = _evaluate_completed_week(
                entry,
                shadow,
                target_week=target_week,
                actual_next_state=actual_next_state,
                gap_relatives=gap_relatives,
                open_to_close_relatives=open_to_close_relatives,
                price_observations=price_observations,
                evaluated_at=evaluation_clock,
                prior_evaluation=prior_evaluation,
                genesis_source=genesis_source,
            )
        except (ForecastLedgerError, KeyError, TypeError, ValueError) as exc:
            evaluation = _partial_evaluation(
                entry,
                evaluated_at=evaluation_clock,
                target_week=target_week,
                reason=f"structural_contract_error:{exc}",
            )
        ledger.append_evaluation(evaluation)
        existing[key_tuple] = evaluation
        appended.append(evaluation)
        previous_v2 = entry

    return ForecastMaturityReport(
        appended=tuple(appended),
        pending=tuple(pending),
        unresolved_due=unresolved,
    )


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


def build_research_replay_input_document(
    *,
    input_vintages: Iterable[AsOfValue],
    availability_basis: str,
    source_observation_count: int,
    canonical: pd.DataFrame,
    states: pd.Series,
    data_as_of: str | datetime,
    operational_input_snapshot_sha256: str,
) -> dict[str, Any]:
    """Bind reconstructed OOS research to its exact private input identity."""

    if availability_basis != "reconstructed_market":
        raise ValueError(
            "research replay availability_basis must be reconstructed_market"
        )
    if (
        isinstance(source_observation_count, bool)
        or not isinstance(source_observation_count, int)
        or source_observation_count <= 0
    ):
        raise ValueError("research replay source_observation_count must be positive")
    try:
        replay_inputs = tuple(
            OperationalInput.from_asof_value(value) for value in input_vintages
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"research replay input vintages are invalid: {exc}") from exc
    if not replay_inputs:
        raise ValueError("research replay has no bound input vintages")
    if len(replay_inputs) > source_observation_count:
        raise ValueError(
            "research replay input-vintage count exceeds source observations"
        )
    if canonical.empty or not isinstance(canonical.index, pd.DatetimeIndex):
        raise ValueError("research replay canonical panel must be non-empty and dated")
    if canonical.index.has_duplicates or not canonical.index.is_monotonic_increasing:
        raise ValueError("research replay canonical dates must be unique and ordered")
    if not isinstance(states, pd.Series) or not states.index.equals(canonical.index):
        raise ValueError("research replay states must align exactly with canonical")
    if states.isna().any():
        raise ValueError("research replay states must be complete")
    try:
        parsed_as_of = (
            data_as_of
            if isinstance(data_as_of, datetime)
            else datetime.fromisoformat(str(data_as_of).replace("Z", "+00:00"))
        )
    except ValueError as exc:
        raise ValueError("research replay data_as_of must be ISO-8601") from exc
    if parsed_as_of.tzinfo is None or parsed_as_of.utcoffset() is None:
        raise ValueError("research replay data_as_of must include a timezone")
    if canonical.index[-1].date() != parsed_as_of.date():
        raise ValueError("research replay canonical end differs from data_as_of")

    def frame_sha256(frame: pd.DataFrame | pd.Series, *, index_label: str) -> str:
        materialized = (
            frame.to_frame() if isinstance(frame, pd.Series) else frame.copy()
        )
        normalized = materialized.copy()
        normalized.index = [
            value.isoformat() if hasattr(value, "isoformat") else str(value)
            for value in normalized.index
        ]
        raw = normalized.to_csv(
            index=True,
            index_label=index_label,
            lineterminator="\n",
            float_format="%.17g",
            na_rep="",
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    return {
        "schema_version": "regime-research-replay-input/1",
        "evidence_track": "reconstructed_oos",
        "data_as_of": parsed_as_of.isoformat(),
        "availability_basis": availability_basis,
        "source_observation_count": source_observation_count,
        "input_vintages": {
            "count": len(replay_inputs),
            "sha256": operational_input_manifest_sha256(replay_inputs),
        },
        "canonical_panel": {
            "start": canonical.index[0].date().isoformat(),
            "end": canonical.index[-1].date().isoformat(),
            "rows": len(canonical),
            "columns": int(canonical.shape[1]),
            "sha256": frame_sha256(canonical, index_label="week"),
        },
        "state_membership": {
            "rows": len(states),
            "sha256": frame_sha256(states.rename("state"), index_label="week"),
        },
        "operational_generation_input_snapshot_sha256": _sha256(
            operational_input_snapshot_sha256,
            field_name="operational_input_snapshot_sha256",
        ),
    }


__all__ = [
    "ConflictingEvaluationError",
    "ConflictingForecastError",
    "DuplicateEvaluationError",
    "DuplicateForecastError",
    "ForecastEvaluationEntry",
    "ForecastLedger",
    "ForecastLedgerEntry",
    "ForecastLedgerError",
    "ForecastLedgerKey",
    "ForecastMaturityReport",
    "OperationalInput",
    "build_research_replay_input_document",
    "mature_forecast_evaluations",
    "operational_input_manifest_sha256",
    "operational_inputs_for_generation",
]
