"""Immutable regime-only outcomes, independent of portfolio execution data.

The legacy investment ledger is never rewritten. These records freeze labels,
probabilities and issue-time evidence for the same original forecast identity.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, date
import json
import math
import sqlite3
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from regime_lab.data.contracts import ensure_utc
from regime_lab.integrity import canonical_json_sha256_v1
from regime_lab.payload import normalized_probabilities

if TYPE_CHECKING:
    from regime_lab.forecast_ledger import ForecastLedger, ForecastLedgerEntry

STATES = ("risk_on", "transition", "risk_off")
SCHEMA = "regime-probability-evaluation/1"
KEY_FIELDS = ("origin_week", "decision_at", "target_at", "label_spec_sha256",
              "model_manifest_sha256", "input_snapshot_sha256")
WHERE_KEY = " AND ".join(f"{name} = ?" for name in KEY_FIELDS)
PROBABILITY_SCHEMA = """
CREATE TABLE IF NOT EXISTS forecast_probability_evaluations (
    origin_week TEXT NOT NULL, decision_at TEXT NOT NULL, target_at TEXT NOT NULL,
    label_spec_sha256 TEXT NOT NULL, model_manifest_sha256 TEXT NOT NULL,
    input_snapshot_sha256 TEXT NOT NULL, evaluated_at TEXT NOT NULL,
    evaluation_json TEXT NOT NULL, evaluation_sha256 TEXT NOT NULL,
    PRIMARY KEY (origin_week, decision_at, target_at, label_spec_sha256,
                 model_manifest_sha256, input_snapshot_sha256),
    FOREIGN KEY (origin_week, decision_at, target_at, label_spec_sha256,
                 model_manifest_sha256, input_snapshot_sha256)
    REFERENCES forecast_ledger (origin_week, decision_at, target_at, label_spec_sha256,
                               model_manifest_sha256, input_snapshot_sha256)
);
CREATE TRIGGER IF NOT EXISTS forecast_probability_no_update
BEFORE UPDATE ON forecast_probability_evaluations BEGIN
    SELECT RAISE(ABORT, 'forecast probability evaluation is append-only'); END;
CREATE TRIGGER IF NOT EXISTS forecast_probability_no_delete
BEFORE DELETE ON forecast_probability_evaluations BEGIN
    SELECT RAISE(ABORT, 'forecast probability evaluation is append-only'); END;
"""


@dataclass(frozen=True)
class ProbabilityForecast:
    """Only the issued forecast is needed to score labels, not provider rows."""
    key: Any
    forecast: Mapping[str, Any]
    forecast_sha256: str
    inserted_at: datetime

    @property
    def origin_week(self): return self.key.origin_week

    @property
    def decision_at(self): return self.key.decision_at

    @property
    def target_at(self): return self.key.target_at

    @property
    def label_spec_sha256(self): return self.key.label_spec_sha256

    @property
    def model_manifest_sha256(self): return self.key.model_manifest_sha256


FORECAST_PROJECTION = ", ".join((*KEY_FIELDS, "forecast_json", "forecast_sha256", "inserted_at"))


def probability_forecast_from_row(row: sqlite3.Row) -> ProbabilityForecast:
    from regime_lab.forecast_ledger import ForecastLedgerKey, ForecastLedgerError
    document = json.loads(row["forecast_json"])
    if canonical_json_sha256_v1(document) != row["forecast_sha256"]:
        raise ForecastLedgerError("stored forecast hash is invalid")
    key = ForecastLedgerKey(origin_week=date.fromisoformat(row["origin_week"]),
        decision_at=datetime.fromisoformat(row["decision_at"]),
        target_at=datetime.fromisoformat(row["target_at"]),
        **{k: row[k] for k in KEY_FIELDS[3:]})
    return ProbabilityForecast(key, document, row["forecast_sha256"], datetime.fromisoformat(row["inserted_at"]))


def issue_evidence(entry: ForecastLedgerEntry, *, deadline: datetime | None = None) -> dict[str, Any]:
    """Use both the actual database receipt and publication receipt, not a supplied decision clock."""
    limit = ensure_utc(deadline or entry.target_at, field_name="deadline")
    inserted = entry.inserted_at
    published_raw = entry.forecast.get("local_publication_at")
    published = None
    reasons = []
    if inserted is None:
        reasons.append("missing_storage_receipt")
    if published_raw is not None:
        try:
            published = ensure_utc(datetime.fromisoformat(str(published_raw)), field_name="local_publication_at")
        except (TypeError, ValueError):
            reasons.append("invalid_publication_receipt")
    else:
        reasons.append("missing_publication_receipt")
    if entry.forecast.get("evidence_track") == "reconstructed_oos" or entry.forecast.get("research_replay"):
        reasons.append("research_replay")
    if inserted is not None and inserted < entry.decision_at:
        reasons.append("storage_before_decision")
    if published is not None and published < entry.decision_at:
        reasons.append("publication_before_decision")
    issued = max((v for v in (inserted, published, entry.decision_at) if v is not None), default=None)
    if issued is not None and issued >= limit:
        reasons.append("deadline_missed")
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "inserted_at": inserted.isoformat() if inserted else None,
        "published_at": published.isoformat() if published else None,
        "issued_at": issued.isoformat() if inserted is not None and published is not None else None,
        "deadline_at": limit.isoformat(),
        "basis": "latest_of_decision_database_receipt_and_local_publication",
    }


def _frozen_models(entry: ForecastLedgerEntry) -> tuple[str, list[dict[str, Any]]]:
    from regime_lab.forecast_ledger import ForecastLedgerError
    champion = str(entry.forecast.get("selection", {}).get("operating_champion") or entry.forecast.get("champion") or "")
    raw_rows = entry.forecast.get("model_forecasts", [])
    if not isinstance(raw_rows, list):
        raise ForecastLedgerError("probability model suite is invalid")
    # Older entries may freeze just the official distribution. It is usable
    # for label scores even without an investment execution contract.
    if not raw_rows and isinstance(entry.forecast.get("probabilities"), Mapping):
        champion = champion or "legacy_official"
        raw_rows = [{"model": champion, "probabilities": entry.forecast["probabilities"],
                     "date": entry.target_at.date().isoformat()}]
    if not raw_rows and isinstance(entry.forecast.get("official"), Mapping):
        raw_rows = [dict(entry.forecast["official"], model=champion)]
    rows = []
    names = set()
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise ForecastLedgerError("probability model record is invalid")
        name = str(raw.get("model", ""))
        if not name or name in names or raw.get("date") != entry.target_at.date().isoformat():
            raise ForecastLedgerError("probability model identity or target is invalid")
        probabilities = normalized_probabilities(raw.get("probabilities", {}))
        names.add(name)
        rows.append({"model": name, "probabilities": probabilities,
                     "predicted_state": max(STATES, key=probabilities.__getitem__),
                     "fallback": bool(raw.get("fallback", False))})
    if champion not in names:
        raise ForecastLedgerError("probability operating champion is missing")
    return champion, rows


def make_probability_evaluation(entry: ForecastLedgerEntry, *, actual: str,
                                evaluated_at: datetime, current: str | None = None) -> dict[str, Any]:
    from regime_lab.forecast_ledger import ForecastLedgerError
    clock = ensure_utc(evaluated_at, field_name="evaluated_at")
    if clock < entry.target_at or actual not in STATES:
        raise ForecastLedgerError("probability target is not mature or actual state is invalid")
    champion, models = _frozen_models(entry)
    for row in models:
        p = row["probabilities"]
        row["log_loss"] = -math.log(max(p[actual], 1e-9))
        row["brier"] = sum((p[s] - float(s == actual)) ** 2 for s in STATES)
        row["correct"] = row["predicted_state"] == actual
    return {
        "schema_version": SCHEMA,
        "forecast_key": entry.key.as_dict(),
        "forecast_sha256": entry.forecast_sha256,
        "evaluated_at": clock.isoformat(),
        "target_week": entry.target_at.date().isoformat(),
        "actual_next_state": actual,
        "current_state": current if current in STATES else None,
        "operating_champion": champion,
        "models": models,
        "issue_evidence": issue_evidence(entry),
        "label_spec_sha256": entry.label_spec_sha256,
        "evidence_track": "operational_oos" if issue_evidence(entry)["eligible"] else "late_or_unverified_issue",
    }


def _validate_document(document: Mapping[str, Any], original: ForecastLedgerEntry) -> None:
    from regime_lab.forecast_ledger import ForecastLedgerError
    if document.get("schema_version") != SCHEMA or document.get("forecast_key") != original.key.as_dict():
        raise ForecastLedgerError("probability evaluation schema/key differs")
    if document.get("forecast_sha256") != original.forecast_sha256:
        raise ForecastLedgerError("probability evaluation reference hash differs from original")
    expected = make_probability_evaluation(original, actual=str(document.get("actual_next_state")),
        evaluated_at=datetime.fromisoformat(str(document.get("evaluated_at"))),
        current=document.get("current_state"))
    if dict(document) != expected:
        raise ForecastLedgerError("probability evaluation differs from frozen forecast calculation")


def append_probability_evaluation(ledger: ForecastLedger, document: Mapping[str, Any]) -> None:
    from regime_lab.forecast_ledger import (ForecastLedgerError,
        DuplicateEvaluationError, ConflictingEvaluationError)
    raw = json.dumps(document, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    key_doc = document["forecast_key"]
    key_tuple = tuple(str(key_doc[k]) for k in KEY_FIELDS)
    digest = canonical_json_sha256_v1(document)
    with ledger._lock:
        connection = ledger._connection
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(f"SELECT {FORECAST_PROJECTION} FROM forecast_ledger WHERE {WHERE_KEY}", key_tuple).fetchone()
            if row is None:
                raise ForecastLedgerError("probability evaluation requires an existing forecast")
            original = probability_forecast_from_row(row)
            _validate_document(document, original)
            existing = connection.execute(f"SELECT evaluation_sha256 FROM forecast_probability_evaluations WHERE {WHERE_KEY}", key_tuple).fetchone()
            if existing is not None:
                if existing["evaluation_sha256"] == digest:
                    raise DuplicateEvaluationError("probability evaluation already exists")
                raise ConflictingEvaluationError("probability evaluation is immutable")
            connection.execute("INSERT INTO forecast_probability_evaluations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (*key_tuple, document["evaluated_at"], raw, digest))
        except BaseException:
            connection.rollback()
            raise
        else:
            connection.commit()


def read_probability_evaluations(connection: sqlite3.Connection) -> tuple[dict[str, Any], ...]:
    """Also supports pre-migration read-only databases without creating tables."""
    from regime_lab.forecast_ledger import ForecastLedgerError
    exists = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='forecast_probability_evaluations'").fetchone()
    if exists is None:
        return ()
    output = []
    for row in connection.execute("SELECT * FROM forecast_probability_evaluations ORDER BY target_at, decision_at"):
        document = json.loads(row["evaluation_json"])
        if document.get("evaluated_at") != row["evaluated_at"]:
            raise ForecastLedgerError("stored probability evaluation clock is inconsistent")
        if canonical_json_sha256_v1(document) != row["evaluation_sha256"]:
            raise ForecastLedgerError("stored probability evaluation hash is invalid")
        key_tuple = tuple(row[k] for k in KEY_FIELDS)
        original = connection.execute(f"SELECT {FORECAST_PROJECTION} FROM forecast_ledger WHERE {WHERE_KEY}", key_tuple).fetchone()
        if original is None:
            raise ForecastLedgerError("stored probability evaluation has no original forecast")
        _validate_document(document, probability_forecast_from_row(original))
        output.append(document)
    return tuple(output)


def mature_probability_evaluations(ledger: ForecastLedger, *, states: pd.Series,
                                  evaluated_at: datetime, label_spec_sha256: str | None = None) -> dict[str, Any]:
    from regime_lab.forecast_ledger import ForecastLedgerError, DuplicateEvaluationError
    clock = ensure_utc(evaluated_at, field_name="evaluated_at")
    if not isinstance(states.index, pd.DatetimeIndex) or states.index.has_duplicates:
        raise ValueError("probability actual states require a unique dated index")
    dates = {at.date(): at for at in states.index}
    if len(dates) != len(states):
        raise ValueError("probability actual states have duplicate week dates")
    existing = {tuple(d["forecast_key"][k] for k in KEY_FIELDS) for d in ledger.list_probability_evaluations()}
    appended, unresolved, pending = [], [], 0
    for entry in ledger.list_probability_forecasts():
        if entry.key.as_sql_tuple() in existing:
            continue
        if entry.target_at > clock:
            pending += 1
            continue
        when = dates.get(entry.target_at.date())
        if label_spec_sha256 is not None and entry.label_spec_sha256 != label_spec_sha256:
            unresolved.append({"forecast_key": entry.key.as_dict(), "reason": "label_spec_mismatch"})
            continue
        if when is None or states.loc[when] not in STATES:
            unresolved.append({"forecast_key": entry.key.as_dict(), "reason": "actual_next_state_unavailable"})
            continue
        origin = dates.get(entry.origin_week)
        try:
            document = make_probability_evaluation(entry, actual=str(states.loc[when]),
                current=str(states.loc[origin]) if origin is not None else None, evaluated_at=clock)
            append_probability_evaluation(ledger, document)
        except DuplicateEvaluationError:
            continue
        except (ForecastLedgerError, ValueError, TypeError) as exc:
            unresolved.append({"forecast_key": entry.key.as_dict(), "reason": f"probability_contract:{exc}"})
            continue
        appended.append(document)
    return {"appended": appended, "pending_count": pending, "unresolved": unresolved}


def probability_summary(documents: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    # Re-publications are separate immutable records, not extra independent
    # weeks. The first verified issue is a fixed, outcome-independent policy.
    eligible_entries = [d for d in documents if d["issue_evidence"]["eligible"]]
    label_versions = {d["label_spec_sha256"] for d in eligible_entries}
    if len(label_versions) > 1:
        raise ValueError("probability summary must compare one label specification at a time")
    eligible_by_week = {}
    for document in sorted(eligible_entries, key=lambda d: (
        d["issue_evidence"]["issued_at"], d["forecast_key"]["decision_at"],
        canonical_json_sha256_v1(d["forecast_key"]),
    )):
        eligible_by_week.setdefault(document["target_week"], document)
    eligible = [eligible_by_week[k] for k in sorted(eligible_by_week)]
    completed_weeks = len({d["target_week"] for d in documents})
    rows = []
    for document in eligible:
        champion = next(r for r in document["models"] if r["model"] == document["operating_champion"])
        rows.append((document, champion))
    benchmarks = {}
    for baseline in ("markov", "persistence"):
        pairs = [(champion, b) for d, champion in rows for b in d["models"] if b["model"] == baseline]
        benchmarks[baseline] = {"matched_n": len(pairs),
            "log_loss_improvement": float(np.mean([b["log_loss"] - c["log_loss"] for c,b in pairs])) if pairs else None,
            "brier_improvement": float(np.mean([b["brier"] - c["brier"] for c,b in pairs])) if pairs else None,
            "status": "available" if pairs else "not_frozen_in_completed_entries"}
    return {
        "schema_version": "regime-probability-score-summary/1",
        "evaluation_basis": "independent_of_investment_execution",
        "completed_entries": len(documents),
        "completed_weeks": completed_weeks,
        "prospective_completed_weeks": len(eligible),
        "excluded_late_or_unverified_weeks": completed_weeks - len(eligible),
        "duplicate_target_entries": len(documents) - completed_weeks,
        "weekly_aggregation_policy": "earliest_verified_issue_per_target_week",
        "log_loss": float(np.mean([r["log_loss"] for _,r in rows])) if rows else None,
        "brier": float(np.mean([r["brier"] for _,r in rows])) if rows else None,
        "benchmarks": benchmarks,
        "evaluation_manifest_sha256": canonical_json_sha256_v1(list(documents)),
        "rows": [{"target_week": d["target_week"], "model": r["model"],
                  "log_loss": r["log_loss"], "brier": r["brier"]} for d,r in rows],
    }
