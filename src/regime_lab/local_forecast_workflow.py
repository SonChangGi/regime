"""Opt-in local forecast issuance and scoring on a marked SQLite copy."""
from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping

import pandas as pd

from regime_lab.forecast_ledger import (ForecastLedger, ForecastLedgerEntry,
    build_operational_diagnostics, operational_input_manifest_sha256)
from regime_lab.forecast_probability import mature_probability_evaluations
from regime_lab.integrity import canonical_json_sha256_v1
from regime_lab.operational_forecast import frame_sha256, validate_preparation_recipe

WORKSPACE_SCHEMA = "regime-local-forecast-workspace/1"
LEDGER_NAME = "forecast-ledger.local.sqlite3"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_once(path: Path, document: Mapping[str, Any]) -> None:
    encoded = json.dumps(document, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"local artifact already exists with different content: {path.name}")
        return
    with path.open("x", encoding="utf-8") as output:
        output.write(encoded)


def initialize_local_copy(source: Path, directory: Path) -> dict:
    source = source.expanduser().resolve(strict=True)
    directory = directory.expanduser().resolve()
    root = Path(__file__).resolve().parents[2]
    protected = [root / name for name in ("publication", "web", "data", "artifacts")]
    if directory == source.parent or any(directory == p or p in directory.parents for p in protected):
        raise ValueError("choose a separate local preview output directory")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / LEDGER_NAME
    marker = directory / "local-workspace.json"
    if target.exists() or marker.exists():
        raise ValueError("local workspace already exists; use its score or summary command")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as original:
        if original.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='forecast_ledger'").fetchone() is None:
            raise ValueError("source is not a forecast ledger")
        with sqlite3.connect(target) as copied:
            original.backup(copied)
            if copied.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("copied ledger failed SQLite integrity check")
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise ValueError("source ledger changed while making the isolated copy")
    document = {"schema_version": WORKSPACE_SCHEMA, "source": str(source), "source_sha256": digest,
        "ledger": str(target), "created_at": _now().isoformat(),
        "scope": "local_preview_only", "source_modified": False}
    _write_once(marker, document)
    # Schema extensions apply exclusively to the marked copy.
    with ForecastLedger(target):
        pass
    return document


def _workspace(directory: Path) -> tuple[Path, dict]:
    raw = directory.expanduser()
    if raw.is_symlink():
        raise ValueError("local workspace must not be a symlink")
    directory = raw.resolve(strict=True)
    marker = directory / "local-workspace.json"
    if marker.is_symlink():
        raise ValueError("local workspace marker must not be a symlink")
    document = _json(marker)
    target = directory / LEDGER_NAME
    if document.get("schema_version") != WORKSPACE_SCHEMA or document.get("scope") != "local_preview_only":
        raise ValueError("an explicitly initialized local workspace is required")
    if target.is_symlink() or str(target) != document.get("ledger") or target.resolve() == Path(document["source"]).resolve():
        raise ValueError("local ledger identity differs from its copy receipt")
    return target, document


def local_summary(directory: Path, *, probability_maturity: Mapping[str, Any] | None = None) -> dict:
    target, receipt = _workspace(directory)
    with ForecastLedger(target) as ledger:
        entries = ledger.list_probability_forecasts()
        operating = [e for e in entries if e.forecast.get("evidence_track") != "local_preview"]
        trials = [e for e in entries if e.forecast.get("evidence_track") == "local_preview"]
        keys = {e.key.as_sql_tuple() for e in operating}
        from regime_lab.forecast_probability import KEY_FIELDS
        belongs = lambda d: tuple(d["forecast_key"][k] for k in KEY_FIELDS) in keys
        scores = [d for d in ledger.list_probability_evaluations() if belongs(d)]
        revisions = [d for d in ledger.list_probability_revisions() if belongs(d)]
        diagnostics = build_operational_diagnostics(operating,
            [e for e in ledger.list_evaluations() if e.forecast_key.as_sql_tuple() in keys],
            probability_evaluations=scores, probability_revision_conflicts=revisions, as_of=_now())
        if probability_maturity:
            maturity = diagnostics["probability_maturity"]
            maturity["label_snapshot"] = probability_maturity["label_snapshot"]
            maturity["unresolved"] = [d for d in probability_maturity["unresolved"] if belongs(d)]
            maturity["coverage"]["unresolved_entries"] = len(maturity["unresolved"])
            if maturity["unresolved"]:
                maturity["coverage"]["status"] = "needs_attention"
    return {"schema_version": "regime-local-forecast-summary/1", "scope": "local_preview_only",
        "source_sha256": receipt["source_sha256"], "ledger": str(target), "operational_diagnostics": diagnostics,
        "local_trial": {"issued_entries": len(trials), "pending_entries": sum(e.target_at > _now() for e in trials),
                        "included_in_operational_scores": False}}


def score_local_copy(directory: Path, *, states: pd.Series, state_manifest: Mapping[str, Any],
                     label_spec_sha256: str, states_available_at: datetime | None = None) -> dict:
    target, _ = _workspace(directory)
    if frame_sha256(states) != state_manifest.get("frames", {}).get("states"):
        raise ValueError("states differ from the independent input manifest")
    with ForecastLedger(target) as ledger:
        maturity = mature_probability_evaluations(ledger, states=states, evaluated_at=_now(),
            label_spec_sha256=label_spec_sha256, states_available_at=states_available_at)
    return local_summary(directory, probability_maturity=maturity)


def issue_local_preparation(directory: Path, *, prepared: Mapping[str, Any],
                            recipe_lock: Mapping[str, Any], locked_payload: Mapping[str, Any]) -> dict:
    """Freeze a fresh local preview; never turn a historical replay into issuance."""
    target, _ = _workspace(directory)
    validate_preparation_recipe(recipe_lock, locked_payload)
    body = {k: v for k, v in prepared.items() if k != "document_sha256"}
    if prepared.get("document_sha256") != canonical_json_sha256_v1(body):
        raise ValueError("prepared forecast content hash differs")
    if (prepared.get("research_replay") or not prepared.get("local_issue_eligible")
            or prepared.get("status") != "prepared_for_review"
            or prepared.get("recipe_lock_sha256") != recipe_lock["sha256"]):
        raise ValueError("only a fresh independently verified preparation can be issued locally")
    now = _now()
    prepared_at = datetime.fromisoformat(prepared["prepared_at"])
    if not prepared_at <= now < datetime.fromisoformat(prepared["issue_deadline_at"]):
        raise ValueError("local preparation is expired or backdated")
    origin = date.fromisoformat(prepared["origin_at"][:10])
    with ForecastLedger(target) as ledger:
        entries = ledger.list_probability_forecasts()
        previous = [e for e in entries if e.forecast.get("preparation_key_sha256") == prepared["key_sha256"]]
        if previous:
            return {"status": "already_issued", "forecast_sha256": previous[0].forecast_sha256, "scope": "local_preview_only"}
        templates = [e for e in entries if e.origin_week == origin
            and e.model_manifest_sha256 == locked_payload["model"]["candidate_manifest_sha256"]
            and e.label_spec_sha256 == locked_payload["label"]["spec_sha256"]]
        if not templates:
            raise ValueError("copy has no matching frozen operational input receipt for this origin and model")
        template = ledger.read(max(templates, key=lambda e: e.decision_at).key)
        if template is None:
            raise ValueError("matching copied input receipt is missing")
        artifact = directory.resolve() / f"issued-{prepared['key_sha256']}.json"
        _write_once(artifact, prepared)
        published = _now()
        if published >= datetime.fromisoformat(prepared["issue_deadline_at"]):
            raise ValueError("local issue deadline elapsed before publication completed")
        forecast = {"schema_version": "regime-local-operational-forecast/1", "evidence_track": "local_preview",
            "local_publication_at": published.isoformat(), "issue_deadline_at": prepared["issue_deadline_at"],
            "input_cutoff_at": prepared["input_cutoff_at"], "preparation_key_sha256": prepared["key_sha256"],
            "prepared_document_sha256": prepared["document_sha256"], "prepared_input_hashes": prepared["input_hashes"],
            "recipe_lock_sha256": recipe_lock["sha256"], "champion": prepared["champion"],
            "selection": {"operating_champion": prepared["champion"]},
            "current": template.forecast.get("current", {}), "official": prepared["forecast"],
            "model_forecasts": prepared["model_forecasts"]}
        entry = ForecastLedgerEntry(origin_week=origin, decision_at=prepared_at,
            target_at=datetime.fromisoformat(prepared["target_at"]),
            label_spec_sha256=template.label_spec_sha256, model_manifest_sha256=template.model_manifest_sha256,
            input_snapshot_sha256=operational_input_manifest_sha256(template.operational_inputs),
            operational_inputs=template.operational_inputs, forecast=forecast)
        ledger.append(entry)
    return {"status": "issued_local_preview", "forecast_sha256": entry.forecast_sha256,
            "artifact": str(artifact), "scope": "local_preview_only", "automatic_promotion": False}
