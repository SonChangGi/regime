"""Local candidate packets: actual-time issuance and immutable endpoint scores."""
from __future__ import annotations

from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import fcntl
from io import BytesIO
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Mapping

import pandas as pd

from regime_lab.analysis.decision_shadow import _scheduled_nyse_entry_at
from regime_lab.integrity import canonical_json_sha256_v1 as digest
from regime_lab.operational_forecast import frame_sha256
from regime_lab.schema import STATE_ORDER
from regime_lab.candidate_recipes import candidate_model_recipes

SCHEMA = "regime-weekly-candidates-summary/1"
LEDGER_NAME = "candidate-forecasts.local.sqlite3"


def _now():
    return datetime.now(timezone.utc)


def _encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _at(value):
    at = pd.Timestamp(value)
    if at.tzinfo is None or pd.isna(at):
        raise ValueError("candidate timestamps must be timezone aware")
    return at.tz_convert("UTC")


def _connect(directory: Path):
    raw = directory.expanduser().absolute()
    root = Path(__file__).resolve().parents[2]
    resolved = raw.resolve()
    if any(part.is_symlink() for part in (raw, *raw.parents)):
        raise ValueError("candidate workspace must not use symlinks")
    if any(resolved == root / name or root / name in resolved.parents
           for name in ("publication", "web", "data", "artifacts")):
        raise ValueError("candidate workspace must be an isolated local output")
    resolved.mkdir(parents=True, exist_ok=True)
    marker = resolved / "candidate-workspace.json"
    db_path = resolved / LEDGER_NAME
    identity = {"schema_version": "regime-candidate-workspace/1", "scope": "local_preview_only",
                "ledger": str(db_path)}
    if marker.is_symlink() or db_path.is_symlink():
        raise ValueError("candidate workspace identity must not be redirected")
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError("candidate workspace identity differs")
    elif db_path.exists():
        raise ValueError("unmarked candidate ledger")
    else:
        with tempfile.NamedTemporaryFile(mode="w", dir=resolved, prefix=".candidate-marker-", delete=False) as handle:
            handle.write(_encoded(identity) + "\n")
            temporary = Path(handle.name)
        try:
            os.link(temporary, marker)
        except FileExistsError:
            if json.loads(marker.read_text()) != identity:
                raise ValueError("candidate workspace identity differs")
        finally:
            temporary.unlink()
    db = sqlite3.connect(db_path, timeout=30)
    db.row_factory = sqlite3.Row
    db.executescript("""
        CREATE TABLE IF NOT EXISTS packets (
            origin TEXT PRIMARY KEY, content TEXT NOT NULL, sha256 TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS scores (
            origin TEXT NOT NULL, model TEXT NOT NULL, horizon INTEGER NOT NULL,
            content TEXT NOT NULL, sha256 TEXT NOT NULL, PRIMARY KEY(origin, model, horizon));
        CREATE TABLE IF NOT EXISTS revisions (
            origin TEXT NOT NULL, model TEXT NOT NULL, horizon INTEGER NOT NULL,
            actual TEXT NOT NULL, observed_at TEXT NOT NULL, states_sha256 TEXT NOT NULL,
            PRIMARY KEY(origin, model, horizon, actual));
    """)
    return db


def _records(db, table):
    result = []
    for row in db.execute(f"SELECT content, sha256 FROM {table}"):
        value = json.loads(row["content"])
        if digest(value) != row["sha256"]:
            raise ValueError("frozen candidate ledger content changed")
        result.append(value)
    return result


def read_candidate_evidence(payload_raw: bytes, enhancement_raw: bytes, states_raw: bytes,
                            artifact_manifest: Mapping):
    """Verify a completed research run or the normal weekly generation binding."""
    payload, enhancement = json.loads(payload_raw), json.loads(enhancement_raw)
    hashes = {"payload": hashlib.sha256(payload_raw).hexdigest(),
              "states": hashlib.sha256(states_raw).hexdigest()}
    enhancement_hash = hashlib.sha256(enhancement_raw).hexdigest()
    if artifact_manifest.get("schema_version") == "regime-generation-manifest/2":
        from regime_lab.forecast_enhancement_publication import DECLARATION, validate_binding
        from regime_lab.integrity import canonical_json_sha256_v1_without_generation_binding
        validate_binding(enhancement, payload)
        if (DECLARATION not in payload.get("research", {})
                or artifact_manifest.get("generation_id") != payload["meta"]["generation_id"]
                or payload["meta"].get("generation_manifest_sha256") != digest(artifact_manifest)
                or artifact_manifest.get("payload", {}).get("payload_contract_sha256")
                != canonical_json_sha256_v1_without_generation_binding(payload)):
            raise ValueError("candidate weekly generation manifest differs")
        states = pd.read_pickle(BytesIO(states_raw))
        if enhancement.get("provenance", {}).get("input_frames", {}).get("states") != frame_sha256(states):
            raise ValueError("candidate weekly generation state snapshot differs")
        evidence_type = "weekly_generation"
    else:
        if (artifact_manifest.get("status") != "validated_local_research"
                or artifact_manifest.get("files", {}).get("forecast-enhancements.json") != enhancement_hash
                or any(artifact_manifest.get("source_hashes", {}).get(k) != v for k, v in hashes.items())
                or any(enhancement.get("provenance", {}).get("input_sha256", {}).get(k) != v for k, v in hashes.items())):
            raise ValueError("candidate artifacts differ from their completed run receipt")
        states = pd.read_pickle(BytesIO(states_raw))
        evidence_type = "completed_research_run"
    receipt = {"source_bytes_sha256": hashes, "enhancement_bytes_sha256": enhancement_hash,
               "artifact_manifest_sha256": digest(artifact_manifest), "payload_sha256": digest(payload),
               "enhancement_sha256": digest(enhancement), "states_sha256": frame_sha256(states),
               "evidence_type": evidence_type}
    return payload, enhancement, states, receipt


def _validate_inputs(payload: Mapping, enhancement: Mapping, states: pd.Series,
                     state_manifest: Mapping, evidence: Mapping, now):
    if (evidence.get("payload_sha256") != digest(payload)
            or evidence.get("enhancement_sha256") != digest(enhancement)
            or evidence.get("states_sha256") != frame_sha256(states)
            or not evidence.get("artifact_manifest_sha256")):
        raise ValueError("candidate evidence receipt differs from inputs")
    model_key = enhancement.get("provenance", {}).get("model_cache_key", {})
    if (model_key.get("schema_version") != "forecast-model-cache/2"
            or model_key.get("protocol") != enhancement.get("protocol")
            or not model_key.get("source_sha256") or not model_key.get("runtime")):
        raise ValueError("candidate numerical recipe binding is missing or differs")
    origin = _at(payload["meta"]["data_as_of"])
    if (enhancement.get("schema_version") != "regime-forecast-enhancements/1"
            or enhancement.get("automatic_promotion") is not False
            or enhancement.get("source_generation_id") != payload["meta"]["generation_id"]
            or _at(enhancement["data_as_of"]) != origin):
        raise ValueError("candidate source generation differs")
    if not origin <= _at(enhancement["generated_at"]) <= now:
        raise ValueError("candidate generation is future dated or predates its inputs")
    if (not isinstance(states.index, pd.DatetimeIndex) or states.index.tz is None
            or states.empty or states.index.has_duplicates or not states.index.is_monotonic_increasing
            or not states.isin(STATE_ORDER).all()):
        raise ValueError("candidate labels require unique ordered dated states")
    if (_at(states.index[-1]) != origin or origin > now
            or frame_sha256(states) != state_manifest.get("frames", {}).get("states")):
        raise ValueError("candidate states differ from independent input manifest or cutoff")
    if state_manifest.get("data_as_of") and _at(state_manifest["data_as_of"]) != origin:
        raise ValueError("candidate input manifest cutoff differs")
    for week in payload["weekly"]:
        at = _at(week["data_as_of"])
        if at not in states.index or states.loc[at] != week["current"]["state"]:
            raise ValueError("candidate scoring labels differ from published states")
    selected = []
    seen = set()
    for row in enhancement["latest"]:
        h = row["horizon_weeks"]
        if type(h) is not int or h not in (1, 4, 13) or row.get("target") != "endpoint":
            raise ValueError("candidate endpoint horizon is unsupported")
        key = (row["model"], h)
        if key in seen or _at(row["origin_date"]) != origin:
            raise ValueError("duplicate or stale candidate prediction")
        seen.add(key)
        target = (origin.tz_convert("America/New_York") + pd.DateOffset(weeks=h)).tz_convert("UTC")
        p = row["probabilities"]
        if (_at(row["target_date"]) != target or row.get("actual") is not None
                or row["current_state"] != states.iloc[-1]
                or set(p) != set(STATE_ORDER)
                or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in p.values())
                or not math.isclose(sum(p.values()), 1.0, abs_tol=1e-8)):
            raise ValueError("candidate target, state or probabilities differ")
        if row.get("last_train_target") and _at(row["last_train_target"]) >= origin:
            raise ValueError("candidate training contains unresolved outcomes")
        selected.append({"model": row["model"], "horizon_weeks": h, "target_at": target.isoformat(),
                         "probabilities": dict(p), "fallback": bool(row.get("fallback", False))})
    if not selected or {r["horizon_weeks"] for r in selected} != {1, 4, 13}:
        raise ValueError("candidate packet requires all three endpoint horizons")
    return origin, sorted(selected, key=lambda r: (r["model"], r["horizon_weeks"]))


def run_weekly_candidates(directory: Path, *, payload: Mapping, enhancement: Mapping,
                          states: pd.Series, state_manifest: Mapping, evidence: Mapping,
                          verify_inputs=None) -> dict:
    """Score prior packets, then freeze this week's forecasts before next market open.

    Each origin can be issued once. Re-runs retain its first forecast and score;
    changed labels are recorded as revisions rather than rewriting performance.
    No operational ledger, source payload, or automation schedule is touched.
    """
    now = _at(_now())
    origin, predictions = _validate_inputs(payload, enhancement, states, state_manifest, evidence, now)
    label_hash = payload["label"]["spec_sha256"]
    identities = candidate_model_recipes(payload, enhancement)
    for row in predictions:
        row["recipe_sha256"] = identities[row["model"]]["recipe_sha256"]
    recipe = {"schema_version": "regime-candidate-recipe-set/2", "models": identities}
    recipe_hash = digest(recipe)
    deadline = _scheduled_nyse_entry_at((origin + pd.DateOffset(weeks=1)).date().isoformat()).tz_convert("UTC")
    state_hash = frame_sha256(states)
    with closing(_connect(directory)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        packets = _records(db, "packets")
        if packets and origin < max(_at(p["origin_at"]) for p in packets):
            raise ValueError("candidate generation cannot move backward")
        scores = _records(db, "scores")
        existing_scores = {(s["origin_at"], s["model"], s["horizon_weeks"]): s for s in scores}
        for packet in packets:
            if packet["label_spec_sha256"] != label_hash:
                continue
            for row in packet["predictions"]:
                target = _at(row["target_at"])
                if target > now or target not in states.index:
                    continue
                actual = states.loc[target]
                key = (packet["origin_at"], row["model"], row["horizon_weeks"])
                if key in existing_scores:
                    if existing_scores[key]["actual"] != actual:
                        db.execute("INSERT OR IGNORE INTO revisions VALUES (?, ?, ?, ?, ?, ?)",
                                   (*key, actual, now.isoformat(), state_hash))
                    continue
                p = row["probabilities"]
                score = {"origin_at": key[0], "model": key[1], "horizon_weeks": key[2],
                         "recipe_sha256": row.get("recipe_sha256") or digest({"legacy_packet": digest(packet), "model": row["model"]}),
                         "recipe_binding": "model" if row.get("recipe_sha256") else "legacy_packet",
                         "target_at": row["target_at"],
                         "actual": actual, "evaluated_at": now.isoformat(), "states_sha256": state_hash,
                         "packet_sha256": digest(packet), "log_loss": -math.log(max(p[actual], 1e-15)),
                         "brier": sum((p[s] - int(s == actual)) ** 2 for s in STATE_ORDER),
                         "correct": max(STATE_ORDER, key=lambda s: p[s]) == actual}
                db.execute("INSERT INTO scores VALUES (?, ?, ?, ?, ?)", (*key, _encoded(score), digest(score)))
        previous = next((p for p in packets if _at(p["origin_at"]) == origin), None)
        if previous:
            # No substitution even if a later generation changes this week's candidate.
            latest = {**previous, "status": "already_issued"}
        elif now >= deadline:
            latest = {"status": "deadline_missed", "origin_at": origin.isoformat(),
                      "issued_at": None, "issue_deadline_at": deadline.isoformat(), "predictions": []}
        else:
            published = _at(_now())
            if published < now or published >= deadline:
                raise ValueError("candidate issue clock changed or deadline elapsed")
            packet = {"schema_version": "regime-candidate-packet/1", "scope": "local_preview_only",
                      "origin_at": origin.isoformat(), "issued_at": published.isoformat(),
                      "issue_deadline_at": deadline.isoformat(), "source_generation_id": payload["meta"]["generation_id"],
                      "source_payload_sha256": digest(payload), "enhancement_sha256": digest(enhancement),
                      "state_manifest_sha256": digest(state_manifest), "states_sha256": state_hash,
                      "evidence_receipt": dict(evidence),
                      "recipe_sha256": recipe_hash, "recipe": recipe, "label_spec_sha256": label_hash,
                      "current_state": states.iloc[-1], "predictions": predictions}
            db.execute("INSERT INTO packets VALUES (?, ?, ?)", (origin.isoformat(), _encoded(packet), digest(packet)))
            latest = {**packet, "status": "issued_local_preview"}
        packets = _records(db, "packets")
        scores = _records(db, "scores")
        revisions = [dict(r) for r in db.execute("SELECT * FROM revisions")]
        if verify_inputs is not None:
            verify_inputs()
    groups = defaultdict(list)
    for score in scores:
        # Old receipts remain immutable. Incomplete old identities are never
        # pooled across packets or silently promoted to a current model version.
        version = (score["recipe_sha256"] if score.get("recipe_binding") == "model"
                   else digest({"legacy_packet": score["packet_sha256"], "model": score["model"]}))
        groups[(version, score["model"], score["horizon_weeks"])].append(score)
    metrics = [{"recipe_sha256": key[0], "model": key[1], "horizon_weeks": key[2], "n": len(rows),
                "log_loss": sum(r["log_loss"] for r in rows) / len(rows),
                "brier": sum(r["brier"] for r in rows) / len(rows),
                "accuracy": sum(r["correct"] for r in rows) / len(rows)} for key, rows in sorted(groups.items())]
    total = sum(len(p["predictions"]) for p in packets)
    scored_keys = {(r["origin_at"], r["model"], r["horizon_weeks"]) for r in scores}
    overdue = sum(_at(r["target_at"]) <= now and (p["origin_at"], r["model"], r["horizon_weeks"]) not in scored_keys
                  for p in packets for r in p["predictions"])
    return {"schema_version": SCHEMA, "scope": "local_preview_only", "as_of": now.isoformat(),
            "source_generation_id": payload["meta"]["generation_id"], "data_as_of": origin.isoformat(),
            "latest_origin_at": origin.isoformat(), "status": latest["status"], "latest": latest,
            "issued_packets": len(packets), "issued_predictions": total, "pending_predictions": total - len(scores),
            "matured_predictions": len(scores), "overdue_predictions": overdue, "late_packets": 0,
            "models": metrics, "revision_conflicts": revisions, "active_recipe_sha256": recipe_hash,
            "active_model_recipe_sha256": {name: value["recipe_sha256"] for name, value in identities.items()},
            "score_population": "issued_before_next_nyse_open", "automatic_promotion": False}


def run_weekly_candidate_files(directory: Path, *, payload_path: Path, enhancement_path: Path,
                               states_path: Path, input_manifest_path: Path,
                               artifact_manifest_path: Path) -> dict:
    """Shared CLI/automation entry: serialize issuance and summary replacement."""
    directory = Path(directory)
    _connect(directory).close()
    descriptor = os.open(directory.resolve() / "weekly-run.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        paths = tuple(map(Path, (payload_path, enhancement_path, states_path, input_manifest_path, artifact_manifest_path)))
        originals = {path: path.read_bytes() for path in paths}
        payload, enhancement, states, evidence = read_candidate_evidence(
            originals[paths[0]], originals[paths[1]], originals[paths[2]], json.loads(originals[paths[4]]))

        def verify():
            if any(path.read_bytes() != raw for path, raw in originals.items()):
                raise RuntimeError("candidate inputs changed while recording")

        result = run_weekly_candidates(directory, payload=payload, enhancement=enhancement,
                                       states=states, state_manifest=json.loads(originals[paths[3]]),
                                       evidence=evidence, verify_inputs=verify)
        from regime_lab.io import write_json_atomic
        write_json_atomic(directory.resolve() / "candidate-summary.json", result)
        return result
