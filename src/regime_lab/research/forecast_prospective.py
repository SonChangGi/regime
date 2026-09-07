"""Frozen, append-only prospective research evidence, separate from execution.

Preview is pure. Registering a protocol and issuing a local publication are
explicit operations on a caller-selected research database. Production methods
take their timestamps from the clock, never from a replay argument.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
from importlib.metadata import version as dependency_version
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import shutil

import numpy as np

from regime_lab.schema import STATE_ORDER
from regime_lab.data.release_archive import weekly_decision_at


PROTOCOL_SCHEMA = "regime-forecast-prospective-protocol/1"
FORECAST_SCHEMA = "regime-forecast-prospective-frozen/1"
STATES_SCHEMA = "regime-forecast-official-states/1"
SUMMARY_SCHEMA = "regime-forecast-prospective-evidence/1"
CAPTURE_SCHEMA = "regime-forecast-prospective-input-capture/1"
HORIZONS = (1, 4, 13)
TARGETS = ("endpoint", "first_departure", "any_risk_off_entry", "any_risk_off_occupancy")
FIRST_DESTINATIONS = ("no_departure", *STATE_ORDER)
APPLICATION_ID = 0x52465231
EPSILON = 1e-15
TOLERANCE = 1e-10


class ResearchLedgerError(ValueError):
    """Invalid research contract, chronology, or immutable record."""


class ResearchConflictError(ResearchLedgerError):
    """An existing immutable key or evaluated official path has changed."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc(value, name="timestamp") -> datetime:
    try:
        result = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ResearchLedgerError(f"{name} must be an explicit timezone-aware timestamp") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ResearchLedgerError(f"{name} must be an explicit timezone-aware timestamp")
    return result.astimezone(timezone.utc)


def _json(value) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ResearchLedgerError("content must be finite JSON") from exc


def content_sha256(value) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _copy(value):
    return json.loads(_json(value))


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ResearchLedgerError(f"{name} must be a lowercase SHA-256")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ResearchLedgerError(f"{name} must be a nonempty trimmed string")
    return value


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ResearchLedgerError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value, name, lower=None, upper=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ResearchLedgerError(f"{name} must be finite")
    if (lower is not None and value < lower) or (upper is not None and value > upper):
        raise ResearchLedgerError(f"{name} is outside the allowed range")
    return value


def _object(value, name):
    if not isinstance(value, Mapping):
        raise ResearchLedgerError(f"{name} must be an object")
    return value


def _required(value, fields, name):
    _object(value, name)
    if not set(fields) <= set(value):
        raise ResearchLedgerError(f"{name} missing fields: {sorted(set(fields) - set(value))}")


def _distribution(value, keys, name):
    _object(value, name)
    if set(value) != set(keys):
        raise ResearchLedgerError(f"{name} must have exactly {list(keys)}")
    for key in keys:
        _number(value[key], f"{name}.{key}", 0, 1)
    if not math.isclose(sum(value.values()), 1, abs_tol=TOLERANCE, rel_tol=0):
        raise ResearchLedgerError(f"{name} must sum to one; probabilities are never repaired")
    return dict(value)


def validate_protocol(protocol: Mapping) -> dict:
    """Validate an explicitly supplied recipe, calendar, and readiness gates.

    Calendar entries are exact official observation timestamps (including
    holidays/DST). All but the final 13 are required issuance origins. Extending
    or changing a protocol requires a new version and evidence series.
    """
    p = _copy(protocol)
    _required(p, ("schema_version", "version", "recipe_sha256", "label", "models", "calendar", "criteria",
                  "issue_deadline_seconds", "max_generation_age_seconds"), "protocol")
    if p["schema_version"] != PROTOCOL_SCHEMA:
        raise ResearchLedgerError("unsupported protocol schema")
    _text(p["version"], "protocol.version")
    _sha(p["recipe_sha256"], "protocol.recipe_sha256")
    _required(p["label"], ("version", "sha256"), "label")
    _text(p["label"]["version"], "label.version")
    _sha(p["label"]["sha256"], "label.sha256")
    for key in ("issue_deadline_seconds", "max_generation_age_seconds"):
        _integer(p[key], key)
    if not isinstance(p["calendar"], list) or len(p["calendar"]) <= 13:
        raise ResearchLedgerError("calendar requires issuance origins and 13 maturity-tail dates")
    dates = [_utc(value, "calendar") for value in p["calendar"]]
    if len(set(dates)) != len(dates) or dates != sorted(dates):
        raise ResearchLedgerError("calendar must be unique and chronological")
    if any(not timedelta(days=4) <= b - a <= timedelta(days=10) for a, b in zip(dates, dates[1:])):
        raise ResearchLedgerError("calendar must contain every weekly observation without gaps")
    if any(a + timedelta(seconds=p["issue_deadline_seconds"]) >= b for a, b in zip(dates, dates[1:])):
        raise ResearchLedgerError("issuance deadline must precede the next observation")
    p["calendar"] = [d.isoformat() for d in dates]
    if not isinstance(p["models"], list) or not p["models"]:
        raise ResearchLedgerError("models must be a nonempty list")
    models = {}
    for model in p["models"]:
        _required(model, ("id", "version", "recipe_sha256", "role", "horizons"), "model")
        for key in ("id", "version"):
            _text(model[key], f"model.{key}")
        _sha(model["recipe_sha256"], "model.recipe_sha256")
        if model["id"] in models:
            raise ResearchLedgerError("model ids must be unique")
        if model["role"] not in ("candidate", "baseline"):
            raise ResearchLedgerError("model role must be candidate or baseline")
        if model["horizons"] not in ([1], [1, 4, 13]) or any(isinstance(h, bool) for h in model["horizons"]):
            raise ResearchLedgerError("model horizons must be [1] or [1, 4, 13]")
        models[model["id"]] = model
    candidates = [m for m in models.values() if m["role"] == "candidate"]
    if not candidates:
        raise ResearchLedgerError("at least one candidate is required")
    for model in candidates:
        baseline = models.get(model.get("baseline_id"))
        if baseline is None or baseline["role"] != "baseline" or not set(model["horizons"]) <= set(baseline["horizons"]):
            raise ResearchLedgerError("each candidate requires a frozen baseline covering its horizons")
    criteria = p["criteria"]
    integer_fields = ("minimum_paired_origins", "minimum_worsening_events", "minimum_recovery_events",
                      "minimum_risk_off_episodes", "minimum_binary_events", "minimum_binary_nonevents",
                      "block_length", "bootstrap_draws")
    _required(criteria, (*integer_fields, "maximum_missing_origin_fraction", "maximum_log_loss_delta_ci_high",
                         "maximum_brier_delta", "minimum_worsening_recall", "maximum_false_alarm_fraction"), "criteria")
    for key in integer_fields:
        _integer(criteria[key], f"criteria.{key}")
    if criteria["bootstrap_draws"] < 199:
        raise ResearchLedgerError("bootstrap_draws must be >= 199")
    if criteria["minimum_paired_origins"] < 2 * criteria["block_length"]:
        raise ResearchLedgerError("minimum_paired_origins must cover at least two bootstrap blocks")
    for key in ("maximum_missing_origin_fraction", "minimum_worsening_recall", "maximum_false_alarm_fraction"):
        _number(criteria[key], f"criteria.{key}", 0, 1)
    for key in ("maximum_log_loss_delta_ci_high", "maximum_brier_delta"):
        _number(criteria[key], f"criteria.{key}", upper=0)
    return p


def build_protocol_template(block: Mapping, *, label_version: str, label_sha256: str,
                            version: str, monitoring_origins: int = 104,
                            first_origin=None, criteria: Mapping | None = None) -> dict:
    """Build a reviewable contract for the fixed four-model audit family.

    The default calendar starts at the supplied block's origin for a concrete
    preview. For a future trial pass a future Friday cutoff; registration still
    requires a separate explicit command. The proposed support gates are a
    starting policy for review, not an empirical power or promotion guarantee.
    """
    _required(block, ("protocol", "source_hashes", "data_as_of", "latest"), "research block")
    _integer(monitoring_origins, "monitoring_origins")
    recipe = _sha(block["protocol"].get("sha256"), "research protocol.sha256")
    origin = _utc(first_origin if first_origin is not None else block["data_as_of"])
    # Existing official weekly convention is a Friday 16:00 New York cutoff,
    # including weeks with a market holiday, not a last-trading-session timestamp.
    from zoneinfo import ZoneInfo
    local_date = origin.astimezone(ZoneInfo("America/New_York")).date()
    if weekly_decision_at(local_date) != origin:
        raise ResearchLedgerError("template origin must match the existing Friday 16:00 New York cutoff")
    calendar = [weekly_decision_at(local_date + timedelta(weeks=i)).isoformat()
                for i in range(monitoring_origins + 13)]
    family = {"directional_duration_hazard": ("candidate", "markov_duration_path_baseline", [1, 4, 13]),
              "markov_duration_path_baseline": ("baseline", None, [1, 4, 13]),
              "boundary_filtered_history": ("baseline", None, [1]),
              "boundary_asymmetric_ewma": ("candidate", "boundary_filtered_history", [1])}
    if {row["model"] for row in block["latest"]} != set(family):
        raise ResearchLedgerError("template builder supports the fixed four-model audit family; supply an explicit protocol for other models")
    code_hashes = {name: _sha(digest, name) for name, digest in block["source_hashes"].items()
                   if name.startswith(("src/", "scripts/", "docs/"))}
    if not code_hashes:
        raise ResearchLedgerError("source code recipe hashes are required for the protocol template")
    root = Path(__file__).resolve().parents[3]
    for name in ("src/regime_lab/research/forecast_prospective.py", "scripts/manage_forecast_research_ledger.py"):
        code_hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    models = []
    for row in block["latest"]:
        name = row["model"]
        role, baseline, horizons = family[name]
        components = {"model": name, "audit_protocol_sha256": recipe, "source_code_hashes": code_hashes,
                      "runtime_versions": {name: dependency_version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn")}}
        model = {"id": name, "version": f"{block['protocol']['version']}:{name}",
                 "recipe_sha256": content_sha256(components), "recipe_components": components,
                 "role": role, "horizons": horizons}
        if baseline:
            model["baseline_id"] = baseline
        models.append(model)
    proposed = {"minimum_paired_origins": 52, "minimum_worsening_events": 12, "minimum_recovery_events": 12,
                "minimum_risk_off_episodes": 6, "minimum_binary_events": 8, "minimum_binary_nonevents": 20,
                "block_length": 13, "bootstrap_draws": 999, "maximum_missing_origin_fraction": 0,
                "maximum_log_loss_delta_ci_high": 0, "maximum_brier_delta": 0,
                "minimum_worsening_recall": .25, "maximum_false_alarm_fraction": .10}
    return validate_protocol({"schema_version": PROTOCOL_SCHEMA, "version": version, "recipe_sha256": recipe,
                              "label": {"version": label_version, "sha256": label_sha256}, "calendar": calendar,
                              "models": models, "criteria": _copy(criteria) if criteria is not None else proposed,
                              "issue_deadline_seconds": 60 * 3600, "max_generation_age_seconds": 3600,
                              "criteria_basis": "Proposed review policy; event support and proper scores are required, not elapsed time alone. No power guarantee.",
                              "calendar_convention": "Existing Friday 16:00 America/New_York weekly cutoff, including holidays; DST-aware."})


def reconstructed_preview_information(block: Mapping) -> dict:
    """Preserve unknown historical availability; never invent first-seen times."""
    if block.get("evidence_track") != "reconstructed_market":
        raise ResearchLedgerError("this helper only describes reconstructed research previews")
    hashes = _object(block.get("source_hashes"), "source_hashes")
    inputs = {name: digest for name, digest in hashes.items() if not name.startswith(("src/", "scripts/", "docs/"))}
    return {"manifest_sha256": _sha(inputs.get("input-manifest.json"), "input manifest hash"),
            "sources": [{"id": name, "sha256": _sha(digest, name), "available_at": None, "retrieved_at": None,
                         "availability_basis": "reconstructed"} for name, digest in sorted(inputs.items())],
            "note": "Saved derived inputs have content hashes. Historical availability/retrieval is not attested by this block; timestamps remain null. Preview only."}


def _timely_origin(protocol, origin, now):
    if origin.isoformat() not in protocol["calendar"][:-13]:
        raise ResearchLedgerError("input origin is outside the frozen issuance calendar")
    if not origin <= now < origin + timedelta(seconds=protocol["issue_deadline_seconds"]):
        raise ResearchLedgerError("expired or future input capture/preparation; historical inputs cannot be reissued")


def _check_recipe_files(protocol):
    """The optional fixed-audit producer attests code and runtime, not a JSON flag."""
    root = Path(__file__).resolve().parents[3]
    label_bytes = (root / "config/label-spec.json").read_bytes()
    label = json.loads(label_bytes)
    if (hashlib.sha256(label_bytes).hexdigest() != protocol["label"]["sha256"]
            or label["specs"][label["default_spec"]]["version"] != protocol["label"]["version"]):
        raise ResearchLedgerError("official label specification differs from the frozen protocol")
    for model in protocol["models"]:
        components = model.get("recipe_components")
        if not components or content_sha256(components) != model["recipe_sha256"]:
            raise ResearchLedgerError("fixed producer requires builder-created recipe components")
        if components["model"] != model["id"] or components["audit_protocol_sha256"] != protocol["recipe_sha256"]:
            raise ResearchLedgerError("producer recipe components do not match the frozen model")
        for name, digest in components["source_code_hashes"].items():
            path = (root / name).resolve()
            if not path.is_relative_to(root) or not name.startswith(("src/", "scripts/", "docs/")):
                raise ResearchLedgerError("recipe source path is outside the repository")
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ResearchLedgerError(f"frozen producer source changed: {name}")
        for name, value in components["runtime_versions"].items():
            if dependency_version(name) != value:
                raise ResearchLedgerError(f"frozen producer runtime changed: {name}")


def _load_captured_frames(directory):
    # These are trusted local derived caches, never downloaded/untrusted pickles.
    import pandas as pd
    from regime_lab.operational_forecast import frame_sha256
    manifest = json.loads((directory / "input-manifest.json").read_text())
    canonical = pd.read_pickle(directory / "canonical.pkl")
    states = pd.read_pickle(directory / "states.pkl")
    if not isinstance(states, pd.Series) or not isinstance(canonical, pd.DataFrame):
        raise ResearchLedgerError("captured states/canonical cache types are invalid")
    origin = _utc(manifest["data_as_of"])
    for name, frame in (("canonical", canonical), ("states", states)):
        if (frame.empty or frame.index.tz is None or frame.index.has_duplicates or not frame.index.is_monotonic_increasing
                or _utc(frame.index[-1]) != origin or frame_sha256(frame) != manifest["frames"][name]):
            raise ResearchLedgerError(f"captured {name} does not match its current input manifest")
    if not canonical.index.equals(states.index) or not states.isin(STATE_ORDER).all():
        raise ResearchLedgerError("captured canonical and official state origins differ")
    official = pd.read_csv(directory / "official-state-history.csv")
    official.index = pd.to_datetime(official["date"], utc=True, format="ISO8601")
    if not states.index.equals(official.index) or not np.array_equal(states.to_numpy(), official["state"].to_numpy()):
        raise ResearchLedgerError("captured states differ from official state history")
    baseline = pd.read_csv(directory / "source-oos.csv")
    return origin, canonical, states, baseline


def capture_research_inputs(*, protocol: Mapping, input_directory: str | Path,
                            source_oos: str | Path, official_state_history: str | Path,
                            output_directory: str | Path) -> dict:
    """Capture future/current derived inputs at the actual local storage time.

    This is first-seen evidence for this new local snapshot. It does not invent
    historical provider release times or convert any historical model results.
    Input bytes, semantic frame hashes, official states and frozen producer code
    are checked before a new immutable capture directory becomes visible.
    """
    p = validate_protocol(protocol)
    _check_recipe_files(p)
    inputs, output = Path(input_directory), Path(output_directory)
    if output.exists():
        raise FileExistsError(f"capture directory already exists: {output}")
    paths = {"canonical.pkl": inputs / "canonical.pkl", "states.pkl": inputs / "states.pkl",
             "input-manifest.json": inputs / "input-manifest.json", "source-oos.csv": Path(source_oos),
             "official-state-history.csv": Path(official_state_history)}
    initial_manifest = json.loads(paths["input-manifest.json"].read_text())
    origin = _utc(initial_manifest["data_as_of"])
    _timely_origin(p, origin, _utc_now())
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=output.parent, prefix=".research-input-capture-"))
    try:
        files = []
        for name, path in paths.items():
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            with (staging / name).open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            stored = _utc_now().isoformat()
            files.append({"name": name, "sha256": digest, "bytes": len(raw), "stored_at": stored})
        captured_origin, _, _, _ = _load_captured_frames(staging)
        if captured_origin != origin:
            raise ResearchLedgerError("input origin changed while capturing")
        now = _utc_now()
        _timely_origin(p, origin, now)
        _check_recipe_files(p)
        for row in files:
            if hashlib.sha256(paths[row["name"]].read_bytes()).hexdigest() != row["sha256"]:
                raise ResearchLedgerError("input changed while capturing; no snapshot installed")
        capture = {"schema_version": CAPTURE_SCHEMA, "protocol_sha256": content_sha256(p), "label": p["label"],
                   "origin_date": origin.isoformat(), "captured_at": now.isoformat(), "files": files,
                   "availability_basis": "first_local_capture", "historical_availability_claim": False,
                   "note": "These bytes were durably observed now. This does not attest historical provider release dates."}
        with (staging / "capture-manifest.json").open("xb") as stream:
            stream.write((_json(capture) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"capture_directory": str(output.resolve()), "capture_manifest_sha256": content_sha256(capture),
            "captured_at": capture["captured_at"], "origin_date": origin.isoformat(), "issued_forecasts": 0}


def _run_fixed_latest(canonical, states, baseline):
    """Run the fixed recipe afresh; retrospective rows never enter the ledger."""
    from regime_lab.analysis.forecast_audit_research import AuditResearchProtocol, run_audit_research
    recipe = AuditResearchProtocol()
    result = run_audit_research(canonical, states, baseline, protocol=recipe)
    return recipe.record(), result.document["latest"]


def prepare_research_forecast(*, protocol: Mapping, capture_directory: str | Path) -> dict:
    """Executable forward producer from captured inputs, not a relabel adapter.

    Model/recipe attestations are generated only after a fresh fixed-model run
    whose current inputs, clock, source code and runtime match the frozen trial.
    This still only prepares JSON; protocol registration and issue are separate.
    """
    p = validate_protocol(protocol)
    _check_recipe_files(p)
    directory = Path(capture_directory)
    capture_bytes = (directory / "capture-manifest.json").read_bytes()
    capture = json.loads(capture_bytes)
    _required(capture, ("schema_version", "protocol_sha256", "label", "origin_date", "captured_at", "files", "availability_basis", "historical_availability_claim"), "capture manifest")
    if (capture["schema_version"] != CAPTURE_SCHEMA or capture["protocol_sha256"] != content_sha256(p)
            or capture["label"] != p["label"] or capture["availability_basis"] != "first_local_capture"
            or capture["historical_availability_claim"] is not False):
        raise ResearchLedgerError("capture does not match the frozen protocol and actual local availability contract")
    origin, captured, now = _utc(capture["origin_date"]), _utc(capture["captured_at"]), _utc_now()
    _timely_origin(p, origin, now)
    if not origin <= captured <= now:
        raise ResearchLedgerError("capture timestamp is backdated or future")
    required = {"canonical.pkl", "states.pkl", "input-manifest.json", "source-oos.csv", "official-state-history.csv"}
    if not isinstance(capture["files"], list) or len(capture["files"]) != len(required) or {f.get("name") for f in capture["files"]} != required:
        raise ResearchLedgerError("capture must contain the exact required input file set")

    def check_files():
        if (directory / "capture-manifest.json").read_bytes() != capture_bytes:
            raise ResearchLedgerError("capture manifest changed during preparation")
        for row in capture["files"]:
            _sha(row["sha256"], "captured file hash")
            raw = (directory / row["name"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != row["sha256"] or len(raw) != row["bytes"]:
                raise ResearchLedgerError("captured input bytes differ from their storage receipt")
            if not origin <= _utc(row["stored_at"]) <= captured:
                raise ResearchLedgerError("captured input storage time is backdated or future")

    check_files()
    observed_origin, canonical, states, baseline = _load_captured_frames(directory)
    if observed_origin != origin:
        raise ResearchLedgerError("capture origin differs from the actual input frames")
    recipe, latest = _run_fixed_latest(canonical, states, baseline)
    if recipe["sha256"] != p["recipe_sha256"]:
        raise ResearchLedgerError("fresh producer recipe differs from frozen protocol")
    by_model = {row["model"]: row for row in latest}
    if len(latest) != len(p["models"]) or set(by_model) != {model["id"] for model in p["models"]}:
        raise ResearchLedgerError("fresh producer returned a different model family")
    for model in p["models"]:
        by_model[model["id"]].update({"version": model["version"], "recipe_sha256": model["recipe_sha256"]})
    check_files()
    _check_recipe_files(p)
    generated = _utc_now()
    _timely_origin(p, origin, generated)
    block = {"schema_version": "regime-forecast-prospective-latest/1", "evidence_track": "prospective_inputs",
             "data_as_of": origin.isoformat(), "generated_at": generated.isoformat(), "latest": latest,
             "protocol": recipe, "capture_manifest_sha256": content_sha256(capture),
             "historical_diagnostics_included": False, "automatic_promotion": False}
    information = {"manifest_sha256": content_sha256(capture),
                   "sources": [{"id": row["name"], "sha256": row["sha256"], "available_at": row["stored_at"],
                                "retrieved_at": row["stored_at"], "availability_basis": "first_seen"} for row in capture["files"]],
                   "availability_scope": "first_local_capture_of_exact_derived_inputs; no historical provider-release assertion"}
    frozen = freeze_research_forecast(block, protocol=p, information_set=information)
    return {"producer_block": block, "information_set": information, "frozen": frozen,
            "status": "prepared_only", "issued_forecasts": 0, "automatic_promotion": False}


def _one_week_path(probability, current):
    return {"horizon_weeks": 1, "endpoint": dict(probability),
            "first_departure": {"no_departure": probability[current],
                                **{s: 0 if s == current else probability[s] for s in STATE_ORDER}},
            "any_risk_off_entry": 0 if current == "risk_off" else probability["risk_off"],
            "any_risk_off_occupancy": probability["risk_off"]}


def _paths(model, current, required_horizons):
    rows = model.get("paths", [])
    if not isinstance(rows, list):
        raise ResearchLedgerError("model.paths must be a list")
    paths = {}
    for raw in rows:
        _required(raw, ("horizon_weeks", "any_risk_off_entry", "any_risk_off_occupancy"), "path")
        horizon = _integer(raw["horizon_weeks"], "horizon_weeks")
        if horizon not in required_horizons or horizon in paths:
            raise ResearchLedgerError("unexpected or duplicate horizon")
        if "endpoint" in raw and "endpoint_probabilities" in raw and raw["endpoint"] != raw["endpoint_probabilities"]:
            raise ResearchLedgerError("conflicting endpoint aliases")
        endpoint = _distribution(raw.get("endpoint", raw.get("endpoint_probabilities")), STATE_ORDER, "endpoint")
        departure = raw.get("first_departure")
        if "first_destination" in raw:
            _required(raw, ("no_departure",), "path")
            alias = {"no_departure": raw["no_departure"], **_object(raw["first_destination"], "first_destination")}
            if departure is not None and departure != alias:
                raise ResearchLedgerError("conflicting first-departure aliases")
            departure = alias
        departure = _distribution(departure, FIRST_DESTINATIONS, "first_departure")
        entry = _number(raw["any_risk_off_entry"], "any_risk_off_entry", 0, 1)
        occupancy = _number(raw["any_risk_off_occupancy"], "any_risk_off_occupancy", 0, 1)
        if departure[current] != 0 or endpoint[current] + TOLERANCE < departure["no_departure"]:
            raise ResearchLedgerError("first departure is inconsistent with the origin state")
        if entry > occupancy + TOLERANCE or endpoint["risk_off"] > occupancy + TOLERANCE:
            raise ResearchLedgerError("risk-off entry/occupancy/endpoint are inconsistent")
        if current != "risk_off" and (abs(entry - occupancy) > TOLERANCE or departure["risk_off"] > entry + TOLERANCE):
            raise ResearchLedgerError("non-risk-off origin requires entry equal to occupancy")
        if current == "risk_off" and entry > 1 - departure["no_departure"] + TOLERANCE:
            raise ResearchLedgerError("risk-off re-entry requires a departure")
        paths[horizon] = {"horizon_weeks": horizon, "endpoint": endpoint, "first_departure": departure,
                          "any_risk_off_entry": entry, "any_risk_off_occupancy": occupancy}
        if "target_date" in raw:
            paths[horizon]["target_date"] = _utc(raw["target_date"], "target_date").isoformat()
    if "next_state" in model:
        next_state = _distribution(model["next_state"], STATE_ORDER, "next_state")
        derived = _one_week_path(next_state, current)
        if 1 in paths and any(paths[1][target] != derived[target] for target in TARGETS):
            # Arithmetic from a path recursion may differ by harmless ulps only.
            def flat(path):
                return [path["endpoint"][s] for s in STATE_ORDER] + [path["first_departure"][s] for s in FIRST_DESTINATIONS] + [path[t] for t in TARGETS[2:]]
            if not np.allclose(flat(paths[1]), flat(derived), atol=TOLERANCE, rtol=0):
                raise ResearchLedgerError("one-week path and next_state disagree")
        paths.setdefault(1, derived)
    if set(paths) != set(required_horizons):
        raise ResearchLedgerError("every frozen model horizon must be present")
    one = paths[1]
    derived = _one_week_path(one["endpoint"], current)
    if any(abs(one["first_departure"][s] - derived["first_departure"][s]) > TOLERANCE for s in FIRST_DESTINATIONS):
        raise ResearchLedgerError("one-week first-departure probabilities are inconsistent")
    if any(abs(one[t] - derived[t]) > TOLERANCE for t in TARGETS[2:]):
        raise ResearchLedgerError("one-week risk-off probabilities are inconsistent")
    ordered = [paths[h] for h in required_horizons]
    for before, after in zip(ordered, ordered[1:]):
        if after["first_departure"]["no_departure"] > before["first_departure"]["no_departure"] + TOLERANCE:
            raise ResearchLedgerError("no-departure probability must not increase with horizon")
        if any(after["first_departure"][s] + TOLERANCE < before["first_departure"][s] for s in STATE_ORDER):
            raise ResearchLedgerError("first-destination probability must not decrease with horizon")
        if any(after[t] + TOLERANCE < before[t] for t in TARGETS[2:]):
            raise ResearchLedgerError("entry/occupancy probability must not decrease with horizon")
    return ordered


def freeze_research_forecast(block: Mapping, *, protocol: Mapping, information_set: Mapping) -> dict:
    """Freeze an exact latest-only forecast for preview without writing anything.

    Retrospective/replay input is allowed in preview and is explicitly ineligible
    for issue. A fresh prospective producer must declare prospective_inputs and
    provide the frozen model recipes and actual input-availability manifest.
    """
    p = validate_protocol(protocol)
    block, information = _copy(block), _copy(information_set)
    _required(block, ("latest", "data_as_of", "generated_at", "evidence_track"), "research block")
    _required(information, ("manifest_sha256", "sources"), "information_set")
    _sha(information["manifest_sha256"], "information_set.manifest_sha256")
    if not isinstance(information["sources"], list) or not information["sources"]:
        raise ResearchLedgerError("input source availability records are required")
    generated = _utc(block["generated_at"], "generated_at")
    now = _utc_now()
    if generated > now:
        raise ResearchLedgerError("generated_at cannot be in the future")
    for source in information["sources"]:
        _required(source, ("id", "sha256", "available_at", "retrieved_at", "availability_basis"), "source")
        _text(source["id"], "source.id")
        _sha(source["sha256"], "source.sha256")
        if source["availability_basis"] == "reconstructed" and (source["available_at"] is None or source["retrieved_at"] is None):
            # A concrete historical preview must preserve unknown timestamps.
            if source["available_at"] is not None:
                _utc(source["available_at"])
            if source["retrieved_at"] is not None:
                _utc(source["retrieved_at"])
            continue
        available, retrieved = _utc(source["available_at"]), _utc(source["retrieved_at"])
        if available > generated or retrieved > generated or available > retrieved:
            raise ResearchLedgerError("source availability/retrieval must precede generation")
        if source["availability_basis"] not in ("first_seen", "archived_release", "reconstructed"):
            raise ResearchLedgerError("unknown source availability basis")
        if source["availability_basis"] == "first_seen" and available != retrieved:
            raise ResearchLedgerError("first_seen availability must equal actual retrieval")
    origin = _utc(block["data_as_of"], "data_as_of")
    if origin > generated:
        raise ResearchLedgerError("origin cannot follow generation")
    calendar = [_utc(d) for d in p["calendar"]]
    if origin not in calendar[:-13]:
        raise ResearchLedgerError("origin is outside the frozen issuance calendar")
    position = calendar.index(origin)
    if not isinstance(block["latest"], list):
        raise ResearchLedgerError("latest must contain all frozen models and baselines")
    supplied = {}
    for model in block["latest"]:
        _required(model, ("model", "origin_date", "current_state"), "latest model")
        if model["model"] in supplied:
            raise ResearchLedgerError("duplicate latest model")
        if _utc(model["origin_date"]) != origin:
            raise ResearchLedgerError("latest models must have the same exact origin")
        if model["current_state"] not in STATE_ORDER:
            raise ResearchLedgerError("unknown current_state")
        supplied[model["model"]] = model
    if set(supplied) != {m["id"] for m in p["models"]}:
        raise ResearchLedgerError("latest model set must exactly match the frozen candidate/baseline set")
    if len({m["current_state"] for m in supplied.values()}) != 1:
        raise ResearchLedgerError("models disagree about the official origin state")
    current = next(iter(supplied.values()))["current_state"]
    frozen_models = []
    for model in p["models"]:
        source = supplied[model["id"]]
        for field in ("version", "recipe_sha256"):
            if field in source and source[field] != model[field]:
                raise ResearchLedgerError(f"model {field} differs from the protocol")
        paths = _paths(source, current, model["horizons"])
        for path in paths:
            target = calendar[position + path["horizon_weeks"]].isoformat()
            if "target_date" in path and path["target_date"] != target:
                raise ResearchLedgerError("model target does not match frozen calendar")
            path["target_date"] = target
        training = _copy(source.get("training", {}))
        if "last_train_target" in training and training["last_train_target"] is not None:
            if _utc(training["last_train_target"]) >= origin:
                raise ResearchLedgerError("training target must be strictly before the forecast origin")
        frozen_models.append({**model, "paths": paths, "training": training})
    disqualifiers = []
    if block["evidence_track"] != "prospective_inputs":
        disqualifiers.append("non_prospective_evidence_track")
    if block.get("replay", False) or block.get("mode") in ("replay", "backtest", "historical"):
        disqualifiers.append("replay_forecast")
    if any(s["availability_basis"] == "reconstructed" for s in information["sources"]):
        disqualifiers.append("reconstructed_input_availability")
    # Missing producer attestation is fine for viewing legacy research, never issue.
    if any(any(source.get(field) != model[field] for field in ("version", "recipe_sha256"))
           for model in p["models"] for source in [supplied[model["id"]]]):
        disqualifiers.append("missing_producer_recipe_attestation")
    deadline = origin + timedelta(seconds=p["issue_deadline_seconds"])
    if now >= deadline:
        disqualifiers.append("issuance_deadline_passed")
    if (now - generated).total_seconds() > p["max_generation_age_seconds"]:
        disqualifiers.append("generation_too_old")
    document = {"schema_version": FORECAST_SCHEMA, "protocol_sha256": content_sha256(p),
                "protocol_version": p["version"], "label": p["label"], "origin_date": origin.isoformat(),
                "current_state": current, "generated_at": generated.isoformat(), "prepared_at": now.isoformat(),
                "publication_deadline": deadline.isoformat(), "evidence_track": block["evidence_track"],
                "source_block_sha256": content_sha256(block), "information_set": information,
                "models": frozen_models, "issue_disqualifiers": disqualifiers,
                "automatic_promotion": False}
    return {"forecast": document, "forecast_sha256": content_sha256(document),
            "preview_only": True, "eligible_for_issue_at_preview": not disqualifiers}


_SCHEMA = """
CREATE TABLE research_protocols (
    protocol_sha256 TEXT PRIMARY KEY, version TEXT NOT NULL UNIQUE,
    stored_at TEXT NOT NULL, content TEXT NOT NULL
);
CREATE TABLE research_forecasts (
    forecast_sha256 TEXT PRIMARY KEY, protocol_sha256 TEXT NOT NULL REFERENCES research_protocols,
    origin_date TEXT NOT NULL, stored_at TEXT NOT NULL, content TEXT NOT NULL,
    UNIQUE(protocol_sha256, origin_date)
);
CREATE TABLE research_publications (
    forecast_sha256 TEXT PRIMARY KEY REFERENCES research_forecasts,
    published_at TEXT NOT NULL, receipt_sha256 TEXT NOT NULL, content TEXT NOT NULL
);
CREATE TABLE research_evaluations (
    forecast_sha256 TEXT NOT NULL REFERENCES research_forecasts, horizon_weeks INTEGER NOT NULL,
    evaluated_at TEXT NOT NULL, evaluation_sha256 TEXT NOT NULL, content TEXT NOT NULL,
    PRIMARY KEY(forecast_sha256, horizon_weeks)
);
"""
_TABLES = ("research_protocols", "research_forecasts", "research_publications", "research_evaluations")


class ForecastResearchLedger:
    """An explicit, separate SQLite research ledger; no live-database discovery."""

    def __init__(self, path: str | Path, *, create: bool = False):
        if not str(path) or str(path) == ":memory:":
            raise ResearchLedgerError("an explicit filesystem research ledger path is required")
        self.path = Path(path).expanduser().resolve()
        exists = self.path.exists()
        if not exists and not create:
            raise ResearchLedgerError("research ledger does not exist; use explicit init")
        if not exists:
            # Exclusive creation avoids silently taking ownership of a racing DB.
            with self.path.open("xb"):
                pass
        with self._connect() as conn:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
            if not tables and create:
                conn.executescript(_SCHEMA)
                for table in _TABLES:
                    for operation in ("UPDATE", "DELETE"):
                        conn.execute(f"CREATE TRIGGER {table}_{operation.lower()} BEFORE {operation} ON {table} "
                                     "BEGIN SELECT RAISE(ABORT, 'append-only research ledger'); END")
                conn.execute(f"PRAGMA application_id = {APPLICATION_ID}")
                conn.execute("PRAGMA user_version = 1")
            elif (tables != set(_TABLES) or conn.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID
                  or conn.execute("PRAGMA user_version").fetchone()[0] != 1):
                raise ResearchLedgerError("path is not a dedicated version-1 research ledger")
        self.verify_integrity()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(f"{self.path.as_uri()}?mode=rw", uri=True, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA synchronous = FULL")
            with conn:
                yield conn
        finally:
            conn.close()

    @staticmethod
    def _read(row, hash_column):
        result = json.loads(row["content"])
        if content_sha256(result) != row[hash_column]:
            raise ResearchLedgerError("stored content hash mismatch")
        return result

    def verify_integrity(self) -> None:
        """Check content hashes, immutable links and denormalized timestamps."""
        with self._connect() as conn:
            if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok" or conn.execute("PRAGMA foreign_key_check").fetchall():
                raise ResearchLedgerError("research database integrity check failed")
            triggers = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
            if any(f"{table}_{op}" not in triggers for table in _TABLES for op in ("update", "delete")):
                raise ResearchLedgerError("append-only triggers are missing")
            protocols = {}
            for row in conn.execute("SELECT * FROM research_protocols"):
                p = self._read(row, "protocol_sha256")
                if p != validate_protocol(p) or p["version"] != row["version"]:
                    raise ResearchLedgerError("stored protocol contract mismatch")
                if _utc(row["stored_at"]) >= _utc(p["calendar"][0]):
                    raise ResearchLedgerError("protocol was not registered before its first origin")
                protocols[row["protocol_sha256"]] = (p, row["stored_at"])
            forecasts = {}
            for row in conn.execute("SELECT * FROM research_forecasts"):
                f = self._read(row, "forecast_sha256")
                if f["protocol_sha256"] != row["protocol_sha256"] or f["origin_date"] != row["origin_date"]:
                    raise ResearchLedgerError("forecast protocol/origin linkage mismatch")
                p, registered = protocols[row["protocol_sha256"]]
                self._validate_issue(f, p, registered, _utc(row["stored_at"]))
                forecasts[row["forecast_sha256"]] = (f, row["stored_at"])
            published = set()
            for row in conn.execute("SELECT * FROM research_publications"):
                receipt = self._read(row, "receipt_sha256")
                f, stored = forecasts[row["forecast_sha256"]]
                if (receipt["forecast_sha256"] != row["forecast_sha256"] or receipt["published_at"] != row["published_at"]
                        or receipt["stored_at"] != stored or receipt["publication_kind"] != "local_artifact"):
                    raise ResearchLedgerError("publication receipt linkage mismatch")
                if not _utc(stored) <= _utc(row["published_at"]) < _utc(f["publication_deadline"]):
                    raise ResearchLedgerError("publication was late or backdated")
                published.add(row["forecast_sha256"])
            for row in conn.execute("SELECT * FROM research_evaluations"):
                result = self._read(row, "evaluation_sha256")
                f, _ = forecasts[row["forecast_sha256"]]
                if (row["forecast_sha256"] not in published or result["forecast_sha256"] != row["forecast_sha256"]
                        or result["horizon_weeks"] != row["horizon_weeks"] or result["evaluated_at"] != row["evaluated_at"]
                        or result["label"] != f["label"] or result["origin_date"] != f["origin_date"]):
                    raise ResearchLedgerError("evaluation immutable linkage mismatch")
                if _utc(result["target_date"]) > _utc(result["evaluated_at"]):
                    raise ResearchLedgerError("evaluation predates its target")

    def register_protocol(self, protocol: Mapping) -> dict:
        p = validate_protocol(protocol)
        digest = content_sha256(p)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute("SELECT * FROM research_protocols WHERE version=?", (p["version"],)).fetchone()
            if existing:
                if existing["protocol_sha256"] != digest:
                    raise ResearchConflictError("protocol version is already frozen with different content")
                return {"protocol_sha256": digest, "stored_at": existing["stored_at"], "created": False}
            now = _utc_now()
            if now >= _utc(p["calendar"][0]):
                raise ResearchLedgerError("register protocol before its first scheduled origin; no backfilled protocol")
            conn.execute("INSERT INTO research_protocols VALUES (?,?,?,?)", (digest, p["version"], now.isoformat(), _json(p)))
        return {"protocol_sha256": digest, "stored_at": now.isoformat(), "created": True}

    @staticmethod
    def _validate_issue(f, protocol, registered, now):
        _required(f, ("schema_version", "protocol_sha256", "protocol_version", "label", "origin_date", "current_state",
                      "generated_at", "prepared_at", "publication_deadline", "evidence_track", "models",
                      "source_block_sha256", "information_set", "issue_disqualifiers", "automatic_promotion"), "frozen forecast")
        if f["schema_version"] != FORECAST_SCHEMA or f["automatic_promotion"] is not False:
            raise ResearchLedgerError("invalid frozen forecast schema/promotion policy")
        if f["protocol_sha256"] != content_sha256(protocol) or f["protocol_version"] != protocol["version"] or f["label"] != protocol["label"]:
            raise ResearchLedgerError("forecast does not match the frozen protocol/label")
        if f["evidence_track"] != "prospective_inputs" or f["issue_disqualifiers"]:
            raise ResearchLedgerError("preview/reconstructed/replay forecasts cannot be issued")
        origin, generated, prepared = (_utc(f[k]) for k in ("origin_date", "generated_at", "prepared_at"))
        deadline = origin + timedelta(seconds=protocol["issue_deadline_seconds"])
        if deadline != _utc(f["publication_deadline"]):
            raise ResearchLedgerError("publication deadline differs from the frozen protocol")
        if not _utc(registered) < origin <= generated <= prepared <= now < deadline:
            raise ResearchLedgerError("expired, backdated, or future forecast issuance")
        if (now - generated).total_seconds() > protocol["max_generation_age_seconds"]:
            raise ResearchLedgerError("generation is too old for real issuance")
        calendar = [_utc(d) for d in protocol["calendar"]]
        if origin not in calendar[:-13] or f["current_state"] not in STATE_ORDER:
            raise ResearchLedgerError("origin/current state is outside the frozen contract")
        _sha(f["source_block_sha256"], "source_block_sha256")
        info = f["information_set"]
        _required(info, ("manifest_sha256", "sources"), "information_set")
        _sha(info["manifest_sha256"], "manifest_sha256")
        if not isinstance(info["sources"], list) or not info["sources"]:
            raise ResearchLedgerError("source availability records are required")
        for source in info["sources"]:
            _required(source, ("id", "sha256", "available_at", "retrieved_at", "availability_basis"), "source")
            _sha(source["sha256"], "source.sha256")
            available, retrieved = _utc(source["available_at"]), _utc(source["retrieved_at"])
            if source["availability_basis"] not in ("first_seen", "archived_release") or not available <= retrieved <= generated:
                raise ResearchLedgerError("reconstructed or future input availability cannot support issuance")
            if source["availability_basis"] == "first_seen" and available != retrieved:
                raise ResearchLedgerError("first_seen availability cannot be backdated")
        if not isinstance(f["models"], list) or [m.get("id") for m in f["models"]] != [m["id"] for m in protocol["models"]]:
            raise ResearchLedgerError("frozen model set/order mismatch")
        position = calendar.index(origin)
        for model, frozen in zip(f["models"], protocol["models"]):
            if {k: model.get(k) for k in frozen} != frozen:
                raise ResearchLedgerError("model recipe/version differs from protocol")
            checked = _paths(model, f["current_state"], frozen["horizons"])
            if checked != model["paths"]:
                raise ResearchLedgerError("frozen paths are not canonical")
            if any(path["target_date"] != calendar[position + path["horizon_weeks"]].isoformat() for path in checked):
                raise ResearchLedgerError("target schedule differs from protocol")
            training = _object(model.get("training", {}), "training")
            if training.get("last_train_target") is not None and _utc(training["last_train_target"]) >= origin:
                raise ResearchLedgerError("training target must be strictly before the forecast origin")

    def issue(self, frozen: Mapping, *, publication_path: str | Path) -> dict:
        """Explicitly store and publish a fresh forecast to a new local file.

        Storage commits before publication. A failed write leaves an immutable
        unpublished record, which contributes no evidence. Retrying the identical
        frozen object can publish while still timely. There is no timestamp flag,
        replacement of an existing file, remote deploy, or automatic promotion.
        """
        self.verify_integrity()
        _required(frozen, ("forecast", "forecast_sha256"), "frozen envelope")
        f = _copy(frozen["forecast"])
        digest = content_sha256(f)
        if digest != frozen["forecast_sha256"]:
            raise ResearchLedgerError("frozen forecast hash mismatch")
        output = Path(publication_path).expanduser().resolve()
        if output == self.path:
            raise ResearchLedgerError("publication cannot overwrite the ledger")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM research_protocols WHERE protocol_sha256=?", (f.get("protocol_sha256"),)).fetchone()
            if row is None:
                raise ResearchLedgerError("register the frozen protocol before issuance")
            p = self._read(row, "protocol_sha256")
            existing = conn.execute("SELECT * FROM research_forecasts WHERE protocol_sha256=? AND origin_date=?",
                                    (f.get("protocol_sha256"), f.get("origin_date"))).fetchone()
            if existing and existing["forecast_sha256"] != digest:
                raise ResearchConflictError("this origin already has a different immutable forecast")
            receipt = conn.execute("SELECT * FROM research_publications WHERE forecast_sha256=?", (digest,)).fetchone()
            if receipt:
                return {**self._read(receipt, "receipt_sha256"), "created": False}
            now = _utc_now()
            self._validate_issue(f, p, row["stored_at"], now)
            if existing:
                stored = existing["stored_at"]
            else:
                stored = now.isoformat()
                conn.execute("INSERT INTO research_forecasts VALUES (?,?,?,?,?)",
                             (digest, f["protocol_sha256"], f["origin_date"], stored, _json(f)))
        publication = {"schema_version": FORECAST_SCHEMA, "forecast_sha256": digest, "stored_at": stored,
                       "forecast": f, "publication_kind": "local_artifact", "automatic_promotion": False}
        raw = (_json(publication) + "\n").encode("utf-8")
        # Serialize publication attempts; publish via exclusive hard-link after fsync.
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute("SELECT * FROM research_publications WHERE forecast_sha256=?", (digest,)).fetchone()
            if existing:
                return {**self._read(existing, "receipt_sha256"), "created": False}
            self._validate_issue(f, p, row["stored_at"], _utc_now())
            temporary = None
            linked = False
            try:
                with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".research-issue-", delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(temporary, output)  # fails if the destination already exists
                linked = True
                published = _utc_now()
                self._validate_issue(f, p, row["stored_at"], published)
                receipt = {"forecast_sha256": digest, "protocol_sha256": f["protocol_sha256"], "stored_at": stored,
                           "published_at": published.isoformat(), "publication_kind": "local_artifact",
                           "publication_path": str(output), "publication_sha256": hashlib.sha256(raw).hexdigest(),
                           "automatic_promotion": False}
                conn.execute("INSERT INTO research_publications VALUES (?,?,?,?)",
                             (digest, published.isoformat(), content_sha256(receipt), _json(receipt)))
                conn.commit()
            except BaseException:
                if linked:
                    output.unlink(missing_ok=True)
                raise
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return {**receipt, "created": True}

    def evaluate_matured(self, official_snapshot: Mapping) -> dict:
        """Append completed official 1/4/13 paths, with no prices or dividends.

        Missing observations remain pending. A changed already-evaluated path
        raises a conflict and rolls back this evaluation batch; immutable earlier
        evidence is never silently restated using a revised official label.
        """
        self.verify_integrity()
        snapshot = _copy(official_snapshot)
        _required(snapshot, ("schema_version", "label", "available_at", "observations"), "official snapshot")
        if snapshot["schema_version"] != STATES_SCHEMA:
            raise ResearchLedgerError("unsupported official states schema")
        now = _utc_now()
        available = _utc(snapshot["available_at"])
        if available > now:
            raise ResearchLedgerError("official states snapshot is not yet available")
        if not isinstance(snapshot["observations"], list):
            raise ResearchLedgerError("official observations must be a list")
        observations = {}
        for observation in snapshot["observations"]:
            _required(observation, ("date", "state"), "official observation")
            date = _utc(observation["date"])
            if date in observations or date > available or observation["state"] not in STATE_ORDER:
                raise ResearchLedgerError("duplicate, future, or invalid official observation")
            observations[date] = observation["state"]
        if list(observations) != sorted(observations):
            raise ResearchLedgerError("official observations must be chronological")
        result = {"created": 0, "unchanged": 0, "pending": [], "automatic_promotion": False}
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            forecasts = conn.execute("SELECT f.* FROM research_forecasts f JOIN research_publications p USING(forecast_sha256)").fetchall()
            for row in forecasts:
                forecast = self._read(row, "forecast_sha256")
                if forecast["label"] != snapshot["label"]:
                    raise ResearchLedgerError("official state label version/hash does not match an issued forecast")
                protocol_row = conn.execute("SELECT * FROM research_protocols WHERE protocol_sha256=?", (forecast["protocol_sha256"],)).fetchone()
                protocol = self._read(protocol_row, "protocol_sha256")
                calendar = [_utc(d) for d in protocol["calendar"]]
                origin = _utc(forecast["origin_date"])
                position = calendar.index(origin)
                for horizon in sorted({h for model in forecast["models"] for h in model["horizons"]}):
                    dates = calendar[position:position + horizon + 1]
                    missing = [d.isoformat() for d in dates if d not in observations]
                    if dates[-1] > now or missing:
                        result["pending"].append({"origin_date": origin.isoformat(), "horizon_weeks": horizon,
                                                  "reason": "not_matured" if dates[-1] > now else "official_observations_missing",
                                                  "missing_dates": missing})
                        continue
                    actual_path = [{"date": d.isoformat(), "state": observations[d]} for d in dates]
                    if actual_path[0]["state"] != forecast["current_state"]:
                        raise ResearchConflictError("issued origin state differs from the official state")
                    outcome = _outcome(actual_path)
                    existing = conn.execute("SELECT * FROM research_evaluations WHERE forecast_sha256=? AND horizon_weeks=?",
                                            (row["forecast_sha256"], horizon)).fetchone()
                    if existing:
                        previous = self._read(existing, "evaluation_sha256")
                        if previous["official_path"] != actual_path:
                            raise ResearchConflictError("an already-evaluated official path changed")
                        result["unchanged"] += 1
                        continue
                    scores = []
                    for model in forecast["models"]:
                        if horizon not in model["horizons"]:
                            continue
                        path = next(path for path in model["paths"] if path["horizon_weeks"] == horizon)
                        scores.append({"model": model["id"], "targets": _scores(path, outcome),
                                       **_direction_metrics(path, outcome, forecast["current_state"])})
                    evaluation = {"forecast_sha256": row["forecast_sha256"], "protocol_sha256": forecast["protocol_sha256"],
                                  "label": snapshot["label"], "origin_date": origin.isoformat(), "target_date": dates[-1].isoformat(),
                                  "horizon_weeks": horizon, "evaluated_at": now.isoformat(),
                                  "state_snapshot_sha256": content_sha256(snapshot), "state_snapshot_available_at": available.isoformat(),
                                  "official_path": actual_path, "outcome": outcome, "scores": scores,
                                  "score_contract": {"log_loss_clip_epsilon": EPSILON, "multiclass_brier": "sum_squared_errors"}}
                    conn.execute("INSERT INTO research_evaluations VALUES (?,?,?,?,?)",
                                 (row["forecast_sha256"], horizon, now.isoformat(), content_sha256(evaluation), _json(evaluation)))
                    result["created"] += 1
        return result

    def readiness(self, protocol_sha256: str) -> dict:
        """Continuous frozen-cohort evidence; readiness means manual review only."""
        self.verify_integrity()
        _sha(protocol_sha256, "protocol_sha256")
        now = _utc_now()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM research_protocols WHERE protocol_sha256=?", (protocol_sha256,)).fetchone()
            if row is None:
                raise ResearchLedgerError("unknown protocol")
            protocol = self._read(row, "protocol_sha256")
            forecasts = conn.execute("SELECT f.*, p.published_at FROM research_forecasts f LEFT JOIN research_publications p USING(forecast_sha256) WHERE protocol_sha256=? ORDER BY origin_date",
                                     (protocol_sha256,)).fetchall()
            evaluation_rows = conn.execute("SELECT e.* FROM research_evaluations e JOIN research_forecasts f USING(forecast_sha256) WHERE f.protocol_sha256=? ORDER BY f.origin_date,e.horizon_weeks",
                                           (protocol_sha256,)).fetchall()
        evaluations = [self._read(row, "evaluation_sha256") for row in evaluation_rows]
        calendar = [_utc(d) for d in protocol["calendar"]]
        expected = [d.isoformat() for d in calendar[:-13] if d + timedelta(seconds=protocol["issue_deadline_seconds"]) <= now]
        published = {f["origin_date"] for f in forecasts if f["published_at"] is not None}
        missing = sorted(set(expected) - published)
        missing_fraction = len(missing) / len(expected) if expected else None
        criteria = protocol["criteria"]
        candidates = []
        for model in protocol["models"]:
            if model["role"] != "candidate":
                continue
            checks = {"issuance_coverage": missing_fraction is not None and missing_fraction <= criteria["maximum_missing_origin_fraction"]}
            one_week = [e for e in evaluations if e["horizon_weeks"] == 1]
            event_counts = {"worsening": sum(e["outcome"]["worsening"] for e in one_week),
                            "recovery": sum(e["outcome"]["recovery"] for e in one_week),
                            "risk_off_episodes": len({e["target_date"] for e in one_week if e["outcome"]["any_risk_off_entry"]})}
            for event, key in (("worsening", "minimum_worsening_events"), ("recovery", "minimum_recovery_events"), ("risk_off_episodes", "minimum_risk_off_episodes")):
                checks[event + "_count"] = event_counts[event] >= criteria[key]
            direction_rows = [next(s for s in e["scores"] if s["model"] == model["id"]) for e in one_week]
            worsening_recall = (sum(s["worsening_hit"] for s in direction_rows) / event_counts["worsening"]) if event_counts["worsening"] else None
            stays = sum(not e["outcome"]["worsening"] and not e["outcome"]["recovery"] for e in one_week)
            false_fraction = sum(s["false_alarm"] for s in direction_rows) / stays if stays else None
            checks["worsening_recall"] = worsening_recall is not None and worsening_recall >= criteria["minimum_worsening_recall"]
            checks["false_alarm_fraction"] = false_fraction is not None and false_fraction <= criteria["maximum_false_alarm_fraction"]
            comparisons = []
            pending_matured = []
            for horizon in model["horizons"]:
                group = [e for e in evaluations if e["horizon_weeks"] == horizon]
                observed_origins = {e["origin_date"] for e in group}
                expected_matured = {date.isoformat() for i, date in enumerate(calendar[:-13])
                                    if date.isoformat() in published and calendar[i + horizon] <= now}
                absent = sorted(expected_matured - observed_origins)
                pending_matured.extend({"origin_date": d, "horizon_weeks": horizon} for d in absent)
                checks[f"{horizon}w_evaluation_coverage"] = not absent
                checks[f"{horizon}w_paired_origin_count"] = len(group) >= criteria["minimum_paired_origins"]
                # Skipped issue origins must not be collapsed into adjacent bootstrap weeks.
                contiguous = len(group) > 0 and all(calendar.index(_utc(b["origin_date"])) - calendar.index(_utc(a["origin_date"])) == 1
                                                   for a, b in zip(group, group[1:]))
                checks[f"{horizon}w_continuous_evidence"] = contiguous
                for target in TARGETS:
                    comparison = _paired_summary(group, model["id"], model["baseline_id"], target, criteria, contiguous)
                    comparison.update({"horizon_weeks": horizon, "target": target})
                    comparisons.append(comparison)
                    checks[f"{horizon}w_{target}_log_loss"] = comparison["log_loss_delta_ci_high"] is not None and comparison["log_loss_delta_ci_high"] <= criteria["maximum_log_loss_delta_ci_high"]
                    checks[f"{horizon}w_{target}_brier"] = comparison["brier_delta"] is not None and comparison["brier_delta"] <= criteria["maximum_brier_delta"]
                    if target in TARGETS[2:]:
                        # For risk-off origins, one-week new entry is structurally
                        # impossible; the aggregate still needs real events/non-events.
                        checks[f"{horizon}w_{target}_event_support"] = comparison["events"] >= criteria["minimum_binary_events"] and comparison["nonevents"] >= criteria["minimum_binary_nonevents"]
            candidates.append({"model": model["id"], "baseline_id": model["baseline_id"], "event_counts": event_counts,
                               "worsening_recall": worsening_recall, "false_alarm_fraction_among_stays": false_fraction,
                               "comparisons": comparisons, "pending_matured_evaluations": pending_matured,
                               "checks": checks, "unmet_criteria": [key for key, passed in checks.items() if not passed],
                               "status": "manual_review_ready" if all(checks.values()) else "collecting_evidence",
                               "automatic_promotion": False})
        return {"schema_version": SUMMARY_SCHEMA, "protocol_sha256": protocol_sha256, "protocol_version": protocol["version"],
                "as_of": now.isoformat(), "registered_at": row["stored_at"], "criteria": criteria,
                "expected_issuance_origins": len(expected), "published_origins": len(published), "missing_origins": missing,
                "missing_origin_fraction": missing_fraction, "unpublished_origins": [f["origin_date"] for f in forecasts if f["published_at"] is None],
                "completed_evaluations": len(evaluations), "candidates": candidates, "automatic_promotion": False,
                "issuance_calendar_exhausted": now >= calendar[-14] + timedelta(seconds=protocol["issue_deadline_seconds"]),
                "final_maturity_at": calendar[-1].isoformat(),
                "uncertainty": "Paired circular block bootstrap, fixed 95% interval; descriptive during continuous monitoring, not sequential or familywise inference.",
                "publication_scope": "Timestamped local research artifacts; no claim of remote delivery or deployment."}


def _outcome(path):
    current, future = path[0]["state"], [row["state"] for row in path[1:]]
    previous = [current, *future[:-1]]
    delta = STATE_ORDER.index(future[-1]) - STATE_ORDER.index(current)
    return {"endpoint": future[-1], "first_departure": next((s for s in future if s != current), "no_departure"),
            "any_risk_off_entry": any(a != "risk_off" and b == "risk_off" for a, b in zip(previous, future)),
            "any_risk_off_occupancy": "risk_off" in future, "worsening": delta > 0, "recovery": delta < 0}


def _scores(path, outcome):
    scores = {}
    for target in TARGETS:
        probability, actual = path[target], outcome[target]
        if target in TARGETS[:2]:
            log_loss = -math.log(max(EPSILON, min(1, probability[actual])))
            brier = sum((p - int(state == actual)) ** 2 for state, p in probability.items())
        else:
            p = min(1 - EPSILON, max(EPSILON, probability))
            log_loss = -math.log(p if actual else 1 - p)
            brier = (probability - int(actual)) ** 2
        scores[target] = {"log_loss": log_loss, "brier": brier}
    return scores


def _direction_metrics(path, outcome, current):
    prediction = max(STATE_ORDER, key=lambda state: path["endpoint"][state])
    change = STATE_ORDER.index(prediction) - STATE_ORDER.index(current)
    return {"argmax_state": prediction, "worsening_hit": outcome["worsening"] and change > 0,
            "recovery_hit": outcome["recovery"] and change < 0,
            "false_alarm": change != 0 and not outcome["worsening"] and not outcome["recovery"]}


def _paired_summary(evaluations, model_id, baseline_id, target, criteria, contiguous):
    candidate, baseline = [], []
    for evaluation in evaluations:
        by_model = {s["model"]: s["targets"][target] for s in evaluation["scores"]}
        if model_id not in by_model or baseline_id not in by_model:
            raise ResearchLedgerError("same-origin baseline or candidate score is missing")
        candidate.append(by_model[model_id])
        baseline.append(by_model[baseline_id])
    n = len(candidate)
    events = sum(bool(e["outcome"][target]) for e in evaluations) if target in TARGETS[2:] else None
    result = {"paired_origins": n, "first_origin": evaluations[0]["origin_date"] if n else None,
              "last_origin": evaluations[-1]["origin_date"] if n else None, "events": events,
              "nonevents": n - events if events is not None else None,
              "candidate_log_loss": None, "baseline_log_loss": None, "log_loss_delta": None,
              "candidate_brier": None, "baseline_brier": None, "brier_delta": None,
              "log_loss_delta_ci_low": None, "log_loss_delta_ci_high": None}
    if not n:
        return result
    for metric in ("log_loss", "brier"):
        a = np.asarray([row[metric] for row in candidate], dtype=float)
        b = np.asarray([row[metric] for row in baseline], dtype=float)
        result[f"candidate_{metric}"] = float(a.mean())
        result[f"baseline_{metric}"] = float(b.mean())
        result[f"{metric}_delta"] = float((a - b).mean())
        if metric == "log_loss" and contiguous and n >= 2 * criteria["block_length"]:
            difference = a - b
            rng = np.random.default_rng(20260907)
            length = criteria["block_length"]
            means = np.empty(criteria["bootstrap_draws"])
            for index in range(len(means)):
                starts = rng.integers(0, n, size=math.ceil(n / length))
                positions = ((starts[:, None] + np.arange(length)) % n).ravel()[:n]
                means[index] = difference[positions].mean()
            result["log_loss_delta_ci_low"], result["log_loss_delta_ci_high"] = map(float, np.quantile(means, [.025, .975]))
    return result
