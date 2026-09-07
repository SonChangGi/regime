"""Generation-bound forecast research for the normal weekly publication path.

Only derived public blocks leave this module. Source snapshots and full study
receipts remain in the private cache; no forecast ledger or publication is edited.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import tempfile
from typing import Callable

import numpy as np
import pandas as pd

from regime_lab.analysis import forecast_audit_research as audit
from regime_lab.integrity import canonical_json_sha256_v1
from regime_lab.operational_forecast import frame_sha256
from regime_lab.payload import normalized_probabilities
from regime_lab.publication_contract import reject_raw_provider_material
from regime_lab.research import forecast_new_information as information
from regime_lab.research.contract import validate_research_extensions
from regime_lab.research.forecast_calibration import build_calibration_audit_from_frames
from regime_lab.research.forecast_information_study import run_information_study
from regime_lab.schema import STATE_ORDER

BLOCKS = ("forecast_research", "calibration_audit", "forecast_information")
SCHEMA_VERSION = "regime-forecast-publication-build/1"
ROOT = Path(__file__).resolve().parents[3]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_json(path: Path, document: dict) -> None:
    raw = json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    path.write_text(raw)


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recipe() -> dict:
    package = Path(__file__).resolve().parents[1]
    files = ["research/forecast_publication.py", "research/forecast_calibration.py",
             "research/forecast_information_study.py", "research/forecast_new_information.py",
             "research/contract.py", "analysis/forecast_audit_research.py", "analysis/forecast_paths.py",
             "analysis/boundary_forecast.py", "analysis/causal_calibration.py",
             "analysis/forecast_research_evaluation.py", "analysis/labels.py", "analysis/label_spec.py",
             "analysis/validation.py", "v5.py", "payload.py"]
    return {"source_hashes": {name: _file_sha(package / name) for name in files},
            "runtime": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn")},
            "model_protocol": audit.AuditResearchProtocol().record(),
            "information_protocol": _read_json(ROOT / "config/forecast-new-information.json")}


def _inputs(payload: dict, frames: dict) -> dict:
    # Existing optional research does not influence these studies. This stable
    # basis also permits rebuilding a release from its archived official inputs.
    official = {name: payload[name] for name in ("meta", "model", "label", "selection", "forecast", "weekly") if name in payload}
    return {"source_generation_id": payload["meta"]["generation_id"],
            "data_as_of": payload["meta"]["data_as_of"],
            "official_payload_sha256": canonical_json_sha256_v1(official),
            "input_frames": {name: {"sha256": frame_sha256(value), "rows": len(value)}
                             for name, value in frames.items()}}


def _validate_inputs(payload: dict, frames: dict) -> None:
    canonical, states = frames["canonical"], frames["states"]
    cutoff = information.utc(payload["meta"]["data_as_of"])
    if (canonical.empty or not canonical.index.equals(states.index)
            or states.index.has_duplicates or not states.index.is_monotonic_increasing
            or information.utc(states.index[-1]) != cutoff or not states.isin(STATE_ORDER).all()):
        raise ValueError("forecast publication canonical/state cutoff or history is invalid")
    for week in payload["weekly"]:
        origin = information.utc(week["data_as_of"])
        if origin not in states.index or states.loc[origin] != week["current"]["state"]:
            raise ValueError("forecast publication weekly states differ from the generation")
    baseline = frames["baseline"]
    required = {"model", "origin_date", "target_date", "current_state", "actual", "evaluation_split", *audit.PROBABILITIES}
    if not required <= set(baseline) or not set(audit.BASELINES) <= set(baseline.model):
        raise ValueError("forecast publication requires both fixed baseline OOS models")
    for row in baseline.loc[baseline.model.isin(audit.BASELINES)].itertuples():
        origin, target = information.utc(row.origin_date), information.utc(row.target_date)
        if (origin not in states.index or target not in states.index
                or states.loc[origin] != row.current_state or states.loc[target] != row.actual):
            raise ValueError("forecast publication baseline states differ from generation")
        position = states.index.get_loc(origin)
        if position + 1 >= len(states) or states.index[position + 1] != target:
            raise ValueError("forecast publication baseline target is not the next week")
        normalized_probabilities({s: getattr(row, f"p_{s}") for s in STATE_ORDER})
    data, future = frames["transition_predictions"], frames["transition_candidates"]
    required = {"model", "horizon", "origin_date", "target_end", "current_state", "actual_change", "raw_p_change", "p_change", "evaluation_split"}
    for frame in (data, future):
        if frame.empty or not required <= set(frame):
            raise ValueError("forecast publication requires resolved and future transition candidates")
        if frame.duplicated(["model", "horizon", "origin_date"]).any():
            raise ValueError("forecast publication duplicate transition origins")
        for column in ("raw_p_change", "p_change"):
            values = frame[column].to_numpy(dtype=float)
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError("forecast publication invalid transition probability")
        for row in frame.itertuples():
            origin, target = information.utc(row.origin_date), information.utc(row.target_end)
            if origin not in states.index or states.loc[origin] != row.current_state:
                raise ValueError("forecast publication transition state differs from generation")
            position, horizon = states.index.get_loc(origin), int(row.horizon)
            expected = (states.index[-1] + pd.Timedelta(weeks=position + horizon - len(states) + 1)
                        if position + horizon >= len(states) else states.index[position + horizon])
            if horizon not in (1, 4, 13) or target != expected:
                raise ValueError("forecast publication transition target differs from generation")
            if frame is data:
                if target > cutoff or pd.isna(row.actual_change):
                    raise ValueError("forecast publication resolved target is not mature")
                actual = bool(states.iloc[position + 1:position + horizon + 1].ne(row.current_state).any())
                if row.actual_change not in (True, False, 0, 1) or bool(row.actual_change) != actual:
                    raise ValueError("forecast publication transition outcome differs from official states")
            elif target <= cutoff or not pd.isna(row.actual_change) or row.evaluation_split != "prospective":
                raise ValueError("forecast publication future candidate is matured or labelled")
    resolved_pairs = set(zip(data.horizon.astype(int), data.model))
    latest = future.loc[pd.to_datetime(future.origin_date, utc=True).eq(cutoff)]
    if set(zip(latest.horizon.astype(int), latest.model)) != resolved_pairs:
        raise ValueError("forecast publication is missing a latest candidate/horizon")
    for horizon, model in resolved_pairs:
        part = future.loc[future.horizon.eq(horizon) & future.model.eq(model)]
        if set(pd.to_datetime(part.origin_date, utc=True)) != set(states.index[-horizon:]):
            raise ValueError("forecast publication is missing unresolved candidate origins")
    cutoff = pd.to_datetime(payload["model"].get("transition_selection_end", "2023-01-01"), utc=True)
    for row in data.itertuples():
        expected_split = "selection" if information.utc(row.target_end) < cutoff else "retrospective_diagnostic" if information.utc(row.origin_date) >= cutoff else None
        if expected_split is None or row.evaluation_split != expected_split:
            raise ValueError("forecast publication transition split crosses the frozen boundary")


def _inventory(directory: Path) -> dict:
    return {str(path.relative_to(directory)): {"sha256": _file_sha(path), "bytes": path.stat().st_size}
            for path in sorted(directory.rglob("*")) if path.is_file() and path.name != "snapshot.json"}


def _validate_snapshot(directory: Path, cutoff: str, control_sha: str | None) -> dict:
    record = _read_json(directory / "snapshot.json")
    if record.get("data_as_of") != cutoff or record.get("existing_sources_manifest_sha256") != control_sha:
        raise ValueError("forecast information snapshot cutoff or existing source identity differs")
    if record.get("files") != _inventory(directory):
        raise ValueError("forecast information source snapshot checksum differs")
    return record


def _merge_versions(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.glob("*")):
        if not path.is_file() or path.suffix not in (".json", ".blob"):
            raise ValueError("unexpected optional source snapshot file")
        target = destination / path.name
        if target.exists():
            if target.read_bytes() != path.read_bytes():
                raise ValueError("optional source first-seen record conflicts with cache")
        else:
            shutil.copyfile(path, target)
    for source_id in {p.name.rsplit("-", 1)[0] for p in destination.glob("*.json")}:
        information.load_snapshots(destination, source_id)


def _boundary_history(forecast: dict, states: pd.Series) -> pd.DataFrame:
    model = next(m for m in forecast["models"] if m["id"] == audit.BOUNDARY_BASELINE)
    rows = []
    for row in model["history"]:
        origin = information.utc(row["origin_date"])
        target = states.index[states.index.get_loc(origin) + 1]
        rows.append({"origin_date": origin, "target_date": target,
                     "current_state": row["current_state"], "actual": states.loc[target],
                     "evaluation_split": "selection" if target < audit.CUTOFF else "holdout",
                     **{f"p_{s}": row["next_state"][s] for s in STATE_ORDER}})
    return information.validate_history(pd.DataFrame(rows))


def _information_snapshot(*, cutoff: str, history: pd.DataFrame, cache: Path,
                          existing_sources: Path | None, information_sources: Path | None,
                          offline: bool, config: dict, progress) -> tuple[Path, dict]:
    root = cache / "information-inputs"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / pd.Timestamp(cutoff).date().isoformat()
    control_sha = _file_sha(existing_sources / "manifest.json") if existing_sources else None
    if destination.exists():
        return destination, _validate_snapshot(destination, cutoff, control_sha)
    versions = cache / "information-source-versions"
    versions.mkdir(parents=True, exist_ok=True)
    if information_sources:
        _merge_versions(information_sources, versions)
    staging = Path(tempfile.mkdtemp(prefix=".information-", dir=root))
    try:
        shutil.copytree(versions, staging / "sources")
        if existing_sources:
            manifest = _read_json(existing_sources / "manifest.json")
            controls = staging / "existing-sources"
            controls.mkdir()
            chosen = {}
            for key in ("vix", "vix9d", "vvix"):
                row = manifest["sources"][key]
                if Path(row["file"]).name != row["file"] or _file_sha(existing_sources / row["file"]) != row["sha256"]:
                    raise ValueError("existing control snapshot identity is invalid")
                shutil.copyfile(existing_sources / row["file"], controls / row["file"])
                chosen[key] = row
            _write_json(controls / "manifest.json", {"sources": chosen})
        study = run_information_study(history, data_as_of=cutoff, config=config,
            output=staging / "initial-study", sources_dir=staging / "sources",
            existing_sources=staging / "existing-sources" if existing_sources else None,
            offline=offline, refresh=not offline, progress=progress)
        record = {"schema_version": "regime-forecast-information-snapshot/1", "data_as_of": cutoff,
                  "existing_sources_manifest_sha256": control_sha,
                  "runtime_as_of": study["runtime_as_of"],
                  "source_statuses": study["sources"],
                  "study_recipe_sha256": canonical_json_sha256_v1(_recipe()),
                  "history_sha256": frame_sha256(history), "files": _inventory(staging)}
        _write_json(staging / "snapshot.json", record)
        _validate_snapshot(staging, cutoff, control_sha)
        _merge_versions(staging / "sources", versions)
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination, record


def _validate_outputs(blocks: dict, *, payload: dict, frames: dict, provenance: dict) -> None:
    if set(blocks) != set(BLOCKS):
        raise ValueError("forecast publication requires all three research blocks")
    cutoff = payload["meta"]["data_as_of"]
    validate_research_extensions(blocks, data_as_of=cutoff)
    reject_raw_provider_material(blocks)
    forecast = blocks["forecast_research"]
    expected = set(pd.to_datetime(frames["baseline"].loc[frames["baseline"].model.eq(audit.BASELINES[0]), "origin_date"], utc=True))
    expected_models = {*audit.BASELINES, audit.BOUNDARY_BASELINE, audit.ASYMMETRIC_MODEL, audit.PATH_MODEL, audit.MARKOV_PATH_MODEL}
    if set(m["id"] for m in forecast["models"]) != expected_models:
        raise ValueError("forecast publication model roster is incomplete")
    for model in forecast["models"]:
        if set(information.utc(row["origin_date"]) for row in model["history"]) != expected:
            raise ValueError("forecast publication lost matched model origins")
        if model["id"] not in audit.BASELINES and "latest" not in model:
            raise ValueError("forecast publication missing latest research model")
        for row in [*model["history"], *([model["latest"]] if "latest" in model else [])]:
            if model["id"] in {audit.PATH_MODEL, audit.MARKOV_PATH_MODEL} and set(row["horizons"]) != {"1w", "4w", "13w"}:
                raise ValueError("forecast publication path model requires all future horizons")
            if frames["states"].loc[information.utc(row["origin_date"])] != row["current_state"]:
                raise ValueError("forecast publication model output state differs")
    calibration = blocks["calibration_audit"]
    if not calibration["rows"] or not calibration.get("latest_rows"):
        raise ValueError("forecast publication calibration results are empty")
    pairs = set(zip(frames["transition_predictions"].horizon.astype(int), frames["transition_predictions"].model))
    if {(r["horizon_weeks"], r["model"]) for r in calibration["latest_rows"]} != pairs:
        raise ValueError("forecast publication calibration latest models are incomplete")
    for row in calibration["rows"]:
        data = frames["transition_predictions"]
        part = data.loc[data.horizon.eq(row["horizon_weeks"]) & data.model.eq(row["model"]) & data.evaluation_split.eq(row["evaluation_split"])]
        if len(part) != row["n_predictions"]:
            raise ValueError("forecast publication calibration sample differs")
    info = blocks["forecast_information"]
    if not info["rows"] and (info.get("status") != "not_evaluable" or not info.get("blocks")):
        raise ValueError("forecast publication missing explicit optional-information status")
    reference = frames["baseline"].loc[frames["baseline"].model.eq(audit.BASELINES[0])]
    for row in info["rows"]:
        if row["matched_n"] > int(reference.evaluation_split.eq(row["evaluation_split"]).sum()):
            raise ValueError("forecast publication information sample exceeds generation origins")
    for name, block in blocks.items():
        if block.get("publication_provenance") != provenance:
            raise ValueError("forecast publication block provenance differs")
        if block.get("artifacts") != [{"label": "전체 결과 JSON", "url": f"./data/{name}.json"}]:
            raise ValueError("forecast publication derived artifact link differs")


def build_forecast_publication_research(
    payload: dict, canonical: pd.DataFrame, states: pd.Series, baseline: pd.DataFrame,
    transition_predictions: pd.DataFrame, transition_candidates: pd.DataFrame,
    cache_directory: Path, *, existing_sources: Path | None = None,
    information_sources: Path | None = None, offline: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Return ``{'blocks': three_public_blocks, 'provenance': receipt}``.

    The output cache is addressed by official inputs, source snapshot and the
    full computation recipe. A corrupt cache raises instead of falling back.
    First-seen source clocks are immutable across retries and future cutoffs.
    """
    frames = {"canonical": canonical.copy(deep=True), "states": states.copy(deep=True),
              "baseline": baseline.copy(deep=True),
              "transition_predictions": transition_predictions.copy(deep=True),
              "transition_candidates": transition_candidates.copy(deep=True)}
    source_payload = deepcopy(payload)
    _validate_inputs(source_payload, frames)
    identity, recipe = _inputs(source_payload, frames), _recipe()
    cache = Path(cache_directory)
    cache.mkdir(parents=True, exist_ok=True)
    snapshot_dir = cache / "information-inputs" / pd.Timestamp(identity["data_as_of"]).date().isoformat()
    control_sha = _file_sha(existing_sources / "manifest.json") if existing_sources else None
    snapshot = _validate_snapshot(snapshot_dir, identity["data_as_of"], control_sha) if snapshot_dir.exists() else None

    def provenance_for(record):
        value = {"schema_version": SCHEMA_VERSION, **identity,
                 "recipe_sha256": canonical_json_sha256_v1(recipe),
                 "information_snapshot_sha256": canonical_json_sha256_v1(record),
                 "evidence_track": "reconstructed_market", "automatic_promotion": False}
        value["cache_key"] = canonical_json_sha256_v1(value)
        return value

    if snapshot is not None:
        provenance = provenance_for(snapshot)
        destination = cache / "results" / provenance["cache_key"]
        if destination.exists():
            receipt = _read_json(destination / "receipt.json")
            blocks = {name: _read_json(destination / f"{name}.json") for name in BLOCKS}
            if receipt.get("provenance") != provenance or receipt.get("outputs") != {name: canonical_json_sha256_v1(value) for name, value in blocks.items()}:
                raise ValueError("forecast publication cache checksum or recipe differs")
            _validate_outputs(blocks, payload=source_payload, frames=frames, provenance=provenance)
            return {"blocks": blocks, "provenance": provenance}
    if progress:
        progress("방향·기간별 국면 연구")
    result = audit.run_audit_research(frames["canonical"], frames["states"], frames["baseline"], progress=progress)
    forecast = audit.forecast_research_extension(result.document)
    history = _boundary_history(forecast, frames["states"])
    if progress:
        progress("확률 보정 비교")
    calibration = build_calibration_audit_from_frames(source_payload, frames["transition_predictions"], frames["transition_candidates"],
        sources=[{"id": name, "sha256": row["sha256"], "rows": row["rows"]} for name, row in identity["input_frames"].items() if name.startswith("transition_")])
    snapshot_dir, snapshot = _information_snapshot(cutoff=identity["data_as_of"], history=history, cache=cache,
        existing_sources=existing_sources, information_sources=information_sources, offline=offline,
        config=recipe["information_protocol"], progress=progress)
    if snapshot["study_recipe_sha256"] == canonical_json_sha256_v1(recipe) and snapshot["history_sha256"] == frame_sha256(history):
        study = _read_json(snapshot_dir / "initial-study/summary.json")
    else:
        with tempfile.TemporaryDirectory(prefix=".information-replay-", dir=cache) as temporary:
            study = run_information_study(history, data_as_of=identity["data_as_of"], config=recipe["information_protocol"],
                output=Path(temporary), sources_dir=snapshot_dir / "sources",
                existing_sources=snapshot_dir / "existing-sources" if (snapshot_dir / "existing-sources").exists() else None,
                offline=True, as_of=snapshot["runtime_as_of"],
                source_statuses=snapshot["source_statuses"], progress=progress)
    public_study = deepcopy(study)
    for row in public_study["blocks"].values():
        reason = str(row.get("reason", ""))
        if row.get("status") in {"unavailable", "invalid_optional_block"} or any(marker in reason for marker in ("/Users/", "/private/", "file://", "\\Users\\")):
            row["reason"] = "optional_source_unavailable"
    info = information.build_forecast_information(public_study)
    info["status"] = "research_only" if info["rows"] else "not_evaluable"
    # Publish availability/counts only. Provider observations and private paths
    # stay in the local full study; this is deliberately not its raw summary.
    info["blocks"] = {name: {key: value for key, value in row.items() if key in {"status", "reason", "historical_origins_with_features", "stored_versions", "parsed_rows", "alternate_representation"}}
                      for name, row in public_study["blocks"].items()}
    info["sources"] = [{"id": name, **{key: row[key] for key in ("sha256", "retrieved_at", "status") if key in row}}
                       for name, row in study["sources"].items()]
    info["generated_at"] = study["completed_at"]
    provenance = provenance_for(snapshot)
    blocks = {"forecast_research": forecast, "calibration_audit": calibration, "forecast_information": info}
    for name, block in blocks.items():
        block["publication_provenance"] = provenance
        block["artifacts"] = [{"label": "전체 결과 JSON", "url": f"./data/{name}.json"}]
    _validate_outputs(blocks, payload=source_payload, frames=frames, provenance=provenance)
    if _inputs(payload, frames) != identity or _recipe() != recipe:
        raise ValueError("forecast publication inputs or source recipe changed during build")
    root = cache / "results"
    root.mkdir(exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".forecast-publication-", dir=root))
    try:
        for name, block in blocks.items():
            _write_json(staging / f"{name}.json", block)
        _write_json(staging / "receipt.json", {"provenance": provenance, "recipe": recipe,
            "compiled_at": datetime.now(timezone.utc).isoformat(),
            "outputs": {name: canonical_json_sha256_v1(value) for name, value in blocks.items()}})
        staging.rename(root / provenance["cache_key"])
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {"blocks": blocks, "provenance": provenance}
