#!/usr/bin/env python3
"""Build an immutable local preview with forecast research and source evidence."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT))
import pandas as pd
from regime_lab.contract_v5 import validate_v5_payload
from regime_lab.dashboard_split import build_dashboard_split, build_history_chunks
from regime_lab.dataset import evidence_drivers
from regime_lab.publication_contract import rewrite_index_asset_versions
from regime_lab.v5 import _context_extremes
from scripts.package_public_demo import STATIC_ALLOWLIST


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))+"\n").encode()


def require_local_output(path: Path):
    allowed = (ROOT/"build/forecast-upgrade-20260907").resolve()
    if allowed not in path.resolve().parents:
        raise ValueError("preview must stay inside the isolated forecast-upgrade workspace")


def read_information_artifacts(information: Path, macro: dict) -> dict[Path, bytes]:
    manifest = macro.get("artifact_manifest", {})
    generation = manifest.get("generation", "")
    if not re.fullmatch(r"[0-9a-f]{24}", generation) or not isinstance(manifest.get("files"), list) or not manifest["files"]:
        raise ValueError("information requires an immutable artifact manifest")
    artifacts = {}
    for record in manifest["files"]:
        relative = Path(record["path"])
        path = information.parent/relative
        expected = Path("runs")/generation/relative.name
        if relative != expected or path.suffix != ".csv" or information.parent.resolve() not in path.resolve().parents or path.is_symlink():
            raise ValueError("information artifact escapes its immutable run")
        if path in artifacts:
            raise ValueError("duplicate information artifact")
        content = path.read_bytes()
        if len(content) != record["bytes"] or hashlib.sha256(content).hexdigest() != record["sha256"]:
            raise ValueError("information artifact differs from its completed run")
        artifacts[path] = content
    return artifacts


def build_preview(source: Path, modeling: Path, information: Path, features: Path,
                  output: Path, operational: Path | None = None,
                  weekly_candidates: Path | None = None) -> dict:
    require_local_output(output)
    paths = [source, modeling, information, features, source.parent/"v5-vs-v4-comparison.json",
             source.parent/"selection-family-audit.json"] + ([operational] if operational else []) + ([weekly_candidates] if weekly_candidates else [])
    if any(output.resolve() == path.resolve() or output.absolute() == path.absolute() for path in paths):
        raise ValueError("preview output must not replace a source input")
    originals = {path: path.read_bytes() for path in paths}
    payload = json.loads(originals[source])
    archived = deepcopy(payload)
    enhancement = json.loads(originals[modeling])
    if enhancement["source_generation_id"] != payload["meta"]["generation_id"] or pd.Timestamp(enhancement["data_as_of"]) != pd.Timestamp(payload["meta"]["data_as_of"]):
        raise ValueError("modeling artifacts do not match the source generation")
    enhancement["additional_information"] = json.loads(originals[information])
    if pd.Timestamp(enhancement["additional_information"]["data_as_of"]) != pd.Timestamp(enhancement["data_as_of"]):
        raise ValueError("information cutoff differs")
    macro = enhancement["additional_information"]
    expected_source_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()).hexdigest()
    if macro.get("source_generation_id") != payload["meta"]["generation_id"] or macro.get("source_payload_sha256") != expected_source_hash:
        raise ValueError("information artifacts do not match the source generation")
    if operational:
        diagnostics = json.loads(originals[operational])
        payload["research"]["operational_diagnostics"] = diagnostics
        enhancement["operational"] = diagnostics
    if weekly_candidates:
        summary = json.loads(originals[weekly_candidates])
        validate_weekly_summary(summary, enhancement)
        enhancement["weekly_candidates"] = summary
    feature_frame = pd.read_pickle(io.BytesIO(originals[features]))
    for week in payload["weekly"]:
        at = pd.Timestamp(week["data_as_of"])
        if at not in feature_frame.index:
            raise ValueError("feature frame does not cover every displayed week")
        week["extreme_context"] = _context_extremes(evidence_drivers(feature_frame, at))
    for before, after in zip(archived["weekly"], payload["weekly"]):
        if {k:v for k,v in before.items() if k != "extreme_context"} != {k:v for k,v in after.items() if k != "extreme_context"}:
            raise ValueError("archived forecast was modified")
    for key in ("forecast", "selection", "label"):
        if payload[key] != archived[key]:
            raise ValueError(f"archived {key} changed")
    payload["meta"]["publication_status"] = "unpublished"
    payload["meta"].pop("publication_review", None)
    payload["meta"].pop("generation_manifest_sha256", None)
    payload["model"]["lifecycle"]["publication"] = {"status": "unpublished"}
    payload["model"]["lifecycle"]["deployment"] = {"status": "candidate"}
    validate_v5_payload(payload)
    raw = encoded(payload)
    core, research = build_dashboard_split(payload, payload_raw=raw)
    history, _ = build_history_chunks(payload, payload_raw=raw)
    files = {"data/regime-results.json": raw, "data/regime-core.json": core,
             "data/regime-research.json": research, **history}
    # Keep downloadable research evidence inside the served preview directory.
    artifact_links = []
    information_artifacts = read_information_artifacts(information, macro)
    originals.update(information_artifacts)
    for artifacts, prefix in ((sorted(modeling.parent.glob("*.csv")), "models"), (information_artifacts, "information")):
        for path in artifacts:
            destination = f"data/research/{prefix}-{path.name}"
            originals.setdefault(path, path.read_bytes())
            files[destination] = originals[path]
            artifact_links.append({"label": path.stem, "url": f"./{destination}"})
    enhancement.setdefault("provenance", {})["artifacts"] = artifact_links
    files["data/forecast-enhancements.json"] = encoded(enhancement)
    files["data/additional-information.json"] = originals[information]
    if weekly_candidates:
        files["data/candidate-summary.json"] = originals[weekly_candidates]
    for key in ("forecast_research", "calibration_audit", "forecast_information"):
        if key in payload["research"]:
            files[f"data/{key}.json"] = encoded(payload["research"][key])
    comparison = json.loads(originals[source.parent/"v5-vs-v4-comparison.json"])
    comparison["inputs"]["v5"]["regime_results"]["sha256"] = hashlib.sha256(raw).hexdigest()
    files["data/v5-vs-v4-comparison.json"] = encoded(comparison)
    files["data/selection-family-audit.json"] = originals[source.parent/"selection-family-audit.json"]
    for name in STATIC_ALLOWLIST:
        path = ROOT/"web"/name
        originals[path] = path.read_bytes()
        files[name] = originals[path]
    extras = {name: files[name] for name in ("insights.js", "insights.css", "forecast-enhancements.js", "forecast-enhancements.css")}
    files["index.html"] = rewrite_index_asset_versions(files["index.html"], styles_raw=files["styles.css"],
        app_raw=files["app.js"], operating_contract_raw=files["operating-contract.generated.js"], extra_assets=extras)
    inventory = {name: {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)} for name, content in files.items()}
    identity = hashlib.sha256(encoded(inventory)).hexdigest()
    generations = output.parent/"preview-generations"
    if generations.is_symlink():
        raise ValueError("preview generation directory must not be a symlink")
    generations.mkdir(parents=True, exist_ok=True)
    destination = generations/identity[:24]
    if destination.is_symlink():
        raise ValueError("preview generation must not be a symlink")
    if not destination.exists():
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=generations))
        try:
            for name, content in files.items():
                target = staging/name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                if target.read_bytes() != content:
                    raise ValueError("preview write verification failed")
            (staging/"preview-inventory.json").write_bytes(encoded(inventory))
            staging.rename(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    elif any((destination/name).read_bytes() != content for name, content in files.items()):
        raise ValueError("existing preview generation changed")
    if any(path.read_bytes() != content for path, content in originals.items()):
        raise ValueError("an input changed while building the preview")
    if output.exists() and not output.is_symlink():
        raise ValueError("preview output must be a new path or an owned symlink")
    pointer = output.parent/(".preview-"+uuid.uuid4().hex)
    pointer.symlink_to(os.path.relpath(destination, output.parent), target_is_directory=True)
    os.replace(pointer, output)
    result = {"ok": True, "output": str(output), "generation": identity,
              "files": len(files), "source_sha256": hashlib.sha256(originals[source]).hexdigest(),
              "archived_forecasts_unchanged": True, "source_files_unchanged": True,
              "external_publication": False}
    (output.parent/"preview-build.json").write_bytes(encoded(result))
    return result


def validate_weekly_summary(summary: dict, enhancement: dict):
    if (summary.get("schema_version") != "regime-weekly-candidates-summary/1"
            or summary.get("scope") != "local_preview_only"
            or summary.get("automatic_promotion") is not False
            or summary.get("source_generation_id") != enhancement["source_generation_id"]
            or pd.Timestamp(summary.get("data_as_of")) != pd.Timestamp(enhancement["data_as_of"])):
        raise ValueError("weekly candidate summary differs from source generation")
    if summary["issued_predictions"] != summary["pending_predictions"] + summary["matured_predictions"]:
        raise ValueError("weekly candidate score counts differ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT/"publication/live/regime-results.json")
    parser.add_argument("--modeling", type=Path, default=ROOT/"build/forecast-upgrade-20260907/modeling/forecast-enhancements.json")
    parser.add_argument("--information", type=Path, default=ROOT/"build/forecast-upgrade-20260907/information/additional-information.json")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--operational", type=Path)
    parser.add_argument("--weekly-candidates", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT/"build/forecast-upgrade-20260907/preview")
    args = parser.parse_args()
    print(json.dumps(build_preview(args.source, args.modeling, args.information, args.features, args.output, args.operational, args.weekly_candidates), ensure_ascii=False))


if __name__ == "__main__":
    main()
