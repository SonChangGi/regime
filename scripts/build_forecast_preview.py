#!/usr/bin/env python3
"""Build an atomic local dashboard with the new causal forecasting models."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import pandas as pd

from regime_lab.contract_v5 import validate_v5_payload
from regime_lab.dashboard_split import build_dashboard_split, build_history_chunks
from regime_lab.operational_forecast import frame_sha256
from regime_lab.publication_contract import rewrite_index_asset_versions
from regime_lab.research.forecast_improvement import build_forecast_improvement
from regime_lab.research.forecast_contract import validate_forecast_improvement
from regime_lab.research.forecast_assets import build_forecast_asset_statistics
from scripts.package_public_demo import STATIC_ALLOWLIST


def encoded(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()


def build_preview(source: Path, inputs: Path, baseline: Path, output: Path, cache: Path) -> dict:
    source_raw = source.read_bytes()
    original = json.loads(source_raw)
    manifest = json.loads((inputs / "input-manifest.json").read_text())
    canonical, states = pd.read_pickle(inputs / "canonical.pkl"), pd.read_pickle(inputs / "states.pkl")
    for name, frame in [("canonical", canonical), ("states", states)]:
        if frame_sha256(frame) != manifest["frames"][name]:
            raise ValueError(f"frozen {name} frame hash differs")
    if states.index[-1] != pd.Timestamp(original["meta"]["data_as_of"]):
        raise ValueError("frozen forecast inputs and source snapshot have different cutoffs")
    for week in original["weekly"]:
        when = pd.Timestamp(week["data_as_of"])
        if str(states.loc[when]) != week["current"]["state"]:
            raise ValueError("frozen labels disagree with an issued observation")
    baseline_raw = baseline.read_bytes()
    block = build_forecast_improvement(canonical, states, pd.read_csv(baseline), cache)
    resamples = original["model"]["execution_parameters"]["conditional_outcome_bootstrap_resamples"]
    block["asset_statistics"] = build_forecast_asset_statistics(canonical, block,
        bootstrap_resamples=resamples, cache_directory=cache.parent / "forecast-assets-cache")
    validate_forecast_improvement(block, data_as_of=original["meta"]["data_as_of"], outcome_resamples=resamples)
    payload = deepcopy(original)
    payload["research"]["forecast_improvement"] = block
    payload["meta"]["publication_status"] = "unpublished"
    payload["meta"].pop("publication_review", None)
    payload["meta"].pop("generation_manifest_sha256", None)
    payload["model"]["lifecycle"]["publication"] = {"status": "unpublished"}
    payload["model"]["lifecycle"]["deployment"] = {"status": "candidate"}
    for field in ("weekly", "forecast", "selection"):
        if payload[field] != original[field]:
            raise ValueError(f"preview changed official {field}")
    validate_v5_payload(payload)
    raw = encoded(payload)
    core, research = build_dashboard_split(payload, payload_raw=raw)
    history, _ = build_history_chunks(payload, payload_raw=raw)
    files = {"data/regime-results.json": raw, "data/regime-core.json": core,
             "data/regime-research.json": research, **history}
    # The old benchmark calculations are identical; only their enclosing
    # candidate document gains research. Rebind its source bytes for preview.
    comparison = json.loads((source.parent / "v5-vs-v4-comparison.json").read_text())
    comparison["inputs"]["v5"]["regime_results"]["sha256"] = hashlib.sha256(raw).hexdigest()
    files["data/v5-vs-v4-comparison.json"] = encoded(comparison)
    files["data/selection-family-audit.json"] = (source.parent / "selection-family-audit.json").read_bytes()
    for name in STATIC_ALLOWLIST:
        files[name] = (ROOT / "web" / name).read_bytes()
    files["index.html"] = rewrite_index_asset_versions(files["index.html"], styles_raw=files["styles.css"],
        app_raw=files["app.js"], operating_contract_raw=files["operating-contract.generated.js"],
        extra_assets={name: files[name] for name in ("insights.js", "insights.css")})
    inventory = {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)} for name, data in files.items()}
    identity = hashlib.sha256(encoded(inventory)).hexdigest()
    generations = output.parent / "preview-generations"
    generations.mkdir(parents=True, exist_ok=True)
    destination = generations / identity[:24]
    if not destination.exists():
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=generations))
        try:
            for name, data in files.items():
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                if hashlib.sha256(target.read_bytes()).hexdigest() != inventory[name]["sha256"]:
                    raise ValueError("preview staging hash mismatch")
            (staging / "preview-inventory.json").write_bytes(encoded(inventory))
            staging.rename(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    elif any((destination / name).read_bytes() != data for name, data in files.items()):
        raise ValueError("existing preview generation differs from its content hash")
    if source.read_bytes() != source_raw or baseline.read_bytes() != baseline_raw:
        raise ValueError("source changed while building preview")
    if output.exists() and not output.is_symlink():
        output.rename(generations / ("previous-" + uuid.uuid4().hex[:12]))
    pointer = output.parent / (".preview-" + uuid.uuid4().hex)
    pointer.symlink_to(os.path.relpath(destination, output.parent), target_is_directory=True)
    os.replace(pointer, output)
    return {"ok": True, "output": str(output.resolve()), "files": len(files),
            "source_sha256": hashlib.sha256(source_raw).hexdigest(), "research_cache_key": block["provenance"]["cache_key"],
            "forecast_and_weekly_preserved": True, "source_unchanged": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "publication/live/regime-results.json")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "build/forecast-performance/preview-root/regime")
    parser.add_argument("--cache", type=Path, default=ROOT / "build/forecast-performance/publication-cache")
    args = parser.parse_args()
    report = build_preview(args.source, args.inputs, args.baseline, args.output, args.cache)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
