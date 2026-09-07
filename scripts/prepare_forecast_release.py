#!/usr/bin/env python3
"""Prepare a reviewed forecast-research release in an isolated build directory."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import pandas as pd

from regime_lab.artifact_inventory import verify_artifact_inventory, write_artifact_inventory
from regime_lab.config import project_root
from regime_lab.integrity import bind_payload_to_generation_manifest, build_generation_manifest, canonical_json_sha256_v1, reviewed_candidate_payload, validate_generation_manifest
from regime_lab.io import write_json_atomic
from regime_lab.operational_forecast import frame_sha256
from regime_lab.path_safety import confined_mutable_path
from regime_lab.research.forecast_assets import _identity as asset_identity, build_forecast_asset_statistics
from regime_lab.research.forecast_contract import validate_forecast_improvement
from regime_lab.research.forecast_improvement import build_forecast_improvement
from regime_lab.schema import validate_dashboard_payload
from regime_lab.v5_artifacts import verify_staged_v5_research_artifacts
from scripts.package_public_demo import STATIC_ALLOWLIST, package_public_dashboard, validate_public_live_derived_payload
from scripts.prepare_comprehensive_release import PUBLICATION_MEMBERS, ReleasePreparationError, _first_difference, _json, _sha256
from scripts.promote_v5_publication import _build_expected_comparison, promote
from scripts.verify_public_package import verify_public_package


def _protected(payload: dict) -> dict:
    """Only review bookkeeping and the one requested research block may differ."""
    value = deepcopy(payload)
    for name in ("publication_status", "publication_review", "generation_manifest_sha256"):
        value["meta"].pop(name, None)
    for name in ("publication", "deployment"):
        value["model"]["lifecycle"].pop(name, None)
    value["research"].pop("forecast_improvement", None)
    return value


def validate_forecast_only_update(source: dict, candidate: dict) -> None:
    if source["meta"].get("publication_status") != "reviewed_publication":
        raise ReleasePreparationError("source must be a reviewed publication")
    if candidate["meta"].get("publication_status") != "unpublished":
        raise ReleasePreparationError("forecast preview must be unpublished")
    if any(name in candidate["meta"] for name in ("publication_review", "generation_manifest_sha256")):
        raise ReleasePreparationError("forecast preview inherited a previous review")
    if "forecast_improvement" not in candidate.get("research", {}):
        raise ReleasePreparationError("forecast preview has no new model evidence")
    difference = _first_difference(_protected(source), _protected(candidate))
    if difference:
        raise ReleasePreparationError(f"forecast release changed protected content: {difference}")


def _require_staging_destination(output: Path, inputs: list[Path]) -> Path:
    root = project_root()
    output = confined_mutable_path(output, project_directory=root, label="forecast release candidate")
    if not output.is_relative_to((root / "build").resolve()):
        raise ReleasePreparationError("forecast release candidate must stay below build/")
    if output.exists() or output.is_symlink():
        raise ReleasePreparationError("forecast release candidate must not exist")
    for item in inputs:
        source = item.resolve()
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ReleasePreparationError("forecast release candidate overlaps a read-only input")
    return output


def prepare_forecast_release(*, source_path: Path, source_artifacts: Path, preview_path: Path,
                             input_directory: Path, forecast_cache: Path, asset_cache: Path,
                             output_root: Path, reviewed_at: datetime) -> dict:
    if reviewed_at.tzinfo is None or reviewed_at.utcoffset() is None:
        raise ReleasePreparationError("forecast review time must include a timezone")
    output_root = _require_staging_destination(output_root, [source_path.parent, source_artifacts, preview_path, input_directory, forecast_cache, asset_cache])
    source_manifest = source_path.with_name("generation-manifest.json")
    tracked = {f"source:{name}": source_path.parent / name for name in PUBLICATION_MEMBERS}
    tracked.update({f"input:{name}": input_directory / name for name in ("canonical.pkl", "states.pkl", "input-manifest.json")})
    tracked["preview"] = preview_path
    tracked["source-inventory"] = source_artifacts / "SHA256SUMS"
    tracked.update({f"web:{name}": ROOT / "web" / name for name in STATIC_ALLOWLIST})
    before = {name: _sha256(path) for name, path in tracked.items()}
    generation = validate_generation_manifest(source_manifest, require_comparison=True, require_selection_family=True,
        artifact_directory=source_artifacts, payload_path_override=source_path,
        comparison_path_override=source_path.with_name("v5-vs-v4-comparison.json"),
        selection_family_path_override=source_path.with_name("selection-family-audit.json"))
    source = generation["payload"]
    validate_public_live_derived_payload(source)
    preview = _json(preview_path)
    validate_dashboard_payload(preview)
    validate_forecast_only_update(source, preview)

    input_manifest = _json(input_directory / "input-manifest.json")
    canonical = pd.read_pickle(input_directory / "canonical.pkl")
    states = pd.read_pickle(input_directory / "states.pkl")
    for name, frame in (("canonical", canonical), ("states", states)):
        if input_manifest["frames"].get(name) != frame_sha256(frame):
            raise ReleasePreparationError(f"frozen forecast {name} hash differs")
    if states.index[-1] != pd.Timestamp(source["meta"]["data_as_of"]):
        raise ReleasePreparationError("forecast input cutoff differs from reviewed source")
    for week in source["weekly"]:
        if states.loc[pd.Timestamp(week["data_as_of"])] != week["current"]["state"]:
            raise ReleasePreparationError("forecast input labels differ from issued observations")
    baseline = pd.read_csv(source_artifacts / "oos-predictions.csv")
    reproduced = build_forecast_improvement(canonical, states, baseline, forecast_cache)
    model_cache = forecast_cache / reproduced["provenance"]["cache_key"]
    resamples = source["model"]["execution_parameters"]["conditional_outcome_bootstrap_resamples"]
    assets_key = canonical_json_sha256_v1(asset_identity(canonical, reproduced, resamples))
    assets_directory = asset_cache / assets_key
    reproduced["asset_statistics"] = build_forecast_asset_statistics(canonical, reproduced, bootstrap_resamples=resamples, cache_directory=asset_cache)
    validate_forecast_improvement(reproduced, data_as_of=source["meta"]["data_as_of"], outcome_resamples=resamples)
    difference = _first_difference(reproduced, preview["research"]["forecast_improvement"])
    if difference:
        raise ReleasePreparationError(f"forecast preview differs from reproduced research: {difference}")
    evidence = {
        "forecast-oos-predictions.csv": model_cache / "oos-predictions.csv",
        "forecast-latest.json": model_cache / "latest.json",
        "forecast-model-cache-manifest.json": model_cache / "manifest.json",
        "forecast-asset-outcomes.csv": assets_directory / "outcomes.csv",
        "forecast-asset-statistics.csv": assets_directory / "statistics.csv",
        "forecast-asset-cache-manifest.json": assets_directory / "manifest.json",
        "forecast-input-manifest.json": input_directory / "input-manifest.json",
    }
    for name, path in evidence.items():
        tracked[f"evidence:{name}"] = path
        before[f"evidence:{name}"] = _sha256(path)
    if before != {name: _sha256(path) for name, path in tracked.items()}:
        raise ReleasePreparationError("forecast inputs changed during review")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent))
    installed = False
    try:
        artifacts = staging / "artifacts"
        shutil.copytree(source_artifacts, artifacts)
        for name, path in evidence.items():
            shutil.copyfile(path, artifacts / name)
        write_json_atomic(artifacts / "forecast-improvement.json", reproduced)
        write_json_atomic(artifacts / "forecast-release-inputs.json", {
            "schema_version": "regime-forecast-release-inputs/1", "source_payload_sha256": _sha256(source_path),
            "generation_id": source["meta"]["generation_id"], "data_as_of": source["meta"]["data_as_of"],
            "read_only_inputs_sha256": before, "official_forecast_and_selection_preserved": True,
            "model_cache_key": reproduced["provenance"]["cache_key"], "asset_cache_key": assets_key,
        })
        write_artifact_inventory(artifacts)
        candidate = reviewed_candidate_payload(preview)
        verify_staged_v5_research_artifacts(candidate["model"]["research_artifacts"], artifacts)
        private = staging / "candidate"
        private.mkdir()
        payload_path, manifest_path = private / "regime-results.json", private / "generation-manifest.json"
        manifest = build_generation_manifest(payload=candidate, payload_path=payload_path, artifact_directory=artifacts,
            input_snapshot=generation["input_snapshot"], label_spec_path=ROOT / generation["label_spec"]["path"],
            selection_family=generation["selection_family"], selection_family_path=artifacts / "selection-family-audit.json")
        candidate = bind_payload_to_generation_manifest(candidate, manifest)
        write_json_atomic(payload_path, candidate)
        write_json_atomic(manifest_path, manifest)
        comparison_path = private / "v5-vs-v4-comparison.json"
        write_json_atomic(comparison_path, _build_expected_comparison(v5_artifacts=artifacts, candidate_path=payload_path))
        publication = staging / "publication/live"
        promote(candidate_path=payload_path, v5_artifacts=artifacts, comparison_path=comparison_path,
                generation_manifest_path=manifest_path, output_path=publication / "regime-results.json",
                publication_contract_directory=ROOT / "publication/live", reviewed_at=reviewed_at)
        reviewed = _json(publication / "regime-results.json")
        if _first_difference(_protected(source), _protected(reviewed)):
            raise ReleasePreparationError("review promotion changed protected operating results")
        package_public_dashboard(web_root=ROOT / "web", payload_path=publication / "regime-results.json",
            output_directory=staging / "public-dashboard", publication_mode="live-derived", rights_acknowledged=True,
            comparison_path=publication / "v5-vs-v4-comparison.json", generation_manifest_path=publication / "generation-manifest.json",
            selection_family_path=publication / "selection-family-audit.json", staged_generation_contract_directory=ROOT / "publication/live")
        verify_public_package(staging / "public-dashboard")
        verify_artifact_inventory(source_artifacts)
        if before != {name: _sha256(path) for name, path in tracked.items()}:
            raise ReleasePreparationError("read-only forecast release inputs changed during preparation")
        shutil.rmtree(private)
        if output_root.exists() or output_root.is_symlink():
            raise ReleasePreparationError("forecast release destination appeared during preparation")
        os.replace(staging, output_root)
        installed = True
        final = output_root / "publication/live"
        validate_generation_manifest(final / "generation-manifest.json", require_comparison=True, require_selection_family=True,
            artifact_directory=output_root / "artifacts", payload_path_override=final / "regime-results.json",
            comparison_path_override=final / "v5-vs-v4-comparison.json", selection_family_path_override=final / "selection-family-audit.json")
        report = {"schema_version": "regime-forecast-release-preparation/1", "ok": True,
            "publication_action": "staged_only_no_commit_push_or_deploy", "data_as_of": source["meta"]["data_as_of"],
            "generation_id": source["meta"]["generation_id"], "reviewed_at": reviewed_at.isoformat(),
            "weekly_rows": len(source["weekly"]), "official_forecast_and_selection_preserved": True,
            "research_reproduced_from_frozen_inputs": True, "logical_publication_directory": "publication/live",
            "publication_members": {name: _sha256(final / name) for name in PUBLICATION_MEMBERS},
            "public_package": verify_public_package(output_root / "public-dashboard")}
        write_json_atomic(output_root / "release-preparation.json", report)
        return report
    except BaseException:
        shutil.rmtree(output_root if installed else staging, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "source-artifacts", "preview", "input-directory", "forecast-cache", "asset-cache", "output-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--reviewed-at")
    args = parser.parse_args()
    parameters = vars(args)
    parameters["source_path"] = parameters.pop("source")
    parameters["preview_path"] = parameters.pop("preview")
    parameters["reviewed_at"] = datetime.fromisoformat(args.reviewed_at.replace("Z", "+00:00")) if args.reviewed_at else datetime.now(timezone.utc)
    print(json.dumps(prepare_forecast_release(**parameters), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
