#!/usr/bin/env python3
"""Reproduce forecast audit blocks and stage a reviewed, same-generation release.

The issued forecasts, model selection, labels, source artifacts and source
ledger are read-only. This command never installs publication/live or uses Git.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
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
from regime_lab.integrity import (
    bind_payload_to_generation_manifest, build_generation_manifest,
    canonical_json_sha256_v1, reviewed_candidate_payload, validate_generation_manifest,
)
from regime_lab.io import write_json_atomic
from regime_lab.operational_forecast import frame_sha256
from regime_lab.path_safety import confined_mutable_path
from regime_lab.publication_contract import reject_raw_provider_material
from regime_lab.research.contract import validate_research_extensions
from regime_lab.schema import validate_dashboard_payload
from regime_lab.v5_artifacts import verify_staged_v5_research_artifacts
from scripts.evaluate_regime_forecasts import evaluate_copy
from scripts.package_public_demo import STATIC_ALLOWLIST, package_public_dashboard, validate_public_live_derived_payload
from scripts.prepare_comprehensive_release import PUBLICATION_MEMBERS, ReleasePreparationError, _first_difference, _json, _sha256
from scripts.promote_v5_publication import _build_expected_comparison, promote
from scripts.verify_public_package import verify_public_package

BLOCKS = ("forecast_research", "calibration_audit", "forecast_information")
BUILD_RECEIPTS = ("forecast_audit_present", "forecast_audit", "forecast_audit_operational")
OUTPUT_RECEIPTS = (*BLOCKS, "operational_diagnostics")


def _protected(payload: dict) -> dict:
    """Remove only explicitly authorized research and review bookkeeping."""
    value = deepcopy(payload)
    for name in ("publication_status", "publication_review", "generation_manifest_sha256"):
        value["meta"].pop(name, None)
    for name in ("publication", "deployment"):
        value["model"]["lifecycle"].pop(name, None)
    research = value["research"]
    for name in BLOCKS:
        research.pop(name, None)
    operational = research.get("operational_diagnostics", {})
    for name in ("as_of", "probability_scores"):
        operational.pop(name, None)
    # Actual receipt/publication clocks replace the previous decision-only lead
    # estimate. Investment results and original forecast/evaluation hashes stay.
    timing = operational.get("timing", {})
    timing.pop("median_lead_seconds", None)
    for row in timing.get("rows", []):
        for name in ("lead_seconds", "issued_at", "issue_evidence", "on_time"):
            row.pop(name, None)
    for name in ("on_time_entries", "on_time_rate"):
        timing.pop(name, None)
    build = research.get("extensions", {}).get("build", {})
    for name in ("generation_id", "data_as_of"):
        if build.get(name) == value["meta"].get(name):
            build.pop(name, None)
    for name in BUILD_RECEIPTS:
        build.pop(name, None)
    outputs = build.get("outputs", {})
    for name in OUTPUT_RECEIPTS:
        outputs.pop(name, None)
    if not outputs:
        build.pop("outputs", None)
    return value


def validate_forecast_audit_only_update(source: dict, candidate: dict) -> None:
    if source["meta"].get("publication_status") != "reviewed_publication":
        raise ReleasePreparationError("source must be a reviewed publication")
    if candidate["meta"].get("publication_status") != "unpublished":
        raise ReleasePreparationError("forecast audit candidate must be unpublished")
    if any(name in candidate["meta"] for name in ("publication_review", "generation_manifest_sha256")):
        raise ReleasePreparationError("forecast audit candidate inherited a previous review")
    if not set(BLOCKS) <= set(candidate.get("research", {})):
        raise ReleasePreparationError("forecast audit candidate requires all three research blocks")
    build = candidate["research"].get("extensions", {}).get("build", {})
    original_build = source["research"].get("extensions", {}).get("build", {})
    for name in ("generation_id", "data_as_of"):
        if name in original_build and build.get(name) != original_build[name]:
            raise ReleasePreparationError(f"forecast audit build receipt changed protected {name}")
        if name in build and build[name] != source["meta"][name]:
            raise ReleasePreparationError(f"forecast audit build receipt changed protected {name}")
    difference = _first_difference(_protected(source), _protected(candidate))
    if difference:
        raise ReleasePreparationError(f"forecast audit release changed protected content: {difference}")


def _evidence_view(block: dict, name: str) -> dict:
    """Preview and production may differ in declared provenance, never results."""
    value = deepcopy(block)
    for key in ("generated_at", "artifacts", "publication_provenance"):
        value.pop(key, None)
    if name == "forecast_research":
        for key in ("source_hashes", "source_context"):
            value.pop(key, None)
    elif name == "calibration_audit":
        value.pop("sources", None)
    elif name == "forecast_information":
        # The normal producer adds availability receipts absent in the earlier
        # local preview. The complete original table and notes still compare.
        for key in ("sources", "blocks"):
            value.pop(key, None)
    elif name == "operational_diagnostics":
        value.pop("as_of", None)
        value.get("probability_scores", {}).pop("evaluation_manifest_sha256", None)
    return value


def validate_reproduced_preview(preview: dict, blocks: dict, operational: dict) -> dict:
    evidence = {**blocks, "operational_diagnostics": operational}
    report = {}
    for name, block in evidence.items():
        if name not in preview.get("research", {}):
            raise ReleasePreparationError(f"reviewed preview is missing {name}")
        expected = _evidence_view(preview["research"][name], name)
        actual = _evidence_view(block, name)
        difference = _first_difference(expected, actual)
        if difference:
            raise ReleasePreparationError(f"forecast preview differs from reproduced {name}: {difference}")
        report[name] = {"results_exact": True, "evidence_sha256": canonical_json_sha256_v1(actual)}
    return report


def _require_destination(output: Path, inputs: list[Path]) -> Path:
    root = project_root()
    output = confined_mutable_path(output, project_directory=root, label="forecast audit release candidate")
    if not output.is_relative_to((root / "build").resolve()):
        raise ReleasePreparationError("forecast audit release candidate must stay below build/")
    if output.exists() or output.is_symlink():
        raise ReleasePreparationError("forecast audit release candidate must not exist")
    for item in inputs:
        source = item.resolve()
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ReleasePreparationError("forecast audit release candidate overlaps a read-only input or cache")
    return output


def _frozen_inputs(source: dict, directory: Path) -> tuple[pd.DataFrame, pd.Series]:
    manifest = _json(directory / "input-manifest.json")
    canonical = pd.read_pickle(directory / "canonical.pkl")
    states = pd.read_pickle(directory / "states.pkl")
    for name, frame in (("canonical", canonical), ("states", states)):
        if manifest.get("frames", {}).get(name) != frame_sha256(frame):
            raise ReleasePreparationError(f"frozen forecast {name} hash differs")
    cutoff = pd.Timestamp(source["meta"]["data_as_of"])
    if (manifest.get("data_as_of") != source["meta"]["data_as_of"] or states.empty
            or not canonical.index.equals(states.index) or states.index.has_duplicates
            or not states.index.is_monotonic_increasing or states.index[-1] != cutoff):
        raise ReleasePreparationError("forecast input cutoff or history differs from reviewed source")
    for week in source["weekly"]:
        origin = pd.Timestamp(week["data_as_of"])
        if origin not in states.index or states.loc[origin] != week["current"]["state"]:
            raise ReleasePreparationError("forecast input labels differ from issued observations")
    return canonical, states


def _check_unchanged(tracked: dict[str, Path], before: dict[str, str]) -> None:
    if before != {name: _sha256(path) for name, path in tracked.items()}:
        raise ReleasePreparationError("read-only forecast audit inputs changed during preparation")


def prepare_forecast_audit_release(
    *, source_path: Path, source_artifacts: Path, preview_path: Path,
    input_directory: Path, ledger_path: Path, research_cache: Path,
    existing_sources: Path, information_sources: Path, output_root: Path,
    reviewed_at: datetime | None = None, progress=None,
) -> dict:
    """Prepare and verify a separate reviewed package; keep all inputs intact."""
    if reviewed_at is not None and (reviewed_at.tzinfo is None or reviewed_at.utcoffset() is None):
        raise ReleasePreparationError("forecast audit review time must include a timezone")
    read_only = [source_path.parent, source_artifacts, preview_path, input_directory,
                 ledger_path, existing_sources, information_sources]
    output_root = _require_destination(output_root, [*read_only, research_cache])
    research_cache = confined_mutable_path(research_cache, project_directory=project_root(), label="forecast audit research cache")
    if not research_cache.is_relative_to((project_root() / "build").resolve()):
        raise ReleasePreparationError("forecast audit research cache must stay below build/")
    for path in read_only:
        if (research_cache == path.resolve() or research_cache.is_relative_to(path.resolve())
                or path.resolve().is_relative_to(research_cache)):
            raise ReleasePreparationError("forecast audit research cache overlaps a read-only input")
    tracked = {f"source:{name}": source_path.parent / name for name in PUBLICATION_MEMBERS}
    tracked.update({f"input:{name}": input_directory / name for name in ("canonical.pkl", "states.pkl", "input-manifest.json")})
    tracked.update({"preview": preview_path, "ledger": ledger_path})
    tracked.update({f"artifact:{path.name}": path for path in source_artifacts.iterdir() if path.is_file()})
    for name, directory in (("existing-source", existing_sources), ("information-source", information_sources)):
        tracked.update({f"{name}:{path.relative_to(directory)}": path for path in directory.rglob("*") if path.is_file()})
    tracked.update({f"web:{name}": ROOT / "web" / name for name in STATIC_ALLOWLIST})
    before = {name: _sha256(path) for name, path in tracked.items()}
    generation = validate_generation_manifest(source_path.with_name("generation-manifest.json"),
        require_comparison=True, require_selection_family=True, artifact_directory=source_artifacts,
        payload_path_override=source_path, comparison_path_override=source_path.with_name("v5-vs-v4-comparison.json"),
        selection_family_path_override=source_path.with_name("selection-family-audit.json"))
    source = generation["payload"]
    validate_public_live_derived_payload(source)
    preview = _json(preview_path)
    validate_dashboard_payload(preview)
    validate_forecast_audit_only_update(source, preview)
    canonical, states = _frozen_inputs(source, input_directory)
    from regime_lab.research.forecast_publication import build_forecast_publication_research
    reproduced = build_forecast_publication_research(source, canonical, states,
        pd.read_csv(source_artifacts / "oos-predictions.csv"),
        pd.read_csv(source_artifacts / "transition-oos-predictions.csv"),
        pd.read_csv(source_artifacts / "transition-candidate-forecasts.csv"), research_cache,
        existing_sources=existing_sources, information_sources=information_sources, offline=True, progress=progress)
    blocks, provenance = reproduced["blocks"], reproduced["provenance"]
    if set(blocks) != set(BLOCKS):
        raise ReleasePreparationError("normal production wrapper did not return all three research blocks")
    validate_research_extensions(blocks, data_as_of=source["meta"]["data_as_of"])
    reject_raw_provider_material(blocks)
    _check_unchanged(tracked, before)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent))
    installed = False
    try:
        if progress:
            progress("실제 발행 원장의 읽기 전용 복사본에서 확률 평가")
        evaluation = evaluate_copy(ledger_path, input_directory, source_path, staging / "operational-evaluation")
        operational = _json(staging / "operational-evaluation/operational-diagnostics.json")
        preview_parity = validate_reproduced_preview(preview, blocks, operational)
        candidate = reviewed_candidate_payload(source)
        candidate["research"].update(blocks)
        candidate["research"]["operational_diagnostics"] = operational
        build = candidate["research"]["extensions"]["build"]
        for name in ("generation_id", "data_as_of"):
            build.setdefault(name, source["meta"][name])
        build["forecast_audit_present"] = True
        build["forecast_audit"] = provenance
        build["forecast_audit_operational"] = {
            "scope": "isolated_copy_of_actual_issued_ledger", "as_of": operational["as_of"],
            "source_ledger_sha256": before["ledger"],
            "source_payload_sha256": before["source:regime-results.json"],
            "source_label_spec_sha256": source["label"]["spec_sha256"],
            "issued_forecasts_unchanged": evaluation["issued_forecasts_unchanged"],
            "investment_evaluations_unchanged": evaluation["investment_evaluations_unchanged"],
        }
        build.setdefault("outputs", {}).update({name: canonical_json_sha256_v1(block)
            for name, block in {**blocks, "operational_diagnostics": operational}.items()})
        validate_forecast_audit_only_update(source, candidate)
        validate_dashboard_payload(candidate)
        reject_raw_provider_material(candidate)
        artifacts = staging / "artifacts"
        shutil.copytree(source_artifacts, artifacts)
        for name, block in blocks.items():
            write_json_atomic(artifacts / f"forecast-audit-{name}.json", block)
        write_json_atomic(artifacts / "forecast-audit-operational-diagnostics.json", operational)
        write_json_atomic(artifacts / "forecast-audit-build.json", provenance)
        write_json_atomic(artifacts / "forecast-audit-release-inputs.json", {
            "schema_version": "regime-forecast-audit-release-inputs/1",
            "generation_id": source["meta"]["generation_id"], "data_as_of": source["meta"]["data_as_of"],
            "source_payload_sha256": before["source:regime-results.json"],
            "read_only_inputs_sha256": before, "preview_results_parity": preview_parity,
            "official_forecast_and_selection_preserved": True, "normal_production_recipe": provenance,
        })
        write_artifact_inventory(artifacts)
        verify_staged_v5_research_artifacts(candidate["model"]["research_artifacts"], artifacts)
        private = staging / "candidate"
        private.mkdir()
        payload_path = private / "regime-results.json"
        manifest_path = private / "generation-manifest.json"
        manifest = build_generation_manifest(payload=candidate, payload_path=payload_path, artifact_directory=artifacts,
            input_snapshot=generation["input_snapshot"], label_spec_path=ROOT / generation["label_spec"]["path"],
            selection_family=generation["selection_family"], selection_family_path=artifacts / "selection-family-audit.json")
        candidate = bind_payload_to_generation_manifest(candidate, manifest)
        write_json_atomic(payload_path, candidate)
        write_json_atomic(manifest_path, manifest)
        if progress:
            progress("기존 독립 V5 감사 및 선택군·비교 결과 재검증")
        from scripts.audit_outputs import audit
        independent_audit = audit(payload_path, artifacts, expected_mode="live")
        write_json_atomic(staging / "independent-v5-audit.json", {
            **independent_audit, "payload": "reviewed_candidate_before_promotion",
            "payload_sha256": _sha256(payload_path), "artifacts": "artifacts",
        })
        comparison_path = private / "v5-vs-v4-comparison.json"
        write_json_atomic(comparison_path, _build_expected_comparison(v5_artifacts=artifacts, candidate_path=payload_path))
        publication = staging / "publication/live"
        review_clock = reviewed_at or datetime.now(timezone.utc)
        promote(candidate_path=payload_path, v5_artifacts=artifacts, comparison_path=comparison_path,
            generation_manifest_path=manifest_path, output_path=publication / "regime-results.json",
            publication_contract_directory=ROOT / "publication/live", reviewed_at=review_clock)
        reviewed = _json(publication / "regime-results.json")
        difference = _first_difference(_protected(source), _protected(reviewed))
        if difference:
            raise ReleasePreparationError(f"review promotion changed protected operating results: {difference}")
        if _json(publication / "selection-family-audit.json") != generation["selection_family"]:
            raise ReleasePreparationError("review promotion changed the independently verified selection family")
        package_public_dashboard(web_root=ROOT / "web", payload_path=publication / "regime-results.json",
            output_directory=staging / "public-dashboard", publication_mode="live-derived", rights_acknowledged=True,
            comparison_path=publication / "v5-vs-v4-comparison.json", generation_manifest_path=publication / "generation-manifest.json",
            selection_family_path=publication / "selection-family-audit.json", staged_generation_contract_directory=ROOT / "publication/live")
        verify_public_package(staging / "public-dashboard")
        verify_artifact_inventory(source_artifacts)
        _check_unchanged(tracked, before)
        shutil.rmtree(private)
        if output_root.exists() or output_root.is_symlink():
            raise ReleasePreparationError("forecast audit destination appeared during preparation")
        os.replace(staging, output_root)
        installed = True
        final = output_root / "publication/live"
        validate_generation_manifest(final / "generation-manifest.json", require_comparison=True, require_selection_family=True,
            artifact_directory=output_root / "artifacts", payload_path_override=final / "regime-results.json",
            comparison_path_override=final / "v5-vs-v4-comparison.json", selection_family_path_override=final / "selection-family-audit.json")
        report = {
            "schema_version": "regime-forecast-audit-release-preparation/1", "ok": True,
            "publication_action": "staged_only_no_commit_push_or_deploy", "data_as_of": source["meta"]["data_as_of"],
            "generation_id": source["meta"]["generation_id"], "reviewed_at": review_clock.isoformat(),
            "weekly_rows": len(source["weekly"]), "official_forecast_and_selection_preserved": True,
            "issued_forecasts_unchanged": True, "source_ledger_unchanged": True,
            "research_reproduced_with_normal_production_wrapper": True,
            "preview_results_parity": preview_parity, "independent_v5_audit_ok": independent_audit["ok"],
            "logical_publication_directory": "publication/live", "research_provenance": provenance,
            "source_publication_members": {name: before[f"source:{name}"] for name in PUBLICATION_MEMBERS},
            "publication_members": {name: _sha256(final / name) for name in PUBLICATION_MEMBERS},
            "public_package": verify_public_package(output_root / "public-dashboard"),
        }
        write_json_atomic(output_root / "release-preparation.json", report)
        return report
    except BaseException:
        shutil.rmtree(output_root if installed else staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "source-artifacts", "preview", "input-directory", "ledger", "research-cache",
                 "existing-sources", "information-sources", "output-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--reviewed-at")
    args = parser.parse_args()
    parameters = vars(args)
    for name in ("source", "preview", "ledger"):
        parameters[f"{name}_path"] = parameters.pop(name)
    parameters["reviewed_at"] = datetime.fromisoformat(args.reviewed_at.replace("Z", "+00:00")) if args.reviewed_at else None
    parameters["progress"] = lambda message: print(message, file=sys.stderr, flush=True)
    print(json.dumps(prepare_forecast_audit_release(**parameters), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
