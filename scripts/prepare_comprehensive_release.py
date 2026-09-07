#!/usr/bin/env python3
"""Stage a fully reviewed Pages bundle without replacing publication/live.

This is a research-only release: the issued forecast, original model selection,
and original ledger-derived decisions remain bound to their reviewed source.
The existing V5 promotion, frozen-V4 comparison, selection-family audit and
public package verifier remain authoritative.
"""

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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import pandas as pd

from regime_lab.analysis.directional_coherence import apply_full_directional_research
from regime_lab.analysis.duration import duration_context
from regime_lab.artifact_inventory import (
    verify_artifact_inventory,
    write_artifact_inventory,
)
from regime_lab.config import project_root
from regime_lab.integrity import (
    bind_payload_to_generation_manifest,
    build_generation_manifest,
    reviewed_candidate_payload,
    validate_generation_manifest,
)
from regime_lab.io import write_json_atomic
from regime_lab.operational_forecast import frame_sha256
from regime_lab.path_safety import confined_mutable_path
from regime_lab.research.additional_sources import source_summary
from regime_lab.research.diagnostics import ablation_diagnostics
from regime_lab.schema import validate_dashboard_payload
from regime_lab.v5_artifacts import (
    V5_RESEARCH_ARTIFACTS,
    canonical_v5_artifact_csv_bytes,
    verify_staged_v5_research_artifacts,
)
from scripts.package_public_demo import (
    package_public_dashboard,
    validate_public_live_derived_payload,
)
from scripts.promote_v5_publication import _build_expected_comparison, promote
from scripts.verify_public_package import verify_public_package

DIRECTIONAL_FRAMES = {
    "directional_oos_predictions": "predictions",
    "directional_model_leaderboard": "leaderboard",
    "directional_walk_forward_splits": "split_audit",
    "directional_selection_diagnostics": "selection_diagnostics",
    "directional_forecasts": "latest_forecasts",
}
RESEARCH_DOCUMENTS = {
    "research/allocation-v2.json": (
        "research",
        "prospective_decision_shadow",
        "allocation_research_v2",
    ),
    "research/decision-research-v2.json": ("research", "decision_research_v2"),
    "research/operational-diagnostics.json": ("research", "operational_diagnostics"),
    "model-economics/label-sensitivity.json": ("research", "label_sensitivity"),
    "downside.json": ("research", "extensions", "downside"),
    "diagnostics.json": ("research", "extensions", "diagnostics"),
}
PUBLICATION_MEMBERS = (
    "regime-results.json",
    "v5-vs-v4-comparison.json",
    "generation-manifest.json",
    "selection-family-audit.json",
)


class ReleasePreparationError(ValueError):
    """A release input or a preserved operating contract is inconsistent."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReleasePreparationError(f"expected a regular input file: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReleasePreparationError(f"expected a JSON object: {path.name}")
    return value


def _first_difference(left: Any, right: Any, path: str = "") -> str | None:
    if type(left) is not type(right):
        return path or "/"
    if isinstance(left, dict):
        for key in sorted(left.keys() | right.keys()):
            child = f"{path}/{key}"
            if key not in left or key not in right:
                return child
            changed = _first_difference(left[key], right[key], child)
            if changed:
                return changed
    elif isinstance(left, list):
        if len(left) != len(right):
            return path
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            changed = _first_difference(a, b, f"{path}/{index}")
            if changed:
                return changed
    elif left != right:
        return path
    return None


def _protected_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove only the explicitly approved research surfaces before comparison."""
    protected = deepcopy(payload)
    for name in (
        "publication_status",
        "publication_review",
        "generation_manifest_sha256",
    ):
        protected["meta"].pop(name, None)
    model = protected["model"]
    for name in ("deployment", "publication"):
        model["lifecycle"].pop(name, None)
    model.pop("directional_transition", None)
    for name in (
        "directional_maximum_selection_origins",
        "directional_maximum_diagnostic_origins",
        "sha256",
    ):
        model["execution_parameters"].pop(name, None)
    for name in DIRECTIONAL_FRAMES:
        reference = model["research_artifacts"][name]
        reference.pop("sha256", None)
        reference.pop("row_count", None)
    for week in protected["weekly"]:
        for name in ("directional_risk", "directional_risk_raw", "duration_context"):
            week.pop(name, None)
    research = protected["research"]
    for name in (
        "extensions",
        "decision_research_v2",
        "operational_diagnostics",
        "label_sensitivity",
    ):
        research.pop(name, None)
    research["prospective_decision_shadow"].pop("allocation_research_v2", None)
    return protected


def validate_research_only_update(
    source: dict[str, Any], preview: dict[str, Any]
) -> None:
    """Reject every change outside the approved research-only allowlist."""
    if source["meta"].get("publication_status") != "reviewed_publication":
        raise ReleasePreparationError("source must be a reviewed publication")
    if preview["meta"].get("publication_status") != "unpublished":
        raise ReleasePreparationError("preview must be an unpublished candidate")
    if (
        "publication_review" in preview["meta"]
        or "generation_manifest_sha256" in preview["meta"]
    ):
        raise ReleasePreparationError(
            "preview must not inherit a source review or binding"
        )
    difference = _first_difference(
        _protected_payload(source), _protected_payload(preview)
    )
    if difference:
        raise ReleasePreparationError(
            f"research release changed protected content: {difference}"
        )


def validate_decision_run(
    run: dict[str, Any],
    *,
    paths: dict[str, Path],
    source: dict[str, Any],
    research_root: Path,
) -> None:
    """Bind both the actual original inputs and all three decision outputs."""
    if run.get("schema_version") != "regime-decision-research-run/1":
        raise ReleasePreparationError("unsupported decision research run manifest")
    for key, path in paths.items():
        if run.get("source_inputs", {}).get(key, {}).get("sha256") != _sha256(path):
            raise ReleasePreparationError(f"decision research input mismatch: {key}")
    if run.get("origin") != source["weekly"][-1]["date"]:
        raise ReleasePreparationError("decision research has a different final origin")
    if run.get("forecast_model") != source["model"]["champion"]:
        raise ReleasePreparationError(
            "decision research uses a different forecast model"
        )
    if run.get("selection_end") != source["model"]["selection_end"]:
        raise ReleasePreparationError("decision research changed the selection cutoff")
    for name in (
        "allocation-v2.json",
        "decision-research-v2.json",
        "operational-diagnostics.json",
    ):
        raw = (research_root / "research" / name).read_bytes()
        if run.get("outputs", {}).get(name) != {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }:
            raise ReleasePreparationError(f"decision research output mismatch: {name}")


def validate_directional_coverage(
    predictions: pd.DataFrame,
    states: pd.Series,
    *,
    selection_end: str,
    minimum_train_weeks: int = 520,
) -> None:
    """Require every purged origin for all four fixed research candidates."""
    models = {
        "empirical_first_passage",
        "markov_first_passage",
        "regularized_multinomial",
        "shallow_multiclass_xgboost",
    }
    cutoff = pd.Timestamp(selection_end)
    if states.index.tz is not None:
        cutoff = cutoff.tz_localize(states.index.tz)
    expected = set()
    outcomes = {}
    for horizon in (1, 4, 13):
        for position in range(minimum_train_weeks + horizon, len(states) - horizon):
            origin = states.index[position]
            target = states.index[position + horizon]
            if target < cutoff:
                split = "selection"
            elif origin >= cutoff:
                split = "retrospective_diagnostic"
            else:
                continue
            outcome = next(
                (
                    str(state)
                    for state in states.iloc[position + 1 : position + horizon + 1]
                    if state != states.iloc[position]
                ),
                "no_departure",
            )
            outcomes[(horizon, origin)] = (
                target,
                split,
                str(states.iloc[position]),
                outcome,
            )
            expected.update((horizon, model, origin) for model in models)
    actual = set()
    for row in predictions.itertuples(index=False):
        origin = pd.Timestamp(row.origin_date)
        key = (int(row.horizon_weeks), str(row.model), origin)
        if key in actual:
            raise ReleasePreparationError(
                "directional research contains duplicate origins"
            )
        actual.add(key)
        evidence = (
            pd.Timestamp(row.target_end),
            str(row.evaluation_split),
            str(row.current_state),
            str(row.actual_outcome),
        )
        if outcomes.get((int(row.horizon_weeks), origin)) != evidence:
            raise ReleasePreparationError(
                "directional research target or purged split differs from source"
            )
    if actual != expected:
        raise ReleasePreparationError(
            "directional research does not cover every full evaluation origin"
        )


def _research_input_paths(
    *,
    source_path: Path,
    source_manifest_path: Path,
    source_artifacts: Path,
    ledger_path: Path,
    preview_path: Path,
    research_root: Path,
) -> dict[str, Path]:
    names = (
        set(RESEARCH_DOCUMENTS)
        | {
            "input/input-manifest.json",
            "input/canonical.pkl",
            "input/features.pkl",
            "input/states.pkl",
            "research/decision-research-run.json",
            "model-economics/directional-execution.json",
            "model-economics/directional-result.pkl",
        }
        | {
            f"model-economics/{V5_RESEARCH_ARTIFACTS[key].path}"
            for key in DIRECTIONAL_FRAMES
        }
    )
    paths = {name: research_root / name for name in names}
    paths.update(
        {
            "source-manifest": source_manifest_path,
            "preview": preview_path,
            "ledger": ledger_path,
            "source-inventory": source_artifacts / "SHA256SUMS",
        }
    )
    paths.update(
        {
            f"source-publication:{name}": source_path.with_name(name)
            for name in PUBLICATION_MEMBERS
        }
    )
    paths.update(
        {
            str(path.relative_to(research_root)): path
            for path in (research_root / "additional-sources").rglob("*")
            if path.is_file()
        }
    )
    return paths


def _validate_research_inputs(
    *,
    source_path: Path,
    source_artifacts: Path,
    ledger_path: Path,
    source: dict[str, Any],
    preview: dict[str, Any],
    research_root: Path,
) -> dict[str, Path]:
    files: dict[str, Path] = {}

    def read(name: str) -> dict[str, Any]:
        files[name] = research_root / name
        return _json(files[name])

    inputs = read("input/input-manifest.json")
    source_sha = _sha256(source_path)
    if (
        inputs.get("source_payload_sha256") != source_sha
        or inputs.get("data_as_of") != source["meta"]["data_as_of"]
    ):
        raise ReleasePreparationError(
            "research cache belongs to a different source snapshot"
        )
    frames = {}
    for name in ("canonical", "states", "features"):
        files[f"input/{name}.pkl"] = research_root / f"input/{name}.pkl"
        frames[name] = pd.read_pickle(files[f"input/{name}.pkl"])
        if frame_sha256(frames[name]) != inputs.get("frames", {}).get(name):
            raise ReleasePreparationError(f"research cache checksum mismatch: {name}")
        if frames[name].index[-1] != pd.Timestamp(source["meta"]["data_as_of"]):
            raise ReleasePreparationError(f"research cache cutoff mismatch: {name}")

    extensions = preview["research"]["extensions"]
    build = extensions["build"]
    if (
        build.get("source_payload_sha256") != source_sha
        or build.get("issued_forecast_preserved") is not True
        or build.get("full_directional_evaluated") is not True
    ):
        raise ReleasePreparationError(
            "preview lacks complete source-bound research evidence"
        )
    if extensions["additional_data"] != source_summary(
        research_root / "additional-sources"
    ):
        raise ReleasePreparationError(
            "additional research source summary differs from preview"
        )
    for relative, key_path in RESEARCH_DOCUMENTS.items():
        expected = read(relative)
        actual = preview
        for key in key_path:
            actual = actual[key]
        if actual != expected:
            raise ReleasePreparationError(
                f"preview differs from research output: {relative}"
            )

    if pd.Timestamp(extensions["downside"]["as_of"]) != pd.Timestamp(
        source["meta"]["data_as_of"]
    ):
        raise ReleasePreparationError("downside research has a different input cutoff")
    if extensions["downside"]["selection_end"] != source["model"]["selection_end"]:
        raise ReleasePreparationError("downside research changed the selection cutoff")
    diagnostics = ablation_diagnostics(
        pd.read_csv(source_artifacts / "feature-ablation-oos-predictions.csv")
    )
    if diagnostics != extensions["diagnostics"]:
        raise ReleasePreparationError(
            "ablation diagnostics differ from original OOS evidence"
        )
    label_execution = preview["research"]["label_sensitivity"]["execution"]
    cutoff = pd.Timestamp(source["model"]["selection_end"])
    price = pd.to_numeric(frames["canonical"]["spy_close"]).astype(float)
    if price.index.tz is not None:
        cutoff = cutoff.tz_localize(price.index.tz)
    price = price.loc[price.index < cutoff]
    control = frames["states"].reindex(price.index)
    for name, frame in (
        ("source_price_sha256", price),
        ("control_state_sha256", control),
    ):
        actual_hash = hashlib.sha256(
            pd.util.hash_pandas_object(frame).values.tobytes()
        ).hexdigest()
        if label_execution.get(name) != actual_hash:
            raise ReleasePreparationError(f"label research input mismatch: {name}")
    for week in preview["weekly"]:
        expected_duration = duration_context(
            frames["states"],
            as_of=week.get("data_as_of", week["date"]),
            bootstrap_resamples=(
                1999 if week["date"] == preview["weekly"][-1]["date"] else 0
            ),
        )
        if week["duration_context"] != expected_duration:
            raise ReleasePreparationError(
                f"duration research cannot be reproduced: {week['date']}"
            )

    decision_paths = {
        "canonical_cache": files["input/canonical.pkl"],
        "source_payload": source_path,
        "ledger": ledger_path,
        "oos_predictions": source_artifacts / "oos-predictions.csv",
        "transition_predictions": source_artifacts / "transition-oos-predictions.csv",
        "outcome_rows": source_artifacts / "model-conditioned-asset-outcomes.csv",
    }
    validate_decision_run(
        read("research/decision-research-run.json"),
        paths=decision_paths,
        source=source,
        research_root=research_root,
    )
    files.update({f"source:{key}": path for key, path in decision_paths.items()})
    execution = read("model-economics/directional-execution.json")
    if (
        execution.get("schema_version") != "full-directional-research/2"
        or not {"maximum_selection_origins", "maximum_diagnostic_origins"}.issubset(
            execution
        )
        or execution.get("maximum_selection_origins") is not None
        or execution.get("maximum_diagnostic_origins") is not None
        or execution.get("profile") != "standard"
        or execution.get("automatic_promotion_eligible") is not False
    ):
        raise ReleasePreparationError(
            "directional research is not the full standard evaluation"
        )
    state_sha = hashlib.sha256(
        pd.util.hash_pandas_object(frames["states"], index=True).values.tobytes()
    ).hexdigest()
    if execution.get("state_frame_sha256") != state_sha:
        raise ReleasePreparationError("directional research state snapshot mismatch")
    files["model-economics/directional-result.pkl"] = (
        research_root / "model-economics/directional-result.pkl"
    )
    benchmark = pd.read_pickle(files["model-economics/directional-result.pkl"])
    if execution.get("prediction_count") != len(benchmark.predictions):
        raise ReleasePreparationError("directional execution prediction count mismatch")
    validate_directional_coverage(
        benchmark.predictions,
        frames["states"],
        selection_end=source["model"]["selection_end"],
    )
    for key, attribute in DIRECTIONAL_FRAMES.items():
        name = f"model-economics/{V5_RESEARCH_ARTIFACTS[key].path}"
        files[name] = research_root / name
        if files[name].read_bytes() != canonical_v5_artifact_csv_bytes(
            key, getattr(benchmark, attribute)
        ):
            raise ReleasePreparationError(
                f"directional CSV differs from completed benchmark: {key}"
            )
    reproduced = apply_full_directional_research(preview, frames["states"], benchmark)
    if reproduced != preview:
        raise ReleasePreparationError(
            f"directional preview cannot be reproduced: {_first_difference(preview, reproduced)}"
        )
    return files


def prepare_release(
    *,
    source_path: Path,
    source_manifest_path: Path,
    source_artifacts: Path,
    ledger_path: Path,
    preview_path: Path,
    research_root: Path,
    output_root: Path,
    reviewed_at: datetime,
) -> dict[str, Any]:
    root = project_root()
    output_root = confined_mutable_path(
        output_root, project_directory=root, label="research release candidate"
    )
    if not output_root.is_relative_to((root / "build").resolve()):
        raise ReleasePreparationError("release candidate must stay below build/")
    if output_root.exists() or output_root.is_symlink():
        raise ReleasePreparationError("release candidate directory must not exist")
    # research_root may be an ancestor; only concrete read-only members are protected.
    inputs = [
        source_path,
        source_manifest_path,
        source_artifacts,
        ledger_path,
        preview_path,
    ]
    inputs.extend(
        research_root / name
        for name in ("input", "research", "model-economics", "additional-sources")
    )
    for item in inputs:
        resolved = item.resolve()
        if (
            output_root == resolved
            or output_root.is_relative_to(resolved)
            or resolved.is_relative_to(output_root)
        ):
            raise ReleasePreparationError(
                "release candidate overlaps a read-only input"
            )
    if reviewed_at.tzinfo is None or reviewed_at.utcoffset() is None:
        raise ReleasePreparationError("review time must include a timezone")

    tracked = _research_input_paths(
        source_path=source_path,
        source_manifest_path=source_manifest_path,
        source_artifacts=source_artifacts,
        ledger_path=ledger_path,
        preview_path=preview_path,
        research_root=research_root,
    )
    before = {name: _sha256(path) for name, path in tracked.items()}
    source_generation = validate_generation_manifest(
        source_manifest_path,
        require_comparison=True,
        require_selection_family=True,
        artifact_directory=source_artifacts,
        payload_path_override=source_path,
        comparison_path_override=source_path.with_name("v5-vs-v4-comparison.json"),
        selection_family_path_override=source_path.with_name(
            "selection-family-audit.json"
        ),
    )
    source = source_generation["payload"]
    validate_public_live_derived_payload(source)
    preview = _json(preview_path)
    validate_dashboard_payload(preview)
    validate_research_only_update(source, preview)
    _validate_research_inputs(
        source_path=source_path,
        source_artifacts=source_artifacts,
        ledger_path=ledger_path,
        source=source,
        preview=preview,
        research_root=research_root,
    )
    if before != {name: _sha256(path) for name, path in tracked.items()}:
        raise ReleasePreparationError(
            "read-only release inputs changed during validation"
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent)
    )
    installed = False
    try:
        artifacts = staging / "artifacts"
        shutil.copytree(source_artifacts, artifacts)
        candidate = reviewed_candidate_payload(preview)
        for key in DIRECTIONAL_FRAMES:
            name = V5_RESEARCH_ARTIFACTS[key].path
            shutil.copyfile(research_root / "model-economics" / name, artifacts / name)
        # These are private evidence members; only their inventory digest is public.
        for name in RESEARCH_DOCUMENTS:
            shutil.copyfile(
                research_root / name, artifacts / f"comprehensive-{Path(name).name}"
            )
        for name in (
            "input/input-manifest.json",
            "research/decision-research-run.json",
            "model-economics/directional-execution.json",
        ):
            shutil.copyfile(
                research_root / name, artifacts / f"comprehensive-{Path(name).name}"
            )
        write_json_atomic(
            artifacts / "comprehensive-release-inputs.json",
            {
                "schema_version": "regime-comprehensive-release-inputs/1",
                "source_payload_sha256": _sha256(source_path),
                "source_generation_id": source["meta"]["generation_id"],
                "data_as_of": source["meta"]["data_as_of"],
                "read_only_inputs_sha256": before,
                "official_forecast_and_selection_preserved": True,
            },
        )
        write_artifact_inventory(artifacts)
        verify_staged_v5_research_artifacts(
            candidate["model"]["research_artifacts"], artifacts
        )
        private = staging / "candidate"
        private.mkdir()
        payload_path = private / "regime-results.json"
        manifest_path = private / "generation-manifest.json"
        manifest = build_generation_manifest(
            payload=candidate,
            payload_path=payload_path,
            artifact_directory=artifacts,
            input_snapshot=source_generation["input_snapshot"],
            label_spec_path=root / source_generation["label_spec"]["path"],
            selection_family=source_generation["selection_family"],
            selection_family_path=artifacts / "selection-family-audit.json",
        )
        candidate = bind_payload_to_generation_manifest(candidate, manifest)
        write_json_atomic(payload_path, candidate)
        write_json_atomic(manifest_path, manifest)
        comparison_path = private / "v5-vs-v4-comparison.json"
        write_json_atomic(
            comparison_path,
            _build_expected_comparison(
                v5_artifacts=artifacts, candidate_path=payload_path
            ),
        )
        publication = staging / "publication/live"
        promote(
            candidate_path=payload_path,
            v5_artifacts=artifacts,
            comparison_path=comparison_path,
            generation_manifest_path=manifest_path,
            output_path=publication / "regime-results.json",
            publication_contract_directory=root / "publication/live",
            reviewed_at=reviewed_at,
        )
        package_public_dashboard(
            web_root=root / "web",
            payload_path=publication / "regime-results.json",
            output_directory=staging / "public-dashboard",
            publication_mode="live-derived",
            rights_acknowledged=True,
            comparison_path=publication / "v5-vs-v4-comparison.json",
            generation_manifest_path=publication / "generation-manifest.json",
            selection_family_path=publication / "selection-family-audit.json",
            staged_generation_contract_directory=root / "publication/live",
        )
        verify_public_package(staging / "public-dashboard")
        verify_artifact_inventory(source_artifacts)
        if before != {name: _sha256(path) for name, path in tracked.items()}:
            raise ReleasePreparationError(
                "read-only release inputs changed during preparation"
            )
        # The review identity is reproducible from the reviewed payload. Remove
        # temporary candidate manifests so no retained contract names a staging path.
        shutil.rmtree(private)
        if output_root.exists() or output_root.is_symlink():
            raise ReleasePreparationError(
                "release candidate appeared during preparation"
            )
        os.replace(staging, output_root)
        installed = True
        final_publication = output_root / "publication/live"
        verified = verify_public_package(output_root / "public-dashboard")
        validate_generation_manifest(
            final_publication / "generation-manifest.json",
            require_comparison=True,
            require_selection_family=True,
            artifact_directory=output_root / "artifacts",
            payload_path_override=final_publication / "regime-results.json",
            comparison_path_override=final_publication / "v5-vs-v4-comparison.json",
            selection_family_path_override=final_publication
            / "selection-family-audit.json",
        )
        result = {
            "schema_version": "regime-comprehensive-release-preparation/1",
            "ok": True,
            "publication_action": "staged_only_no_commit_push_or_deploy",
            "generation_id": source["meta"]["generation_id"],
            "data_as_of": source["meta"]["data_as_of"],
            "weekly_rows": len(source["weekly"]),
            "reviewed_at": reviewed_at.isoformat(),
            "official_forecast_and_selection_preserved": True,
            "logical_publication_directory": "publication/live",
            "publication_members": {
                name: _sha256(final_publication / name) for name in PUBLICATION_MEMBERS
            },
            "public_package": verified,
        }
        write_json_atomic(output_root / "release-preparation.json", result)
        return result
    except BaseException:
        shutil.rmtree(output_root if installed else staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source",
        "source-manifest",
        "source-artifacts",
        "ledger",
        "preview",
        "research-root",
        "output-root",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--reviewed-at", help="Timezone-aware ISO time; defaults to current UTC"
    )
    args = parser.parse_args()
    result = prepare_release(
        source_path=args.source,
        source_manifest_path=args.source_manifest,
        source_artifacts=args.source_artifacts,
        ledger_path=args.ledger,
        preview_path=args.preview,
        research_root=args.research_root,
        output_root=args.output_root,
        reviewed_at=(
            datetime.fromisoformat(args.reviewed_at.replace("Z", "+00:00"))
            if args.reviewed_at
            else datetime.now(timezone.utc)
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
