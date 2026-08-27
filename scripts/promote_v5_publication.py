#!/usr/bin/env python3
"""Create a reviewed public V5 snapshot without changing model decisions."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping

from regime_lab.contract_v5 import (
    V5_MINIMUM_PROMOTION_LOG_LOSS_IMPROVEMENT,
    V5_MULTISCALE_MODEL,
    V5_PUBLICATION_REVIEW_SCHEMA,
    V5_PUBLICATION_STATUS,
    V5ContractError,
    validate_v5_champion_selection_evidence,
)
from regime_lab.config import project_root
from regime_lab.io import write_json_atomic
from regime_lab.integrity import (
    IntegrityError,
    bind_payload_to_generation_manifest,
    build_generation_manifest,
    reviewed_candidate_sha256_v1,
    validate_generation_manifest,
    validate_lifecycle_consistency,
    validate_reviewed_candidate_hash,
)
from regime_lab.publication_contract import (
    PublicContractError,
    validate_v5_comparison_sidecar,
)
from regime_lab.schema import ContractError, validate_dashboard_payload
from regime_lab.selection_family_audit import (
    build_selection_family_audit_from_artifacts,
    validate_selection_family_audit,
)


UTC = timezone.utc
COMPARISON_SCHEMA = "regime-v5-v4-matched-comparison/1"


class PromotionError(RuntimeError):
    """The candidate is not eligible for reviewed public V5 publication."""


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PromotionError(f"{label} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PromotionError(f"{label} must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise PromotionError(f"{label} must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _published_json_bytes(value: object) -> bytes:
    """Mirror ``write_json_atomic`` so raw sidecar bindings are precomputed."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=False,
        )
        + "\n"
    ).encode("utf-8")


def _comparison_builder_module() -> ModuleType:
    module_name = "_regime_v5_comparison_for_promotion"
    loaded = sys.modules.get(module_name)
    if isinstance(loaded, ModuleType):
        return loaded
    script = Path(__file__).with_name("compare_v5_to_frozen_v4.py")
    spec = importlib.util.spec_from_file_location(module_name, script)
    if spec is None or spec.loader is None:
        raise PromotionError("V5 comparison builder could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _build_expected_comparison(
    *,
    v5_artifacts: Path,
    candidate_path: Path,
) -> dict[str, Any]:
    module = _comparison_builder_module()
    try:
        report = module.build_comparison(
            v5_artifacts,
            v5_payload=candidate_path,
        )
    except Exception as exc:
        comparison_error = getattr(module, "ComparisonError", ())
        if comparison_error and isinstance(exc, comparison_error):
            raise PromotionError(
                f"candidate artifact comparison could not be independently reproduced: {exc}"
            ) from exc
        raise
    if not isinstance(report, dict):
        raise PromotionError("independently reproduced comparison is invalid")
    return report


def _validate_reproducible_comparison(
    supplied: Mapping[str, Any],
    *,
    v5_artifacts: Path,
    candidate_path: Path,
) -> None:
    expected = _build_expected_comparison(
        v5_artifacts=v5_artifacts,
        candidate_path=candidate_path,
    )
    if _canonical_json_bytes(supplied) != _canonical_json_bytes(expected):
        raise PromotionError(
            "supplied comparison differs from the independently reproduced artifact comparison"
        )


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PromotionError(f"{label} must be an object")
    return value


def _exact_parity(split: object) -> bool:
    split = _mapping(split, label="comparison split")
    parity = _mapping(split.get("probability_parity"), label="probability parity")
    numeric = _mapping(parity.get("probability_numeric"), label="numeric parity")
    tokens = _mapping(parity.get("probability_token_bytes"), label="token parity")
    deltas = _mapping(split.get("delta_left_minus_right"), label="metric deltas")
    return (
        numeric.get("exact_float_parity") is True
        and numeric.get("maximum_absolute_difference") == 0
        and numeric.get("mismatch_rows") == 0
        and numeric.get("mismatch_values") == 0
        and tokens.get("exact_parity") is True
        and tokens.get("mismatch_rows") == 0
        and tokens.get("mismatch_values") == 0
        and all(
            deltas.get(metric) == 0
            for metric in (
                "log_loss",
                "brier",
                "balanced_accuracy",
                "fallback_rate",
            )
        )
    )


def _validate_publication_health(
    meta: Mapping[str, Any],
    model: Mapping[str, Any],
) -> None:
    """Allow only healthy output or pre-reviewed model-only warnings."""

    health = _mapping(model.get("model_health"), label="candidate.model.model_health")
    reasons = health.get("reasons")
    if not isinstance(reasons, list) or any(not isinstance(reason, str) for reason in reasons):
        raise PromotionError("candidate model health reasons are invalid")
    if len(reasons) != len(set(reasons)):
        raise PromotionError("candidate model health reasons must be unique")
    status = meta.get("status")
    if status == "ok":
        if health.get("status") != "ok" or reasons:
            raise PromotionError("candidate ok status requires healthy model diagnostics")
        return
    allowed_reasons = {
        "weak_generalization",
        "calibration_drift",
        "low_transition_recall",
    }
    if (
        status != "degraded"
        or health.get("status") != "review_due"
        or not reasons
        or not set(reasons).issubset(allowed_reasons)
    ):
        raise PromotionError(
            "candidate degradation is not an allowed model-only review warning"
        )


def _validate_review_evidence(
    candidate: Mapping[str, Any],
    comparison: Mapping[str, Any],
    *,
    candidate_sha256: str,
) -> None:
    meta = _mapping(candidate.get("meta"), label="candidate.meta")
    model = _mapping(candidate.get("model"), label="candidate.model")
    if meta.get("mode") != "live" or meta.get("freshness", {}).get("status") != "current":
        raise PromotionError("candidate must be a current live result")
    if meta.get("publication_status") not in {None, "unpublished"} or meta.get(
        "publication_review"
    ) is not None:
        raise PromotionError("candidate is already marked for publication")
    if model.get("profile") != "standard":
        raise PromotionError("candidate must use the reviewed standard profile")
    diagnostics = model.get("selection_diagnostics")
    if not isinstance(diagnostics, list):
        raise PromotionError("candidate selection diagnostics are missing")
    for position, value in enumerate(diagnostics):
        row = _mapping(
            value,
            label=f"candidate.model.selection_diagnostics[{position}]",
        )
        threshold = row.get("minimum_log_loss_improvement")
        if (
            type(threshold) not in {int, float}
            or not math.isfinite(threshold)
            or float(threshold) != V5_MINIMUM_PROMOTION_LOG_LOSS_IMPROVEMENT
        ):
            raise PromotionError(
                "new V5 promotion candidates must use minimum_log_loss_improvement=0.01"
            )
    try:
        validate_v5_champion_selection_evidence(model)
    except V5ContractError as exc:
        raise PromotionError(f"candidate champion selection evidence failed: {exc}") from exc
    _validate_publication_health(meta, model)
    if model.get("latest_forecast_fallback") is not False:
        raise PromotionError("candidate latest forecast must not use fallback")
    weekly = candidate.get("weekly")
    if not isinstance(weekly, list) or not weekly:
        raise PromotionError("candidate weekly history is missing")
    latest = _mapping(weekly[-1], label="candidate latest week")
    if latest.get("health", {}).get("status") != "ok":
        raise PromotionError("candidate latest week health must be ok")
    sources = candidate.get("sources")
    if not isinstance(sources, list) or any(
        not isinstance(source, Mapping)
        or source.get("status") != "ok"
        or bool(source.get("issues"))
        for source in sources
    ):
        raise PromotionError("candidate sources must be complete and issue-free")

    fx = _mapping(model.get("fx_ablation"), label="candidate.model.fx_ablation")
    fx_gate = _mapping(fx.get("gate"), label="candidate.model.fx_ablation.gate")
    if (
        fx.get("promotion_allowed") is not False
        or fx.get("core_champion_promoted") is not False
        or fx_gate.get("passed_variants") != []
    ):
        raise PromotionError("FX must remain a non-promoted shadow evaluation")
    diagnostic_index: dict[str, Mapping[str, Any]] = {}
    for position, value in enumerate(diagnostics):
        row = _mapping(
            value,
            label=f"candidate.model.selection_diagnostics[{position}]",
        )
        name = row.get("model")
        if not isinstance(name, str) or not name or name in diagnostic_index:
            raise PromotionError("candidate selection diagnostics model names are invalid")
        diagnostic_index[name] = row

    multiscale = diagnostic_index.get(V5_MULTISCALE_MODEL)
    if multiscale is None:
        raise PromotionError("Multiscale selection diagnostics are missing")
    champion = model.get("champion")
    if not isinstance(champion, str) or champion not in diagnostic_index:
        raise PromotionError("candidate champion selection diagnostics are missing")
    champion_reference = diagnostic_index[champion].get("reference_model")
    if (
        not isinstance(champion_reference, str)
        or champion_reference not in diagnostic_index
    ):
        raise PromotionError("candidate champion reference diagnostics are missing")

    if (
        comparison.get("schema_version") != COMPARISON_SCHEMA
        or comparison.get("report_role") != "derived_only_diagnostic_comparison"
        or comparison.get("promotion_interpretation") != "prohibited"
        or comparison.get("provider_or_raw_feature_values_included") is not False
    ):
        raise PromotionError("comparison role is invalid")
    inputs = _mapping(comparison.get("inputs"), label="comparison.inputs")
    v5_input = _mapping(inputs.get("v5"), label="comparison.inputs.v5")
    payload_input = _mapping(
        v5_input.get("regime_results"),
        label="comparison.inputs.v5.regime_results",
    )
    if payload_input.get("sha256") != candidate_sha256:
        raise PromotionError("comparison is not bound to the candidate bytes")
    parity = _mapping(
        comparison.get("v5_markov_vs_frozen_v4_markov"),
        label="comparison Markov parity",
    )
    if not _exact_parity(parity.get("primary_selection")) or not _exact_parity(
        parity.get("post_selection_holdout")
    ):
        raise PromotionError("candidate Markov does not preserve exact frozen V4 parity")
    multiscale_comparison = _mapping(
        comparison.get("v5_causal_multiscale_ensemble_vs_v5_markov"),
        label="comparison Multiscale result",
    )
    gate = _mapping(
        multiscale_comparison.get("selection_gate_crosscheck"),
        label="comparison Multiscale gate",
    )
    if gate.get("artifact_role") != "selection_family_independently_recomputed":
        raise PromotionError("comparison selection-family audit role is invalid")
    selection_reference_gate = gate.get(
        "multiscale_gate_against_selection_reference"
    )
    if not isinstance(selection_reference_gate, bool):
        raise PromotionError("comparison Multiscale selection-reference gate is invalid")
    if selection_reference_gate is not multiscale.get("gate_passed"):
        raise PromotionError(
            "comparison Multiscale selection-reference gate differs from candidate evidence"
        )
    gate_models = _mapping(
        gate.get("models"),
        label="comparison Multiscale gate models",
    )
    required_gate_models = {
        V5_MULTISCALE_MODEL,
        "markov",
        champion,
        champion_reference,
    }
    if set(gate_models) != required_gate_models:
        raise PromotionError(
            "comparison gate models do not bind the champion and its reference"
        )
    for name in sorted(required_gate_models):
        comparison_row = _mapping(
            gate_models.get(name),
            label=f"comparison selection gate model {name}",
        )
        candidate_row = diagnostic_index[name]
        if comparison_row.get("matched_metric_crosscheck") is not True:
            raise PromotionError(
                f"comparison selection metrics are not cross-checked for {name}"
            )
        for field in (
            "reference_model",
            "selected",
            "gate_passed",
            "gate_reason",
            "fallback_count",
            "n_predictions",
        ):
            if comparison_row.get(field) != candidate_row.get(field):
                raise PromotionError(
                    f"comparison selection {field} differs from candidate evidence for {name}"
                )
        for field in ("log_loss", "brier"):
            try:
                candidate_value = float(candidate_row[field])
                comparison_value = float(comparison_row[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise PromotionError(
                    f"comparison selection {field} is invalid for {name}"
                ) from exc
            if not math.isclose(
                comparison_value,
                candidate_value,
                abs_tol=1e-8,
                rel_tol=0.0,
            ):
                raise PromotionError(
                    f"comparison selection {field} differs from candidate evidence for {name}"
                )


def promote(
    *,
    candidate_path: Path,
    v5_artifacts: Path,
    comparison_path: Path,
    generation_manifest_path: Path,
    output_path: Path,
    reviewed_at: datetime,
    output_comparison_path: Path | None = None,
    output_manifest_path: Path | None = None,
    output_selection_family_path: Path | None = None,
    publication_contract_directory: Path | None = None,
) -> dict[str, Any]:
    reviewed_comparison_path = output_comparison_path or output_path.with_name(
        "v5-vs-v4-comparison.json"
    )
    reviewed_manifest_path = output_manifest_path or output_path.with_name(
        "generation-manifest.json"
    )
    reviewed_selection_family_path = (
        output_selection_family_path
        or output_path.with_name("selection-family-audit.json")
    )
    contract_directory = (
        publication_contract_directory
        if publication_contract_directory is not None
        else output_path.parent
    )
    if not contract_directory.is_absolute():
        contract_directory = project_root() / contract_directory
    contract_payload_path = contract_directory / "regime-results.json"
    contract_comparison_path = contract_directory / "v5-vs-v4-comparison.json"
    contract_selection_family_path = (
        contract_directory / "selection-family-audit.json"
    )
    outputs = (
        output_path,
        reviewed_comparison_path,
        reviewed_selection_family_path,
        reviewed_manifest_path,
    )
    if len({path.resolve() for path in outputs}) != len(outputs):
        raise PromotionError("reviewed generation output paths must be distinct")
    for path in outputs:
        if path.exists() or path.is_symlink():
            raise PromotionError(f"output already exists: {path}")
    if reviewed_at.tzinfo is None or reviewed_at.utcoffset() is None:
        raise PromotionError("reviewed_at must include a timezone")
    candidate = _read_json(candidate_path, label="V5 candidate")
    comparison = _read_json(comparison_path, label="V5 comparison")
    try:
        generation = validate_generation_manifest(
            generation_manifest_path,
            # The unreviewed build manifest deliberately precedes the
            # independently rebuilt V5/V4 comparison.  Promotion validates
            # that comparison below and binds it only into the reviewed
            # generation.
            require_comparison=False,
            require_selection_family=True,
            artifact_directory=v5_artifacts,
        )
        if generation["payload_path"].resolve() != candidate_path.resolve():
            raise PromotionError("generation manifest points to a different candidate")
        if generation["artifact_directory"].resolve() != v5_artifacts.resolve():
            raise PromotionError("generation manifest points to different artifacts")
        source_selection_family = build_selection_family_audit_from_artifacts(
            candidate,
            v5_artifacts,
        )
        validate_selection_family_audit(
            source_selection_family,
            expected_generation_id=str(candidate["meta"]["generation_id"]),
        )
        if generation["selection_family"] != source_selection_family:
            raise PromotionError(
                "generation selection-family differs from source evidence"
            )
        if generation["selection_family_path"].resolve() != (
            v5_artifacts / "selection-family-audit.json"
        ).resolve():
            raise PromotionError(
                "generation manifest points to a different selection-family sidecar"
            )
        candidate_lifecycle = validate_lifecycle_consistency(candidate)
        if candidate_lifecycle["publication"] != "unpublished":
            raise PromotionError("candidate generation must be unpublished")
    except (IntegrityError, TypeError, ValueError) as exc:
        raise PromotionError(f"candidate generation manifest failed: {exc}") from exc
    try:
        validate_dashboard_payload(candidate)
    except ContractError as exc:
        raise PromotionError(f"candidate contract failed: {exc}") from exc
    candidate_raw_sha256 = _sha256(candidate_path)
    _validate_reproducible_comparison(
        comparison,
        v5_artifacts=v5_artifacts,
        candidate_path=candidate_path,
    )
    _validate_review_evidence(
        candidate,
        comparison,
        candidate_sha256=candidate_raw_sha256,
    )

    reviewed = deepcopy(candidate)
    meta = _mapping(reviewed["meta"], label="candidate.meta")
    model = _mapping(reviewed["model"], label="candidate.model")
    multiscale_promoted = model["champion"] == V5_MULTISCALE_MODEL
    candidate_sha256 = reviewed_candidate_sha256_v1(reviewed)
    lifecycle = _mapping(model["lifecycle"], label="candidate.model.lifecycle")
    deployment = _mapping(
        lifecycle["deployment"],
        label="candidate.model.lifecycle.deployment",
    )
    publication = _mapping(
        lifecycle["publication"],
        label="candidate.model.lifecycle.publication",
    )
    deployment["status"] = "operating"  # type: ignore[index]
    publication["status"] = V5_PUBLICATION_STATUS  # type: ignore[index]
    meta["publication_status"] = V5_PUBLICATION_STATUS  # type: ignore[index]
    meta["publication_review"] = {  # type: ignore[index]
        "schema_version": V5_PUBLICATION_REVIEW_SCHEMA,
        "decision": "publish_v5_research_snapshot",
        "reviewed_at": reviewed_at.astimezone(UTC).isoformat(),
        "reviewed_candidate_sha256": candidate_sha256,
        "champion": model["champion"],
        "multiscale_promoted": multiscale_promoted,
        "fx_promoted": False,
    }
    try:
        validate_lifecycle_consistency(reviewed)
        validate_dashboard_payload(reviewed)
        validate_reviewed_candidate_hash(reviewed)
    except (ContractError, IntegrityError) as exc:
        raise PromotionError(f"reviewed publication contract failed: {exc}") from exc

    reviewed_comparison = deepcopy(comparison)
    try:
        comparison_payload = reviewed_comparison["inputs"]["v5"]["regime_results"]
    except (KeyError, TypeError) as exc:
        raise PromotionError("comparison payload binding is missing") from exc
    if not isinstance(comparison_payload, dict):
        raise PromotionError("comparison payload binding is invalid")
    # The sidecar describes the eventual public package member rather than the
    # private review filename (which may be ``reviewed-regime-results.json``).
    comparison_payload["path"] = "regime-results.json"
    # The raw payload hash is excluded from the comparison contract hash, so a
    # placeholder safely breaks the payload/manifest/sidecar cycle.
    comparison_payload["sha256"] = "0" * 64
    label_spec_path = project_root() / str(generation["label_spec"]["path"])
    try:
        reviewed_manifest = build_generation_manifest(
            payload=reviewed,
            payload_path=output_path,
            artifact_directory=v5_artifacts,
            input_snapshot=generation["input_snapshot"],
            label_spec_path=label_spec_path,
            comparison=reviewed_comparison,
            comparison_path=reviewed_comparison_path,
            selection_family=source_selection_family,
            selection_family_path=reviewed_selection_family_path,
            payload_contract_path=contract_payload_path,
            comparison_contract_path=contract_comparison_path,
            selection_family_contract_path=contract_selection_family_path,
        )
        reviewed = bind_payload_to_generation_manifest(reviewed, reviewed_manifest)
        reviewed_raw = _published_json_bytes(reviewed)
        comparison_payload["sha256"] = hashlib.sha256(reviewed_raw).hexdigest()
        validate_lifecycle_consistency(reviewed)
        validate_dashboard_payload(reviewed)
        validate_reviewed_candidate_hash(reviewed)
        validate_v5_comparison_sidecar(
            reviewed_comparison,
            payload=reviewed,
            payload_raw=reviewed_raw,
        )
        validate_selection_family_audit(
            source_selection_family,
            expected_generation_id=str(reviewed["meta"]["generation_id"]),
        )
    except (ContractError, IntegrityError, PublicContractError) as exc:
        raise PromotionError(f"final reviewed generation contract failed: {exc}") from exc

    # The manifest is the commit marker and is written last.  All paths were
    # required to be absent, so rollback can remove only files created here and
    # can never overwrite or delete a prior last-good generation.
    created: list[Path] = []
    try:
        write_json_atomic(output_path, reviewed)
        created.append(output_path)
        write_json_atomic(reviewed_comparison_path, reviewed_comparison)
        created.append(reviewed_comparison_path)
        write_json_atomic(
            reviewed_selection_family_path,
            source_selection_family,
        )
        created.append(reviewed_selection_family_path)
        write_json_atomic(reviewed_manifest_path, reviewed_manifest)
        created.append(reviewed_manifest_path)
        final_generation = validate_generation_manifest(
            reviewed_manifest_path,
            require_comparison=True,
            require_selection_family=True,
            artifact_directory=v5_artifacts,
            payload_path_override=output_path,
            comparison_path_override=reviewed_comparison_path,
            selection_family_path_override=reviewed_selection_family_path,
        )
        if final_generation["payload_path"].resolve() != output_path.resolve():
            raise PromotionError("final manifest points to a different payload")
        if (
            final_generation["comparison_path"].resolve()
            != reviewed_comparison_path.resolve()
        ):
            raise PromotionError("final manifest points to a different comparison")
        if final_generation["selection_family_path"].resolve() != (
            reviewed_selection_family_path.resolve()
        ):
            raise PromotionError(
                "final manifest points to a different selection-family sidecar"
            )
        declared_members = {
            final_generation["declared_payload_path"].resolve(),
            final_generation["declared_comparison_path"].resolve(),
            final_generation["declared_selection_family_path"].resolve(),
        }
        expected_declared_members = {
            contract_payload_path.resolve(),
            contract_comparison_path.resolve(),
            contract_selection_family_path.resolve(),
        }
        if declared_members != expected_declared_members:
            raise PromotionError(
                "final manifest publication member paths are inconsistent"
            )
    except (IntegrityError, OSError, PromotionError) as exc:
        for path in reversed(created):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        if isinstance(exc, PromotionError):
            raise
        raise PromotionError(f"final reviewed generation verification failed: {exc}") from exc
    return {
        "ok": True,
        "output": str(output_path),
        "comparison_output": str(reviewed_comparison_path),
        "manifest_output": str(reviewed_manifest_path),
        "selection_family_output": str(reviewed_selection_family_path),
        "candidate_sha256": candidate_sha256,
        "candidate_raw_sha256": candidate_raw_sha256,
        "publication_sha256": _sha256(output_path),
        "comparison_sha256": _sha256(reviewed_comparison_path),
        "generation_manifest_sha256": final_generation["manifest_sha256"],
        "selection_family_sha256": source_selection_family["sha256"],
        "reviewed_at": reviewed_at.astimezone(UTC).isoformat(),
        "champion": model["champion"],
        "multiscale_promoted": multiscale_promoted,
        "fx_promoted": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote an audited V5 candidate to a reviewed public research snapshot"
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--v5-artifacts",
        type=Path,
        required=True,
        help="private V5 artifacts used to independently reproduce all selection evidence",
    )
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="candidate generation manifest; defaults to the candidate sibling",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--output-comparison",
        type=Path,
        help="reviewed comparison path; defaults beside --output",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        help="reviewed generation manifest path; defaults beside --output",
    )
    parser.add_argument(
        "--output-selection-family",
        type=Path,
        help="reviewed selection-family audit path; defaults beside --output",
    )
    parser.add_argument(
        "--publication-contract-directory",
        type=Path,
        help=(
            "Logical final directory recorded in the reviewed generation "
            "manifest when outputs are written to a private staging directory"
        ),
    )
    parser.add_argument(
        "--reviewed-at",
        help="ISO-8601 decision time; defaults to the current UTC time",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        reviewed_at = (
            datetime.fromisoformat(args.reviewed_at.replace("Z", "+00:00"))
            if args.reviewed_at
            else datetime.now(UTC)
        )
        result = promote(
            candidate_path=args.candidate,
            v5_artifacts=args.v5_artifacts,
            comparison_path=args.comparison,
            generation_manifest_path=(
                args.manifest
                if args.manifest is not None
                else args.candidate.with_name("generation-manifest.json")
            ),
            output_path=args.output,
            output_comparison_path=args.output_comparison,
            output_manifest_path=args.output_manifest,
            output_selection_family_path=args.output_selection_family,
            publication_contract_directory=args.publication_contract_directory,
            reviewed_at=reviewed_at,
        )
    except (PromotionError, OSError, ValueError) as exc:
        print(f"V5 publication promotion refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
