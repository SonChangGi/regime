#!/usr/bin/env python3
"""Recompose investment-aligned research into a local V5 preview payload.

The replay consumes an already-issued live payload plus the read-only
last-good snapshot store.  It does not recollect data, retrain a model, alter
the issued forecast envelope, or mutate publication inputs.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from regime_lab.analysis.decision_shadow import build_decision_shadow  # noqa: E402
from regime_lab.analysis.labels import (  # noqa: E402
    CausalRegimeLabeler,
    RegimeLabelConfig,
)
from regime_lab.artifact_inventory import write_artifact_inventory  # noqa: E402
from regime_lab.allocation.shadow import (  # noqa: E402
    allocation_calibration_evidence,
    build_allocation_shadow_candidate,
)
from regime_lab.collection import last_completed_week_cutoff, weekly_cutoffs  # noqa: E402
from regime_lab.config import load_config, project_root  # noqa: E402
from regime_lab.contract_v5 import validate_v5_payload  # noqa: E402
from regime_lab.data import SQLiteSnapshotStore  # noqa: E402
from regime_lab.dataset import build_weekly_dataset  # noqa: E402
from regime_lab.forecast_ledger import (  # noqa: E402
    build_research_replay_input_document,
)
from regime_lab.integrity import (  # noqa: E402
    IntegrityError,
    bind_payload_to_generation_manifest,
    build_generation_manifest,
    canonical_json_sha256_v1,
    reviewed_candidate_payload,
    validate_generation_manifest,
)
from regime_lab.io import write_json_atomic  # noqa: E402
from regime_lab.path_safety import confined_mutable_path  # noqa: E402
from regime_lab.selection_family_audit import (  # noqa: E402
    build_selection_family_audit_from_artifacts,
)
from regime_lab.v5 import _conditional_research, _model_conditioned_research  # noqa: E402
from regime_lab.v5_artifacts import (  # noqa: E402
    canonical_v5_artifact_csv_bytes,
    canonical_v5_artifact_frame,
    verify_staged_v5_research_artifacts,
)


class PreviewRecomposeError(RuntimeError):
    """Raised before an unsafe or semantically inconsistent preview write."""


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PreviewRecomposeError(f"{label} must be an existing regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreviewRecomposeError(f"{label} must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise PreviewRecomposeError(f"{label} must be a JSON object")
    return value


def _aware_datetime(value: object, *, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise PreviewRecomposeError(f"{label} must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PreviewRecomposeError(f"{label} must include a timezone")
    return parsed


def _pending_performance() -> dict[str, Any]:
    return {
        "status": "pending",
        "weeks": 0,
        "gross_cumulative_return": None,
        "net_cumulative_return": None,
        "turnover_sum": None,
        "transaction_cost_rate_sum": None,
        "transaction_cost_bps": None,
        "forecast_hit_count": None,
        "forecast_accuracy": None,
        "actual_state_counts": None,
    }


def _prospective_ledger_summary(
    payload: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Preserve available ledger counts while upgrading a legacy pending view."""

    forecast = payload.get("forecast")
    if not isinstance(forecast, Mapping):
        raise PreviewRecomposeError("source payload forecast is invalid")
    ledger = forecast.get("prospective_ledger")
    if not isinstance(ledger, Mapping):
        return None
    if ledger.get("schema_version") == "regime-prospective-ledger-summary/2":
        return deepcopy(dict(ledger))
    if ledger.get("schema_version") != "regime-prospective-ledger-summary/1":
        raise PreviewRecomposeError("source prospective ledger schema is invalid")

    status = str(ledger.get("status", ""))
    if status == "not_applicable":
        entry_count = 0
    elif status == "recorded":
        raw_count = ledger.get("entry_count")
        if isinstance(raw_count, bool) or not isinstance(raw_count, int):
            raise PreviewRecomposeError("legacy prospective ledger count is invalid")
        entry_count = raw_count
    elif status == "pending_append":
        return None
    else:
        raise PreviewRecomposeError("legacy prospective ledger status is invalid")
    if entry_count < 0:
        raise PreviewRecomposeError("legacy prospective ledger count is invalid")

    legacy_shadow = payload.get("research", {}).get(
        "prospective_decision_shadow", {}
    )
    legacy_prospective = (
        legacy_shadow.get("prospective_ledger", {})
        if isinstance(legacy_shadow, Mapping)
        else {}
    )
    realized = (
        legacy_prospective.get("realized_evaluation_count", 0)
        if isinstance(legacy_prospective, Mapping)
        else 0
    )
    if realized != 0:
        raise PreviewRecomposeError(
            "legacy payload reports realized ledger outcomes without V2 metrics"
        )
    legacy_count = (
        legacy_prospective.get("ledger_entry_count")
        if isinstance(legacy_prospective, Mapping)
        else None
    )
    if legacy_count is not None and legacy_count != entry_count:
        raise PreviewRecomposeError(
            "forecast and decision-shadow ledger counts disagree"
        )
    return {
        "schema_version": "regime-prospective-ledger-summary/2",
        "status": "empty" if entry_count == 0 else "pending",
        "entry_count": entry_count,
        "pending_evaluation_count": entry_count,
        "unresolved_due_evaluation_count": 0,
        "realized_evaluation_count": 0,
        "partial_evaluation_count": 0,
        "key_manifest_sha256": str(
            ledger.get("key_manifest_sha256")
            or canonical_json_sha256_v1([])
        ),
        "evaluation_manifest_sha256": canonical_json_sha256_v1([]),
        "hash_scope": "ordered_ledger_primary_keys_only",
        "evaluation_hash_scope": (
            "ordered_forecast_primary_keys_status_and_evaluation_sha256"
        ),
        "performance": _pending_performance(),
    }


def _canonical_and_states(
    payload: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    database: Path,
    operational_input_snapshot_sha256: str | None = None,
) -> tuple[Any, Any, int, dict[str, Any]]:
    meta = payload.get("meta")
    label = payload.get("label")
    if not isinstance(meta, Mapping) or not isinstance(label, Mapping):
        raise PreviewRecomposeError("source payload meta/label is invalid")
    data_as_of = _aware_datetime(meta.get("data_as_of"), label="meta.data_as_of")
    if data_as_of != last_completed_week_cutoff(data_as_of):
        raise PreviewRecomposeError(
            "meta.data_as_of must be an exact completed Friday 16:00 ET cutoff"
        )
    print("Read-only last-good snapshot 조립", flush=True)
    with SQLiteSnapshotStore(database, read_only=True) as store:
        observations = store.read_last_good_observations()
    if not observations:
        raise PreviewRecomposeError("snapshot store has no last-good observations")
    dataset = build_weekly_dataset(
        config,
        weekly_cutoffs(date(2006, 1, 1), data_as_of),
        observations,
        availability_basis="reconstructed_market",
    )
    canonical = dataset.canonical.loc[
        dataset.canonical["spy_close"].notna()
    ].copy()

    fit_period = label.get("fit_period")
    if not isinstance(fit_period, Mapping):
        raise PreviewRecomposeError("source label fit period is invalid")
    fit_weeks = fit_period.get("weeks")
    if (
        isinstance(fit_weeks, bool)
        or not isinstance(fit_weeks, int)
        or fit_weeks < 260
        or fit_weeks > len(canonical)
    ):
        raise PreviewRecomposeError("source label fit length is invalid")
    resolved_start = canonical.index[0].date().isoformat()
    resolved_end = canonical.index[fit_weeks - 1].date().isoformat()
    if (
        fit_period.get("start") != resolved_start
        or fit_period.get("end") != resolved_end
    ):
        raise PreviewRecomposeError(
            "reconstructed canonical panel differs from the source label fit period"
        )
    labeler = CausalRegimeLabeler(
        RegimeLabelConfig(
            price_column="spy_close",
            minimum_fit_observations=260,
        )
    )
    labeler.fit(canonical.iloc[:fit_weeks])
    states = labeler.transform(canonical)
    if operational_input_snapshot_sha256 is None:
        research_input_snapshot: dict[str, Any] = {}
    else:
        try:
            research_input_snapshot = build_research_replay_input_document(
                input_vintages=dataset.input_vintages,
                availability_basis=str(dataset.availability_basis),
                source_observation_count=len(observations),
                canonical=canonical,
                states=states,
                data_as_of=data_as_of,
                operational_input_snapshot_sha256=(
                    operational_input_snapshot_sha256
                ),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise PreviewRecomposeError(str(exc)) from exc
    print(
        f"Canonical panel 준비: {len(canonical):,} weeks × "
        f"{canonical.shape[1]:,} columns",
        flush=True,
    )
    return canonical, states, len(observations), research_input_snapshot


def _recompose_payload_with_frames(
    payload: Mapping[str, Any],
    *,
    canonical: Any,
    states: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replace only research calculations while preserving forecast identity."""

    weekly = payload.get("weekly")
    selection = payload.get("selection")
    model = payload.get("model")
    forecast = payload.get("forecast")
    meta = payload.get("meta")
    if (
        not isinstance(weekly, list)
        or not weekly
        or not isinstance(selection, Mapping)
        or not isinstance(model, Mapping)
        or not isinstance(forecast, Mapping)
        or not isinstance(meta, Mapping)
    ):
        raise PreviewRecomposeError("source payload research inputs are invalid")
    champion = str(selection.get("operating_champion", ""))
    if not champion:
        raise PreviewRecomposeError("source operating champion is missing")
    execution = model.get("execution_parameters")
    if not isinstance(execution, Mapping):
        raise PreviewRecomposeError("source execution parameters are missing")
    bootstrap_resamples = execution.get("conditional_outcome_bootstrap_resamples")
    if (
        isinstance(bootstrap_resamples, bool)
        or not isinstance(bootstrap_resamples, int)
        or bootstrap_resamples < 1
    ):
        raise PreviewRecomposeError("source conditional bootstrap budget is invalid")

    print(
        f"조건부 성과 재평가: {bootstrap_resamples:,} bootstrap resamples",
        flush=True,
    )
    conditional, conditional_result, matched_origins = _conditional_research(
        canonical,
        states,
        weekly,
        bootstrap_resamples=bootstrap_resamples,
    )
    comparison = model.get("forecast_comparison")
    comparison_models = (
        tuple(str(value) for value in comparison.get("models", ()))
        if isinstance(comparison, Mapping)
        else ()
    )
    if not comparison_models:
        raise PreviewRecomposeError("source forecast comparison models are missing")
    print(
        f"모델별 성과 재평가: {len(comparison_models):,} OOS models",
        flush=True,
    )
    (
        model_conditioned,
        model_conditioned_outcomes,
        model_conditioned_statistics,
    ) = _model_conditioned_research(
        canonical,
        weekly,
        comparison_models,
        bootstrap_resamples=bootstrap_resamples,
        matched_origins=matched_origins,
    )
    decision_at = forecast.get("decision_at") or meta.get("generated_at")
    decision_shadow = build_decision_shadow(
        weekly,
        canonical,
        forecast_model=champion,
        prospective_ledger_summary=_prospective_ledger_summary(payload),
        decision_at=decision_at,
    )
    decision_shadow["allocation_candidate"] = build_allocation_shadow_candidate(
        weekly,
        canonical,
        states,
        forecast_model=champion,
        selection_end=str(model["selection_end"]),
        current_signal=decision_shadow["current_signal"],
        calibration_evidence=allocation_calibration_evidence(
            model,
            forecast_model=champion,
        ),
    )
    print("V2 decision shadow 및 전체 V5 계약 검증", flush=True)

    candidate = deepcopy(dict(payload))
    research = deepcopy(dict(candidate.get("research", {})))
    research.update(conditional)
    research.update(model_conditioned)
    research["prospective_decision_shadow"] = decision_shadow
    candidate["research"] = research
    candidate = reviewed_candidate_payload(candidate)
    validate_v5_payload(candidate)
    frames = {
        "conditional_asset_outcomes": conditional_result.outcomes,
        "conditional_asset_statistics": conditional_result.statistics,
        "model_conditioned_asset_outcomes": model_conditioned_outcomes,
        "model_conditioned_asset_statistics": model_conditioned_statistics,
    }
    return candidate, frames


def recompose_payload(
    payload: Mapping[str, Any],
    *,
    canonical: Any,
    states: Any,
) -> dict[str, Any]:
    candidate, _ = _recompose_payload_with_frames(
        payload,
        canonical=canonical,
        states=states,
    )
    return candidate


def _bind_research_artifact_frames(
    candidate: Mapping[str, Any],
    frames: Mapping[str, Any],
) -> dict[str, Any]:
    bound = deepcopy(dict(candidate))
    model = bound.get("model")
    if not isinstance(model, dict):
        raise PreviewRecomposeError("candidate model contract is invalid")
    manifest = model.get("research_artifacts")
    if not isinstance(manifest, Mapping):
        raise PreviewRecomposeError("candidate research artifact manifest is missing")
    updated_manifest = deepcopy(dict(manifest))
    for key, frame in frames.items():
        metadata = updated_manifest.get(key)
        if not isinstance(metadata, Mapping):
            raise PreviewRecomposeError(f"candidate research artifact is missing: {key}")
        canonical = canonical_v5_artifact_frame(key, frame)
        payload = canonical_v5_artifact_csv_bytes(key, frame)
        updated_manifest[key] = {
            "path": str(metadata.get("path", "")),
            "row_count": int(len(canonical)),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    model["research_artifacts"] = updated_manifest
    validate_v5_payload(bound)
    return bound


def build_release_candidate(
    *,
    payload_path: str | Path,
    database_path: str | Path,
    config_path: str | Path,
    source_manifest_path: str | Path,
    source_artifacts_path: str | Path,
    release_root_path: str | Path,
) -> dict[str, Any]:
    """Create a complete private candidate generation without changing forecasts."""

    root = project_root()

    def project_path(value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    payload_file = project_path(payload_path)
    database = project_path(database_path)
    config_file = project_path(config_path)
    source_manifest = project_path(source_manifest_path)
    source_artifacts = project_path(source_artifacts_path)
    release_root = confined_mutable_path(
        release_root_path,
        project_directory=root,
        label="recomposed release candidate",
    )
    build_root = (root / "build").resolve()
    if not release_root.is_relative_to(build_root):
        raise PreviewRecomposeError("release candidate must stay below build/")
    if release_root.exists() or release_root.is_symlink():
        raise PreviewRecomposeError("release candidate directory must not exist")
    release_resolved = release_root.resolve()
    for read_only_input in (
        payload_file.resolve(),
        database.resolve(),
        source_manifest.resolve(),
        source_artifacts.resolve(),
    ):
        if (
            release_resolved == read_only_input
            or release_resolved.is_relative_to(read_only_input)
            or read_only_input.is_relative_to(release_resolved)
        ):
            raise PreviewRecomposeError(
                "release candidate overlaps a read-only input"
            )

    release_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{release_root.name}-",
            dir=release_root.parent,
        )
    )
    staged_artifacts = staging / "artifacts"
    final_payload = release_root / "regime-results.json"
    final_manifest = release_root / "generation-manifest.json"
    final_artifacts = release_root / "artifacts"
    installed = False
    try:
        shutil.copytree(source_artifacts, staged_artifacts)
        try:
            source_generation = validate_generation_manifest(
                source_manifest,
                require_comparison=True,
                require_selection_family=True,
                artifact_directory=staged_artifacts,
                payload_path_override=payload_file,
                comparison_path_override=payload_file.with_name(
                    "v5-vs-v4-comparison.json"
                ),
                selection_family_path_override=payload_file.with_name(
                    "selection-family-audit.json"
                ),
            )
        except (IntegrityError, OSError, TypeError, ValueError) as exc:
            raise PreviewRecomposeError(
                f"source reviewed generation is invalid: {exc}"
            ) from exc

        payload = deepcopy(source_generation["payload"])
        validate_v5_payload(payload)
        if payload.get("meta", {}).get("mode") != "live":
            raise PreviewRecomposeError("source payload must be live")
        config = load_config(config_file)
        canonical, states, observation_count, research_input_snapshot = (
            _canonical_and_states(
                payload,
                config=config,
                database=database,
                operational_input_snapshot_sha256=str(
                    source_generation["input_snapshot"]["sha256"]
                ),
            )
        )
        candidate, frames = _recompose_payload_with_frames(
            payload,
            canonical=canonical,
            states=states,
        )
        candidate = _bind_research_artifact_frames(candidate, frames)

        for key, frame in frames.items():
            metadata = candidate["model"]["research_artifacts"][key]
            artifact_path = staged_artifacts / str(metadata["path"])
            artifact_path.write_bytes(canonical_v5_artifact_csv_bytes(key, frame))

        selection_family = build_selection_family_audit_from_artifacts(
            candidate,
            staged_artifacts,
        )
        staged_selection = staged_artifacts / "selection-family-audit.json"
        write_json_atomic(staged_selection, selection_family)
        staged_research_input = staged_artifacts / "research-replay-input.json"
        write_json_atomic(staged_research_input, research_input_snapshot)
        write_artifact_inventory(staged_artifacts)
        verify_staged_v5_research_artifacts(
            candidate["model"]["research_artifacts"],
            staged_artifacts,
        )

        label_spec_path = root / str(source_generation["label_spec"]["path"])
        generation_manifest = build_generation_manifest(
            payload=candidate,
            payload_path=final_payload,
            artifact_directory=staged_artifacts,
            input_snapshot=source_generation["input_snapshot"],
            label_spec_path=label_spec_path,
            selection_family=selection_family,
            selection_family_path=staged_selection,
            selection_family_contract_path=(
                final_artifacts / "selection-family-audit.json"
            ),
        )
        candidate = bind_payload_to_generation_manifest(
            candidate,
            generation_manifest,
        )
        validate_v5_payload(candidate)
        staged_payload = staging / "regime-results.json"
        staged_manifest = staging / "generation-manifest.json"
        write_json_atomic(staged_payload, candidate)
        write_json_atomic(staged_manifest, generation_manifest)
        try:
            validate_generation_manifest(
                staged_manifest,
                require_comparison=False,
                require_selection_family=True,
                artifact_directory=staged_artifacts,
                payload_path_override=staged_payload,
                selection_family_path_override=staged_selection,
            )
            verify_staged_v5_research_artifacts(
                candidate["model"]["research_artifacts"],
                staged_artifacts,
            )
        except (IntegrityError, OSError, TypeError, ValueError) as exc:
            raise PreviewRecomposeError(
                f"staged release candidate is invalid: {exc}"
            ) from exc
        if release_root.exists() or release_root.is_symlink():
            raise PreviewRecomposeError(
                "release candidate directory appeared during generation"
            )
        os.replace(staging, release_root)
        installed = True
        try:
            validated = validate_generation_manifest(
                final_manifest,
                require_comparison=False,
                require_selection_family=True,
                artifact_directory=final_artifacts,
            )
            verify_staged_v5_research_artifacts(
                candidate["model"]["research_artifacts"],
                final_artifacts,
            )
        except (IntegrityError, OSError, TypeError, ValueError) as exc:
            raise PreviewRecomposeError(
                f"persisted release candidate is invalid: {exc}"
            ) from exc
    except BaseException:
        shutil.rmtree(release_root if installed else staging, ignore_errors=True)
        raise

    shadow = candidate["research"]["prospective_decision_shadow"]
    historical = shadow["historical_reconstructed_shadow"]
    return {
        "ok": True,
        "release_root": str(release_root),
        "output": str(final_payload),
        "artifacts": str(final_artifacts),
        "manifest": str(final_manifest),
        "generation_id": validated["generation_id"],
        "manifest_sha256": validated["manifest_sha256"],
        "data_as_of": candidate["meta"]["data_as_of"],
        "weekly_rows": len(candidate["weekly"]),
        "canonical_rows": len(canonical),
        "last_good_observations": observation_count,
        "research_input_snapshot": str(
            final_artifacts / "research-replay-input.json"
        ),
        "research_input_vintages": research_input_snapshot["input_vintages"][
            "count"
        ],
        "conditional_rows": len(
            candidate["research"]["conditional_asset_stats"]["rows"]
        ),
        "model_conditioned_rows": len(
            candidate["research"]["model_conditioned_asset_stats"]["rows"]
        ),
        "decision_shadow_schema": shadow["schema_version"],
        "decision_shadow_weeks": historical["strategies"][
            "probability_shadow"
        ]["weeks"],
        "publication_status": candidate["meta"]["publication_status"],
        "contract_valid": True,
    }


def build_local_preview(
    *,
    payload_path: str | Path,
    database_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    root = project_root()
    payload_file = Path(payload_path)
    if not payload_file.is_absolute():
        payload_file = root / payload_file
    database = Path(database_path)
    if not database.is_absolute():
        database = root / database
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    output = confined_mutable_path(
        output_path,
        project_directory=root,
        label="local preview payload",
    )
    build_root = (root / "build").resolve()
    if not output.is_relative_to(build_root):
        raise PreviewRecomposeError("local preview output must stay below build/")
    if output in {payload_file.resolve(), database.resolve()}:
        raise PreviewRecomposeError("local preview output overlaps a read-only input")

    payload = _read_json(payload_file, label="source live payload")
    validate_v5_payload(payload)
    if payload.get("meta", {}).get("mode") != "live":
        raise PreviewRecomposeError("source payload must be live")
    config = load_config(config_file)
    canonical, states, observation_count, _ = _canonical_and_states(
        payload,
        config=config,
        database=database,
    )
    candidate = recompose_payload(
        payload,
        canonical=canonical,
        states=states,
    )
    write_json_atomic(output, candidate)
    persisted = _read_json(output, label="persisted local preview payload")
    validate_v5_payload(persisted)
    if persisted != candidate:
        raise PreviewRecomposeError("persisted preview differs from validated memory")

    shadow = candidate["research"]["prospective_decision_shadow"]
    historical = shadow["historical_reconstructed_shadow"]
    return {
        "ok": True,
        "output": str(output),
        "mode": candidate["meta"]["mode"],
        "generated_at": candidate["meta"]["generated_at"],
        "data_as_of": candidate["meta"]["data_as_of"],
        "weekly_rows": len(candidate["weekly"]),
        "canonical_rows": len(canonical),
        "last_good_observations": observation_count,
        "conditional_rows": len(
            candidate["research"]["conditional_asset_stats"]["rows"]
        ),
        "model_conditioned_rows": len(
            candidate["research"]["model_conditioned_asset_stats"]["rows"]
        ),
        "decision_shadow_schema": shadow["schema_version"],
        "decision_shadow_weeks": historical["strategies"][
            "probability_shadow"
        ]["weeks"],
        "decision_status": shadow["current_signal"]["status"],
        "publication_status": candidate["meta"]["publication_status"],
        "contract_valid": True,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompose a contract-valid local V5 preview from the issued live "
            "payload and the read-only last-good store"
        )
    )
    parser.add_argument(
        "--payload",
        type=Path,
        default=Path("publication/live/regime-results.json"),
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/regime.sqlite3"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/series.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/local-preview-live-v5/regime-results.json"),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("publication/live/generation-manifest.json"),
    )
    parser.add_argument(
        "--source-artifacts",
        type=Path,
        help="verified private artifacts for the issued source generation",
    )
    parser.add_argument(
        "--release-root",
        type=Path,
        help="new build/ directory for a complete unpublished candidate generation",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if (args.release_root is None) != (args.source_artifacts is None):
            raise PreviewRecomposeError(
                "--release-root and --source-artifacts must be supplied together"
            )
        if args.release_root is not None:
            result = build_release_candidate(
                payload_path=args.payload,
                database_path=args.database,
                config_path=args.config,
                source_manifest_path=args.source_manifest,
                source_artifacts_path=args.source_artifacts,
                release_root_path=args.release_root,
            )
        else:
            result = build_local_preview(
                payload_path=args.payload,
                database_path=args.database,
                config_path=args.config,
                output_path=args.output,
            )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"local preview recompose refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
