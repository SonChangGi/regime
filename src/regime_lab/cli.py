"""Command-line entrypoints for collection, analysis, validation, and serving."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping

from regime_lab.artifact_inventory import (
    verify_artifact_inventory,
    write_artifact_inventory,
)
from regime_lab.collection import (
    CollectionGateError,
    collect_live_data,
    collection_report_document,
    last_completed_week_cutoff,
    validate_collection_for_training,
)
from regime_lab.analysis.ablation import feature_ablation_manifest_document
from regime_lab.analysis.models import model_manifest, model_manifest_sha256
from regime_lab.config import default_config_path, load_config, project_root
from regime_lab.dataset import build_weekly_dataset
from regime_lab.database_backup import create_database_backup
from regime_lab.demo import generate_demo_payload
from regime_lab.evidence import (
    STATE_LABEL_HISTORY_COLUMNS,
    STATE_MEMBERSHIP_HISTORY_COLUMNS,
    WEEKLY_STATE_FORECAST_COLUMNS,
    WEEKLY_STATE_FORECAST_V5_COLUMNS,
    canonical_evidence_csv_bytes,
)
from regime_lab.feature_quality import (
    canonical_feature_quality_json_bytes,
    verify_feature_quality_artifact,
)
from regime_lab.forecast_ledger import (
    ForecastLedger,
    ForecastLedgerEntry,
    ForecastLedgerKey,
    OperationalInput,
    operational_input_manifest_sha256,
)
from regime_lab.keychain import provider_environment_from_keychain
from regime_lab.payload import write_dashboard_payload
from regime_lab.io import write_json_atomic
from regime_lab.integrity import (
    bind_payload_to_generation_manifest,
    build_generation_manifest,
    canonical_json_sha256_v1,
    validate_generation_manifest,
)
from regime_lab.analysis.label_spec import default_label_spec_path
from regime_lab.path_safety import confined_mutable_path
from regime_lab.provider_rights import (
    ProviderRightsError,
    providers_for_live_config,
    verify_provider_rights,
)
from regime_lab.run_registry import append_run_event, current_run_status
from regime_lab.selection_family_audit import (
    build_selection_family_audit_from_artifacts,
)
from regime_lab.publication_contract import (
    V5_RESULT_VERSION,
    validate_v5_comparison_sidecar,
)
from regime_lab.pipeline import build_dashboard_result
from regime_lab.schema import validate_dashboard_payload
from regime_lab.server import serve_dashboard
from regime_lab.smoke import main as smoke_main
from regime_lab.automation import AlreadyRunning, automation_lock, command_automation
from regime_lab.v5_artifacts import (
    V5_RESEARCH_ARTIFACTS_BY_PATH,
    canonical_v5_artifact_csv_bytes,
    verify_staged_v5_core_artifacts,
    verify_staged_v5_research_artifacts,
)
from regime_lab.v5_preflight import (
    require_v5_analysis_source_unchanged,
    verify_v5_preflight,
)


def _operational_inputs_for_generation(
    dataset: object,
    *,
    additional_records: tuple[object, ...] = (),
    origin_at: datetime,
    decision_at: datetime,
) -> tuple[OperationalInput, ...]:
    """Bind the exact inputs used by a real forecast decision.

    Historical walk-forward rows can only be classified as reconstructed OOS
    when the provider supplied a current-adjusted backfill.  Requiring an
    ``operational`` as-of join for every historical row would therefore erase
    the training history rather than make it prospective.  The operational
    claim belongs to the *current decision*: every revision used anywhere in
    the assembled training/forecast matrix must have been retrieved by that
    decision, and no observation period may extend beyond the forecast origin.
    The append-only ledger enforces the same clocks independently on append.
    """

    if getattr(dataset, "availability_basis", None) not in {
        "source",
        "operational",
        "reconstructed_market",
    }:
        raise ValueError("forecast ledger dataset has an invalid availability basis")
    if origin_at.tzinfo is None or origin_at.utcoffset() is None:
        raise ValueError("forecast origin must include a timezone")
    if decision_at.tzinfo is None or decision_at.utcoffset() is None:
        raise ValueError("forecast decision must include a timezone")
    origin = origin_at.astimezone(timezone.utc)
    decision = decision_at.astimezone(timezone.utc)
    if origin > decision:
        raise ValueError("forecast origin must not follow the decision")
    values = tuple(getattr(dataset, "input_vintages", ()))
    inputs = [OperationalInput.from_asof_value(value) for value in values]
    for item in inputs:
        if item.observed_period_end > origin.date():
            raise ValueError("forecast input period exceeds the forecast origin")
        if item.operating_available_at > decision:
            raise ValueError("forecast input was first seen after the decision")
        if item.system_retrieved_at > decision:
            raise ValueError("forecast input was retrieved after the decision")
    for record in additional_records:
        if (
            record.observed_period_end > origin.date()
            or record.operating_available_at > decision
            or record.system_retrieved_at > decision
        ):
            continue
        inputs.append(OperationalInput.from_observation(record))
    unique = {
        (
            item.source,
            item.series_id,
            item.observed_period_end,
            item.revision_seq,
            item.raw_sha256,
        ): item
        for item in inputs
    }
    if not unique:
        raise ValueError("operational generation has no bound input vintages")
    return tuple(unique.values())


def _forecast_ledger_entry(
    payload: Mapping[str, Any],
    *,
    operational_inputs: tuple[OperationalInput, ...],
    published_at: datetime,
) -> ForecastLedgerEntry:
    forecast_contract = payload.get("forecast")
    if not isinstance(forecast_contract, Mapping):
        raise ValueError("operational forecast ledger requires payload.forecast")
    if forecast_contract.get("status") != "active":
        raise ValueError("expired forecast cannot enter the operational ledger")
    decision_at = datetime.fromisoformat(str(forecast_contract["decision_at"]))
    target_at = datetime.fromisoformat(str(forecast_contract["target_at"]))
    latest = payload["weekly"][-1]
    model = payload["model"]
    label = payload["label"]
    input_snapshot_sha256 = operational_input_manifest_sha256(operational_inputs)
    publication_at = published_at.astimezone(timezone.utc)
    return ForecastLedgerEntry(
        origin_week=date.fromisoformat(str(latest["date"])),
        decision_at=decision_at,
        target_at=target_at,
        label_spec_sha256=str(label["spec_sha256"]),
        model_manifest_sha256=str(model["candidate_manifest_sha256"]),
        input_snapshot_sha256=input_snapshot_sha256,
        operational_inputs=operational_inputs,
        forecast={
            "schema_version": "regime-operational-forecast-ledger/1",
            "evidence_track": "operational_oos",
            "generation_id": payload["meta"]["generation_id"],
            "generated_at": payload["meta"]["generated_at"],
            "local_publication_at": publication_at.isoformat(),
            "current": latest["current"],
            "official": latest["next_week"],
            "model_forecasts": latest.get("model_forecasts", []),
            "champion": model["champion"],
            "selection": payload["selection"],
            "lifecycle": model["lifecycle"],
        },
    )


def _root_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root() / path


def _mutable_path(value: str | Path, *, label: str) -> Path:
    """Confine a CLI write target without restricting read-only inputs."""

    return confined_mutable_path(
        value,
        project_directory=project_root(),
        label=label,
    )


def _backup_database_before_mutation(
    database: Path,
    *,
    backup_directory: str | Path | None,
    source_code_fingerprint_sha256: str | None = None,
) -> None:
    """Create a verified private generation before a command mutates SQLite."""

    if not (database.exists() or database.is_symlink()):
        return
    selected = (
        database.with_name(f"{database.name}.backups")
        if backup_directory is None
        else _mutable_path(backup_directory, label="database backup directory")
    )
    create_database_backup(
        database,
        selected,
        retain=4,
        source_code_fingerprint_sha256=source_code_fingerprint_sha256,
    )


_CONTRACT_DEFAULT_TARGETS: dict[tuple[str, str], tuple[str, str]] = {
    ("build", "v4"): ("web/data/regime-results.json", "artifacts/latest"),
    ("build", "v5"): (
        "build/v5-live/regime-results.json",
        "build/v5-live/artifacts",
    ),
    ("demo", "v4"): ("web/data/regime-results.json", "artifacts/demo"),
    ("demo", "v5"): (
        "build/v5-demo/regime-results.json",
        "build/v5-demo/artifacts",
    ),
}

# These paths either hold the active v4 result or feed its atomic promotion.
# A v5 research command must not replace, contain, or be contained by them.
_V4_OWNED_WRITE_TARGETS: tuple[str, ...] = (
    "web/data/regime-results.json",
    "artifacts/latest",
    "artifacts/demo",
    "publication/live/regime-results.json",
    "build/weekly-automation/generation/regime-results.json",
    "build/weekly-automation/generation/artifacts",
)
_V5_H10_DEFAULT_RECEIPT = "build/v5-h10/collection-receipt.json"
_V5_H10_ARCHIVE_DEFAULT_RECEIPT = (
    "build/v5-h10/archive-collection-receipt.json"
)
_V6_OFR_FSI_DEFAULT_DATABASE = "data/ofr-fsi-shadow.sqlite3"
_V6_OFR_FSI_DEFAULT_RECEIPT = "build/v6-ofr-fsi/collection-receipt.json"
_H10_RECEIPT_PROTECTED_WRITE_TARGETS: tuple[str, ...] = (
    "build/weekly-automation",
    "build/v5-live/regime-results.json",
    "build/v5-live/artifacts",
    "build/v5-demo/regime-results.json",
    "build/v5-demo/artifacts",
)
_OFR_FSI_PROTECTED_WRITE_TARGETS: tuple[str, ...] = (
    "data/regime.sqlite3",
    "build/weekly-automation",
    "build/v5-live",
    "build/v5-demo",
    "publication/live",
    "web",
    "artifacts/latest",
    "artifacts/demo",
)


def _paths_overlap(left: Path, right: Path) -> bool:
    left_absolute = left.absolute()
    right_absolute = right.absolute()
    return (
        left_absolute == right_absolute
        or left_absolute.is_relative_to(right_absolute)
        or right_absolute.is_relative_to(left_absolute)
    )


def _resolve_contract_write_targets(
    *,
    command: str,
    contract_version: str,
    output: str | Path | None,
    artifacts: str | Path | None,
) -> tuple[Path, Path]:
    """Resolve contract-specific defaults and protect every v4-owned target."""

    key = (str(command), str(contract_version))
    try:
        default_output, default_artifacts = _CONTRACT_DEFAULT_TARGETS[key]
    except KeyError as exc:
        raise ValueError(
            "contract write targets require build/demo and v4/v5"
        ) from exc
    output_path = _mutable_path(
        default_output if output is None else output,
        label="dashboard output",
    )
    artifact_path = _mutable_path(
        default_artifacts if artifacts is None else artifacts,
        label="artifact output",
    )
    if _paths_overlap(output_path, artifact_path):
        raise ValueError(
            "dashboard output and artifact output must not overlap: "
            f"{output_path} conflicts with {artifact_path}"
        )
    if contract_version == "v5":
        protected = tuple(
            _root_path(target).absolute() for target in _V4_OWNED_WRITE_TARGETS
        )
        for label, candidate in (
            ("dashboard output", output_path),
            ("artifact output", artifact_path),
        ):
            conflict = next(
                (target for target in protected if _paths_overlap(candidate, target)),
                None,
            )
            if conflict is not None:
                raise ValueError(
                    f"v5 {label} overlaps a v4-owned target: "
                    f"{candidate} conflicts with {conflict}"
                )
    return output_path, artifact_path


def _resolve_live_checkpoint_directory(
    *,
    contract_version: str,
    output: Path,
    artifacts: Path,
    value: str | Path | None,
) -> Path | None:
    """Resolve the private base-walk-forward checkpoint for a live V5 build."""

    if contract_version != "v5":
        if value is not None:
            raise ValueError("--checkpoint-directory is available only for V5 builds")
        return None
    candidate = (
        output.parent / ".private-checkpoints" / "base-walk-forward"
        if value is None
        else value
    )
    checkpoint = _mutable_path(candidate, label="private V5 checkpoint directory")
    for label, target in (
        ("dashboard output", output),
        ("artifact output", artifacts),
    ):
        if _paths_overlap(checkpoint, target):
            raise ValueError(
                "private V5 checkpoint directory must not overlap the "
                f"{label}: {checkpoint} conflicts with {target}"
            )
    protected = tuple(
        _root_path(target).absolute() for target in _V4_OWNED_WRITE_TARGETS
    )
    conflict = next(
        (target for target in protected if _paths_overlap(checkpoint, target)),
        None,
    )
    if conflict is not None:
        raise ValueError(
            "private V5 checkpoint directory overlaps a V4-owned target: "
            f"{checkpoint} conflicts with {conflict}"
        )
    return checkpoint


def _resolve_h10_collection_receipt(
    value: str | Path | None,
    *,
    database: Path,
) -> Path:
    """Resolve one v5-only receipt without permitting operational overlap."""

    receipt = _mutable_path(
        _V5_H10_DEFAULT_RECEIPT if value is None else value,
        label="H.10 collection receipt",
    )
    protected = tuple(
        _root_path(target).absolute() for target in _V4_OWNED_WRITE_TARGETS
    )
    conflict = next(
        (target for target in protected if _paths_overlap(receipt, target)),
        None,
    )
    if conflict is not None:
        raise ValueError(
            "v5 H.10 receipt overlaps a v4-owned target: "
            f"{receipt} conflicts with {conflict}"
        )
    protected_research = tuple(
        _root_path(target).absolute()
        for target in _H10_RECEIPT_PROTECTED_WRITE_TARGETS
    )
    conflict = next(
        (
            target
            for target in protected_research
            if _paths_overlap(receipt, target)
        ),
        None,
    )
    if conflict is not None:
        raise ValueError(
            "v5 H.10 receipt overlaps an automation or model-result target: "
            f"{receipt} conflicts with {conflict}"
        )
    live_build_lock = database.with_name(f"{database.name}.live-build.lock")
    for label, target in (
        ("snapshot database", database),
        ("live-build lock", live_build_lock),
    ):
        if _paths_overlap(receipt, target):
            raise ValueError(f"v5 H.10 receipt overlaps the {label}: {target}")
    return receipt


def _resolve_ofr_fsi_collection_targets(
    *,
    database_value: str | Path | None,
    receipt_value: str | Path | None,
) -> tuple[Path, Path, Path]:
    """Resolve private V6 OFR targets and reject every operating surface."""

    database = _mutable_path(
        _V6_OFR_FSI_DEFAULT_DATABASE if database_value is None else database_value,
        label="OFR FSI shadow database",
    )
    receipt = _mutable_path(
        _V6_OFR_FSI_DEFAULT_RECEIPT if receipt_value is None else receipt_value,
        label="OFR FSI collection receipt",
    )
    lock = database.with_name(f"{database.name}.ofr-fsi-collect.lock")
    protected = tuple(
        _root_path(target).absolute()
        for target in _OFR_FSI_PROTECTED_WRITE_TARGETS
    )
    for label, candidate in (
        ("shadow database", database),
        ("collection receipt", receipt),
        ("collection lock", lock),
    ):
        conflict = next(
            (target for target in protected if _paths_overlap(candidate, target)),
            None,
        )
        if conflict is not None:
            raise ValueError(
                f"V6 OFR FSI {label} overlaps an operating/public target: "
                f"{candidate} conflicts with {conflict}"
            )
    if _paths_overlap(database, receipt):
        raise ValueError("V6 OFR FSI receipt must not overlap its private database")
    if _paths_overlap(receipt, lock):
        raise ValueError("V6 OFR FSI receipt must not overlap its collection lock")
    return database, receipt, lock


def _flush_progress(message: str) -> None:
    print(message, flush=True)


def _aware_datetime_argument(raw: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected cutoff must be an ISO-8601 datetime"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("expected cutoff must include a timezone")
    return parsed.astimezone(timezone.utc)


def _date_argument(raw: str) -> date:
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an ISO-8601 date") from exc


class AnalysisPreconditionError(RuntimeError):
    """A post-collection condition blocks expensive model analysis."""


def _require_ac_power_before_analysis(*, enabled: bool) -> None:
    if not enabled or sys.platform != "darwin":
        return
    completed = subprocess.run(
        ["/usr/bin/pmset", "-g", "batt"],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0 or b"AC Power" not in completed.stdout:
        raise AnalysisPreconditionError("AC power is required before analysis")


def _write_supporting_results(
    benchmark: object,
    directory: Path,
    *,
    generation_id: str | None = None,
    write_inventory: bool = False,
    selection_context: Mapping[str, Any] | None = None,
) -> None:
    """Stage a self-consistent artifact generation before replacing latest.

    CSV generation and manifest serialization must all succeed in the private
    sibling staging directory.  Only then is the directory entry swapped,
    avoiding a mixed generation after interruption.
    """

    directory.parent.mkdir(parents=True, exist_ok=True)
    leaderboard = getattr(benchmark, "leaderboard")
    predictions = getattr(benchmark, "predictions")
    split_audit = getattr(benchmark, "split_audit")
    selection_diagnostics = getattr(benchmark, "selection_diagnostics", None)
    model_names = tuple(
        predictions["model"].astype(str).drop_duplicates().tolist()
    )
    manifest = model_manifest(
        getattr(benchmark, "profile"), random_state=17, names=model_names
    )
    manifest["sha256"] = model_manifest_sha256(
        getattr(benchmark, "profile"), random_state=17, names=model_names
    )
    transition = getattr(benchmark, "transition_benchmark", None)
    feature_ablation = getattr(benchmark, "feature_ablation", None)
    feature_manifest = getattr(benchmark, "feature_manifest", None)
    feature_quality = getattr(benchmark, "feature_quality_report", None)
    structural_forecasts = getattr(benchmark, "structural_forecasts", None)
    joint_survival_forecasts = getattr(
        benchmark, "joint_survival_forecasts", None
    )
    stacking_weights = getattr(benchmark, "stacking_weights", None)
    multiscale_scale_forecasts = getattr(
        benchmark,
        "multiscale_scale_forecasts",
        None,
    )
    state_labels = getattr(benchmark, "state_label_history", None)
    weekly_forecasts = getattr(benchmark, "weekly_state_forecasts", None)
    directional = getattr(benchmark, "directional_benchmark", None)
    conditional_outcomes = getattr(benchmark, "conditional_asset_outcomes", None)
    conditional_statistics = getattr(
        benchmark, "conditional_asset_statistics", None
    )
    model_conditioned_outcomes = getattr(
        benchmark, "model_conditioned_asset_outcomes", None
    )
    model_conditioned_statistics = getattr(
        benchmark, "model_conditioned_asset_statistics", None
    )
    fx_features = getattr(benchmark, "fx_features", None)
    fx_coverage = getattr(benchmark, "fx_coverage", None)
    fx_ablation_oos = getattr(benchmark, "fx_ablation_oos", None)
    membership_history = getattr(benchmark, "state_membership_history", None)
    weekly_forecasts_v5 = getattr(
        benchmark, "weekly_state_forecasts_v5", None
    )
    staging = Path(
        tempfile.mkdtemp(prefix=f".{directory.name}-staging-", dir=directory.parent)
    )
    backup = directory.parent / f".{directory.name}-previous"
    try:
        frames = {
            "model-leaderboard.csv": leaderboard,
            "oos-predictions.csv": predictions,
            "walk-forward-splits.csv": split_audit,
        }
        if selection_diagnostics is not None:
            frames["selection-diagnostics.csv"] = selection_diagnostics
        if transition is not None:
            frames.update(
                {
                    "transition-oos-predictions.csv": transition.predictions,
                    "transition-model-leaderboard.csv": transition.leaderboard,
                    "transition-walk-forward-splits.csv": transition.split_audit,
                    "nested-selection.csv": transition.nested_selection,
                    "transition-forecasts.csv": transition.latest_forecasts(),
                    "transition-candidate-status.csv": transition.candidate_status,
                    "transition-candidate-forecasts.csv": (
                        transition.latest_candidate_forecasts()
                    ),
                }
            )
        if stacking_weights is not None:
            frames["stacking-weights.csv"] = stacking_weights
        if multiscale_scale_forecasts is not None:
            frames["multiscale-ensemble-scales.csv"] = (
                multiscale_scale_forecasts
            )
        if structural_forecasts is not None:
            frames["structural-forecasts.csv"] = structural_forecasts
        if joint_survival_forecasts is not None:
            frames["joint-survival-forecasts.csv"] = joint_survival_forecasts
        if feature_ablation is not None:
            frames.update(
                {
                    "feature-ablation-oos-predictions.csv": (
                        feature_ablation.predictions
                    ),
                    "feature-ablation-leaderboard.csv": (
                        feature_ablation.leaderboard
                    ),
                }
            )
        if directional is not None:
            frames.update(
                {
                    "directional-oos-predictions.csv": directional.predictions,
                    "directional-model-leaderboard.csv": directional.leaderboard,
                    "directional-walk-forward-splits.csv": directional.split_audit,
                    "directional-selection-diagnostics.csv": (
                        directional.selection_diagnostics
                    ),
                    "directional-forecasts.csv": directional.latest_forecasts,
                }
            )
        if conditional_outcomes is not None:
            frames["conditional-asset-outcomes.csv"] = conditional_outcomes
        if conditional_statistics is not None:
            frames["conditional-asset-statistics.csv"] = conditional_statistics
        if model_conditioned_outcomes is not None:
            frames["model-conditioned-asset-outcomes.csv"] = (
                model_conditioned_outcomes
            )
        if model_conditioned_statistics is not None:
            frames["model-conditioned-asset-statistics.csv"] = (
                model_conditioned_statistics
            )
        if fx_features is not None:
            frames["fx-features.csv"] = fx_features
        if fx_coverage is not None:
            frames["fx-coverage.csv"] = fx_coverage
        if fx_ablation_oos is not None:
            frames["fx-ablation-oos.csv"] = fx_ablation_oos
        for filename, frame in frames.items():
            if filename in V5_RESEARCH_ARTIFACTS_BY_PATH:
                (staging / filename).write_bytes(
                    canonical_v5_artifact_csv_bytes(filename, frame)
                )
            else:
                frame.to_csv(staging / filename, index=False)
        if state_labels is not None:
            (staging / "state-label-history.csv").write_bytes(
                canonical_evidence_csv_bytes(
                    state_labels,
                    STATE_LABEL_HISTORY_COLUMNS,
                )
            )
        if weekly_forecasts is not None:
            (staging / "weekly-state-forecasts.csv").write_bytes(
                canonical_evidence_csv_bytes(
                    weekly_forecasts,
                    WEEKLY_STATE_FORECAST_COLUMNS,
                )
            )
        if membership_history is not None:
            (staging / "state-membership-history.csv").write_bytes(
                canonical_evidence_csv_bytes(
                    membership_history,
                    STATE_MEMBERSHIP_HISTORY_COLUMNS,
                )
            )
        if weekly_forecasts_v5 is not None:
            (staging / "weekly-state-forecasts-v5.csv").write_bytes(
                canonical_evidence_csv_bytes(
                    weekly_forecasts_v5,
                    WEEKLY_STATE_FORECAST_V5_COLUMNS,
                )
            )
        write_json_atomic(staging / "candidate-manifest.json", manifest)
        if feature_manifest is not None:
            write_json_atomic(staging / "feature-manifest.json", feature_manifest)
        if feature_quality is not None:
            (staging / "feature-quality.json").write_bytes(
                canonical_feature_quality_json_bytes(feature_quality)
            )
        if feature_ablation is not None:
            write_json_atomic(
                staging / "feature-ablation-manifest.json",
                feature_ablation_manifest_document(feature_ablation.manifest),
            )
        if selection_context is not None:
            if selection_diagnostics is None:
                raise RuntimeError(
                    "selection-family audit requires selection diagnostics"
                )
            if not isinstance(selection_context.get("selection"), Mapping):
                raise RuntimeError("selection-family audit context is invalid")
            selection_family = build_selection_family_audit_from_artifacts(
                selection_context,
                staging,
            )
            write_json_atomic(
                staging / "selection-family-audit.json",
                selection_family,
            )
        if generation_id is not None:
            write_json_atomic(
                staging / "build-generation.json",
                {"generation_id": str(generation_id)},
            )
        if not any(staging.iterdir()):
            raise RuntimeError("supporting artifact staging produced no files")
        if write_inventory:
            write_artifact_inventory(staging)
        if backup.exists():
            shutil.rmtree(backup)
        if directory.exists():
            os.replace(directory, backup)
        try:
            os.replace(staging, directory)
        except BaseException:
            if backup.exists() and not directory.exists():
                os.replace(backup, directory)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _path_exists(path: Path) -> bool:
    """Return true for regular paths and dangling symlinks."""

    return path.exists() or path.is_symlink()


def _remove_path(path: Path) -> None:
    """Remove one explicitly resolved publication path."""

    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _verify_staged_evidence_artifacts(
    payload: dict[str, Any],
    directory: Path,
) -> None:
    """Fail closed when a v4 payload does not match its staged evidence bytes."""

    result_version = payload.get("meta", {}).get("result_version")
    if result_version not in {
        "weekly-regime-result-v4",
        "weekly-regime-result-v5",
    }:
        return
    evidence = payload["model"]["evidence_artifacts"]
    keys = (
        ("state_label_history", "weekly_state_forecasts")
        if result_version == "weekly-regime-result-v4"
        else ("state_membership_history", "weekly_state_forecasts")
    )
    for key in keys:
        metadata = evidence[key]
        path = directory / str(metadata["path"])
        if not path.is_file():
            raise RuntimeError(f"staged evidence artifact is missing: {path.name}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != str(metadata["sha256"]):
            raise RuntimeError(f"staged evidence artifact hash mismatch: {path.name}")


def _verify_staged_research_artifacts(
    payload: dict[str, Any],
    directory: Path,
) -> None:
    if payload.get("meta", {}).get("result_version") != "weekly-regime-result-v5":
        return
    verify_staged_v5_research_artifacts(
        payload["model"]["research_artifacts"],
        directory,
    )


def _verify_staged_core_artifacts(
    payload: dict[str, Any],
    directory: Path,
) -> None:
    if payload.get("meta", {}).get("result_version") != "weekly-regime-result-v5":
        return
    verify_staged_v5_core_artifacts(
        payload["model"]["core_artifacts"],
        directory,
    )


def _verify_staged_feature_quality_artifact(
    payload: dict[str, Any],
    directory: Path,
) -> None:
    if payload.get("meta", {}).get("result_version") != "weekly-regime-result-v5":
        return
    manifest = payload.get("model", {}).get("feature_quality_artifact")
    if manifest is None:
        return
    verify_feature_quality_artifact(
        manifest,
        directory,
    )


def _publish_active_generation(
    payload: dict[str, Any],
    benchmark: object,
    *,
    output: Path,
    artifacts: Path,
    input_snapshot_sha256: str | None = None,
    finalization: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Publish one payload/artifact generation with rollback-safe cutover.

    Both outputs are fully serialized in private sibling directories before an
    active path is touched.  Cutover first moves *both* previous outputs into
    transaction-specific recovery directories, so an interruption can expose
    either the complete old generation, no complete generation, or the complete
    new generation -- never a new artifact directory beside an old payload.
    Ordinary exceptions roll both paths back.  If rollback itself fails, the
    recovery directories are deliberately retained and named in the error.
    """

    output = _mutable_path(output, label="dashboard output")
    artifacts = _mutable_path(artifacts, label="artifact output")
    manifest_path = output.with_name("generation-manifest.json")
    output_absolute = output.absolute()
    artifacts_absolute = artifacts.absolute()
    if _paths_overlap(output_absolute, artifacts_absolute):
        raise ValueError(
            "dashboard payload and artifacts must not overlap; dashboard payload "
            "must not be stored inside artifacts"
        )

    validate_dashboard_payload(payload)
    generation_id = str(payload.get("meta", {}).get("generation_id", ""))
    if not generation_id:
        raise ValueError("payload meta.generation_id must be non-empty")

    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts.parent.mkdir(parents=True, exist_ok=True)
    artifact_transaction = Path(
        tempfile.mkdtemp(
            prefix=f".{artifacts.name}-publish-",
            dir=artifacts.parent,
        )
    )
    payload_transaction = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}-publish-",
            dir=output.parent,
        )
    )
    staged_artifacts = artifact_transaction / "next"
    previous_artifacts = artifact_transaction / "previous"
    staged_payload = payload_transaction / "next.json"
    previous_payload = payload_transaction / "previous.json"
    staged_manifest = payload_transaction / "generation-manifest.json"
    previous_manifest = payload_transaction / "previous-generation-manifest.json"

    moved_previous_artifacts = False
    moved_previous_payload = False
    moved_previous_manifest = False
    published_artifacts = False
    published_payload = False
    published_manifest = False
    preserve_recovery = False
    try:
        is_v5 = (
            payload.get("meta", {}).get("result_version")
            == "weekly-regime-result-v5"
        )
        if is_v5:
            _write_supporting_results(
                benchmark,
                staged_artifacts,
                generation_id=generation_id,
                write_inventory=True,
                selection_context=payload,
            )
            verify_artifact_inventory(staged_artifacts)
        else:
            _write_supporting_results(
                benchmark,
                staged_artifacts,
                generation_id=generation_id,
            )
        _verify_staged_evidence_artifacts(payload, staged_artifacts)
        _verify_staged_research_artifacts(payload, staged_artifacts)
        _verify_staged_core_artifacts(payload, staged_artifacts)
        _verify_staged_feature_quality_artifact(payload, staged_artifacts)
        if is_v5:
            snapshot_sha256 = input_snapshot_sha256 or canonical_json_sha256_v1(
                {
                    "mode": payload.get("meta", {}).get("mode"),
                    "data_as_of": payload.get("meta", {}).get("data_as_of"),
                    "sources": payload.get("sources", ()),
                }
            )
            selection_family_path = staged_artifacts / "selection-family-audit.json"
            try:
                selection_family = json.loads(
                    selection_family_path.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    "staged selection-family audit cannot be read"
                ) from exc
            if not isinstance(selection_family, Mapping):
                raise RuntimeError("staged selection-family audit must be an object")
            mcs = (
                selection_family.get("supplemental_evaluation", {})
                .get("model_confidence_set", {})
            )
            retained_models = (
                mcs.get("retained_models") if isinstance(mcs, Mapping) else None
            )
            if isinstance(retained_models, list) and retained_models:
                payload_selection = payload.get("selection")
                if isinstance(payload_selection, dict) and (
                    "statistically_indistinguishable_models" in payload_selection
                ):
                    payload_selection[
                        "statistically_indistinguishable_models"
                    ] = [str(name) for name in retained_models]
                    payload_selection[
                        "statistical_equivalence_status"
                    ] = "completed_selection_mcs"
            generation_manifest = build_generation_manifest(
                payload=payload,
                payload_path=output,
                artifact_directory=staged_artifacts,
                input_snapshot={
                    "data_as_of": payload["meta"]["data_as_of"],
                    "sha256": snapshot_sha256,
                },
                label_spec_path=default_label_spec_path(),
                selection_family=selection_family,
                selection_family_path=artifacts / "selection-family-audit.json",
            )
            payload = bind_payload_to_generation_manifest(
                payload,
                generation_manifest,
            )
            validate_dashboard_payload(payload)
            write_json_atomic(staged_manifest, generation_manifest)
        write_dashboard_payload(payload, staged_payload)

        # Move both previous outputs away before installing either new output.
        # A hard interruption during cutover therefore fails closed rather than
        # pairing one new side with one old side.
        if _path_exists(artifacts):
            os.replace(artifacts, previous_artifacts)
            moved_previous_artifacts = True
        if _path_exists(output):
            os.replace(output, previous_payload)
            moved_previous_payload = True
        if is_v5 and _path_exists(manifest_path):
            os.replace(manifest_path, previous_manifest)
            moved_previous_manifest = True

        os.replace(staged_artifacts, artifacts)
        published_artifacts = True
        os.replace(staged_payload, output)
        published_payload = True
        if is_v5:
            os.replace(staged_manifest, manifest_path)
            published_manifest = True
            validate_generation_manifest(
                manifest_path,
                require_comparison=False,
                require_selection_family=True,
                require_artifacts=True,
                artifact_directory=artifacts,
            )
        if finalization is not None:
            finalization(payload)
    except BaseException as publication_error:
        try:
            if published_manifest and _path_exists(manifest_path):
                _remove_path(manifest_path)
            if published_payload and _path_exists(output):
                _remove_path(output)
            if published_artifacts and _path_exists(artifacts):
                _remove_path(artifacts)
            if moved_previous_artifacts and _path_exists(previous_artifacts):
                os.replace(previous_artifacts, artifacts)
                moved_previous_artifacts = False
            if moved_previous_payload and _path_exists(previous_payload):
                os.replace(previous_payload, output)
                moved_previous_payload = False
            if moved_previous_manifest and _path_exists(previous_manifest):
                os.replace(previous_manifest, manifest_path)
                moved_previous_manifest = False
        except BaseException as rollback_error:
            preserve_recovery = True
            raise RuntimeError(
                "active generation publication and rollback failed; recovery "
                f"paths retained at {artifact_transaction} and "
                f"{payload_transaction}; publication error: "
                f"{type(publication_error).__name__}: {publication_error}"
            ) from rollback_error
        raise
    finally:
        if not preserve_recovery:
            shutil.rmtree(artifact_transaction, ignore_errors=True)
            shutil.rmtree(payload_transaction, ignore_errors=True)
    return payload


def command_collect_h10(args: argparse.Namespace) -> int:
    """Collect one prospective H.10 snapshot without training or publication."""

    if getattr(args, "contract", None) != "v5":
        raise SystemExit("H.10 collection requires explicit --contract v5")
    try:
        verify_provider_rights(
            ("frb_h10",),
            policy_path=project_root() / "config/provider_rights.json",
            capabilities=("collection", "local_storage"),
        )
    except ProviderRightsError as exc:
        raise SystemExit(str(exc)) from exc
    database = _mutable_path(args.database, label="snapshot database")
    archive_ingest = bool(
        getattr(args, "official_release_archive_ingest", False)
    )
    archive_start = getattr(args, "archive_start", None)
    archive_through = getattr(args, "archive_through", None)
    if not archive_ingest and (
        archive_start is not None or archive_through is not None
    ):
        raise SystemExit(
            "archive date bounds require --official-release-archive-ingest"
        )
    requested_at = datetime.now(timezone.utc)
    as_of = getattr(args, "as_of", None) or last_completed_week_cutoff(
        requested_at
    )
    if as_of > requested_at:
        raise SystemExit("H.10 as-of cutoff must not be in the future")
    if as_of != last_completed_week_cutoff(as_of):
        raise SystemExit(
            "H.10 as-of must be an exact completed Friday 16:00 ET cutoff"
        )
    receipt_value = getattr(args, "receipt", None)
    if receipt_value is None and archive_ingest:
        receipt_value = _V5_H10_ARCHIVE_DEFAULT_RECEIPT
    receipt = _resolve_h10_collection_receipt(
        receipt_value,
        database=database,
    )
    live_build_lock = database.with_name(f"{database.name}.live-build.lock")
    try:
        with automation_lock(live_build_lock):
            _backup_database_before_mutation(
                database,
                backup_directory=getattr(args, "backup_directory", None),
                source_code_fingerprint_sha256=getattr(
                    args,
                    "backup_source_code_fingerprint_sha256",
                    None,
                ),
            )
            from regime_lab.data import (
                H10ArchiveClient,
                H10Client,
                SQLiteSnapshotStore,
            )
            from regime_lab.h10_store import (
                h10_collection_receipt_document,
                refresh_h10_archive_store,
                refresh_h10_store,
            )

            with SQLiteSnapshotStore(database) as store:
                if archive_ingest:
                    refresh = refresh_h10_archive_store(
                        store,
                        H10ArchiveClient(),
                        requested_at=requested_at,
                        as_of=as_of,
                        start_date=(
                            archive_start
                            if archive_start is not None
                            else date(2022, 1, 1)
                        ),
                        end_date=archive_through,
                    )
                else:
                    refresh = refresh_h10_store(
                        store,
                        H10Client(),
                        requested_at=requested_at,
                        as_of=as_of,
                    )
            document = h10_collection_receipt_document(
                refresh,
                requested_at=requested_at,
                as_of=as_of,
            )
            write_json_atomic(receipt, document)
    except AlreadyRunning as exc:
        raise SystemExit(
            "H.10 collection refused because another build owns "
            f"{live_build_lock}: {exc}"
        ) from exc
    print(
        json.dumps(
            {**document, "receipt": str(receipt)},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


def command_collect_ofr_fsi(args: argparse.Namespace) -> int:
    """Collect one private prospective OFR FSI shadow snapshot only."""

    if getattr(args, "contract", None) != "v6":
        raise SystemExit("OFR FSI collection requires explicit --contract v6")
    try:
        verify_provider_rights(
            ("ofr_fsi",),
            policy_path=project_root() / "config/provider_rights.json",
            capabilities=("collection", "local_storage"),
        )
    except ProviderRightsError as exc:
        raise SystemExit(str(exc)) from exc
    database, receipt, collection_lock = _resolve_ofr_fsi_collection_targets(
        database_value=getattr(args, "database", None),
        receipt_value=getattr(args, "receipt", None),
    )
    requested_at = datetime.now(timezone.utc)
    as_of = getattr(args, "as_of", None)
    if as_of is not None and as_of > requested_at:
        raise SystemExit("OFR FSI as-of cutoff must not be in the future")
    try:
        with automation_lock(collection_lock):
            _backup_database_before_mutation(
                database,
                backup_directory=getattr(args, "backup_directory", None),
                source_code_fingerprint_sha256=getattr(
                    args,
                    "backup_source_code_fingerprint_sha256",
                    None,
                ),
            )
            from regime_lab.data import (
                OFRFSIClient,
                OFRFSIConfig,
                SQLiteSnapshotStore,
                load_ofr_fsi_contract,
            )
            from regime_lab.ofr_fsi_store import (
                ofr_fsi_collection_receipt_document,
                refresh_ofr_fsi_store,
            )

            contract = load_ofr_fsi_contract()
            client = OFRFSIClient(OFRFSIConfig(contract))
            with SQLiteSnapshotStore(database) as store:
                refresh = refresh_ofr_fsi_store(
                    store,
                    client,
                    requested_at=requested_at,
                    as_of=as_of,
                )
            document = ofr_fsi_collection_receipt_document(
                refresh,
                requested_at=requested_at,
                as_of=refresh.as_of,
            )
            write_json_atomic(receipt, document)
    except AlreadyRunning as exc:
        raise SystemExit(
            "OFR FSI collection refused because another shadow collection owns "
            f"{collection_lock}: {exc}"
        ) from exc
    print(
        json.dumps(
            {**document, "receipt": str(receipt)},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


def _h10_collection_report_source(*, status: str, issue: str | None = None) -> dict[str, Any]:
    """Return a fixed, secret-free H.10 source state for the local build receipt."""

    return {
        "id": "frb_h10",
        "name": "Federal Reserve H.10 foreign exchange rates",
        "status": status,
        "issues": [issue] if issue is not None else [],
    }


def _h10_training_gate(source: object) -> str | None:
    """Require a complete H.10 refresh before an expensive V5 analysis starts."""

    if not isinstance(source, Mapping) or source.get("id") != "frb_h10":
        return "frb_h10 source result is missing"
    if source.get("status") != "ok":
        return "frb_h10 source health is not ok"
    issues = source.get("issues")
    if not isinstance(issues, list) or any(not isinstance(issue, str) for issue in issues):
        return "frb_h10 source issues are invalid"
    if issues:
        return "frb_h10 source issues must be empty"
    return None


def _v5_collection_report_document(
    collection: Any,
    *,
    expected_cutoff: datetime,
    h10_source: Mapping[str, Any],
    gate_error: str | None,
) -> dict[str, Any]:
    """Extend the core-provider receipt with the pending or final H.10 state."""

    document = collection_report_document(
        collection,
        expected_cutoff=expected_cutoff,
        gate_error=gate_error,
    )
    source = dict(h10_source)
    document["sources"] = [
        *(
            row
            for row in document["sources"]
            if row.get("id") != "frb_h10"
        ),
        source,
    ]
    source_issues = source.get("issues")
    if isinstance(source_issues, list):
        document["issues"] = list(
            dict.fromkeys([*document["issues"], *source_issues])
        )
    h10_status = source.get("status")
    document["ready_for_training"] = gate_error is None and h10_status == "ok"
    if h10_status == "pending":
        document["overall_health"] = "pending"
    elif h10_status not in {"ok", "not_attempted"}:
        document["overall_health"] = "degraded"
    return document


def command_build(args: argparse.Namespace) -> int:
    if args.profile == "quick":
        raise SystemExit(
            "live build does not permit the three-origin quick smoke profile; "
            "use standard or full"
        )
    contract_version = getattr(args, "contract", "v5")
    output, artifacts = _resolve_contract_write_targets(
        command="build",
        contract_version=contract_version,
        output=getattr(args, "output", None),
        artifacts=getattr(args, "artifacts", None),
    )
    checkpoint_directory = _resolve_live_checkpoint_directory(
        contract_version=contract_version,
        output=output,
        artifacts=artifacts,
        value=getattr(args, "checkpoint_directory", None),
    )
    config = load_config(args.config)
    rights_policy_value = config.get(
        "provider_rights_policy",
        "config/provider_rights.json",
    )
    rights_policy_path = Path(str(rights_policy_value))
    if not rights_policy_path.is_absolute():
        rights_policy_path = project_root() / rights_policy_path
    try:
        required_provider_ids = list(providers_for_live_config(config))
        if contract_version == "v5" and "frb_h10" not in required_provider_ids:
            required_provider_ids.append("frb_h10")
        verify_provider_rights(
            required_provider_ids,
            policy_path=rights_policy_path,
            capabilities=("collection", "local_storage", "model_training"),
        )
    except ProviderRightsError as exc:
        raise SystemExit(str(exc)) from exc
    database = _mutable_path(args.database, label="snapshot database")
    raw_report = getattr(args, "collection_report", None)
    collection_report = (
        _mutable_path(raw_report, label="collection report")
        if raw_report is not None
        else None
    )
    build_started_at = datetime.now(timezone.utc)
    run_id = build_started_at.strftime("%Y%m%dT%H%M%S.%fZ-live-build")
    configured_run_registry = getattr(args, "run_registry", None)
    run_registry = _mutable_path(
        output.parent / "run-registry.jsonl"
        if configured_run_registry is None
        else configured_run_registry,
        label="run registry",
    )
    append_run_event(
        run_registry,
        run_id=run_id,
        status="started",
        occurred_at=build_started_at,
        detail={"contract": contract_version, "profile": args.profile},
    )
    expected_cutoff = getattr(args, "expected_cutoff", None) or (
        last_completed_week_cutoff(build_started_at)
    )
    live_build_lock = database.with_name(f"{database.name}.live-build.lock")
    try:
        with automation_lock(live_build_lock):
            v5_preflight = None
            if contract_version == "v5":
                print("V5 실행 사전점검", flush=True)
                v5_preflight = verify_v5_preflight(
                    profile=args.profile,
                    database_path=database,
                    config=config,
                )
                print(
                    "V5 사전점검 완료: "
                    f"source={v5_preflight['source_fingerprint_sha256'][:12]}",
                    flush=True,
                )
            append_run_event(
                run_registry,
                run_id=run_id,
                status="collecting",
                detail={"expected_cutoff": expected_cutoff.isoformat()},
            )
            _backup_database_before_mutation(
                database,
                backup_directory=getattr(args, "backup_directory", None),
                source_code_fingerprint_sha256=getattr(
                    args,
                    "backup_source_code_fingerprint_sha256",
                    None,
                ),
            )
            credentials = (
                nullcontext()
                if args.from_env
                else provider_environment_from_keychain(rights_acknowledged=True)
            )
            with credentials:
                collection = collect_live_data(
                    config,
                    database_path=database,
                    now=build_started_at,
                    expected_cutoff=expected_cutoff,
                    progress=_flush_progress,
                )
            gate_error: CollectionGateError | AnalysisPreconditionError | None = None
            try:
                validate_collection_for_training(
                    collection,
                    expected_cutoff=expected_cutoff,
                )
            except CollectionGateError as exc:
                gate_error = exc
            if gate_error is None:
                try:
                    _require_ac_power_before_analysis(
                        enabled=bool(getattr(args, "require_ac_power", False))
                    )
                except AnalysisPreconditionError as exc:
                    gate_error = exc
            if collection_report is not None:
                report_gate_error = str(gate_error) if gate_error else None
                report_document = collection_report_document(
                    collection,
                    expected_cutoff=expected_cutoff,
                    gate_error=report_gate_error,
                )
                if contract_version == "v5":
                    report_document = _v5_collection_report_document(
                        collection,
                        expected_cutoff=expected_cutoff,
                        h10_source=_h10_collection_report_source(
                            status=("not_attempted" if gate_error else "pending")
                        ),
                        gate_error=(
                            report_gate_error
                            if gate_error
                            else "Federal Reserve H.10 collection is pending"
                        ),
                    )
                write_json_atomic(collection_report, report_document)
            if gate_error is not None:
                raise SystemExit(
                    "live build stopped before analysis: "
                    f"{gate_error}"
                )
            fx_result = None
            latest_fx_context = None
            h10_source = None
            additional_operational_records: tuple[object, ...] = ()
            if contract_version == "v5":
                from regime_lab.data import H10Client, SQLiteSnapshotStore
                from regime_lab.h10_store import refresh_h10_store

                print("Federal Reserve H.10 FX snapshot 갱신", flush=True)
                try:
                    with SQLiteSnapshotStore(database) as h10_store:
                        h10_refresh = refresh_h10_store(
                            h10_store,
                            H10Client(),
                            requested_at=datetime.now(timezone.utc),
                            as_of=expected_cutoff,
                        )
                except Exception:
                    if collection_report is not None:
                        write_json_atomic(
                            collection_report,
                            _v5_collection_report_document(
                                collection,
                                expected_cutoff=expected_cutoff,
                                h10_source=_h10_collection_report_source(
                                    status="unavailable",
                                    issue="h10_refresh_failed_before_training",
                                ),
                                gate_error="frb_h10 refresh failed before training",
                            ),
                        )
                    raise
                h10_source = h10_refresh.source_row
                # Older frozen/replay adapters did not expose the bitemporal
                # record collection.  They remain valid for reproducing those
                # runs; live refresh results provide the field and therefore
                # participate in the operational first-seen snapshot.
                additional_operational_records = tuple(
                    getattr(h10_refresh, "effective_records", ())
                )
                h10_gate_error = _h10_training_gate(h10_source)
                if collection_report is not None:
                    write_json_atomic(
                        collection_report,
                        _v5_collection_report_document(
                            collection,
                            expected_cutoff=expected_cutoff,
                            h10_source=(
                                h10_source
                                if isinstance(h10_source, Mapping)
                                else _h10_collection_report_source(
                                    status="unavailable",
                                    issue="h10_source_result_missing",
                                )
                            ),
                            gate_error=h10_gate_error,
                        ),
                    )
                fx_result = h10_refresh.fx_features
                latest_fx_context = h10_refresh.fx_context
                if h10_gate_error is not None:
                    raise SystemExit(
                        "live build stopped before analysis: "
                        f"{h10_gate_error}"
                    )
            print("Point-in-time weekly frame 조립", flush=True)
            dataset = build_weekly_dataset(
                config,
                collection.cutoffs,
                collection.records,
                # Historical Alpha Vantage rows are a current-adjusted
                # backfill, so the walk-forward evidence remains explicitly
                # reconstructed OOS.  The current live decision is bound to
                # actual first-seen/retrieval clocks below in ForecastLedger.
                availability_basis="reconstructed_market",
            )
            print(
                f"모델 비교 시작: {len(dataset.features):,} weeks × "
                f"{dataset.features.shape[1]:,} features ({args.profile})",
                flush=True,
            )
            append_run_event(
                run_registry,
                run_id=run_id,
                status="analyzing",
                detail={"weeks": len(dataset.features)},
            )
            payload, benchmark = build_dashboard_result(
                dataset,
                collection,
                profile_name=args.profile,
                mode="live",
                selection_end=str(config["model"]["final_holdout_start"]),
                progress=_flush_progress,
                contract_version=contract_version,
                fx_result=fx_result,
                latest_fx_context=latest_fx_context,
                h10_source=h10_source,
                checkpoint_directory=checkpoint_directory,
                source_fingerprint_sha256=(
                    None
                    if v5_preflight is None
                    else str(v5_preflight["source_fingerprint_sha256"])
                ),
                minimum_log_loss_improvement=(
                    config["model"].get("minimum_log_loss_improvement")
                    if contract_version == "v5"
                    else None
                ),
            )
            if v5_preflight is not None:
                require_v5_analysis_source_unchanged(
                    str(v5_preflight["source_fingerprint_sha256"]),
                    config=config,
                )
            forecast_contract = payload["forecast"]
            decision_value = forecast_contract.get("decision_at")
            if decision_value is None:
                raise ValueError("expired forecast cannot bind operational inputs")
            operational_inputs = _operational_inputs_for_generation(
                dataset,
                additional_records=additional_operational_records,
                origin_at=datetime.fromisoformat(
                    str(forecast_contract["origin_at"])
                ),
                decision_at=datetime.fromisoformat(str(decision_value)),
            )
            input_snapshot_sha256 = operational_input_manifest_sha256(
                operational_inputs
            )
            configured_forecast_ledger = getattr(args, "forecast_ledger", None)
            forecast_ledger_path = _mutable_path(
                output.parent / "forecast-ledger.sqlite3"
                if configured_forecast_ledger is None
                else configured_forecast_ledger,
                label="forecast ledger",
            )
            prospective_key = ForecastLedgerKey(
                origin_week=date.fromisoformat(str(payload["weekly"][-1]["date"])),
                decision_at=datetime.fromisoformat(str(forecast_contract["decision_at"])),
                target_at=datetime.fromisoformat(str(forecast_contract["target_at"])),
                label_spec_sha256=str(payload["label"]["spec_sha256"]),
                model_manifest_sha256=str(
                    payload["model"]["candidate_manifest_sha256"]
                ),
                input_snapshot_sha256=input_snapshot_sha256,
            )
            with ForecastLedger(forecast_ledger_path) as ledger:
                ledger_summary = ledger.public_summary(
                    pending_key=prospective_key
                )
                payload["forecast"]["prospective_ledger"] = ledger_summary
                decision_shadow = payload.get("research", {}).get(
                    "prospective_decision_shadow"
                )
                if isinstance(decision_shadow, dict) and isinstance(
                    decision_shadow.get("prospective_ledger"), dict
                ):
                    decision_shadow["prospective_ledger"].update(
                        {
                            "status": "ledger_recorded_outcomes_pending",
                            "ledger_entry_count": ledger_summary["entry_count"],
                        }
                    )

            def append_forecast_ledger(bound_payload: dict[str, Any]) -> None:
                entry = _forecast_ledger_entry(
                    bound_payload,
                    operational_inputs=operational_inputs,
                    published_at=datetime.now(timezone.utc),
                )
                with ForecastLedger(forecast_ledger_path) as ledger:
                    ledger.append(entry)

            payload = _publish_active_generation(
                payload,
                benchmark,
                output=output,
                artifacts=artifacts,
                input_snapshot_sha256=input_snapshot_sha256,
                finalization=(
                    append_forecast_ledger if contract_version == "v5" else None
                ),
            )
            append_run_event(
                run_registry,
                run_id=run_id,
                status="completed",
                generation_id=str(payload["meta"]["generation_id"]),
                detail={"data_as_of": str(payload["meta"]["data_as_of"])},
            )
    except AlreadyRunning as exc:
        if current_run_status(run_registry, run_id) not in {"interrupted", "completed"}:
            append_run_event(
                run_registry,
                run_id=run_id,
                status="interrupted",
                detail={"error_type": type(exc).__name__},
            )
        if collection_report is not None:
            write_json_atomic(
                collection_report,
                {
                    "schema_version": 1,
                    "expected_cutoff": expected_cutoff.isoformat(),
                    "model_cutoff": expected_cutoff.isoformat(),
                    "ready_for_training": False,
                    "overall_health": "unknown",
                    "issues": [],
                    "sources": [],
                    "gate_error": "live build database lock is busy",
                    "error_code": "database_build_lock_busy",
                },
            )
        raise SystemExit(
            f"live build refused because another build owns {live_build_lock}: {exc}"
        ) from exc
    except BaseException as exc:
        if current_run_status(run_registry, run_id) not in {"interrupted", "completed"}:
            append_run_event(
                run_registry,
                run_id=run_id,
                status="interrupted",
                detail={"error_type": type(exc).__name__},
            )
        raise
    print(
        json.dumps(
            {
                "output": str(output),
                "database": str(database),
                "mode": payload["meta"]["mode"],
                "status": payload["meta"]["status"],
                "data_as_of": payload["meta"]["data_as_of"],
                "weeks": len(payload["weekly"]),
                "features": dataset.features.shape[1],
                "champion": benchmark.champion,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


def command_demo(args: argparse.Namespace) -> int:
    contract_version = getattr(args, "contract", "v5")
    output, artifacts = _resolve_contract_write_targets(
        command="demo",
        contract_version=contract_version,
        output=getattr(args, "output", None),
        artifacts=getattr(args, "artifacts", None),
    )
    config = load_config(args.config)
    print("고정 seed 모의자료로 전체 분석 경로 실행", flush=True)
    payload, benchmark = generate_demo_payload(
        config,
        profile_name=args.profile,
        progress=_flush_progress,
        contract_version=contract_version,
    )
    _publish_active_generation(
        payload,
        benchmark,
        output=output,
        artifacts=artifacts,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "mode": "demo",
                "weeks": len(payload["weekly"]),
                "champion": benchmark.champion,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_validate(args: argparse.Namespace) -> int:
    path = _root_path(args.path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_dashboard_payload(payload)
    print(
        json.dumps(
            {
                "valid": True,
                "path": str(path),
                "mode": payload["meta"]["mode"],
                "weeks": len(payload["weekly"]),
                "data_as_of": payload["meta"]["data_as_of"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_serve(args: argparse.Namespace) -> int:
    payload = _root_path(args.payload)
    comparison_value = getattr(args, "comparison", None)
    if comparison_value is None:
        sibling = payload.with_name("v5-vs-v4-comparison.json")
        comparison = sibling if sibling.is_file() and not sibling.is_symlink() else None
    else:
        comparison = _root_path(comparison_value)
    payload_raw = payload.read_bytes()
    payload_document = json.loads(payload_raw)
    validate_dashboard_payload(payload_document)
    comparison_raw = None
    if comparison is not None:
        if payload_document.get("meta", {}).get("result_version") != V5_RESULT_VERSION:
            raise ValueError("a V5/V4 comparison sidecar requires a V5 payload")
        comparison_raw = comparison.read_bytes()
        comparison_document = json.loads(comparison_raw)
        validate_v5_comparison_sidecar(
            comparison_document,
            payload=payload_document,
            payload_raw=payload_raw,
        )
    serve_dashboard(
        _root_path(args.web_root),
        host=args.host,
        port=args.port,
        payload_bytes=payload_raw,
        comparison_bytes=comparison_raw,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="regime-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="collect live PIT data and train models")
    build.add_argument("--config", type=Path, default=default_config_path())
    build.add_argument("--database", default="data/regime.sqlite3")
    build.add_argument(
        "--backup-directory",
        default=None,
        help="verified SQLite backup root (default: <database>.backups)",
    )
    build.add_argument(
        "--backup-source-code-fingerprint-sha256",
        default=None,
        help=argparse.SUPPRESS,
    )
    build.add_argument(
        "--output",
        default=None,
        help="payload target (v4: web/data; v5: build/v5-live)",
    )
    build.add_argument(
        "--artifacts",
        default=None,
        help="artifact target (v4: artifacts/latest; v5: build/v5-live)",
    )
    build.add_argument("--profile", choices=("standard", "full"), default="standard")
    build.add_argument(
        "--contract",
        choices=("v4", "v5"),
        default="v5",
        help="result contract (default: active V5; V4 is frozen regression only)",
    )
    build.add_argument(
        "--expected-cutoff",
        type=_aware_datetime_argument,
        help="require collection to match this exact ISO-8601 weekly cutoff",
    )
    build.add_argument(
        "--collection-report",
        help="atomically write a secret-free collection health receipt",
    )
    build.add_argument(
        "--run-registry",
        default=None,
        help="append-only local run lifecycle registry (default: payload sibling)",
    )
    build.add_argument(
        "--forecast-ledger",
        default=None,
        help=(
            "append-only operational forecast ledger "
            "(default: payload sibling forecast-ledger.sqlite3)"
        ),
    )
    build.add_argument(
        "--checkpoint-directory",
        default=None,
        help=(
            "private V5 base walk-forward checkpoint directory "
            "(default: output sibling .private-checkpoints/base-walk-forward)"
        ),
    )
    build.add_argument(
        "--require-ac-power",
        action="store_true",
        help="on macOS, recheck AC power after collection and before analysis",
    )
    build.add_argument("--from-env", action="store_true", help="use existing process environment instead of macOS Keychain")
    build.add_argument("--alfred-rights-confirmed", action="store_true")
    build.set_defaults(func=command_build)

    collect_h10 = subparsers.add_parser(
        "collect-h10",
        help="collect one isolated prospective Fed H.10 snapshot for v5 research",
    )
    collect_h10.add_argument(
        "--contract",
        choices=("v5",),
        required=True,
        help="explicitly opt in to the local v5 research contract",
    )
    collect_h10.add_argument("--database", default="data/regime.sqlite3")
    collect_h10.add_argument(
        "--backup-directory",
        default=None,
        help="verified SQLite backup root (default: <database>.backups)",
    )
    collect_h10.add_argument(
        "--backup-source-code-fingerprint-sha256",
        default=None,
        help=argparse.SUPPRESS,
    )
    collect_h10.add_argument(
        "--receipt",
        default=None,
        help="derived-only receipt target (default: build/v5-h10)",
    )
    collect_h10.add_argument(
        "--as-of",
        type=_aware_datetime_argument,
        help="evaluate stored H.10 availability at this ISO-8601 cutoff",
    )
    collect_h10.add_argument(
        "--official-release-archive-ingest",
        action="store_true",
        help="ingest the official release archive into its isolated v5 dataset",
    )
    collect_h10.add_argument(
        "--archive-start",
        type=_date_argument,
        help="archive release-event start date (minimum/default: 2022-01-01)",
    )
    collect_h10.add_argument(
        "--archive-through",
        type=_date_argument,
        help="archive release-event end date (default: the as-of date)",
    )
    collect_h10.set_defaults(func=command_collect_h10)

    collect_ofr_fsi = subparsers.add_parser(
        "collect-ofr-fsi",
        help="collect one isolated prospective OFR FSI shadow snapshot for v6",
    )
    collect_ofr_fsi.add_argument(
        "--contract",
        choices=("v6",),
        required=True,
        help="explicitly opt in to the private v6 prospective-shadow contract",
    )
    collect_ofr_fsi.add_argument(
        "--database",
        default=None,
        help="private append-only store (default: data/ofr-fsi-shadow.sqlite3)",
    )
    collect_ofr_fsi.add_argument(
        "--backup-directory",
        default=None,
        help="verified SQLite backup root (default: <database>.backups)",
    )
    collect_ofr_fsi.add_argument(
        "--backup-source-code-fingerprint-sha256",
        default=None,
        help=argparse.SUPPRESS,
    )
    collect_ofr_fsi.add_argument(
        "--receipt",
        default=None,
        help="value-free local receipt (default: build/v6-ofr-fsi)",
    )
    collect_ofr_fsi.add_argument(
        "--as-of",
        type=_aware_datetime_argument,
        help="evaluate first-seen eligibility at this ISO-8601 cutoff",
    )
    collect_ofr_fsi.set_defaults(func=command_collect_ofr_fsi)

    demo = subparsers.add_parser("demo", help="generate a clearly labelled synthetic result")
    demo.add_argument("--config", type=Path, default=default_config_path())
    demo.add_argument(
        "--output",
        default=None,
        help="payload target (v4: web/data; v5: build/v5-demo)",
    )
    demo.add_argument(
        "--artifacts",
        default=None,
        help="artifact target (v4: artifacts/demo; v5: build/v5-demo)",
    )
    demo.add_argument("--profile", choices=("quick", "standard"), default="quick")
    demo.add_argument(
        "--contract",
        choices=("v4", "v5"),
        default="v5",
        help="result contract (default: active V5; V4 is frozen regression only)",
    )
    demo.set_defaults(func=command_demo)

    validate = subparsers.add_parser("validate", help="validate a dashboard result JSON")
    validate.add_argument(
        "path",
        nargs="?",
        default="publication/live/regime-results.json",
    )
    validate.set_defaults(func=command_validate)

    serve = subparsers.add_parser("serve", help="serve the static local dashboard")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--web-root", default="web")
    serve.add_argument(
        "--payload",
        default="publication/live/regime-results.json",
        help="dashboard payload (default: reviewed live V5 publication)",
    )
    serve.add_argument(
        "--comparison",
        default=None,
        help="optional V5/V4 sidecar (default: payload sibling when present)",
    )
    serve.set_defaults(func=command_serve)

    smoke = subparsers.add_parser("smoke", help="run a bounded provider smoke test")
    smoke.add_argument("provider", choices=("all", "alfred", "alpha_vantage"), default="all", nargs="?")
    smoke.add_argument("--database", default="data/regime.sqlite3")
    smoke.add_argument("--alfred-rights-confirmed", action="store_true")
    smoke.set_defaults(
        func=lambda args: smoke_main(
            [
                args.provider,
                "--budget-database",
                str(_mutable_path(args.database, label="smoke budget database")),
                *(
                    ["--alfred-rights-confirmed"]
                    if args.alfred_rights_confirmed
                    else []
                ),
            ]
        )
    )

    automation = subparsers.add_parser(
        "automation",
        help="preflight, run, or manage the local weekly release automation",
    )
    automation.add_argument(
        "action",
        choices=("preflight", "run", "install", "uninstall", "status"),
    )
    automation.add_argument(
        "--config",
        type=Path,
        default=Path("config/automation.json"),
    )
    automation.add_argument(
        "--alfred-rights-confirmed",
        action="store_true",
        help="confirm ALFRED local storage and ML-training permission for install",
    )
    automation.add_argument(
        "--acknowledge-personal-noncommercial-publication",
        action="store_true",
        help="confirm personal noncommercial derived-output publication for install",
    )
    automation.add_argument(
        "--force-retry",
        action="store_true",
        help=(
            "for automation run only, bypass a transient retry delay; quota "
            "and blocked guards remain enforced"
        ),
    )
    automation.add_argument(
        "--force-blocked-recovery",
        action="store_true",
        help=(
            "for automation run only, explicitly retry a repaired blocked "
            "state while retaining every ordinary preflight"
        ),
    )
    automation.set_defaults(func=command_automation)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("중단됨", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
