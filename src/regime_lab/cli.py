"""Command-line entrypoints for collection, analysis, validation, and serving."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

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
from regime_lab.demo import generate_demo_payload
from regime_lab.evidence import (
    STATE_LABEL_HISTORY_COLUMNS,
    WEEKLY_STATE_FORECAST_COLUMNS,
    canonical_evidence_csv_bytes,
)
from regime_lab.keychain import provider_environment_from_keychain
from regime_lab.payload import write_dashboard_payload
from regime_lab.io import write_json_atomic
from regime_lab.path_safety import confined_mutable_path
from regime_lab.pipeline import build_dashboard_result
from regime_lab.schema import validate_dashboard_payload
from regime_lab.server import serve_dashboard
from regime_lab.smoke import main as smoke_main
from regime_lab.automation import AlreadyRunning, automation_lock, command_automation


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
    structural_forecasts = getattr(benchmark, "structural_forecasts", None)
    joint_survival_forecasts = getattr(
        benchmark, "joint_survival_forecasts", None
    )
    stacking_weights = getattr(benchmark, "stacking_weights", None)
    state_labels = getattr(benchmark, "state_label_history", None)
    weekly_forecasts = getattr(benchmark, "weekly_state_forecasts", None)
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
        for filename, frame in frames.items():
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
        write_json_atomic(staging / "candidate-manifest.json", manifest)
        if feature_manifest is not None:
            write_json_atomic(staging / "feature-manifest.json", feature_manifest)
        if feature_ablation is not None:
            write_json_atomic(
                staging / "feature-ablation-manifest.json",
                feature_ablation_manifest_document(feature_ablation.manifest),
            )
        if generation_id is not None:
            write_json_atomic(
                staging / "build-generation.json",
                {"generation_id": str(generation_id)},
            )
        if not any(staging.iterdir()):
            raise RuntimeError("supporting artifact staging produced no files")
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

    if payload.get("meta", {}).get("result_version") != "weekly-regime-result-v4":
        return
    evidence = payload["model"]["evidence_artifacts"]
    for key in ("state_label_history", "weekly_state_forecasts"):
        metadata = evidence[key]
        path = directory / str(metadata["path"])
        if not path.is_file():
            raise RuntimeError(f"staged evidence artifact is missing: {path.name}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != str(metadata["sha256"]):
            raise RuntimeError(f"staged evidence artifact hash mismatch: {path.name}")


def _publish_active_generation(
    payload: dict[str, Any],
    benchmark: object,
    *,
    output: Path,
    artifacts: Path,
) -> None:
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
    output_absolute = output.absolute()
    artifacts_absolute = artifacts.absolute()
    if output_absolute == artifacts_absolute or output_absolute.is_relative_to(
        artifacts_absolute
    ):
        raise ValueError("dashboard payload must not be stored inside artifacts")

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

    moved_previous_artifacts = False
    moved_previous_payload = False
    published_artifacts = False
    published_payload = False
    preserve_recovery = False
    try:
        _write_supporting_results(
            benchmark,
            staged_artifacts,
            generation_id=generation_id,
        )
        _verify_staged_evidence_artifacts(payload, staged_artifacts)
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

        os.replace(staged_artifacts, artifacts)
        published_artifacts = True
        os.replace(staged_payload, output)
        published_payload = True
    except BaseException as publication_error:
        try:
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


def command_build(args: argparse.Namespace) -> int:
    if not args.alfred_rights_confirmed:
        raise SystemExit(
            "live build requires --alfred-rights-confirmed after verifying "
            "storage, ML training, and derived-output permission"
        )
    if args.profile == "quick":
        raise SystemExit(
            "live build does not permit the three-origin quick smoke profile; "
            "use standard or full"
        )
    config = load_config(args.config)
    database = _mutable_path(args.database, label="snapshot database")
    output = _mutable_path(args.output, label="dashboard output")
    artifacts = _mutable_path(args.artifacts, label="artifact output")
    raw_report = getattr(args, "collection_report", None)
    collection_report = (
        _mutable_path(raw_report, label="collection report")
        if raw_report is not None
        else None
    )
    build_started_at = datetime.now(timezone.utc)
    expected_cutoff = getattr(args, "expected_cutoff", None) or (
        last_completed_week_cutoff(build_started_at)
    )
    live_build_lock = database.with_name(f"{database.name}.live-build.lock")
    try:
        with automation_lock(live_build_lock):
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
                write_json_atomic(
                    collection_report,
                    collection_report_document(
                        collection,
                        expected_cutoff=expected_cutoff,
                        gate_error=str(gate_error) if gate_error else None,
                    ),
                )
            if gate_error is not None:
                raise SystemExit(
                    "live build stopped before analysis: "
                    f"{gate_error}"
                )
            print("Point-in-time weekly frame 조립", flush=True)
            dataset = build_weekly_dataset(
                config, collection.cutoffs, collection.records
            )
            print(
                f"모델 비교 시작: {len(dataset.features):,} weeks × "
                f"{dataset.features.shape[1]:,} features ({args.profile})",
                flush=True,
            )
            payload, benchmark = build_dashboard_result(
                dataset,
                collection,
                profile_name=args.profile,
                mode="live",
                selection_end=str(config["model"]["final_holdout_start"]),
                progress=_flush_progress,
            )
            _publish_active_generation(
                payload,
                benchmark,
                output=output,
                artifacts=artifacts,
            )
    except AlreadyRunning as exc:
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
    config = load_config(args.config)
    output = _mutable_path(args.output, label="dashboard output")
    print("고정 seed 모의자료로 전체 분석 경로 실행", flush=True)
    payload, benchmark = generate_demo_payload(
        config,
        profile_name=args.profile,
        progress=_flush_progress,
    )
    _publish_active_generation(
        payload,
        benchmark,
        output=output,
        artifacts=_mutable_path(args.artifacts, label="artifact output"),
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="regime-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="collect live PIT data and train models")
    build.add_argument("--config", type=Path, default=default_config_path())
    build.add_argument("--database", default="data/regime.sqlite3")
    build.add_argument("--output", default="web/data/regime-results.json")
    build.add_argument("--artifacts", default="artifacts/latest")
    build.add_argument("--profile", choices=("standard", "full"), default="standard")
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
        "--require-ac-power",
        action="store_true",
        help="on macOS, recheck AC power after collection and before analysis",
    )
    build.add_argument("--from-env", action="store_true", help="use existing process environment instead of macOS Keychain")
    build.add_argument("--alfred-rights-confirmed", action="store_true")
    build.set_defaults(func=command_build)

    demo = subparsers.add_parser("demo", help="generate a clearly labelled synthetic result")
    demo.add_argument("--config", type=Path, default=default_config_path())
    demo.add_argument("--output", default="web/data/regime-results.json")
    demo.add_argument("--artifacts", default="artifacts/demo")
    demo.add_argument("--profile", choices=("quick", "standard"), default="quick")
    demo.set_defaults(func=command_demo)

    validate = subparsers.add_parser("validate", help="validate a dashboard result JSON")
    validate.add_argument("path", nargs="?", default="web/data/regime-results.json")
    validate.set_defaults(func=command_validate)

    serve = subparsers.add_parser("serve", help="serve the static local dashboard")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--web-root", default="web")
    serve.set_defaults(
        func=lambda args: serve_dashboard(
            _root_path(args.web_root), host=args.host, port=args.port
        )
        or 0
    )

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
