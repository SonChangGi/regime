"""Real-data runner and durable artifacts for the five-track mechanism ablation.

The public forecast selector never consumes these results.  This module loads
the same reconstructed source matrix as the private research comparison,
requires an exact-once feature-role manifest, runs fixed model families on
identical origins, and writes only derived forecasts and diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping, Sequence
import uuid

import numpy as np
import pandas as pd

from regime_lab.analysis.labels import STATE_ORDER
from regime_lab.analysis.mechanism_ablation import (
    FEATURE_ROLES,
    MECHANISM_TRACKS,
    MechanismAblationResult,
    load_mechanism_ablation_spec,
    mechanism_ablation_manifest_document,
    run_mechanism_ablation,
)
from regime_lab.analysis.validation import (
    PROBABILITY_COLUMNS,
    expected_calibration_error,
)
from regime_lab.config import project_root
from regime_lab.integrity import canonical_json_sha256_v1
from regime_lab.io import write_json_atomic
from regime_lab.pipeline import _profile
from regime_lab.research_comparison import (
    _prepare_matrix,
    research_source_fingerprint,
)
from regime_lab.walkforward_checkpoint import runtime_version_manifest


UTC = timezone.utc
FEATURE_ROLE_MANIFEST_SCHEMA_VERSION = "regime-feature-role-manifest/2"
MECHANISM_RUN_SCHEMA_VERSION = "regime-mechanism-ablation-run/1"
MECHANISM_EVIDENCE_STATUS = "historical_reconstructed_oos"
WEEKS_PER_YEAR = 52.1775
ARTIFACT_FRAMES: tuple[tuple[str, str], ...] = (
    ("oos_predictions", "oos-predictions.csv"),
    ("leaderboard", "leaderboard.csv"),
    ("independent_metrics", "independent-metrics.csv"),
    ("state_recall", "state-recall.csv"),
    ("transition_diagnostics", "transition-diagnostics.csv"),
    ("transition_events", "transition-events.csv"),
    ("common_origins", "common-origins.csv"),
    ("track_manifest", "track-manifest.csv"),
    ("role_manifest", "role-manifest.csv"),
)


@dataclass(frozen=True)
class FeatureRoleManifest:
    """Hash-verified exact feature ownership for one source-matrix contract."""

    path: Path
    document: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]
    feature_count: int
    feature_name_set_sha256: str
    sha256: str


def _feature_name_set_sha256(features: Sequence[str]) -> str:
    return canonical_json_sha256_v1(sorted(str(value) for value in features))


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be an existing regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


def load_feature_role_manifest(
    path: str | Path | None = None,
) -> FeatureRoleManifest:
    """Load a self-hashed role manifest and prove internal exact-once ownership."""

    resolved = (
        project_root() / "config" / "feature-role-manifest-v2.json"
        if path is None
        else Path(path)
    )
    document = _read_json_object(resolved, label="feature role manifest")
    expected_fields = {
        "schema_version",
        "source_matrix_contract",
        "feature_count",
        "feature_name_set_sha256",
        "role_order",
        "roles",
        "direct_label_component_exclusions",
        "validated_against",
        "selection_effect",
        "automatic_promotion_eligible",
        "sha256",
    }
    if set(document) != expected_fields:
        raise ValueError("feature role manifest fields are not exact")
    if document.get("schema_version") != FEATURE_ROLE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("feature role manifest schema_version is unsupported")
    if document.get("source_matrix_contract") != (
        "research-comparison-v6-reconstructed"
    ):
        raise ValueError("feature role source matrix contract is invalid")
    body = dict(document)
    digest = str(body.pop("sha256", ""))
    if canonical_json_sha256_v1(body) != digest:
        raise ValueError("feature role manifest canonical hash mismatch")
    if tuple(document.get("role_order", ())) != FEATURE_ROLES:
        raise ValueError("feature role order differs from the ablation contract")
    if document.get("selection_effect") != "none":
        raise ValueError("feature role manifest must not affect model selection")
    if document.get("automatic_promotion_eligible") is not False:
        raise ValueError("feature role manifest must prohibit automatic promotion")

    raw_roles = document.get("roles")
    if not isinstance(raw_roles, list) or len(raw_roles) != len(FEATURE_ROLES):
        raise ValueError("feature role rows are incomplete")
    owner: dict[str, str] = {}
    rows: list[Mapping[str, Any]] = []
    for expected_role, raw in zip(FEATURE_ROLES, raw_roles, strict=True):
        if not isinstance(raw, dict) or set(raw) != {
            "id",
            "description",
            "features",
        }:
            raise ValueError(f"feature role {expected_role} row is invalid")
        if raw.get("id") != expected_role:
            raise ValueError("feature role rows do not follow role_order")
        if not isinstance(raw.get("description"), str) or not raw["description"].strip():
            raise ValueError(f"feature role {expected_role} description is missing")
        features = raw.get("features")
        if not isinstance(features, list) or not features:
            raise ValueError(f"feature role {expected_role} must be non-empty")
        if any(not isinstance(value, str) or not value for value in features):
            raise ValueError(f"feature role {expected_role} has invalid names")
        if len(features) != len(set(features)):
            raise ValueError(f"feature role {expected_role} contains duplicates")
        overlap = sorted(value for value in features if value in owner)
        if overlap:
            raise ValueError(f"feature roles overlap: {overlap}")
        for feature in features:
            owner[feature] = expected_role
        rows.append({"id": expected_role, "features": tuple(features)})

    count = document.get("feature_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("feature role manifest feature_count is invalid")
    if count != len(owner):
        raise ValueError("feature role manifest feature_count does not match roles")
    feature_hash = str(document.get("feature_name_set_sha256", ""))
    if feature_hash != _feature_name_set_sha256(tuple(owner)):
        raise ValueError("feature role manifest feature-name hash mismatch")
    exclusions = document.get("direct_label_component_exclusions")
    if not isinstance(exclusions, list) or exclusions != sorted(
        rows[-1]["features"]
    ):
        raise ValueError("direct label exclusions must equal full_only_control")
    reference = document.get("validated_against")
    if not isinstance(reference, dict):
        raise ValueError("feature role validation reference is missing")
    if reference.get("evidence_status") != MECHANISM_EVIDENCE_STATUS:
        raise ValueError("feature role reference evidence status is invalid")
    if reference.get("historical_market_vintage_certified") is not False:
        raise ValueError("reconstructed market history must not be called certified")
    if reference.get("identity_gate") != "feature_count_and_feature_name_set_only":
        raise ValueError("feature role identity gate is invalid")
    return FeatureRoleManifest(
        path=resolved,
        document=document,
        rows=tuple(rows),
        feature_count=count,
        feature_name_set_sha256=feature_hash,
        sha256=digest,
    )


def validate_feature_role_manifest_for_columns(
    manifest: FeatureRoleManifest,
    columns: Sequence[object] | pd.Index,
) -> None:
    """Reject any added, removed, renamed, or duplicate source predictor."""

    names = tuple(str(value) for value in columns)
    if len(names) != len(set(names)):
        raise ValueError("source feature columns must be unique")
    expected = {
        str(feature)
        for row in manifest.rows
        for feature in row["features"]
    }
    actual = set(names)
    if len(names) != manifest.feature_count:
        raise ValueError(
            "source feature count differs from role manifest: "
            f"{len(names)} != {manifest.feature_count}"
        )
    if _feature_name_set_sha256(names) != manifest.feature_name_set_sha256:
        missing = sorted(expected.difference(actual))
        unexpected = sorted(actual.difference(expected))
        raise ValueError(
            "source feature names differ from role manifest; "
            f"missing={missing}, unexpected={unexpected}"
        )
    if actual != expected:
        raise RuntimeError("feature name hash matched but exact sets differ")


def mechanism_ablation_source_fingerprint(
    config: Mapping[str, Any],
    *,
    role_manifest_path: Path,
    specification_path: Path,
) -> str:
    """Bind checkpoint reuse to code, effective config, and both manifests."""

    base = research_source_fingerprint(config=config)
    digest = hashlib.sha256()
    for label, payload in (
        ("research_source_fingerprint", base.encode("ascii")),
        ("feature_role_manifest", role_manifest_path.read_bytes()),
        ("mechanism_specification", specification_path.read_bytes()),
        (
            "runner_script",
            (project_root() / "scripts" / "run_mechanism_ablation.py").read_bytes(),
        ),
    ):
        encoded = label.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def build_mechanism_metric_tables(
    result: MechanismAblationResult,
    *,
    tolerance: float = 1e-12,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recompute score, recall, and transition diagnostics from prediction rows."""

    predictions = result.predictions.copy()
    leaderboard = result.leaderboard.copy()
    metric_rows: list[dict[str, Any]] = []
    recall_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    positions = {state: index for index, state in enumerate(STATE_ORDER)}

    for (track, model, split), group in predictions.groupby(
        ["track", "model", "evaluation_split"], sort=False
    ):
        group = group.sort_values(["origin_date", "target_date"], ignore_index=True)
        probability = group[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
        probability = np.clip(probability, 1e-9, 1.0)
        probability /= probability.sum(axis=1, keepdims=True)
        actual = group["actual"].astype(str).to_numpy()
        current = group["current_state"].astype(str).to_numpy()
        predicted = np.asarray(STATE_ORDER, dtype=object)[probability.argmax(axis=1)]
        actual_positions = np.asarray([positions[value] for value in actual], dtype=int)
        actual_probability = probability[np.arange(len(group)), actual_positions]
        one_hot = np.zeros_like(probability)
        one_hot[np.arange(len(group)), actual_positions] = 1.0
        log_loss = float(np.mean(-np.log(actual_probability)))
        brier = float(np.mean(np.sum((probability - one_hot) ** 2, axis=1)))
        calibration = expected_calibration_error(actual, probability)
        entropy = np.zeros_like(probability)
        positive = probability > 0.0
        entropy[positive] = -probability[positive] * np.log(probability[positive])
        sharpness = float(
            np.clip(1.0 - entropy.sum(axis=1).mean() / math.log(len(STATE_ORDER)), 0, 1)
        )
        recorded = leaderboard.loc[
            leaderboard["track"].eq(track)
            & leaderboard["model"].eq(model)
            & leaderboard["evaluation_split"].eq(split)
        ]
        if len(recorded) != 1:
            raise ValueError(f"leaderboard row is not exact for {track}/{model}/{split}")
        row = recorded.iloc[0]
        for name, value in (
            ("log_loss", log_loss),
            ("brier", brier),
            ("calibration_error", calibration),
        ):
            if not math.isclose(float(row[name]), value, rel_tol=0.0, abs_tol=tolerance):
                raise ValueError(
                    f"independent {name} differs for {track}/{model}/{split}"
                )
        if int(row["n_predictions"]) != len(group):
            raise ValueError(f"prediction count differs for {track}/{model}/{split}")
        metric_rows.append(
            {
                "track": str(track),
                "model": str(model),
                "evaluation_split": str(split),
                "n_predictions": int(len(group)),
                "log_loss": log_loss,
                "brier": brier,
                "calibration_error": calibration,
                "normalized_entropy_sharpness": sharpness,
                "mean_max_probability": float(probability.max(axis=1).mean()),
                "leaderboard_crosscheck": "matched",
            }
        )

        for state in STATE_ORDER:
            mask = actual == state
            support = int(mask.sum())
            true_positive = int(np.count_nonzero(mask & (predicted == state)))
            recall_rows.append(
                {
                    "track": str(track),
                    "model": str(model),
                    "evaluation_split": str(split),
                    "state": state,
                    "support": support,
                    "true_positive": true_positive,
                    "recall": None if support == 0 else float(true_positive / support),
                    "status": "insufficient_support" if support == 0 else "computed",
                }
            )

        actual_transition = actual != current
        predicted_transition = predicted != current
        event_positions = np.flatnonzero(actual_transition)
        detected_delays: list[int] = []
        on_time_departure = 0
        on_time_destination = 0
        for event_number, raw_position in enumerate(event_positions, start=1):
            event_position = int(raw_position)
            next_event_position = (
                int(event_positions[event_number])
                if event_number < len(event_positions)
                else len(group)
            )
            destination = str(actual[event_position])
            if predicted_transition[event_position]:
                on_time_departure += 1
            if str(predicted[event_position]) == destination:
                on_time_destination += 1
            detected_position: int | None = None
            for candidate in range(event_position, next_event_position):
                if str(predicted[candidate]) == destination:
                    detected_position = candidate
                    break
            delay = (
                None
                if detected_position is None
                else int(detected_position - event_position)
            )
            if delay is not None:
                detected_delays.append(delay)
            event_rows.append(
                {
                    "track": str(track),
                    "model": str(model),
                    "evaluation_split": str(split),
                    "event_number": event_number,
                    "target_at": pd.Timestamp(
                        group.iloc[event_position]["target_date"]
                    ).isoformat(),
                    "source_state": str(current[event_position]),
                    "destination_state": destination,
                    "status": (
                        "missed_before_next_transition_or_split_end"
                        if delay is None
                        else "detected"
                    ),
                    "detected_target_at": (
                        None
                        if detected_position is None
                        else pd.Timestamp(
                            group.iloc[detected_position]["target_date"]
                        ).isoformat()
                    ),
                    "detection_delay_forecast_weeks": delay,
                }
            )
        false_alarms = int(
            np.count_nonzero(predicted_transition & ~actual_transition)
        )
        exposure_years = float(len(group) / WEEKS_PER_YEAR)
        transition_rows.append(
            {
                "track": str(track),
                "model": str(model),
                "evaluation_split": str(split),
                "transition_event_count": int(len(event_positions)),
                "detected_event_count": int(len(detected_delays)),
                "missed_event_count": int(len(event_positions) - len(detected_delays)),
                "on_time_departure_count": int(on_time_departure),
                "on_time_destination_count": int(on_time_destination),
                "destination_detection_rate": (
                    None
                    if len(event_positions) == 0
                    else float(len(detected_delays) / len(event_positions))
                ),
                "mean_detection_delay_forecast_weeks": (
                    None if not detected_delays else float(np.mean(detected_delays))
                ),
                "median_detection_delay_forecast_weeks": (
                    None if not detected_delays else float(np.median(detected_delays))
                ),
                "maximum_detection_delay_forecast_weeks": (
                    None if not detected_delays else int(max(detected_delays))
                ),
                "false_alarm_count": false_alarms,
                "exposure_years": exposure_years,
                "false_alarms_per_year": float(false_alarms / exposure_years),
            }
        )
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(recall_rows),
        pd.DataFrame(transition_rows),
        pd.DataFrame(event_rows),
    )


def run_real_mechanism_ablation(
    config: Mapping[str, Any],
    *,
    database: Path,
    as_of: datetime,
    profile_name: str,
    role_manifest_path: Path,
    specification_path: Path,
    checkpoint_directory: Path | None,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Run the fixed five tracks on the reconstructed real-data source matrix."""

    if profile_name not in {"quick", "standard", "full"}:
        raise ValueError("profile_name must be quick, standard, or full")
    manifest = load_feature_role_manifest(role_manifest_path)
    specification = load_mechanism_ablation_spec(specification_path)
    source_fingerprint = mechanism_ablation_source_fingerprint(
        config,
        role_manifest_path=role_manifest_path,
        specification_path=specification_path,
    )
    features, states, input_metadata = _prepare_matrix(
        config,
        database=database,
        as_of=as_of,
    )
    validate_feature_role_manifest_for_columns(manifest, features.columns)
    if mechanism_ablation_source_fingerprint(
        config,
        role_manifest_path=role_manifest_path,
        specification_path=specification_path,
    ) != source_fingerprint:
        raise RuntimeError("source changed while preparing the mechanism matrix")

    profile = _profile(profile_name, len(features))
    split_minimum = 3 if profile_name == "quick" else 12
    result = run_mechanism_ablation(
        features,
        states,
        manifest.rows,
        profile=profile,
        selection_end=str(config["model"]["final_holdout_start"]),
        specification=specification,
        gap=1,
        minimum_train_weeks=profile.minimum_train_weeks,
        selection_max_origins=3 if profile_name == "quick" else None,
        minimum_selection_predictions=split_minimum,
        minimum_holdout_predictions=split_minimum,
        random_state=17,
        model_workers=1 if profile_name == "quick" else 4,
        checkpoint_directory=checkpoint_directory,
        source_fingerprint_sha256=(
            source_fingerprint if checkpoint_directory is not None else None
        ),
        progress=progress,
    )
    if mechanism_ablation_source_fingerprint(
        config,
        role_manifest_path=role_manifest_path,
        specification_path=specification_path,
    ) != source_fingerprint:
        raise RuntimeError("source changed during mechanism ablation")

    independent, recalls, transitions, events = build_mechanism_metric_tables(result)
    run_manifest = mechanism_ablation_manifest_document(result)
    frames = {
        "oos_predictions": result.predictions,
        "leaderboard": result.leaderboard,
        "independent_metrics": independent,
        "state_recall": recalls,
        "transition_diagnostics": transitions,
        "transition_events": events,
        "common_origins": result.common_origins,
        "track_manifest": result.track_manifest,
        "role_manifest": result.role_manifest,
    }
    report: dict[str, Any] = {
        "schema_version": MECHANISM_RUN_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "data_as_of": as_of.astimezone(UTC).isoformat(),
        "profile": profile.name,
        "mode": "private_real_data_retrospective_research",
        "evidence_status": MECHANISM_EVIDENCE_STATUS,
        "synthetic_fixture": False,
        "historical_market_vintage_certified": False,
        "operational_oos": False,
        "derived_only_artifacts": True,
        "public_release_eligible": False,
        "selection_effect": "none",
        "champion_selection_performed": False,
        "automatic_promotion_eligible": False,
        "cross_family_ranking": False,
        "track_order": list(MECHANISM_TRACKS),
        "selection_end": str(config["model"]["final_holdout_start"]),
        "evaluation_split_contract": {
            "selection": "pre-registered pre-cutoff diagnostic rows",
            "holdout": "post-selection retrospective diagnostic only",
            "holdout_used_for_selection": False,
        },
        "input": {
            **input_metadata,
            "feature_name_set_sha256": _feature_name_set_sha256(
                tuple(str(value) for value in features.columns)
            ),
            "analysis_source_fingerprint_sha256": source_fingerprint,
            "runtime_versions": runtime_version_manifest(),
        },
        "contracts": {
            "mechanism_specification_sha256": result.specification_sha256,
            "feature_role_manifest_sha256": manifest.sha256,
            "feature_role_manifest_schema": FEATURE_ROLE_MANIFEST_SCHEMA_VERSION,
            "common_origin_manifest": run_manifest,
        },
        "metric_definitions": {
            "primary": "multiclass_log_loss",
            "secondary": [
                "multiclass_brier",
                "top_label_expected_calibration_error_10_bins",
                "state_argmax_recall",
                "transition_destination_detection_delay",
                "false_alarms_per_year",
            ],
            "transition_event": (
                "actual next state differs from the current state at origin"
            ),
            "transition_detection_window": (
                "within the same evaluation split and before the next actual transition"
            ),
        },
        "result_counts": {
            "common_origins": int(len(result.common_origins)),
            "prediction_rows": int(len(result.predictions)),
            "leaderboard_rows": int(len(result.leaderboard)),
            "state_recall_rows": int(len(recalls)),
            "transition_event_rows": int(len(events)),
        },
        "interpretation": {
            "negative_paired_log_loss_delta_vs_full_favors_reduced_track": True,
            "state_baselines_are_not_ranked_against_fixed_xgboost_tracks": True,
            "result_is_not_a_performance_or_return_guarantee": True,
        },
    }
    return report, frames


def write_mechanism_ablation_generation(
    output_root: Path,
    report: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    *,
    expected_source_fingerprint_sha256: str,
    source_config: Mapping[str, Any],
    role_manifest_path: Path,
    specification_path: Path,
) -> Path:
    """Atomically expose one immutable derived-only mechanism generation."""

    current = mechanism_ablation_source_fingerprint(
        source_config,
        role_manifest_path=role_manifest_path,
        specification_path=specification_path,
    )
    if current != expected_source_fingerprint_sha256:
        raise RuntimeError("source changed before writing mechanism artifacts")
    required = {key for key, _ in ARTIFACT_FRAMES}
    if set(frames) != required:
        raise ValueError("mechanism artifact frame inventory is not exact")
    if report.get("evidence_status") != MECHANISM_EVIDENCE_STATUS:
        raise ValueError("mechanism report evidence status is invalid")
    if report.get("derived_only_artifacts") is not True:
        raise ValueError("mechanism report must declare derived-only artifacts")
    if report.get("automatic_promotion_eligible") is not False:
        raise ValueError("mechanism report must prohibit automatic promotion")

    root = output_root.resolve()
    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    generation_id = (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    staging = Path(tempfile.mkdtemp(prefix=".mechanism-generation-", dir=runs))
    final = runs / generation_id
    try:
        artifact_manifest: dict[str, dict[str, Any]] = {}
        for key, filename in ARTIFACT_FRAMES:
            frame = frames[key]
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"mechanism artifact {key} must be a DataFrame")
            path = staging / filename
            frame.to_csv(path, index=False, lineterminator="\n")
            raw = path.read_bytes()
            artifact_manifest[key] = {
                "path": filename,
                "row_count": int(len(frame)),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        report_document = {**dict(report), "artifact_manifest": artifact_manifest}
        report_body_hash = canonical_json_sha256_v1(report_document)
        report_document["sha256"] = report_body_hash
        write_json_atomic(staging / "mechanism-ablation-report.json", report_document)
        report_raw = (staging / "mechanism-ablation-report.json").read_bytes()
        current = mechanism_ablation_source_fingerprint(
            source_config,
            role_manifest_path=role_manifest_path,
            specification_path=specification_path,
        )
        if current != expected_source_fingerprint_sha256:
            raise RuntimeError("source changed while writing mechanism artifacts")
        os.replace(staging, final)
        write_json_atomic(
            root / "latest.json",
            {
                "schema_version": 1,
                "generation": f"runs/{generation_id}",
                "report": "mechanism-ablation-report.json",
                "report_sha256": hashlib.sha256(report_raw).hexdigest(),
            },
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return final


__all__ = [
    "ARTIFACT_FRAMES",
    "FEATURE_ROLE_MANIFEST_SCHEMA_VERSION",
    "FeatureRoleManifest",
    "MECHANISM_EVIDENCE_STATUS",
    "MECHANISM_RUN_SCHEMA_VERSION",
    "build_mechanism_metric_tables",
    "load_feature_role_manifest",
    "mechanism_ablation_source_fingerprint",
    "run_real_mechanism_ablation",
    "validate_feature_role_manifest_for_columns",
    "write_mechanism_ablation_generation",
]
