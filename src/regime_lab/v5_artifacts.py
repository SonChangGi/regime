"""Canonical, hash-linked sidecars for the opt-in v5 research contract."""

from __future__ import annotations

from collections.abc import Mapping
import csv
from dataclasses import dataclass
from datetime import date
import hashlib
from io import StringIO
from pathlib import Path
import re
from typing import Any

import pandas as pd

from regime_lab.analysis.fx import CORE_DOLLAR_INDEXES
from regime_lab.analysis.fx_ablation import FX_ABLATION_OOS_COLUMNS
from regime_lab.analysis.outcomes import OUTCOME_COLUMNS, POINT_METRICS
from regime_lab.data.h10 import FIXED_BILATERAL_PANEL


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class V5ResearchArtifactSpec:
    key: str
    path: str
    columns: tuple[str, ...]
    sort_columns: tuple[str, ...]
    date_columns: tuple[str, ...] = ()
    timestamp_columns: tuple[str, ...] = ()
    materialize_observation_week: bool = False


DIRECTIONAL_OOS_COLUMNS: tuple[str, ...] = (
    "horizon_weeks",
    "origin_date",
    "target_end",
    "evaluation_split",
    "model",
    "current_state",
    "actual_outcome",
    "actual_change",
    "p_no_departure",
    "p_risk_on",
    "p_transition",
    "p_risk_off",
    "fallback",
    "fallback_reason",
)

DIRECTIONAL_LEADERBOARD_COLUMNS: tuple[str, ...] = (
    "horizon_weeks",
    "evaluation_split",
    "model",
    "selected",
    "score_target",
    "log_loss",
    "brier",
    "n_predictions",
    "event_count",
    "destination_class_count",
    "effective_event_blocks",
    "fallback_count",
)

DIRECTIONAL_SPLIT_COLUMNS: tuple[str, ...] = (
    "horizon_weeks",
    "origin_date",
    "target_end",
    "evaluation_split",
    "train_size",
    "last_train_origin",
    "last_train_target_end",
    "purged_origin_count",
)

DIRECTIONAL_SELECTION_COLUMNS: tuple[str, ...] = (
    "horizon_weeks",
    "model",
    "reference_model",
    "selected",
    "gate_passed",
    "gate_reason",
    "score_target",
    "selection_event_count",
    "selection_destination_class_count",
    "selection_effective_event_blocks",
    "minimum_selection_events",
    "minimum_destination_classes",
    "minimum_event_blocks",
    "log_loss",
    "brier",
    "absolute_log_loss_improvement",
    "holm_adjusted_p_value",
    "fallback_count",
)

DIRECTIONAL_FORECAST_COLUMNS: tuple[str, ...] = (
    "horizon_weeks",
    "origin_date",
    "target_end",
    "model",
    "current_state",
    "p_no_departure",
    "p_risk_on",
    "p_transition",
    "p_risk_off",
    "fallback",
    "fallback_reason",
)

CONDITIONAL_STATISTICS_COLUMNS: tuple[str, ...] = (
    "state",
    "asset",
    "horizon_weeks",
    "execution_lag_weeks",
    "return_currency",
    "sample_start",
    "sample_end",
    "n",
    "unique_episodes",
    "status",
    "minimum_observations",
    "minimum_unique_episodes",
    "bootstrap_method",
    "bootstrap_block_weeks",
    "bootstrap_resamples",
    "bootstrap_seed",
    *POINT_METRICS,
    *(
        field
        for metric in POINT_METRICS
        for field in (f"{metric}_ci95_lower", f"{metric}_ci95_upper")
    ),
)


def _fx_feature_columns() -> tuple[str, ...]:
    columns: list[str] = ["observation_week"]
    for code in (*CORE_DOLLAR_INDEXES, *FIXED_BILATERAL_PANEL):
        stem = code.lower()
        columns.append(f"fx__{stem}__usd_log_level")
        columns.extend(
            f"fx__{stem}__usd_log_return_{horizon}w"
            for horizon in (1, 4, 13)
        )
        columns.extend(
            f"fx__{stem}__realized_vol_{window}w" for window in (13, 26)
        )
    for horizon in (1, 4, 13):
        columns.extend(
            (
                f"fx__eme_minus_afe__usd_log_return_{horizon}w",
                f"fx__broad_minus_afe__usd_log_return_{horizon}w",
                f"fx__broad_minus_eme__usd_log_return_{horizon}w",
                f"fx__dollar_indexes__return_mad_{horizon}w",
                f"fx__bilateral__median_usd_log_return_{horizon}w",
                f"fx__bilateral__usd_appreciating_share_{horizon}w",
                f"fx__bilateral__return_mad_{horizon}w",
            )
        )
    return tuple(columns)


FX_FEATURE_COLUMNS = _fx_feature_columns()
FX_COVERAGE_COLUMNS: tuple[str, ...] = (
    "observation_week",
    "core_level_count",
    "core_level_ratio",
    "bilateral_level_count",
    "bilateral_level_ratio",
    "non_normal_observation_count",
    "bilateral_return_1w_count",
    "bilateral_return_1w_ratio",
    "bilateral_return_4w_count",
    "bilateral_return_4w_ratio",
    "bilateral_return_13w_count",
    "bilateral_return_13w_ratio",
    "source_available_at",
    "feature_available_at",
    "source_status",
    "feature_status",
    "archive_correction_quarantined",
    "archive_correction_available_at",
    "archive_correction_quarantine_until_week",
)

V5_RESEARCH_ARTIFACT_SPECS: tuple[V5ResearchArtifactSpec, ...] = (
    V5ResearchArtifactSpec(
        "directional_oos_predictions",
        "directional-oos-predictions.csv",
        DIRECTIONAL_OOS_COLUMNS,
        ("horizon_weeks", "origin_date", "model"),
        ("origin_date", "target_end"),
    ),
    V5ResearchArtifactSpec(
        "directional_model_leaderboard",
        "directional-model-leaderboard.csv",
        DIRECTIONAL_LEADERBOARD_COLUMNS,
        ("horizon_weeks", "evaluation_split", "log_loss", "brier", "model"),
    ),
    V5ResearchArtifactSpec(
        "directional_walk_forward_splits",
        "directional-walk-forward-splits.csv",
        DIRECTIONAL_SPLIT_COLUMNS,
        ("horizon_weeks", "origin_date"),
        (
            "origin_date",
            "target_end",
            "last_train_origin",
            "last_train_target_end",
        ),
    ),
    V5ResearchArtifactSpec(
        "directional_selection_diagnostics",
        "directional-selection-diagnostics.csv",
        DIRECTIONAL_SELECTION_COLUMNS,
        ("horizon_weeks", "model"),
    ),
    V5ResearchArtifactSpec(
        "directional_forecasts",
        "directional-forecasts.csv",
        DIRECTIONAL_FORECAST_COLUMNS,
        ("horizon_weeks", "origin_date"),
        ("origin_date", "target_end"),
    ),
    V5ResearchArtifactSpec(
        "conditional_asset_outcomes",
        "conditional-asset-outcomes.csv",
        tuple(OUTCOME_COLUMNS),
        ("origin_position", "horizon_weeks", "asset"),
        ("origin_date", "entry_date", "exit_date"),
    ),
    V5ResearchArtifactSpec(
        "conditional_asset_statistics",
        "conditional-asset-statistics.csv",
        CONDITIONAL_STATISTICS_COLUMNS,
        ("state", "asset", "horizon_weeks"),
        ("sample_start", "sample_end"),
    ),
    V5ResearchArtifactSpec(
        "fx_features",
        "fx-features.csv",
        FX_FEATURE_COLUMNS,
        ("observation_week",),
        ("observation_week",),
        materialize_observation_week=True,
    ),
    V5ResearchArtifactSpec(
        "fx_coverage",
        "fx-coverage.csv",
        FX_COVERAGE_COLUMNS,
        ("observation_week",),
        ("observation_week", "archive_correction_quarantine_until_week"),
        (
            "source_available_at",
            "feature_available_at",
            "archive_correction_available_at",
        ),
        materialize_observation_week=True,
    ),
    V5ResearchArtifactSpec(
        "fx_ablation_oos",
        "fx-ablation-oos.csv",
        FX_ABLATION_OOS_COLUMNS,
        ("origin_date", "target_date", "variant"),
        ("origin_date", "target_date", "last_train_target"),
    ),
)

V5_RESEARCH_ARTIFACTS = {
    spec.key: spec for spec in V5_RESEARCH_ARTIFACT_SPECS
}
V5_RESEARCH_ARTIFACTS_BY_PATH = {
    spec.path: spec for spec in V5_RESEARCH_ARTIFACT_SPECS
}
REQUIRED_RESEARCH_ARTIFACT_KEYS = frozenset(
    spec.key for spec in V5_RESEARCH_ARTIFACT_SPECS if not spec.key.startswith("fx_")
)
FX_RESEARCH_ARTIFACT_KEYS = frozenset(
    ("fx_features", "fx_coverage", "fx_ablation_oos")
)
V5_CORE_ARTIFACT_PATHS: tuple[tuple[str, str], ...] = (
    ("oos_predictions", "oos-predictions.csv"),
    ("model_leaderboard", "model-leaderboard.csv"),
    ("walk_forward_splits", "walk-forward-splits.csv"),
    ("selection_diagnostics", "selection-diagnostics.csv"),
    ("stacking_weights", "stacking-weights.csv"),
    ("multiscale_ensemble_scales", "multiscale-ensemble-scales.csv"),
)
V5_CORE_ARTIFACTS = dict(V5_CORE_ARTIFACT_PATHS)


def _spec(key_or_path: str) -> V5ResearchArtifactSpec:
    value = str(key_or_path)
    spec = V5_RESEARCH_ARTIFACTS.get(value)
    if spec is None:
        spec = V5_RESEARCH_ARTIFACTS_BY_PATH.get(value)
    if spec is None:
        raise KeyError(f"unknown v5 research artifact: {value}")
    return spec


def _date_value(value: object, *, context: str) -> str | None:
    if value is None or value is pd.NA or value is pd.NaT or pd.isna(value):
        return None
    try:
        resolved = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} is not a date") from exc
    return resolved.date().isoformat()


def _timestamp_value(value: object, *, context: str) -> str | None:
    if value is None or value is pd.NA or value is pd.NaT or pd.isna(value):
        return None
    try:
        resolved = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} is not a timestamp") from exc
    if resolved.tzinfo is None or resolved.utcoffset() is None:
        raise ValueError(f"{context} must be timezone-aware")
    return resolved.isoformat()


def canonical_v5_artifact_frame(
    key_or_path: str,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return one frozen-column, stable-order frame ready for CSV encoding."""

    spec = _spec(key_or_path)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{spec.key} must be a pandas DataFrame")
    if frame.columns.has_duplicates:
        raise ValueError(f"{spec.key} contains duplicate columns")
    canonical = frame.copy()
    if spec.materialize_observation_week:
        if "observation_week" in canonical.columns:
            raise ValueError(
                f"{spec.key} must carry observation_week in its DatetimeIndex"
            )
        if not isinstance(canonical.index, pd.DatetimeIndex):
            raise TypeError(f"{spec.key} must use a DatetimeIndex")
        if canonical.index.has_duplicates:
            raise ValueError(f"{spec.key} observation weeks must be unique")
        canonical.insert(0, "observation_week", canonical.index)
        canonical = canonical.reset_index(drop=True)

    actual = tuple(str(column) for column in canonical.columns)
    if actual != spec.columns:
        raise ValueError(
            f"{spec.key} columns must exactly match the v5 contract: "
            f"expected {spec.columns}, got {actual}"
        )

    for column in spec.date_columns:
        canonical[column] = [
            _date_value(value, context=f"{spec.key}.{column}")
            for value in canonical[column]
        ]
    for column in spec.timestamp_columns:
        canonical[column] = [
            _timestamp_value(value, context=f"{spec.key}.{column}")
            for value in canonical[column]
        ]
    canonical = canonical.sort_values(
        list(spec.sort_columns),
        kind="mergesort",
        ignore_index=True,
    )
    return canonical.loc[:, spec.columns]


def canonical_v5_artifact_csv_bytes(
    key_or_path: str,
    frame: pd.DataFrame,
) -> bytes:
    """Serialize one v5 sidecar with a byte-stable CSV contract."""

    canonical = canonical_v5_artifact_frame(key_or_path, frame)
    stream = StringIO(newline="")
    canonical.to_csv(
        stream,
        index=False,
        lineterminator="\n",
        na_rep="",
        float_format="%.17g",
    )
    return stream.getvalue().encode("utf-8")


def build_v5_research_artifact_manifest(
    frames: Mapping[str, pd.DataFrame],
) -> dict[str, dict[str, object]]:
    """Build the payload manifest from the exact bytes the CLI must stage."""

    keys = frozenset(str(key) for key in frames)
    allowed = REQUIRED_RESEARCH_ARTIFACT_KEYS | FX_RESEARCH_ARTIFACT_KEYS
    if not REQUIRED_RESEARCH_ARTIFACT_KEYS.issubset(keys) or not keys.issubset(
        allowed
    ):
        raise ValueError("v5 research artifact set is incomplete or contains unknown keys")
    present_fx = keys & FX_RESEARCH_ARTIFACT_KEYS
    if present_fx and present_fx != FX_RESEARCH_ARTIFACT_KEYS:
        raise ValueError("v5 FX research artifacts must be supplied as a complete set")

    manifest: dict[str, dict[str, object]] = {}
    for spec in V5_RESEARCH_ARTIFACT_SPECS:
        if spec.key not in frames:
            continue
        canonical = canonical_v5_artifact_frame(spec.key, frames[spec.key])
        payload = canonical_v5_artifact_csv_bytes(spec.key, frames[spec.key])
        manifest[spec.key] = {
            "path": spec.path,
            "row_count": int(len(canonical)),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return manifest


def verify_staged_v5_research_artifacts(
    manifest: Mapping[str, Any],
    directory: Path,
) -> None:
    """Fail closed when staged sidecars diverge from the payload manifest."""

    if not isinstance(manifest, Mapping):
        raise TypeError("v5 research artifact manifest must be a mapping")
    keys = frozenset(str(key) for key in manifest)
    allowed = REQUIRED_RESEARCH_ARTIFACT_KEYS | FX_RESEARCH_ARTIFACT_KEYS
    if not REQUIRED_RESEARCH_ARTIFACT_KEYS.issubset(keys) or not keys.issubset(
        allowed
    ):
        raise ValueError("v5 research artifact manifest keys are invalid")
    present_fx = keys & FX_RESEARCH_ARTIFACT_KEYS
    if present_fx and present_fx != FX_RESEARCH_ARTIFACT_KEYS:
        raise ValueError("v5 FX research artifact manifest must contain the complete set")

    root = Path(directory)
    for key in (spec.key for spec in V5_RESEARCH_ARTIFACT_SPECS if spec.key in manifest):
        spec = V5_RESEARCH_ARTIFACTS[key]
        metadata = manifest[key]
        if not isinstance(metadata, Mapping) or set(metadata) != {
            "path",
            "row_count",
            "sha256",
        }:
            raise ValueError(f"v5 research artifact metadata is invalid: {key}")
        if metadata["path"] != spec.path:
            raise ValueError(f"v5 research artifact path is invalid: {key}")
        row_count = metadata["row_count"]
        if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 0:
            raise ValueError(f"v5 research artifact row_count is invalid: {key}")
        expected_hash = metadata["sha256"]
        if not isinstance(expected_hash, str) or SHA256_PATTERN.fullmatch(expected_hash) is None:
            raise ValueError(f"v5 research artifact sha256 is invalid: {key}")

        path = root / spec.path
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(
                f"staged v5 research artifact is missing or non-regular: {spec.path}"
            )
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_hash:
            raise RuntimeError(
                f"staged v5 research artifact hash mismatch: {spec.path}"
            )
        if b"\r\n" in payload or not payload.endswith(b"\n"):
            raise RuntimeError(
                f"staged v5 research artifact is not canonical LF CSV: {spec.path}"
            )
        try:
            reader = csv.reader(StringIO(payload.decode("utf-8"), newline=""))
            header = tuple(next(reader))
        except (UnicodeDecodeError, StopIteration, csv.Error) as exc:
            raise RuntimeError(
                f"staged v5 research artifact is not valid UTF-8 CSV: {spec.path}"
            ) from exc
        if header != spec.columns:
            raise RuntimeError(
                f"staged v5 research artifact header mismatch: {spec.path}"
            )
        actual_rows = 0
        previous_week: date | None = None
        try:
            for row in reader:
                actual_rows += 1
                if len(row) != len(spec.columns):
                    raise RuntimeError(
                        f"staged v5 research artifact row width mismatch: {spec.path}"
                    )
                if spec.materialize_observation_week:
                    try:
                        observation_week = date.fromisoformat(row[0])
                    except ValueError as exc:
                        raise RuntimeError(
                            f"staged v5 FX observation_week is invalid: {spec.path}"
                        ) from exc
                    if previous_week is not None and observation_week <= previous_week:
                        raise RuntimeError(
                            f"staged v5 FX observation_week is not increasing: {spec.path}"
                        )
                    previous_week = observation_week
        except csv.Error as exc:
            raise RuntimeError(
                f"staged v5 research artifact CSV is malformed: {spec.path}"
            ) from exc
        if actual_rows != row_count:
            raise RuntimeError(
                f"staged v5 research artifact row count mismatch: {spec.path}"
            )


def canonical_v5_core_artifact_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize one core frame exactly as the V5 staging writer does."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("v5 core artifact must be a pandas DataFrame")
    if frame.empty:
        raise ValueError("v5 core artifact must not be empty")
    if frame.columns.has_duplicates:
        raise ValueError("v5 core artifact columns must not duplicate")
    stream = StringIO(newline="")
    frame.to_csv(stream, index=False, lineterminator="\n")
    return stream.getvalue().encode("utf-8")


def build_v5_core_artifact_manifest(
    frames: Mapping[str, pd.DataFrame],
) -> dict[str, dict[str, object]]:
    """Bind every V5 core/model sidecar to its exact staged bytes."""

    if tuple(frames) != tuple(key for key, _ in V5_CORE_ARTIFACT_PATHS):
        raise ValueError("v5 core artifact keys/order are invalid")
    manifest: dict[str, dict[str, object]] = {}
    for key, path in V5_CORE_ARTIFACT_PATHS:
        frame = frames[key]
        payload = canonical_v5_core_artifact_csv_bytes(frame)
        manifest[key] = {
            "path": path,
            "row_count": int(len(frame)),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return manifest


def verify_staged_v5_core_artifacts(
    manifest: Mapping[str, Any],
    directory: Path,
) -> None:
    """Fail closed when staged core/model sidecars differ from V5 payload."""

    if not isinstance(manifest, Mapping):
        raise TypeError("v5 core artifact manifest must be a mapping")
    if tuple(manifest) != tuple(key for key, _ in V5_CORE_ARTIFACT_PATHS):
        raise ValueError("v5 core artifact manifest keys/order are invalid")
    root = Path(directory)
    for key, expected_path in V5_CORE_ARTIFACT_PATHS:
        metadata = manifest[key]
        if not isinstance(metadata, Mapping) or set(metadata) != {
            "path",
            "row_count",
            "sha256",
        }:
            raise ValueError(f"v5 core artifact metadata is invalid: {key}")
        if metadata["path"] != expected_path:
            raise ValueError(f"v5 core artifact path is invalid: {key}")
        row_count = metadata["row_count"]
        if (
            isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or row_count < 1
        ):
            raise ValueError(f"v5 core artifact row_count is invalid: {key}")
        expected_hash = metadata["sha256"]
        if (
            not isinstance(expected_hash, str)
            or SHA256_PATTERN.fullmatch(expected_hash) is None
        ):
            raise ValueError(f"v5 core artifact sha256 is invalid: {key}")
        path = root / expected_path
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(
                f"staged v5 core artifact is missing/non-regular: {expected_path}"
            )
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_hash:
            raise RuntimeError(
                f"staged v5 core artifact hash mismatch: {expected_path}"
            )
        if b"\r\n" in payload or not payload.endswith(b"\n"):
            raise RuntimeError(
                f"staged v5 core artifact is not canonical LF CSV: {expected_path}"
            )
        try:
            frame = pd.read_csv(path)
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            raise RuntimeError(
                f"staged v5 core artifact is not valid CSV: {expected_path}"
            ) from exc
        if len(frame) != row_count:
            raise RuntimeError(
                f"staged v5 core artifact row count mismatch: {expected_path}"
            )


__all__ = [
    "CONDITIONAL_STATISTICS_COLUMNS",
    "DIRECTIONAL_FORECAST_COLUMNS",
    "DIRECTIONAL_LEADERBOARD_COLUMNS",
    "DIRECTIONAL_OOS_COLUMNS",
    "DIRECTIONAL_SELECTION_COLUMNS",
    "DIRECTIONAL_SPLIT_COLUMNS",
    "FX_ABLATION_OOS_COLUMNS",
    "FX_COVERAGE_COLUMNS",
    "FX_FEATURE_COLUMNS",
    "FX_RESEARCH_ARTIFACT_KEYS",
    "REQUIRED_RESEARCH_ARTIFACT_KEYS",
    "V5_CORE_ARTIFACTS",
    "V5_CORE_ARTIFACT_PATHS",
    "V5_RESEARCH_ARTIFACTS",
    "V5_RESEARCH_ARTIFACTS_BY_PATH",
    "build_v5_research_artifact_manifest",
    "build_v5_core_artifact_manifest",
    "canonical_v5_core_artifact_csv_bytes",
    "canonical_v5_artifact_csv_bytes",
    "canonical_v5_artifact_frame",
    "verify_staged_v5_research_artifacts",
    "verify_staged_v5_core_artifacts",
]
