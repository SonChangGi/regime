"""Five-track, matched-origin mechanism ablation for the regime forecast.

This is deliberately separate from :mod:`regime_lab.analysis.ablation`, whose
seven frozen feature-family variants remain the V4/V5 reproduction contract.
The v2 tracks answer a different question: how much evidence comes from state
history, label mechanics, market structure, macro/rates/credit, and the full
feature set.

Feature roles are supplied as an exact-once manifest.  This module does not
guess roles from column names: such guessing would silently move features
between tracks as naming conventions evolve.  Likewise, the state-history
baselines and fixed-XGBoost tracks are separate comparison families so a
feature conclusion is never disguised as a model-class conclusion.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .labels import STATE_ORDER
from .models import BenchmarkProfile, resolve_profile
from .validation import PROBABILITY_COLUMNS, evaluate_predictions, run_benchmark


MECHANISM_ABLATION_SCHEMA_VERSION = "regime-mechanism-ablation/2"
MECHANISM_TRACKS: tuple[str, ...] = (
    "state_only",
    "label_mechanics",
    "market_ex_label",
    "macro_rates_credit",
    "full",
)
FEATURE_ROLES: tuple[str, ...] = (
    "label_mechanics",
    "market_ex_label_components",
    "macro_rates_credit",
    "full_only_control",
)
_KEY_COLUMNS: tuple[str, ...] = ("origin_date", "target_date")
_AUDIT_COLUMNS: tuple[str, ...] = (
    "origin_date",
    "target_date",
    "evaluation_split",
    "current_state",
    "actual",
    "train_size",
    "gap",
)
_REQUIRED_PREDICTION_COLUMNS = frozenset(
    (
        *_AUDIT_COLUMNS,
        "model",
        "predicted",
        *PROBABILITY_COLUMNS,
        "fallback",
        "fallback_reason",
    )
)


@dataclass(frozen=True)
class MechanismTrackSpec:
    track_id: str
    feature_roles: tuple[str, ...]
    benchmark_models: tuple[str, ...]
    reported_models: tuple[str, ...]
    comparison_family: str
    interpretation: str


@dataclass(frozen=True)
class MechanismAblationSpec:
    schema_version: str
    tracks: tuple[MechanismTrackSpec, ...]
    feature_roles: tuple[str, ...]
    evaluation: Mapping[str, Any]
    sha256: str


@dataclass(frozen=True)
class MechanismAblationResult:
    """Auditable predictions and summaries for the five fixed tracks."""

    predictions: pd.DataFrame
    leaderboard: pd.DataFrame
    track_manifest: pd.DataFrame
    role_manifest: pd.DataFrame
    common_origins: pd.DataFrame
    specification_sha256: str


def _canonical_sha256(value: Mapping[str, Any] | Sequence[Any]) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _default_spec_path() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "mechanism-ablation-v2.json"


def load_mechanism_ablation_spec(
    path: str | Path | None = None,
) -> MechanismAblationSpec:
    """Load and fail-closed validate the immutable five-track contract."""

    resolved = _default_spec_path() if path is None else Path(path)
    document = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("mechanism ablation specification must be an object")
    if document.get("schema_version") != MECHANISM_ABLATION_SCHEMA_VERSION:
        raise ValueError("unsupported mechanism ablation schema_version")
    if tuple(document.get("track_order", ())) != MECHANISM_TRACKS:
        raise ValueError("mechanism ablation track_order is not the frozen five tracks")
    if tuple(document.get("feature_roles", ())) != FEATURE_ROLES:
        raise ValueError("mechanism ablation feature_roles are invalid")
    if document.get("role_assignment") != "caller_supplied_exact_once_manifest":
        raise ValueError("feature role assignment must be explicit and exact-once")

    raw_tracks = document.get("tracks")
    if not isinstance(raw_tracks, dict) or tuple(raw_tracks) != MECHANISM_TRACKS:
        raise ValueError("mechanism ablation tracks must follow track_order exactly")
    tracks: list[MechanismTrackSpec] = []
    for track_id in MECHANISM_TRACKS:
        raw = raw_tracks.get(track_id)
        if not isinstance(raw, dict):
            raise ValueError(f"track {track_id} must be an object")
        roles = tuple(str(value) for value in raw.get("feature_roles", ()))
        if len(roles) != len(set(roles)) or not set(roles).issubset(FEATURE_ROLES):
            raise ValueError(f"track {track_id} has invalid feature roles")
        benchmark_models = tuple(
            str(value) for value in raw.get("benchmark_models", ())
        )
        reported_models = tuple(
            str(value) for value in raw.get("reported_models", ())
        )
        if not benchmark_models or not reported_models:
            raise ValueError(
                f"track {track_id} must declare benchmark and reported models"
            )
        if not set(reported_models).issubset(benchmark_models):
            raise ValueError(f"track {track_id} reports a model it does not run")
        family = str(raw.get("comparison_family", "")).strip()
        interpretation = str(raw.get("interpretation", "")).strip()
        if not family or not interpretation:
            raise ValueError(f"track {track_id} needs family and interpretation")
        tracks.append(
            MechanismTrackSpec(
                track_id=track_id,
                feature_roles=roles,
                benchmark_models=benchmark_models,
                reported_models=reported_models,
                comparison_family=family,
                interpretation=interpretation,
            )
        )

    if tracks[0].feature_roles:
        raise ValueError("state_only must not consume measured feature roles")
    fixed_family = {track.comparison_family for track in tracks[1:]}
    fixed_models = {track.reported_models for track in tracks[1:]}
    if len(fixed_family) != 1 or fixed_models != {("xgboost",)}:
        raise ValueError("non-state tracks must share one fixed-XGBoost family")
    if tracks[-1].feature_roles != FEATURE_ROLES:
        raise ValueError("full must contain every feature role")

    evaluation = document.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("mechanism ablation evaluation must be an object")
    if (
        evaluation.get("origin_contract")
        != "exact_origin_target_actual_and_split_match"
    ):
        raise ValueError(
            "mechanism ablation must use the exact matched-origin contract"
        )
    if evaluation.get("cross_family_ranking") is not False:
        raise ValueError("cross-family ranking must remain disabled")
    return MechanismAblationSpec(
        schema_version=MECHANISM_ABLATION_SCHEMA_VERSION,
        tracks=tuple(tracks),
        feature_roles=FEATURE_ROLES,
        evaluation=dict(evaluation),
        sha256=_canonical_sha256(document),
    )


def _normalise_role_manifest(
    feature_role_manifest: Sequence[Mapping[str, Any]],
    columns: pd.Index,
) -> tuple[dict[str, tuple[str, ...]], pd.DataFrame]:
    if isinstance(feature_role_manifest, (str, bytes)):
        raise TypeError("feature_role_manifest must be a sequence of mappings")
    if columns.has_duplicates:
        raise ValueError("feature columns must be unique")
    available = tuple(str(column) for column in columns)
    available_set = set(available)
    records = list(feature_role_manifest)
    if not records:
        raise ValueError("feature_role_manifest must not be empty")

    by_role: dict[str, tuple[str, ...]] = {}
    owner: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for raw in records:
        if not isinstance(raw, Mapping):
            raise TypeError("each feature role row must be a mapping")
        role = str(raw.get("id", "")).strip()
        if role not in FEATURE_ROLES or role in by_role:
            raise ValueError(f"invalid or duplicate feature role: {role!r}")
        raw_features = raw.get("features")
        if isinstance(raw_features, (str, bytes)) or not isinstance(
            raw_features, Sequence
        ):
            raise TypeError(f"feature role {role} features must be a sequence")
        features = tuple(str(value) for value in raw_features)
        if len(features) != len(set(features)):
            raise ValueError(f"feature role {role} contains duplicates")
        unknown = sorted(set(features).difference(available_set))
        if unknown:
            raise ValueError(
                f"feature role {role} contains unknown features: {unknown}"
            )
        overlap = sorted(feature for feature in features if feature in owner)
        if overlap:
            raise ValueError(f"features assigned to more than one role: {overlap}")
        for feature in features:
            owner[feature] = role
        ordered_features = tuple(
            feature for feature in available if feature in features
        )
        by_role[role] = ordered_features
        rows.append(
            {
                "role": role,
                "feature_count": len(ordered_features),
                "features": ordered_features,
                "feature_sha256": _canonical_sha256(list(ordered_features)),
            }
        )
    if tuple(by_role) != FEATURE_ROLES:
        raise ValueError("feature role rows must follow the frozen role order")
    missing = [feature for feature in available if feature not in owner]
    if missing:
        raise ValueError(f"feature role manifest leaves features unassigned: {missing}")
    if len(owner) != len(available):
        raise RuntimeError("feature role manifest is not exact-once")
    return by_role, pd.DataFrame(rows)


def _track_columns(
    features: pd.DataFrame,
    roles: Mapping[str, tuple[str, ...]],
    track: MechanismTrackSpec,
) -> tuple[str, ...]:
    if track.track_id == "state_only":
        return ("__state_only_constant__",)
    selected = {
        feature
        for role in track.feature_roles
        for feature in roles[role]
    }
    ordered = tuple(
        str(column) for column in features.columns if str(column) in selected
    )
    if not ordered:
        raise ValueError(f"mechanism track {track.track_id} has no features")
    if track.track_id == "full" and ordered != tuple(str(c) for c in features.columns):
        raise ValueError("full track must contain every input feature exactly once")
    return ordered


def _validate_predictions(frame: pd.DataFrame, *, context: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{context} predictions must be a DataFrame")
    missing = sorted(_REQUIRED_PREDICTION_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"{context} predictions missing columns: {missing}")
    result = frame.copy()
    result["origin_date"] = pd.to_datetime(result["origin_date"], errors="raise")
    result["target_date"] = pd.to_datetime(result["target_date"], errors="raise")
    if result.empty:
        raise ValueError(f"{context} predictions must not be empty")
    if result.duplicated(["model", *_KEY_COLUMNS]).any():
        raise ValueError(f"{context} predictions contain duplicate model origins")
    if not (result["origin_date"] < result["target_date"]).all():
        raise ValueError(f"{context} target dates must follow origins")
    if set(result["evaluation_split"].astype(str)) != {"selection", "holdout"}:
        raise ValueError(f"{context} must contain selection and holdout rows")
    if not pd.to_numeric(result["gap"], errors="coerce").eq(1).all():
        raise ValueError(f"{context} must preserve the official gap=1")
    probability = result[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(probability).all():
        raise ValueError(f"{context} probabilities must be finite")
    if ((probability < 0.0) | (probability > 1.0)).any():
        raise ValueError(f"{context} probabilities must be in [0, 1]")
    if not np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError(f"{context} probability rows must sum to one")
    invalid_actual = sorted(set(result["actual"].astype(str)).difference(STATE_ORDER))
    if invalid_actual:
        raise ValueError(f"{context} has unsupported actual states: {invalid_actual}")
    invalid_current = sorted(
        set(result["current_state"].astype(str)).difference(STATE_ORDER)
    )
    if invalid_current:
        raise ValueError(
            f"{context} has unsupported current states: {invalid_current}"
        )
    expected = np.asarray(
        [STATE_ORDER[position] for position in probability.argmax(axis=1)],
        dtype=object,
    )
    if not np.array_equal(expected, result["predicted"].astype(str).to_numpy()):
        raise ValueError(f"{context} predicted states do not match probabilities")
    return result.sort_values(["model", *_KEY_COLUMNS], ignore_index=True)


def _assert_matched_origins(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    context: str,
) -> None:
    left = reference[list(_AUDIT_COLUMNS)].sort_values(
        list(_KEY_COLUMNS), ignore_index=True
    )
    right = candidate[list(_AUDIT_COLUMNS)].sort_values(
        list(_KEY_COLUMNS), ignore_index=True
    )
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False, check_like=False)
    except AssertionError as exc:
        raise ValueError(
            f"mechanism ablation origin mismatch for {context}; "
            "all tracks must share exact origins, targets, actuals, and splits"
        ) from exc


def _per_row_losses(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    probability = frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    positions = {state: position for position, state in enumerate(STATE_ORDER)}
    actual_position = np.asarray(
        [positions[state] for state in frame["actual"].astype(str)], dtype=int
    )
    actual_probability = probability[np.arange(len(frame)), actual_position]
    one_hot = np.zeros_like(probability)
    one_hot[np.arange(len(frame)), actual_position] = 1.0
    return (
        -np.log(np.clip(actual_probability, 1e-9, 1.0)),
        np.sum((probability - one_hot) ** 2, axis=1),
    )


def _build_leaderboard(predictions: pd.DataFrame) -> pd.DataFrame:
    full = predictions.loc[
        predictions["track"].eq("full") & predictions["model"].eq("xgboost")
    ]
    rows: list[dict[str, Any]] = []
    for (track, model, split), subset in predictions.groupby(
        ["track", "model", "evaluation_split"], sort=False
    ):
        subset = subset.sort_values(list(_KEY_COLUMNS), ignore_index=True)
        metrics = evaluate_predictions(subset).iloc[0].to_dict()
        comparison_family = str(subset["comparison_family"].iloc[0])
        fixed_xgb = comparison_family == "fixed_xgboost_feature_tracks"
        paired_log_loss_delta: float | None = None
        paired_brier_delta: float | None = None
        if fixed_xgb:
            reference = full.loc[full["evaluation_split"].eq(split)].sort_values(
                list(_KEY_COLUMNS), ignore_index=True
            )
            _assert_matched_origins(
                reference, subset, context=f"{track}/{model}/{split}"
            )
            candidate_log, candidate_brier = _per_row_losses(subset)
            reference_log, reference_brier = _per_row_losses(reference)
            paired_log_loss_delta = float(np.mean(candidate_log - reference_log))
            paired_brier_delta = float(np.mean(candidate_brier - reference_brier))
        rows.append(
            {
                "track": str(track),
                "model": str(model),
                "evaluation_split": str(split),
                "comparison_family": comparison_family,
                "model_mechanics_comparable_to_full": fixed_xgb,
                "cross_family_ranked": False,
                **{key: value for key, value in metrics.items() if key != "model"},
                "paired_log_loss_delta_vs_full": paired_log_loss_delta,
                "paired_brier_delta_vs_full": paired_brier_delta,
            }
        )
    return pd.DataFrame(rows)


def run_mechanism_ablation(
    features: pd.DataFrame,
    states: pd.Series,
    feature_role_manifest: Sequence[Mapping[str, Any]],
    *,
    profile: str | BenchmarkProfile,
    selection_end: str | pd.Timestamp,
    specification: MechanismAblationSpec | None = None,
    gap: int = 1,
    minimum_train_weeks: int | None = None,
    selection_max_origins: int | None = None,
    minimum_selection_predictions: int = 12,
    minimum_holdout_predictions: int = 12,
    random_state: int = 17,
    model_workers: int = 1,
    checkpoint_directory: str | Path | None = None,
    source_fingerprint_sha256: str | None = None,
    benchmark_runner: Callable[..., Any] = run_benchmark,
    progress: Callable[[str], None] | None = None,
) -> MechanismAblationResult:
    """Run all five tracks and reject any common-origin discrepancy.

    The feature-role manifest must assign every column exactly once.  The
    function intentionally does not take a preselected champion: each reported
    model is fixed by the versioned specification before the run.
    """

    if not isinstance(features, pd.DataFrame) or not isinstance(states, pd.Series):
        raise TypeError("features and states must be pandas objects")
    if not isinstance(features.index, pd.DatetimeIndex):
        raise TypeError("features must use a DatetimeIndex")
    if features.empty or features.shape[1] == 0:
        raise ValueError("features must not be empty")
    if not features.index.equals(states.index):
        raise ValueError("features and states must use the same index")
    if features.columns.has_duplicates:
        raise ValueError("feature columns must be unique")
    numeric = features.apply(pd.to_numeric, errors="coerce")
    coercion_missing = numeric.isna() & ~features.isna()
    if bool(coercion_missing.to_numpy().any()):
        raise TypeError("all features must be numeric")
    if bool(np.isinf(numeric.to_numpy(dtype=float)).any()):
        raise ValueError("features contain infinite values")
    if int(gap) != 1:
        raise ValueError("mechanism ablation v2 is frozen to the official gap=1")
    if checkpoint_directory is None and source_fingerprint_sha256 is not None:
        raise ValueError(
            "source_fingerprint_sha256 requires checkpoint_directory"
        )
    if checkpoint_directory is not None and not source_fingerprint_sha256:
        raise ValueError(
            "checkpoint_directory requires source_fingerprint_sha256"
        )

    spec = specification or load_mechanism_ablation_spec()
    if tuple(track.track_id for track in spec.tracks) != MECHANISM_TRACKS:
        raise ValueError("specification does not contain the exact five tracks")
    roles, role_manifest = _normalise_role_manifest(
        feature_role_manifest, features.columns
    )
    resolved_profile = resolve_profile(profile)
    cutoff = pd.Timestamp(selection_end)
    track_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    reference: pd.DataFrame | None = None

    for track in spec.tracks:
        columns = _track_columns(features, roles, track)
        track_features = (
            pd.DataFrame(
                {"__state_only_constant__": np.zeros(len(features), dtype=float)},
                index=features.index,
            )
            if track.track_id == "state_only"
            else numeric.loc[:, list(columns)]
        )
        if progress is not None:
            progress(f"mechanism ablation: {track.track_id}")
        track_progress = (
            None
            if progress is None
            else lambda message, track_id=track.track_id: progress(
                f"mechanism ablation: {track_id}: {message}"
            )
        )
        benchmark = benchmark_runner(
            track_features,
            states,
            profile=resolved_profile,
            models=track.benchmark_models,
            include_hmm=False,
            gap=1,
            minimum_train_weeks=minimum_train_weeks,
            random_state=random_state,
            selection_end=cutoff,
            selection_max_origins=selection_max_origins,
            model_workers=model_workers,
            minimum_selection_predictions=minimum_selection_predictions,
            minimum_holdout_predictions=minimum_holdout_predictions,
            progress=track_progress,
            checkpoint_directory=(
                None
                if checkpoint_directory is None
                else Path(checkpoint_directory) / track.track_id
            ),
            source_fingerprint_sha256=source_fingerprint_sha256,
        )
        source = getattr(benchmark, "predictions", None)
        if not isinstance(source, pd.DataFrame):
            raise TypeError(
                f"benchmark for {track.track_id} did not return predictions"
            )
        candidate = source.loc[
            source["model"].astype(str).isin(track.reported_models)
        ].copy()
        candidate = _validate_predictions(candidate, context=track.track_id)
        for model in track.reported_models:
            model_rows = candidate.loc[candidate["model"].astype(str).eq(model)]
            if model_rows.empty:
                raise ValueError(
                    f"track {track.track_id} omitted reported model {model}"
                )
            if reference is None:
                reference = model_rows.copy()
            else:
                _assert_matched_origins(
                    reference,
                    model_rows,
                    context=f"{track.track_id}/{model}",
                )
        candidate.insert(0, "track", track.track_id)
        candidate.insert(2, "comparison_family", track.comparison_family)
        candidate.insert(3, "interpretation", track.interpretation)
        track_frames.append(candidate)
        manifest_rows.append(
            {
                "track": track.track_id,
                "feature_roles": track.feature_roles,
                "feature_count": 0 if track.track_id == "state_only" else len(columns),
                "feature_columns": () if track.track_id == "state_only" else columns,
                "feature_sha256": _canonical_sha256(
                    [] if track.track_id == "state_only" else list(columns)
                ),
                "benchmark_models": track.benchmark_models,
                "reported_models": track.reported_models,
                "comparison_family": track.comparison_family,
                "interpretation": track.interpretation,
                "gap": 1,
            }
        )

    if reference is None:
        raise RuntimeError("mechanism ablation produced no reference origins")
    predictions = pd.concat(track_frames, ignore_index=True, sort=False).sort_values(
        ["origin_date", "target_date", "track", "model"], ignore_index=True
    )
    common_origins = reference[list(_AUDIT_COLUMNS)].sort_values(
        list(_KEY_COLUMNS), ignore_index=True
    )
    return MechanismAblationResult(
        predictions=predictions,
        leaderboard=_build_leaderboard(predictions),
        track_manifest=pd.DataFrame(manifest_rows),
        role_manifest=role_manifest,
        common_origins=common_origins,
        specification_sha256=spec.sha256,
    )


def mechanism_ablation_manifest_document(
    result: MechanismAblationResult,
) -> dict[str, Any]:
    """Build a canonical, hash-bound audit sidecar without prediction rows."""

    if tuple(result.track_manifest["track"].astype(str)) != MECHANISM_TRACKS:
        raise ValueError("track manifest order is invalid")
    origins = result.common_origins.sort_values(list(_KEY_COLUMNS), ignore_index=True)
    origin_records = [
        {
            "origin_date": pd.Timestamp(row.origin_date).isoformat(),
            "target_date": pd.Timestamp(row.target_date).isoformat(),
            "evaluation_split": str(row.evaluation_split),
            "current_state": str(row.current_state),
            "actual": str(row.actual),
            "train_size": int(row.train_size),
            "gap": int(row.gap),
        }
        for row in origins.itertuples(index=False)
    ]
    tracks = []
    for row in result.track_manifest.to_dict(orient="records"):
        tracks.append(
            {
                "track": str(row["track"]),
                "feature_roles": list(row["feature_roles"]),
                "feature_count": int(row["feature_count"]),
                "feature_columns": list(row["feature_columns"]),
                "feature_sha256": str(row["feature_sha256"]),
                "benchmark_models": list(row["benchmark_models"]),
                "reported_models": list(row["reported_models"]),
                "comparison_family": str(row["comparison_family"]),
                "interpretation": str(row["interpretation"]),
                "gap": int(row["gap"]),
            }
        )
    body = {
        "schema_version": MECHANISM_ABLATION_SCHEMA_VERSION,
        "specification_sha256": result.specification_sha256,
        "origin_contract": "exact_origin_target_actual_and_split_match",
        "origin_count": len(origin_records),
        "origins_sha256": _canonical_sha256(origin_records),
        "cross_family_ranking": False,
        "tracks": tracks,
    }
    return {**body, "sha256": _canonical_sha256(body)}


__all__ = [
    "FEATURE_ROLES",
    "MECHANISM_ABLATION_SCHEMA_VERSION",
    "MECHANISM_TRACKS",
    "MechanismAblationResult",
    "MechanismAblationSpec",
    "MechanismTrackSpec",
    "load_mechanism_ablation_spec",
    "mechanism_ablation_manifest_document",
    "run_mechanism_ablation",
]
