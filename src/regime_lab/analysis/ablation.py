"""Pre-registered, common-origin feature-family ablations.

The production feature set is frozen before the post-2023 diagnostic period.
This module therefore reports the diagnostic period, but derives every variant
ranking from the pre-cutoff selection rows only.  It deliberately reuses the
main benchmark's all-structural XGBoost forecasts so the reference model is not
silently refit under a different walk-forward contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .labels import STATE_ORDER
from .models import BenchmarkProfile, resolve_profile
from .validation import PROBABILITY_COLUMNS, evaluate_predictions, run_benchmark


LEGACY_GROUP = "legacy_v3"
STRUCTURAL_GROUPS: tuple[str, ...] = (
    "sector_breadth",
    "broad_size_style_breadth",
    "cross_asset_breadth",
    "treasury_curve",
    "bank_credit",
    "financial_conditions",
    "release_innovation",
)
EXPECTED_GROUPS: tuple[str, ...] = (LEGACY_GROUP, *STRUCTURAL_GROUPS)

VARIANT_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("legacy_v3", (LEGACY_GROUP,)),
    (
        "legacy_plus_market_structure",
        (
            LEGACY_GROUP,
            "sector_breadth",
            "broad_size_style_breadth",
            "cross_asset_breadth",
        ),
    ),
    ("legacy_plus_treasury_curve", (LEGACY_GROUP, "treasury_curve")),
    ("legacy_plus_bank_credit", (LEGACY_GROUP, "bank_credit")),
    (
        "legacy_plus_financial_conditions",
        (LEGACY_GROUP, "financial_conditions"),
    ),
    (
        "legacy_plus_release_innovation",
        (LEGACY_GROUP, "release_innovation"),
    ),
    ("all_structural", EXPECTED_GROUPS),
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
_REQUIRED_PREDICTION_COLUMNS: frozenset[str] = frozenset(
    (
        *_AUDIT_COLUMNS,
        "model",
        "predicted",
        *PROBABILITY_COLUMNS,
        "train_size",
        "gap",
        "fallback",
        "fallback_reason",
    )
)
FEATURE_ABLATION_MANIFEST_SCHEMA_VERSION = "1.0.0"
FEATURE_ABLATION_CONTRACT: dict[str, object] = {
    "anchor_model": "xgboost",
    "reference_variant": "legacy_v3",
    "published_variant": "all_structural",
    "primary_period": "pre_2023_selection_oos",
    "post_2023_role": "retrospective_diagnostic_only",
    "may_change_published_variant": False,
}


@dataclass(frozen=True)
class FeatureAblationResult:
    """Auditable outputs of the fixed feature-family comparison."""

    predictions: pd.DataFrame
    leaderboard: pd.DataFrame
    manifest: pd.DataFrame
    common_origins: pd.DataFrame


def _normalise_manifest(
    feature_group_manifest: Sequence[Mapping[str, Any]],
    feature_columns: pd.Index,
) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
    if isinstance(feature_group_manifest, (str, bytes)):
        raise TypeError("feature_group_manifest must be a sequence of mappings")
    records = list(feature_group_manifest)
    if not records:
        raise ValueError("feature_group_manifest must not be empty")
    if feature_columns.has_duplicates:
        raise ValueError("features must not contain duplicate columns")
    available = {str(column) for column in feature_columns}
    groups: dict[str, tuple[str, ...]] = {}
    assigned: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("every feature manifest row must be a mapping")
        group_id = str(record.get("id", "")).strip()
        if not group_id:
            raise ValueError("feature manifest ids must not be empty")
        if group_id in groups:
            raise ValueError(f"duplicate feature manifest id: {group_id}")
        raw_features = record.get("features")
        if isinstance(raw_features, (str, bytes)) or not isinstance(
            raw_features, Sequence
        ):
            raise TypeError(f"manifest group {group_id} features must be a sequence")
        columns = tuple(str(column) for column in raw_features)
        if not columns:
            raise ValueError(f"manifest group {group_id} must not be empty")
        if len(columns) != len(set(columns)):
            raise ValueError(f"manifest group {group_id} contains duplicates")
        declared_count = record.get("feature_count", len(columns))
        if int(declared_count) != len(columns):
            raise ValueError(f"manifest group {group_id} feature_count mismatch")
        unknown_columns = sorted(set(columns).difference(available))
        if unknown_columns:
            raise ValueError(
                f"manifest group {group_id} contains unknown features: "
                f"{unknown_columns}"
            )
        overlap = sorted(set(columns).intersection(assigned))
        if overlap:
            raise ValueError(
                "feature manifest assigns columns more than once: "
                f"{overlap}"
            )
        groups[group_id] = columns
        assigned.update(columns)

    expected = set(EXPECTED_GROUPS)
    unknown_groups = sorted(set(groups).difference(expected))
    missing_groups = sorted(expected.difference(groups))
    if unknown_groups or missing_groups:
        raise ValueError(
            "feature manifest must contain exactly the frozen v4 groups; "
            f"missing={missing_groups}, unknown={unknown_groups}"
        )

    # The dataset manifest is built before pipeline-only boundary and duration
    # features are appended.  Those unassigned columns are intentionally common
    # controls in every variant, rather than a separate candidate family.
    extras = tuple(str(column) for column in feature_columns if str(column) not in assigned)
    return groups, extras


def feature_columns_sha256(columns: Sequence[str]) -> str:
    """Hash an exact ordered feature list using canonical JSON."""

    encoded = json.dumps(
        [str(column) for column in columns],
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def feature_ablation_manifest_document(manifest: pd.DataFrame) -> dict[str, Any]:
    """Build the canonical payload-linked feature-ablation sidecar."""

    required = {
        "variant",
        "group_ids",
        "feature_count",
        "extra_feature_count",
        "feature_columns",
        "feature_sha256",
        "reused_main_benchmark",
    }
    if not isinstance(manifest, pd.DataFrame) or not required.issubset(manifest.columns):
        raise ValueError("feature ablation manifest columns are incomplete")
    expected_order = [name for name, _groups in VARIANT_GROUPS]
    if manifest["variant"].astype(str).tolist() != expected_order:
        raise ValueError("feature ablation manifest variant order is invalid")
    variants: list[dict[str, Any]] = []
    for raw in manifest.to_dict(orient="records"):
        columns = [str(column) for column in raw["feature_columns"]]
        if len(columns) != len(set(columns)):
            raise ValueError(f"feature variant {raw['variant']} contains duplicates")
        expected_hash = feature_columns_sha256(columns)
        if str(raw["feature_sha256"]) != expected_hash:
            raise ValueError(f"feature variant {raw['variant']} hash mismatch")
        if int(raw["feature_count"]) != len(columns):
            raise ValueError(f"feature variant {raw['variant']} count mismatch")
        variants.append(
            {
                "variant": str(raw["variant"]),
                "group_ids": str(raw["group_ids"]),
                "feature_count": int(raw["feature_count"]),
                "extra_feature_count": int(raw["extra_feature_count"]),
                "feature_columns": columns,
                "feature_sha256": expected_hash,
                "reused_main_benchmark": bool(raw["reused_main_benchmark"]),
            }
        )
    body: dict[str, Any] = {
        "schema_version": FEATURE_ABLATION_MANIFEST_SCHEMA_VERSION,
        **FEATURE_ABLATION_CONTRACT,
        "variants": variants,
    }
    serialized = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {**body, "sha256": hashlib.sha256(serialized).hexdigest()}


def _variant_columns(
    feature_columns: pd.Index,
    groups: Mapping[str, tuple[str, ...]],
    extras: tuple[str, ...],
) -> tuple[dict[str, tuple[str, ...]], pd.DataFrame]:
    common = set(extras)
    variants: dict[str, tuple[str, ...]] = {}
    manifest_rows: list[dict[str, object]] = []
    for variant, group_ids in VARIANT_GROUPS:
        selected = set(common)
        for group_id in group_ids:
            selected.update(groups[group_id])
        columns = tuple(
            str(column) for column in feature_columns if str(column) in selected
        )
        if not columns:
            raise ValueError(f"feature variant {variant} is empty")
        variants[variant] = columns
        manifest_rows.append(
            {
                "variant": variant,
                "group_ids": "|".join(group_ids),
                "feature_count": int(len(columns)),
                "extra_feature_count": int(len(extras)),
                "feature_columns": columns,
                "feature_sha256": feature_columns_sha256(columns),
                "reused_main_benchmark": variant == "all_structural",
            }
        )
    if variants["all_structural"] != tuple(str(item) for item in feature_columns):
        raise ValueError("all_structural must contain every input feature exactly once")
    return variants, pd.DataFrame(manifest_rows)


def _validate_probability_predictions(
    frame: pd.DataFrame,
    *,
    context: str,
) -> pd.DataFrame:
    missing = sorted(_REQUIRED_PREDICTION_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"{context} predictions missing columns: {missing}")
    result = frame.copy()
    result["origin_date"] = pd.to_datetime(result["origin_date"], errors="raise")
    result["target_date"] = pd.to_datetime(result["target_date"], errors="raise")
    if result.empty:
        raise ValueError(f"{context} predictions must not be empty")
    if result.duplicated(list(_KEY_COLUMNS)).any():
        raise ValueError(f"{context} predictions contain duplicate origins")
    if not (result["origin_date"] < result["target_date"]).all():
        raise ValueError(f"{context} target dates must follow origin dates")
    if set(result["evaluation_split"].astype(str)) != {"selection", "holdout"}:
        raise ValueError(f"{context} must contain selection and holdout rows")
    for column in ("actual", "current_state", "predicted"):
        invalid = sorted(set(result[column].astype(str)).difference(STATE_ORDER))
        if invalid:
            raise ValueError(f"{context} contains unsupported {column}: {invalid}")
    probability = result[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(probability).all():
        raise ValueError(f"{context} predictions contain non-finite probabilities")
    if (probability < 0.0).any() or (probability > 1.0).any():
        raise ValueError(f"{context} probabilities must be in [0, 1]")
    if not np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError(f"{context} probability rows must sum to one")
    expected = np.asarray(
        [STATE_ORDER[position] for position in probability.argmax(axis=1)],
        dtype=object,
    )
    if not np.array_equal(expected, result["predicted"].astype(str).to_numpy()):
        raise ValueError(f"{context} predicted labels do not match probabilities")
    return result.sort_values(list(_KEY_COLUMNS), ignore_index=True)


def _extract_main_xgboost(main_benchmark: Any) -> pd.DataFrame:
    if not hasattr(main_benchmark, "predictions"):
        raise TypeError("main_benchmark must expose predictions")
    source = main_benchmark.predictions
    if not isinstance(source, pd.DataFrame):
        raise TypeError("main_benchmark.predictions must be a DataFrame")
    frame = source.loc[source["model"].astype(str).eq("xgboost")].copy()
    if frame.empty:
        raise ValueError("main_benchmark does not contain XGBoost OOS predictions")
    return _validate_probability_predictions(frame, context="all_structural")


def _assert_same_origins(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    variant: str,
) -> None:
    left = reference[list(_AUDIT_COLUMNS)].reset_index(drop=True)
    right = candidate[list(_AUDIT_COLUMNS)].reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False, check_like=False)
    except AssertionError as exc:
        raise ValueError(
            f"feature variant {variant} does not share exact origins and actuals"
        ) from exc


def _per_row_losses(frame: pd.DataFrame) -> pd.DataFrame:
    probability = frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    positions = {state: index for index, state in enumerate(STATE_ORDER)}
    actual = frame["actual"].astype(str).to_numpy()
    actual_probability = np.asarray(
        [probability[row, positions[state]] for row, state in enumerate(actual)],
        dtype=float,
    )
    one_hot = np.zeros_like(probability)
    one_hot[
        np.arange(len(frame)),
        np.asarray([positions[state] for state in actual], dtype=int),
    ] = 1.0
    return pd.DataFrame(
        {
            "log_loss": -np.log(np.clip(actual_probability, 1e-9, 1.0)),
            "brier": np.sum((probability - one_hot) ** 2, axis=1),
        },
        index=frame.index,
    )


def _build_leaderboard(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    legacy = predictions.loc[predictions["variant"].eq("legacy_v3")]
    for evaluation_split, role in (
        ("selection", "selection_primary"),
        ("holdout", "post_2023_retrospective_diagnostic"),
    ):
        legacy_split = legacy.loc[
            legacy["evaluation_split"].eq(evaluation_split)
        ].sort_values(list(_KEY_COLUMNS), ignore_index=True)
        legacy_losses = _per_row_losses(legacy_split)
        for variant, _ in VARIANT_GROUPS:
            subset = predictions.loc[
                predictions["variant"].eq(variant)
                & predictions["evaluation_split"].eq(evaluation_split)
            ].sort_values(list(_KEY_COLUMNS), ignore_index=True)
            _assert_same_origins(legacy_split, subset, variant=variant)
            metrics = evaluate_predictions(subset).iloc[0].to_dict()
            losses = _per_row_losses(subset)
            rows.append(
                {
                    "variant": variant,
                    "evaluation_split": evaluation_split,
                    "role": role,
                    **metrics,
                    "paired_n": int(len(subset)),
                    # Candidate minus legacy: negative values indicate an
                    # improvement and remain exactly paired by target date.
                    "paired_log_loss_delta_vs_legacy": float(
                        (losses["log_loss"] - legacy_losses["log_loss"]).mean()
                    ),
                    "paired_brier_delta_vs_legacy": float(
                        (losses["brier"] - legacy_losses["brier"]).mean()
                    ),
                }
            )
    result = pd.DataFrame(rows)
    selection = result.loc[result["evaluation_split"].eq("selection")].copy()
    eligible = selection.loc[selection["fallback_count"].eq(0)].copy()
    if eligible.empty:
        eligible = selection.loc[selection["variant"].eq("legacy_v3")].copy()
    ranked = eligible.sort_values(
        ["log_loss", "brier", "calibration_error", "variant"],
        kind="stable",
    )
    ranks = {str(value): rank for rank, value in enumerate(ranked["variant"], 1)}
    ineligible = [
        variant for variant, _ in VARIANT_GROUPS if variant not in ranks
    ]
    for variant in ineligible:
        ranks[variant] = len(ranks) + 1
    winner = str(ranked.iloc[0]["variant"])
    result["selection_rank"] = result["variant"].map(ranks).astype(int)
    result["selection_winner"] = result["variant"].eq(winner)
    ordered = [
        "variant",
        "evaluation_split",
        "role",
        "selection_rank",
        "selection_winner",
        "model",
        *[
            column
            for column in result.columns
            if column
            not in {
                "variant",
                "evaluation_split",
                "role",
                "selection_rank",
                "selection_winner",
                "model",
            }
        ],
    ]
    return result[ordered].sort_values(
        ["evaluation_split", "selection_rank", "variant"],
        key=lambda series: series.map({"selection": 0, "holdout": 1})
        if series.name == "evaluation_split"
        else series,
        ignore_index=True,
    )


def run_feature_ablation(
    features: pd.DataFrame,
    states: pd.Series,
    feature_group_manifest: Sequence[Mapping[str, Any]],
    main_benchmark: Any,
    *,
    profile: str | BenchmarkProfile,
    selection_end: str | pd.Timestamp,
    gap: int = 1,
    minimum_train_weeks: int | None = None,
    selection_max_origins: int | None = None,
    minimum_selection_predictions: int = 12,
    minimum_holdout_predictions: int = 12,
    random_state: int = 17,
    model_workers: int = 1,
    progress: Callable[[str], None] | None = None,
) -> FeatureAblationResult:
    """Run the seven fixed XGBoost feature variants on identical OOS origins.

    ``all_structural`` is copied from ``main_benchmark``.  The other six
    variants are rerun with the same profile, purge gap, training floor, split
    cutoff, and origin budgets.  Variant ranks are computed from selection rows
    only; holdout rows are never consulted by that calculation.
    """

    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a DataFrame")
    if not isinstance(states, pd.Series):
        raise TypeError("states must be a Series")
    if features.empty or features.shape[1] == 0:
        raise ValueError("features must not be empty")
    if not isinstance(features.index, pd.DatetimeIndex):
        raise TypeError("features must use a DatetimeIndex")
    if not features.index.equals(states.index):
        raise ValueError("features and states must use the same index")
    numeric = features.apply(pd.to_numeric, errors="coerce")
    original_missing = features.isna()
    coercion_missing = numeric.isna() & ~original_missing
    if bool(coercion_missing.to_numpy().any()):
        raise TypeError("all feature columns must be numeric")
    if bool(np.isinf(numeric.to_numpy(dtype=float)).any()):
        raise ValueError("features contain infinite values")
    resolved_profile = resolve_profile(profile)
    if hasattr(main_benchmark, "profile") and main_benchmark.profile != resolved_profile:
        raise ValueError("profile does not match main_benchmark.profile")
    cutoff = pd.Timestamp(selection_end)
    if features.index.tz is None:
        if cutoff.tzinfo is not None:
            cutoff = cutoff.tz_localize(None)
    elif cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize(features.index.tz)
    else:
        cutoff = cutoff.tz_convert(features.index.tz)
    main_cutoff = getattr(main_benchmark, "selection_end", None)
    if main_cutoff is not None:
        left = pd.Timestamp(main_cutoff)
        right = cutoff
        if left.tzinfo is not None and right.tzinfo is None:
            right = right.tz_localize(left.tzinfo)
        elif left.tzinfo is None and right.tzinfo is not None:
            right = right.tz_localize(None)
        elif left.tzinfo is not None and right.tzinfo is not None:
            right = right.tz_convert(left.tzinfo)
        if left != right:
            raise ValueError("selection_end does not match main_benchmark")

    groups, extras = _normalise_manifest(feature_group_manifest, features.columns)
    variants, manifest = _variant_columns(features.columns, groups, extras)
    reference = _extract_main_xgboost(main_benchmark)
    if not (reference.loc[
        reference["evaluation_split"].eq("selection"), "target_date"
    ] < cutoff).all():
        raise ValueError("selection predictions cross selection_end")
    if not (reference.loc[
        reference["evaluation_split"].eq("holdout"), "target_date"
    ] >= cutoff).all():
        raise ValueError("holdout predictions precede selection_end")

    output_frames: list[pd.DataFrame] = []
    for variant, _ in VARIANT_GROUPS:
        if progress is not None:
            progress(f"feature ablation: {variant}")
        if variant == "all_structural":
            candidate = reference.copy()
        else:
            nested_progress = None
            if progress is not None:
                nested_progress = lambda message, name=variant: progress(
                    f"feature ablation {name}: {message}"
                )
            benchmark = run_benchmark(
                features.loc[:, list(variants[variant])],
                states,
                profile=resolved_profile,
                models=("majority", "xgboost"),
                include_hmm=False,
                gap=gap,
                minimum_train_weeks=minimum_train_weeks,
                random_state=random_state,
                selection_end=cutoff,
                selection_max_origins=selection_max_origins,
                model_workers=model_workers,
                minimum_selection_predictions=minimum_selection_predictions,
                minimum_holdout_predictions=minimum_holdout_predictions,
                progress=nested_progress,
            )
            candidate = benchmark.predictions.loc[
                benchmark.predictions["model"].astype(str).eq("xgboost")
            ].copy()
            candidate = _validate_probability_predictions(
                candidate, context=variant
            )
            # Structural augmentation can apply a stricter common-origin
            # intersection than the base XGBoost rerun (the one-week event
            # track starts diagnostics at the cutoff origin, while the base
            # multiclass holdout has one target crossing that cutoff).  Every
            # ablation must therefore be evaluated on the final published
            # benchmark's exact keys, never on that extra base-only origin.
            candidate = candidate.merge(
                reference.loc[:, list(_KEY_COLUMNS)],
                on=list(_KEY_COLUMNS),
                how="inner",
                validate="one_to_one",
            )
            candidate = _validate_probability_predictions(
                candidate, context=f"{variant} common-origin"
            )
            _assert_same_origins(reference, candidate, variant=variant)
        candidate.insert(0, "variant", variant)
        candidate["reused_main_benchmark"] = variant == "all_structural"
        output_frames.append(candidate)

    predictions = pd.concat(output_frames, ignore_index=True, sort=False)
    predictions = predictions.sort_values(
        ["origin_date", "target_date", "variant"], ignore_index=True
    )
    leaderboard = _build_leaderboard(predictions)
    common_origins = reference[list(_AUDIT_COLUMNS)].reset_index(drop=True)
    return FeatureAblationResult(
        predictions=predictions,
        leaderboard=leaderboard,
        manifest=manifest,
        common_origins=common_origins,
    )


__all__ = [
    "EXPECTED_GROUPS",
    "STRUCTURAL_GROUPS",
    "VARIANT_GROUPS",
    "FeatureAblationResult",
    "run_feature_ablation",
]
