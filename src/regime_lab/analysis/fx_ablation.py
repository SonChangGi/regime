"""Prospective, common-origin gate for the preregistered H.10 FX variants."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .fx import FXFeatureResult
from .labels import STATE_ORDER
from .models import align_probabilities, class_prior_probabilities
from .validation import _holm_adjusted_pvalues
from .validation import _moving_block_bootstrap_pvalues


FX_VARIANT_ORDER = (
    "v4_control",
    "v4_plus_broad_index",
    "v4_plus_bilateral_panel",
    "v4_plus_all_fx",
)

FX_ABLATION_OOS_COLUMNS: tuple[str, ...] = (
    "origin_date",
    "target_date",
    "variant",
    "evaluation_split",
    "current_state",
    "actual",
    "p_risk_on",
    "p_transition",
    "p_risk_off",
    "train_size",
    "gap",
    "last_train_target",
    "purged_origin_count",
    "fallback",
    "fallback_reason",
    "common_origins_sha256",
)

FX_ABLATION_MINIMUM_COMMON_WEEKS = 156
FX_ABLATION_MINIMUM_TRAIN_WEEKS = 104
FX_ABLATION_COMMON_ORIGIN_REQUIRED_PAIRS = 9
FX_ABLATION_PURGE_WEEKS = 1
FX_ABLATION_BOOTSTRAP_BLOCK_WEEKS = 13
FX_ABLATION_BOOTSTRAP_RESAMPLES = 1_999
FX_ABLATION_BOOTSTRAP_SEED = 17
FX_ABLATION_ALPHA = 0.05
FX_ABLATION_MINIMUM_LOG_LOSS_IMPROVEMENT = 0.05
FX_ABLATION_BRIER_TOLERANCE = 0.01
FX_ABLATION_REGULARIZATION_C = 0.10

_FIXED_BILATERAL_CODES = (
    "eur",
    "jpy",
    "gbp",
    "chf",
    "cad",
    "aud",
    "cny",
    "mxn",
    "brl",
)


def _model_feature_columns(features: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        str(column)
        for column in features.columns
        if not str(column).endswith("__usd_log_level")
    )


def fx_ablation_variants(features: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    """Return the frozen four-way feature contract in deterministic order."""

    columns = _model_feature_columns(features)
    broad = tuple(
        column for column in columns if column.startswith("fx__brd__")
    )
    bilateral = tuple(
        column
        for column in columns
        if column.startswith("fx__bilateral__")
        or any(
            column.startswith(f"fx__{code}__")
            for code in ("eur", "jpy", "gbp", "chf", "cad", "aud", "cny", "mxn", "brl")
        )
    )
    all_fx = columns
    if not broad or not bilateral or not all_fx:
        raise ValueError("FX feature result does not satisfy the v5 ablation contract")
    return {
        "v4_control": (),
        "v4_plus_broad_index": broad,
        "v4_plus_bilateral_panel": bilateral,
        "v4_plus_all_fx": all_fx,
    }


def align_fx_features_to_cutoffs(
    result: FXFeatureResult,
    cutoffs: pd.DatetimeIndex,
) -> pd.DataFrame:
    """As-of join each model cutoff to the latest available H.10 feature row."""

    if not isinstance(result, FXFeatureResult):
        raise TypeError("result must be an FXFeatureResult")
    if not isinstance(cutoffs, pd.DatetimeIndex):
        raise TypeError("cutoffs must be a DatetimeIndex")
    if cutoffs.empty or cutoffs.has_duplicates or not cutoffs.is_monotonic_increasing:
        raise ValueError("cutoffs must be non-empty, unique, and increasing")

    cutoff_utc = (
        (cutoffs + pd.offsets.Hour(16))
        .tz_localize("America/New_York")
        .tz_convert("UTC")
        if cutoffs.tz is None
        else cutoffs.tz_convert("UTC")
    )
    source = result.features.copy()
    source.insert(0, "fx_observation_week", pd.DatetimeIndex(source.index))
    source.insert(
        0,
        "fx_feature_available_at",
        pd.to_datetime(
            result.coverage["feature_available_at"], utc=True, errors="coerce"
        ),
    )
    quarantine = result.coverage.get(
        "archive_correction_quarantined",
        pd.Series(False, index=result.coverage.index, dtype="bool"),
    )
    correction_available = result.coverage.get(
        "archive_correction_available_at",
        pd.Series(
            pd.NaT,
            index=result.coverage.index,
            dtype="datetime64[ns, UTC]",
        ),
    )
    quarantine_until = result.coverage.get(
        "archive_correction_quarantine_until_week",
        pd.Series(
            pd.NaT,
            index=result.coverage.index,
            dtype="datetime64[ns]",
        ),
    )
    source.insert(
        0,
        "fx_archive_correction_quarantined",
        quarantine.fillna(False).astype(bool),
    )
    source.insert(
        0,
        "fx_archive_correction_available_at",
        pd.to_datetime(correction_available, utc=True, errors="coerce"),
    )
    source.insert(
        0,
        "fx_archive_correction_quarantine_until_week",
        pd.to_datetime(quarantine_until, errors="coerce"),
    )
    source = source.loc[source["fx_feature_available_at"].notna()].copy()
    source = source.sort_values(
        ["fx_feature_available_at", "fx_observation_week"],
        kind="mergesort",
    )
    target = pd.DataFrame({"model_cutoff": cutoff_utc}).sort_values("model_cutoff")
    if source.empty:
        aligned = target.copy()
        for column in ("fx_feature_available_at", "fx_observation_week", *result.features.columns):
            aligned[column] = pd.NA
    else:
        aligned = pd.merge_asof(
            target,
            source,
            left_on="model_cutoff",
            right_on="fx_feature_available_at",
            direction="backward",
            allow_exact_matches=True,
        )
    aligned.index = cutoffs
    cutoff_calendar = pd.DatetimeIndex(cutoff_utc).tz_convert(
        "America/New_York"
    ).tz_localize(None).normalize()
    observation_calendar = pd.to_datetime(
        aligned["fx_observation_week"], errors="coerce"
    ).dt.tz_localize(None).dt.normalize()
    aligned["fx_observation_age_days"] = (
        pd.Series(cutoff_calendar, index=aligned.index) - observation_calendar
    ).dt.days.astype("Int64")
    return aligned.drop(columns="model_cutoff")


def _manifest(variants: Mapping[str, Sequence[str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for name in FX_VARIANT_ORDER:
        columns = tuple(str(column) for column in variants[name])
        encoded = json.dumps(
            list(columns), separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        output.append(
            {
                "variant": name,
                "feature_count": len(columns),
                "feature_columns_sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
    return output


def fx_ablation_readiness(
    result: FXFeatureResult | None,
    cutoffs: pd.DatetimeIndex,
    *,
    minimum_common_weeks: int = 156,
) -> dict[str, Any]:
    """Report whether the four-way prospective ablation has enough PIT rows."""

    if minimum_common_weeks < 1:
        raise ValueError("minimum_common_weeks must be positive")
    base = {
        "role": "prospective_shadow",
        "variants": list(FX_VARIANT_ORDER),
        "minimum_common_weeks": int(minimum_common_weeks),
        "historical_availability_backfill": False,
        "official_release_archive_ingest": (
            bool(result.official_release_archive_ingest)
            if result is not None
            else False
        ),
        "availability_basis": (
            str(result.availability_basis)
            if result is not None
            else "collection_first_seen_at"
        ),
        "archive_revision_policy": (
            str(result.archive_revision_policy)
            if result is not None
            else "later_official_release_preserved_as_new_vintage"
        ),
        "archive_correction_availability_basis": (
            str(result.archive_correction_availability_basis)
            if result is not None
            else "date_only_conservative_next_day"
        ),
    }
    if result is None:
        return {
            **base,
            "status": "unavailable",
            "eligible_common_weeks": 0,
            "first_eligible_cutoff": None,
            "last_eligible_cutoff": None,
            "manifest": [],
        }

    variants = fx_ablation_variants(result.features)
    aligned = align_fx_features_to_cutoffs(result, cutoffs)
    required = list(variants["v4_plus_all_fx"])
    eligible = aligned.loc[
        aligned[required].notna().all(axis=1)
        & aligned["fx_observation_age_days"].eq(7)
        & ~aligned["fx_archive_correction_quarantined"].eq(True)
    ]
    count = int(len(eligible))
    return {
        **base,
        "status": (
            "ready_for_evaluation"
            if count >= minimum_common_weeks
            else "insufficient_history"
        ),
        "eligible_common_weeks": count,
        "first_eligible_cutoff": (
            eligible.index[0].date().isoformat() if count else None
        ),
        "last_eligible_cutoff": (
            eligible.index[-1].date().isoformat() if count else None
        ),
        "manifest": _manifest(variants),
    }


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fixed_multinomial_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    solver="lbfgs",
                    C=FX_ABLATION_REGULARIZATION_C,
                    class_weight=None,
                    max_iter=2_000,
                    tol=1e-6,
                    random_state=FX_ABLATION_BOOTSTRAP_SEED,
                ),
            ),
        ]
    )


def _model_contract() -> dict[str, Any]:
    return {
        "name": "fixed_l2_multinomial_logistic",
        "horizon_weeks": 1,
        "multiclass": "multinomial",
        "regularization": "l2",
        "regularization_c": FX_ABLATION_REGULARIZATION_C,
        "class_weight": None,
        "solver": "lbfgs",
        "max_iter": 2_000,
        "tolerance": 1e-6,
        "random_state": FX_ABLATION_BOOTSTRAP_SEED,
        "imputation": "expanding_train_median",
        "scaling": "expanding_train_standard",
        "fit_window": "expanding",
        "state_order": list(STATE_ORDER),
    }


def _empty_common_origins() -> dict[str, Any]:
    return {
        "count": 0,
        "first_origin": None,
        "last_origin": None,
        "sha256": None,
        "rows": [],
    }


def _gate_contract(
    *,
    bootstrap_block_weeks: int,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    alpha: float,
    minimum_log_loss_improvement: float,
    brier_tolerance: float,
) -> dict[str, Any]:
    return {
        "reference_variant": "v4_control",
        "method": "paired_circular_moving_block_bootstrap_holm",
        "bootstrap_block_weeks": int(bootstrap_block_weeks),
        "bootstrap_effective_block_weeks": None,
        "bootstrap_resamples": int(bootstrap_resamples),
        "bootstrap_seed": int(bootstrap_seed),
        "alpha": float(alpha),
        "minimum_log_loss_improvement": float(
            minimum_log_loss_improvement
        ),
        "brier_tolerance": float(brier_tolerance),
        "comparisons": [],
        "passed_variants": [],
    }


def _shadow_output(
    readiness: Mapping[str, Any],
    *,
    status: str,
    status_reason: str | None,
    minimum_train_weeks: int,
    bootstrap_block_weeks: int,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    alpha: float,
    minimum_log_loss_improvement: float,
    brier_tolerance: float,
) -> dict[str, Any]:
    return {
        **dict(readiness),
        "status": status,
        "status_reason": status_reason,
        "common_origin_required_pairs": (
            FX_ABLATION_COMMON_ORIGIN_REQUIRED_PAIRS
        ),
        "minimum_train_weeks": int(minimum_train_weeks),
        "target_horizon_weeks": 1,
        "purge_weeks": FX_ABLATION_PURGE_WEEKS,
        "target_availability_rule": (
            "last_train_target_strictly_before_evaluation_origin"
        ),
        "model": _model_contract(),
        "common_evaluation_origins": _empty_common_origins(),
        "variant_metrics": [],
        "gate": _gate_contract(
            bootstrap_block_weeks=bootstrap_block_weeks,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
            alpha=alpha,
            minimum_log_loss_improvement=minimum_log_loss_improvement,
            brier_tolerance=brier_tolerance,
        ),
        "promotion_allowed": False,
        "promotion_candidate": None,
        "core_champion_promoted": False,
    }


def _validate_shadow_inputs(
    core_features: pd.DataFrame,
    states: pd.Series,
    cutoffs: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.Series]:
    if not isinstance(core_features, pd.DataFrame):
        raise TypeError("core_features must be a pandas DataFrame")
    if not isinstance(states, pd.Series):
        raise TypeError("states must be a pandas Series")
    if not isinstance(core_features.index, pd.DatetimeIndex):
        raise TypeError("core_features must use a DatetimeIndex")
    if not isinstance(states.index, pd.DatetimeIndex):
        raise TypeError("states must use a DatetimeIndex")
    if not isinstance(cutoffs, pd.DatetimeIndex):
        raise TypeError("cutoffs must be a DatetimeIndex")
    if (
        core_features.index.empty
        or core_features.index.has_duplicates
        or not core_features.index.is_monotonic_increasing
    ):
        raise ValueError(
            "core_features index must be non-empty, unique, and increasing"
        )
    if not core_features.index.equals(states.index):
        raise ValueError("core_features and states must have exactly the same index")
    if cutoffs.empty or cutoffs.has_duplicates or not cutoffs.is_monotonic_increasing:
        raise ValueError("cutoffs must be non-empty, unique, and increasing")
    if (core_features.index.get_indexer(cutoffs) < 0).any():
        raise ValueError("every cutoff must be present in core_features and states")
    if core_features.shape[1] < 1:
        raise ValueError("core_features must contain at least one column")
    column_names = tuple(str(column) for column in core_features.columns)
    if len(set(column_names)) != len(column_names):
        raise ValueError("core feature names must be unique after string coercion")
    non_numeric = [
        str(column)
        for column in core_features.columns
        if not pd.api.types.is_numeric_dtype(core_features[column])
    ]
    if non_numeric:
        raise TypeError(f"all core feature columns must be numeric: {non_numeric}")

    numeric = core_features.astype(float).copy()
    numeric.columns = list(column_names)
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("core_features must not contain infinite values")
    invalid_states = sorted(set(states.dropna().astype(str)).difference(STATE_ORDER))
    if invalid_states:
        raise ValueError(f"states contain unsupported labels: {invalid_states}")
    if states.isna().any():
        raise ValueError("states must be complete")
    return numeric, states.astype(str).copy()


def _fixed_pair_contract_available(features: pd.DataFrame) -> bool:
    columns = _model_feature_columns(features)
    return all(
        any(column.startswith(f"fx__{code}__") for column in columns)
        for code in _FIXED_BILATERAL_CODES
    )


def _calendar_days_between(left: pd.Timestamp, right: pd.Timestamp) -> int:
    return (right.date() - left.date()).days


def _date_string(value: pd.Timestamp) -> str:
    return value.date().isoformat()


def _variant_scores(
    actual: Sequence[str],
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    positions = {state: index for index, state in enumerate(STATE_ORDER)}
    actual_positions = np.asarray([positions[str(value)] for value in actual])
    probability = np.asarray(probabilities, dtype=float)
    actual_probability = probability[
        np.arange(len(actual_positions)), actual_positions
    ]
    log_losses = -np.log(np.clip(actual_probability, 1e-9, 1.0))
    one_hot = np.zeros_like(probability)
    one_hot[np.arange(len(actual_positions)), actual_positions] = 1.0
    brier = np.sum((probability - one_hot) ** 2, axis=1)
    predicted_positions = np.argmax(probability, axis=1)
    accuracy = (predicted_positions == actual_positions).astype(float)
    class_recalls = [
        float(accuracy[actual_positions == position].mean())
        for position in range(len(STATE_ORDER))
        if np.any(actual_positions == position)
    ]
    balanced_accuracy = float(np.mean(class_recalls))
    return log_losses, brier, accuracy, balanced_accuracy


def run_fx_shadow_ablation(
    core_features: pd.DataFrame,
    states: pd.Series,
    result: FXFeatureResult | None,
    cutoffs: pd.DatetimeIndex,
    *,
    minimum_train_weeks: int = FX_ABLATION_MINIMUM_TRAIN_WEEKS,
    bootstrap_block_weeks: int = FX_ABLATION_BOOTSTRAP_BLOCK_WEEKS,
    bootstrap_resamples: int = FX_ABLATION_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = FX_ABLATION_BOOTSTRAP_SEED,
    alpha: float = FX_ABLATION_ALPHA,
    minimum_log_loss_improvement: float = (
        FX_ABLATION_MINIMUM_LOG_LOSS_IMPROVEMENT
    ),
    brier_tolerance: float = FX_ABLATION_BRIER_TOLERANCE,
    evidence_sink: Callable[[pd.DataFrame], Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the four frozen FX variants on one prospective origin set.

    Every fold predicts the next weekly state.  Training targets must be
    strictly earlier than the evaluation origin, which purges the immediately
    preceding supervised origin.  FX rows are the same first-seen, one-week-old
    PIT rows used by :func:`fx_ablation_readiness`; no historical availability
    is reconstructed.  The returned gate is diagnostic-only and can never
    replace or promote the core v4 champion.
    """

    if (
        isinstance(minimum_train_weeks, bool)
        or int(minimum_train_weeks) != minimum_train_weeks
        or int(minimum_train_weeks) < FX_ABLATION_MINIMUM_TRAIN_WEEKS
    ):
        raise ValueError("minimum_train_weeks must be an integer of at least 104")
    if (
        isinstance(bootstrap_block_weeks, bool)
        or int(bootstrap_block_weeks) != bootstrap_block_weeks
        or int(bootstrap_block_weeks) < 1
    ):
        raise ValueError("bootstrap_block_weeks must be a positive integer")
    if (
        isinstance(bootstrap_resamples, bool)
        or int(bootstrap_resamples) != bootstrap_resamples
        or int(bootstrap_resamples) < 1
    ):
        raise ValueError("bootstrap_resamples must be a positive integer")
    if (
        isinstance(bootstrap_seed, bool)
        or int(bootstrap_seed) != bootstrap_seed
        or int(bootstrap_seed) < 0
    ):
        raise ValueError("bootstrap_seed must be a non-negative integer")
    if not np.isfinite(float(alpha)) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be between zero and one")
    if (
        not np.isfinite(float(minimum_log_loss_improvement))
        or float(minimum_log_loss_improvement) < 0.0
    ):
        raise ValueError("minimum_log_loss_improvement must be non-negative")
    if not np.isfinite(float(brier_tolerance)) or float(brier_tolerance) < 0.0:
        raise ValueError("brier_tolerance must be non-negative")
    if evidence_sink is not None and not callable(evidence_sink):
        raise TypeError("evidence_sink must be callable or None")

    minimum_train_weeks = int(minimum_train_weeks)
    bootstrap_block_weeks = int(bootstrap_block_weeks)
    bootstrap_resamples = int(bootstrap_resamples)
    bootstrap_seed = int(bootstrap_seed)
    alpha = float(alpha)
    minimum_log_loss_improvement = float(minimum_log_loss_improvement)
    brier_tolerance = float(brier_tolerance)
    core, state_series = _validate_shadow_inputs(core_features, states, cutoffs)

    output_kwargs = {
        "minimum_train_weeks": minimum_train_weeks,
        "bootstrap_block_weeks": bootstrap_block_weeks,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
        "alpha": alpha,
        "minimum_log_loss_improvement": minimum_log_loss_improvement,
        "brier_tolerance": brier_tolerance,
    }
    if result is None:
        readiness = fx_ablation_readiness(
            None,
            cutoffs,
            minimum_common_weeks=FX_ABLATION_MINIMUM_COMMON_WEEKS,
        )
        return _shadow_output(
            readiness,
            status="unavailable",
            status_reason="fx_feature_result_unavailable",
            **output_kwargs,
        )
    if not isinstance(result, FXFeatureResult):
        raise TypeError("result must be an FXFeatureResult or None")

    try:
        variants = fx_ablation_variants(result.features)
        readiness = fx_ablation_readiness(
            result,
            cutoffs,
            minimum_common_weeks=FX_ABLATION_MINIMUM_COMMON_WEEKS,
        )
    except (KeyError, TypeError, ValueError):
        unavailable = fx_ablation_readiness(
            None,
            cutoffs,
            minimum_common_weeks=FX_ABLATION_MINIMUM_COMMON_WEEKS,
        )
        return _shadow_output(
            unavailable,
            status="unavailable",
            status_reason="fx_feature_contract_unavailable",
            **output_kwargs,
        )

    if not _fixed_pair_contract_available(result.features):
        unavailable = fx_ablation_readiness(
            None,
            cutoffs,
            minimum_common_weeks=FX_ABLATION_MINIMUM_COMMON_WEEKS,
        )
        return _shadow_output(
            unavailable,
            status="unavailable",
            status_reason="fixed_nine_pair_contract_unavailable",
            **output_kwargs,
        )
    if readiness["status"] != "ready_for_evaluation":
        return _shadow_output(
            readiness,
            status=str(readiness["status"]),
            status_reason="eligible_common_weeks_below_156",
            **output_kwargs,
        )

    aligned = align_fx_features_to_cutoffs(result, cutoffs)
    all_fx_columns = list(variants["v4_plus_all_fx"])
    non_numeric_fx = [
        column
        for column in all_fx_columns
        if not pd.api.types.is_numeric_dtype(aligned[column])
    ]
    if non_numeric_fx:
        return _shadow_output(
            readiness,
            status="unavailable",
            status_reason="fx_model_features_non_numeric",
            **output_kwargs,
        )
    fx_values = aligned.loc[:, all_fx_columns].astype(float)
    finite_fx = pd.Series(
        np.isfinite(fx_values.to_numpy(dtype=float)).all(axis=1),
        index=aligned.index,
    )
    eligible_mask = (
        finite_fx
        & aligned["fx_observation_age_days"].eq(7)
        & ~aligned["fx_archive_correction_quarantined"].eq(True)
    )
    common_cutoffs = pd.DatetimeIndex(aligned.index[eligible_mask])

    core_at_cutoff = core.reindex(cutoffs)
    if set(core_at_cutoff.columns).intersection(all_fx_columns):
        raise ValueError("core and FX feature names must not overlap")
    feature_frames: dict[str, pd.DataFrame] = {}
    for variant in FX_VARIANT_ORDER:
        additions = list(variants[variant])
        feature_frames[variant] = pd.concat(
            [core_at_cutoff, aligned.loc[:, additions]],
            axis=1,
        )

    core_positions = core.index.get_indexer(common_cutoffs)
    supervised_rows: list[dict[str, Any]] = []
    for cutoff, core_position in zip(
        common_cutoffs,
        core_positions,
        strict=True,
    ):
        target_position = int(core_position) + 1
        if target_position >= len(core.index):
            continue
        origin_date = pd.Timestamp(core.index[int(core_position)])
        target_date = pd.Timestamp(core.index[target_position])
        if _calendar_days_between(origin_date, target_date) != 7:
            continue
        supervised_rows.append(
            {
                "cutoff": pd.Timestamp(cutoff),
                "origin_date": origin_date,
                "target_date": target_date,
                "current_state": str(state_series.iloc[int(core_position)]),
                "actual": str(state_series.iloc[target_position]),
            }
        )

    folds: list[dict[str, Any]] = []
    for test_row in supervised_rows:
        origin = pd.Timestamp(test_row["origin_date"])
        train_rows = [
            row
            for row in supervised_rows
            if pd.Timestamp(row["target_date"]) < origin
        ]
        purged_rows = [
            row
            for row in supervised_rows
            if pd.Timestamp(row["target_date"]) == origin
        ]
        if len(train_rows) < minimum_train_weeks or len(purged_rows) != 1:
            continue
        last_train_target = pd.Timestamp(train_rows[-1]["target_date"])
        if not last_train_target < origin:
            raise RuntimeError("FX ablation target purge failed closed")
        folds.append(
            {
                "test": test_row,
                "train": train_rows,
                "purged_origin_count": len(purged_rows),
            }
        )

    if not folds:
        insufficient = dict(readiness)
        insufficient["status"] = "insufficient_history"
        return _shadow_output(
            insufficient,
            status="insufficient_history",
            status_reason="no_origin_has_104_strictly_available_training_targets",
            **output_kwargs,
        )

    origin_rows: list[dict[str, Any]] = []
    for fold in folds:
        train_rows = fold["train"]
        test_row = fold["test"]
        origin_rows.append(
            {
                "origin_date": _date_string(test_row["origin_date"]),
                "target_date": _date_string(test_row["target_date"]),
                "train_size": len(train_rows),
                "train_start_origin": _date_string(train_rows[0]["origin_date"]),
                "last_train_origin": _date_string(train_rows[-1]["origin_date"]),
                "last_train_target": _date_string(train_rows[-1]["target_date"]),
                "purged_origin_count": int(fold["purged_origin_count"]),
            }
        )
    origin_hash = _sha256_json(
        [
            [row["origin_date"], row["target_date"]]
            for row in origin_rows
        ]
    )
    common_origin_output = {
        "count": len(origin_rows),
        "first_origin": origin_rows[0]["origin_date"],
        "last_origin": origin_rows[-1]["origin_date"],
        "sha256": origin_hash,
        "rows": origin_rows,
    }

    probability_by_variant: dict[str, np.ndarray] = {}
    fallback_reasons_by_variant: dict[str, dict[str, int]] = {}
    fallback_reason_rows_by_variant: dict[str, list[str]] = {}
    actual = [str(fold["test"]["actual"]) for fold in folds]
    for variant in FX_VARIANT_ORDER:
        variant_frame = feature_frames[variant]
        probability_rows: list[np.ndarray] = []
        fallback_reasons: dict[str, int] = {}
        fallback_reason_rows: list[str] = []
        for fold in folds:
            train_rows = fold["train"]
            train_cutoffs = [row["cutoff"] for row in train_rows]
            y_train = pd.Series(
                [row["actual"] for row in train_rows],
                dtype="object",
            )
            x_train = variant_frame.loc[train_cutoffs]
            x_test = variant_frame.loc[[fold["test"]["cutoff"]]]
            fallback_reason: str | None = None
            if set(y_train.astype(str)) != set(STATE_ORDER):
                fallback_reason = "training_class_coverage"
                probability = class_prior_probabilities(y_train).reshape(1, -1)
            else:
                estimator = _fixed_multinomial_model()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", category=ConvergenceWarning)
                        warnings.simplefilter("ignore", category=FutureWarning)
                        estimator.fit(x_train, y_train)
                        raw = estimator.predict_proba(x_test)
                    classes = estimator.named_steps["classifier"].classes_
                    probability = align_probabilities(
                        raw,
                        classes,
                        expected_rows=1,
                    )
                except Exception:
                    fallback_reason = "model_fit_or_prediction_error"
                    probability = class_prior_probabilities(y_train).reshape(1, -1)
            if fallback_reason is not None:
                fallback_reasons[fallback_reason] = (
                    fallback_reasons.get(fallback_reason, 0) + 1
                )
            fallback_reason_rows.append(fallback_reason or "")
            probability_rows.append(np.asarray(probability[0], dtype=float))
        probability_by_variant[variant] = np.vstack(probability_rows)
        fallback_reasons_by_variant[variant] = fallback_reasons
        fallback_reason_rows_by_variant[variant] = fallback_reason_rows

    evidence_rows: list[dict[str, Any]] = []
    for variant in FX_VARIANT_ORDER:
        for position, fold in enumerate(folds):
            test_row = fold["test"]
            probability = probability_by_variant[variant][position]
            fallback_reason = fallback_reason_rows_by_variant[variant][position]
            evidence_rows.append(
                {
                    "origin_date": _date_string(test_row["origin_date"]),
                    "target_date": _date_string(test_row["target_date"]),
                    "variant": variant,
                    "evaluation_split": "prospective_shadow",
                    "current_state": str(test_row["current_state"]),
                    "actual": str(test_row["actual"]),
                    "p_risk_on": float(probability[0]),
                    "p_transition": float(probability[1]),
                    "p_risk_off": float(probability[2]),
                    "train_size": len(fold["train"]),
                    "gap": FX_ABLATION_PURGE_WEEKS,
                    "last_train_target": _date_string(
                        fold["train"][-1]["target_date"]
                    ),
                    "purged_origin_count": int(fold["purged_origin_count"]),
                    "fallback": bool(fallback_reason),
                    "fallback_reason": fallback_reason,
                    "common_origins_sha256": origin_hash,
                }
            )
    evidence = pd.DataFrame(evidence_rows, columns=FX_ABLATION_OOS_COLUMNS)
    evidence = evidence.sort_values(
        ["origin_date", "target_date", "variant"],
        kind="mergesort",
        ignore_index=True,
    )
    log_loss_by_variant: dict[str, np.ndarray] = {}
    brier_by_variant: dict[str, np.ndarray] = {}
    metric_rows: list[dict[str, Any]] = []
    for variant in FX_VARIANT_ORDER:
        probability = probability_by_variant[variant]
        log_losses, brier, accuracy, balanced_accuracy = _variant_scores(
            actual, probability
        )
        log_loss_by_variant[variant] = log_losses
        brier_by_variant[variant] = brier
        fallback_reasons = fallback_reasons_by_variant[variant]
        fallback_count = int(sum(fallback_reasons.values()))
        all_columns = tuple(str(column) for column in feature_frames[variant].columns)
        metric_rows.append(
            {
                "variant": variant,
                "feature_count": len(all_columns),
                "fx_feature_count": len(variants[variant]),
                "feature_columns_sha256": _sha256_json(list(all_columns)),
                "log_loss": float(log_losses.mean()),
                "brier": float(brier.mean()),
                "accuracy": float(accuracy.mean()),
                "balanced_accuracy": balanced_accuracy,
                "n": len(actual),
                "n_predictions": len(actual),
                "fallback": fallback_count > 0,
                "fallback_count": fallback_count,
                "fallback_reasons": dict(sorted(fallback_reasons.items())),
                "first_origin": origin_rows[0]["origin_date"],
                "last_origin": origin_rows[-1]["origin_date"],
                "origin_sha256": origin_hash,
            }
        )

    challenger_variants = FX_VARIANT_ORDER[1:]
    paired_improvements = {
        variant: (
            log_loss_by_variant["v4_control"] - log_loss_by_variant[variant]
        )
        for variant in challenger_variants
    }
    raw_pvalues, effective_block = _moving_block_bootstrap_pvalues(
        paired_improvements,
        block_length=bootstrap_block_weeks,
        resamples=bootstrap_resamples,
        random_state=bootstrap_seed,
    )
    adjusted_pvalues = _holm_adjusted_pvalues(raw_pvalues)
    metric_index = {row["variant"]: row for row in metric_rows}
    comparisons: list[dict[str, Any]] = []
    passed_variants: list[str] = []
    control_fallback_count = int(metric_index["v4_control"]["fallback_count"])
    for variant in challenger_variants:
        improvement = float(paired_improvements[variant].mean())
        brier_difference = float(
            brier_by_variant[variant].mean()
            - brier_by_variant["v4_control"].mean()
        )
        fallback_count = int(metric_index[variant]["fallback_count"])
        failures: list[str] = []
        if control_fallback_count:
            failures.append("control_fallback_present")
        if fallback_count:
            failures.append("fallback_present")
        if improvement + 1e-12 < minimum_log_loss_improvement:
            failures.append("insufficient_log_loss_improvement")
        if adjusted_pvalues[variant] > alpha:
            failures.append("holm_not_significant")
        if brier_difference > brier_tolerance + 1e-12:
            failures.append("brier_degradation")
        passed = not failures
        if passed:
            passed_variants.append(variant)
        comparisons.append(
            {
                "variant": variant,
                "reference_variant": "v4_control",
                "mean_log_loss_improvement": improvement,
                "brier_difference": brier_difference,
                "control_fallback_count": control_fallback_count,
                "fallback_count": fallback_count,
                "raw_p_value": float(raw_pvalues[variant]),
                "holm_adjusted_p_value": float(adjusted_pvalues[variant]),
                "gate_passed": passed,
                "gate_reasons": ["passed"] if passed else failures,
            }
        )

    evaluated = _shadow_output(
        readiness,
        status="evaluated",
        status_reason=None,
        **output_kwargs,
    )
    evaluated["common_evaluation_origins"] = common_origin_output
    evaluated["variant_metrics"] = metric_rows
    evaluated["gate"] = {
        **evaluated["gate"],
        "bootstrap_effective_block_weeks": int(effective_block),
        "comparisons": comparisons,
        "passed_variants": passed_variants,
    }
    if evidence_sink is not None:
        evidence_sink(evidence.copy())
    return evaluated


__all__ = [
    "FX_ABLATION_OOS_COLUMNS",
    "FX_VARIANT_ORDER",
    "align_fx_features_to_cutoffs",
    "fx_ablation_readiness",
    "fx_ablation_variants",
    "run_fx_shadow_ablation",
]
