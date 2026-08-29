#!/usr/bin/env python3
"""Read-only audit of a completed regime dashboard build.

The audit intentionally reads only the published payload and supporting CSVs.
It does not open the snapshot database, refit a model, or modify artifacts.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression

from regime_lab.artifact_inventory import (
    ArtifactInventoryError,
    verify_artifact_inventory,
)
from regime_lab.contract_v5 import (
    V5_FORECAST_COMPARISON_MODELS as CONTRACT_V5_FORECAST_COMPARISON_MODELS,
    V5_SCHEMA_VERSION as CONTRACT_V5_SCHEMA_VERSION,
    V5_STANDARD_CORE_MODELS,
)
from regime_lab.feature_quality import verify_feature_quality_artifact
from regime_lab.frozen_v4 import (
    FROZEN_V4_BASELINE,
    FrozenV4BaselineError,
    verify_frozen_v4_baseline,
)
from regime_lab.integrity import (
    GENERATION_MANIFEST_SCHEMA_VERSION,
    IntegrityError,
    validate_generation_manifest,
    validate_lifecycle_consistency,
    validate_reviewed_candidate_hash,
)
from regime_lab.operating_contract import load_operating_contract
from regime_lab.publication_contract import (
    PublicContractError,
    validate_v5_comparison_sidecar,
)
from regime_lab.schema import ContractError, validate_dashboard_payload
from regime_lab.selection_family_audit import (
    build_selection_family_audit_from_artifacts,
    validate_selection_family_audit,
)


STATE_ORDER = ("risk_on", "transition", "risk_off")
PROBABILITY_COLUMNS = tuple(f"p_{state}" for state in STATE_ORDER)
V5_SERIALIZED_PROBABILITY_DECIMALS = 8
V5_SERIALIZED_SIMPLEX_ATOL = 1.1 * 10 ** (-V5_SERIALIZED_PROBABILITY_DECIMALS)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINES = ("majority", "persistence", "markov")
COMPLEXITY = {
    "majority": 0,
    "persistence": 1,
    "markov": 2,
    "ridge_logistic": 3,
    "shrinkage_lda": 4,
    "elastic_net_logistic": 5,
    "transition_logistic": 6,
    "duration_tvtp_hurdle": 7,
    "calibrated_linear_svm": 7,
    "spline_logistic": 8,
    "gaussian_hmm": 9,
    "hist_gradient_boosting": 10,
    "random_forest": 11,
    "extra_trees": 12,
    "xgboost": 13,
    "xgb_hazard_destination": 14,
    "causal_dynamic_ensemble": 15,
    "causal_multiscale_ensemble": 16,
}
V2_RESULT_VERSION = "weekly-regime-result-v2"
V3_RESULT_VERSION = "weekly-regime-result-v3"
V3_MODEL_VERSION = "weekly-nondl-structural-v3"
V3_FEATURE_SET_VERSION = "weekly-pit-market-internals-v3"
V4_RESULT_VERSION = "weekly-regime-result-v4"
V4_MODEL_VERSION = "weekly-nondl-structural-v4"
V4_FEATURE_SET_VERSION = "weekly-pit-structural-v4"
V5_RESULT_VERSION = "weekly-regime-result-v5"
V5_SCHEMA_VERSION = CONTRACT_V5_SCHEMA_VERSION
V5_RESEARCH_ARTIFACTS = (
    ("directional_oos_predictions", "directional-oos-predictions.csv"),
    ("directional_model_leaderboard", "directional-model-leaderboard.csv"),
    ("directional_walk_forward_splits", "directional-walk-forward-splits.csv"),
    ("directional_selection_diagnostics", "directional-selection-diagnostics.csv"),
    ("directional_forecasts", "directional-forecasts.csv"),
    ("conditional_asset_outcomes", "conditional-asset-outcomes.csv"),
    ("conditional_asset_statistics", "conditional-asset-statistics.csv"),
)
V5_MODEL_CONDITIONED_ARTIFACTS = (
    (
        "model_conditioned_asset_outcomes",
        "model-conditioned-asset-outcomes.csv",
    ),
    (
        "model_conditioned_asset_statistics",
        "model-conditioned-asset-statistics.csv",
    ),
)
V5_FX_ARTIFACTS = (
    ("fx_features", "fx-features.csv"),
    ("fx_coverage", "fx-coverage.csv"),
    ("fx_ablation_oos", "fx-ablation-oos.csv"),
)
V5_FX_VARIANTS = (
    "v4_control",
    "v4_plus_broad_index",
    "v4_plus_bilateral_panel",
    "v4_plus_all_fx",
)
V5_FX_ABLATION_OOS_COLUMNS = (
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
V5_DIRECTIONAL_OUTCOMES = ("no_departure", *STATE_ORDER)
V5_DIRECTIONAL_BASELINES = (
    "empirical_first_passage",
    "markov_first_passage",
)
V5_DIRECTIONAL_SCORE_TARGET = "first_destination_given_departure"
V5_DIRECTIONAL_MINIMUM_EVENTS = 8
V5_DIRECTIONAL_MINIMUM_DESTINATION_CLASSES = 2
V5_DIRECTIONAL_MINIMUM_EVENT_BLOCKS = 3
V5_DIRECTIONAL_BOOTSTRAP_RESAMPLES = 999
V5_OUTCOME_ASSETS = ("SPY", "QQQ", "IWM", "TLT", "HYG", "UUP")
V5_OUTCOME_HORIZONS = (1, 4, 13)
V5_OUTCOME_POINT_METRICS = (
    "mean_return",
    "median_return",
    "positive_rate",
    "annualized_volatility",
    "downside_volatility",
    "cvar_5",
    "mean_max_drawdown",
)
V5_CONDITIONAL_STATISTIC_FIELDS = (
    "execution_lag_weeks",
    "return_currency",
    "sample_start",
    "sample_end",
    "n",
    "non_overlapping_n",
    "unique_episodes",
    "status",
    "minimum_observations",
    "minimum_unique_episodes",
    "minimum_non_overlapping_observations",
    "bootstrap_method",
    "bootstrap_block_weeks",
    "bootstrap_resamples",
    "bootstrap_seed",
    "unconditional_benchmark_method",
    "unconditional_benchmark_n",
    "unconditional_benchmark_mean_return",
    "excess_mean_return",
    "episode_equal_mean_return",
    "episode_equal_unconditional_benchmark_method",
    "episode_equal_unconditional_benchmark_episode_n",
    "episode_equal_unconditional_benchmark_mean_return",
    "episode_equal_excess_return",
    "episode_bootstrap_method",
    "episode_bootstrap_resamples",
    "episode_bootstrap_seed",
    "episode_equal_mean_return_ci95_lower",
    "episode_equal_mean_return_ci95_upper",
    *V5_OUTCOME_POINT_METRICS,
    *tuple(
        field
        for metric in V5_OUTCOME_POINT_METRICS
        for field in (
            f"{metric}_ci95_lower",
            f"{metric}_ci95_upper",
        )
    ),
)
V5_INVESTMENT_CONDITIONAL_FIELDS = frozenset(
    {
        "non_overlapping_n",
        "minimum_non_overlapping_observations",
        "unconditional_benchmark_method",
        "unconditional_benchmark_n",
        "unconditional_benchmark_mean_return",
        "excess_mean_return",
        "episode_equal_mean_return",
        "episode_equal_unconditional_benchmark_method",
        "episode_equal_unconditional_benchmark_episode_n",
        "episode_equal_unconditional_benchmark_mean_return",
        "episode_equal_excess_return",
        "episode_bootstrap_method",
        "episode_bootstrap_resamples",
        "episode_bootstrap_seed",
        "episode_equal_mean_return_ci95_lower",
        "episode_equal_mean_return_ci95_upper",
    }
)
V4_PREREGISTRATION_SHA256 = (
    "2f53ada564efca770261f16ce6eb16ec9c9782bde014de7a7d85b7b24dbe407b"
)
V4_STRUCTURAL_MODELS = {
    "xgb_hazard_destination",
    "causal_dynamic_ensemble",
}
V4_BASE_MODELS = {
    "majority",
    "persistence",
    "markov",
    "elastic_net_logistic",
    "calibrated_linear_svm",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "ridge_logistic",
    "transition_logistic",
    "duration_tvtp_hurdle",
    "shrinkage_lda",
    "spline_logistic",
    "xgboost",
}
V4_STANDARD_MODELS = V4_BASE_MODELS | V4_STRUCTURAL_MODELS
V4_FULL_MODELS = V4_STANDARD_MODELS | {"gaussian_hmm"}
V5_MULTISCALE_MODEL = "causal_multiscale_ensemble"
V5_FORECAST_COMPARISON_MODELS = CONTRACT_V5_FORECAST_COMPARISON_MODELS
V5_FORECAST_COMPARISON_METHOD = "model_comparison_walk_forward_probability"
V5_STANDARD_MODELS = set(V5_STANDARD_CORE_MODELS)
V5_FULL_MODELS = V5_STANDARD_MODELS | {"gaussian_hmm"}
OPERATING_CONTRACT = load_operating_contract()
# Legacy ranks remain available for frozen V2-V4 reproduction.  Active V5
# names and ranks come from the same typed source of truth as model selection.
COMPLEXITY.update(
    {
        str(name): int(rank)
        for name, rank in OPERATING_CONTRACT.selection_policy[
            "complexity_registry"
        ].items()
    }
)
V4_STRUCTURAL_EXPERTS = (
    "markov",
    "xgboost",
    "xgb_hazard_destination",
)
V4_DIRECT_JUMP_FLOOR = 1e-6
V4_ENSEMBLE_HALF_LIFE_WEEKS = 52.0
V4_ENSEMBLE_MINIMUM_HISTORY_ROWS = 26
V4_ABLATION_VARIANTS = {
    "legacy_v3": ("legacy_v3",),
    "legacy_plus_market_structure": (
        "legacy_v3",
        "sector_breadth",
        "broad_size_style_breadth",
        "cross_asset_breadth",
    ),
    "legacy_plus_treasury_curve": ("legacy_v3", "treasury_curve"),
    "legacy_plus_bank_credit": ("legacy_v3", "bank_credit"),
    "legacy_plus_financial_conditions": (
        "legacy_v3",
        "financial_conditions",
    ),
    "legacy_plus_release_innovation": ("legacy_v3", "release_innovation"),
    "all_structural": (
        "legacy_v3",
        "sector_breadth",
        "broad_size_style_breadth",
        "cross_asset_breadth",
        "treasury_curve",
        "bank_credit",
        "financial_conditions",
        "release_innovation",
    ),
}
V4_FEATURE_GROUP_PREFIXES = {
    "sector_breadth": ("market_group__gics_sector__",),
    "broad_size_style_breadth": ("market_group__broad_size_style__",),
    "cross_asset_breadth": ("market_group__cross_asset__",),
    "treasury_curve": ("treasury_curve__",),
    "bank_credit": ("bank_credit__",),
    "financial_conditions": ("anfci__",),
    "release_innovation": ("release_innovation__",),
}
V4_FINANCIAL_CONDITION_FEATURES = {
    "anfci__level",
    "anfci__change_1w",
    "anfci__change_4w",
    "anfci__z_52w",
}
V4_ANFCI_LEGACY_AVAILABILITY_FEATURES = {
    "anfci__age_days",
    "anfci__release_lag_days",
    "anfci__is_filled",
}
V4_STATE_EVIDENCE_COLUMNS = (
    "date",
    "state",
    "p_risk_on",
    "p_transition",
    "p_risk_off",
    "risk_score",
    "lower_threshold",
    "upper_threshold",
    "hysteresis_margin",
    "previous_state",
    "probability_temperature",
)
V4_WEEKLY_FORECAST_EVIDENCE_COLUMNS = (
    "origin_date",
    "current_state",
    "current_p_risk_on",
    "current_p_transition",
    "current_p_risk_off",
    "target_date",
    "model",
    "next_p_risk_on",
    "next_p_transition",
    "next_p_risk_off",
    "fallback",
    "fallback_reason",
)
TRANSITION_HORIZONS = (1, 4, 13)
TRANSITION_REQUIRED_MODELS = {
    "empirical_hazard",
    "markov_hazard",
    "duration_tvtp_hurdle",
    "regularized_logistic",
}
TRANSITION_ALLOWED_MODELS = TRANSITION_REQUIRED_MODELS | {
    "binary_xgboost",
    "joint_survival_hazard",
}
V2_BASELINE = {
    "result_version": V2_RESULT_VERSION,
    "label_version": "market-causal-3state-v1",
    "model_version": "weekly-nondl-walkforward-v2",
    "champion": "markov",
    "payload_sha256": (
        "50ab693b15f5100b1e39d98356c88455b76a4a2c4a4c335e5882509568c5fe98"
    ),
    "artifacts_inventory_sha256": (
        "09603aca14244fc00ee56f0d75a45192fc29a77c8f1a47b9927aef32d4fcbf0f"
    ),
    "captured_at": "2026-08-12",
}
V3_BASELINE = {
    "result_version": V3_RESULT_VERSION,
    "label_version": "market-causal-3state-v1",
    "model_version": V3_MODEL_VERSION,
    "champion": "markov",
    "payload_sha256": (
        "de93c585117b2784750f586a4f84ad99964c63081b252ad7affd7a75bd797095"
    ),
    "artifacts_inventory_sha256": (
        "8ef3778cc8c36faff0c80e2bf094f1f11bd6966ab3b7b2d6edb84ba292aff6b9"
    ),
    "captured_at": "2026-08-13",
}
V3_BASELINE_RELATIVE_DIRECTORY = Path("artifacts/baselines/v3-20260813")
V4_PREREGISTRATION_RELATIVE_PATH = Path("config/structural_v4.json")
BOOTSTRAP_BLOCK_WEEKS = 13
BOOTSTRAP_RESAMPLES = 1_999
BOOTSTRAP_SEED = 17
SELECTION_ALPHA = 0.05
ALLOWED_LOG_LOSS_IMPROVEMENT_THRESHOLDS = (0.01, 0.05)
DIRECTIONAL_MINIMUM_LOG_LOSS_IMPROVEMENT = 0.05
BRIER_TOLERANCE = 0.01


class AuditFailure(AssertionError):
    """Raised when an output contract is violated."""


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AuditFailure(message)


def anchored_transition_projection(
    raw_probabilities: Mapping[str, object],
) -> dict[str, float]:
    """Independently reproduce the published 1w-anchored L2 projection."""

    require(
        set(raw_probabilities) == {"1w", "4w", "13w"},
        "transition term-structure raw probability keys mismatch",
    )
    values = {
        key: float(raw_probabilities[key]) for key in ("1w", "4w", "13w")
    }
    require(
        all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in values.values()),
        "transition term-structure raw probabilities are invalid",
    )
    p1 = values["1w"]
    p4 = max(values["4w"], p1)
    p13 = max(values["13w"], p1)
    if p4 > p13:
        p4 = p13 = (p4 + p13) / 2.0
    return {"1w": p1, "4w": p4, "13w": p13}


def require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    require(not missing, f"{name} missing columns: {missing}")


def boolean_series(series: pd.Series, name: str) -> pd.Series:
    lowered = series.astype(str).str.strip().str.lower()
    require(lowered.isin({"true", "false"}).all(), f"{name} is not boolean")
    return lowered.eq("true")


def read_csv(path: Path, date_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    require(path.is_file() and path.stat().st_size > 0, f"missing/empty file: {path}")
    frame = pd.read_csv(path)
    for column in date_columns:
        require(column in frame, f"{path.name} missing date column: {column}")
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    require(not frame.empty, f"empty CSV: {path}")
    return frame


def require_exact_columns(
    frame: pd.DataFrame,
    expected: Sequence[str],
    name: str,
) -> None:
    require(
        tuple(str(column) for column in frame.columns) == tuple(expected),
        f"{name} columns/order mismatch",
    )


def canonical_json_sha256(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def ordered_feature_sha256(columns: Sequence[str]) -> str:
    """Match the preregistered ablation runner's ordered-column digest."""

    serialized = json.dumps(
        [str(column) for column in columns],
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash one materialized artifact without following an implicit contract."""

    require(path.is_file() and not path.is_symlink(), f"missing/non-regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_frozen_v3_baseline(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Verify the immutable v3 snapshot itself, not only copied metadata."""

    baseline = project_root / V3_BASELINE_RELATIVE_DIRECTORY
    inventory = baseline / "SHA256SUMS"
    payload = baseline / "regime-results.json"
    require(baseline.is_dir(), f"frozen v3 baseline directory is missing: {baseline}")
    require(
        file_sha256(payload) == V3_BASELINE["payload_sha256"],
        "frozen v3 payload SHA-256 mismatch",
    )
    inventory_hash = file_sha256(inventory)
    require(
        inventory_hash == V3_BASELINE["artifacts_inventory_sha256"],
        "frozen v3 inventory SHA-256 mismatch",
    )

    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        inventory.read_text(encoding="utf-8").splitlines(), start=1
    ):
        require(raw_line and "  " in raw_line,
                f"frozen v3 inventory line {line_number} is malformed")
        expected_hash, relative_name = raw_line.split("  ", 1)
        require(
            len(expected_hash) == 64
            and all(character in "0123456789abcdef" for character in expected_hash),
            f"frozen v3 inventory line {line_number} has an invalid SHA-256",
        )
        relative_path = Path(relative_name)
        require(
            relative_name
            and not relative_path.is_absolute()
            and relative_path.name == relative_name
            and relative_name not in entries,
            f"frozen v3 inventory line {line_number} has an unsafe/duplicate path",
        )
        entries[relative_name] = expected_hash
    require(entries, "frozen v3 inventory is empty")
    actual_names = {
        path.name
        for path in baseline.iterdir()
        if path.is_file() and path.name != inventory.name
    }
    require(
        actual_names == set(entries),
        "frozen v3 inventory does not exactly cover the snapshot files",
    )
    for relative_name, expected_hash in entries.items():
        require(
            file_sha256(baseline / relative_name) == expected_hash,
            f"frozen v3 inventory member SHA-256 mismatch: {relative_name}",
        )
    return {
        "payload_sha256": V3_BASELINE["payload_sha256"],
        "inventory_sha256": inventory_hash,
        "files": len(entries),
    }


def audit_structural_preregistration(
    published: Mapping[str, Any],
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Hash the preregistration file named by the frozen v4 contract."""

    require(
        set(published) == {"path", "sha256"}
        and published.get("path") == V4_PREREGISTRATION_RELATIVE_PATH.as_posix(),
        "v4 structural preregistration path/fields mismatch",
    )
    path = project_root / V4_PREREGISTRATION_RELATIVE_PATH
    actual_hash = file_sha256(path)
    require(
        actual_hash == V4_PREREGISTRATION_SHA256,
        "v4 structural preregistration materialized SHA-256 mismatch",
    )
    require(
        published.get("sha256") == actual_hash,
        "v4 structural preregistration published SHA-256 mismatch",
    )
    return {"path": str(path), "sha256": actual_hash}


def _evidence_path(
    artifacts: Path,
    metadata: Mapping[str, Any],
    *,
    expected_name: str,
    context: str,
) -> Path:
    require(
        metadata.get("path") == expected_name,
        f"{context} path mismatch",
    )
    path = artifacts / expected_name
    require(
        path.parent.resolve() == artifacts.resolve(),
        f"{context} path escapes artifact directory",
    )
    require(
        file_sha256(path) == metadata.get("sha256"),
        f"{context} raw CSV SHA-256 mismatch",
    )
    return path


def audit_v4_state_evidence(
    payload: dict[str, Any],
    artifacts: Path,
    *,
    transition_predictions: pd.DataFrame,
    main_predictions: pd.DataFrame | None = None,
    prospective_transition_frames: Sequence[pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Rebuild causal 3-state labels/probabilities and all transition targets."""

    contract = payload["model"].get("evidence_artifacts")
    require(isinstance(contract, dict), "payload v4 evidence_artifacts is missing")
    require(
        set(contract) == {"state_label_history", "weekly_state_forecasts"},
        "payload v4 evidence_artifacts keys mismatch",
    )
    metadata = contract["state_label_history"]
    require(isinstance(metadata, dict), "state-label evidence metadata is invalid")
    require(
        set(metadata) == {
            "path", "row_count", "sha256", "label_fit_weeks",
            "label_fit_end", "initial_state",
        },
        "state-label evidence metadata fields mismatch",
    )
    require(int(metadata.get("label_fit_weeks", -1)) == 520,
            "state-label evidence label_fit_weeks mismatch")
    require(metadata.get("initial_state") == "transition",
            "state-label evidence initial state mismatch")
    state_path = _evidence_path(
        artifacts,
        metadata,
        expected_name="state-label-history.csv",
        context="state-label evidence",
    )
    states = pd.read_csv(state_path, keep_default_na=True)
    require(not states.empty, "state-label history is empty")
    require_exact_columns(
        states, V4_STATE_EVIDENCE_COLUMNS, "state-label-history.csv"
    )
    require(len(states) == int(metadata.get("row_count", -1)),
            "state-label evidence row_count mismatch")
    states["date"] = pd.to_datetime(states["date"], utc=True, errors="raise")
    require(states["date"].is_monotonic_increasing
            and not states["date"].duplicated().any(),
            "state-label dates must be unique and increasing")
    local_dates = states["date"].dt.tz_convert("America/New_York").dt.date
    require(
        all(
            (current - previous).days == 7
            for previous, current in zip(
                local_dates.iloc[:-1], local_dates.iloc[1:], strict=True
            )
        ),
        "state-label history is not a contiguous weekly timeline",
    )
    require(set(states["state"].astype(str)).issubset(STATE_ORDER),
            "state-label history contains an invalid state")
    require(
        pd.to_datetime(metadata.get("label_fit_end"), utc=True, errors="raise")
        == states["date"].iloc[519],
        "state-label evidence label_fit_end mismatch",
    )
    numeric_fields = [
        *PROBABILITY_COLUMNS, "risk_score", "lower_threshold", "upper_threshold",
        "hysteresis_margin", "probability_temperature",
    ]
    numeric = states[numeric_fields].apply(pd.to_numeric, errors="raise")
    finite_fields = [
        *PROBABILITY_COLUMNS, "lower_threshold", "upper_threshold",
        "hysteresis_margin", "probability_temperature",
    ]
    require(np.isfinite(numeric[finite_fields].to_numpy(dtype=float)).all(),
            "state-label evidence contains non-finite fitted/probability primitives")
    lower = numeric["lower_threshold"].to_numpy(dtype=float)
    upper = numeric["upper_threshold"].to_numpy(dtype=float)
    margin = numeric["hysteresis_margin"].to_numpy(dtype=float)
    temperature = numeric["probability_temperature"].to_numpy(dtype=float)
    require((lower < upper).all() and (margin >= 0.0).all()
            and (temperature > 0.0).all(),
            "state-label thresholds/temperature are invalid")
    require(np.allclose(lower, lower[0], atol=0.0)
            and np.allclose(upper, upper[0], atol=0.0)
            and np.allclose(margin, margin[0], atol=0.0)
            and np.allclose(temperature, temperature[0], atol=0.0),
            "state-label fitted primitives change over time")
    require(np.isclose(margin[0], (upper[0] - lower[0]) * 0.15, atol=1e-12),
            "state-label hysteresis margin mismatch")

    risk_score = numeric["risk_score"].to_numpy(dtype=float)
    expected_states: list[str] = []
    state = "transition"
    for value in risk_score:
        if not np.isfinite(value):
            expected_states.append(state)
            continue
        if state == "transition":
            if value <= lower[0]:
                state = "risk_off"
            elif value >= upper[0]:
                state = "risk_on"
        elif state == "risk_on":
            if value <= lower[0] - margin[0]:
                state = "risk_off"
            elif value < upper[0] - margin[0]:
                state = "transition"
        else:
            if value >= upper[0] + margin[0]:
                state = "risk_on"
            elif value > lower[0] + margin[0]:
                state = "transition"
        expected_states.append(state)
    require(
        np.array_equal(states["state"].astype(str).to_numpy(), expected_states),
        "state-label sequential hysteresis recomputation mismatch",
    )
    previous = states["previous_state"]
    require(pd.isna(previous.iloc[0]) or str(previous.iloc[0]).strip() == "",
            "state-label first previous_state must be empty")
    require(
        np.array_equal(
            previous.iloc[1:].astype(str).to_numpy(),
            states["state"].iloc[:-1].astype(str).to_numpy(),
        ),
        "state-label previous_state sequence mismatch",
    )
    width = upper[0] - lower[0]
    anchors = np.asarray(
        [upper[0] + width / 2.0, (lower[0] + upper[0]) / 2.0,
         lower[0] - width / 2.0],
        dtype=float,
    )
    distance = (risk_score[:, None] - anchors[None, :]) / width
    logits = -(distance ** 2) / temperature[0]
    logits[~np.isfinite(risk_score)] = np.asarray([-20.0, 0.0, -20.0])
    logits -= logits.max(axis=1, keepdims=True)
    expected_probability = np.exp(logits)
    expected_probability /= expected_probability.sum(axis=1, keepdims=True)
    actual_probability = numeric[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    require(np.allclose(actual_probability, expected_probability, atol=1e-12),
            "state-label probability recomputation mismatch")

    indexed = states.set_index("date", drop=False)
    if main_predictions is not None:
        main_contract = main_predictions[
            ["origin_date", "target_date", "current_state", "actual"]
        ].drop_duplicates()
        require(
            not main_contract.duplicated(["origin_date", "target_date"]).any(),
            "main models disagree on state-label evidence",
        )
        for _, row in main_contract.iterrows():
            origin = pd.Timestamp(row["origin_date"])
            target = pd.Timestamp(row["target_date"])
            context = f"main state target {origin.isoformat()}"
            require(origin in indexed.index and target in indexed.index,
                    f"{context} is outside state-label history")
            require(str(row["current_state"]) == str(indexed.loc[origin, "state"]),
                    f"{context} current state mismatch")
            require(str(row["actual"]) == str(indexed.loc[target, "state"]),
                    f"{context} t+1 actual state mismatch")
    for _, row in transition_predictions.iterrows():
        origin = pd.Timestamp(row["origin_date"])
        target_end = pd.Timestamp(row["target_end"])
        horizon = int(row["horizon"])
        context = f"transition target {horizon}w/{origin.isoformat()}"
        require(origin in indexed.index and target_end in indexed.index,
                f"{context} is outside state-label history")
        current_state = str(indexed.loc[origin, "state"])
        require(str(row["current_state"]) == current_state,
                f"{context} current state mismatch")
        origin_position = int(states.index[states["date"].eq(origin)][0])
        target_position = int(states.index[states["date"].eq(target_end)][0])
        require(target_position - origin_position == horizon,
                f"{context} state timeline is not weekly contiguous")
        future_states = states.iloc[
            origin_position + 1 : target_position + 1
        ]["state"].astype(str)
        expected_change = bool(future_states.ne(current_state).any())
        require(bool(row["actual_change"]) == expected_change,
                f"{context} any-departure target mismatch")

    if prospective_transition_frames is not None:
        require(bool(prospective_transition_frames),
                "prospective transition state evidence is empty")
        prospective_frames = []
        for position, frame in enumerate(prospective_transition_frames):
            require_columns(
                frame,
                {"origin_date", "current_state"},
                f"prospective transition evidence[{position}]",
            )
            prospective_frames.append(frame[["origin_date", "current_state"]])
        prospective = pd.concat(
            prospective_frames, ignore_index=True
        ).drop_duplicates()
        require(not prospective.duplicated("origin_date").any(),
                "prospective transition rows disagree on current state")
        for _, row in prospective.iterrows():
            origin = pd.Timestamp(row["origin_date"])
            require(origin in indexed.index,
                    "prospective transition origin is outside state-label history")
            require(str(row["current_state"]) == str(indexed.loc[origin, "state"]),
                    f"prospective transition current state mismatch at "
                    f"{origin.isoformat()}")
    return {
        "rows": len(states),
        "sha256": metadata["sha256"],
        "first_date": states["date"].iloc[0].isoformat(),
        "last_date": states["date"].iloc[-1].isoformat(),
    }


def audit_v4_weekly_forecast_evidence(
    payload: dict[str, Any],
    artifacts: Path,
) -> dict[str, Any]:
    """Require exact source parity for every dashboard current/t+1 estimate."""

    contract = payload["model"]["evidence_artifacts"]
    metadata = contract["weekly_state_forecasts"]
    require(isinstance(metadata, dict), "weekly forecast evidence metadata is invalid")
    require(set(metadata) == {"path", "row_count", "sha256"},
            "weekly forecast evidence metadata fields mismatch")
    path = _evidence_path(
        artifacts,
        metadata,
        expected_name="weekly-state-forecasts.csv",
        context="weekly forecast evidence",
    )
    evidence = pd.read_csv(path, keep_default_na=True)
    require(not evidence.empty, "weekly forecast evidence is empty")
    require_exact_columns(
        evidence,
        V4_WEEKLY_FORECAST_EVIDENCE_COLUMNS,
        "weekly-state-forecasts.csv",
    )
    require(len(evidence) == int(metadata.get("row_count", -1)),
            "weekly forecast evidence row_count mismatch")
    evidence["origin_date"] = pd.to_datetime(
        evidence["origin_date"], utc=True, errors="raise"
    )
    target_dates = pd.to_datetime(
        evidence["target_date"], errors="raise"
    ).dt.date
    evidence["fallback"] = boolean_series(
        evidence["fallback"], "weekly forecast evidence fallback"
    )
    require(not evidence["origin_date"].duplicated().any(),
            "weekly forecast evidence duplicates an origin")
    origin_dates = evidence["origin_date"].dt.tz_convert(
        "America/New_York"
    ).dt.date
    require(
        all(
            (target_date - origin_date).days == 7
            for origin_date, target_date in zip(
                origin_dates, target_dates, strict=True
            )
        ),
        "weekly forecast evidence target is not exactly one week later",
    )
    weekly = payload["weekly"]
    require(len(evidence) == len(weekly),
            "weekly forecast evidence/payload row count mismatch")
    by_origin = evidence.set_index(
        evidence["origin_date"].dt.date.astype(str), drop=False
    )
    require(set(by_origin.index) == {str(row["date"]) for row in weekly},
            "weekly forecast evidence/payload origin sets differ")
    state_metadata = contract["state_label_history"]
    state_path = artifacts / str(state_metadata["path"])
    state_history = pd.read_csv(state_path, keep_default_na=True)
    require_exact_columns(
        state_history, V4_STATE_EVIDENCE_COLUMNS, "state-label-history.csv"
    )
    state_history["date"] = pd.to_datetime(
        state_history["date"], utc=True, errors="raise"
    )
    state_by_origin = state_history.set_index(
        state_history["date"].dt.date.astype(str), drop=False
    )
    require(set(by_origin.index).issubset(set(state_by_origin.index)),
            "weekly forecast origins are missing from state-label history")
    for index, week in enumerate(weekly):
        origin = str(week["date"])
        row = by_origin.loc[origin]
        state_row = state_by_origin.loc[origin]
        require(pd.Timestamp(row["origin_date"]).isoformat()
                == pd.to_datetime(week["data_as_of"], utc=True).isoformat(),
                f"weekly[{index}] data_as_of/evidence origin mismatch")
        require(str(row["current_state"]) == str(week["current"]["state"]),
                f"weekly[{index}] current state/evidence mismatch")
        require(str(row["current_state"]) == str(state_row["state"]),
                f"weekly[{index}] current state/state-history mismatch")
        require(str(row["model"]) == str(week["next_week"]["model"]),
                f"weekly[{index}] next model/evidence mismatch")
        require(pd.Timestamp(row["target_date"]).date().isoformat()
                == str(week["next_week"]["date"]),
                f"weekly[{index}] next target/evidence mismatch")
        for state_name in STATE_ORDER:
            require(
                np.isclose(
                    float(row[f"current_p_{state_name}"]),
                    float(week["current"]["probabilities"][state_name]),
                    atol=1e-12,
                ),
                f"weekly[{index}] current {state_name} evidence mismatch",
            )
            require(
                np.isclose(
                    float(row[f"current_p_{state_name}"]),
                    round(float(state_row[f"p_{state_name}"]), 8),
                    atol=1e-12,
                ),
                f"weekly[{index}] current {state_name}/state-history mismatch",
            )
            require(
                np.isclose(
                    float(row[f"next_p_{state_name}"]),
                    float(week["next_week"]["probabilities"][state_name]),
                    atol=1e-12,
                ),
                f"weekly[{index}] next {state_name} evidence mismatch",
            )
        next_probability = np.asarray(
            [float(row[f"next_p_{state_name}"]) for state_name in STATE_ORDER]
        )
        require(
            str(week["next_week"]["state"])
            == STATE_ORDER[int(np.argmax(next_probability))],
            f"weekly[{index}] next state/evidence argmax mismatch",
        )
        require(bool(row["fallback"]) == bool(week["next_week"]["fallback"]),
                f"weekly[{index}] next fallback/evidence mismatch")
        reason = "" if pd.isna(row["fallback_reason"]) else str(row["fallback_reason"])
        require(reason == str(week["next_week"].get("fallback_reason", "")),
                f"weekly[{index}] next fallback reason/evidence mismatch")
    return {"rows": len(evidence), "sha256": metadata["sha256"]}


def compose_joint_probability(
    xgboost_probability: Sequence[float],
    hazard_probability: float,
    current_state: str,
    *,
    direct_jump_floor: float = V4_DIRECT_JUMP_FLOOR,
) -> np.ndarray:
    """Independently rebuild the hazard/destination probability identity."""

    require(current_state in STATE_ORDER, "joint current_state is invalid")
    destination = np.asarray(xgboost_probability, dtype=float)
    require(
        destination.shape == (len(STATE_ORDER),)
        and np.isfinite(destination).all()
        and ((destination >= 0.0) & (destination <= 1.0)).all()
        and np.isclose(destination.sum(), 1.0, atol=1e-8),
        "joint XGBoost source probability is invalid",
    )
    hazard = float(hazard_probability)
    require(
        np.isfinite(hazard) and 0.0 < hazard < 1.0,
        "joint hazard probability must be strictly in (0,1)",
    )
    floor = float(direct_jump_floor)
    require(
        np.isfinite(floor) and 0.0 < floor < 0.5,
        "joint direct-jump floor is invalid",
    )
    current_position = STATE_ORDER.index(current_state)
    leave_positions = [
        position for position in range(len(STATE_ORDER))
        if position != current_position
    ]
    conditional = np.maximum(destination[leave_positions], floor)
    conditional /= conditional.sum()
    result = np.zeros(len(STATE_ORDER), dtype=float)
    result[current_position] = 1.0 - hazard
    result[leave_positions] = hazard * conditional
    require(np.isclose(result.sum(), 1.0, atol=1e-12),
            "joint probability does not sum to one")
    return result


def _require_nullable_timestamp_close(
    actual: Any,
    expected: pd.Timestamp | pd.NaT,
    context: str,
) -> None:
    if pd.isna(expected):
        require(pd.isna(actual), f"{context} must be null")
        return
    require(not pd.isna(actual), f"{context} is unexpectedly null")
    require(pd.Timestamp(actual) == pd.Timestamp(expected), f"{context} mismatch")


def _discounted_weight_evidence(
    expert_predictions: pd.DataFrame,
    *,
    origin_date: pd.Timestamp,
    current_fallbacks: Mapping[str, bool],
    expert_names: Sequence[str] = V4_STRUCTURAL_EXPERTS,
    half_life_weeks: float = V4_ENSEMBLE_HALF_LIFE_WEEKS,
    minimum_history_rows: int = V4_ENSEMBLE_MINIMUM_HISTORY_ROWS,
) -> dict[str, dict[str, Any]]:
    """Recompute one origin's causal discounted-loss weights from primitives."""

    eligible_history = expert_predictions.loc[
        expert_predictions["target_date"] < origin_date
    ].copy()
    key_columns = ["origin_date", "target_date"]
    if eligible_history.empty:
        history = eligible_history.copy()
        common_history_rows = 0
    else:
        nonfallback = eligible_history.assign(
            _eligible=~eligible_history["fallback"].astype(bool)
        ).pivot(index=key_columns, columns="model", values="_eligible")
        common_keys = nonfallback.loc[
            nonfallback.reindex(columns=expert_names, fill_value=False).all(axis=1)
        ].index
        history = eligible_history.merge(
            pd.DataFrame(list(common_keys), columns=key_columns),
            on=key_columns,
            how="inner",
            validate="many_to_one",
        )
        common_history_rows = int(len(common_keys))
    latest_target = history["target_date"].max() if not history.empty else pd.NaT
    eligible = [
        name for name in expert_names if not bool(current_fallbacks.get(name, False))
    ]
    if not eligible:
        return {
            name: {
                "weight": 0.0,
                "eligible": False,
                "current_fallback": True,
                "history_rows": common_history_rows,
                "common_history_rows": common_history_rows,
                "latest_eligible_target_date": latest_target,
                "discounted_log_loss": np.nan,
                "warmup": common_history_rows < minimum_history_rows,
            }
            for name in expert_names
        }

    positions = {state: index for index, state in enumerate(STATE_ORDER)}
    decay = float(np.exp(np.log(0.5) / float(half_life_weeks)))
    losses: dict[str, float] = {}
    counts: dict[str, int] = {}
    for name in expert_names:
        rows = history.loc[history["model"].astype(str).eq(name)].copy()
        counts[name] = int(len(rows))
        if rows.empty:
            losses[name] = 0.0
            continue
        actual_positions = np.asarray(
            [positions[str(value)] for value in rows["actual"]], dtype=int
        )
        probability = rows[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
        realised = np.clip(
            probability[np.arange(len(rows)), actual_positions], 1e-12, 1.0
        )
        age_weeks = (
            (origin_date - rows["target_date"]).dt.total_seconds()
            / (7.0 * 24.0 * 60.0 * 60.0)
        ).to_numpy(dtype=float)
        require((age_weeks > 0.0).all(),
                "ensemble eligible loss is not strictly pre-origin")
        losses[name] = float(np.sum((decay ** age_weeks) * -np.log(realised)))

    warmup = common_history_rows < int(minimum_history_rows)
    weights = {name: 0.0 for name in expert_names}
    if warmup:
        for name in eligible:
            weights[name] = 1.0 / len(eligible)
    else:
        scores = np.asarray([-losses[name] for name in eligible], dtype=float)
        scores -= scores.max()
        raw = np.exp(scores)
        raw /= raw.sum()
        for index, name in enumerate(eligible):
            weights[name] = float(raw[index])
    return {
        name: {
            "weight": weights[name],
            "eligible": name in eligible,
            "current_fallback": bool(current_fallbacks.get(name, False)),
            "history_rows": counts[name],
            "common_history_rows": common_history_rows,
            "latest_eligible_target_date": latest_target,
            "discounted_log_loss": losses[name],
            "warmup": warmup,
        }
        for name in expert_names
    }


def probability_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    positions = {state: index for index, state in enumerate(STATE_ORDER)}
    rows: list[dict[str, Any]] = []
    for model, group in predictions.groupby("model", sort=False):
        probability = group[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
        actual = group["actual"].astype(str).to_numpy()
        actual_probability = np.asarray(
            [probability[row, positions[state]] for row, state in enumerate(actual)]
        )
        predicted = np.asarray(STATE_ORDER, dtype=object)[probability.argmax(axis=1)]
        one_hot = np.zeros_like(probability)
        for row_index, state in enumerate(actual):
            one_hot[row_index, positions[state]] = 1.0
        confidence = probability.max(axis=1)
        correct = (predicted == actual).astype(float)
        current = group["current_state"].astype(str).to_numpy()
        calibration_error = 0.0
        boundaries = np.linspace(0.0, 1.0, 11)
        for index in range(10):
            lower, upper = boundaries[index : index + 2]
            mask = (
                (confidence >= lower) & (confidence <= upper)
                if index == 0
                else (confidence > lower) & (confidence <= upper)
            )
            if mask.any():
                calibration_error += float(mask.mean()) * abs(
                    float(correct[mask].mean()) - float(confidence[mask].mean())
                )
        rows.append(
            {
                "model": str(model),
                "log_loss": float(-np.log(actual_probability).mean()),
                "brier": float(np.mean(np.sum((probability - one_hot) ** 2, axis=1))),
                "accuracy": float(accuracy_score(actual, predicted)),
                "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
                "macro_f1": float(
                    f1_score(
                        actual,
                        predicted,
                        labels=STATE_ORDER,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "transition_recall": float(
                    recall_score(
                        actual != current,
                        predicted != current,
                        zero_division=0,
                    )
                ),
                "calibration_error": float(calibration_error),
                "n_predictions": int(len(group)),
                "fallback_count": int(group["fallback"].sum()),
            }
        )
    return pd.DataFrame(rows).set_index("model")


def audit_feature_manifest(
    payload: dict[str, Any],
    artifacts: Path,
) -> dict[str, Any]:
    path = artifacts / "feature-manifest.json"
    require(path.is_file() and path.stat().st_size > 0, f"missing/empty file: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(document, dict), "feature manifest must be an object")
    published_hash = str(document.get("sha256", ""))
    body = dict(document)
    body.pop("sha256", None)
    require(
        published_hash == canonical_json_sha256(body),
        "feature manifest SHA-256 is invalid",
    )
    require(
        payload["model"].get("feature_manifest_sha256") == published_hash,
        "payload/artifact feature manifest hash mismatch",
    )
    require(
        body.get("feature_set_version") == V4_FEATURE_SET_VERSION,
        "feature manifest version mismatch",
    )
    groups = body.get("groups")
    require(isinstance(groups, list) and groups, "feature manifest groups are missing")
    group_ids: list[str] = []
    all_features: list[str] = []
    for index, group in enumerate(groups):
        context = f"feature manifest group[{index}]"
        require(isinstance(group, dict), f"{context} must be an object")
        group_id = str(group.get("id", ""))
        require(group_id, f"{context} id is empty")
        group_ids.append(group_id)
        features = group.get("features")
        require(isinstance(features, list) and features, f"{context} features missing")
        names = [str(value) for value in features]
        require(all(names), f"{context} contains an empty feature name")
        require(len(names) == len(set(names)), f"{context} duplicates a feature")
        require(
            int(group.get("feature_count", -1)) == len(names),
            f"{context} feature_count mismatch",
        )
        prefixes = V4_FEATURE_GROUP_PREFIXES.get(group_id)
        if prefixes is not None:
            require(
                all(name.startswith(prefixes) for name in names),
                f"{context} contains a feature outside its frozen namespace",
            )
        if group_id == "financial_conditions":
            require(
                set(names) == V4_FINANCIAL_CONDITION_FEATURES,
                "financial_conditions must contain exactly the four preregistered "
                "ANFCI features",
            )
        all_features.extend(names)
    require(len(group_ids) == len(set(group_ids)), "feature manifest group IDs duplicate")
    require(
        set(group_ids) == set(V4_ABLATION_VARIANTS["all_structural"]),
        "feature manifest group set differs from preregistration",
    )
    require(
        len(all_features) == len(set(all_features)),
        "a model feature is assigned to more than one feature group",
    )
    legacy_features = next(
        (group["features"] for group in groups if group["id"] == "legacy_v3"),
        [],
    )
    non_anfci_structural_prefixes = tuple(
        prefix
        for group_id, prefixes in V4_FEATURE_GROUP_PREFIXES.items()
        if group_id != "financial_conditions"
        for prefix in prefixes
    )
    require(
        not any(
            str(name).startswith(non_anfci_structural_prefixes)
            for name in legacy_features
        ),
        "legacy_v3 contains a structural feature namespace",
    )
    legacy_anfci = {
        str(name) for name in legacy_features if str(name).startswith("anfci__")
    }
    require(
        legacy_anfci.issubset(V4_ANFCI_LEGACY_AVAILABILITY_FEATURES),
        "legacy_v3 contains an unregistered ANFCI model feature",
    )
    require(
        int(body.get("feature_count", -1)) == len(all_features),
        "feature manifest total count mismatch",
    )
    return {
        "sha256": published_hash,
        "feature_count": len(all_features),
        "groups": len(groups),
        "group_feature_counts": {
            str(group["id"]): int(group["feature_count"]) for group in groups
        },
        "group_features": {
            str(group["id"]): [str(value) for value in group["features"]]
            for group in groups
        },
    }


def audit_joint_predictions(
    predictions: pd.DataFrame,
    transition_predictions: pd.DataFrame,
) -> dict[str, Any]:
    """Rebuild every joint structural OOS forecast from its two sources."""

    require_columns(
        transition_predictions,
        {
            "origin_date", "target_end", "horizon", "model", "evaluation_split",
            "current_state", "actual_change", "p_change", "fallback",
            "calibration_fallback",
        },
        "transition-oos-predictions.csv for structural audit",
    )
    transition = transition_predictions.copy()
    transition["fallback"] = boolean_series(
        transition["fallback"], "joint binary_xgboost fallback"
    )
    transition["calibration_fallback"] = boolean_series(
        transition["calibration_fallback"],
        "joint binary_xgboost calibration_fallback",
    )
    transition["actual_change"] = boolean_series(
        transition["actual_change"], "joint binary_xgboost actual_change"
    )
    binary = transition.loc[
        transition["model"].astype(str).eq("binary_xgboost")
        & transition["horizon"].astype(int).eq(1)
    ].copy()
    xgboost = predictions.loc[predictions["model"].eq("xgboost")].copy()
    joint = predictions.loc[
        predictions["model"].eq("xgb_hazard_destination")
    ].copy()
    require(not xgboost.empty and not joint.empty and not binary.empty,
            "structural joint source or result rows are missing")
    for frame, context in ((xgboost, "xgboost"), (joint, "joint")):
        require(not frame.duplicated(["origin_date", "target_date"]).any(),
                f"duplicate {context} structural origin")
    require(not binary.duplicated(["origin_date", "target_end"]).any(),
            "duplicate binary_xgboost structural origin")
    require(
        len(xgboost) == len(joint) == len(binary),
        "joint sources do not have strict common origins",
    )
    binary = binary.rename(
        columns={
            "target_end": "binary_target_date",
            "evaluation_split": "binary_split",
            "current_state": "binary_current_state",
            "fallback": "binary_fallback",
            "calibration_fallback": "binary_calibration_fallback",
        }
    )
    joined = xgboost.merge(
        joint,
        on=["origin_date", "target_date"],
        suffixes=("_xgb", "_joint"),
        validate="one_to_one",
    ).merge(binary, on="origin_date", validate="one_to_one")
    require(len(joined) == len(xgboost), "joint source origin join is incomplete")
    split_mapping = {
        "selection": "selection",
        "holdout": "retrospective_diagnostic",
    }
    audited = 0
    for _, row in joined.iterrows():
        context = f"joint origin {pd.Timestamp(row['origin_date']).isoformat()}"
        require(
            pd.Timestamp(row["target_date"])
            == pd.Timestamp(row["binary_target_date"]),
            f"{context} target mismatch",
        )
        require(
            str(row["current_state_xgb"])
            == str(row["current_state_joint"])
            == str(row["binary_current_state"]),
            f"{context} current state mismatch",
        )
        require(
            str(row["actual_xgb"]) == str(row["actual_joint"]),
            f"{context} actual state mismatch",
        )
        require(
            str(row["evaluation_split_xgb"])
            == str(row["evaluation_split_joint"]),
            f"{context} main evaluation split mismatch",
        )
        expected_change = str(row["actual_xgb"]) != str(row["current_state_xgb"])
        require(bool(row["actual_change"]) == expected_change,
                f"{context} binary actual is inconsistent")
        require(
            str(row["binary_split"])
            == split_mapping.get(str(row["evaluation_split_xgb"])),
            f"{context} evaluation split mismatch",
        )
        expected = compose_joint_probability(
            [row[f"{column}_xgb"] for column in PROBABILITY_COLUMNS],
            float(row["p_change"]),
            str(row["current_state_xgb"]),
        )
        actual = np.asarray(
            [row[f"{column}_joint"] for column in PROBABILITY_COLUMNS], dtype=float
        )
        require(np.allclose(actual, expected, atol=1e-12),
                f"{context} probability recomputation mismatch")
        expected_fallback = bool(
            row["fallback_xgb"]
            or row["binary_fallback"]
            or row["binary_calibration_fallback"]
        )
        require(bool(row["fallback_joint"]) == expected_fallback,
                f"{context} fallback mismatch")
        if "direct_jump_floor" in row and not pd.isna(row["direct_jump_floor"]):
            require(np.isclose(float(row["direct_jump_floor"]),
                               V4_DIRECT_JUMP_FLOOR, atol=0.0),
                    f"{context} direct-jump floor mismatch")
        audited += 1
    return {"origins": audited, "direct_jump_floor": V4_DIRECT_JUMP_FLOOR}


def audit_stacking_weights(
    predictions: pd.DataFrame,
    artifacts: Path,
    *,
    allow_v5_multiscale_rows: bool = False,
) -> dict[str, Any]:
    path = artifacts / "stacking-weights.csv"
    weights = read_csv(
        path,
        ("origin_date", "target_date", "latest_eligible_target_date"),
    )
    require_columns(
        weights,
        {
            "origin_date", "target_date", "evaluation_split", "ensemble_model",
            "expert", "weight", "eligible", "current_fallback", "history_rows",
            "common_history_rows", "latest_eligible_target_date",
            "discounted_log_loss", "warmup", "half_life_weeks",
            "minimum_history_rows", "eligible_loss_rule",
        },
        "stacking-weights.csv",
    )
    for field in ("eligible", "current_fallback", "warmup"):
        weights[field] = boolean_series(weights[field], f"stacking {field}")
    if allow_v5_multiscale_rows:
        weights = weights.loc[
            weights["ensemble_model"].astype(str).eq(
                "causal_dynamic_ensemble"
            )
        ].copy()
        require(
            not weights.empty,
            "v5 stacking sidecar omits causal_dynamic_ensemble",
        )
    require(set(weights["ensemble_model"].astype(str)) == {"causal_dynamic_ensemble"},
            "stacking ensemble model mismatch")
    require(set(weights["expert"].astype(str)) == set(V4_STRUCTURAL_EXPERTS),
            "stacking expert set mismatch")
    require((weights["half_life_weeks"].astype(float)
             == V4_ENSEMBLE_HALF_LIFE_WEEKS).all(),
            "stacking half-life mismatch")
    require((weights["minimum_history_rows"].astype(int)
             == V4_ENSEMBLE_MINIMUM_HISTORY_ROWS).all(),
            "stacking minimum history mismatch")
    require(set(weights["eligible_loss_rule"].astype(str))
            == {"target_date_strictly_before_origin"},
            "stacking eligible-loss rule mismatch")
    require(not weights.duplicated(["origin_date", "target_date", "expert"]).any(),
            "duplicate stacking origin/expert")

    experts = predictions.loc[
        predictions["model"].isin(V4_STRUCTURAL_EXPERTS)
    ].copy()
    ensemble = predictions.loc[
        predictions["model"].eq("causal_dynamic_ensemble")
    ].copy()
    require(not experts.empty and not ensemble.empty,
            "structural experts or ensemble predictions are missing")
    reference_keys: pd.DataFrame | None = None
    for name in V4_STRUCTURAL_EXPERTS:
        keys = experts.loc[experts["model"].eq(name), [
            "origin_date", "target_date", "evaluation_split", "current_state", "actual"
        ]].sort_values(["origin_date", "target_date"], ignore_index=True)
        require(not keys.empty, f"stacking expert {name} is missing")
        if reference_keys is None:
            reference_keys = keys
        else:
            require(keys.equals(reference_keys),
                    f"stacking expert {name} lacks strict common origins")
    assert reference_keys is not None
    require(len(weights) == len(reference_keys) * len(V4_STRUCTURAL_EXPERTS),
            "stacking weight row count mismatch")
    weight_keys = weights[["origin_date", "target_date"]].drop_duplicates().sort_values(
        ["origin_date", "target_date"], ignore_index=True
    )
    require(weight_keys.equals(reference_keys[["origin_date", "target_date"]]),
            "stacking weights do not cover every common OOS origin")
    ensemble_keys = ensemble[["origin_date", "target_date"]].sort_values(
        ["origin_date", "target_date"], ignore_index=True
    )
    require(ensemble_keys.equals(weight_keys),
            "ensemble predictions and stacking weights have different origins")

    for (origin, target), current_weights in weights.groupby(
        ["origin_date", "target_date"], sort=True
    ):
        context = f"stacking origin {pd.Timestamp(origin).isoformat()}"
        current_experts = experts.loc[
            experts["origin_date"].eq(origin) & experts["target_date"].eq(target)
        ].set_index("model")
        require(set(current_experts.index) == set(V4_STRUCTURAL_EXPERTS),
                f"{context} expert rows mismatch")
        fallbacks = {
            name: bool(current_experts.loc[name, "fallback"])
            for name in V4_STRUCTURAL_EXPERTS
        }
        expected = _discounted_weight_evidence(
            experts,
            origin_date=pd.Timestamp(origin),
            current_fallbacks=fallbacks,
        )
        current_weights = current_weights.set_index("expert")
        require(set(current_weights.index) == set(expected),
                f"{context} expert evidence mismatch")
        for name, evidence in expected.items():
            actual = current_weights.loc[name]
            require(np.isclose(float(actual["weight"]), evidence["weight"], atol=1e-12),
                    f"{context}/{name} weight mismatch")
            require(bool(actual["eligible"]) == bool(evidence["eligible"]),
                    f"{context}/{name} eligible mismatch")
            require(bool(actual["current_fallback"])
                    == bool(evidence["current_fallback"]),
                    f"{context}/{name} current fallback mismatch")
            for field in ("history_rows", "common_history_rows"):
                require(int(actual[field]) == int(evidence[field]),
                        f"{context}/{name} {field} mismatch")
            require(bool(actual["warmup"]) == bool(evidence["warmup"]),
                    f"{context}/{name} warmup mismatch")
            _require_nullable_timestamp_close(
                actual["latest_eligible_target_date"],
                evidence["latest_eligible_target_date"],
                f"{context}/{name} latest eligible target",
            )
            nullable_close(
                actual["discounted_log_loss"],
                None if pd.isna(evidence["discounted_log_loss"])
                else float(evidence["discounted_log_loss"]),
                f"{context}/{name} discounted log loss",
                atol=1e-10,
            )
        expected_weights = np.asarray(
            [expected[name]["weight"] for name in V4_STRUCTURAL_EXPERTS]
        )
        weight_sum = float(expected_weights.sum())
        require(np.isclose(weight_sum, 0.0 if all(fallbacks.values()) else 1.0,
                           atol=1e-12),
                f"{context} weight sum mismatch")
        current_ensemble = ensemble.loc[
            ensemble["origin_date"].eq(origin) & ensemble["target_date"].eq(target)
        ]
        require(len(current_ensemble) == 1, f"{context} ensemble row missing")
        current_ensemble = current_ensemble.iloc[0]
        expected_probability = (
            np.full(len(STATE_ORDER), 1.0 / len(STATE_ORDER))
            if weight_sum == 0.0
            else sum(
                expected[name]["weight"]
                * current_experts.loc[name, list(PROBABILITY_COLUMNS)].to_numpy(
                    dtype=float
                )
                for name in V4_STRUCTURAL_EXPERTS
            )
        )
        expected_probability = np.asarray(expected_probability, dtype=float)
        expected_probability /= expected_probability.sum()
        actual_probability = current_ensemble[list(PROBABILITY_COLUMNS)].to_numpy(
            dtype=float
        )
        require(np.allclose(actual_probability, expected_probability, atol=1e-12),
                f"{context} ensemble probability mismatch")
        require(bool(current_ensemble["fallback"]) == (weight_sum == 0.0),
                f"{context} ensemble fallback mismatch")
        require(set(current_weights["evaluation_split"].astype(str))
                == {str(current_ensemble["evaluation_split"])},
                f"{context} split mismatch")
    return {
        "origins": len(reference_keys),
        "experts": list(V4_STRUCTURAL_EXPERTS),
        "rows": len(weights),
    }


def audit_v5_multiscale_ensemble(
    predictions: pd.DataFrame,
    stacking_weights: pd.DataFrame,
    scale_predictions: pd.DataFrame,
    artifacts: Path,
) -> dict[str, Any]:
    """Rebuild every V5 multiscale pool and its fixed outer average."""

    from regime_lab.analysis.structural_models import (
        MULTISCALE_SCALE_FORECAST_COLUMNS,
    )

    experts = tuple(V4_STRUCTURAL_EXPERTS)
    scales = (26, 52, 104)
    outer_weight = 1.0 / 3.0
    require(
        tuple(scale_predictions.columns) == MULTISCALE_SCALE_FORECAST_COLUMNS,
        "v5 multiscale scale sidecar columns mismatch",
    )
    scales_frame = scale_predictions.copy()
    for column in ("origin_date", "target_date"):
        scales_frame[column] = pd.to_datetime(
            scales_frame[column], utc=True, errors="raise"
        )
    scales_frame["fallback"] = boolean_series(
        scales_frame["fallback"],
        "v5 multiscale scale fallback",
    )
    require(
        not scales_frame.duplicated(
            ["row_role", "origin_date", "target_date", "scale_half_life_weeks"]
        ).any(),
        "v5 multiscale scale sidecar keys duplicate",
    )
    scale_probability = scales_frame.loc[:, PROBABILITY_COLUMNS].apply(
        pd.to_numeric, errors="raise"
    )
    require(
        np.isfinite(scale_probability.to_numpy(dtype=float)).all()
        and ((scale_probability >= 0.0) & (scale_probability <= 1.0)).all().all()
        and np.allclose(
            scale_probability.sum(axis=1),
            1.0,
            atol=1e-12,
            rtol=0.0,
        ),
        "v5 multiscale scale probabilities are invalid",
    )
    require(
        set(scales_frame["row_role"].astype(str)) == {"oos", "latest_forecast"}
        and set(scales_frame["ensemble_model"].astype(str))
        == {V5_MULTISCALE_MODEL}
        and set(scales_frame["scale_half_life_weeks"].astype(int)) == set(scales),
        "v5 multiscale scale identity mismatch",
    )
    exact_columns = {
        "minimum_history_rows": 26,
        "eligible_loss_rule": "target_date_strictly_before_origin",
        "inner_pool_method": "causal_discounted_completed_oos_log_score",
        "expert_models": ";".join(experts),
    }
    for column, expected in exact_columns.items():
        require(
            scales_frame[column].eq(expected).all(),
            f"v5 multiscale scale {column} mismatch",
        )
    require(
        np.allclose(
            pd.to_numeric(
                scales_frame["outer_scale_weight"], errors="raise"
            ),
            outer_weight,
            atol=1e-15,
            rtol=0.0,
        ),
        "v5 multiscale outer scale weights are not exact thirds",
    )

    expert_oos = predictions.loc[
        predictions["model"].astype(str).isin(experts)
    ].copy()
    multiscale_oos = predictions.loc[
        predictions["model"].astype(str).eq(V5_MULTISCALE_MODEL)
    ].copy()
    require(
        not expert_oos.empty and not multiscale_oos.empty,
        "v5 multiscale OOS candidate or experts are missing",
    )
    oos_scales = scales_frame.loc[
        scales_frame["row_role"].astype(str).eq("oos")
    ].copy()
    latest_scales = scales_frame.loc[
        scales_frame["row_role"].astype(str).eq("latest_forecast")
    ].copy()
    oos_keys = multiscale_oos.loc[:, ["origin_date", "target_date"]].sort_values(
        ["origin_date", "target_date"], ignore_index=True
    )
    scale_oos_keys = oos_scales.loc[
        :, ["origin_date", "target_date"]
    ].drop_duplicates().sort_values(
        ["origin_date", "target_date"], ignore_index=True
    )
    require(
        oos_keys.equals(scale_oos_keys)
        and len(oos_scales) == len(oos_keys) * len(scales),
        "v5 multiscale scale sidecar does not cover every OOS origin",
    )
    require(
        len(latest_scales) == len(scales)
        and latest_scales[["origin_date", "target_date"]].drop_duplicates().shape[0]
        == 1,
        "v5 multiscale latest scale sidecar must contain exactly three rows",
    )

    weights = stacking_weights.copy()
    require_columns(
        weights,
        {
            "origin_date",
            "target_date",
            "evaluation_split",
            "ensemble_model",
            "expert",
            "weight",
            "eligible",
            "current_fallback",
            "history_rows",
            "common_history_rows",
            "latest_eligible_target_date",
            "discounted_log_loss",
            "warmup",
            "half_life_weeks",
            "outer_scale_weight",
            "minimum_history_rows",
            "eligible_loss_rule",
            "inner_pool_method",
        },
        "v5 multiscale stacking weights",
    )
    for column in ("origin_date", "target_date", "latest_eligible_target_date"):
        weights[column] = pd.to_datetime(weights[column], utc=True, errors="coerce")
    for column in ("eligible", "current_fallback", "warmup"):
        weights[column] = boolean_series(
            weights[column], f"v5 multiscale stacking {column}"
        )
    weights = weights.loc[
        weights["ensemble_model"].astype(str).eq(V5_MULTISCALE_MODEL)
    ].copy()
    require(
        not weights.empty
        and set(weights["expert"].astype(str)) == set(experts)
        and set(weights["half_life_weeks"].astype(int)) == set(scales),
        "v5 multiscale stacking model/expert/scale identity mismatch",
    )
    require(
        not weights.duplicated(
            ["origin_date", "target_date", "half_life_weeks", "expert"]
        ).any(),
        "v5 multiscale stacking keys duplicate",
    )
    scale_keys = scales_frame.loc[
        :, ["origin_date", "target_date", "scale_half_life_weeks"]
    ].rename(columns={"scale_half_life_weeks": "half_life_weeks"})
    weight_keys = weights.loc[
        :, ["origin_date", "target_date", "half_life_weeks"]
    ].drop_duplicates()
    require(
        len(weights) == len(scale_keys) * len(experts)
        and set(map(tuple, weight_keys.to_numpy()))
        == set(map(tuple, scale_keys.to_numpy())),
        "v5 multiscale stacking/scale sidecar keys mismatch",
    )

    structural = read_csv(
        artifacts / "structural-forecasts.csv",
        ("origin_date", "target_date"),
    )
    structural["fallback"] = boolean_series(
        structural["fallback"], "v5 structural forecast fallback"
    )
    latest_origin = pd.Timestamp(latest_scales["origin_date"].iloc[0])
    latest_target = pd.Timestamp(latest_scales["target_date"].iloc[0])
    latest_experts = structural.loc[
        structural["origin_date"].eq(latest_origin)
        & structural["target_date"].eq(latest_target)
        & structural["model"].astype(str).isin(experts)
    ].copy()
    require(
        set(latest_experts["model"].astype(str)) == set(experts),
        "v5 multiscale latest structural experts are missing",
    )

    scale_vectors: dict[tuple[pd.Timestamp, pd.Timestamp, int], np.ndarray] = {}
    for row in scales_frame.itertuples(index=False):
        origin = pd.Timestamp(row.origin_date)
        target = pd.Timestamp(row.target_date)
        scale = int(row.scale_half_life_weeks)
        role = str(row.row_role)
        if role == "oos":
            current = expert_oos.loc[
                expert_oos["origin_date"].eq(origin)
                & expert_oos["target_date"].eq(target)
            ].copy()
            expected_artifact = "oos-predictions.csv"
            expected_split = str(
                multiscale_oos.loc[
                    multiscale_oos["origin_date"].eq(origin)
                    & multiscale_oos["target_date"].eq(target),
                    "evaluation_split",
                ].iloc[0]
            )
        else:
            current = latest_experts.copy()
            expected_artifact = "structural-forecasts.csv"
            expected_split = "prospective"
        current = current.set_index(current["model"].astype(str))
        require(
            set(current.index) == set(experts),
            "v5 multiscale scale is missing an expert forecast",
        )
        require(
            str(row.evaluation_split) == expected_split
            and str(row.expert_forecast_artifact) == expected_artifact,
            "v5 multiscale scale source role/split mismatch",
        )
        expected_key = (
            f"{expected_artifact}|origin={origin.date().isoformat()}"
            f"|target={target.date().isoformat()}"
            f"|models={';'.join(experts)}"
        )
        require(
            str(row.expert_forecast_key) == expected_key,
            "v5 multiscale expert forecast key mismatch",
        )
        fallbacks = {
            name: bool(current.loc[name, "fallback"]) for name in experts
        }
        expected = _discounted_weight_evidence(
            expert_oos,
            origin_date=origin,
            current_fallbacks=fallbacks,
            expert_names=experts,
            half_life_weeks=float(scale),
            minimum_history_rows=26,
        )
        current_weights = weights.loc[
            weights["origin_date"].eq(origin)
            & weights["target_date"].eq(target)
            & weights["half_life_weeks"].astype(int).eq(scale)
        ].set_index(weights.loc[
            weights["origin_date"].eq(origin)
            & weights["target_date"].eq(target)
            & weights["half_life_weeks"].astype(int).eq(scale),
            "expert",
        ].astype(str))
        require(
            set(current_weights.index) == set(experts),
            "v5 multiscale scale stacking expert set mismatch",
        )
        for name, evidence in expected.items():
            actual = current_weights.loc[name]
            _require_v5_recomputed_value(
                evidence["weight"],
                actual["weight"],
                context=f"v5 multiscale {origin}/{scale}/{name} weight",
                tolerance=1e-12,
            )
            for field in (
                "eligible",
                "current_fallback",
                "history_rows",
                "common_history_rows",
                "warmup",
                "latest_eligible_target_date",
                "discounted_log_loss",
            ):
                _require_v5_recomputed_value(
                    evidence[field],
                    actual[field],
                    context=f"v5 multiscale {origin}/{scale}/{name} {field}",
                    tolerance=1e-10,
                )
            require(
                np.isclose(
                    float(actual["outer_scale_weight"]),
                    outer_weight,
                    atol=1e-15,
                    rtol=0.0,
                )
                and int(actual["minimum_history_rows"]) == 26
                and str(actual["eligible_loss_rule"])
                == "target_date_strictly_before_origin"
                and str(actual["inner_pool_method"])
                == "causal_discounted_completed_oos_log_score",
                "v5 multiscale stacking frozen contract mismatch",
            )
        weight_sum = sum(float(expected[name]["weight"]) for name in experts)
        expected_fallback = np.isclose(weight_sum, 0.0, atol=1e-12, rtol=0.0)
        probability = (
            np.full(len(STATE_ORDER), 1.0 / len(STATE_ORDER))
            if expected_fallback
            else sum(
                float(expected[name]["weight"])
                * current.loc[name, list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
                for name in experts
            )
        )
        probability = np.asarray(probability, dtype=float)
        probability /= probability.sum()
        actual_probability = np.asarray(
            [getattr(row, column) for column in PROBABILITY_COLUMNS], dtype=float
        )
        actual_fallback_reason = (
            "" if pd.isna(row.fallback_reason) else str(row.fallback_reason)
        )
        require(
            np.allclose(
                actual_probability,
                probability,
                atol=1e-12,
                rtol=0.0,
            )
            and bool(row.fallback) == bool(expected_fallback)
            and actual_fallback_reason
            == ("all_structural_experts_fallback" if expected_fallback else ""),
            "v5 multiscale scale probability/fallback mismatch",
        )
        scale_vectors[(origin, target, scale)] = probability

    for row in multiscale_oos.itertuples(index=False):
        origin = pd.Timestamp(row.origin_date)
        target = pd.Timestamp(row.target_date)
        expected = np.mean(
            np.stack([scale_vectors[(origin, target, scale)] for scale in scales]),
            axis=0,
        )
        actual = np.asarray(
            [getattr(row, column) for column in PROBABILITY_COLUMNS], dtype=float
        )
        require(
            np.allclose(actual, expected, atol=1e-12, rtol=0.0)
            and str(row.multiscale_half_lives_weeks) == "26;52;104"
            and all(
                np.isclose(float(value), outer_weight, atol=1e-15, rtol=0.0)
                for value in str(row.multiscale_outer_weights).split(";")
            )
            and str(row.multiscale_aggregation)
            == "fixed_equal_probability_average"
            and str(row.inner_pool_method)
            == "causal_discounted_completed_oos_log_score"
            and str(row.eligible_loss_rule)
            == "target_date_strictly_before_origin"
            and int(row.ensemble_minimum_history_rows) == 26,
            "v5 multiscale aggregate OOS probability/metadata mismatch",
        )

    latest_aggregate = structural.loc[
        structural["origin_date"].eq(latest_origin)
        & structural["target_date"].eq(latest_target)
        & structural["model"].astype(str).eq(V5_MULTISCALE_MODEL)
    ]
    require(
        len(latest_aggregate) == 1,
        "v5 multiscale latest aggregate forecast is missing",
    )
    expected_latest = np.mean(
        np.stack(
            [
                scale_vectors[(latest_origin, latest_target, scale)]
                for scale in scales
            ]
        ),
        axis=0,
    )
    require(
        np.allclose(
            latest_aggregate.iloc[0][list(PROBABILITY_COLUMNS)].to_numpy(
                dtype=float
            ),
            expected_latest,
            atol=1e-12,
            rtol=0.0,
        ),
        "v5 multiscale latest aggregate probability mismatch",
    )
    return {
        "oos_origins": len(oos_keys),
        "latest_origins": 1,
        "scale_rows": len(scales_frame),
        "stacking_rows": len(weights),
        "scales": list(scales),
        "outer_scale_weights": [outer_weight] * 3,
    }


def _ablation_row_losses(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    positions = {state: index for index, state in enumerate(STATE_ORDER)}
    probability = frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    actual_positions = np.asarray(
        [positions[str(value)] for value in frame["actual"]], dtype=int
    )
    realised = np.clip(
        probability[np.arange(len(frame)), actual_positions], 1e-9, 1.0
    )
    one_hot = np.zeros_like(probability)
    one_hot[np.arange(len(frame)), actual_positions] = 1.0
    return -np.log(realised), np.sum((probability - one_hot) ** 2, axis=1)


def audit_feature_ablation(
    payload: dict[str, Any],
    artifacts: Path,
    *,
    main_predictions: pd.DataFrame,
    feature_manifest: dict[str, Any],
) -> dict[str, Any]:
    predictions = read_csv(
        artifacts / "feature-ablation-oos-predictions.csv",
        ("origin_date", "target_date"),
    )
    leaderboard = read_csv(artifacts / "feature-ablation-leaderboard.csv")
    manifest_path = artifacts / "feature-ablation-manifest.json"
    require(manifest_path.is_file() and manifest_path.stat().st_size > 0,
            f"missing/empty file: {manifest_path}")
    manifest_document = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(isinstance(manifest_document, dict),
            "feature ablation manifest must be an object")
    manifest_hash = str(manifest_document.get("sha256", ""))
    manifest_body = dict(manifest_document)
    manifest_body.pop("sha256", None)
    require(manifest_hash == canonical_json_sha256(manifest_body),
            "feature ablation manifest SHA-256 is invalid")
    require(
        set(manifest_body) == {
            "schema_version", "anchor_model", "reference_variant",
            "published_variant", "primary_period", "post_2023_role",
            "may_change_published_variant", "variants",
        },
        "feature ablation manifest fields mismatch",
    )
    require(
        {
            "schema_version": manifest_body.get("schema_version"),
            "anchor_model": manifest_body.get("anchor_model"),
            "reference_variant": manifest_body.get("reference_variant"),
            "published_variant": manifest_body.get("published_variant"),
            "primary_period": manifest_body.get("primary_period"),
            "post_2023_role": manifest_body.get("post_2023_role"),
            "may_change_published_variant": manifest_body.get(
                "may_change_published_variant"
            ),
        }
        == {
            "schema_version": "1.0.0",
            "anchor_model": "xgboost",
            "reference_variant": "legacy_v3",
            "published_variant": "all_structural",
            "primary_period": "pre_2023_selection_oos",
            "post_2023_role": "retrospective_diagnostic_only",
            "may_change_published_variant": False,
        },
        "feature ablation manifest contract values mismatch",
    )
    manifest_rows = manifest_body.get("variants")
    require(isinstance(manifest_rows, list) and manifest_rows,
            "feature ablation manifest variants are missing")

    require_columns(
        predictions,
        {
            "variant", "origin_date", "target_date", "evaluation_split",
            "model", "current_state", "actual", "predicted", "train_size",
            "gap", "fallback", "fallback_reason", "reused_main_benchmark",
            *PROBABILITY_COLUMNS,
        },
        "feature-ablation-oos-predictions.csv",
    )
    require_columns(
        leaderboard,
        {
            "variant", "evaluation_split", "role", "selection_rank",
            "selection_winner", "model", "log_loss", "brier", "accuracy",
            "balanced_accuracy", "macro_f1", "transition_recall",
            "calibration_error", "n_predictions", "fallback_count", "paired_n",
            "paired_log_loss_delta_vs_legacy",
            "paired_brier_delta_vs_legacy",
        },
        "feature-ablation-leaderboard.csv",
    )
    predictions["fallback"] = boolean_series(
        predictions["fallback"], "feature ablation fallback"
    )
    predictions["reused_main_benchmark"] = boolean_series(
        predictions["reused_main_benchmark"],
        "feature ablation reused_main_benchmark",
    )
    leaderboard["selection_winner"] = boolean_series(
        leaderboard["selection_winner"], "feature ablation selection_winner"
    )
    expected_variants = set(V4_ABLATION_VARIANTS)
    require(set(predictions["variant"].astype(str)) == expected_variants,
            "feature ablation prediction variant set mismatch")
    require(set(leaderboard["variant"].astype(str)) == expected_variants,
            "feature ablation leaderboard variant set mismatch")
    require(set(predictions["model"].astype(str)) == {"xgboost"},
            "feature ablation must use only XGBoost")
    require(set(leaderboard["model"].astype(str)) == {"xgboost"},
            "feature ablation leaderboard model mismatch")
    require(set(predictions["evaluation_split"].astype(str))
            == {"selection", "holdout"},
            "feature ablation split set mismatch")
    require(
        not predictions.duplicated(["variant", "origin_date", "target_date"]).any(),
        "duplicate feature ablation variant/origin",
    )
    require_calendar_horizon(
        predictions.assign(_horizon=1),
        "origin_date",
        "target_date",
        "_horizon",
        "feature ablation OOS",
    )
    require((predictions["gap"].astype(int) == 1).all(),
            "feature ablation gap must be one week")
    probability = predictions[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    require(np.isfinite(probability).all()
            and ((probability >= 0.0) & (probability <= 1.0)).all()
            and np.allclose(probability.sum(axis=1), 1.0, atol=1e-10),
            "feature ablation probabilities are invalid")
    expected_labels = np.asarray(STATE_ORDER, dtype=object)[probability.argmax(axis=1)]
    require(np.array_equal(predictions["predicted"].astype(str), expected_labels),
            "feature ablation predicted label mismatch")
    require(
        predictions.loc[predictions["variant"].eq("all_structural"),
                        "reused_main_benchmark"].all()
        and not predictions.loc[~predictions["variant"].eq("all_structural"),
                                "reused_main_benchmark"].any(),
        "feature ablation main-benchmark reuse flags mismatch",
    )
    cutoff = pd.to_datetime(payload["model"].get("selection_end"), utc=True)
    require((predictions.loc[predictions["evaluation_split"].eq("selection"),
                             "target_date"] < cutoff).all(),
            "feature ablation selection reaches the cutoff")
    require((predictions.loc[predictions["evaluation_split"].eq("holdout"),
                             "target_date"] >= cutoff).all(),
            "feature ablation diagnostic precedes the cutoff")

    reference = predictions.loc[
        predictions["variant"].eq("legacy_v3"),
        [
            "origin_date", "target_date", "evaluation_split", "current_state",
            "actual", "train_size", "gap",
        ],
    ].sort_values(["origin_date", "target_date"], ignore_index=True)
    for variant in V4_ABLATION_VARIANTS:
        candidate = predictions.loc[
            predictions["variant"].eq(variant), reference.columns
        ].sort_values(["origin_date", "target_date"], ignore_index=True)
        require(candidate.equals(reference),
                f"feature ablation {variant} lacks exact common origins/actuals")

    all_structural = predictions.loc[
        predictions["variant"].eq("all_structural")
    ].sort_values(["origin_date", "target_date"], ignore_index=True)
    main_xgb = main_predictions.loc[
        main_predictions["model"].eq("xgboost")
    ].sort_values(["origin_date", "target_date"], ignore_index=True)
    comparison_fields = [
        "origin_date", "target_date", "evaluation_split", "current_state", "actual",
        "predicted", *PROBABILITY_COLUMNS, "train_size", "gap", "fallback",
        "fallback_reason",
    ]
    require(len(all_structural) == len(main_xgb),
            "all_structural does not reuse every main XGBoost origin")
    for field in comparison_fields:
        if field in PROBABILITY_COLUMNS:
            require(np.allclose(all_structural[field].astype(float),
                                main_xgb[field].astype(float), atol=1e-12),
                    f"all_structural/main XGBoost {field} mismatch")
        else:
            require(all_structural[field].fillna("").astype(str).equals(
                    main_xgb[field].fillna("").astype(str)),
                    f"all_structural/main XGBoost {field} mismatch")

    require(not leaderboard.duplicated(["variant", "evaluation_split"]).any(),
            "duplicate feature ablation leaderboard row")
    require(len(leaderboard) == 2 * len(expected_variants),
            "feature ablation leaderboard row count mismatch")
    published_by_key = leaderboard.set_index(["variant", "evaluation_split"])
    selection_order: list[tuple[str, float, float, float]] = []
    for evaluation_split, expected_role in (
        ("selection", "selection_primary"),
        ("holdout", "post_2023_retrospective_diagnostic"),
    ):
        legacy = predictions.loc[
            predictions["variant"].eq("legacy_v3")
            & predictions["evaluation_split"].eq(evaluation_split)
        ].sort_values(["origin_date", "target_date"], ignore_index=True)
        legacy_log_loss, legacy_brier = _ablation_row_losses(legacy)
        for variant in V4_ABLATION_VARIANTS:
            subset = predictions.loc[
                predictions["variant"].eq(variant)
                & predictions["evaluation_split"].eq(evaluation_split)
            ].sort_values(["origin_date", "target_date"], ignore_index=True)
            metric_input = subset.copy()
            metric_input["model"] = variant
            metrics = probability_metrics(metric_input).loc[variant]
            row = published_by_key.loc[(variant, evaluation_split)]
            require(str(row["role"]) == expected_role,
                    f"feature ablation {variant}/{evaluation_split} role mismatch")
            for field in (
                "log_loss", "brier", "accuracy", "balanced_accuracy", "macro_f1",
                "transition_recall", "calibration_error",
            ):
                require(np.isclose(float(row[field]), float(metrics[field]), atol=1e-10),
                        f"feature ablation {variant}/{evaluation_split} {field} mismatch")
            for field in ("n_predictions", "fallback_count"):
                require(int(row[field]) == int(metrics[field]),
                        f"feature ablation {variant}/{evaluation_split} {field} mismatch")
            candidate_log_loss, candidate_brier = _ablation_row_losses(subset)
            expected_log_delta = float((candidate_log_loss - legacy_log_loss).mean())
            expected_brier_delta = float((candidate_brier - legacy_brier).mean())
            require(int(row["paired_n"]) == len(subset),
                    f"feature ablation {variant}/{evaluation_split} paired_n mismatch")
            require(np.isclose(float(row["paired_log_loss_delta_vs_legacy"]),
                               expected_log_delta, atol=1e-10),
                    f"feature ablation {variant}/{evaluation_split} paired log-loss mismatch")
            require(np.isclose(float(row["paired_brier_delta_vs_legacy"]),
                               expected_brier_delta, atol=1e-10),
                    f"feature ablation {variant}/{evaluation_split} paired Brier mismatch")
            if evaluation_split == "selection" and int(metrics["fallback_count"]) == 0:
                selection_order.append(
                    (
                        variant,
                        float(metrics["log_loss"]),
                        float(metrics["brier"]),
                        float(metrics["calibration_error"]),
                    )
                )
    require(selection_order, "feature ablation has no eligible selection variant")
    selection_order.sort(key=lambda row: (row[1], row[2], row[3], row[0]))
    ranks = {row[0]: index for index, row in enumerate(selection_order, 1)}
    for variant in V4_ABLATION_VARIANTS:
        if variant not in ranks:
            ranks[variant] = len(ranks) + 1
    winner = selection_order[0][0]
    for variant in V4_ABLATION_VARIANTS:
        rows = leaderboard.loc[leaderboard["variant"].eq(variant)]
        require(set(rows["selection_rank"].astype(int)) == {ranks[variant]},
                f"feature ablation {variant} rank used non-selection data")
        require(set(rows["selection_winner"].astype(bool)) == {variant == winner},
                f"feature ablation {variant} winner used non-selection data")

    manifest_by_variant = {
        str(row.get("variant")): row for row in manifest_rows
        if isinstance(row, dict)
    }
    require(set(manifest_by_variant) == expected_variants,
            "feature ablation manifest variant set mismatch")
    group_counts = feature_manifest["group_feature_counts"]
    group_features = feature_manifest["group_features"]
    feature_hashes: set[str] = set()
    for variant, expected_groups in V4_ABLATION_VARIANTS.items():
        row = manifest_by_variant[variant]
        actual_groups = tuple(str(row.get("group_ids", "")).split("|"))
        require(actual_groups == expected_groups,
                f"feature ablation {variant} group IDs mismatch")
        extra_count = int(row.get("extra_feature_count", -1))
        require(extra_count >= 0, f"feature ablation {variant} extra count invalid")
        # The published final-matrix manifest folds pipeline-only controls into
        # legacy_v3, whereas the ablation runner records their count separately.
        expected_count = sum(group_counts[name] for name in expected_groups)
        require(int(row.get("feature_count", -1)) == expected_count,
                f"feature ablation {variant} feature count mismatch")
        columns = row.get("feature_columns")
        require(
            isinstance(columns, list)
            and columns
            and all(isinstance(column, str) and column for column in columns)
            and len(columns) == len(set(columns)),
            f"feature ablation {variant} ordered feature columns are missing/invalid",
        )
        require(
            len(columns) == expected_count
            and set(columns)
            == {
                feature
                for group_id in expected_groups
                for feature in group_features[group_id]
            },
            f"feature ablation {variant} feature membership mismatch",
        )
        feature_hash = str(row.get("feature_sha256", ""))
        require(
            feature_hash == ordered_feature_sha256(columns),
            f"feature ablation {variant} feature hash recomputation mismatch",
        )
        feature_hashes.add(feature_hash)
        require(bool(row.get("reused_main_benchmark"))
                == (variant == "all_structural"),
                f"feature ablation {variant} manifest reuse mismatch")
    require(len(feature_hashes) == len(expected_variants),
            "feature ablation variants do not have distinct feature hashes")

    contract = payload["model"].get("ablation")
    require(isinstance(contract, dict), "payload v4 ablation contract is missing")
    require(contract.get("reference_variant") == "legacy_v3",
            "payload ablation reference variant mismatch")
    require(contract.get("published_variant") == "all_structural",
            "payload ablation published variant mismatch")
    require(contract.get("may_change_published_variant") is False,
            "payload ablation may not change the published variant")
    require(
        contract.get("manifest_sha256") == manifest_hash,
        "payload/artifact ablation manifest hash mismatch",
    )
    return {
        "variants": len(expected_variants),
        "common_origins": len(reference),
        "selection_winner": winner,
        "published_variant": "all_structural",
        "manifest_sha256": manifest_hash,
    }


def audit_structural_forecasts(
    artifacts: Path,
    *,
    historical_predictions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Audit latest joint/ensemble forecasts and their explicit source rows."""

    forecasts = read_csv(
        artifacts / "structural-forecasts.csv", ("origin_date", "target_date")
    )
    require_columns(
        forecasts,
        {
            "origin_date", "target_date", "model", "current_state", "predicted", "fallback",
            "fallback_reason", *PROBABILITY_COLUMNS,
        },
        "structural-forecasts.csv",
    )
    forecasts["fallback"] = boolean_series(
        forecasts["fallback"], "structural forecast fallback"
    )
    require(forecasts["origin_date"].nunique() == 1,
            "structural forecasts must share one latest origin")
    require(forecasts["target_date"].nunique() == 1,
            "structural forecasts must share one latest target")
    require_calendar_horizon(
        forecasts.assign(_horizon=1),
        "origin_date",
        "target_date",
        "_horizon",
        "latest structural forecast",
    )
    models = set(forecasts["model"].astype(str))
    required_models = {
        "markov", "xgboost", "binary_xgboost",
        "xgb_hazard_destination", "causal_dynamic_ensemble",
    }
    require(models == required_models,
            "structural latest forecast model set mismatch")
    require(not forecasts["model"].duplicated().any(),
            "duplicate structural latest forecast model")
    state_values = set(forecasts["current_state"].astype(str))
    require(len(state_values) == 1 and state_values.issubset(STATE_ORDER),
            "structural latest forecasts disagree on current state")
    for model in required_models.difference({"binary_xgboost"}):
        row = forecasts.loc[forecasts["model"].eq(model)].iloc[0]
        probability = row[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
        require(np.isfinite(probability).all()
                and ((probability >= 0.0) & (probability <= 1.0)).all()
                and np.isclose(probability.sum(), 1.0, atol=1e-10),
                f"latest structural {model} probability is invalid")
        require(str(row["predicted"]) == STATE_ORDER[int(np.argmax(probability))],
                f"latest structural {model} predicted state mismatch")
    binary = forecasts.loc[forecasts["model"].eq("binary_xgboost")].iloc[0]
    p_change_field = "p_change" if "p_change" in forecasts else "binary_p_change"
    require(p_change_field in forecasts,
            "latest structural binary hazard source is missing p_change")
    p_change = float(binary[p_change_field])
    expected_joint = compose_joint_probability(
        forecasts.loc[forecasts["model"].eq("xgboost"),
                      list(PROBABILITY_COLUMNS)].iloc[0].to_numpy(dtype=float),
        p_change,
        str(binary["current_state"]),
    )
    joint = forecasts.loc[
        forecasts["model"].eq("xgb_hazard_destination"),
        list(PROBABILITY_COLUMNS),
    ].iloc[0].to_numpy(dtype=float)
    require(np.allclose(joint, expected_joint, atol=1e-12),
            "latest structural joint probability mismatch")
    xgboost_row = forecasts.loc[forecasts["model"].eq("xgboost")].iloc[0]
    joint_row = forecasts.loc[
        forecasts["model"].eq("xgb_hazard_destination")
    ].iloc[0]
    require(
        bool(joint_row["fallback"])
        == bool(xgboost_row["fallback"] or binary["fallback"]),
        "latest structural joint fallback mismatch",
    )

    expert_rows = forecasts.loc[
        forecasts["model"].isin(V4_STRUCTURAL_EXPERTS)
    ].set_index("model")
    ensemble = forecasts.loc[
        forecasts["model"].eq("causal_dynamic_ensemble")
    ].iloc[0]
    weight_columns = {
        "weight_markov": "markov",
        "weight_xgboost": "xgboost",
        "weight_xgb_hazard_destination": "xgb_hazard_destination",
    }
    require(
        set(weight_columns).issubset(forecasts.columns),
        "latest structural ensemble weight columns are missing",
    )
    weights = {
        expert: float(ensemble[column])
        for column, expert in weight_columns.items()
    }
    require(all(np.isfinite(value) and value >= 0.0 for value in weights.values()),
            "latest structural ensemble weights are invalid")
    require(historical_predictions is not None,
            "latest structural ensemble requires historical OOS evidence")
    history = historical_predictions.loc[
        historical_predictions["model"].isin(V4_STRUCTURAL_EXPERTS)
    ].copy()
    current_fallbacks = {
        name: bool(expert_rows.loc[name, "fallback"])
        for name in V4_STRUCTURAL_EXPERTS
    }
    expected_evidence = _discounted_weight_evidence(
        history,
        origin_date=pd.Timestamp(forecasts["origin_date"].iloc[0]),
        current_fallbacks=current_fallbacks,
    )
    for name in V4_STRUCTURAL_EXPERTS:
        require(np.isclose(weights[name],
                           float(expected_evidence[name]["weight"]),
                           atol=1e-12),
                f"latest structural {name} weight mismatch")
    expected_weight_sum = (
        0.0 if all(current_fallbacks.values()) else 1.0
    )
    require(np.isclose(sum(weights.values()), expected_weight_sum, atol=1e-12),
            "latest structural ensemble weight sum mismatch")
    expected_ensemble = (
        np.full(len(STATE_ORDER), 1.0 / len(STATE_ORDER))
        if expected_weight_sum == 0.0
        else sum(
            weights[name]
            * expert_rows.loc[name, list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
            for name in V4_STRUCTURAL_EXPERTS
        )
    )
    expected_ensemble = np.asarray(expected_ensemble, dtype=float)
    expected_ensemble /= expected_ensemble.sum()
    actual_ensemble = ensemble[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    require(np.allclose(actual_ensemble, expected_ensemble, atol=1e-12),
            "latest structural ensemble probability mismatch")
    require(
        bool(ensemble["fallback"]) == (expected_weight_sum == 0.0),
        "latest structural ensemble fallback mismatch",
    )
    return {"origin": forecasts["origin_date"].iloc[0].isoformat(), "models": 5}


def audit_joint_survival_forecasts(artifacts: Path) -> dict[str, Any]:
    forecasts = read_csv(
        artifacts / "joint-survival-forecasts.csv", ("origin_date",)
    )
    require_columns(
        forecasts,
        {
            "origin_date", "horizon_weeks", "cumulative_p_change", "role",
            "one_week_hazard", "step_hazards",
        },
        "joint-survival-forecasts.csv",
    )
    require(set(forecasts["role"].astype(str)) == {"shadow_coherence_benchmark"},
            "joint survival forecasts must remain shadow coherence evidence")
    require(forecasts["origin_date"].nunique() == 1,
            "joint survival rows must share one origin")
    forecasts = forecasts.sort_values("horizon_weeks", ignore_index=True)
    require(tuple(forecasts["horizon_weeks"].astype(int)) == TRANSITION_HORIZONS,
            "joint survival horizons must be exactly 1/4/13")
    cumulative = forecasts["cumulative_p_change"].to_numpy(dtype=float)
    require(np.isfinite(cumulative).all()
            and ((cumulative > 0.0) & (cumulative < 1.0)).all(),
            "joint survival probabilities must be strictly in (0,1)")
    require(np.all(np.diff(cumulative) >= -1e-12),
            "joint survival probabilities are not monotone")

    hazard = forecasts["one_week_hazard"].to_numpy(dtype=float)
    require(np.isfinite(hazard).all()
            and ((hazard > 0.0) & (hazard < 1.0)).all(),
            "joint survival one-week hazard is invalid")
    require(np.allclose(hazard, hazard[0], atol=1e-12),
            "joint survival one-week source changes across horizons")
    structural = read_csv(
        artifacts / "structural-forecasts.csv", ("origin_date", "target_date")
    )
    require_columns(
        structural,
        {"origin_date", "model", "p_change"},
        "structural-forecasts.csv for joint survival audit",
    )
    binary = structural.loc[
        structural["model"].astype(str).eq("binary_xgboost")
    ]
    require(len(binary) == 1, "joint survival binary source row is missing/duplicated")
    binary = binary.iloc[0]
    require(
        pd.Timestamp(binary["origin_date"]) == forecasts["origin_date"].iloc[0],
        "joint survival/binary source origin mismatch",
    )
    require(np.isclose(float(binary["p_change"]), hazard[0], atol=1e-12),
            "joint survival one-week hazard differs from binary source")
    for _, row in forecasts.iterrows():
        raw = row["step_hazards"]
        try:
            hazards = json.loads(str(raw))
        except json.JSONDecodeError as exc:
            raise AuditFailure("joint survival step_hazards is not JSON") from exc
        require(isinstance(hazards, list)
                and len(hazards) == int(row["horizon_weeks"]),
                "joint survival step hazard count mismatch")
        values = np.asarray(hazards, dtype=float)
        require(np.isfinite(values).all()
                and ((values > 0.0) & (values < 1.0)).all(),
                "joint survival step hazard is invalid")
        require(
            np.allclose(values, float(row["one_week_hazard"]), atol=1e-12),
            "joint survival step hazards do not repeat the frozen one-week source",
        )
        expected = 1.0 - float(np.prod(1.0 - values))
        require(np.isclose(float(row["cumulative_p_change"]), expected,
                           atol=1e-12),
                "joint survival product identity mismatch")
    return {
        "origin": forecasts["origin_date"].iloc[0].isoformat(),
        "horizons": list(TRANSITION_HORIZONS),
    }


def choose_legacy_champion(metrics: pd.DataFrame) -> str:
    baselines = [name for name in BASELINES if name in metrics.index]
    require(bool(baselines), "no baseline model available for champion audit")
    baseline_table = metrics.loc[baselines].sort_values(
        ["log_loss", "calibration_error"]
    )
    baseline = baseline_table.iloc[0]
    baseline_name = str(baseline_table.index[0])
    challengers = metrics.loc[
        ~metrics.index.isin(BASELINES)
        & (metrics["fallback_count"] == 0)
        & (metrics["log_loss"] <= float(baseline["log_loss"]) - 0.001)
        & (
            metrics["calibration_error"]
            <= float(baseline["calibration_error"]) + 0.02
        )
    ].copy()
    if challengers.empty:
        return baseline_name
    best_loss = float(challengers["log_loss"].min())
    near_best = challengers.loc[challengers["log_loss"] <= best_loss + 0.01].copy()
    near_best["complexity"] = [COMPLEXITY.get(str(name), 99) for name in near_best.index]
    near_best["model_name"] = near_best.index.astype(str)
    return str(
        near_best.sort_values(
            ["complexity", "calibration_error", "log_loss", "model_name"]
        ).index[0]
    )


def holm_adjusted_pvalues(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (model, pvalue) in enumerate(ordered):
        candidate = min(1.0, pvalue * (len(ordered) - rank))
        running = max(running, candidate)
        adjusted[model] = running
    return adjusted


def paired_losses(predictions: pd.DataFrame) -> dict[str, np.ndarray]:
    positions = {state: index for index, state in enumerate(STATE_ORDER)}
    losses: dict[str, np.ndarray] = {}
    reference_targets: pd.Series | None = None
    reference_actual: pd.Series | None = None
    for model, group in predictions.groupby("model", sort=False):
        ordered = group.sort_values("target_date").reset_index(drop=True)
        targets = ordered["target_date"]
        actual = ordered["actual"].astype(str)
        if reference_targets is None:
            reference_targets = targets
            reference_actual = actual
        else:
            require(targets.equals(reference_targets),
                    f"selection targets differ for {model}")
            require(actual.equals(reference_actual),
                    f"selection actuals differ for {model}")
        probability = ordered[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
        actual_probability = np.asarray(
            [probability[row, positions[state]] for row, state in enumerate(actual)]
        )
        losses[str(model)] = -np.log(actual_probability)
    return losses


def bootstrap_pvalues(improvements: dict[str, np.ndarray]) -> tuple[dict[str, float], int]:
    require(bool(improvements), "selection has no learned challengers")
    lengths = {len(values) for values in improvements.values()}
    require(len(lengths) == 1, "paired selection loss lengths differ")
    observation_count = lengths.pop()
    effective_block = min(BOOTSTRAP_BLOCK_WEEKS, max(1, observation_count // 2))
    blocks_per_sample = int(np.ceil(observation_count / effective_block))
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    starts = generator.integers(
        0, observation_count, size=(BOOTSTRAP_RESAMPLES, blocks_per_sample)
    )
    offsets = np.arange(effective_block)
    indices = (starts[..., np.newaxis] + offsets) % observation_count
    indices = indices.reshape(BOOTSTRAP_RESAMPLES, -1)[:, :observation_count]
    pvalues: dict[str, float] = {}
    for model, differential in improvements.items():
        observed = float(differential.mean())
        null_means = (differential - observed)[indices].mean(axis=1)
        pvalues[model] = float(
            (1 + np.count_nonzero(null_means >= observed))
            / (BOOTSTRAP_RESAMPLES + 1)
        )
    return pvalues, effective_block


def choose_selection_champion(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    minimum_log_loss_improvement: float,
) -> tuple[str, pd.DataFrame]:
    require(
        np.isfinite(minimum_log_loss_improvement)
        and any(
            np.isclose(
                minimum_log_loss_improvement,
                allowed,
                atol=1e-12,
                rtol=0.0,
            )
            for allowed in ALLOWED_LOG_LOSS_IMPROVEMENT_THRESHOLDS
        ),
        "selection minimum log-loss improvement is invalid",
    )
    baselines = [name for name in BASELINES if name in metrics.index]
    require(bool(baselines), "no probability baseline for selection audit")
    baseline_table = metrics.loc[baselines].copy()
    baseline_table["model_name"] = baseline_table.index.astype(str)
    baseline_table = baseline_table.sort_values(
        ["log_loss", "calibration_error", "model_name"]
    )
    reference_model = str(baseline_table.index[0])
    reference = baseline_table.iloc[0]
    losses = paired_losses(predictions)
    challengers = [name for name in metrics.index if name not in BASELINES]
    improvements = {
        model: losses[reference_model] - losses[model] for model in challengers
    }
    raw_pvalues, effective_block = bootstrap_pvalues(improvements)
    adjusted = holm_adjusted_pvalues(raw_pvalues)
    rows: list[dict[str, Any]] = []
    passing: list[str] = []
    for model in metrics.index:
        row = metrics.loc[model]
        improvement = float(reference["log_loss"] - row["log_loss"])
        brier_difference = float(row["brier"] - reference["brier"])
        failures: list[str] = []
        if model not in BASELINES:
            if int(row["fallback_count"]) != 0:
                failures.append("fallback_present")
            if improvement + 1e-12 < minimum_log_loss_improvement:
                failures.append("insufficient_log_loss_improvement")
            if adjusted[model] > SELECTION_ALPHA:
                failures.append("holm_not_significant")
            if brier_difference > BRIER_TOLERANCE + 1e-12:
                failures.append("brier_degradation")
            if not failures:
                passing.append(str(model))
        elif model != reference_model:
            failures.append("non_reference_baseline")
        rows.append(
            {
                "model": str(model),
                "reference_model": reference_model,
                "is_reference": model == reference_model,
                "gate_passed": model == reference_model or not failures,
                "gate_reason": "passed" if not failures else ";".join(failures),
                "log_loss": float(row["log_loss"]),
                "reference_log_loss": float(reference["log_loss"]),
                "absolute_log_loss_improvement": improvement,
                "brier": float(row["brier"]),
                "reference_brier": float(reference["brier"]),
                "brier_difference": brier_difference,
                "fallback_count": int(row["fallback_count"]),
                "raw_p_value": raw_pvalues.get(str(model), np.nan),
                "holm_adjusted_p_value": adjusted.get(str(model), np.nan),
                "n_predictions": int(row["n_predictions"]),
                "bootstrap_block_weeks": BOOTSTRAP_BLOCK_WEEKS,
                "bootstrap_effective_block_weeks": effective_block,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "alpha": SELECTION_ALPHA,
                "minimum_log_loss_improvement": minimum_log_loss_improvement,
                "brier_tolerance": BRIER_TOLERANCE,
            }
        )
    champion = reference_model
    if passing:
        candidates = metrics.loc[passing].copy()
        best_loss = float(candidates["log_loss"].min())
        candidates = candidates.loc[candidates["log_loss"] <= best_loss + 0.01].copy()
        candidates["complexity"] = [COMPLEXITY.get(str(name), 99) for name in candidates.index]
        candidates["model_name"] = candidates.index.astype(str)
        champion = str(
            candidates.sort_values(
                ["complexity", "calibration_error", "log_loss", "model_name"]
            ).index[0]
        )
    diagnostics = pd.DataFrame(rows)
    diagnostics["selected"] = diagnostics["model"].eq(champion)
    return champion, diagnostics


def selection_minimum_log_loss_improvement(
    diagnostics: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    context: str,
) -> float:
    if isinstance(diagnostics, pd.DataFrame):
        require(
            "minimum_log_loss_improvement" in diagnostics.columns,
            f"{context} minimum_log_loss_improvement is missing",
        )
        values = pd.to_numeric(
            diagnostics["minimum_log_loss_improvement"], errors="coerce"
        ).to_numpy(dtype=float)
    else:
        try:
            values = np.asarray(
                [
                    float(row.get("minimum_log_loss_improvement", np.nan))
                    if isinstance(row, Mapping)
                    else np.nan
                    for row in diagnostics
                ],
                dtype=float,
            )
        except (TypeError, ValueError) as exc:
            raise AuditFailure(
                f"{context} minimum_log_loss_improvement is invalid"
            ) from exc
    require(
        len(values) > 0 and np.isfinite(values).all(),
        f"{context} minimum_log_loss_improvement is invalid",
    )
    threshold = float(values[0])
    require(
        np.allclose(values, threshold, atol=1e-12, rtol=0.0)
        and any(
            np.isclose(threshold, allowed, atol=1e-12, rtol=0.0)
            for allowed in ALLOWED_LOG_LOSS_IMPROVEMENT_THRESHOLDS
        ),
        f"{context} minimum_log_loss_improvement is inconsistent",
    )
    return threshold


def validate_probability_object(value: Any, context: str) -> None:
    require(isinstance(value, dict), f"{context} must be an object")
    require(set(value) == set(STATE_ORDER), f"{context} state keys mismatch")
    probability = np.asarray([float(value[state]) for state in STATE_ORDER])
    require(np.isfinite(probability).all(), f"{context} has non-finite values")
    require(((probability >= -1e-9) & (probability <= 1.0 + 1e-9)).all(),
            f"{context} values outside [0,1]")
    require(np.isclose(probability.sum(), 1.0, atol=1e-5),
            f"{context} does not sum to one")


def nullable_close(
    actual: Any,
    expected: float | None,
    context: str,
    *,
    atol: float = 1e-10,
) -> None:
    """Compare a published nullable metric without turning null into zero."""

    if expected is None:
        require(pd.isna(actual), f"{context} must be null when no event is observed")
        return
    require(not pd.isna(actual), f"{context} is unexpectedly null")
    require(np.isclose(float(actual), expected, atol=atol), f"{context} mismatch")


def transition_probability_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Independently recompute the published binary departure metrics."""

    rows: list[dict[str, Any]] = []
    grouped = predictions.groupby(
        ["horizon", "evaluation_split", "model"], sort=False
    )
    for (horizon, split, model), group in grouped:
        actual = group["actual_change"].to_numpy(dtype=bool)
        probability = group["p_change"].to_numpy(dtype=float)
        predicted = group["predicted_change"].to_numpy(dtype=bool)
        require(
            np.isfinite(probability).all()
            and (probability > 0.0).all()
            and (probability < 1.0).all(),
            f"transition probability invalid for {horizon}w/{split}/{model}",
        )
        average_precision: float | None = None
        if actual.any():
            average_precision = float(average_precision_score(actual, probability))
        false_positives = int(np.count_nonzero(predicted & ~actual))
        years = max(float(len(group)) / 52.1775, 1.0 / 52.1775)
        rows.append(
            {
                "horizon": int(horizon),
                "evaluation_split": str(split),
                "model": str(model),
                "log_loss": float(
                    -np.mean(
                        actual * np.log(probability)
                        + (~actual) * np.log(1.0 - probability)
                    )
                ),
                "brier": float(
                    np.mean((probability - actual.astype(float)) ** 2)
                ),
                "average_precision": average_precision,
                "precision": float(
                    np.count_nonzero(predicted & actual)
                    / max(1, np.count_nonzero(predicted))
                ),
                "recall": float(
                    np.count_nonzero(predicted & actual)
                    / max(1, np.count_nonzero(actual))
                ),
                "false_alarms_per_year": float(false_positives / years),
                "n_predictions": int(len(group)),
                "event_count": int(actual.sum()),
                "non_event_count": int((~actual).sum()),
                "fallback_count": int(group["fallback"].astype(bool).sum()),
                "calibration_fallback_count": int(
                    group["calibration_fallback"].astype(bool).sum()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["horizon", "evaluation_split", "log_loss", "brier", "model"],
        ignore_index=True,
    )


def choose_transition_champion(
    predictions: pd.DataFrame,
    candidates: set[str],
) -> str:
    """Rebuild the conservative probability-baseline family gate."""

    table = transition_probability_metrics(predictions)
    baselines = table.loc[
        table["model"].isin(("empirical_hazard", "markov_hazard"))
    ].sort_values(["log_loss", "brier", "model"])
    require(not baselines.empty, "transition selection has no probability baseline")
    baseline = baselines.iloc[0]
    eligible = table.loc[
        ~table["model"].isin(("empirical_hazard", "markov_hazard"))
        & table["model"].isin(candidates)
        & table["fallback_count"].eq(0)
        & (table["log_loss"] <= float(baseline["log_loss"]) - 0.005)
        & (table["brier"] <= float(baseline["brier"]) + 0.005)
    ].copy()
    if eligible.empty:
        return str(baseline["model"])
    complexity = {
        "regularized_logistic": 0,
        "duration_tvtp_hurdle": 1,
        "binary_xgboost": 2,
        "joint_survival_hazard": 3,
    }
    eligible["complexity"] = eligible["model"].map(complexity).fillna(99)
    return str(
        eligible.sort_values(
            ["log_loss", "brier", "complexity", "model"]
        ).iloc[0]["model"]
    )


def transition_threshold(
    history: pd.DataFrame,
    *,
    minimum_rows: int,
) -> tuple[float, str]:
    """Independently rebuild the selection-only balanced-accuracy threshold."""

    if len(history) < minimum_rows:
        return 0.5, f"fallback_0.5:insufficient_rows:{len(history)}<{minimum_rows}"
    actual = history["actual_change"].to_numpy(dtype=bool)
    if not actual.any() or actual.all():
        return 0.5, "fallback_0.5:insufficient_event_classes"
    probability = history["p_change"].to_numpy(dtype=float)
    scored: list[tuple[float, float]] = []
    for threshold in np.linspace(0.05, 0.95, 91):
        predicted = probability >= threshold
        true_positive_rate = float(predicted[actual].mean())
        true_negative_rate = float((~predicted[~actual]).mean())
        scored.append(
            ((true_positive_rate + true_negative_rate) / 2.0, float(threshold))
        )
    best_score = max(score for score, _ in scored)
    tied = [threshold for score, threshold in scored if np.isclose(score, best_score)]
    return (
        float(min(tied, key=lambda value: (abs(value - 0.5), -value))),
        "prequential_balanced_accuracy",
    )


def transition_calibration(
    raw_probability: float,
    history: pd.DataFrame,
    *,
    minimum_rows: int,
) -> tuple[float, str, bool, str]:
    """Independently rebuild the causal Platt calibration contract."""

    raw_value = float(raw_probability)
    require(np.isfinite(raw_value) and 0.0 < raw_value < 1.0,
            "transition raw probability is invalid")
    if len(history) < minimum_rows:
        return (
            raw_value,
            "identity",
            True,
            f"insufficient_prequential_rows:{len(history)}<{minimum_rows}",
        )
    target = history["actual_change"].astype(int)
    if target.nunique() < 2 or int(target.sum()) < 3 or int((1 - target).sum()) < 3:
        return raw_value, "identity", True, "insufficient_event_classes"
    raw_history = np.clip(
        history["raw_p_change"].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6
    )
    design = np.log(raw_history / (1.0 - raw_history)).reshape(-1, 1)
    estimator = LogisticRegression(
        C=0.10,
        solver="lbfgs",
        max_iter=1_000,
        random_state=17,
    )
    estimator.fit(design, target)
    clipped = float(np.clip(raw_value, 1e-6, 1.0 - 1e-6))
    probability = float(
        estimator.predict_proba(
            np.asarray([[np.log(clipped / (1.0 - clipped))]], dtype=float)
        )[0, 1]
    )
    return (
        float(np.clip(probability, 1e-6, 1.0 - 1e-6)),
        "prequential_platt_logit",
        False,
        "",
    )


def effective_transition_fallback(row: pd.Series) -> tuple[bool, str]:
    """Rebuild the conservative dashboard fallback from primitive evidence.

    The artifact keeps fit, calibration, and threshold fallback channels
    separate for diagnosis.  The dashboard combines them because any one of
    the three means the published structural probability is degraded.
    """

    def reason_text(value: Any) -> str:
        return "" if pd.isna(value) else str(value)

    fit_fallback = bool(row.get("fallback", False))
    calibration_fallback = bool(row.get("calibration_fallback", False))
    threshold_method = reason_text(row.get("threshold_method", ""))
    threshold_fallback = threshold_method.startswith("fallback_0.5")
    reasons = (
        reason_text(row.get("fallback_reason", "")),
        (
            f"calibration:{reason_text(row.get('calibration_fallback_reason', ''))}"
            if calibration_fallback
            else ""
        ),
        f"threshold:{threshold_method}" if threshold_fallback else "",
    )
    return (
        bool(fit_fallback or calibration_fallback or threshold_fallback),
        "; ".join(reason for reason in reasons if reason),
    )


def require_calendar_horizon(
    frame: pd.DataFrame,
    start: str,
    end: str,
    horizon_column: str,
    context: str,
) -> None:
    """Validate Friday-to-Friday horizon length without DST false positives."""

    start_dates = frame[start].dt.tz_convert("America/New_York").dt.date
    end_dates = frame[end].dt.tz_convert("America/New_York").dt.date
    actual_days = np.asarray(
        [(end_date - start_date).days for start_date, end_date in zip(start_dates, end_dates, strict=True)]
    )
    expected_days = frame[horizon_column].to_numpy(dtype=int) * 7
    require(
        np.array_equal(actual_days, expected_days),
        f"{context} target_end-origin is not exactly 7*h calendar days",
    )


def _compare_transition_metric_rows(
    recomputed: pd.DataFrame,
    published: pd.DataFrame,
    *,
    context: str,
    atol: float = 1e-10,
) -> None:
    keys = ["horizon", "evaluation_split", "model"]
    require(not recomputed.duplicated(keys).any(), f"{context} recomputed keys duplicate")
    require(not published.duplicated(keys).any(), f"{context} keys duplicate")
    merged = recomputed.merge(
        published,
        on=keys,
        suffixes=("_expected", "_published"),
        validate="one_to_one",
    )
    require(
        len(merged) == len(recomputed) == len(published),
        f"{context} group set differs from transition predictions",
    )
    for _, row in merged.iterrows():
        key = f"{int(row['horizon'])}w/{row['evaluation_split']}/{row['model']}"
        for field in (
            "log_loss",
            "brier",
            "precision",
            "recall",
            "false_alarms_per_year",
        ):
            require(
                np.isclose(
                    float(row[f"{field}_published"]),
                    float(row[f"{field}_expected"]),
                    atol=atol,
                ),
                f"{context} {field} mismatch for {key}",
            )
        expected_ap = row["average_precision_expected"]
        nullable_close(
            row["average_precision_published"],
            None if pd.isna(expected_ap) else float(expected_ap),
            f"{context} average_precision for {key}",
            atol=atol,
        )
        for field in (
            "n_predictions",
            "event_count",
            "non_event_count",
            "fallback_count",
            "calibration_fallback_count",
        ):
            require(
                int(row[f"{field}_published"])
                == int(row[f"{field}_expected"]),
                f"{context} {field} mismatch for {key}",
            )


def audit_transition_candidate_forecasts(
    payload: dict[str, Any],
    artifacts: Path,
    *,
    evaluated_predictions: pd.DataFrame,
    candidate_models: set[str],
    minimum_inner_predictions: int,
) -> dict[str, Any]:
    path = artifacts / "transition-candidate-forecasts.csv"
    forecasts = read_csv(
        path,
        (
            "origin_date", "target_start", "target_end",
            "last_train_origin", "last_train_target_end",
        ),
    )
    require_columns(
        forecasts,
        {
            "origin_date", "target_start", "target_end", "horizon", "model",
            "evaluation_split", "current_state", "actual_change", "raw_p_change",
            "p_change", "threshold", "predicted_change", "train_size",
            "last_train_origin", "last_train_target_end", "gap", "fallback",
            "fallback_reason", "calibration_method", "calibration_fallback",
            "calibration_fallback_reason", "threshold_method", "selection_scope",
            "selection_locked",
        },
        "transition-candidate-forecasts.csv",
    )
    for field in (
        "predicted_change", "fallback", "calibration_fallback", "selection_locked"
    ):
        forecasts[field] = boolean_series(
            forecasts[field], f"transition candidate forecast {field}"
        )
    require(set(forecasts["model"].astype(str)) == candidate_models,
            "transition candidate forecast model set mismatch")
    require(set(forecasts["horizon"].astype(int)) == set(TRANSITION_HORIZONS),
            "transition candidate forecast horizon set mismatch")
    require(set(forecasts["evaluation_split"].astype(str)) == {"prospective"},
            "transition candidate forecasts must be prospective")
    require(forecasts["actual_change"].isna().all(),
            "transition candidate prospective actual must be null")
    require(not forecasts.duplicated(["horizon", "model", "origin_date"]).any(),
            "duplicate transition candidate prospective row")
    expected_rows = len(candidate_models) * sum(TRANSITION_HORIZONS)
    require(len(forecasts) == expected_rows,
            "transition candidate prospective row count mismatch")
    expected_counts = {
        (horizon, model): horizon
        for horizon in TRANSITION_HORIZONS for model in candidate_models
    }
    actual_counts = forecasts.groupby(["horizon", "model"]).size().to_dict()
    require(actual_counts == expected_counts,
            "transition candidate prospective group counts mismatch")
    for horizon in TRANSITION_HORIZONS:
        horizon_rows = forecasts.loc[forecasts["horizon"].astype(int).eq(horizon)]
        reference_keys: pd.DataFrame | None = None
        contract_fields = [
            "origin_date", "target_start", "target_end", "evaluation_split",
            "current_state", "train_size", "last_train_origin",
            "last_train_target_end", "gap",
        ]
        for model_name, model_rows in horizon_rows.groupby("model", sort=False):
            keys = model_rows[contract_fields].sort_values(
                ["origin_date", "target_end"], ignore_index=True
            )
            if reference_keys is None:
                reference_keys = keys
            else:
                require(
                    keys.equals(reference_keys),
                    f"transition candidate {horizon}w/{model_name} lacks strict "
                    "common origins/train contract",
                )
    require_calendar_horizon(
        forecasts, "origin_date", "target_end", "horizon",
        "transition candidate forecasts",
    )
    one_week = forecasts.assign(_horizon=1)
    require_calendar_horizon(
        one_week, "origin_date", "target_start", "_horizon",
        "transition candidate forecasts target_start",
    )
    require((forecasts["gap"].astype(int) == forecasts["horizon"].astype(int)).all(),
            "transition candidate forecast gap must equal horizon")
    require((forecasts["last_train_target_end"] < forecasts["origin_date"]).all(),
            "transition candidate training target reaches origin")
    require((forecasts["last_train_origin"] < forecasts["origin_date"]).all(),
            "transition candidate training origin reaches forecast origin")
    require(set(forecasts["selection_scope"].astype(str)) == {"selection_oos_only"},
            "transition candidate selection scope mismatch")
    require(forecasts["selection_locked"].all(),
            "transition candidate family selection is not locked")
    for field in ("raw_p_change", "p_change"):
        values = forecasts[field].to_numpy(dtype=float)
        require(np.isfinite(values).all()
                and ((values > 0.0) & (values < 1.0)).all(),
                f"transition candidate {field} must be strictly in (0,1)")
    require((forecasts["predicted_change"]
             == (forecasts["p_change"] >= forecasts["threshold"])).all(),
            "transition candidate predicted flag mismatches threshold")
    for _, row in forecasts.iterrows():
        history = evaluated_predictions.loc[
            evaluated_predictions["horizon"].eq(int(row["horizon"]))
            & evaluated_predictions["model"].eq(str(row["model"]))
            & evaluated_predictions["evaluation_split"].eq("selection")
        ]
        (
            expected_probability,
            expected_calibration_method,
            expected_calibration_fallback,
            expected_calibration_reason,
        ) = transition_calibration(
            float(row["raw_p_change"]),
            history,
            minimum_rows=minimum_inner_predictions,
        )
        expected_threshold, expected_method = transition_threshold(
            history, minimum_rows=minimum_inner_predictions
        )
        context = f"candidate {row['horizon']}w/{row['model']}"
        require(np.isclose(float(row["p_change"]), expected_probability,
                           atol=1e-12),
                f"{context} calibration probability mismatch")
        require(str(row["calibration_method"]) == expected_calibration_method,
                f"{context} calibration method mismatch")
        require(bool(row["calibration_fallback"])
                == expected_calibration_fallback,
                f"{context} calibration fallback mismatch")
        actual_reason = "" if pd.isna(row["calibration_fallback_reason"]) else str(
            row["calibration_fallback_reason"]
        )
        require(actual_reason == expected_calibration_reason,
                f"{context} calibration reason mismatch")
        require(np.isclose(float(row["threshold"]), expected_threshold, atol=1e-12),
                f"{context} threshold mismatch")
        require(str(row["threshold_method"]) == expected_method,
                f"{context} threshold method mismatch")
        if str(row["model"]) == "joint_survival_hazard":
            require("one_week_hazard" in forecasts,
                    "joint survival OOS audit lacks one_week_hazard")
            hazard = float(row["one_week_hazard"])
            require(np.isfinite(hazard) and 0.0 < hazard < 1.0,
                    f"{context} one-week hazard is invalid")
            expected_raw = 1.0 - (1.0 - hazard) ** int(row["horizon"])
            require(np.isclose(float(row["raw_p_change"]), expected_raw,
                               atol=1e-12),
                    f"{context} survival raw probability identity mismatch")
    return {"models": len(candidate_models), "rows": len(forecasts)}


def audit_transition_outputs(
    payload: dict[str, Any],
    artifacts: Path,
    *,
    main_predictions: pd.DataFrame,
    main_champion: str,
    main_published_split: str,
) -> dict[str, Any]:
    """Audit every v3 transition artifact and its dashboard projection."""

    minimum_inner_predictions = (
        3 if str(payload["model"].get("profile")) == "quick" else 12
    )

    predictions = read_csv(
        artifacts / "transition-oos-predictions.csv",
        ("origin_date", "target_start", "target_end"),
    )
    leaderboard = read_csv(artifacts / "transition-model-leaderboard.csv")
    splits = read_csv(
        artifacts / "transition-walk-forward-splits.csv",
        (
            "origin_date",
            "target_start",
            "target_end",
            "train_start",
            "last_train_origin",
            "last_train_target_end",
        ),
    )
    nested = read_csv(
        artifacts / "nested-selection.csv", ("origin_date",)
    )
    forecasts = read_csv(
        artifacts / "transition-forecasts.csv",
        (
            "origin_date",
            "target_start",
            "target_end",
            "last_train_origin",
            "last_train_target_end",
        ),
    )
    candidate_status = read_csv(artifacts / "transition-candidate-status.csv")

    require_columns(
        predictions,
        {
            "origin_date", "target_start", "target_end", "horizon", "model",
            "evaluation_split", "current_state", "actual_change", "raw_p_change",
            "p_change", "threshold", "predicted_change", "train_size", "gap",
            "fallback", "fallback_reason", "calibration_method",
            "calibration_fallback", "calibration_fallback_reason",
            "threshold_method",
        },
        "transition-oos-predictions.csv",
    )
    require_columns(
        leaderboard,
        {
            "horizon", "evaluation_split", "model", "selected", "log_loss",
            "brier", "average_precision", "precision", "recall",
            "false_alarms_per_year", "n_predictions", "event_count",
            "non_event_count", "fallback_count", "calibration_fallback_count",
        },
        "transition-model-leaderboard.csv",
    )
    require_columns(
        splits,
        {
            "origin_date", "target_start", "target_end", "horizon",
            "evaluation_split", "train_size", "train_start",
            "last_train_origin", "last_train_target_end",
            "purged_origin_count", "gap",
        },
        "transition-walk-forward-splits.csv",
    )
    require_columns(
        nested,
        {
            "origin_date", "horizon", "evaluation_split", "selected_model",
            "selection_reason", "threshold", "threshold_method",
            "selection_history_origins", "selection_scope", "selection_locked",
        },
        "nested-selection.csv",
    )
    require_columns(
        forecasts,
        {
            "origin_date", "target_start", "target_end", "horizon", "model",
            "evaluation_split", "current_state", "actual_change", "raw_p_change",
            "p_change", "threshold", "predicted_change", "train_size",
            "last_train_origin", "last_train_target_end", "gap", "fallback",
            "fallback_reason", "calibration_method", "calibration_fallback",
            "calibration_fallback_reason", "threshold_method", "selection_scope",
            "selection_locked",
        },
        "transition-forecasts.csv",
    )
    candidate_status_columns = {
        "model", "requested", "available", "published", "reason"
    }
    uses_structural_v4_contract = payload.get("meta", {}).get(
        "result_version"
    ) in {V4_RESULT_VERSION, V5_RESULT_VERSION}
    if uses_structural_v4_contract:
        candidate_status_columns.update({"selection_eligible", "role"})
    require_columns(
        candidate_status,
        candidate_status_columns,
        "transition-candidate-status.csv",
    )

    for frame, fields, prefix in (
        (
            predictions,
            ("actual_change", "predicted_change", "fallback", "calibration_fallback"),
            "transition OOS",
        ),
        (leaderboard, ("selected",), "transition leaderboard"),
        (nested, ("selection_locked",), "nested selection"),
        (
            forecasts,
            ("predicted_change", "fallback", "calibration_fallback", "selection_locked"),
            "transition forecast",
        ),
        (
            candidate_status,
            (
                "requested",
                "available",
                "published",
                *(
                    ("selection_eligible",)
                    if uses_structural_v4_contract
                    else ()
                ),
            ),
            "transition candidate status",
        ),
    ):
        for field in fields:
            frame[field] = boolean_series(frame[field], f"{prefix} {field}")

    horizons = set(predictions["horizon"].astype(int))
    require(horizons == set(TRANSITION_HORIZONS), "transition OOS horizons must be 1/4/13")
    require(
        set(splits["horizon"].astype(int)) == horizons
        and set(nested["horizon"].astype(int)) == horizons
        and set(forecasts["horizon"].astype(int)) == horizons,
        "transition artifact horizon sets differ",
    )
    split_values = set(predictions["evaluation_split"].astype(str))
    require(
        split_values == {"selection", "retrospective_diagnostic"},
        f"transition OOS split values invalid: {sorted(split_values)}",
    )
    require(
        set(splits["evaluation_split"].astype(str)) == split_values
        and set(nested["evaluation_split"].astype(str)) == split_values,
        "transition split labels differ across artifacts",
    )
    require(
        set(forecasts["evaluation_split"].astype(str)) == {"prospective"},
        "transition forecasts must be prospective",
    )
    require(forecasts["actual_change"].isna().all(), "prospective actual_change must be null")
    require(not predictions["actual_change"].isna().any(), "evaluated actual_change is null")
    require(
        set(predictions["current_state"].astype(str)).issubset(set(STATE_ORDER))
        and set(forecasts["current_state"].astype(str)).issubset(set(STATE_ORDER)),
        "transition artifacts contain an invalid current state",
    )

    candidate_status["model"] = candidate_status["model"].astype(str)
    require(not candidate_status["model"].duplicated().any(), "candidate status model duplicates")
    requested_models = set(candidate_status.loc[candidate_status["requested"], "model"])
    published_models = set(candidate_status.loc[candidate_status["published"], "model"])
    selection_eligible_models = (
        set(candidate_status.loc[candidate_status["selection_eligible"], "model"])
        if uses_structural_v4_contract
        else published_models
    )
    prediction_models = set(predictions["model"].astype(str))
    require(
        TRANSITION_REQUIRED_MODELS.issubset(requested_models)
        and requested_models.issubset(TRANSITION_ALLOWED_MODELS),
        "transition requested candidate set is invalid",
    )
    if uses_structural_v4_contract:
        require(requested_models == TRANSITION_ALLOWED_MODELS,
                "v4 transition candidate set must contain exactly six models")
        require(
            selection_eligible_models
            == TRANSITION_ALLOWED_MODELS.difference({"joint_survival_hazard"}),
            "v4 transition selection-eligible set mismatch",
        )
        shadow_status = candidate_status.loc[
            candidate_status["model"].eq("joint_survival_hazard")
        ]
        require(
            len(shadow_status) == 1
            and str(shadow_status.iloc[0]["role"])
            == "shadow_coherence_benchmark"
            and not bool(shadow_status.iloc[0]["selection_eligible"]),
            "joint survival candidate must remain a non-selectable shadow",
        )
        require(
            set(candidate_status.loc[
                ~candidate_status["model"].eq("joint_survival_hazard"), "role"
            ].astype(str)) == {"candidate"},
            "selectable transition candidate role mismatch",
        )
    require(published_models == prediction_models, "candidate/prediction model sets differ")
    require(
        candidate_status.loc[candidate_status["published"], "available"].all(),
        "an unavailable transition candidate was published",
    )
    require(
        set(leaderboard["model"].astype(str)) == prediction_models,
        "transition leaderboard/prediction model sets differ",
    )

    require(
        not predictions.duplicated(["horizon", "model", "origin_date"]).any(),
        "duplicate transition OOS horizon/model/origin",
    )
    require(not splits.duplicated(["horizon", "origin_date"]).any(), "duplicate transition split")
    require(not nested.duplicated(["horizon", "origin_date"]).any(), "duplicate nested selection")
    require(not forecasts.duplicated(["horizon", "origin_date"]).any(), "duplicate prospective forecast")
    require_calendar_horizon(predictions, "origin_date", "target_end", "horizon", "transition OOS")
    require_calendar_horizon(splits, "origin_date", "target_end", "horizon", "transition splits")
    require_calendar_horizon(forecasts, "origin_date", "target_end", "horizon", "transition forecasts")
    for frame, context in ((predictions, "transition OOS"), (splits, "transition splits"), (forecasts, "transition forecasts")):
        one_week = frame.assign(_horizon=1)
        require_calendar_horizon(one_week, "origin_date", "target_start", "_horizon", f"{context} target_start")
        require((frame["gap"].astype(int) == frame["horizon"].astype(int)).all(), f"{context} gap must equal horizon")
    require(
        (splits["purged_origin_count"].astype(int) == splits["horizon"].astype(int)).all(),
        "transition split purged count must equal horizon",
    )
    require((splits["last_train_target_end"] < splits["origin_date"]).all(), "transition split training target reaches origin")
    require((forecasts["last_train_target_end"] < forecasts["origin_date"]).all(), "prospective training target reaches origin")
    require(
        (splits["last_train_origin"] < splits["origin_date"]).all()
        and (forecasts["last_train_origin"] < forecasts["origin_date"]).all(),
        "transition last training origin reaches forecast origin",
    )
    transition_selection_end = payload["model"].get("transition_selection_end")
    require(
        isinstance(transition_selection_end, str)
        and len(transition_selection_end) == 10,
        "payload transition_selection_end must be an ISO date",
    )
    cutoff = pd.to_datetime(transition_selection_end, utc=True, errors="raise")
    require(
        cutoff.date().isoformat() == transition_selection_end,
        "payload transition_selection_end must be a canonical ISO date",
    )
    require(
        (predictions.loc[predictions["evaluation_split"].eq("selection"), "target_end"] < cutoff).all(),
        "transition selection target_end reaches 2023 cutoff",
    )
    require(
        (predictions.loc[predictions["evaluation_split"].eq("retrospective_diagnostic"), "origin_date"] >= cutoff).all(),
        "transition retrospective diagnostic origin precedes cutoff",
    )
    for horizon in sorted(predictions["horizon"].astype(int).unique()):
        horizon_rows = predictions.loc[
            predictions["horizon"].astype(int).eq(horizon)
        ]
        selection_rows = horizon_rows.loc[
            horizon_rows["evaluation_split"].eq("selection")
        ]
        diagnostic_rows = horizon_rows.loc[
            horizon_rows["evaluation_split"].eq("retrospective_diagnostic")
        ]
        require(
            selection_rows["target_end"].max()
            < diagnostic_rows["target_start"].min(),
            f"transition horizon-{horizon} selection/diagnostic event windows overlap",
        )

    origin_contract_fields = [
        "horizon", "origin_date", "target_start", "target_end",
        "evaluation_split", "current_state", "actual_change", "train_size", "gap",
    ]
    origin_contract = predictions[origin_contract_fields].drop_duplicates()
    require(
        not origin_contract.duplicated(["horizon", "origin_date"]).any(),
        "transition models disagree on an origin target/split/label/train contract",
    )
    reference_transition_keys = origin_contract[
        ["horizon", "origin_date", "target_end"]
    ].sort_values(["horizon", "origin_date", "target_end"], ignore_index=True)
    for model_name, model_rows in predictions.groupby("model", sort=False):
        keys = model_rows[["horizon", "origin_date", "target_end"]].sort_values(
            ["horizon", "origin_date", "target_end"], ignore_index=True
        )
        require(
            keys.equals(reference_transition_keys),
            f"transition model {model_name} lacks strict common OOS origins",
        )
    merged_splits = origin_contract.merge(
        splits[["horizon", "origin_date", "target_start", "target_end", "evaluation_split", "train_size", "gap"]],
        on=["horizon", "origin_date"],
        suffixes=("_prediction", "_split"),
        validate="one_to_one",
    )
    require(len(merged_splits) == len(origin_contract) == len(splits), "transition prediction/split origin sets differ")
    for field in ("target_start", "target_end", "evaluation_split", "train_size", "gap"):
        require((merged_splits[f"{field}_prediction"] == merged_splits[f"{field}_split"]).all(), f"transition prediction/split {field} mismatch")
    merged_nested = origin_contract[["horizon", "origin_date", "evaluation_split"]].merge(
        nested[["horizon", "origin_date", "evaluation_split"]],
        on=["horizon", "origin_date"],
        suffixes=("_prediction", "_nested"),
        validate="one_to_one",
    )
    require(len(merged_nested) == len(origin_contract) == len(nested), "transition prediction/nested origin sets differ")
    require((merged_nested["evaluation_split_prediction"] == merged_nested["evaluation_split_nested"]).all(), "nested selection split mismatch")

    for frame, context in ((predictions, "transition OOS"), (forecasts, "transition forecast")):
        for field in ("raw_p_change", "p_change"):
            values = frame[field].to_numpy(dtype=float)
            require(np.isfinite(values).all() and (values > 0.0).all() and (values < 1.0).all(), f"{context} {field} must be strictly in (0,1)")
        threshold = frame["threshold"].to_numpy(dtype=float)
        require(np.isfinite(threshold).all() and (threshold >= 0.0).all() and (threshold <= 1.0).all(), f"{context} threshold outside [0,1]")
        require((frame["predicted_change"] == (frame["p_change"] >= frame["threshold"])).all(), f"{context} predicted_change does not match threshold")
        fallback = frame.loc[frame["fallback"]]
        require(fallback["fallback_reason"].fillna("").astype(str).str.len().gt(0).all(), f"{context} fallback reason missing")
        calibration_fallback = frame.loc[frame["calibration_fallback"]]
        require(calibration_fallback["calibration_fallback_reason"].fillna("").astype(str).str.len().gt(0).all(), f"{context} calibration fallback reason missing")

    if uses_structural_v4_contract:
        require("one_week_hazard" in predictions,
                "v4 transition OOS lacks joint one_week_hazard evidence")
        survival_rows = predictions.loc[
            predictions["model"].astype(str).eq("joint_survival_hazard")
        ]
        require(not survival_rows.empty, "v4 transition OOS lacks joint survival rows")
        hazards = pd.to_numeric(
            survival_rows["one_week_hazard"], errors="raise"
        ).to_numpy(dtype=float)
        require(np.isfinite(hazards).all()
                and ((hazards > 0.0) & (hazards < 1.0)).all(),
                "joint survival OOS one-week hazard is invalid")
        expected_raw = 1.0 - (1.0 - hazards) ** survival_rows["horizon"].to_numpy(
            dtype=int
        )
        require(np.allclose(
            survival_rows["raw_p_change"].to_numpy(dtype=float),
            expected_raw,
            atol=1e-12,
        ), "joint survival OOS raw probability identity mismatch")

    for _, row in predictions.iterrows():
        history = predictions.loc[
            predictions["horizon"].eq(int(row["horizon"]))
            & predictions["model"].eq(str(row["model"]))
            & predictions["evaluation_split"].eq("selection")
            & (predictions["target_end"] < pd.Timestamp(row["origin_date"]))
        ]
        (
            expected_probability,
            expected_calibration_method,
            expected_calibration_fallback,
            expected_calibration_reason,
        ) = transition_calibration(
            float(row["raw_p_change"]),
            history,
            minimum_rows=minimum_inner_predictions,
        )
        calibration_context = (
            f"{row['horizon']}w/{row['model']}/{row['origin_date']}"
        )
        require(np.isclose(float(row["p_change"]), expected_probability,
                           atol=1e-12),
                f"prequential calibration probability mismatch at {calibration_context}")
        require(str(row["calibration_method"]) == expected_calibration_method,
                f"prequential calibration method mismatch at {calibration_context}")
        require(bool(row["calibration_fallback"])
                == expected_calibration_fallback,
                f"prequential calibration fallback mismatch at {calibration_context}")
        actual_reason = "" if pd.isna(row["calibration_fallback_reason"]) else str(
            row["calibration_fallback_reason"]
        )
        require(actual_reason == expected_calibration_reason,
                f"prequential calibration reason mismatch at {calibration_context}")
        expected_threshold, expected_method = transition_threshold(
            history,
            minimum_rows=minimum_inner_predictions,
        )
        require(np.isclose(float(row["threshold"]), expected_threshold, atol=1e-12), f"prequential threshold mismatch at {row['horizon']}w/{row['model']}/{row['origin_date']}")
        require(str(row["threshold_method"]) == expected_method, f"prequential threshold method mismatch at {row['horizon']}w/{row['model']}/{row['origin_date']}")

    evaluated_keys = set(zip(predictions["horizon"].astype(int), predictions["origin_date"], strict=True))
    prospective_keys = set(zip(forecasts["horizon"].astype(int), forecasts["origin_date"], strict=True))
    require(evaluated_keys.isdisjoint(prospective_keys), "evaluated and prospective transition origins overlap")
    forecast_counts = forecasts.groupby("horizon")["origin_date"].nunique().to_dict()
    require(forecast_counts == {1: 1, 4: 4, 13: 13}, "prospective horizon forecast counts must be exactly h")

    recomputed = transition_probability_metrics(predictions)
    _compare_transition_metric_rows(recomputed, leaderboard, context="transition leaderboard")

    champions: dict[int, str] = {}
    for horizon in TRANSITION_HORIZONS:
        selection_predictions = predictions.loc[
            predictions["horizon"].eq(horizon)
            & predictions["evaluation_split"].eq("selection")
        ]
        champion = choose_transition_champion(
            selection_predictions, selection_eligible_models
        )
        champions[horizon] = champion
        horizon_rows = leaderboard.loc[leaderboard["horizon"].eq(horizon)]
        for split, group in horizon_rows.groupby("evaluation_split"):
            require(int(group["selected"].sum()) == 1, f"transition leaderboard must select one {horizon}w/{split} model")
            require(str(group.loc[group["selected"], "model"].iloc[0]) == champion, f"transition selected model mismatch for {horizon}w/{split}")

        forecast_history = predictions.loc[
            predictions["horizon"].eq(horizon)
            & predictions["model"].eq(champion)
            & predictions["evaluation_split"].eq("selection")
        ]
        expected_forecast_threshold, expected_forecast_method = transition_threshold(
            forecast_history,
            minimum_rows=minimum_inner_predictions,
        )
        horizon_forecasts = forecasts.loc[forecasts["horizon"].eq(horizon)]
        require(set(horizon_forecasts["model"].astype(str)) == {champion}, f"prospective model differs from {horizon}w champion")
        require(np.allclose(horizon_forecasts["threshold"].astype(float), expected_forecast_threshold, atol=1e-12), f"prospective threshold mismatch for {horizon}w")
        require(set(horizon_forecasts["threshold_method"].astype(str)) == {expected_forecast_method}, f"prospective threshold method mismatch for {horizon}w")
        require(set(horizon_forecasts["selection_scope"].astype(str)) == {"selection_oos_only"}, f"prospective selection scope mismatch for {horizon}w")

    for _, row in nested.iterrows():
        horizon = int(row["horizon"])
        origin = pd.Timestamp(row["origin_date"])
        history = predictions.loc[
            predictions["horizon"].eq(horizon)
            & predictions["evaluation_split"].eq("selection")
            & (predictions["target_end"] < origin)
        ]
        history_origins = int(history["origin_date"].nunique())
        require(int(row["selection_history_origins"]) == history_origins, f"nested selection history count mismatch at {horizon}w/{origin}")
        require(str(row["selection_scope"]) == "earlier_selection_oos_only", "nested selection scope is not earlier selection OOS only")
        require(bool(row["selection_locked"]) == (str(row["evaluation_split"]) == "retrospective_diagnostic"), "nested selection lock mismatch")
        expected_nested = (
            "markov_hazard"
            if history_origins < minimum_inner_predictions
            and "markov_hazard" in prediction_models
            else choose_transition_champion(history, selection_eligible_models)
        )
        require(str(row["selected_model"]) == expected_nested, f"nested selected model mismatch at {horizon}w/{origin}")
        selected_prediction = predictions.loc[
            predictions["horizon"].eq(horizon)
            & predictions["origin_date"].eq(origin)
            & predictions["model"].eq(expected_nested)
        ]
        require(len(selected_prediction) == 1, f"nested selected prediction missing at {horizon}w/{origin}")
        selected_prediction = selected_prediction.iloc[0]
        require(np.isclose(float(row["threshold"]), float(selected_prediction["threshold"]), atol=1e-12), f"nested threshold mismatch at {horizon}w/{origin}")
        require(str(row["threshold_method"]) == str(selected_prediction["threshold_method"]), f"nested threshold method mismatch at {horizon}w/{origin}")

    model_contract = payload["model"]
    payload_champions = model_contract.get("transition_champions")
    require(isinstance(payload_champions, dict), "payload transition_champions missing")
    require(payload_champions == {f"{horizon}w": model for horizon, model in champions.items()}, "payload transition champions mismatch")
    payload_leaderboard = model_contract.get("transition_leaderboard")
    require(isinstance(payload_leaderboard, list) and payload_leaderboard, "payload transition leaderboard missing")
    payload_table = pd.DataFrame(payload_leaderboard).rename(columns={"horizon_weeks": "horizon", "binary_log_loss": "log_loss"})
    _compare_transition_metric_rows(
        recomputed,
        payload_table,
        context="payload transition leaderboard",
        atol=1e-7,
    )
    for published_row in payload_leaderboard:
        match = leaderboard.loc[
            leaderboard["horizon"].eq(int(published_row["horizon_weeks"]))
            & leaderboard["evaluation_split"].eq(str(published_row["evaluation_split"]))
            & leaderboard["model"].eq(str(published_row["model"]))
        ]
        require(len(match) == 1 and bool(published_row.get("selected")) == bool(match.iloc[0]["selected"]), "payload transition selected flag mismatch")
    payload_status = model_contract.get("transition_candidate_status")
    require(isinstance(payload_status, list) and payload_status, "payload transition candidate status missing")
    payload_status_by_model = {str(row.get("model")): row for row in payload_status}
    require(set(payload_status_by_model) == set(candidate_status["model"]), "payload/CSV transition candidate model sets differ")
    for _, row in candidate_status.iterrows():
        item = payload_status_by_model[str(row["model"])]
        for field in ("requested", "available", "published"):
            require(bool(item.get(field)) == bool(row[field]), f"payload transition candidate {field} mismatch for {row['model']}")
        if uses_structural_v4_contract:
            require(
                bool(item.get("selection_eligible"))
                == bool(row["selection_eligible"]),
                f"payload transition candidate selection_eligible mismatch for {row['model']}",
            )
            require(
                str(item.get("role")) == str(row["role"]),
                f"payload transition candidate role mismatch for {row['model']}",
            )
        require(str(item.get("reason", "")) == ("" if pd.isna(row["reason"]) else str(row["reason"])), f"payload transition candidate reason mismatch for {row['model']}")

    source_rows = pd.concat(
        [
            predictions.loc[predictions["evaluation_split"].eq("retrospective_diagnostic")],
            forecasts,
        ],
        ignore_index=True,
    )
    source_rows = source_rows.loc[
        [str(model) == champions[int(horizon)] for horizon, model in zip(source_rows["horizon"], source_rows["model"], strict=True)]
    ]
    require(not source_rows.duplicated(["horizon", "origin_date"]).any(), "published transition source duplicate")
    source_by_key = {
        (int(row["horizon"]), pd.Timestamp(row["origin_date"]).date().isoformat()): row
        for _, row in source_rows.iterrows()
    }
    weekly = payload["weekly"]
    main_published = main_predictions.loc[
        main_predictions["evaluation_split"].eq(main_published_split)
        & main_predictions["model"].eq(main_champion)
    ]
    main_by_origin = {
        pd.Timestamp(row["origin_date"]).date().isoformat(): row
        for _, row in main_published.iterrows()
    }
    for index, week in enumerate(weekly):
        origin = str(week["date"])
        risk = week.get("transition_risk")
        require(isinstance(risk, dict) and set(risk) == {"1w", "4w", "13w"}, f"weekly[{index}] transition_risk horizons mismatch")
        current_state = str(week["current"]["state"])
        next_probability = float(week["next_week"]["probabilities"][current_state])
        canonical = 1.0 - next_probability
        require(np.isclose(float(week["transition_probability"]), canonical, atol=1e-7), f"weekly[{index}] canonical transition alias mismatch")
        require(np.isclose(float(risk["1w"]["probability"]), canonical, atol=1e-7), f"weekly[{index}] 1w transition alias mismatch")
        require(str(risk["1w"]["target_end"]) == str(week["next_week"]["date"]), f"weekly[{index}] 1w target mismatch")
        require(str(risk["1w"]["model"]) == main_champion, f"weekly[{index}] 1w model is not main champion")
        require(np.isclose(float(risk["1w"]["threshold"]), 0.5, atol=1e-12), f"weekly[{index}] authoritative 1w threshold must be 0.5")
        require(bool(risk["1w"]["fallback"]) == bool(week["next_week"]["fallback"]), f"weekly[{index}] authoritative 1w fallback mismatch")
        term_structure = week.get("transition_term_structure")
        raw_probabilities: Mapping[str, object] | None = None
        if term_structure is not None:
            require(
                isinstance(term_structure, dict)
                and set(term_structure)
                == {
                    "semantics",
                    "coherence_method",
                    "one_week_anchor",
                    "raw_probabilities",
                    "adjusted",
                },
                f"weekly[{index}] transition term-structure fields mismatch",
            )
            require(
                term_structure["semantics"]
                == "cumulative_first_departure_probability"
                and term_structure["coherence_method"]
                == "one_week_anchored_l2_isotonic_projection_v1"
                and term_structure["one_week_anchor"]
                == "official_multiclass_departure_probability",
                f"weekly[{index}] transition term-structure identity mismatch",
            )
            raw_value = term_structure.get("raw_probabilities")
            require(
                isinstance(raw_value, dict),
                f"weekly[{index}] transition raw probabilities missing",
            )
            raw_probabilities = raw_value
            projected = anchored_transition_projection(raw_probabilities)
            for key in ("1w", "4w", "13w"):
                require(
                    np.isclose(
                        float(risk[key]["probability"]),
                        projected[key],
                        atol=1e-7,
                    ),
                    f"weekly[{index}] {key} anchored projection mismatch",
                )
            expected_adjusted = any(
                not np.isclose(
                    projected[key],
                    float(raw_probabilities[key]),
                    atol=1e-12,
                )
                for key in ("1w", "4w", "13w")
            )
            require(
                term_structure.get("adjusted") is expected_adjusted,
                f"weekly[{index}] transition adjusted flag mismatch",
            )
            require(
                np.isclose(float(raw_probabilities["1w"]), canonical, atol=1e-7),
                f"weekly[{index}] raw 1w probability mismatch",
            )
        if origin in main_by_origin:
            main_row = main_by_origin[origin]
            require(np.isclose(canonical, 1.0 - float(main_row[f"p_{main_row['current_state']}"]), atol=1e-7), f"weekly[{index}] 1w probability does not map to main OOS champion")
        for horizon in (4, 13):
            source = source_by_key.get((horizon, origin))
            require(source is not None, f"weekly[{index}] has no selected transition source for {horizon}w")
            published_risk = risk[f"{horizon}w"]
            require(np.isclose(float(published_risk["threshold"]), float(source["threshold"]), atol=1e-7), f"weekly[{index}] {horizon}w threshold/source mismatch")
            require(str(published_risk["target_end"]) == pd.Timestamp(source["target_end"]).date().isoformat(), f"weekly[{index}] {horizon}w target/source mismatch")
            require(published_risk["model"] == source["model"], f"weekly[{index}] {horizon}w model/source mismatch")
            source_probability = (
                float(raw_probabilities[f"{horizon}w"])
                if raw_probabilities is not None
                else float(published_risk["probability"])
            )
            require(np.isclose(source_probability, float(source["p_change"]), atol=1e-7), f"weekly[{index}] {horizon}w probability/source mismatch")
            effective_fallback, effective_reason = effective_transition_fallback(source)
            require(bool(published_risk["fallback"]) == effective_fallback, f"weekly[{index}] {horizon}w effective fallback/source mismatch")
            require(str(published_risk.get("fallback_reason", "")) == effective_reason, f"weekly[{index}] {horizon}w effective fallback reason/source mismatch")

    candidate_forecast_summary = None
    if uses_structural_v4_contract:
        candidate_forecast_summary = audit_transition_candidate_forecasts(
            payload,
            artifacts,
            evaluated_predictions=predictions,
            candidate_models=prediction_models,
            minimum_inner_predictions=minimum_inner_predictions,
        )

    return {
        "horizons": list(TRANSITION_HORIZONS),
        "models": len(prediction_models),
        "champions": {f"{horizon}w": model for horizon, model in champions.items()},
        "evaluated_rows": len(predictions),
        "prospective_rows": len(forecasts),
        "candidate_forecasts": candidate_forecast_summary,
    }


def _v5_artifact_path(
    artifacts: Path,
    relative_name: object,
    *,
    context: str,
) -> Path:
    name = str(relative_name)
    relative = Path(name)
    require(
        bool(name)
        and not relative.is_absolute()
        and len(relative.parts) == 1
        and relative.name == name
        and name not in {".", ".."},
        f"{context} path is unsafe",
    )
    path = artifacts / relative
    require(
        path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
        f"missing/empty/non-regular {context}: {path}",
    )
    return path


def _audit_v5_file_contracts(
    contracts: object,
    artifacts: Path,
    *,
    context: str,
) -> dict[str, pd.DataFrame]:
    require(isinstance(contracts, dict), f"{context} must be an object")
    frames: dict[str, pd.DataFrame] = {}
    for key, raw in contracts.items():
        row_context = f"{context}.{key}"
        require(isinstance(raw, dict), f"{row_context} must be an object")
        require(
            set(raw) == {"path", "row_count", "sha256"},
            f"{row_context} fields mismatch",
        )
        name = str(raw["path"])
        require(name not in frames, f"{context} duplicates {name}")
        path = _v5_artifact_path(artifacts, name, context=row_context)
        require(
            file_sha256(path) == str(raw["sha256"]),
            f"{row_context} SHA-256 mismatch",
        )
        frame = pd.read_csv(path)
        require(
            not frame.empty or int(raw["row_count"]) == 0,
            f"{row_context} CSV is empty",
        )
        require(
            len(frame) == int(raw["row_count"]),
            f"{row_context} row_count mismatch",
        )
        frames[name] = frame
    return frames


def _calendar_days(series: pd.Series, *, context: str) -> pd.Series:
    text = series.astype("string").str.strip()
    date_only = text.str.fullmatch(r"\d{4}-\d{2}-\d{2}").fillna(False)
    normalized = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    if date_only.any():
        normalized.loc[date_only] = pd.to_datetime(
            text.loc[date_only],
            format="%Y-%m-%d",
            errors="raise",
        ).dt.normalize()

    timestamped = ~date_only
    if timestamped.any():
        parsed = pd.to_datetime(series.loc[timestamped], utc=True, errors="raise")
        normalized.loc[timestamped] = (
            parsed.dt.tz_convert("America/New_York")
            .dt.tz_localize(None)
            .dt.normalize()
        )

    require(normalized.notna().all(), f"{context} contains a missing date")
    return normalized


def _market_date(value: object) -> str:
    text = str(value)
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        return text
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        return parsed.date().isoformat()
    return parsed.tz_convert("America/New_York").date().isoformat()


def _audit_v5_probability_rows(
    frame: pd.DataFrame,
    *,
    context: str,
    include_actual: bool,
) -> None:
    probability_columns = (
        "p_no_departure",
        "p_risk_on",
        "p_transition",
        "p_risk_off",
    )
    required = {
        "horizon_weeks",
        "origin_date",
        "target_end",
        "model",
        "current_state",
        *probability_columns,
    }
    if include_actual:
        required.update({"actual_outcome", "actual_change"})
    require_columns(frame, required, context)
    numeric = frame.loc[:, probability_columns].apply(
        pd.to_numeric, errors="raise"
    )
    require(
        np.isfinite(numeric.to_numpy(dtype=float)).all(),
        f"{context} contains non-finite probabilities",
    )
    require(
        ((numeric >= 0.0) & (numeric <= 1.0)).all().all(),
        f"{context} probabilities are outside [0, 1]",
    )
    require(
        np.allclose(numeric.sum(axis=1), 1.0, atol=1e-10, rtol=0.0),
        f"{context} probabilities do not sum to one",
    )
    require(
        frame["current_state"].astype(str).isin(STATE_ORDER).all(),
        f"{context} current state is invalid",
    )
    for state in STATE_ORDER:
        mask = frame["current_state"].astype(str).eq(state)
        require(
            np.allclose(
                numeric.loc[mask, f"p_{state}"], 0.0, atol=1e-12, rtol=0.0
            ),
            f"{context} assigns first-departure mass to the origin state",
        )
    horizons = pd.to_numeric(frame["horizon_weeks"], errors="raise").astype(int)
    require(
        set(horizons) == {1, 4, 13},
        f"{context} horizons must be exactly 1/4/13",
    )
    origin = _calendar_days(frame["origin_date"], context=f"{context}.origin_date")
    target = _calendar_days(frame["target_end"], context=f"{context}.target_end")
    require(
        ((target - origin).dt.days == 7 * horizons).all(),
        f"{context} target_end is not exactly 7*h calendar days after origin",
    )
    if include_actual:
        outcomes = frame["actual_outcome"].astype(str)
        require(
            outcomes.isin(("no_departure", *STATE_ORDER)).all(),
            f"{context} actual outcome is invalid",
        )
        require(
            (
                outcomes.eq("no_departure")
                | ~outcomes.eq(frame["current_state"].astype(str))
            ).all(),
            f"{context} actual first departure equals the origin state",
        )
        actual_change = boolean_series(
            frame["actual_change"], f"{context}.actual_change"
        )
        require(
            actual_change.eq(~outcomes.eq("no_departure")).all(),
            f"{context} actual_change disagrees with actual_outcome",
        )


def _audit_v5_embedded_records(
    records: object,
    frame: pd.DataFrame,
    *,
    keys: tuple[str, ...],
    context: str,
) -> None:
    require(isinstance(records, list), f"{context} payload rows must be an array")
    require(len(records) == len(frame), f"{context} row count mismatch")
    require_columns(frame, set(keys), context)
    require(not frame.duplicated(list(keys)).any(), f"{context} CSV keys duplicate")
    lookup = {
        tuple(str(row[key]) for key in keys): row
        for _, row in frame.iterrows()
    }
    embedded_identities: list[tuple[str, ...]] = []
    for position, expected in enumerate(records):
        require(
            isinstance(expected, dict),
            f"{context} payload row {position} must be an object",
        )
        require_columns(
            frame,
            set(expected),
            f"{context} payload row {position}",
        )
        require(
            set(keys).issubset(expected),
            f"{context} payload row {position} keys are incomplete",
        )
        identity = tuple(str(expected[key]) for key in keys)
        embedded_identities.append(identity)
        require(identity in lookup, f"{context} payload/CSV keys differ")
        actual = lookup[identity]
        for field, expected_value in expected.items():
            actual_value = actual[field]
            actual_missing = bool(pd.isna(actual_value))
            if expected_value is None:
                require(
                    actual_missing,
                    f"{context} {identity} {field} nullability mismatch",
                )
            elif isinstance(expected_value, bool):
                require(
                    not actual_missing
                    and str(actual_value).strip().lower()
                    == str(expected_value).lower(),
                    f"{context} {identity} {field} mismatch",
                )
            elif isinstance(expected_value, (int, float)):
                require(
                    not actual_missing
                    and np.isclose(
                        float(actual_value),
                        float(expected_value),
                        atol=5e-8,
                        rtol=0.0,
                    ),
                    f"{context} {identity} {field} mismatch",
                )
            else:
                require(
                    not actual_missing and str(actual_value) == str(expected_value),
                    f"{context} {identity} {field} mismatch",
                )
    require(
        len(set(embedded_identities)) == len(embedded_identities),
        f"{context} payload keys duplicate",
    )
    require(
        set(embedded_identities) == set(lookup),
        f"{context} payload/CSV keys differ",
    )


def _v5_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _v5_boolean(value: object, *, context: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    require(normalized in {"true", "false"}, f"{context} is not boolean")
    return normalized == "true"


def _require_v5_recomputed_value(
    expected: object,
    actual: object,
    *,
    context: str,
    tolerance: float = 5e-8,
) -> None:
    expected_missing = _v5_missing(expected)
    actual_missing = _v5_missing(actual)
    if expected_missing:
        require(actual_missing, f"{context} nullability mismatch")
        return
    require(not actual_missing, f"{context} nullability mismatch")
    if isinstance(expected, (bool, np.bool_)):
        require(
            _v5_boolean(actual, context=context) == bool(expected),
            f"{context} mismatch",
        )
    elif isinstance(expected, (int, np.integer)) and not isinstance(expected, bool):
        try:
            resolved = int(actual)
        except (TypeError, ValueError) as exc:
            raise AuditFailure(f"{context} is not an integer") from exc
        require(float(actual) == float(resolved), f"{context} is not an integer")
        require(resolved == int(expected), f"{context} mismatch")
    elif isinstance(expected, (float, np.floating)):
        try:
            resolved = float(actual)
        except (TypeError, ValueError) as exc:
            raise AuditFailure(f"{context} is not numeric") from exc
        require(
            np.isclose(resolved, float(expected), atol=tolerance, rtol=0.0),
            f"{context} mismatch",
        )
    else:
        require(str(actual) == str(expected), f"{context} mismatch")


def _audit_v5_recomputed_records(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    *,
    keys: tuple[str, ...],
    fields: tuple[str, ...],
    context: str,
) -> None:
    require_columns(expected, {*keys, *fields}, f"recomputed {context}")
    require_columns(actual, {*keys, *fields}, context)
    require(
        not expected.duplicated(list(keys)).any(),
        f"recomputed {context} keys duplicate",
    )
    require(not actual.duplicated(list(keys)).any(), f"{context} keys duplicate")
    expected_lookup = {
        tuple(str(row[key]) for key in keys): row
        for _, row in expected.iterrows()
    }
    actual_lookup = {
        tuple(str(row[key]) for key in keys): row
        for _, row in actual.iterrows()
    }
    require(
        set(expected_lookup) == set(actual_lookup),
        f"{context} keys differ from independent recomputation",
    )
    for identity, expected_row in expected_lookup.items():
        actual_row = actual_lookup[identity]
        for field in fields:
            _require_v5_recomputed_value(
                expected_row[field],
                actual_row[field],
                context=f"{context} {identity} {field}",
            )


def _audit_v5_core_model(
    payload: Mapping[str, Any],
    artifacts: Path,
) -> dict[str, Any]:
    model = payload["model"]
    profile = str(model["profile"])
    champion = str(model["champion"])
    is_v5 = (
        str(payload.get("meta", {}).get("result_version"))
        == V5_RESULT_VERSION
    )
    core_frames: dict[str, pd.DataFrame] | None = None
    if is_v5:
        from regime_lab.v5_artifacts import V5_CORE_ARTIFACT_PATHS

        core_frames = _audit_v5_file_contracts(
            model.get("core_artifacts"),
            artifacts,
            context="payload.model.core_artifacts",
        )
        expected_core_paths = [path for _, path in V5_CORE_ARTIFACT_PATHS]
        require(
            list(core_frames) == expected_core_paths,
            "v5 core artifact manifest path/order mismatch",
        )
        predictions = core_frames["oos-predictions.csv"].copy()
        splits = core_frames["walk-forward-splits.csv"].copy()
        leaderboard = core_frames["model-leaderboard.csv"].copy()
        diagnostics = core_frames["selection-diagnostics.csv"].copy()
        for frame, columns in (
            (predictions, ("origin_date", "target_date")),
            (
                splits,
                (
                    "origin_date",
                    "target_date",
                    "train_start",
                    "last_train_origin",
                    "last_train_target",
                    "first_purged_origin",
                ),
            ),
        ):
            for column in columns:
                require_columns(frame, {column}, "v5 core artifact")
                frame[column] = pd.to_datetime(
                    frame[column],
                    utc=True,
                    errors="raise",
                )
    else:
        predictions = read_csv(
            artifacts / "oos-predictions.csv",
            ("origin_date", "target_date"),
        )
        splits = read_csv(
            artifacts / "walk-forward-splits.csv",
            (
                "origin_date",
                "target_date",
                "train_start",
                "last_train_origin",
                "last_train_target",
                "first_purged_origin",
            ),
        )
        leaderboard = read_csv(artifacts / "model-leaderboard.csv")
        diagnostics = read_csv(artifacts / "selection-diagnostics.csv")

    manifest_path = artifacts / "candidate-manifest.json"
    require(
        manifest_path.is_file()
        and not manifest_path.is_symlink()
        and manifest_path.stat().st_size > 0,
        "v5 core candidate manifest is missing/non-regular",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(isinstance(manifest, dict), "v5 core candidate manifest must be an object")
    manifest_hash = str(manifest.get("sha256", ""))
    manifest_body = dict(manifest)
    manifest_body.pop("sha256", None)
    require(
        manifest_hash == canonical_json_sha256(manifest_body),
        "v5 core candidate manifest SHA-256 mismatch",
    )
    require(
        model.get("candidate_manifest_sha256") == manifest_hash
        and model.get("candidate_manifest") == manifest_body,
        "v5 core payload/candidate manifest mismatch",
    )
    require(
        str(manifest_body.get("profile")) == profile,
        "v5 core candidate manifest profile mismatch",
    )
    manifest_models = [
        str(row.get("name"))
        for row in manifest_body.get("models", [])
        if isinstance(row, Mapping)
    ]
    require(
        manifest_models and len(manifest_models) == len(set(manifest_models)),
        "v5 core candidate manifest model names are invalid",
    )
    historical_roster = (
        OPERATING_CONTRACT.historical_reviewed_roster_by_manifest_sha256(
            manifest_hash
        )
        if is_v5
        else None
    )
    if historical_roster is not None:
        expected_models = {
            str(name) for name in historical_roster["candidate_models"]
        }
    elif is_v5:
        expected_models = (
            V5_FULL_MODELS if profile == "full" else V5_STANDARD_MODELS
        )
    else:
        expected_models = (
            V4_FULL_MODELS if profile == "full" else V4_STANDARD_MODELS
        )
    require(
        set(manifest_models) == expected_models,
        "v5 core candidate manifest differs from the permitted V5 suite",
    )

    required_prediction_columns = {
        "origin_date",
        "target_date",
        "model",
        "evaluation_split",
        "current_state",
        "actual",
        "predicted",
        "train_size",
        "gap",
        "fallback",
        "fallback_reason",
        *PROBABILITY_COLUMNS,
    }
    require_columns(
        predictions,
        required_prediction_columns,
        "v5 core oos-predictions.csv",
    )
    require_columns(
        splits,
        {
            "origin_date",
            "target_date",
            "evaluation_split",
            "train_size",
            "last_train_target",
            "first_purged_origin",
            "purged_origin_count",
            "gap",
        },
        "v5 core walk-forward-splits.csv",
    )
    predictions["fallback"] = boolean_series(
        predictions["fallback"], "v5 core fallback"
    )
    leaderboard["selected"] = boolean_series(
        leaderboard["selected"], "v5 core leaderboard selected"
    )
    probability = predictions.loc[:, PROBABILITY_COLUMNS].to_numpy(dtype=float)
    require(
        np.isfinite(probability).all()
        and ((probability >= 0.0) & (probability <= 1.0)).all()
        and np.allclose(probability.sum(axis=1), 1.0, atol=1e-10, rtol=0.0),
        "v5 core OOS probabilities are invalid",
    )
    expected_prediction = np.asarray(STATE_ORDER, dtype=object)[
        probability.argmax(axis=1)
    ]
    require(
        predictions["predicted"].astype(str).to_numpy().tolist()
        == expected_prediction.tolist(),
        "v5 core predicted state disagrees with ordered argmax",
    )
    require(
        not predictions.duplicated(["model", "origin_date"]).any(),
        "v5 core OOS model/origin keys duplicate",
    )
    require(
        set(predictions["model"].astype(str)) == set(manifest_models),
        "v5 core candidate manifest/prediction model sets differ",
    )
    require(
        set(predictions["evaluation_split"].astype(str))
        == {"selection", "holdout"},
        "v5 core OOS split values are invalid",
    )
    require_calendar_horizon(
        predictions.assign(_horizon=1),
        "origin_date",
        "target_date",
        "_horizon",
        "v5 core OOS",
    )
    require(
        predictions["gap"].astype(int).eq(1).all(),
        "v5 core OOS gap must be one week",
    )

    origin_fields = (
        "origin_date",
        "target_date",
        "evaluation_split",
        "current_state",
        "actual",
        "train_size",
        "gap",
    )
    origins = predictions.loc[:, origin_fields].drop_duplicates()
    require(
        not origins.duplicated("origin_date").any(),
        "v5 core models disagree on an OOS origin contract",
    )
    reference_keys = origins.loc[:, ["origin_date", "target_date"]].sort_values(
        ["origin_date", "target_date"], ignore_index=True
    )
    for name, rows in predictions.groupby("model", sort=False):
        keys = rows.loc[:, ["origin_date", "target_date"]].sort_values(
            ["origin_date", "target_date"], ignore_index=True
        )
        require(
            keys.equals(reference_keys),
            f"v5 core model {name} lacks strict common OOS origins",
        )
    require(not splits.duplicated("origin_date").any(), "v5 core split origins duplicate")
    merged = origins.merge(
        splits.loc[
            :,
            [
                "origin_date",
                "target_date",
                "evaluation_split",
                "train_size",
                "gap",
            ],
        ],
        on="origin_date",
        suffixes=("_prediction", "_split"),
        validate="one_to_one",
    )
    require(
        len(merged) == len(origins) == len(splits),
        "v5 core prediction/split origin sets differ",
    )
    for field in ("target_date", "evaluation_split", "train_size", "gap"):
        require(
            merged[f"{field}_prediction"].astype(str).equals(
                merged[f"{field}_split"].astype(str)
            ),
            f"v5 core prediction/split {field} mismatch",
        )
    require(
        splits["gap"].astype(int).eq(1).all()
        and splits["purged_origin_count"].astype(int).eq(1).all()
        and (splits["last_train_target"] < splits["origin_date"]).all()
        and (splits["first_purged_origin"] < splits["origin_date"]).all(),
        "v5 core split purge contract is invalid",
    )

    selection_end = pd.to_datetime(str(model["selection_end"]), utc=True)
    selection = predictions.loc[
        predictions["evaluation_split"].astype(str).eq("selection")
    ]
    holdout = predictions.loc[
        predictions["evaluation_split"].astype(str).eq("holdout")
    ]
    minimum_origins = 3 if profile == "quick" else 12
    require(
        selection["origin_date"].nunique() >= minimum_origins
        and holdout["origin_date"].nunique() >= minimum_origins,
        "v5 core OOS split has insufficient origins",
    )
    require(
        (selection["target_date"] < selection_end).all()
        and (holdout["target_date"] >= selection_end).all(),
        "v5 core selection boundary is invalid",
    )
    if profile in {"standard", "full"}:
        require(
            selection["origin_date"].nunique() >= 300,
            "v5 core standard/full omits the full pre-2023 selection era",
        )

    selection_metrics = probability_metrics(selection)
    holdout_metrics = probability_metrics(holdout)
    selection_threshold = selection_minimum_log_loss_improvement(
        diagnostics,
        context="v5 core selection diagnostics",
    )
    expected_champion, expected_diagnostics = choose_selection_champion(
        selection_metrics,
        selection,
        minimum_log_loss_improvement=selection_threshold,
    )
    require(
        expected_champion == champion,
        "v5 core payload champion disagrees with independent selection",
    )
    require(
        set(leaderboard["model"].astype(str)) == set(manifest_models)
        and int(leaderboard["selected"].sum()) == 1
        and str(leaderboard.loc[leaderboard["selected"], "model"].iloc[0])
        == champion,
        "v5 core leaderboard champion mismatch",
    )
    leaderboard_index = leaderboard.set_index(leaderboard["model"].astype(str))
    metric_fields = (
        "log_loss",
        "brier",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "transition_recall",
        "calibration_error",
        "n_predictions",
        "fallback_count",
    )
    for name in manifest_models:
        csv_row = leaderboard_index.loc[name]
        for field in metric_fields:
            _require_v5_recomputed_value(
                holdout_metrics.loc[name, field],
                csv_row[field],
                context=f"v5 core leaderboard {name} {field}",
                tolerance=1e-10,
            )
            _require_v5_recomputed_value(
                selection_metrics.loc[name, field],
                csv_row[f"selection_{field}"],
                context=f"v5 core leaderboard {name} selection_{field}",
                tolerance=1e-10,
            )
    payload_leaderboard = model["leaderboard"]
    payload_frame = leaderboard.rename(columns={"model": "name"})
    require(
        [int(row.get("rank", -1)) for row in payload_leaderboard]
        == list(range(1, len(payload_leaderboard) + 1)),
        "v5 core embedded leaderboard ranks are invalid",
    )
    require(
        all(
            bool(row.get("is_champion"))
            == (str(row.get("name")) == champion)
            for row in payload_leaderboard
        ),
        "v5 core embedded leaderboard champion flags are invalid",
    )
    common_payload_leaderboard = [
        {
            key: value
            for key, value in row.items()
            if key in payload_frame.columns
        }
        for row in payload_leaderboard
    ]
    _audit_v5_embedded_records(
        common_payload_leaderboard,
        payload_frame,
        keys=("name",),
        context="v5 core embedded leaderboard",
    )

    diagnostic_fields = tuple(
        field
        for field in expected_diagnostics.columns
        if field != "model"
    )
    _audit_v5_recomputed_records(
        expected_diagnostics,
        diagnostics,
        keys=("model",),
        fields=diagnostic_fields,
        context="v5 core selection diagnostics",
    )
    _audit_v5_embedded_records(
        model["selection_diagnostics"],
        diagnostics,
        keys=("model",),
        context="v5 core embedded selection diagnostics",
    )

    if str(payload["meta"]["mode"]) == "live":
        diagnostic = model["holdout_diagnostic"]
        ranked = holdout_metrics.reset_index().sort_values(
            ["log_loss", "model"], ignore_index=True
        )
        champion_row = ranked.loc[ranked["model"].astype(str).eq(champion)].iloc[0]
        best_row = ranked.iloc[0]
        regret = max(
            0.0,
            float(champion_row["log_loss"]) - float(best_row["log_loss"]),
        )
        expected_status = "weak_generalization" if regret > 0.05 else "ok"
        require(
            diagnostic.get("status") == expected_status
            and diagnostic.get("applicable") is True
            and diagnostic.get("selection_locked") is True
            and diagnostic.get("champion_model") == champion
            and diagnostic.get("best_model") == str(best_row["model"]),
            "v5 core holdout diagnostic identity mismatch",
        )
        for field, expected in (
            ("champion_log_loss", float(champion_row["log_loss"])),
            ("best_log_loss", float(best_row["log_loss"])),
            ("absolute_regret", regret),
        ):
            _require_v5_recomputed_value(
                expected,
                diagnostic[field],
                context=f"v5 core holdout diagnostic {field}",
            )
    stacking = audit_stacking_weights(
        predictions,
        artifacts,
        allow_v5_multiscale_rows=True,
    )
    multiscale = None
    if is_v5:
        assert core_frames is not None
        structural_contract = model.get("structural_models", {}).get(
            V5_MULTISCALE_MODEL
        )
        require(
            isinstance(structural_contract, Mapping)
            and dict(structural_contract.get("sidecar", {}))
            == dict(model["core_artifacts"]["multiscale_ensemble_scales"]),
            "v5 multiscale structural/core sidecar contract mismatch",
        )
        multiscale = audit_v5_multiscale_ensemble(
            predictions,
            core_frames["stacking-weights.csv"],
            core_frames["multiscale-ensemble-scales.csv"],
            artifacts,
        )
    return {
        "profile": profile,
        "champion": champion,
        "models": len(manifest_models),
        "selection_origins": int(selection["origin_date"].nunique()),
        "holdout_origins": int(holdout["origin_date"].nunique()),
        "fallback_rows": int(predictions["fallback"].sum()),
        "stacking": stacking,
        "multiscale": multiscale,
        "core_artifacts": (
            list(core_frames) if core_frames is not None else None
        ),
    }


def _audit_v5_model_forecasts(
    payload: Mapping[str, Any],
    artifacts: Path,
) -> dict[str, Any]:
    """Bind every embedded model comparison forecast to its source CSV row."""

    model = payload.get("model", {})
    require(isinstance(model, Mapping), "v5 model forecast model metadata is invalid")
    weekly = payload.get("weekly")
    require(
        isinstance(weekly, list) and bool(weekly),
        "v5 model forecast weekly payload is missing",
    )
    comparison = model.get("forecast_comparison")
    if comparison is None:
        raise AuditFailure("v5 model forecast comparison metadata is required")

    context = "v5 model forecast comparison"
    require(isinstance(comparison, Mapping), f"{context} metadata is invalid")
    require(
        set(comparison) == {"role", "horizon_weeks", "models"},
        f"{context} metadata fields mismatch",
    )
    require(
        comparison.get("role") == "research_comparison"
        and comparison.get("horizon_weeks") == 1,
        f"{context} role/horizon mismatch",
    )
    models = comparison.get("models")
    historical_roster = (
        OPERATING_CONTRACT.historical_reviewed_roster_by_manifest_sha256(
            str(model.get("candidate_manifest_sha256", ""))
        )
    )
    comparison_models = (
        tuple(
            str(name)
            for name in historical_roster["forecast_comparison_models"]
        )
        if historical_roster is not None
        else V5_FORECAST_COMPARISON_MODELS
    )
    champion = str(model.get("champion", ""))
    if champion and champion not in comparison_models:
        comparison_models = (*comparison_models, champion)
    require(
        isinstance(models, list)
        and tuple(str(name) for name in models) == comparison_models,
        f"{context} model order mismatch",
    )
    leaderboard = model.get("leaderboard")
    require(isinstance(leaderboard, list), f"{context} leaderboard is invalid")
    leaderboard_names = {
        str(row.get("name"))
        for row in leaderboard
        if isinstance(row, Mapping)
    }
    require(
        set(comparison_models).issubset(leaderboard_names),
        f"{context} models are missing from the leaderboard",
    )

    core_contracts = model.get("core_artifacts")
    require(isinstance(core_contracts, Mapping), f"{context} core artifacts missing")
    oos_contract = core_contracts.get("oos_predictions")
    require(isinstance(oos_contract, Mapping), f"{context} OOS contract missing")
    require(
        set(oos_contract) == {"path", "row_count", "sha256"}
        and oos_contract.get("path") == "oos-predictions.csv",
        f"{context} OOS contract fields/path mismatch",
    )
    oos_path = _v5_artifact_path(
        artifacts,
        oos_contract["path"],
        context=f"{context} historical source",
    )
    require(
        file_sha256(oos_path) == str(oos_contract["sha256"]),
        f"{context} OOS source SHA-256 mismatch",
    )
    historical = read_csv(oos_path, ("origin_date", "target_date"))
    require(
        len(historical) == int(oos_contract["row_count"]),
        f"{context} OOS source row_count mismatch",
    )
    latest = read_csv(
        _v5_artifact_path(
            artifacts,
            "structural-forecasts.csv",
            context=f"{context} latest source",
        ),
        ("origin_date", "target_date"),
    )

    required_source_columns = {
        "origin_date",
        "target_date",
        "model",
        "predicted",
        "fallback",
        "fallback_reason",
        *PROBABILITY_COLUMNS,
    }
    for source_name, source in (
        ("historical OOS", historical),
        ("latest structural", latest),
    ):
        require_columns(source, required_source_columns, f"{context} {source_name}")
        source["fallback"] = boolean_series(
            source["fallback"], f"{context} {source_name} fallback"
        )
        source["_origin_day"] = source["origin_date"].map(_market_date)
        source["_target_day"] = source["target_date"].map(_market_date)
        selected = source.loc[
            source["model"].astype(str).isin(comparison_models)
        ]
        require(
            not selected.duplicated(["_origin_day", "model"]).any(),
            f"{context} {source_name} origin/model keys duplicate",
        )

    weekly_dates = [
        _market_date(week.get("date"))
        for week in weekly
        if isinstance(week, Mapping)
    ]
    require(
        len(weekly_dates) == len(weekly) and len(set(weekly_dates)) == len(weekly),
        f"{context} weekly dates are invalid/duplicate",
    )

    historical_rows = 0
    latest_rows = 0
    for position, week in enumerate(weekly):
        require(isinstance(week, Mapping), f"{context} weekly[{position}] is invalid")
        origin_day = _market_date(week["date"])
        source_name = (
            "latest structural"
            if position == len(weekly) - 1
            else "historical OOS"
        )
        source = latest if position == len(weekly) - 1 else historical
        source_rows = source.loc[
            source["_origin_day"].eq(origin_day)
            & source["model"].astype(str).isin(comparison_models)
        ].copy()
        require(
            len(source_rows) == len(comparison_models)
            and set(source_rows["model"].astype(str))
            == set(comparison_models),
            f"{context} weekly[{position}] {source_name} rows are incomplete",
        )
        source_lookup = source_rows.set_index(source_rows["model"].astype(str))
        published_rows = week.get("model_forecasts")
        require(
            isinstance(published_rows, list)
            and len(published_rows) == len(comparison_models),
            f"{context} weekly[{position}] payload rows are incomplete",
        )
        for model_position, name in enumerate(comparison_models):
            row_context = f"{context} weekly[{position}] {name}"
            published = published_rows[model_position]
            require(isinstance(published, Mapping), f"{row_context} is invalid")
            require(
                published.get("model") == name,
                f"{row_context} model/order mismatch",
            )
            require(
                published.get("method") == V5_FORECAST_COMPARISON_METHOD,
                f"{row_context} method mismatch",
            )
            source_row = source_lookup.loc[name]
            require(
                str(published.get("state")) == str(source_row["predicted"]),
                f"{row_context} predicted state/source mismatch",
            )
            require(
                _market_date(published.get("date"))
                == str(source_row["_target_day"]),
                f"{row_context} target date/source mismatch",
            )
            probabilities = published.get("probabilities")
            require(
                isinstance(probabilities, Mapping)
                and set(probabilities) == set(STATE_ORDER),
                f"{row_context} probabilities are invalid",
            )
            for state in STATE_ORDER:
                require(
                    np.isclose(
                        float(probabilities[state]),
                        float(source_row[f"p_{state}"]),
                        atol=5e-8,
                        rtol=0.0,
                    ),
                    f"{row_context} {state} probability/source mismatch",
                )
            require(
                _v5_boolean(
                    published.get("fallback"),
                    context=f"{row_context} payload fallback",
                )
                == bool(source_row["fallback"]),
                f"{row_context} fallback/source mismatch",
            )
            expected_reason = (
                ""
                if _v5_missing(source_row["fallback_reason"])
                else str(source_row["fallback_reason"])
            )
            require(
                str(published.get("fallback_reason", "")) == expected_reason,
                f"{row_context} fallback reason/source mismatch",
            )
        if position == len(weekly) - 1:
            latest_rows += len(published_rows)
        else:
            historical_rows += len(published_rows)

    return {
        "status": "verified",
        "models": len(comparison_models),
        "weeks": len(weekly),
        "historical_rows": historical_rows,
        "latest_rows": latest_rows,
    }


def _audit_v5_feature_quality(
    payload: Mapping[str, Any],
    artifacts: Path,
    *,
    expected_features: Sequence[str],
) -> dict[str, Any]:
    model = payload.get("model", {})
    require(isinstance(model, Mapping), "v5 feature quality model is invalid")
    manifest = model.get("feature_quality_artifact")
    if manifest is None:
        return {"status": "legacy_absent"}
    require(
        isinstance(manifest, dict),
        "v5 feature quality manifest is invalid",
    )
    try:
        document = verify_feature_quality_artifact(manifest, artifacts)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise AssertionError(f"v5 feature quality artifact failed: {exc}") from exc
    rows = document.get("features")
    require(isinstance(rows, list), "v5 feature quality rows are invalid")
    names = [
        str(row.get("feature"))
        for row in rows
        if isinstance(row, Mapping)
    ]
    require(
        len(names) == len(rows)
        and len(names) == len(set(names))
        and set(names) == {str(name) for name in expected_features},
        "v5 feature quality rows do not match the model feature manifest",
    )
    return {
        "status": str(document["status"]),
        "feature_count": int(document["feature_count"]),
        "warning_feature_count": int(document["warning_feature_count"]),
        "unavailable_feature_count": int(document["unavailable_feature_count"]),
        "sha256": str(manifest["sha256"]),
    }


def _audit_v5_execution_parameters(payload: Mapping[str, Any]) -> dict[str, Any]:
    context = "v5 execution parameters"
    raw = payload.get("model", {}).get("execution_parameters")
    require(isinstance(raw, Mapping), f"{context} must be an object")
    parameters = dict(raw)
    expected_fields = {
        "profile",
        "directional_minimum_selection_predictions",
        "directional_minimum_diagnostic_predictions",
        "directional_maximum_selection_origins",
        "directional_maximum_diagnostic_origins",
        "duration_bootstrap_resamples",
        "conditional_outcome_bootstrap_resamples",
        "preregistered_bootstrap_resamples",
        "preregistration_overrides",
        "sha256",
    }
    require(set(parameters) == expected_fields, f"{context} fields mismatch")
    profile = str(parameters["profile"])
    require(profile in {"quick", "standard", "full"}, f"{context} profile invalid")
    model_profile = payload.get("model", {}).get("profile")
    require(model_profile is not None, f"{context} model profile missing")
    require(str(model_profile) == profile, f"{context} profile/model mismatch")
    expected_minimum = 3 if profile == "quick" else 12
    for field in (
        "directional_minimum_selection_predictions",
        "directional_minimum_diagnostic_predictions",
    ):
        require(
            int(parameters[field]) == expected_minimum,
            f"{context} {field} mismatch",
        )
    expected_maximum = {"quick": 3, "standard": 60, "full": None}[profile]
    for field in (
        "directional_maximum_selection_origins",
        "directional_maximum_diagnostic_origins",
    ):
        value = parameters[field]
        require(
            (value is None and expected_maximum is None)
            or (value is not None and int(value) == expected_maximum),
            f"{context} {field} mismatch",
        )
    preregistered = int(parameters["preregistered_bootstrap_resamples"])
    duration_resamples = int(parameters["duration_bootstrap_resamples"])
    outcome_resamples = int(parameters["conditional_outcome_bootstrap_resamples"])
    require(preregistered == 1_999, f"{context} preregistered resamples mismatch")
    require(duration_resamples >= 1, f"{context} duration resamples invalid")
    require(outcome_resamples >= 1, f"{context} outcome resamples invalid")
    expected_overrides: list[str] = []
    if duration_resamples != preregistered:
        expected_overrides.append("duration.bootstrap_resamples")
    if outcome_resamples != preregistered:
        expected_overrides.append(
            "conditional_asset_statistics.bootstrap_resamples"
        )
    require(
        list(parameters["preregistration_overrides"]) == expected_overrides,
        f"{context} override linkage mismatch",
    )
    unhashed = {key: parameters[key] for key in parameters if key != "sha256"}
    require(
        str(parameters["sha256"]) == canonical_json_sha256(unhashed),
        f"{context} SHA-256 mismatch",
    )
    return parameters


def _v5_directional_conditional_matrix(rows: pd.DataFrame) -> np.ndarray:
    matrix = rows[[f"p_{state}" for state in STATE_ORDER]].to_numpy(dtype=float)
    current_states = rows["current_state"].astype(str).to_numpy()
    positions = {state: index for index, state in enumerate(STATE_ORDER)}
    result = np.zeros_like(matrix)
    for row_index, current_state in enumerate(current_states):
        vector = np.maximum(matrix[row_index], 0.0)
        vector[positions[current_state]] = 0.0
        total = float(vector.sum())
        if not np.isfinite(total) or total <= 0.0:
            alternatives = [state for state in STATE_ORDER if state != current_state]
            vector = np.asarray(
                [
                    1.0 / len(alternatives) if state in alternatives else 0.0
                    for state in STATE_ORDER
                ],
                dtype=float,
            )
        else:
            vector = vector / total
        result[row_index] = vector
    return result


def _v5_directional_event_support(rows: pd.DataFrame) -> dict[str, int]:
    ordered = rows.sort_values("origin_date", kind="mergesort").reset_index(drop=True)
    events = boolean_series(
        ordered["actual_change"], "v5 directional actual_change"
    ).to_numpy(dtype=bool)
    destinations = ordered.loc[events, "actual_outcome"].astype(str)
    event_blocks = {
        int(position) // BOOTSTRAP_BLOCK_WEEKS for position in np.flatnonzero(events)
    }
    return {
        "event_count": int(events.sum()),
        "destination_class_count": int(destinations.nunique()),
        "effective_event_blocks": int(len(event_blocks)),
    }


def _v5_directional_conditional_losses(rows: pd.DataFrame) -> np.ndarray:
    ordered = rows.sort_values("origin_date", kind="mergesort").reset_index(drop=True)
    events = boolean_series(
        ordered["actual_change"], "v5 directional actual_change"
    ).to_numpy(dtype=bool)
    losses = np.full(len(ordered), np.nan, dtype=float)
    if not events.any():
        return losses
    matrix = _v5_directional_conditional_matrix(ordered)
    positions = {state: index for index, state in enumerate(STATE_ORDER)}
    actual = ordered["actual_outcome"].astype(str).to_numpy()
    event_positions = np.flatnonzero(events)
    actual_probability = np.asarray(
        [matrix[row, positions[actual[row]]] for row in event_positions],
        dtype=float,
    )
    losses[event_positions] = -np.log(
        np.clip(actual_probability, 1e-8, 1.0)
    )
    return losses


def _v5_directional_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (horizon, split, model), group in predictions.groupby(
        ["horizon_weeks", "evaluation_split", "model"], sort=False
    ):
        support = _v5_directional_event_support(group)
        events = boolean_series(
            group["actual_change"], "v5 directional actual_change"
        )
        scored = group.loc[events].copy()
        if scored.empty:
            log_loss = float("nan")
            brier = float("nan")
        else:
            matrix = _v5_directional_conditional_matrix(scored)
            actual = scored["actual_outcome"].astype(str).to_numpy()
            positions = {state: index for index, state in enumerate(STATE_ORDER)}
            actual_probability = np.asarray(
                [matrix[row, positions[state]] for row, state in enumerate(actual)]
            )
            one_hot = np.zeros_like(matrix)
            one_hot[
                np.arange(len(scored)),
                [positions[state] for state in actual],
            ] = 1.0
            log_loss = float(
                -np.log(np.clip(actual_probability, 1e-8, 1.0)).mean()
            )
            brier = float(np.square(matrix - one_hot).sum(axis=1).mean())
        rows.append(
            {
                "horizon_weeks": int(horizon),
                "evaluation_split": str(split),
                "model": str(model),
                "score_target": V5_DIRECTIONAL_SCORE_TARGET,
                "log_loss": log_loss,
                "brier": brier,
                "n_predictions": int(len(group)),
                **support,
                "fallback_count": int(
                    boolean_series(
                        group["fallback"], "v5 directional fallback"
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["horizon_weeks", "evaluation_split", "log_loss", "brier", "model"],
        ignore_index=True,
        na_position="last",
    )


def _v5_directional_bootstrap_pvalue(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return None
    observed = float(values[finite].mean())
    centered = values.copy()
    centered[finite] -= observed
    block = min(BOOTSTRAP_BLOCK_WEEKS, max(1, len(values) // 2))
    blocks = int(np.ceil(len(values) / block))
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    starts = generator.integers(
        0,
        len(values),
        size=(V5_DIRECTIONAL_BOOTSTRAP_RESAMPLES, blocks),
    )
    offsets = np.arange(block)
    indexes = (starts[..., None] + offsets) % len(values)
    indexes = indexes.reshape(V5_DIRECTIONAL_BOOTSTRAP_RESAMPLES, -1)[
        :, : len(values)
    ]
    sampled = centered[indexes]
    counts = np.isfinite(sampled).sum(axis=1)
    valid = counts > 0
    if int(valid.sum()) < int(
        np.ceil(V5_DIRECTIONAL_BOOTSTRAP_RESAMPLES * 0.8)
    ):
        return None
    null = np.nansum(sampled[valid], axis=1) / counts[valid]
    return float(
        (1 + np.count_nonzero(null >= observed)) / (len(null) + 1)
    )


def _v5_validate_directional_model_origins(
    predictions: pd.DataFrame,
    *,
    horizon: int,
    split: str,
) -> None:
    frame = predictions.loc[
        predictions["horizon_weeks"].astype(int).eq(horizon)
        & predictions["evaluation_split"].astype(str).eq(split)
    ].copy()
    require(not frame.empty, f"v5 directional horizon-{horizon} {split} is empty")
    reference: pd.DataFrame | None = None
    for model, rows in frame.groupby("model", sort=True):
        ordered = rows.sort_values("origin_date", kind="mergesort").reset_index(drop=True)
        require(
            not ordered["origin_date"].duplicated().any(),
            f"v5 directional {model} duplicates an origin",
        )
        identity = ordered[
            [
                "origin_date",
                "target_end",
                "current_state",
                "actual_outcome",
                "actual_change",
            ]
        ].copy()
        identity["actual_change"] = boolean_series(
            identity["actual_change"], "v5 directional actual_change"
        ).to_numpy(dtype=bool)
        if reference is None:
            reference = identity
        else:
            try:
                pd.testing.assert_frame_equal(
                    reference,
                    identity,
                    check_dtype=False,
                    check_exact=True,
                    obj=f"v5 directional horizon-{horizon} {split} common origins",
                )
            except AssertionError as exc:
                raise AuditFailure(
                    f"v5 directional horizon-{horizon} {split} model origins differ"
                ) from exc


def _v5_select_directional_horizon(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    horizon: int,
) -> tuple[str, list[dict[str, Any]]]:
    frame = predictions.loc[
        predictions["horizon_weeks"].astype(int).eq(horizon)
        & predictions["evaluation_split"].astype(str).eq("selection")
    ].copy()
    table = metrics.loc[
        metrics["horizon_weeks"].astype(int).eq(horizon)
        & metrics["evaluation_split"].astype(str).eq("selection")
    ].set_index("model", drop=False)
    baseline_names = [
        model for model in V5_DIRECTIONAL_BASELINES if model in table.index
    ]
    require(baseline_names, f"v5 directional horizon-{horizon} has no baseline")
    baseline_rows = table.loc[baseline_names].reset_index(drop=True)
    reference_name = (
        "empirical_first_passage"
        if "empirical_first_passage" in table.index
        else str(baseline_rows.sort_values("model").iloc[0]["model"])
    )
    support = _v5_directional_event_support(
        frame.loc[frame["model"].astype(str).eq(reference_name)]
    )
    support_failures: list[str] = []
    if support["event_count"] < V5_DIRECTIONAL_MINIMUM_EVENTS:
        support_failures.append("insufficient_departure_events")
    if (
        support["destination_class_count"]
        < V5_DIRECTIONAL_MINIMUM_DESTINATION_CLASSES
    ):
        support_failures.append("insufficient_destination_classes")
    if support["effective_event_blocks"] < V5_DIRECTIONAL_MINIMUM_EVENT_BLOCKS:
        support_failures.append("insufficient_event_blocks")
    if support_failures:
        reason = ";".join(support_failures)
        diagnostics: list[dict[str, Any]] = []
        for model, row in table.iterrows():
            log_loss = float(row["log_loss"])
            brier = float(row["brier"])
            diagnostics.append(
                {
                    "horizon_weeks": horizon,
                    "model": str(model),
                    "reference_model": reference_name,
                    "selected": str(model) == reference_name,
                    "gate_passed": False,
                    "gate_reason": reason,
                    "score_target": V5_DIRECTIONAL_SCORE_TARGET,
                    "selection_event_count": support["event_count"],
                    "selection_destination_class_count": support[
                        "destination_class_count"
                    ],
                    "selection_effective_event_blocks": support[
                        "effective_event_blocks"
                    ],
                    "minimum_selection_events": V5_DIRECTIONAL_MINIMUM_EVENTS,
                    "minimum_destination_classes": (
                        V5_DIRECTIONAL_MINIMUM_DESTINATION_CLASSES
                    ),
                    "minimum_event_blocks": V5_DIRECTIONAL_MINIMUM_EVENT_BLOCKS,
                    "log_loss": log_loss if np.isfinite(log_loss) else None,
                    "brier": brier if np.isfinite(brier) else None,
                    "absolute_log_loss_improvement": None,
                    "holm_adjusted_p_value": None,
                    "fallback_count": int(row["fallback_count"]),
                }
            )
        return reference_name, diagnostics

    baseline = baseline_rows.sort_values(
        ["log_loss", "brier", "model"], na_position="last"
    ).iloc[0]
    reference = str(baseline["model"])
    reference_rows = frame.loc[
        frame["model"].astype(str).eq(reference)
    ].sort_values("origin_date", kind="mergesort")
    reference_loss = _v5_directional_conditional_losses(reference_rows)
    raw_pvalues: dict[str, float] = {}
    improvements: dict[str, float] = {}
    for model in table.index:
        if model in V5_DIRECTIONAL_BASELINES:
            continue
        candidate_rows = frame.loc[
            frame["model"].astype(str).eq(str(model))
        ].sort_values("origin_date", kind="mergesort")
        candidate_loss = _v5_directional_conditional_losses(candidate_rows)
        differential = reference_loss - candidate_loss
        improvements[str(model)] = float(np.nanmean(differential))
        pvalue = _v5_directional_bootstrap_pvalue(differential)
        if pvalue is not None:
            raw_pvalues[str(model)] = pvalue
    ordered_pvalues = sorted(
        raw_pvalues.items(), key=lambda item: (item[1], item[0])
    )
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (model, value) in enumerate(ordered_pvalues):
        running = max(
            running,
            min(1.0, value * (len(ordered_pvalues) - rank)),
        )
        adjusted[model] = running

    diagnostics = []
    passing: list[str] = []
    for model, row in table.iterrows():
        model_name = str(model)
        failures: list[str] = []
        if model_name not in V5_DIRECTIONAL_BASELINES:
            if int(row["fallback_count"]) != 0:
                failures.append("fallback_present")
            if (
                improvements[model_name]
                < DIRECTIONAL_MINIMUM_LOG_LOSS_IMPROVEMENT
            ):
                failures.append("insufficient_log_loss_improvement")
            if float(row["brier"]) > float(baseline["brier"]) + BRIER_TOLERANCE:
                failures.append("brier_degradation")
            if model_name not in adjusted:
                failures.append("bootstrap_insufficient")
            elif adjusted[model_name] > SELECTION_ALPHA:
                failures.append("holm_not_significant")
            if not failures:
                passing.append(model_name)
        elif model_name != reference:
            failures.append("non_reference_baseline")
        diagnostics.append(
            {
                "horizon_weeks": horizon,
                "model": model_name,
                "reference_model": reference,
                "selected": False,
                "gate_passed": not failures,
                "gate_reason": "passed" if not failures else ";".join(failures),
                "score_target": V5_DIRECTIONAL_SCORE_TARGET,
                "selection_event_count": support["event_count"],
                "selection_destination_class_count": support[
                    "destination_class_count"
                ],
                "selection_effective_event_blocks": support[
                    "effective_event_blocks"
                ],
                "minimum_selection_events": V5_DIRECTIONAL_MINIMUM_EVENTS,
                "minimum_destination_classes": (
                    V5_DIRECTIONAL_MINIMUM_DESTINATION_CLASSES
                ),
                "minimum_event_blocks": V5_DIRECTIONAL_MINIMUM_EVENT_BLOCKS,
                "log_loss": float(row["log_loss"]),
                "brier": float(row["brier"]),
                "absolute_log_loss_improvement": float(
                    float(baseline["log_loss"]) - float(row["log_loss"])
                ),
                "holm_adjusted_p_value": adjusted.get(model_name),
                "fallback_count": int(row["fallback_count"]),
            }
        )
    champion = reference
    if passing:
        champion = str(
            table.loc[passing]
            .reset_index(drop=True)
            .sort_values(["log_loss", "brier", "model"])
            .iloc[0]["model"]
        )
    for row in diagnostics:
        row["selected"] = row["model"] == champion
    return champion, diagnostics


def _recompute_v5_directional(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str]]:
    for horizon in V5_OUTCOME_HORIZONS:
        for split in ("selection", "retrospective_diagnostic"):
            _v5_validate_directional_model_origins(
                predictions,
                horizon=horizon,
                split=split,
            )
    leaderboard = _v5_directional_metrics(predictions)
    champions: dict[int, str] = {}
    diagnostic_rows: list[dict[str, Any]] = []
    for horizon in V5_OUTCOME_HORIZONS:
        champion, rows = _v5_select_directional_horizon(
            predictions,
            leaderboard,
            horizon=horizon,
        )
        champions[horizon] = champion
        diagnostic_rows.extend(rows)
    leaderboard = leaderboard.copy()
    leaderboard.insert(
        3,
        "selected",
        [
            str(model) == champions[int(horizon)]
            for horizon, model in zip(
                leaderboard["horizon_weeks"],
                leaderboard["model"],
                strict=True,
            )
        ],
    )
    diagnostics = pd.DataFrame(diagnostic_rows).sort_values(
        ["horizon_weeks", "model"], ignore_index=True
    )
    return leaderboard, diagnostics, champions


def _audit_v5_directional(
    payload: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    predictions = frames["directional-oos-predictions.csv"].copy()
    forecasts = frames["directional-forecasts.csv"].copy()
    splits = frames["directional-walk-forward-splits.csv"].copy()
    diagnostics = frames["directional-selection-diagnostics.csv"].copy()
    _audit_v5_probability_rows(
        predictions,
        context="v5 directional OOS predictions",
        include_actual=True,
    )
    _audit_v5_probability_rows(
        forecasts,
        context="v5 directional prospective forecasts",
        include_actual=False,
    )
    require_columns(
        splits,
        {
            "horizon_weeks",
            "origin_date",
            "target_end",
            "last_train_target_end",
            "purged_origin_count",
        },
        "v5 directional splits",
    )
    horizon = pd.to_numeric(splits["horizon_weeks"], errors="raise").astype(int)
    purged = pd.to_numeric(
        splits["purged_origin_count"], errors="raise"
    ).astype(int)
    require(
        set(horizon) == {1, 4, 13} and purged.eq(horizon).all(),
        "v5 directional split purge does not equal the forecast horizon",
    )
    origin = pd.to_datetime(splits["origin_date"], utc=True, errors="raise")
    last_target = pd.to_datetime(
        splits["last_train_target_end"], utc=True, errors="raise"
    )
    require(
        (last_target < origin).all(),
        "v5 directional split training target reaches the prediction origin",
    )
    origin_day = _calendar_days(
        splits["origin_date"], context="v5 directional splits.origin_date"
    )
    target_day = _calendar_days(
        splits["target_end"], context="v5 directional splits.target_end"
    )
    require(
        ((target_day - origin_day).dt.days == 7 * horizon).all(),
        "v5 directional split target_end is not exactly 7*h calendar days later",
    )
    require(
        not splits.duplicated(["horizon_weeks", "origin_date"]).any(),
        "v5 directional splits duplicate an origin/horizon",
    )
    require_columns(
        splits,
        {"evaluation_split"},
        "v5 directional splits",
    )
    prediction_origins = predictions.loc[
        :,
        ["horizon_weeks", "origin_date", "target_end", "evaluation_split"],
    ].copy()
    prediction_origins["horizon_weeks"] = pd.to_numeric(
        prediction_origins["horizon_weeks"], errors="raise"
    ).astype(int)
    prediction_origins["origin_date"] = pd.to_datetime(
        prediction_origins["origin_date"], utc=True, errors="raise"
    )
    prediction_origins["target_end"] = pd.to_datetime(
        prediction_origins["target_end"], utc=True, errors="raise"
    )
    prediction_origins["evaluation_split"] = prediction_origins[
        "evaluation_split"
    ].astype(str)
    prediction_origins = prediction_origins.drop_duplicates(ignore_index=True)
    split_origins = splits.loc[
        :,
        ["horizon_weeks", "origin_date", "target_end", "evaluation_split"],
    ].copy()
    split_origins["horizon_weeks"] = pd.to_numeric(
        split_origins["horizon_weeks"], errors="raise"
    ).astype(int)
    split_origins["origin_date"] = pd.to_datetime(
        split_origins["origin_date"], utc=True, errors="raise"
    )
    split_origins["target_end"] = pd.to_datetime(
        split_origins["target_end"], utc=True, errors="raise"
    )
    split_origins["evaluation_split"] = split_origins[
        "evaluation_split"
    ].astype(str)
    require(
        {
            tuple(row)
            for row in prediction_origins.itertuples(index=False, name=None)
        }
        == {tuple(row) for row in split_origins.itertuples(index=False, name=None)},
        "v5 directional predictions/splits origin contract mismatch",
    )
    execution = _audit_v5_execution_parameters(payload)
    split_limits = {
        "selection": (
            int(execution["directional_minimum_selection_predictions"]),
            execution["directional_maximum_selection_origins"],
        ),
        "retrospective_diagnostic": (
            int(execution["directional_minimum_diagnostic_predictions"]),
            execution["directional_maximum_diagnostic_origins"],
        ),
    }
    for horizon_value in V5_OUTCOME_HORIZONS:
        for split_name, (minimum, maximum) in split_limits.items():
            count = int(
                len(
                    splits.loc[
                        splits["horizon_weeks"].astype(int).eq(horizon_value)
                        & splits["evaluation_split"].astype(str).eq(split_name)
                    ]
                )
            )
            require(
                count >= minimum,
                f"v5 directional horizon-{horizon_value} {split_name} "
                "origin count is below execution parameters",
            )
            if maximum is not None:
                require(
                    count <= int(maximum),
                    f"v5 directional horizon-{horizon_value} {split_name} "
                    "origin count exceeds execution parameters",
                )

    require_columns(
        diagnostics,
        {"horizon_weeks", "model", "selected"},
        "v5 directional selection diagnostics",
    )
    selected = diagnostics.loc[
        boolean_series(
            diagnostics["selected"],
            "v5 directional selection diagnostics.selected",
        )
    ].copy()
    require(
        selected["horizon_weeks"].astype(int).value_counts().to_dict()
        == {1: 1, 4: 1, 13: 1},
        "v5 directional selection must choose one model per horizon",
    )
    champions = payload["model"]["directional_transition"]["champions"]
    directional_contract = payload["model"]["directional_transition"]
    require(
        directional_contract.get("target")
        == "first_departure_state_within_h_or_no_departure",
        "v5 directional target mismatch",
    )
    require(
        directional_contract.get("deployed_direction_role")
        == V5_DIRECTIONAL_SCORE_TARGET,
        "v5 directional deployed role mismatch",
    )
    require(
        directional_contract.get("selection_metric")
        == "conditional_destination_log_loss",
        "v5 directional selection metric mismatch",
    )
    require(
        int(directional_contract.get("minimum_selection_departure_events", -1))
        == V5_DIRECTIONAL_MINIMUM_EVENTS,
        "v5 directional minimum event gate mismatch",
    )
    require(
        int(directional_contract.get("minimum_selection_destination_classes", -1))
        == V5_DIRECTIONAL_MINIMUM_DESTINATION_CLASSES,
        "v5 directional minimum destination-class gate mismatch",
    )
    require(
        int(directional_contract.get("minimum_selection_event_blocks", -1))
        == V5_DIRECTIONAL_MINIMUM_EVENT_BLOCKS,
        "v5 directional minimum event-block gate mismatch",
    )
    _audit_v5_embedded_records(
        directional_contract["leaderboard"],
        frames["directional-model-leaderboard.csv"],
        keys=("horizon_weeks", "evaluation_split", "model"),
        context="v5 directional leaderboard",
    )
    _audit_v5_embedded_records(
        directional_contract["selection_diagnostics"],
        diagnostics,
        keys=("horizon_weeks", "model"),
        context="v5 directional selection diagnostics",
    )
    recomputed_leaderboard, recomputed_diagnostics, recomputed_champions = (
        _recompute_v5_directional(predictions)
    )
    leaderboard_fields = (
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
    _audit_v5_recomputed_records(
        recomputed_leaderboard,
        frames["directional-model-leaderboard.csv"],
        keys=("horizon_weeks", "evaluation_split", "model"),
        fields=leaderboard_fields,
        context="v5 directional leaderboard",
    )
    diagnostic_fields = (
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
    _audit_v5_recomputed_records(
        recomputed_diagnostics,
        diagnostics,
        keys=("horizon_weeks", "model"),
        fields=diagnostic_fields,
        context="v5 directional selection diagnostics",
    )
    for horizon_value in (1, 4, 13):
        require(
            str(champions[f"{horizon_value}w"])
            == recomputed_champions[horizon_value],
            f"v5 directional horizon-{horizon_value} champion disagrees with "
            "independent recomputation",
        )
        selected_model = str(
            selected.loc[
                selected["horizon_weeks"].astype(int).eq(horizon_value), "model"
            ].iloc[0]
        )
        require(
            selected_model == str(champions[f"{horizon_value}w"]),
            f"v5 directional horizon-{horizon_value} champion mismatch",
        )
        require(
            forecasts.loc[
                forecasts["horizon_weeks"].astype(int).eq(horizon_value), "model"
            ].astype(str).eq(selected_model).all(),
            f"v5 directional horizon-{horizon_value} forecast model mismatch",
        )
    return {
        "oos_rows": len(predictions),
        "forecast_rows": len(forecasts),
        "split_rows": len(splits),
        "champions": dict(champions),
    }


def _v5_reconciled_destinations(
    *,
    probability: float,
    current_state: str,
    source: Mapping[str, object],
) -> dict[str, float]:
    destination = {
        state: (
            0.0
            if state == current_state
            else max(0.0, float(source.get(f"p_{state}", 0.0)))
        )
        for state in STATE_ORDER
    }
    total = float(sum(destination.values()))
    if not np.isfinite(total) or total <= 0.0:
        alternatives = [state for state in STATE_ORDER if state != current_state]
        destination = {
            state: (1.0 / len(alternatives) if state in alternatives else 0.0)
            for state in STATE_ORDER
        }
    else:
        destination = {
            state: value / total for state, value in destination.items()
        }
    scaled = {
        state: round(float(probability) * value, 8)
        for state, value in destination.items()
    }
    residual = round(float(probability) - sum(scaled.values()), 8)
    if residual:
        chosen = max(
            (state for state in STATE_ORDER if state != current_state),
            key=scaled.get,
        )
        scaled[chosen] = round(scaled[chosen] + residual, 8)
    return scaled


def _v5_markov_direction_source(
    membership: pd.DataFrame,
    *,
    cutoff: object,
    current_state: str,
) -> dict[str, float]:
    dates = pd.to_datetime(membership["date"], utc=True, errors="raise")
    at = pd.Timestamp(cutoff)
    at = at.tz_localize("UTC") if at.tzinfo is None else at.tz_convert("UTC")
    states = membership.loc[dates.le(at), "state"].astype(str).reset_index(drop=True)
    require(len(states) >= 2, "v5 weekly directional fallback history is too short")
    current = states.iloc[:-1]
    following = states.iloc[1:].to_numpy(dtype=object)
    mask = current.eq(current_state).to_numpy(dtype=bool)
    counts = {
        state: float(np.count_nonzero(following[mask] == state)) + 1.0
        for state in STATE_ORDER
    }
    return {f"p_{state}": value for state, value in counts.items()}


def _audit_v5_weekly_directional(
    payload: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    membership: pd.DataFrame,
) -> dict[str, int]:
    """Bind published weekly directions to the selected sidecar or PIT fallback."""

    require_columns(membership, {"date", "state"}, "v5 directional membership")
    champions = payload["model"]["directional_transition"]["champions"]
    lookup: dict[tuple[str, int], Mapping[str, object]] = {}
    for path in (
        "directional-oos-predictions.csv",
        "directional-forecasts.csv",
    ):
        frame = frames[path]
        for row in frame.to_dict(orient="records"):
            horizon = int(row["horizon_weeks"])
            if str(row["model"]) != str(champions[f"{horizon}w"]):
                continue
            key = (_market_date(row["origin_date"]), horizon)
            previous = lookup.get(key)
            if previous is not None:
                fields = (
                    "model",
                    "current_state",
                    "p_no_departure",
                    *PROBABILITY_COLUMNS,
                )
                require(
                    all(str(previous[field]) == str(row[field]) for field in fields),
                    f"v5 directional sidecars conflict at {key}",
                )
            lookup[key] = row

    matched = 0
    fallback = 0
    for week_index, week in enumerate(payload["weekly"]):
        origin = str(week["date"])
        current_state = str(week["current"]["state"])
        cutoff = week.get("data_as_of", origin)
        for horizon in V5_OUTCOME_HORIZONS:
            context = f"v5 weekly[{week_index}].directional_risk.{horizon}w"
            published = week["directional_risk"][f"{horizon}w"]
            source = lookup.get((origin, horizon))
            if source is None:
                expected_model = "markov_first_passage"
                source = _v5_markov_direction_source(
                    membership,
                    cutoff=cutoff,
                    current_state=current_state,
                )
                fallback += 1
            else:
                expected_model = str(champions[f"{horizon}w"])
                require(
                    str(source["current_state"]) == current_state,
                    f"{context} source current state mismatch",
                )
                matched += 1
            require(
                str(published["model"]) == expected_model,
                f"{context} model/source mismatch",
            )
            probability = float(published["probability"])
            expected_destinations = _v5_reconciled_destinations(
                probability=probability,
                current_state=current_state,
                source=source,
            )
            for state in STATE_ORDER:
                require(
                    np.isclose(
                        float(published["first_destination"][state]),
                        expected_destinations[state],
                        atol=5e-8,
                        rtol=0.0,
                    ),
                    f"{context} {state} destination/source mismatch",
                )
    return {"matched_rows": matched, "fallback_rows": fallback}


def _v5_conditional_point_metrics(
    returns: np.ndarray,
    drawdowns: np.ndarray,
    *,
    horizon_weeks: int,
) -> dict[str, float]:
    returns = np.asarray(returns, dtype=float)
    drawdowns = np.asarray(drawdowns, dtype=float)
    tail_count = max(1, int(np.ceil(0.05 * len(returns))))
    annualization = np.sqrt(52.0 / horizon_weeks)
    return {
        "mean_return": float(np.mean(returns)),
        "median_return": float(np.median(returns)),
        "positive_rate": float(np.mean(returns > 0.0)),
        "annualized_volatility": (
            float(np.std(returns, ddof=1) * annualization)
            if len(returns) > 1
            else float("nan")
        ),
        "downside_volatility": float(
            np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2)) * annualization
        ),
        "cvar_5": float(np.mean(np.sort(returns)[:tail_count])),
        "mean_max_drawdown": float(np.mean(drawdowns)),
    }


def _v5_conditional_block_indexes(
    frame: pd.DataFrame,
    *,
    block_length: int,
) -> np.ndarray:
    ordered = frame.sort_values("origin_position", kind="mergesort").reset_index(
        drop=True
    )
    blocks: list[np.ndarray] = []
    for _, episode in ordered.groupby("episode_id", sort=False):
        episode = episode.sort_values(
            "origin_position", kind="mergesort"
        ).reset_index()
        gap_group = episode["origin_position"].diff().ne(1).cumsum()
        for _, contiguous in episode.groupby(gap_group, sort=False):
            base_positions = contiguous["index"].to_numpy(dtype=int)
            offsets = np.arange(block_length)
            for start in range(len(base_positions)):
                blocks.append(base_positions[(start + offsets) % len(base_positions)])
    require(bool(blocks), "v5 conditional bootstrap has no episode-bounded blocks")
    return np.asarray(blocks, dtype=int)


def _v5_conditional_bootstrap_intervals(
    frame: pd.DataFrame,
    *,
    horizon_weeks: int,
    block_length: int,
    resamples: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    if resamples == 0:
        return {
            metric: (float("nan"), float("nan"))
            for metric in V5_OUTCOME_POINT_METRICS
        }
    ordered = frame.sort_values("origin_position", kind="mergesort").reset_index(
        drop=True
    )
    block_indexes = _v5_conditional_block_indexes(
        ordered,
        block_length=block_length,
    )
    returns = ordered["forward_return"].to_numpy(dtype=float)
    drawdowns = ordered["max_drawdown"].to_numpy(dtype=float)
    target_rows = len(ordered)
    blocks_per_replicate = int(np.ceil(target_rows / block_length))
    generator = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {
        metric: [] for metric in V5_OUTCOME_POINT_METRICS
    }
    for _ in range(resamples):
        chosen = generator.integers(
            0,
            len(block_indexes),
            size=blocks_per_replicate,
        )
        positions = block_indexes[chosen].reshape(-1)[:target_rows]
        metrics = _v5_conditional_point_metrics(
            returns[positions],
            drawdowns[positions],
            horizon_weeks=horizon_weeks,
        )
        for metric, value in metrics.items():
            draws[metric].append(value)
    intervals: dict[str, tuple[float, float]] = {}
    for metric, values in draws.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        require(len(finite) > 0, f"v5 conditional {metric} bootstrap is empty")
        lower, upper = np.quantile(finite, [0.025, 0.975])
        intervals[metric] = (float(lower), float(upper))
    return intervals


def _v5_non_overlapping_count(frame: pd.DataFrame) -> int:
    """Greedily count disjoint holding windows in chronological order."""

    if frame.empty:
        return 0
    ordered = frame.assign(
        _entry=pd.to_datetime(frame["entry_date"], utc=True),
        _exit=pd.to_datetime(frame["exit_date"], utc=True),
    ).sort_values(["_entry", "_exit"], kind="mergesort")
    count = 0
    previous_exit: pd.Timestamp | None = None
    for entry_value, exit_value in zip(
        ordered["_entry"], ordered["_exit"], strict=True
    ):
        entry = pd.Timestamp(entry_value)
        exit_date = pd.Timestamp(exit_value)
        if previous_exit is None or entry > previous_exit:
            count += 1
            previous_exit = exit_date
    return count


def _v5_episode_equal_mean(frame: pd.DataFrame) -> float:
    if frame.empty:
        return float("nan")
    means = frame.groupby("episode_id", sort=False)["forward_return"].mean()
    return float(means.mean())


def _v5_whole_episode_bootstrap_interval(
    frame: pd.DataFrame,
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    if resamples == 0 or frame.empty:
        return float("nan"), float("nan")
    episode_means = (
        frame.groupby("episode_id", sort=False)["forward_return"]
        .mean()
        .to_numpy(dtype=float)
    )
    require(
        len(episode_means) > 0,
        "v5 conditional episode bootstrap has no episodes",
    )
    generator = np.random.default_rng(seed)
    draws = np.asarray(
        [
            float(
                np.mean(
                    episode_means[
                        generator.integers(
                            0,
                            len(episode_means),
                            len(episode_means),
                        )
                    ]
                )
            )
            for _ in range(resamples)
        ],
        dtype=float,
    )
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return float(lower), float(upper)


def _recompute_v5_conditional_statistics(
    outcomes: pd.DataFrame,
    statistics: pd.DataFrame,
    *,
    expected_resamples: int,
) -> pd.DataFrame:
    keys = ("state", "asset", "horizon_weeks")
    require_columns(statistics, set(keys), "v5 conditional statistics")
    require(
        not statistics.duplicated(list(keys)).any(),
        "v5 conditional statistics keys duplicate",
    )
    statistic_keys = {
        (str(row.state), str(row.asset), int(row.horizon_weeks))
        for row in statistics.itertuples(index=False)
    }
    outcome_keys = {
        (str(row.state), str(row.asset), int(row.horizon_weeks))
        for row in outcomes.loc[:, list(keys)].drop_duplicates().itertuples(index=False)
    }
    require(
        outcome_keys.issubset(statistic_keys),
        "v5 conditional outcomes contain an unreported group",
    )
    state_positions = {state: index for index, state in enumerate(STATE_ORDER)}
    asset_positions = {
        asset: index for index, asset in enumerate(V5_OUTCOME_ASSETS)
    }
    unconditional = {
        (str(asset), int(horizon)): group.sort_values(
            "origin_position", kind="mergesort"
        )
        for (asset, horizon), group in outcomes.groupby(
            ["asset", "horizon_weeks"], sort=False
        )
    }
    rows: list[dict[str, Any]] = []
    for statistic in statistics.itertuples(index=False):
        state = str(statistic.state)
        asset = str(statistic.asset)
        horizon = int(statistic.horizon_weeks)
        require(state in state_positions, f"v5 conditional state {state} is invalid")
        require(asset in asset_positions, f"v5 conditional asset {asset} is invalid")
        require(
            horizon in V5_OUTCOME_HORIZONS,
            f"v5 conditional horizon {horizon} is invalid",
        )
        group = outcomes.loc[
            outcomes["state"].astype(str).eq(state)
            & outcomes["asset"].astype(str).eq(asset)
            & outcomes["horizon_weeks"].astype(int).eq(horizon)
        ].sort_values("origin_position", kind="mergesort")
        count = int(len(group))
        unique_episodes = int(group["episode_id"].nunique())
        minimum_observations = 20
        minimum_episodes = 5
        non_overlapping_n = _v5_non_overlapping_count(group)
        reported_minimum_non_overlapping = getattr(
            statistic,
            "minimum_non_overlapping_observations",
            None,
        )
        minimum_non_overlapping = (
            1
            if reported_minimum_non_overlapping is None
            else int(reported_minimum_non_overlapping)
        )
        if reported_minimum_non_overlapping is not None:
            require(
                minimum_non_overlapping == 5,
                "v5 conditional minimum non-overlapping support is invalid",
            )
        supported = (
            count >= minimum_observations
            and unique_episodes >= minimum_episodes
            and non_overlapping_n >= minimum_non_overlapping
        )
        metrics = (
            _v5_conditional_point_metrics(
                group["forward_return"].to_numpy(dtype=float),
                group["max_drawdown"].to_numpy(dtype=float),
                horizon_weeks=horizon,
            )
            if count
            else {
                metric: float("nan") for metric in V5_OUTCOME_POINT_METRICS
            }
        )
        seed = (
            17
            + state_positions[state] * 10_000
            + asset_positions[asset] * 100
            + horizon
        )
        intervals = (
            _v5_conditional_bootstrap_intervals(
                group,
                horizon_weeks=horizon,
                block_length=13,
                resamples=expected_resamples,
                seed=seed,
            )
            if supported
            else {
                metric: (float("nan"), float("nan"))
                for metric in V5_OUTCOME_POINT_METRICS
            }
        )
        benchmark = unconditional.get((asset, horizon), outcomes.iloc[0:0])
        benchmark_mean = (
            float(benchmark["forward_return"].mean())
            if len(benchmark)
            else float("nan")
        )
        episode_equal_mean = _v5_episode_equal_mean(group)
        episode_equal_benchmark_mean = _v5_episode_equal_mean(benchmark)
        episode_equal_benchmark_episodes = int(
            benchmark["episode_id"].nunique()
        )
        episode_seed = seed + 1_000_000
        episode_interval = (
            _v5_whole_episode_bootstrap_interval(
                group,
                resamples=expected_resamples,
                seed=episode_seed,
            )
            if supported
            else (float("nan"), float("nan"))
        )
        reported_benchmark_method = getattr(
            statistic,
            "unconditional_benchmark_method",
            "same_asset_horizon_all_origins_mean",
        )
        require(
            reported_benchmark_method
            in {
                "same_asset_horizon_all_origins_buy_and_hold",
                "same_asset_horizon_all_origins_mean",
            },
            "v5 conditional unconditional benchmark method is invalid",
        )
        row: dict[str, Any] = {
            "state": state,
            "asset": asset,
            "horizon_weeks": horizon,
            "execution_lag_weeks": (
                int(group["execution_lag_weeks"].iloc[0]) if count else 1
            ),
            "return_currency": (
                str(group["return_currency"].iloc[0]) if count else "USD"
            ),
            "sample_start": (
                pd.Timestamp(group["origin_date"].min()).date().isoformat()
                if count
                else None
            ),
            "sample_end": (
                pd.Timestamp(group["origin_date"].max()).date().isoformat()
                if count
                else None
            ),
            "n": count,
            "non_overlapping_n": non_overlapping_n,
            "unique_episodes": unique_episodes,
            "status": "ok" if supported else "insufficient_support",
            "minimum_observations": minimum_observations,
            "minimum_unique_episodes": minimum_episodes,
            "minimum_non_overlapping_observations": minimum_non_overlapping,
            "bootstrap_method": "episode_bounded_circular_block",
            "bootstrap_block_weeks": 13,
            "bootstrap_resamples": expected_resamples,
            "bootstrap_seed": seed,
            "unconditional_benchmark_method": reported_benchmark_method,
            "unconditional_benchmark_n": int(len(benchmark)),
            "unconditional_benchmark_mean_return": benchmark_mean,
            "excess_mean_return": (
                float(metrics["mean_return"] - benchmark_mean)
                if count and np.isfinite(benchmark_mean)
                else float("nan")
            ),
            "episode_equal_mean_return": episode_equal_mean,
            "episode_equal_unconditional_benchmark_method": (
                "same_asset_horizon_all_state_episodes_equal_weight"
            ),
            "episode_equal_unconditional_benchmark_episode_n": (
                episode_equal_benchmark_episodes
            ),
            "episode_equal_unconditional_benchmark_mean_return": (
                episode_equal_benchmark_mean
            ),
            "episode_equal_excess_return": (
                float(
                    episode_equal_mean
                    - episode_equal_benchmark_mean
                )
                if np.isfinite(episode_equal_mean)
                and np.isfinite(episode_equal_benchmark_mean)
                else float("nan")
            ),
            "episode_bootstrap_method": "whole_episode_resampling",
            "episode_bootstrap_resamples": expected_resamples,
            "episode_bootstrap_seed": episode_seed,
            "episode_equal_mean_return_ci95_lower": episode_interval[0],
            "episode_equal_mean_return_ci95_upper": episode_interval[1],
            **metrics,
        }
        for metric, (lower, upper) in intervals.items():
            row[f"{metric}_ci95_lower"] = lower
            row[f"{metric}_ci95_upper"] = upper
        rows.append(row)
    return pd.DataFrame(rows)


def _v5_conditional_comparison_fields(
    statistics: pd.DataFrame,
    *,
    require_investment_fields: bool,
) -> tuple[str, ...]:
    present = set(str(column) for column in statistics.columns)
    legacy_required = set(V5_CONDITIONAL_STATISTIC_FIELDS).difference(
        V5_INVESTMENT_CONDITIONAL_FIELDS
    )
    require(
        legacy_required.issubset(present),
        "v5 conditional statistics omit required legacy fields",
    )
    if require_investment_fields:
        require(
            set(V5_CONDITIONAL_STATISTIC_FIELDS).issubset(present),
            "v5 investment-aligned conditional statistics omit required fields",
        )
    return tuple(
        field for field in V5_CONDITIONAL_STATISTIC_FIELDS if field in present
    )


def _audit_v5_conditional(
    payload: Mapping[str, Any], frames: Mapping[str, pd.DataFrame]
) -> dict[str, Any]:
    statistics = frames["conditional-asset-statistics.csv"]
    metadata = payload["research"]["conditional_asset_stats"]
    method = metadata.get("method")
    investment_aligned = (
        method
        == "matched_oos_actual_next_state_target_week_adjusted_forward_return"
    )
    require(
        method
        in {
            "state_conditioned_forward_total_return",
            "matched_oos_actual_next_state_target_week_adjusted_forward_return",
        },
        "v5 conditional asset method is invalid",
    )
    if investment_aligned:
        require(
            metadata.get("role") == "matched_oracle_diagnostic"
            and metadata.get("conditioning")
            == "actual_next_state_on_matched_oos_origins"
            and metadata.get("state_horizon_weeks") == 1
            and metadata.get("execution_lag_weeks") == 1
            and metadata.get("entry_price_basis") == "next_week_adjusted_open"
            and metadata.get("exit_price_basis")
            == "horizon_week_adjusted_close"
            and metadata.get("rebalance_policy") == "none_fixed_asset_hold"
            and metadata.get("origin_sampling") == "weekly_rolling_overlapping"
            and metadata.get("return_measure")
            == "provider_adjusted_forward_return"
            and metadata.get("entry_week_distribution_policy")
            == "conservative_excluded_without_ex_date"
            and metadata.get("corporate_action_policy")
            == "same_row_adjustment_factor_split_consistent"
            and metadata.get("drawdown_observation_basis")
            == "entry_adjusted_open_then_weekly_adjusted_closes",
            "v5 conditional investment semantics mismatch",
        )
    embedded = metadata["rows"]
    _audit_v5_embedded_records(
        embedded,
        statistics,
        keys=("state", "asset", "horizon_weeks"),
        context="v5 conditional asset statistics",
    )
    outcomes = frames["conditional-asset-outcomes.csv"]
    require_columns(
        outcomes,
        {
            "origin_position",
            "origin_date",
            "entry_date",
            "exit_date",
            "state",
            "episode_id",
            "asset",
            "horizon_weeks",
            "execution_lag_weeks",
            "return_currency",
            "forward_return",
            "max_drawdown",
        },
        "v5 conditional asset outcomes",
    )
    origin = _calendar_days(
        outcomes["origin_date"], context="v5 conditional outcomes.origin_date"
    )
    entry = _calendar_days(
        outcomes["entry_date"], context="v5 conditional outcomes.entry_date"
    )
    exit_date = _calendar_days(
        outcomes["exit_date"], context="v5 conditional outcomes.exit_date"
    )
    horizon = pd.to_numeric(outcomes["horizon_weeks"], errors="raise").astype(int)
    require(
        ((entry - origin).dt.days == 7).all(),
        "v5 conditional outcomes entry is not t+1 week",
    )
    require(
        ((exit_date - entry).dt.days == 7 * (horizon - 1)).all(),
        "v5 conditional outcomes exit horizon is invalid",
    )
    require(
        (horizon.ne(1) | entry.eq(exit_date)).all(),
        "v5 conditional one-week outcome must include target-week open-to-close",
    )
    require(
        outcomes["execution_lag_weeks"].astype(int).eq(1).all()
        and outcomes["return_currency"].astype(str).eq("USD").all(),
        "v5 conditional outcomes execution/currency contract mismatch",
    )
    require(
        outcomes["state"].astype(str).isin(STATE_ORDER).all()
        and outcomes["asset"].astype(str).isin(V5_OUTCOME_ASSETS).all()
        and set(horizon).issubset(set(V5_OUTCOME_HORIZONS))
        and bool(set(horizon)),
        "v5 conditional outcomes state/asset/horizon is invalid",
    )
    require(
        not outcomes.duplicated(
            ["origin_position", "horizon_weeks", "asset"]
        ).any(),
        "v5 conditional outcomes duplicate an origin/horizon/asset",
    )
    numeric_outcomes = outcomes.loc[:, ["forward_return", "max_drawdown"]].apply(
        pd.to_numeric, errors="raise"
    )
    require(
        np.isfinite(numeric_outcomes.to_numpy(dtype=float)).all(),
        "v5 conditional outcomes contain non-finite values",
    )
    if investment_aligned:
        weekly = payload.get("weekly")
        require(
            isinstance(weekly, list) and len(weekly) >= 2,
            "v5 matched conditional outcomes require weekly target states",
        )
        current_state_by_date: dict[str, str] = {}
        for position, week in enumerate(weekly):
            require(
                isinstance(week, Mapping),
                f"v5 matched conditional weekly[{position}] is invalid",
            )
            current = week.get("current")
            require(
                isinstance(current, Mapping),
                f"v5 matched conditional weekly[{position}].current is invalid",
            )
            current_state_by_date[_market_date(week.get("date"))] = str(
                current.get("state")
            )
        for row in outcomes.itertuples(index=False):
            target_day = (
                pd.Timestamp(row.origin_date) + timedelta(weeks=1)
            ).date().isoformat()
            require(
                current_state_by_date.get(target_day) == str(row.state),
                "v5 matched conditional outcome is not bound to actual next state",
            )
    execution = _audit_v5_execution_parameters(payload)
    expected_resamples = int(
        execution["conditional_outcome_bootstrap_resamples"]
    )
    if investment_aligned:
        require(
            statistics["unconditional_benchmark_method"]
            .astype(str)
            .eq("same_asset_horizon_all_origins_mean")
            .all(),
            "v5 conditional investment benchmark method is mislabeled",
        )
    recomputed = _recompute_v5_conditional_statistics(
        outcomes,
        statistics,
        expected_resamples=expected_resamples,
    )
    _audit_v5_recomputed_records(
        recomputed,
        statistics,
        keys=("state", "asset", "horizon_weeks"),
        fields=_v5_conditional_comparison_fields(
            statistics,
            require_investment_fields=investment_aligned,
        ),
        context="v5 conditional asset statistics",
    )
    return {
        "outcome_rows": len(outcomes),
        "statistics_rows": len(statistics),
        "bootstrap_resamples": expected_resamples,
    }


def _audit_v5_model_conditioned(
    payload: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    """Bind model-conditioned outcomes to OOS forecasts and recompute statistics."""

    outcome_path = "model-conditioned-asset-outcomes.csv"
    statistics_path = "model-conditioned-asset-statistics.csv"
    present = {path for path in (outcome_path, statistics_path) if path in frames}
    research = payload.get("research")
    require(isinstance(research, Mapping), "v5 research payload is invalid")
    embedded_metadata = research.get("model_conditioned_asset_stats")
    if not present:
        require(
            embedded_metadata is None,
            "v5 model-conditioned payload requires its artifact pair",
        )
        return {
            "status": "legacy_absent",
            "models": 0,
            "outcome_rows": 0,
            "statistics_rows": 0,
        }
    require(
        present == {outcome_path, statistics_path},
        "v5 model-conditioned artifacts must be a complete pair",
    )
    require(
        isinstance(embedded_metadata, Mapping),
        "v5 model-conditioned payload statistics are missing",
    )

    model_metadata = payload.get("model")
    require(
        isinstance(model_metadata, Mapping),
        "v5 model-conditioned model metadata is invalid",
    )
    comparison = model_metadata.get("forecast_comparison")
    require(
        isinstance(comparison, Mapping),
        "v5 model-conditioned outcomes require forecast comparison metadata",
    )
    raw_models = comparison.get("models")
    require(
        isinstance(raw_models, list) and bool(raw_models),
        "v5 model-conditioned forecast models are invalid",
    )
    models = tuple(str(name) for name in raw_models)
    require(
        len(models) == len(set(models)),
        "v5 model-conditioned forecast models duplicate",
    )
    method = embedded_metadata.get("method")
    investment_aligned = (
        method
        == "matched_oos_predicted_next_state_target_week_adjusted_forward_return"
    )
    require(
        method
        in {
            "oos_one_week_forecast_conditioned_forward_total_return",
            "matched_oos_predicted_next_state_target_week_adjusted_forward_return",
        }
        and embedded_metadata.get("role") == "retrospective_model_diagnostic"
        and embedded_metadata.get("conditioning") == "hard_argmax_oos_forecast"
        and embedded_metadata.get("forecast_horizon_weeks") == 1
        and embedded_metadata.get("execution_lag_weeks") == 1
        and tuple(embedded_metadata.get("horizons_weeks", ()))
        == V5_OUTCOME_HORIZONS
        and tuple(embedded_metadata.get("assets", ())) == V5_OUTCOME_ASSETS
        and tuple(str(name) for name in embedded_metadata.get("models", ()))
        == models
        and embedded_metadata.get("return_currency") == "USD",
        "v5 model-conditioned metadata contract mismatch",
    )
    if investment_aligned:
        require(
            embedded_metadata.get("entry_price_basis")
            == "next_week_adjusted_open"
            and embedded_metadata.get("exit_price_basis")
            == "horizon_week_adjusted_close"
            and embedded_metadata.get("rebalance_policy")
            == "none_fixed_asset_hold"
            and embedded_metadata.get("origin_sampling")
            == "weekly_rolling_overlapping"
            and embedded_metadata.get("return_measure")
            == "provider_adjusted_forward_return"
            and embedded_metadata.get("entry_week_distribution_policy")
            == "conservative_excluded_without_ex_date"
            and embedded_metadata.get("corporate_action_policy")
            == "same_row_adjustment_factor_split_consistent"
            and embedded_metadata.get("drawdown_observation_basis")
            == "entry_adjusted_open_then_weekly_adjusted_closes",
            "v5 model-conditioned investment semantics mismatch",
        )

    statistics = frames[statistics_path]
    if investment_aligned:
        require(
            statistics["unconditional_benchmark_method"]
            .astype(str)
            .eq("same_asset_horizon_all_origins_mean")
            .all(),
            "v5 model-conditioned investment benchmark method is mislabeled",
        )
    outcomes = frames[outcome_path].copy()
    _audit_v5_embedded_records(
        embedded_metadata.get("rows"),
        statistics,
        keys=("conditioning_model", "state", "asset", "horizon_weeks"),
        context="v5 model-conditioned asset statistics",
    )
    require_columns(
        outcomes,
        {
            "conditioning_model",
            "origin_position",
            "origin_date",
            "entry_date",
            "exit_date",
            "state",
            "episode_id",
            "asset",
            "horizon_weeks",
            "execution_lag_weeks",
            "return_currency",
            "forward_return",
            "max_drawdown",
        },
        "v5 model-conditioned asset outcomes",
    )
    model_statistic_fields = _v5_conditional_comparison_fields(
        statistics,
        require_investment_fields=investment_aligned,
    )
    require_columns(
        statistics,
        {
            "conditioning_model",
            "state",
            "asset",
            "horizon_weeks",
            *model_statistic_fields,
        },
        "v5 model-conditioned asset statistics",
    )
    require(
        set(statistics["conditioning_model"].astype(str)) == set(models)
        and len(statistics)
        == len(models)
        * len(STATE_ORDER)
        * len(V5_OUTCOME_ASSETS)
        * len(V5_OUTCOME_HORIZONS),
        "v5 model-conditioned statistics model/row coverage is invalid",
    )

    origin = _calendar_days(
        outcomes["origin_date"],
        context="v5 model-conditioned outcomes.origin_date",
    )
    entry = _calendar_days(
        outcomes["entry_date"],
        context="v5 model-conditioned outcomes.entry_date",
    )
    exit_date = _calendar_days(
        outcomes["exit_date"],
        context="v5 model-conditioned outcomes.exit_date",
    )
    horizon = pd.to_numeric(outcomes["horizon_weeks"], errors="raise")
    positions = pd.to_numeric(outcomes["origin_position"], errors="raise")
    episodes = pd.to_numeric(outcomes["episode_id"], errors="raise")
    require(
        np.isfinite(horizon.to_numpy(dtype=float)).all()
        and np.isfinite(positions.to_numpy(dtype=float)).all()
        and np.isfinite(episodes.to_numpy(dtype=float)).all()
        and np.equal(horizon, np.floor(horizon)).all()
        and np.equal(positions, np.floor(positions)).all()
        and np.equal(episodes, np.floor(episodes)).all()
        and (positions >= 0).all()
        and (episodes >= 0).all(),
        "v5 model-conditioned outcome indexes are invalid",
    )
    horizon = horizon.astype(int)
    outcomes["_origin_position"] = positions.astype(int)
    outcomes["_episode_id"] = episodes.astype(int)
    outcomes["_origin_day"] = origin.dt.strftime("%Y-%m-%d")
    outcomes["_entry_day"] = entry.dt.strftime("%Y-%m-%d")
    outcomes["_exit_day"] = exit_date.dt.strftime("%Y-%m-%d")
    outcomes["_horizon"] = horizon
    require(
        ((entry - origin).dt.days == 7).all(),
        "v5 model-conditioned outcomes entry is not t+1 week",
    )
    require(
        ((exit_date - entry).dt.days == 7 * (horizon - 1)).all(),
        "v5 model-conditioned outcomes exit horizon is invalid",
    )
    require(
        (horizon.ne(1) | entry.eq(exit_date)).all(),
        "v5 model-conditioned one-week outcome must include target-week open-to-close",
    )
    require(
        outcomes["execution_lag_weeks"].astype(int).eq(1).all()
        and outcomes["return_currency"].astype(str).eq("USD").all(),
        "v5 model-conditioned outcomes execution/currency contract mismatch",
    )
    require(
        set(outcomes["conditioning_model"].astype(str)) == set(models)
        and outcomes["state"].astype(str).isin(STATE_ORDER).all()
        and outcomes["asset"].astype(str).isin(V5_OUTCOME_ASSETS).all()
        and set(horizon).issubset(set(V5_OUTCOME_HORIZONS))
        and bool(set(horizon)),
        "v5 model-conditioned outcomes model/state/asset/horizon is invalid",
    )
    require(
        not outcomes.duplicated(
            [
                "conditioning_model",
                "_origin_position",
                "_horizon",
                "asset",
            ]
        ).any(),
        "v5 model-conditioned outcomes duplicate a model/origin/horizon/asset",
    )
    numeric_outcomes = outcomes.loc[:, ["forward_return", "max_drawdown"]].apply(
        pd.to_numeric,
        errors="raise",
    )
    require(
        np.isfinite(numeric_outcomes.to_numpy(dtype=float)).all(),
        "v5 model-conditioned outcomes contain non-finite values",
    )

    weekly = payload.get("weekly")
    require(
        isinstance(weekly, list) and bool(weekly),
        "v5 model-conditioned weekly forecasts are missing",
    )
    forecast_lookup: dict[tuple[str, str], tuple[int, str, int]] = {}
    episode_by_model = {name: -1 for name in models}
    prior_state_by_model: dict[str, str | None] = {name: None for name in models}
    weekly_days: set[str] = set()
    for position, week in enumerate(weekly):
        require(
            isinstance(week, Mapping),
            f"v5 model-conditioned weekly[{position}] is invalid",
        )
        origin_day = _market_date(week.get("date"))
        require(
            origin_day not in weekly_days,
            "v5 model-conditioned weekly origin dates duplicate",
        )
        weekly_days.add(origin_day)
        published = week.get("model_forecasts")
        require(
            isinstance(published, list) and len(published) == len(models),
            f"v5 model-conditioned weekly[{position}] forecasts are incomplete",
        )
        target_day = (
            pd.Timestamp(origin_day) + timedelta(weeks=1)
        ).date().isoformat()
        for model_position, name in enumerate(models):
            forecast = published[model_position]
            require(
                isinstance(forecast, Mapping) and forecast.get("model") == name,
                f"v5 model-conditioned weekly[{position}] model/order mismatch",
            )
            state = str(forecast.get("state"))
            require(
                state in STATE_ORDER,
                f"v5 model-conditioned weekly[{position}] state is invalid",
            )
            require(
                forecast.get("date") is not None
                and _market_date(forecast.get("date")) == target_day,
                f"v5 model-conditioned weekly[{position}] target is not t+1 week",
            )
            if prior_state_by_model[name] != state:
                episode_by_model[name] += 1
            prior_state_by_model[name] = state
            forecast_lookup[(origin_day, name)] = (
                position,
                state,
                episode_by_model[name],
            )

    for _, row in outcomes.iterrows():
        name = str(row["conditioning_model"])
        origin_day = str(row["_origin_day"])
        binding = forecast_lookup.get((origin_day, name))
        require(
            binding is not None,
            "v5 model-conditioned outcome has no weekly OOS forecast",
        )
        expected_position, expected_state, expected_episode = binding
        require(
            int(row["_origin_position"]) == expected_position,
            "v5 model-conditioned outcome origin position/weekly forecast mismatch",
        )
        require(
            str(row["state"]) == expected_state,
            "v5 model-conditioned outcome state/weekly OOS forecast mismatch",
        )
        require(
            int(row["_episode_id"]) == expected_episode,
            "v5 model-conditioned outcome episode/forecast sequence mismatch",
        )

    signature_columns = (
        "_origin_position",
        "_origin_day",
        "_entry_day",
        "_exit_day",
        "asset",
        "_horizon",
    )
    reference_signature: set[tuple[str, ...]] | None = None
    for name in models:
        selected = outcomes.loc[
            outcomes["conditioning_model"].astype(str).eq(name),
            list(signature_columns),
        ]
        signature = {
            tuple(str(row[field]) for field in signature_columns)
            for _, row in selected.iterrows()
        }
        if reference_signature is None:
            reference_signature = signature
        else:
            require(
                signature == reference_signature,
                "v5 model-conditioned outcome coverage differs by model",
            )

    expected_groups = {
        (state, asset, horizon_value)
        for state in STATE_ORDER
        for asset in V5_OUTCOME_ASSETS
        for horizon_value in V5_OUTCOME_HORIZONS
    }
    expected_resamples = int(
        _audit_v5_execution_parameters(payload)[
            "conditional_outcome_bootstrap_resamples"
        ]
    )
    for name in models:
        model_statistics = statistics.loc[
            statistics["conditioning_model"].astype(str).eq(name)
        ].drop(columns="conditioning_model")
        actual_groups = {
            (str(row.state), str(row.asset), int(row.horizon_weeks))
            for row in model_statistics.itertuples(index=False)
        }
        require(
            len(model_statistics) == len(expected_groups)
            and actual_groups == expected_groups,
            f"v5 model-conditioned statistics coverage is incomplete for {name}",
        )
        model_outcomes = outcomes.loc[
            outcomes["conditioning_model"].astype(str).eq(name)
        ].drop(
            columns=[
                "conditioning_model",
                "_origin_position",
                "_episode_id",
                "_origin_day",
                "_entry_day",
                "_exit_day",
                "_horizon",
            ]
        )
        recomputed = _recompute_v5_conditional_statistics(
            model_outcomes,
            model_statistics,
            expected_resamples=expected_resamples,
        )
        _audit_v5_recomputed_records(
            recomputed,
            model_statistics,
            keys=("state", "asset", "horizon_weeks"),
            fields=model_statistic_fields,
            context=f"v5 model-conditioned asset statistics {name}",
        )

    if investment_aligned and "conditional-asset-outcomes.csv" in frames:
        actual_outcomes = frames["conditional-asset-outcomes.csv"]
        signature_columns_actual = (
            "origin_position",
            "origin_date",
            "entry_date",
            "exit_date",
            "asset",
            "horizon_weeks",
        )
        require_columns(
            actual_outcomes,
            set(signature_columns_actual),
            "v5 matched actual conditional outcomes",
        )
        actual_signature = {
            tuple(str(row[field]) for field in signature_columns_actual)
            for _, row in actual_outcomes.loc[:, signature_columns_actual].iterrows()
        }
        for name in models:
            predicted = outcomes.loc[
                outcomes["conditioning_model"].astype(str).eq(name)
            ]
            predicted_signature = {
                (
                    str(row["_origin_position"]),
                    str(row["origin_date"]),
                    str(row["entry_date"]),
                    str(row["exit_date"]),
                    str(row["asset"]),
                    str(row["_horizon"]),
                )
                for _, row in predicted.iterrows()
            }
            require(
                predicted_signature == actual_signature,
                f"v5 model-conditioned outcomes are not matched to actual OOS origins for {name}",
            )

    return {
        "status": "verified",
        "models": len(models),
        "outcome_rows": len(outcomes),
        "statistics_rows": len(statistics),
        "bootstrap_resamples": expected_resamples,
    }


def _audit_v5_decision_shadow(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the decision shadow to its immutable execution spec and accounting."""

    research = payload.get("research")
    require(isinstance(research, Mapping), "v5 decision shadow research is invalid")
    raw = research.get("prospective_decision_shadow")
    if raw is None:
        return {"status": "legacy_absent"}
    require(isinstance(raw, Mapping), "v5 decision shadow is invalid")
    schema_version = raw.get("schema_version")
    require(
        schema_version
        in {
            "regime-prospective-decision-shadow/1",
            "regime-prospective-decision-shadow/2",
        },
        "v5 decision shadow schema is invalid",
    )
    investment_aligned = schema_version == "regime-prospective-decision-shadow/2"
    expected_shadow_fields = {
        "schema_version",
        "role",
        "spec",
        "execution_contract",
        "historical_reconstructed_shadow",
        "prospective_ledger",
    }
    if investment_aligned:
        expected_shadow_fields.add("current_signal")
    require(
        set(raw) == expected_shadow_fields
        and raw.get("role")
        == "research_only_no_forecast_or_champion_effect",
        "v5 decision shadow fields/role are invalid",
    )
    spec = raw.get("spec")
    execution = raw.get("execution_contract")
    historical = raw.get("historical_reconstructed_shadow")
    prospective = raw.get("prospective_ledger")
    require(
        isinstance(spec, Mapping)
        and isinstance(execution, Mapping)
        and isinstance(historical, Mapping)
        and isinstance(prospective, Mapping),
        "v5 decision shadow components are invalid",
    )
    expected_historical_fields = {
        "status",
        "evidence_track",
        "evidence_status",
        "minimum_evaluation_weeks",
        "strategies",
    }
    if investment_aligned:
        expected_historical_fields.update(
            {
                "first_tradable_week",
                "evaluation_start_week",
                "evaluation_end_week",
                "latest_target_weights",
                "allocation_policy",
            }
        )
    else:
        expected_historical_fields.add("first_tradable_at")
    require(
        set(historical) == expected_historical_fields,
        "v5 decision shadow historical fields are invalid",
    )
    spec_id = str(spec.get("spec_id"))
    expected_paths = {
        "spy-tlt-probability-shadow-v1": "config/decision-shadow.json",
        "spy-tlt-probability-shadow-v2": "config/decision-shadow-v2.json",
    }
    expected_path = expected_paths.get(spec_id)
    require(expected_path is not None, "v5 decision shadow spec id is invalid")
    require(
        spec.get("path") == expected_path,
        "v5 decision shadow spec id/path mismatch",
    )
    require(
        (investment_aligned and spec_id == "spy-tlt-probability-shadow-v2")
        or (
            not investment_aligned
            and spec_id == "spy-tlt-probability-shadow-v1"
        ),
        "v5 decision shadow schema/spec generation mismatch",
    )
    path = PROJECT_ROOT / expected_path
    require(
        path.is_file() and not path.is_symlink(),
        "v5 decision shadow spec is missing or non-regular",
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    require(
        isinstance(document, Mapping)
        and document.get("spec_id") == spec_id
        and document.get("role")
        == "research_only_no_forecast_or_champion_effect",
        "v5 decision shadow local spec identity mismatch",
    )
    require(
        spec.get("sha256") == canonical_json_sha256(document),
        "v5 decision shadow spec SHA-256 mismatch",
    )
    require(
        dict(execution) == dict(document.get("execution", {})),
        "v5 decision shadow execution differs from its spec",
    )

    strategies = historical.get("strategies")
    require(isinstance(strategies, Mapping), "v5 decision shadow strategies are invalid")
    expected_strategies = {
        "probability_shadow",
        "spy_buy_and_hold",
        "static_60_40",
        "vol_target_60_40",
    }
    require(
        set(strategies) == expected_strategies,
        "v5 decision shadow benchmark set is invalid",
    )
    weeks: set[int] = set()
    transaction_cost_field = (
        "transaction_cost_rate_sum"
        if investment_aligned
        else "total_transaction_cost"
    )
    expected_metric_fields = {
        "weeks",
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "certainty_equivalent_return",
        "maximum_drawdown",
        "annualized_turnover",
        "gross_cumulative_return",
        transaction_cost_field,
        "transaction_cost_bps",
    }
    for name, raw_metrics in strategies.items():
        require(
            isinstance(raw_metrics, Mapping),
            f"v5 decision shadow {name} metrics are invalid",
        )
        require(
            set(raw_metrics) == expected_metric_fields,
            f"v5 decision shadow {name} metric fields are invalid",
        )
        strategy_weeks = raw_metrics.get("weeks")
        require(
            isinstance(strategy_weeks, int) and strategy_weeks >= 0,
            f"v5 decision shadow {name} weeks are invalid",
        )
        weeks.add(strategy_weeks)
        for field in (transaction_cost_field, "transaction_cost_bps"):
            value = raw_metrics.get(field)
            require(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and np.isfinite(float(value))
                and float(value) >= 0.0,
                f"v5 decision shadow {name}.{field} is invalid",
            )
        annualized_turnover = raw_metrics.get("annualized_turnover")
        if strategy_weeks == 0:
            require(
                annualized_turnover is None
                and np.isclose(
                    float(raw_metrics[transaction_cost_field]),
                    0.0,
                    atol=1e-12,
                ),
                f"v5 decision shadow {name} empty turnover/cost is invalid",
            )
        else:
            require(
                isinstance(annualized_turnover, (int, float))
                and not isinstance(annualized_turnover, bool)
                and np.isfinite(float(annualized_turnover))
                and float(annualized_turnover) >= 0.0,
                f"v5 decision shadow {name}.annualized_turnover is invalid",
            )
        net = raw_metrics.get("cumulative_return")
        gross = raw_metrics.get("gross_cumulative_return")
        if net is not None and gross is not None:
            require(
                np.isfinite(float(net))
                and np.isfinite(float(gross))
                and float(net) <= float(gross) + 1e-12,
                f"v5 decision shadow {name} net return exceeds gross return",
            )
    require(len(weeks) == 1, "v5 decision shadow strategies use unmatched weeks")
    common_weeks = next(iter(weeks))

    if investment_aligned:
        evaluation_start_raw = historical.get("evaluation_start_week")
        evaluation_end_raw = historical.get("evaluation_end_week")
        require(
            (evaluation_start_raw is None) == (evaluation_end_raw is None),
            "v5 decision shadow evaluation bounds differ in nullability",
        )
        if common_weeks == 0:
            require(
                evaluation_start_raw is None and evaluation_end_raw is None,
                "v5 decision shadow empty evaluation has date bounds",
            )
        else:
            require(
                evaluation_start_raw is not None and evaluation_end_raw is not None,
                "v5 decision shadow populated evaluation lacks date bounds",
            )
            evaluation_start = datetime.fromisoformat(
                str(evaluation_start_raw)
            ).date()
            evaluation_end = datetime.fromisoformat(str(evaluation_end_raw)).date()
            evaluation_days = (evaluation_end - evaluation_start).days
            require(
                evaluation_days >= 0
                and evaluation_days % 7 == 0
                and evaluation_days // 7 + 1 == common_weeks,
                "v5 decision shadow evaluation bounds do not match weeks",
            )
        allocation = historical.get("allocation_policy")
        require(
            isinstance(allocation, Mapping),
            "v5 decision shadow allocation policy is invalid",
        )
        require(
            set(allocation)
            == {
                "method",
                "assets",
                "forecast_model",
                "latest_signal_origin",
                "latest_target_weights",
            }
            and allocation.get("method")
            == "probability_weighted_state_portfolios"
            and allocation.get("assets") == ["SPY", "TLT"],
            "v5 decision shadow allocation policy fields are invalid",
        )
        weights = allocation.get("latest_target_weights")
        if weights is not None:
            require(
                isinstance(weights, Mapping) and set(weights) == {"SPY", "TLT"},
                "v5 decision shadow latest target weights are invalid",
            )
            values = np.asarray([weights["SPY"], weights["TLT"]], dtype=float)
            require(
                np.isfinite(values).all()
                and (values >= 0.0).all()
                and (values <= 1.0).all()
                and np.isclose(float(values.sum()), 1.0, atol=1e-8),
                "v5 decision shadow latest target weights are invalid",
            )
            require(
                historical.get("latest_target_weights") == weights,
                "v5 decision shadow latest target weight copies differ",
            )

    expected_prospective_fields = {
        "status",
        "evidence_track",
        "ledger_entry_count",
        "realized_evaluation_count",
        "affects_official_forecast",
        "affects_champion_selection",
    }
    if investment_aligned:
        expected_prospective_fields.update(
            {
                "pending_evaluation_count",
                "unresolved_due_evaluation_count",
                "partial_evaluation_count",
                "evaluation_manifest_sha256",
                "performance",
            }
        )
    require(
        set(prospective) == expected_prospective_fields,
        "v5 decision shadow prospective ledger fields are invalid",
    )
    require(
        prospective.get("evidence_track") == "operational_oos"
        and prospective.get("affects_official_forecast") is False
        and prospective.get("affects_champion_selection") is False,
        "v5 decision shadow prospective isolation is invalid",
    )
    ledger_entries = prospective.get("ledger_entry_count")
    realized_evaluations = prospective.get("realized_evaluation_count")
    require(
        isinstance(ledger_entries, int)
        and not isinstance(ledger_entries, bool)
        and ledger_entries >= 0
        and isinstance(realized_evaluations, int)
        and not isinstance(realized_evaluations, bool)
        and 0 <= realized_evaluations <= ledger_entries,
        "v5 decision shadow prospective ledger counts are invalid",
    )
    if investment_aligned:
        pending_evaluations = prospective.get("pending_evaluation_count")
        unresolved_evaluations = prospective.get(
            "unresolved_due_evaluation_count"
        )
        partial_evaluations = prospective.get("partial_evaluation_count")
        require(
            all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
                for value in (
                    pending_evaluations,
                    unresolved_evaluations,
                    partial_evaluations,
                )
            )
            and pending_evaluations
            + unresolved_evaluations
            + realized_evaluations
            + partial_evaluations
            == ledger_entries,
            "v5 decision shadow prospective maturity counts are invalid",
        )
        evaluation_hash = prospective.get("evaluation_manifest_sha256")
        require(
            isinstance(evaluation_hash, str)
            and len(evaluation_hash) == 64
            and all(character in "0123456789abcdef" for character in evaluation_hash),
            "v5 decision shadow evaluation manifest hash is invalid",
        )
        expected_status = (
            "completed"
            if ledger_entries > 0 and realized_evaluations == ledger_entries
            else "pending"
            if pending_evaluations == ledger_entries
            else "partial"
        )
        require(
            prospective.get("status") == expected_status,
            "v5 decision shadow prospective status/counts are inconsistent",
        )
        performance = prospective.get("performance")
        expected_performance_fields = {
            "status",
            "weeks",
            "gross_cumulative_return",
            "net_cumulative_return",
            "turnover_sum",
            "transaction_cost_rate_sum",
            "transaction_cost_bps",
            "forecast_hit_count",
            "forecast_accuracy",
            "actual_state_counts",
        }
        require(
            isinstance(performance, Mapping)
            and set(performance) == expected_performance_fields
            and performance.get("weeks") == realized_evaluations,
            "v5 decision shadow prospective performance fields are invalid",
        )
        expected_performance_status = (
            "completed" if expected_status == "completed" else
            "pending" if expected_status == "pending" else "partial"
        )
        require(
            performance.get("status") == expected_performance_status,
            "v5 decision shadow prospective performance status is invalid",
        )
        if realized_evaluations == 0:
            require(
                all(
                    performance.get(field) is None
                    for field in expected_performance_fields - {"status", "weeks"}
                ),
                "v5 decision shadow empty prospective performance is not null",
            )
        else:
            numeric_fields = (
                "gross_cumulative_return",
                "net_cumulative_return",
                "turnover_sum",
                "transaction_cost_rate_sum",
                "transaction_cost_bps",
                "forecast_accuracy",
            )
            require(
                all(
                    isinstance(performance.get(field), (int, float))
                    and not isinstance(performance.get(field), bool)
                    and np.isfinite(float(performance[field]))
                    for field in numeric_fields
                ),
                "v5 decision shadow prospective performance values are invalid",
            )
            hits = performance.get("forecast_hit_count")
            counts = performance.get("actual_state_counts")
            require(
                isinstance(hits, int)
                and not isinstance(hits, bool)
                and 0 <= hits <= realized_evaluations
                and isinstance(counts, Mapping)
                and set(counts) == {"risk_on", "transition", "risk_off"}
                and all(
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 0
                    for value in counts.values()
                )
                and sum(counts.values()) == realized_evaluations
                and np.isclose(
                    float(performance["forecast_accuracy"]),
                    hits / realized_evaluations,
                )
                and np.isclose(
                    float(performance["transaction_cost_bps"]), 10.0
                )
                and np.isclose(
                    float(performance["transaction_cost_rate_sum"]),
                    float(performance["turnover_sum"]) / 1_000.0,
                )
                and float(performance["net_cumulative_return"])
                <= float(performance["gross_cumulative_return"]) + 1e-12,
                "v5 decision shadow prospective performance is inconsistent",
            )
        forecast_ledger = payload.get("forecast", {}).get("prospective_ledger")
        if (
            isinstance(forecast_ledger, Mapping)
            and forecast_ledger.get("schema_version")
            == "regime-prospective-ledger-summary/2"
        ):
            expected_copy = {
                "status": (
                    "pending"
                    if forecast_ledger.get("status") == "empty"
                    else forecast_ledger.get("status")
                ),
                "ledger_entry_count": forecast_ledger.get("entry_count"),
                "pending_evaluation_count": forecast_ledger.get(
                    "pending_evaluation_count"
                ),
                "unresolved_due_evaluation_count": forecast_ledger.get(
                    "unresolved_due_evaluation_count"
                ),
                "realized_evaluation_count": forecast_ledger.get(
                    "realized_evaluation_count"
                ),
                "partial_evaluation_count": forecast_ledger.get(
                    "partial_evaluation_count"
                ),
                "evaluation_manifest_sha256": forecast_ledger.get(
                    "evaluation_manifest_sha256"
                ),
                "performance": forecast_ledger.get("performance"),
            }
            require(
                all(prospective.get(key) == value for key, value in expected_copy.items()),
                "v5 decision shadow prospective summary differs from forecast",
            )
    elif prospective.get("status") == "awaiting_realized_targets":
        require(
            ledger_entries == 0 and realized_evaluations == 0,
            "v5 decision shadow empty ledger status is inconsistent",
        )
    elif prospective.get("status") == "ledger_recorded_outcomes_pending":
        require(
            ledger_entries > 0 and realized_evaluations < ledger_entries,
            "v5 decision shadow pending ledger status is inconsistent",
        )
    else:
        require(False, "v5 decision shadow prospective ledger status is invalid")

    current_action = "legacy_weekly_close_contract"
    if investment_aligned:
        current_signal = raw.get("current_signal")
        require(
            isinstance(current_signal, Mapping)
            and set(current_signal)
            == {
                "origin_date",
                "target_week",
                "scheduled_entry_at",
                "decision_at",
                "forecast_model",
                "status",
                "action",
            },
            "v5 decision shadow current signal is invalid",
        )
        origin_date = datetime.fromisoformat(
            str(current_signal["origin_date"])
        ).date()
        target_week = datetime.fromisoformat(
            str(current_signal["target_week"])
        ).date()
        require(
            target_week == origin_date + timedelta(days=7),
            "v5 decision shadow current target week is invalid",
        )
        scheduled_entry = pd.Timestamp(current_signal["scheduled_entry_at"])
        decision_at = pd.Timestamp(current_signal["decision_at"])
        require(
            scheduled_entry.tzinfo is not None
            and decision_at.tzinfo is not None,
            "v5 decision shadow current signal timestamps require timezones",
        )
        scheduled_entry = scheduled_entry.tz_convert("UTC")
        decision_at = decision_at.tz_convert("UTC")
        expected = (
            ("scheduled", "trade_at_scheduled_open")
            if decision_at < scheduled_entry
            else ("missed_entry", "no_trade")
        )
        require(
            (current_signal.get("status"), current_signal.get("action"))
            == expected,
            "v5 decision shadow current signal timing is inconsistent",
        )
        selection = payload.get("selection")
        forecast_envelope = payload.get("forecast")
        weekly_rows = payload.get("weekly")
        require(
            isinstance(selection, Mapping)
            and isinstance(forecast_envelope, Mapping)
            and isinstance(weekly_rows, Sequence)
            and not isinstance(weekly_rows, (str, bytes))
            and len(weekly_rows) > 0,
            "v5 decision shadow payload binding inputs are invalid",
        )
        latest_week = weekly_rows[-1]
        require(
            isinstance(latest_week, Mapping),
            "v5 decision shadow latest weekly row is invalid",
        )
        operating_champion = selection.get("operating_champion")
        model_forecasts = latest_week.get("model_forecasts")
        require(
            isinstance(operating_champion, str)
            and operating_champion
            and isinstance(model_forecasts, Sequence)
            and not isinstance(model_forecasts, (str, bytes)),
            "v5 decision shadow operating forecast binding is invalid",
        )
        operating_rows = [
            row
            for row in model_forecasts
            if isinstance(row, Mapping)
            and row.get("model") == operating_champion
        ]
        require(
            len(operating_rows) == 1,
            "v5 decision shadow operating forecast is not unique",
        )
        operating_row = operating_rows[0]
        latest_origin = datetime.fromisoformat(str(latest_week.get("date"))).date()
        operating_target = datetime.fromisoformat(
            str(operating_row.get("date"))
        ).date()
        require(
            origin_date == latest_origin
            and target_week == operating_target
            and current_signal.get("forecast_model") == operating_champion
            and allocation.get("forecast_model") == operating_champion
            and allocation.get("latest_signal_origin")
            == latest_origin.isoformat(),
            "v5 decision shadow current signal/allocation binding is invalid",
        )
        forecast_origin = pd.Timestamp(forecast_envelope.get("origin_at"))
        forecast_target = pd.Timestamp(forecast_envelope.get("target_at"))
        require(
            forecast_origin.tzinfo is not None
            and forecast_target.tzinfo is not None
            and forecast_origin.tz_convert("America/New_York").date()
            == latest_origin
            and forecast_target.tz_convert("America/New_York").date()
            == operating_target,
            "v5 decision shadow dates differ from forecast envelope",
        )
        raw_forecast_decision = forecast_envelope.get("decision_at")
        if raw_forecast_decision is None:
            meta = payload.get("meta")
            require(
                isinstance(meta, Mapping),
                "v5 decision shadow expired decision lacks payload meta",
            )
            expected_decision = pd.Timestamp(meta.get("generated_at"))
        else:
            expected_decision = pd.Timestamp(raw_forecast_decision)
        require(
            expected_decision.tzinfo is not None
            and decision_at == expected_decision.tz_convert("UTC"),
            "v5 decision shadow decision differs from forecast envelope",
        )
        probabilities = operating_row.get("probabilities")
        weight_mapping = document.get("probability_weight_mapping")
        require(
            isinstance(probabilities, Mapping)
            and set(probabilities) == set(STATE_ORDER)
            and isinstance(weight_mapping, Mapping),
            "v5 decision shadow operating probability mapping is invalid",
        )
        expected_weights = {
            asset: sum(
                float(probabilities[state])
                * float(weight_mapping[state][asset])
                for state in STATE_ORDER
            )
            for asset in ("SPY", "TLT")
        }
        require(
            weights is not None
            and all(
                np.isclose(
                    float(weights[asset]),
                    expected_weights[asset],
                    atol=1e-8,
                )
                for asset in ("SPY", "TLT")
            ),
            "v5 decision shadow latest weights differ from operating forecast",
        )
        current_action = str(current_signal.get("action"))
    return {
        "status": "verified",
        "spec_id": spec_id,
        "weeks": common_weeks,
        "current_signal_action": current_action,
        "ledger_entries": ledger_entries,
        "realized_evaluations": realized_evaluations,
    }


def _v5_serialized_probability_rows_are_valid(values: pd.DataFrame) -> bool:
    matrix = values.to_numpy(dtype=float)
    return bool(
        np.isfinite(matrix).all()
        and ((values >= 0.0) & (values <= 1.0)).all().all()
        and np.allclose(
            values.sum(axis=1),
            1.0,
            atol=V5_SERIALIZED_SIMPLEX_ATOL,
            rtol=0.0,
        )
    )


def _audit_v5_evidence(
    payload: Mapping[str, Any], artifacts: Path
) -> tuple[dict[str, Any], pd.DataFrame]:
    contracts = payload["model"]["evidence_artifacts"]
    rows: dict[str, Any] = {}
    membership_frame: pd.DataFrame | None = None
    forecast_frame: pd.DataFrame | None = None
    for key in ("state_membership_history", "weekly_state_forecasts"):
        metadata = contracts[key]
        context = f"payload.model.evidence_artifacts.{key}"
        path = _v5_artifact_path(artifacts, metadata["path"], context=context)
        require(
            file_sha256(path) == str(metadata["sha256"]),
            f"{context} SHA-256 mismatch",
        )
        frame = pd.read_csv(path)
        require(
            len(frame) == int(metadata["row_count"]),
            f"{context} row_count mismatch",
        )
        rows[key] = {"rows": len(frame), "sha256": str(metadata["sha256"])}
        if key == "state_membership_history":
            membership_frame = frame
        else:
            forecast_frame = frame
    require(membership_frame is not None, "v5 membership evidence is missing")
    require_columns(
        membership_frame,
        {"date", "state", "m_risk_on", "m_transition", "m_risk_off"},
        "v5 membership evidence",
    )
    membership_dates = pd.to_datetime(
        membership_frame["date"], utc=True, errors="raise"
    )
    require(
        membership_dates.is_monotonic_increasing
        and not membership_dates.duplicated().any(),
        "v5 membership evidence dates are not unique/increasing",
    )
    memberships = membership_frame.loc[
        :, ["m_risk_on", "m_transition", "m_risk_off"]
    ].apply(pd.to_numeric, errors="raise")
    require(
        np.isfinite(memberships.to_numpy(dtype=float)).all()
        and ((memberships >= 0.0) & (memberships <= 1.0)).all().all()
        and np.allclose(memberships.sum(axis=1), 1.0, atol=1e-8, rtol=0.0),
        "v5 membership evidence is not a probability simplex",
    )
    require(forecast_frame is not None, "v5 weekly forecast evidence is missing")
    require_columns(
        forecast_frame,
        {
            "origin_date",
            "current_state",
            "current_m_risk_on",
            "current_m_transition",
            "current_m_risk_off",
            "target_date",
            "model",
            "next_p_risk_on",
            "next_p_transition",
            "next_p_risk_off",
            "fallback",
            "fallback_reason",
        },
        "v5 weekly forecast evidence",
    )
    weekly = payload["weekly"]
    require(
        len(forecast_frame) == len(weekly),
        "v5 weekly forecast evidence/payload row count mismatch",
    )
    require(
        not forecast_frame["origin_date"].duplicated().any(),
        "v5 weekly forecast evidence duplicates an origin",
    )
    current_columns = [f"current_m_{state}" for state in STATE_ORDER]
    next_columns = [f"next_p_{state}" for state in STATE_ORDER]
    current_values = forecast_frame.loc[:, current_columns].apply(
        pd.to_numeric, errors="raise"
    )
    next_values = forecast_frame.loc[:, next_columns].apply(
        pd.to_numeric, errors="raise"
    )
    for label, values in (("current", current_values), ("next", next_values)):
        require(
            _v5_serialized_probability_rows_are_valid(values),
            f"v5 weekly forecast evidence {label} probabilities are invalid",
        )
    fallback = boolean_series(
        forecast_frame["fallback"], "v5 weekly forecast evidence.fallback"
    )
    for index, week in enumerate(weekly):
        row = forecast_frame.iloc[index]
        require(
            _market_date(row["origin_date"]) == str(week["date"]),
            f"v5 weekly[{index}] origin/evidence mismatch",
        )
        require(
            _market_date(row["target_date"]) == str(week["next_week"]["date"]),
            f"v5 weekly[{index}] target/evidence mismatch",
        )
        require(
            str(row["current_state"]) == str(week["current"]["state"]),
            f"v5 weekly[{index}] current state/evidence mismatch",
        )
        require(
            str(row["model"]) == str(week["next_week"]["model"]),
            f"v5 weekly[{index}] model/evidence mismatch",
        )
        require(
            bool(fallback.iloc[index]) == bool(week["next_week"]["fallback"]),
            f"v5 weekly[{index}] fallback/evidence mismatch",
        )
        for state in STATE_ORDER:
            require(
                np.isclose(
                    float(row[f"current_m_{state}"]),
                    float(week["current"]["memberships"][state]),
                    atol=1e-8,
                    rtol=0.0,
                ),
                f"v5 weekly[{index}] current {state} evidence mismatch",
            )
            require(
                np.isclose(
                    float(row[f"next_p_{state}"]),
                    float(week["next_week"]["probabilities"][state]),
                    atol=1e-8,
                    rtol=0.0,
                ),
                f"v5 weekly[{index}] next {state} evidence mismatch",
            )
    return rows, membership_frame


def _v5_duration_spells(states: Sequence[str]) -> list[dict[str, Any]]:
    require(bool(states), "v5 duration state history is empty")
    spells: list[dict[str, Any]] = []
    start = 0
    for position in range(1, len(states) + 1):
        departed = position < len(states) and states[position] != states[start]
        final = position == len(states)
        if not departed and not final:
            continue
        spells.append(
            {
                "state": str(states[start]),
                "duration_weeks": int(position - start),
                "event_observed": bool(departed),
            }
        )
        start = position
    return spells


def _v5_duration_km(
    spells: Sequence[Mapping[str, Any]],
    *,
    state: str,
) -> list[dict[str, Any]]:
    selected = [spell for spell in spells if str(spell["state"]) == state]
    require(selected, f"v5 duration has no spells for {state}")
    durations = np.asarray(
        [int(spell["duration_weeks"]) for spell in selected], dtype=int
    )
    observed = np.asarray(
        [bool(spell["event_observed"]) for spell in selected], dtype=bool
    )
    survival = 1.0
    rows: list[dict[str, Any]] = []
    for duration in sorted(set(durations.tolist())):
        at_risk = int(np.count_nonzero(durations >= duration))
        at_time = durations == duration
        events = int(np.count_nonzero(at_time & observed))
        if events:
            survival *= 1.0 - events / at_risk
        rows.append(
            {
                "duration_weeks": int(duration),
                "survival": float(survival),
                "events": events,
            }
        )
    return rows


def _v5_duration_survival_at(
    km: Sequence[Mapping[str, Any]], elapsed_weeks: int
) -> float:
    eligible = [
        row for row in km if int(row["duration_weeks"]) <= elapsed_weeks
    ]
    return 1.0 if not eligible else float(eligible[-1]["survival"])


def _recompute_v5_duration(
    states: Sequence[str],
    *,
    as_of: str,
) -> dict[str, Any]:
    spells = _v5_duration_spells(states)
    current = spells[-1]
    state = str(current["state"])
    elapsed = int(current["duration_weeks"])
    selected = [spell for spell in spells if str(spell["state"]) == state]
    completed = int(sum(bool(spell["event_observed"]) for spell in selected))
    censored = int(sum(not bool(spell["event_observed"]) for spell in selected))
    base: dict[str, Any] = {
        "as_of": as_of,
        "method": "state_specific_kaplan_meier",
        "state": state,
        "elapsed_weeks": elapsed,
        "episodes": int(len(selected)),
        "completed_spells": completed,
        "censored_spells": censored,
        "minimum_completed_spells": 5,
        "restriction_weeks": 52,
    }
    if completed < 5:
        return {
            **base,
            "status": "insufficient_history",
            "conditional_survival": {"4w": None, "13w": None},
            "departure_probability": {"4w": None, "13w": None},
            "median_remaining_weeks": None,
            "restricted_mean_remaining_weeks": None,
        }
    km = _v5_duration_km(spells, state=state)
    conditioning_time = elapsed - 1
    denominator = _v5_duration_survival_at(km, conditioning_time)
    if not np.isfinite(denominator) or denominator <= 0.0:
        return {
            **base,
            "status": "unavailable",
            "conditional_survival": {"4w": None, "13w": None},
            "departure_probability": {"4w": None, "13w": None},
            "median_remaining_weeks": None,
            "restricted_mean_remaining_weeks": None,
        }
    conditional_survival = {
        f"{horizon}w": float(
            np.clip(
                _v5_duration_survival_at(km, conditioning_time + horizon)
                / denominator,
                0.0,
                1.0,
            )
        )
        for horizon in (4, 13)
    }
    event_times = [
        int(row["duration_weeks"]) for row in km if int(row["events"]) > 0
    ]
    maximum_search = max(0, max(event_times) - conditioning_time) if event_times else 0
    median_remaining: int | None = None
    for remaining in range(1, maximum_search + 1):
        ratio = (
            _v5_duration_survival_at(km, conditioning_time + remaining)
            / denominator
        )
        if ratio <= 0.5 + 1e-12:
            median_remaining = remaining
            break
    rmst = sum(
        float(
            np.clip(
                _v5_duration_survival_at(km, conditioning_time + remaining)
                / denominator,
                0.0,
                1.0,
            )
        )
        for remaining in range(52)
    )
    return {
        **base,
        "status": "ok",
        "conditional_survival": conditional_survival,
        "departure_probability": {
            key: float(1.0 - value)
            for key, value in conditional_survival.items()
        },
        "median_remaining_weeks": median_remaining,
        "restricted_mean_remaining_weeks": float(rmst),
    }


def _audit_v5_duration_interval(
    value: object,
    *,
    context: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[float, float] | None:
    if value is None:
        return None
    require(isinstance(value, Mapping), f"{context} must be an interval or null")
    require(set(value) == {"lower", "upper"}, f"{context} fields mismatch")
    try:
        lower = float(value["lower"])
        upper = float(value["upper"])
    except (TypeError, ValueError) as exc:
        raise AuditFailure(f"{context} is not numeric") from exc
    require(
        np.isfinite(lower) and np.isfinite(upper) and lower <= upper,
        f"{context} bounds are invalid",
    )
    if minimum is not None:
        require(lower >= minimum, f"{context} lower bound is invalid")
    if maximum is not None:
        require(upper <= maximum, f"{context} upper bound is invalid")
    return lower, upper


def _audit_v5_duration_ci(
    duration: Mapping[str, Any],
    *,
    expected_resamples: int,
    context: str,
) -> None:
    bootstrap = duration.get("bootstrap")
    require(isinstance(bootstrap, Mapping), f"{context}.bootstrap must be an object")
    require(
        set(bootstrap) == {"unit", "resamples", "valid_resamples", "seed", "interval"},
        f"{context}.bootstrap fields mismatch",
    )
    require(bootstrap["unit"] == "episode", f"{context}.bootstrap unit mismatch")
    require(
        int(bootstrap["resamples"]) == expected_resamples,
        f"{context}.bootstrap resamples/execution mismatch",
    )
    valid_resamples = int(bootstrap["valid_resamples"])
    require(
        0 <= valid_resamples <= expected_resamples,
        f"{context}.bootstrap valid_resamples invalid",
    )
    require(int(bootstrap["seed"]) == 17, f"{context}.bootstrap seed mismatch")
    require(
        np.isclose(float(bootstrap["interval"]), 0.95, atol=1e-12, rtol=0.0),
        f"{context}.bootstrap interval mismatch",
    )
    if expected_resamples == 0:
        require(valid_resamples == 0, f"{context}.bootstrap valid_resamples must be zero")
    ci95 = duration.get("ci95")
    if duration.get("status") != "ok":
        require(ci95 is None, f"{context}.ci95 must be null without an estimate")
        return
    require(isinstance(ci95, Mapping), f"{context}.ci95 must be an object")
    require(
        set(ci95)
        == {
            "conditional_survival",
            "departure_probability",
            "median_remaining_weeks",
            "restricted_mean_remaining_weeks",
        },
        f"{context}.ci95 fields mismatch",
    )
    stay = ci95["conditional_survival"]
    depart = ci95["departure_probability"]
    require(isinstance(stay, Mapping), f"{context}.ci95 survival must be an object")
    require(isinstance(depart, Mapping), f"{context}.ci95 departure must be an object")
    require(set(stay) == {"4w", "13w"}, f"{context}.ci95 survival keys mismatch")
    require(set(depart) == {"4w", "13w"}, f"{context}.ci95 departure keys mismatch")
    for horizon in ("4w", "13w"):
        stay_interval = _audit_v5_duration_interval(
            stay[horizon],
            context=f"{context}.ci95.conditional_survival.{horizon}",
            minimum=0.0,
            maximum=1.0,
        )
        depart_interval = _audit_v5_duration_interval(
            depart[horizon],
            context=f"{context}.ci95.departure_probability.{horizon}",
            minimum=0.0,
            maximum=1.0,
        )
        require(
            (stay_interval is None) == (depart_interval is None),
            f"{context}.ci95 {horizon} complement nullability mismatch",
        )
        if stay_interval is not None and depart_interval is not None:
            require(
                np.isclose(
                    depart_interval[0],
                    1.0 - stay_interval[1],
                    atol=5e-8,
                    rtol=0.0,
                )
                and np.isclose(
                    depart_interval[1],
                    1.0 - stay_interval[0],
                    atol=5e-8,
                    rtol=0.0,
                ),
                f"{context}.ci95 {horizon} complement mismatch",
            )
    median_interval = _audit_v5_duration_interval(
        ci95["median_remaining_weeks"],
        context=f"{context}.ci95.median_remaining_weeks",
        minimum=0.0,
    )
    rmst_interval = _audit_v5_duration_interval(
        ci95["restricted_mean_remaining_weeks"],
        context=f"{context}.ci95.restricted_mean_remaining_weeks",
        minimum=0.0,
        maximum=52.0,
    )
    if expected_resamples == 0:
        require(
            all(stay[horizon] is None and depart[horizon] is None for horizon in ("4w", "13w"))
            and ci95["median_remaining_weeks"] is None
            and ci95["restricted_mean_remaining_weeks"] is None,
            f"{context}.ci95 must be null-valued when bootstrap is disabled",
        )
        return
    minimum_valid = max(1, int(np.ceil(expected_resamples * 0.8)))
    enough_valid = valid_resamples >= minimum_valid
    require(
        all(
            (stay[horizon] is not None) == enough_valid
            and (depart[horizon] is not None) == enough_valid
            for horizon in ("4w", "13w")
        )
        and (rmst_interval is not None) == enough_valid,
        f"{context}.ci95 valid-resample linkage mismatch",
    )
    if median_interval is not None:
        require(
            enough_valid,
            f"{context}.ci95 median interval lacks valid resamples",
        )


def _audit_v5_duration(
    payload: Mapping[str, Any],
    membership: pd.DataFrame,
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    require_columns(membership, {"date", "state"}, "v5 duration membership evidence")
    dates = pd.to_datetime(membership["date"], utc=True, errors="raise")
    local_dates = dates.dt.tz_convert("America/New_York").dt.date
    require(
        local_dates.is_monotonic_increasing and not local_dates.duplicated().any(),
        "v5 duration membership dates are not unique/increasing",
    )
    require(
        all(
            (current - previous).days == 7
            for previous, current in zip(
                local_dates.iloc[:-1], local_dates.iloc[1:], strict=True
            )
        ),
        "v5 duration membership history is not weekly",
    )
    states = membership["state"].astype(str).tolist()
    require(
        set(states).issubset(STATE_ORDER),
        "v5 duration membership states are invalid",
    )
    duration_resamples = int(execution["duration_bootstrap_resamples"])
    weekly = payload["weekly"]
    latest_date = str(weekly[-1]["date"])
    for index, week in enumerate(weekly):
        context = f"v5 weekly[{index}].duration_context"
        duration = week.get("duration_context")
        require(isinstance(duration, Mapping), f"{context} must be an object")
        cutoff_value = week.get("data_as_of", week.get("date"))
        cutoff = pd.Timestamp(cutoff_value)
        cutoff_date = cutoff.date()
        if cutoff.tzinfo is not None:
            cutoff_date = cutoff.tz_convert(
                "America/New_York"
            ).date()
            cutoff_utc = cutoff.tz_convert("UTC")
        else:
            cutoff_utc = cutoff.tz_localize("UTC")
        require(
            str(week["date"]) == cutoff_date.isoformat(),
            f"v5 weekly[{index}] date/data_as_of mismatch",
        )
        eligible_positions = [
            position
            for position, observed_at in enumerate(dates)
            if observed_at <= cutoff_utc
        ]
        require(eligible_positions, f"{context} cutoff precedes membership history")
        last_position = eligible_positions[-1]
        as_of = local_dates.iloc[last_position].isoformat()
        require(as_of == str(week["date"]), f"{context} as-of evidence mismatch")
        expected = _recompute_v5_duration(states[: last_position + 1], as_of=as_of)
        for field in (
            "as_of",
            "method",
            "state",
            "elapsed_weeks",
            "episodes",
            "completed_spells",
            "censored_spells",
            "minimum_completed_spells",
            "status",
            "median_remaining_weeks",
            "restricted_mean_remaining_weeks",
            "restriction_weeks",
        ):
            _require_v5_recomputed_value(
                expected[field],
                duration.get(field),
                context=f"{context}.{field}",
            )
        require(
            str(week["current"]["state"]) == expected["state"],
            f"{context} current state/evidence mismatch",
        )
        for block_name in ("conditional_survival", "departure_probability"):
            actual_block = duration.get(block_name)
            require(
                isinstance(actual_block, Mapping),
                f"{context}.{block_name} must be an object",
            )
            require(
                set(actual_block) == {"4w", "13w"},
                f"{context}.{block_name} keys mismatch",
            )
            for horizon in ("4w", "13w"):
                _require_v5_recomputed_value(
                    expected[block_name][horizon],
                    actual_block[horizon],
                    context=f"{context}.{block_name}.{horizon}",
                )
        expected_resamples = (
            duration_resamples if str(week["date"]) == latest_date else 0
        )
        _audit_v5_duration_ci(
            duration,
            expected_resamples=expected_resamples,
            context=context,
        )
    return {"weeks": len(weekly), "latest_bootstrap_resamples": duration_resamples}


def _v5_fx_correction_start(available_at: pd.Timestamp) -> pd.Timestamp:
    """Return the first Friday 16:00 ET cutoff at/after a correction."""

    local = pd.Timestamp(available_at).tz_convert("America/New_York")
    candidate_date = local.date() + timedelta(
        days=(4 - local.weekday()) % 7
    )
    candidate = (
        pd.Timestamp(candidate_date)
        .tz_localize("America/New_York")
        + timedelta(hours=16)
    )
    if candidate < local:
        candidate = (
            pd.Timestamp(candidate_date + timedelta(weeks=1))
            .tz_localize("America/New_York")
            + timedelta(hours=16)
        )
    return candidate


def _audit_v5_fx_correction_quarantine(
    coverage: pd.DataFrame,
    *,
    correction_events: Sequence[pd.Timestamp] | None = None,
) -> dict[str, Any]:
    """Independently rebuild the 27-origin H.10 correction quarantine."""

    required = {
        "archive_correction_quarantined",
        "archive_correction_available_at",
        "archive_correction_quarantine_until_week",
    }
    require_columns(coverage, required, "v5 FX correction quarantine")
    quarantined = boolean_series(
        coverage["archive_correction_quarantined"],
        "v5 FX correction quarantine flag",
    )
    available_raw = coverage["archive_correction_available_at"]
    until_raw = coverage["archive_correction_quarantine_until_week"]
    available = pd.to_datetime(available_raw, utc=True, errors="coerce")
    until = pd.to_datetime(until_raw, errors="coerce")
    require(
        available.notna().equals(available_raw.notna()),
        "v5 FX correction availability timestamp is invalid",
    )
    require(
        until.notna().equals(until_raw.notna()),
        "v5 FX correction quarantine end is invalid",
    )
    require(
        (available.notna() == quarantined).all()
        and (until.notna() == quarantined).all(),
        "v5 FX correction quarantine evidence/flag parity mismatch",
    )
    require(
        bool((until.dropna() == until.dropna().dt.normalize()).all()),
        "v5 FX correction quarantine end must be a calendar date",
    )
    if "feature_status" in coverage:
        require(
            coverage["feature_status"].astype(str).eq(
                "correction_quarantine"
            ).equals(quarantined),
            "v5 FX correction quarantine feature status mismatch",
        )

    evidence = pd.DataFrame(
        {"available_at": available, "until_week": until}
    ).loc[quarantined]
    evidence_events: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    for row in evidence.drop_duplicates().itertuples(index=False):
        correction = pd.Timestamp(row.available_at)
        start = _v5_fx_correction_start(correction)
        expected_end = start + timedelta(weeks=26)
        stored_end = pd.Timestamp(row.until_week)
        require(
            stored_end.date() == expected_end.date(),
            "v5 FX correction quarantine is not exactly 27 origins",
        )
        evidence_events.append((correction, start, expected_end))

    if correction_events is None:
        events = evidence_events
    else:
        resolved_events = [pd.Timestamp(value) for value in correction_events]
        require(
            all(value.tzinfo is not None for value in resolved_events),
            "v5 FX source correction availability must include a timezone",
        )
        normalized = [value.tz_convert("UTC") for value in resolved_events]
        require(
            normalized == sorted(set(normalized)),
            "v5 FX source correction availability is not unique/increasing",
        )
        events = [
            (
                correction,
                _v5_fx_correction_start(correction),
                _v5_fx_correction_start(correction) + timedelta(weeks=26),
            )
            for correction in normalized
        ]
        source_pairs = {
            (correction, end.date()) for correction, _, end in events
        }
        require(
            {
                (correction.tz_convert("UTC"), end.date())
                for correction, _, end in evidence_events
            }.issubset(source_pairs),
            "v5 FX coverage correction evidence is absent from source provenance",
        )

    intended_cutoffs = pd.DatetimeIndex(
        [
            (
                pd.Timestamp(week + timedelta(weeks=1))
                .tz_localize("America/New_York")
                + timedelta(hours=16)
            )
            for week in coverage.index
        ]
    )
    expected_quarantine = pd.Series(False, index=coverage.index)
    for _, start, end in events:
        expected_quarantine |= pd.Series(
            (intended_cutoffs >= start) & (intended_cutoffs <= end),
            index=coverage.index,
        )
    require(
        expected_quarantine.equals(quarantined),
        "v5 FX correction quarantine mask disagrees with 27-origin evidence",
    )

    for position in np.flatnonzero(quarantined.to_numpy()):
        cutoff = intended_cutoffs[int(position)]
        correction = pd.Timestamp(available.iloc[int(position)])
        start = _v5_fx_correction_start(correction)
        end = start + timedelta(weeks=26)
        require(
            start <= cutoff <= end,
            "v5 FX correction evidence does not cover its model cutoff",
        )
        active = [
            (candidate_correction, candidate_start, candidate_end)
            for candidate_correction, candidate_start, candidate_end in events
            if candidate_start <= cutoff <= candidate_end
        ]
        require(active, "v5 FX correction quarantine lacks an active window")
        selected = max(active, key=lambda item: (item[2], item[0]))
        require(
            correction.tz_convert("UTC") == selected[0].tz_convert("UTC")
            and end == selected[2],
            "v5 FX overlapping correction evidence is not bound to latest end",
        )

    return {
        "correction_events": len(events),
        "visible_correction_events": len(evidence_events),
        "quarantined_weeks": int(quarantined.sum()),
        "first_quarantined_cutoff": (
            intended_cutoffs[np.flatnonzero(quarantined.to_numpy())[0]]
            .date()
            .isoformat()
            if bool(quarantined.any())
            else None
        ),
        "last_quarantined_cutoff": (
            intended_cutoffs[np.flatnonzero(quarantined.to_numpy())[-1]]
            .date()
            .isoformat()
            if bool(quarantined.any())
            else None
        ),
    }


def _v5_fx_evaluation_origins(
    result: Any,
    cutoffs: pd.DatetimeIndex,
) -> tuple[dict[str, Any], Mapping[str, Sequence[str]]]:
    from regime_lab.analysis.fx_ablation import (
        align_fx_features_to_cutoffs,
        fx_ablation_variants,
    )

    variants = fx_ablation_variants(result.features)
    aligned = align_fx_features_to_cutoffs(result, cutoffs)
    required = list(variants["v4_plus_all_fx"])
    numeric = aligned.loc[:, required].apply(pd.to_numeric, errors="coerce")
    finite = pd.Series(
        np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1),
        index=aligned.index,
    )
    eligible = (
        finite
        & aligned["fx_observation_age_days"].eq(7)
        & ~aligned["fx_archive_correction_quarantined"].eq(True)
    )
    common_cutoffs = pd.DatetimeIndex(aligned.index[eligible])
    positions = cutoffs.get_indexer(common_cutoffs)

    supervised: list[dict[str, Any]] = []
    for cutoff, position in zip(common_cutoffs, positions, strict=True):
        require(position >= 0, "v5 FX eligible cutoff is absent from state evidence")
        target_position = int(position) + 1
        if target_position >= len(cutoffs):
            continue
        origin = pd.Timestamp(cutoffs[int(position)])
        target = pd.Timestamp(cutoffs[target_position])
        if (target.date() - origin.date()).days != 7:
            continue
        supervised.append(
            {
                "cutoff": pd.Timestamp(cutoff),
                "origin_date": origin,
                "target_date": target,
            }
        )

    rows: list[dict[str, Any]] = []
    for test in supervised:
        origin = pd.Timestamp(test["origin_date"])
        train = [
            row
            for row in supervised
            if pd.Timestamp(row["target_date"]) < origin
        ]
        purged = [
            row
            for row in supervised
            if pd.Timestamp(row["target_date"]) == origin
        ]
        if len(train) < 104 or len(purged) != 1:
            continue
        rows.append(
            {
                "origin_date": origin.date().isoformat(),
                "target_date": pd.Timestamp(test["target_date"]).date().isoformat(),
                "train_size": len(train),
                "train_start_origin": pd.Timestamp(
                    train[0]["origin_date"]
                ).date().isoformat(),
                "last_train_origin": pd.Timestamp(
                    train[-1]["origin_date"]
                ).date().isoformat(),
                "last_train_target": pd.Timestamp(
                    train[-1]["target_date"]
                ).date().isoformat(),
                "purged_origin_count": len(purged),
            }
        )

    pairs = [[row["origin_date"], row["target_date"]] for row in rows]
    contract = {
        "count": len(rows),
        "first_origin": rows[0]["origin_date"] if rows else None,
        "last_origin": rows[-1]["origin_date"] if rows else None,
        "sha256": canonical_json_sha256(pairs) if rows else None,
        "rows": rows,
    }
    return contract, variants


def _v5_fx_bootstrap_pvalues(
    improvements: Mapping[str, np.ndarray],
    *,
    block_length: int,
    resamples: int,
    seed: int,
) -> tuple[dict[str, float], int]:
    """Independently reproduce the preregistered common circular block draws."""

    require(bool(improvements), "v5 FX paired improvements are empty")
    lengths = {len(np.asarray(values)) for values in improvements.values()}
    require(len(lengths) == 1, "v5 FX paired improvement lengths differ")
    observation_count = lengths.pop()
    require(observation_count > 0, "v5 FX paired improvements are empty")
    effective_block = min(block_length, max(1, observation_count // 2))
    blocks_per_sample = int(np.ceil(observation_count / effective_block))
    generator = np.random.default_rng(seed)
    starts = generator.integers(
        0,
        observation_count,
        size=(resamples, blocks_per_sample),
    )
    offsets = np.arange(effective_block)
    indices = (starts[..., np.newaxis] + offsets) % observation_count
    indices = indices.reshape(resamples, -1)[:, :observation_count]
    pvalues: dict[str, float] = {}
    for variant, raw in improvements.items():
        differential = np.asarray(raw, dtype=float)
        require(
            np.isfinite(differential).all(),
            f"v5 FX {variant} paired improvements are non-finite",
        )
        observed = float(differential.mean())
        null_means = (differential - observed)[indices].mean(axis=1)
        pvalues[variant] = float(
            (1 + np.count_nonzero(null_means >= observed)) / (resamples + 1)
        )
    return pvalues, effective_block


def _audit_v5_fx_metrics(
    ablation: Mapping[str, Any],
    evidence: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    variants: Mapping[str, Sequence[str]],
    core_feature_count: int | None,
) -> dict[str, Any]:
    require_exact_columns(
        evidence,
        V5_FX_ABLATION_OOS_COLUMNS,
        "v5 FX ablation OOS evidence",
    )
    status = str(ablation["status"])
    if status != "evaluated":
        require(evidence.empty, "v5 FX non-evaluated OOS evidence must be empty")
        return {"variant_rows": 0, "comparison_rows": 0, "oos_rows": 0}

    origins = ablation["common_evaluation_origins"]
    origin_count = int(origins["count"])
    require(
        len(evidence) == origin_count * len(V5_FX_VARIANTS),
        "v5 FX OOS evidence row count disagrees with common origins",
    )
    frame = evidence.copy()
    for column in ("origin_date", "target_date", "last_train_target"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    canonical_keys = list(
        zip(
            frame["origin_date"],
            frame["target_date"],
            frame["variant"].astype(str),
            strict=True,
        )
    )
    require(
        canonical_keys == sorted(canonical_keys),
        "v5 FX OOS evidence is not in canonical origin/target/variant order",
    )
    require(
        not frame.duplicated(["origin_date", "target_date", "variant"]).any(),
        "v5 FX OOS evidence duplicates a variant/common origin",
    )
    require(
        set(frame["variant"].astype(str)) == set(V5_FX_VARIANTS),
        "v5 FX OOS evidence variant set mismatch",
    )
    require(
        frame["evaluation_split"].astype(str).eq("prospective_shadow").all(),
        "v5 FX OOS evidence split mismatch",
    )
    require(
        frame["current_state"].astype(str).isin(STATE_ORDER).all()
        and frame["actual"].astype(str).isin(STATE_ORDER).all(),
        "v5 FX OOS evidence contains an invalid state",
    )
    require(
        ((frame["target_date"] - frame["origin_date"]).dt.days == 7).all(),
        "v5 FX OOS target is not exactly one week after origin",
    )
    require(
        (frame["last_train_target"] < frame["origin_date"]).all(),
        "v5 FX OOS training target reaches its evaluation origin",
    )
    train_size_raw = pd.to_numeric(frame["train_size"], errors="raise")
    gap_raw = pd.to_numeric(frame["gap"], errors="raise")
    purged_raw = pd.to_numeric(frame["purged_origin_count"], errors="raise")
    require(
        np.isfinite(train_size_raw).all()
        and np.isfinite(gap_raw).all()
        and np.isfinite(purged_raw).all()
        and np.equal(train_size_raw, np.floor(train_size_raw)).all()
        and np.equal(gap_raw, np.floor(gap_raw)).all()
        and np.equal(purged_raw, np.floor(purged_raw)).all(),
        "v5 FX OOS train/purge counts must be integers",
    )
    train_size = train_size_raw.astype(int)
    gap = gap_raw.astype(int)
    purged = purged_raw.astype(int)
    require(
        train_size.ge(int(ablation["minimum_train_weeks"])).all()
        and gap.eq(int(ablation["purge_weeks"])).all()
        and purged.eq(1).all(),
        "v5 FX OOS train/purge evidence mismatch",
    )

    probability = frame[list(PROBABILITY_COLUMNS)].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(dtype=float)
    require(
        np.isfinite(probability).all()
        and ((probability >= 0.0) & (probability <= 1.0)).all()
        and np.allclose(probability.sum(axis=1), 1.0, atol=1e-8, rtol=0.0),
        "v5 FX OOS probabilities are invalid",
    )
    fallback = boolean_series(frame["fallback"], "v5 FX OOS fallback")
    fallback_reason = frame["fallback_reason"].fillna("").astype(str)
    allowed_fallback_reasons = {
        "training_class_coverage",
        "model_fit_or_prediction_error",
    }
    require(
        fallback_reason.loc[~fallback].eq("").all()
        and fallback_reason.loc[fallback].isin(allowed_fallback_reasons).all(),
        "v5 FX OOS fallback reason mismatch",
    )
    frame["fallback"] = fallback.to_numpy()
    frame["fallback_reason"] = fallback_reason.to_numpy()

    membership_frame = membership.copy()
    membership_frame["market_date"] = pd.to_datetime(
        membership_frame["date"], utc=True, errors="raise"
    ).dt.date.astype(str)
    require(
        not membership_frame["market_date"].duplicated().any(),
        "v5 FX membership dates are duplicated",
    )
    state_by_date = dict(
        zip(
            membership_frame["market_date"],
            membership_frame["state"].astype(str),
            strict=True,
        )
    )
    origin_states = frame["origin_date"].dt.date.astype(str).map(state_by_date)
    target_states = frame["target_date"].dt.date.astype(str).map(state_by_date)
    require(
        origin_states.notna().all()
        and target_states.notna().all()
        and origin_states.equals(frame["current_state"].astype(str))
        and target_states.equals(frame["actual"].astype(str)),
        "v5 FX OOS actual/current states disagree with membership evidence",
    )

    identity_columns = [
        "origin_date",
        "target_date",
        "evaluation_split",
        "current_state",
        "actual",
        "train_size",
        "gap",
        "last_train_target",
        "purged_origin_count",
        "common_origins_sha256",
    ]
    ordered_by_variant: dict[str, pd.DataFrame] = {}
    reference_identity: pd.DataFrame | None = None
    for variant in V5_FX_VARIANTS:
        ordered = frame.loc[frame["variant"].astype(str).eq(variant)].sort_values(
            ["origin_date", "target_date"], kind="mergesort", ignore_index=True
        )
        require(
            len(ordered) == origin_count,
            f"v5 FX {variant} OOS origin count mismatch",
        )
        identity = ordered.loc[:, identity_columns].reset_index(drop=True)
        if reference_identity is None:
            reference_identity = identity
        else:
            require(
                identity.equals(reference_identity),
                f"v5 FX {variant} does not use exact common-origin evidence",
            )
        ordered_by_variant[variant] = ordered

    if reference_identity is None:
        raise AuditFailure("v5 FX OOS control identity is missing")
    expected_rows = list(origins["rows"])
    evidence_pairs = [
        [row.origin_date.date().isoformat(), row.target_date.date().isoformat()]
        for row in reference_identity.itertuples(index=False)
    ]
    expected_pairs = [
        [str(row["origin_date"]), str(row["target_date"])]
        for row in expected_rows
    ]
    common_hash = canonical_json_sha256(evidence_pairs)
    require(
        evidence_pairs == expected_pairs
        and origins["sha256"] == common_hash
        and reference_identity["common_origins_sha256"]
        .astype(str)
        .eq(common_hash)
        .all(),
        "v5 FX OOS common-origin hash/pairs mismatch",
    )
    for position, expected in enumerate(expected_rows):
        actual_row = reference_identity.iloc[position]
        require(
            int(actual_row["train_size"]) == int(expected["train_size"])
            and actual_row["last_train_target"].date().isoformat()
            == str(expected["last_train_target"])
            and int(actual_row["purged_origin_count"])
            == int(expected["purged_origin_count"]),
            "v5 FX OOS purge evidence disagrees with common-origin payload",
        )

    metrics = list(ablation["variant_metrics"])
    require(
        [str(row.get("variant")) for row in metrics] == list(V5_FX_VARIANTS),
        "v5 FX metric variant order mismatch",
    )
    inferred_core_counts: set[int] = set()
    metric_index: dict[str, Mapping[str, Any]] = {}
    loss_by_variant: dict[str, np.ndarray] = {}
    brier_by_variant: dict[str, np.ndarray] = {}
    state_positions = {state: index for index, state in enumerate(STATE_ORDER)}
    for row in metrics:
        variant = str(row["variant"])
        expected_fx_count = len(variants[variant])
        fx_count = int(row["fx_feature_count"])
        feature_count = int(row["feature_count"])
        require(
            fx_count == expected_fx_count,
            f"v5 FX {variant} feature count disagrees with sidecar columns",
        )
        inferred_core_counts.add(feature_count - fx_count)
        if core_feature_count is not None:
            require(
                feature_count == core_feature_count + expected_fx_count,
                f"v5 FX {variant} feature count disagrees with feature manifest",
            )
        ordered = ordered_by_variant[variant]
        variant_probability = ordered[list(PROBABILITY_COLUMNS)].to_numpy(
            dtype=float
        )
        actual_positions = np.asarray(
            [state_positions[value] for value in ordered["actual"].astype(str)]
        )
        actual_probability = variant_probability[
            np.arange(origin_count), actual_positions
        ]
        losses = -np.log(np.clip(actual_probability, 1e-9, 1.0))
        one_hot = np.zeros_like(variant_probability)
        one_hot[np.arange(origin_count), actual_positions] = 1.0
        brier = np.sum((variant_probability - one_hot) ** 2, axis=1)
        correct = (
            np.argmax(variant_probability, axis=1) == actual_positions
        ).astype(float)
        recalls = [
            float(correct[actual_positions == position].mean())
            for position in range(len(STATE_ORDER))
            if np.any(actual_positions == position)
        ]
        variant_fallback = ordered["fallback"].astype(bool)
        reasons = dict(
            sorted(
                ordered.loc[variant_fallback, "fallback_reason"]
                .astype(str)
                .value_counts()
                .astype(int)
                .to_dict()
                .items()
            )
        )
        fallback_count = int(variant_fallback.sum())
        require(
            np.isclose(
                float(row["log_loss"]),
                float(losses.mean()),
                atol=1e-12,
                rtol=0.0,
            )
            and np.isclose(
                float(row["brier"]),
                float(brier.mean()),
                atol=1e-12,
                rtol=0.0,
            )
            and np.isclose(
                float(row["accuracy"]),
                float(correct.mean()),
                atol=1e-12,
                rtol=0.0,
            )
            and np.isclose(
                float(row["balanced_accuracy"]),
                float(np.mean(recalls)),
                atol=1e-12,
                rtol=0.0,
            ),
            f"v5 FX {variant} metrics disagree with OOS evidence",
        )
        require(
            int(row["n"]) == origin_count
            and int(row["n_predictions"]) == origin_count
            and int(row["fallback_count"]) == fallback_count
            and bool(row["fallback"]) == (fallback_count > 0)
            and dict(row["fallback_reasons"]) == reasons
            and row["first_origin"] == origins["first_origin"]
            and row["last_origin"] == origins["last_origin"]
            and row["origin_sha256"] == common_hash,
            f"v5 FX {variant} metrics are not bound to OOS/common origins",
        )
        loss_by_variant[variant] = losses
        brier_by_variant[variant] = brier
        metric_index[variant] = row
    require(
        len(inferred_core_counts) == 1,
        "v5 FX variants disagree on the core feature count",
    )

    gate = ablation["gate"]
    comparisons = list(gate["comparisons"])
    require(
        [str(row.get("variant")) for row in comparisons]
        == list(V5_FX_VARIANTS[1:]),
        "v5 FX gate comparison order mismatch",
    )
    improvements = {
        variant: loss_by_variant["v4_control"] - loss_by_variant[variant]
        for variant in V5_FX_VARIANTS[1:]
    }
    raw_pvalues, effective_block = _v5_fx_bootstrap_pvalues(
        improvements,
        block_length=int(gate["bootstrap_block_weeks"]),
        resamples=int(gate["bootstrap_resamples"]),
        seed=int(gate["bootstrap_seed"]),
    )
    adjusted = holm_adjusted_pvalues(raw_pvalues)
    require(
        int(gate["bootstrap_effective_block_weeks"]) == effective_block,
        "v5 FX bootstrap effective block mismatch",
    )
    expected_passed: list[str] = []
    control = metric_index["v4_control"]
    for row in comparisons:
        variant = str(row["variant"])
        challenger = metric_index[variant]
        improvement = float(improvements[variant].mean())
        brier_difference = float(
            (brier_by_variant[variant] - brier_by_variant["v4_control"]).mean()
        )
        require(
            np.isclose(
                float(row["mean_log_loss_improvement"]),
                improvement,
                atol=1e-12,
                rtol=0.0,
            )
            and np.isclose(
                float(row["brier_difference"]),
                brier_difference,
                atol=1e-12,
                rtol=0.0,
            ),
            f"v5 FX {variant} paired metric mismatch",
        )
        require(
            np.isclose(
                float(row["raw_p_value"]),
                raw_pvalues[variant],
                atol=1e-12,
                rtol=0.0,
            ),
            f"v5 FX {variant} raw bootstrap p-value mismatch",
        )
        require(
            np.isclose(
                float(row["holm_adjusted_p_value"]),
                adjusted[variant],
                atol=1e-12,
                rtol=0.0,
            ),
            f"v5 FX {variant} Holm adjustment mismatch",
        )
        require(
            int(row["control_fallback_count"]) == int(control["fallback_count"])
            and int(row["fallback_count"]) == int(challenger["fallback_count"]),
            f"v5 FX {variant} gate fallback count mismatch",
        )
        failures: list[str] = []
        if int(control["fallback_count"]):
            failures.append("control_fallback_present")
        if int(challenger["fallback_count"]):
            failures.append("fallback_present")
        if improvement + 1e-12 < float(gate["minimum_log_loss_improvement"]):
            failures.append("insufficient_log_loss_improvement")
        if adjusted[variant] > float(gate["alpha"]):
            failures.append("holm_not_significant")
        if brier_difference > float(gate["brier_tolerance"]) + 1e-12:
            failures.append("brier_degradation")
        passed = not failures
        require(
            bool(row["gate_passed"]) == passed
            and list(row["gate_reasons"]) == (["passed"] if passed else failures),
            f"v5 FX {variant} gate decision mismatch",
        )
        if passed:
            expected_passed.append(variant)
    require(
        list(gate["passed_variants"]) == expected_passed,
        "v5 FX passed-variant summary mismatch",
    )
    return {
        "variant_rows": len(metrics),
        "comparison_rows": len(comparisons),
        "oos_rows": len(frame),
    }


def _audit_v5_fx_provenance(
    payload: Mapping[str, Any],
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    fx_config = preregistration.get("fx")
    require(isinstance(fx_config, Mapping), "v5 FX preregistration is missing")
    expected_archive_contract = {
        "historical_availability_backfill": False,
        "official_release_archive_ingest": True,
        "availability_basis": "official_archive_release_schedule",
        "archive_revision_policy": (
            "later_official_release_preserved_as_new_vintage"
        ),
        "archive_correction_availability_basis": (
            "date_only_conservative_next_day"
        ),
    }
    for field, expected in expected_archive_contract.items():
        require(
            fx_config.get(field) == expected,
            f"v5 FX preregistration {field} mismatch",
        )

    ablation = payload["model"]["fx_ablation"]
    shadow = fx_config.get("shadow_evaluation")
    require(
        isinstance(shadow, Mapping),
        "v5 FX shadow-evaluation preregistration is missing",
    )
    require(
        list(fx_config.get("ablations", ())) == list(V5_FX_VARIANTS),
        "v5 FX ablation variants disagree with preregistration",
    )
    preregistered_payload_fields = {
        "minimum_common_weeks": fx_config.get("common_origin_minimum_weeks"),
        "common_origin_required_pairs": shadow.get("required_bilateral_pairs"),
        "minimum_train_weeks": shadow.get("minimum_train_weeks"),
        "target_horizon_weeks": shadow.get("target_horizon_weeks"),
        "purge_weeks": shadow.get("purge_weeks"),
        "target_availability_rule": shadow.get("target_availability_rule"),
    }
    for field, expected in preregistered_payload_fields.items():
        require(
            ablation.get(field) == expected,
            f"v5 FX payload {field} disagrees with preregistration",
        )
    model = ablation.get("model")
    require(isinstance(model, Mapping), "v5 FX model payload is missing")
    preregistered_model_fields = {
        "name": shadow.get("model"),
        "regularization_c": shadow.get("regularization_c"),
        "imputation": shadow.get("imputation"),
        "scaling": shadow.get("scaling"),
        "fit_window": shadow.get("fit_window"),
    }
    for field, expected in preregistered_model_fields.items():
        require(
            model.get(field) == expected,
            f"v5 FX model {field} disagrees with preregistration",
        )
    gate = ablation.get("gate")
    require(isinstance(gate, Mapping), "v5 FX gate payload is missing")
    preregistered_gate_fields = {
        "method": shadow.get("bootstrap_method"),
        "bootstrap_block_weeks": shadow.get("bootstrap_block_weeks"),
        "bootstrap_resamples": shadow.get("bootstrap_resamples"),
        "bootstrap_seed": shadow.get("bootstrap_seed"),
        "alpha": shadow.get("holm_alpha"),
        "minimum_log_loss_improvement": shadow.get(
            "minimum_log_loss_improvement"
        ),
        "brier_tolerance": shadow.get("maximum_brier_degradation"),
    }
    for field, expected in preregistered_gate_fields.items():
        require(
            gate.get(field) == expected,
            f"v5 FX gate {field} disagrees with preregistration",
        )
    actual = {
        field: ablation[field]
        for field in (
            "official_release_archive_ingest",
            "availability_basis",
            "archive_revision_policy",
            "archive_correction_availability_basis",
        )
    }
    require(
        actual["availability_basis"]
        in {"official_archive_release_schedule", "collection_first_seen_at"},
        "v5 FX availability basis is invalid",
    )
    require(
        bool(actual["official_release_archive_ingest"])
        == (
            actual["availability_basis"]
            == "official_archive_release_schedule"
        ),
        "v5 FX archive ingest/basis mismatch",
    )
    require(
        actual["archive_revision_policy"]
        == expected_archive_contract["archive_revision_policy"],
        "v5 FX archive revision policy mismatch",
    )
    require(
        actual["archive_correction_availability_basis"]
        == expected_archive_contract["archive_correction_availability_basis"],
        "v5 FX archive correction availability basis mismatch",
    )

    if str(payload["meta"]["mode"]) == "live":
        sources = {
            str(row.get("id")): row
            for row in payload.get("sources", [])
            if isinstance(row, Mapping)
        }
        h10 = sources.get("frb_h10")
        require(isinstance(h10, Mapping), "v5 live H.10 source is missing")
        for field, value in actual.items():
            require(
                h10.get(field) == value,
                f"v5 H.10 source/model {field} mismatch",
            )

    ready = int(ablation["eligible_common_weeks"]) >= 156
    if ready:
        for field, expected in expected_archive_contract.items():
            require(
                ablation[field] == expected,
                f"v5 FX ready/evaluated {field} lacks archive provenance",
            )
    return {**actual, "archive_required_for_readiness": ready}


def _audit_v5_fx(
    payload: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    membership: pd.DataFrame,
    *,
    core_feature_count: int | None = None,
) -> dict[str, Any]:
    fx_paths = {path for _, path in V5_FX_ARTIFACTS}
    present = set(frames).intersection(fx_paths)
    status = str(payload["model"]["fx_ablation"]["status"])
    if not present:
        require(status == "unavailable", "v5 FX artifacts are missing")
        require(
            payload["model"]["fx_ablation"].get("status_reason")
            == "fx_feature_result_unavailable",
            "v5 FX missing-artifact reason mismatch",
        )
        return {"status": status, "eligible_common_weeks": 0}
    require(
        present == fx_paths,
        "v5 FX research sidecars must be present as a complete set",
    )
    features = frames["fx-features.csv"].copy()
    coverage = frames["fx-coverage.csv"].copy()
    ablation_oos = frames["fx-ablation-oos.csv"].copy()
    for name, frame in (("features", features), ("coverage", coverage)):
        require_columns(frame, {"observation_week"}, f"v5 FX {name}")
        dates = pd.to_datetime(
            frame.pop("observation_week"), utc=True, errors="raise"
        )
        require(
            dates.is_monotonic_increasing and not dates.duplicated().any(),
            f"v5 FX {name} observation_week is not unique/increasing",
        )
        frame.index = pd.DatetimeIndex(dates).tz_localize(None).normalize()
    require(features.index.equals(coverage.index), "v5 FX artifact indexes differ")

    ablation = payload["model"]["fx_ablation"]
    sources = {
        str(row.get("id")): row
        for row in payload.get("sources", [])
        if isinstance(row, Mapping)
    }
    h10 = sources.get("frb_h10")
    correction_events: Sequence[pd.Timestamp] | None = None
    if isinstance(h10, Mapping):
        raw_events = h10.get("archive_correction_available_at")
        require(
            isinstance(raw_events, list),
            "v5 H.10 correction availability must be a list",
        )
        parsed_events = pd.to_datetime(raw_events, utc=True, errors="coerce")
        require(
            not bool(pd.isna(parsed_events).any()),
            "v5 H.10 correction availability contains an invalid timestamp",
        )
        correction_events = list(pd.DatetimeIndex(parsed_events))
        release_count = int(h10.get("archive_release_count", -1))
        correction_count = int(h10.get("archive_correction_count", -1))
        require(
            release_count >= correction_count == len(correction_events) >= 0,
            "v5 H.10 archive release/correction counts mismatch",
        )
        require(
            h10.get("archive_correction_quarantine_weeks") == 27,
            "v5 H.10 correction quarantine length mismatch",
        )
        require(
            h10.get("archive_evaluation_start") == "2022-01-01",
            "v5 H.10 archive evaluation start mismatch",
        )
        require(
            h10.get("archive_evaluation_start_rationale")
            == "post_2019_06_24_jan06_index_rebase_common_scale",
            "v5 H.10 archive evaluation-start rationale mismatch",
        )
        if bool(ablation["official_release_archive_ingest"]):
            require(
                release_count > 0,
                "v5 H.10 official archive provenance has no releases",
            )
        else:
            require(
                release_count == 0
                and correction_count == 0
                and not correction_events,
                "v5 H.10 first-seen fallback contains archive events",
            )
    correction_quarantine = _audit_v5_fx_correction_quarantine(
        coverage,
        correction_events=correction_events,
    )

    from regime_lab.analysis.fx import FXFeatureResult
    from regime_lab.analysis.fx_ablation import fx_ablation_readiness

    empty = pd.DataFrame(index=features.index)
    result = FXFeatureResult(
        features=features,
        weekly_usd_log_levels=empty,
        weekly_availability=empty,
        coverage=coverage,
        status=coverage.loc[
            :, [column for column in ("source_status", "feature_status") if column in coverage]
        ].copy(),
    )
    cutoffs = pd.DatetimeIndex(
        pd.to_datetime(membership["date"], utc=True, errors="raise")
    )
    recomputed = fx_ablation_readiness(result, cutoffs)
    for field in (
        "role",
        "variants",
        "minimum_common_weeks",
        "historical_availability_backfill",
        "eligible_common_weeks",
        "first_eligible_cutoff",
        "last_eligible_cutoff",
        "manifest",
    ):
        require(
            recomputed[field] == ablation[field],
            f"v5 FX prospective readiness {field} mismatch",
        )
    readiness_status = str(recomputed["status"])
    reason = ablation.get("status_reason")
    expected_readiness_status = (
        "ready_for_evaluation"
        if status == "evaluated"
        or reason == "no_origin_has_104_strictly_available_training_targets"
        or reason == "fx_model_features_non_numeric"
        else "insufficient_history"
    )
    require(
        readiness_status == expected_readiness_status,
        "v5 FX prospective readiness status mismatch",
    )
    expected_origins, variants = _v5_fx_evaluation_origins(result, cutoffs)
    published_origins = ablation["common_evaluation_origins"]
    require(
        published_origins == (
            expected_origins
            if status == "evaluated"
            else {
                "count": 0,
                "first_origin": None,
                "last_origin": None,
                "sha256": None,
                "rows": [],
            }
        ),
        "v5 FX common evaluation origins disagree with sidecar evidence",
    )
    metric_summary = _audit_v5_fx_metrics(
        ablation,
        ablation_oos,
        membership,
        variants=variants,
        core_feature_count=core_feature_count,
    )
    return {
        "status": status,
        "eligible_common_weeks": int(recomputed["eligible_common_weeks"]),
        "observation_weeks": len(features),
        "evaluation_origins": int(expected_origins["count"]),
        "correction_quarantine": correction_quarantine,
        **metric_summary,
    }


def _audit_v5_inherited_structural_outputs(
    payload: Mapping[str, Any],
    artifacts: Path,
    *,
    feature_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit the V4 transition and feature-ablation contracts inherited by V5."""

    predictions = read_csv(
        artifacts / "oos-predictions.csv",
        ("origin_date", "target_date"),
    )
    return {
        "transition": audit_transition_outputs(
            payload,
            artifacts,
            main_predictions=predictions,
            main_champion=str(payload["model"]["champion"]),
            main_published_split="holdout",
        ),
        "feature_ablation": audit_feature_ablation(
            payload,
            artifacts,
            main_predictions=predictions,
            feature_manifest=dict(feature_manifest),
        ),
        "joint_survival": audit_joint_survival_forecasts(artifacts),
    }


def audit_v5_artifact_inventory(artifacts: Path) -> dict[str, Any]:
    """Verify the complete V5 generation before reading individual sidecars."""

    try:
        return verify_artifact_inventory(artifacts)
    except (ArtifactInventoryError, OSError) as exc:
        raise AuditFailure(f"v5 artifact inventory failed: {exc}") from exc


def _audit_v5_research_replay_input(
    payload: Mapping[str, Any],
    artifacts: Path,
) -> dict[str, Any]:
    """Verify the private input identity for reconstructed research outputs."""

    path = artifacts / "research-replay-input.json"
    shadow = payload.get("research", {}).get("prospective_decision_shadow")
    schema_version = shadow.get("schema_version") if isinstance(shadow, Mapping) else None
    live_v2 = (
        payload.get("meta", {}).get("mode") == "live"
        and schema_version == "regime-prospective-decision-shadow/2"
    )
    if not live_v2:
        require(
            not path.exists() and not path.is_symlink(),
            "non-live or historical generation has an unexpected replay input",
        )
        return {"status": "not_required_non_live_or_historical_contract"}

    require(
        path.is_file() and not path.is_symlink(),
        "decision-shadow V2 generation is missing research-replay-input.json",
    )
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditFailure(f"research replay input contract failed: {exc}") from exc
    require(isinstance(document, Mapping), "research replay input must be an object")
    require(
        set(document)
        == {
            "schema_version",
            "evidence_track",
            "data_as_of",
            "availability_basis",
            "source_observation_count",
            "input_vintages",
            "canonical_panel",
            "state_membership",
            "operational_generation_input_snapshot_sha256",
        },
        "research replay input fields changed",
    )
    require(
        document["schema_version"] == "regime-research-replay-input/1",
        "research replay input schema is invalid",
    )
    require(
        document["evidence_track"] == "reconstructed_oos"
        and document["availability_basis"] == "reconstructed_market",
        "research replay input evidence track is invalid",
    )
    require(
        document["data_as_of"] == payload["meta"]["data_as_of"],
        "research replay input data_as_of differs from payload",
    )
    source_observations = document["source_observation_count"]
    require(
        type(source_observations) is int and source_observations > 0,
        "research replay source observation count is invalid",
    )

    def verified_sha256(value: object, *, context: str) -> str:
        digest = str(value)
        require(
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            f"{context} is not a SHA-256 digest",
        )
        return digest

    input_vintages = document["input_vintages"]
    require(
        isinstance(input_vintages, Mapping)
        and set(input_vintages) == {"count", "sha256"},
        "research replay input-vintage contract is invalid",
    )
    require(
        type(input_vintages["count"]) is int
        and 0 < input_vintages["count"] <= source_observations,
        "research replay input-vintage count is invalid",
    )
    input_sha256 = verified_sha256(
        input_vintages["sha256"],
        context="research replay input-vintage hash",
    )

    canonical = document["canonical_panel"]
    require(
        isinstance(canonical, Mapping)
        and set(canonical)
        == {"start", "end", "rows", "columns", "sha256"},
        "research replay canonical-panel contract is invalid",
    )
    membership = document["state_membership"]
    require(
        isinstance(membership, Mapping)
        and set(membership) == {"rows", "sha256"},
        "research replay state-membership contract is invalid",
    )
    require(
        type(canonical["rows"]) is int
        and canonical["rows"] >= len(payload["weekly"])
        and type(canonical["columns"]) is int
        and canonical["columns"] > 0
        and membership["rows"] == canonical["rows"],
        "research replay canonical dimensions are invalid",
    )
    try:
        start = date.fromisoformat(str(canonical["start"]))
        end = date.fromisoformat(str(canonical["end"]))
        data_as_of = datetime.fromisoformat(
            str(document["data_as_of"]).replace("Z", "+00:00")
        ).date()
    except ValueError as exc:
        raise AuditFailure("research replay input dates are invalid") from exc
    require(
        start <= end == data_as_of,
        "research replay canonical coverage differs from data_as_of",
    )
    canonical_sha256 = verified_sha256(
        canonical["sha256"],
        context="research replay canonical-panel hash",
    )
    membership_sha256 = verified_sha256(
        membership["sha256"],
        context="research replay state-membership hash",
    )
    operational_sha256 = verified_sha256(
        document["operational_generation_input_snapshot_sha256"],
        context="operational generation input snapshot hash",
    )
    return {
        "status": "verified",
        "source_observation_count": source_observations,
        "input_vintage_count": input_vintages["count"],
        "input_vintage_sha256": input_sha256,
        "canonical_rows": canonical["rows"],
        "canonical_columns": canonical["columns"],
        "canonical_sha256": canonical_sha256,
        "state_membership_sha256": membership_sha256,
        "operational_generation_input_snapshot_sha256": operational_sha256,
    }


def _v5_expected_research_artifacts(
    research_contract: object,
) -> tuple[tuple[str, str], ...]:
    require(
        isinstance(research_contract, Mapping),
        "v5 research artifact manifest must be an object",
    )
    optional_keys = tuple(key for key, _ in V5_MODEL_CONDITIONED_ARTIFACTS)
    optional_present = tuple(
        key for key in optional_keys if key in research_contract
    )
    require(
        not optional_present or optional_present == optional_keys,
        "v5 model-conditioned artifacts must be a complete pair",
    )
    fx_keys = tuple(key for key, _ in V5_FX_ARTIFACTS)
    fx_present = tuple(key for key in fx_keys if key in research_contract)
    require(
        not fx_present or fx_present == fx_keys,
        "v5 FX artifacts must be a complete set",
    )
    expected = list(V5_RESEARCH_ARTIFACTS)
    if optional_present:
        expected.extend(V5_MODEL_CONDITIONED_ARTIFACTS)
    if fx_present:
        expected.extend(V5_FX_ARTIFACTS)
    require(
        list(research_contract) == [key for key, _ in expected],
        "v5 research artifact key/order contract mismatch",
    )
    return tuple(expected)


def _audit_v5_selection_family(
    payload: Mapping[str, Any],
    artifacts: Path,
) -> dict[str, Any]:
    """Rebuild the generic all-candidate selection audit from source rows."""

    model = payload["model"]
    manifest_sha256 = str(model.get("candidate_manifest_sha256", ""))
    historical = (
        OPERATING_CONTRACT.historical_reviewed_roster_by_manifest_sha256(
            manifest_sha256
        )
    )
    path = artifacts / "selection-family-audit.json"
    if historical is not None and not path.exists() and not path.is_symlink():
        return {
            "status": "not_emitted_by_historical_reviewed_generation",
            "candidate_manifest_sha256": historical[
                "candidate_manifest_sha256"
            ],
        }
    require(
        path.is_file() and not path.is_symlink(),
        "current V5 generation is missing selection-family-audit.json",
    )
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        validate_selection_family_audit(
            document,
            expected_generation_id=str(payload["meta"]["generation_id"]),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise AuditFailure(f"selection-family audit contract failed: {exc}") from exc
    require(isinstance(document, dict), "selection-family audit must be an object")

    try:
        expected = build_selection_family_audit_from_artifacts(payload, artifacts)
    except (OSError, TypeError, ValueError) as exc:
        raise AuditFailure(
            f"selection-family source evidence failed: {exc}"
        ) from exc
    require(
        document == expected,
        "selection-family audit differs from independently rebuilt source evidence",
    )
    return {
        "status": "verified",
        "candidate_count": document["candidate_count"],
        "common_origins": document["common_origin_contract"]["origin_count"],
        "champion": document["champion"],
        "runner_up": document["runner_up"],
        "sha256": document["sha256"],
        "supplemental_role": document["supplemental_evaluation"]["role"],
        "mcs_retained_models": document["supplemental_evaluation"][
            "model_confidence_set"
        ]["retained_models"],
    }


def audit_v5(
    payload: Mapping[str, Any],
    payload_path: Path,
    artifacts: Path,
    expected_mode: str,
) -> dict[str, Any]:
    """Independently verify the opt-in v5 payload and its research sidecars."""

    from regime_lab.schema import validate_dashboard_payload

    validate_dashboard_payload(payload)
    meta = payload["meta"]
    mode = str(meta["mode"])
    if expected_mode != "auto":
        require(mode == expected_mode, f"expected mode={expected_mode}, got {mode}")
    require(meta["schema_version"] == V5_SCHEMA_VERSION, "unexpected v5 schema")
    artifact_inventory = audit_v5_artifact_inventory(artifacts)
    research_replay_input = _audit_v5_research_replay_input(payload, artifacts)

    generation_path = _v5_artifact_path(
        artifacts, "build-generation.json", context="v5 generation contract"
    )
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    require(
        generation == {"generation_id": meta["generation_id"]},
        "v5 payload/artifact generation IDs differ",
    )

    preregistration = payload["model"]["structural_preregistration"]
    preregistration_path = PROJECT_ROOT / str(preregistration["path"])
    require(
        preregistration_path.is_file() and not preregistration_path.is_symlink(),
        "v5 structural preregistration file is missing/non-regular",
    )
    require(
        file_sha256(preregistration_path) == str(preregistration["sha256"]),
        "v5 structural preregistration SHA-256 mismatch",
    )
    preregistration_document = json.loads(
        preregistration_path.read_text(encoding="utf-8")
    )
    require(
        isinstance(preregistration_document, dict),
        "v5 structural preregistration must be an object",
    )
    fx_provenance = _audit_v5_fx_provenance(
        payload,
        preregistration_document,
    )

    research_contract = payload["model"].get("research_artifacts")
    expected_contracts = _v5_expected_research_artifacts(research_contract)
    frames = _audit_v5_file_contracts(
        research_contract,
        artifacts,
        context="payload.model.research_artifacts",
    )
    require(
        [str(research_contract[key]["path"]) for key, _ in expected_contracts]
        == [path for _, path in expected_contracts],
        "v5 research artifact path contract mismatch",
    )
    expected_paths = [path for _, path in expected_contracts]
    require(
        list(frames) == expected_paths,
        "v5 research artifact path/order contract mismatch",
    )
    execution = _audit_v5_execution_parameters(payload)
    core = _audit_v5_core_model(payload, artifacts)
    selection_family = _audit_v5_selection_family(payload, artifacts)
    model_forecasts = _audit_v5_model_forecasts(payload, artifacts)
    feature_summary = audit_feature_manifest(payload, artifacts)
    inherited_structural = _audit_v5_inherited_structural_outputs(
        payload,
        artifacts,
        feature_manifest=feature_summary,
    )
    manifest_features = [
        feature
        for group_features in feature_summary["group_features"].values()
        for feature in group_features
    ]
    feature_quality = _audit_v5_feature_quality(
        payload,
        artifacts,
        expected_features=manifest_features,
    )
    evidence, membership = _audit_v5_evidence(payload, artifacts)
    directional = _audit_v5_directional(payload, frames)
    directional["weekly_binding"] = _audit_v5_weekly_directional(
        payload,
        frames,
        membership,
    )
    conditional = _audit_v5_conditional(payload, frames)
    model_conditioned = _audit_v5_model_conditioned(payload, frames)
    decision_shadow = _audit_v5_decision_shadow(payload)
    duration = _audit_v5_duration(payload, membership, execution)
    fx = _audit_v5_fx(
        payload,
        frames,
        membership,
        core_feature_count=int(feature_summary["feature_count"]),
    )

    from regime_lab.frozen_v4 import verify_frozen_v4_baseline

    baseline = verify_frozen_v4_baseline(project_directory=PROJECT_ROOT)
    require(
        dict(payload["model"]["baseline_v4"]) == baseline,
        "v5 payload frozen-v4 baseline metadata mismatch",
    )
    return {
        "ok": True,
        "mode": mode,
        "contract": "v5",
        "champion": payload["model"]["champion"],
        "weeks": len(payload["weekly"]),
        "directional": directional,
        "conditional": conditional,
        "model_conditioned": model_conditioned,
        "decision_shadow": decision_shadow,
        "duration": duration,
        "execution_parameters_sha256": execution["sha256"],
        "core": core,
        "selection_family": selection_family,
        "model_forecasts": model_forecasts,
        "feature_quality": feature_quality,
        "core_feature_manifest": {
            key: value
            for key, value in feature_summary.items()
            if key != "group_features"
        },
        "transition": inherited_structural["transition"],
        "feature_ablation": inherited_structural["feature_ablation"],
        "joint_survival": inherited_structural["joint_survival"],
        "evidence": evidence,
        "fx": fx,
        "fx_provenance": fx_provenance,
        "artifact_inventory": artifact_inventory,
        "research_replay_input": research_replay_input,
        "research_artifacts": list(frames),
        "payload": str(payload_path),
        "artifacts": str(artifacts),
    }


def audit(payload_path: Path, artifacts: Path, expected_mode: str) -> dict[str, Any]:
    require(payload_path.is_file() and payload_path.stat().st_size > 0,
            f"missing/empty payload: {payload_path}")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), "payload root must be an object")
    if payload.get("meta", {}).get("result_version") == V5_RESULT_VERSION:
        return audit_v5(payload, payload_path, artifacts, expected_mode)
    leaderboard = read_csv(artifacts / "model-leaderboard.csv")
    predictions = read_csv(
        artifacts / "oos-predictions.csv", ("origin_date", "target_date")
    )
    splits = read_csv(
        artifacts / "walk-forward-splits.csv",
        (
            "origin_date",
            "target_date",
            "train_start",
            "last_train_origin",
            "last_train_target",
            "first_purged_origin",
        ),
    )
    manifest_path = artifacts / "candidate-manifest.json"
    require(manifest_path.is_file() and manifest_path.stat().st_size > 0,
            f"missing/empty file: {manifest_path}")
    artifact_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    meta = payload.get("meta", {})
    model_contract = payload.get("model", {})
    mode = str(meta.get("mode"))
    if expected_mode != "auto":
        require(mode == expected_mode, f"expected mode={expected_mode}, got {mode}")
    require(meta.get("schema_version") == "1.0.0", "unexpected schema_version")
    require(meta.get("timezone") == "America/New_York", "unexpected timezone")
    require(model_contract.get("state_order") == list(STATE_ORDER),
            "payload model.state_order mismatch")
    result_version = meta.get("result_version")
    is_v3 = result_version == V3_RESULT_VERSION
    is_v4 = result_version == V4_RESULT_VERSION
    frozen_v3_summary = None
    preregistration_summary = None
    if is_v3 or is_v4:
        generation_path = artifacts / "build-generation.json"
        require(
            generation_path.is_file() and generation_path.stat().st_size > 0,
            f"missing/empty file: {generation_path}",
        )
        generation_contract = json.loads(
            generation_path.read_text(encoding="utf-8")
        )
        require(
            isinstance(generation_contract, dict)
            and isinstance(meta.get("generation_id"), str)
            and generation_contract.get("generation_id") == meta.get("generation_id"),
            "payload/artifact generation IDs differ",
        )
        expected_model_version = V4_MODEL_VERSION if is_v4 else V3_MODEL_VERSION
        expected_feature_version = (
            V4_FEATURE_SET_VERSION if is_v4 else V3_FEATURE_SET_VERSION
        )
        require(model_contract.get("version") == expected_model_version,
                f"unexpected {'v4' if is_v4 else 'v3'} model suite version")
        require(model_contract.get("label_version") == "market-causal-3state-v1",
                f"unexpected {'v4' if is_v4 else 'v3'} label version")
        require(model_contract.get("feature_set_version") == expected_feature_version,
                f"unexpected {'v4' if is_v4 else 'v3'} feature-set version")
        require(model_contract.get("primary_horizon_weeks") == 1,
                f"{'v4' if is_v4 else 'v3'} primary horizon must be one week")
        require(model_contract.get("transition_horizons_weeks") == [1, 4, 13],
                f"{'v4' if is_v4 else 'v3'} transition horizons must be exactly 1/4/13")
        if is_v3:
            require(model_contract.get("baseline_v2") == V2_BASELINE,
                    "v3 frozen v2 baseline hashes/metadata differ")
        else:
            require(model_contract.get("baseline_v3") == V3_BASELINE,
                    "v4 frozen v3 baseline hashes/metadata differ")
            frozen_v3_summary = audit_frozen_v3_baseline()
            preregistration = model_contract.get("structural_preregistration")
            require(isinstance(preregistration, dict),
                    "v4 structural preregistration metadata missing")
            preregistration_summary = audit_structural_preregistration(
                preregistration
            )
            structural_models = model_contract.get("structural_models")
            require(isinstance(structural_models, dict),
                    "v4 structural model metadata missing")
            require(set(structural_models) == {
                "xgb_hazard_destination",
                "causal_dynamic_ensemble",
                "joint_survival_hazard",
            }, "v4 structural model metadata keys mismatch")
            require(
                structural_models == {
                    "xgb_hazard_destination": {
                        "hazard_model": "binary_xgboost",
                        "destination_model": "xgboost",
                        "direct_jump_floor": V4_DIRECT_JUMP_FLOOR,
                    },
                    "causal_dynamic_ensemble": {
                        "experts": list(V4_STRUCTURAL_EXPERTS),
                        "half_life_weeks": 52,
                        "minimum_history_rows": 26,
                        "eligible_loss_rule": "target_date_strictly_before_origin",
                    },
                    "joint_survival_hazard": {
                        "base_target_weeks": 1,
                        "horizons_weeks": [1, 4, 13],
                        "future_covariates": "origin_values_frozen",
                        "identity": "one_minus_product_one_minus_weekly_hazard",
                    },
                },
                "v4 structural model metadata values mismatch",
            )
    else:
        require(result_version in (None, V2_RESULT_VERSION),
                f"unsupported result_version: {result_version}")
        require(model_contract.get("version") == "weekly-nondl-walkforward-v2",
                "unexpected model suite version")
    require(
        model_contract.get("selection_status") == "provisional_predeployment",
        "selection status must remain provisional before deployment approval",
    )
    require(
        model_contract.get("post_selection_period_role")
        == "retrospective_external_period_diagnostic",
        "post-selection period role is not retrospective diagnostic",
    )
    require(isinstance(artifact_manifest, dict), "candidate manifest must be an object")
    artifact_hash = str(artifact_manifest.get("sha256", ""))
    manifest_body = dict(artifact_manifest)
    manifest_body.pop("sha256", None)
    serialized_manifest = json.dumps(
        manifest_body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    recomputed_manifest_hash = hashlib.sha256(
        serialized_manifest.encode("utf-8")
    ).hexdigest()
    require(artifact_hash == recomputed_manifest_hash,
            "candidate manifest SHA-256 is invalid")
    require(model_contract.get("candidate_manifest_sha256") == artifact_hash,
            "payload/artifact candidate manifest hash mismatch")
    require(model_contract.get("candidate_manifest") == manifest_body,
            "payload/artifact candidate manifests differ")
    manifest_models = manifest_body.get("models")
    require(isinstance(manifest_models, list) and manifest_models,
            "candidate manifest models must be a non-empty array")
    manifest_names = [str(row.get("name")) for row in manifest_models]
    require(len(manifest_names) == len(set(manifest_names)),
            "candidate manifest model names are duplicated")
    require(set(manifest_names) == set(predictions["model"].astype(str)),
            "candidate manifest/prediction model sets differ")
    require(manifest_body.get("profile") == model_contract.get("profile"),
            "candidate manifest profile differs from payload")
    require(int(manifest_body.get("random_state")) == 17,
            "candidate manifest random_state must be 17")
    if is_v3 or is_v4:
        expected_models = (
            (17 if str(model_contract.get("profile")) == "full" else 16)
            if is_v4
            else (15 if str(model_contract.get("profile")) == "full" else 14)
        )
        require(len(manifest_names) == expected_models,
                f"{'v4' if is_v4 else 'v3'} {model_contract.get('profile')} "
                f"candidate manifest must contain exactly {expected_models} models")
        require("duration_tvtp_hurdle" in manifest_names,
                f"{'v4' if is_v4 else 'v3'} candidate manifest omits duration_tvtp_hurdle")
        if is_v4:
            expected_v4_names = (
                V4_FULL_MODELS
                if str(model_contract.get("profile")) == "full"
                else V4_STANDARD_MODELS
            )
            require(
                set(manifest_names) == expected_v4_names,
                "v4 candidate manifest model set differs from the frozen suite",
            )
        transition_manifest = manifest_body.get("transition_research")
        require(isinstance(transition_manifest, dict),
                f"{'v4' if is_v4 else 'v3'} manifest transition_research metadata missing")
        require(transition_manifest.get("horizons_weeks") == [1, 4, 13],
                f"{'v4' if is_v4 else 'v3'} manifest transition horizons mismatch")
        require(
            transition_manifest.get("feature_set_version")
            == (V4_FEATURE_SET_VERSION if is_v4 else V3_FEATURE_SET_VERSION),
            f"{'v4' if is_v4 else 'v3'} manifest transition feature-set version mismatch",
        )
    require(
        tuple(item.get("id") for item in payload.get("states", [])) == STATE_ORDER,
        "payload state definitions are missing or misordered",
    )

    required_prediction_columns = {
        "origin_date", "target_date", "model", "evaluation_split",
        "current_state", "actual", "predicted", "train_size", "gap",
        "fallback", "fallback_reason", *PROBABILITY_COLUMNS,
    }
    require_columns(predictions, required_prediction_columns, "oos-predictions.csv")
    require_columns(
        splits,
        {
            "origin_date", "target_date", "train_size", "train_start",
            "last_train_origin", "last_train_target", "purged_origin_count",
            "first_purged_origin", "gap", "evaluation_split",
        },
        "walk-forward-splits.csv",
    )
    require_columns(
        leaderboard,
        {
            "model", "selected", "log_loss", "calibration_error",
            "n_predictions", "fallback_count",
        },
        "model-leaderboard.csv",
    )

    predictions["fallback"] = boolean_series(predictions["fallback"], "fallback")
    leaderboard["selected"] = boolean_series(leaderboard["selected"], "selected")
    probability = predictions[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    require(np.isfinite(probability).all(), "OOS probabilities contain non-finite values")
    require(((probability >= -1e-12) & (probability <= 1.0 + 1e-12)).all(),
            "OOS probabilities outside [0,1]")
    require(np.allclose(probability.sum(axis=1), 1.0, atol=1e-10),
            "OOS probability rows do not sum to one")
    expected_prediction = np.asarray(STATE_ORDER, dtype=object)[probability.argmax(axis=1)]
    require((predictions["predicted"].astype(str).to_numpy() == expected_prediction).all(),
            "OOS predicted state does not match ordered argmax")
    valid_states = set(STATE_ORDER)
    for column in ("current_state", "actual", "predicted"):
        require(set(predictions[column].astype(str)).issubset(valid_states),
                f"invalid state in OOS {column}")
    require(not predictions.duplicated(["model", "origin_date"]).any(),
            "duplicate model/origin OOS rows")
    one_week_predictions = predictions.assign(_horizon=1)
    require_calendar_horizon(
        one_week_predictions,
        "origin_date",
        "target_date",
        "_horizon",
        "main OOS",
    )
    require((predictions["gap"].astype(int) == 1).all(), "OOS gap must be one week")
    fallback_rows = predictions.loc[predictions["fallback"]]
    require(fallback_rows["fallback_reason"].fillna("").astype(str).str.len().gt(0).all(),
            "fallback OOS row is missing fallback_reason")

    origin_contract_fields = [
        "origin_date", "target_date", "evaluation_split", "current_state",
        "actual", "train_size", "gap",
    ]
    origin_contract = predictions[origin_contract_fields].drop_duplicates()
    require(not origin_contract.duplicated("origin_date").any(),
            "models disagree on an origin target/split/label/train contract")
    reference_origin_keys = origin_contract[["origin_date", "target_date"]].sort_values(
        ["origin_date", "target_date"], ignore_index=True
    )
    for model_name, model_rows in predictions.groupby("model", sort=False):
        model_keys = model_rows[["origin_date", "target_date"]].sort_values(
            ["origin_date", "target_date"], ignore_index=True
        )
        require(
            model_keys.equals(reference_origin_keys),
            f"main model {model_name} lacks strict common OOS origins",
        )
    require(not splits.duplicated("origin_date").any(), "duplicate split audit origin")
    merged = origin_contract.merge(
        splits[[
            "origin_date", "target_date", "evaluation_split", "train_size", "gap"
        ]],
        on="origin_date", suffixes=("_prediction", "_audit"), validate="one_to_one",
    )
    require(len(merged) == len(origin_contract) == len(splits),
            "prediction/split origin sets differ")
    require((merged["target_date_prediction"] == merged["target_date_audit"]).all(),
            "prediction/split target_date mismatch")
    require((merged["evaluation_split_prediction"] == merged["evaluation_split_audit"]).all(),
            "prediction/split segment mismatch")
    require((merged["train_size_prediction"].astype(int)
             == merged["train_size_audit"].astype(int)).all(),
            "prediction/split train_size mismatch")
    require((merged["gap_prediction"].astype(int)
             == merged["gap_audit"].astype(int)).all(),
            "prediction/split gap mismatch")
    require_calendar_horizon(
        splits.assign(_horizon=1),
        "origin_date",
        "target_date",
        "_horizon",
        "main split audit",
    )
    require((splits["gap"].astype(int) == 1).all(), "split gap must be one week")
    require((splits["purged_origin_count"].astype(int) == 1).all(),
            "split must purge exactly one origin")
    require((splits["last_train_target"] < splits["origin_date"]).all(),
            "last training target reaches or exceeds OOS origin")
    require((splits["first_purged_origin"] < splits["origin_date"]).all(),
            "purged origin must precede OOS origin")
    require(splits["train_size"].astype(int).is_monotonic_increasing,
            "expanding train_size is not monotonic")

    champion = str(model_contract.get("champion"))
    require(set(leaderboard["model"]) == set(predictions["model"]),
            "leaderboard/prediction model sets differ")
    require(int(leaderboard["selected"].sum()) == 1,
            "leaderboard must select exactly one model")
    selected_model = str(leaderboard.loc[leaderboard["selected"], "model"].iloc[0])
    require(selected_model == champion, "CSV selected model differs from payload champion")

    split_values = set(predictions["evaluation_split"].astype(str))
    profile = str(model_contract.get("profile"))
    chronological_split = mode == "live" or (mode == "demo" and is_v4)
    if chronological_split:
        require(split_values == {"selection", "holdout"},
                f"chronological OOS split values are invalid: {sorted(split_values)}")
        require_columns(
            leaderboard,
            {
                "selection_log_loss", "selection_calibration_error",
                "selection_n_predictions", "selection_fallback_count",
            },
            "chronological model-leaderboard.csv",
        )
        selection_end = model_contract.get("selection_end")
        require(selection_end, "chronological payload is missing model.selection_end")
        cutoff = pd.to_datetime(selection_end, utc=True)
        selection = predictions.loc[predictions["evaluation_split"] == "selection"]
        holdout = predictions.loc[predictions["evaluation_split"] == "holdout"]
        required_origins = 3 if profile == "quick" else 12
        require(selection["origin_date"].nunique() >= required_origins,
                "insufficient selection origins")
        require(holdout["origin_date"].nunique() >= required_origins,
                "insufficient holdout origins")
        require((selection["target_date"] < cutoff).all(),
                "selection target at/after selection_end")
        require((holdout["target_date"] >= cutoff).all(),
                "holdout target before selection_end")
        selection_metrics = probability_metrics(selection)
        holdout_metrics = probability_metrics(holdout)
        payload_selection_diagnostics = model_contract.get(
            "selection_diagnostics"
        )
        require(
            isinstance(payload_selection_diagnostics, list)
            and payload_selection_diagnostics,
            "payload selection diagnostics are missing",
        )
        selection_threshold = selection_minimum_log_loss_improvement(
            payload_selection_diagnostics,
            context="payload selection diagnostics",
        )
        expected_champion, recomputed_selection_diagnostics = (
            choose_selection_champion(
                selection_metrics,
                selection,
                minimum_log_loss_improvement=selection_threshold,
            )
        )
        published_segment = "holdout"
    else:
        require(split_values == {"legacy"},
                f"demo/legacy split values are invalid: {sorted(split_values)}")
        require(model_contract.get("selection_end") is None,
                "legacy payload unexpectedly has selection_end")
        selection_metrics = None
        holdout_metrics = probability_metrics(predictions)
        expected_champion = choose_legacy_champion(holdout_metrics)
        recomputed_selection_diagnostics = None
        selection = predictions.iloc[0:0]
        holdout = predictions
        published_segment = "legacy"

    require(expected_champion == champion,
            f"champion was not selected from the permitted segment: "
            f"expected {expected_champion}, got {champion}")
    if chronological_split:
        require(profile == "quick" or selection["origin_date"].nunique() >= 300,
                "standard/full must evaluate the full pre-2023 OOS selection era")
        minimum_models = (
            (17 if profile == "full" else 16)
            if is_v4
            else ((15 if profile == "full" else 14) if is_v3 else 13)
        )
        require(predictions["model"].nunique() >= minimum_models,
                f"expanded live suite must contain at least {minimum_models} models")
        if profile in {"standard", "full"}:
            expected_count = (
                (17 if profile == "full" else 16)
                if is_v4
                else ((15 if profile == "full" else 14) if is_v3 else 13)
            )
            require(len(manifest_names) == expected_count,
                    f"{profile} live manifest must contain exactly "
                    f"{expected_count} models")
        diagnostics_path = artifacts / "selection-diagnostics.csv"
        diagnostics_csv = read_csv(diagnostics_path)
        diagnostics_csv["selected"] = boolean_series(
            diagnostics_csv["selected"], "selection diagnostic selected"
        )
        diagnostics_csv["gate_passed"] = boolean_series(
            diagnostics_csv["gate_passed"], "selection diagnostic gate_passed"
        )
        payload_diagnostics = model_contract.get("selection_diagnostics")
        require(isinstance(payload_diagnostics, list) and payload_diagnostics,
                "payload selection diagnostics are missing")
        require(set(diagnostics_csv["model"].astype(str)) == set(predictions["model"]),
                "selection diagnostics model set mismatch")
        assert recomputed_selection_diagnostics is not None
        recomputed_by_model = recomputed_selection_diagnostics.set_index("model")
        csv_by_model = diagnostics_csv.set_index(
            diagnostics_csv["model"].astype(str)
        )
        payload_by_model = {
            str(row.get("model")): row for row in payload_diagnostics
        }
        require(set(payload_by_model) == set(recomputed_by_model.index),
                "payload selection diagnostics model set mismatch")
        for model_name, expected in recomputed_by_model.iterrows():
            csv_row = csv_by_model.loc[model_name]
            payload_row = payload_by_model[model_name]
            require(bool(csv_row["selected"]) == bool(expected["selected"]),
                    f"selection diagnostic selected mismatch for {model_name}")
            require(str(csv_row["gate_reason"]) == str(expected["gate_reason"]),
                    f"selection diagnostic gate reason mismatch for {model_name}")
            require(bool(payload_row.get("selected")) == bool(expected["selected"]),
                    f"payload selection diagnostic mismatch for {model_name}")
            require(bool(csv_row["gate_passed"]) == bool(expected["gate_passed"]),
                    f"selection diagnostic gate_passed mismatch for {model_name}")
            require(bool(payload_row.get("gate_passed")) == bool(expected["gate_passed"]),
                    f"payload selection gate_passed mismatch for {model_name}")
            require(str(payload_row.get("gate_reason")) == str(expected["gate_reason"]),
                    f"payload selection gate_reason mismatch for {model_name}")
            for field in ("reference_model", "gate_reason"):
                require(str(csv_row[field]) == str(expected[field]),
                        f"selection CSV {field} mismatch for {model_name}")
                require(str(payload_row.get(field)) == str(expected[field]),
                        f"selection payload {field} mismatch for {model_name}")
            for field in (
                "fallback_count",
                "n_predictions",
                "bootstrap_block_weeks",
                "bootstrap_effective_block_weeks",
                "bootstrap_resamples",
                "bootstrap_seed",
            ):
                expected_value = int(expected[field])
                require(int(csv_row[field]) == expected_value,
                        f"selection CSV {field} mismatch for {model_name}")
                require(int(payload_row.get(field)) == expected_value,
                        f"selection payload {field} mismatch for {model_name}")
            for field in (
                "log_loss",
                "reference_log_loss",
                "absolute_log_loss_improvement",
                "brier",
                "reference_brier",
                "brier_difference",
                "raw_p_value",
                "holm_adjusted_p_value",
                "alpha",
                "minimum_log_loss_improvement",
                "brier_tolerance",
            ):
                expected_value = float(expected[field])
                csv_value = float(csv_row[field])
                payload_value = payload_row.get(field)
                if np.isnan(expected_value):
                    require(np.isnan(csv_value) and payload_value is None,
                            f"selection diagnostic {field} mismatch for {model_name}")
                else:
                    require(np.isclose(csv_value, expected_value, atol=1e-10),
                            f"selection CSV {field} mismatch for {model_name}")
                    require(np.isclose(float(payload_value), expected_value, atol=1e-7),
                            f"selection payload {field} mismatch for {model_name}")
    leaderboard_by_model = leaderboard.set_index("model")
    for model_name, metrics in holdout_metrics.iterrows():
        row = leaderboard_by_model.loc[model_name]
        for field in (
            "log_loss",
            "brier",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "transition_recall",
        ):
            require(np.isclose(float(row[field]), float(metrics[field]), atol=1e-10),
                    f"{model_name} primary {field} is not {published_segment} performance")
        require(np.isclose(float(row["calibration_error"]),
                           metrics["calibration_error"], atol=1e-10),
                f"{model_name} primary calibration_error mismatch")
        require(int(row["n_predictions"]) == int(metrics["n_predictions"]),
                f"{model_name} primary n_predictions mismatch")
        require(int(row["fallback_count"]) == int(metrics["fallback_count"]),
                f"{model_name} primary fallback_count mismatch")
        if selection_metrics is not None:
            selected_metrics = selection_metrics.loc[model_name]
            require(np.isclose(float(row["selection_log_loss"]),
                               selected_metrics["log_loss"], atol=1e-10),
                    f"{model_name} selection_log_loss mismatch")
            require(int(row["selection_n_predictions"]) ==
                    int(selected_metrics["n_predictions"]),
                    f"{model_name} selection_n_predictions mismatch")

    payload_leaderboard = model_contract.get("leaderboard", [])
    require(isinstance(payload_leaderboard, list), "payload leaderboard must be a list")
    payload_models = {str(row.get("name")): row for row in payload_leaderboard}
    require(set(payload_models) == set(leaderboard_by_model.index),
            "payload/CSV leaderboard model sets differ")
    require(sum(bool(row.get("selected")) for row in payload_leaderboard) == 1,
            "payload leaderboard must select exactly one model")
    require(bool(payload_models[champion].get("selected")),
            "payload champion is not selected in leaderboard")
    for model_name, csv_row in leaderboard_by_model.iterrows():
        json_row = payload_models[str(model_name)]
        for field in (
            "log_loss",
            "brier",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "transition_recall",
            "calibration_error",
        ):
            require(np.isclose(float(json_row[field]), float(csv_row[field]), atol=1e-7),
                    f"payload/CSV {field} mismatch for {model_name}")

    diagnostic = model_contract.get("holdout_diagnostic")
    if mode == "live":
        require(isinstance(diagnostic, dict),
                "live payload is missing model.holdout_diagnostic")
        ranked_holdout = leaderboard[["model", "log_loss"]].copy()
        ranked_holdout["model"] = ranked_holdout["model"].astype(str)
        ranked_holdout["log_loss"] = pd.to_numeric(
            ranked_holdout["log_loss"], errors="raise"
        )
        ranked_holdout = ranked_holdout.sort_values(
            ["log_loss", "model"], ignore_index=True
        )
        champion_row = ranked_holdout.loc[
            ranked_holdout["model"] == champion
        ].iloc[0]
        best_row = ranked_holdout.iloc[0]
        champion_rank = int(
            ranked_holdout.index[ranked_holdout["model"] == champion][0]
        ) + 1
        regret = max(
            0.0, float(champion_row["log_loss"]) - float(best_row["log_loss"])
        )
        threshold = 0.05
        expected_status = "weak_generalization" if regret > threshold else "ok"
        require(diagnostic.get("status") == expected_status,
                "holdout diagnostic status mismatch")
        require(diagnostic.get("applicable") is True,
                "live holdout diagnostic must be applicable")
        require(diagnostic.get("selection_locked") is True,
                "live holdout diagnostic must keep selection locked")
        require(int(diagnostic.get("champion_rank")) == champion_rank,
                "holdout diagnostic champion_rank mismatch")
        require(int(diagnostic.get("model_count")) == len(ranked_holdout),
                "holdout diagnostic model_count mismatch")
        require(diagnostic.get("champion_model") == champion,
                "holdout diagnostic champion_model mismatch")
        require(diagnostic.get("best_model") == str(best_row["model"]),
                "holdout diagnostic best_model mismatch")
        for field, expected in (
            ("champion_log_loss", float(champion_row["log_loss"])),
            ("best_log_loss", float(best_row["log_loss"])),
            ("absolute_regret", regret),
            ("material_regret_threshold", threshold),
        ):
            require(np.isclose(float(diagnostic.get(field)), expected, atol=1e-7),
                    f"holdout diagnostic {field} mismatch")
        if expected_status == "weak_generalization":
            require(meta.get("status") == "degraded",
                    "weak holdout generalization must degrade meta.status")
            warnings = meta.get("warnings", [])
            require(
                isinstance(warnings, list)
                and any(
                    "일반화 경고" in str(item)
                    and "champion을 교체하지 않았습니다" in str(item)
                    for item in warnings
                ),
                "weak holdout generalization warning is missing",
            )
    elif diagnostic is not None:
        require(isinstance(diagnostic, dict), "demo holdout_diagnostic must be an object")
        require(diagnostic.get("status") == "ok",
                "demo holdout diagnostic status must be ok")
        require(diagnostic.get("applicable") is False,
                "demo holdout diagnostic must be non-applicable")
        require(diagnostic.get("selection_locked") is False,
                "demo holdout diagnostic must not claim frozen selection")

    weekly = payload.get("weekly")
    require(isinstance(weekly, list) and weekly, "payload weekly must be non-empty")
    weekly_dates = [str(row.get("date")) for row in weekly]
    require(weekly_dates == sorted(set(weekly_dates)),
            "payload weekly dates must be unique and increasing")
    for index, row in enumerate(weekly):
        for estimate_name in ("current", "next_week"):
            estimate = row.get(estimate_name, {})
            require(str(estimate.get("state")) in STATE_ORDER,
                    f"weekly[{index}].{estimate_name}.state invalid")
            validate_probability_object(
                estimate.get("probabilities"),
                f"weekly[{index}].{estimate_name}.probabilities",
            )
        transition = float(row.get("transition_probability"))
        require(0.0 <= transition <= 1.0,
                f"weekly[{index}].transition_probability outside [0,1]")

    champion_history = holdout.loc[holdout["model"] == champion].copy()
    if chronological_split and (is_v3 or is_v4):
        # The h=13 transition diagnostic applies a clean embargo: diagnostic
        # origins themselves begin on/after the cutoff, whereas the legacy
        # one-week main holdout contains one origin immediately before it.
        # The dashboard publishes the intersection of both honest tracks.
        transition_cutoff = pd.to_datetime(
            model_contract["transition_selection_end"], utc=True
        )
        champion_history = champion_history.loc[
            champion_history["origin_date"] >= transition_cutoff
        ]
    history_dates = {
        timestamp.date().isoformat() for timestamp in champion_history["origin_date"]
    }
    payload_dates = set(weekly_dates)
    require(history_dates.issubset(payload_dates),
            "payload omits champion published-segment origins")
    require(len(payload_dates - history_dates) == 1,
            "payload must add exactly one latest forecast beyond OOS history")
    if chronological_split:
        selection_dates = {
            timestamp.date().isoformat() for timestamp in selection["origin_date"]
        }
        require(selection_dates.isdisjoint(payload_dates),
                "selection-period origin leaked into published weekly history")

    weekly_by_date = {str(row["date"]): row for row in weekly}
    for _, prediction in champion_history.iterrows():
        origin_key = prediction["origin_date"].date().isoformat()
        row = weekly_by_date[origin_key]
        next_week = row["next_week"]
        require(next_week.get("model") == champion,
                f"payload model mismatch at {origin_key}")
        require(str(next_week.get("date")) == prediction["target_date"].date().isoformat(),
                f"payload target date mismatch at {origin_key}")
        for state in STATE_ORDER:
            require(np.isclose(float(next_week["probabilities"][state]),
                               float(prediction[f"p_{state}"]), atol=1e-7),
                    f"payload/OOS probability mismatch at {origin_key}: {state}")
        expected_transition = round(
            1.0 - float(prediction[f"p_{prediction['current_state']}"]), 8
        )
        require(np.isclose(float(row["transition_probability"]),
                           expected_transition, atol=1e-7),
                f"payload transition_probability mismatch at {origin_key}")
        require(bool(next_week.get("fallback")) == bool(prediction["fallback"]),
                f"payload fallback mismatch at {origin_key}")

    latest_date = next(iter(payload_dates - history_dates))
    latest = weekly_by_date[latest_date]["next_week"]
    require(latest.get("model") == champion, "latest forecast model differs from champion")
    require(bool(model_contract.get("latest_forecast_fallback")) ==
            bool(latest.get("fallback")), "latest fallback metadata mismatch")

    transition_summary = None
    if is_v3 or is_v4:
        transition_summary = audit_transition_outputs(
            payload,
            artifacts,
            main_predictions=predictions,
            main_champion=champion,
            main_published_split=published_segment,
        )

    structural_summary = None
    if is_v4:
        feature_summary = audit_feature_manifest(payload, artifacts)
        transition_predictions = read_csv(
            artifacts / "transition-oos-predictions.csv",
            ("origin_date", "target_start", "target_end"),
        )
        prospective_transition_frames = [
            read_csv(
                artifacts / filename,
                ("origin_date", "target_start", "target_end"),
            )
            for filename in (
                "transition-forecasts.csv",
                "transition-candidate-forecasts.csv",
            )
        ]
        state_evidence_summary = audit_v4_state_evidence(
            payload,
            artifacts,
            transition_predictions=transition_predictions,
            main_predictions=predictions,
            prospective_transition_frames=prospective_transition_frames,
        )
        weekly_evidence_summary = audit_v4_weekly_forecast_evidence(
            payload, artifacts
        )
        joint_summary = audit_joint_predictions(predictions, transition_predictions)
        stacking_summary = audit_stacking_weights(predictions, artifacts)
        ablation_summary = audit_feature_ablation(
            payload,
            artifacts,
            main_predictions=predictions,
            feature_manifest=feature_summary,
        )
        latest_structural_summary = audit_structural_forecasts(
            artifacts, historical_predictions=predictions
        )
        survival_summary = audit_joint_survival_forecasts(artifacts)
        structural_summary = {
            "feature_manifest": {
                key: value
                for key, value in feature_summary.items()
                if key != "group_features"
            },
            "state_evidence": state_evidence_summary,
            "weekly_forecast_evidence": weekly_evidence_summary,
            "joint_oos": joint_summary,
            "stacking": stacking_summary,
            "ablation": ablation_summary,
            "latest_forecast": latest_structural_summary,
            "survival": survival_summary,
        }

    return {
        "ok": True,
        "mode": mode,
        "profile": profile,
        "champion": champion,
        "selection_end": model_contract.get("selection_end"),
        "models": int(predictions["model"].nunique()),
        "origins": {
            "selection": int(selection["origin_date"].nunique()),
            published_segment: int(holdout["origin_date"].nunique()),
        },
        "payload_weeks": len(weekly),
        "oos_fallback_rows": int(predictions["fallback"].sum()),
        "latest_forecast_fallback": bool(latest.get("fallback")),
        "holdout_diagnostic": diagnostic,
        "transition": transition_summary,
        "structural": structural_summary,
        "frozen_v3_baseline": frozen_v3_summary,
        "structural_preregistration": preregistration_summary,
        "payload": str(payload_path),
        "artifacts": str(artifacts),
    }


def _audit_publication_live() -> dict[str, Any]:
    """Audit the authoritative reviewed publication, never an ignored V4 path."""

    payload_path = PROJECT_ROOT / "publication/live/regime-results.json"
    comparison_path = PROJECT_ROOT / "publication/live/v5-vs-v4-comparison.json"
    manifest_path = PROJECT_ROOT / "publication/live/generation-manifest.json"
    require(
        payload_path.is_file() and not payload_path.is_symlink(),
        "active publication payload is missing/non-regular",
    )
    payload_raw = payload_path.read_bytes()
    payload = json.loads(payload_raw.decode("utf-8"))
    require(isinstance(payload, dict), "active publication payload must be an object")
    meta = payload.get("meta")
    require(isinstance(meta, Mapping), "active publication metadata is missing")
    require(
        meta.get("result_version") == V5_RESULT_VERSION,
        "publication-live requires the active V5 contract; frozen/stale V4 cannot pass",
    )
    require(meta.get("mode") == "live", "active publication must use live mode")
    require(
        isinstance(meta.get("freshness"), Mapping)
        and meta["freshness"].get("status") == "current",
        "active publication is not current",
    )
    _require_wall_clock_freshness(meta)
    validate_dashboard_payload(payload)
    lifecycle = validate_lifecycle_consistency(payload)
    require(
        lifecycle["publication"] == "reviewed_publication"
        and lifecycle["deployment"] == "operating",
        "active publication lifecycle is not operating+reviewed_publication",
    )
    validate_reviewed_candidate_hash(payload)
    require(
        comparison_path.is_file() and not comparison_path.is_symlink(),
        "active publication comparison sidecar is missing/non-regular",
    )
    comparison_raw = comparison_path.read_bytes()
    comparison = json.loads(comparison_raw.decode("utf-8"))
    require(isinstance(comparison, dict), "active comparison sidecar must be an object")
    validate_v5_comparison_sidecar(
        comparison,
        payload=payload,
        payload_raw=payload_raw,
    )
    manifest_document = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(
        isinstance(manifest_document, Mapping),
        "active generation manifest must be an object",
    )
    selection_required = (
        manifest_document.get("schema_version")
        == GENERATION_MANIFEST_SCHEMA_VERSION
    )
    generation = validate_generation_manifest(
        manifest_path,
        require_comparison=True,
        require_selection_family=selection_required,
        require_artifacts=False,
    )
    require(
        generation["payload_path"].resolve() == payload_path.resolve(),
        "active generation manifest points to a different payload",
    )
    require(
        generation["comparison_path"].resolve() == comparison_path.resolve(),
        "active generation manifest points to a different comparison",
    )
    if selection_required:
        selection_path = PROJECT_ROOT / "publication/live/selection-family-audit.json"
        require(
            generation["selection_family_path"].resolve()
            == selection_path.resolve(),
            "active generation manifest points to a different selection-family audit",
        )
    return {
        "ok": True,
        "target": "publication-live",
        "contract": "v5",
        "generation_id": meta.get("generation_id"),
        "data_as_of": meta.get("data_as_of"),
        "champion": payload.get("model", {}).get("champion"),
        "lifecycle": lifecycle,
        "generation_manifest_sha256": generation["manifest_sha256"],
        "payload": str(payload_path),
        "comparison": str(comparison_path),
    }


def _require_wall_clock_freshness(
    meta: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> None:
    """Reject an old live snapshot even if its serialized status says current."""

    freshness = meta.get("freshness")
    require(isinstance(freshness, Mapping), "active freshness metadata is missing")
    maximum_age = freshness.get("maximum_age_days")
    require(
        type(maximum_age) is int and maximum_age >= 0,
        "active freshness maximum_age_days is invalid",
    )
    raw_data_as_of = meta.get("data_as_of")
    require(isinstance(raw_data_as_of, str), "active data_as_of is invalid")
    try:
        data_as_of = datetime.fromisoformat(raw_data_as_of.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AuditFailure("active data_as_of is invalid") from exc
    require(
        data_as_of.tzinfo is not None and data_as_of.utcoffset() is not None,
        "active data_as_of must include timezone",
    )
    checked_at = now or datetime.now(timezone.utc)
    require(
        checked_at.tzinfo is not None and checked_at.utcoffset() is not None,
        "audit time must include timezone",
    )
    age = checked_at.astimezone(timezone.utc) - data_as_of.astimezone(timezone.utc)
    require(age >= timedelta(0), "active data_as_of cannot be in the future")
    require(
        age <= timedelta(days=maximum_age),
        "active publication is stale at audit time",
    )


def _audit_local_generation(
    manifest_path: Path,
    *,
    payload_path: Path | None = None,
    comparison_path: Path | None = None,
    selection_family_path: Path | None = None,
    artifact_directory: Path | None = None,
) -> dict[str, Any]:
    generation = validate_generation_manifest(
        manifest_path,
        artifact_directory=artifact_directory,
        payload_path_override=payload_path,
        comparison_path_override=comparison_path,
        selection_family_path_override=selection_family_path,
    )
    summary = audit(
        generation["payload_path"],
        generation["artifact_directory"],
        "auto",
    )
    replay_input = summary.get("research_replay_input")
    if isinstance(replay_input, Mapping) and replay_input.get("status") == "verified":
        require(
            replay_input["operational_generation_input_snapshot_sha256"]
            == generation["input_snapshot"]["sha256"],
            "research replay input refers to a different operational generation",
        )
    return {
        **summary,
        "target": "local-generation",
        "generation_id": generation["generation_id"],
        "generation_manifest_sha256": generation["manifest_sha256"],
        "manifest": str(manifest_path),
    }


def _audit_frozen_v4_target() -> dict[str, Any]:
    baseline = verify_frozen_v4_baseline(project_directory=PROJECT_ROOT)
    return {
        "ok": True,
        "target": "frozen-v4",
        "role": "immutable_research_baseline_not_active_publication",
        "result_version": baseline["result_version"],
        "generation_id": baseline["generation_id"],
        "data_as_of": baseline["data_as_of"],
        "payload": str(PROJECT_ROOT / str(FROZEN_V4_BASELINE["payload_path"])),
        "artifacts": str(PROJECT_ROOT / str(FROZEN_V4_BASELINE["artifacts_path"])),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only audit of one explicitly selected Regime target"
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=("publication-live", "local-generation", "frozen-v4"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Required generation-manifest.json for --target local-generation",
    )
    parser.add_argument("--payload", type=Path)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--selection-family", type=Path)
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args(argv)
    if args.target == "local-generation" and args.manifest is None:
        parser.error("--target local-generation requires --manifest PATH")
    if args.target != "local-generation" and args.manifest is not None:
        parser.error("--manifest is only valid with --target local-generation")
    staged_values = (
        args.payload,
        args.comparison,
        args.selection_family,
        args.artifacts,
    )
    if args.target != "local-generation" and any(
        value is not None for value in staged_values
    ):
        parser.error("staged member overrides require --target local-generation")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.target == "publication-live":
            summary = _audit_publication_live()
        elif args.target == "local-generation":
            summary = _audit_local_generation(
                args.manifest,
                payload_path=args.payload,
                comparison_path=args.comparison,
                selection_family_path=args.selection_family,
                artifact_directory=args.artifacts,
            )
        else:
            summary = _audit_frozen_v4_target()
    except (
        AuditFailure,
        ContractError,
        FrozenV4BaselineError,
        IntegrityError,
        PublicContractError,
        KeyError,
        OSError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(
            json.dumps(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
