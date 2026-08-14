#!/usr/bin/env python3
"""Read-only audit of a completed regime dashboard build.

The audit intentionally reads only the published payload and supporting CSVs.
It does not open the snapshot database, refit a model, or modify artifacts.
"""

from __future__ import annotations

import argparse
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


STATE_ORDER = ("risk_on", "transition", "risk_off")
PROBABILITY_COLUMNS = tuple(f"p_{state}" for state in STATE_ORDER)
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
}
V2_RESULT_VERSION = "weekly-regime-result-v2"
V3_RESULT_VERSION = "weekly-regime-result-v3"
V3_MODEL_VERSION = "weekly-nondl-structural-v3"
V3_FEATURE_SET_VERSION = "weekly-pit-market-internals-v3"
V4_RESULT_VERSION = "weekly-regime-result-v4"
V4_MODEL_VERSION = "weekly-nondl-structural-v4"
V4_FEATURE_SET_VERSION = "weekly-pit-structural-v4"
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
MINIMUM_LOG_LOSS_IMPROVEMENT = 0.05
BRIER_TOLERANCE = 0.01


class AuditFailure(AssertionError):
    """Raised when an output contract is violated."""


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AuditFailure(message)


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
) -> tuple[str, pd.DataFrame]:
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
            if improvement + 1e-12 < MINIMUM_LOG_LOSS_IMPROVEMENT:
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
                "minimum_log_loss_improvement": MINIMUM_LOG_LOSS_IMPROVEMENT,
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
    is_v4 = payload.get("meta", {}).get("result_version") == V4_RESULT_VERSION
    if is_v4:
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
                "requested", "available", "published",
                *( ("selection_eligible",) if is_v4 else () ),
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
        if is_v4
        else published_models
    )
    prediction_models = set(predictions["model"].astype(str))
    require(
        TRANSITION_REQUIRED_MODELS.issubset(requested_models)
        and requested_models.issubset(TRANSITION_ALLOWED_MODELS),
        "transition requested candidate set is invalid",
    )
    if is_v4:
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

    if is_v4:
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
        if is_v4:
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
            require(np.isclose(float(published_risk["probability"]), float(source["p_change"]), atol=1e-7), f"weekly[{index}] {horizon}w probability/source mismatch")
            effective_fallback, effective_reason = effective_transition_fallback(source)
            require(bool(published_risk["fallback"]) == effective_fallback, f"weekly[{index}] {horizon}w effective fallback/source mismatch")
            require(str(published_risk.get("fallback_reason", "")) == effective_reason, f"weekly[{index}] {horizon}w effective fallback reason/source mismatch")

    candidate_forecast_summary = None
    if is_v4:
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


def audit(payload_path: Path, artifacts: Path, expected_mode: str) -> dict[str, Any]:
    require(payload_path.is_file() and payload_path.stat().st_size > 0,
            f"missing/empty payload: {payload_path}")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
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

    require(isinstance(payload, dict), "payload root must be an object")
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
        expected_champion, recomputed_selection_diagnostics = (
            choose_selection_champion(selection_metrics, selection)
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only audit of regime payload and walk-forward CSV artifacts"
    )
    parser.add_argument(
        "--payload", type=Path, default=Path("web/data/regime-results.json")
    )
    parser.add_argument(
        "--artifacts", type=Path, default=Path("artifacts/latest")
    )
    parser.add_argument(
        "--mode", choices=("live", "demo", "auto"), default="live",
        help="expected payload mode; default catches a stale demo payload",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = audit(args.payload, args.artifacts, args.mode)
    except (AuditFailure, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
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
