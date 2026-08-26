"""End-to-end causal regime analysis and dashboard payload assembly."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from regime_lab.analysis import (
    BenchmarkProfile,
    CausalRegimeLabeler,
    RegimeLabelConfig,
    augment_benchmark_with_structural_models,
    causal_state_durations,
    derive_causal_transition_features,
    FEATURE_ABLATION_CONTRACT,
    feature_ablation_manifest_document,
    filter_shadow_nowcast,
    forecast_next_regime,
    forecast_structural_probabilities,
    run_benchmark,
    run_feature_ablation,
    run_transition_benchmark,
)
from regime_lab.analysis.models import model_manifest, model_manifest_sha256
from regime_lab.collection import LiveCollection
from regime_lab.contract_v5 import V5_FORECAST_COMPARISON_MODELS
from regime_lab.dataset import WeeklyDataset, evidence_drivers, factor_scores
from regime_lab.evidence import (
    STATE_LABEL_HISTORY_COLUMNS,
    STATE_MEMBERSHIP_HISTORY_COLUMNS,
    WEEKLY_STATE_FORECAST_COLUMNS,
    WEEKLY_STATE_FORECAST_V5_COLUMNS,
    evidence_csv_sha256,
    state_label_history,
    state_membership_history,
    weekly_state_forecasts as build_weekly_state_forecasts,
    weekly_state_forecasts_v5,
)
from regime_lab.feature_manifest import (
    complete_feature_group_manifest,
    feature_manifest_document,
)
from regime_lab.feature_quality import (
    feature_quality_artifact_manifest,
    feature_quality_document,
)
from regime_lab.frozen_v4 import (
    FROZEN_V4_BASELINE,
    verify_frozen_v4_baseline,
)
from regime_lab.payload import STATE_DEFINITIONS, estimate_from_probabilities
from regime_lab.schema import STATE_ORDER
from regime_lab.v5_artifacts import build_v5_core_artifact_manifest
from regime_lab.v5_preflight import STRUCTURAL_V5_PREREGISTRATION_SHA256


LABEL_VERSION = "market-causal-3state-v1"
MODEL_VERSION = "weekly-nondl-structural-v4"
RESULT_VERSION = "weekly-regime-result-v4"
FEATURE_SET_VERSION = "weekly-pit-structural-v4"
TRANSITION_HORIZONS = (1, 4, 13)
BASELINE_V2 = {
    "result_version": "weekly-regime-result-v2",
    "label_version": LABEL_VERSION,
    "model_version": "weekly-nondl-walkforward-v2",
    "champion": "markov",
    "payload_sha256": "50ab693b15f5100b1e39d98356c88455b76a4a2c4a4c335e5882509568c5fe98",
    "artifacts_inventory_sha256": "09603aca14244fc00ee56f0d75a45192fc29a77c8f1a47b9927aef32d4fcbf0f",
    "captured_at": "2026-08-12",
}
BASELINE_V3 = {
    "result_version": "weekly-regime-result-v3",
    "label_version": LABEL_VERSION,
    "model_version": "weekly-nondl-structural-v3",
    "champion": "markov",
    "payload_sha256": "de93c585117b2784750f586a4f84ad99964c63081b252ad7affd7a75bd797095",
    "artifacts_inventory_sha256": "8ef3778cc8c36faff0c80e2bf094f1f11bd6966ab3b7b2d6edb84ba292aff6b9",
    "captured_at": "2026-08-13",
}
STRUCTURAL_PREREGISTRATION = {
    "path": "config/structural_v4.json",
    "sha256": "2f53ada564efca770261f16ce6eb16ec9c9782bde014de7a7d85b7b24dbe407b",
}
FROZEN_V5_BASELINE_V4 = FROZEN_V4_BASELINE
STRUCTURAL_MODEL_CONTRACT = {
    "xgb_hazard_destination": {
        "hazard_model": "binary_xgboost",
        "destination_model": "xgboost",
        "direct_jump_floor": 0.000001,
    },
    "causal_dynamic_ensemble": {
        "experts": ["markov", "xgboost", "xgb_hazard_destination"],
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
}
ABLATION_CONTRACT = FEATURE_ABLATION_CONTRACT
MATERIAL_HOLDOUT_LOG_LOSS_REGRET = 0.05
DEFAULT_SELECTION_MINIMUM_LOG_LOSS_IMPROVEMENT = 0.05
V5_SELECTION_MINIMUM_LOG_LOSS_IMPROVEMENT = 0.01


def _profile(name: str, row_count: int) -> BenchmarkProfile:
    if name == "quick":
        return BenchmarkProfile.quick()
    if name == "full":
        return BenchmarkProfile.full()
    if name != "standard":
        raise ValueError("profile must be quick, standard, or full")
    # The bounded standard profile retains the full available 2023+ holdout;
    # selection origins are added separately by the time-split allocator.
    origins = min(260, max(36, row_count - 522))
    return BenchmarkProfile(
        name="standard",
        max_origins=origins,
        minimum_train_weeks=520,
        random_forest_trees=160,
        extra_trees=160,
        hist_gradient_iterations=160,
        svm_calibration_splits=3,
        hmm_iterations=60,
    )


def _json_number(value: object, digits: int = 8) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits) if np.isfinite(number) else None


def _current_estimate(
    probabilities: pd.Series,
    hard_state: str,
) -> dict[str, Any]:
    estimate = estimate_from_probabilities(
        {state: float(probabilities[state]) for state in STATE_ORDER}
    )
    estimate["state"] = hard_state
    estimate["confidence"] = estimate["probabilities"][hard_state]
    estimate["method"] = "causal_rule_filtered_evidence"
    return estimate


def _market_context(
    canonical: pd.DataFrame,
    features: pd.DataFrame,
    at: pd.Timestamp,
) -> dict[str, Any]:
    spy = canonical["spy_close"].astype(float)
    log_return = np.log(spy.where(spy > 0)).diff()
    simple_return_26w = spy.div(spy.shift(26)).sub(1.0)
    realized = log_return.rolling(13, min_periods=7).std(ddof=0) * np.sqrt(52.0)
    drawdown = spy / spy.rolling(52, min_periods=26).max() - 1.0

    def feature(name: str) -> pd.Series:
        if name not in features:
            return pd.Series(np.nan, index=canonical.index, dtype=float)
        return pd.to_numeric(features[name], errors="coerce").reindex(
            canonical.index
        )

    sector_breadth = feature(
        "market_group__gics_sector__positive_return_share_4w"
    )
    credit_relative_log = feature(
        "market_spread__high_yield_investment_grade__relative_return_13w"
    )
    credit_relative = np.expm1(credit_relative_log)
    anfci_change = feature("anfci__change_4w")

    def percentile(series: pd.Series) -> float | None:
        current_value = pd.to_numeric(
            series.reindex(pd.DatetimeIndex([at])), errors="coerce"
        ).iloc[0]
        if not np.isfinite(current_value):
            return None
        numeric = pd.to_numeric(series, errors="coerce").loc[:at].tail(52).dropna()
        if len(numeric) < 26:
            return None
        return _json_number(float((numeric <= float(current_value)).mean()))

    def metric(
        *,
        label: str,
        series: pd.Series,
        display_format: str,
        method: str,
        window_weeks: int,
    ) -> dict[str, Any]:
        current_value = pd.to_numeric(
            series.reindex(pd.DatetimeIndex([at])), errors="coerce"
        ).iloc[0]
        return {
            "label": label,
            "value": _json_number(current_value),
            "format": display_format,
            "method": method,
            "window_weeks": window_weeks,
            "percentile_52w": percentile(series),
        }

    return {
        "spy_trend_26w": metric(
            label="SPY 26주 추세",
            series=simple_return_26w,
            display_format="signed_percent",
            method="simple_total_return",
            window_weeks=26,
        ),
        "spy_realized_vol_13w": metric(
            label="SPY 13주 변동성",
            series=realized,
            display_format="plain_percent",
            method="weekly_log_return_std_annualized",
            window_weeks=13,
        ),
        "spy_drawdown_52w": metric(
            label="SPY 52주 고점 대비",
            series=drawdown,
            display_format="signed_percent",
            method="price_over_trailing_high_minus_one",
            window_weeks=52,
        ),
        "gics_sector_breadth_4w": metric(
            label="섹터 상승 비중 · 4주",
            series=sector_breadth,
            display_format="plain_percent",
            method="positive_return_share",
            window_weeks=4,
        ),
        "hyg_lqd_relative_13w": metric(
            label="HYG − LQD · 13주",
            series=credit_relative,
            display_format="signed_percent",
            method="relative_simple_total_return",
            window_weeks=13,
        ),
        "anfci_change_4w": metric(
            label="ANFCI 변화 · 4주",
            series=anfci_change,
            display_format="signed_number",
            method="index_point_change",
            window_weeks=4,
        ),
    }


def _leaderboard_rows(table: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, (_, row) in enumerate(table.iterrows(), start=1):
        values: dict[str, Any] = {
            "rank": rank,
            "name": str(row["model"]),
            "selected": bool(row.get("selected", False)),
            "is_champion": bool(row.get("selected", False)),
        }
        for name in (
            "log_loss",
            "brier",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "transition_precision",
            "transition_recall",
            "calibration_error",
        ):
            values[name] = _json_number(row.get(name))
        values["n_predictions"] = int(row.get("n_predictions", 0))
        values["fallback_count"] = int(row.get("fallback_count", 0))
        values["selection_log_loss"] = _json_number(
            row.get("selection_log_loss")
        )
        values["selection_calibration_error"] = _json_number(
            row.get("selection_calibration_error")
        )
        values["selection_n_predictions"] = int(
            row.get("selection_n_predictions", 0)
        )
        rows.append(values)
    return rows


def holdout_diagnostic(
    leaderboard: pd.DataFrame,
    champion: str,
    *,
    selection_locked: bool,
) -> dict[str, Any]:
    """Summarize frozen-champion generalization without changing selection.

    The primary leaderboard is the holdout table only when ``selection_locked``
    is true.  Legacy/demo runs therefore return a safe, explicitly
    non-applicable diagnostic instead of relabelling in-sample/OOS metrics as a
    frozen holdout.
    """

    if not isinstance(leaderboard, pd.DataFrame) or leaderboard.empty:
        raise ValueError("holdout diagnostic requires a non-empty leaderboard")
    missing = {"model", "log_loss"}.difference(leaderboard.columns)
    if missing:
        raise ValueError(f"holdout diagnostic missing columns: {sorted(missing)}")
    model_count = int(len(leaderboard))
    if champion not in set(leaderboard["model"].astype(str)):
        raise ValueError("champion is absent from leaderboard")
    base: dict[str, Any] = {
        "status": "ok",
        "applicable": bool(selection_locked),
        "selection_locked": bool(selection_locked),
        "metric": "multiclass_log_loss",
        "material_regret_threshold": MATERIAL_HOLDOUT_LOG_LOSS_REGRET,
        "champion_rank": None,
        "model_count": model_count,
        "champion_model": champion,
        "champion_log_loss": None,
        "best_model": None,
        "best_log_loss": None,
        "absolute_regret": None,
    }
    if not selection_locked:
        return base

    ranked = leaderboard[["model", "log_loss"]].copy()
    ranked["model"] = ranked["model"].astype(str)
    ranked["log_loss"] = pd.to_numeric(ranked["log_loss"], errors="raise")
    if not np.isfinite(ranked["log_loss"].to_numpy(dtype=float)).all():
        raise ValueError("holdout leaderboard contains non-finite log_loss")
    ranked = ranked.sort_values(["log_loss", "model"], ignore_index=True)
    champion_row = ranked.loc[ranked["model"] == champion].iloc[0]
    best_row = ranked.iloc[0]
    champion_loss = float(champion_row["log_loss"])
    best_loss = float(best_row["log_loss"])
    regret = max(0.0, champion_loss - best_loss)
    base.update(
        {
            "status": (
                "weak_generalization"
                if regret > MATERIAL_HOLDOUT_LOG_LOSS_REGRET
                else "ok"
            ),
            "champion_rank": int(ranked.index[ranked["model"] == champion][0]) + 1,
            "champion_log_loss": round(champion_loss, 8),
            "best_model": str(best_row["model"]),
            "best_log_loss": round(best_loss, 8),
            "absolute_regret": round(regret, 8),
        }
    )
    return base


def _holdout_generalization_warning(diagnostic: Mapping[str, Any]) -> str:
    return (
        "2023+ 사후 진단 일반화 경고: selection 구간에서 고정한 champion "
        f"{diagnostic['champion_model']}의 log loss는 "
        f"{float(diagnostic['champion_log_loss']):.4f}로, 진단 구간 최우수 "
        f"{diagnostic['best_model']}({float(diagnostic['best_log_loss']):.4f})보다 "
        f"{float(diagnostic['absolute_regret']):.4f} 높습니다. "
        "이 사후 진단 결과를 이용해 champion을 교체하지 않았습니다."
    )


def _selection_diagnostic_rows(table: pd.DataFrame | None) -> list[dict[str, Any]]:
    if table is None or table.empty:
        return []
    rows: list[dict[str, Any]] = []
    for _, row in table.iterrows():
        values: dict[str, Any] = {
            "model": str(row["model"]),
            "reference_model": str(row["reference_model"]),
            "selected": bool(row["selected"]),
            "gate_passed": bool(row["gate_passed"]),
            "gate_reason": str(row["gate_reason"]),
        }
        for name in (
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
            values[name] = _json_number(row.get(name))
        for name in (
            "fallback_count",
            "n_predictions",
            "bootstrap_block_weeks",
            "bootstrap_effective_block_weeks",
            "bootstrap_resamples",
            "bootstrap_seed",
        ):
            values[name] = int(row.get(name, 0))
        rows.append(values)
    return rows


def _transition_leaderboard_rows(table: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialize the binary event benchmark without inventing missing metrics."""

    rows: list[dict[str, Any]] = []
    for _, row in table.iterrows():
        rows.append(
            {
                "horizon_weeks": int(row["horizon"]),
                "model": str(row["model"]),
                "selected": bool(row.get("selected", False)),
                "evaluation_split": str(row["evaluation_split"]),
                "binary_log_loss": _json_number(row.get("log_loss")),
                "brier": _json_number(row.get("brier")),
                "average_precision": _json_number(row.get("average_precision")),
                "precision": _json_number(row.get("precision")),
                "recall": _json_number(row.get("recall")),
                "false_alarms_per_year": _json_number(
                    row.get("false_alarms_per_year")
                ),
                "n_predictions": int(row.get("n_predictions", 0)),
                "event_count": int(row.get("event_count", 0)),
                "non_event_count": int(row.get("non_event_count", 0)),
                "fallback_count": int(row.get("fallback_count", 0)),
                "calibration_fallback_count": int(
                    row.get("calibration_fallback_count", 0)
                ),
            }
        )
    return rows


def _shadow_nowcast_summary(
    current_probabilities: pd.DataFrame,
    canonical_states: pd.Series,
) -> dict[str, Any]:
    shadow = filter_shadow_nowcast(current_probabilities.loc[:, list(STATE_ORDER)])
    summary = shadow.summary()
    agreement = shadow.states.eq(canonical_states.reindex(shadow.states.index))
    return {
        "status": "shadow_only",
        "method": "causal_explicit_duration_filter",
        "canonical_target": False,
        "canonical_agreement_rate": _json_number(agreement.mean()),
        "latest_state": str(shadow.states.iloc[-1]),
        **summary,
    }


def _transition_risk_history(
    transition_benchmark: Any,
) -> dict[pd.Timestamp, dict[str, dict[str, Any]]]:
    """Build the published champion event-risk track plus latest forecasts."""

    frame = transition_benchmark.predictions
    published = frame.loc[
        frame["evaluation_split"].eq("retrospective_diagnostic")
    ].copy()
    champion_rows: list[pd.DataFrame] = []
    for horizon, champion in transition_benchmark.champions_by_horizon.items():
        champion_rows.append(
            published.loc[
                published["horizon"].eq(int(horizon))
                & published["model"].eq(str(champion))
            ]
        )
    published_champions = (
        pd.concat(champion_rows, ignore_index=True)
        if champion_rows
        else published.iloc[0:0]
    )
    combined = pd.concat(
        [published_champions, transition_benchmark.latest_forecasts()],
        ignore_index=True,
    )
    output: dict[pd.Timestamp, dict[str, dict[str, Any]]] = {}
    for _, row in combined.iterrows():
        origin = pd.Timestamp(row["origin_date"])
        horizon = int(row["horizon"])
        output.setdefault(origin, {})[f"{horizon}w"] = {
            "probability": round(float(row["p_change"]), 8),
            "target_end": pd.Timestamp(row["target_end"]).date().isoformat(),
            "model": str(row["model"]),
            "threshold": round(float(row["threshold"]), 8),
            "fallback": bool(
                row.get("fallback", False)
                or row.get("calibration_fallback", False)
                or str(row.get("threshold_method", "")).startswith("fallback_0.5")
            ),
            "fallback_reason": "; ".join(
                reason
                for reason in (
                    str(row.get("fallback_reason", "")),
                    (
                        f"calibration:{row.get('calibration_fallback_reason', '')}"
                        if bool(row.get("calibration_fallback", False))
                        else ""
                    ),
                    (
                        f"threshold:{row.get('threshold_method', '')}"
                        if str(row.get("threshold_method", "")).startswith(
                            "fallback_0.5"
                        )
                        else ""
                    ),
                )
                if reason
            ),
        }
    return output


def _next_week_estimate(probabilities: Mapping[str, float], target_date: pd.Timestamp) -> dict[str, Any]:
    estimate = estimate_from_probabilities(probabilities)
    estimate["date"] = target_date.date().isoformat()
    estimate["method"] = "champion_walk_forward_probability"
    return estimate


def _comparison_forecast_record(row: Mapping[str, Any]) -> dict[str, Any]:
    probabilities = {
        state: float(row[f"p_{state}"])
        for state in STATE_ORDER
    }
    estimate = _next_week_estimate(
        probabilities,
        pd.Timestamp(row["target_date"]),
    )
    estimate.update(
        {
            "method": "model_comparison_walk_forward_probability",
            "model": str(row["model"]),
            "fallback": bool(row.get("fallback", False)),
            "fallback_reason": (
                ""
                if pd.isna(row.get("fallback_reason", ""))
                else str(row.get("fallback_reason", ""))
            ),
        }
    )
    return estimate


def _comparison_forecasts_by_origin(
    predictions: pd.DataFrame,
    latest_forecasts: pd.DataFrame,
    *,
    model_names: Sequence[str],
) -> dict[pd.Timestamp, list[dict[str, Any]]]:
    comparison_models = tuple(dict.fromkeys(str(name) for name in model_names))
    if not comparison_models:
        raise RuntimeError("model comparison requires at least one model")
    required_columns = {
        "origin_date",
        "target_date",
        "model",
        "fallback",
        "fallback_reason",
        *(f"p_{state}" for state in STATE_ORDER),
    }
    for label, frame in (
        ("OOS predictions", predictions),
        ("latest structural forecasts", latest_forecasts),
    ):
        missing = required_columns.difference(frame.columns)
        if missing:
            raise RuntimeError(
                f"{label} lack model comparison columns: {sorted(missing)}"
            )

    selected = pd.concat(
        [
            predictions.loc[
                predictions["model"].astype(str).isin(
                    comparison_models
                )
            ],
            latest_forecasts.loc[
                latest_forecasts["model"].astype(str).isin(
                    comparison_models
                )
            ],
        ],
        ignore_index=True,
        sort=False,
    ).copy()
    selected["origin_date"] = pd.to_datetime(
        selected["origin_date"], utc=True, errors="raise"
    )
    selected["target_date"] = pd.to_datetime(
        selected["target_date"], utc=True, errors="raise"
    )
    if selected.duplicated(["origin_date", "model"]).any():
        raise RuntimeError("model comparison forecasts duplicate an origin/model")

    result: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    for origin, group in selected.groupby("origin_date", sort=True):
        by_model = {
            str(row["model"]): row
            for row in group.to_dict(orient="records")
        }
        if set(by_model) != set(comparison_models):
            raise RuntimeError(
                f"model comparison forecasts are incomplete at {origin}"
            )
        result[pd.Timestamp(origin)] = [
            _comparison_forecast_record(by_model[model])
            for model in comparison_models
        ]
    return result


def build_dashboard_result(
    dataset: WeeklyDataset,
    collection: LiveCollection | None,
    *,
    profile_name: str = "standard",
    mode: str = "live",
    sources: Sequence[dict[str, Any]] | None = None,
    warnings: Sequence[str] = (),
    selection_end: str | pd.Timestamp | None = None,
    progress: Callable[[str], None] | None = None,
    contract_version: str = "v5",
    fx_result: Any | None = None,
    latest_fx_context: Mapping[str, Any] | None = None,
    h10_source: Mapping[str, Any] | None = None,
    checkpoint_directory: str | Path | None = None,
    source_fingerprint_sha256: str | None = None,
    minimum_log_loss_improvement: float | None = None,
) -> tuple[dict[str, Any], Any]:
    if contract_version not in {"v4", "v5"}:
        raise ValueError("contract_version must be v4 or v5")
    if contract_version != "v5" and (
        checkpoint_directory is not None
        or source_fingerprint_sha256 is not None
    ):
        raise ValueError("walk-forward checkpoints are V5-only")
    if mode == "live" and profile_name == "quick":
        raise ValueError(
            "live mode does not permit the three-origin quick smoke profile"
        )
    if contract_version == "v5":
        verify_frozen_v4_baseline()
    contract_minimum_log_loss_improvement = (
        V5_SELECTION_MINIMUM_LOG_LOSS_IMPROVEMENT
        if contract_version == "v5"
        else DEFAULT_SELECTION_MINIMUM_LOG_LOSS_IMPROVEMENT
    )
    resolved_minimum_log_loss_improvement = (
        contract_minimum_log_loss_improvement
        if minimum_log_loss_improvement is None
        else float(minimum_log_loss_improvement)
    )
    if not np.isclose(
        resolved_minimum_log_loss_improvement,
        contract_minimum_log_loss_improvement,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            f"{contract_version} minimum_log_loss_improvement must be "
            f"{contract_minimum_log_loss_improvement:.2f}"
        )
    canonical = dataset.canonical.loc[dataset.canonical["spy_close"].notna()].copy()
    features = dataset.features.reindex(canonical.index)
    if len(canonical) < 650:
        raise RuntimeError(
            f"insufficient weekly history for a 520-week train window: {len(canonical)}"
        )

    labeler = CausalRegimeLabeler(
        RegimeLabelConfig(price_column="spy_close", minimum_fit_observations=260)
    )
    label_fit_weeks = 520
    labeler.fit(canonical.iloc[:label_fit_weeks])
    states = labeler.transform(canonical)
    current_probabilities = labeler.state_probabilities(canonical)
    label_score_frame = labeler.score_frame(canonical)
    label_history = state_label_history(
        states,
        current_probabilities,
        label_score_frame["risk_score"],
        lower_threshold=float(labeler.lower_threshold_),
        upper_threshold=float(labeler.upper_threshold_),
        hysteresis_fraction=labeler.config.hysteresis_fraction,
        probability_temperature=labeler.config.probability_temperature,
    )
    features = features.copy()
    for column in label_score_frame.columns:
        features[f"regime_boundary__{column}"] = label_score_frame[column]
    features = derive_causal_transition_features(
        features,
        states,
        risk_score_col="regime_boundary__risk_score",
        lower_threshold=labeler.lower_threshold_,
        upper_threshold=labeler.upper_threshold_,
    ).drop(columns=["current_state"])
    scores = factor_scores(features, label_score_frame)

    benchmark_profile = _profile(profile_name, len(features))
    if mode == "live" and selection_end is None:
        raise ValueError("live mode requires an explicit model selection_end")
    benchmark_selection_end = selection_end or "2023-01-01"
    # The interactive quick budget contains only ten holdout origins, so the
    # production 12-row split floor cannot fit by construction.  Three rows per
    # side is an explicit smoke-profile compromise; standard/full retain the
    # stricter default and are the only profiles suitable for model claims.
    split_prediction_minimum = 3 if profile_name == "quick" else 12
    base_benchmark = run_benchmark(
        features,
        states,
        profile=benchmark_profile,
        # HMM is retained as an optional full-profile challenger.  Re-fitting it
        # at every standard weekly origin is disproportionately expensive and is
        # not needed to satisfy the broad statistical/ML comparison.
        include_hmm=profile_name == "full",
        gap=1,
        minimum_train_weeks=benchmark_profile.minimum_train_weeks,
        random_state=17,
        selection_end=benchmark_selection_end,
        selection_max_origins=3 if profile_name == "quick" else None,
        model_workers=1 if profile_name == "quick" else 4,
        minimum_selection_predictions=split_prediction_minimum,
        minimum_holdout_predictions=split_prediction_minimum,
        minimum_log_loss_improvement=(
            resolved_minimum_log_loss_improvement
        ),
        progress=progress,
        checkpoint_directory=checkpoint_directory,
        source_fingerprint_sha256=source_fingerprint_sha256,
    )
    if progress is not None:
        progress("1·4·13주 국면 이탈 위험 워크포워드 시작")
    transition_benchmark = run_transition_benchmark(
        features,
        states,
        horizons=TRANSITION_HORIZONS,
        profile=benchmark_profile,
        include_xgboost=True,
        include_joint_survival=True,
        minimum_train_weeks=benchmark_profile.minimum_train_weeks,
        selection_end=benchmark_selection_end,
        selection_max_origins=3 if profile_name == "quick" else None,
        minimum_selection_predictions=split_prediction_minimum,
        minimum_diagnostic_predictions=split_prediction_minimum,
        minimum_inner_predictions=split_prediction_minimum,
        random_state=17,
        progress=progress,
    )
    benchmark = augment_benchmark_with_structural_models(
        base_benchmark,
        transition_benchmark,
        include_multiscale=contract_version == "v5",
        random_state=17,
    )
    if progress is not None:
        progress("구조 피처 공통표본 ablation 시작")
    ablation = run_feature_ablation(
        features,
        states,
        dataset.feature_group_manifest,
        benchmark,
        profile=benchmark_profile,
        selection_end=benchmark_selection_end,
        gap=1,
        minimum_train_weeks=benchmark_profile.minimum_train_weeks,
        selection_max_origins=3 if profile_name == "quick" else None,
        minimum_selection_predictions=split_prediction_minimum,
        minimum_holdout_predictions=split_prediction_minimum,
        random_state=17,
        model_workers=1 if profile_name == "quick" else 4,
        progress=progress,
    )
    ablation_manifest = feature_ablation_manifest_document(ablation.manifest)
    ablation_contract = {
        **ABLATION_CONTRACT,
        "manifest_sha256": ablation_manifest["sha256"],
    }
    completed_feature_groups = complete_feature_group_manifest(
        tuple(features.columns),
        dataset.feature_group_manifest,
    )
    feature_manifest = feature_manifest_document(
        completed_feature_groups,
        feature_set_version=FEATURE_SET_VERSION,
    )
    feature_quality = feature_quality_document(features)
    champion_predictions = benchmark.champion_predictions().copy()
    manifest_model_frame = getattr(benchmark, "predictions", benchmark.leaderboard)
    evaluated_model_names = tuple(
        manifest_model_frame["model"].astype(str).drop_duplicates().tolist()
    )
    suite_manifest = model_manifest(
        benchmark_profile,
        random_state=17,
        names=evaluated_model_names,
    )
    suite_manifest_hash = model_manifest_sha256(
        benchmark_profile,
        random_state=17,
        names=evaluated_model_names,
    )
    generalization = holdout_diagnostic(
        benchmark.leaderboard,
        benchmark.champion,
        # Demo runs exercise the same chronological split as production, but
        # synthetic outcomes are not an external holdout claim.  Keep that
        # distinction explicit so the payload and independent audit agree.
        selection_locked=(mode == "live" and benchmark.selection_end is not None),
    )
    latest_date = pd.Timestamp(features.index[-1])
    latest_base_probabilities: dict[str, pd.Series] = {}
    for base_name in ("markov", "xgboost"):
        latest_base_probabilities[base_name] = forecast_next_regime(
            features,
            states,
            champion_name=base_name,
            as_of=latest_date,
            profile=benchmark_profile,
            gap=1,
            minimum_train_weeks=benchmark_profile.minimum_train_weeks,
            random_state=17,
        )
    binary_latest = transition_benchmark.latest_candidate_forecasts(
        horizon=1,
        model="binary_xgboost",
    ).sort_values("origin_date").iloc[-1]
    if pd.Timestamp(binary_latest["origin_date"]) != latest_date:
        raise RuntimeError("latest binary hazard origin does not match model cutoff")
    binary_fallback = bool(
        binary_latest.get("fallback", False)
        or binary_latest.get("calibration_fallback", False)
    )
    structural_latest = forecast_structural_probabilities(
        origin_date=latest_date,
        current_state=str(states.loc[latest_date]),
        markov_probability=[
            float(latest_base_probabilities["markov"][state])
            for state in STATE_ORDER
        ],
        xgboost_probability=[
            float(latest_base_probabilities["xgboost"][state])
            for state in STATE_ORDER
        ],
        binary_xgboost_p_change=float(binary_latest["p_change"]),
        historical_oos_predictions=benchmark.predictions.loc[
            benchmark.predictions["model"].isin(
                ("markov", "xgboost", "xgb_hazard_destination")
            )
        ],
        expert_fallbacks={
            "markov": bool(
                latest_base_probabilities["markov"].attrs.get("fallback", False)
            ),
            "xgboost": bool(
                latest_base_probabilities["xgboost"].attrs.get("fallback", False)
            ),
            "xgb_hazard_destination": binary_fallback,
        },
        current_duration_weeks=int(causal_state_durations(states).iloc[-1]),
        include_multiscale=contract_version == "v5",
    )
    if contract_version == "v5":
        oos_scale_forecasts = benchmark.multiscale_scale_forecasts
        latest_scale_forecasts = structural_latest.multiscale_scale_predictions
        if oos_scale_forecasts is None or latest_scale_forecasts is None:
            raise RuntimeError("V5 multiscale ensemble evidence was not produced")
        multiscale_scale_forecasts = pd.concat(
            [oos_scale_forecasts, latest_scale_forecasts],
            ignore_index=True,
            sort=False,
        ).sort_values(
            ["origin_date", "target_date", "row_role", "scale_half_life_weeks"],
            ignore_index=True,
        )
        latest_multiscale_weights = structural_latest.stacking_weights.loc[
            structural_latest.stacking_weights["ensemble_model"].astype(str).eq(
                "causal_multiscale_ensemble"
            )
        ]
        if len(latest_multiscale_weights) != 9:
            raise RuntimeError(
                "V5 latest multiscale ensemble must emit nine expert weights"
            )
        benchmark = replace(
            benchmark,
            stacking_weights=pd.concat(
                [benchmark.stacking_weights, latest_multiscale_weights],
                ignore_index=True,
                sort=False,
            ).sort_values(
                [
                    "origin_date",
                    "ensemble_model",
                    "half_life_weeks",
                    "expert",
                ],
                ignore_index=True,
            ),
            multiscale_scale_forecasts=multiscale_scale_forecasts,
        )
    if benchmark.champion in {
        "xgb_hazard_destination",
        "causal_dynamic_ensemble",
        "causal_multiscale_ensemble",
    }:
        structural_row = structural_latest.probabilities.loc[
            structural_latest.probabilities["model"].eq(benchmark.champion)
        ].iloc[0]
        latest_probability = pd.Series(
            [float(structural_row[f"p_{state}"]) for state in STATE_ORDER],
            index=STATE_ORDER,
            name="next_week_probability",
        )
        latest_probability.attrs.update(
            {
                "fallback": bool(structural_row["fallback"]),
                "fallback_reason": str(structural_row["fallback_reason"]),
            }
        )
    elif benchmark.champion in latest_base_probabilities:
        latest_probability = latest_base_probabilities[benchmark.champion]
    else:
        latest_probability = forecast_next_regime(
            features,
            states,
            champion_name=benchmark.champion,
            as_of=latest_date,
            profile=benchmark_profile,
            gap=1,
            minimum_train_weeks=benchmark_profile.minimum_train_weeks,
            random_state=17,
        )

    predictions: dict[pd.Timestamp, dict[str, Any]] = {}
    for _, row in champion_predictions.iterrows():
        origin = pd.Timestamp(row["origin_date"])
        predictions[origin] = {
            "probabilities": {
                state: float(row[f"p_{state}"])
                for state in STATE_ORDER
            },
            "target_date": pd.Timestamp(row["target_date"]),
            "fallback": bool(row.get("fallback", False)),
            "fallback_reason": str(row.get("fallback_reason", "")),
        }
    latest_fallback = bool(latest_probability.attrs.get("fallback", False))
    latest_fallback_reason = str(
        latest_probability.attrs.get("fallback_reason", "")
    )
    predictions[latest_date] = {
        "probabilities": {
            state: float(latest_probability[state]) for state in STATE_ORDER
        },
        "target_date": latest_date + timedelta(days=7),
        "fallback": latest_fallback,
        "fallback_reason": latest_fallback_reason,
    }
    latest_weight_map = {
        str(row["expert"]): float(row["weight"])
        for _, row in structural_latest.stacking_weights.loc[
            structural_latest.stacking_weights["ensemble_model"].astype(str).eq(
                "causal_dynamic_ensemble"
            )
        ].iterrows()
    }
    structural_forecasts = structural_latest.probabilities.copy()
    structural_forecasts["target_date"] = latest_date + timedelta(days=7)
    structural_forecasts["p_change"] = float(binary_latest["p_change"])
    for expert in ("markov", "xgboost", "xgb_hazard_destination"):
        structural_forecasts[f"weight_{expert}"] = latest_weight_map[expert]
    binary_forecast_row = {
        "origin_date": latest_date,
        "target_date": latest_date + timedelta(days=7),
        "model": "binary_xgboost",
        "current_state": str(states.loc[latest_date]),
        "p_risk_on": np.nan,
        "p_transition": np.nan,
        "p_risk_off": np.nan,
        "predicted": "",
        "fallback": binary_fallback,
        "fallback_reason": ";".join(
            reason
            for reason in (
                str(binary_latest.get("fallback_reason", "")),
                (
                    f"calibration:{binary_latest.get('calibration_fallback_reason', '')}"
                    if bool(binary_latest.get("calibration_fallback", False))
                    else ""
                ),
            )
            if reason
        ),
        "source_role": "departure_hazard",
        "ensemble_weight": np.nan,
        "binary_xgboost_p_change": float(binary_latest["p_change"]),
        "p_change": float(binary_latest["p_change"]),
        **{
            f"weight_{expert}": latest_weight_map[expert]
            for expert in ("markov", "xgboost", "xgb_hazard_destination")
        },
    }
    structural_forecasts = pd.concat(
        [structural_forecasts, pd.DataFrame([binary_forecast_row])],
        ignore_index=True,
        sort=False,
    ).sort_values("model", ignore_index=True)
    comparison_models = tuple(V5_FORECAST_COMPARISON_MODELS)
    if contract_version == "v5" and benchmark.champion not in comparison_models:
        comparison_models = (*comparison_models, benchmark.champion)
        champion_forecast_row = {
            "origin_date": latest_date,
            "target_date": latest_date + timedelta(days=7),
            "model": benchmark.champion,
            "current_state": str(states.loc[latest_date]),
            **{
                f"p_{state}": float(latest_probability[state])
                for state in STATE_ORDER
            },
            "predicted": str(latest_probability.astype(float).idxmax()),
            "fallback": latest_fallback,
            "fallback_reason": latest_fallback_reason,
            "source_role": "selected_champion_comparison",
        }
        structural_forecasts = pd.concat(
            [structural_forecasts, pd.DataFrame([champion_forecast_row])],
            ignore_index=True,
            sort=False,
        ).sort_values("model", ignore_index=True)
    comparison_forecasts = (
        _comparison_forecasts_by_origin(
            benchmark.predictions,
            structural_forecasts,
            model_names=comparison_models,
        )
        if contract_version == "v5"
        else {}
    )
    joint_survival_forecasts = structural_latest.survival_probabilities.rename(
        columns={
            "horizon": "horizon_weeks",
            "p_change": "cumulative_p_change",
        }
    ).copy()
    joint_survival_forecasts.insert(0, "origin_date", latest_date)
    joint_survival_forecasts["role"] = "shadow_coherence_benchmark"
    joint_survival_forecasts["one_week_hazard"] = float(binary_latest["p_change"])
    joint_survival_forecasts["step_hazards"] = [
        json.dumps(
            [float(binary_latest["p_change"])] * int(horizon),
            separators=(",", ":"),
        )
        for horizon in joint_survival_forecasts["horizon_weeks"]
    ]
    transition_history = _transition_risk_history(transition_benchmark)

    weekly: list[dict[str, Any]] = []
    for origin in sorted(predictions):
        prediction = predictions[origin]
        next_probabilities = prediction["probabilities"]
        target_date = prediction["target_date"]
        prediction_fallback = bool(prediction["fallback"])
        current_state = str(states.loc[origin])
        current = _current_estimate(current_probabilities.loc[origin], current_state)
        next_week = _next_week_estimate(next_probabilities, target_date)
        next_week["model"] = benchmark.champion
        next_week["fallback"] = prediction_fallback
        next_week["fallback_reason"] = prediction["fallback_reason"]
        if prediction_fallback:
            next_week["method"] = "class_prior_fallback"
        transition_probability = 1.0 - float(next_probabilities[current_state])
        transition_risk = transition_history.get(origin)
        if transition_risk is None:
            # The multiclass forecast remains the authoritative 1w alias.  A
            # row without all three event horizons must not masquerade as v3.
            continue
        transition_risk["1w"] = {
            **transition_risk["1w"],
            "probability": round(transition_probability, 8),
            "target_end": pd.Timestamp(target_date).date().isoformat(),
            "model": benchmark.champion,
            "threshold": 0.5,
            "fallback": prediction_fallback,
            "fallback_reason": prediction["fallback_reason"],
        }
        if set(transition_risk) != {"1w", "4w", "13w"}:
            continue
        source_health = str(dataset.health.reindex([origin]).iloc[0])
        structural_fallback = any(
            bool(transition_risk[key]["fallback"]) for key in ("4w", "13w")
        )
        result_health = (
            "degraded"
            if prediction_fallback or structural_fallback
            else source_health
        )
        fallback_reasons = [
            f"{key}: {transition_risk[key]['fallback_reason']}"
            for key in ("4w", "13w")
            if transition_risk[key]["fallback"]
        ]
        week_result = {
            "date": origin.date().isoformat(),
            "data_as_of": origin.isoformat(),
            "current": current,
            "next_week": next_week,
            "transition_probability": round(transition_probability, 8),
            "transition_risk": transition_risk,
            "scores": {
                name: _json_number(scores.loc[origin, name]) or 0.0
                for name in ("trend", "stress", "macro", "financial_conditions")
            },
            "market": _market_context(canonical, dataset.features, origin),
            "top_drivers": evidence_drivers(features, origin),
            "health": {
                "status": result_health,
                "reason": (
                    f"champion fit 실패로 class prior 사용: {prediction['fallback_reason']}"
                    if prediction_fallback
                    else (
                        "구조적 전환위험 fallback: " + " | ".join(fallback_reasons)
                        if structural_fallback
                        else (
                            "모든 required series가 cutoff 이전 값으로 결합됨"
                            if source_health == "ok"
                            else "하나 이상의 required series가 누락·지연 상태"
                        )
                    )
                ),
            },
        }
        if contract_version == "v5":
            model_forecasts = comparison_forecasts.get(origin)
            if model_forecasts is None:
                raise RuntimeError(
                    f"model comparison forecasts are missing for {origin}"
                )
            week_result["model_forecasts"] = model_forecasts
        weekly.append(week_result)

    forecast_evidence = build_weekly_state_forecasts(weekly)
    evidence_artifacts = {
        "state_label_history": {
            "path": "state-label-history.csv",
            "row_count": int(len(label_history)),
            "sha256": evidence_csv_sha256(
                label_history,
                STATE_LABEL_HISTORY_COLUMNS,
            ),
            "label_fit_weeks": label_fit_weeks,
            "label_fit_end": pd.Timestamp(labeler.train_end_).isoformat(),
            "initial_state": "transition",
        },
        "weekly_state_forecasts": {
            "path": "weekly-state-forecasts.csv",
            "row_count": int(len(forecast_evidence)),
            "sha256": evidence_csv_sha256(
                forecast_evidence,
                WEEKLY_STATE_FORECAST_COLUMNS,
            ),
        },
    }

    source_rows = list(sources or (collection.sources if collection else ()))
    collection_status = collection.overall_health.value if collection else "degraded"
    latest_week_health = weekly[-1]["health"]["status"]
    overall_status = (
        "ok"
        if (
            collection_status == "ok"
            and latest_week_health == "ok"
            and mode == "live"
            and generalization["status"] != "weak_generalization"
        )
        else "degraded"
    )
    model_start = pd.Timestamp(champion_predictions["origin_date"].min())
    model_end = pd.Timestamp(champion_predictions["target_date"].max())
    generated_at = pd.Timestamp.now(tz="UTC")
    cutoff = collection.model_cutoff if collection else latest_date.to_pydatetime()
    all_warnings = list(warnings)
    if latest_fallback:
        all_warnings.append(
            "최신 다음 주 예측은 champion fit 실패로 class-prior fallback을 사용했습니다."
        )
    if mode == "live" and generalization["status"] == "weak_generalization":
        all_warnings.append(_holdout_generalization_warning(generalization))
    if mode == "live":
        all_warnings.extend(
            [
                "Alpha Vantage adjusted history는 현재 제공된 조정계수 기준이며 신규 snapshot부터 변경을 추적합니다.",
                "ALFRED는 일자 단위 vintage이므로 정확한 발표시각이 없으면 18:00 ET proxy를 적용해 다음 주부터 사용합니다.",
                "Top drivers는 SHAP가 아니라 과거 52주 표준화 evidence proxy입니다.",
            ]
        )
    payload: dict[str, Any] = {
        "meta": {
            "schema_version": "1.0.0",
            "result_version": RESULT_VERSION,
            "generated_at": generated_at.isoformat(),
            "generation_id": generated_at.strftime("%Y%m%dT%H%M%S.%fZ"),
            "data_as_of": cutoff.isoformat(),
            "mode": mode,
            "status": overall_status,
            "timezone": "America/New_York",
            "cutoff_policy": "completed US market week, Friday 16:00 ET; unknown release time uses next week",
            "transition_alert_thresholds": {"medium": 0.4, "high": 0.65},
            "transition_probability_definition": "1 - P(next_week equals current regime)",
            "transition_risk_definition": (
                "P(at least one departure from the origin regime within h weeks)"
            ),
            "supported_date_range": f"{weekly[0]['date']}–{weekly[-1]['date']}",
            "warnings": list(dict.fromkeys(all_warnings)),
        },
        "states": STATE_DEFINITIONS,
        "model": {
            "champion": benchmark.champion,
            "version": MODEL_VERSION,
            "label_version": LABEL_VERSION,
            "feature_set_version": FEATURE_SET_VERSION,
            "baseline_v2": BASELINE_V2,
            "baseline_v3": BASELINE_V3,
            "structural_preregistration": STRUCTURAL_PREREGISTRATION,
            "feature_manifest_sha256": feature_manifest["sha256"],
            "evidence_artifacts": evidence_artifacts,
            "structural_models": STRUCTURAL_MODEL_CONTRACT,
            "ablation": ablation_contract,
            "primary_horizon_weeks": 1,
            "transition_horizons_weeks": list(TRANSITION_HORIZONS),
            "selection_metric": "multiclass_log_loss",
            "selection_protocol": "material paired block-bootstrap gate on pre-2023 OOS only",
            "selection_status": "provisional_predeployment",
            "post_selection_period_role": "retrospective_external_period_diagnostic",
            "profile": benchmark.profile.name,
            "validation_period": f"{model_start.date()}–{model_end.date()}",
            "selection_end": (
                benchmark.selection_end.date().isoformat()
                if benchmark.selection_end is not None
                else None
            ),
            "selection_period": (
                f"{pd.Timestamp(benchmark.predictions_for_split('selection')['target_date'].min()).date()}–"
                f"{pd.Timestamp(benchmark.predictions_for_split('selection')['target_date'].max()).date()}"
                if benchmark.selection_end is not None
                else None
            ),
            "holdout_period": (
                f"{pd.Timestamp(benchmark.predictions_for_split('holdout')['target_date'].min()).date()}–"
                f"{pd.Timestamp(benchmark.predictions_for_split('holdout')['target_date'].max()).date()}"
                if benchmark.selection_end is not None
                else None
            ),
            "latest_forecast_fallback": latest_fallback,
            "candidate_manifest_sha256": suite_manifest_hash,
            "candidate_manifest": suite_manifest,
            "selection_diagnostics": _selection_diagnostic_rows(
                getattr(benchmark, "selection_diagnostics", None)
            ),
            "holdout_diagnostic": generalization,
            "state_order": list(STATE_ORDER),
            "leaderboard": _leaderboard_rows(benchmark.leaderboard),
            "transition_champions": {
                f"{int(horizon)}w": str(name)
                for horizon, name in transition_benchmark.champions_by_horizon.items()
            },
            "transition_selection_end": (
                transition_benchmark.selection_end.date().isoformat()
                if transition_benchmark.selection_end is not None
                else "2023-01-01"
            ),
            "transition_leaderboard": _transition_leaderboard_rows(
                transition_benchmark.leaderboard
            ),
            "transition_candidate_status": transition_benchmark.candidate_status.to_dict(
                orient="records"
            ),
            "shadow_nowcast": _shadow_nowcast_summary(
                current_probabilities,
                states,
            ),
        },
        "weekly": weekly,
        "sources": source_rows,
        "feature_catalog": list(dataset.feature_catalog),
    }
    if contract_version == "v5":
        payload["model"]["forecast_comparison"] = {
            "role": "research_comparison",
            "horizon_weeks": 1,
            "models": list(comparison_models),
        }
    # The primary multiclass result remains the public return for API
    # compatibility; the CLI reads these additive attributes for the v4 audit
    # bundle.  They stay outside the public JSON to keep the page concise.
    benchmark = replace(
        benchmark,
        state_label_history=label_history,
        weekly_state_forecasts=forecast_evidence,
    )
    object.__setattr__(benchmark, "transition_benchmark", transition_benchmark)
    object.__setattr__(benchmark, "feature_ablation", ablation)
    object.__setattr__(benchmark, "feature_manifest", feature_manifest)
    object.__setattr__(
        benchmark,
        "feature_quality_report",
        feature_quality,
    )
    object.__setattr__(benchmark, "structural_forecasts", structural_forecasts)
    object.__setattr__(
        benchmark,
        "joint_survival_forecasts",
        joint_survival_forecasts,
    )
    if contract_version == "v5":
        from regime_lab.v5 import build_v5_payload, run_v5_directional_benchmark

        if progress is not None:
            progress("v5 최초 이탈 방향 benchmark 시작")
        directional = run_v5_directional_benchmark(
            features,
            states,
            profile_name=profile_name,
            selection_end=benchmark_selection_end,
        )
        bootstrap_resamples = 199 if profile_name == "quick" else 1_999
        if progress is not None:
            progress("v5 지속기간·조건부 자산 통계 조립")
        fx_ablation_evidence: list[pd.DataFrame] = []
        payload, conditional_outcomes = build_v5_payload(
            payload,
            canonical=canonical,
            features=features,
            states=states,
            directional=directional,
            baseline_v4=FROZEN_V5_BASELINE_V4,
            structural_preregistration_sha256=(
                STRUCTURAL_V5_PREREGISTRATION_SHA256
            ),
            fx_result=fx_result,
            latest_fx_context=latest_fx_context,
            h10_source=h10_source,
            duration_bootstrap_resamples=bootstrap_resamples,
            outcome_bootstrap_resamples=bootstrap_resamples,
            fx_ablation_evidence_sink=fx_ablation_evidence.append,
        )
        membership_history = state_membership_history(label_history)
        forecast_evidence_v5 = weekly_state_forecasts_v5(payload["weekly"])
        payload["model"]["champion_core_feature_set_version"] = (
            FEATURE_SET_VERSION
        )
        payload["model"]["feature_quality_artifact"] = (
            feature_quality_artifact_manifest(feature_quality)
        )
        payload["model"]["evidence_artifacts"] = {
            "state_membership_history": {
                "path": "state-membership-history.csv",
                "row_count": int(len(membership_history)),
                "sha256": evidence_csv_sha256(
                    membership_history,
                    STATE_MEMBERSHIP_HISTORY_COLUMNS,
                ),
                "label_fit_weeks": label_fit_weeks,
                "label_fit_end": pd.Timestamp(labeler.train_end_).isoformat(),
                "initial_state": "transition",
                "method": "risk_score_anchor_membership",
            },
            "weekly_state_forecasts": {
                "path": "weekly-state-forecasts-v5.csv",
                "row_count": int(len(forecast_evidence_v5)),
                "sha256": evidence_csv_sha256(
                    forecast_evidence_v5,
                    WEEKLY_STATE_FORECAST_V5_COLUMNS,
                ),
            },
        }
        core_artifacts = build_v5_core_artifact_manifest(
            {
                "oos_predictions": benchmark.predictions,
                "model_leaderboard": benchmark.leaderboard,
                "walk_forward_splits": benchmark.split_audit,
                "selection_diagnostics": benchmark.selection_diagnostics,
                "stacking_weights": benchmark.stacking_weights,
                "multiscale_ensemble_scales": (
                    benchmark.multiscale_scale_forecasts
                ),
            }
        )
        payload["model"]["core_artifacts"] = core_artifacts
        multiscale_contract = payload["model"].get(
            "structural_models",
            {},
        ).get("causal_multiscale_ensemble")
        if not isinstance(multiscale_contract, dict):
            raise RuntimeError("V5 multiscale structural metadata is missing")
        multiscale_contract["sidecar"] = dict(
            core_artifacts["multiscale_ensemble_scales"]
        )
        object.__setattr__(benchmark, "directional_benchmark", directional)
        object.__setattr__(
            benchmark,
            "conditional_asset_outcomes",
            conditional_outcomes.outcomes,
        )
        object.__setattr__(
            benchmark,
            "conditional_asset_statistics",
            conditional_outcomes.statistics,
        )
        model_conditioned_outcomes = getattr(
            conditional_outcomes,
            "model_conditioned_outcomes",
            None,
        )
        model_conditioned_statistics = getattr(
            conditional_outcomes,
            "model_conditioned_statistics",
            None,
        )
        if model_conditioned_outcomes is not None:
            object.__setattr__(
                benchmark,
                "model_conditioned_asset_outcomes",
                model_conditioned_outcomes,
            )
        if model_conditioned_statistics is not None:
            object.__setattr__(
                benchmark,
                "model_conditioned_asset_statistics",
                model_conditioned_statistics,
            )
        object.__setattr__(
            benchmark,
            "state_membership_history",
            membership_history,
        )
        object.__setattr__(
            benchmark,
            "weekly_state_forecasts_v5",
            forecast_evidence_v5,
        )
        if fx_result is not None:
            object.__setattr__(benchmark, "fx_features", fx_result.features)
            object.__setattr__(benchmark, "fx_coverage", fx_result.coverage)
            if len(fx_ablation_evidence) != 1:
                raise RuntimeError("V5 FX ablation evidence was not captured exactly once")
            object.__setattr__(
                benchmark,
                "fx_ablation_oos",
                fx_ablation_evidence[0],
            )
    return payload, benchmark
