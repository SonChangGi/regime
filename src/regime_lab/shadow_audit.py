"""Derived-only real-data audit for non-canonical regime shadows.

The runner in this module deliberately consumes the retrospective reconstructed
matrix.  It does not alter the supervised label, enter the weekly candidate
set, or make any result promotion-eligible.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from regime_lab.analysis.dynamic_factor_tvtp import (
    DynamicFactorTVTPConfig,
    run_dynamic_factor_tvtp_shadow,
)
from regime_lab.analysis.labels import CausalRegimeLabeler, RegimeLabelConfig, STATE_ORDER
from regime_lab.analysis.models import BenchmarkProfile
from regime_lab.analysis.pagan_sossounov import pagan_sossounov_chronology
from regime_lab.analysis.shadow_regimes import (
    bayesian_online_changepoint_shadow,
    filter_direct_jump_hsmm_shadow,
)
from regime_lab.analysis.validation import evaluate_predictions, run_benchmark
from regime_lab.collection import last_completed_week_cutoff, weekly_cutoffs
from regime_lab.data import SQLiteSnapshotStore
from regime_lab.dataset import build_weekly_dataset
from regime_lab.integrity import canonical_json_sha256_v1
from regime_lab.io import write_json_atomic
from regime_lab.research_comparison import (
    _prepare_matrix,
    research_source_fingerprint,
)


UTC = timezone.utc
SHADOW_AUDIT_SCHEMA_VERSION = "regime-real-data-shadow-audit/1"


def _json_number(value: Any) -> float | None:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _iso(value: Any) -> str:
    return pd.Timestamp(value).isoformat()


def _load_label_evidence(
    config: Mapping[str, Any],
    *,
    database: Path,
    as_of: datetime,
    expected_index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    cutoff = as_of.astimezone(UTC)
    if cutoff != last_completed_week_cutoff(cutoff):
        raise ValueError("as_of must be an exact completed Friday 16:00 ET cutoff")
    with SQLiteSnapshotStore(database, read_only=True) as store:
        observations = store.read_last_good_observations()
    dataset = build_weekly_dataset(
        config,
        weekly_cutoffs(date(2006, 1, 1), cutoff),
        observations,
        availability_basis="reconstructed_market",
    )
    canonical = dataset.canonical.loc[
        dataset.canonical["spy_close"].notna()
    ].copy()
    canonical = canonical.reindex(expected_index)
    if canonical["spy_close"].isna().any():
        raise RuntimeError("shadow audit canonical rows differ from research matrix")
    labeler = CausalRegimeLabeler(
        RegimeLabelConfig(price_column="spy_close", minimum_fit_observations=260)
    )
    labeler.fit(canonical.iloc[:520])
    memberships = labeler.state_memberships(canonical)
    scores = labeler.score_frame(canonical)["risk_score"]
    return canonical, memberships, scores


def _monthly_raw_price(canonical: pd.DataFrame) -> pd.Series:
    if "spy_raw_close" not in canonical:
        raise KeyError("SPY raw close is required for the ex-post chronology")
    raw = pd.to_numeric(canonical["spy_raw_close"], errors="coerce").dropna()
    if raw.empty or (raw <= 0.0).any():
        raise ValueError("SPY raw close must be positive")
    if raw.index.tz is not None:
        raw.index = raw.index.tz_convert(UTC).tz_localize(None)
    # Alpha Vantage supplies weekly bars, so this is the last reported weekly
    # close in each month rather than an exact month-end daily close.
    monthly = raw.resample("ME").last().dropna()
    monthly.name = "spy_monthly_raw_close_weekly_last_proxy"
    return monthly


def _prediction_metrics(frame: pd.DataFrame, states: pd.Series, model: str) -> dict[str, Any]:
    predictions = frame.copy()
    target_lookup = states.copy()
    target_lookup.index = pd.to_datetime(target_lookup.index)
    predictions["actual"] = pd.to_datetime(predictions["target_date"]).map(target_lookup)
    if predictions["actual"].isna().any():
        raise RuntimeError("shadow predictions cannot be matched to their targets")
    predictions["model"] = model
    predictions["fallback"] = predictions.get("used_fallback", False)
    metrics = evaluate_predictions(predictions).iloc[0]
    return {
        "log_loss": _json_number(metrics["log_loss"]),
        "brier": _json_number(metrics["brier"]),
        "calibration_error": _json_number(metrics["calibration_error"]),
        "accuracy": _json_number(metrics["accuracy"]),
        "balanced_accuracy": _json_number(metrics["balanced_accuracy"]),
        "transition_precision": _json_number(metrics["transition_precision"]),
        "transition_recall": _json_number(metrics["transition_recall"]),
        "n_predictions": int(metrics["n_predictions"]),
        "fallback_count": int(metrics["fallback_count"]),
    }


def build_shadow_audit_from_inputs(
    *,
    features: pd.DataFrame,
    states: pd.Series,
    memberships: pd.DataFrame,
    risk_score: pd.Series,
    monthly_raw_price: pd.Series,
    data_as_of: datetime,
    input_metadata: Mapping[str, Any],
    source_fingerprint_sha256: str,
    minimum_train_weeks: int = 520,
    direct_jump_origins: int = 10,
    dynamic_factor_origins: int = 10,
) -> dict[str, Any]:
    """Execute bounded, matched real-data shadows without selecting a model."""

    if data_as_of.tzinfo is None or data_as_of.utcoffset() is None:
        raise ValueError("data_as_of must include a timezone")
    if not features.index.equals(states.index):
        raise ValueError("features and states must use the same index")
    if not memberships.index.equals(states.index) or tuple(memberships.columns) != STATE_ORDER:
        raise ValueError("memberships must align exactly and follow STATE_ORDER")
    finite = pd.to_numeric(risk_score, errors="coerce").dropna()
    if len(finite) < minimum_train_weeks:
        raise ValueError("risk score history is shorter than the shadow training prefix")
    emissions = memberships.reindex(finite.index)

    hsmm = filter_direct_jump_hsmm_shadow(emissions)
    bocpd = bayesian_online_changepoint_shadow(finite)
    pagan = pagan_sossounov_chronology(monthly_raw_price)

    profile = BenchmarkProfile.quick().with_overrides(
        max_origins=direct_jump_origins,
        minimum_train_weeks=minimum_train_weeks,
    )
    direct = run_benchmark(
        features,
        states,
        profile=profile,
        models=("markov", "direct_jump_tvtp_hurdle"),
        gap=1,
        minimum_train_weeks=minimum_train_weeks,
        random_state=17,
        model_workers=1,
    )
    direct_metrics = {
        str(row["model"]): {
            "log_loss": _json_number(row["log_loss"]),
            "brier": _json_number(row["brier"]),
            "calibration_error": _json_number(row["calibration_error"]),
            "n_predictions": int(row["n_predictions"]),
            "fallback_count": int(row["fallback_count"]),
        }
        for _, row in direct.leaderboard.iterrows()
    }

    dynamic = run_dynamic_factor_tvtp_shadow(
        features,
        states,
        config=DynamicFactorTVTPConfig(
            min_train_size=minimum_train_weeks,
            max_origins=dynamic_factor_origins,
        ),
    )
    dynamic_metrics = _prediction_metrics(
        dynamic.predictions,
        states,
        "dynamic_factor_tvtp",
    )
    latest_dynamic = dynamic.predictions.iloc[-1]

    turns = [
        {
            "at": _iso(row["at"]),
            "kind": str(row["kind"]),
            "confirmed_at": _iso(row["confirmed_at"]),
            "future_confirmation_months": int(row["future_confirmation_months"]),
        }
        for _, row in pagan.turning_points.iterrows()
    ]
    report: dict[str, Any] = {
        "schema_version": SHADOW_AUDIT_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "data_as_of": data_as_of.astimezone(UTC).isoformat(),
        "evidence_track": "reconstructed_oos",
        "canonical_target": False,
        "automatic_promotion_eligible": False,
        "public_release_eligible": False,
        "input": {
            **dict(input_metadata),
            "source_fingerprint_sha256": source_fingerprint_sha256,
            "membership_semantics": "distance_to_anchor_not_posterior",
        },
        "methods": {
            "filtered_hsmm": {
                "status": "executed_real_data_shadow",
                "role": hsmm.role,
                "configuration_sha256": hsmm.configuration_sha256,
                "observation_count": int(len(hsmm.states)),
                "latest_as_of": _iso(hsmm.states.index[-1]),
                "latest_state": str(hsmm.states.iloc[-1]),
                "latest_membership_filtered": {
                    state: _json_number(hsmm.probabilities.iloc[-1][state])
                    for state in STATE_ORDER
                },
                "map_direct_jump_count": int(
                    hsmm.diagnostics["map_direct_jump"].astype(bool).sum()
                ),
                "uses_backward_smoothing": False,
                "uses_supervised_target": False,
            },
            "bayesian_online_changepoint": {
                "status": "executed_real_data_shadow",
                "role": bocpd.role,
                "configuration_sha256": bocpd.configuration_sha256,
                "observation_count": int(len(bocpd.diagnostics)),
                "latest_as_of": _iso(bocpd.diagnostics.index[-1]),
                "latest_changepoint_probability": _json_number(
                    bocpd.diagnostics.iloc[-1]["changepoint_probability"]
                ),
                "latest_map_run_length": int(
                    bocpd.diagnostics.iloc[-1]["map_run_length"]
                ),
                "maximum_changepoint_probability": _json_number(
                    bocpd.diagnostics["changepoint_probability"].max()
                ),
                "posterior_probability_sum": _json_number(
                    bocpd.final_run_length_posterior.sum()
                ),
                "uses_future_observation": False,
                "uses_supervised_target": False,
            },
            "direct_jump_tvtp_hurdle": {
                "status": "executed_bounded_matched_oos_shadow",
                "role": "model_shadow_only",
                "origin_count": int(
                    direct.predictions["origin_date"].nunique()
                ),
                "gap_weeks": 1,
                "direct_jump_allowed": True,
                "metrics": direct_metrics,
                "selection_result_ignored": str(direct.champion),
            },
            "dynamic_factor_tvtp": {
                "status": "executed_bounded_matched_oos_shadow",
                "role": dynamic.role,
                "configuration_sha256": dynamic.configuration_sha256,
                "origin_count": int(len(dynamic.predictions)),
                "gap_weeks": 1,
                "causality_scope": dynamic.causality_scope,
                "vintage_safety": dynamic.vintage_safety,
                "operational_oos_eligible": dynamic.operational_oos_eligible,
                "metrics": dynamic_metrics,
                "latest": {
                    "origin_date": _iso(latest_dynamic["origin_date"]),
                    "target_date": _iso(latest_dynamic["target_date"]),
                    "predicted": str(latest_dynamic["predicted"]),
                    "probabilities": {
                        state: _json_number(latest_dynamic[f"p_{state}"])
                        for state in STATE_ORDER
                    },
                },
            },
            "pagan_sossounov": {
                "status": "executed_real_data_ex_post_audit",
                "role": pagan.role,
                "configuration_sha256": pagan.configuration_sha256,
                "label_method_spec_sha256": pagan.label_method_spec_sha256,
                "monthly_observation_count": int(len(monthly_raw_price)),
                "turning_point_count": int(len(turns)),
                "turning_points": turns,
                "price_sampling": "last_weekly_raw_close_observed_in_month_proxy",
                "exact_month_end_daily_close_available": False,
                "uses_future_observations": True,
            },
        },
        "interpretation": {
            "official_label_changed": False,
            "official_model_changed": False,
            "bounded_model_metrics_are_selection_evidence": False,
            "pagan_chronology_is_forecast_eligible": False,
        },
    }
    report["report_sha256"] = canonical_json_sha256_v1(report)
    return report


def run_real_data_shadow_audit(
    config: Mapping[str, Any],
    *,
    database: Path,
    as_of: datetime,
    direct_jump_origins: int = 10,
    dynamic_factor_origins: int = 10,
) -> dict[str, Any]:
    source_fingerprint = research_source_fingerprint(config=config)
    features, states, metadata = _prepare_matrix(
        config,
        database=database,
        as_of=as_of,
    )
    canonical, memberships, scores = _load_label_evidence(
        config,
        database=database,
        as_of=as_of,
        expected_index=features.index,
    )
    report = build_shadow_audit_from_inputs(
        features=features,
        states=states,
        memberships=memberships,
        risk_score=scores,
        monthly_raw_price=_monthly_raw_price(canonical),
        data_as_of=as_of,
        input_metadata=metadata,
        source_fingerprint_sha256=source_fingerprint,
        direct_jump_origins=direct_jump_origins,
        dynamic_factor_origins=dynamic_factor_origins,
    )
    if research_source_fingerprint(config=config) != source_fingerprint:
        raise RuntimeError("research source changed during the shadow audit")
    return report


def write_shadow_audit(path: Path, report: Mapping[str, Any]) -> Path:
    output = Path(path)
    write_json_atomic(output, dict(report))
    return output


__all__ = [
    "SHADOW_AUDIT_SCHEMA_VERSION",
    "build_shadow_audit_from_inputs",
    "run_real_data_shadow_audit",
    "write_shadow_audit",
]
