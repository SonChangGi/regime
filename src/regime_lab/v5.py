"""Opt-in v5 payload composition on top of the frozen v4 analysis path."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from regime_lab.analysis.directional import (
    DirectionalBenchmarkResult,
    MINIMUM_DESTINATION_CLASSES,
    MINIMUM_EVENT_BLOCKS,
    MINIMUM_SELECTION_EVENTS,
    markov_first_passage_probabilities,
    reconcile_directional_risk,
    run_directional_transition_benchmark,
)
from regime_lab.analysis.duration import duration_context
from regime_lab.analysis.decision_shadow import build_decision_shadow
from regime_lab.analysis.fx import FXFeatureResult, fx_context_at, unavailable_fx_context
from regime_lab.analysis.fx_ablation import (
    FX_ABLATION_OOS_COLUMNS,
    run_fx_shadow_ablation,
)
from regime_lab.analysis.outcomes import ASSETS, HORIZONS, build_conditional_asset_statistics
from regime_lab.analysis.label_spec import load_label_spec
from regime_lab.operating_contract import load_operating_contract
from regime_lab.v5_artifacts import build_v5_research_artifact_manifest


V5_SCHEMA_VERSION = "2.1.0"
V5_RESULT_VERSION = "weekly-regime-result-v5"
V5_MODEL_VERSION = "weekly-nondl-structural-v5"
V5_FEATURE_SET_VERSION = "weekly-pit-structural-v5"
STATE_ORDER = ("risk_on", "transition", "risk_off")
STATE_LABELS_KO = {
    "risk_on": "위험선호",
    "transition": "전환",
    "risk_off": "위험회피",
}
MATERIAL_CALIBRATION_DRIFT = 0.05
MINIMUM_TRANSITION_EVENTS_FOR_HEALTH = 12
LOW_TRANSITION_RECALL = 0.20
MINIMUM_FULL_HORIZON_REMAINING_FRACTION = 4.0 / 7.0
MINIMUM_CONTEXT_COVERAGE_FRACTION = 0.75
MACRO_CONTEXT_COLUMNS = (
    "payems__z_52w",
    "indpro__z_52w",
    "rsafs__z_52w",
    "houst__z_52w",
    "gdpc1__z_52w",
    "unrate__z_52w",
    "icsa__z_52w",
    "ccsa__z_52w",
)
FINANCIAL_CONTEXT_COLUMNS = (
    "nfci__z_52w",
    "nfcirisk__z_52w",
    "nfcicredit__z_52w",
    "nfcileverage__z_52w",
    "nfcinonfinleverage__z_52w",
    "stlfsi4__z_52w",
    "t10y2y__z_52w",
    "walcl__z_52w",
)
_OPERATING_CONTRACT = load_operating_contract()


def _config_document(name: str) -> tuple[dict[str, Any], str]:
    path = Path(__file__).resolve().parents[2] / "config" / name
    document = json.loads(path.read_text(encoding="utf-8"))
    return document, hashlib.sha256(
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _json_value(value: Any, *, digits: int = 8) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return round(number, digits) if math.isfinite(number) else None
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): _json_value(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _selection_contract(model: Mapping[str, Any]) -> dict[str, Any]:
    policy = _OPERATING_CONTRACT.selection_policy
    diagnostics = [
        dict(row)
        for row in model.get("selection_diagnostics", ())
        if isinstance(row, Mapping)
    ]
    champion = str(model["champion"])
    candidate_set = [str(row.get("model")) for row in diagnostics]
    passing = [
        row
        for row in diagnostics
        if row.get("gate_passed") is True and str(row.get("model")) != champion
        and row.get("is_reference") is not True
    ]
    passing.sort(
        key=lambda row: (
            float(row.get("log_loss", float("inf"))),
            float(row.get("brier", float("inf"))),
            str(row.get("model")),
        )
    )
    runner_up = str(passing[0]["model"]) if passing else None
    champion_row = next(
        (row for row in diagnostics if str(row.get("model")) == champion),
        None,
    )
    best_gate_loss = min(
        (
            float(row["log_loss"])
            for row in diagnostics
            if row.get("gate_passed") is True
            and row.get("log_loss") is not None
        ),
        default=None,
    )
    champion_loss = (
        None
        if champion_row is None or champion_row.get("log_loss") is None
        else float(champion_row["log_loss"])
    )
    if champion_row is None or champion_row.get("gate_passed") is not True:
        reason = "reference_fallback_no_challenger_passed"
    elif (
        best_gate_loss is not None
        and champion_loss is not None
        and champion_loss > best_gate_loss + 1e-12
    ):
        reason = "simplicity_tiebreak_within_tolerance"
    else:
        reason = "best_gate_passing_log_loss"
    registry, registry_sha256 = _config_document("selection-release-epochs.json")
    manifest_sha256 = model.get("candidate_manifest_sha256")
    epoch = next(
        (
            row
            for row in registry.get("epochs", ())
            if isinstance(row, Mapping)
            and row.get("candidate_manifest_sha256") == manifest_sha256
        ),
        None,
    )
    return {
        "schema_version": "regime-selection-evidence/1",
        "status": "selected_by_gate",
        "policy_sha256": _OPERATING_CONTRACT.selection_policy_sha256,
        "complexity_registry_sha256": (
            _OPERATING_CONTRACT.complexity_registry_sha256
        ),
        "candidate_set": candidate_set,
        "runner_up": runner_up,
        "selection_reason": reason,
        "simplicity_tolerance": float(policy["simplicity_tolerance"]),
        "tie_break_order": list(policy["tie_break_order"]),
        "operating_champion": str(
            _OPERATING_CONTRACT.document["models"]["official_champion"]
        ),
        # Selection rows are reconstructed from the frozen historical
        # selection window.  A live forecast clock must never relabel them as
        # prospective operational evidence.
        "selection_evidence_track": "reconstructed_oos",
        "evidence_status": "historical_reconstructed_oos",
        "selected_champion": champion,
        "statistically_indistinguishable_models": [],
        "statistical_equivalence_status": "pending_selection_sidecar",
        "release_epoch_registry": {
            "path": "config/selection-release-epochs.json",
            "sha256": registry_sha256,
            "mode": "append_only",
            "epoch_count": len(registry.get("epochs", ())),
            "current_epoch_id": None if epoch is None else epoch.get("epoch_id"),
            "current_epoch_status": (
                "unregistered_candidate_manifest"
                if epoch is None
                else epoch.get("status")
            ),
        },
        "multiplicity_defense": {
            "current_epoch_method": "holm_step_down_plus_model_confidence_set",
            "cumulative_status": (
                "legacy_not_alpha_spent"
                if epoch is not None
                else "future_epoch_registration_required"
            ),
            "future_policy": dict(registry["future_epoch_policy"]),
            "automatic_promotion_eligible": False,
        },
    }


def _forecast_contract(
    latest_week: Mapping[str, Any],
    *,
    generated_at: datetime,
    mode: str,
    evidence_track: str,
) -> dict[str, Any]:
    origin = pd.Timestamp(f"{latest_week['date']} 16:00:00", tz="America/New_York")
    target = pd.Timestamp(
        f"{latest_week['next_week']['date']} 16:00:00",
        tz="America/New_York",
    )
    decision = origin if mode == "demo" else pd.Timestamp(generated_at)
    if decision.tzinfo is None:
        decision = decision.tz_localize("UTC")
    decision = decision.tz_convert("UTC")
    origin = origin.tz_convert("UTC")
    target = target.tz_convert("UTC")
    active = decision < target
    scheduled_horizon = int((target - origin).total_seconds())
    remaining_horizon = int((target - decision).total_seconds()) if active else 0
    issue_latency = int((decision - origin).total_seconds()) if active else None
    remaining_fraction = (
        float(remaining_horizon / scheduled_horizon)
        if active and scheduled_horizon > 0
        else 0.0
    )
    timing_status = (
        "expired"
        if not active
        else (
            "full_horizon_forecast"
            if remaining_fraction + 1e-12
            >= MINIMUM_FULL_HORIZON_REMAINING_FRACTION
            else "late_nowcast"
        )
    )
    empty_ledger_hash = hashlib.sha256(b"[]").hexdigest()
    return {
        "status": "active" if active else "expired",
        "origin_at": origin.isoformat(),
        "decision_at": decision.isoformat() if active else None,
        "target_at": target.isoformat(),
        "remaining_horizon": remaining_horizon,
        "evidence_track": str(evidence_track),
        "forecast_evidence_track": str(evidence_track),
        "issue_latency_seconds": issue_latency,
        "scheduled_horizon_seconds": scheduled_horizon,
        "remaining_horizon_fraction": round(remaining_fraction, 8),
        "minimum_full_horizon_remaining_fraction": (
            MINIMUM_FULL_HORIZON_REMAINING_FRACTION
        ),
        "timing_status": timing_status,
        "prospective_ledger": {
            "schema_version": "regime-prospective-ledger-summary/1",
            "status": "pending_append" if evidence_track == "operational_oos" else "not_applicable",
            "entry_count": None if evidence_track == "operational_oos" else 0,
            "key_manifest_sha256": (
                None if evidence_track == "operational_oos" else empty_ledger_hash
            ),
            "hash_scope": "ordered_ledger_primary_keys_only",
        },
    }


def _anchored_isotonic_transition_risk(
    risk: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Project 4w/13w cumulative risk onto p1 <= p4 <= p13.

    The official one-week departure probability is an immutable anchor.  The
    two longer horizons use the exact L2 isotonic solution under that lower
    bound: apply the anchor bound, then pool an adjacent 4w/13w violation.
    """

    result = {key: deepcopy(dict(value)) for key, value in risk.items()}
    raw = {key: float(result[key]["probability"]) for key in ("1w", "4w", "13w")}
    p1 = raw["1w"]
    p4 = max(raw["4w"], p1)
    p13 = max(raw["13w"], p1)
    if p4 > p13:
        pooled = (p4 + p13) / 2.0
        p4 = pooled
        p13 = pooled
    projected = {"1w": p1, "4w": p4, "13w": p13}
    for key, value in projected.items():
        result[key]["probability"] = round(float(value), 8)
    metadata = {
        "semantics": "cumulative_first_departure_probability",
        "coherence_method": "one_week_anchored_l2_isotonic_projection_v1",
        "one_week_anchor": "official_multiclass_departure_probability",
        "raw_probabilities": {key: round(value, 8) for key, value in raw.items()},
        "adjusted": any(abs(projected[key] - raw[key]) > 1e-12 for key in raw),
    }
    return result, metadata


def _term_structure_selection_evidence(
    transition_predictions: pd.DataFrame,
    champions_by_horizon: Mapping[int, str],
    *,
    selection_end: object,
) -> dict[str, Any]:
    """Matched selection-only Brier comparison for the shape projection.

    The public weekly payload intentionally contains only the display/history
    window and can therefore begin after the selection cutoff.  The auditable
    transition OOS sidecar is the authoritative source for this comparison.
    Each retained origin must have the selected 1/4/13-week forecast and its
    realised cumulative departure event; unmatched horizon rows are excluded.
    """

    required = {
        "origin_date",
        "target_end",
        "horizon",
        "model",
        "evaluation_split",
        "actual_change",
        "p_change",
    }
    missing = sorted(required.difference(transition_predictions.columns))
    if missing:
        raise ValueError(
            "transition selection evidence is missing columns: "
            + ", ".join(missing)
        )
    if set(champions_by_horizon) != set(HORIZONS):
        raise ValueError("transition champions must cover exactly 1/4/13 weeks")

    cutoff = pd.Timestamp(selection_end).date()
    selection = transition_predictions.loc[
        transition_predictions["evaluation_split"].astype(str).eq("selection")
    ].copy()
    selection["origin_key"] = selection["origin_date"].map(
        lambda value: pd.Timestamp(value).date().isoformat()
    )
    selection["target_end_key"] = selection["target_end"].map(
        lambda value: pd.Timestamp(value).date()
    )
    selection = selection.loc[selection["target_end_key"] < cutoff]
    champion_rows: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        champion_rows.append(
            selection.loc[
                selection["horizon"].astype(int).eq(horizon)
                & selection["model"].astype(str).eq(
                    str(champions_by_horizon[horizon])
                )
            ]
        )
    selected = pd.concat(champion_rows, ignore_index=True)
    if selected.empty:
        raise ValueError(
            "transition selection evidence has no selected champion rows"
        )
    if selected.duplicated(["origin_key", "horizon"]).any():
        raise ValueError(
            "transition selection evidence has duplicate origin/horizon rows"
        )

    probability_by_origin = selected.pivot(
        index="origin_key", columns="horizon", values="p_change"
    )
    actual_by_origin = selected.pivot(
        index="origin_key", columns="horizon", values="actual_change"
    )
    if not set(HORIZONS).issubset(probability_by_origin.columns) or not set(
        HORIZONS
    ).issubset(actual_by_origin.columns):
        raise ValueError(
            "transition selection evidence has no matched 1/4/13-week origin"
        )
    common_origins = probability_by_origin.dropna(subset=list(HORIZONS)).index
    common_origins = common_origins.intersection(
        actual_by_origin.dropna(subset=list(HORIZONS)).index
    ).sort_values()
    if len(common_origins) < 1:
        raise ValueError(
            "transition selection evidence has no matched 1/4/13-week origin"
        )

    raw_losses: list[float] = []
    projected_losses: list[float] = []
    for origin_key in common_origins:
        raw_risk = {
            f"{horizon}w": {
                "probability": float(probability_by_origin.loc[origin_key, horizon])
            }
            for horizon in HORIZONS
        }
        projected_risk, metadata = _anchored_isotonic_transition_risk(
            raw_risk
        )
        realised = [
            int(bool(actual_by_origin.loc[origin_key, horizon]))
            for horizon in HORIZONS
        ]
        if realised != sorted(realised):
            raise ValueError(
                "transition cumulative events are not monotone at origin "
                f"{origin_key}"
            )
        for horizon, actual in zip(HORIZONS, realised, strict=True):
            raw_probability = float(metadata["raw_probabilities"][f"{horizon}w"])
            projected_probability = float(
                projected_risk[f"{horizon}w"]["probability"]
            )
            raw_losses.append((raw_probability - actual) ** 2)
            projected_losses.append((projected_probability - actual) ** 2)
    return {
        "evidence_track": "reconstructed_oos",
        "evidence_status": "historical_reconstructed_oos",
        "evaluation_split": "selection",
        "selection_end": cutoff.isoformat(),
        "source_artifact": "transition-oos-predictions.csv",
        "probability_source": "selected_horizon_champion_calibrated_p_change",
        "projection_fit": "parameter_free_fixed_l2_order_constraint",
        "matched_origin_count": len(common_origins),
        "matched_probability_count": len(raw_losses),
        "raw_brier": round(float(np.mean(raw_losses)), 8) if raw_losses else None,
        "projected_brier": (
            round(float(np.mean(projected_losses)), 8)
            if projected_losses
            else None
        ),
        "brier_difference_projected_minus_raw": (
            round(float(np.mean(projected_losses) - np.mean(raw_losses)), 8)
            if raw_losses
            else None
        ),
        "selection_effect": "semantic_coherence_only_no_model_selection",
    }


def _context_score_contract(
    scores: Mapping[str, Any],
    features: pd.DataFrame,
    *,
    at: object,
) -> tuple[dict[str, Any], dict[str, Any]]:
    timestamp = pd.Timestamp(at)
    comparable = timestamp
    if features.index.tz is None and comparable.tzinfo is not None:
        comparable = comparable.tz_localize(None)
    elif features.index.tz is not None and comparable.tzinfo is None:
        comparable = comparable.tz_localize(features.index.tz)
    elif features.index.tz is not None and comparable.tzinfo is not None:
        comparable = comparable.tz_convert(features.index.tz)
    row = features.loc[comparable] if comparable in features.index else pd.Series(dtype=float)
    output = dict(scores)
    coverage: dict[str, Any] = {}
    definitions = {
        "trend": (),
        "stress": (),
        "macro": MACRO_CONTEXT_COLUMNS,
        "financial_conditions": FINANCIAL_CONTEXT_COLUMNS,
    }
    for name, columns in definitions.items():
        if columns:
            available = sum(
                column in row and pd.notna(row[column]) for column in columns
            )
            expected = len(columns)
        else:
            available = int(pd.notna(output.get(name)))
            expected = 1
        minimum = max(1, math.ceil(expected * MINIMUM_CONTEXT_COVERAGE_FRACTION))
        sufficient = available >= minimum
        if not sufficient:
            output[name] = None
        coverage[name] = {
            "available_count": int(available),
            "expected_count": int(expected),
            "minimum_required_count": int(minimum),
            "status": "sufficient" if sufficient else "insufficient_coverage",
        }
    return output, coverage


def _aware_cutoff(value: object) -> pd.Timestamp:
    at = pd.Timestamp(value)
    if at.tzinfo is None:
        at = at.tz_localize("America/New_York")
    return at.tz_convert("UTC")


def _execution_parameters(
    profile_name: str,
    *,
    duration_bootstrap_resamples: int,
    outcome_bootstrap_resamples: int,
) -> dict[str, Any]:
    if profile_name == "quick":
        minimum_predictions = 3
        maximum_selection_origins: int | None = 3
        maximum_diagnostic_origins: int | None = 3
    elif profile_name == "standard":
        minimum_predictions = 12
        maximum_selection_origins = 60
        maximum_diagnostic_origins = 60
    elif profile_name == "full":
        minimum_predictions = 12
        maximum_selection_origins = None
        maximum_diagnostic_origins = None
    else:
        raise ValueError("profile_name must be quick, standard, or full")
    preregistered_resamples = 1_999
    overrides: list[str] = []
    if duration_bootstrap_resamples != preregistered_resamples:
        overrides.append("duration.bootstrap_resamples")
    if outcome_bootstrap_resamples != preregistered_resamples:
        overrides.append("conditional_asset_statistics.bootstrap_resamples")
    parameters: dict[str, Any] = {
        "profile": profile_name,
        "directional_minimum_selection_predictions": minimum_predictions,
        "directional_minimum_diagnostic_predictions": minimum_predictions,
        "directional_maximum_selection_origins": maximum_selection_origins,
        "directional_maximum_diagnostic_origins": maximum_diagnostic_origins,
        "duration_bootstrap_resamples": int(duration_bootstrap_resamples),
        "conditional_outcome_bootstrap_resamples": int(
            outcome_bootstrap_resamples
        ),
        "preregistered_bootstrap_resamples": preregistered_resamples,
        "preregistration_overrides": overrides,
    }
    encoded = json.dumps(
        parameters,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return {**parameters, "sha256": hashlib.sha256(encoded).hexdigest()}


def run_v5_directional_benchmark(
    features: pd.DataFrame,
    states: pd.Series,
    *,
    profile_name: str,
    selection_end: str | pd.Timestamp,
) -> DirectionalBenchmarkResult:
    if profile_name == "quick":
        minimum = 3
        selection_max = 3
        diagnostic_max = 3
    elif profile_name == "standard":
        minimum = 12
        selection_max = 60
        diagnostic_max = 60
    elif profile_name == "full":
        minimum = 12
        selection_max = None
        diagnostic_max = None
    else:
        raise ValueError("profile_name must be quick, standard, or full")
    return run_directional_transition_benchmark(
        features,
        states,
        horizons=HORIZONS,
        minimum_train_weeks=520,
        selection_end=selection_end,
        minimum_selection_predictions=minimum,
        minimum_diagnostic_predictions=minimum,
        selection_max_origins=selection_max,
        maximum_diagnostic_origins=diagnostic_max,
        random_state=17,
    )


def _directional_lookup(
    result: DirectionalBenchmarkResult,
) -> dict[tuple[str, int], Mapping[str, Any]]:
    lookup: dict[tuple[str, int], Mapping[str, Any]] = {}
    for source in (result.predictions, result.latest_forecasts):
        for row in source.to_dict(orient="records"):
            horizon = int(row["horizon_weeks"])
            if str(row["model"]) != str(result.champions_by_horizon[horizon]):
                continue
            key = (pd.Timestamp(row["origin_date"]).date().isoformat(), horizon)
            lookup[key] = row
    return lookup


def _fallback_directional(
    states: pd.Series,
    *,
    origin: pd.Timestamp,
    horizon: int,
    current_state: str,
) -> Mapping[str, Any]:
    comparable = origin
    if states.index.tz is None and comparable.tzinfo is not None:
        comparable = comparable.tz_localize(None)
    elif states.index.tz is not None and comparable.tzinfo is None:
        comparable = comparable.tz_localize(states.index.tz)
    elif states.index.tz is not None and comparable.tzinfo is not None:
        comparable = comparable.tz_convert(states.index.tz)
    history = states.loc[states.index <= comparable]
    probabilities = markov_first_passage_probabilities(
        history,
        current_state,
        horizon,
    )
    return {
        "model": "markov_first_passage",
        **{f"p_{name}": value for name, value in probabilities.items()},
    }


def _directional_rows(
    week: Mapping[str, Any],
    *,
    states: pd.Series,
    result: DirectionalBenchmarkResult,
    lookup: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    origin = pd.Timestamp(week["date"])
    current_state = str(week["current"]["state"])
    output: dict[str, Any] = {}
    for horizon in HORIZONS:
        key = f"{horizon}w"
        row = lookup.get((origin.date().isoformat(), horizon))
        if row is None:
            row = _fallback_directional(
                states,
                origin=origin,
                horizon=horizon,
                current_state=current_state,
            )
        p_change = float(week["transition_risk"][key]["probability"])
        probabilities = {
            state: float(row.get(f"p_{state}", 0.0)) for state in STATE_ORDER
        }
        reconciled = reconcile_directional_risk(
            p_change,
            current_state,
            probabilities,
        )
        output[key] = {
            "probability": round(p_change, 8),
            "no_departure": reconciled["no_departure"],
            "first_destination": reconciled["first_destination"],
            "target_end": str(week["transition_risk"][key]["target_end"]),
            "model": str(row["model"]),
            "method": str(reconciled["definition"]),
        }
    return output


def _current_membership(current: Mapping[str, Any]) -> dict[str, Any]:
    memberships = {
        state: round(float(current["probabilities"][state]), 8)
        for state in STATE_ORDER
    }
    state = str(current["state"])
    return {
        "state": state,
        "memberships": memberships,
        "primary_membership": memberships[state],
        "membership_entropy": round(float(current["entropy"]), 8),
        "method": "risk_score_anchor_membership",
    }


def _context_extremes(
    drivers: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for driver in drivers:
        z_score = float(driver.get("value", 0.0))
        if not math.isfinite(z_score):
            continue
        output.append(
            {
                "feature": str(driver["feature"]),
                "label": str(driver["label"]),
                "z_score": round(z_score, 4),
                "position": "high" if z_score >= 0.0 else "low",
                "method": "trailing_52w_z_score",
            }
        )
    return output


def _summary(week: Mapping[str, Any]) -> str:
    current = week["current"]
    forecast = week["next_week"]
    risk = week["transition_risk"]
    return (
        f"현재 {STATE_LABELS_KO[str(current['state'])]} 멤버십 "
        f"{float(current['primary_membership']):.0%}, 다음 주 "
        f"{STATE_LABELS_KO[str(forecast['state'])]} 전망 "
        f"{float(forecast['confidence']):.0%}. "
        f"국면 이탈 위험은 4주 {float(risk['4w']['probability']):.0%}, "
        f"13주 {float(risk['13w']['probability']):.0%}."
    )


def _model_health(model: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    probability_reasons: list[str] = []
    early_warning_reasons: list[str] = []
    holdout = model.get("holdout_diagnostic", {})
    if isinstance(holdout, Mapping) and holdout.get("status") == "weak_generalization":
        reasons.append("weak_generalization")
        probability_reasons.append("weak_generalization")

    champion = str(model.get("champion", ""))
    leaderboard = model.get("leaderboard", [])
    if isinstance(leaderboard, Sequence):
        champion_rows = [
            row
            for row in leaderboard
            if isinstance(row, Mapping) and str(row.get("name")) == champion
        ]
        if champion_rows:
            row = champion_rows[0]
            selection = row.get("selection_calibration_error")
            diagnostic = row.get("calibration_error")
            if selection is not None and diagnostic is not None:
                if float(diagnostic) - float(selection) > MATERIAL_CALIBRATION_DRIFT:
                    reasons.append("calibration_drift")
                    probability_reasons.append("calibration_drift")
    else:
        champion_rows = []

    champion_row = champion_rows[0] if champion_rows else {}
    calibration_error = champion_row.get("calibration_error")
    selection_calibration_error = champion_row.get("selection_calibration_error")
    calibration_drift = (
        float(calibration_error) - float(selection_calibration_error)
        if calibration_error is not None and selection_calibration_error is not None
        else None
    )
    probability_health = {
        "status": (
            "review_due"
            if probability_reasons
            else ("ok" if calibration_error is not None else "insufficient_evidence")
        ),
        "reasons": list(dict.fromkeys(probability_reasons)),
        "champion": champion,
        "evaluation_split": "retrospective_diagnostic",
        "calibration_method": "top_label_ece_10_equal_width_bins",
        "calibration_error": _json_value(calibration_error),
        "selection_calibration_error": _json_value(selection_calibration_error),
        "calibration_drift": _json_value(calibration_drift),
        "log_loss": _json_value(champion_row.get("log_loss")),
        "brier": _json_value(champion_row.get("brier")),
        "n_predictions": int(champion_row.get("n_predictions", 0)),
    }

    event_count = int(champion_row.get("transition_event_count", 0))
    recall = champion_row.get("transition_recall")
    precision = champion_row.get("transition_precision")
    if (
        event_count >= MINIMUM_TRANSITION_EVENTS_FOR_HEALTH
        and recall is not None
        and float(recall) < LOW_TRANSITION_RECALL
    ):
        reasons.append("low_transition_recall")
        early_warning_reasons.append("low_transition_recall")
    early_warning_health = {
        "status": (
            "review_due"
            if early_warning_reasons
            else (
                "ok"
                if event_count >= MINIMUM_TRANSITION_EVENTS_FOR_HEALTH
                else "insufficient_evidence"
            )
        ),
        "reasons": list(dict.fromkeys(early_warning_reasons)),
        "champion": champion,
        "evaluation_split": "retrospective_diagnostic",
        "event_definition": "actual_next_state_differs_from_current_state",
        "n_predictions": int(champion_row.get("n_predictions", 0)),
        "exposure_years": _json_value(champion_row.get("exposure_years")),
        "event_count": event_count,
        "on_time_departure_count": int(
            champion_row.get("on_time_departure_count", 0)
        ),
        "on_time_recall": _json_value(recall),
        "precision": _json_value(precision),
        "false_alarm_count": int(champion_row.get("false_alarm_count", 0)),
        "false_alarms_per_year": _json_value(
            champion_row.get("false_alarms_per_year")
        ),
        "detected_event_count": int(champion_row.get("detected_event_count", 0)),
        "mean_detection_delay_forecast_weeks": _json_value(
            champion_row.get("mean_detection_delay_forecast_weeks")
        ),
        "minimum_event_count": MINIMUM_TRANSITION_EVENTS_FOR_HEALTH,
    }
    return {
        "aggregate": {
            "status": "review_due" if reasons else "ok",
            "reasons": list(dict.fromkeys(reasons)),
        },
        "probability_health": probability_health,
        "early_warning_health": early_warning_health,
    }


def _freshness(meta: Mapping[str, Any]) -> dict[str, Any]:
    generated = _aware_cutoff(meta["generated_at"])
    data_as_of = _aware_cutoff(meta["data_as_of"])
    age_days = max(0, int((generated - data_as_of).total_seconds() // 86_400))
    maximum = 10
    return {
        "cadence": "weekly",
        "maximum_age_days": maximum,
        "age_days": age_days,
        "status": "current" if age_days <= maximum else "stale",
        "data_as_of": data_as_of.isoformat(),
    }


def _conditional_research(
    canonical: pd.DataFrame,
    states: pd.Series,
    weekly: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int,
) -> tuple[dict[str, Any], Any, pd.DatetimeIndex]:
    """Compare OOS forecasts with the actual state they targeted.

    Each weekly row is a forecast origin ``t`` for state ``t+1``.  The
    comparator is intentionally an oracle diagnostic: it assigns the realized
    ``t+1`` state back to the same origin used by the forecast.  This keeps the
    observed and predicted studies on identical, investable return windows.
    """

    canonical_index_by_date = {
        pd.Timestamp(index).date().isoformat(): index for index in canonical.index
    }
    state_by_date = {
        pd.Timestamp(index).date().isoformat(): str(value)
        for index, value in states.items()
    }
    matched_origins: list[pd.Timestamp] = []
    actual_next_states: list[str] = []
    for week in weekly:
        origin_date = str(week.get("date", ""))
        origin = canonical_index_by_date.get(origin_date)
        if origin is None:
            raise ValueError(
                f"conditional outcome origin is absent from canonical data: {origin_date}"
            )
        origin_position = canonical.index.get_loc(origin)
        if not isinstance(origin_position, (int, np.integer)):
            raise ValueError("conditional outcome origin must resolve uniquely")
        target_position = int(origin_position) + 1
        if target_position >= len(canonical.index):
            continue
        target = canonical.index[target_position]
        target_date = pd.Timestamp(target).date().isoformat()
        actual_state = state_by_date.get(target_date)
        if actual_state is None:
            continue
        if actual_state not in STATE_ORDER:
            raise ValueError(
                f"conditional outcome target state is invalid: {actual_state}"
            )
        matched_origins.append(pd.Timestamp(origin))
        actual_next_states.append(actual_state)

    if not matched_origins:
        raise ValueError("conditional outcomes require realized OOS forecast targets")
    matched_index = pd.DatetimeIndex(matched_origins)
    if matched_index.has_duplicates or not matched_index.is_monotonic_increasing:
        raise ValueError("conditional outcome origins must be unique and increasing")
    conditioning_states = pd.Series(
        actual_next_states,
        index=matched_index,
        dtype="object",
        name="state",
    )
    result = build_conditional_asset_statistics(
        canonical,
        conditioning_states,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=17,
    )
    return (
        {
            "conditional_asset_stats": {
                "method": (
                    "matched_oos_actual_next_state_target_week_adjusted_forward_return"
                ),
                "role": "matched_oracle_diagnostic",
                "conditioning": "actual_next_state_on_matched_oos_origins",
                "return_measure": "provider_adjusted_forward_return",
                "entry_week_distribution_policy": (
                    "conservative_excluded_without_ex_date"
                ),
                "corporate_action_policy": (
                    "same_row_adjustment_factor_split_consistent"
                ),
                "drawdown_observation_basis": (
                    "entry_adjusted_open_then_weekly_adjusted_closes"
                ),
                "state_horizon_weeks": 1,
                "execution_lag_weeks": 1,
                "entry_price_basis": "next_week_adjusted_open",
                "exit_price_basis": "horizon_week_adjusted_close",
                "rebalance_policy": "none_fixed_asset_hold",
                "origin_sampling": "weekly_rolling_overlapping",
                "horizons_weeks": list(HORIZONS),
                "assets": list(ASSETS),
                "return_currency": "USD",
                "rows": _records(result.statistics),
            }
        },
        result,
        matched_index,
    )


def _model_conditioned_research(
    canonical: pd.DataFrame,
    weekly: Sequence[Mapping[str, Any]],
    model_names: Sequence[str],
    *,
    bootstrap_resamples: int,
    matched_origins: pd.DatetimeIndex | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Condition forward outcomes on each model's completed OOS forecast.

    Forecasts made at origin ``t`` describe the state expected at ``t+1``.
    The outcome study enters at the adjusted open of ``t+1`` and includes that
    target week's open-to-close return.  When ``matched_origins`` is supplied,
    the forecast and actual-state oracle use exactly the same OOS origins.
    """

    resolved_models = tuple(str(name) for name in model_names)
    if not resolved_models or len(resolved_models) != len(set(resolved_models)):
        raise ValueError("model-conditioned outcomes require unique model names")

    index_by_date = {
        pd.Timestamp(index).date().isoformat(): index for index in canonical.index
    }
    allowed_dates = (
        {
            pd.Timestamp(index).date().isoformat()
            for index in pd.DatetimeIndex(matched_origins)
        }
        if matched_origins is not None
        else None
    )
    origin_index: list[pd.Timestamp] = []
    states_by_model: dict[str, list[str]] = {
        name: [] for name in resolved_models
    }
    for week in weekly:
        date_value = str(week.get("date", ""))
        if allowed_dates is not None and date_value not in allowed_dates:
            continue
        index = index_by_date.get(date_value)
        if index is None:
            raise ValueError(
                f"model-conditioned outcome origin is absent from canonical data: {date_value}"
            )
        raw_forecasts = week.get("model_forecasts")
        if not isinstance(raw_forecasts, Sequence) or isinstance(
            raw_forecasts, (str, bytes)
        ):
            raise ValueError(
                f"model-conditioned forecasts are missing for {date_value}"
            )
        by_name = {
            str(row.get("model")): row
            for row in raw_forecasts
            if isinstance(row, Mapping)
        }
        if set(by_name) != set(resolved_models):
            raise ValueError(
                f"model-conditioned forecast suite is incomplete for {date_value}"
            )
        origin_index.append(index)
        for name in resolved_models:
            forecast_state = str(by_name[name].get("state", ""))
            if forecast_state not in STATE_ORDER:
                raise ValueError(
                    f"model-conditioned forecast state is invalid for {name}: {forecast_state}"
                )
            states_by_model[name].append(forecast_state)

    if not origin_index:
        raise ValueError("model-conditioned outcomes require OOS forecasts")
    if pd.DatetimeIndex(origin_index).has_duplicates:
        raise ValueError("model-conditioned outcome origins must be unique")

    prices_index = pd.DatetimeIndex(origin_index)
    if allowed_dates is not None and set(
        pd.Timestamp(index).date().isoformat() for index in prices_index
    ) != allowed_dates:
        raise ValueError("model-conditioned origins differ from matched OOS origins")
    statistics_frames: list[pd.DataFrame] = []
    outcome_frames: list[pd.DataFrame] = []
    for name in resolved_models:
        predicted_states = pd.Series(
            states_by_model[name],
            index=prices_index,
            dtype="object",
            name="state",
        )
        result = build_conditional_asset_statistics(
            canonical,
            predicted_states,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=17,
        )
        statistics = result.statistics.copy()
        statistics.insert(0, "conditioning_model", name)
        outcomes = result.outcomes.copy()
        outcomes.insert(0, "conditioning_model", name)
        statistics_frames.append(statistics)
        outcome_frames.append(outcomes)

    combined_statistics = pd.concat(statistics_frames, ignore_index=True)
    combined_outcomes = pd.concat(outcome_frames, ignore_index=True)
    return (
        {
            "model_conditioned_asset_stats": {
                "method": (
                    "matched_oos_predicted_next_state_target_week_adjusted_forward_return"
                ),
                "role": "retrospective_model_diagnostic",
                "conditioning": "hard_argmax_oos_forecast",
                "return_measure": "provider_adjusted_forward_return",
                "entry_week_distribution_policy": (
                    "conservative_excluded_without_ex_date"
                ),
                "corporate_action_policy": (
                    "same_row_adjustment_factor_split_consistent"
                ),
                "drawdown_observation_basis": (
                    "entry_adjusted_open_then_weekly_adjusted_closes"
                ),
                "forecast_horizon_weeks": 1,
                "execution_lag_weeks": 1,
                "entry_price_basis": "next_week_adjusted_open",
                "exit_price_basis": "horizon_week_adjusted_close",
                "rebalance_policy": "none_fixed_asset_hold",
                "origin_sampling": "weekly_rolling_overlapping",
                "horizons_weeks": list(HORIZONS),
                "assets": list(ASSETS),
                "models": list(resolved_models),
                "return_currency": "USD",
                "rows": _records(combined_statistics),
            }
        },
        combined_outcomes,
        combined_statistics,
    )


def build_v5_payload(
    payload_v4: Mapping[str, Any],
    *,
    canonical: pd.DataFrame,
    features: pd.DataFrame,
    states: pd.Series,
    directional: DirectionalBenchmarkResult,
    transition_selection_predictions: pd.DataFrame,
    transition_champions_by_horizon: Mapping[int, str],
    transition_selection_end: str | pd.Timestamp,
    baseline_v4: Mapping[str, Any],
    structural_preregistration_sha256: str,
    label_fit_start: str | pd.Timestamp | None = None,
    label_fit_end: str | pd.Timestamp | None = None,
    label_fit_weeks: int = 520,
    evidence_track: str = "reconstructed_oos",
    fx_result: FXFeatureResult | None = None,
    latest_fx_context: Mapping[str, Any] | None = None,
    h10_source: Mapping[str, Any] | None = None,
    duration_bootstrap_resamples: int = 1_999,
    outcome_bootstrap_resamples: int = 1_999,
    fx_ablation_evidence_sink: Callable[[pd.DataFrame], Any] | None = None,
) -> tuple[dict[str, Any], Any]:
    """Compose the semantic v5 contract without changing v4 model selection."""

    payload = deepcopy(dict(payload_v4))
    original_model = deepcopy(dict(payload["model"]))
    execution_parameters = _execution_parameters(
        str(original_model["profile"]),
        duration_bootstrap_resamples=duration_bootstrap_resamples,
        outcome_bootstrap_resamples=outcome_bootstrap_resamples,
    )
    health = _model_health(original_model)
    meta = deepcopy(dict(payload["meta"]))
    generated_at = datetime.now(timezone.utc)
    meta["generated_at"] = generated_at.isoformat()
    meta["generation_id"] = generated_at.strftime("%Y%m%dT%H%M%S.%fZ")
    meta.update(
        {
            "schema_version": V5_SCHEMA_VERSION,
            "result_version": V5_RESULT_VERSION,
            "transition_probability_definition": (
                "P(first departure from the origin regime within one week)"
            ),
            "transition_risk_definition": (
                "P(first departure from the origin regime within h weeks)"
            ),
            "current_membership_definition": (
                "distance-to-threshold-anchor observational membership; not posterior"
            ),
        }
    )
    meta["freshness"] = _freshness(meta)
    warnings = [
        warning
        for warning in meta.get("warnings", [])
        if "Top drivers" not in str(warning)
    ]
    meta["warnings"] = warnings
    if health["aggregate"]["status"] == "review_due":
        meta["status"] = "degraded"

    if fx_ablation_evidence_sink is not None and not callable(
        fx_ablation_evidence_sink
    ):
        raise TypeError("fx_ablation_evidence_sink must be callable or None")
    fx_ablation_evidence: list[pd.DataFrame] = []
    fx_ablation = run_fx_shadow_ablation(
        features,
        states,
        fx_result,
        pd.DatetimeIndex(canonical.index),
        evidence_sink=fx_ablation_evidence.append,
    )
    fx_ablation_oos = (
        fx_ablation_evidence[0]
        if fx_ablation_evidence
        else pd.DataFrame(columns=FX_ABLATION_OOS_COLUMNS)
    )

    model = original_model
    for obsolete in ("baseline_v2", "baseline_v3"):
        model.pop(obsolete, None)
    model.update(
        {
            "version": V5_MODEL_VERSION,
            "feature_set_version": V5_FEATURE_SET_VERSION,
            "baseline_v4": dict(baseline_v4),
            "structural_preregistration": {
                "path": "config/structural_v5.json",
                "sha256": str(structural_preregistration_sha256),
            },
            "execution_parameters": execution_parameters,
            "directional_transition": {
                "target": "first_departure_state_within_h_or_no_departure",
                "deployed_direction_role": "first_destination_given_departure",
                "selection_metric": "conditional_destination_log_loss",
                "minimum_selection_departure_events": MINIMUM_SELECTION_EVENTS,
                "minimum_selection_destination_classes": MINIMUM_DESTINATION_CLASSES,
                "minimum_selection_event_blocks": MINIMUM_EVENT_BLOCKS,
                "champions": {
                    f"{horizon}w": str(directional.champions_by_horizon[horizon])
                    for horizon in HORIZONS
                },
                "leaderboard": _records(directional.leaderboard),
                "selection_diagnostics": _records(
                    directional.selection_diagnostics
                ),
                "selection_end": directional.selection_end.date().isoformat(),
            },
            "model_health": health["aggregate"],
            "probability_health": health["probability_health"],
            "early_warning_health": health["early_warning_health"],
            "fx_role": "context_and_preregistered_shadow_ablation",
            "fx_ablation": fx_ablation,
            "selection_status": "selected_by_gate",
        }
    )
    structural_models = deepcopy(dict(model.get("structural_models", {})))
    structural_models["causal_multiscale_ensemble"] = {
        "role": "v5_opt_in_candidate",
        "experts": ["markov", "xgboost", "xgb_hazard_destination"],
        "scale_half_lives_weeks": [26, 52, 104],
        "outer_scale_weights": [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        "aggregation": "fixed_equal_probability_average",
        "inner_pool_method": "causal_discounted_completed_oos_log_score",
        "minimum_history_rows": 26,
        "eligible_loss_rule": "target_date_strictly_before_origin",
        "selection_gate": "existing_multiclass_holm_log_loss_brier_zero_fallback",
        "automatic_promotion_bypass": False,
    }
    model["structural_models"] = structural_models
    model["transition_term_structure_evidence"] = (
        _term_structure_selection_evidence(
            transition_selection_predictions,
            transition_champions_by_horizon,
            selection_end=transition_selection_end,
        )
    )

    lookup = _directional_lookup(directional)
    latest_date = str(payload["weekly"][-1]["date"])
    weekly: list[dict[str, Any]] = []
    for source_week in payload["weekly"]:
        week = deepcopy(dict(source_week))
        week["current"] = _current_membership(source_week["current"])
        reconciled_risk, term_structure = _anchored_isotonic_transition_risk(
            week["transition_risk"]
        )
        week["transition_risk"] = reconciled_risk
        week["transition_term_structure"] = term_structure
        week["context_scores"], week["context_score_coverage"] = (
            _context_score_contract(
                week.pop("scores"),
                features,
                at=week.get("data_as_of", week["date"]),
            )
        )
        week["extreme_context"] = _context_extremes(week.pop("top_drivers"))
        week["directional_risk"] = _directional_rows(
            week,
            states=states,
            result=directional,
            lookup=lookup,
        )
        week["duration_context"] = duration_context(
            states,
            as_of=source_week.get("data_as_of", week["date"]),
            bootstrap_resamples=(
                duration_bootstrap_resamples
                if str(week["date"]) == latest_date
                else 0
            ),
            bootstrap_seed=17,
        )
        cutoff = _aware_cutoff(week.get("data_as_of", week["date"]))
        week["fx_context"] = (
            fx_context_at(fx_result, cutoff=cutoff)
            if fx_result is not None
            else (
                deepcopy(dict(latest_fx_context))
                if str(week["date"]) == latest_date
                and latest_fx_context is not None
                else unavailable_fx_context()
            )
        )
        week["summary"] = _summary(week)
        weekly.append(week)

    selection = _selection_contract(model)
    forecast_contract = _forecast_contract(
        weekly[-1],
        generated_at=generated_at,
        mode=str(meta["mode"]),
        evidence_track=evidence_track,
    )
    shadow_decision_at = forecast_contract.get("decision_at")
    if shadow_decision_at is None:
        # Non-live expired fixtures suppress decision_at in the forecast envelope,
        # but the shadow still records the same generation decision for no-trade.
        shadow_decision_at = pd.Timestamp(generated_at).tz_convert("UTC").isoformat()

    research, conditional_result, matched_outcome_origins = _conditional_research(
        canonical,
        states,
        weekly,
        bootstrap_resamples=outcome_bootstrap_resamples,
    )
    research["prospective_decision_shadow"] = build_decision_shadow(
        weekly,
        canonical,
        forecast_model=str(selection["operating_champion"]),
        decision_at=shadow_decision_at,
    )
    forecast_comparison = model.get("forecast_comparison", {})
    comparison_models = (
        tuple(forecast_comparison.get("models", ()))
        if isinstance(forecast_comparison, Mapping)
        else ()
    )
    if comparison_models:
        (
            model_conditioned_research,
            model_conditioned_outcomes,
            model_conditioned_statistics,
        ) = _model_conditioned_research(
            canonical,
            weekly,
            comparison_models,
            bootstrap_resamples=outcome_bootstrap_resamples,
            matched_origins=matched_outcome_origins,
        )
        research.update(model_conditioned_research)
    research_frames = {
        "directional_oos_predictions": directional.predictions,
        "directional_model_leaderboard": directional.leaderboard,
        "directional_walk_forward_splits": directional.split_audit,
        "directional_selection_diagnostics": directional.selection_diagnostics,
        "directional_forecasts": directional.latest_forecasts,
        "conditional_asset_outcomes": conditional_result.outcomes,
        "conditional_asset_statistics": conditional_result.statistics,
    }
    if comparison_models:
        research_frames.update(
            {
                "model_conditioned_asset_outcomes": model_conditioned_outcomes,
                "model_conditioned_asset_statistics": model_conditioned_statistics,
            }
        )
    if fx_result is not None:
        research_frames.update(
            {
                "fx_features": fx_result.features,
                "fx_coverage": fx_result.coverage,
                "fx_ablation_oos": fx_ablation_oos,
            }
        )
    model["research_artifacts"] = build_v5_research_artifact_manifest(
        research_frames
    )
    sources = [deepcopy(dict(row)) for row in payload.get("sources", [])]
    if h10_source is not None:
        sources = [row for row in sources if row.get("id") != "frb_h10"]
        sources.append(deepcopy(dict(h10_source)))
    feature_catalog = [
        deepcopy(dict(row)) for row in payload.get("feature_catalog", [])
    ]
    feature_catalog.append(
        {
            "id": "fx_usd_structure",
            "label": "Fed H.10 달러 강도·통화 횡단면",
            "category": "FX context",
            "frequency": "weekly point-in-time",
            "source": "derived from Federal Reserve H.10",
        }
    )

    label_spec = load_label_spec()
    label_grid, label_grid_sha256 = _config_document(
        "label-sensitivity-grid.json"
    )
    research["label_sensitivity"] = {
        "schema_version": "regime-label-sensitivity-summary/1",
        "status": "preregistered_pending_execution",
        "evidence_track": "reconstructed_oos",
        "evaluation_split": "selection_only",
        "control": {
            "spec_id": label_spec.spec_id,
            "spec_version": label_spec.version,
            "spec_sha256": label_spec.spec_sha256,
            "remains_operating_control": True,
        },
        "grid": {
            "path": "config/label-sensitivity-grid.json",
            "sha256": label_grid_sha256,
            "dimensions": dict(label_grid["grid"]),
        },
        "execution_summary": {
            "evaluated_spec_count": 0,
            "state_occupancy": None,
            "episode_count": None,
            "weekly_flip_rate": None,
            "transition_jaccard": None,
            "forward_return_separation": None,
            "model_rank_robustness": None,
        },
        "automatic_promotion_eligible": False,
    }
    if label_fit_weeks < 1 or label_fit_weeks > len(canonical):
        raise ValueError("label_fit_weeks is outside the canonical history")
    resolved_fit_start = pd.Timestamp(
        canonical.index[0] if label_fit_start is None else label_fit_start
    )
    resolved_fit_end = pd.Timestamp(
        canonical.index[label_fit_weeks - 1]
        if label_fit_end is None
        else label_fit_end
    )
    if resolved_fit_end < resolved_fit_start:
        raise ValueError("label fit period is reversed")
    lifecycle = {
        "selection": {"status": "selected_by_gate"},
        "deployment": {"status": "candidate"},
        "publication": {"status": "unpublished"},
    }
    model["lifecycle"] = lifecycle
    meta["publication_status"] = "unpublished"
    result_payload = {
        "meta": meta,
        "states": [dict(row) for row in _OPERATING_CONTRACT.state_definitions],
        "label": {
            "spec_id": label_spec.spec_id,
            "spec_version": label_spec.version,
            "spec_sha256": label_spec.spec_sha256,
            "fit_period": {
                "start": resolved_fit_start.date().isoformat(),
                "end": resolved_fit_end.date().isoformat(),
                "weeks": int(label_fit_weeks),
            },
            "input_scope": "SPY adjusted close only",
            "membership_semantics": label_spec.membership.semantics,
        },
        "forecast": forecast_contract,
        "selection": selection,
        "model": model,
        "weekly": weekly,
        "sources": sources,
        "feature_catalog": feature_catalog,
        "research": research,
    }
    if comparison_models:
        object.__setattr__(
            conditional_result,
            "model_conditioned_outcomes",
            model_conditioned_outcomes,
        )
        object.__setattr__(
            conditional_result,
            "model_conditioned_statistics",
            model_conditioned_statistics,
        )
    if fx_result is not None and fx_ablation_evidence_sink is not None:
        fx_ablation_evidence_sink(fx_ablation_oos.copy())
    return result_payload, conditional_result


__all__ = [
    "V5_FEATURE_SET_VERSION",
    "V5_MODEL_VERSION",
    "V5_RESULT_VERSION",
    "V5_SCHEMA_VERSION",
    "build_v5_payload",
    "run_v5_directional_benchmark",
]
