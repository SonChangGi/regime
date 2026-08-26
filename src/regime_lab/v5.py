"""Opt-in v5 payload composition on top of the frozen v4 analysis path."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
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
_OPERATING_CONTRACT = load_operating_contract()


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
    return {
        "status": "active" if active else "expired",
        "origin_at": origin.isoformat(),
        "decision_at": decision.isoformat() if active else None,
        "target_at": target.isoformat(),
        "remaining_horizon": (
            int((target - decision).total_seconds()) if active else 0
        ),
        "evidence_track": str(evidence_track),
    }


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
    holdout = model.get("holdout_diagnostic", {})
    if isinstance(holdout, Mapping) and holdout.get("status") == "weak_generalization":
        reasons.append("weak_generalization")

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

    transition_rows = model.get("transition_leaderboard", [])
    if isinstance(transition_rows, Sequence):
        for row in transition_rows:
            if not isinstance(row, Mapping):
                continue
            if (
                int(row.get("horizon_weeks", 0)) == 1
                and bool(row.get("selected", False))
                and row.get("evaluation_split") == "retrospective_diagnostic"
                and int(row.get("event_count", 0)) >= MINIMUM_TRANSITION_EVENTS_FOR_HEALTH
                and float(row.get("recall", 1.0)) < LOW_TRANSITION_RECALL
            ):
                reasons.append("low_transition_recall")
                break
    return {
        "status": "review_due" if reasons else "ok",
        "reasons": list(dict.fromkeys(reasons)),
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
    *,
    bootstrap_resamples: int,
) -> tuple[dict[str, Any], Any]:
    result = build_conditional_asset_statistics(
        canonical,
        states,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=17,
    )
    return (
        {
            "conditional_asset_stats": {
                "method": "state_conditioned_forward_total_return",
                "role": "descriptive_only",
                "execution_lag_weeks": 1,
                "horizons_weeks": list(HORIZONS),
                "assets": list(ASSETS),
                "return_currency": "USD",
                "rows": _records(result.statistics),
            }
        },
        result,
    )


def _model_conditioned_research(
    canonical: pd.DataFrame,
    weekly: Sequence[Mapping[str, Any]],
    model_names: Sequence[str],
    *,
    bootstrap_resamples: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Condition forward outcomes on each model's completed OOS forecast.

    Forecasts made at origin ``t`` describe the state expected at ``t+1``.
    The outcome study therefore enters at the completed ``t+1`` close, which
    is the same one-week execution lag used by the observed-state study.
    """

    resolved_models = tuple(str(name) for name in model_names)
    if not resolved_models or len(resolved_models) != len(set(resolved_models)):
        raise ValueError("model-conditioned outcomes require unique model names")

    index_by_date = {
        pd.Timestamp(index).date().isoformat(): index for index in canonical.index
    }
    origin_index: list[pd.Timestamp] = []
    states_by_model: dict[str, list[str]] = {
        name: [] for name in resolved_models
    }
    for week in weekly:
        date_value = str(week.get("date", ""))
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

    prices = canonical.loc[origin_index]
    statistics_frames: list[pd.DataFrame] = []
    outcome_frames: list[pd.DataFrame] = []
    for name in resolved_models:
        predicted_states = pd.Series(
            states_by_model[name],
            index=prices.index,
            dtype="object",
            name="state",
        )
        result = build_conditional_asset_statistics(
            prices,
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
                "method": "oos_one_week_forecast_conditioned_forward_total_return",
                "role": "retrospective_model_diagnostic",
                "conditioning": "hard_argmax_oos_forecast",
                "forecast_horizon_weeks": 1,
                "execution_lag_weeks": 1,
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
    if health["status"] == "review_due":
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
            "model_health": health,
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

    lookup = _directional_lookup(directional)
    latest_date = str(payload["weekly"][-1]["date"])
    weekly: list[dict[str, Any]] = []
    for source_week in payload["weekly"]:
        week = deepcopy(dict(source_week))
        week["current"] = _current_membership(source_week["current"])
        week["context_scores"] = week.pop("scores")
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

    research, conditional_result = _conditional_research(
        canonical,
        states,
        bootstrap_resamples=outcome_bootstrap_resamples,
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
    selection = _selection_contract(model)
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
        "forecast": _forecast_contract(
            weekly[-1],
            generated_at=generated_at,
            mode=str(meta["mode"]),
            evidence_track=evidence_track,
        ),
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
