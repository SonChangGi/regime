"""Contract shared by the analytical pipeline and the static dashboard."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
import math
from typing import Any

SCHEMA_VERSION = "1.0.0"
V3_RESULT_VERSION = "weekly-regime-result-v3"
V3_MODEL_VERSION = "weekly-nondl-structural-v3"
V3_LABEL_VERSION = "market-causal-3state-v1"
V3_FEATURE_SET_VERSION = "weekly-pit-market-internals-v3"
V4_RESULT_VERSION = "weekly-regime-result-v4"
V4_MODEL_VERSION = "weekly-nondl-structural-v4"
V4_LABEL_VERSION = "market-causal-3state-v1"
V4_FEATURE_SET_VERSION = "weekly-pit-structural-v4"
FROZEN_V4_BASELINE_V3 = {
    "result_version": V3_RESULT_VERSION,
    "label_version": V3_LABEL_VERSION,
    "model_version": V3_MODEL_VERSION,
    "champion": "markov",
    "payload_sha256": "de93c585117b2784750f586a4f84ad99964c63081b252ad7affd7a75bd797095",
    "artifacts_inventory_sha256": "8ef3778cc8c36faff0c80e2bf094f1f11bd6966ab3b7b2d6edb84ba292aff6b9",
    "captured_at": "2026-08-13",
}
FROZEN_V4_STRUCTURAL_PREREGISTRATION = {
    "path": "config/structural_v4.json",
    "sha256": "2f53ada564efca770261f16ce6eb16ec9c9782bde014de7a7d85b7b24dbe407b",
}
TRANSITION_HORIZONS = (1, 4, 13)
STATE_ORDER = ("risk_on", "transition", "risk_off")
HEALTH_STATUSES = {
    "ok",
    "stale",
    "degraded",
    "quota_exhausted",
    "schema_changed",
    "revision_gap",
    "rights_unconfirmed",
    "license_blocked",
    "unavailable",
}


class ContractError(ValueError):
    """Raised when a result payload cannot be safely shown by the dashboard."""


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ContractError(f"{context}.{key} is required")
    return mapping[key]


def _validate_probabilities(value: Any, context: str) -> None:
    if not isinstance(value, Mapping):
        raise ContractError(f"{context} must be an object")
    actual_keys = set(value)
    expected_keys = set(STATE_ORDER)
    if actual_keys != expected_keys:
        raise ContractError(f"{context} state keys must be exactly {STATE_ORDER}")
    probs = [float(value[state]) for state in STATE_ORDER]
    if not all(math.isfinite(prob) for prob in probs):
        raise ContractError(f"{context} probabilities must be finite")
    if any(prob < -1e-9 or prob > 1.0 + 1e-9 for prob in probs):
        raise ContractError(f"{context} probabilities must be in [0, 1]")
    if abs(sum(probs) - 1.0) > 1e-5:
        raise ContractError(f"{context} probabilities must sum to one")


def _finite_number(
    value: Any,
    context: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise ContractError(f"{context} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{context} must be a finite number") from exc
    if not math.isfinite(number):
        raise ContractError(f"{context} must be finite")
    if minimum is not None and number < minimum:
        raise ContractError(f"{context} must be at least {minimum}")
    if maximum is not None and number > maximum:
        raise ContractError(f"{context} must be at most {maximum}")
    return number


def _nonnegative_integer(value: Any, context: str) -> int:
    number = _finite_number(value, context, minimum=0.0)
    if not number.is_integer():
        raise ContractError(f"{context} must be an integer")
    return int(number)


def _validate_transition_model(
    model: Mapping[str, Any],
    *,
    expected_model_version: str,
    expected_label_version: str,
    expected_feature_set_version: str,
) -> None:
    context = "payload.model"
    if _require(model, "version", context) != expected_model_version:
        raise ContractError(f"{context}.version must be {expected_model_version}")
    if _require(model, "label_version", context) != expected_label_version:
        raise ContractError(f"{context}.label_version must be {expected_label_version}")
    if _require(model, "feature_set_version", context) != expected_feature_set_version:
        raise ContractError(
            f"{context}.feature_set_version must be {expected_feature_set_version}"
        )
    if _require(model, "primary_horizon_weeks", context) != 1:
        raise ContractError(f"{context}.primary_horizon_weeks must be one")
    transition_selection_end = _require(model, "transition_selection_end", context)
    try:
        date.fromisoformat(str(transition_selection_end))
    except ValueError as exc:
        raise ContractError(
            f"{context}.transition_selection_end must be an ISO date"
        ) from exc
    horizons = _require(model, "transition_horizons_weeks", context)
    if (
        not isinstance(horizons, Sequence)
        or isinstance(horizons, (str, bytes))
        or tuple(horizons) != TRANSITION_HORIZONS
    ):
        raise ContractError(
            f"{context}.transition_horizons_weeks must be {TRANSITION_HORIZONS}"
        )
    baseline = _require(model, "baseline_v2", context)
    if not isinstance(baseline, Mapping):
        raise ContractError(f"{context}.baseline_v2 must be an object")
    if _require(baseline, "result_version", f"{context}.baseline_v2") != "weekly-regime-result-v2":
        raise ContractError(f"{context}.baseline_v2 result version is invalid")
    for name in ("label_version", "model_version", "champion"):
        value = _require(baseline, name, f"{context}.baseline_v2")
        if not isinstance(value, str) or not value:
            raise ContractError(f"{context}.baseline_v2.{name} must be non-empty")
    for name in ("payload_sha256", "artifacts_inventory_sha256"):
        value = str(_require(baseline, name, f"{context}.baseline_v2"))
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ContractError(f"{context}.baseline_v2.{name} must be SHA-256")

    champions = _require(model, "transition_champions", context)
    if not isinstance(champions, Mapping) or set(champions) != {"1w", "4w", "13w"}:
        raise ContractError(f"{context}.transition_champions keys are invalid")
    if any(not isinstance(value, str) or not value for value in champions.values()):
        raise ContractError(f"{context}.transition_champions values are invalid")
    leaderboard = _require(model, "transition_leaderboard", context)
    if not isinstance(leaderboard, Sequence) or isinstance(leaderboard, (str, bytes)) or not leaderboard:
        raise ContractError(f"{context}.transition_leaderboard must be non-empty")
    covered_horizons: set[int] = set()
    for index, row in enumerate(leaderboard):
        row_context = f"{context}.transition_leaderboard[{index}]"
        if not isinstance(row, Mapping):
            raise ContractError(f"{row_context} must be an object")
        horizon = _nonnegative_integer(_require(row, "horizon_weeks", row_context), f"{row_context}.horizon_weeks")
        if horizon not in TRANSITION_HORIZONS:
            raise ContractError(f"{row_context}.horizon_weeks is invalid")
        covered_horizons.add(horizon)
        model_name = _require(row, "model", row_context)
        if not isinstance(model_name, str) or not model_name:
            raise ContractError(f"{row_context}.model must be non-empty")
        if not isinstance(_require(row, "selected", row_context), bool):
            raise ContractError(f"{row_context}.selected must be boolean")
        if _require(row, "evaluation_split", row_context) not in {
            "selection",
            "retrospective_diagnostic",
        }:
            raise ContractError(f"{row_context}.evaluation_split is invalid")
        _finite_number(_require(row, "binary_log_loss", row_context), f"{row_context}.binary_log_loss", minimum=0.0)
        for metric in ("brier", "precision", "recall"):
            _finite_number(_require(row, metric, row_context), f"{row_context}.{metric}", minimum=0.0, maximum=1.0)
        event_count = _nonnegative_integer(_require(row, "event_count", row_context), f"{row_context}.event_count")
        non_event_count = _nonnegative_integer(_require(row, "non_event_count", row_context), f"{row_context}.non_event_count")
        prediction_count = _nonnegative_integer(_require(row, "n_predictions", row_context), f"{row_context}.n_predictions")
        if prediction_count != event_count + non_event_count:
            raise ContractError(f"{row_context} event counts do not sum to n_predictions")
        average_precision = _require(row, "average_precision", row_context)
        if average_precision is None:
            if event_count != 0:
                raise ContractError(f"{row_context}.average_precision may be null only without events")
        else:
            _finite_number(average_precision, f"{row_context}.average_precision", minimum=0.0, maximum=1.0)
        _finite_number(_require(row, "false_alarms_per_year", row_context), f"{row_context}.false_alarms_per_year", minimum=0.0)
        for metric in ("fallback_count", "calibration_fallback_count"):
            _nonnegative_integer(_require(row, metric, row_context), f"{row_context}.{metric}")
    if covered_horizons != set(TRANSITION_HORIZONS):
        raise ContractError(f"{context}.transition_leaderboard lacks a horizon")

    shadow = _require(model, "shadow_nowcast", context)
    if not isinstance(shadow, Mapping) or shadow.get("status") != "shadow_only":
        raise ContractError(f"{context}.shadow_nowcast must be shadow_only")
    if shadow.get("canonical_target") is not False:
        raise ContractError(f"{context}.shadow_nowcast cannot be canonical")


def _validate_v3_model(model: Mapping[str, Any]) -> None:
    _validate_transition_model(
        model,
        expected_model_version=V3_MODEL_VERSION,
        expected_label_version=V3_LABEL_VERSION,
        expected_feature_set_version=V3_FEATURE_SET_VERSION,
    )


def _validate_exact_object(
    value: Any,
    context: str,
    expected_keys: set[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ContractError(f"{context} fields must be exactly {tuple(sorted(expected_keys))}")
    return value


def _validate_sha256(value: Any, context: str) -> None:
    candidate = str(value)
    if len(candidate) != 64 or any(
        character not in "0123456789abcdef" for character in candidate
    ):
        raise ContractError(f"{context} must be a lowercase SHA-256")


def _validate_v4_evidence_artifacts(model: Mapping[str, Any]) -> None:
    context = "payload.model.evidence_artifacts"
    artifacts = _validate_exact_object(
        _require(model, "evidence_artifacts", "payload.model"),
        context,
        {"state_label_history", "weekly_state_forecasts"},
    )
    labels_context = f"{context}.state_label_history"
    labels = _validate_exact_object(
        artifacts["state_label_history"],
        labels_context,
        {
            "path",
            "row_count",
            "sha256",
            "label_fit_weeks",
            "label_fit_end",
            "initial_state",
        },
    )
    if labels["path"] != "state-label-history.csv":
        raise ContractError(f"{labels_context}.path is invalid")
    if _nonnegative_integer(labels["row_count"], f"{labels_context}.row_count") < 520:
        raise ContractError(f"{labels_context}.row_count must cover label fitting")
    _validate_sha256(labels["sha256"], f"{labels_context}.sha256")
    if labels["label_fit_weeks"] != 520:
        raise ContractError(f"{labels_context}.label_fit_weeks must be 520")
    try:
        datetime.fromisoformat(str(labels["label_fit_end"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError(f"{labels_context}.label_fit_end must be ISO-8601") from exc
    if labels["initial_state"] != "transition":
        raise ContractError(f"{labels_context}.initial_state must be transition")

    forecasts_context = f"{context}.weekly_state_forecasts"
    forecasts = _validate_exact_object(
        artifacts["weekly_state_forecasts"],
        forecasts_context,
        {"path", "row_count", "sha256"},
    )
    if forecasts["path"] != "weekly-state-forecasts.csv":
        raise ContractError(f"{forecasts_context}.path is invalid")
    if _nonnegative_integer(
        forecasts["row_count"], f"{forecasts_context}.row_count"
    ) < 1:
        raise ContractError(f"{forecasts_context}.row_count must be positive")
    _validate_sha256(forecasts["sha256"], f"{forecasts_context}.sha256")


def _validate_v4_model(model: Mapping[str, Any]) -> None:
    context = "payload.model"
    _validate_transition_model(
        model,
        expected_model_version=V4_MODEL_VERSION,
        expected_label_version=V4_LABEL_VERSION,
        expected_feature_set_version=V4_FEATURE_SET_VERSION,
    )

    baseline_context = f"{context}.baseline_v3"
    baseline = _validate_exact_object(
        _require(model, "baseline_v3", context),
        baseline_context,
        {
            "result_version",
            "label_version",
            "model_version",
            "champion",
            "payload_sha256",
            "artifacts_inventory_sha256",
            "captured_at",
        },
    )
    for field, expected in FROZEN_V4_BASELINE_V3.items():
        if _require(baseline, field, baseline_context) != expected:
            raise ContractError(f"{baseline_context}.{field} must be {expected}")
    for field in ("payload_sha256", "artifacts_inventory_sha256"):
        _validate_sha256(_require(baseline, field, baseline_context), f"{baseline_context}.{field}")

    prereg_context = f"{context}.structural_preregistration"
    preregistration = _validate_exact_object(
        _require(model, "structural_preregistration", context),
        prereg_context,
        {"path", "sha256"},
    )
    for field, expected in FROZEN_V4_STRUCTURAL_PREREGISTRATION.items():
        if _require(preregistration, field, prereg_context) != expected:
            raise ContractError(f"{prereg_context}.{field} must be {expected}")
    _validate_sha256(
        _require(preregistration, "sha256", prereg_context),
        f"{prereg_context}.sha256",
    )
    _validate_sha256(
        _require(model, "feature_manifest_sha256", context),
        f"{context}.feature_manifest_sha256",
    )
    _validate_v4_evidence_artifacts(model)

    models_context = f"{context}.structural_models"
    structural_models = _validate_exact_object(
        _require(model, "structural_models", context),
        models_context,
        {
            "xgb_hazard_destination",
            "causal_dynamic_ensemble",
            "joint_survival_hazard",
        },
    )
    hazard_context = f"{models_context}.xgb_hazard_destination"
    hazard = _validate_exact_object(
        structural_models["xgb_hazard_destination"],
        hazard_context,
        {"hazard_model", "destination_model", "direct_jump_floor"},
    )
    if hazard != {
        "hazard_model": "binary_xgboost",
        "destination_model": "xgboost",
        "direct_jump_floor": 0.000001,
    }:
        raise ContractError(f"{hazard_context} values are invalid")

    ensemble_context = f"{models_context}.causal_dynamic_ensemble"
    ensemble = _validate_exact_object(
        structural_models["causal_dynamic_ensemble"],
        ensemble_context,
        {
            "experts",
            "half_life_weeks",
            "minimum_history_rows",
            "eligible_loss_rule",
        },
    )
    if (
        not isinstance(ensemble["experts"], Sequence)
        or isinstance(ensemble["experts"], (str, bytes))
        or tuple(ensemble["experts"])
        != ("markov", "xgboost", "xgb_hazard_destination")
        or ensemble["half_life_weeks"] != 52
        or ensemble["minimum_history_rows"] != 26
        or ensemble["eligible_loss_rule"] != "target_date_strictly_before_origin"
    ):
        raise ContractError(f"{ensemble_context} values are invalid")

    survival_context = f"{models_context}.joint_survival_hazard"
    survival = _validate_exact_object(
        structural_models["joint_survival_hazard"],
        survival_context,
        {
            "base_target_weeks",
            "horizons_weeks",
            "future_covariates",
            "identity",
        },
    )
    horizons = survival["horizons_weeks"]
    if (
        survival["base_target_weeks"] != 1
        or not isinstance(horizons, Sequence)
        or isinstance(horizons, (str, bytes))
        or tuple(horizons) != TRANSITION_HORIZONS
        or survival["future_covariates"] != "origin_values_frozen"
        or survival["identity"]
        != "one_minus_product_one_minus_weekly_hazard"
    ):
        raise ContractError(f"{survival_context} values are invalid")

    ablation_context = f"{context}.ablation"
    ablation = _validate_exact_object(
        _require(model, "ablation", context),
        ablation_context,
        {
            "anchor_model",
            "reference_variant",
            "published_variant",
            "primary_period",
            "post_2023_role",
            "may_change_published_variant",
            "manifest_sha256",
        },
    )
    expected_ablation = {
        "anchor_model": "xgboost",
        "reference_variant": "legacy_v3",
        "published_variant": "all_structural",
        "primary_period": "pre_2023_selection_oos",
        "post_2023_role": "retrospective_diagnostic_only",
        "may_change_published_variant": False,
    }
    if any(ablation.get(field) != expected for field, expected in expected_ablation.items()):
        raise ContractError(f"{ablation_context} values are invalid")
    _validate_sha256(
        _require(ablation, "manifest_sha256", ablation_context),
        f"{ablation_context}.manifest_sha256",
    )


def _validate_v3_week(row: Mapping[str, Any], context: str) -> None:
    transition_risk = _require(row, "transition_risk", context)
    if not isinstance(transition_risk, Mapping) or set(transition_risk) != {"1w", "4w", "13w"}:
        raise ContractError(f"{context}.transition_risk keys are invalid")
    origin = str(_require(row, "date", context))
    for horizon in TRANSITION_HORIZONS:
        key = f"{horizon}w"
        result = _require(transition_risk, key, f"{context}.transition_risk")
        result_context = f"{context}.transition_risk.{key}"
        expected_fields = {
            "probability", "target_end", "model", "threshold", "fallback", "fallback_reason"
        }
        if not isinstance(result, Mapping) or set(result) != expected_fields:
            raise ContractError(f"{result_context} fields are invalid")
        _finite_number(_require(result, "probability", result_context), f"{result_context}.probability", minimum=0.0, maximum=1.0)
        _finite_number(_require(result, "threshold", result_context), f"{result_context}.threshold", minimum=0.0, maximum=1.0)
        if not isinstance(_require(result, "model", result_context), str) or not result["model"]:
            raise ContractError(f"{result_context}.model must be non-empty")
        if not isinstance(_require(result, "fallback", result_context), bool):
            raise ContractError(f"{result_context}.fallback must be boolean")
        if not isinstance(_require(result, "fallback_reason", result_context), str):
            raise ContractError(f"{result_context}.fallback_reason must be a string")
        try:
            origin_date = date.fromisoformat(origin)
            target_end = date.fromisoformat(
                str(_require(result, "target_end", result_context))
            )
        except ValueError as exc:
            raise ContractError(f"{result_context}.target_end must be an ISO date") from exc
        if (target_end - origin_date).days != 7 * horizon:
            raise ContractError(f"{result_context}.target_end must be exactly {horizon} weeks later")
    transition_probability = _finite_number(row["transition_probability"], f"{context}.transition_probability", minimum=0.0, maximum=1.0)
    primary = float(transition_risk["1w"]["probability"])
    if not math.isclose(transition_probability, primary, abs_tol=1e-8, rel_tol=0.0):
        raise ContractError(f"{context} 1w transition aliases disagree")
    current_state = str(row["current"]["state"])
    stay_probability = float(row["next_week"]["probabilities"][current_state])
    if not math.isclose(primary, 1.0 - stay_probability, abs_tol=1e-8, rel_tol=0.0):
        raise ContractError(f"{context} 1w departure probability is inconsistent")
    if str(row["next_week"].get("date")) != str(transition_risk["1w"]["target_end"]):
        raise ContractError(f"{context} next_week date and 1w target disagree")


def validate_dashboard_payload(payload: Mapping[str, Any]) -> None:
    """Validate the minimum fail-closed dashboard payload contract."""

    if not isinstance(payload, Mapping):
        raise ContractError("payload must be an object")
    meta = _require(payload, "meta", "payload")
    if not isinstance(meta, Mapping):
        raise ContractError("payload.meta must be an object")
    if _require(meta, "schema_version", "payload.meta") != SCHEMA_VERSION:
        raise ContractError(f"unsupported schema version: {meta.get('schema_version')}")
    _require(meta, "generated_at", "payload.meta")
    _require(meta, "data_as_of", "payload.meta")
    _require(meta, "mode", "payload.meta")
    if _require(meta, "timezone", "payload.meta") != "America/New_York":
        raise ContractError("payload.meta.timezone must be America/New_York")
    result_version = meta.get("result_version")
    supported_result_versions = {V3_RESULT_VERSION, V4_RESULT_VERSION}
    if result_version is not None and result_version not in supported_result_versions:
        raise ContractError(f"unsupported result version: {result_version}")
    is_v3 = result_version == V3_RESULT_VERSION
    is_v4 = result_version == V4_RESULT_VERSION
    is_transition_contract = is_v3 or is_v4
    if is_transition_contract:
        generation_id = _require(meta, "generation_id", "payload.meta")
        if not isinstance(generation_id, str) or not generation_id:
            raise ContractError("payload.meta.generation_id must be non-empty")

    states = _require(payload, "states", "payload")
    if not isinstance(states, Sequence) or isinstance(states, (str, bytes)):
        raise ContractError("payload.states must be an array")
    state_ids = [state.get("id") for state in states if isinstance(state, Mapping)]
    if tuple(state_ids) != STATE_ORDER:
        raise ContractError(f"payload.states must be ordered as {STATE_ORDER}")

    model = _require(payload, "model", "payload")
    if not isinstance(model, Mapping):
        raise ContractError("payload.model must be an object")
    _require(model, "champion", "payload.model")
    selection_status = _require(model, "selection_status", "payload.model")
    if selection_status != "provisional_predeployment":
        raise ContractError(
            "payload.model.selection_status must be provisional_predeployment"
        )
    leaderboard = _require(model, "leaderboard", "payload.model")
    if not isinstance(leaderboard, Sequence) or isinstance(leaderboard, (str, bytes)):
        raise ContractError("payload.model.leaderboard must be an array")
    if is_v3:
        _validate_v3_model(model)
    elif is_v4:
        _validate_v4_model(model)

    weekly = _require(payload, "weekly", "payload")
    if not isinstance(weekly, Sequence) or isinstance(weekly, (str, bytes)) or not weekly:
        raise ContractError("payload.weekly must be a non-empty array")
    previous_date: str | None = None
    for index, row in enumerate(weekly):
        context = f"payload.weekly[{index}]"
        if not isinstance(row, Mapping):
            raise ContractError(f"{context} must be an object")
        date = str(_require(row, "date", context))
        if previous_date is not None and date <= previous_date:
            raise ContractError("payload.weekly dates must be strictly increasing")
        previous_date = date
        for name in ("current", "next_week"):
            estimate = _require(row, name, context)
            if not isinstance(estimate, Mapping):
                raise ContractError(f"{context}.{name} must be an object")
            if _require(estimate, "state", f"{context}.{name}") not in STATE_ORDER:
                raise ContractError(f"{context}.{name}.state is invalid")
            _validate_probabilities(
                _require(estimate, "probabilities", f"{context}.{name}"),
                f"{context}.{name}.probabilities",
            )
        transition = float(_require(row, "transition_probability", context))
        if not math.isfinite(transition) or transition < 0 or transition > 1:
            raise ContractError(f"{context}.transition_probability must be in [0, 1]")
        scores = _require(row, "scores", context)
        if not isinstance(scores, Mapping):
            raise ContractError(f"{context}.scores must be an object")
        for score_name in ("trend", "stress", "macro", "financial_conditions"):
            score = float(_require(scores, score_name, f"{context}.scores"))
            if not math.isfinite(score) or score < -1.000001 or score > 1.000001:
                raise ContractError(f"{context}.scores.{score_name} must be in [-1, 1]")
        if is_transition_contract:
            _validate_v3_week(row, context)

    if is_v4:
        artifact_count = model["evidence_artifacts"]["weekly_state_forecasts"][
            "row_count"
        ]
        if int(artifact_count) != len(weekly):
            raise ContractError(
                "payload.model.evidence_artifacts.weekly_state_forecasts.row_count "
                "must equal payload.weekly length"
            )

    sources = _require(payload, "sources", "payload")
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        raise ContractError("payload.sources must be an array")
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise ContractError(f"payload.sources[{index}] must be an object")
        status = _require(source, "status", f"payload.sources[{index}]")
        if status not in HEALTH_STATUSES:
            raise ContractError(f"payload.sources[{index}].status is invalid")

    feature_catalog = _require(payload, "feature_catalog", "payload")
    if not isinstance(feature_catalog, Sequence) or isinstance(
        feature_catalog, (str, bytes)
    ):
        raise ContractError("payload.feature_catalog must be an array")
    if not feature_catalog:
        raise ContractError("payload.feature_catalog must not be empty")
    for index, feature in enumerate(feature_catalog):
        if not isinstance(feature, Mapping):
            raise ContractError(f"payload.feature_catalog[{index}] must be an object")
        for key in ("id", "category", "frequency", "source"):
            _require(feature, key, f"payload.feature_catalog[{index}]")
