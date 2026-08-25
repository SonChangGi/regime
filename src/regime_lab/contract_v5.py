"""Fail-closed public contract for the opt-in v5 research payload."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
import hashlib
import json
import math
from typing import Any

from regime_lab.frozen_v4 import FROZEN_V4_BASELINE
from regime_lab.v5_artifacts import (
    CONDITIONAL_STATISTICS_COLUMNS,
    FX_RESEARCH_ARTIFACT_KEYS,
    MODEL_CONDITIONED_STATISTICS_COLUMNS,
    OPTIONAL_RESEARCH_ARTIFACT_KEYS,
    REQUIRED_RESEARCH_ARTIFACT_KEYS,
    V5_CORE_ARTIFACT_PATHS,
    V5_RESEARCH_ARTIFACTS,
)


V5_SCHEMA_VERSION = "2.0.0"
V5_RESULT_VERSION = "weekly-regime-result-v5"
V5_MODEL_VERSION = "weekly-nondl-structural-v5"
V5_LABEL_VERSION = "market-causal-3state-v1"
V5_FEATURE_SET_VERSION = "weekly-pit-structural-v5"
V5_PUBLICATION_STATUS = "reviewed_publication"
V5_PUBLICATION_REVIEW_SCHEMA = "regime-v5-publication-review/1"
V5_PAYLOAD_FIELDS = frozenset(
    {"meta", "states", "model", "weekly", "sources", "feature_catalog", "research"}
)
V5_META_FIELDS = frozenset(
    {
        "schema_version",
        "result_version",
        "generated_at",
        "generation_id",
        "data_as_of",
        "mode",
        "status",
        "timezone",
        "cutoff_policy",
        "transition_alert_thresholds",
        "transition_probability_definition",
        "transition_risk_definition",
        "supported_date_range",
        "warnings",
        "current_membership_definition",
        "freshness",
    }
)
V5_PUBLICATION_META_FIELDS = frozenset(
    {*V5_META_FIELDS, "publication_status", "publication_review"}
)
STATE_ORDER = ("risk_on", "transition", "risk_off")
HORIZONS = (1, 4, 13)
OUTCOME_ASSETS = ("SPY", "QQQ", "IWM", "TLT", "HYG", "UUP")
FX_VARIANTS = (
    "v4_control",
    "v4_plus_broad_index",
    "v4_plus_bilateral_panel",
    "v4_plus_all_fx",
)
V5_MULTISCALE_MODEL = "causal_multiscale_ensemble"
V5_FORECAST_COMPARISON_MODELS = (
    "markov",
    "xgboost",
    "xgb_hazard_destination",
    "causal_dynamic_ensemble",
    V5_MULTISCALE_MODEL,
)
V5_STANDARD_CORE_MODELS = frozenset(
    {
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
        "xgb_hazard_destination",
        "causal_dynamic_ensemble",
        V5_MULTISCALE_MODEL,
    }
)
DEMO_SOURCE_LICENSES = {
    "synthetic_market": "synthetic_fixture",
    "synthetic_macro": "synthetic_fixture",
}
LIVE_SOURCE_LICENSES = {
    "alpha_vantage": "private_noncommercial",
    "alfred": "user_confirmed_ml_storage_derived",
    "frb_h10": "federal_reserve_board_public_domain_citation_requested",
}


class V5ContractError(ValueError):
    pass


def _require(value: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in value:
        raise V5ContractError(f"{context}.{key} is required")
    return value[key]


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise V5ContractError(f"{context} must be an object")
    return value


def _sequence(value: Any, context: str, *, nonempty: bool = False) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise V5ContractError(f"{context} must be an array")
    if nonempty and not value:
        raise V5ContractError(f"{context} must not be empty")
    return value


def _number(
    value: Any,
    context: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise V5ContractError(f"{context} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise V5ContractError(f"{context} must be a finite number") from exc
    if not math.isfinite(result):
        raise V5ContractError(f"{context} must be finite")
    if minimum is not None and result < minimum:
        raise V5ContractError(f"{context} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise V5ContractError(f"{context} must be at most {maximum}")
    return result


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    number = _number(value, context, minimum=float(minimum))
    if not number.is_integer():
        raise V5ContractError(f"{context} must be an integer")
    return int(number)


def _optional_number(
    value: Any,
    context: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if value is None:
        return None
    return _number(value, context, minimum=minimum, maximum=maximum)


def _probabilities(value: Any, context: str) -> Mapping[str, Any]:
    result = _mapping(value, context)
    if set(result) != set(STATE_ORDER):
        raise V5ContractError(f"{context} keys must be exactly {STATE_ORDER}")
    total = sum(
        _number(result[state], f"{context}.{state}", minimum=0.0, maximum=1.0)
        for state in STATE_ORDER
    )
    if not math.isclose(total, 1.0, abs_tol=1e-6, rel_tol=0.0):
        raise V5ContractError(f"{context} must sum to one")
    return result


def _sha256(value: Any, context: str) -> None:
    candidate = str(value)
    if len(candidate) != 64 or any(char not in "0123456789abcdef" for char in candidate):
        raise V5ContractError(f"{context} must be a lowercase SHA-256")


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _holm_adjusted_pvalues(pvalues: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running_maximum = 0.0
    hypothesis_count = len(ordered)
    for rank, (name, pvalue) in enumerate(ordered):
        candidate = min(1.0, float(pvalue) * (hypothesis_count - rank))
        running_maximum = max(running_maximum, candidate)
        adjusted[name] = running_maximum
    return adjusted


def _iso_date(value: Any, context: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise V5ContractError(f"{context} must be an ISO date") from exc


def _iso_datetime(value: Any, context: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise V5ContractError(f"{context} must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise V5ContractError(f"{context} must include a timezone")
    return parsed


def _validate_current(value: Any, context: str) -> str:
    current = _mapping(value, context)
    expected = {
        "state",
        "memberships",
        "primary_membership",
        "membership_entropy",
        "method",
    }
    if set(current) != expected:
        raise V5ContractError(f"{context} fields must be exactly {tuple(sorted(expected))}")
    state = str(current["state"])
    if state not in STATE_ORDER:
        raise V5ContractError(f"{context}.state is invalid")
    memberships = _probabilities(current["memberships"], f"{context}.memberships")
    primary = _number(
        current["primary_membership"],
        f"{context}.primary_membership",
        minimum=0.0,
        maximum=1.0,
    )
    if not math.isclose(primary, float(memberships[state]), abs_tol=1e-8, rel_tol=0.0):
        raise V5ContractError(f"{context}.primary_membership must match the hard state")
    _number(
        current["membership_entropy"],
        f"{context}.membership_entropy",
        minimum=0.0,
        maximum=1.0,
    )
    if current["method"] != "risk_score_anchor_membership":
        raise V5ContractError(f"{context}.method is invalid")
    return state


def _validate_forecast(value: Any, context: str) -> date:
    forecast = _mapping(value, context)
    expected = {
        "state",
        "probabilities",
        "confidence",
        "entropy",
        "date",
        "method",
        "model",
        "fallback",
        "fallback_reason",
    }
    if set(forecast) != expected:
        raise V5ContractError(f"{context} fields are invalid")
    state = str(_require(forecast, "state", context))
    if state not in STATE_ORDER:
        raise V5ContractError(f"{context}.state is invalid")
    probabilities = _probabilities(
        _require(forecast, "probabilities", context), f"{context}.probabilities"
    )
    confidence = _number(
        _require(forecast, "confidence", context),
        f"{context}.confidence",
        minimum=0.0,
        maximum=1.0,
    )
    if not math.isclose(confidence, float(probabilities[state]), abs_tol=1e-8, rel_tol=0.0):
        raise V5ContractError(f"{context}.confidence must match the predicted state")
    _number(
        _require(forecast, "entropy", context),
        f"{context}.entropy",
        minimum=0.0,
        maximum=1.0,
    )
    target_date = _iso_date(_require(forecast, "date", context), f"{context}.date")
    for name in ("method", "model", "fallback_reason"):
        if not isinstance(_require(forecast, name, context), str):
            raise V5ContractError(f"{context}.{name} must be a string")
    if not isinstance(_require(forecast, "fallback", context), bool):
        raise V5ContractError(f"{context}.fallback must be boolean")
    return target_date


def _validate_transition_risk(
    value: Any,
    context: str,
    *,
    origin: date,
) -> dict[int, float]:
    risk = _mapping(value, context)
    if set(risk) != {"1w", "4w", "13w"}:
        raise V5ContractError(f"{context} horizons are invalid")
    probabilities: dict[int, float] = {}
    for horizon in HORIZONS:
        key = f"{horizon}w"
        row = _mapping(risk[key], f"{context}.{key}")
        probability = _number(
            _require(row, "probability", f"{context}.{key}"),
            f"{context}.{key}.probability",
            minimum=0.0,
            maximum=1.0,
        )
        target_end = _iso_date(
            _require(row, "target_end", f"{context}.{key}"),
            f"{context}.{key}.target_end",
        )
        if (target_end - origin).days != 7 * horizon:
            raise V5ContractError(f"{context}.{key}.target_end is inconsistent")
        probabilities[horizon] = probability
    return probabilities


def _validate_directional_risk(
    value: Any,
    context: str,
    *,
    current_state: str,
    departure: Mapping[int, float],
    origin: date,
) -> None:
    directional = _mapping(value, context)
    if set(directional) != {"1w", "4w", "13w"}:
        raise V5ContractError(f"{context} horizons are invalid")
    for horizon in HORIZONS:
        key = f"{horizon}w"
        row_context = f"{context}.{key}"
        row = _mapping(directional[key], row_context)
        expected = {
            "probability",
            "no_departure",
            "first_destination",
            "target_end",
            "model",
            "method",
        }
        if set(row) != expected:
            raise V5ContractError(f"{row_context} fields are invalid")
        probability = _number(
            row["probability"], f"{row_context}.probability", minimum=0.0, maximum=1.0
        )
        if not math.isclose(probability, departure[horizon], abs_tol=1e-8, rel_tol=0.0):
            raise V5ContractError(f"{row_context}.probability must match transition_risk")
        no_departure = _number(
            row["no_departure"],
            f"{row_context}.no_departure",
            minimum=0.0,
            maximum=1.0,
        )
        if not math.isclose(no_departure, 1.0 - probability, abs_tol=1e-8, rel_tol=0.0):
            raise V5ContractError(f"{row_context}.no_departure is inconsistent")
        destinations = _mapping(row["first_destination"], f"{row_context}.first_destination")
        if set(destinations) != set(STATE_ORDER):
            raise V5ContractError(f"{row_context}.first_destination keys are invalid")
        destination_sum = sum(
            _number(
                destinations[state],
                f"{row_context}.first_destination.{state}",
                minimum=0.0,
                maximum=1.0,
            )
            for state in STATE_ORDER
        )
        if not math.isclose(destination_sum, probability, abs_tol=1e-6, rel_tol=0.0):
            raise V5ContractError(f"{row_context}.first_destination must sum to departure risk")
        if not math.isclose(float(destinations[current_state]), 0.0, abs_tol=1e-8):
            raise V5ContractError(f"{row_context} cannot first-depart into the origin state")
        target_end = _iso_date(row["target_end"], f"{row_context}.target_end")
        if (target_end - origin).days != 7 * horizon:
            raise V5ContractError(f"{row_context}.target_end is inconsistent")
        if not isinstance(row["model"], str) or not row["model"]:
            raise V5ContractError(f"{row_context}.model must be non-empty")
        if row["method"] != "first_departure_state_within_h_or_no_departure":
            raise V5ContractError(f"{row_context}.method is invalid")


def _validate_duration(value: Any, context: str, current_state: str) -> None:
    duration = _mapping(value, context)
    if duration.get("status") not in {"ok", "insufficient_history", "unavailable"}:
        raise V5ContractError(f"{context}.status is invalid")
    if duration.get("method") != "state_specific_kaplan_meier":
        raise V5ContractError(f"{context}.method is invalid")
    if duration.get("state") != current_state:
        raise V5ContractError(f"{context}.state must match current.state")
    _integer(_require(duration, "elapsed_weeks", context), f"{context}.elapsed_weeks", minimum=1)
    _integer(_require(duration, "episodes", context), f"{context}.episodes", minimum=1)
    _integer(_require(duration, "completed_spells", context), f"{context}.completed_spells")
    _integer(_require(duration, "censored_spells", context), f"{context}.censored_spells")
    _integer(
        _require(duration, "minimum_completed_spells", context),
        f"{context}.minimum_completed_spells",
        minimum=1,
    )
    for field in ("median_remaining_weeks", "restricted_mean_remaining_weeks"):
        _optional_number(duration.get(field), f"{context}.{field}", minimum=0.0)
    if _integer(
        _require(duration, "restriction_weeks", context),
        f"{context}.restriction_weeks",
        minimum=1,
    ) != 52:
        raise V5ContractError(f"{context}.restriction_weeks must be 52")
    stay = _mapping(
        _require(duration, "conditional_survival", context),
        f"{context}.conditional_survival",
    )
    depart = _mapping(
        _require(duration, "departure_probability", context),
        f"{context}.departure_probability",
    )
    if set(stay) != {"4w", "13w"} or set(depart) != {"4w", "13w"}:
        raise V5ContractError(f"{context} duration horizon keys are invalid")
    for key in ("4w", "13w"):
        stay_value = _optional_number(
            stay[key], f"{context}.conditional_survival.{key}", minimum=0.0, maximum=1.0
        )
        depart_value = _optional_number(
            depart[key], f"{context}.departure_probability.{key}", minimum=0.0, maximum=1.0
        )
        if (stay_value is None) != (depart_value is None):
            raise V5ContractError(f"{context}.{key} duration values must share nullability")
        if stay_value is not None and not math.isclose(
            stay_value + float(depart_value), 1.0, abs_tol=1e-8, rel_tol=0.0
        ):
            raise V5ContractError(f"{context}.{key} duration values are inconsistent")
    bootstrap = _mapping(_require(duration, "bootstrap", context), f"{context}.bootstrap")
    if bootstrap.get("unit") != "episode":
        raise V5ContractError(f"{context}.bootstrap.unit is invalid")
    _integer(_require(bootstrap, "resamples", f"{context}.bootstrap"), f"{context}.bootstrap.resamples")
    _integer(_require(bootstrap, "valid_resamples", f"{context}.bootstrap"), f"{context}.bootstrap.valid_resamples")
    _integer(_require(bootstrap, "seed", f"{context}.bootstrap"), f"{context}.bootstrap.seed")
    _number(_require(bootstrap, "interval", f"{context}.bootstrap"), f"{context}.bootstrap.interval", minimum=0.0, maximum=1.0)
    ci95 = duration.get("ci95")
    if ci95 is not None and not isinstance(ci95, Mapping):
        raise V5ContractError(f"{context}.ci95 must be an object or null")


def _validate_fx_context(value: Any, context: str) -> None:
    fx = _mapping(value, context)
    if fx.get("status") not in {
        "ok",
        "partial",
        "degraded",
        "stale",
        "insufficient_history",
        "unavailable",
    }:
        raise V5ContractError(f"{context}.status is invalid")
    if fx.get("method") != "fed_h10_usd_strength":
        raise V5ContractError(f"{context}.method is invalid")
    if tuple(fx.get("bilateral_panel", ())) != (
        "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "CNY", "MXN", "BRL"
    ):
        raise V5ContractError(f"{context}.bilateral_panel is invalid")
    coverage = _mapping(_require(fx, "coverage", context), f"{context}.coverage")
    _integer(_require(coverage, "available_pairs", f"{context}.coverage"), f"{context}.coverage.available_pairs")
    _integer(_require(coverage, "required_pairs", f"{context}.coverage"), f"{context}.coverage.required_pairs")
    for block_name in ("indexes", "bilateral"):
        block = _mapping(_require(fx, block_name, context), f"{context}.{block_name}")
        for name, value in block.items():
            _optional_number(value, f"{context}.{block_name}.{name}")


def _validate_fx_ablation(value: Any) -> None:
    context = "payload.model.fx_ablation"
    ablation = _mapping(value, context)
    expected_fields = {
        "role",
        "variants",
        "minimum_common_weeks",
        "historical_availability_backfill",
        "official_release_archive_ingest",
        "availability_basis",
        "archive_revision_policy",
        "archive_correction_availability_basis",
        "status",
        "eligible_common_weeks",
        "first_eligible_cutoff",
        "last_eligible_cutoff",
        "manifest",
        "status_reason",
        "common_origin_required_pairs",
        "minimum_train_weeks",
        "target_horizon_weeks",
        "purge_weeks",
        "target_availability_rule",
        "model",
        "common_evaluation_origins",
        "variant_metrics",
        "gate",
        "promotion_allowed",
        "promotion_candidate",
        "core_champion_promoted",
    }
    if set(ablation) != expected_fields:
        raise V5ContractError(f"{context} fields are invalid")
    if ablation["role"] != "prospective_shadow":
        raise V5ContractError(f"{context}.role is invalid")
    variants = _sequence(ablation["variants"], f"{context}.variants")
    if tuple(variants) != FX_VARIANTS:
        raise V5ContractError(f"{context}.variants are invalid")
    minimum = _integer(
        ablation["minimum_common_weeks"],
        f"{context}.minimum_common_weeks",
        minimum=1,
    )
    if minimum != 156:
        raise V5ContractError(f"{context}.minimum_common_weeks must be 156")
    if ablation["historical_availability_backfill"] is not False:
        raise V5ContractError(
            f"{context}.historical_availability_backfill must be false"
        )
    archive_ingest = ablation["official_release_archive_ingest"]
    if not isinstance(archive_ingest, bool):
        raise V5ContractError(
            f"{context}.official_release_archive_ingest must be boolean"
        )
    availability_basis = ablation["availability_basis"]
    allowed_bases = {
        "official_archive_release_schedule",
        "collection_first_seen_at",
    }
    if availability_basis not in allowed_bases or archive_ingest != (
        availability_basis == "official_archive_release_schedule"
    ):
        raise V5ContractError(f"{context}.availability_basis is inconsistent")
    if (
        ablation["archive_revision_policy"]
        != "later_official_release_preserved_as_new_vintage"
    ):
        raise V5ContractError(f"{context}.archive_revision_policy is invalid")
    if (
        ablation["archive_correction_availability_basis"]
        != "date_only_conservative_next_day"
    ):
        raise V5ContractError(
            f"{context}.archive_correction_availability_basis is invalid"
        )
    status = str(ablation["status"])
    status_reasons = {
        "unavailable": {
            "fx_feature_result_unavailable",
            "fx_feature_contract_unavailable",
            "fixed_nine_pair_contract_unavailable",
            "fx_model_features_non_numeric",
        },
        "insufficient_history": {
            "eligible_common_weeks_below_156",
            "no_origin_has_104_strictly_available_training_targets",
        },
        "evaluated": {None},
    }
    if status not in status_reasons:
        raise V5ContractError(f"{context}.status is invalid")
    if ablation["status_reason"] not in status_reasons[status]:
        raise V5ContractError(f"{context}.status_reason is invalid")
    count = _integer(
        ablation["eligible_common_weeks"],
        f"{context}.eligible_common_weeks",
    )
    if count >= minimum and not archive_ingest:
        raise V5ContractError(
            f"{context} ready/evaluated history requires official archive provenance"
        )
    first_raw = ablation["first_eligible_cutoff"]
    last_raw = ablation["last_eligible_cutoff"]
    if count == 0:
        if first_raw is not None or last_raw is not None:
            raise V5ContractError(f"{context} empty date bounds are invalid")
    else:
        first = _iso_date(first_raw, f"{context}.first_eligible_cutoff")
        last = _iso_date(last_raw, f"{context}.last_eligible_cutoff")
        if last < first:
            raise V5ContractError(f"{context} date bounds are reversed")
    manifest = _sequence(ablation["manifest"], f"{context}.manifest")
    if manifest:
        if len(manifest) != len(FX_VARIANTS):
            raise V5ContractError(f"{context}.manifest is incomplete")
        for index, variant in enumerate(FX_VARIANTS):
            row_context = f"{context}.manifest[{index}]"
            row = _mapping(manifest[index], row_context)
            if set(row) != {
                "variant",
                "feature_count",
                "feature_columns_sha256",
            }:
                raise V5ContractError(f"{row_context} fields are invalid")
            if row["variant"] != variant:
                raise V5ContractError(f"{row_context}.variant is invalid")
            feature_count = _integer(
                row["feature_count"], f"{row_context}.feature_count"
            )
            if (variant == "v4_control") != (feature_count == 0):
                raise V5ContractError(f"{row_context}.feature_count is invalid")
            _sha256(
                row["feature_columns_sha256"],
                f"{row_context}.feature_columns_sha256",
            )
    elif count != 0:
        raise V5ContractError(f"{context}.manifest is required for eligible rows")

    if status == "evaluated" and (count < minimum or not manifest):
        raise V5ContractError(f"{context} evaluated state is inconsistent")
    if (
        ablation["status_reason"] == "eligible_common_weeks_below_156"
        and count >= minimum
    ):
        raise V5ContractError(f"{context} readiness is inconsistent")
    if (
        status == "unavailable"
        and ablation["status_reason"] != "fx_model_features_non_numeric"
        and (count != 0 or manifest)
    ):
        raise V5ContractError(f"{context} unavailable state is inconsistent")

    if _integer(
        ablation["common_origin_required_pairs"],
        f"{context}.common_origin_required_pairs",
        minimum=1,
    ) != 9:
        raise V5ContractError(f"{context}.common_origin_required_pairs must be 9")
    if _integer(
        ablation["minimum_train_weeks"],
        f"{context}.minimum_train_weeks",
        minimum=1,
    ) != 104:
        raise V5ContractError(f"{context}.minimum_train_weeks must be 104")
    if _integer(
        ablation["target_horizon_weeks"],
        f"{context}.target_horizon_weeks",
        minimum=1,
    ) != 1:
        raise V5ContractError(f"{context}.target_horizon_weeks must be 1")
    if _integer(
        ablation["purge_weeks"],
        f"{context}.purge_weeks",
        minimum=1,
    ) != 1:
        raise V5ContractError(f"{context}.purge_weeks must be 1")
    if (
        ablation["target_availability_rule"]
        != "last_train_target_strictly_before_evaluation_origin"
    ):
        raise V5ContractError(f"{context}.target_availability_rule is invalid")

    model_context = f"{context}.model"
    model = _mapping(ablation["model"], model_context)
    model_contract = {
        "name": "fixed_l2_multinomial_logistic",
        "horizon_weeks": 1,
        "multiclass": "multinomial",
        "regularization": "l2",
        "regularization_c": 0.10,
        "class_weight": None,
        "solver": "lbfgs",
        "max_iter": 2_000,
        "tolerance": 1e-6,
        "random_state": 17,
        "imputation": "expanding_train_median",
        "scaling": "expanding_train_standard",
        "fit_window": "expanding",
        "state_order": list(STATE_ORDER),
    }
    if set(model) != set(model_contract):
        raise V5ContractError(f"{model_context} fields are invalid")
    for key, expected in model_contract.items():
        supplied = model[key]
        if isinstance(expected, float):
            if not math.isclose(
                _number(supplied, f"{model_context}.{key}"),
                expected,
                abs_tol=1e-12,
                rel_tol=0.0,
            ):
                raise V5ContractError(f"{model_context}.{key} is invalid")
        elif supplied != expected:
            raise V5ContractError(f"{model_context}.{key} is invalid")

    origins_context = f"{context}.common_evaluation_origins"
    origins = _mapping(ablation["common_evaluation_origins"], origins_context)
    if set(origins) != {"count", "first_origin", "last_origin", "sha256", "rows"}:
        raise V5ContractError(f"{origins_context} fields are invalid")
    origin_count = _integer(origins["count"], f"{origins_context}.count")
    origin_rows = _sequence(origins["rows"], f"{origins_context}.rows")
    if origin_count != len(origin_rows):
        raise V5ContractError(f"{origins_context}.count is inconsistent")
    origin_pairs: list[list[str]] = []
    previous_origin: date | None = None
    for index, raw_row in enumerate(origin_rows):
        row_context = f"{origins_context}.rows[{index}]"
        row = _mapping(raw_row, row_context)
        if set(row) != {
            "origin_date",
            "target_date",
            "train_size",
            "train_start_origin",
            "last_train_origin",
            "last_train_target",
            "purged_origin_count",
        }:
            raise V5ContractError(f"{row_context} fields are invalid")
        origin = _iso_date(row["origin_date"], f"{row_context}.origin_date")
        target = _iso_date(row["target_date"], f"{row_context}.target_date")
        train_start = _iso_date(
            row["train_start_origin"], f"{row_context}.train_start_origin"
        )
        last_train_origin = _iso_date(
            row["last_train_origin"], f"{row_context}.last_train_origin"
        )
        last_train_target = _iso_date(
            row["last_train_target"], f"{row_context}.last_train_target"
        )
        if (target - origin).days != 7:
            raise V5ContractError(f"{row_context} target horizon is invalid")
        if not (
            train_start <= last_train_origin < last_train_target < origin
            and (last_train_target - last_train_origin).days == 7
        ):
            raise V5ContractError(f"{row_context} training target purge is invalid")
        if previous_origin is not None and origin <= previous_origin:
            raise V5ContractError(f"{origins_context}.rows are not chronological")
        previous_origin = origin
        if _integer(row["train_size"], f"{row_context}.train_size") < 104:
            raise V5ContractError(f"{row_context}.train_size is invalid")
        if _integer(
            row["purged_origin_count"],
            f"{row_context}.purged_origin_count",
        ) != 1:
            raise V5ContractError(f"{row_context}.purged_origin_count must be 1")
        origin_pairs.append([origin.isoformat(), target.isoformat()])

    if origin_count == 0:
        if any(origins[key] is not None for key in ("first_origin", "last_origin", "sha256")):
            raise V5ContractError(f"{origins_context} empty state is inconsistent")
    else:
        first_origin = _iso_date(
            origins["first_origin"], f"{origins_context}.first_origin"
        )
        last_origin = _iso_date(
            origins["last_origin"], f"{origins_context}.last_origin"
        )
        supplied_origin_hash = str(origins["sha256"])
        _sha256(supplied_origin_hash, f"{origins_context}.sha256")
        if (
            first_origin.isoformat() != origin_pairs[0][0]
            or last_origin.isoformat() != origin_pairs[-1][0]
            or supplied_origin_hash != _canonical_sha256(origin_pairs)
        ):
            raise V5ContractError(f"{origins_context} summary is inconsistent")

    metrics = _sequence(ablation["variant_metrics"], f"{context}.variant_metrics")
    metric_index: dict[str, Mapping[str, Any]] = {}
    core_feature_counts: set[int] = set()
    for index, raw_row in enumerate(metrics):
        row_context = f"{context}.variant_metrics[{index}]"
        row = _mapping(raw_row, row_context)
        if set(row) != {
            "variant",
            "feature_count",
            "fx_feature_count",
            "feature_columns_sha256",
            "log_loss",
            "brier",
            "accuracy",
            "balanced_accuracy",
            "n",
            "n_predictions",
            "fallback",
            "fallback_count",
            "fallback_reasons",
            "first_origin",
            "last_origin",
            "origin_sha256",
        }:
            raise V5ContractError(f"{row_context} fields are invalid")
        if index >= len(FX_VARIANTS) or row["variant"] != FX_VARIANTS[index]:
            raise V5ContractError(f"{row_context}.variant is invalid")
        variant = str(row["variant"])
        feature_count = _integer(
            row["feature_count"], f"{row_context}.feature_count", minimum=1
        )
        fx_feature_count = _integer(
            row["fx_feature_count"], f"{row_context}.fx_feature_count"
        )
        if feature_count < fx_feature_count or (
            (variant == "v4_control") != (fx_feature_count == 0)
        ):
            raise V5ContractError(f"{row_context}.fx_feature_count is invalid")
        core_feature_counts.add(feature_count - fx_feature_count)
        _sha256(
            row["feature_columns_sha256"],
            f"{row_context}.feature_columns_sha256",
        )
        _number(row["log_loss"], f"{row_context}.log_loss", minimum=0.0)
        _number(row["brier"], f"{row_context}.brier", minimum=0.0, maximum=2.0)
        _number(row["accuracy"], f"{row_context}.accuracy", minimum=0.0, maximum=1.0)
        _number(
            row["balanced_accuracy"],
            f"{row_context}.balanced_accuracy",
            minimum=0.0,
            maximum=1.0,
        )
        n = _integer(row["n"], f"{row_context}.n")
        n_predictions = _integer(
            row["n_predictions"], f"{row_context}.n_predictions"
        )
        fallback_count = _integer(
            row["fallback_count"], f"{row_context}.fallback_count"
        )
        if n != origin_count or n_predictions != origin_count or fallback_count > n:
            raise V5ContractError(f"{row_context} sample counts are inconsistent")
        if not isinstance(row["fallback"], bool) or row["fallback"] != (
            fallback_count > 0
        ):
            raise V5ContractError(f"{row_context}.fallback is inconsistent")
        reasons = _mapping(row["fallback_reasons"], f"{row_context}.fallback_reasons")
        allowed_reasons = {
            "training_class_coverage",
            "model_fit_or_prediction_error",
        }
        if not set(reasons).issubset(allowed_reasons) or sum(
            _integer(value, f"{row_context}.fallback_reasons.{key}", minimum=1)
            for key, value in reasons.items()
        ) != fallback_count:
            raise V5ContractError(f"{row_context}.fallback_reasons are inconsistent")
        if (
            row["first_origin"] != origins["first_origin"]
            or row["last_origin"] != origins["last_origin"]
            or row["origin_sha256"] != origins["sha256"]
        ):
            raise V5ContractError(f"{row_context} origin contract is inconsistent")
        metric_index[variant] = row

    if status == "evaluated":
        if origin_count < 1 or len(metrics) != len(FX_VARIANTS):
            raise V5ContractError(f"{context} evaluated outputs are incomplete")
        if len(core_feature_counts) != 1:
            raise V5ContractError(
                f"{context}.variant_metrics core feature counts are inconsistent"
            )
    elif origin_count != 0 or metrics:
        raise V5ContractError(f"{context} non-evaluated outputs must be empty")

    gate_context = f"{context}.gate"
    gate = _mapping(ablation["gate"], gate_context)
    if set(gate) != {
        "reference_variant",
        "method",
        "bootstrap_block_weeks",
        "bootstrap_effective_block_weeks",
        "bootstrap_resamples",
        "bootstrap_seed",
        "alpha",
        "minimum_log_loss_improvement",
        "brier_tolerance",
        "comparisons",
        "passed_variants",
    }:
        raise V5ContractError(f"{gate_context} fields are invalid")
    if (
        gate["reference_variant"] != "v4_control"
        or gate["method"] != "paired_circular_moving_block_bootstrap_holm"
        or _integer(gate["bootstrap_block_weeks"], f"{gate_context}.bootstrap_block_weeks") != 13
        or _integer(gate["bootstrap_resamples"], f"{gate_context}.bootstrap_resamples") != 1_999
        or _integer(gate["bootstrap_seed"], f"{gate_context}.bootstrap_seed") != 17
        or not math.isclose(_number(gate["alpha"], f"{gate_context}.alpha"), 0.05, abs_tol=1e-12, rel_tol=0.0)
        or not math.isclose(_number(gate["minimum_log_loss_improvement"], f"{gate_context}.minimum_log_loss_improvement"), 0.05, abs_tol=1e-12, rel_tol=0.0)
        or not math.isclose(_number(gate["brier_tolerance"], f"{gate_context}.brier_tolerance"), 0.01, abs_tol=1e-12, rel_tol=0.0)
    ):
        raise V5ContractError(f"{gate_context} preregistered parameters are invalid")
    comparisons = _sequence(gate["comparisons"], f"{gate_context}.comparisons")
    passed_variants = list(
        _sequence(gate["passed_variants"], f"{gate_context}.passed_variants")
    )
    effective_block = gate["bootstrap_effective_block_weeks"]
    if status != "evaluated":
        if effective_block is not None or comparisons or passed_variants:
            raise V5ContractError(f"{gate_context} non-evaluated state is inconsistent")
    else:
        expected_block = min(13, max(1, origin_count // 2))
        if _integer(effective_block, f"{gate_context}.bootstrap_effective_block_weeks") != expected_block:
            raise V5ContractError(f"{gate_context}.bootstrap_effective_block_weeks is invalid")
        if len(comparisons) != len(FX_VARIANTS) - 1:
            raise V5ContractError(f"{gate_context}.comparisons are incomplete")
        expected_passed: list[str] = []
        raw_pvalues: dict[str, float] = {}
        supplied_adjusted_pvalues: dict[str, float] = {}
        control = metric_index["v4_control"]
        control_fallback = int(control["fallback_count"])
        for index, variant in enumerate(FX_VARIANTS[1:]):
            comparison_context = f"{gate_context}.comparisons[{index}]"
            comparison = _mapping(comparisons[index], comparison_context)
            if set(comparison) != {
                "variant",
                "reference_variant",
                "mean_log_loss_improvement",
                "brier_difference",
                "control_fallback_count",
                "fallback_count",
                "raw_p_value",
                "holm_adjusted_p_value",
                "gate_passed",
                "gate_reasons",
            }:
                raise V5ContractError(f"{comparison_context} fields are invalid")
            if comparison["variant"] != variant or comparison["reference_variant"] != "v4_control":
                raise V5ContractError(f"{comparison_context}.variant is invalid")
            challenger = metric_index[variant]
            improvement = _number(
                comparison["mean_log_loss_improvement"],
                f"{comparison_context}.mean_log_loss_improvement",
            )
            brier_difference = _number(
                comparison["brier_difference"],
                f"{comparison_context}.brier_difference",
            )
            raw_p = _number(
                comparison["raw_p_value"],
                f"{comparison_context}.raw_p_value",
                minimum=0.0,
                maximum=1.0,
            )
            adjusted_p = _number(
                comparison["holm_adjusted_p_value"],
                f"{comparison_context}.holm_adjusted_p_value",
                minimum=0.0,
                maximum=1.0,
            )
            raw_pvalues[variant] = raw_p
            supplied_adjusted_pvalues[variant] = adjusted_p
            if (
                not math.isclose(improvement, float(control["log_loss"]) - float(challenger["log_loss"]), abs_tol=1e-10, rel_tol=0.0)
                or not math.isclose(brier_difference, float(challenger["brier"]) - float(control["brier"]), abs_tol=1e-10, rel_tol=0.0)
                or adjusted_p + 1e-12 < raw_p
                or _integer(comparison["control_fallback_count"], f"{comparison_context}.control_fallback_count") != control_fallback
                or _integer(comparison["fallback_count"], f"{comparison_context}.fallback_count") != int(challenger["fallback_count"])
            ):
                raise V5ContractError(f"{comparison_context} metrics are inconsistent")
            expected_failures: list[str] = []
            if control_fallback:
                expected_failures.append("control_fallback_present")
            if int(challenger["fallback_count"]):
                expected_failures.append("fallback_present")
            if improvement + 1e-12 < 0.05:
                expected_failures.append("insufficient_log_loss_improvement")
            if adjusted_p > 0.05:
                expected_failures.append("holm_not_significant")
            if brier_difference > 0.01 + 1e-12:
                expected_failures.append("brier_degradation")
            expected_pass = not expected_failures
            if not isinstance(comparison["gate_passed"], bool) or comparison["gate_passed"] != expected_pass:
                raise V5ContractError(f"{comparison_context}.gate_passed is inconsistent")
            expected_reasons = ["passed"] if expected_pass else expected_failures
            if list(_sequence(comparison["gate_reasons"], f"{comparison_context}.gate_reasons")) != expected_reasons:
                raise V5ContractError(f"{comparison_context}.gate_reasons are inconsistent")
            if expected_pass:
                expected_passed.append(variant)
        expected_adjusted_pvalues = _holm_adjusted_pvalues(raw_pvalues)
        for variant, expected_adjusted in expected_adjusted_pvalues.items():
            if not math.isclose(
                supplied_adjusted_pvalues[variant],
                expected_adjusted,
                abs_tol=1e-12,
                rel_tol=0.0,
            ):
                raise V5ContractError(
                    f"{gate_context}.comparisons Holm adjustment is inconsistent"
                )
        if passed_variants != expected_passed:
            raise V5ContractError(f"{gate_context}.passed_variants are inconsistent")

    if (
        ablation["promotion_allowed"] is not False
        or ablation["promotion_candidate"] is not None
        or ablation["core_champion_promoted"] is not False
    ):
        raise V5ContractError(f"{context} promotion must remain disabled")


def _validate_evidence_artifacts(model: Mapping[str, Any]) -> int:
    evidence = _mapping(
        _require(model, "evidence_artifacts", "payload.model"),
        "payload.model.evidence_artifacts",
    )
    if set(evidence) != {"state_membership_history", "weekly_state_forecasts"}:
        raise V5ContractError("payload.model.evidence_artifacts keys are invalid")
    membership = _mapping(
        evidence["state_membership_history"],
        "payload.model.evidence_artifacts.state_membership_history",
    )
    membership_context = "payload.model.evidence_artifacts.state_membership_history"
    expected_membership = {
        "path",
        "row_count",
        "sha256",
        "label_fit_weeks",
        "label_fit_end",
        "initial_state",
        "method",
    }
    if set(membership) != expected_membership:
        raise V5ContractError(f"{membership_context} fields are invalid")
    if membership["path"] != "state-membership-history.csv":
        raise V5ContractError(f"{membership_context}.path is invalid")
    if _integer(membership["row_count"], f"{membership_context}.row_count") < 520:
        raise V5ContractError(f"{membership_context}.row_count is too small")
    _sha256(membership["sha256"], f"{membership_context}.sha256")
    if membership["label_fit_weeks"] != 520:
        raise V5ContractError(f"{membership_context}.label_fit_weeks is invalid")
    _iso_datetime(membership["label_fit_end"], f"{membership_context}.label_fit_end")
    if membership["initial_state"] not in STATE_ORDER:
        raise V5ContractError(f"{membership_context}.initial_state is invalid")
    if membership["method"] != "risk_score_anchor_membership":
        raise V5ContractError(f"{membership_context}.method is invalid")

    forecast = _mapping(
        evidence["weekly_state_forecasts"],
        "payload.model.evidence_artifacts.weekly_state_forecasts",
    )
    forecast_context = "payload.model.evidence_artifacts.weekly_state_forecasts"
    if set(forecast) != {"path", "row_count", "sha256"}:
        raise V5ContractError(f"{forecast_context} fields are invalid")
    if forecast["path"] != "weekly-state-forecasts-v5.csv":
        raise V5ContractError(f"{forecast_context}.path is invalid")
    row_count = _integer(forecast["row_count"], f"{forecast_context}.row_count")
    _sha256(forecast["sha256"], f"{forecast_context}.sha256")
    return row_count


def _validate_research_artifacts(
    model: Mapping[str, Any],
    directional: Mapping[str, Any],
) -> None:
    context = "payload.model.research_artifacts"
    artifacts = _mapping(_require(model, "research_artifacts", "payload.model"), context)
    keys = set(artifacts)
    allowed = (
        REQUIRED_RESEARCH_ARTIFACT_KEYS
        | OPTIONAL_RESEARCH_ARTIFACT_KEYS
        | FX_RESEARCH_ARTIFACT_KEYS
    )
    if not REQUIRED_RESEARCH_ARTIFACT_KEYS.issubset(keys) or not keys.issubset(
        allowed
    ):
        raise V5ContractError(f"{context} keys are invalid")
    present_model_conditioned = keys & OPTIONAL_RESEARCH_ARTIFACT_KEYS
    if present_model_conditioned and present_model_conditioned != OPTIONAL_RESEARCH_ARTIFACT_KEYS:
        raise V5ContractError(
            f"{context} model-conditioned keys must be supplied as a complete set"
        )
    present_fx = keys & FX_RESEARCH_ARTIFACT_KEYS
    if present_fx and present_fx != FX_RESEARCH_ARTIFACT_KEYS:
        raise V5ContractError(f"{context} FX keys must be supplied as a complete set")

    counts: dict[str, int] = {}
    for key in keys:
        item_context = f"{context}.{key}"
        item = _mapping(artifacts[key], item_context)
        if set(item) != {"path", "row_count", "sha256"}:
            raise V5ContractError(f"{item_context} fields are invalid")
        spec = V5_RESEARCH_ARTIFACTS[key]
        if item["path"] != spec.path:
            raise V5ContractError(f"{item_context}.path is invalid")
        counts[key] = _integer(
            item["row_count"],
            f"{item_context}.row_count",
            minimum=0 if key == "fx_ablation_oos" else 1,
        )
        _sha256(item["sha256"], f"{item_context}.sha256")

    if counts["conditional_asset_statistics"] != (
        len(OUTCOME_ASSETS) * len(STATE_ORDER) * len(HORIZONS)
    ):
        raise V5ContractError(
            f"{context}.conditional_asset_statistics.row_count is invalid"
        )
    leaderboard = _sequence(
        _require(directional, "leaderboard", "payload.model.directional_transition"),
        "payload.model.directional_transition.leaderboard",
        nonempty=True,
    )
    if counts["directional_model_leaderboard"] != len(leaderboard):
        raise V5ContractError(
            f"{context}.directional_model_leaderboard.row_count is inconsistent"
        )
    diagnostics = _sequence(
        _require(
            directional,
            "selection_diagnostics",
            "payload.model.directional_transition",
        ),
        "payload.model.directional_transition.selection_diagnostics",
        nonempty=True,
    )
    if counts["directional_selection_diagnostics"] != len(diagnostics):
        raise V5ContractError(
            f"{context}.directional_selection_diagnostics.row_count is inconsistent"
        )
    if present_fx and counts["fx_features"] != counts["fx_coverage"]:
        raise V5ContractError(f"{context} FX row counts are inconsistent")
    fx_ablation = _mapping(
        _require(model, "fx_ablation", "payload.model"),
        "payload.model.fx_ablation",
    )
    fx_status = str(_require(fx_ablation, "status", "payload.model.fx_ablation"))
    origin_count = _integer(
        _mapping(
            _require(
                fx_ablation,
                "common_evaluation_origins",
                "payload.model.fx_ablation",
            ),
            "payload.model.fx_ablation.common_evaluation_origins",
        )["count"],
        "payload.model.fx_ablation.common_evaluation_origins.count",
    )
    if fx_status == "evaluated" and not present_fx:
        raise V5ContractError(f"{context}.fx_ablation_oos is required when evaluated")
    if present_fx:
        expected_oos_rows = origin_count * len(FX_VARIANTS) if fx_status == "evaluated" else 0
        if counts["fx_ablation_oos"] != expected_oos_rows:
            raise V5ContractError(
                f"{context}.fx_ablation_oos.row_count is inconsistent"
            )


def _validate_core_artifacts(model: Mapping[str, Any]) -> None:
    context = "payload.model.core_artifacts"
    artifacts = _mapping(_require(model, "core_artifacts", "payload.model"), context)
    if tuple(artifacts) != tuple(key for key, _ in V5_CORE_ARTIFACT_PATHS):
        raise V5ContractError(f"{context} keys/order are invalid")
    for key, expected_path in V5_CORE_ARTIFACT_PATHS:
        item_context = f"{context}.{key}"
        item = _mapping(artifacts[key], item_context)
        if set(item) != {"path", "row_count", "sha256"}:
            raise V5ContractError(f"{item_context} fields are invalid")
        if item["path"] != expected_path:
            raise V5ContractError(f"{item_context}.path is invalid")
        _integer(item["row_count"], f"{item_context}.row_count", minimum=1)
        _sha256(item["sha256"], f"{item_context}.sha256")


def _validate_feature_quality_artifact(model: Mapping[str, Any]) -> None:
    context = "payload.model.feature_quality_artifact"
    artifact = _mapping(
        _require(model, "feature_quality_artifact", "payload.model"),
        context,
    )
    expected_fields = {
        "path",
        "row_count",
        "feature_count",
        "status",
        "warning_feature_count",
        "unavailable_feature_count",
        "content_sha256",
        "sha256",
    }
    if set(artifact) != expected_fields:
        raise V5ContractError(f"{context} fields are invalid")
    if artifact["path"] != "feature-quality.json":
        raise V5ContractError(f"{context}.path is invalid")
    _integer(artifact["row_count"], f"{context}.row_count", minimum=1)
    feature_count = _integer(
        artifact["feature_count"],
        f"{context}.feature_count",
        minimum=1,
    )
    warning_count = _integer(
        artifact["warning_feature_count"],
        f"{context}.warning_feature_count",
    )
    unavailable_count = _integer(
        artifact["unavailable_feature_count"],
        f"{context}.unavailable_feature_count",
    )
    if warning_count + unavailable_count > feature_count:
        raise V5ContractError(f"{context} feature counts are inconsistent")
    if artifact["status"] not in {"ok", "warning"}:
        raise V5ContractError(f"{context}.status is invalid")
    if bool(warning_count or unavailable_count) != (
        artifact["status"] == "warning"
    ):
        raise V5ContractError(f"{context}.status is inconsistent")
    _sha256(artifact["content_sha256"], f"{context}.content_sha256")
    _sha256(artifact["sha256"], f"{context}.sha256")


def _validate_v5_core_candidate_contract(
    model: Mapping[str, Any],
    *,
    profile: str,
) -> None:
    context = "payload.model.candidate_manifest"
    manifest = _mapping(_require(model, "candidate_manifest", "payload.model"), context)
    supplied_hash_raw = _require(
        model,
        "candidate_manifest_sha256",
        "payload.model",
    )
    _sha256(
        supplied_hash_raw,
        "payload.model.candidate_manifest_sha256",
    )
    supplied_hash = str(supplied_hash_raw)
    encoded = json.dumps(
        dict(manifest),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != supplied_hash:
        raise V5ContractError("payload.model.candidate_manifest_sha256 is inconsistent")
    if manifest.get("profile") != profile or manifest.get("random_state") != 17:
        raise V5ContractError(f"{context} profile/random_state is invalid")
    rows = _sequence(
        _require(manifest, "models", context),
        f"{context}.models",
        nonempty=True,
    )
    names: list[str] = []
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"{context}.models[{index}]")
        name = _require(row, "name", f"{context}.models[{index}]")
        if not isinstance(name, str) or not name:
            raise V5ContractError(f"{context}.models[{index}].name is invalid")
        names.append(name)
    expected = set(V5_STANDARD_CORE_MODELS)
    if profile == "full":
        expected.add("gaussian_hmm")
    if len(names) != len(set(names)) or set(names) != expected:
        raise V5ContractError(f"{context} model set is invalid")

    structural = _mapping(
        _require(model, "structural_models", "payload.model"),
        "payload.model.structural_models",
    )
    multiscale = _mapping(
        _require(
            structural,
            V5_MULTISCALE_MODEL,
            "payload.model.structural_models",
        ),
        f"payload.model.structural_models.{V5_MULTISCALE_MODEL}",
    )
    expected_fields = {
        "role",
        "experts",
        "scale_half_lives_weeks",
        "outer_scale_weights",
        "aggregation",
        "inner_pool_method",
        "minimum_history_rows",
        "eligible_loss_rule",
        "selection_gate",
        "automatic_promotion_bypass",
        "sidecar",
    }
    if set(multiscale) != expected_fields:
        raise V5ContractError(
            f"payload.model.structural_models.{V5_MULTISCALE_MODEL} fields are invalid"
        )
    exact = {
        "role": "v5_opt_in_candidate",
        "experts": ["markov", "xgboost", "xgb_hazard_destination"],
        "scale_half_lives_weeks": [26, 52, 104],
        "aggregation": "fixed_equal_probability_average",
        "inner_pool_method": "causal_discounted_completed_oos_log_score",
        "minimum_history_rows": 26,
        "eligible_loss_rule": "target_date_strictly_before_origin",
        "selection_gate": "existing_multiclass_holm_log_loss_brier_zero_fallback",
        "automatic_promotion_bypass": False,
    }
    for field, expected_value in exact.items():
        if multiscale[field] != expected_value:
            raise V5ContractError(
                f"payload.model.structural_models.{V5_MULTISCALE_MODEL}.{field} is invalid"
            )
    weights = list(
        _sequence(
            multiscale["outer_scale_weights"],
            f"payload.model.structural_models.{V5_MULTISCALE_MODEL}.outer_scale_weights",
        )
    )
    if len(weights) != 3 or any(
        not math.isclose(
            _number(value, "multiscale outer weight"),
            1.0 / 3.0,
            abs_tol=1e-15,
            rel_tol=0.0,
        )
        for value in weights
    ):
        raise V5ContractError(
            f"payload.model.structural_models.{V5_MULTISCALE_MODEL}.outer_scale_weights is invalid"
        )
    sidecar = _mapping(
        multiscale["sidecar"],
        f"payload.model.structural_models.{V5_MULTISCALE_MODEL}.sidecar",
    )
    if dict(sidecar) != dict(model["core_artifacts"]["multiscale_ensemble_scales"]):
        raise V5ContractError(
            f"payload.model.structural_models.{V5_MULTISCALE_MODEL}.sidecar is inconsistent"
        )


def _validate_execution_parameters(value: Any) -> str:
    context = "payload.model.execution_parameters"
    parameters = _mapping(value, context)
    expected = {
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
    if set(parameters) != expected:
        raise V5ContractError(f"{context} fields are invalid")
    profile = parameters.get("profile")
    if profile not in {"quick", "standard", "full"}:
        raise V5ContractError(f"{context}.profile is invalid")
    minimum_selection = _integer(
        parameters["directional_minimum_selection_predictions"],
        f"{context}.directional_minimum_selection_predictions",
        minimum=1,
    )
    minimum_diagnostic = _integer(
        parameters["directional_minimum_diagnostic_predictions"],
        f"{context}.directional_minimum_diagnostic_predictions",
        minimum=1,
    )
    expected_minimum = 3 if profile == "quick" else 12
    if minimum_selection != expected_minimum or minimum_diagnostic != expected_minimum:
        raise V5ContractError(f"{context} directional minima are inconsistent")
    expected_maximum = {"quick": 3, "standard": 60, "full": None}[profile]
    for field in (
        "directional_maximum_selection_origins",
        "directional_maximum_diagnostic_origins",
    ):
        raw = parameters[field]
        resolved = None if raw is None else _integer(raw, f"{context}.{field}", minimum=1)
        if resolved != expected_maximum:
            raise V5ContractError(f"{context}.{field} is inconsistent")
    preregistered = _integer(
        parameters["preregistered_bootstrap_resamples"],
        f"{context}.preregistered_bootstrap_resamples",
        minimum=1,
    )
    if preregistered != 1_999:
        raise V5ContractError(f"{context}.preregistered_bootstrap_resamples is invalid")
    duration = _integer(
        parameters["duration_bootstrap_resamples"],
        f"{context}.duration_bootstrap_resamples",
        minimum=1,
    )
    outcomes = _integer(
        parameters["conditional_outcome_bootstrap_resamples"],
        f"{context}.conditional_outcome_bootstrap_resamples",
        minimum=1,
    )
    overrides = list(
        _sequence(parameters["preregistration_overrides"], f"{context}.preregistration_overrides")
    )
    expected_overrides = []
    if duration != preregistered:
        expected_overrides.append("duration.bootstrap_resamples")
    if outcomes != preregistered:
        expected_overrides.append("conditional_asset_statistics.bootstrap_resamples")
    if overrides != expected_overrides:
        raise V5ContractError(f"{context}.preregistration_overrides is inconsistent")
    _sha256(parameters["sha256"], f"{context}.sha256")
    supplied_hash = str(parameters["sha256"])
    unhashed = {key: parameters[key] for key in parameters if key != "sha256"}
    encoded = json.dumps(
        unhashed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    if supplied_hash != hashlib.sha256(encoded).hexdigest():
        raise V5ContractError(f"{context}.sha256 is inconsistent")
    return str(profile)


def _validate_model(model: Any, *, mode: str) -> int:
    model = _mapping(model, "payload.model")
    expected_versions = {
        "version": V5_MODEL_VERSION,
        "label_version": V5_LABEL_VERSION,
        "feature_set_version": V5_FEATURE_SET_VERSION,
    }
    for field, expected in expected_versions.items():
        if _require(model, field, "payload.model") != expected:
            raise V5ContractError(f"payload.model.{field} must be {expected}")
    if model.get("selection_status") != "provisional_predeployment":
        raise V5ContractError("payload.model.selection_status is invalid")
    if not isinstance(model.get("champion"), str) or not model["champion"]:
        raise V5ContractError("payload.model.champion must be non-empty")
    _sequence(_require(model, "leaderboard", "payload.model"), "payload.model.leaderboard", nonempty=True)
    baseline = _mapping(
        _require(model, "baseline_v4", "payload.model"),
        "payload.model.baseline_v4",
    )
    if dict(baseline) != dict(FROZEN_V4_BASELINE):
        raise V5ContractError(
            "payload.model.baseline_v4 must match the frozen v4 contract"
        )
    prereg = _mapping(
        _require(model, "structural_preregistration", "payload.model"),
        "payload.model.structural_preregistration",
    )
    if prereg.get("path") != "config/structural_v5.json":
        raise V5ContractError("payload.model.structural_preregistration.path is invalid")
    _sha256(prereg.get("sha256"), "payload.model.structural_preregistration.sha256")
    execution_profile = _validate_execution_parameters(
        _require(model, "execution_parameters", "payload.model")
    )
    model_profile = _require(model, "profile", "payload.model")
    if model_profile != execution_profile:
        raise V5ContractError(
            "payload.model.profile must match execution_parameters.profile"
        )
    if mode == "live" and execution_profile == "quick":
        raise V5ContractError("payload live mode does not permit the quick profile")
    directional = _mapping(
        _require(model, "directional_transition", "payload.model"),
        "payload.model.directional_transition",
    )
    if directional.get("target") != "first_departure_state_within_h_or_no_departure":
        raise V5ContractError("payload.model.directional_transition.target is invalid")
    if directional.get("deployed_direction_role") != "first_destination_given_departure":
        raise V5ContractError(
            "payload.model.directional_transition.deployed_direction_role is invalid"
        )
    if directional.get("selection_metric") != "conditional_destination_log_loss":
        raise V5ContractError(
            "payload.model.directional_transition.selection_metric is invalid"
        )
    support_contract = {
        "minimum_selection_departure_events": 8,
        "minimum_selection_destination_classes": 2,
        "minimum_selection_event_blocks": 3,
    }
    for field, expected in support_contract.items():
        if directional.get(field) != expected:
            raise V5ContractError(
                f"payload.model.directional_transition.{field} is invalid"
            )
    champions = _mapping(
        _require(directional, "champions", "payload.model.directional_transition"),
        "payload.model.directional_transition.champions",
    )
    if set(champions) != {"1w", "4w", "13w"}:
        raise V5ContractError("payload.model.directional_transition.champions keys are invalid")
    _sequence(
        _require(directional, "leaderboard", "payload.model.directional_transition"),
        "payload.model.directional_transition.leaderboard",
        nonempty=True,
    )
    _iso_date(
        _require(directional, "selection_end", "payload.model.directional_transition"),
        "payload.model.directional_transition.selection_end",
    )
    health = _mapping(_require(model, "model_health", "payload.model"), "payload.model.model_health")
    if health.get("status") not in {"ok", "review_due"}:
        raise V5ContractError("payload.model.model_health.status is invalid")
    _sequence(_require(health, "reasons", "payload.model.model_health"), "payload.model.model_health.reasons")
    if model.get("champion_core_feature_set_version") != "weekly-pit-structural-v4":
        raise V5ContractError(
            "payload.model.champion_core_feature_set_version is invalid"
        )
    if model.get("fx_role") != "context_and_preregistered_shadow_ablation":
        raise V5ContractError("payload.model.fx_role is invalid")
    _validate_fx_ablation(_require(model, "fx_ablation", "payload.model"))
    _validate_core_artifacts(model)
    if "feature_quality_artifact" in model:
        _validate_feature_quality_artifact(model)
    _validate_v5_core_candidate_contract(model, profile=str(model_profile))
    _validate_research_artifacts(model, directional)
    return _validate_evidence_artifacts(model)


def _validate_forecast_comparison(model: Mapping[str, Any]) -> tuple[str, ...] | None:
    raw = model.get("forecast_comparison")
    if raw is None:
        return None
    context = "payload.model.forecast_comparison"
    comparison = _mapping(raw, context)
    if set(comparison) != {"role", "horizon_weeks", "models"}:
        raise V5ContractError(f"{context} fields are invalid")
    if comparison["role"] != "research_comparison":
        raise V5ContractError(f"{context}.role is invalid")
    if comparison["horizon_weeks"] != 1:
        raise V5ContractError(f"{context}.horizon_weeks must be one")
    models = tuple(
        _sequence(comparison["models"], f"{context}.models", nonempty=True)
    )
    champion = str(_require(model, "champion", "payload.model"))
    valid_models = models == V5_FORECAST_COMPARISON_MODELS
    if champion not in V5_FORECAST_COMPARISON_MODELS:
        valid_models = models == (*V5_FORECAST_COMPARISON_MODELS, champion)
    if not valid_models:
        raise V5ContractError(f"{context}.models are invalid")
    leaderboard = _sequence(
        _require(model, "leaderboard", "payload.model"),
        "payload.model.leaderboard",
        nonempty=True,
    )
    leaderboard_names = {
        str(row.get("name"))
        for row in leaderboard
        if isinstance(row, Mapping) and isinstance(row.get("name"), str)
    }
    if not set(models).issubset(leaderboard_names):
        raise V5ContractError(f"{context}.models must exist in the leaderboard")
    return models


def _validate_model_forecasts(
    value: Any,
    context: str,
    *,
    origin: date,
    models: tuple[str, ...],
    champion: str,
    official: Mapping[str, Any],
) -> None:
    rows = _sequence(value, context, nonempty=True)
    if len(rows) != len(models):
        raise V5ContractError(f"{context} model count is invalid")
    validated: dict[str, Mapping[str, Any]] = {}
    for index, expected_model in enumerate(models):
        row_context = f"{context}[{index}]"
        row = _mapping(rows[index], row_context)
        target_date = _validate_forecast(row, row_context)
        if row.get("model") != expected_model:
            raise V5ContractError(f"{row_context}.model order is invalid")
        if row.get("method") != "model_comparison_walk_forward_probability":
            raise V5ContractError(f"{row_context}.method is invalid")
        if (target_date - origin).days != 7:
            raise V5ContractError(f"{row_context}.date is inconsistent")
        probabilities = _mapping(
            row["probabilities"], f"{row_context}.probabilities"
        )
        maximum = max(float(probabilities[state]) for state in STATE_ORDER)
        if not math.isclose(
            float(probabilities[str(row["state"])]),
            maximum,
            abs_tol=1e-8,
            rel_tol=0.0,
        ):
            raise V5ContractError(
                f"{row_context}.state is not the probability argmax"
            )
        validated[expected_model] = row

    champion_row = validated.get(champion)
    if champion_row is None:
        raise V5ContractError(f"{context} does not include the champion")
    parity_fields = (
        "state",
        "probabilities",
        "confidence",
        "entropy",
        "date",
        "model",
        "fallback",
        "fallback_reason",
    )
    if any(champion_row[field] != official[field] for field in parity_fields):
        raise V5ContractError(
            f"{context} champion forecast differs from next_week"
        )


def _validate_publication_review(
    meta: Mapping[str, Any],
    model: Mapping[str, Any],
    *,
    mode: str,
) -> None:
    status = meta.get("publication_status")
    review = meta.get("publication_review")
    if status is None:
        if review is not None:
            raise V5ContractError(
                "payload.meta.publication_review requires publication_status"
            )
        return
    if mode != "live" or status != V5_PUBLICATION_STATUS:
        raise V5ContractError("payload.meta.publication_status is invalid")
    review = _mapping(review, "payload.meta.publication_review")
    expected_fields = {
        "schema_version",
        "decision",
        "reviewed_at",
        "reviewed_candidate_sha256",
        "champion",
        "multiscale_promoted",
        "fx_promoted",
    }
    if set(review) != expected_fields:
        raise V5ContractError("payload.meta.publication_review fields are invalid")
    if review.get("schema_version") != V5_PUBLICATION_REVIEW_SCHEMA:
        raise V5ContractError(
            "payload.meta.publication_review.schema_version is invalid"
        )
    if review.get("decision") != "publish_v5_research_snapshot":
        raise V5ContractError("payload.meta.publication_review.decision is invalid")
    _iso_datetime(
        _require(review, "reviewed_at", "payload.meta.publication_review"),
        "payload.meta.publication_review.reviewed_at",
    )
    _sha256(
        review.get("reviewed_candidate_sha256"),
        "payload.meta.publication_review.reviewed_candidate_sha256",
    )
    if model.get("champion") != "markov" or review.get("champion") != "markov":
        raise V5ContractError(
            "reviewed V5 publication must retain the Markov champion"
        )
    if review.get("multiscale_promoted") is not False:
        raise V5ContractError(
            "payload.meta.publication_review.multiscale_promoted must be false"
        )
    if review.get("fx_promoted") is not False:
        raise V5ContractError(
            "payload.meta.publication_review.fx_promoted must be false"
        )

    leaderboard = _sequence(
        _require(model, "leaderboard", "payload.model"),
        "payload.model.leaderboard",
        nonempty=True,
    )
    selected_leaderboard: list[str] = []
    champion_leaderboard: list[str] = []
    for index, raw in enumerate(leaderboard):
        context = f"payload.model.leaderboard[{index}]"
        row = _mapping(raw, context)
        name = _require(row, "name", context)
        if not isinstance(name, str) or not name:
            raise V5ContractError(f"{context}.name must be non-empty")
        if not isinstance(row.get("selected"), bool) or not isinstance(
            row.get("is_champion"), bool
        ):
            raise V5ContractError(
                f"{context}.selected/is_champion must be boolean"
            )
        if row["selected"]:
            selected_leaderboard.append(name)
        if row["is_champion"]:
            champion_leaderboard.append(name)
    if selected_leaderboard != ["markov"] or champion_leaderboard != ["markov"]:
        raise V5ContractError(
            "reviewed V5 leaderboard must select exactly one Markov champion"
        )

    diagnostics = _sequence(
        _require(model, "selection_diagnostics", "payload.model"),
        "payload.model.selection_diagnostics",
        nonempty=True,
    )
    selected_diagnostics: list[str] = []
    multiscale_rows = 0
    for index, raw in enumerate(diagnostics):
        context = f"payload.model.selection_diagnostics[{index}]"
        row = _mapping(raw, context)
        name = _require(row, "model", context)
        if not isinstance(name, str) or not name:
            raise V5ContractError(f"{context}.model must be non-empty")
        if not isinstance(row.get("selected"), bool) or not isinstance(
            row.get("gate_passed"), bool
        ):
            raise V5ContractError(f"{context}.selected/gate_passed must be boolean")
        if row["selected"]:
            if row["gate_passed"] is not True:
                raise V5ContractError(
                    f"{context} selected model must have passed its gate"
                )
            selected_diagnostics.append(name)
        if name == V5_MULTISCALE_MODEL:
            multiscale_rows += 1
            if row["selected"] is not False or row["gate_passed"] is not False:
                raise V5ContractError(
                    "reviewed V5 Multiscale diagnostic must remain non-promoted"
                )
    if selected_diagnostics != ["markov"] or multiscale_rows != 1:
        raise V5ContractError(
            "reviewed V5 selection diagnostics must select Markov and retain Multiscale"
        )


def _validate_conditional_stats(value: Any, *, expected_resamples: int) -> None:
    research = _mapping(value, "payload.research")
    expected_stats_fields = {
        "method",
        "role",
        "execution_lag_weeks",
        "horizons_weeks",
        "assets",
        "return_currency",
        "rows",
    }
    stats = _mapping(
        _require(research, "conditional_asset_stats", "payload.research"),
        "payload.research.conditional_asset_stats",
    )
    if set(stats) != expected_stats_fields:
        raise V5ContractError(
            "payload.research.conditional_asset_stats fields are invalid"
        )
    if stats.get("method") != "state_conditioned_forward_total_return":
        raise V5ContractError("conditional asset method is invalid")
    if stats.get("role") != "descriptive_only":
        raise V5ContractError("conditional asset role is invalid")
    if stats.get("execution_lag_weeks") != 1:
        raise V5ContractError("conditional asset execution lag must be one week")
    if tuple(stats.get("horizons_weeks", ())) != HORIZONS:
        raise V5ContractError("conditional asset horizons are invalid")
    if tuple(stats.get("assets", ())) != OUTCOME_ASSETS:
        raise V5ContractError("conditional asset list is invalid")
    if stats.get("return_currency") != "USD":
        raise V5ContractError("conditional asset return currency must be USD")
    rows = _sequence(_require(stats, "rows", "payload.research.conditional_asset_stats"), "payload.research.conditional_asset_stats.rows")
    forbidden = {"weight", "allocation", "position", "signal", "target_weight"}
    combinations: set[tuple[str, str, int]] = set()
    for index, raw in enumerate(rows):
        context = f"payload.research.conditional_asset_stats.rows[{index}]"
        row = _mapping(raw, context)
        if forbidden.intersection(row):
            raise V5ContractError(f"{context} contains an allocation field")
        if set(row) != set(CONDITIONAL_STATISTICS_COLUMNS):
            raise V5ContractError(f"{context} fields are invalid")
        if row.get("asset") not in OUTCOME_ASSETS or row.get("state") not in STATE_ORDER:
            raise V5ContractError(f"{context} asset/state is invalid")
        horizon = _integer(
            _require(row, "horizon_weeks", context),
            f"{context}.horizon_weeks",
            minimum=1,
        )
        if horizon not in HORIZONS:
            raise V5ContractError(f"{context}.horizon_weeks is invalid")
        combination = (
            str(row["asset"]),
            str(row["state"]),
            horizon,
        )
        if combination in combinations:
            raise V5ContractError(f"{context} duplicates an asset/state/horizon")
        combinations.add(combination)
        if row.get("execution_lag_weeks") != 1:
            raise V5ContractError(f"{context}.execution_lag_weeks is invalid")
        if row.get("return_currency") != "USD":
            raise V5ContractError(f"{context}.return_currency is invalid")
        if row.get("bootstrap_method") != "episode_bounded_circular_block":
            raise V5ContractError(f"{context}.bootstrap_method is invalid")
        if row.get("bootstrap_block_weeks") != 13:
            raise V5ContractError(f"{context}.bootstrap_block_weeks is invalid")
        if row.get("bootstrap_resamples") != expected_resamples:
            raise V5ContractError(f"{context}.bootstrap_resamples is inconsistent")
        if row.get("minimum_observations") != 20:
            raise V5ContractError(f"{context}.minimum_observations is invalid")
        if row.get("minimum_unique_episodes") != 5:
            raise V5ContractError(f"{context}.minimum_unique_episodes is invalid")
        _integer(_require(row, "n", context), f"{context}.n")
        _integer(_require(row, "unique_episodes", context), f"{context}.unique_episodes")
        for field in (
            "mean_return",
            "median_return",
            "positive_rate",
            "annualized_volatility",
            "downside_volatility",
            "cvar_5",
            "mean_max_drawdown",
        ):
            _optional_number(row.get(field), f"{context}.{field}")
        if row.get("status") not in {"ok", "insufficient_support"}:
            raise V5ContractError(f"{context}.status is invalid")
        for metric in (
            "mean_return",
            "median_return",
            "positive_rate",
            "annualized_volatility",
            "downside_volatility",
            "cvar_5",
            "mean_max_drawdown",
        ):
            lower = _optional_number(row.get(f"{metric}_ci95_lower"), f"{context}.{metric}_ci95_lower")
            upper = _optional_number(row.get(f"{metric}_ci95_upper"), f"{context}.{metric}_ci95_upper")
            if (lower is None) != (upper is None):
                raise V5ContractError(f"{context}.{metric} CI must share nullability")
            if lower is not None and upper < lower:
                raise V5ContractError(f"{context}.{metric} CI is reversed")
    expected = {
        (asset, state, horizon)
        for asset in OUTCOME_ASSETS
        for state in STATE_ORDER
        for horizon in HORIZONS
    }
    if combinations != expected:
        raise V5ContractError(
            "conditional asset rows must cover every asset/state/horizon"
        )


def _validate_model_conditioned_stats(
    value: Any,
    *,
    expected_models: tuple[str, ...] | None,
    expected_resamples: int,
    model: Mapping[str, Any],
) -> None:
    research = _mapping(value, "payload.research")
    raw = research.get("model_conditioned_asset_stats")
    artifacts = _mapping(
        _require(model, "research_artifacts", "payload.model"),
        "payload.model.research_artifacts",
    )
    artifact_keys = set(artifacts)
    if raw is None:
        if artifact_keys & OPTIONAL_RESEARCH_ARTIFACT_KEYS:
            raise V5ContractError(
                "model-conditioned artifacts require public derived statistics"
            )
        return
    if expected_models is None:
        raise V5ContractError(
            "model-conditioned statistics require forecast comparison metadata"
        )
    if not OPTIONAL_RESEARCH_ARTIFACT_KEYS.issubset(artifact_keys):
        raise V5ContractError(
            "model-conditioned statistics require their complete artifact pair"
        )

    context = "payload.research.model_conditioned_asset_stats"
    stats = _mapping(raw, context)
    expected_stats_fields = {
        "method",
        "role",
        "conditioning",
        "forecast_horizon_weeks",
        "execution_lag_weeks",
        "horizons_weeks",
        "assets",
        "models",
        "return_currency",
        "rows",
    }
    if set(stats) != expected_stats_fields:
        raise V5ContractError(f"{context} fields are invalid")
    if stats.get("method") != "oos_one_week_forecast_conditioned_forward_total_return":
        raise V5ContractError(f"{context}.method is invalid")
    if stats.get("role") != "retrospective_model_diagnostic":
        raise V5ContractError(f"{context}.role is invalid")
    if stats.get("conditioning") != "hard_argmax_oos_forecast":
        raise V5ContractError(f"{context}.conditioning is invalid")
    if stats.get("forecast_horizon_weeks") != 1:
        raise V5ContractError(f"{context}.forecast_horizon_weeks must be one")
    if stats.get("execution_lag_weeks") != 1:
        raise V5ContractError(f"{context}.execution_lag_weeks must be one")
    if tuple(stats.get("horizons_weeks", ())) != HORIZONS:
        raise V5ContractError(f"{context}.horizons_weeks is invalid")
    if tuple(stats.get("assets", ())) != OUTCOME_ASSETS:
        raise V5ContractError(f"{context}.assets is invalid")
    if tuple(stats.get("models", ())) != expected_models:
        raise V5ContractError(f"{context}.models is invalid")
    if stats.get("return_currency") != "USD":
        raise V5ContractError(f"{context}.return_currency is invalid")

    rows = _sequence(_require(stats, "rows", context), f"{context}.rows")
    expected_rows = len(expected_models) * len(OUTCOME_ASSETS) * len(STATE_ORDER) * len(HORIZONS)
    if len(rows) != expected_rows:
        raise V5ContractError(f"{context}.rows has invalid coverage")
    for name in expected_models:
        model_rows: list[dict[str, Any]] = []
        for index, raw_row in enumerate(rows):
            row_context = f"{context}.rows[{index}]"
            row = _mapping(raw_row, row_context)
            if set(row) != set(MODEL_CONDITIONED_STATISTICS_COLUMNS):
                raise V5ContractError(f"{row_context} fields are invalid")
            if row.get("conditioning_model") != name:
                continue
            model_rows.append(
                {
                    str(key): item
                    for key, item in row.items()
                    if key != "conditioning_model"
                }
            )
        canonical = {
            "method": "state_conditioned_forward_total_return",
            "role": "descriptive_only",
            "execution_lag_weeks": 1,
            "horizons_weeks": list(HORIZONS),
            "assets": list(OUTCOME_ASSETS),
            "return_currency": "USD",
            "rows": model_rows,
        }
        _validate_conditional_stats(
            {"conditional_asset_stats": canonical},
            expected_resamples=expected_resamples,
        )

    statistics_count = _integer(
        artifacts["model_conditioned_asset_statistics"]["row_count"],
        "payload.model.research_artifacts.model_conditioned_asset_statistics.row_count",
        minimum=1,
    )
    if statistics_count != expected_rows:
        raise V5ContractError(
            "model-conditioned asset statistics artifact row count is inconsistent"
        )


def validate_v5_payload(payload: Mapping[str, Any]) -> None:
    payload = _mapping(payload, "payload")
    if set(payload) != V5_PAYLOAD_FIELDS:
        raise V5ContractError("payload fields are invalid")
    meta = _mapping(_require(payload, "meta", "payload"), "payload.meta")
    if meta.get("schema_version") != V5_SCHEMA_VERSION:
        raise V5ContractError(f"payload.meta.schema_version must be {V5_SCHEMA_VERSION}")
    if meta.get("result_version") != V5_RESULT_VERSION:
        raise V5ContractError(f"payload.meta.result_version must be {V5_RESULT_VERSION}")
    if meta.get("timezone") != "America/New_York":
        raise V5ContractError("payload.meta.timezone is invalid")
    mode = _require(meta, "mode", "payload.meta")
    if not isinstance(mode, str) or mode not in {"demo", "live"}:
        raise V5ContractError("payload.meta.mode must be demo or live")
    warnings = _sequence(
        _require(meta, "warnings", "payload.meta"),
        "payload.meta.warnings",
    )
    if any(not isinstance(warning, str) for warning in warnings):
        raise V5ContractError("payload.meta.warnings must contain only strings")
    generated_at = _iso_datetime(
        _require(meta, "generated_at", "payload.meta"),
        "payload.meta.generated_at",
    )
    data_as_of = _iso_datetime(
        _require(meta, "data_as_of", "payload.meta"),
        "payload.meta.data_as_of",
    )
    if not isinstance(meta.get("generation_id"), str) or not meta["generation_id"]:
        raise V5ContractError("payload.meta.generation_id must be non-empty")
    freshness = _mapping(_require(meta, "freshness", "payload.meta"), "payload.meta.freshness")
    if set(freshness) != {
        "cadence",
        "maximum_age_days",
        "age_days",
        "status",
        "data_as_of",
    }:
        raise V5ContractError("payload.meta.freshness fields are invalid")
    if freshness.get("cadence") != "weekly":
        raise V5ContractError("payload.meta.freshness.cadence is invalid")
    maximum_age = _integer(
        _require(freshness, "maximum_age_days", "payload.meta.freshness"),
        "payload.meta.freshness.maximum_age_days",
        minimum=1,
    )
    if maximum_age != 10:
        raise V5ContractError("payload.meta.freshness.maximum_age_days must be 10")
    expected_age = max(
        0,
        int((generated_at - data_as_of).total_seconds() // 86_400),
    )
    age_days = _integer(
        _require(freshness, "age_days", "payload.meta.freshness"),
        "payload.meta.freshness.age_days",
    )
    if age_days != expected_age:
        raise V5ContractError("payload.meta.freshness.age_days is inconsistent")
    expected_status = "current" if age_days <= maximum_age else "stale"
    if freshness.get("status") != expected_status:
        raise V5ContractError("payload.meta.freshness.status is inconsistent")
    freshness_cutoff = _iso_datetime(
        _require(freshness, "data_as_of", "payload.meta.freshness"),
        "payload.meta.freshness.data_as_of",
    )
    if freshness_cutoff != data_as_of:
        raise V5ContractError("payload.meta.freshness.data_as_of is inconsistent")
    expected_meta_fields = (
        V5_PUBLICATION_META_FIELDS
        if "publication_status" in meta or "publication_review" in meta
        else V5_META_FIELDS
    )
    if set(meta) != expected_meta_fields:
        raise V5ContractError("payload.meta fields are invalid")

    states = _sequence(_require(payload, "states", "payload"), "payload.states")
    if tuple(item.get("id") if isinstance(item, Mapping) else None for item in states) != STATE_ORDER:
        raise V5ContractError(f"payload.states must be ordered as {STATE_ORDER}")
    model = _mapping(_require(payload, "model", "payload"), "payload.model")
    evidence_week_count = _validate_model(model, mode=str(mode))
    forecast_comparison_models = _validate_forecast_comparison(model)
    _validate_publication_review(meta, model, mode=str(mode))

    weekly = _sequence(_require(payload, "weekly", "payload"), "payload.weekly", nonempty=True)
    previous: date | None = None
    for index, raw in enumerate(weekly):
        context = f"payload.weekly[{index}]"
        row = _mapping(raw, context)
        origin = _iso_date(_require(row, "date", context), f"{context}.date")
        if previous is not None and origin <= previous:
            raise V5ContractError("payload.weekly dates must be strictly increasing")
        previous = origin
        current_state = _validate_current(_require(row, "current", context), f"{context}.current")
        forecast_date = _validate_forecast(
            _require(row, "next_week", context), f"{context}.next_week"
        )
        if (forecast_date - origin).days != 7:
            raise V5ContractError(f"{context}.next_week.date is inconsistent")
        if forecast_comparison_models is None:
            if "model_forecasts" in row:
                raise V5ContractError(
                    f"{context}.model_forecasts requires model forecast_comparison metadata"
                )
        else:
            _validate_model_forecasts(
                _require(row, "model_forecasts", context),
                f"{context}.model_forecasts",
                origin=origin,
                models=forecast_comparison_models,
                champion=str(model["champion"]),
                official=_mapping(row["next_week"], f"{context}.next_week"),
            )
        departure = _validate_transition_risk(
            _require(row, "transition_risk", context),
            f"{context}.transition_risk",
            origin=origin,
        )
        transition_alias = _number(
            _require(row, "transition_probability", context),
            f"{context}.transition_probability",
            minimum=0.0,
            maximum=1.0,
        )
        if not math.isclose(transition_alias, departure[1], abs_tol=1e-8, rel_tol=0.0):
            raise V5ContractError(f"{context}.transition_probability is inconsistent")
        _validate_directional_risk(
            _require(row, "directional_risk", context),
            f"{context}.directional_risk",
            current_state=current_state,
            departure=departure,
            origin=origin,
        )
        _validate_duration(_require(row, "duration_context", context), f"{context}.duration_context", current_state)
        _validate_fx_context(_require(row, "fx_context", context), f"{context}.fx_context")
        scores = _mapping(_require(row, "context_scores", context), f"{context}.context_scores")
        if set(scores) != {"trend", "stress", "macro", "financial_conditions"}:
            raise V5ContractError(f"{context}.context_scores keys are invalid")
        for name, value in scores.items():
            _number(value, f"{context}.context_scores.{name}", minimum=-1.0, maximum=1.0)
        extremes = _sequence(_require(row, "extreme_context", context), f"{context}.extreme_context")
        for position, raw_extreme in enumerate(extremes):
            extreme_context = f"{context}.extreme_context[{position}]"
            extreme = _mapping(raw_extreme, extreme_context)
            if {"impact", "direction"}.intersection(extreme):
                raise V5ContractError(f"{extreme_context} contains attribution semantics")
            if set(extreme) != {"feature", "label", "z_score", "position", "method"}:
                raise V5ContractError(f"{extreme_context} fields are invalid")
            _number(extreme["z_score"], f"{extreme_context}.z_score")
            if extreme["position"] not in {"high", "low"}:
                raise V5ContractError(f"{extreme_context}.position is invalid")
        if not isinstance(_require(row, "summary", context), str) or not row["summary"].strip():
            raise V5ContractError(f"{context}.summary must be non-empty")
        _mapping(_require(row, "market", context), f"{context}.market")
        _mapping(_require(row, "health", context), f"{context}.health")
        if "scores" in row or "top_drivers" in row:
            raise V5ContractError(f"{context} contains removed v4 semantic fields")

    if evidence_week_count != len(weekly):
        raise V5ContractError(
            "payload.model evidence forecast row count must equal payload.weekly"
        )

    sources = _sequence(_require(payload, "sources", "payload"), "payload.sources")
    sources_by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw_source in enumerate(sources):
        source = _mapping(raw_source, f"payload.sources[{index}]")
        source_id = _require(source, "id", f"payload.sources[{index}]")
        if not isinstance(source_id, str):
            raise V5ContractError(f"payload.sources[{index}].id must be a string")
        if source_id in sources_by_id:
            raise V5ContractError(f"payload.sources[{index}].id is duplicated")
        sources_by_id[source_id] = source
        if source.get("status") not in {
            "ok",
            "stale",
            "degraded",
            "quota_exhausted",
            "schema_changed",
            "revision_gap",
            "rights_unconfirmed",
            "license_blocked",
            "unavailable",
        }:
            raise V5ContractError(f"payload.sources[{index}].status is invalid")
    expected_source_licenses = (
        DEMO_SOURCE_LICENSES if mode == "demo" else LIVE_SOURCE_LICENSES
    )
    if set(sources_by_id) != set(expected_source_licenses):
        raise V5ContractError(
            f"payload.sources identities are invalid for mode={mode}"
        )
    for source_id, expected_license in expected_source_licenses.items():
        if sources_by_id[source_id].get("license_class") != expected_license:
            raise V5ContractError(
                f"payload.sources[{source_id}].license_class is invalid"
            )
    if mode == "live":
        h10 = sources_by_id["frb_h10"]
        ablation = _mapping(
            _mapping(payload["model"], "payload.model")["fx_ablation"],
            "payload.model.fx_ablation",
        )
        for field in (
            "official_release_archive_ingest",
            "availability_basis",
            "archive_revision_policy",
            "archive_correction_availability_basis",
        ):
            if field not in h10 or h10[field] != ablation[field]:
                raise V5ContractError(
                    f"payload.sources[frb_h10].{field} must match model FX provenance"
                )
        release_count = _integer(
            _require(
                h10,
                "archive_release_count",
                "payload.sources[frb_h10]",
            ),
            "payload.sources[frb_h10].archive_release_count",
        )
        correction_count = _integer(
            _require(
                h10,
                "archive_correction_count",
                "payload.sources[frb_h10]",
            ),
            "payload.sources[frb_h10].archive_correction_count",
        )
        correction_values = _sequence(
            _require(
                h10,
                "archive_correction_available_at",
                "payload.sources[frb_h10]",
            ),
            "payload.sources[frb_h10].archive_correction_available_at",
        )
        corrections = [
            _iso_datetime(
                value,
                f"payload.sources[frb_h10].archive_correction_available_at[{index}]",
            )
            for index, value in enumerate(correction_values)
        ]
        if any(value.utcoffset().total_seconds() != 0 for value in corrections):
            raise V5ContractError(
                "payload.sources[frb_h10].archive_correction_available_at must be UTC"
            )
        if corrections != sorted(set(corrections)):
            raise V5ContractError(
                "payload.sources[frb_h10].archive_correction_available_at "
                "must be unique/increasing"
            )
        if release_count < correction_count or correction_count != len(corrections):
            raise V5ContractError(
                "payload.sources[frb_h10] archive release/correction counts mismatch"
            )
        if h10.get("archive_correction_quarantine_weeks") != 27:
            raise V5ContractError(
                "payload.sources[frb_h10].archive_correction_quarantine_weeks is invalid"
            )
        if h10.get("archive_evaluation_start") != "2022-01-01":
            raise V5ContractError(
                "payload.sources[frb_h10].archive_evaluation_start is invalid"
            )
        if (
            h10.get("archive_evaluation_start_rationale")
            != "post_2019_06_24_jan06_index_rebase_common_scale"
        ):
            raise V5ContractError(
                "payload.sources[frb_h10].archive_evaluation_start_rationale is invalid"
            )
        if bool(ablation["official_release_archive_ingest"]):
            if release_count == 0:
                raise V5ContractError(
                    "payload.sources[frb_h10] archive ingest has no releases"
                )
        elif release_count != 0 or correction_count != 0 or corrections:
            raise V5ContractError(
                "payload.sources[frb_h10] first-seen fallback has archive events"
            )
    catalog = _sequence(
        _require(payload, "feature_catalog", "payload"),
        "payload.feature_catalog",
        nonempty=True,
    )
    for index, raw_feature in enumerate(catalog):
        feature = _mapping(raw_feature, f"payload.feature_catalog[{index}]")
        for field in ("id", "category", "frequency", "source"):
            if not isinstance(
                _require(feature, field, f"payload.feature_catalog[{index}]"), str
            ):
                raise V5ContractError(
                    f"payload.feature_catalog[{index}].{field} must be a string"
                )
    expected_outcome_resamples = int(
        _mapping(payload["model"], "payload.model")["execution_parameters"][
            "conditional_outcome_bootstrap_resamples"
        ]
    )
    _validate_conditional_stats(
        _require(payload, "research", "payload"),
        expected_resamples=expected_outcome_resamples,
    )
    _validate_model_conditioned_stats(
        _require(payload, "research", "payload"),
        expected_models=forecast_comparison_models,
        expected_resamples=expected_outcome_resamples,
        model=model,
    )


__all__ = [
    "OUTCOME_ASSETS",
    "STATE_ORDER",
    "V5ContractError",
    "V5_FEATURE_SET_VERSION",
    "V5_FORECAST_COMPARISON_MODELS",
    "V5_LABEL_VERSION",
    "V5_MODEL_VERSION",
    "V5_PUBLICATION_REVIEW_SCHEMA",
    "V5_PUBLICATION_STATUS",
    "V5_RESULT_VERSION",
    "V5_SCHEMA_VERSION",
    "validate_v5_payload",
]
