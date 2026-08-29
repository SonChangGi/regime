"""Fail-closed public contract for the opt-in v5 research payload."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta
import hashlib
import json
import math
from typing import Any
from zoneinfo import ZoneInfo

from regime_lab.analysis.decision_shadow import load_decision_shadow_spec
from regime_lab.analysis.label_spec import load_label_spec
from regime_lab.frozen_v4 import FROZEN_V4_BASELINE
from regime_lab.integrity import (
    IntegrityError,
    validate_lifecycle_consistency,
    validate_reviewed_candidate_hash,
)
from regime_lab.operating_contract import load_operating_contract
from regime_lab.v5_artifacts import (
    CONDITIONAL_STATISTICS_COLUMNS,
    FX_RESEARCH_ARTIFACT_KEYS,
    MODEL_CONDITIONED_STATISTICS_COLUMNS,
    OPTIONAL_RESEARCH_ARTIFACT_KEYS,
    REQUIRED_RESEARCH_ARTIFACT_KEYS,
    V5_CORE_ARTIFACT_PATHS,
    V5_RESEARCH_ARTIFACTS,
)


V5_SCHEMA_VERSION = "2.1.0"
V5_RESULT_VERSION = "weekly-regime-result-v5"
V5_MODEL_VERSION = "weekly-nondl-structural-v5"
V5_LABEL_VERSION = "market-causal-3state-v1"
V5_FEATURE_SET_VERSION = "weekly-pit-structural-v5"
V5_PUBLICATION_STATUS = "reviewed_publication"
V5_PUBLICATION_REVIEW_SCHEMA = "regime-v5-publication-review/1"
V5_PAYLOAD_FIELDS = frozenset(
    {
        "meta",
        "states",
        "label",
        "forecast",
        "selection",
        "model",
        "weekly",
        "sources",
        "feature_catalog",
        "research",
    }
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
        "publication_status",
    }
)
V5_PUBLICATION_META_FIELDS = frozenset(
    {*V5_META_FIELDS, "publication_review"}
)
V5_MANIFEST_META_FIELDS = frozenset(
    {*V5_META_FIELDS, "generation_manifest_sha256"}
)
V5_PUBLICATION_MANIFEST_META_FIELDS = frozenset(
    {*V5_PUBLICATION_META_FIELDS, "generation_manifest_sha256"}
)
STATE_ORDER = ("risk_on", "transition", "risk_off")
HORIZONS = (1, 4, 13)
OUTCOME_ASSETS = ("SPY", "QQQ", "IWM", "TLT", "HYG", "UUP")
LEGACY_ENHANCED_CONDITIONAL_STATISTICS_FIELDS = frozenset(
    {
        "unconditional_benchmark_method",
        "unconditional_benchmark_n",
        "unconditional_benchmark_mean_return",
        "excess_mean_return",
        "episode_equal_mean_return",
        "episode_equal_excess_return",
        "episode_bootstrap_method",
        "episode_bootstrap_resamples",
        "episode_bootstrap_seed",
        "episode_equal_mean_return_ci95_lower",
        "episode_equal_mean_return_ci95_upper",
    }
)
MATCHED_EPISODE_BENCHMARK_FIELDS = frozenset(
    {
        "episode_equal_unconditional_benchmark_method",
        "episode_equal_unconditional_benchmark_episode_n",
        "episode_equal_unconditional_benchmark_mean_return",
    }
)
ENHANCED_CONDITIONAL_STATISTICS_FIELDS = frozenset(
    {
        *LEGACY_ENHANCED_CONDITIONAL_STATISTICS_FIELDS,
        *MATCHED_EPISODE_BENCHMARK_FIELDS,
    }
)
FX_VARIANTS = (
    "v4_control",
    "v4_plus_broad_index",
    "v4_plus_bilateral_panel",
    "v4_plus_all_fx",
)
V5_MULTISCALE_MODEL = "causal_multiscale_ensemble"
V5_MINIMUM_PROMOTION_LOG_LOSS_IMPROVEMENT = 0.01
V5_MAXIMUM_PROMOTION_BRIER_DEGRADATION = 0.01
V5_MAXIMUM_PROMOTION_ALPHA = 0.05
# The only reviewed V5 publication produced before the 0.01 promotion policy
# took effect.  The digest is over the decoded payload serialized as canonical
# JSON, so whitespace and object-key order do not affect the identity while any
# semantic change does.
V5_LEGACY_REVIEWED_005_SNAPSHOT_SHA256 = (
    "3fb67126917d6f0ec178de01d9aebc7698476921d1b4d183b2dd0da50e992704"
)
V5_FORECAST_COMPARISON_MODELS = (
    "majority",
    "persistence",
    "markov",
    "xgboost",
    "pca_ridge_logistic",
    "recency_weighted_xgboost_208w",
    "recency_weighted_ridge_logistic_208w",
    "discounted_markov_208w",
    "xgb_hazard_destination",
    "causal_dynamic_ensemble",
    V5_MULTISCALE_MODEL,
)
_OPERATING_CONTRACT = load_operating_contract()
V5_STANDARD_CORE_MODELS = frozenset(
    {
        *_OPERATING_CONTRACT.weekly_base_models,
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


def _observed_weekday_holiday(value: date) -> date:
    if value.weekday() == 5:
        return value - timedelta(days=1)
    if value.weekday() == 6:
        return value + timedelta(days=1)
    return value


def _gregorian_easter(year: int) -> date:
    """Return Gregorian Easter Sunday using the Meeus/Jones/Butcher rule."""

    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = (h + ell - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def _nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (occurrence - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    last = next_month - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _is_nyse_session(value: date) -> bool:
    if value.weekday() >= 5:
        return False
    year = value.year
    holidays = {
        _observed_weekday_holiday(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _gregorian_easter(year) - timedelta(days=2),
        _last_weekday(year, 5, 0),
        _observed_weekday_holiday(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        _observed_weekday_holiday(date(year, 12, 25)),
    }
    if year >= 2022:
        holidays.add(_observed_weekday_holiday(date(year, 6, 19)))
    # Unscheduled full-day closures that can change a week's first session.
    holidays.update(
        {
            date(2007, 1, 2),
            date(2012, 10, 29),
            date(2012, 10, 30),
        }
    )
    return value not in holidays


def _first_nyse_session_of_week(target_week: date) -> date:
    week_start = target_week - timedelta(days=target_week.weekday())
    for offset in range(7):
        candidate = week_start + timedelta(days=offset)
        if _is_nyse_session(candidate):
            return candidate
    raise V5ContractError("target week has no NYSE trading session")


def _validate_label_contract(value: Any) -> None:
    context = "payload.label"
    label = _mapping(value, context)
    expected = {
        "spec_id",
        "spec_version",
        "spec_sha256",
        "fit_period",
        "input_scope",
        "membership_semantics",
    }
    if set(label) != expected:
        raise V5ContractError(f"{context} fields are invalid")
    specification = load_label_spec()
    if label.get("spec_id") != specification.spec_id:
        raise V5ContractError(f"{context}.spec_id is not the frozen canonical label")
    if label.get("spec_version") != V5_LABEL_VERSION:
        raise V5ContractError(f"{context}.spec_version is invalid")
    _sha256(label.get("spec_sha256"), f"{context}.spec_sha256")
    if label.get("spec_sha256") != specification.spec_sha256:
        raise V5ContractError(f"{context}.spec_sha256 differs from label-spec.json")
    if label.get("input_scope") != "SPY adjusted close only":
        raise V5ContractError(f"{context}.input_scope is invalid")
    if label.get("membership_semantics") != "distance_to_anchor_not_posterior":
        raise V5ContractError(f"{context}.membership_semantics is invalid")
    fit = _mapping(_require(label, "fit_period", context), f"{context}.fit_period")
    if set(fit) != {"start", "end", "weeks"}:
        raise V5ContractError(f"{context}.fit_period fields are invalid")
    start = _iso_date(fit["start"], f"{context}.fit_period.start")
    end = _iso_date(fit["end"], f"{context}.fit_period.end")
    if end < start or _integer(fit["weeks"], f"{context}.fit_period.weeks", minimum=1) != 520:
        raise V5ContractError(f"{context}.fit_period is invalid")


def _validate_prospective_performance(
    value: Any,
    *,
    context: str,
    ledger_status: str,
    realized_count: int,
) -> None:
    performance = _mapping(value, context)
    expected = {
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
    if set(performance) != expected:
        raise V5ContractError(f"{context} fields are invalid")
    weeks = _integer(performance.get("weeks"), f"{context}.weeks")
    if weeks != realized_count:
        raise V5ContractError(f"{context}.weeks differs from realized count")
    expected_status = (
        "completed" if ledger_status == "completed" else
        "pending" if ledger_status in {"empty", "pending"} else "partial"
    )
    if performance.get("status") != expected_status:
        raise V5ContractError(f"{context}.status is inconsistent")
    metric_names = expected - {"status", "weeks"}
    if weeks == 0:
        if any(performance.get(field) is not None for field in metric_names):
            raise V5ContractError(f"{context} empty metrics must be null")
        return
    gross = _number(
        performance.get("gross_cumulative_return"),
        f"{context}.gross_cumulative_return",
        minimum=-1.0,
    )
    net = _number(
        performance.get("net_cumulative_return"),
        f"{context}.net_cumulative_return",
        minimum=-1.0,
    )
    if net > gross + 1e-12:
        raise V5ContractError(f"{context} net return exceeds gross return")
    turnover = _number(
        performance.get("turnover_sum"), f"{context}.turnover_sum", minimum=0.0
    )
    cost = _number(
        performance.get("transaction_cost_rate_sum"),
        f"{context}.transaction_cost_rate_sum",
        minimum=0.0,
    )
    cost_bps = _number(
        performance.get("transaction_cost_bps"),
        f"{context}.transaction_cost_bps",
        minimum=0.0,
    )
    if not math.isclose(cost_bps, 10.0, abs_tol=1e-12) or not math.isclose(
        cost, turnover * cost_bps / 10_000.0, abs_tol=1e-12
    ):
        raise V5ContractError(f"{context} transaction cost is inconsistent")
    hits = _integer(
        performance.get("forecast_hit_count"),
        f"{context}.forecast_hit_count",
    )
    if hits > weeks:
        raise V5ContractError(f"{context}.forecast_hit_count is invalid")
    accuracy = _number(
        performance.get("forecast_accuracy"),
        f"{context}.forecast_accuracy",
        minimum=0.0,
        maximum=1.0,
    )
    if not math.isclose(accuracy, hits / weeks, abs_tol=1e-12):
        raise V5ContractError(f"{context}.forecast_accuracy is inconsistent")
    counts = _mapping(
        performance.get("actual_state_counts"), f"{context}.actual_state_counts"
    )
    if set(counts) != set(STATE_ORDER):
        raise V5ContractError(f"{context}.actual_state_counts fields are invalid")
    total = sum(
        _integer(counts.get(state), f"{context}.actual_state_counts.{state}")
        for state in STATE_ORDER
    )
    if total != weeks:
        raise V5ContractError(f"{context}.actual_state_counts is inconsistent")


def _validate_prospective_ledger_v2(value: Any, *, context: str) -> dict[str, Any]:
    ledger = _mapping(value, context)
    expected = {
        "schema_version",
        "status",
        "entry_count",
        "pending_evaluation_count",
        "unresolved_due_evaluation_count",
        "realized_evaluation_count",
        "partial_evaluation_count",
        "key_manifest_sha256",
        "evaluation_manifest_sha256",
        "hash_scope",
        "evaluation_hash_scope",
        "performance",
    }
    if set(ledger) != expected:
        raise V5ContractError(f"{context} fields are invalid")
    if (
        ledger.get("schema_version") != "regime-prospective-ledger-summary/2"
        or ledger.get("hash_scope") != "ordered_ledger_primary_keys_only"
        or ledger.get("evaluation_hash_scope")
        != "ordered_forecast_primary_keys_status_and_evaluation_sha256"
    ):
        raise V5ContractError(f"{context} identity is invalid")
    _sha256(ledger.get("key_manifest_sha256"), f"{context}.key_manifest_sha256")
    _sha256(
        ledger.get("evaluation_manifest_sha256"),
        f"{context}.evaluation_manifest_sha256",
    )
    count_fields = (
        "entry_count",
        "pending_evaluation_count",
        "unresolved_due_evaluation_count",
        "realized_evaluation_count",
        "partial_evaluation_count",
    )
    counts = {
        field: _integer(ledger.get(field), f"{context}.{field}")
        for field in count_fields
    }
    if sum(counts[field] for field in count_fields[1:]) != counts["entry_count"]:
        raise V5ContractError(f"{context} counts are inconsistent")
    status = str(ledger.get("status", ""))
    expected_status = (
        "empty"
        if counts["entry_count"] == 0
        else "completed"
        if counts["realized_evaluation_count"] == counts["entry_count"]
        else "pending"
        if counts["pending_evaluation_count"] == counts["entry_count"]
        else "partial"
    )
    if status != expected_status:
        raise V5ContractError(f"{context}.status is inconsistent")
    _validate_prospective_performance(
        ledger.get("performance"),
        context=f"{context}.performance",
        ledger_status=status,
        realized_count=counts["realized_evaluation_count"],
    )
    return ledger


def _validate_forecast_contract(value: Any, *, mode: str) -> tuple[datetime, datetime]:
    context = "payload.forecast"
    forecast = _mapping(value, context)
    legacy_expected = {
        "status",
        "origin_at",
        "decision_at",
        "target_at",
        "remaining_horizon",
        "evidence_track",
    }
    enhanced_expected = {
        *legacy_expected,
        "forecast_evidence_track",
        "issue_latency_seconds",
        "scheduled_horizon_seconds",
        "remaining_horizon_fraction",
        "minimum_full_horizon_remaining_fraction",
        "timing_status",
        "prospective_ledger",
    }
    enhanced = set(forecast) == enhanced_expected
    if set(forecast) not in (legacy_expected, enhanced_expected):
        raise V5ContractError(f"{context} fields are invalid")
    status = forecast.get("status")
    if status not in {"active", "expired"}:
        raise V5ContractError(f"{context}.status is invalid")
    if forecast.get("evidence_track") not in {"operational_oos", "reconstructed_oos"}:
        raise V5ContractError(f"{context}.evidence_track is invalid")
    origin = _iso_datetime(forecast.get("origin_at"), f"{context}.origin_at")
    target = _iso_datetime(forecast.get("target_at"), f"{context}.target_at")
    scheduled_seconds = int((target - origin).total_seconds())
    if target <= origin or scheduled_seconds not in {
        7 * 86_400 - 3_600,
        7 * 86_400,
        7 * 86_400 + 3_600,
    }:
        raise V5ContractError(f"{context} origin/target horizon is invalid")
    remaining = _integer(
        forecast.get("remaining_horizon"),
        f"{context}.remaining_horizon",
    )
    if status == "active":
        decision = _iso_datetime(
            forecast.get("decision_at"), f"{context}.decision_at"
        )
        if not origin <= decision < target:
            raise V5ContractError(
                f"{context} requires origin_at <= decision_at < target_at"
            )
        expected_remaining = int((target - decision).total_seconds())
        if remaining != expected_remaining or remaining <= 0:
            raise V5ContractError(f"{context}.remaining_horizon is inconsistent")
    else:
        if forecast.get("decision_at") is not None or remaining != 0:
            raise V5ContractError(f"{context} expired state is inconsistent")
        if mode == "live":
            raise V5ContractError("live payload cannot expose an expired current forecast")
    if enhanced:
        if forecast.get("forecast_evidence_track") != forecast.get("evidence_track"):
            raise V5ContractError(
                f"{context}.forecast_evidence_track differs from legacy alias"
            )
        if forecast.get("scheduled_horizon_seconds") != scheduled_seconds:
            raise V5ContractError(f"{context}.scheduled_horizon_seconds is invalid")
        minimum_fraction = _number(
            forecast.get("minimum_full_horizon_remaining_fraction"),
            f"{context}.minimum_full_horizon_remaining_fraction",
            minimum=0.0,
            maximum=1.0,
        )
        if not math.isclose(minimum_fraction, 4.0 / 7.0, abs_tol=1e-12):
            raise V5ContractError(f"{context} full-horizon threshold is invalid")
        fraction = _number(
            forecast.get("remaining_horizon_fraction"),
            f"{context}.remaining_horizon_fraction",
            minimum=0.0,
            maximum=1.0,
        )
        expected_fraction = remaining / scheduled_seconds if status == "active" else 0.0
        if not math.isclose(fraction, expected_fraction, abs_tol=1e-8):
            raise V5ContractError(f"{context}.remaining_horizon_fraction is inconsistent")
        expected_timing = (
            "expired"
            if status == "expired"
            else (
                "full_horizon_forecast"
                if fraction + 1e-12 >= minimum_fraction
                else "late_nowcast"
            )
        )
        if forecast.get("timing_status") != expected_timing:
            raise V5ContractError(f"{context}.timing_status is inconsistent")
        latency = forecast.get("issue_latency_seconds")
        if status == "active":
            if latency != int((decision - origin).total_seconds()):
                raise V5ContractError(f"{context}.issue_latency_seconds is inconsistent")
        elif latency is not None:
            raise V5ContractError(f"{context}.issue_latency_seconds must be null")
        ledger = _mapping(
            forecast.get("prospective_ledger"), f"{context}.prospective_ledger"
        )
        if ledger.get("schema_version") == "regime-prospective-ledger-summary/2":
            _validate_prospective_ledger_v2(
                ledger, context=f"{context}.prospective_ledger"
            )
        else:
            if set(ledger) != {
                "schema_version",
                "status",
                "entry_count",
                "key_manifest_sha256",
                "hash_scope",
            }:
                raise V5ContractError(
                    f"{context}.prospective_ledger fields are invalid"
                )
            if ledger.get("schema_version") != "regime-prospective-ledger-summary/1":
                raise V5ContractError(f"{context}.prospective_ledger schema is invalid")
            if ledger.get("hash_scope") != "ordered_ledger_primary_keys_only":
                raise V5ContractError(
                    f"{context}.prospective_ledger hash scope is invalid"
                )
            if ledger.get("status") == "not_applicable":
                if ledger.get("entry_count") != 0:
                    raise V5ContractError(
                        f"{context}.prospective_ledger count is invalid"
                    )
                _sha256(
                    ledger.get("key_manifest_sha256"),
                    f"{context}.prospective_ledger.key_manifest_sha256",
                )
            elif ledger.get("status") == "pending_append":
                if (
                    ledger.get("entry_count") is not None
                    or ledger.get("key_manifest_sha256") is not None
                ):
                    raise V5ContractError(
                        f"{context}.prospective_ledger pending state is invalid"
                    )
            elif ledger.get("status") == "recorded":
                _integer(
                    ledger.get("entry_count"),
                    f"{context}.prospective_ledger.entry_count",
                    minimum=1,
                )
                _sha256(
                    ledger.get("key_manifest_sha256"),
                    f"{context}.prospective_ledger.key_manifest_sha256",
                )
            else:
                raise V5ContractError(
                    f"{context}.prospective_ledger status is invalid"
                )
    return origin, target


def _validate_selection_contract(value: Any, model: Mapping[str, Any]) -> None:
    context = "payload.selection"
    selection = _mapping(value, context)
    legacy_expected = {
        "schema_version",
        "status",
        "policy_sha256",
        "complexity_registry_sha256",
        "candidate_set",
        "runner_up",
        "selection_reason",
        "simplicity_tolerance",
        "tie_break_order",
        "operating_champion",
    }
    enhanced_expected = {
        *legacy_expected,
        "selection_evidence_track",
        "evidence_status",
    }
    decision_grade_expected = {
        *enhanced_expected,
        "selected_champion",
        "statistically_indistinguishable_models",
        "statistical_equivalence_status",
        "release_epoch_registry",
        "multiplicity_defense",
    }
    enhanced = set(selection) in (enhanced_expected, decision_grade_expected)
    decision_grade = set(selection) == decision_grade_expected
    if set(selection) not in (
        legacy_expected,
        enhanced_expected,
        decision_grade_expected,
    ):
        raise V5ContractError(f"{context} fields are invalid")
    if selection.get("schema_version") != "regime-selection-evidence/1":
        raise V5ContractError(f"{context}.schema_version is invalid")
    if selection.get("status") != "selected_by_gate":
        raise V5ContractError(f"{context}.status is invalid")
    if selection.get("status") != model.get("selection_status"):
        raise V5ContractError(f"{context}.status differs from model alias")
    if enhanced and (
        selection.get("selection_evidence_track") != "reconstructed_oos"
        or selection.get("evidence_status") != "historical_reconstructed_oos"
    ):
        raise V5ContractError(f"{context} historical evidence identity is invalid")
    if decision_grade:
        if selection.get("selected_champion") != model.get("champion"):
            raise V5ContractError(f"{context}.selected_champion is inconsistent")
        indistinguishable = list(
            _sequence(
                selection.get("statistically_indistinguishable_models"),
                f"{context}.statistically_indistinguishable_models",
            )
        )
        if any(name not in selection.get("candidate_set", ()) for name in indistinguishable):
            raise V5ContractError(f"{context} indistinguishable set is invalid")
        status = selection.get("statistical_equivalence_status")
        if status not in {"pending_selection_sidecar", "completed_selection_mcs"}:
            raise V5ContractError(f"{context}.statistical_equivalence_status is invalid")
        registry = _mapping(
            selection.get("release_epoch_registry"), f"{context}.release_epoch_registry"
        )
        if registry.get("mode") != "append_only":
            raise V5ContractError(f"{context}.release_epoch_registry mode is invalid")
        _sha256(registry.get("sha256"), f"{context}.release_epoch_registry.sha256")
        _integer(registry.get("epoch_count"), f"{context}.release_epoch_registry.epoch_count")
        multiplicity = _mapping(
            selection.get("multiplicity_defense"), f"{context}.multiplicity_defense"
        )
        if multiplicity.get("current_epoch_method") != (
            "holm_step_down_plus_model_confidence_set"
        ) or multiplicity.get("automatic_promotion_eligible") is not False:
            raise V5ContractError(f"{context}.multiplicity_defense is invalid")
    policy = _OPERATING_CONTRACT.selection_policy
    if selection.get("policy_sha256") != _OPERATING_CONTRACT.selection_policy_sha256:
        raise V5ContractError(f"{context}.policy_sha256 is invalid")
    if selection.get("complexity_registry_sha256") != _OPERATING_CONTRACT.complexity_registry_sha256:
        raise V5ContractError(f"{context}.complexity_registry_sha256 is invalid")
    tolerance = _number(
        selection.get("simplicity_tolerance"),
        f"{context}.simplicity_tolerance",
        minimum=0.0,
    )
    if not math.isclose(tolerance, 0.01, abs_tol=1e-12, rel_tol=0.0):
        raise V5ContractError(f"{context}.simplicity_tolerance is invalid")
    if selection.get("tie_break_order") != policy["tie_break_order"]:
        raise V5ContractError(f"{context}.tie_break_order is invalid")
    candidates = list(
        _sequence(selection.get("candidate_set"), f"{context}.candidate_set", nonempty=True)
    )
    if any(not isinstance(name, str) or not name for name in candidates) or len(candidates) != len(set(candidates)):
        raise V5ContractError(f"{context}.candidate_set is invalid")
    diagnostic_rows = [
        row
        for row in _sequence(
            _require(model, "selection_diagnostics", "payload.model"),
            "payload.model.selection_diagnostics",
            nonempty=True,
        )
        if isinstance(row, Mapping)
    ]
    diagnostic_names = [
        str(row.get("model"))
        for row in diagnostic_rows
    ]
    if candidates != diagnostic_names:
        raise V5ContractError(f"{context}.candidate_set differs from diagnostics")
    runner_up = selection.get("runner_up")
    if runner_up is not None and (
        not isinstance(runner_up, str)
        or runner_up not in candidates
        or runner_up == model.get("champion")
    ):
        raise V5ContractError(f"{context}.runner_up is invalid")
    allowed_reasons = {
        "reference_fallback_no_challenger_passed",
        "simplicity_tiebreak_within_tolerance",
        "best_gate_passing_log_loss",
    }
    reason = selection.get("selection_reason")
    if reason not in allowed_reasons:
        raise V5ContractError(f"{context}.selection_reason is invalid")

    champion = str(model.get("champion", ""))
    champion_row = next(
        (row for row in diagnostic_rows if str(row.get("model")) == champion),
        None,
    )
    if champion_row is None:
        raise V5ContractError(f"{context} champion diagnostics are missing")
    passing_rows = [
        row for row in diagnostic_rows if row.get("gate_passed") is True
    ]
    runner_rows = [
        row
        for row in passing_rows
        if str(row.get("model")) != champion
        and row.get("is_reference") is not True
    ]
    runner_rows.sort(
        key=lambda row: (
            float(row.get("log_loss", float("inf"))),
            float(row.get("brier", float("inf"))),
            str(row.get("model")),
        )
    )
    expected_runner = str(runner_rows[0]["model"]) if runner_rows else None
    if runner_up != expected_runner:
        raise V5ContractError(f"{context}.runner_up differs from gate evidence")

    if champion_row.get("gate_passed") is not True:
        expected_reason = "reference_fallback_no_challenger_passed"
    else:
        champion_loss = float(champion_row.get("log_loss", float("inf")))
        best_gate_row = min(
            passing_rows,
            key=lambda row: (
                float(row.get("log_loss", float("inf"))),
                str(row.get("model")),
            ),
        )
        best_gate_loss = float(best_gate_row.get("log_loss", float("inf")))
        if champion_loss <= best_gate_loss + 1e-12:
            expected_reason = "best_gate_passing_log_loss"
        else:
            if champion_loss > best_gate_loss + tolerance + 1e-12:
                raise V5ContractError(
                    f"{context} simplicity tie exceeds its tolerance"
                )
            registry = policy["complexity_registry"]
            best_name = str(best_gate_row.get("model"))
            if (
                champion not in registry
                or best_name not in registry
                or int(registry[champion]) >= int(registry[best_name])
            ):
                raise V5ContractError(
                    f"{context} simplicity tie lacks a lower-complexity champion"
                )
            expected_reason = "simplicity_tiebreak_within_tolerance"
    if reason != expected_reason:
        raise V5ContractError(f"{context}.selection_reason differs from gate evidence")
    expected_operating = _OPERATING_CONTRACT.document["models"]["official_champion"]
    if selection.get("operating_champion") != expected_operating:
        raise V5ContractError(f"{context}.operating_champion is invalid")


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


def _validate_transition_term_structure(
    value: Any,
    context: str,
    *,
    probabilities: Mapping[int, float],
) -> None:
    term = _mapping(value, context)
    expected = {
        "semantics",
        "coherence_method",
        "one_week_anchor",
        "raw_probabilities",
        "adjusted",
    }
    if set(term) != expected:
        raise V5ContractError(f"{context} fields are invalid")
    if term.get("semantics") != "cumulative_first_departure_probability":
        raise V5ContractError(f"{context}.semantics is invalid")
    if term.get("coherence_method") != (
        "one_week_anchored_l2_isotonic_projection_v1"
    ):
        raise V5ContractError(f"{context}.coherence_method is invalid")
    if term.get("one_week_anchor") != "official_multiclass_departure_probability":
        raise V5ContractError(f"{context}.one_week_anchor is invalid")
    raw = _mapping(term.get("raw_probabilities"), f"{context}.raw_probabilities")
    if set(raw) != {"1w", "4w", "13w"}:
        raise V5ContractError(f"{context}.raw_probabilities fields are invalid")
    raw_values = {
        horizon: _number(
            raw[f"{horizon}w"],
            f"{context}.raw_probabilities.{horizon}w",
            minimum=0.0,
            maximum=1.0,
        )
        for horizon in HORIZONS
    }
    if not isinstance(term.get("adjusted"), bool):
        raise V5ContractError(f"{context}.adjusted must be boolean")
    if not math.isclose(probabilities[1], raw_values[1], abs_tol=1e-8):
        raise V5ContractError(f"{context} changed the one-week anchor")
    if not (
        probabilities[1] <= probabilities[4] + 1e-12
        and probabilities[4] <= probabilities[13] + 1e-12
    ):
        raise V5ContractError(f"{context} cumulative probabilities are not monotone")
    expected_adjusted = any(
        not math.isclose(probabilities[horizon], raw_values[horizon], abs_tol=1e-8)
        for horizon in HORIZONS
    )
    if term.get("adjusted") is not expected_adjusted:
        raise V5ContractError(f"{context}.adjusted is inconsistent")


def _validate_context_score_coverage(
    scores: Mapping[str, Any],
    value: Any,
    context: str,
) -> None:
    coverage = _mapping(value, context)
    expected_names = {"trend", "stress", "macro", "financial_conditions"}
    if set(coverage) != expected_names:
        raise V5ContractError(f"{context} keys are invalid")
    for name in expected_names:
        row_context = f"{context}.{name}"
        row = _mapping(coverage[name], row_context)
        if set(row) != {
            "available_count",
            "expected_count",
            "minimum_required_count",
            "status",
        }:
            raise V5ContractError(f"{row_context} fields are invalid")
        available = _integer(row.get("available_count"), f"{row_context}.available_count")
        expected = _integer(
            row.get("expected_count"), f"{row_context}.expected_count", minimum=1
        )
        minimum = _integer(
            row.get("minimum_required_count"),
            f"{row_context}.minimum_required_count",
            minimum=1,
        )
        if available > expected or minimum > expected:
            raise V5ContractError(f"{row_context} counts are inconsistent")
        sufficient = available >= minimum
        expected_status = "sufficient" if sufficient else "insufficient_coverage"
        if row.get("status") != expected_status:
            raise V5ContractError(f"{row_context}.status is inconsistent")
        score = scores.get(name)
        if sufficient:
            _number(score, f"payload weekly context_scores.{name}", minimum=-1.0, maximum=1.0)
        elif score is not None:
            raise V5ContractError(
                f"payload weekly context_scores.{name} must be null when coverage is insufficient"
            )


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
    *,
    mode: str,
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
            minimum=(
                0
                if key == "fx_ablation_oos"
                or (key == "model_conditioned_asset_outcomes" and mode == "demo")
                else 1
            ),
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
    roster_id = manifest.get("roster_id")
    historical = _OPERATING_CONTRACT.historical_reviewed_roster_by_manifest_sha256(
        supplied_hash
    )
    if roster_id is not None:
        named_historical = _OPERATING_CONTRACT.historical_reviewed_roster(
            str(roster_id)
        )
        if historical is None or named_historical != historical:
            raise V5ContractError(f"{context}.roster_id is invalid")
    historical_expected = (
        set(str(name) for name in historical["candidate_models"])
        if historical is not None
        else None
    )
    current_roster = set(names) == expected and roster_id is None and historical is None
    historical_roster = (
        historical_expected is not None and set(names) == historical_expected
    )
    if len(names) != len(set(names)) or not (current_roster or historical_roster):
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
    if model.get("selection_status") != "selected_by_gate":
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
    for field in ("probability_health", "early_warning_health"):
        if field not in model:
            continue
        track = _mapping(model[field], f"payload.model.{field}")
        if track.get("status") not in {"ok", "review_due", "insufficient_evidence"}:
            raise V5ContractError(f"payload.model.{field}.status is invalid")
        _sequence(
            _require(track, "reasons", f"payload.model.{field}"),
            f"payload.model.{field}.reasons",
        )
        if track.get("champion") != model.get("champion"):
            raise V5ContractError(f"payload.model.{field}.champion is inconsistent")
    probability_health = model.get("probability_health")
    if isinstance(probability_health, Mapping):
        if probability_health.get("calibration_method") != (
            "top_label_ece_10_equal_width_bins"
        ):
            raise V5ContractError(
                "payload.model.probability_health.calibration_method is invalid"
            )
        for field in (
            "calibration_error",
            "selection_calibration_error",
            "calibration_drift",
            "log_loss",
            "brier",
        ):
            _optional_number(
                probability_health.get(field),
                f"payload.model.probability_health.{field}",
            )
        _integer(
            probability_health.get("n_predictions"),
            "payload.model.probability_health.n_predictions",
        )
    early_warning_health = model.get("early_warning_health")
    if isinstance(early_warning_health, Mapping):
        for field in (
            "event_count",
            "on_time_departure_count",
            "false_alarm_count",
            "detected_event_count",
            "minimum_event_count",
            "n_predictions",
        ):
            _integer(
                early_warning_health.get(field),
                f"payload.model.early_warning_health.{field}",
            )
        for field in (
            "on_time_recall",
            "precision",
            "false_alarms_per_year",
            "mean_detection_delay_forecast_weeks",
            "exposure_years",
        ):
            _optional_number(
                early_warning_health.get(field),
                f"payload.model.early_warning_health.{field}",
                minimum=0.0,
            )
        events = int(early_warning_health.get("event_count", 0))
        on_time = int(early_warning_health.get("on_time_departure_count", 0))
        if on_time > events:
            raise V5ContractError(
                "payload.model.early_warning_health on-time count exceeds events"
            )
        recall = early_warning_health.get("on_time_recall")
        if events > 0 and recall is not None and not math.isclose(
            float(recall), on_time / events, abs_tol=1e-8
        ):
            raise V5ContractError(
                "payload.model.early_warning_health recall is inconsistent"
            )
        false_alarms = int(early_warning_health.get("false_alarm_count", 0))
        predicted_departures = on_time + false_alarms
        precision = early_warning_health.get("precision")
        if predicted_departures > 0 and precision is not None and not math.isclose(
            float(precision), on_time / predicted_departures, abs_tol=1e-8
        ):
            raise V5ContractError(
                "payload.model.early_warning_health precision is inconsistent"
            )
    term_evidence = model.get("transition_term_structure_evidence")
    if term_evidence is not None:
        term_evidence = _mapping(
            term_evidence, "payload.model.transition_term_structure_evidence"
        )
        if set(term_evidence) != {
            "evidence_track",
            "evidence_status",
            "evaluation_split",
            "selection_end",
            "source_artifact",
            "probability_source",
            "projection_fit",
            "matched_origin_count",
            "matched_probability_count",
            "raw_brier",
            "projected_brier",
            "brier_difference_projected_minus_raw",
            "selection_effect",
        }:
            raise V5ContractError(
                "payload.model.transition_term_structure_evidence fields are invalid"
            )
        if (
            term_evidence.get("evidence_track") != "reconstructed_oos"
            or term_evidence.get("evidence_status")
            != "historical_reconstructed_oos"
            or term_evidence.get("evaluation_split") != "selection"
            or term_evidence.get("source_artifact")
            != "transition-oos-predictions.csv"
            or term_evidence.get("probability_source")
            != "selected_horizon_champion_calibrated_p_change"
            or term_evidence.get("projection_fit")
            != "parameter_free_fixed_l2_order_constraint"
            or term_evidence.get("selection_effect")
            != "semantic_coherence_only_no_model_selection"
        ):
            raise V5ContractError(
                "payload.model.transition_term_structure_evidence identity is invalid"
            )
        origin_count = _integer(
            term_evidence.get("matched_origin_count"),
            "payload.model.transition_term_structure_evidence.matched_origin_count",
            minimum=1,
        )
        probability_count = _integer(
            term_evidence.get("matched_probability_count"),
            "payload.model.transition_term_structure_evidence.matched_probability_count",
        )
        if probability_count != len((1, 4, 13)) * origin_count:
            raise V5ContractError(
                "payload.model.transition_term_structure_evidence coverage is inconsistent"
            )
        for field in ("raw_brier", "projected_brier"):
            _number(
                term_evidence.get(field),
                f"payload.model.transition_term_structure_evidence.{field}",
                minimum=0.0,
                maximum=1.0,
            )
        _number(
            term_evidence.get("brier_difference_projected_minus_raw"),
            "payload.model.transition_term_structure_evidence."
            "brier_difference_projected_minus_raw",
            minimum=-1.0,
            maximum=1.0,
        )
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
    _validate_research_artifacts(model, directional, mode=mode)
    return _validate_evidence_artifacts(model)


def _validate_forecast_comparison(model: Mapping[str, Any]) -> tuple[str, ...]:
    raw = _require(model, "forecast_comparison", "payload.model")
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
    historical = _OPERATING_CONTRACT.historical_reviewed_roster_by_manifest_sha256(
        str(model.get("candidate_manifest_sha256", ""))
    )
    expected_models = (
        tuple(str(name) for name in historical["forecast_comparison_models"])
        if historical is not None
        else V5_FORECAST_COMPARISON_MODELS
    )
    valid_models = models == expected_models
    if champion not in expected_models:
        valid_models = models == (*expected_models, champion)
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


def _validate_selected_gate_evidence(
    row: Mapping[str, Any],
    *,
    context: str,
) -> None:
    if row.get("gate_passed") is not True or row.get("gate_reason") != "passed":
        raise V5ContractError(f"{context} selected model must have passed its gate")

    model_name = _require(row, "model", context)
    reference_model = _require(row, "reference_model", context)
    if not isinstance(model_name, str) or not model_name:
        raise V5ContractError(f"{context}.model must be non-empty")
    if not isinstance(reference_model, str) or not reference_model:
        raise V5ContractError(f"{context}.reference_model must be non-empty")

    log_loss = _number(
        _require(row, "log_loss", context),
        f"{context}.log_loss",
        minimum=0.0,
    )
    reference_log_loss = _number(
        _require(row, "reference_log_loss", context),
        f"{context}.reference_log_loss",
        minimum=0.0,
    )
    improvement = _number(
        _require(row, "absolute_log_loss_improvement", context),
        f"{context}.absolute_log_loss_improvement",
    )
    brier = _number(
        _require(row, "brier", context),
        f"{context}.brier",
        minimum=0.0,
    )
    reference_brier = _number(
        _require(row, "reference_brier", context),
        f"{context}.reference_brier",
        minimum=0.0,
    )
    brier_difference = _number(
        _require(row, "brier_difference", context),
        f"{context}.brier_difference",
    )
    if not math.isclose(
        improvement,
        reference_log_loss - log_loss,
        abs_tol=1e-10,
        rel_tol=0.0,
    ):
        raise V5ContractError(f"{context}.absolute_log_loss_improvement is inconsistent")
    if not math.isclose(
        brier_difference,
        brier - reference_brier,
        abs_tol=1e-10,
        rel_tol=0.0,
    ):
        raise V5ContractError(f"{context}.brier_difference is inconsistent")

    fallback_count = _integer(
        _require(row, "fallback_count", context),
        f"{context}.fallback_count",
    )
    if fallback_count != 0:
        raise V5ContractError(f"{context}.fallback_count must be zero")
    _integer(
        _require(row, "n_predictions", context),
        f"{context}.n_predictions",
        minimum=1,
    )
    block_weeks = _integer(
        _require(row, "bootstrap_block_weeks", context),
        f"{context}.bootstrap_block_weeks",
        minimum=1,
    )
    effective_block_weeks = _integer(
        _require(row, "bootstrap_effective_block_weeks", context),
        f"{context}.bootstrap_effective_block_weeks",
        minimum=1,
    )
    if block_weeks != 13 or effective_block_weeks > block_weeks:
        raise V5ContractError(f"{context} bootstrap block contract is invalid")
    if (
        _integer(
            _require(row, "bootstrap_resamples", context),
            f"{context}.bootstrap_resamples",
            minimum=1,
        )
        != 1_999
        or _integer(
            _require(row, "bootstrap_seed", context),
            f"{context}.bootstrap_seed",
        )
        != 17
    ):
        raise V5ContractError(f"{context} bootstrap contract is invalid")

    alpha = _number(
        _require(row, "alpha", context),
        f"{context}.alpha",
        minimum=0.0,
        maximum=V5_MAXIMUM_PROMOTION_ALPHA,
    )
    if alpha <= 0.0:
        raise V5ContractError(f"{context}.alpha must be positive")
    minimum_improvement = _number(
        _require(row, "minimum_log_loss_improvement", context),
        f"{context}.minimum_log_loss_improvement",
        minimum=V5_MINIMUM_PROMOTION_LOG_LOSS_IMPROVEMENT,
    )
    if not any(
        math.isclose(
            minimum_improvement,
            allowed,
            abs_tol=1e-12,
            rel_tol=0.0,
        )
        for allowed in (0.01, 0.05)
    ):
        raise V5ContractError(
            f"{context}.minimum_log_loss_improvement is not an approved threshold"
        )
    brier_tolerance = _number(
        _require(row, "brier_tolerance", context),
        f"{context}.brier_tolerance",
        minimum=0.0,
        maximum=V5_MAXIMUM_PROMOTION_BRIER_DEGRADATION,
    )

    if model_name == reference_model:
        if (
            not math.isclose(improvement, 0.0, abs_tol=1e-10, rel_tol=0.0)
            or not math.isclose(brier_difference, 0.0, abs_tol=1e-10, rel_tol=0.0)
            or row.get("raw_p_value") is not None
            or row.get("holm_adjusted_p_value") is not None
        ):
            raise V5ContractError(f"{context} reference-model gate evidence is invalid")
        return

    raw_p_value = _number(
        _require(row, "raw_p_value", context),
        f"{context}.raw_p_value",
        minimum=0.0,
        maximum=1.0,
    )
    adjusted_p_value = _number(
        _require(row, "holm_adjusted_p_value", context),
        f"{context}.holm_adjusted_p_value",
        minimum=0.0,
        maximum=1.0,
    )
    if (
        improvement + 1e-12 < minimum_improvement
        or adjusted_p_value > alpha + 1e-12
        or adjusted_p_value + 1e-12 < raw_p_value
        or brier_difference > brier_tolerance + 1e-12
    ):
        raise V5ContractError(f"{context} selected challenger gate evidence is invalid")


def validate_v5_champion_selection_evidence(model: Mapping[str, Any]) -> str:
    """Bind the declared champion to one audited selection-gate decision."""

    champion = _require(model, "champion", "payload.model")
    if not isinstance(champion, str) or not champion:
        raise V5ContractError("payload.model.champion must be non-empty")

    leaderboard = _sequence(
        _require(model, "leaderboard", "payload.model"),
        "payload.model.leaderboard",
        nonempty=True,
    )
    selected_leaderboard: list[str] = []
    champion_leaderboard: list[str] = []
    leaderboard_names: set[str] = set()
    for index, raw in enumerate(leaderboard):
        context = f"payload.model.leaderboard[{index}]"
        row = _mapping(raw, context)
        name = _require(row, "name", context)
        if not isinstance(name, str) or not name:
            raise V5ContractError(f"{context}.name must be non-empty")
        if name in leaderboard_names:
            raise V5ContractError(f"{context}.name must be unique")
        leaderboard_names.add(name)
        if not isinstance(row.get("selected"), bool) or not isinstance(
            row.get("is_champion"), bool
        ):
            raise V5ContractError(
                f"{context}.selected/is_champion must be boolean"
            )
        if row["selected"] != row["is_champion"]:
            raise V5ContractError(f"{context}.selected/is_champion must agree")
        if row["selected"]:
            selected_leaderboard.append(name)
        if row["is_champion"]:
            champion_leaderboard.append(name)
    if selected_leaderboard != [champion] or champion_leaderboard != [champion]:
        raise V5ContractError(
            "payload.model.leaderboard must select exactly the declared champion"
        )

    diagnostics = _sequence(
        _require(model, "selection_diagnostics", "payload.model"),
        "payload.model.selection_diagnostics",
        nonempty=True,
    )
    selected_diagnostics: list[tuple[str, Mapping[str, Any], str]] = []
    diagnostic_rows: dict[str, tuple[Mapping[str, Any], str]] = {}
    reference_models: set[str] = set()
    for index, raw in enumerate(diagnostics):
        context = f"payload.model.selection_diagnostics[{index}]"
        row = _mapping(raw, context)
        name = _require(row, "model", context)
        if not isinstance(name, str) or not name:
            raise V5ContractError(f"{context}.model must be non-empty")
        if name in diagnostic_rows:
            raise V5ContractError(f"{context}.model must be unique")
        diagnostic_rows[name] = (row, context)
        reference_model = _require(row, "reference_model", context)
        if not isinstance(reference_model, str) or not reference_model:
            raise V5ContractError(f"{context}.reference_model must be non-empty")
        reference_models.add(reference_model)
        if not isinstance(row.get("selected"), bool) or not isinstance(
            row.get("gate_passed"), bool
        ):
            raise V5ContractError(f"{context}.selected/gate_passed must be boolean")
        if row["selected"]:
            selected_diagnostics.append((name, row, context))
    if len(selected_diagnostics) != 1 or selected_diagnostics[0][0] != champion:
        raise V5ContractError(
            "payload.model.selection_diagnostics must select exactly the declared "
            "champion"
        )
    if len(reference_models) != 1:
        raise V5ContractError(
            "payload.model.selection_diagnostics reference model is inconsistent"
        )
    reference_model = next(iter(reference_models))
    if reference_model not in diagnostic_rows or reference_model not in leaderboard_names:
        raise V5ContractError(
            "payload.model.selection_diagnostics reference model is missing"
        )

    selected_row, selected_context = (
        selected_diagnostics[0][1],
        selected_diagnostics[0][2],
    )
    reference_row, reference_context = diagnostic_rows[reference_model]
    _validate_selected_gate_evidence(selected_row, context=selected_context)
    if selected_row is not reference_row:
        _validate_selected_gate_evidence(reference_row, context=reference_context)
        metric_bindings = (
            ("reference_log_loss", "log_loss"),
            ("reference_brier", "brier"),
            ("n_predictions", "n_predictions"),
            ("bootstrap_block_weeks", "bootstrap_block_weeks"),
            ("bootstrap_effective_block_weeks", "bootstrap_effective_block_weeks"),
            ("bootstrap_resamples", "bootstrap_resamples"),
            ("bootstrap_seed", "bootstrap_seed"),
            ("alpha", "alpha"),
            ("minimum_log_loss_improvement", "minimum_log_loss_improvement"),
            ("brier_tolerance", "brier_tolerance"),
        )
        for selected_field, reference_field in metric_bindings:
            if selected_row.get(selected_field) != reference_row.get(reference_field):
                raise V5ContractError(
                    f"{selected_context}.{selected_field} differs from reference "
                    "model evidence"
                )
    return champion


def _canonical_payload_sha256(payload: Mapping[str, Any]) -> str:
    try:
        raw = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise V5ContractError("payload cannot be canonicalized") from exc
    return hashlib.sha256(raw).hexdigest()


def _validate_reviewed_selection_threshold(
    model: Mapping[str, Any],
    *,
    expected: float,
) -> None:
    diagnostics = _sequence(
        _require(model, "selection_diagnostics", "payload.model"),
        "payload.model.selection_diagnostics",
        nonempty=True,
    )
    for index, raw in enumerate(diagnostics):
        context = f"payload.model.selection_diagnostics[{index}]"
        row = _mapping(raw, context)
        threshold = _number(
            _require(row, "minimum_log_loss_improvement", context),
            f"{context}.minimum_log_loss_improvement",
            minimum=0.0,
        )
        if not math.isclose(threshold, expected, abs_tol=1e-12, rel_tol=0.0):
            raise V5ContractError(
                "reviewed V5 publication selection threshold must be exactly "
                f"{expected:.2f}"
            )


def _validate_publication_review(
    meta: Mapping[str, Any],
    model: Mapping[str, Any],
    *,
    mode: str,
    payload: Mapping[str, Any] | None = None,
) -> None:
    status = meta.get("publication_status")
    review = meta.get("publication_review")
    champion = validate_v5_champion_selection_evidence(model)
    is_legacy_reviewed_snapshot = (
        status == V5_PUBLICATION_STATUS
        and payload is not None
        and _canonical_payload_sha256(payload)
        == V5_LEGACY_REVIEWED_005_SNAPSHOT_SHA256
    )
    _validate_reviewed_selection_threshold(
        model,
        expected=(0.05 if is_legacy_reviewed_snapshot else 0.01),
    )
    if status == "unpublished":
        if review is not None:
            raise V5ContractError(
                "unpublished payload cannot carry publication_review"
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
    if review.get("champion") != champion:
        raise V5ContractError(
            "payload.meta.publication_review.champion must match payload.model.champion"
        )
    expected_multiscale_promotion = champion == V5_MULTISCALE_MODEL
    if review.get("multiscale_promoted") is not expected_multiscale_promotion:
        raise V5ContractError(
            "payload.meta.publication_review.multiscale_promoted is inconsistent"
        )
    if review.get("fx_promoted") is not False:
        raise V5ContractError(
            "payload.meta.publication_review.fx_promoted must be false"
        )
    if payload is not None:
        try:
            validate_reviewed_candidate_hash(payload)
        except IntegrityError as exc:
            raise V5ContractError(str(exc)) from exc


def _validate_conditional_stats(value: Any, *, expected_resamples: int) -> None:
    research = _mapping(value, "payload.research")
    legacy_stats_fields = {
        "method",
        "role",
        "execution_lag_weeks",
        "horizons_weeks",
        "assets",
        "return_currency",
        "rows",
    }
    investment_stats_fields = {
        *legacy_stats_fields,
        "conditioning",
        "state_horizon_weeks",
        "entry_price_basis",
        "exit_price_basis",
        "rebalance_policy",
        "origin_sampling",
        "return_measure",
        "entry_week_distribution_policy",
        "corporate_action_policy",
        "drawdown_observation_basis",
    }
    stats = _mapping(
        _require(research, "conditional_asset_stats", "payload.research"),
        "payload.research.conditional_asset_stats",
    )
    method = stats.get("method")
    expected_stats_fields = (
        investment_stats_fields
        if method
        == "matched_oos_actual_next_state_target_week_adjusted_forward_return"
        else legacy_stats_fields
    )
    if set(stats) != expected_stats_fields:
        raise V5ContractError(
            "payload.research.conditional_asset_stats fields are invalid"
        )
    if method not in {
        "state_conditioned_forward_total_return",
        "matched_oos_actual_next_state_target_week_adjusted_forward_return",
    }:
        raise V5ContractError("conditional asset method is invalid")
    if method == "matched_oos_actual_next_state_target_week_adjusted_forward_return":
        if (
            stats.get("role") != "matched_oracle_diagnostic"
            or stats.get("conditioning")
            != "actual_next_state_on_matched_oos_origins"
            or stats.get("state_horizon_weeks") != 1
            or stats.get("entry_price_basis") != "next_week_adjusted_open"
            or stats.get("exit_price_basis") != "horizon_week_adjusted_close"
            or stats.get("rebalance_policy") != "none_fixed_asset_hold"
            or stats.get("origin_sampling") != "weekly_rolling_overlapping"
            or stats.get("return_measure") != "provider_adjusted_forward_return"
            or stats.get("entry_week_distribution_policy")
            != "conservative_excluded_without_ex_date"
            or stats.get("corporate_action_policy")
            != "same_row_adjustment_factor_split_consistent"
            or stats.get("drawdown_observation_basis")
            != "entry_adjusted_open_then_weekly_adjusted_closes"
        ):
            raise V5ContractError("conditional asset investment semantics are invalid")
    elif stats.get("role") != "descriptive_only":
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
        enhanced_fields = set(CONDITIONAL_STATISTICS_COLUMNS)
        current_core_fields = enhanced_fields.difference(
            ENHANCED_CONDITIONAL_STATISTICS_FIELDS
        )
        non_overlapping_contract_fields = {
            "non_overlapping_n",
            "minimum_non_overlapping_observations",
        }
        historical_enhanced_fields = enhanced_fields.difference(
            non_overlapping_contract_fields
        )
        legacy_enhanced_fields = enhanced_fields.difference(
            MATCHED_EPISODE_BENCHMARK_FIELDS
        )
        historical_legacy_enhanced_fields = legacy_enhanced_fields.difference(
            non_overlapping_contract_fields
        )
        historical_core_fields = current_core_fields.difference(
            non_overlapping_contract_fields
        )
        allowed_row_fields = {
            frozenset(historical_core_fields),
            frozenset(historical_enhanced_fields),
            frozenset(current_core_fields),
            frozenset(enhanced_fields),
            frozenset(legacy_enhanced_fields),
            frozenset(historical_legacy_enhanced_fields),
        }
        if method == "matched_oos_actual_next_state_target_week_adjusted_forward_return":
            allowed_row_fields = {frozenset(enhanced_fields)}
        if frozenset(row) not in allowed_row_fields:
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
        count = _integer(_require(row, "n", context), f"{context}.n")
        non_overlapping: int | None = None
        if "non_overlapping_n" in row:
            non_overlapping = _integer(
                row.get("non_overlapping_n"),
                f"{context}.non_overlapping_n",
            )
            if non_overlapping > count:
                raise V5ContractError(
                    f"{context}.non_overlapping_n exceeds n"
                )
        unique_episodes = _integer(
            _require(row, "unique_episodes", context),
            f"{context}.unique_episodes",
        )
        if "minimum_non_overlapping_observations" in row and (
            row.get("minimum_non_overlapping_observations") != 5
        ):
            raise V5ContractError(
                f"{context}.minimum_non_overlapping_observations is invalid"
            )
        if LEGACY_ENHANCED_CONDITIONAL_STATISTICS_FIELDS.issubset(row):
            benchmark_method = row.get("unconditional_benchmark_method")
            if method == (
                "matched_oos_actual_next_state_target_week_adjusted_forward_return"
            ):
                benchmark_method_is_valid = (
                    benchmark_method == "same_asset_horizon_all_origins_mean"
                )
            else:
                benchmark_method_is_valid = benchmark_method in {
                    "same_asset_horizon_all_origins_buy_and_hold",
                    "same_asset_horizon_all_origins_mean",
                }
            if not benchmark_method_is_valid:
                raise V5ContractError(
                    f"{context}.unconditional_benchmark_method is invalid"
                )
            _integer(
                row.get("unconditional_benchmark_n"),
                f"{context}.unconditional_benchmark_n",
            )
            for field in (
                "unconditional_benchmark_mean_return",
                "excess_mean_return",
                "episode_equal_mean_return",
                "episode_equal_excess_return",
            ):
                _optional_number(row.get(field), f"{context}.{field}")
            if row.get("episode_bootstrap_method") != "whole_episode_resampling":
                raise V5ContractError(f"{context}.episode_bootstrap_method is invalid")
            if row.get("episode_bootstrap_resamples") != expected_resamples:
                raise V5ContractError(
                    f"{context}.episode_bootstrap_resamples is inconsistent"
                )
            _integer(
                row.get("episode_bootstrap_seed"),
                f"{context}.episode_bootstrap_seed",
            )
            episode_lower = _optional_number(
                row.get("episode_equal_mean_return_ci95_lower"),
                f"{context}.episode_equal_mean_return_ci95_lower",
            )
            episode_upper = _optional_number(
                row.get("episode_equal_mean_return_ci95_upper"),
                f"{context}.episode_equal_mean_return_ci95_upper",
            )
            if (episode_lower is None) != (episode_upper is None):
                raise V5ContractError(f"{context} episode CI must share nullability")
            if episode_lower is not None and episode_upper < episode_lower:
                raise V5ContractError(f"{context} episode CI is reversed")
            if MATCHED_EPISODE_BENCHMARK_FIELDS.issubset(row):
                if (
                    row.get("episode_equal_unconditional_benchmark_method")
                    != "same_asset_horizon_all_state_episodes_equal_weight"
                ):
                    raise V5ContractError(
                        f"{context}.episode_equal_unconditional_benchmark_method "
                        "is invalid"
                    )
                _integer(
                    row.get("episode_equal_unconditional_benchmark_episode_n"),
                    f"{context}.episode_equal_unconditional_benchmark_episode_n",
                )
                episode_benchmark = _optional_number(
                    row.get(
                        "episode_equal_unconditional_benchmark_mean_return"
                    ),
                    f"{context}.episode_equal_unconditional_benchmark_mean_return",
                )
                episode_mean = _optional_number(
                    row.get("episode_equal_mean_return"),
                    f"{context}.episode_equal_mean_return",
                )
                episode_excess = _optional_number(
                    row.get("episode_equal_excess_return"),
                    f"{context}.episode_equal_excess_return",
                )
                complete_episode_estimand = None not in (
                    episode_benchmark,
                    episode_mean,
                    episode_excess,
                )
                if complete_episode_estimand and not math.isclose(
                    episode_excess,
                    episode_mean - episode_benchmark,
                    abs_tol=2e-8,
                    rel_tol=0.0,
                ):
                    raise V5ContractError(
                        f"{context}.episode_equal_excess_return is inconsistent"
                    )
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
        if non_overlapping is not None:
            expected_status = (
                "ok"
                if count >= 20
                and unique_episodes >= 5
                and non_overlapping >= 5
                else "insufficient_support"
            )
            if row.get("status") != expected_status:
                raise V5ContractError(f"{context}.status is inconsistent with support")
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
    expected_models: tuple[str, ...],
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
        raise V5ContractError(
            "payload.research.model_conditioned_asset_stats is required"
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
    investment_stats_fields = {
        *expected_stats_fields,
        "entry_price_basis",
        "exit_price_basis",
        "rebalance_policy",
        "origin_sampling",
        "return_measure",
        "entry_week_distribution_policy",
        "corporate_action_policy",
        "drawdown_observation_basis",
    }
    method = stats.get("method")
    expected_fields = (
        investment_stats_fields
        if method
        == "matched_oos_predicted_next_state_target_week_adjusted_forward_return"
        else expected_stats_fields
    )
    if set(stats) != expected_fields:
        raise V5ContractError(f"{context} fields are invalid")
    if method not in {
        "oos_one_week_forecast_conditioned_forward_total_return",
        "matched_oos_predicted_next_state_target_week_adjusted_forward_return",
    }:
        raise V5ContractError(f"{context}.method is invalid")
    if stats.get("role") != "retrospective_model_diagnostic":
        raise V5ContractError(f"{context}.role is invalid")
    if stats.get("conditioning") != "hard_argmax_oos_forecast":
        raise V5ContractError(f"{context}.conditioning is invalid")
    if stats.get("forecast_horizon_weeks") != 1:
        raise V5ContractError(f"{context}.forecast_horizon_weeks must be one")
    if method == (
        "matched_oos_predicted_next_state_target_week_adjusted_forward_return"
    ) and (
        stats.get("entry_price_basis") != "next_week_adjusted_open"
        or stats.get("exit_price_basis") != "horizon_week_adjusted_close"
        or stats.get("rebalance_policy") != "none_fixed_asset_hold"
        or stats.get("origin_sampling") != "weekly_rolling_overlapping"
        or stats.get("return_measure") != "provider_adjusted_forward_return"
        or stats.get("entry_week_distribution_policy")
        != "conservative_excluded_without_ex_date"
        or stats.get("corporate_action_policy")
        != "same_row_adjustment_factor_split_consistent"
        or stats.get("drawdown_observation_basis")
        != "entry_adjusted_open_then_weekly_adjusted_closes"
    ):
        raise V5ContractError(f"{context} investment semantics are invalid")
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
            enhanced_model_fields = frozenset(MODEL_CONDITIONED_STATISTICS_COLUMNS)
            current_core_model_fields = enhanced_model_fields.difference(
                ENHANCED_CONDITIONAL_STATISTICS_FIELDS
            )
            non_overlapping_contract_fields = {
                "non_overlapping_n",
                "minimum_non_overlapping_observations",
            }
            historical_enhanced_model_fields = enhanced_model_fields.difference(
                non_overlapping_contract_fields
            )
            legacy_enhanced_model_fields = enhanced_model_fields.difference(
                MATCHED_EPISODE_BENCHMARK_FIELDS
            )
            historical_legacy_enhanced_model_fields = (
                legacy_enhanced_model_fields.difference(
                    non_overlapping_contract_fields
                )
            )
            historical_core_model_fields = current_core_model_fields.difference(
                non_overlapping_contract_fields
            )
            allowed_model_row_fields = {
                enhanced_model_fields,
                frozenset(current_core_model_fields),
                frozenset(historical_enhanced_model_fields),
                frozenset(historical_core_model_fields),
                frozenset(legacy_enhanced_model_fields),
                frozenset(historical_legacy_enhanced_model_fields),
            }
            if method == (
                "matched_oos_predicted_next_state_target_week_adjusted_forward_return"
            ):
                allowed_model_row_fields = {enhanced_model_fields}
            if frozenset(row) not in allowed_model_row_fields:
                raise V5ContractError(f"{row_context} fields are invalid")
            if (
                method
                == "matched_oos_predicted_next_state_target_week_adjusted_forward_return"
                and row.get("unconditional_benchmark_method")
                != "same_asset_horizon_all_origins_mean"
            ):
                raise V5ContractError(
                    f"{row_context}.unconditional_benchmark_method is invalid"
                )
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


def _validate_decision_shadow_current_signal(
    value: Any,
    *,
    context: str,
) -> dict[str, Any]:
    signal = _mapping(value, context)
    if set(signal) != {
        "origin_date",
        "target_week",
        "scheduled_entry_at",
        "decision_at",
        "forecast_model",
        "status",
        "action",
    }:
        raise V5ContractError(f"{context} fields are invalid")
    origin_date = _iso_date(signal.get("origin_date"), f"{context}.origin_date")
    target_week = _iso_date(signal.get("target_week"), f"{context}.target_week")
    if target_week != origin_date + timedelta(days=7):
        raise V5ContractError(f"{context}.target_week must be origin_date plus 7 days")
    scheduled_entry_at = _iso_datetime(
        signal.get("scheduled_entry_at"), f"{context}.scheduled_entry_at"
    )
    decision_at = _iso_datetime(signal.get("decision_at"), f"{context}.decision_at")
    forecast_model = signal.get("forecast_model")
    if not isinstance(forecast_model, str) or not forecast_model:
        raise V5ContractError(f"{context}.forecast_model must be non-empty")
    scheduled_eastern = scheduled_entry_at.astimezone(
        ZoneInfo("America/New_York")
    )
    expected_session = _first_nyse_session_of_week(target_week)
    if (
        scheduled_eastern.date() != expected_session
        or scheduled_eastern.timetz().replace(tzinfo=None) != time(9, 30)
    ):
        raise V5ContractError(
            f"{context}.scheduled_entry_at must be the target week's first "
            "NYSE session at 09:30 America/New_York"
        )
    if decision_at < scheduled_entry_at:
        expected_status_action = ("scheduled", "trade_at_scheduled_open")
    else:
        expected_status_action = ("missed_entry", "no_trade")
    if (signal.get("status"), signal.get("action")) != expected_status_action:
        raise V5ContractError(f"{context} status/action is inconsistent with timing")
    return {
        "origin_date": origin_date,
        "target_week": target_week,
        "scheduled_entry_at": scheduled_entry_at,
        "decision_at": decision_at,
        "forecast_model": forecast_model,
    }


def _validate_decision_shadow(
    research: Mapping[str, Any],
    *,
    latest_week: Mapping[str, Any] | None = None,
    forecast: Mapping[str, Any] | None = None,
    operating_champion: str | None = None,
    generated_at: datetime | None = None,
) -> None:
    raw = research.get("prospective_decision_shadow")
    if raw is None:
        # Backward-compatible with reviewed schema-2.1 snapshots.
        return
    context = "payload.research.prospective_decision_shadow"
    shadow = _mapping(raw, context)
    schema_version = shadow.get("schema_version")
    if schema_version == "regime-prospective-decision-shadow/1":
        is_v2 = False
    elif schema_version == "regime-prospective-decision-shadow/2":
        is_v2 = True
    else:
        raise V5ContractError(f"{context} schema is invalid")
    expected_shadow_fields = {
        "schema_version",
        "role",
        "spec",
        "execution_contract",
        "historical_reconstructed_shadow",
        "prospective_ledger",
    }
    if is_v2:
        expected_shadow_fields.add("current_signal")
    if set(shadow) != expected_shadow_fields:
        raise V5ContractError(f"{context} fields are invalid")
    if shadow.get("role") != "research_only_no_forecast_or_champion_effect":
        raise V5ContractError(f"{context} identity is invalid")
    validated_current_signal: dict[str, Any] | None = None
    if is_v2:
        validated_current_signal = _validate_decision_shadow_current_signal(
            shadow.get("current_signal"), context=f"{context}.current_signal"
        )
    spec = _mapping(shadow.get("spec"), f"{context}.spec")
    if set(spec) != {"path", "sha256", "spec_id"}:
        raise V5ContractError(f"{context}.spec fields are invalid")
    if spec.get("path") not in {
        "config/decision-shadow.json",
        "config/decision-shadow-v2.json",
    }:
        raise V5ContractError(f"{context}.spec.path is invalid")
    _sha256(spec.get("sha256"), f"{context}.spec.sha256")
    execution = _mapping(
        shadow.get("execution_contract"), f"{context}.execution_contract"
    )
    legacy_execution = {
        "first_tradable_point": "next_completed_weekly_close",
        "execution_lag_weeks": 1,
        "holding_period_weeks": 1,
    }
    investment_execution = {
        "signal_origin": "completed_weekly_close",
        "first_tradable_point": "next_week_adjusted_open",
        "target_return_window": "next_week_open_to_close",
        "rebalance_frequency": "weekly",
        "late_signal_policy": "no_trade",
        "holding_period_weeks": 1,
    }
    expected_execution = investment_execution if is_v2 else legacy_execution
    if dict(execution) != expected_execution:
        raise V5ContractError(f"{context}.execution_contract is invalid")
    if not is_v2 and (
        spec.get("path") != "config/decision-shadow.json"
        or spec.get("spec_id") != "spy-tlt-probability-shadow-v1"
    ):
        raise V5ContractError(f"{context}.spec v1 identity is invalid")
    if is_v2 and (
        spec.get("path") != "config/decision-shadow-v2.json"
        or spec.get("spec_id") != "spy-tlt-probability-shadow-v2"
    ):
        raise V5ContractError(f"{context}.spec v2 identity is invalid")
    historical = _mapping(
        shadow.get("historical_reconstructed_shadow"),
        f"{context}.historical_reconstructed_shadow",
    )
    historical_fields = {
        "status",
        "evidence_track",
        "evidence_status",
        "minimum_evaluation_weeks",
        "strategies",
    }
    if is_v2:
        historical_fields.update(
            {
                "first_tradable_week",
                "evaluation_start_week",
                "evaluation_end_week",
                "latest_target_weights",
                "allocation_policy",
            }
        )
    else:
        historical_fields.add("first_tradable_at")
    if set(historical) != historical_fields:
        raise V5ContractError(f"{context} historical fields are invalid")
    if (
        historical.get("evidence_track") != "reconstructed_oos"
        or historical.get("evidence_status") != "historical_reconstructed_shadow"
        or historical.get("status") not in {"completed", "insufficient_history"}
    ):
        raise V5ContractError(f"{context} historical evidence identity is invalid")
    if is_v2:
        if historical.get("first_tradable_week") is not None:
            _iso_date(
                historical.get("first_tradable_week"),
                f"{context}.historical_reconstructed_shadow.first_tradable_week",
            )
        evaluation_start = historical.get("evaluation_start_week")
        evaluation_end = historical.get("evaluation_end_week")
        if (evaluation_start is None) != (evaluation_end is None):
            raise V5ContractError(f"{context} evaluation bounds differ in nullability")
        if evaluation_start is not None:
            evaluation_start = _iso_date(
                evaluation_start,
                f"{context}.historical_reconstructed_shadow.evaluation_start_week",
            )
            evaluation_end = _iso_date(
                evaluation_end,
                f"{context}.historical_reconstructed_shadow.evaluation_end_week",
            )
            if evaluation_end < evaluation_start:
                raise V5ContractError(f"{context} evaluation bounds are reversed")
    elif historical.get("first_tradable_at") is not None:
        _iso_datetime(
            historical.get("first_tradable_at"),
            f"{context}.historical_reconstructed_shadow.first_tradable_at",
        )
    minimum_evaluation_weeks = _integer(
        historical.get("minimum_evaluation_weeks"),
        f"{context}.historical_reconstructed_shadow.minimum_evaluation_weeks",
        minimum=1,
    )
    if is_v2:
        allocation = _mapping(
            historical.get("allocation_policy"),
            f"{context}.historical_reconstructed_shadow.allocation_policy",
        )
        if set(allocation) != {
            "method",
            "assets",
            "forecast_model",
            "latest_signal_origin",
            "latest_target_weights",
        }:
            raise V5ContractError(f"{context} allocation policy fields are invalid")
        if (
            allocation.get("method")
            != "probability_weighted_state_portfolios"
            or tuple(allocation.get("assets", ())) != ("SPY", "TLT")
        ):
            raise V5ContractError(f"{context} allocation policy is invalid")
        forecast_model = allocation.get("forecast_model")
        if not isinstance(forecast_model, str) or not forecast_model:
            raise V5ContractError(f"{context} allocation forecast_model is invalid")
        latest_weights = allocation.get("latest_target_weights")
        if historical.get("latest_target_weights") != latest_weights:
            raise V5ContractError(f"{context} latest target weight copies differ")
        latest_signal_origin = allocation.get("latest_signal_origin")
        if latest_signal_origin is not None:
            _iso_date(
                latest_signal_origin,
                f"{context}.historical_reconstructed_shadow.allocation_policy.latest_signal_origin",
            )
        if latest_weights is not None:
            weights = _mapping(
                latest_weights,
                f"{context}.historical_reconstructed_shadow.allocation_policy.latest_target_weights",
            )
            if set(weights) != {"SPY", "TLT"}:
                raise V5ContractError(f"{context} latest target weights are invalid")
            values = [
                _number(weights[asset], f"{context}.latest_target_weights.{asset}")
                for asset in ("SPY", "TLT")
            ]
            if any(value < 0.0 or value > 1.0 for value in values) or not math.isclose(
                sum(values), 1.0, abs_tol=1e-8
            ):
                raise V5ContractError(f"{context} latest target weights are invalid")
        binding_inputs = (
            latest_week,
            forecast,
            operating_champion,
            generated_at,
        )
        if any(value is not None for value in binding_inputs):
            if any(value is None for value in binding_inputs):
                raise V5ContractError(
                    f"{context} payload binding inputs are incomplete"
                )
            assert validated_current_signal is not None
            assert latest_week is not None
            assert forecast is not None
            assert operating_champion is not None
            assert generated_at is not None
            latest_origin = _iso_date(
                latest_week.get("date"),
                "payload.weekly[-1].date",
            )
            official_next = _mapping(
                latest_week.get("next_week"),
                "payload.weekly[-1].next_week",
            )
            official_target = _iso_date(
                official_next.get("date"),
                "payload.weekly[-1].next_week.date",
            )
            forecast_rows = _sequence(
                latest_week.get("model_forecasts"),
                "payload.weekly[-1].model_forecasts",
                nonempty=True,
            )
            operating_rows = [
                _mapping(row, "payload.weekly[-1].model_forecasts[]")
                for row in forecast_rows
                if isinstance(row, Mapping)
                and row.get("model") == operating_champion
            ]
            if len(operating_rows) != 1:
                raise V5ContractError(
                    f"{context} operating champion forecast is not unique"
                )
            operating_row = operating_rows[0]
            operating_target = _iso_date(
                operating_row.get("date"),
                "payload.weekly[-1].model_forecasts[operating].date",
            )
            if operating_target != official_target:
                raise V5ContractError(
                    f"{context} operating champion target differs from next_week"
                )
            if (
                validated_current_signal["origin_date"] != latest_origin
                or validated_current_signal["target_week"] != operating_target
                or validated_current_signal["forecast_model"]
                != operating_champion
                or allocation.get("forecast_model") != operating_champion
                or allocation.get("latest_signal_origin")
                != latest_origin.isoformat()
            ):
                raise V5ContractError(
                    f"{context} current signal/allocation payload binding is invalid"
                )
            eastern = ZoneInfo("America/New_York")
            forecast_origin = _iso_datetime(
                forecast.get("origin_at"),
                "payload.forecast.origin_at",
            )
            forecast_target = _iso_datetime(
                forecast.get("target_at"),
                "payload.forecast.target_at",
            )
            if (
                forecast_origin.astimezone(eastern).date() != latest_origin
                or forecast_target.astimezone(eastern).date() != operating_target
            ):
                raise V5ContractError(
                    f"{context} current signal dates differ from forecast envelope"
                )
            raw_forecast_decision = forecast.get("decision_at")
            expected_decision = (
                generated_at
                if raw_forecast_decision is None
                else _iso_datetime(
                    raw_forecast_decision,
                    "payload.forecast.decision_at",
                )
            )
            if validated_current_signal["decision_at"] != expected_decision:
                raise V5ContractError(
                    f"{context} current signal decision differs from forecast"
                )
            probabilities = _mapping(
                operating_row.get("probabilities"),
                "payload.weekly[-1].model_forecasts[operating].probabilities",
            )
            if set(probabilities) != set(STATE_ORDER):
                raise V5ContractError(
                    f"{context} operating champion probabilities are invalid"
                )
            try:
                weight_mapping = load_decision_shadow_spec()[
                    "probability_weight_mapping"
                ]
            except (OSError, TypeError, ValueError, KeyError) as exc:
                raise V5ContractError(
                    f"{context} local weight mapping is invalid"
                ) from exc
            if latest_weights is None:
                raise V5ContractError(
                    f"{context} current operating forecast lacks target weights"
                )
            probability_values = {
                state: _number(
                    probabilities[state],
                    f"payload.weekly[-1].model_forecasts[operating]."
                    f"probabilities.{state}",
                )
                for state in STATE_ORDER
            }
            probability_total = sum(probability_values.values())
            expected_weights = {
                asset: sum(
                    probability_values[state]
                    / probability_total
                    * float(weight_mapping[state][asset])
                    for state in STATE_ORDER
                )
                for asset in ("SPY", "TLT")
            }
            if any(
                not math.isclose(
                    float(latest_weights[asset]),
                    expected_weights[asset],
                    abs_tol=1e-8,
                )
                for asset in ("SPY", "TLT")
            ):
                raise V5ContractError(
                    f"{context} latest target weights differ from operating forecast"
                )
    strategies = _mapping(
        historical.get("strategies"),
        f"{context}.historical_reconstructed_shadow.strategies",
    )
    if set(strategies) != {
        "probability_shadow",
        "spy_buy_and_hold",
        "static_60_40",
        "vol_target_60_40",
    }:
        raise V5ContractError(f"{context} benchmark set is invalid")
    metric_fields = {
        "weeks",
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "certainty_equivalent_return",
        "maximum_drawdown",
        "annualized_turnover",
        "gross_cumulative_return",
        "transaction_cost_bps",
    }
    transaction_cost_field = (
        "transaction_cost_rate_sum" if is_v2 else "total_transaction_cost"
    )
    metric_fields.add(transaction_cost_field)
    common_weeks: int | None = None
    common_transaction_cost_bps: float | None = None
    for name, raw_metrics in strategies.items():
        metrics_context = f"{context}.strategies.{name}"
        metrics = _mapping(raw_metrics, metrics_context)
        if set(metrics) != metric_fields:
            raise V5ContractError(f"{metrics_context} fields are invalid")
        weeks = _integer(metrics.get("weeks"), f"{metrics_context}.weeks")
        if common_weeks is None:
            common_weeks = weeks
        elif weeks != common_weeks:
            raise V5ContractError(f"{context} strategy weeks are inconsistent")
        for field in (
            "annualized_return",
            "annualized_volatility",
            "sharpe",
            "certainty_equivalent_return",
            "maximum_drawdown",
        ):
            _optional_number(metrics.get(field), f"{metrics_context}.{field}")
        annualized_turnover = _optional_number(
            metrics.get("annualized_turnover"),
            f"{metrics_context}.annualized_turnover",
            minimum=0.0,
        )
        transaction_cost_sum = _number(
            metrics.get(transaction_cost_field),
            f"{metrics_context}.{transaction_cost_field}",
            minimum=0.0,
        )
        transaction_cost_bps = _number(
            metrics.get("transaction_cost_bps"),
            f"{metrics_context}.transaction_cost_bps",
            minimum=0.0,
        )
        if common_transaction_cost_bps is None:
            common_transaction_cost_bps = transaction_cost_bps
        elif transaction_cost_bps != common_transaction_cost_bps:
            raise V5ContractError(
                f"{context} strategy transaction_cost_bps are inconsistent"
            )
        if transaction_cost_bps != 10.0:
            raise V5ContractError(
                f"{metrics_context}.transaction_cost_bps differs from spec"
            )
        net_cumulative = _optional_number(
            metrics.get("cumulative_return"), f"{metrics_context}.cumulative_return"
        )
        gross_cumulative = _optional_number(
            metrics.get("gross_cumulative_return"),
            f"{metrics_context}.gross_cumulative_return",
        )
        if (net_cumulative is None) != (gross_cumulative is None):
            raise V5ContractError(
                f"{metrics_context} net/gross cumulative return nullability differs"
            )
        if (
            net_cumulative is not None
            and gross_cumulative is not None
            and net_cumulative > gross_cumulative
        ):
            raise V5ContractError(
                f"{metrics_context}.cumulative_return exceeds gross_cumulative_return"
            )
        if weeks == 0:
            if annualized_turnover is not None or not math.isclose(
                transaction_cost_sum,
                0.0,
                abs_tol=1e-12,
            ):
                raise V5ContractError(
                    f"{metrics_context} empty metrics have non-empty turnover/cost"
                )
        elif net_cumulative is None or annualized_turnover is None:
            raise V5ContractError(f"{metrics_context} populated metrics are incomplete")
    assert common_weeks is not None
    if is_v2:
        if common_weeks == 0:
            if evaluation_start is not None or evaluation_end is not None:
                raise V5ContractError(f"{context} empty evaluation has date bounds")
        else:
            if evaluation_start is None or evaluation_end is None:
                raise V5ContractError(f"{context} populated evaluation lacks date bounds")
            expected_weeks = (evaluation_end - evaluation_start).days // 7 + 1
            if (
                (evaluation_end - evaluation_start).days % 7 != 0
                or expected_weeks != common_weeks
            ):
                raise V5ContractError(
                    f"{context} evaluation bounds do not match strategy weeks"
                )
    expected_historical_status = (
        "completed"
        if common_weeks >= minimum_evaluation_weeks
        else "insufficient_history"
    )
    if historical.get("status") != expected_historical_status:
        raise V5ContractError(f"{context} historical status is inconsistent")
    prospective = _mapping(
        shadow.get("prospective_ledger"), f"{context}.prospective_ledger"
    )
    expected_prospective_fields = {
        "status",
        "evidence_track",
        "ledger_entry_count",
        "realized_evaluation_count",
        "affects_official_forecast",
        "affects_champion_selection",
    }
    if is_v2:
        expected_prospective_fields.update(
            {
                "pending_evaluation_count",
                "unresolved_due_evaluation_count",
                "partial_evaluation_count",
                "evaluation_manifest_sha256",
                "performance",
            }
        )
    if set(prospective) != expected_prospective_fields:
        raise V5ContractError(f"{context} prospective fields are invalid")
    if (
        prospective.get("evidence_track") != "operational_oos"
        or prospective.get("affects_official_forecast") is not False
        or prospective.get("affects_champion_selection") is not False
    ):
        raise V5ContractError(f"{context} prospective isolation is invalid")
    ledger_entry_count = _integer(
        prospective.get("ledger_entry_count"),
        f"{context}.prospective_ledger.ledger_entry_count",
    )
    realized_evaluation_count = _integer(
        prospective.get("realized_evaluation_count"),
        f"{context}.prospective_ledger.realized_evaluation_count",
    )
    if realized_evaluation_count > ledger_entry_count:
        raise V5ContractError(f"{context} prospective ledger counts are inconsistent")
    ledger_status = prospective.get("status")
    if is_v2:
        pending_count = _integer(
            prospective.get("pending_evaluation_count"),
            f"{context}.prospective_ledger.pending_evaluation_count",
        )
        unresolved_count = _integer(
            prospective.get("unresolved_due_evaluation_count"),
            f"{context}.prospective_ledger.unresolved_due_evaluation_count",
        )
        partial_count = _integer(
            prospective.get("partial_evaluation_count"),
            f"{context}.prospective_ledger.partial_evaluation_count",
        )
        _sha256(
            prospective.get("evaluation_manifest_sha256"),
            f"{context}.prospective_ledger.evaluation_manifest_sha256",
        )
        if (
            pending_count + unresolved_count + realized_evaluation_count + partial_count
            != ledger_entry_count
        ):
            raise V5ContractError(
                f"{context} prospective ledger counts are inconsistent"
            )
        expected_status = (
            "completed"
            if ledger_entry_count > 0
            and realized_evaluation_count == ledger_entry_count
            else "pending"
            if pending_count == ledger_entry_count
            else "partial"
        )
        if ledger_status != expected_status:
            raise V5ContractError(f"{context} prospective status is inconsistent")
        _validate_prospective_performance(
            prospective.get("performance"),
            context=f"{context}.prospective_ledger.performance",
            ledger_status=str(ledger_status),
            realized_count=realized_evaluation_count,
        )
        if forecast is not None:
            public_ledger = forecast.get("prospective_ledger")
            if (
                isinstance(public_ledger, Mapping)
                and public_ledger.get("schema_version")
                == "regime-prospective-ledger-summary/2"
            ):
                expected_copy = {
                    "status": (
                        "pending"
                        if public_ledger.get("status") == "empty"
                        else public_ledger.get("status")
                    ),
                    "ledger_entry_count": public_ledger.get("entry_count"),
                    "pending_evaluation_count": public_ledger.get(
                        "pending_evaluation_count"
                    ),
                    "unresolved_due_evaluation_count": public_ledger.get(
                        "unresolved_due_evaluation_count"
                    ),
                    "realized_evaluation_count": public_ledger.get(
                        "realized_evaluation_count"
                    ),
                    "partial_evaluation_count": public_ledger.get(
                        "partial_evaluation_count"
                    ),
                    "evaluation_manifest_sha256": public_ledger.get(
                        "evaluation_manifest_sha256"
                    ),
                    "performance": public_ledger.get("performance"),
                }
                if any(prospective.get(key) != value for key, value in expected_copy.items()):
                    raise V5ContractError(
                        f"{context} prospective ledger differs from forecast summary"
                    )
        return
    if ledger_status == "awaiting_realized_targets":
        status_is_consistent = (
            ledger_entry_count == 0 and realized_evaluation_count == 0
        )
    elif ledger_status == "ledger_recorded_outcomes_pending":
        status_is_consistent = (
            ledger_entry_count > 0
            and realized_evaluation_count < ledger_entry_count
        )
    else:
        raise V5ContractError(f"{context} prospective status is invalid")
    if not status_is_consistent:
        raise V5ContractError(f"{context} prospective status/counts are inconsistent")


def _validate_label_sensitivity(research: Mapping[str, Any], label: Mapping[str, Any]) -> None:
    raw = research.get("label_sensitivity")
    if raw is None:
        return
    context = "payload.research.label_sensitivity"
    summary = _mapping(raw, context)
    if (
        summary.get("schema_version") != "regime-label-sensitivity-summary/1"
        or summary.get("status") not in {
            "preregistered_pending_execution",
            "completed",
        }
        or summary.get("evidence_track") != "reconstructed_oos"
        or summary.get("evaluation_split") != "selection_only"
        or summary.get("automatic_promotion_eligible") is not False
    ):
        raise V5ContractError(f"{context} identity is invalid")
    control = _mapping(summary.get("control"), f"{context}.control")
    if (
        control.get("spec_id") != label.get("spec_id")
        or control.get("spec_version") != label.get("spec_version")
        or control.get("spec_sha256") != label.get("spec_sha256")
        or control.get("remains_operating_control") is not True
    ):
        raise V5ContractError(f"{context}.control differs from payload.label")
    grid = _mapping(summary.get("grid"), f"{context}.grid")
    _sha256(grid.get("sha256"), f"{context}.grid.sha256")
    execution = _mapping(
        summary.get("execution_summary"), f"{context}.execution_summary"
    )
    required = {
        "evaluated_spec_count",
        "state_occupancy",
        "episode_count",
        "weekly_flip_rate",
        "transition_jaccard",
        "forward_return_separation",
        "model_rank_robustness",
    }
    if set(execution) != required:
        raise V5ContractError(f"{context}.execution_summary fields are invalid")
    evaluated = _integer(
        execution.get("evaluated_spec_count"),
        f"{context}.execution_summary.evaluated_spec_count",
    )
    if summary.get("status") == "preregistered_pending_execution" and (
        evaluated != 0
        or any(execution[field] is not None for field in required if field != "evaluated_spec_count")
    ):
        raise V5ContractError(f"{context} pending execution summary is inconsistent")


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
    has_review = "publication_review" in meta
    has_manifest = "generation_manifest_sha256" in meta
    expected_meta_fields = (
        V5_PUBLICATION_MANIFEST_META_FIELDS
        if has_review and has_manifest
        else V5_PUBLICATION_META_FIELDS
        if has_review
        else V5_MANIFEST_META_FIELDS
        if has_manifest
        else V5_META_FIELDS
    )
    if set(meta) != expected_meta_fields:
        raise V5ContractError("payload.meta fields are invalid")
    if has_manifest:
        _sha256(
            meta.get("generation_manifest_sha256"),
            "payload.meta.generation_manifest_sha256",
        )

    states = _sequence(_require(payload, "states", "payload"), "payload.states")
    if [dict(item) for item in states if isinstance(item, Mapping)] != [
        dict(item) for item in _OPERATING_CONTRACT.state_definitions
    ]:
        raise V5ContractError("payload.states differ from operating-contract.json")
    _validate_label_contract(_require(payload, "label", "payload"))
    forecast_origin, forecast_target = _validate_forecast_contract(
        _require(payload, "forecast", "payload"), mode=str(mode)
    )
    model = _mapping(_require(payload, "model", "payload"), "payload.model")
    evidence_week_count = _validate_model(model, mode=str(mode))
    try:
        lifecycle = validate_lifecycle_consistency(payload)
    except IntegrityError as exc:
        raise V5ContractError(str(exc)) from exc
    if lifecycle["publication"] == V5_PUBLICATION_STATUS and not has_manifest:
        raise V5ContractError(
            "reviewed publication requires generation_manifest_sha256"
        )
    is_reviewed_publication = (
        meta.get("publication_status") == V5_PUBLICATION_STATUS
    )
    # For reviewed output, validate policy identity before recomputing the
    # richer selection explanation.  A forged legacy 0.05 policy must fail for
    # that exact reason rather than being masked by downstream metric changes.
    # Unpublished candidates retain the earlier diagnostic ordering so drift
    # in their selection evidence is reported at its narrowest contract edge.
    if is_reviewed_publication:
        _validate_publication_review(meta, model, mode=str(mode), payload=payload)
    selection = _mapping(
        _require(payload, "selection", "payload"),
        "payload.selection",
    )
    _validate_selection_contract(selection, model)
    forecast_comparison_models = _validate_forecast_comparison(model)
    if not is_reviewed_publication:
        _validate_publication_review(meta, model, mode=str(mode), payload=payload)

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
        if "transition_term_structure" in row:
            _validate_transition_term_structure(
                row["transition_term_structure"],
                f"{context}.transition_term_structure",
                probabilities=departure,
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
        if "context_score_coverage" in row:
            _validate_context_score_coverage(
                scores,
                row["context_score_coverage"],
                f"{context}.context_score_coverage",
            )
        else:
            for name, value in scores.items():
                _number(
                    value,
                    f"{context}.context_scores.{name}",
                    minimum=-1.0,
                    maximum=1.0,
                )
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
    latest_week = _mapping(weekly[-1], "payload.weekly[-1]")
    latest_origin = _iso_date(latest_week["date"], "payload.weekly[-1].date")
    latest_target = _iso_date(
        _mapping(latest_week["next_week"], "payload.weekly[-1].next_week")["date"],
        "payload.weekly[-1].next_week.date",
    )
    if forecast_origin.date() != latest_origin or forecast_target.date() != latest_target:
        raise V5ContractError("payload.forecast differs from the latest weekly row")

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
    _validate_decision_shadow(
        _mapping(_require(payload, "research", "payload"), "payload.research"),
        latest_week=_mapping(weekly[-1], "payload.weekly[-1]"),
        forecast=_mapping(payload["forecast"], "payload.forecast"),
        operating_champion=str(selection["operating_champion"]),
        generated_at=generated_at,
    )
    _validate_label_sensitivity(
        _mapping(_require(payload, "research", "payload"), "payload.research"),
        _mapping(_require(payload, "label", "payload"), "payload.label"),
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
    "V5_MAXIMUM_PROMOTION_ALPHA",
    "V5_MAXIMUM_PROMOTION_BRIER_DEGRADATION",
    "V5_MINIMUM_PROMOTION_LOG_LOSS_IMPROVEMENT",
    "V5_MODEL_VERSION",
    "V5_MULTISCALE_MODEL",
    "V5_PUBLICATION_REVIEW_SCHEMA",
    "V5_PUBLICATION_STATUS",
    "V5_RESULT_VERSION",
    "V5_SCHEMA_VERSION",
    "validate_v5_champion_selection_evidence",
    "validate_v5_payload",
]
