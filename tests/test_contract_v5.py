from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from regime_lab.contract_v5 import (
    V5_FORECAST_COMPARISON_MODELS,
    V5ContractError,
    _validate_publication_review,
    validate_v5_payload,
)
from regime_lab.frozen_v4 import FROZEN_V4_BASELINE
from regime_lab.pipeline import STRUCTURAL_V5_PREREGISTRATION_SHA256
from regime_lab.schema import validate_dashboard_payload
from regime_lab.v5 import _execution_parameters


def _states() -> list[dict[str, str]]:
    return [
        {"id": "risk_on"},
        {"id": "transition"},
        {"id": "risk_off"},
    ]


def _model_forecasts(official: dict[str, object]) -> list[dict[str, object]]:
    probabilities = {
        "xgboost": {"risk_on": 0.2, "transition": 0.7, "risk_off": 0.1},
        "xgb_hazard_destination": {
            "risk_on": 0.25,
            "transition": 0.65,
            "risk_off": 0.1,
        },
        "causal_dynamic_ensemble": {
            "risk_on": 0.15,
            "transition": 0.75,
            "risk_off": 0.1,
        },
        "causal_multiscale_ensemble": {
            "risk_on": 0.18,
            "transition": 0.72,
            "risk_off": 0.1,
        },
    }
    rows = []
    for model in V5_FORECAST_COMPARISON_MODELS:
        if model == "markov":
            row = deepcopy(official)
        else:
            row_probabilities = probabilities[model]
            row = {
                "state": "transition",
                "probabilities": row_probabilities,
                "confidence": row_probabilities["transition"],
                "entropy": 0.75,
                "date": official["date"],
                "model": model,
                "fallback": False,
                "fallback_reason": "",
            }
        row["method"] = "model_comparison_walk_forward_probability"
        rows.append(row)
    return rows


def _conditional_rows() -> list[dict[str, object]]:
    metrics = (
        "mean_return",
        "median_return",
        "positive_rate",
        "annualized_volatility",
        "downside_volatility",
        "cvar_5",
        "mean_max_drawdown",
    )
    rows: list[dict[str, object]] = []
    for asset in ("SPY", "QQQ", "IWM", "TLT", "HYG", "UUP"):
        for state in ("risk_on", "transition", "risk_off"):
            for horizon in (1, 4, 13):
                row: dict[str, object] = {
                    "asset": asset,
                    "state": state,
                    "horizon_weeks": horizon,
                    "execution_lag_weeks": 1,
                    "return_currency": "USD",
                    "sample_start": None,
                    "sample_end": None,
                    "n": 0,
                    "unique_episodes": 0,
                    "status": "insufficient_support",
                    "minimum_observations": 20,
                    "minimum_unique_episodes": 5,
                    "bootstrap_method": "episode_bounded_circular_block",
                    "bootstrap_block_weeks": 13,
                    "bootstrap_resamples": 1_999,
                    "bootstrap_seed": 17,
                }
                for metric in metrics:
                    row[metric] = None
                    row[f"{metric}_ci95_lower"] = None
                    row[f"{metric}_ci95_upper"] = None
                rows.append(row)
    return rows


def _research_artifacts() -> dict[str, dict[str, object]]:
    paths = {
        "directional_oos_predictions": "directional-oos-predictions.csv",
        "directional_model_leaderboard": "directional-model-leaderboard.csv",
        "directional_walk_forward_splits": "directional-walk-forward-splits.csv",
        "directional_selection_diagnostics": "directional-selection-diagnostics.csv",
        "directional_forecasts": "directional-forecasts.csv",
        "conditional_asset_outcomes": "conditional-asset-outcomes.csv",
        "conditional_asset_statistics": "conditional-asset-statistics.csv",
    }
    counts = {
        "directional_oos_predictions": 10,
        "directional_model_leaderboard": 1,
        "directional_walk_forward_splits": 10,
        "directional_selection_diagnostics": 1,
        "directional_forecasts": 3,
        "conditional_asset_outcomes": 100,
        "conditional_asset_statistics": 54,
    }
    return {
        key: {"path": path, "row_count": counts[key], "sha256": "f" * 64}
        for key, path in paths.items()
    }


def _add_model_conditioned_stats(payload: dict[str, object]) -> None:
    rows = [
        {"conditioning_model": model, **deepcopy(row)}
        for model in V5_FORECAST_COMPARISON_MODELS
        for row in _conditional_rows()
    ]
    payload["research"]["model_conditioned_asset_stats"] = {
        "method": "oos_one_week_forecast_conditioned_forward_total_return",
        "role": "retrospective_model_diagnostic",
        "conditioning": "hard_argmax_oos_forecast",
        "forecast_horizon_weeks": 1,
        "execution_lag_weeks": 1,
        "horizons_weeks": [1, 4, 13],
        "assets": ["SPY", "QQQ", "IWM", "TLT", "HYG", "UUP"],
        "models": list(V5_FORECAST_COMPARISON_MODELS),
        "return_currency": "USD",
        "rows": rows,
    }
    payload["model"]["research_artifacts"].update(
        {
            "model_conditioned_asset_outcomes": {
                "path": "model-conditioned-asset-outcomes.csv",
                "row_count": 100,
                "sha256": "e" * 64,
            },
            "model_conditioned_asset_statistics": {
                "path": "model-conditioned-asset-statistics.csv",
                "row_count": len(rows),
                "sha256": "d" * 64,
            },
        }
    )


def _core_artifacts() -> dict[str, dict[str, object]]:
    paths = {
        "oos_predictions": "oos-predictions.csv",
        "model_leaderboard": "model-leaderboard.csv",
        "walk_forward_splits": "walk-forward-splits.csv",
        "selection_diagnostics": "selection-diagnostics.csv",
        "stacking_weights": "stacking-weights.csv",
        "multiscale_ensemble_scales": "multiscale-ensemble-scales.csv",
    }
    return {
        key: {"path": path, "row_count": 1, "sha256": "a" * 64}
        for key, path in paths.items()
    }


def _candidate_manifest() -> tuple[dict[str, object], str]:
    names = (
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
        "causal_multiscale_ensemble",
    )
    manifest: dict[str, object] = {
        "schema_version": "1.0.0",
        "profile": "standard",
        "random_state": 17,
        "models": [{"name": name} for name in names],
    }
    encoded = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return manifest, hashlib.sha256(encoded).hexdigest()


def _structural_models() -> dict[str, object]:
    sidecar = _core_artifacts()["multiscale_ensemble_scales"]
    return {
        "causal_multiscale_ensemble": {
            "role": "v5_opt_in_candidate",
            "experts": ["markov", "xgboost", "xgb_hazard_destination"],
            "scale_half_lives_weeks": [26, 52, 104],
            "outer_scale_weights": [1.0 / 3.0] * 3,
            "aggregation": "fixed_equal_probability_average",
            "inner_pool_method": "causal_discounted_completed_oos_log_score",
            "minimum_history_rows": 26,
            "eligible_loss_rule": "target_date_strictly_before_origin",
            "selection_gate": (
                "existing_multiclass_holm_log_loss_brier_zero_fallback"
            ),
            "automatic_promotion_bypass": False,
            "sidecar": sidecar,
        }
    }


def _unavailable_fx_ablation() -> dict[str, object]:
    return {
        "role": "prospective_shadow",
        "variants": [
            "v4_control",
            "v4_plus_broad_index",
            "v4_plus_bilateral_panel",
            "v4_plus_all_fx",
        ],
        "minimum_common_weeks": 156,
        "historical_availability_backfill": False,
        "official_release_archive_ingest": False,
        "availability_basis": "collection_first_seen_at",
        "archive_revision_policy": (
            "later_official_release_preserved_as_new_vintage"
        ),
        "archive_correction_availability_basis": (
            "date_only_conservative_next_day"
        ),
        "status": "unavailable",
        "eligible_common_weeks": 0,
        "first_eligible_cutoff": None,
        "last_eligible_cutoff": None,
        "manifest": [],
        "status_reason": "fx_feature_result_unavailable",
        "common_origin_required_pairs": 9,
        "minimum_train_weeks": 104,
        "target_horizon_weeks": 1,
        "purge_weeks": 1,
        "target_availability_rule": (
            "last_train_target_strictly_before_evaluation_origin"
        ),
        "model": {
            "name": "fixed_l2_multinomial_logistic",
            "horizon_weeks": 1,
            "multiclass": "multinomial",
            "regularization": "l2",
            "regularization_c": 0.1,
            "class_weight": None,
            "solver": "lbfgs",
            "max_iter": 2000,
            "tolerance": 1e-6,
            "random_state": 17,
            "imputation": "expanding_train_median",
            "scaling": "expanding_train_standard",
            "fit_window": "expanding",
            "state_order": ["risk_on", "transition", "risk_off"],
        },
        "common_evaluation_origins": {
            "count": 0,
            "first_origin": None,
            "last_origin": None,
            "sha256": None,
            "rows": [],
        },
        "variant_metrics": [],
        "gate": {
            "reference_variant": "v4_control",
            "method": "paired_circular_moving_block_bootstrap_holm",
            "bootstrap_block_weeks": 13,
            "bootstrap_effective_block_weeks": None,
            "bootstrap_resamples": 1999,
            "bootstrap_seed": 17,
            "alpha": 0.05,
            "minimum_log_loss_improvement": 0.05,
            "brier_tolerance": 0.01,
            "comparisons": [],
            "passed_variants": [],
        },
        "promotion_allowed": False,
        "promotion_candidate": None,
        "core_champion_promoted": False,
    }


def valid_payload() -> dict[str, object]:
    candidate_manifest, candidate_manifest_sha256 = _candidate_manifest()
    transition_risk = {
        horizon: {
            "probability": probability,
            "target_end": target,
            "model": "markov",
            "threshold": 0.5,
            "fallback": False,
            "fallback_reason": "",
        }
        for horizon, probability, target in (
            ("1w", 0.2, "2026-08-14"),
            ("4w", 0.4, "2026-09-04"),
            ("13w", 0.7, "2026-11-06"),
        )
    }
    directional_risk = {
        horizon: {
            "probability": probability,
            "no_departure": 1.0 - probability,
            "first_destination": {
                "risk_on": 0.0,
                "transition": probability * 0.75,
                "risk_off": probability * 0.25,
            },
            "target_end": target,
            "model": "markov_first_passage",
            "method": "first_departure_state_within_h_or_no_departure",
        }
        for horizon, probability, target in (
            ("1w", 0.2, "2026-08-14"),
            ("4w", 0.4, "2026-09-04"),
            ("13w", 0.7, "2026-11-06"),
        )
    }
    next_week: dict[str, object] = {
        "state": "risk_on",
        "probabilities": {
            "risk_on": 0.6,
            "transition": 0.25,
            "risk_off": 0.15,
        },
        "confidence": 0.6,
        "entropy": 0.85,
        "date": "2026-08-14",
        "method": "champion_walk_forward_probability",
        "model": "markov",
        "fallback": False,
        "fallback_reason": "",
    }
    return {
        "meta": {
            "schema_version": "2.0.0",
            "result_version": "weekly-regime-result-v5",
            "generated_at": "2026-08-08T01:00:00+00:00",
            "data_as_of": "2026-08-07T20:00:00+00:00",
            "generation_id": "v5-test",
            "mode": "demo",
            "status": "ok",
            "warnings": [],
            "timezone": "America/New_York",
            "cutoff_policy": "completed US market week",
            "transition_alert_thresholds": {"medium": 0.4, "high": 0.65},
            "transition_probability_definition": "one-week first departure",
            "transition_risk_definition": "first departure within horizon",
            "supported_date_range": "2026-08-07–2026-08-07",
            "current_membership_definition": "soft causal membership",
            "freshness": {
                "cadence": "weekly",
                "maximum_age_days": 10,
                "age_days": 0,
                "status": "current",
                "data_as_of": "2026-08-07T20:00:00+00:00",
            },
        },
        "states": _states(),
        "model": {
            "champion": "markov",
            "version": "weekly-nondl-structural-v5",
            "label_version": "market-causal-3state-v1",
            "feature_set_version": "weekly-pit-structural-v5",
            "selection_status": "provisional_predeployment",
            "profile": "standard",
            "candidate_manifest": candidate_manifest,
            "candidate_manifest_sha256": candidate_manifest_sha256,
            "structural_models": _structural_models(),
            "leaderboard": [
                {"name": name} for name in V5_FORECAST_COMPARISON_MODELS
            ],
            "forecast_comparison": {
                "role": "research_comparison",
                "horizon_weeks": 1,
                "models": list(V5_FORECAST_COMPARISON_MODELS),
            },
            "baseline_v4": dict(FROZEN_V4_BASELINE),
            "structural_preregistration": {
                "path": "config/structural_v5.json",
                "sha256": "c" * 64,
            },
            "execution_parameters": _execution_parameters(
                "standard",
                duration_bootstrap_resamples=1_999,
                outcome_bootstrap_resamples=1_999,
            ),
            "directional_transition": {
                "target": "first_departure_state_within_h_or_no_departure",
                "deployed_direction_role": "first_destination_given_departure",
                "selection_metric": "conditional_destination_log_loss",
                "minimum_selection_departure_events": 8,
                "minimum_selection_destination_classes": 2,
                "minimum_selection_event_blocks": 3,
                "champions": {
                    "1w": "markov_first_passage",
                    "4w": "markov_first_passage",
                    "13w": "markov_first_passage",
                },
                "leaderboard": [{}],
                "selection_diagnostics": [{}],
                "selection_end": "2023-01-01",
            },
            "model_health": {"status": "ok", "reasons": []},
            "champion_core_feature_set_version": "weekly-pit-structural-v4",
            "fx_role": "context_and_preregistered_shadow_ablation",
            "fx_ablation": _unavailable_fx_ablation(),
            "evidence_artifacts": {
                "state_membership_history": {
                    "path": "state-membership-history.csv",
                    "row_count": 700,
                    "sha256": "d" * 64,
                    "label_fit_weeks": 520,
                    "label_fit_end": "2021-12-31T21:00:00+00:00",
                    "initial_state": "transition",
                    "method": "risk_score_anchor_membership",
                },
                "weekly_state_forecasts": {
                    "path": "weekly-state-forecasts-v5.csv",
                    "row_count": 1,
                    "sha256": "e" * 64,
                },
            },
            "core_artifacts": _core_artifacts(),
            "research_artifacts": _research_artifacts(),
        },
        "weekly": [
            {
                "date": "2026-08-07",
                "current": {
                    "state": "risk_on",
                    "memberships": {
                        "risk_on": 0.7,
                        "transition": 0.2,
                        "risk_off": 0.1,
                    },
                    "primary_membership": 0.7,
                    "membership_entropy": 0.73,
                    "method": "risk_score_anchor_membership",
                },
                "next_week": next_week,
                "model_forecasts": _model_forecasts(next_week),
                "transition_probability": 0.2,
                "transition_risk": transition_risk,
                "directional_risk": directional_risk,
                "duration_context": {
                    "as_of": "2026-08-07",
                    "status": "ok",
                    "method": "state_specific_kaplan_meier",
                    "state": "risk_on",
                    "elapsed_weeks": 8,
                    "episodes": 20,
                    "completed_spells": 19,
                    "censored_spells": 1,
                    "minimum_completed_spells": 5,
                    "bootstrap": {
                        "unit": "episode",
                        "resamples": 0,
                        "valid_resamples": 0,
                        "seed": 17,
                        "interval": 0.95,
                    },
                    "conditional_survival": {"4w": 0.8, "13w": 0.5},
                    "departure_probability": {"4w": 0.2, "13w": 0.5},
                    "median_remaining_weeks": 13,
                    "restricted_mean_remaining_weeks": 21.5,
                    "restriction_weeks": 52,
                    "ci95": None,
                },
                "fx_context": {
                    "status": "unavailable",
                    "method": "fed_h10_usd_strength",
                    "bilateral_panel": [
                        "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "CNY", "MXN", "BRL"
                    ],
                    "coverage": {"available_pairs": 0, "required_pairs": 9},
                    "indexes": {},
                    "bilateral": {},
                },
                "context_scores": {
                    "trend": 0.5,
                    "stress": -0.1,
                    "macro": 0.2,
                    "financial_conditions": 0.1,
                },
                "extreme_context": [
                    {
                        "feature": "spy_close__return_13w",
                        "label": "SPY 13주 수익률",
                        "z_score": 1.3,
                        "position": "high",
                        "method": "trailing_52w_z_score",
                    }
                ],
                "summary": "위험선호 멤버십이 가장 높고 4주 이탈위험은 40%입니다.",
                "market": {},
                "health": {},
            }
        ],
        "sources": [
            {
                "id": "synthetic_market",
                "status": "degraded",
                "license_class": "synthetic_fixture",
            },
            {
                "id": "synthetic_macro",
                "status": "degraded",
                "license_class": "synthetic_fixture",
            },
        ],
        "feature_catalog": [
            {
                "id": "fixture",
                "category": "test",
                "frequency": "weekly",
                "source": "fixture",
            }
        ],
        "research": {
            "conditional_asset_stats": {
                "method": "state_conditioned_forward_total_return",
                "role": "descriptive_only",
                "execution_lag_weeks": 1,
                "horizons_weeks": [1, 4, 13],
                "assets": ["SPY", "QQQ", "IWM", "TLT", "HYG", "UUP"],
                "return_currency": "USD",
                "rows": _conditional_rows(),
            }
        },
    }


def test_valid_v5_payload() -> None:
    payload = valid_payload()
    validate_v5_payload(payload)
    validate_dashboard_payload(payload)


def test_valid_v5_payload_with_model_conditioned_stats() -> None:
    payload = valid_payload()
    _add_model_conditioned_stats(payload)

    validate_v5_payload(payload)


def test_tracked_public_v5_snapshot_still_validates() -> None:
    path = Path(__file__).parents[1] / "publication" / "live" / "regime-results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    validate_v5_payload(payload)


def test_v5_forecast_comparison_remains_optional_for_older_payloads() -> None:
    payload = valid_payload()
    payload["model"].pop("forecast_comparison")
    for week in payload["weekly"]:
        week.pop("model_forecasts")

    validate_v5_payload(payload)


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        (
            lambda value: value["weekly"][0]["model_forecasts"].__setitem__(
                slice(0, 2),
                list(reversed(value["weekly"][0]["model_forecasts"][:2])),
            ),
            "model order is invalid",
        ),
        (
            lambda value: value["weekly"][0]["model_forecasts"].pop(),
            "model count is invalid",
        ),
        (
            lambda value: value["weekly"][0]["model_forecasts"][1].update(
                {"date": "2026-08-21"}
            ),
            "date is inconsistent",
        ),
        (
            lambda value: value["weekly"][0]["model_forecasts"][1][
                "probabilities"
            ].update({"risk_on": 0.3}),
            "probabilities must sum to one",
        ),
        (
            lambda value: value["weekly"][0]["model_forecasts"][0].update(
                {"fallback": True, "fallback_reason": "test"}
            ),
            "champion forecast differs from next_week",
        ),
    ),
)
def test_v5_forecast_comparison_rejects_drift(mutate, match: str) -> None:
    payload = valid_payload()
    mutate(payload)

    with pytest.raises(V5ContractError, match=match):
        validate_v5_payload(payload)


def test_v5_forecast_comparison_rejects_metadata_order_and_orphan_rows() -> None:
    payload = valid_payload()
    payload["model"]["forecast_comparison"]["models"][:2] = list(
        reversed(payload["model"]["forecast_comparison"]["models"][:2])
    )
    with pytest.raises(V5ContractError, match="models are invalid"):
        validate_v5_payload(payload)

    payload = valid_payload()
    payload["model"].pop("forecast_comparison")
    with pytest.raises(V5ContractError, match="requires model forecast_comparison"):
        validate_v5_payload(payload)


def test_v5_freshness_is_recomputed_from_timestamp_contract() -> None:
    payload = valid_payload()
    payload["meta"]["freshness"]["age_days"] = 11
    payload["meta"]["freshness"]["status"] = "stale"

    with pytest.raises(V5ContractError, match="freshness.age_days"):
        validate_v5_payload(payload)


@pytest.mark.parametrize("warnings", (None, ["ok", 3]))
def test_v5_warnings_are_a_string_array(warnings) -> None:
    payload = valid_payload()
    if warnings is None:
        payload["meta"].pop("warnings")
    else:
        payload["meta"]["warnings"] = warnings

    with pytest.raises(V5ContractError, match="payload.meta.warnings"):
        validate_v5_payload(payload)


def test_v5_execution_parameters_are_hash_bound() -> None:
    payload = valid_payload()
    payload["model"]["execution_parameters"][
        "duration_bootstrap_resamples"
    ] = 199

    with pytest.raises(V5ContractError, match="preregistration_overrides"):
        validate_v5_payload(payload)

    payload = valid_payload()
    payload["model"]["execution_parameters"]["sha256"] = "0" * 64
    with pytest.raises(V5ContractError, match="sha256 is inconsistent"):
        validate_v5_payload(payload)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda value: value["meta"].update({"mode": "live"}),
            "sources identities are invalid for mode=live",
        ),
        (
            lambda value: value["model"].pop("profile"),
            "payload.model.profile is required",
        ),
        (
            lambda value: value["model"].update({"profile": "full"}),
            "profile must match execution_parameters.profile",
        ),
    ],
)
def test_v5_mode_profile_and_source_identity_are_bound(mutate, match: str) -> None:
    payload = valid_payload()
    mutate(payload)

    with pytest.raises(V5ContractError, match=match):
        validate_v5_payload(payload)


def _live_sources() -> list[dict[str, object]]:
    return [
        {
            "id": "alpha_vantage",
            "status": "ok",
            "license_class": "private_noncommercial",
        },
        {
            "id": "alfred",
            "status": "ok",
            "license_class": "user_confirmed_ml_storage_derived",
        },
        {
            "id": "frb_h10",
            "status": "unavailable",
            "license_class": (
                "federal_reserve_board_public_domain_citation_requested"
            ),
            "official_release_archive_ingest": False,
            "availability_basis": "collection_first_seen_at",
            "archive_revision_policy": (
                "later_official_release_preserved_as_new_vintage"
            ),
            "archive_correction_availability_basis": (
                "date_only_conservative_next_day"
            ),
            "archive_release_count": 0,
            "archive_correction_count": 0,
            "archive_correction_available_at": [],
            "archive_correction_quarantine_weeks": 27,
            "archive_evaluation_start": "2022-01-01",
            "archive_evaluation_start_rationale": (
                "post_2019_06_24_jan06_index_rebase_common_scale"
            ),
        },
    ]


def test_v5_live_contract_rejects_quick_profile() -> None:
    payload = valid_payload()
    payload["meta"]["mode"] = "live"
    payload["sources"] = _live_sources()
    payload["model"]["profile"] = "quick"
    payload["model"]["execution_parameters"] = _execution_parameters(
        "quick",
        duration_bootstrap_resamples=1_999,
        outcome_bootstrap_resamples=1_999,
    )

    with pytest.raises(V5ContractError, match="does not permit the quick profile"):
        validate_v5_payload(payload)


def test_v5_live_contract_binds_archive_correction_provenance() -> None:
    payload = valid_payload()
    payload["meta"]["mode"] = "live"
    payload["sources"] = _live_sources()
    ablation = payload["model"]["fx_ablation"]
    ablation["official_release_archive_ingest"] = True
    ablation["availability_basis"] = "official_archive_release_schedule"
    h10 = payload["sources"][2]
    h10["official_release_archive_ingest"] = True
    h10["availability_basis"] = "official_archive_release_schedule"
    h10["archive_release_count"] = 2
    h10["archive_correction_count"] = 1
    h10["archive_correction_available_at"] = ["2023-01-10T05:00:00Z"]

    validate_v5_payload(payload)

    h10["archive_correction_count"] = 0
    with pytest.raises(V5ContractError, match="release/correction counts"):
        validate_v5_payload(payload)


def test_v5_preregistration_hash_is_frozen_to_tracked_config() -> None:
    path = Path(__file__).parents[1] / "config" / "structural_v5.json"

    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        STRUCTURAL_V5_PREREGISTRATION_SHA256
    )


def test_v5_core_artifact_manifest_is_exact_and_hash_linked() -> None:
    payload = valid_payload()
    validate_v5_payload(payload)

    payload["model"]["core_artifacts"]["stacking_weights"]["path"] = (
        "other.csv"
    )
    with pytest.raises(V5ContractError, match="stacking_weights.path"):
        validate_v5_payload(payload)


def test_v5_fx_research_manifest_binds_non_evaluated_oos_to_zero_rows() -> None:
    payload = valid_payload()
    artifacts = payload["model"]["research_artifacts"]
    artifacts.update(
        {
            "fx_features": {
                "path": "fx-features.csv",
                "row_count": 50,
                "sha256": "1" * 64,
            },
            "fx_coverage": {
                "path": "fx-coverage.csv",
                "row_count": 50,
                "sha256": "2" * 64,
            },
            "fx_ablation_oos": {
                "path": "fx-ablation-oos.csv",
                "row_count": 0,
                "sha256": "3" * 64,
            },
        }
    )
    validate_v5_payload(payload)

    artifacts["fx_ablation_oos"]["row_count"] = 1
    with pytest.raises(V5ContractError, match="fx_ablation_oos.row_count"):
        validate_v5_payload(payload)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda value: value["weekly"][0]["current"].update(
                {"probabilities": value["weekly"][0]["current"].pop("memberships")}
            ),
            "current fields",
        ),
        (
            lambda value: value["weekly"][0]["next_week"].update(
                {"date": "2026-08-21"}
            ),
            "next_week.date is inconsistent",
        ),
        (
            lambda value: value["weekly"][0]["directional_risk"]["4w"][
                "first_destination"
            ].update({"transition": 0.1}),
            "first_destination must sum",
        ),
        (
            lambda value: value["weekly"][0]["duration_context"][
                "departure_probability"
            ].update({"4w": 0.3}),
            "duration values are inconsistent",
        ),
        (
            lambda value: value["model"]["fx_ablation"].update(
                {"historical_availability_backfill": True}
            ),
            "historical_availability_backfill must be false",
        ),
        (
            lambda value: value["model"]["fx_ablation"].update(
                {
                    "official_release_archive_ingest": True,
                    "availability_basis": "collection_first_seen_at",
                }
            ),
            "availability_basis is inconsistent",
        ),
        (
            lambda value: value["model"]["fx_ablation"].update(
                {"eligible_common_weeks": 156}
            ),
            "requires official archive provenance",
        ),
        (
            lambda value: value["model"]["fx_ablation"].update(
                {"common_origin_required_pairs": 6}
            ),
            "common_origin_required_pairs must be 9",
        ),
        (
            lambda value: value["model"]["fx_ablation"].update(
                {"core_champion_promoted": True}
            ),
            "promotion must remain disabled",
        ),
        (
            lambda value: value["model"]["baseline_v4"].update(
                {"payload_sha256": "f" * 64}
            ),
            "baseline_v4 must match the frozen v4 contract",
        ),
        (
            lambda value: value["model"]["research_artifacts"][
                "directional_forecasts"
            ].update({"path": "../directional-forecasts.csv"}),
            "directional_forecasts.path is invalid",
        ),
        (
            lambda value: value["model"]["research_artifacts"][
                "conditional_asset_statistics"
            ].update({"row_count": 53}),
            "conditional_asset_statistics.row_count is invalid",
        ),
    ],
)
def test_v5_contract_rejects_semantic_drift(mutate, match: str) -> None:
    value = deepcopy(valid_payload())
    mutate(value)
    with pytest.raises(V5ContractError, match=match):
        validate_v5_payload(value)


def test_conditional_stats_reject_allocation_fields() -> None:
    value = valid_payload()
    value["research"]["conditional_asset_stats"]["rows"] = [
        {
            "asset": "SPY",
            "state": "risk_on",
            "horizon_weeks": 1,
            "n": 50,
            "unique_episodes": 8,
            "status": "ok",
            "allocation": 0.75,
        }
    ]
    with pytest.raises(V5ContractError, match="allocation field"):
        validate_v5_payload(value)


@pytest.mark.parametrize(
    "stats_key",
    ("conditional_asset_stats", "model_conditioned_asset_stats"),
)
def test_conditional_stats_reject_raw_like_extra_row_fields(stats_key: str) -> None:
    value = valid_payload()
    if stats_key == "model_conditioned_asset_stats":
        _add_model_conditioned_stats(value)
    value["research"][stats_key]["rows"][0]["spy_close"] = 640.25

    with pytest.raises(V5ContractError, match=r"rows\[0\] fields are invalid"):
        validate_v5_payload(value)


@pytest.mark.parametrize(
    "stats_key",
    ("conditional_asset_stats", "model_conditioned_asset_stats"),
)
def test_conditional_stats_reject_extra_top_level_metadata(stats_key: str) -> None:
    value = valid_payload()
    if stats_key == "model_conditioned_asset_stats":
        _add_model_conditioned_stats(value)
    value["research"][stats_key]["raw_fields"] = ["spy_close"]

    with pytest.raises(V5ContractError, match=rf"{stats_key} fields are invalid"):
        validate_v5_payload(value)


@pytest.mark.parametrize("location", ("payload", "meta"))
def test_v5_contract_rejects_unreviewed_extension_fields(location: str) -> None:
    value = valid_payload()
    if location == "payload":
        value["data"] = [{"date": "2026-08-07", "value": 1.0}]
    else:
        value["meta"]["provider_panel"] = [1.0, 2.0]
    with pytest.raises(V5ContractError, match="fields are invalid"):
        validate_v5_payload(value)


def _review_contract() -> tuple[dict[str, object], dict[str, object]]:
    meta: dict[str, object] = {
        "publication_status": "reviewed_publication",
        "publication_review": {
            "schema_version": "regime-v5-publication-review/1",
            "decision": "publish_v5_research_snapshot",
            "reviewed_at": "2026-08-23T02:11:21+00:00",
            "reviewed_candidate_sha256": "a" * 64,
            "champion": "markov",
            "multiscale_promoted": False,
            "fx_promoted": False,
        },
    }
    model: dict[str, object] = {
        "champion": "markov",
        "leaderboard": [
            {"name": "markov", "selected": True, "is_champion": True},
            {
                "name": "causal_multiscale_ensemble",
                "selected": False,
                "is_champion": False,
            },
        ],
        "selection_diagnostics": [
            {"model": "markov", "selected": True, "gate_passed": True},
            {
                "model": "causal_multiscale_ensemble",
                "selected": False,
                "gate_passed": False,
            },
        ],
    }
    return meta, model


def test_reviewed_v5_contract_binds_markov_decision_evidence() -> None:
    meta, model = _review_contract()
    _validate_publication_review(meta, model, mode="live")


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        (
            lambda meta, model: (
                model.__setitem__("champion", "not_a_model"),
                meta["publication_review"].__setitem__("champion", "not_a_model"),
            ),
            "Markov champion",
        ),
        (
            lambda _meta, model: model["leaderboard"][1].__setitem__(
                "selected", True
            ),
            "leaderboard must select exactly one Markov",
        ),
        (
            lambda _meta, model: model["selection_diagnostics"][1].__setitem__(
                "gate_passed", True
            ),
            "Multiscale diagnostic must remain non-promoted",
        ),
    ),
)
def test_reviewed_v5_contract_rejects_decision_drift(mutate, match: str) -> None:
    meta, model = _review_contract()
    mutate(meta, model)
    with pytest.raises(V5ContractError, match=match):
        _validate_publication_review(meta, model, mode="live")
