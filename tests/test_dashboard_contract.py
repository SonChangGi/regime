"""Static contract checks for the dependency-free regime dashboard."""

from copy import deepcopy
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
HTML_PATH = WEB / "index.html"
CSS_PATH = WEB / "styles.css"
JS_PATH = WEB / "app.js"

V5_FORECAST_COMPARISON_MODELS = [
    "markov",
    "xgboost",
    "xgb_hazard_destination",
    "causal_dynamic_ensemble",
    "causal_multiscale_ensemble",
]


def _valid_v3_browser_payload() -> dict:
    current = {
        "state": "transition",
        "probabilities": {"risk_on": 0.25, "transition": 0.5, "risk_off": 0.25},
        "confidence": 0.5,
        "entropy": 0.95,
    }
    next_week = {
        "state": "transition",
        "date": "2026-08-14",
        "probabilities": {"risk_on": 0.1, "transition": 0.8, "risk_off": 0.1},
        "confidence": 0.8,
        "entropy": 0.5,
    }
    model = {
        "champion": "markov",
        "selection_status": "provisional_predeployment",
        "leaderboard": [],
        "version": "weekly-nondl-structural-v3",
        "label_version": "market-causal-3state-v1",
        "feature_set_version": "weekly-pit-market-internals-v3",
        "primary_horizon_weeks": 1,
        "transition_selection_end": "2023-01-01",
        "transition_horizons_weeks": [1, 4, 13],
        "baseline_v2": {
            "result_version": "weekly-regime-result-v2",
            "label_version": "market-causal-3state-v1",
            "model_version": "weekly-nondl-walkforward-v2",
            "champion": "markov",
            "payload_sha256": "a" * 64,
            "artifacts_inventory_sha256": "b" * 64,
        },
        "transition_champions": {"1w": "hazard", "4w": "hazard", "13w": "duration"},
        "transition_leaderboard": [
            {
                "horizon_weeks": horizon,
                "model": "hazard",
                "selected": True,
                "evaluation_split": "selection",
                "binary_log_loss": 1.2,
                "brier": 0.2,
                "average_precision": None,
                "precision": 0.0,
                "recall": 0.0,
                "false_alarms_per_year": 0.0,
                "n_predictions": 10,
                "event_count": 0,
                "non_event_count": 10,
                "fallback_count": 0,
                "calibration_fallback_count": 0,
            }
            for horizon in (1, 4, 13)
        ],
        "shadow_nowcast": {"status": "shadow_only", "canonical_target": False},
    }
    transition_risk = {
        "1w": {
            "probability": 0.2,
            "target_end": "2026-08-14",
            "model": "hazard",
            "threshold": 0.5,
            "fallback": False,
            "fallback_reason": "",
        },
        "4w": {
            "probability": 0.3,
            "target_end": "2026-09-04",
            "model": "hazard",
            "threshold": 0.5,
            "fallback": False,
            "fallback_reason": "",
        },
        "13w": {
            "probability": 0.5,
            "target_end": "2026-11-06",
            "model": "duration",
            "threshold": 0.5,
            "fallback": False,
            "fallback_reason": "",
        },
    }
    return {
        "meta": {
            "schema_version": "1.0.0",
            "result_version": "weekly-regime-result-v3",
            "generation_id": "20260812T000000Z-example",
            "generated_at": "2026-08-12T00:00:00Z",
            "data_as_of": "2026-08-07",
            "mode": "demo",
            "timezone": "America/New_York",
        },
        "states": [
            {"id": "risk_on"},
            {"id": "transition"},
            {"id": "risk_off"},
        ],
        "model": model,
        "weekly": [{
            "date": "2026-08-07",
            "current": current,
            "next_week": next_week,
            "transition_probability": 0.2,
            "transition_risk": transition_risk,
            "scores": {"trend": 0.1, "stress": -0.1, "macro": 0.0, "financial_conditions": 0.0},
        }],
        "sources": [{"id": "fixture", "status": "degraded"}],
        "feature_catalog": [{"id": "fixture"}],
    }


def _valid_v4_browser_payload() -> dict:
    payload = deepcopy(_valid_v3_browser_payload())
    payload["meta"]["result_version"] = "weekly-regime-result-v4"
    payload["model"].update(
        {
            "version": "weekly-nondl-structural-v4",
            "feature_set_version": "weekly-pit-structural-v4",
            "baseline_v3": {
                "result_version": "weekly-regime-result-v3",
                "label_version": "market-causal-3state-v1",
                "model_version": "weekly-nondl-structural-v3",
                "champion": "markov",
                "payload_sha256": "de93c585117b2784750f586a4f84ad99964c63081b252ad7affd7a75bd797095",
                "artifacts_inventory_sha256": "8ef3778cc8c36faff0c80e2bf094f1f11bd6966ab3b7b2d6edb84ba292aff6b9",
                "captured_at": "2026-08-13",
            },
            "structural_preregistration": {
                "path": "config/structural_v4.json",
                "sha256": "2f53ada564efca770261f16ce6eb16ec9c9782bde014de7a7d85b7b24dbe407b",
            },
            "feature_manifest_sha256": "f" * 64,
            "evidence_artifacts": {
                "state_label_history": {
                    "path": "state-label-history.csv",
                    "row_count": 700,
                    "sha256": "b" * 64,
                    "label_fit_weeks": 520,
                    "label_fit_end": "2021-12-17T00:00:00",
                    "initial_state": "transition",
                },
                "weekly_state_forecasts": {
                    "path": "weekly-state-forecasts.csv",
                    "row_count": len(payload["weekly"]),
                    "sha256": "c" * 64,
                },
            },
            "structural_models": {
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
            },
            "ablation": {
                "anchor_model": "xgboost",
                "reference_variant": "legacy_v3",
                "published_variant": "all_structural",
                "primary_period": "pre_2023_selection_oos",
                "post_2023_role": "retrospective_diagnostic_only",
                "may_change_published_variant": False,
                "manifest_sha256": "a" * 64,
            },
        }
    )
    return payload


def _unavailable_fx_ablation() -> dict:
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
        "target_availability_rule": "last_train_target_strictly_before_evaluation_origin",
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


def _valid_v5_browser_payload() -> dict:
    research_paths = {
        "directional_oos_predictions": "directional-oos-predictions.csv",
        "directional_model_leaderboard": "directional-model-leaderboard.csv",
        "directional_walk_forward_splits": "directional-walk-forward-splits.csv",
        "directional_selection_diagnostics": "directional-selection-diagnostics.csv",
        "directional_forecasts": "directional-forecasts.csv",
        "conditional_asset_outcomes": "conditional-asset-outcomes.csv",
        "conditional_asset_statistics": "conditional-asset-statistics.csv",
        "fx_features": "fx-features.csv",
        "fx_coverage": "fx-coverage.csv",
        "fx_ablation_oos": "fx-ablation-oos.csv",
    }
    research_counts = {
        "directional_oos_predictions": 10,
        "directional_model_leaderboard": 1,
        "directional_walk_forward_splits": 10,
        "directional_selection_diagnostics": 1,
        "directional_forecasts": 3,
        "conditional_asset_outcomes": 100,
        "conditional_asset_statistics": 54,
        "fx_features": 50,
        "fx_coverage": 50,
        "fx_ablation_oos": 0,
    }
    research_artifacts = {
        key: {"path": path, "row_count": research_counts[key], "sha256": "f" * 64}
        for key, path in research_paths.items()
    }
    core_paths = {
        "oos_predictions": "oos-predictions.csv",
        "model_leaderboard": "model-leaderboard.csv",
        "walk_forward_splits": "walk-forward-splits.csv",
        "selection_diagnostics": "selection-diagnostics.csv",
        "stacking_weights": "stacking-weights.csv",
        "multiscale_ensemble_scales": "multiscale-ensemble-scales.csv",
    }
    core_artifacts = {
        key: {"path": path, "row_count": 10, "sha256": "a" * 64}
        for key, path in core_paths.items()
    }
    candidate_names = [
        "majority", "persistence", "markov", "elastic_net_logistic",
        "calibrated_linear_svm", "random_forest", "extra_trees",
        "hist_gradient_boosting", "ridge_logistic", "transition_logistic",
        "duration_tvtp_hurdle", "shrinkage_lda", "spline_logistic", "xgboost",
        "xgb_hazard_destination", "causal_dynamic_ensemble",
        "causal_multiscale_ensemble",
    ]

    leaderboard = [
        {
            "name": name,
            "rank": rank,
            "selection_log_loss": 0.90 + rank / 100,
            "log_loss": 0.80 + rank / 100,
            "brier": 0.50 + rank / 100,
            "calibration_error": 0.05 + rank / 1000,
        }
        for name, rank in zip(V5_FORECAST_COMPARISON_MODELS, (3, 1, 4, 2, 5), strict=True)
    ]

    comparison_probabilities = {
        "markov": {"risk_on": 0.20, "transition": 0.60, "risk_off": 0.20},
        "xgboost": {"risk_on": 0.12, "transition": 0.76, "risk_off": 0.12},
        "xgb_hazard_destination": {"risk_on": 0.15, "transition": 0.70, "risk_off": 0.15},
        "causal_dynamic_ensemble": {"risk_on": 0.18, "transition": 0.68, "risk_off": 0.14},
        "causal_multiscale_ensemble": {"risk_on": 0.17, "transition": 0.69, "risk_off": 0.14},
    }
    model_forecasts = [
        {
            "state": "transition",
            "probabilities": comparison_probabilities[name],
            "confidence": comparison_probabilities[name]["transition"],
            "entropy": 0.70 if name == "markov" else 0.65,
            "date": "2026-08-14",
            "method": "model_comparison_walk_forward_probability",
            "model": name,
            "fallback": False,
            "fallback_reason": "",
        }
        for name in V5_FORECAST_COMPARISON_MODELS
    ]

    def outcome_row(asset: str, state: str, horizon: int, mean: float) -> dict:
        points = {
            "mean_return": mean,
            "median_return": mean - 0.005,
            "positive_rate": 0.55,
            "annualized_volatility": 0.20,
            "downside_volatility": 0.11,
            "cvar_5": -0.18,
            "mean_max_drawdown": -0.12,
        }
        row = {
            "asset": asset,
            "state": state,
            "horizon_weeks": horizon,
            "execution_lag_weeks": 1,
            "return_currency": "USD",
            "sample_start": "2020-01-03",
            "sample_end": "2026-05-08",
            "n": 80,
            "unique_episodes": 12,
            "status": "ok",
            "minimum_observations": 20,
            "minimum_unique_episodes": 5,
            "bootstrap_method": "episode_bounded_circular_block",
            "bootstrap_block_weeks": 13,
            "bootstrap_resamples": 1999,
            "bootstrap_seed": 17,
            **points,
        }
        for metric, value in points.items():
            row[f"{metric}_ci95_lower"] = value - 0.02
            row[f"{metric}_ci95_upper"] = value + 0.02
        return row

    transition_risk = {
        "1w": {"probability": 0.20, "target_end": "2026-08-14"},
        "4w": {"probability": 0.30, "target_end": "2026-09-04"},
        "13w": {"probability": 0.50, "target_end": "2026-11-06"},
    }
    directional_risk = {
        "1w": {
            "probability": 0.20,
            "no_departure": 0.80,
            "first_destination": {"risk_on": 0.12, "transition": 0.0, "risk_off": 0.08},
            "target_end": "2026-08-14",
            "model": "directional_hazard",
            "method": "first_departure_state_within_h_or_no_departure",
        },
        "4w": {
            "probability": 0.30,
            "no_departure": 0.70,
            "first_destination": {"risk_on": 0.18, "transition": 0.0, "risk_off": 0.12},
            "target_end": "2026-09-04",
            "model": "directional_hazard",
            "method": "first_departure_state_within_h_or_no_departure",
        },
        "13w": {
            "probability": 0.50,
            "no_departure": 0.50,
            "first_destination": {"risk_on": 0.20, "transition": 0.0, "risk_off": 0.30},
            "target_end": "2026-11-06",
            "model": "directional_hazard",
            "method": "first_departure_state_within_h_or_no_departure",
        },
    }
    return {
        "meta": {
            "schema_version": "2.0.0",
            "result_version": "weekly-regime-result-v5",
            "generation_id": "20260812T000000Z-v5-example",
            "generated_at": "2026-08-12T00:00:00Z",
            "data_as_of": "2026-08-07T21:00:00-04:00",
            "mode": "demo",
            "timezone": "America/New_York",
            "warnings": [],
            "freshness": {
                "cadence": "weekly",
                "maximum_age_days": 10,
                "age_days": 3,
                "status": "current",
                "data_as_of": "2026-08-07T21:00:00-04:00",
            },
        },
        "states": [{"id": state} for state in ("risk_on", "transition", "risk_off")],
        "model": {
            "champion": "markov",
            "selection_status": "provisional_predeployment",
            "leaderboard": leaderboard,
            "forecast_comparison": {
                "role": "research_comparison",
                "horizon_weeks": 1,
                "models": V5_FORECAST_COMPARISON_MODELS,
            },
            "profile": "standard",
            "version": "weekly-nondl-structural-v5",
            "label_version": "market-causal-3state-v1",
            "feature_set_version": "weekly-pit-structural-v5",
            "baseline_v4": {
                "result_version": "weekly-regime-result-v4",
                "label_version": "market-causal-3state-v1",
                "model_version": "weekly-nondl-structural-v4",
                "feature_set_version": "weekly-pit-structural-v4",
                "champion": "markov",
                "payload_sha256": "a" * 64,
                "artifacts_inventory_sha256": "b" * 64,
                "captured_at": "2026-08-13",
                "profile": "standard",
            },
            "structural_preregistration": {
                "path": "config/structural_v5.json",
                "sha256": "c" * 64,
            },
            "execution_parameters": {
                "profile": "standard",
                "directional_minimum_selection_predictions": 12,
                "directional_minimum_diagnostic_predictions": 12,
                "directional_maximum_selection_origins": 60,
                "directional_maximum_diagnostic_origins": 60,
                "duration_bootstrap_resamples": 1999,
                "conditional_outcome_bootstrap_resamples": 1999,
                "preregistered_bootstrap_resamples": 1999,
                "preregistration_overrides": [],
                "sha256": "f" * 64,
            },
            "directional_transition": {
                "target": "first_departure_state_within_h_or_no_departure",
                "deployed_direction_role": "first_destination_given_departure",
                "selection_metric": "conditional_destination_log_loss",
                "minimum_selection_departure_events": 8,
                "minimum_selection_destination_classes": 2,
                "minimum_selection_event_blocks": 3,
                "champions": {"1w": "directional_hazard", "4w": "directional_hazard", "13w": "directional_hazard"},
                "leaderboard": [{"horizon_weeks": 1, "model": "directional_hazard"}],
                "selection_diagnostics": [{"horizon_weeks": 1}],
                "selection_end": "2023-01-01",
            },
            "model_health": {"status": "ok", "reasons": []},
            "champion_core_feature_set_version": "weekly-pit-structural-v4",
            "fx_role": "context_and_preregistered_shadow_ablation",
            "fx_ablation": _unavailable_fx_ablation(),
            "core_artifacts": core_artifacts,
            "candidate_manifest_sha256": "9" * 64,
            "candidate_manifest": {
                "profile": "standard",
                "random_state": 17,
                "models": [{"name": name} for name in candidate_names],
            },
            "structural_models": {
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
                "causal_multiscale_ensemble": {
                    "role": "v5_opt_in_candidate",
                    "experts": ["markov", "xgboost", "xgb_hazard_destination"],
                    "scale_half_lives_weeks": [26, 52, 104],
                    "outer_scale_weights": [1 / 3, 1 / 3, 1 / 3],
                    "aggregation": "fixed_equal_probability_average",
                    "inner_pool_method": "causal_discounted_completed_oos_log_score",
                    "minimum_history_rows": 26,
                    "eligible_loss_rule": "target_date_strictly_before_origin",
                    "selection_gate": "existing_multiclass_holm_log_loss_brier_zero_fallback",
                    "automatic_promotion_bypass": False,
                    "sidecar": dict(core_artifacts["multiscale_ensemble_scales"]),
                },
            },
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
            "research_artifacts": research_artifacts,
        },
        "weekly": [{
            "date": "2026-08-07",
            "current": {
                "state": "transition",
                "memberships": {"risk_on": 0.20, "transition": 0.50, "risk_off": 0.30},
                "primary_membership": 0.50,
                "membership_entropy": 0.90,
                "method": "risk_score_anchor_membership",
            },
            "next_week": {
                "state": "transition",
                "probabilities": {"risk_on": 0.20, "transition": 0.60, "risk_off": 0.20},
                "confidence": 0.60,
                "entropy": 0.70,
                "date": "2026-08-14",
                "method": "one_week_state_forecast",
                "model": "markov",
                "fallback": False,
                "fallback_reason": "",
            },
            "model_forecasts": model_forecasts,
            "transition_probability": 0.20,
            "transition_risk": transition_risk,
            "directional_risk": directional_risk,
            "duration_context": {
                "as_of": "2026-08-07",
                "status": "ok",
                "method": "state_specific_kaplan_meier",
                "state": "transition",
                "elapsed_weeks": 3,
                "episodes": 20,
                "completed_spells": 18,
                "censored_spells": 2,
                "minimum_completed_spells": 5,
                "median_remaining_weeks": 6.0,
                "restricted_mean_remaining_weeks": 8.5,
                "restriction_weeks": 52,
                "conditional_survival": {"4w": 0.70, "13w": 0.45},
                "departure_probability": {"4w": 0.30, "13w": 0.55},
                "bootstrap": {"unit": "episode", "resamples": 1999, "valid_resamples": 1900, "seed": 17, "interval": 0.95},
                "ci95": {},
            },
            "fx_context": {
                "status": "ok",
                "method": "fed_h10_usd_strength",
                "bilateral_panel": ["EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "CNY", "MXN", "BRL"],
                "coverage": {
                    "available_pairs": 9,
                    "required_pairs": 9,
                    "available_indexes": 3,
                    "required_indexes": 3,
                },
                "indexes": {"broad_4w_return": 0.01},
                "bilateral": {"usd_up_breadth_4w": 0.67},
                "observation_week": "2026-08-07",
                "feature_available_at": "2026-08-07T21:00:00-04:00",
                "direction": "positive_is_usd_appreciation",
            },
            "context_scores": {"trend": 0.1, "stress": -0.1, "macro": 0.0, "financial_conditions": 0.0},
            "extreme_context": [{"feature": "credit_spread", "label": "Credit spread", "z_score": 2.1, "position": "high", "method": "rolling_52w_zscore"}],
            "summary": "Transition membership with balanced next-week forecast.",
            "market": {},
            "health": {},
        }],
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
        "feature_catalog": [{"id": "fixture"}],
        "research": {
            "conditional_asset_stats": {
                "method": "state_conditioned_forward_total_return",
                "role": "descriptive_only",
                "execution_lag_weeks": 1,
                "horizons_weeks": [1, 4, 13],
                "assets": ["SPY", "QQQ", "IWM", "TLT", "HYG", "UUP"],
                "return_currency": "USD",
                "rows": [
                    outcome_row(asset, state, horizon, mean)
                    for asset in ("SPY", "QQQ", "IWM", "TLT", "HYG", "UUP")
                    for state, mean in (
                        ("risk_on", 0.08),
                        ("transition", 0.02),
                        ("risk_off", -0.05),
                    )
                    for horizon in (1, 4, 13)
                ],
            }
        },
    }


def _browser_validation_errors(payload: dict) -> list[str]:
    program = """
const api = require(process.argv[1]);
let input = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", chunk => { input += chunk; });
process.stdin.on("end", () => {
  process.stdout.write(JSON.stringify(api.validatePayload(JSON.parse(input)).errors));
});
"""
    completed = subprocess.run(
        ["node", "-e", program, str(JS_PATH)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(completed.stdout)


def _browser_history_window_cases() -> list[object]:
    program = """
const api = require(process.argv[1]);
process.stdout.write(JSON.stringify([
  api.resolveHistoryWindow(11, 52),
  api.resolveHistoryWindow(52, 52),
  api.resolveHistoryWindow(51, 52),
  api.resolveHistoryWindow(120, 104),
  api.resolveHistoryWindow(11, "all"),
  api.resolveHistoryWindow(0, 26),
]));
"""
    completed = subprocess.run(
        ["node", "-e", program, str(JS_PATH)],
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(completed.stdout)


class DashboardParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.scripts: list[dict[str, str | None]] = []
        self.links: list[dict[str, str | None]] = []
        self.anchors: list[dict[str, str | None]] = []
        self.inputs: list[dict[str, str | None]] = []
        self.selects: list[dict[str, str | None]] = []
        self.landmarks: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if element_id := attributes.get("id"):
            self.ids.add(element_id)
        if tag == "script":
            self.scripts.append(attributes)
        elif tag == "link":
            self.links.append(attributes)
        elif tag == "a":
            self.anchors.append(attributes)
        elif tag == "input":
            self.inputs.append(attributes)
        elif tag == "select":
            self.selects.append(attributes)
        if tag in {"header", "main", "footer", "nav", "section", "article"}:
            self.landmarks.add(tag)


def parsed_html() -> DashboardParser:
    parser = DashboardParser()
    parser.feed(HTML_PATH.read_text(encoding="utf-8"))
    return parser


def test_dashboard_assets_are_local_and_present() -> None:
    parser = parsed_html()
    assert HTML_PATH.is_file()
    assert CSS_PATH.is_file()
    assert JS_PATH.is_file()
    assert any(str(script.get("src", "")).startswith("./app.js?") and "defer" in script for script in parser.scripts)
    assert any(str(link.get("href", "")).startswith("./styles.css?") for link in parser.links)
    asset_versions = {
        str(asset.get(attribute)).split("?v=", 1)[1]
        for asset, attribute in [
            (script, "src") for script in parser.scripts if str(script.get("src", "")).startswith("./app.js?v=")
        ] + [
            (link, "href") for link in parser.links if str(link.get("href", "")).startswith("./styles.css?v=")
        ]
    }
    assert len(asset_versions) == 1
    assert asset_versions == {"20260824-v5-7"}

    assert all(not str(script.get("src", "")).startswith(("http://", "https://", "//")) for script in parser.scripts)
    assert all(not str(link.get("href", "")).startswith(("http://", "https://", "//")) for link in parser.links)

    allowed_external_links = {
        "https://sonchanggi.github.io/quant-dashboard/",
        "https://sonchanggi.github.io/fearNgreed/",
        "https://sonchanggi.github.io/momentum-factor-lab/",
        "https://sonchanggi.github.io/dram-price/",
        "https://sonchanggi.github.io/best-factor/",
        "https://sonchanggi.github.io/etf-tracking/",
        "https://sonchanggi.github.io/sox/",
        "https://sonchanggi.github.io/regime/",
        "https://fred.stlouisfed.org/",
        "https://www.alphavantage.co/",
        "https://www.federalreserve.gov/releases/h10/",
    }
    external_links = {
        href
        for anchor in parser.anchors
        if (href := anchor.get("href")) and href.startswith(("http://", "https://", "//"))
    }
    assert external_links == allowed_external_links

    document = HTML_PATH.read_text(encoding="utf-8").lower()
    assert "//cdn" not in document


def test_required_result_surfaces_exist() -> None:
    required_ids = {
        "app-state",
        "loading-state",
        "error-state",
        "empty-state",
        "dashboard",
        "header-result-identity",
        "analysis-date",
        "week-select",
        "latest-week",
        "current-regime-card",
        "next-regime-card",
        "current-probabilities",
        "next-probabilities",
        "header-model-health",
        "transition-card",
        "probability-chart",
        "probability-chart-wrap",
        "chart-selection-readout",
        "chart-readout-date",
        "history-data-body",
        "factor-scores",
        "regime-timeline",
        "top-drivers",
        "market-context",
        "leaderboard-body",
        "source-health-body",
        "feature-catalog",
        "header-data-as-of",
        "header-analysis-date",
        "model-loss-chart",
        "model-forecast-field",
        "model-forecast-select",
        "model-forecast-scope",
        "model-forecast-explorer",
        "model-forecast-role",
        "model-forecast-title",
        "model-forecast-caption",
        "model-forecast-symbol",
        "model-forecast-state",
        "model-forecast-confidence",
        "model-forecast-probabilities",
        "model-forecast-rank",
        "model-forecast-log-loss",
        "model-forecast-brier",
        "model-forecast-calibration",
        "model-evidence-summary",
        "transition-horizon-bars",
        "transition-model-section",
        "transition-horizon-select",
        "transition-model-summary",
        "transition-leaderboard-body",
        "duration-context-card",
        "duration-context",
        "fx-context-card",
        "fx-ablation-status",
        "fx-context",
        "conditional-stats",
        "conditional-asset-select",
        "conditional-horizon-select",
        "conditional-stat-grid",
        "conditional-stat-body",
    }
    assert required_ids <= parsed_html().ids


def test_date_controls_support_arbitrary_date_and_exact_week_selection() -> None:
    parser = parsed_html()
    assert any(item.get("id") == "analysis-date" and item.get("type") == "date" for item in parser.inputs)
    assert any(item.get("id") == "week-select" for item in parser.selects)

    script = JS_PATH.read_text(encoding="utf-8")
    assert 'const DATA_URL = "./data/regime-results.json"' in script
    assert "function snapToPriorDate" in script
    assert "dates[middle] <= targetDate" in script
    assert "preserveSnapNote" in script
    assert "window.__REGIME_DASHBOARD__" in script


def test_date_controls_share_one_control_row_and_one_helper_row() -> None:
    document = HTML_PATH.read_text(encoding="utf-8")
    styles = CSS_PATH.read_text(encoding="utf-8")
    form_start = document.index('id="date-form"')
    form_end = document.index("</form>", form_start)
    form = document[form_start:form_end]
    first_group_end = form.index("</div>")
    assert 'id="snap-note"' not in form[:first_group_end]
    assert 'id="analysis-date"' in form and 'aria-describedby="snap-note"' in form
    assert 'id="snap-note" class="control-note sr-only"' in form
    assert 'role="status" aria-live="polite"' in form
    assert '"date week steps latest"' in styles
    assert '"note note note note"' in styles
    assert ".date-controls :is(input, select, button)" in styles
    assert "min-height: 44px" in styles


def test_shared_navigation_and_theme_contract_are_explicit() -> None:
    document = HTML_PATH.read_text(encoding="utf-8")
    script = JS_PATH.read_text(encoding="utf-8")
    assert 'class="site-nav"' in document
    assert (
        'href="https://sonchanggi.github.io/regime/" '
        'aria-current="page">Regime</a>'
    ) in document
    assert 'class="section-nav"' in document
    for anchor in ("#overview", "#history", "#evidence", "#models", "#data-health"):
        assert f'href="{anchor}"' in document
    assert 'const THEME_STORAGE_KEY = "quant-research-theme"' in script


def test_chart_exploration_is_single_focus_and_state_isolated() -> None:
    document = HTML_PATH.read_text(encoding="utf-8")
    script = JS_PATH.read_text(encoding="utf-8")
    assert 'id="probability-chart-wrap"' in document
    assert 'tabindex="0"' in document
    assert 'id="chart-selection-readout"' in document
    assert "function handleChartKeydown" in script
    assert "function previewChartDateFromPointer" in script
    assert "chartPinnedDate" in script and "chartPreviewDate" in script
    assert 'circle.setAttribute("tabindex"' not in script


def test_model_results_remain_visible_without_generalization_warning_surface() -> None:
    document = HTML_PATH.read_text(encoding="utf-8")
    script = JS_PATH.read_text(encoding="utf-8")
    assert 'id="model-diagnostic"' not in document
    assert "holdout_diagnostic" in script
    assert "선정 구간" in script
    assert "holdoutBestName" in script
    assert "is-holdout-best" in script
    assert "2023+ 1위" in script
    assert "진단 주의" not in script


def test_model_forecast_selector_is_labeled_and_scoped_to_comparison() -> None:
    document = HTML_PATH.read_text(encoding="utf-8")
    script = JS_PATH.read_text(encoding="utf-8")
    styles = CSS_PATH.read_text(encoding="utf-8")
    assert '<label for="model-forecast-select">예측 비교 모델</label>' in document
    assert 'id="model-forecast-select"' in document
    assert 'aria-controls="model-forecast-explorer"' in document
    assert 'aria-describedby="model-forecast-scope"' in document
    assert (
        'id="model-forecast-scope" class="sr-only">'
        "공식 선정 모델은 변경하지 않고 비교 예측만 전환합니다."
    ) in document
    assert 'id="model-forecast-explorer"' in document
    assert 'aria-labelledby="model-forecast-title"' in document
    assert 'id="model-forecast-probabilities"' in document
    assert "V5_FORECAST_COMPARISON_MODELS" in script
    assert "function renderModelForecast()" in script
    assert 'dom["model-forecast-select"].addEventListener("change"' in script
    assert '.model-forecast-field select' in styles
    assert "min-height: 44px;" in styles


def test_current_membership_and_forecast_probability_are_separate_surfaces() -> None:
    document = HTML_PATH.read_text(encoding="utf-8")
    script = JS_PATH.read_text(encoding="utf-8")
    assert 'id="probability-shifts"' not in document
    assert 'id="model-loss-chart"' in document
    assert 'id="model-loss-axis"' in document
    assert "function renderProbabilityShifts" not in script
    assert "function renderModelLossChart" in script
    assert "function getCurrentMeasure" in script
    assert 'version === V5_RESULT_VERSION ? "membership" : "probability"' in script
    assert 'field = currentMeasureKind(version) === "membership" ? "memberships" : "probabilities"' in script
    assert "isCurrent ? getCurrentMeasure(result, stateCode) : getProbability(result, stateCode)" in script
    assert 'metricValue(row, ["selection_log_loss"])' in script
    assert 'metricValue(row, ["log_loss", "multiclass_log_loss"])' in script
    assert 'dom["model-loss-axis"].replaceChildren' in script


def test_results_first_layout_keeps_equal_cards_and_wide_history() -> None:
    document = HTML_PATH.read_text(encoding="utf-8")
    script = JS_PATH.read_text(encoding="utf-8")
    styles = CSS_PATH.read_text(encoding="utf-8")
    assert 'class="probability-shift-card card"' not in document
    assert 'id="duration-context-card" class="card v5-only"' in document
    assert 'id="fx-context-card" class="card v5-only"' in document
    assert 'id="conditional-stats" class="dashboard-section conditional-stats-section card v5-only"' in document
    assert 'viewBox="0 0 1200 300"' in document
    assert "width: 1200" in script
    assert "const desiredTicks = Math.min(7, history.length)" in script
    assert "function scrollChartDateIntoView" in script
    assert "requestAnimationFrame(() => scrollChartDateIntoView(state.chartPinnedDate))" in script
    assert ".hero-grid {\n  grid-template-columns: repeat(3, minmax(0, 1fr));" in styles
    assert ".hero-grid > .transition-card {\n    grid-column: auto;" in styles
    assert "#history.analysis-grid" in styles
    assert ".factor-list {\n  margin-top: 16px;\n  grid-template-columns: repeat(4" in styles
    assert ".chart-point.is-active" in styles


def test_v5_only_sections_fail_closed_for_v4_payloads() -> None:
    document = HTML_PATH.read_text(encoding="utf-8")
    script = JS_PATH.read_text(encoding="utf-8")
    styles = CSS_PATH.read_text(encoding="utf-8")
    for section_id in ("conditional-stats-nav", "duration-context-card", "fx-context-card", "conditional-stats", "research-evidence"):
        assert f'id="{section_id}"' in document
        start = document.index(f'id="{section_id}"')
        assert "hidden" in document[start : start + 180]
    assert ".v5-only[hidden]" in styles
    assert "if (!isV5Payload() || !isObject(duration))" in script
    assert "if (!isV5Payload() || !isObject(fx))" in script
    assert "if (!isV5Payload())" in script


def test_full_model_and_conditional_tables_are_collapsed_by_default() -> None:
    document = HTML_PATH.read_text(encoding="utf-8")
    assert document.count('class="compact-table-details"') == 3
    assert "<summary>전체 모델 표</summary>" in document
    assert "<summary>전체 이탈 모델 표</summary>" in document
    assert "상세 성과 표" in document
    assert 'class="compact-table-details" open' not in document


def test_browser_contract_rejects_probability_keys_beyond_three_states() -> None:
    script = JS_PATH.read_text(encoding="utf-8")
    assert "probabilityKeys.length !== STATE_ORDER.length" in script
    assert "확률 키는 표준 세 상태와 정확히 일치" in script


def test_v3_transition_contract_is_additive_and_fail_closed() -> None:
    document = HTML_PATH.read_text(encoding="utf-8")
    script = JS_PATH.read_text(encoding="utf-8")
    assert 'const V3_RESULT_VERSION = "weekly-regime-result-v3"' in script
    assert "TRANSITION_HORIZONS" in script
    assert 'const expectedKeys = ["1w", "4w", "13w"]' in script
    assert 'const exactRiskKeys = ["probability", "target_end", "model", "threshold", "fallback", "fallback_reason"]' in script
    assert "transition_probability와 transition_risk.1w.probability가 일치하지 않습니다" in script
    assert "1주 이탈 확률과 next_week 현재 국면 잔류 확률이 일치하지 않습니다" in script
    assert "payload.model.primary_horizon_weeks !== 1" in script
    assert "payload.model.transition_leaderboard" in script
    assert "payload.model.shadow_nowcast" in script
    assert 'id="transition-horizon-bars"' in document


def test_v4_structural_contract_is_additive_and_fail_closed() -> None:
    script = JS_PATH.read_text(encoding="utf-8")
    assert 'const V4_RESULT_VERSION = "weekly-regime-result-v4"' in script
    assert 'const V4_MODEL_VERSION = "weekly-nondl-structural-v4"' in script
    assert 'const V4_FEATURE_SET_VERSION = "weekly-pit-structural-v4"' in script
    assert "FROZEN_V4_BASELINE_V3" in script
    assert "FROZEN_V4_STRUCTURAL_PREREGISTRATION" in script
    assert "baseline_v3" in script
    assert "structural_preregistration" in script
    assert "feature_manifest_sha256" in script
    assert "evidence_artifacts" in script
    assert "state-label-history.csv" in script
    assert "weekly-state-forecasts.csv" in script
    assert "xgb_hazard_destination" in script
    assert "causal_dynamic_ensemble" in script
    assert "joint_survival_hazard" in script
    assert "retrospective_diagnostic_only" in script


def test_declared_result_versions_and_no_event_average_precision_fail_closed() -> None:
    script = JS_PATH.read_text(encoding="utf-8")
    assert "![V3_RESULT_VERSION, V4_RESULT_VERSION, V5_RESULT_VERSION].includes(declaredResultVersion)" in script
    assert "지원하지 않는 meta.result_version입니다" in script
    assert "row.average_precision === null && eventCount === 0" in script
    assert "무이벤트 구간의 null이어야 합니다" in script
    assert "const binaryLogLoss = strictFiniteNumber(row.binary_log_loss)" in script
    assert 'for (const metric of ["brier", "precision", "recall"])' in script
    assert "function strictFiniteNumber" in script
    assert "function strictProbability" in script
    assert "strictProbability(row.average_precision)" in script


def test_v3_transition_metric_ranges_counts_and_selection_cutoff_are_explicit() -> None:
    script = JS_PATH.read_text(encoding="utf-8")
    assert "isIsoDate(payload.model.transition_selection_end)" in script
    assert "model.transition_selection_end는 YYYY-MM-DD 형식의 실제 날짜" in script
    assert "binary_log_loss가 0 이상의 유한한 숫자" in script
    assert '"non_event_count", "fallback_count", "calibration_fallback_count"' in script
    assert "!Number.isInteger(value)" in script
    assert "predictionCount !== eventCount + nonEventCount" in script
    assert "n_predictions는 event_count와 non_event_count의 합" in script


def test_browser_validator_executes_valid_v3_semantic_contract() -> None:
    assert _browser_validation_errors(_valid_v3_browser_payload()) == []


def test_browser_validator_executes_valid_v4_semantic_contract() -> None:
    assert _browser_validation_errors(_valid_v4_browser_payload()) == []


def test_browser_validator_accepts_valid_v5_without_changing_v4() -> None:
    assert _browser_validation_errors(_valid_v4_browser_payload()) == []
    assert _browser_validation_errors(_valid_v5_browser_payload()) == []


def test_v5_browser_binds_model_comparison_order_inventory_and_official_parity() -> None:
    payload = _valid_v5_browser_payload()
    assert _browser_validation_errors(payload) == []

    missing_row = deepcopy(payload)
    missing_row["weekly"][0]["model_forecasts"].pop()
    assert any(
        "model_forecasts 모델 수" in error
        for error in _browser_validation_errors(missing_row)
    )

    wrong_order = deepcopy(payload)
    rows = wrong_order["weekly"][0]["model_forecasts"]
    rows[1], rows[2] = rows[2], rows[1]
    assert any(
        "model 순서가 forecast_comparison" in error
        for error in _browser_validation_errors(wrong_order)
    )

    missing_leaderboard_model = deepcopy(payload)
    missing_leaderboard_model["model"]["leaderboard"] = [
        row
        for row in missing_leaderboard_model["model"]["leaderboard"]
        if row["name"] != "causal_multiscale_ensemble"
    ]
    assert any(
        "forecast_comparison 모델이 leaderboard" in error
        for error in _browser_validation_errors(missing_leaderboard_model)
    )

    champion_mismatch = deepcopy(payload)
    champion_mismatch["weekly"][0]["model_forecasts"][0]["probabilities"] = {
        "risk_on": 0.19,
        "transition": 0.60,
        "risk_off": 0.21,
    }
    assert any(
        "선정 모델이 공식 next_week와 일치하지 않습니다" in error
        for error in _browser_validation_errors(champion_mismatch)
    )

    orphan_forecasts = deepcopy(payload)
    orphan_forecasts["model"].pop("forecast_comparison")
    assert any(
        "model_forecasts에는 forecast_comparison 메타데이터" in error
        for error in _browser_validation_errors(orphan_forecasts)
    )


def test_v5_browser_rejects_model_forecast_state_that_is_not_probability_argmax() -> None:
    payload = _valid_v5_browser_payload()
    forecast = payload["weekly"][0]["model_forecasts"][1]
    forecast["state"] = "risk_on"
    forecast["confidence"] = forecast["probabilities"]["risk_on"]
    assert any(
        "state가 최대 예측확률과 일치하지 않습니다" in error
        for error in _browser_validation_errors(payload)
    )


def test_v5_browser_requires_explicit_mode_and_warning_array() -> None:
    for label, mutate in (
        ("missing mode", lambda payload: payload["meta"].pop("mode")),
        ("unknown mode", lambda payload: payload["meta"].__setitem__("mode", "research")),
        ("missing warnings", lambda payload: payload["meta"].pop("warnings")),
        ("non-array warnings", lambda payload: payload["meta"].__setitem__("warnings", "demo")),
        ("blank warning", lambda payload: payload["meta"].__setitem__("warnings", [" "])),
    ):
        payload = _valid_v5_browser_payload()
        mutate(payload)
        assert _browser_validation_errors(payload), f"browser validator accepted {label}"


def test_v5_browser_binds_mode_profile_and_source_identity() -> None:
    profile_mismatch = _valid_v5_browser_payload()
    profile_mismatch["model"]["profile"] = "full"
    assert any("model.profile" in error for error in _browser_validation_errors(profile_mismatch))

    live_with_demo_sources = _valid_v5_browser_payload()
    live_with_demo_sources["meta"]["mode"] = "live"
    assert any("sources identity" in error for error in _browser_validation_errors(live_with_demo_sources))

    valid_live = _valid_v5_browser_payload()
    valid_live["meta"]["mode"] = "live"
    valid_live["sources"] = [
        {"id": "alpha_vantage", "status": "ok", "license_class": "private_noncommercial"},
        {"id": "alfred", "status": "ok", "license_class": "user_confirmed_ml_storage_derived"},
        {
            "id": "frb_h10",
            "status": "unavailable",
            "license_class": "federal_reserve_board_public_domain_citation_requested",
            "official_release_archive_ingest": False,
            "availability_basis": "collection_first_seen_at",
            "archive_revision_policy": "later_official_release_preserved_as_new_vintage",
            "archive_correction_availability_basis": "date_only_conservative_next_day",
            "archive_release_count": 0,
            "archive_correction_count": 0,
            "archive_correction_available_at": [],
            "archive_correction_quarantine_weeks": 27,
            "archive_evaluation_start": "2022-01-01",
            "archive_evaluation_start_rationale": "post_2019_06_24_jan06_index_rebase_common_scale",
        },
    ]
    assert _browser_validation_errors(valid_live) == []

    bad_license = deepcopy(valid_live)
    bad_license["sources"][0]["license_class"] = "synthetic_fixture"
    assert any("license_class" in error for error in _browser_validation_errors(bad_license))

    bad_source_status = deepcopy(valid_live)
    bad_source_status["sources"][0]["status"] = "partial"
    assert any("sources.alpha_vantage.status" in error for error in _browser_validation_errors(bad_source_status))

    live_quick = deepcopy(valid_live)
    live_quick["model"]["profile"] = "quick"
    execution = live_quick["model"]["execution_parameters"]
    execution["profile"] = "quick"
    execution["directional_minimum_selection_predictions"] = 3
    execution["directional_minimum_diagnostic_predictions"] = 3
    execution["directional_maximum_selection_origins"] = 3
    execution["directional_maximum_diagnostic_origins"] = 3
    assert any("quick profile" in error for error in _browser_validation_errors(live_quick))


def test_v5_browser_binds_h10_archive_provenance_and_inventory() -> None:
    payload = _valid_v5_browser_payload()
    payload["meta"]["mode"] = "live"
    payload["model"]["fx_ablation"].update(
        {
            "official_release_archive_ingest": True,
            "availability_basis": "official_archive_release_schedule",
        }
    )
    payload["sources"] = [
        {"id": "alpha_vantage", "status": "ok", "license_class": "private_noncommercial"},
        {"id": "alfred", "status": "ok", "license_class": "user_confirmed_ml_storage_derived"},
        {
            "id": "frb_h10",
            "status": "ok",
            "license_class": "federal_reserve_board_public_domain_citation_requested",
            "official_release_archive_ingest": True,
            "availability_basis": "official_archive_release_schedule",
            "archive_revision_policy": "later_official_release_preserved_as_new_vintage",
            "archive_correction_availability_basis": "date_only_conservative_next_day",
            "archive_release_count": 245,
            "archive_correction_count": 1,
            "archive_correction_available_at": ["2024-08-08T04:00:00+00:00"],
            "archive_correction_quarantine_weeks": 27,
            "archive_evaluation_start": "2022-01-01",
            "archive_evaluation_start_rationale": "post_2019_06_24_jan06_index_rebase_common_scale",
        },
    ]
    assert _browser_validation_errors(payload) == []

    for label, mutate in (
        (
            "model/source basis mismatch",
            lambda value: value["sources"][2].__setitem__(
                "availability_basis", "collection_first_seen_at"
            ),
        ),
        (
            "missing archive releases",
            lambda value: value["sources"][2].__setitem__("archive_release_count", 0),
        ),
        (
            "correction count mismatch",
            lambda value: value["sources"][2].__setitem__("archive_correction_count", 0),
        ),
        (
            "non-UTC correction",
            lambda value: value["sources"][2].__setitem__(
                "archive_correction_available_at", ["2024-08-08T00:00:00-04:00"]
            ),
        ),
        (
            "wrong quarantine",
            lambda value: value["sources"][2].__setitem__(
                "archive_correction_quarantine_weeks", 26
            ),
        ),
    ):
        changed = deepcopy(payload)
        mutate(changed)
        assert _browser_validation_errors(changed), f"browser validator accepted {label}"


def test_v5_browser_binds_core_artifacts_and_multiscale_candidate() -> None:
    payload = _valid_v5_browser_payload()
    assert _browser_validation_errors(payload) == []

    for label, mutate in (
        (
            "missing core artifact",
            lambda value: value["model"]["core_artifacts"].pop("stacking_weights"),
        ),
        (
            "empty core artifact",
            lambda value: value["model"]["core_artifacts"]["oos_predictions"].__setitem__(
                "row_count", 0
            ),
        ),
        (
            "sidecar hash mismatch",
            lambda value: value["model"]["structural_models"][
                "causal_multiscale_ensemble"
            ]["sidecar"].__setitem__("sha256", "b" * 64),
        ),
        (
            "half-life drift",
            lambda value: value["model"]["structural_models"][
                "causal_multiscale_ensemble"
            ].__setitem__("scale_half_lives_weeks", [13, 52, 104]),
        ),
        (
            "outer weight drift",
            lambda value: value["model"]["structural_models"][
                "causal_multiscale_ensemble"
            ].__setitem__("outer_scale_weights", [0.5, 0.25, 0.25]),
        ),
        (
            "candidate omitted",
            lambda value: value["model"]["candidate_manifest"]["models"].pop(),
        ),
    ):
        changed = deepcopy(payload)
        mutate(changed)
        assert _browser_validation_errors(changed), f"browser validator accepted {label}"


def test_result_identity_and_dynamic_freshness_are_payload_driven() -> None:
    program = f"""
const api = require({json.dumps(str(JS_PATH))});
process.stdout.write(JSON.stringify({{
  identities: [
    api.resultIdentity({{meta: {{mode: "demo"}}, model: {{execution_parameters: {{profile: "quick"}}}}}}).label,
    api.resultIdentity({{meta: {{mode: "live"}}, model: {{execution_parameters: {{profile: "standard"}}}}}}).label,
    api.resultIdentity({{meta: {{mode: "live"}}, model: {{profile: "full"}}}}).label,
    api.resultIdentity({{meta: {{mode: "live"}}, model: {{profile: "standard", selection_status: "provisional_predeployment"}}}}).label,
    api.resultIdentity({{meta: {{mode: "live", publication_status: "reviewed_publication"}}, model: {{profile: "standard", selection_status: "provisional_predeployment"}}}}).label
  ],
  fxStatuses: [
    api.fxStatusLabel("unavailable"),
    api.fxStatusLabel("insufficient_history"),
    api.fxStatusLabel("evaluated")
  ],
  current: api.displayFreshness("2026-08-12T00:00:00Z", 10, Date.parse("2026-08-22T23:59:59Z")),
  stale: api.displayFreshness("2026-08-07T20:00:00Z", 10, Date.parse("2026-08-22T03:07:56Z"))
}}));
"""
    completed = subprocess.run(["node", "-e", program], text=True, capture_output=True, check=True)
    result = json.loads(completed.stdout)
    assert result["identities"] == [
        "모의자료 · QUICK · 파이프라인 검증",
        "실데이터 · STANDARD",
        "실데이터 · FULL",
        "실데이터 · STANDARD · 배포 전 잠정",
        "실데이터 · STANDARD · 공개 검토 완료",
    ]
    assert result["fxStatuses"] == ["사용 불가", "표본 축적 중", "완료"]
    assert result["current"] == {
        "age_days": 10,
        "maximum_age_days": 10,
        "status": "current",
    }
    assert result["stale"] == {
        "age_days": 14,
        "maximum_age_days": 10,
        "status": "stale",
    }


def test_v5_hard_state_does_not_have_to_be_membership_argmax() -> None:
    payload = _valid_v5_browser_payload()
    current = payload["weekly"][0]["current"]
    current["memberships"] = {"risk_on": 0.50, "transition": 0.40, "risk_off": 0.10}
    current["primary_membership"] = 0.40
    assert _browser_validation_errors(payload) == []


def test_v5_rejects_cross_version_current_fields_and_directional_mass_errors() -> None:
    mixed = _valid_v5_browser_payload()
    mixed["weekly"][0]["current"]["probabilities"] = mixed["weekly"][0]["current"]["memberships"]
    assert any("current" in error for error in _browser_validation_errors(mixed))

    bad_mass = _valid_v5_browser_payload()
    bad_mass["weekly"][0]["directional_risk"]["13w"]["first_destination"]["risk_off"] = 0.20
    assert any("first_destination" in error for error in _browser_validation_errors(bad_mass))

    bad_duration = _valid_v5_browser_payload()
    bad_duration["weekly"][0]["duration_context"]["departure_probability"]["4w"] = 0.40
    assert any("생존·이탈 합계" in error for error in _browser_validation_errors(bad_duration))

    bad_fx = _valid_v5_browser_payload()
    bad_fx["weekly"][0]["fx_context"]["bilateral_panel"].pop()
    assert any("9통화 panel" in error for error in _browser_validation_errors(bad_fx))

    backfilled_fx = _valid_v5_browser_payload()
    backfilled_fx["model"]["fx_ablation"]["historical_availability_backfill"] = True
    assert any("비소급 prospective shadow" in error for error in _browser_validation_errors(backfilled_fx))

    partial_panel_model = _valid_v5_browser_payload()
    partial_panel_model["model"]["fx_ablation"]["common_origin_required_pairs"] = 6
    assert any("공통 origin·purge" in error for error in _browser_validation_errors(partial_panel_model))

    promoted_fx = _valid_v5_browser_payload()
    promoted_fx["model"]["fx_ablation"]["core_champion_promoted"] = True
    assert any("자동 승격" in error for error in _browser_validation_errors(promoted_fx))


def test_v5_browser_accepts_degraded_last_good_fx_context() -> None:
    payload = _valid_v5_browser_payload()
    payload["weekly"][0]["fx_context"]["status"] = "degraded"
    assert _browser_validation_errors(payload) == []


def test_v5_browser_rejects_research_artifact_manifest_drift() -> None:
    missing_pair = _valid_v5_browser_payload()
    missing_pair["model"]["research_artifacts"].pop("fx_coverage")
    assert any(
        "research_artifacts manifest" in error
        for error in _browser_validation_errors(missing_pair)
    )

    unsafe_path = _valid_v5_browser_payload()
    unsafe_path["model"]["research_artifacts"]["directional_forecasts"][
        "path"
    ] = "../directional-forecasts.csv"
    assert any(
        "directional_forecasts" in error
        for error in _browser_validation_errors(unsafe_path)
    )

    wrong_count = _valid_v5_browser_payload()
    wrong_count["model"]["research_artifacts"][
        "conditional_asset_statistics"
    ]["row_count"] = 53
    assert any(
        "conditional asset statistics artifact" in error
        for error in _browser_validation_errors(wrong_count)
    )

    wrong_fx_oos_count = _valid_v5_browser_payload()
    wrong_fx_oos_count["model"]["research_artifacts"]["fx_ablation_oos"][
        "row_count"
    ] = 1
    assert any(
        "FX research artifact" in error
        for error in _browser_validation_errors(wrong_fx_oos_count)
    )


def test_v5_rejects_allocation_semantics_in_conditional_statistics() -> None:
    payload = _valid_v5_browser_payload()
    payload["research"]["conditional_asset_stats"]["rows"][0]["allocation"] = 0.5
    assert any("allocation" in error for error in _browser_validation_errors(payload))


def test_exported_current_measure_adapter_is_version_explicit() -> None:
    program = f"""
const api = require({json.dumps(str(JS_PATH))});
process.stdout.write(JSON.stringify([
  api.currentMeasureKind("weekly-regime-result-v4"),
  api.currentMeasureKind("weekly-regime-result-v5"),
  api.getCurrentMeasure({{probabilities: {{risk_on: 0.25}}}}, "risk_on", "weekly-regime-result-v4"),
  api.getCurrentMeasure({{memberships: {{risk_on: 0.75}}}}, "risk_on", "weekly-regime-result-v5")
]));
"""
    completed = subprocess.run(["node", "-e", program], text=True, capture_output=True, check=True)
    assert json.loads(completed.stdout) == ["probability", "membership", 0.25, 0.75]


def test_v3_browser_contract_requires_nonempty_generation_id() -> None:
    payload = _valid_v3_browser_payload()
    payload["meta"]["generation_id"] = None
    errors = _browser_validation_errors(payload)
    assert any("generation_id" in error for error in errors)


def test_browser_validator_executes_python_v3_semantic_rejections() -> None:
    cases = [
        ("generation id", lambda payload: payload["meta"].__setitem__("generation_id", "")),
        ("model.version", lambda payload: payload["model"].__setitem__("version", "wrong-v3")),
        ("label version", lambda payload: payload["model"].__setitem__("label_version", "wrong-label")),
        ("feature version", lambda payload: payload["model"].__setitem__("feature_set_version", "wrong-features")),
        (
            "baseline hash",
            lambda payload: payload["model"]["baseline_v2"].__setitem__("payload_sha256", "A" * 64),
        ),
        (
            "baseline required field",
            lambda payload: payload["model"]["baseline_v2"].pop("champion"),
        ),
        (
            "shadow canonical",
            lambda payload: payload["model"]["shadow_nowcast"].__setitem__("canonical_target", True),
        ),
        (
            "horizon target",
            lambda payload: payload["weekly"][0]["transition_risk"]["13w"].__setitem__("target_end", "2026-11-13"),
        ),
        (
            "next-week alias",
            lambda payload: payload["weekly"][0]["next_week"].__setitem__("date", "2026-08-21"),
        ),
        (
            "fallback reason",
            lambda payload: payload["weekly"][0]["transition_risk"]["1w"].__setitem__("fallback_reason", None),
        ),
        (
            "champion keys",
            lambda payload: payload["model"]["transition_champions"].__setitem__("26w", "hazard"),
        ),
    ]
    for label, mutate in cases:
        payload = deepcopy(_valid_v3_browser_payload())
        mutate(payload)
        assert _browser_validation_errors(payload), f"browser validator accepted invalid {label}"


def test_browser_validator_executes_python_v4_semantic_rejections() -> None:
    cases = [
        ("v4 model", lambda payload: payload["model"].__setitem__("version", "wrong-v4")),
        ("v4 feature", lambda payload: payload["model"].__setitem__("feature_set_version", "wrong")),
        ("v3 baseline hash", lambda payload: payload["model"]["baseline_v3"].__setitem__("payload_sha256", "A" * 64)),
        ("v3 baseline valid wrong hash", lambda payload: payload["model"]["baseline_v3"].__setitem__("payload_sha256", "0" * 64)),
        ("v3 baseline inventory valid wrong hash", lambda payload: payload["model"]["baseline_v3"].__setitem__("artifacts_inventory_sha256", "1" * 64)),
        ("v3 baseline champion", lambda payload: payload["model"]["baseline_v3"].__setitem__("champion", "xgboost")),
        ("v3 baseline captured at", lambda payload: payload["model"]["baseline_v3"].__setitem__("captured_at", "2026-08-14")),
        ("v3 baseline shape", lambda payload: payload["model"]["baseline_v3"].__setitem__("extra", True)),
        ("prereg path", lambda payload: payload["model"]["structural_preregistration"].__setitem__("path", "config/structural_v5.json")),
        ("prereg valid wrong hash", lambda payload: payload["model"]["structural_preregistration"].__setitem__("sha256", "0" * 64)),
        ("feature manifest", lambda payload: payload["model"].__setitem__("feature_manifest_sha256", "bad")),
        ("label evidence path", lambda payload: payload["model"]["evidence_artifacts"]["state_label_history"].__setitem__("path", "../state-label-history.csv")),
        ("label evidence hash", lambda payload: payload["model"]["evidence_artifacts"]["state_label_history"].__setitem__("sha256", "bad")),
        ("weekly evidence count", lambda payload: payload["model"]["evidence_artifacts"]["weekly_state_forecasts"].__setitem__("row_count", 2)),
        ("expert order", lambda payload: payload["model"]["structural_models"]["causal_dynamic_ensemble"].__setitem__("experts", ["xgboost", "markov", "xgb_hazard_destination"])),
        ("hazard floor", lambda payload: payload["model"]["structural_models"]["xgb_hazard_destination"].__setitem__("direct_jump_floor", 0.0)),
        ("survival horizon", lambda payload: payload["model"]["structural_models"]["joint_survival_hazard"].__setitem__("horizons_weeks", [1, 4])),
        ("ablation authority", lambda payload: payload["model"]["ablation"].__setitem__("may_change_published_variant", True)),
        ("ablation manifest hash", lambda payload: payload["model"]["ablation"].__setitem__("manifest_sha256", "bad")),
    ]
    for label, mutate in cases:
        payload = _valid_v4_browser_payload()
        mutate(payload)
        assert _browser_validation_errors(payload), f"browser validator accepted invalid {label}"


def test_v3_transition_models_have_horizon_specific_diagnostic_surface() -> None:
    document = HTML_PATH.read_text(encoding="utf-8")
    script = JS_PATH.read_text(encoding="utf-8")
    styles = CSS_PATH.read_text(encoding="utf-8")
    assert 'id="transition-model-section"' in document
    assert 'for="transition-horizon-select"' in document
    for value in ("1", "4", "13"):
        assert f'<option value="{value}"' in document
    for label in ("AP ↑", "Precision ↑", "Recall ↑", "False alarms / 연 ↓"):
        assert label in document
    assert "선정 구간 · 2023+ 진단" in document
    assert "${horizon}주 이탈 · 선정 구간 / 2023+ 진단" in script
    assert "function renderTransitionModels" in script
    assert "function renderTransitionHorizons" in script
    assert "section.hidden = true" in script
    assert "min-height: 44px" in styles
    assert "#transition-leaderboard-table" in styles


def test_browser_contract_requires_provisional_predeployment_selection_status() -> None:
    script = JS_PATH.read_text(encoding="utf-8")
    assert 'payload.model.selection_status !== "provisional_predeployment"' in script
    assert "model.selection_status는 provisional_predeployment여야 합니다" in script
    assert 'createElement("span", null, "선정 모델")' in script
    assert 'labels.push("공개 검토 완료")' in script
    assert 'labels.push("배포 전 잠정")' in script
    assert 'meta.publication_status !== V5_PUBLICATION_STATUS' in script


def test_v5_comparison_sidecar_requires_current_payload_and_frozen_v4_identity() -> None:
    payload_sha = "b" * 64
    inventory_sha = "a" * 64
    split = {
        "common_keys": {"count": 1},
        "delta_left_minus_right": {
            "log_loss": 0,
            "brier": 0,
            "balanced_accuracy": 0,
            "fallback_rate": 0,
        },
        "probability_parity": {
            "probability_numeric": {
                "exact_float_parity": True,
                "maximum_absolute_difference": 0,
                "mismatch_rows": 0,
                "mismatch_values": 0,
            },
            "probability_token_bytes": {
                "exact_parity": True,
                "mismatch_rows": 0,
                "mismatch_values": 0,
            },
        },
    }
    report = {
        "schema_version": "regime-v5-v4-matched-comparison/1",
        "report_role": "derived_only_diagnostic_comparison",
        "promotion_interpretation": "prohibited",
        "inputs": {
            "v5": {"regime_results": {"sha256": payload_sha}},
            "frozen_v4": {"sha256sums": {"sha256": inventory_sha}},
        },
        "v5_markov_vs_frozen_v4_markov": {
            "common_keys": {"count": 2},
            "primary_selection": split,
            "post_selection_holdout": split,
        },
    }
    payload = {"model": {"baseline_v4": {"artifacts_inventory_sha256": inventory_sha}}}
    program = f"""
const api = require({json.dumps(str(JS_PATH))});
const report = {json.dumps(report)};
const payload = {json.dumps(payload)};
process.stdout.write(JSON.stringify([
  api.validateV5ComparisonSummary(report, payload, {json.dumps(payload_sha)}),
  api.validateV5ComparisonSummary(report, payload, {json.dumps('c' * 64)}),
]));
"""
    completed = subprocess.run(["node", "-e", program], text=True, capture_output=True, check=True)
    accepted, rejected = json.loads(completed.stdout)
    assert accepted == {
        "commonKeys": 2,
        "selectionKeys": 1,
        "holdoutKeys": 1,
        "exactParity": True,
    }
    assert rejected is None


def test_v5_decision_evidence_is_collapsed_at_the_bottom_and_semantically_distinct() -> None:
    document = HTML_PATH.read_text(encoding="utf-8")
    script = JS_PATH.read_text(encoding="utf-8")
    styles = CSS_PATH.read_text(encoding="utf-8")
    assert 'id="model-evidence-summary"' in document
    assert '"champion-summary", "model-evidence-summary"' in script
    research_position = document.index('id="research-evidence"')
    assert research_position > document.index('id="models"')
    assert research_position < document.index('id="data-health"')
    assert 'id="research-evidence-details"' in document
    assert 'id="research-evidence-details" open' not in document
    assert document.index('id="model-evidence-summary"') > research_position
    assert document.index('id="fx-ablation-status"') > research_position
    for phrase in (
        "가용 공통",
        "실제 OOS",
        "FX 후보 gate",
        "core 비승격",
        "멀티스케일 후보",
        "V4 기준 비교",
        "Markov 확률 완전 일치",
        "일반화 약화",
        "보정 드리프트",
    ):
        assert phrase in script
    assert ".model-evidence-summary" in styles
    assert ".research-evidence-grid" in styles


def test_conditional_performance_reuses_card_spacing_and_has_reachable_wide_table() -> None:
    document = HTML_PATH.read_text(encoding="utf-8")
    styles = CSS_PATH.read_text(encoding="utf-8")
    assert 'id="conditional-stat-scroll" class="table-scroll conditional-table-scroll"' in document
    assert 'class="table-scroll-guide"' in document
    assert "@media (max-width: 1240px) {\n  .table-scroll-guide {\n    display: block;" in styles
    assert ".conditional-stats-section {\n  margin-bottom: 12px;\n  padding: 20px;" in styles
    assert "#conditional-stat-table {\n  width: 100%;\n  min-width: 1160px;\n  table-layout: fixed;" in styles
    assert "#conditional-stat-table th:first-child" in styles
    assert "position: sticky;" in styles
    assert "scrollbar-width: auto;" in styles


def test_conditional_performance_leads_with_asset_class_mean_comparison() -> None:
    document = HTML_PATH.read_text(encoding="utf-8")
    script = JS_PATH.read_text(encoding="utf-8")
    styles = CSS_PATH.read_text(encoding="utf-8")
    section_start = document.index('id="conditional-stats"')
    details_start = document.index('class="compact-table-details"', section_start)
    assert document.index('id="conditional-horizon-select"', section_start) < details_start
    assert document.index('id="conditional-asset-select"', section_start) > details_start
    assert '<label for="conditional-asset-select">상세 자산</label>' in document
    assert "자산군별 평균 수익률" in document
    assert "평균 95% CI" in document
    assert "연율 하방 변동성" in document
    assert 'id="conditional-comparison-caption"' in document
    assert "const OUTCOME_ASSET_LABELS" in script
    assert "function conditionalStatsRows()" in script
    assert "function conditionalDetailRows()" in script
    assert "function conditionalComparisonRows()" in script
    assert "function renderConditionalComparison()" in script
    assert "publicationSnapshotLabel()" in script
    assert '"과거 조회"' in script
    assert "function renderConditionalDetail()" in script
    assert 'createElement("div", "conditional-asset-list")' in script
    assert 'createElement("span", "conditional-return-track")' in script
    assert 'track.setAttribute("aria-hidden", "true")' in script
    assert '`${value > 0 ? "+" : ""}${formatSignedPercent(value, comparisonDigits)}`' in script
    assert 'createElement("small", null, `n ${sample}`)' in script
    assert "표본 부족" in script
    assert "값 없음" in script
    assert 'renderConditionalDetail();\n      dom["screen-reader-status"]' in script
    assert ".conditional-asset-row" in styles
    assert ".conditional-return-track::after" in styles


def test_operations_are_collapsed_below_results_without_warning_surfaces() -> None:
    document = HTML_PATH.read_text(encoding="utf-8")
    script = JS_PATH.read_text(encoding="utf-8")
    overview_end = document.index("</section>", document.index('id="overview"'))
    operations_position = document.index('id="data-health"')
    assert operations_position > overview_end
    assert '<details class="research-notice-details operations-details">' in document
    assert 'id="research-notice-summary"' in document
    assert '<details class="research-notice-details operations-details" open' not in document
    for warning_surface in ("data-alerts", "model-diagnostic", "method-notices", "publication-gate"):
        assert warning_surface not in document
    assert "renderAlerts" not in script
    assert "renderMethodNotices" not in script


def test_default_canvas_uses_compact_copy_and_one_operations_disclosure() -> None:
    document = HTML_PATH.read_text(encoding="utf-8")
    script = JS_PATH.read_text(encoding="utf-8")
    assert document.count('class="eyebrow"') == 1
    assert 'class="table-scroll-hint"' not in document
    assert 'class="method-note"' not in document
    assert document.count('class="research-notice-details operations-details"') == 1
    assert 'class="source-links"' in document
    assert "공개 배포 전 권리 확인 필요:" not in script
    assert "사후 진단 일반화" not in document
    assert "This product uses the FRED® API" not in document
    assert "개인·비상업 파생 결과" not in document
    assert "투자 조언 아님" not in document


def test_h10_source_has_public_rights_copy_and_official_link() -> None:
    document = HTML_PATH.read_text(encoding="utf-8")
    script = JS_PATH.read_text(encoding="utf-8")
    assert (
        '<a href="https://www.federalreserve.gov/releases/h10/">'
        "Federal Reserve H.10</a>"
    ) in document
    assert (
        'if (value === "federal_reserve_board_public_domain_citation_requested") '
        'return "미 연준 공개 · 출처 표기"'
    ) in script


def test_dashboard_uses_only_real_payload_values() -> None:
    script = JS_PATH.read_text(encoding="utf-8")
    assert "fetch(DATA_URL" in script
    assert "validatePayload(payload)" in script
    assert "DataContractError" in script
    assert "Math.random" not in script
    assert "mockData" not in script
    assert "sampleData" not in script
    assert ".innerHTML" not in script
    assert 'payload.meta.mode' in script


def test_history_window_never_claims_more_weeks_than_are_available() -> None:
    script = JS_PATH.read_text(encoding="utf-8")
    assert _browser_history_window_cases() == ["all", 52, "all", 104, "all", "all"]
    assert "function syncHistoryWindowControl()" in script
    assert "preferredHistoryWindow: 52" in script
    assert "const requested = state.preferredHistoryWindow" in script
    assert 'option.textContent = available ? `전체 · ${available}주` : "전체"' in script
    assert "option.disabled = weeks > available" in script
    assert "`${range} · ${history.length}주 관측" in script


def test_three_state_and_health_contracts_are_explicit() -> None:
    script = JS_PATH.read_text(encoding="utf-8")
    for state_code in ("risk_on", "transition", "risk_off"):
        assert state_code in script
    for health_code in (
        "ok",
        "stale",
        "degraded",
        "quota_exhausted",
        "schema_changed",
        "revision_gap",
        "rights_unconfirmed",
        "license_blocked",
    ):
        assert health_code in script


def test_signed_market_percentages_do_not_use_probability_validation() -> None:
    script = JS_PATH.read_text(encoding="utf-8")
    assert "function formatSignedPercent" in script
    assert 'if (format === "percent") return formatSignedPercent(number)' in script
    assert 'if (format === "probability") return formatPercent(number)' in script
    assert "return formatSignedPercent(number);" in script


def test_styles_have_no_remote_assets_or_gradients() -> None:
    styles = CSS_PATH.read_text(encoding="utf-8").lower()
    assert "url(" not in styles
    assert "gradient" not in styles
    assert "@import" not in styles


def test_v4_uses_the_same_transition_dashboard_surfaces_as_v3() -> None:
    script = JS_PATH.read_text(encoding="utf-8")
    assert "[V3_RESULT_VERSION, V4_RESULT_VERSION].includes(resultVersion)" in script
    assert "if (!hasTransitionContract || !allRows.length)" in script
