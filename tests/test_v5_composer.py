from __future__ import annotations

import hashlib
import json
from datetime import timedelta

import numpy as np
import pandas as pd

from regime_lab.contract_v5 import (
    V5_FORECAST_COMPARISON_MODELS,
    validate_v5_payload,
)
from regime_lab.frozen_v4 import FROZEN_V4_BASELINE
from regime_lab.v5 import build_v5_payload, run_v5_directional_benchmark


def _inputs(rows: int = 700):
    index = pd.date_range("2012-01-06", periods=rows, freq="W-FRI", tz="UTC")
    pattern = (
        ["risk_on"] * 8
        + ["transition"] * 2
        + ["risk_off"] * 5
        + ["transition"] * 2
    )
    states = pd.Series(
        [pattern[position % len(pattern)] for position in range(rows)],
        index=index,
        dtype="object",
    )
    position = np.arange(rows, dtype=float)
    features = pd.DataFrame(
        {
            "trend": np.sin(position / 11.0),
            "stress": np.cos(position / 7.0),
        },
        index=index,
    )
    canonical = pd.DataFrame(
        {
            f"{asset.lower()}_close": 100.0
            * np.exp(np.cumsum(0.001 + 0.005 * np.sin(position / (7 + offset))))
            for offset, asset in enumerate(("SPY", "QQQ", "IWM", "TLT", "HYG", "UUP"))
        },
        index=index,
    )
    return index, states, features, canonical


def _transition_risk(origin: pd.Timestamp, current_state: str):
    probabilities = {1: 0.2, 4: 0.45, 13: 0.75}
    return {
        f"{horizon}w": {
            "probability": probability,
            "target_end": (origin + timedelta(days=7 * horizon)).date().isoformat(),
            "model": "markov",
            "threshold": 0.5,
            "fallback": False,
            "fallback_reason": "",
        }
        for horizon, probability in probabilities.items()
    }


def _model_forecasts(official: dict[str, object]) -> list[dict[str, object]]:
    probabilities = {
        "xgboost": {"risk_on": 0.25, "transition": 0.65, "risk_off": 0.1},
        "xgb_hazard_destination": {
            "risk_on": 0.2,
            "transition": 0.7,
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
    rows: list[dict[str, object]] = []
    for model in V5_FORECAST_COMPARISON_MODELS:
        if model == "markov":
            row = dict(official)
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


def _v4_payload(index: pd.DatetimeIndex, states: pd.Series):
    weekly = []
    for origin in index[-2:]:
        current_state = str(states.loc[origin])
        current_probs = {
            state: (0.7 if state == current_state else 0.15)
            for state in ("risk_on", "transition", "risk_off")
        }
        next_state = "transition"
        next_probs = {"risk_on": 0.2, "transition": 0.6, "risk_off": 0.2}
        next_week: dict[str, object] = {
            "state": next_state,
            "probabilities": next_probs,
            "confidence": 0.6,
            "entropy": 0.86,
            "date": (origin + timedelta(days=7)).date().isoformat(),
            "method": "champion_walk_forward_probability",
            "model": "markov",
            "fallback": False,
            "fallback_reason": "",
        }
        weekly.append(
            {
                "date": origin.date().isoformat(),
                "data_as_of": origin.isoformat(),
                "current": {
                    "state": current_state,
                    "probabilities": current_probs,
                    "confidence": 0.7,
                    "entropy": 0.75,
                    "method": "causal_rule_filtered_evidence",
                },
                "next_week": next_week,
                "transition_probability": 0.2,
                "transition_risk": _transition_risk(origin, current_state),
                "scores": {
                    "trend": 0.4,
                    "stress": -0.2,
                    "macro": 0.1,
                    "financial_conditions": 0.2,
                },
                "market": {},
                "top_drivers": [
                    {
                        "feature": "nfci__z_52w",
                        "label": "Chicago Fed 금융여건",
                        "value": -1.2,
                        "impact": 1.2,
                        "direction": "risk_on",
                        "method": "rolling_z_evidence_proxy",
                    }
                ],
                "health": {"status": "ok", "reason": "test"},
            }
        )
    generated = index[-1] + timedelta(hours=1)
    return {
        "meta": {
            "schema_version": "1.0.0",
            "result_version": "weekly-regime-result-v4",
            "generated_at": generated.isoformat(),
            "generation_id": "v5-compose-test",
            "data_as_of": index[-1].isoformat(),
            "mode": "demo",
            "status": "degraded",
            "timezone": "America/New_York",
            "cutoff_policy": "completed US market week",
            "transition_alert_thresholds": {"medium": 0.4, "high": 0.65},
            "supported_date_range": (
                f"{index[0].date().isoformat()}–{index[-1].date().isoformat()}"
            ),
            "warnings": ["Top drivers는 SHAP가 아닙니다."],
        },
        "states": [
            {"id": "risk_on"},
            {"id": "transition"},
            {"id": "risk_off"},
        ],
        "model": {
            "champion": "markov",
            "profile": "quick",
            "version": "weekly-nondl-structural-v4",
            "label_version": "market-causal-3state-v1",
            "feature_set_version": "weekly-pit-structural-v4",
            "selection_status": "provisional_predeployment",
            "leaderboard": [
                {"name": name} for name in V5_FORECAST_COMPARISON_MODELS
            ],
            "transition_leaderboard": [],
            "holdout_diagnostic": {"status": "ok"},
        },
        "weekly": weekly,
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
                "id": "test",
                "category": "test",
                "frequency": "weekly",
                "source": "fixture",
            }
        ],
    }


def test_v5_composer_changes_semantics_without_allocation_output() -> None:
    index, states, features, canonical = _inputs()
    directional = run_v5_directional_benchmark(
        features,
        states,
        profile_name="quick",
        selection_end="2023-01-01",
    )
    baseline = dict(FROZEN_V4_BASELINE)
    prereg_sha = hashlib.sha256(b"preregistered-v5").hexdigest()

    payload, conditional = build_v5_payload(
        _v4_payload(index, states),
        canonical=canonical,
        features=features,
        states=states,
        directional=directional,
        baseline_v4=baseline,
        structural_preregistration_sha256=prereg_sha,
        duration_bootstrap_resamples=1,
        outcome_bootstrap_resamples=1,
    )
    payload["model"]["forecast_comparison"] = {
        "role": "research_comparison",
        "horizon_weeks": 1,
        "models": list(V5_FORECAST_COMPARISON_MODELS),
    }
    for week in payload["weekly"]:
        week["model_forecasts"] = _model_forecasts(week["next_week"])

    core_paths = {
        "oos_predictions": "oos-predictions.csv",
        "model_leaderboard": "model-leaderboard.csv",
        "walk_forward_splits": "walk-forward-splits.csv",
        "selection_diagnostics": "selection-diagnostics.csv",
        "stacking_weights": "stacking-weights.csv",
        "multiscale_ensemble_scales": "multiscale-ensemble-scales.csv",
    }
    payload["model"]["core_artifacts"] = {
        key: {"path": path, "row_count": 1, "sha256": "a" * 64}
        for key, path in core_paths.items()
    }
    payload["model"]["structural_models"]["causal_multiscale_ensemble"][
        "sidecar"
    ] = dict(
        payload["model"]["core_artifacts"]["multiscale_ensemble_scales"]
    )
    candidate_names = (
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
    candidate_manifest = {
        "schema_version": "1.0.0",
        "profile": "quick",
        "random_state": 17,
        "models": [{"name": name} for name in candidate_names],
    }
    payload["model"]["candidate_manifest"] = candidate_manifest
    payload["model"]["candidate_manifest_sha256"] = hashlib.sha256(
        json.dumps(
            candidate_manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    payload["model"]["champion_core_feature_set_version"] = (
        "weekly-pit-structural-v4"
    )
    payload["model"]["evidence_artifacts"] = {
        "state_membership_history": {
            "path": "state-membership-history.csv",
            "row_count": len(states),
            "sha256": "d" * 64,
            "label_fit_weeks": 520,
            "label_fit_end": index[519].isoformat(),
            "initial_state": "transition",
            "method": "risk_score_anchor_membership",
        },
        "weekly_state_forecasts": {
            "path": "weekly-state-forecasts-v5.csv",
            "row_count": len(payload["weekly"]),
            "sha256": "e" * 64,
        },
    }

    validate_v5_payload(payload)
    assert payload["meta"]["schema_version"] == "2.0.0"
    assert "probabilities" not in payload["weekly"][-1]["current"]
    assert "memberships" in payload["weekly"][-1]["current"]
    assert "scores" not in payload["weekly"][-1]
    assert "top_drivers" not in payload["weekly"][-1]
    assert payload["weekly"][-1]["fx_context"]["status"] == "unavailable"
    assert len(payload["research"]["conditional_asset_stats"]["rows"]) == 54
    assert len(conditional.statistics) == 54
    assert "allocation" not in repr(payload["research"])
    assert payload["model"]["forecast_comparison"]["models"] == list(
        V5_FORECAST_COMPARISON_MODELS
    )
    assert [
        row["model"] for row in payload["weekly"][-1]["model_forecasts"]
    ] == list(V5_FORECAST_COMPARISON_MODELS)
    assert payload["weekly"][-1]["model_forecasts"][0] == {
        **payload["weekly"][-1]["next_week"],
        "method": "model_comparison_walk_forward_probability",
    }
