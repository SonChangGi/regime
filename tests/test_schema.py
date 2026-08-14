from __future__ import annotations

import pytest

from regime_lab.pipeline import BASELINE_V3, STRUCTURAL_PREREGISTRATION
from regime_lab.schema import (
    ContractError,
    FROZEN_V4_BASELINE_V3,
    FROZEN_V4_STRUCTURAL_PREREGISTRATION,
    SCHEMA_VERSION,
    validate_dashboard_payload,
)


def _payload() -> dict:
    estimate = {
        "state": "transition",
        "probabilities": {"risk_on": 0.25, "transition": 0.5, "risk_off": 0.25},
        "confidence": 0.5,
        "entropy": 0.95,
    }
    return {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "generated_at": "2026-08-11T00:00:00Z",
            "data_as_of": "2026-08-07",
            "mode": "demo",
            "timezone": "America/New_York",
        },
        "states": [
            {"id": "risk_on", "label": "Risk-on"},
            {"id": "transition", "label": "Transition"},
            {"id": "risk_off", "label": "Risk-off"},
        ],
        "model": {
            "champion": "markov",
            "selection_status": "provisional_predeployment",
            "leaderboard": [],
        },
        "weekly": [
            {
                "date": "2026-08-07",
                "current": estimate,
                "next_week": estimate,
                "transition_probability": 0.2,
                "scores": {
                    "trend": 0.1,
                    "stress": -0.1,
                    "macro": 0.0,
                    "financial_conditions": 0.0,
                },
            }
        ],
        "sources": [{"id": "fixture", "status": "degraded"}],
        "feature_catalog": [
            {
                "id": "fixture",
                "category": "test",
                "frequency": "weekly",
                "source": "fixture",
            }
        ],
    }


def test_dashboard_contract_accepts_valid_payload() -> None:
    validate_dashboard_payload(_payload())


def test_dashboard_contract_rejects_probabilities_that_do_not_sum_to_one() -> None:
    payload = _payload()
    payload["weekly"][0]["next_week"]["probabilities"]["risk_on"] = 0.9
    with pytest.raises(ContractError, match="sum to one"):
        validate_dashboard_payload(payload)


def test_dashboard_contract_rejects_a_fourth_probability_state() -> None:
    payload = _payload()
    payload["weekly"][0]["next_week"]["probabilities"]["extra_state"] = 0.0
    with pytest.raises(ContractError, match="exactly"):
        validate_dashboard_payload(payload)


def test_dashboard_contract_rejects_non_finite_probability() -> None:
    payload = _payload()
    payload["weekly"][0]["next_week"]["probabilities"]["risk_on"] = float("nan")
    with pytest.raises(ContractError, match="finite"):
        validate_dashboard_payload(payload)


@pytest.mark.parametrize(("field", "value"), [("transition_probability", float("nan")), ("scores", float("inf"))])
def test_dashboard_contract_rejects_non_finite_weekly_metrics(field, value) -> None:
    payload = _payload()
    if field == "scores":
        payload["weekly"][0]["scores"]["trend"] = value
    else:
        payload["weekly"][0][field] = value
    with pytest.raises(ContractError):
        validate_dashboard_payload(payload)


def test_dashboard_contract_rejects_non_monotonic_dates() -> None:
    payload = _payload()
    payload["weekly"].append(dict(payload["weekly"][0]))
    with pytest.raises(ContractError, match="strictly increasing"):
        validate_dashboard_payload(payload)


def test_dashboard_contract_requires_feature_catalog() -> None:
    payload = _payload()
    payload.pop("feature_catalog")
    with pytest.raises(ContractError, match="feature_catalog"):
        validate_dashboard_payload(payload)


@pytest.mark.parametrize("selection_status", [None, "final", "provisional_typo"])
def test_dashboard_contract_requires_exact_provisional_selection_status(
    selection_status: str | None,
) -> None:
    payload = _payload()
    if selection_status is None:
        payload["model"].pop("selection_status")
    else:
        payload["model"]["selection_status"] = selection_status
    with pytest.raises(ContractError, match="selection_status"):
        validate_dashboard_payload(payload)


def _upgrade_to_v3(payload: dict) -> dict:
    payload["meta"]["result_version"] = "weekly-regime-result-v3"
    payload["meta"]["generation_id"] = "20260812T000000.000000Z"
    payload["model"].update(
        {
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
            "transition_champions": {
                "1w": "markov_hazard",
                "4w": "markov_hazard",
                "13w": "duration_tvtp_hurdle",
            },
            "transition_leaderboard": [
                {
                    "horizon_weeks": horizon,
                    "model": "markov_hazard",
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
            "shadow_nowcast": {
                "status": "shadow_only",
                "canonical_target": False,
            },
        }
    )
    row = payload["weekly"][0]
    row["next_week"]["date"] = "2026-08-14"
    row["next_week"]["probabilities"] = {
        "risk_on": 0.1,
        "transition": 0.8,
        "risk_off": 0.1,
    }
    row["transition_probability"] = 0.2
    row["transition_risk"] = {
        f"{horizon}w": {
            "probability": 0.2 + 0.01 * horizon,
            "target_end": f"2026-{8 if horizon < 13 else 11:02d}-{14 if horizon == 1 else 0:02d}",
            "model": "markov_hazard",
            "threshold": 0.5,
            "fallback": False,
            "fallback_reason": "",
        }
        for horizon in (1, 4, 13)
    }
    row["transition_risk"]["1w"]["probability"] = 0.2
    row["transition_risk"]["1w"]["target_end"] = "2026-08-14"
    row["transition_risk"]["4w"]["target_end"] = "2026-09-04"
    row["transition_risk"]["13w"]["target_end"] = "2026-11-06"
    return payload


def _upgrade_to_v4(payload: dict) -> dict:
    payload = _upgrade_to_v3(payload)
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


def test_dashboard_contract_accepts_additive_v3_contract() -> None:
    validate_dashboard_payload(_upgrade_to_v3(_payload()))


def test_dashboard_contract_accepts_exact_v4_structural_contract() -> None:
    validate_dashboard_payload(_upgrade_to_v4(_payload()))


def test_v4_producer_uses_schema_canonical_frozen_contracts() -> None:
    assert BASELINE_V3 == FROZEN_V4_BASELINE_V3
    assert STRUCTURAL_PREREGISTRATION == FROZEN_V4_STRUCTURAL_PREREGISTRATION


def test_dashboard_contract_rejects_unknown_declared_result_version() -> None:
    payload = _payload()
    payload["meta"]["result_version"] = "weekly-regime-result-v5"
    with pytest.raises(ContractError, match="unsupported result version"):
        validate_dashboard_payload(payload)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["model"].__setitem__("feature_set_version", "wrong"),
        lambda payload: payload["model"]["baseline_v3"].__setitem__("payload_sha256", "A" * 64),
        lambda payload: payload["model"]["baseline_v3"].__setitem__("payload_sha256", "0" * 64),
        lambda payload: payload["model"]["baseline_v3"].__setitem__("artifacts_inventory_sha256", "1" * 64),
        lambda payload: payload["model"]["baseline_v3"].__setitem__("champion", "xgboost"),
        lambda payload: payload["model"]["baseline_v3"].__setitem__("captured_at", "2026-08-14"),
        lambda payload: payload["model"]["baseline_v3"].__setitem__("extra", True),
        lambda payload: payload["model"]["structural_preregistration"].__setitem__("path", "config/structural_v5.json"),
        lambda payload: payload["model"]["structural_preregistration"].__setitem__("sha256", "0" * 64),
        lambda payload: payload["model"]["evidence_artifacts"]["state_label_history"].__setitem__("path", "../state-label-history.csv"),
        lambda payload: payload["model"]["evidence_artifacts"]["state_label_history"].__setitem__("sha256", "bad"),
        lambda payload: payload["model"]["evidence_artifacts"]["weekly_state_forecasts"].__setitem__("row_count", 2),
        lambda payload: payload["model"]["structural_models"]["causal_dynamic_ensemble"].__setitem__("half_life_weeks", 51),
        lambda payload: payload["model"]["structural_models"]["joint_survival_hazard"].__setitem__("horizons_weeks", [1, 13]),
        lambda payload: payload["model"]["ablation"].__setitem__("may_change_published_variant", True),
        lambda payload: payload["model"]["ablation"].__setitem__("manifest_sha256", "bad"),
    ],
)
def test_dashboard_contract_rejects_v4_contract_tampering(mutation) -> None:
    payload = _upgrade_to_v4(_payload())
    mutation(payload)
    with pytest.raises(ContractError):
        validate_dashboard_payload(payload)


def test_dashboard_contract_rejects_v3_transition_alias_drift() -> None:
    payload = _upgrade_to_v3(_payload())
    payload["weekly"][0]["transition_risk"]["1w"]["probability"] = 0.3
    with pytest.raises(ContractError, match="aliases disagree"):
        validate_dashboard_payload(payload)


def test_dashboard_contract_rejects_v3_horizon_date_drift() -> None:
    payload = _upgrade_to_v3(_payload())
    payload["weekly"][0]["transition_risk"]["13w"]["target_end"] = "2026-11-13"
    with pytest.raises(ContractError, match="exactly 13 weeks"):
        validate_dashboard_payload(payload)


def test_dashboard_contract_rejects_null_ap_when_events_exist() -> None:
    payload = _upgrade_to_v3(_payload())
    row = payload["model"]["transition_leaderboard"][0]
    row["event_count"] = 1
    row["non_event_count"] = 9
    with pytest.raises(ContractError, match="average_precision"):
        validate_dashboard_payload(payload)
