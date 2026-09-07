from copy import deepcopy
import math

import pytest

from regime_lab.research.forecast_contract import validate_forecast_improvement


@pytest.fixture
def block():
    rows = [
        {"origin_date": "2022-12-16T21:00:00Z", "target_date": "2022-12-23T21:00:00Z", "current_state": "risk_on", "actual": "risk_on", "predicted": "risk_on", "evaluation_split": "selection", "probabilities": {"risk_on": .7, "transition": .2, "risk_off": .1}},
        {"origin_date": "2023-01-06T21:00:00Z", "target_date": "2023-01-13T21:00:00Z", "current_state": "risk_on", "actual": "transition", "predicted": "transition", "evaluation_split": "holdout", "probabilities": {"risk_on": .2, "transition": .7, "risk_off": .1}},
    ]
    for row in rows:
        row["raw_probabilities"] = row["probabilities"].copy()
        row["calibration"] = {"temperature": 1., "rows": 0, "last_train_target": "2022-12-09T21:00:00Z"}
    metrics = {split: {"n_predictions": 1, "log_loss": -math.log(.7), "brier": .14,
                       "transition_event_count": events, "on_time_departure_count": events,
                       "false_alarm_count": 0, "false_alarms_per_year": 0,
                       "transition_recall": events, "transition_precision": events,
                       "worsening_event_count": events, "on_time_worsening_count": events,
                       "recovery_event_count": 0, "on_time_recovery_count": 0}
               for split, events in [("selection", 0), ("holdout", 1)]}
    latest = {"origin_date": "2023-01-13T21:00:00Z", "target_date": "2023-01-20T21:00:00Z", "current_state": "transition", "actual": None, "predicted": "transition",
              "calibration": {"temperature": 1., "rows": 0, "last_train_target": "2023-01-06T21:00:00Z"},
              "probabilities": {"risk_on": .2, "transition": .7, "risk_off": .1}, "raw_probabilities": {"risk_on": .2, "transition": .7, "risk_off": .1}}
    return {"schema_version": "regime-forecast-improvement/1", "selected_model": "boundary_filtered_history", "evidence_track": "reconstructed_market",
            "data_as_of": latest["origin_date"], "provenance": {key: "a"*64 for key in ["input_sha256", "code_sha256", "baseline_oos_sha256", "cache_key"]},
            "models": [{"id": model, "history": deepcopy(rows), "latest": deepcopy(latest), "metrics": deepcopy(metrics)} for model in ["boundary_filtered_history", "boundary_student_t"]]}


def test_recomputes_public_forecast_evidence(block):
    validate_forecast_improvement(block, data_as_of="2023-01-13T21:00:00Z")


@pytest.mark.parametrize("mutation", ["metric", "mixed_history", "bad_probability", "predicted", "unresolved_history", "latest_actual", "stale_latest", "latest_state", "hash", "publication_cutoff", "recall", "worsening_count", "recovery_count", "future_calibration", "calibration_count", "calibration_temperature", "latest_predicted"])
def test_rejects_misleading_public_forecast_evidence(block, mutation):
    model = block["models"][0]
    if mutation == "metric":
        model["metrics"]["holdout"]["on_time_departure_count"] = 0
    elif mutation == "mixed_history":
        model["history"][0]["origin_date"] = "2022-12-17T21:00:00Z"
        model["history"][0]["target_date"] = "2022-12-24T21:00:00Z"
    elif mutation == "bad_probability":
        model["history"][0]["probabilities"]["risk_on"] = .9
    elif mutation == "predicted":
        model["history"][0]["predicted"] = "risk_off"
    elif mutation == "unresolved_history":
        model["history"][-1]["actual"] = None
    elif mutation == "latest_actual":
        model["latest"]["actual"] = "risk_on"
    elif mutation == "stale_latest":
        model["latest"]["origin_date"] = "2023-01-06T21:00:00Z"
    elif mutation == "latest_state":
        model["latest"]["current_state"] = "risk_off"
    elif mutation == "hash":
        block["provenance"]["cache_key"] = "missing"
    elif mutation == "recall":
        model["metrics"]["holdout"]["transition_recall"] = .999
    elif mutation == "worsening_count":
        model["metrics"]["holdout"]["on_time_worsening_count"] = 19
    elif mutation == "recovery_count":
        model["metrics"]["holdout"]["on_time_recovery_count"] = 1
    elif mutation == "future_calibration":
        model["history"][0]["calibration"]["last_train_target"] = model["history"][0]["target_date"]
    elif mutation == "calibration_count":
        model["latest"]["calibration"]["rows"] = 99999
    elif mutation == "calibration_temperature":
        model["latest"]["calibration"]["temperature"] = -1
    elif mutation == "latest_predicted":
        model["latest"]["predicted"] = "risk_off"
    with pytest.raises(ValueError):
        validate_forecast_improvement(block, data_as_of="2023-01-20T21:00:00Z" if mutation == "publication_cutoff" else block["data_as_of"])
