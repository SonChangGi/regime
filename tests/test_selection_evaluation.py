from __future__ import annotations

import copy
from datetime import timedelta

import pandas as pd
import pytest

from regime_lab.analysis.labels import STATE_ORDER
from regime_lab.analysis.selection_evaluation import (
    build_selection_evaluation,
    validate_selection_evaluation,
)
from regime_lab.analysis.validation import evaluate_predictions


def _selection_predictions() -> pd.DataFrame:
    origins = pd.date_range("2021-01-01", periods=12, freq="W-FRI", tz="UTC")
    current = (
        "risk_on",
        "risk_on",
        "transition",
        "transition",
        "risk_off",
        "risk_off",
        "risk_on",
        "risk_on",
        "risk_on",
        "transition",
        "risk_off",
        "risk_off",
    )
    actual = (
        "risk_on",
        "transition",
        "transition",
        "risk_off",
        "risk_off",
        "risk_on",
        "risk_on",
        "risk_on",
        "transition",
        "risk_off",
        "risk_off",
        "risk_on",
    )
    rows: list[dict[str, object]] = []
    for model in ("markov", "fast", "noisy"):
        for position, origin in enumerate(origins):
            if model == "markov":
                hard_prediction = current[position]
                confidence = 0.8
            elif model == "fast":
                hard_prediction = actual[position]
                confidence = 0.8
            else:
                hard_prediction = (
                    "risk_off" if position in {0, 6} else actual[position]
                )
                confidence = 0.7
            remaining = (1.0 - confidence) / 2.0
            probability = {
                state: confidence if state == hard_prediction else remaining
                for state in STATE_ORDER
            }
            rows.append(
                {
                    "origin_date": origin,
                    "target_date": origin + timedelta(days=7),
                    "model": model,
                    "evaluation_split": "selection",
                    "current_state": current[position],
                    "actual": actual[position],
                    "predicted": hard_prediction,
                    "p_risk_on": probability["risk_on"],
                    "p_transition": probability["transition"],
                    "p_risk_off": probability["risk_off"],
                    "train_size": 520 + position,
                    "gap": 1,
                    "fallback": False,
                }
            )
    return pd.DataFrame(rows)


def _selection_diagnostics() -> pd.DataFrame:
    result = evaluate_predictions(_selection_predictions())
    result["selected"] = result["model"].eq("fast")
    return result


def _document() -> dict:
    return build_selection_evaluation(
        _selection_predictions(),
        _selection_diagnostics(),
        evidence_status="synthetic_fixture",
        mcs_block_length=4,
        mcs_resamples=99,
        mcs_random_state=17,
    )


def test_selection_evaluation_recomputes_all_supplemental_metrics() -> None:
    document = _document()

    validate_selection_evaluation(document)
    assert document["schema_version"] == "regime-selection-evaluation/1"
    assert document["evidence_status"] == "synthetic_fixture"
    assert document["role"] == "supplemental_not_selection_gate"
    assert document["evaluation_split"] == "selection"
    assert document["holdout_rows_used"] == 0
    assert document["selection_effect"] == "none"
    assert document["selected_champion_unchanged"] == "fast"
    assert document["common_origin_contract"]["origin_count"] == 12
    assert document["primary_metric_crosscheck"]["status"] == "matched"
    assert document["primary_metric_crosscheck"]["changes_holm_gate"] is False
    assert len(document["state_recall"]) == 3 * len(STATE_ORDER)
    assert all(
        0.0 <= row["normalized_entropy_sharpness"] <= 1.0
        for row in document["model_metrics"]
    )

    transitions = {
        row["model"]: row for row in document["transition_diagnostics"]
    }
    assert transitions["fast"]["transition_event_count"] == 6
    assert transitions["fast"]["detected_event_count"] == 6
    assert transitions["fast"]["mean_detection_delay_forecast_weeks"] == 0.0
    assert transitions["fast"]["false_alarm_count"] == 0
    assert transitions["markov"]["detected_event_count"] == 4
    assert transitions["markov"]["missed_event_count"] == 2
    assert transitions["markov"]["mean_detection_delay_forecast_weeks"] == 1.0
    assert transitions["noisy"]["false_alarm_count"] == 2

    mcs = document["model_confidence_set"]
    assert mcs["changes_champion"] is False
    assert mcs["observation_count"] == 12
    assert set(mcs["retained_models"]).union(mcs["eliminated_models"]) == set(
        document["candidate_set"]
    )


def test_selection_evaluation_rejects_any_holdout_or_nonselection_row() -> None:
    predictions = _selection_predictions()
    predictions.loc[predictions.index[-1], "evaluation_split"] = "holdout"

    with pytest.raises(ValueError, match="selection rows only"):
        build_selection_evaluation(
            predictions,
            _selection_diagnostics(),
            evidence_status="synthetic_fixture",
            mcs_resamples=99,
        )


def test_selection_evaluation_rejects_candidate_origin_mismatch() -> None:
    predictions = _selection_predictions()
    mask = predictions["model"].eq("noisy") & predictions["origin_date"].eq(
        predictions["origin_date"].min()
    )
    predictions.loc[mask, "target_date"] = (
        predictions.loc[mask, "target_date"] + timedelta(days=7)
    )

    with pytest.raises(ValueError, match="does not share exact"):
        build_selection_evaluation(
            predictions,
            _selection_diagnostics(),
            evidence_status="synthetic_fixture",
            mcs_resamples=99,
        )


def test_selection_evaluation_rejects_primary_score_or_argmax_drift() -> None:
    diagnostics = _selection_diagnostics()
    diagnostics.loc[diagnostics["model"].eq("fast"), "log_loss"] += 0.001
    with pytest.raises(ValueError, match="recomputed log_loss differs"):
        build_selection_evaluation(
            _selection_predictions(),
            diagnostics,
            evidence_status="synthetic_fixture",
            mcs_resamples=99,
        )

    predictions = _selection_predictions()
    predictions.loc[predictions.index[0], "predicted"] = "transition"
    with pytest.raises(ValueError, match="probability argmax"):
        build_selection_evaluation(
            predictions,
            _selection_diagnostics(),
            evidence_status="synthetic_fixture",
            mcs_resamples=99,
        )


def test_selection_evaluation_rejects_brier_drift_or_nonboolean_selection() -> None:
    diagnostics = _selection_diagnostics()
    diagnostics.loc[diagnostics["model"].eq("fast"), "brier"] += 0.001
    with pytest.raises(ValueError, match="recomputed brier differs"):
        build_selection_evaluation(
            _selection_predictions(),
            diagnostics,
            evidence_status="synthetic_fixture",
            mcs_resamples=99,
        )

    diagnostics = _selection_diagnostics()
    diagnostics["selected"] = diagnostics["selected"].map(str)
    with pytest.raises(ValueError, match="selected values must be booleans"):
        build_selection_evaluation(
            _selection_predictions(),
            diagnostics,
            evidence_status="synthetic_fixture",
            mcs_resamples=99,
        )


def test_selection_evaluation_rejects_nonofficial_gap() -> None:
    predictions = _selection_predictions()
    predictions.loc[predictions.index[0], "gap"] = 0

    with pytest.raises(ValueError, match="official gap=1"):
        build_selection_evaluation(
            predictions,
            _selection_diagnostics(),
            evidence_status="synthetic_fixture",
            mcs_resamples=99,
        )


def test_selection_evaluation_evidence_status_and_hash_fail_closed() -> None:
    with pytest.raises(ValueError, match="evidence_status"):
        build_selection_evaluation(
            _selection_predictions(),
            _selection_diagnostics(),
            evidence_status="real",
            mcs_resamples=99,
        )

    document = _document()
    tampered = copy.deepcopy(document)
    tampered["evidence_status"] = "historical_reconstructed_oos"
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_selection_evaluation(tampered)
