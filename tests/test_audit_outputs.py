from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_outputs.py"
SPEC = importlib.util.spec_from_file_location("audit_outputs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit_outputs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_outputs)


def test_selection_audit_rebuilds_every_published_diagnostic_field() -> None:
    models = ("majority", "persistence", "markov", "ridge_logistic")
    actual = ("risk_on", "transition", "risk_off", "risk_on")
    probabilities = {
        "majority": (0.70, 0.15, 0.15),
        "persistence": (0.55, 0.25, 0.20),
        "markov": (0.60, 0.25, 0.15),
        "ridge_logistic": (0.58, 0.27, 0.15),
    }
    rows: list[dict[str, object]] = []
    for model in models:
        for index, state in enumerate(actual):
            risk_on, transition, risk_off = probabilities[model]
            if state == "transition":
                risk_on, transition, risk_off = transition, risk_on, risk_off
            elif state == "risk_off":
                risk_on, transition, risk_off = risk_off, transition, risk_on
            rows.append(
                {
                    "model": model,
                    "target_date": pd.Timestamp("2020-01-03", tz="UTC")
                    + timedelta(weeks=index),
                    "actual": state,
                    "current_state": actual[max(0, index - 1)],
                    "p_risk_on": risk_on,
                    "p_transition": transition,
                    "p_risk_off": risk_off,
                    "fallback": False,
                }
            )
    predictions = pd.DataFrame(rows)
    metrics = audit_outputs.probability_metrics(predictions)

    _, diagnostics = audit_outputs.choose_selection_champion(metrics, predictions)

    assert {
        "model",
        "reference_model",
        "is_reference",
        "selected",
        "gate_passed",
        "gate_reason",
        "log_loss",
        "reference_log_loss",
        "absolute_log_loss_improvement",
        "brier",
        "reference_brier",
        "brier_difference",
        "fallback_count",
        "raw_p_value",
        "holm_adjusted_p_value",
        "n_predictions",
        "bootstrap_block_weeks",
        "bootstrap_effective_block_weeks",
        "bootstrap_resamples",
        "bootstrap_seed",
        "alpha",
        "minimum_log_loss_improvement",
        "brier_tolerance",
    } <= set(diagnostics.columns)
    assert len(diagnostics) == len(models)
    assert diagnostics["selected"].sum() == 1


def _binary_predictions(*, all_non_events: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    actual = (False, False, False, False) if all_non_events else (
        False,
        True,
        False,
        True,
    )
    probabilities = {
        "empirical_hazard": (0.20, 0.55, 0.30, 0.60),
        "markov_hazard": (0.15, 0.70, 0.20, 0.75),
        "regularized_logistic": (0.10, 0.80, 0.15, 0.85),
    }
    for model, values in probabilities.items():
        for index, (event, probability) in enumerate(zip(actual, values, strict=True)):
            rows.append(
                {
                    "horizon": 4,
                    "evaluation_split": "selection",
                    "model": model,
                    "actual_change": event,
                    "p_change": probability,
                    "predicted_change": probability >= 0.5,
                    "fallback": False,
                    "calibration_fallback": False,
                    "origin_date": pd.Timestamp("2020-01-03", tz="UTC")
                    + timedelta(weeks=index),
                }
            )
    return pd.DataFrame(rows)


def test_transition_metrics_keep_no_event_average_precision_null() -> None:
    metrics = audit_outputs.transition_probability_metrics(
        _binary_predictions(all_non_events=True)
    )

    assert metrics["average_precision"].isna().all()
    assert (metrics["event_count"] == 0).all()
    assert (metrics["precision"] == 0.0).all()
    assert (metrics["recall"] == 0.0).all()


def test_transition_metric_tamper_is_detected_independently() -> None:
    predictions = _binary_predictions()
    expected = audit_outputs.transition_probability_metrics(predictions)
    tampered = expected.copy()
    tampered.loc[tampered.index[0], "brier"] += 0.01

    with pytest.raises(audit_outputs.AuditFailure, match="brier mismatch"):
        audit_outputs._compare_transition_metric_rows(
            expected,
            tampered,
            context="tampered leaderboard",
        )


def test_transition_horizon_tamper_is_detected_on_calendar_dates() -> None:
    frame = pd.DataFrame(
        {
            "origin": pd.to_datetime(["2026-01-02T21:00:00Z"]),
            "target": pd.to_datetime(["2026-01-31T21:00:00Z"]),
            "horizon": [4],
        }
    )

    with pytest.raises(audit_outputs.AuditFailure, match=r"7\*h"):
        audit_outputs.require_calendar_horizon(
            frame,
            "origin",
            "target",
            "horizon",
            "tampered transition split",
        )


def test_transition_probability_tamper_outside_open_interval_is_rejected() -> None:
    predictions = _binary_predictions()
    predictions.loc[predictions.index[0], "p_change"] = 1.0

    with pytest.raises(audit_outputs.AuditFailure, match="probability invalid"):
        audit_outputs.transition_probability_metrics(predictions)


def test_transition_threshold_is_selection_only_and_deterministic() -> None:
    history = _binary_predictions().loc[
        lambda frame: frame["model"].eq("regularized_logistic")
    ]

    threshold, method = audit_outputs.transition_threshold(
        history,
        minimum_rows=3,
    )
    tampered = history.copy()
    tampered.loc[tampered["actual_change"], "p_change"] = (0.20, 0.25)
    tampered_threshold, _ = audit_outputs.transition_threshold(
        tampered,
        minimum_rows=3,
    )

    assert method == "prequential_balanced_accuracy"
    assert 0.05 <= threshold <= 0.95
    assert threshold != tampered_threshold


def test_transition_calibration_is_selection_history_only() -> None:
    history = _binary_predictions().loc[
        lambda frame: frame["model"].eq("regularized_logistic")
    ].copy()
    history["raw_p_change"] = history["p_change"]

    probability, method, fallback, reason = audit_outputs.transition_calibration(
        0.40, history, minimum_rows=3
    )
    changed = history.copy()
    changed["actual_change"] = ~changed["actual_change"]
    changed_probability, _, _, _ = audit_outputs.transition_calibration(
        0.40, changed, minimum_rows=3
    )

    assert method == "identity"
    assert fallback is True
    assert reason == "insufficient_event_classes"
    assert probability == 0.40
    assert changed_probability == 0.40

    longer = pd.concat([history] * 3, ignore_index=True)
    probability, method, fallback, reason = audit_outputs.transition_calibration(
        0.40, longer, minimum_rows=3
    )
    assert 0.0 < probability < 1.0
    assert method == "prequential_platt_logit"
    assert fallback is False
    assert reason == ""


def test_effective_transition_fallback_combines_all_degradation_channels() -> None:
    row = pd.Series(
        {
            "fallback": False,
            "fallback_reason": "forbidden_transitions_routed_adjacent",
            "calibration_fallback": True,
            "calibration_fallback_reason": "insufficient_event_classes",
            "threshold_method": "fallback_0.5:insufficient_event_classes",
        }
    )

    fallback, reason = audit_outputs.effective_transition_fallback(row)

    assert fallback is True
    assert reason == (
        "forbidden_transitions_routed_adjacent; "
        "calibration:insufficient_event_classes; "
        "threshold:fallback_0.5:insufficient_event_classes"
    )


def test_effective_transition_fallback_ignores_csv_nan_reasons() -> None:
    row = pd.Series(
        {
            "fallback": False,
            "fallback_reason": float("nan"),
            "calibration_fallback": False,
            "calibration_fallback_reason": float("nan"),
            "threshold_method": "prequential_balanced_accuracy",
        }
    )

    assert audit_outputs.effective_transition_fallback(row) == (False, "")


def test_v4_joint_probability_recomputation_detects_tamper() -> None:
    origins = pd.to_datetime(
        ["2022-12-16T21:00:00Z", "2023-01-06T21:00:00Z"]
    )
    targets = origins + timedelta(days=7)
    xgb_rows = []
    joint_rows = []
    transition_rows = []
    for index, (origin, target) in enumerate(zip(origins, targets, strict=True)):
        state = "risk_on" if index == 0 else "transition"
        actual = "transition" if index == 0 else "transition"
        probability = (0.65, 0.25, 0.10)
        hazard = 0.20 + 0.05 * index
        joint = audit_outputs.compose_joint_probability(probability, hazard, state)
        split = "selection" if index == 0 else "holdout"
        common = {
            "origin_date": origin,
            "target_date": target,
            "evaluation_split": split,
            "current_state": state,
            "actual": actual,
            "fallback": False,
            "fallback_reason": "",
        }
        xgb_rows.append(
            {
                **common,
                "model": "xgboost",
                "predicted": "risk_on",
                "p_risk_on": probability[0],
                "p_transition": probability[1],
                "p_risk_off": probability[2],
            }
        )
        joint_rows.append(
            {
                **common,
                "model": "xgb_hazard_destination",
                "predicted": audit_outputs.STATE_ORDER[int(np.argmax(joint))],
                "p_risk_on": joint[0],
                "p_transition": joint[1],
                "p_risk_off": joint[2],
                "direct_jump_floor": audit_outputs.V4_DIRECT_JUMP_FLOOR,
            }
        )
        transition_rows.append(
            {
                "origin_date": origin,
                "target_end": target,
                "horizon": 1,
                "model": "binary_xgboost",
                "evaluation_split": (
                    "selection" if split == "selection"
                    else "retrospective_diagnostic"
                ),
                "current_state": state,
                "actual_change": actual != state,
                "p_change": hazard,
                "fallback": False,
                "calibration_fallback": False,
            }
        )
    predictions = pd.DataFrame([*xgb_rows, *joint_rows])
    transitions = pd.DataFrame(transition_rows)

    summary = audit_outputs.audit_joint_predictions(predictions, transitions)
    assert summary["origins"] == 2

    tampered = predictions.copy()
    mask = tampered["model"].eq("xgb_hazard_destination")
    tampered.loc[mask, "p_risk_on"] += 0.01
    tampered.loc[mask, "p_transition"] -= 0.01
    with pytest.raises(audit_outputs.AuditFailure, match="recomputation mismatch"):
        audit_outputs.audit_joint_predictions(tampered, transitions)


def test_v4_feature_manifest_requires_exact_one_time_assignment(
    tmp_path: Path,
) -> None:
    groups = []
    for group_id in audit_outputs.V4_ABLATION_VARIANTS["all_structural"]:
        features = (
            sorted(audit_outputs.V4_FINANCIAL_CONDITION_FEATURES)
            if group_id == "financial_conditions"
            else [
                "legacy_feature"
                if group_id == "legacy_v3"
                else f"{audit_outputs.V4_FEATURE_GROUP_PREFIXES[group_id][0]}value"
            ]
        )
        groups.append(
            {
                "id": group_id,
                "feature_count": len(features),
                "features": features,
            }
        )
    body = {
        "feature_set_version": audit_outputs.V4_FEATURE_SET_VERSION,
        "feature_count": sum(group["feature_count"] for group in groups),
        "groups": groups,
    }
    document = {**body, "sha256": audit_outputs.canonical_json_sha256(body)}
    (tmp_path / "feature-manifest.json").write_text(
        json.dumps(document), encoding="utf-8"
    )
    payload = {"model": {"feature_manifest_sha256": document["sha256"]}}

    result = audit_outputs.audit_feature_manifest(payload, tmp_path)
    assert result["feature_count"] == body["feature_count"]

    body["groups"][0]["features"] = body["groups"][1]["features"]
    document = {**body, "sha256": audit_outputs.canonical_json_sha256(body)}
    (tmp_path / "feature-manifest.json").write_text(
        json.dumps(document), encoding="utf-8"
    )
    payload["model"]["feature_manifest_sha256"] = document["sha256"]
    with pytest.raises(audit_outputs.AuditFailure, match="more than one"):
        audit_outputs.audit_feature_manifest(payload, tmp_path)


def test_v4_feature_manifest_rejects_financial_condition_drift(
    tmp_path: Path,
) -> None:
    groups = []
    for group_id in audit_outputs.V4_ABLATION_VARIANTS["all_structural"]:
        features = (
            sorted(audit_outputs.V4_FINANCIAL_CONDITION_FEATURES)
            if group_id == "financial_conditions"
            else [
                "legacy_feature"
                if group_id == "legacy_v3"
                else f"{audit_outputs.V4_FEATURE_GROUP_PREFIXES[group_id][0]}value"
            ]
        )
        groups.append(
            {
                "id": group_id,
                "feature_count": len(features),
                "features": features,
            }
        )

    financial = next(group for group in groups if group["id"] == "financial_conditions")
    financial["features"].remove("anfci__change_1w")
    financial["features"].append("anfci__change_4w_z_52w")
    body = {
        "feature_set_version": audit_outputs.V4_FEATURE_SET_VERSION,
        "feature_count": sum(group["feature_count"] for group in groups),
        "groups": groups,
    }
    document = {**body, "sha256": audit_outputs.canonical_json_sha256(body)}
    (tmp_path / "feature-manifest.json").write_text(
        json.dumps(document), encoding="utf-8"
    )
    payload = {"model": {"feature_manifest_sha256": document["sha256"]}}

    with pytest.raises(
        audit_outputs.AuditFailure, match="four preregistered ANFCI"
    ):
        audit_outputs.audit_feature_manifest(payload, tmp_path)


def test_v4_feature_manifest_keeps_only_anfci_availability_in_legacy(
    tmp_path: Path,
) -> None:
    groups = []
    for group_id in audit_outputs.V4_ABLATION_VARIANTS["all_structural"]:
        if group_id == "financial_conditions":
            features = sorted(audit_outputs.V4_FINANCIAL_CONDITION_FEATURES)
        elif group_id == "legacy_v3":
            features = [
                "legacy_feature",
                *sorted(audit_outputs.V4_ANFCI_LEGACY_AVAILABILITY_FEATURES),
            ]
        else:
            features = [f"{audit_outputs.V4_FEATURE_GROUP_PREFIXES[group_id][0]}value"]
        groups.append(
            {
                "id": group_id,
                "feature_count": len(features),
                "features": features,
            }
        )

    def write_document() -> dict[str, object]:
        body = {
            "feature_set_version": audit_outputs.V4_FEATURE_SET_VERSION,
            "feature_count": sum(group["feature_count"] for group in groups),
            "groups": groups,
        }
        document = {**body, "sha256": audit_outputs.canonical_json_sha256(body)}
        (tmp_path / "feature-manifest.json").write_text(
            json.dumps(document), encoding="utf-8"
        )
        return document

    document = write_document()
    payload = {"model": {"feature_manifest_sha256": document["sha256"]}}
    audit_outputs.audit_feature_manifest(payload, tmp_path)

    legacy = next(group for group in groups if group["id"] == "legacy_v3")
    legacy["features"].append("anfci__change_13w")
    legacy["feature_count"] += 1
    document = write_document()
    payload["model"]["feature_manifest_sha256"] = document["sha256"]
    with pytest.raises(audit_outputs.AuditFailure, match="unregistered ANFCI"):
        audit_outputs.audit_feature_manifest(payload, tmp_path)


def test_v4_stacking_weight_recomputation_is_strictly_pre_origin() -> None:
    origins = pd.date_range("2020-01-03", periods=4, freq="7D", tz="UTC")
    rows = []
    for model_index, model in enumerate(audit_outputs.V4_STRUCTURAL_EXPERTS):
        for index, origin in enumerate(origins):
            probability = np.asarray((0.60, 0.25, 0.15), dtype=float)
            probability = np.roll(probability, model_index)
            rows.append(
                {
                    "origin_date": origin,
                    "target_date": origin + timedelta(days=7),
                    "evaluation_split": "selection",
                    "model": model,
                    "current_state": "risk_on",
                    "actual": audit_outputs.STATE_ORDER[index % 3],
                    "fallback": False,
                    "p_risk_on": probability[0],
                    "p_transition": probability[1],
                    "p_risk_off": probability[2],
                }
            )
    history = pd.DataFrame(rows)
    origin = origins[-1]
    evidence = audit_outputs._discounted_weight_evidence(
        history,
        origin_date=origin,
        current_fallbacks={name: False for name in audit_outputs.V4_STRUCTURAL_EXPERTS},
        minimum_history_rows=1,
    )
    altered = history.copy()
    altered.loc[
        altered["origin_date"].eq(origin), "actual"
    ] = "risk_off"
    altered_evidence = audit_outputs._discounted_weight_evidence(
        altered,
        origin_date=origin,
        current_fallbacks={name: False for name in audit_outputs.V4_STRUCTURAL_EXPERTS},
        minimum_history_rows=1,
    )

    assert {name: row["weight"] for name, row in evidence.items()} == {
        name: row["weight"] for name, row in altered_evidence.items()
    }
    assert all(
        pd.Timestamp(row["latest_eligible_target_date"]) < origin
        for row in evidence.values()
    )

    degraded = history.copy()
    degraded.loc[
        degraded["origin_date"].eq(origins[0])
        & degraded["model"].eq("xgboost"),
        "fallback",
    ] = True
    common_evidence = audit_outputs._discounted_weight_evidence(
        degraded,
        origin_date=origin,
        current_fallbacks={name: False for name in audit_outputs.V4_STRUCTURAL_EXPERTS},
        minimum_history_rows=1,
    )
    assert {row["common_history_rows"] for row in evidence.values()} == {2}
    assert {row["common_history_rows"] for row in common_evidence.values()} == {1}
    assert len({row["history_rows"] for row in common_evidence.values()}) == 1
    assert all(
        row["history_rows"] == row["common_history_rows"]
        for row in common_evidence.values()
    )
    excluded_tamper = degraded.copy()
    excluded = excluded_tamper["origin_date"].eq(origins[0])
    excluded_tamper.loc[excluded, "actual"] = "risk_off"
    excluded_tamper.loc[
        excluded, ["p_risk_on", "p_transition", "p_risk_off"]
    ] = [0.01, 0.01, 0.98]
    tampered_common_evidence = audit_outputs._discounted_weight_evidence(
        excluded_tamper,
        origin_date=origin,
        current_fallbacks={name: False for name in audit_outputs.V4_STRUCTURAL_EXPERTS},
        minimum_history_rows=1,
    )
    assert {
        name: row["weight"] for name, row in common_evidence.items()
    } == {
        name: row["weight"] for name, row in tampered_common_evidence.items()
    }


def test_v4_joint_survival_forecast_identity_detects_tamper(
    tmp_path: Path,
) -> None:
    hazard = 0.12
    rows = [
        {
            "origin_date": "2026-08-07T20:00:00Z",
            "horizon_weeks": horizon,
            "one_week_hazard": hazard,
            "step_hazards": json.dumps([hazard] * horizon),
            "cumulative_p_change": 1.0 - (1.0 - hazard) ** horizon,
            "role": "shadow_coherence_benchmark",
        }
        for horizon in (1, 4, 13)
    ]
    path = tmp_path / "joint-survival-forecasts.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    pd.DataFrame(
        [
            {
                "origin_date": "2026-08-07T20:00:00Z",
                "target_date": "2026-08-14T20:00:00Z",
                "model": "binary_xgboost",
                "p_change": hazard,
            }
        ]
    ).to_csv(tmp_path / "structural-forecasts.csv", index=False)

    result = audit_outputs.audit_joint_survival_forecasts(tmp_path)
    assert result["horizons"] == [1, 4, 13]

    rows[-1]["cumulative_p_change"] -= 0.01
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(audit_outputs.AuditFailure, match="product identity"):
        audit_outputs.audit_joint_survival_forecasts(tmp_path)


def test_v4_joint_survival_rejects_nonfrozen_steps_and_binary_source_drift(
    tmp_path: Path,
) -> None:
    hazard = 0.12
    rows = [
        {
            "origin_date": "2026-08-07T20:00:00Z",
            "horizon_weeks": horizon,
            "one_week_hazard": hazard,
            "step_hazards": json.dumps([hazard] * horizon),
            "cumulative_p_change": 1.0 - (1.0 - hazard) ** horizon,
            "role": "shadow_coherence_benchmark",
        }
        for horizon in (1, 4, 13)
    ]
    survival_path = tmp_path / "joint-survival-forecasts.csv"
    pd.DataFrame(rows).to_csv(survival_path, index=False)
    structural_path = tmp_path / "structural-forecasts.csv"
    pd.DataFrame(
        [
            {
                "origin_date": "2026-08-07T20:00:00Z",
                "target_date": "2026-08-14T20:00:00Z",
                "model": "binary_xgboost",
                "p_change": hazard,
            }
        ]
    ).to_csv(structural_path, index=False)

    rows[-1]["step_hazards"] = json.dumps(
        [hazard - 0.01] * 6 + [hazard + 0.01] * 7
    )
    rows[-1]["cumulative_p_change"] = 1.0 - np.prod(
        1.0 - np.asarray(json.loads(rows[-1]["step_hazards"]), dtype=float)
    )
    pd.DataFrame(rows).to_csv(survival_path, index=False)
    with pytest.raises(audit_outputs.AuditFailure, match="do not repeat"):
        audit_outputs.audit_joint_survival_forecasts(tmp_path)

    rows[-1]["step_hazards"] = json.dumps([hazard] * 13)
    rows[-1]["cumulative_p_change"] = 1.0 - (1.0 - hazard) ** 13
    pd.DataFrame(rows).to_csv(survival_path, index=False)
    structural = pd.read_csv(structural_path)
    structural.loc[0, "p_change"] = hazard + 0.01
    structural.to_csv(structural_path, index=False)
    with pytest.raises(audit_outputs.AuditFailure, match="differs from binary source"):
        audit_outputs.audit_joint_survival_forecasts(tmp_path)


def test_frozen_v3_baseline_hashes_materialized_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = Path("baseline")
    baseline = tmp_path / relative
    baseline.mkdir()
    payload = baseline / "regime-results.json"
    payload.write_text('{"result":"v3"}\n', encoding="utf-8")
    member = baseline / "oos-predictions.csv"
    member.write_text("origin,actual\n2020-01-03,risk_on\n", encoding="utf-8")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    inventory = baseline / "SHA256SUMS"
    inventory.write_text(
        f"{digest(member)}  {member.name}\n{digest(payload)}  {payload.name}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(audit_outputs, "V3_BASELINE_RELATIVE_DIRECTORY", relative)
    monkeypatch.setattr(
        audit_outputs,
        "V3_BASELINE",
        {
            **audit_outputs.V3_BASELINE,
            "payload_sha256": digest(payload),
            "artifacts_inventory_sha256": digest(inventory),
        },
    )

    result = audit_outputs.audit_frozen_v3_baseline(tmp_path)
    assert result["files"] == 2

    member.write_text("origin,actual\n2020-01-03,risk_off\n", encoding="utf-8")
    with pytest.raises(audit_outputs.AuditFailure, match="member SHA-256"):
        audit_outputs.audit_frozen_v3_baseline(tmp_path)


def test_structural_preregistration_hashes_the_named_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = Path("config/structural_v4.json")
    path = tmp_path / relative
    path.parent.mkdir()
    path.write_text('{"frozen":true}\n', encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(audit_outputs, "V4_PREREGISTRATION_RELATIVE_PATH", relative)
    monkeypatch.setattr(audit_outputs, "V4_PREREGISTRATION_SHA256", digest)

    result = audit_outputs.audit_structural_preregistration(
        {"path": relative.as_posix(), "sha256": digest}, tmp_path
    )
    assert result["sha256"] == digest

    path.write_text('{"frozen":false}\n', encoding="utf-8")
    with pytest.raises(audit_outputs.AuditFailure, match="materialized SHA-256"):
        audit_outputs.audit_structural_preregistration(
            {"path": relative.as_posix(), "sha256": digest}, tmp_path
        )


def _write_state_evidence(tmp_path: Path, rows: int = 523) -> tuple[dict, pd.DataFrame]:
    dates = pd.date_range(
        "2016-01-01T21:00:00Z", periods=rows, freq="W-FRI"
    )
    risk = np.resize(np.asarray([np.nan, 0.8, 0.1, -0.8, -0.1], dtype=float), rows)
    lower = -0.5
    upper = 0.5
    margin = 0.15
    temperature = 0.75
    state = "transition"
    states: list[str] = []
    previous: list[object] = []
    for value in risk:
        previous.append(pd.NA if not states else states[-1])
        if np.isfinite(value):
            if state == "transition":
                if value <= lower:
                    state = "risk_off"
                elif value >= upper:
                    state = "risk_on"
            elif state == "risk_on":
                if value <= lower - margin:
                    state = "risk_off"
                elif value < upper - margin:
                    state = "transition"
            else:
                if value >= upper + margin:
                    state = "risk_on"
                elif value > lower + margin:
                    state = "transition"
        states.append(state)
    width = upper - lower
    anchors = np.asarray([upper + width / 2, (lower + upper) / 2, lower - width / 2])
    distance = (risk[:, None] - anchors[None, :]) / width
    logits = -(distance ** 2) / temperature
    logits[~np.isfinite(risk)] = [-20.0, 0.0, -20.0]
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    probability /= probability.sum(axis=1, keepdims=True)
    frame = pd.DataFrame(
        {
            "date": dates,
            "state": states,
            "p_risk_on": probability[:, 0],
            "p_transition": probability[:, 1],
            "p_risk_off": probability[:, 2],
            "risk_score": risk,
            "lower_threshold": lower,
            "upper_threshold": upper,
            "hysteresis_margin": margin,
            "previous_state": previous,
            "probability_temperature": temperature,
        },
        columns=audit_outputs.V4_STATE_EVIDENCE_COLUMNS,
    )
    path = tmp_path / "state-label-history.csv"
    frame.to_csv(path, index=False, lineterminator="\n")
    metadata = {
        "path": path.name,
        "row_count": len(frame),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "label_fit_weeks": 520,
        "label_fit_end": dates[519].isoformat(),
        "initial_state": "transition",
    }
    return metadata, frame


def test_v4_state_evidence_recomputes_hysteresis_probabilities_and_departures(
    tmp_path: Path,
) -> None:
    metadata, frame = _write_state_evidence(tmp_path)
    origin_position = 520
    horizon = 2
    current = str(frame.iloc[origin_position]["state"])
    actual_change = bool(
        frame.iloc[origin_position + 1 : origin_position + horizon + 1]["state"]
        .astype(str)
        .ne(current)
        .any()
    )
    transition = pd.DataFrame(
        [
            {
                "origin_date": frame.iloc[origin_position]["date"],
                "target_end": frame.iloc[origin_position + horizon]["date"],
                "horizon": horizon,
                "current_state": current,
                "actual_change": actual_change,
            }
        ]
    )
    main = pd.DataFrame(
        [
            {
                "origin_date": frame.iloc[origin_position]["date"],
                "target_date": frame.iloc[origin_position + 1]["date"],
                "current_state": current,
                "actual": str(frame.iloc[origin_position + 1]["state"]),
            }
        ]
    )
    prospective = pd.DataFrame(
        [
            {
                "origin_date": frame.iloc[-1]["date"],
                "current_state": str(frame.iloc[-1]["state"]),
            }
        ]
    )
    payload = {
        "model": {
            "evidence_artifacts": {
                "state_label_history": metadata,
                "weekly_state_forecasts": {},
            }
        }
    }

    result = audit_outputs.audit_v4_state_evidence(
        payload,
        tmp_path,
        transition_predictions=transition,
        main_predictions=main,
        prospective_transition_frames=[prospective],
    )
    assert result["rows"] == len(frame)

    tampered = transition.copy()
    tampered["actual_change"] = ~tampered["actual_change"]
    with pytest.raises(audit_outputs.AuditFailure, match="any-departure"):
        audit_outputs.audit_v4_state_evidence(
            payload, tmp_path, transition_predictions=tampered
        )

    tampered_main = main.copy()
    tampered_main["actual"] = "risk_off" if main.iloc[0]["actual"] != "risk_off" else "risk_on"
    with pytest.raises(audit_outputs.AuditFailure, match=r"t\+1 actual"):
        audit_outputs.audit_v4_state_evidence(
            payload,
            tmp_path,
            transition_predictions=transition,
            main_predictions=tampered_main,
        )

    tampered_prospective = prospective.copy()
    tampered_prospective["current_state"] = (
        "risk_off"
        if prospective.iloc[0]["current_state"] != "risk_off"
        else "risk_on"
    )
    with pytest.raises(audit_outputs.AuditFailure, match="prospective transition"):
        audit_outputs.audit_v4_state_evidence(
            payload,
            tmp_path,
            transition_predictions=transition,
            prospective_transition_frames=[tampered_prospective],
        )


def test_v4_weekly_evidence_has_state_history_and_payload_parity(
    tmp_path: Path,
) -> None:
    state_metadata, states = _write_state_evidence(tmp_path)
    selected = states.iloc[-2:].copy()
    rows = []
    weekly = []
    for index, (_, state_row) in enumerate(selected.iterrows()):
        origin = pd.Timestamp(state_row["date"])
        target = origin + timedelta(days=7)
        next_probability = np.asarray([0.60, 0.25, 0.15], dtype=float)
        current_probability = {
            state: round(float(state_row[f"p_{state}"]), 8)
            for state in audit_outputs.STATE_ORDER
        }
        next_probability_object = {
            state: float(next_probability[position])
            for position, state in enumerate(audit_outputs.STATE_ORDER)
        }
        rows.append(
            {
                "origin_date": origin.isoformat(),
                "current_state": str(state_row["state"]),
                "current_p_risk_on": current_probability["risk_on"],
                "current_p_transition": current_probability["transition"],
                "current_p_risk_off": current_probability["risk_off"],
                "target_date": target.date().isoformat(),
                "model": "markov",
                "next_p_risk_on": next_probability[0],
                "next_p_transition": next_probability[1],
                "next_p_risk_off": next_probability[2],
                "fallback": False,
                "fallback_reason": "",
            }
        )
        weekly.append(
            {
                "date": origin.date().isoformat(),
                "data_as_of": origin.isoformat(),
                "current": {
                    "state": str(state_row["state"]),
                    "probabilities": current_probability,
                },
                "next_week": {
                    "date": target.date().isoformat(),
                    "state": "risk_on",
                    "model": "markov",
                    "probabilities": next_probability_object,
                    "fallback": False,
                    "fallback_reason": "",
                },
            }
        )
    path = tmp_path / "weekly-state-forecasts.csv"
    pd.DataFrame(
        rows, columns=audit_outputs.V4_WEEKLY_FORECAST_EVIDENCE_COLUMNS
    ).to_csv(path, index=False, lineterminator="\n")
    weekly_metadata = {
        "path": path.name,
        "row_count": len(rows),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    payload = {
        "model": {
            "evidence_artifacts": {
                "state_label_history": state_metadata,
                "weekly_state_forecasts": weekly_metadata,
            }
        },
        "weekly": weekly,
    }

    result = audit_outputs.audit_v4_weekly_forecast_evidence(payload, tmp_path)
    assert result["rows"] == 2

    payload["weekly"][0]["next_week"]["probabilities"]["risk_on"] -= 0.01
    with pytest.raises(audit_outputs.AuditFailure, match="next risk_on"):
        audit_outputs.audit_v4_weekly_forecast_evidence(payload, tmp_path)


def test_latest_structural_forecast_requires_weight_columns(tmp_path: Path) -> None:
    origin = pd.Timestamp("2026-08-07T20:00:00Z")
    target = origin + timedelta(days=7)
    source_probability = np.asarray([0.60, 0.25, 0.15])
    hazard = 0.20
    joint_probability = audit_outputs.compose_joint_probability(
        source_probability, hazard, "risk_on"
    )
    probabilities = {
        "markov": np.asarray([0.55, 0.30, 0.15]),
        "xgboost": source_probability,
        "xgb_hazard_destination": joint_probability,
        "causal_dynamic_ensemble": np.asarray([0.55, 0.30, 0.15]),
    }
    rows = []
    for model, probability in probabilities.items():
        rows.append(
            {
                "origin_date": origin,
                "target_date": target,
                "model": model,
                "current_state": "risk_on",
                "p_risk_on": probability[0],
                "p_transition": probability[1],
                "p_risk_off": probability[2],
                "predicted": audit_outputs.STATE_ORDER[int(np.argmax(probability))],
                "fallback": False,
                "fallback_reason": "",
                "p_change": hazard,
            }
        )
    rows.append(
        {
            "origin_date": origin,
            "target_date": target,
            "model": "binary_xgboost",
            "current_state": "risk_on",
            "p_risk_on": np.nan,
            "p_transition": np.nan,
            "p_risk_off": np.nan,
            "predicted": "",
            "fallback": False,
            "fallback_reason": "",
            "p_change": hazard,
        }
    )
    pd.DataFrame(rows).to_csv(tmp_path / "structural-forecasts.csv", index=False)

    with pytest.raises(audit_outputs.AuditFailure, match="weight columns"):
        audit_outputs.audit_structural_forecasts(
            tmp_path, historical_predictions=pd.DataFrame()
        )


@pytest.mark.skipif(
    os.environ.get("REGIME_RUN_V4_E2E") != "1",
    reason="set REGIME_RUN_V4_E2E=1 for the multi-minute offline v4 bundle run",
)
def test_offline_synthetic_v4_bundle_passes_the_full_auditor(tmp_path: Path) -> None:
    """Provider-free regression for the real pipeline → files → audit boundary."""

    from regime_lab.cli import _write_supporting_results
    from regime_lab.config import default_config_path, load_config
    from regime_lab.demo import generate_demo_payload
    from regime_lab.payload import write_dashboard_payload

    payload, benchmark = generate_demo_payload(
        load_config(default_config_path()), profile_name="quick"
    )
    artifacts = tmp_path / "artifacts"
    payload_path = tmp_path / "regime-results.json"
    _write_supporting_results(
        benchmark,
        artifacts,
        generation_id=str(payload["meta"]["generation_id"]),
    )
    write_dashboard_payload(payload, payload_path)

    result = audit_outputs.audit(payload_path, artifacts, "demo")

    assert result["ok"] is True
    assert result["models"] == 16
    assert result["transition"]["models"] == 6
    assert result["structural"]["ablation"]["variants"] == 7
    assert result["structural"]["state_evidence"]["rows"] >= 520
