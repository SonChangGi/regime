from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from regime_lab.analysis.labels import STATE_ORDER
from regime_lab.analysis.mechanism_ablation import MechanismAblationResult
from regime_lab.analysis.validation import expected_calibration_error
from regime_lab.integrity import canonical_json_sha256_v1
from regime_lab.mechanism_ablation_run import (
    ARTIFACT_FRAMES,
    MECHANISM_EVIDENCE_STATUS,
    build_mechanism_metric_tables,
    load_feature_role_manifest,
    validate_feature_role_manifest_for_columns,
    write_mechanism_ablation_generation,
)


def test_repository_feature_role_manifest_is_exact_once_and_pinned() -> None:
    manifest = load_feature_role_manifest()
    counts = {row["id"]: len(row["features"]) for row in manifest.rows}
    assert manifest.feature_count == 510
    assert counts == {
        "label_mechanics": 12,
        "market_ex_label_components": 137,
        "macro_rates_credit": 355,
        "full_only_control": 6,
    }
    assigned = [feature for row in manifest.rows for feature in row["features"]]
    validate_feature_role_manifest_for_columns(manifest, list(reversed(assigned)))
    assert len(assigned) == len(set(assigned)) == 510
    by_role = {row["id"]: set(row["features"]) for row in manifest.rows}
    assert "regime_boundary__risk_score" in by_role["label_mechanics"]
    assert "state_duration_weeks" in by_role["label_mechanics"]
    assert "market_internal__positive_return_share_1w" in by_role[
        "market_ex_label_components"
    ]
    assert "dgs10__level" in by_role["macro_rates_credit"]
    assert "release_innovation__unrate__standardized" in by_role[
        "macro_rates_credit"
    ]
    assert by_role["full_only_control"] == {
        "spy_close__risk_adjusted_trend_13w",
        "spy_close__risk_adjusted_trend_26w",
        "spy_close__realized_vol_4w",
        "spy_close__realized_vol_13w",
        "spy_close__drawdown_13w",
        "spy_close__drawdown_52w",
    }


def test_feature_role_manifest_rejects_matrix_drift() -> None:
    manifest = load_feature_role_manifest()
    assigned = [feature for row in manifest.rows for feature in row["features"]]
    with pytest.raises(ValueError, match="feature count differs"):
        validate_feature_role_manifest_for_columns(manifest, assigned[:-1])
    changed = [*assigned[:-1], "unexpected_feature"]
    with pytest.raises(ValueError, match="feature names differ"):
        validate_feature_role_manifest_for_columns(manifest, changed)


def _metric_result() -> MechanismAblationResult:
    origins = pd.date_range("2023-01-06", periods=6, freq="W-FRI")
    actual = ["risk_on", "transition", "risk_off"] * 2
    current = ["risk_on", "risk_on", "transition"] * 2
    rows = []
    for position, origin in enumerate(origins):
        probabilities = {state: 0.1 for state in STATE_ORDER}
        probabilities[actual[position]] = 0.8
        rows.append(
            {
                "track": "full",
                "model": "xgboost",
                "evaluation_split": "selection" if position < 3 else "holdout",
                "origin_date": origin,
                    "target_date": origin + pd.offsets.Day(7),
                "current_state": current[position],
                "actual": actual[position],
                "predicted": actual[position],
                **{f"p_{state}": probabilities[state] for state in STATE_ORDER},
                "fallback": False,
            }
        )
    predictions = pd.DataFrame(rows)
    probability = predictions[[f"p_{state}" for state in STATE_ORDER]].to_numpy()
    calibration = expected_calibration_error(
        predictions["actual"].iloc[:3], probability[:3]
    )
    leaderboard = pd.DataFrame(
        [
            {
                "track": "full",
                "model": "xgboost",
                "evaluation_split": split,
                "log_loss": float(-np.log(0.8)),
                "brier": 0.06,
                "calibration_error": calibration,
                "n_predictions": 3,
            }
            for split in ("selection", "holdout")
        ]
    )
    return MechanismAblationResult(
        predictions=predictions,
        leaderboard=leaderboard,
        track_manifest=pd.DataFrame(),
        role_manifest=pd.DataFrame(),
        common_origins=pd.DataFrame(),
        specification_sha256="b" * 64,
    )


def test_independent_metric_tables_recompute_scores_recall_and_transitions() -> None:
    metrics, recalls, transitions, events = build_mechanism_metric_tables(
        _metric_result()
    )
    assert len(metrics) == 2
    assert metrics["leaderboard_crosscheck"].eq("matched").all()
    assert np.allclose(metrics["log_loss"], -np.log(0.8))
    assert np.allclose(metrics["brier"], 0.06)
    assert len(recalls) == 2 * len(STATE_ORDER)
    assert recalls["recall"].eq(1.0).all()
    assert transitions["transition_event_count"].eq(2).all()
    assert transitions["mean_detection_delay_forecast_weeks"].eq(0.0).all()
    assert transitions["false_alarm_count"].eq(0).all()
    assert len(events) == 4


def test_metric_crosscheck_fails_on_recorded_primary_score_drift() -> None:
    result = _metric_result()
    result.leaderboard.loc[0, "log_loss"] += 0.01
    with pytest.raises(ValueError, match="independent log_loss differs"):
        build_mechanism_metric_tables(result)


def test_atomic_writer_emits_only_declared_derived_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fingerprint = "c" * 64
    monkeypatch.setattr(
        "regime_lab.mechanism_ablation_run.mechanism_ablation_source_fingerprint",
        lambda *args, **kwargs: fingerprint,
    )
    frames = {
        key: pd.DataFrame({"derived_value": [position]})
        for position, (key, _) in enumerate(ARTIFACT_FRAMES)
    }
    report = {
        "schema_version": "regime-mechanism-ablation-run/1",
        "evidence_status": MECHANISM_EVIDENCE_STATUS,
        "derived_only_artifacts": True,
        "automatic_promotion_eligible": False,
        "selection_effect": "none",
        "input": {"analysis_source_fingerprint_sha256": fingerprint},
    }
    role_path = tmp_path / "role.json"
    spec_path = tmp_path / "spec.json"
    role_path.write_text("{}", encoding="utf-8")
    spec_path.write_text("{}", encoding="utf-8")
    generation = write_mechanism_ablation_generation(
        tmp_path / "output",
        report,
        frames,
        expected_source_fingerprint_sha256=fingerprint,
        source_config={},
        role_manifest_path=role_path,
        specification_path=spec_path,
    )

    output_files = {path.name for path in generation.iterdir()}
    assert output_files == {
        *(filename for _, filename in ARTIFACT_FRAMES),
        "mechanism-ablation-report.json",
    }
    document = json.loads(
        (generation / "mechanism-ablation-report.json").read_text(encoding="utf-8")
    )
    digest = document.pop("sha256")
    assert digest == canonical_json_sha256_v1(document)
    assert document["automatic_promotion_eligible"] is False
    for key, filename in ARTIFACT_FRAMES:
        raw = (generation / filename).read_bytes()
        record = document["artifact_manifest"][key]
        assert record["sha256"] == hashlib.sha256(raw).hexdigest()
        assert record["row_count"] == 1
    latest = json.loads(
        (tmp_path / "output" / "latest.json").read_text(encoding="utf-8")
    )
    assert latest["generation"] == f"runs/{generation.name}"
