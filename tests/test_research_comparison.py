from __future__ import annotations

import hashlib
import json
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from regime_lab.feature_quality import feature_quality_document
from regime_lab.research_comparison import (
    ARTIFACT_FRAMES,
    RESEARCH_MODEL_NAMES,
    RESEARCH_PAIRED_CHALLENGER_NAMES,
    V6_RESEARCH_FEATURE_SET_VERSION,
    fold_feature_availability,
    paired_control_comparison,
    research_source_fingerprint,
    run_research_comparison,
    write_research_generation,
)
import regime_lab.research_comparison as research_comparison_module


def _prediction_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    models = ("markov", *RESEARCH_PAIRED_CHALLENGER_NAMES)
    for split, start in (("selection", "2022-01-07"), ("holdout", "2023-01-06")):
        origins = pd.date_range(start, periods=20, freq="W-FRI", tz="UTC")
        for position, origin in enumerate(origins):
            actual = ("risk_on", "transition", "risk_off")[position % 3]
            for model_position, model in enumerate(models):
                actual_probability = 0.55 + 0.04 * model_position
                remainder = (1.0 - actual_probability) / 2.0
                probabilities = {state: remainder for state in ("risk_on", "transition", "risk_off")}
                probabilities[actual] = actual_probability
                rows.append(
                    {
                        "origin_date": origin,
                        "target_date": origin + timedelta(days=7),
                        "evaluation_split": split,
                        "model": model,
                        "actual": actual,
                        **{f"p_{state}": value for state, value in probabilities.items()},
                    }
                )
    return pd.DataFrame(rows)


def test_paired_control_comparison_uses_common_origins_and_fixed_holdout() -> None:
    rows = paired_control_comparison(_prediction_rows())

    assert len(rows) == 2 * len(RESEARCH_PAIRED_CHALLENGER_NAMES)
    assert {row["candidate"] for row in rows} == set(
        RESEARCH_PAIRED_CHALLENGER_NAMES
    )
    assert {"ridge_logistic", "xgboost"}.issubset(
        {row["candidate"] for row in rows}
    )
    assert {row["split"] for row in rows} == {
        "selection",
        "retrospective_diagnostic",
    }
    assert {row["n_common_origins"] for row in rows} == {20}
    assert all(
        row["mean_log_loss_delta_candidate_minus_control"] < 0
        for row in rows
    )
    assert all(row["effective_block_weeks"] == 10 for row in rows)


def test_research_generation_is_immutable_and_hash_indexed(tmp_path: Path) -> None:
    frames = {
        key: pd.DataFrame({"value": [index]})
        for index, (key, _filename) in enumerate(ARTIFACT_FRAMES)
    }
    index = pd.date_range("2024-01-05", periods=60, freq="W-FRI", tz="UTC")
    quality = feature_quality_document(
        pd.DataFrame({"x": np.arange(60, dtype=float)}, index=index)
    )
    report = {
        "schema_version": "regime-private-model-comparison/1",
        "data_as_of": "2026-08-21T20:00:00+00:00",
    }

    generation = write_research_generation(tmp_path / "research", report, frames, quality)
    pointer = json.loads((tmp_path / "research/latest.json").read_text(encoding="utf-8"))
    report_path = generation / "research-model-comparison.json"

    assert generation.is_dir()
    assert pointer["generation"] == f"runs/{generation.name}"
    assert pointer["report_sha256"] == hashlib.sha256(report_path.read_bytes()).hexdigest()
    assert (generation / "feature-quality.json").is_file()


def test_fold_feature_availability_is_causal_and_hashes_names() -> None:
    index = pd.date_range("2020-01-03", periods=5, freq="W-FRI", tz="UTC")
    features = pd.DataFrame(
        {
            "always": np.arange(5, dtype=float),
            "late": [np.nan, np.nan, np.nan, 1.0, 2.0],
        },
        index=index,
    )
    split = pd.DataFrame(
        {
            "origin_date": [index[3], index[4]],
            "last_train_origin": [index[1], index[3]],
            "train_size": [2, 4],
            "evaluation_split": ["selection", "holdout"],
        }
    )

    result = fold_feature_availability(features, split)

    assert result["unavailable_feature_count"].tolist() == [1, 0]
    assert result["available_feature_count"].tolist() == [1, 2]
    assert result["unavailable_features_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()


def test_research_source_fingerprint_changes_with_code_bytes(tmp_path: Path) -> None:
    for relative in (
        "src/regime_lab/example.py",
        "scripts/evaluate_research_models.py",
        "config/series.json",
        "config/provider_rights.json",
        "config/structural_v6_research.json",
        "pyproject.toml",
        "requirements-ci.lock",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    first = research_source_fingerprint(tmp_path)
    (tmp_path / "src/regime_lab/example.py").write_text("changed", encoding="utf-8")

    assert research_source_fingerprint(tmp_path) != first


def test_research_source_fingerprint_binds_effective_config_and_policy(
    tmp_path: Path,
) -> None:
    for relative in (
        "src/regime_lab/example.py",
        "scripts/evaluate_research_models.py",
        "config/structural_v6_research.json",
        "pyproject.toml",
        "requirements-ci.lock",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    policy = tmp_path / "config/custom-rights.json"
    policy.write_text('{"schema_version":1}', encoding="utf-8")
    config = {
        "provider_rights_policy": "config/custom-rights.json",
        "model": {"final_holdout_start": "2023-01-01"},
    }
    first = research_source_fingerprint(tmp_path, config=config)
    policy.write_text('{"schema_version":1,"changed":true}', encoding="utf-8")

    assert research_source_fingerprint(tmp_path, config=config) != first


def test_research_model_suite_matches_preregistration() -> None:
    document = json.loads(
        Path("config/structural_v6_research.json").read_text(encoding="utf-8")
    )

    assert document["comparison_models"] == list(RESEARCH_MODEL_NAMES)
    assert document["feature_set_version"] == V6_RESEARCH_FEATURE_SET_VERSION


def test_v5_v6_selection_policy_preserves_frozen_v4_and_other_gates() -> None:
    document = json.loads(
        Path("config/structural_v6_research.json").read_text(encoding="utf-8")
    )
    policy = document["selection_policy"]
    evaluation = document["evaluation"]

    assert policy == {
        "id": "selection-policy-v2",
        "effective_date": "2026-08-25",
        "application": "active_v5_and_prospective_v6_runs",
        "reselection_scope": {
            "v4": "prohibited_frozen",
            "legacy_reviewed_v5_snapshot": "preserved_exact_only",
            "new_v5_and_v6_runs": "required_current_policy",
        },
        "minimum_log_loss_improvement": 0.01,
        "unchanged_gates": {
            "holm_alpha": 0.05,
            "maximum_brier_degradation": 0.01,
            "fallback_count_required": 0,
        },
        "version_contracts": {
            "v4": {
                "status": "frozen_regression_baseline",
                "minimum_log_loss_improvement": 0.05,
                "champion_name_locked": True,
            },
            "v5": {
                "status": "active_operating_contract",
                "minimum_log_loss_improvement": 0.01,
                "champion_name_locked": False,
            },
        },
    }
    assert evaluation["minimum_log_loss_improvement"] == 0.01
    assert evaluation["holm_alpha"] == 0.05
    assert evaluation["maximum_brier_degradation"] == 0.01
    assert research_comparison_module._preregistered_selection_policy(document) == {
        "id": "selection-policy-v2",
        "effective_date": "2026-08-25",
        "application": "active_v5_and_prospective_v6_runs",
        "reselection_scope": {
            "v4": "prohibited_frozen",
            "legacy_reviewed_v5_snapshot": "preserved_exact_only",
            "new_v5_and_v6_runs": "required_current_policy",
        },
        "minimum_log_loss_improvement": 0.01,
        "holm_alpha": 0.05,
        "maximum_brier_degradation": 0.01,
        "fallback_count_required": 0,
    }


@pytest.mark.parametrize(
    ("gate", "value"),
    (
        ("holm_alpha", 0.10),
        ("maximum_brier_degradation", 0.02),
        ("fallback_count_required", 1),
        ("fallback_count_required", False),
    ),
)
def test_v6_selection_policy_rejects_changed_frozen_gate(
    gate: str,
    value: object,
) -> None:
    document = json.loads(
        Path("config/structural_v6_research.json").read_text(encoding="utf-8")
    )
    document["selection_policy"]["unchanged_gates"][gate] = value

    with pytest.raises(RuntimeError, match="frozen gate"):
        research_comparison_module._preregistered_selection_policy(document)


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_v6_selection_policy_requires_exact_unchanged_gate_keys(
    mutation: str,
) -> None:
    document = json.loads(
        Path("config/structural_v6_research.json").read_text(encoding="utf-8")
    )
    gates = document["selection_policy"]["unchanged_gates"]
    if mutation == "missing":
        gates.pop("holm_alpha")
    else:
        gates["unregistered_gate"] = 0

    with pytest.raises(RuntimeError, match="unchanged selection gates"):
        research_comparison_module._preregistered_selection_policy(document)


def test_v6_selection_policy_rejects_evaluation_gate_mismatch() -> None:
    document = json.loads(
        Path("config/structural_v6_research.json").read_text(encoding="utf-8")
    )
    document["evaluation"]["holm_alpha"] = 0.04

    with pytest.raises(RuntimeError, match="Holm alpha"):
        research_comparison_module._preregistered_selection_policy(document)


@pytest.mark.parametrize(
    ("version", "field", "value", "message"),
    (
        ("v4", "minimum_log_loss_improvement", 0.01, "V4 frozen"),
        ("v4", "champion_name_locked", False, "V4 frozen"),
        ("v5", "minimum_log_loss_improvement", 0.05, "V5 active"),
        ("v5", "champion_name_locked", True, "V5 active"),
    ),
)
def test_v6_selection_policy_keeps_v4_frozen_and_v5_dynamic(
    version: str,
    field: str,
    value: object,
    message: str,
) -> None:
    document = json.loads(
        Path("config/structural_v6_research.json").read_text(encoding="utf-8")
    )
    document["selection_policy"]["version_contracts"][version][field] = value

    with pytest.raises(RuntimeError, match=message):
        research_comparison_module._preregistered_selection_policy(document)


def test_v6_cross_asset_block_is_unique_and_overlap_fails_closed() -> None:
    index = pd.date_range("2020-01-03", periods=40, freq="W-FRI", tz="UTC")
    position = np.arange(40, dtype=float)
    canonical = pd.DataFrame(
        {
            "spy_close": 100.0 * np.exp(np.cumsum(0.002 + 0.01 * np.sin(position))),
            "tlt_close": 100.0 * np.exp(np.cumsum(0.001 + 0.01 * np.cos(position))),
            "hyg_close": 100.0 * np.exp(np.cumsum(0.002 + 0.008 * np.sin(position / 2))),
            "uup_close": 100.0 * np.exp(np.cumsum(0.001 - 0.004 * np.sin(position))),
        },
        index=index,
    )
    base = pd.DataFrame({"base": np.arange(40, dtype=float)}, index=index)

    combined = research_comparison_module._with_v6_cross_asset_features(
        base,
        canonical,
    )
    cross_asset = [
        name for name in combined if name.startswith("cross_asset__")
    ]

    assert tuple(cross_asset) == research_comparison_module.V6_CROSS_ASSET_FEATURE_NAMES
    assert len(cross_asset) == len(set(cross_asset)) == 8

    overlapping = base.assign(**{cross_asset[0]: 0.0})
    with pytest.raises(RuntimeError, match="overlap"):
        research_comparison_module._with_v6_cross_asset_features(
            overlapping,
            canonical,
        )


def test_v6_preregistration_connects_new_model_and_research_data_blocks() -> None:
    document = json.loads(
        Path("config/structural_v6_research.json").read_text(encoding="utf-8")
    )
    candidates = {item["name"]: item for item in document["model_candidates"]}
    blocks = {item["id"]: item for item in document["feature_blocks"]}
    sources = document["core_sources"]

    assert candidates["recency_weighted_ridge_logistic_208w"] == {
        "name": "recency_weighted_ridge_logistic_208w",
        "role": "low_variance_drift_robust_research_candidate",
        "half_life_weeks": 208,
        "C": 0.1,
    }
    assert blocks["cross_asset_correlations"]["pairs"] == [
        "SPY_TLT",
        "SPY_HYG",
        "SPY_UUP",
        "HYG_TLT",
    ]
    assert blocks["bank_lending_standards"]["required_series"] == [
        "DRTSCILM",
        "DRTSCIS",
        "DRSDCILM",
        "DRSDCIS",
    ]
    h41 = {item["output_id"]: item for item in sources["board_h41"]["series"]}
    assert h41["WLRRAL"]["domain"] == "liquidity_absorption"
    assert "Reverse Repurchase Agreements" in h41["WLRRAL"]["source_label"]
    assert sources["board_sloos"]["status"] == "research_planned"
    assert sources["board_sloos"]["promotion_eligible"] is False


def test_research_run_forwards_v6_preregistered_selection_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BenchmarkIntercept(RuntimeError):
        pass

    index = pd.date_range("2010-01-01", periods=650, freq="W-FRI", tz="UTC")
    features = pd.DataFrame({"x": np.arange(650, dtype=float)}, index=index)
    states = pd.Series(np.resize(("risk_on", "transition", "risk_off"), 650), index=index)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        research_comparison_module,
        "research_source_fingerprint",
        lambda *_args, **_kwargs: "a" * 64,
    )
    monkeypatch.setattr(
        research_comparison_module,
        "_prepare_matrix",
        lambda *_args, **_kwargs: (features, states, {}),
    )

    def intercept(*_args, **kwargs):
        captured.update(kwargs)
        raise BenchmarkIntercept

    monkeypatch.setattr(research_comparison_module, "run_benchmark", intercept)

    with pytest.raises(BenchmarkIntercept):
        run_research_comparison(
            {"model": {"final_holdout_start": "2023-01-01"}},
            database=tmp_path / "unused.sqlite3",
            as_of=pd.Timestamp("2026-08-21T20:00:00Z").to_pydatetime(),
            profile_name="quick",
        )

    assert captured["minimum_log_loss_improvement"] == 0.01


def test_generation_rejects_source_change_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = {
        key: pd.DataFrame({"value": [index]})
        for index, (key, _filename) in enumerate(ARTIFACT_FRAMES)
    }
    index = pd.date_range("2024-01-05", periods=60, freq="W-FRI", tz="UTC")
    quality = feature_quality_document(
        pd.DataFrame({"x": np.arange(60, dtype=float)}, index=index)
    )
    report = {
        "schema_version": "regime-private-model-comparison/1",
        "data_as_of": "2026-08-21T20:00:00+00:00",
        "input": {"analysis_source_fingerprint_sha256": "a" * 64},
    }
    fingerprints = iter(("a" * 64, "b" * 64))
    monkeypatch.setattr(
        research_comparison_module,
        "research_source_fingerprint",
        lambda *_args, **_kwargs: next(fingerprints),
    )

    with pytest.raises(RuntimeError, match="source changed"):
        write_research_generation(
            tmp_path / "research",
            report,
            frames,
            quality,
            expected_source_fingerprint_sha256="a" * 64,
        )

    assert not (tmp_path / "research/latest.json").exists()
    assert not list((tmp_path / "research/runs").glob("20*"))


def test_research_run_rejects_source_change_during_matrix_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = pd.date_range("2020-01-03", periods=650, freq="W-FRI", tz="UTC")
    features = pd.DataFrame({"x": np.arange(650, dtype=float)}, index=index)
    states = pd.Series("risk_on", index=index)
    monkeypatch.setattr(
        research_comparison_module,
        "_prepare_matrix",
        lambda *_args, **_kwargs: (features, states, {}),
    )
    fingerprints = iter(("a" * 64, "b" * 64))
    monkeypatch.setattr(
        research_comparison_module,
        "research_source_fingerprint",
        lambda *_args, **_kwargs: next(fingerprints),
    )

    with pytest.raises(RuntimeError, match="source changed"):
        run_research_comparison(
            {},
            database=tmp_path / "unused.sqlite3",
            as_of=pd.Timestamp("2026-08-21T20:00:00Z").to_pydatetime(),
            profile_name="quick",
        )
