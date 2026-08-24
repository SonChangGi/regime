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
