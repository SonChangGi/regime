from __future__ import annotations

from copy import deepcopy
import hashlib
from types import SimpleNamespace

import pandas as pd
import pytest

from regime_lab import cli
from regime_lab.analysis import BenchmarkProfile
from regime_lab.v5_artifacts import (
    FX_ABLATION_OOS_COLUMNS,
    FX_RESEARCH_ARTIFACT_KEYS,
    OPTIONAL_RESEARCH_ARTIFACT_KEYS,
    REQUIRED_RESEARCH_ARTIFACT_KEYS,
    V5_CORE_ARTIFACT_PATHS,
    V5_RESEARCH_ARTIFACTS,
    build_v5_core_artifact_manifest,
    build_v5_research_artifact_manifest,
    canonical_v5_core_artifact_csv_bytes,
    canonical_v5_artifact_csv_bytes,
    verify_staged_v5_core_artifacts,
    verify_staged_v5_research_artifacts,
)


def _empty_frame(key: str) -> pd.DataFrame:
    spec = V5_RESEARCH_ARTIFACTS[key]
    if spec.materialize_observation_week:
        return pd.DataFrame(
            columns=spec.columns[1:],
            index=pd.DatetimeIndex([], name="week_end"),
        )
    return pd.DataFrame(columns=spec.columns)


def _frames(*, include_fx: bool = False) -> dict[str, pd.DataFrame]:
    keys = set(REQUIRED_RESEARCH_ARTIFACT_KEYS)
    if include_fx:
        keys.update(FX_RESEARCH_ARTIFACT_KEYS)
    return {key: _empty_frame(key) for key in keys}


def test_canonical_directional_csv_is_stable_under_row_permutation() -> None:
    columns = V5_RESEARCH_ARTIFACTS["directional_forecasts"].columns
    rows = [
        {
            "horizon_weeks": horizon,
            "origin_date": pd.Timestamp(origin),
            "target_end": pd.Timestamp(target),
            "model": "markov_first_passage",
            "current_state": "risk_on",
            "p_no_departure": 0.7,
            "p_risk_on": 0.0,
            "p_transition": 0.2,
            "p_risk_off": 0.1,
            "fallback": False,
            "fallback_reason": "",
        }
        for horizon, origin, target in (
            (4, "2026-08-14", "2026-09-11"),
            (1, "2026-08-07", "2026-08-14"),
        )
    ]
    frame = pd.DataFrame(rows, columns=columns)

    encoded = canonical_v5_artifact_csv_bytes("directional_forecasts", frame)
    reversed_encoded = canonical_v5_artifact_csv_bytes(
        "directional_forecasts", frame.iloc[::-1]
    )

    assert encoded == reversed_encoded
    assert b"1,2026-08-07,2026-08-14" in encoded
    assert b"\r\n" not in encoded


def test_fx_canonical_csv_materializes_observation_week() -> None:
    spec = V5_RESEARCH_ARTIFACTS["fx_features"]
    values = {column: [None, None] for column in spec.columns[1:]}
    values["fx__brd__usd_log_level"] = [1.25, 1.5]
    frame = pd.DataFrame(
        values,
        index=pd.DatetimeIndex(["2026-08-14", "2026-08-07"], name="week_end"),
    )

    encoded = canonical_v5_artifact_csv_bytes("fx_features", frame)
    lines = encoded.decode("utf-8").splitlines()

    assert lines[0].startswith("observation_week,fx__brd__usd_log_level,")
    assert lines[1].startswith("2026-08-07,1.5,")
    assert lines[2].startswith("2026-08-14,1.25,")


def test_fx_ablation_oos_schema_and_canonical_sort_are_frozen() -> None:
    assert FX_ABLATION_OOS_COLUMNS == (
        "origin_date",
        "target_date",
        "variant",
        "evaluation_split",
        "current_state",
        "actual",
        "p_risk_on",
        "p_transition",
        "p_risk_off",
        "train_size",
        "gap",
        "last_train_target",
        "purged_origin_count",
        "fallback",
        "fallback_reason",
        "common_origins_sha256",
    )
    row = {
        "origin_date": pd.Timestamp("2026-08-14"),
        "target_date": pd.Timestamp("2026-08-21"),
        "variant": "v4_control",
        "evaluation_split": "prospective_shadow",
        "current_state": "risk_on",
        "actual": "transition",
        "p_risk_on": 0.2,
        "p_transition": 0.7,
        "p_risk_off": 0.1,
        "train_size": 104,
        "gap": 1,
        "last_train_target": pd.Timestamp("2026-08-07"),
        "purged_origin_count": 1,
        "fallback": False,
        "fallback_reason": "",
        "common_origins_sha256": "0" * 64,
    }
    later = {**row, "variant": "v4_plus_all_fx"}
    frame = pd.DataFrame([later, row], columns=FX_ABLATION_OOS_COLUMNS)

    lines = canonical_v5_artifact_csv_bytes(
        "fx_ablation_oos", frame
    ).decode("utf-8").splitlines()

    assert lines[0].split(",") == list(FX_ABLATION_OOS_COLUMNS)
    assert lines[1].split(",")[2] == "v4_control"
    assert lines[2].split(",")[2] == "v4_plus_all_fx"


def test_manifest_requires_complete_core_and_fx_set() -> None:
    frames = _frames()
    frames.pop("directional_forecasts")
    with pytest.raises(ValueError, match="incomplete"):
        build_v5_research_artifact_manifest(frames)

    frames = _frames()
    frames["fx_features"] = _empty_frame("fx_features")
    with pytest.raises(ValueError, match="complete set"):
        build_v5_research_artifact_manifest(frames)

    frames = _frames()
    frames["model_conditioned_asset_statistics"] = _empty_frame(
        "model_conditioned_asset_statistics"
    )
    with pytest.raises(ValueError, match="model-conditioned.*complete set"):
        build_v5_research_artifact_manifest(frames)


def test_manifest_accepts_complete_optional_model_conditioned_pair() -> None:
    frames = _frames()
    frames.update({key: _empty_frame(key) for key in OPTIONAL_RESEARCH_ARTIFACT_KEYS})

    manifest = build_v5_research_artifact_manifest(frames)

    assert OPTIONAL_RESEARCH_ARTIFACT_KEYS.issubset(manifest)


def test_manifest_hashes_are_verified_against_staged_bytes(tmp_path) -> None:
    frames = _frames(include_fx=True)
    manifest = build_v5_research_artifact_manifest(frames)
    for key, metadata in manifest.items():
        payload = canonical_v5_artifact_csv_bytes(key, frames[key])
        (tmp_path / str(metadata["path"])).write_bytes(payload)
        assert metadata["sha256"] == hashlib.sha256(payload).hexdigest()

    verify_staged_v5_research_artifacts(manifest, tmp_path)

    broken = deepcopy(manifest)
    broken["conditional_asset_statistics"]["row_count"] = 1
    with pytest.raises(RuntimeError, match="row count mismatch"):
        verify_staged_v5_research_artifacts(broken, tmp_path)

    path = tmp_path / str(manifest["fx_features"]["path"])
    path.write_bytes(path.read_bytes() + b"tampered\n")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_staged_v5_research_artifacts(manifest, tmp_path)


def test_staged_verifier_rejects_symlinked_sidecar(tmp_path) -> None:
    frames = _frames()
    manifest = build_v5_research_artifact_manifest(frames)
    for key, metadata in manifest.items():
        (tmp_path / str(metadata["path"])).write_bytes(
            canonical_v5_artifact_csv_bytes(key, frames[key])
        )
    target = tmp_path / "external.csv"
    target.write_bytes(
        canonical_v5_artifact_csv_bytes(
            "directional_forecasts", frames["directional_forecasts"]
        )
    )
    sidecar = tmp_path / "directional-forecasts.csv"
    sidecar.unlink()
    sidecar.symlink_to(target)

    with pytest.raises(RuntimeError, match="non-regular"):
        verify_staged_v5_research_artifacts(manifest, tmp_path)


def test_core_manifest_binds_all_six_v5_sidecars(tmp_path) -> None:
    frames = {
        key: pd.DataFrame({"identity": [index], "value": [index / 7.0]})
        for index, (key, _) in enumerate(V5_CORE_ARTIFACT_PATHS)
    }
    manifest = build_v5_core_artifact_manifest(frames)
    for key, path in V5_CORE_ARTIFACT_PATHS:
        payload = canonical_v5_core_artifact_csv_bytes(frames[key])
        (tmp_path / path).write_bytes(payload)
        assert manifest[key] == {
            "path": path,
            "row_count": 1,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    verify_staged_v5_core_artifacts(manifest, tmp_path)

    broken = deepcopy(manifest)
    broken["stacking_weights"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_staged_v5_core_artifacts(broken, tmp_path)


def test_cli_writes_every_v5_sidecar_with_canonical_bytes(tmp_path) -> None:
    frames = _frames(include_fx=True)
    for key in FX_RESEARCH_ARTIFACT_KEYS:
        spec = V5_RESEARCH_ARTIFACTS[key]
        if spec.materialize_observation_week:
            frames[key] = pd.DataFrame(
                {column: [None] for column in spec.columns[1:]},
                index=pd.DatetimeIndex(["2026-08-07"], name="week_end"),
            )
        else:
            frames[key] = pd.DataFrame(
                [
                    {
                        "origin_date": "2026-08-07",
                        "target_date": "2026-08-14",
                        "variant": "v4_control",
                        "evaluation_split": "prospective_shadow",
                        "current_state": "risk_on",
                        "actual": "transition",
                        "p_risk_on": 0.2,
                        "p_transition": 0.7,
                        "p_risk_off": 0.1,
                        "train_size": 104,
                        "gap": 1,
                        "last_train_target": "2026-07-31",
                        "purged_origin_count": 1,
                        "fallback": False,
                        "fallback_reason": "",
                        "common_origins_sha256": "0" * 64,
                    }
                ],
                columns=spec.columns,
            )
    directional = SimpleNamespace(
        predictions=frames["directional_oos_predictions"],
        leaderboard=frames["directional_model_leaderboard"],
        split_audit=frames["directional_walk_forward_splits"],
        selection_diagnostics=frames["directional_selection_diagnostics"],
        latest_forecasts=frames["directional_forecasts"],
    )
    benchmark = SimpleNamespace(
        leaderboard=pd.DataFrame({"model": ["markov"]}),
        predictions=pd.DataFrame({"model": ["markov"]}),
        split_audit=pd.DataFrame({"origin_date": []}),
        profile=BenchmarkProfile.quick(),
        directional_benchmark=directional,
        conditional_asset_outcomes=frames["conditional_asset_outcomes"],
        conditional_asset_statistics=frames["conditional_asset_statistics"],
        fx_features=frames["fx_features"],
        fx_coverage=frames["fx_coverage"],
        fx_ablation_oos=frames["fx_ablation_oos"],
    )

    cli._write_supporting_results(benchmark, tmp_path / "artifacts")

    for key, frame in frames.items():
        spec = V5_RESEARCH_ARTIFACTS[key]
        assert (tmp_path / "artifacts" / spec.path).read_bytes() == (
            canonical_v5_artifact_csv_bytes(key, frame)
        )
    assert (tmp_path / "artifacts" / "fx-features.csv").read_text(
        encoding="utf-8"
    ).startswith("observation_week,")
