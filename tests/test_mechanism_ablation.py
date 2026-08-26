from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from regime_lab.analysis.labels import STATE_ORDER
from regime_lab.analysis.mechanism_ablation import (
    FEATURE_ROLES,
    MECHANISM_TRACKS,
    load_mechanism_ablation_spec,
    mechanism_ablation_manifest_document,
    run_mechanism_ablation,
)
from regime_lab.analysis.models import BenchmarkProfile


def _inputs():
    index = pd.date_range("2019-01-04", periods=40, freq="W-FRI")
    features = pd.DataFrame(
        {
            "label_score": np.sin(np.arange(len(index)) / 4.0),
            "market_breadth": np.cos(np.arange(len(index)) / 5.0),
            "macro_credit": np.linspace(-1.0, 1.0, len(index)),
            "other_control": np.arange(len(index), dtype=float) % 3,
        },
        index=index,
    )
    states = pd.Series(
        [STATE_ORDER[position % 3] for position in range(len(index))],
        index=index,
        dtype="object",
    )
    roles = (
        {"id": "label_mechanics", "features": ("label_score",)},
        {
            "id": "market_ex_label_components",
            "features": ("market_breadth",),
        },
        {"id": "macro_rates_credit", "features": ("macro_credit",)},
        {"id": "full_only_control", "features": ("other_control",)},
    )
    return features, states, roles


def _prediction_rows(models: tuple[str, ...]) -> pd.DataFrame:
    origins = pd.to_datetime(
        ["2020-01-03", "2020-01-10", "2020-01-17", "2020-01-24"]
    )
    targets = origins + pd.to_timedelta(7, unit="D")
    actuals = ("risk_on", "transition", "risk_off", "risk_on")
    currents = ("transition", "risk_on", "transition", "risk_off")
    rows = []
    for model in models:
        for position, (origin, target, actual, current) in enumerate(
            zip(origins, targets, actuals, currents, strict=True)
        ):
            probability = {state: 0.1 for state in STATE_ORDER}
            probability[actual] = 0.8
            rows.append(
                {
                    "origin_date": origin,
                    "target_date": target,
                    "model": model,
                    "evaluation_split": "selection" if position < 2 else "holdout",
                    "current_state": current,
                    "actual": actual,
                    "predicted": actual,
                    **{f"p_{state}": probability[state] for state in STATE_ORDER},
                    "train_size": 104 + position,
                    "gap": 1,
                    "fallback": False,
                    "fallback_reason": "",
                }
            )
    return pd.DataFrame(rows)


def test_spec_freezes_exact_five_tracks_and_separates_model_families() -> None:
    spec = load_mechanism_ablation_spec()
    assert tuple(track.track_id for track in spec.tracks) == MECHANISM_TRACKS
    assert spec.feature_roles == FEATURE_ROLES
    assert spec.tracks[0].reported_models == ("persistence", "markov")
    assert {track.reported_models for track in spec.tracks[1:]} == {("xgboost",)}
    assert spec.evaluation["cross_family_ranking"] is False


def test_five_track_run_uses_exact_roles_and_matched_origins() -> None:
    features, states, roles = _inputs()
    calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def fake_runner(frame, target, **kwargs):
        assert target is states
        calls.append((tuple(frame.columns), tuple(kwargs["models"])))
        return SimpleNamespace(predictions=_prediction_rows(tuple(kwargs["models"])))

    result = run_mechanism_ablation(
        features,
        states,
        roles,
        profile=BenchmarkProfile.quick(),
        selection_end="2020-01-15",
        minimum_selection_predictions=2,
        minimum_holdout_predictions=2,
        benchmark_runner=fake_runner,
    )

    assert [columns for columns, _ in calls] == [
        ("__state_only_constant__",),
        ("label_score",),
        ("market_breadth",),
        ("macro_credit",),
        tuple(features.columns),
    ]
    assert set(result.predictions["track"]) == set(MECHANISM_TRACKS)
    assert result.predictions.groupby(["track", "model"]).size().nunique() == 1
    assert len(result.common_origins) == 4
    state_rows = result.leaderboard.loc[result.leaderboard["track"].eq("state_only")]
    assert not state_rows["model_mechanics_comparable_to_full"].any()
    assert state_rows["paired_log_loss_delta_vs_full"].isna().all()
    feature_rows = result.leaderboard.loc[
        ~result.leaderboard["track"].eq("state_only")
    ]
    assert feature_rows["model_mechanics_comparable_to_full"].all()
    document = mechanism_ablation_manifest_document(result)
    assert document["origin_count"] == 4
    assert document["cross_family_ranking"] is False
    assert [row["track"] for row in document["tracks"]] == list(MECHANISM_TRACKS)


def test_origin_mismatch_fails_instead_of_intersecting() -> None:
    features, states, roles = _inputs()
    call = 0

    def mismatch_runner(frame, target, **kwargs):
        nonlocal call
        del frame, target
        call += 1
        predictions = _prediction_rows(tuple(kwargs["models"]))
        if call == 3:
            first_xgboost = predictions.index[predictions["model"].eq("xgboost")][0]
            predictions.loc[first_xgboost, "actual"] = "risk_off"
        return SimpleNamespace(predictions=predictions)

    with pytest.raises(ValueError, match="origin mismatch"):
        run_mechanism_ablation(
            features,
            states,
            roles,
            profile="quick",
            selection_end="2020-01-15",
            minimum_selection_predictions=2,
            minimum_holdout_predictions=2,
            benchmark_runner=mismatch_runner,
        )


def test_checkpoint_identity_is_forwarded_to_each_track(tmp_path) -> None:
    features, states, roles = _inputs()
    calls: list[tuple[object, object]] = []

    def fake_runner(frame, target, **kwargs):
        del frame, target
        calls.append(
            (
                kwargs["checkpoint_directory"],
                kwargs["source_fingerprint_sha256"],
            )
        )
        return SimpleNamespace(predictions=_prediction_rows(tuple(kwargs["models"])))

    run_mechanism_ablation(
        features,
        states,
        roles,
        profile=BenchmarkProfile.quick(),
        selection_end="2020-01-15",
        minimum_selection_predictions=2,
        minimum_holdout_predictions=2,
        checkpoint_directory=tmp_path / "checkpoints",
        source_fingerprint_sha256="a" * 64,
        benchmark_runner=fake_runner,
    )

    assert [path.name for path, _ in calls] == list(MECHANISM_TRACKS)
    assert {fingerprint for _, fingerprint in calls} == {"a" * 64}


def test_checkpoint_contract_rejects_half_configured_identity(tmp_path) -> None:
    features, states, roles = _inputs()
    with pytest.raises(ValueError, match="requires source_fingerprint"):
        run_mechanism_ablation(
            features,
            states,
            roles,
            profile="quick",
            selection_end="2020-01-15",
            checkpoint_directory=tmp_path / "checkpoints",
        )


@pytest.mark.parametrize("problem", ["missing", "overlap", "unknown"])
def test_role_manifest_is_explicit_exact_once(problem: str) -> None:
    features, states, source = _inputs()
    roles = [dict(row) for row in source]
    if problem == "missing":
        roles[-1]["features"] = ()
    elif problem == "overlap":
        roles[-1]["features"] = ("label_score", "other_control")
    else:
        roles[-1]["features"] = ("unknown_feature",)
    with pytest.raises(ValueError):
        run_mechanism_ablation(
            features,
            states,
            roles,
            profile="quick",
            selection_end="2020-01-15",
            benchmark_runner=lambda *args, **kwargs: pytest.fail(
                "invalid roles reached benchmark"
            ),
        )
