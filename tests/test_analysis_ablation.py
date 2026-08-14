from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import regime_lab.analysis.ablation as ablation
from regime_lab.analysis import BenchmarkProfile, STATE_ORDER


def _inputs() -> tuple[pd.DataFrame, pd.Series, tuple[dict[str, object], ...]]:
    index = pd.date_range("2022-11-04", periods=12, freq="W-FRI")
    columns = {
        "legacy_feature": np.linspace(0.0, 1.0, len(index)),
        "sector_feature": np.linspace(1.0, 2.0, len(index)),
        "broad_feature": np.linspace(2.0, 3.0, len(index)),
        "cross_feature": np.linspace(3.0, 4.0, len(index)),
        "treasury_feature": np.linspace(4.0, 5.0, len(index)),
        "bank_feature": np.linspace(5.0, 6.0, len(index)),
        "financial_conditions_feature": np.linspace(6.0, 7.0, len(index)),
        "release_feature": np.linspace(7.0, 8.0, len(index)),
        # Added after WeeklyDataset creates its manifest; common to all variants.
        "regime_boundary__risk_score": np.linspace(-1.0, 1.0, len(index)),
    }
    features = pd.DataFrame(columns, index=index)
    states = pd.Series(
        [STATE_ORDER[position % 3] for position in range(len(index))],
        index=index,
        dtype="object",
    )
    assignments = (
        ("sector_breadth", "sector_feature"),
        ("broad_size_style_breadth", "broad_feature"),
        ("cross_asset_breadth", "cross_feature"),
        ("treasury_curve", "treasury_feature"),
        ("bank_credit", "bank_feature"),
        ("financial_conditions", "financial_conditions_feature"),
        ("release_innovation", "release_feature"),
        ("legacy_v3", "legacy_feature"),
    )
    manifest = tuple(
        {
            "id": group,
            "description": group,
            "feature_count": 1,
            "features": (column,),
        }
        for group, column in assignments
    )
    return features, states, manifest


def _prediction_frame(
    *,
    selection_probability: float = 0.72,
    holdout_probability: float = 0.62,
) -> pd.DataFrame:
    origins = pd.to_datetime(
        ["2022-12-09", "2022-12-16", "2022-12-30", "2023-01-06"]
    )
    targets = pd.to_datetime(
        ["2022-12-16", "2022-12-23", "2023-01-06", "2023-01-13"]
    )
    actual = ["risk_on", "transition", "risk_off", "risk_on"]
    current = ["risk_on", "risk_on", "transition", "risk_off"]
    rows: list[dict[str, object]] = []
    for position, (origin, target, state, current_state) in enumerate(
        zip(origins, targets, actual, current, strict=True)
    ):
        peak = selection_probability if position < 2 else holdout_probability
        remainder = (1.0 - peak) / 2.0
        probability = {item: remainder for item in STATE_ORDER}
        probability[state] = peak
        rows.append(
            {
                "origin_date": origin,
                "target_date": target,
                "model": "xgboost",
                "evaluation_split": "selection" if position < 2 else "holdout",
                "current_state": current_state,
                "actual": state,
                "predicted": state,
                **{f"p_{item}": probability[item] for item in STATE_ORDER},
                "train_size": 100 + position,
                "gap": 1,
                "fallback": False,
                "fallback_reason": "",
            }
        )
    return pd.DataFrame(rows)


def _main_benchmark(frame: pd.DataFrame | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        predictions=_prediction_frame() if frame is None else frame,
        profile=BenchmarkProfile.quick(),
        selection_end=pd.Timestamp("2023-01-01"),
    )


def _patch_runner(monkeypatch: pytest.MonkeyPatch, reference: pd.DataFrame):
    calls: list[tuple[str, ...]] = []

    def fake_run_benchmark(features: pd.DataFrame, states: pd.Series, **kwargs):
        del states
        assert kwargs["models"] == ("majority", "xgboost")
        assert kwargs["profile"] == BenchmarkProfile.quick()
        assert kwargs["gap"] == 1
        expected_cutoff = pd.Timestamp("2023-01-01")
        if features.index.tz is not None:
            expected_cutoff = expected_cutoff.tz_localize(features.index.tz)
        assert kwargs["selection_end"] == expected_cutoff
        calls.append(tuple(features.columns))
        candidate = reference.copy()
        # Make the legacy-only fit the pre-cutoff winner.  Selection fields
        # must remain unchanged no matter what happens in the holdout rows.
        if tuple(features.columns) == (
            "legacy_feature",
            "regime_boundary__risk_score",
        ):
            candidate = _prediction_frame(
                selection_probability=0.82,
                holdout_probability=0.52,
            )
            if isinstance(reference["origin_date"].dtype, pd.DatetimeTZDtype):
                for column in ("origin_date", "target_date"):
                    candidate[column] = pd.to_datetime(candidate[column], utc=True)
        return SimpleNamespace(predictions=candidate)

    monkeypatch.setattr(ablation, "run_benchmark", fake_run_benchmark)
    return calls


def test_fixed_variants_use_exact_groups_and_reuse_all_structural(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, states, manifest = _inputs()
    reference = _prediction_frame()
    calls = _patch_runner(monkeypatch, reference)

    result = ablation.run_feature_ablation(
        features,
        states,
        manifest,
        _main_benchmark(reference),
        profile=BenchmarkProfile.quick(),
        selection_end="2023-01-01",
        selection_max_origins=3,
        minimum_selection_predictions=2,
        minimum_holdout_predictions=2,
    )

    assert list(result.manifest["variant"]) == [
        name for name, _ in ablation.VARIANT_GROUPS
    ]
    assert len(calls) == 6
    assert (
        "legacy_feature",
        "regime_boundary__risk_score",
    ) in calls
    assert (
        "legacy_feature",
        "sector_feature",
        "broad_feature",
        "cross_feature",
        "regime_boundary__risk_score",
    ) in calls
    all_rows = result.predictions.loc[
        result.predictions["variant"].eq("all_structural")
    ]
    assert all_rows["reused_main_benchmark"].all()
    assert not result.predictions.loc[
        ~result.predictions["variant"].eq("all_structural"),
        "reused_main_benchmark",
    ].any()
    assert result.manifest.loc[
        result.manifest["variant"].eq("all_structural"), "feature_count"
    ].item() == features.shape[1]
    all_structural = result.manifest.loc[
        result.manifest["variant"].eq("all_structural")
    ].iloc[0]
    assert tuple(all_structural["feature_columns"]) == tuple(features.columns)
    assert all_structural["feature_sha256"] == ablation.feature_columns_sha256(
        list(features.columns)
    )
    document = ablation.feature_ablation_manifest_document(result.manifest)
    assert document["variants"][-1]["feature_columns"] == list(features.columns)
    body = dict(document)
    published_hash = body.pop("sha256")
    import hashlib
    import json
    assert published_hash == hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert set(result.leaderboard["role"]) == {
        "selection_primary",
        "post_2023_retrospective_diagnostic",
    }
    assert len(result.common_origins) == 4
    assert result.predictions.groupby("variant").size().nunique() == 1


def test_ablation_normalizes_naive_cutoff_to_aware_weekly_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, states, manifest = _inputs()
    features.index = features.index.tz_localize("UTC")
    states.index = states.index.tz_localize("UTC")
    reference = _prediction_frame()
    for column in ("origin_date", "target_date"):
        reference[column] = pd.to_datetime(reference[column], utc=True)
    benchmark = _main_benchmark(reference)
    benchmark.selection_end = pd.Timestamp("2023-01-01", tz="UTC")
    _patch_runner(monkeypatch, reference)

    result = ablation.run_feature_ablation(
        features,
        states,
        manifest,
        benchmark,
        profile=BenchmarkProfile.quick(),
        selection_end="2023-01-01",
        selection_max_origins=3,
        minimum_selection_predictions=2,
        minimum_holdout_predictions=2,
    )

    assert isinstance(result.predictions["origin_date"].dtype, pd.DatetimeTZDtype)


def test_common_origin_or_actual_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, states, manifest = _inputs()
    reference = _prediction_frame()
    call_count = 0

    def mismatch_runner(*args, **kwargs):
        nonlocal call_count
        del args, kwargs
        call_count += 1
        candidate = reference.copy()
        if call_count == 2:
            candidate.loc[0, "actual"] = "risk_off"
            candidate.loc[0, "predicted"] = "risk_off"
            candidate.loc[0, list(ablation.PROBABILITY_COLUMNS)] = [0.1, 0.1, 0.8]
        return SimpleNamespace(predictions=candidate)

    monkeypatch.setattr(ablation, "run_benchmark", mismatch_runner)
    with pytest.raises(ValueError, match="exact origins and actuals"):
        ablation.run_feature_ablation(
            features,
            states,
            manifest,
            _main_benchmark(reference),
            profile=BenchmarkProfile.quick(),
            selection_end="2023-01-01",
            minimum_selection_predictions=2,
            minimum_holdout_predictions=2,
        )


def test_reruns_are_trimmed_to_augmented_main_common_origins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, states, manifest = _inputs()
    full = _prediction_frame()
    # Mirrors the structural one-week intersection: the base multiclass run
    # has one target crossing the cutoff whose origin predates the transition
    # diagnostic period, so it is absent from the augmented main benchmark.
    reference = full.loc[
        ~full["origin_date"].eq(pd.Timestamp("2022-12-30"))
    ].reset_index(drop=True)

    monkeypatch.setattr(
        ablation,
        "run_benchmark",
        lambda *args, **kwargs: SimpleNamespace(predictions=full.copy()),
    )
    result = ablation.run_feature_ablation(
        features,
        states,
        manifest,
        _main_benchmark(reference),
        profile=BenchmarkProfile.quick(),
        selection_end="2023-01-01",
        minimum_selection_predictions=1,
        minimum_holdout_predictions=1,
    )

    expected_keys = reference[["origin_date", "target_date"]].reset_index(drop=True)
    for variant, rows in result.predictions.groupby("variant", sort=False):
        actual_keys = rows[["origin_date", "target_date"]].sort_values(
            ["origin_date", "target_date"], ignore_index=True
        )
        pd.testing.assert_frame_equal(actual_keys, expected_keys)
        assert variant in dict(ablation.VARIANT_GROUPS)


def test_selection_rank_is_invariant_to_holdout_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, states, manifest = _inputs()
    reference = _prediction_frame()
    _patch_runner(monkeypatch, reference)
    first = ablation.run_feature_ablation(
        features,
        states,
        manifest,
        _main_benchmark(reference),
        profile=BenchmarkProfile.quick(),
        selection_end="2023-01-01",
        minimum_selection_predictions=2,
        minimum_holdout_predictions=2,
    )

    changed = reference.copy()
    holdout = changed["evaluation_split"].eq("holdout")
    changed.loc[holdout, list(ablation.PROBABILITY_COLUMNS)] = [0.02, 0.02, 0.96]
    changed.loc[holdout, "predicted"] = "risk_off"
    _patch_runner(monkeypatch, changed)
    second = ablation.run_feature_ablation(
        features,
        states,
        manifest,
        _main_benchmark(changed),
        profile=BenchmarkProfile.quick(),
        selection_end="2023-01-01",
        minimum_selection_predictions=2,
        minimum_holdout_predictions=2,
    )

    columns = ["variant", "selection_rank", "selection_winner"]
    first_selection = first.leaderboard.loc[
        first.leaderboard["evaluation_split"].eq("selection"), columns
    ].sort_values("variant", ignore_index=True)
    second_selection = second.leaderboard.loc[
        second.leaderboard["evaluation_split"].eq("selection"), columns
    ].sort_values("variant", ignore_index=True)
    pd.testing.assert_frame_equal(first_selection, second_selection)
    selected = first_selection.loc[first_selection["selection_winner"], "variant"]
    assert selected.tolist() == ["legacy_v3"]


@pytest.mark.parametrize("problem", ["unknown", "duplicate", "missing"])
def test_manifest_group_contract_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    problem: str,
) -> None:
    features, states, base = _inputs()
    manifest = [dict(item) for item in base]
    if problem == "unknown":
        manifest[0]["id"] = "future_group"
    elif problem == "duplicate":
        manifest[1]["id"] = manifest[0]["id"]
    else:
        manifest.pop()
    monkeypatch.setattr(
        ablation,
        "run_benchmark",
        lambda *args, **kwargs: pytest.fail("invalid manifest reached benchmark"),
    )
    with pytest.raises(ValueError):
        ablation.run_feature_ablation(
            features,
            states,
            manifest,
            _main_benchmark(),
            profile=BenchmarkProfile.quick(),
            selection_end="2023-01-01",
        )


def test_manifest_duplicate_feature_and_unknown_feature_fail_closed() -> None:
    features, states, base = _inputs()
    for mutation, message in (
        ("duplicate", "more than once"),
        ("unknown", "unknown features"),
    ):
        manifest = [dict(item) for item in base]
        if mutation == "duplicate":
            manifest[1]["features"] = ("sector_feature",)
        else:
            manifest[0]["features"] = ("not_a_column",)
        with pytest.raises(ValueError, match=message):
            ablation.run_feature_ablation(
                features,
                states,
                manifest,
                _main_benchmark(),
                profile=BenchmarkProfile.quick(),
                selection_end="2023-01-01",
            )


def test_empty_infinite_and_nonfinite_probability_inputs_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, states, manifest = _inputs()
    with pytest.raises(ValueError, match="must not be empty"):
        ablation.run_feature_ablation(
            features.iloc[:0],
            states.iloc[:0],
            manifest,
            _main_benchmark(),
            profile=BenchmarkProfile.quick(),
            selection_end="2023-01-01",
        )
    infinite = features.copy()
    infinite.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        ablation.run_feature_ablation(
            infinite,
            states,
            manifest,
            _main_benchmark(),
            profile=BenchmarkProfile.quick(),
            selection_end="2023-01-01",
        )
    broken = _prediction_frame()
    broken.loc[0, "p_risk_on"] = np.nan
    monkeypatch.setattr(
        ablation,
        "run_benchmark",
        lambda *args, **kwargs: pytest.fail("invalid main benchmark reached rerun"),
    )
    with pytest.raises(ValueError, match="non-finite probabilities"):
        ablation.run_feature_ablation(
            features,
            states,
            manifest,
            _main_benchmark(broken),
            profile=BenchmarkProfile.quick(),
            selection_end="2023-01-01",
        )


def test_paired_delta_is_candidate_minus_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, states, manifest = _inputs()
    reference = _prediction_frame(selection_probability=0.62)
    _patch_runner(monkeypatch, reference)
    result = ablation.run_feature_ablation(
        features,
        states,
        manifest,
        _main_benchmark(reference),
        profile=BenchmarkProfile.quick(),
        selection_end="2023-01-01",
        minimum_selection_predictions=2,
        minimum_holdout_predictions=2,
    )
    legacy = result.leaderboard.loc[
        result.leaderboard["variant"].eq("legacy_v3")
        & result.leaderboard["evaluation_split"].eq("selection")
    ].iloc[0]
    all_structural = result.leaderboard.loc[
        result.leaderboard["variant"].eq("all_structural")
        & result.leaderboard["evaluation_split"].eq("selection")
    ].iloc[0]
    assert legacy["paired_log_loss_delta_vs_legacy"] == pytest.approx(0.0)
    assert legacy["paired_brier_delta_vs_legacy"] == pytest.approx(0.0)
    assert all_structural["paired_log_loss_delta_vs_legacy"] > 0.0
    assert all_structural["paired_brier_delta_vs_legacy"] > 0.0
