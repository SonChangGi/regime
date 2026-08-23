from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

import regime_lab.analysis.fx_ablation as fx_ablation_module

from regime_lab.analysis.fx import FXFeatureResult
from regime_lab.analysis.fx_ablation import (
    FX_VARIANT_ORDER,
    align_fx_features_to_cutoffs,
    fx_ablation_readiness,
    run_fx_shadow_ablation,
)
from regime_lab.analysis.labels import STATE_ORDER
from regime_lab.contract_v5 import V5ContractError, _validate_fx_ablation
from regime_lab.v5_artifacts import FX_ABLATION_OOS_COLUMNS


def _result(rows: int) -> FXFeatureResult:
    weeks = pd.date_range("2023-01-06", periods=rows, freq="W-FRI")
    availability = (
        weeks
        + pd.offsets.Day(3)
        + pd.offsets.Hour(20)
        + pd.offsets.Minute(15)
    )
    features = pd.DataFrame(
        {
            "fx__brd__usd_log_return_1w": range(rows),
            "fx__eur__usd_log_return_1w": range(rows),
        },
        index=weeks,
        dtype=float,
    )
    coverage = pd.DataFrame(
        {"feature_available_at": availability.tz_localize("UTC")},
        index=weeks,
    )
    empty = pd.DataFrame(index=weeks)
    return FXFeatureResult(
        features=features,
        weekly_usd_log_levels=empty,
        weekly_availability=empty,
        coverage=coverage,
        status=empty,
    )


def _shadow_inputs(
    rows: int = 157,
) -> tuple[pd.DataFrame, pd.Series, FXFeatureResult, pd.DatetimeIndex]:
    weeks = pd.date_range("2020-01-03", periods=rows, freq="W-FRI")
    position = np.arange(rows, dtype=float)
    latent = np.sin(position / 5.0) + 0.35 * np.cos(position / 11.0)
    states = pd.Series(
        np.select(
            [latent > 0.35, latent < -0.35],
            [STATE_ORDER[0], STATE_ORDER[2]],
            default=STATE_ORDER[1],
        ),
        index=weeks,
        dtype="object",
    )
    core_features = pd.DataFrame(
        {
            "core__trend": np.roll(latent, 2),
            "core__stress": np.cos(position / 7.0),
        },
        index=weeks,
    )
    features = pd.DataFrame(
        {
            "fx__brd__usd_log_return_1w": latent,
            "fx__afe__usd_log_return_1w": np.sin(position / 6.0),
            "fx__eme__usd_log_return_1w": np.cos(position / 9.0),
            **{
                f"fx__{code}__usd_log_return_1w": (
                    latent * (1.0 + offset / 20.0)
                    + 0.01 * np.sin(position / (offset + 2.0))
                )
                for offset, code in enumerate(
                    ("eur", "jpy", "gbp", "chf", "cad", "aud", "cny", "mxn", "brl")
                )
            },
        },
        index=weeks,
        dtype=float,
    )
    availability = (
        weeks
        + pd.offsets.Day(3)
        + pd.offsets.Hour(20)
        + pd.offsets.Minute(15)
    )
    coverage = pd.DataFrame(
        {
            "feature_available_at": availability.tz_localize("UTC"),
            "bilateral_level_count": 9,
            "core_level_count": 3,
        },
        index=weeks,
    )
    empty = pd.DataFrame(index=weeks)
    result = FXFeatureResult(
        features=features,
        weekly_usd_log_levels=empty,
        weekly_availability=empty,
        coverage=coverage,
        status=empty,
        official_release_archive_ingest=True,
        availability_basis="official_archive_release_schedule",
    )
    return core_features, states, result, weeks


def test_alignment_uses_first_seen_and_never_backfills_prior_cutoffs() -> None:
    result = _result(2)
    cutoffs = pd.date_range("2023-01-06", periods=2, freq="W-FRI")

    aligned = align_fx_features_to_cutoffs(result, cutoffs)

    assert pd.isna(aligned.iloc[0]["fx__brd__usd_log_return_1w"])
    assert aligned.iloc[1]["fx__brd__usd_log_return_1w"] == 0.0
    assert pd.Timestamp(aligned.iloc[1]["fx_observation_week"]) == pd.Timestamp(
        "2023-01-06"
    )
    assert aligned.iloc[1]["fx_observation_age_days"] == 7


def test_ablation_readiness_has_common_origin_gate_and_hashed_manifest() -> None:
    result = _result(160)
    cutoffs = pd.date_range("2023-01-06", periods=160, freq="W-FRI")

    readiness = fx_ablation_readiness(result, cutoffs)

    assert readiness["status"] == "ready_for_evaluation"
    assert readiness["eligible_common_weeks"] == 159
    assert readiness["first_eligible_cutoff"] == "2023-01-13"
    assert readiness["historical_availability_backfill"] is False
    assert tuple(readiness["variants"]) == FX_VARIANT_ORDER
    assert [row["variant"] for row in readiness["manifest"]] == list(
        FX_VARIANT_ORDER
    )
    assert readiness["manifest"][0]["feature_count"] == 0
    assert all(len(row["feature_columns_sha256"]) == 64 for row in readiness["manifest"])


def test_missing_fx_stays_unavailable() -> None:
    cutoffs = pd.date_range("2023-01-06", periods=160, freq="W-FRI")

    readiness = fx_ablation_readiness(None, cutoffs)

    assert readiness["status"] == "unavailable"
    assert readiness["eligible_common_weeks"] == 0
    assert readiness["manifest"] == []


def test_one_fx_week_cannot_be_carried_forward_into_156_eligible_origins() -> None:
    result = _result(1)
    cutoffs = pd.date_range("2023-01-06", periods=160, freq="W-FRI")

    readiness = fx_ablation_readiness(result, cutoffs)

    assert readiness["status"] == "insufficient_history"
    assert readiness["eligible_common_weeks"] == 1
    assert readiness["first_eligible_cutoff"] == "2023-01-13"
    assert readiness["last_eligible_cutoff"] == "2023-01-13"


def test_shadow_ablation_fails_closed_when_fx_is_unavailable_or_insufficient() -> None:
    core, states, _result_value, cutoffs = _shadow_inputs()

    unavailable = run_fx_shadow_ablation(
        core,
        states,
        None,
        cutoffs,
        bootstrap_resamples=19,
    )
    assert unavailable["status"] == "unavailable"
    assert unavailable["common_origin_required_pairs"] == 9
    assert unavailable["variant_metrics"] == []
    assert unavailable["gate"]["comparisons"] == []
    assert unavailable["promotion_allowed"] is False
    assert unavailable["core_champion_promoted"] is False

    short_core, short_states, short_result, short_cutoffs = _shadow_inputs(156)
    insufficient = run_fx_shadow_ablation(
        short_core,
        short_states,
        short_result,
        short_cutoffs,
        bootstrap_resamples=19,
    )
    assert insufficient["status"] == "insufficient_history"
    assert insufficient["eligible_common_weeks"] == 155
    assert insufficient["variant_metrics"] == []
    assert insufficient["promotion_candidate"] is None


def test_ready_shadow_ablation_is_evaluated_on_identical_purged_origins() -> None:
    core, states, result, cutoffs = _shadow_inputs()
    kwargs = {"bootstrap_resamples": 99, "bootstrap_seed": 17}

    first = run_fx_shadow_ablation(core, states, result, cutoffs, **kwargs)
    second = run_fx_shadow_ablation(core, states, result, cutoffs, **kwargs)

    assert first["status"] == "evaluated"
    assert first["status_reason"] is None
    assert first["eligible_common_weeks"] == 156
    assert first["minimum_common_weeks"] == 156
    assert first["minimum_train_weeks"] == 104
    assert first["common_origin_required_pairs"] == 9
    assert first["target_horizon_weeks"] == 1
    assert first["purge_weeks"] == 1
    assert first["model"]["name"] == "fixed_l2_multinomial_logistic"

    common = first["common_evaluation_origins"]
    assert common["count"] == 50
    assert len(common["rows"]) == 50
    assert common["rows"][0]["train_size"] == 104
    assert all(row["purged_origin_count"] == 1 for row in common["rows"])
    assert all(
        pd.Timestamp(row["last_train_target"])
        < pd.Timestamp(row["origin_date"])
        for row in common["rows"]
    )

    metrics = first["variant_metrics"]
    assert [row["variant"] for row in metrics] == list(FX_VARIANT_ORDER)
    assert {row["n"] for row in metrics} == {50}
    assert {row["n_predictions"] for row in metrics} == {50}
    assert {row["origin_sha256"] for row in metrics} == {common["sha256"]}
    assert all(row["fallback_count"] == 0 for row in metrics)
    assert all(np.isfinite(row["log_loss"]) for row in metrics)
    assert all(np.isfinite(row["brier"]) for row in metrics)

    assert len(first["gate"]["comparisons"]) == 3
    assert all(
        comparison["reference_variant"] == "v4_control"
        for comparison in first["gate"]["comparisons"]
    )
    assert first["variant_metrics"] == second["variant_metrics"]
    assert first["common_evaluation_origins"] == second["common_evaluation_origins"]
    assert first["gate"] == second["gate"]
    json.dumps(first, allow_nan=False)
    assert first["promotion_allowed"] is False
    assert first["promotion_candidate"] is None
    assert first["core_champion_promoted"] is False


def test_ready_shadow_ablation_emits_frozen_derived_only_oos_evidence() -> None:
    core, states, result, cutoffs = _shadow_inputs()
    captured: list[pd.DataFrame] = []

    evaluated = run_fx_shadow_ablation(
        core,
        states,
        result,
        cutoffs,
        bootstrap_resamples=19,
        evidence_sink=captured.append,
    )

    assert evaluated["status"] == "evaluated"
    assert len(captured) == 1
    evidence = captured[0]
    assert tuple(evidence.columns) == FX_ABLATION_OOS_COLUMNS
    common = evaluated["common_evaluation_origins"]
    assert len(evidence) == common["count"] * len(FX_VARIANT_ORDER)
    assert (
        evidence[["origin_date", "target_date", "variant"]]
        .duplicated()
        .sum()
        == 0
    )
    assert set(evidence["variant"]) == set(FX_VARIANT_ORDER)
    assert set(evidence["evaluation_split"]) == {"prospective_shadow"}
    assert set(evidence["common_origins_sha256"]) == {common["sha256"]}
    assert set(evidence["gap"]) == {1}
    assert set(evidence["purged_origin_count"]) == {1}
    assert not evidence["fallback"].any()
    assert set(evidence["fallback_reason"]) == {""}
    assert all(
        pd.Timestamp(row.last_train_target) < pd.Timestamp(row.origin_date)
        for row in evidence.itertuples(index=False)
    )
    probabilities = evidence[
        ["p_risk_on", "p_transition", "p_risk_off"]
    ].to_numpy(dtype=float)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert not any(
        "fx__" in str(column) or "core__" in str(column)
        for column in evidence.columns
    )


def test_convergence_warning_is_recorded_as_fold_fallback(monkeypatch) -> None:
    class WarningEstimator:
        def fit(self, x_train, y_train):
            warnings.warn("did not converge", ConvergenceWarning)

    monkeypatch.setattr(
        fx_ablation_module,
        "_fixed_multinomial_model",
        WarningEstimator,
    )
    core, states, result, cutoffs = _shadow_inputs()
    captured: list[pd.DataFrame] = []

    evaluated = run_fx_shadow_ablation(
        core,
        states,
        result,
        cutoffs,
        bootstrap_resamples=19,
        evidence_sink=captured.append,
    )

    count = evaluated["common_evaluation_origins"]["count"]
    assert all(
        row["fallback_count"] == count
        for row in evaluated["variant_metrics"]
    )
    assert all(
        row["fallback_reasons"] == {"model_fit_or_prediction_error": count}
        for row in evaluated["variant_metrics"]
    )
    assert captured[0]["fallback"].all()
    assert set(captured[0]["fallback_reason"]) == {
        "model_fit_or_prediction_error"
    }
    assert all(not row["gate_passed"] for row in evaluated["gate"]["comparisons"])


def test_evaluated_shadow_ablation_satisfies_the_v5_payload_contract() -> None:
    core, states, result, cutoffs = _shadow_inputs()

    evaluated = run_fx_shadow_ablation(core, states, result, cutoffs)

    _validate_fx_ablation(evaluated)


def test_v5_fx_contract_rejects_self_consistent_but_forged_holm_value() -> None:
    core, states, result, cutoffs = _shadow_inputs()
    evaluated = run_fx_shadow_ablation(core, states, result, cutoffs)
    comparison = evaluated["gate"]["comparisons"][0]
    assert comparison["gate_passed"] is True
    comparison["holm_adjusted_p_value"] += 0.0001

    with pytest.raises(V5ContractError, match="Holm"):
        _validate_fx_ablation(evaluated)
