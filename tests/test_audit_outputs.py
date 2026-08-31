from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from regime_lab.analysis.decision_shadow import build_decision_shadow
from regime_lab.analysis.labels import STATE_ORDER
from regime_lab.analysis.models import (
    model_manifest,
    model_manifest_sha256,
)
from regime_lab.analysis.structural_models import (
    DEFAULT_ENSEMBLE_EXPERTS,
    ENSEMBLE_MODEL_NAME,
    JOINT_MODEL_NAME,
    PROBABILITY_COLUMNS,
    causal_dynamic_ensemble,
)
from regime_lab.analysis.validation import (
    evaluate_predictions,
    select_champion_with_diagnostics,
)
from regime_lab.feature_quality import (
    canonical_feature_quality_json_bytes,
    feature_quality_artifact_manifest,
    feature_quality_document,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_outputs.py"
SPEC = importlib.util.spec_from_file_location("audit_outputs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit_outputs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_outputs)


def test_anchored_transition_projection_preserves_one_week_and_pools_violations() -> None:
    assert audit_outputs.anchored_transition_projection(
        {"1w": 0.30, "4w": 0.20, "13w": 0.50}
    ) == {"1w": 0.30, "4w": 0.30, "13w": 0.50}
    assert audit_outputs.anchored_transition_projection(
        {"1w": 0.10, "4w": 0.70, "13w": 0.30}
    ) == {"1w": 0.10, "4w": 0.50, "13w": 0.50}


def test_calendar_days_preserves_date_only_and_converts_timestamp_to_market_day() -> None:
    values = pd.Series(
        [
            "2023-01-06",
            "2023-01-06T02:00:00+00:00",
            "2023-01-06T21:00:00+00:00",
        ]
    )

    normalized = audit_outputs._calendar_days(values, context="test dates")

    assert normalized.dt.strftime("%Y-%m-%d").tolist() == [
        "2023-01-06",
        "2023-01-05",
        "2023-01-06",
    ]


def test_v5_feature_quality_binds_to_model_features_not_source_catalog(
    tmp_path: Path,
) -> None:
    index = pd.date_range("2024-01-05", periods=60, freq="W-FRI", tz="UTC")
    document = feature_quality_document(
        pd.DataFrame(
            {
                "signal": np.arange(60, dtype=float),
                "regime_boundary": np.linspace(0.0, 1.0, 60),
            },
            index=index,
        )
    )
    manifest = feature_quality_artifact_manifest(document)
    (tmp_path / "feature-quality.json").write_bytes(
        canonical_feature_quality_json_bytes(document)
    )
    payload = {
        "model": {"feature_quality_artifact": manifest},
        "feature_catalog": [{"id": "source-level-catalog-entry"}],
    }

    result = audit_outputs._audit_v5_feature_quality(
        payload,
        tmp_path,
        expected_features=("regime_boundary", "signal"),
    )
    assert result["feature_count"] == 2

    with pytest.raises(
        audit_outputs.AuditFailure,
        match="model feature manifest",
    ):
        audit_outputs._audit_v5_feature_quality(
            payload,
            tmp_path,
            expected_features=("signal", "missing_feature"),
        )


def test_v5_serialized_probability_tolerance_matches_eight_decimal_contract() -> None:
    values = pd.DataFrame(
        [
            [0.33333333, 0.33333333, 0.33333333],
            [0.33333334, 0.33333334, 0.33333333],
        ]
    )

    assert audit_outputs._v5_serialized_probability_rows_are_valid(values)


def test_v5_serialized_probability_tolerance_rejects_material_tampering() -> None:
    tampered = pd.DataFrame([[0.33333332, 0.33333333, 0.33333333]])

    assert not audit_outputs._v5_serialized_probability_rows_are_valid(tampered)


def _self_contained_v4_core_fixture(tmp_path: Path) -> tuple[dict, Path]:
    artifacts = tmp_path / "v4-core"
    artifacts.mkdir()
    # This fixture audits the frozen V4 branch.  Keep its historical roster
    # explicit instead of inheriting the active weekly V5 training roster.
    model_names = (
        *sorted(audit_outputs.V4_BASE_MODELS),
        JOINT_MODEL_NAME,
        ENSEMBLE_MODEL_NAME,
    )
    manifest_body = model_manifest("quick", names=model_names)
    manifest_hash = model_manifest_sha256("quick", names=model_names)
    (artifacts / "candidate-manifest.json").write_text(
        json.dumps({**manifest_body, "sha256": manifest_hash}, sort_keys=True),
        encoding="utf-8",
    )
    origins = [
        *zip(
            pd.date_range("2022-10-07T20:00:00Z", periods=3, freq="7D"),
            ["selection"] * 3,
        ),
        *zip(
            pd.date_range("2023-01-06T21:00:00Z", periods=3, freq="7D"),
            ["holdout"] * 3,
        ),
    ]
    prediction_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    for position, (origin, split) in enumerate(origins):
        origin = pd.Timestamp(origin)
        target = origin + timedelta(weeks=1)
        actual = STATE_ORDER[position % len(STATE_ORDER)]
        current = STATE_ORDER[(position - 1) % len(STATE_ORDER)]
        for name in model_names:
            if name == ENSEMBLE_MODEL_NAME:
                continue
            if name == "majority":
                probability = np.asarray([0.40, 0.30, 0.30], dtype=float)
            elif name == "persistence":
                probability = np.asarray([0.34, 0.33, 0.33], dtype=float)
            else:
                probability = np.full(len(STATE_ORDER), 0.10, dtype=float)
                probability[STATE_ORDER.index(actual)] = 0.80
            prediction_rows.append(
                {
                    "origin_date": origin,
                    "target_date": target,
                    "model": name,
                    "evaluation_split": split,
                    "current_state": current,
                    "actual": actual,
                    "predicted": STATE_ORDER[int(probability.argmax())],
                    "train_size": 520 + position,
                    "gap": 1,
                    "fallback": False,
                    "fallback_reason": "none",
                    **{
                        column: float(probability[index])
                        for index, column in enumerate(PROBABILITY_COLUMNS)
                    },
                }
            )
        split_rows.append(
            {
                "origin_date": origin,
                "target_date": target,
                "evaluation_split": split,
                "train_start": origin - timedelta(weeks=521),
                "last_train_origin": origin - timedelta(weeks=2),
                "last_train_target": origin - timedelta(weeks=1),
                "first_purged_origin": origin - timedelta(weeks=1),
                "purged_origin_count": 1,
                "train_size": 520 + position,
                "gap": 1,
            }
        )

    base_predictions = pd.DataFrame(prediction_rows)
    ensemble = causal_dynamic_ensemble(
        base_predictions.loc[
            base_predictions["model"].isin(DEFAULT_ENSEMBLE_EXPERTS)
        ]
    )
    predictions = pd.concat(
        [base_predictions, ensemble.predictions],
        ignore_index=True,
        sort=False,
    )
    splits = pd.DataFrame(split_rows)
    selection = predictions.loc[predictions["evaluation_split"].eq("selection")]
    holdout = predictions.loc[predictions["evaluation_split"].eq("holdout")]
    selection_metrics = evaluate_predictions(selection)
    holdout_metrics = evaluate_predictions(holdout)
    champion, diagnostics = select_champion_with_diagnostics(
        selection_metrics,
        selection,
    )
    assert champion == "markov"

    selection_metrics = selection_metrics.set_index("model")
    leaderboard = holdout_metrics.copy()
    leaderboard["selected"] = leaderboard["model"].eq(champion)
    for field in selection_metrics.columns:
        leaderboard[f"selection_{field}"] = leaderboard["model"].map(
            selection_metrics[field]
        )
    ranked = leaderboard.sort_values(["log_loss", "model"], ignore_index=True)
    embedded_leaderboard = [
        {
            "rank": rank,
            "name": str(row.model),
            "selected": bool(row.selected),
            "is_champion": str(row.model) == champion,
            "log_loss": float(row.log_loss),
        }
        for rank, row in enumerate(ranked.itertuples(index=False), start=1)
    ]

    predictions.to_csv(artifacts / "oos-predictions.csv", index=False)
    splits.to_csv(artifacts / "walk-forward-splits.csv", index=False)
    leaderboard.to_csv(artifacts / "model-leaderboard.csv", index=False)
    diagnostics.to_csv(artifacts / "selection-diagnostics.csv", index=False)
    ensemble.weights.to_csv(artifacts / "stacking-weights.csv", index=False)

    payload = {
        "meta": {
            "mode": "demo",
            "result_version": audit_outputs.V4_RESULT_VERSION,
        },
        "model": {
            "profile": "quick",
            "champion": champion,
            "selection_end": "2023-01-01",
            "candidate_manifest_sha256": manifest_hash,
            "candidate_manifest": manifest_body,
            "leaderboard": embedded_leaderboard,
            "selection_diagnostics": _v5_frame_records(diagnostics),
        },
    }
    return payload, artifacts


def _v5_execution_parameters(
    *,
    profile: str = "quick",
    duration_resamples: int = 7,
    outcome_resamples: int = 7,
) -> dict[str, object]:
    minimum = 3 if profile == "quick" else 12
    maximum = {"quick": 3, "standard": 60, "full": None}[profile]
    overrides: list[str] = []
    if duration_resamples != 1_999:
        overrides.append("duration.bootstrap_resamples")
    if outcome_resamples != 1_999:
        overrides.append("conditional_asset_statistics.bootstrap_resamples")
    parameters: dict[str, object] = {
        "profile": profile,
        "directional_minimum_selection_predictions": minimum,
        "directional_minimum_diagnostic_predictions": minimum,
        "directional_maximum_selection_origins": maximum,
        "directional_maximum_diagnostic_origins": maximum,
        "duration_bootstrap_resamples": duration_resamples,
        "conditional_outcome_bootstrap_resamples": outcome_resamples,
        "preregistered_bootstrap_resamples": 1_999,
        "preregistration_overrides": overrides,
    }
    return {
        **parameters,
        "sha256": audit_outputs.canonical_json_sha256(parameters),
    }


def _v5_frame_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {
            str(key): None if bool(pd.isna(value)) else value
            for key, value in row.items()
        }
        for row in frame.to_dict(orient="records")
    ]


def _write_v5_contract_frames(
    directory: Path,
    frames: dict[str, pd.DataFrame],
) -> tuple[dict[str, dict[str, object]], dict[str, pd.DataFrame]]:
    keys_by_path = {
        path: key
        for key, path in (
            *audit_outputs.V5_RESEARCH_ARTIFACTS,
            *audit_outputs.V5_MODEL_CONDITIONED_ARTIFACTS,
            *audit_outputs.V5_FX_ARTIFACTS,
        )
    }
    contracts: dict[str, dict[str, object]] = {}
    for path_name, frame in frames.items():
        path = directory / path_name
        frame.to_csv(path, index=False)
        contracts[keys_by_path[path_name]] = {
            "path": path_name,
            "row_count": len(frame),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    loaded = audit_outputs._audit_v5_file_contracts(
        contracts,
        directory,
        context="mutated fixture manifest",
    )
    return contracts, loaded


def _v5_directional_audit_fixture() -> tuple[dict, dict[str, pd.DataFrame]]:
    probability_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    leaderboard_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    forecast_rows: list[dict[str, object]] = []
    model = "empirical_first_passage"
    support_reason = (
        "insufficient_departure_events;"
        "insufficient_destination_classes;"
        "insufficient_event_blocks"
    )
    for horizon in (1, 4, 13):
        for split, start in (
            ("selection", "2022-10-07T20:00:00Z"),
            ("retrospective_diagnostic", "2024-01-05T21:00:00Z"),
        ):
            origins = pd.date_range(start, periods=3, freq="7D")
            for origin in origins:
                target = origin + timedelta(weeks=horizon)
                probability_rows.append(
                    {
                        "horizon_weeks": horizon,
                        "origin_date": origin,
                        "target_end": target,
                        "evaluation_split": split,
                        "model": model,
                        "current_state": "risk_on",
                        "actual_outcome": "no_departure",
                        "actual_change": False,
                        "p_no_departure": 0.7,
                        "p_risk_on": 0.0,
                        "p_transition": 0.2,
                        "p_risk_off": 0.1,
                        "fallback": False,
                    }
                )
                split_rows.append(
                    {
                        "horizon_weeks": horizon,
                        "origin_date": origin,
                        "target_end": target,
                        "evaluation_split": split,
                        "last_train_target_end": origin - timedelta(weeks=1),
                        "purged_origin_count": horizon,
                    }
                )
            leaderboard_rows.append(
                {
                    "horizon_weeks": horizon,
                    "evaluation_split": split,
                    "model": model,
                    "selected": True,
                    "score_target": "first_destination_given_departure",
                    "log_loss": None,
                    "brier": None,
                    "n_predictions": 3,
                    "event_count": 0,
                    "destination_class_count": 0,
                    "effective_event_blocks": 0,
                    "fallback_count": 0,
                }
            )
        forecast_origin = pd.Timestamp("2026-07-03T20:00:00Z")
        forecast_rows.append(
            {
                "horizon_weeks": horizon,
                "origin_date": forecast_origin,
                "target_end": forecast_origin + timedelta(weeks=horizon),
                "model": model,
                "current_state": "risk_on",
                "p_no_departure": 0.7,
                "p_risk_on": 0.0,
                "p_transition": 0.2,
                "p_risk_off": 0.1,
                "fallback": False,
            }
        )
        diagnostic_rows.append(
            {
                "horizon_weeks": horizon,
                "model": model,
                "reference_model": model,
                "selected": True,
                "gate_passed": False,
                "gate_reason": support_reason,
                "score_target": "first_destination_given_departure",
                "selection_event_count": 0,
                "selection_destination_class_count": 0,
                "selection_effective_event_blocks": 0,
                "minimum_selection_events": 8,
                "minimum_destination_classes": 2,
                "minimum_event_blocks": 3,
                "log_loss": None,
                "brier": None,
                "absolute_log_loss_improvement": None,
                "holm_adjusted_p_value": None,
                "fallback_count": 0,
            }
        )
    predictions = pd.DataFrame(probability_rows)
    forecasts = pd.DataFrame(forecast_rows)
    payload = {
        "model": {
            "profile": "quick",
            "execution_parameters": _v5_execution_parameters(),
            "directional_transition": {
                "target": "first_departure_state_within_h_or_no_departure",
                "deployed_direction_role": "first_destination_given_departure",
                "selection_metric": "conditional_destination_log_loss",
                "minimum_selection_departure_events": 8,
                "minimum_selection_destination_classes": 2,
                "minimum_selection_event_blocks": 3,
                "champions": {
                    "1w": model,
                    "4w": model,
                    "13w": model,
                },
                "leaderboard": leaderboard_rows,
                "selection_diagnostics": diagnostic_rows,
            }
        }
    }
    frames = {
        "directional-oos-predictions.csv": predictions,
        "directional-forecasts.csv": forecasts,
        "directional-walk-forward-splits.csv": pd.DataFrame(split_rows),
        "directional-selection-diagnostics.csv": pd.DataFrame(diagnostic_rows),
        "directional-model-leaderboard.csv": pd.DataFrame(leaderboard_rows),
    }
    return payload, frames


def _v5_supported_directional_audit_fixture() -> tuple[
    dict,
    dict[str, pd.DataFrame],
]:
    prediction_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    forecast_rows: list[dict[str, object]] = []
    models = {
        "empirical_first_passage": 0.60,
        "markov_first_passage": 0.55,
        "regularized_multinomial": 0.95,
        "weak_regularized_multinomial": 0.65,
    }
    event_positions = {0, 1, 2, 3, 13, 14, 26, 27}
    for horizon in (1, 4, 13):
        for split, start, periods in (
            ("selection", "2022-01-07T21:00:00Z", 39),
            ("retrospective_diagnostic", "2024-01-05T21:00:00Z", 12),
        ):
            origins = pd.date_range(start, periods=periods, freq="7D")
            for position, origin in enumerate(origins):
                target = origin + timedelta(weeks=horizon)
                is_event = split == "selection" and position in event_positions
                actual = (
                    ("transition" if position % 2 == 0 else "risk_off")
                    if is_event
                    else "no_departure"
                )
                split_rows.append(
                    {
                        "horizon_weeks": horizon,
                        "origin_date": origin,
                        "target_end": target,
                        "evaluation_split": split,
                        "last_train_target_end": origin - timedelta(weeks=1),
                        "purged_origin_count": horizon,
                    }
                )
                for model, correct_destination_probability in models.items():
                    if actual == "transition":
                        transition = correct_destination_probability
                        risk_off = 1.0 - correct_destination_probability
                    elif actual == "risk_off":
                        transition = 1.0 - correct_destination_probability
                        risk_off = correct_destination_probability
                    else:
                        transition = 0.5
                        risk_off = 0.5
                    departure_mass = 0.30
                    prediction_rows.append(
                        {
                            "horizon_weeks": horizon,
                            "origin_date": origin,
                            "target_end": target,
                            "evaluation_split": split,
                            "model": model,
                            "current_state": "risk_on",
                            "actual_outcome": actual,
                            "actual_change": is_event,
                            "p_no_departure": 1.0 - departure_mass,
                            "p_risk_on": 0.0,
                            "p_transition": departure_mass * transition,
                            "p_risk_off": departure_mass * risk_off,
                            "fallback": False,
                        }
                    )
    predictions = pd.DataFrame(prediction_rows)
    leaderboard, diagnostics, champions = audit_outputs._recompute_v5_directional(
        predictions
    )
    for horizon in (1, 4, 13):
        forecast_origin = pd.Timestamp("2026-07-03T20:00:00Z")
        forecast_rows.append(
            {
                "horizon_weeks": horizon,
                "origin_date": forecast_origin,
                "target_end": forecast_origin + timedelta(weeks=horizon),
                "model": champions[horizon],
                "current_state": "risk_on",
                "p_no_departure": 0.7,
                "p_risk_on": 0.0,
                "p_transition": 0.2,
                "p_risk_off": 0.1,
                "fallback": False,
            }
        )
    payload = {
        "model": {
            "profile": "full",
            "execution_parameters": _v5_execution_parameters(profile="full"),
            "directional_transition": {
                "target": "first_departure_state_within_h_or_no_departure",
                "deployed_direction_role": "first_destination_given_departure",
                "selection_metric": "conditional_destination_log_loss",
                "minimum_selection_departure_events": 8,
                "minimum_selection_destination_classes": 2,
                "minimum_selection_event_blocks": 3,
                "champions": {
                    f"{horizon}w": champion
                    for horizon, champion in champions.items()
                },
                "leaderboard": _v5_frame_records(leaderboard),
                "selection_diagnostics": _v5_frame_records(diagnostics),
            },
        }
    }
    frames = {
        "directional-oos-predictions.csv": predictions,
        "directional-model-leaderboard.csv": leaderboard,
        "directional-walk-forward-splits.csv": pd.DataFrame(split_rows),
        "directional-selection-diagnostics.csv": diagnostics,
        "directional-forecasts.csv": pd.DataFrame(forecast_rows),
    }
    return payload, frames


def _v5_conditional_audit_fixture() -> tuple[dict, dict[str, pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    origins = pd.date_range("2020-01-03T21:00:00Z", periods=20, freq="7D")
    for horizon in (1, 4, 13):
        for position, origin in enumerate(origins):
            entry = origin + timedelta(weeks=1)
            rows.append(
                {
                    "origin_position": position,
                    "origin_date": origin,
                    "entry_date": entry,
                    "exit_date": entry + timedelta(weeks=horizon - 1),
                    "state": "risk_on",
                    "episode_id": position // 4,
                    "asset": "SPY",
                    "horizon_weeks": horizon,
                    "execution_lag_weeks": 1,
                    "return_currency": "USD",
                    "forward_return": (position - 8) / 100.0 + horizon / 1000.0,
                    "max_drawdown": -(position % 5 + horizon) / 100.0,
                }
            )
    outcomes = pd.DataFrame(rows)
    keys = pd.DataFrame(
        [
            {
                "state": "risk_on",
                "asset": "SPY",
                "horizon_weeks": horizon,
                "minimum_non_overlapping_observations": 5,
            }
            for horizon in (1, 4, 13)
        ]
    )
    statistics = audit_outputs._recompute_v5_conditional_statistics(
        outcomes,
        keys,
        expected_resamples=7,
    )
    payload = {
        "model": {
            "profile": "quick",
            "execution_parameters": _v5_execution_parameters(),
        },
        "weekly": [
            {
                "date": origin.date().isoformat(),
                "current": {"state": "risk_on"},
            }
            for origin in pd.date_range(
                origins[0], periods=len(origins) + 1, freq="7D"
            )
        ],
        "research": {
            "conditional_asset_stats": {
                "method": (
                    "matched_oos_actual_next_state_target_week_adjusted_forward_return"
                ),
                "role": "matched_oracle_diagnostic",
                "conditioning": "actual_next_state_on_matched_oos_origins",
                "state_horizon_weeks": 1,
                "execution_lag_weeks": 1,
                "entry_price_basis": "next_week_adjusted_open",
                "exit_price_basis": "horizon_week_adjusted_close",
                "rebalance_policy": "none_fixed_asset_hold",
                "origin_sampling": "weekly_rolling_overlapping",
                "return_measure": "provider_adjusted_forward_return",
                "entry_week_distribution_policy": (
                    "conservative_excluded_without_ex_date"
                ),
                "corporate_action_policy": (
                    "same_row_adjustment_factor_split_consistent"
                ),
                "drawdown_observation_basis": (
                    "entry_adjusted_open_then_weekly_adjusted_closes"
                ),
                "rows": _v5_frame_records(statistics),
            }
        },
    }
    frames = {
        "conditional-asset-outcomes.csv": outcomes,
        "conditional-asset-statistics.csv": statistics,
    }
    return payload, frames


def _v5_model_conditioned_audit_fixture() -> tuple[
    dict,
    dict[str, pd.DataFrame],
]:
    models = ("markov", "xgboost")
    origins = pd.date_range("2024-01-05T21:00:00Z", periods=24, freq="7D")
    states_by_model = {
        "markov": [STATE_ORDER[(position // 2) % 3] for position in range(24)],
        "xgboost": [STATE_ORDER[2 - ((position // 2) % 3)] for position in range(24)],
    }
    episodes_by_model: dict[str, list[int]] = {}
    for name, states in states_by_model.items():
        episodes: list[int] = []
        episode = -1
        prior: str | None = None
        for state in states:
            if state != prior:
                episode += 1
            episodes.append(episode)
            prior = state
        episodes_by_model[name] = episodes

    weekly = [
        {
            "date": origin.date().isoformat(),
            "model_forecasts": [
                {
                    "model": name,
                    "state": states_by_model[name][position],
                    "date": (origin + timedelta(weeks=1)).date().isoformat(),
                }
                for name in models
            ],
        }
        for position, origin in enumerate(origins)
    ]
    outcome_rows: list[dict[str, object]] = []
    for model_position, name in enumerate(models):
        for position, origin in enumerate(origins):
            entry = origin + timedelta(weeks=1)
            for horizon in audit_outputs.V5_OUTCOME_HORIZONS:
                if position + horizon >= len(origins):
                    continue
                for asset_position, asset in enumerate(
                    audit_outputs.V5_OUTCOME_ASSETS
                ):
                    outcome_rows.append(
                        {
                            "conditioning_model": name,
                            "origin_position": position,
                            "origin_date": origin,
                            "entry_date": entry,
                            "exit_date": entry + timedelta(weeks=horizon - 1),
                            "state": states_by_model[name][position],
                            "episode_id": episodes_by_model[name][position],
                            "asset": asset,
                            "horizon_weeks": horizon,
                            "execution_lag_weeks": 1,
                            "return_currency": "USD",
                            "forward_return": (
                                (
                                    position
                                    + asset_position
                                    + horizon
                                    + model_position
                                )
                                % 11
                                - 5
                            )
                            / 100.0,
                            "max_drawdown": -(
                                1 + (position + asset_position + horizon) % 7
                            )
                            / 100.0,
                        }
                    )
    outcomes = pd.DataFrame(outcome_rows)
    statistic_keys = pd.DataFrame(
        [
            {
                "state": state,
                "asset": asset,
                "horizon_weeks": horizon,
                "minimum_non_overlapping_observations": 5,
            }
            for state in STATE_ORDER
            for asset in audit_outputs.V5_OUTCOME_ASSETS
            for horizon in audit_outputs.V5_OUTCOME_HORIZONS
        ]
    )
    statistic_frames: list[pd.DataFrame] = []
    for name in models:
        model_outcomes = outcomes.loc[
            outcomes["conditioning_model"].eq(name)
        ].drop(columns="conditioning_model")
        statistics = audit_outputs._recompute_v5_conditional_statistics(
            model_outcomes,
            statistic_keys,
            expected_resamples=7,
        )
        statistics.insert(0, "conditioning_model", name)
        statistic_frames.append(statistics)
    combined_statistics = pd.concat(statistic_frames, ignore_index=True)
    payload = {
        "model": {
            "profile": "quick",
            "execution_parameters": _v5_execution_parameters(),
            "forecast_comparison": {"models": list(models)},
        },
        "weekly": weekly,
        "research": {
            "model_conditioned_asset_stats": {
                "method": (
                    "matched_oos_predicted_next_state_target_week_adjusted_forward_return"
                ),
                "role": "retrospective_model_diagnostic",
                "conditioning": "hard_argmax_oos_forecast",
                "forecast_horizon_weeks": 1,
                "execution_lag_weeks": 1,
                "entry_price_basis": "next_week_adjusted_open",
                "exit_price_basis": "horizon_week_adjusted_close",
                "rebalance_policy": "none_fixed_asset_hold",
                "origin_sampling": "weekly_rolling_overlapping",
                "return_measure": "provider_adjusted_forward_return",
                "entry_week_distribution_policy": (
                    "conservative_excluded_without_ex_date"
                ),
                "corporate_action_policy": (
                    "same_row_adjustment_factor_split_consistent"
                ),
                "drawdown_observation_basis": (
                    "entry_adjusted_open_then_weekly_adjusted_closes"
                ),
                "horizons_weeks": list(audit_outputs.V5_OUTCOME_HORIZONS),
                "assets": list(audit_outputs.V5_OUTCOME_ASSETS),
                "models": list(models),
                "return_currency": "USD",
                "rows": _v5_frame_records(combined_statistics),
            }
        },
    }
    frames = {
        "model-conditioned-asset-outcomes.csv": outcomes,
        "model-conditioned-asset-statistics.csv": combined_statistics,
    }
    return payload, frames


def _v5_duration_audit_fixture() -> tuple[dict, pd.DataFrame, dict[str, object]]:
    states: list[str] = []
    for duration in (2, 3, 4, 5, 6):
        states.extend(["risk_on"] * duration)
        states.append("transition")
    states.extend(["risk_on"] * 3)
    dates = pd.date_range("2023-01-06T21:00:00Z", periods=len(states), freq="7D")
    membership = pd.DataFrame({"date": dates, "state": states})
    as_of = dates[-1].tz_convert("America/New_York").date().isoformat()
    duration = audit_outputs._recompute_v5_duration(states, as_of=as_of)
    duration.update(
        {
            "bootstrap": {
                "unit": "episode",
                "resamples": 7,
                "valid_resamples": 0,
                "seed": 17,
                "interval": 0.95,
            },
            "ci95": {
                "conditional_survival": {"4w": None, "13w": None},
                "departure_probability": {"4w": None, "13w": None},
                "median_remaining_weeks": None,
                "restricted_mean_remaining_weeks": None,
            },
        }
    )
    execution = _v5_execution_parameters()
    payload = {
        "weekly": [
            {
                "date": as_of,
                "data_as_of": dates[-1].isoformat(),
                "current": {"state": states[-1]},
                "duration_context": duration,
            }
        ]
    }
    return payload, membership, execution


def test_v5_directional_audit_recomputes_purge_and_simplex() -> None:
    payload, frames = _v5_directional_audit_fixture()

    summary = audit_outputs._audit_v5_directional(payload, frames)

    assert summary["oos_rows"] == 18
    assert summary["champions"]["13w"] == "empirical_first_passage"


def test_v5_directional_audit_rejects_origin_state_mass() -> None:
    payload, frames = _v5_directional_audit_fixture()
    frames["directional-oos-predictions.csv"].loc[0, "p_risk_on"] = 0.01
    frames["directional-oos-predictions.csv"].loc[0, "p_no_departure"] = 0.69

    with pytest.raises(audit_outputs.AuditFailure, match="origin state"):
        audit_outputs._audit_v5_directional(payload, frames)


def test_v5_directional_audit_links_predictions_to_split_origins() -> None:
    payload, frames = _v5_directional_audit_fixture()
    splits = frames["directional-walk-forward-splits.csv"]
    splits.loc[0, "origin_date"] += timedelta(days=1)
    splits.loc[0, "target_end"] += timedelta(days=1)

    with pytest.raises(audit_outputs.AuditFailure, match="predictions/splits"):
        audit_outputs._audit_v5_directional(payload, frames)


def test_v5_directional_audit_recomputes_supported_gate_and_holm() -> None:
    payload, frames = _v5_supported_directional_audit_fixture()

    summary = audit_outputs._audit_v5_directional(payload, frames)

    assert summary["champions"] == {
        "1w": "regularized_multinomial",
        "4w": "regularized_multinomial",
        "13w": "regularized_multinomial",
    }
    diagnostics = frames["directional-selection-diagnostics.csv"]
    candidate = diagnostics.loc[
        diagnostics["model"].eq("regularized_multinomial")
    ]
    assert candidate["gate_passed"].all()
    assert candidate["holm_adjusted_p_value"].notna().all()
    assert set(candidate["selection_event_count"]) == {8}
    assert set(candidate["selection_destination_class_count"]) == {2}
    assert set(candidate["selection_effective_event_blocks"]) == {3}


@pytest.mark.parametrize(("field", "delta"), [("event_count", 1), ("log_loss", 0.1)])
def test_v5_directional_audit_rejects_rehashed_leaderboard_tamper(
    tmp_path: Path,
    field: str,
    delta: float,
) -> None:
    payload, frames = _v5_supported_directional_audit_fixture()
    leaderboard = frames["directional-model-leaderboard.csv"]
    mask = (
        leaderboard["horizon_weeks"].eq(1)
        & leaderboard["evaluation_split"].eq("selection")
        & leaderboard["model"].eq("empirical_first_passage")
    )
    leaderboard.loc[mask, field] += delta
    for row in payload["model"]["directional_transition"]["leaderboard"]:
        if (
            row["horizon_weeks"] == 1
            and row["evaluation_split"] == "selection"
            and row["model"] == "empirical_first_passage"
        ):
            row[field] += delta

    _, loaded = _write_v5_contract_frames(tmp_path, frames)

    audit_outputs._audit_v5_embedded_records(
        payload["model"]["directional_transition"]["leaderboard"],
        loaded["directional-model-leaderboard.csv"],
        keys=("horizon_weeks", "evaluation_split", "model"),
        context="mutated leaderboard parity",
    )
    with pytest.raises(audit_outputs.AuditFailure, match=field):
        audit_outputs._audit_v5_directional(payload, loaded)


def test_v5_directional_audit_rejects_rehashed_champion_and_forecast_tamper(
    tmp_path: Path,
) -> None:
    payload, frames = _v5_supported_directional_audit_fixture()
    forged = "weak_regularized_multinomial"
    payload["model"]["directional_transition"]["champions"]["1w"] = forged
    forecasts = frames["directional-forecasts.csv"]
    forecasts.loc[forecasts["horizon_weeks"].eq(1), "model"] = forged

    _, loaded = _write_v5_contract_frames(tmp_path, frames)

    with pytest.raises(audit_outputs.AuditFailure, match="champion disagrees"):
        audit_outputs._audit_v5_directional(payload, loaded)


def test_v5_directional_audit_rejects_rehashed_embedded_and_sidecar_holm_tamper(
    tmp_path: Path,
) -> None:
    payload, frames = _v5_supported_directional_audit_fixture()
    diagnostics = frames["directional-selection-diagnostics.csv"]
    mask = diagnostics["model"].eq("regularized_multinomial")
    diagnostics.loc[mask, "holm_adjusted_p_value"] += 0.10
    for row in payload["model"]["directional_transition"][
        "selection_diagnostics"
    ]:
        if row["model"] == "regularized_multinomial":
            row["holm_adjusted_p_value"] += 0.10

    contracts, loaded = _write_v5_contract_frames(tmp_path, frames)

    assert contracts["directional_selection_diagnostics"]["sha256"] == hashlib.sha256(
        (tmp_path / "directional-selection-diagnostics.csv").read_bytes()
    ).hexdigest()
    audit_outputs._audit_v5_embedded_records(
        payload["model"]["directional_transition"]["selection_diagnostics"],
        loaded["directional-selection-diagnostics.csv"],
        keys=("horizon_weeks", "model"),
        context="mutated directional parity",
    )
    with pytest.raises(audit_outputs.AuditFailure, match="holm_adjusted_p_value"):
        audit_outputs._audit_v5_directional(payload, loaded)


def _v5_weekly_directional_binding_fixture() -> tuple[
    dict,
    dict[str, pd.DataFrame],
    pd.DataFrame,
]:
    payload, frames = _v5_directional_audit_fixture()
    origin = pd.Timestamp("2026-07-03T20:00:00Z")
    payload["weekly"] = [
        {
            "date": origin.date().isoformat(),
            "data_as_of": origin.isoformat(),
            "current": {"state": "risk_on"},
            "directional_risk": {
                f"{horizon}w": {
                    "probability": 0.3,
                    "no_departure": 0.7,
                    "first_destination": {
                        "risk_on": 0.0,
                        "transition": 0.2,
                        "risk_off": 0.1,
                    },
                    "target_end": (
                        origin + timedelta(weeks=horizon)
                    ).date().isoformat(),
                    "model": "empirical_first_passage",
                    "method": "first_departure_state_within_h_or_no_departure",
                }
                for horizon in (1, 4, 13)
            },
        }
    ]
    membership = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-06-19T20:00:00Z", periods=3, freq="7D"
            ),
            "state": ["transition", "risk_on", "risk_on"],
        }
    )
    return payload, frames, membership


def test_v5_weekly_directional_audit_binds_sidecar_and_champion() -> None:
    payload, frames, membership = _v5_weekly_directional_binding_fixture()

    summary = audit_outputs._audit_v5_weekly_directional(
        payload,
        frames,
        membership,
    )

    assert summary == {"matched_rows": 3, "fallback_rows": 0}


@pytest.mark.parametrize("field", ["model", "first_destination"])
def test_v5_weekly_directional_audit_rejects_published_tamper(field: str) -> None:
    payload, frames, membership = _v5_weekly_directional_binding_fixture()
    row = payload["weekly"][0]["directional_risk"]["4w"]
    if field == "model":
        row["model"] = "forged_model"
    else:
        row["first_destination"]["transition"] = 0.15
        row["first_destination"]["risk_off"] = 0.15

    with pytest.raises(audit_outputs.AuditFailure, match="source mismatch"):
        audit_outputs._audit_v5_weekly_directional(payload, frames, membership)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("n", 21),
        ("non_overlapping_n", 999),
        ("unique_episodes", 6),
        ("status", "insufficient_support"),
        ("mean_return", 0.987654321),
        ("unconditional_benchmark_mean_return", 0.7654321),
        ("episode_equal_mean_return", 0.654321),
        ("episode_equal_mean_return_ci95_lower", 0.54321),
        ("mean_return_ci95_lower", 0.87654321),
        ("bootstrap_seed", 999),
        ("bootstrap_resamples", 8),
    ],
)
def test_v5_conditional_audit_rejects_rehashed_embedded_and_sidecar_tamper(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    payload, frames = _v5_conditional_audit_fixture()
    statistics = frames["conditional-asset-statistics.csv"]
    statistics.loc[0, field] = replacement
    payload["research"]["conditional_asset_stats"]["rows"][0][field] = replacement

    contracts, loaded = _write_v5_contract_frames(tmp_path, frames)

    assert contracts["conditional_asset_statistics"]["sha256"] == hashlib.sha256(
        (tmp_path / "conditional-asset-statistics.csv").read_bytes()
    ).hexdigest()
    audit_outputs._audit_v5_embedded_records(
        payload["research"]["conditional_asset_stats"]["rows"],
        loaded["conditional-asset-statistics.csv"],
        keys=("state", "asset", "horizon_weeks"),
        context="mutated conditional parity",
    )
    with pytest.raises(audit_outputs.AuditFailure, match=field):
        audit_outputs._audit_v5_conditional(payload, loaded)


def test_v5_conditional_audit_recomputes_all_point_and_interval_fields() -> None:
    payload, frames = _v5_conditional_audit_fixture()

    summary = audit_outputs._audit_v5_conditional(payload, frames)

    assert summary == {
        "outcome_rows": 60,
        "statistics_rows": 3,
        "bootstrap_resamples": 7,
    }


def test_v5_decision_shadow_audit_binds_v2_spec_and_late_no_trade() -> None:
    index = pd.date_range("2022-01-07", periods=90, freq="W-FRI", tz="UTC")
    position = np.arange(len(index), dtype=float)
    prices = pd.DataFrame(
        {
            "spy_close": 100.0 * np.exp(np.cumsum(0.002 + position / 100_000)),
            "tlt_close": 100.0 * np.exp(np.cumsum(0.001 - position / 200_000)),
        },
        index=index,
    )
    prices["spy_adjusted_open"] = prices["spy_close"] * 0.998
    prices["tlt_adjusted_open"] = prices["tlt_close"] * 0.999
    for asset in ("spy", "tlt"):
        prices[f"{asset}_raw_open"] = prices[f"{asset}_adjusted_open"]
        prices[f"{asset}_raw_close"] = prices[f"{asset}_close"]
        prices[f"{asset}_dividend_amount"] = 0.0
    forecast_model = "causal_dynamic_ensemble"
    weekly = [
        {
            "date": origin.date().isoformat(),
            "next_week": {
                "date": (origin + timedelta(weeks=1)).date().isoformat(),
                "probabilities": {
                    "risk_on": 0.6,
                    "transition": 0.25,
                    "risk_off": 0.15,
                }
            },
            "model_forecasts": [
                {
                    "model": forecast_model,
                    "date": (origin + timedelta(weeks=1)).date().isoformat(),
                    "probabilities": {
                        "risk_on": 0.6,
                        "transition": 0.25,
                        "risk_off": 0.15,
                    },
                }
            ],
        }
        for origin in index[10:-1]
    ]
    target_week = pd.Timestamp(weekly[-1]["next_week"]["date"]).date()
    target_monday = target_week - timedelta(days=4)
    decision_at = pd.Timestamp(
        f"{target_monday.isoformat()} 09:31:00",
        tz="America/New_York",
    )
    payload = {
        "meta": {"generated_at": decision_at.isoformat()},
        "selection": {"operating_champion": forecast_model},
        "weekly": weekly,
        "forecast": {
            "timing_status": "full_horizon_forecast",
            "origin_at": pd.Timestamp(
                f"{weekly[-1]['date']} 16:00:00",
                tz="America/New_York",
            ).isoformat(),
            "decision_at": decision_at.isoformat(),
            "target_at": pd.Timestamp(
                f"{weekly[-1]['next_week']['date']} 16:00:00",
                tz="America/New_York",
            ).isoformat(),
        },
        "research": {
            "prospective_decision_shadow": build_decision_shadow(
                weekly,
                prices,
                forecast_model=forecast_model,
                decision_at=decision_at,
            )
        },
    }

    summary = audit_outputs._audit_v5_decision_shadow(payload)

    assert summary["status"] == "verified"
    assert summary["spec_id"] == "spy-tlt-probability-shadow-v2"
    assert summary["current_signal_action"] == "no_trade"

    populated_shadow = deepcopy(
        payload["research"]["prospective_decision_shadow"]
    )
    historical = payload["research"]["prospective_decision_shadow"][
        "historical_reconstructed_shadow"
    ]
    historical["status"] = "insufficient_history"
    historical["evaluation_start_week"] = None
    historical["evaluation_end_week"] = None
    for metrics in historical["strategies"].values():
        for field in (
            "cumulative_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe",
            "certainty_equivalent_return",
            "maximum_drawdown",
            "annualized_turnover",
            "gross_cumulative_return",
        ):
            metrics[field] = None
        metrics["weeks"] = 0
        metrics["transaction_cost_rate_sum"] = 0.0
    assert audit_outputs._audit_v5_decision_shadow(payload)["weeks"] == 0

    historical["strategies"]["probability_shadow"]["annualized_turnover"] = 0.0
    with pytest.raises(audit_outputs.AuditFailure, match="empty turnover/cost"):
        audit_outputs._audit_v5_decision_shadow(payload)

    payload["research"]["prospective_decision_shadow"] = populated_shadow
    payload["research"]["prospective_decision_shadow"]["spec"]["sha256"] = "0" * 64
    with pytest.raises(audit_outputs.AuditFailure, match="spec SHA-256"):
        audit_outputs._audit_v5_decision_shadow(payload)


def test_v5_allocation_candidate_audit_binds_spec_and_official_forecast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        audit_outputs,
        "_validate_allocation_candidate",
        lambda candidate, *, context: None,
    )
    local_spec = json.loads(
        (ROOT / "config" / "allocation-shadow-v1.json").read_text(
            encoding="utf-8"
        )
    )
    probabilities = {
        "risk_on": 0.2,
        "transition": 0.7,
        "risk_off": 0.1,
    }
    candidate = {
        "policy_status": "baseline_preferred",
        "recommended_target": "realistic_60_40",
        "spec": {
            "path": "config/allocation-shadow-v1.json",
            "spec_id": local_spec["spec_id"],
            "sha256": audit_outputs.canonical_json_sha256(local_spec),
        },
        "current_intent": {
            "forecast": {
                "origin_date": "2026-08-21",
                "target_week": "2026-08-28",
                "model": "causal_dynamic_ensemble",
                "probabilities": probabilities,
            },
            "recommended": {"policy": "realistic_60_40"},
            "target": {
                "weights": {"SPY": 0.6, "TLT": 0.4},
                "cash": 0.0,
            },
        },
    }
    payload = {
        "selection": {"operating_champion": "causal_dynamic_ensemble"},
        "weekly": [
            {
                "date": "2026-08-21",
                "model_forecasts": [
                    {
                        "model": "causal_dynamic_ensemble",
                        "date": "2026-08-28",
                        "probabilities": probabilities,
                    }
                ],
            }
        ],
    }

    assert audit_outputs._audit_v5_allocation_candidate(payload, candidate) == {
        "status": "verified",
        "policy_status": "baseline_preferred",
        "recommended_target": "realistic_60_40",
        "forecast_model": "causal_dynamic_ensemble",
    }

    broken = deepcopy(candidate)
    broken["current_intent"]["forecast"]["model"] = "markov"
    with pytest.raises(
        audit_outputs.AuditFailure,
        match="differs from the official forecast",
    ):
        audit_outputs._audit_v5_allocation_candidate(payload, broken)


def test_current_live_decision_shadow_spec_remains_auditable() -> None:
    payload = json.loads(
        (ROOT / "publication" / "live" / "regime-results.json").read_text(
            encoding="utf-8"
        )
    )

    summary = audit_outputs._audit_v5_decision_shadow(payload)

    assert summary["status"] == "verified"
    assert summary["spec_id"] == "spy-tlt-probability-shadow-v2"


def test_v5_research_replay_input_separates_research_and_operational_hashes(
    tmp_path: Path,
) -> None:
    payload = {
        "meta": {
            "mode": "live",
            "data_as_of": "2026-08-21T20:00:00+00:00",
        },
        "weekly": [{}] * 190,
        "research": {
            "prospective_decision_shadow": {
                "schema_version": "regime-prospective-decision-shadow/2"
            }
        },
    }
    document = {
        "schema_version": "regime-research-replay-input/1",
        "evidence_track": "reconstructed_oos",
        "data_as_of": "2026-08-21T20:00:00+00:00",
        "availability_basis": "reconstructed_market",
        "source_observation_count": 2_801_234,
        "input_vintages": {"count": 179_461, "sha256": "a" * 64},
        "canonical_panel": {
            "start": "2006-01-06",
            "end": "2026-08-21",
            "rows": 1_077,
            "columns": 288,
            "sha256": "b" * 64,
        },
        "state_membership": {"rows": 1_077, "sha256": "c" * 64},
        "operational_generation_input_snapshot_sha256": "d" * 64,
    }
    (tmp_path / "research-replay-input.json").write_text(
        json.dumps(document),
        encoding="utf-8",
    )

    summary = audit_outputs._audit_v5_research_replay_input(payload, tmp_path)

    assert summary["status"] == "verified"
    assert summary["input_vintage_sha256"] == "a" * 64
    assert summary["operational_generation_input_snapshot_sha256"] == "d" * 64


def test_v5_research_replay_input_is_required_for_decision_shadow_v2(
    tmp_path: Path,
) -> None:
    payload = {
        "meta": {
            "mode": "live",
            "data_as_of": "2026-08-21T20:00:00+00:00",
        },
        "weekly": [{}],
        "research": {
            "prospective_decision_shadow": {
                "schema_version": "regime-prospective-decision-shadow/2"
            }
        },
    }

    with pytest.raises(audit_outputs.AuditFailure, match="missing research-replay"):
        audit_outputs._audit_v5_research_replay_input(payload, tmp_path)


def test_v5_research_artifact_contract_keeps_legacy_and_optional_order() -> None:
    legacy = {key: {} for key, _ in audit_outputs.V5_RESEARCH_ARTIFACTS}
    assert audit_outputs._v5_expected_research_artifacts(legacy) == (
        audit_outputs.V5_RESEARCH_ARTIFACTS
    )

    with_optional = {
        key: {}
        for key, _ in (
            *audit_outputs.V5_RESEARCH_ARTIFACTS,
            *audit_outputs.V5_MODEL_CONDITIONED_ARTIFACTS,
            *audit_outputs.V5_FX_ARTIFACTS,
        )
    }
    assert audit_outputs._v5_expected_research_artifacts(with_optional) == (
        *audit_outputs.V5_RESEARCH_ARTIFACTS,
        *audit_outputs.V5_MODEL_CONDITIONED_ARTIFACTS,
        *audit_outputs.V5_FX_ARTIFACTS,
    )


def test_v5_research_artifact_contract_rejects_partial_or_reordered_pair() -> None:
    partial = {
        key: {}
        for key, _ in (
            *audit_outputs.V5_RESEARCH_ARTIFACTS,
            audit_outputs.V5_MODEL_CONDITIONED_ARTIFACTS[0],
        )
    }
    with pytest.raises(audit_outputs.AuditFailure, match="complete pair"):
        audit_outputs._v5_expected_research_artifacts(partial)

    reordered_items = [
        *audit_outputs.V5_RESEARCH_ARTIFACTS[:-1],
        *audit_outputs.V5_MODEL_CONDITIONED_ARTIFACTS,
        audit_outputs.V5_RESEARCH_ARTIFACTS[-1],
    ]
    reordered = {key: {} for key, _ in reordered_items}
    with pytest.raises(audit_outputs.AuditFailure, match="key/order"):
        audit_outputs._v5_expected_research_artifacts(reordered)


def test_v5_model_conditioned_audit_is_backward_compatible_when_absent() -> None:
    summary = audit_outputs._audit_v5_model_conditioned(
        {"research": {}},
        {},
    )

    assert summary == {
        "status": "legacy_absent",
        "models": 0,
        "outcome_rows": 0,
        "statistics_rows": 0,
    }


def test_v5_model_conditioned_audit_binds_forecasts_and_recomputes_stats() -> None:
    payload, frames = _v5_model_conditioned_audit_fixture()

    summary = audit_outputs._audit_v5_model_conditioned(payload, frames)

    assert summary == {
        "status": "verified",
        "models": 2,
        "outcome_rows": len(frames["model-conditioned-asset-outcomes.csv"]),
        "statistics_rows": 108,
        "bootstrap_resamples": 7,
    }


def test_v5_model_conditioned_audit_rejects_timeline_tamper() -> None:
    payload, frames = _v5_model_conditioned_audit_fixture()
    outcomes = frames["model-conditioned-asset-outcomes.csv"]
    outcomes.loc[0, "entry_date"] += timedelta(weeks=1)

    with pytest.raises(audit_outputs.AuditFailure, match=r"entry is not t\+1"):
        audit_outputs._audit_v5_model_conditioned(payload, frames)


def test_v5_model_conditioned_audit_rejects_forecast_state_tamper() -> None:
    payload, frames = _v5_model_conditioned_audit_fixture()
    outcomes = frames["model-conditioned-asset-outcomes.csv"]
    original = str(outcomes.loc[0, "state"])
    outcomes.loc[0, "state"] = next(
        state for state in STATE_ORDER if state != original
    )

    with pytest.raises(audit_outputs.AuditFailure, match="weekly OOS forecast"):
        audit_outputs._audit_v5_model_conditioned(payload, frames)


def test_v5_model_conditioned_audit_rejects_rehashed_statistics_tamper() -> None:
    payload, frames = _v5_model_conditioned_audit_fixture()
    statistics = frames["model-conditioned-asset-statistics.csv"]
    statistics.loc[0, "n"] = int(statistics.loc[0, "n"]) + 1
    payload["research"]["model_conditioned_asset_stats"]["rows"][0]["n"] = int(
        statistics.loc[0, "n"]
    )

    with pytest.raises(audit_outputs.AuditFailure, match=" n mismatch"):
        audit_outputs._audit_v5_model_conditioned(payload, frames)


def test_v5_duration_audit_recomputes_corrected_d_minus_one_km() -> None:
    payload, membership, execution = _v5_duration_audit_fixture()

    summary = audit_outputs._audit_v5_duration(payload, membership, execution)

    duration = payload["weekly"][0]["duration_context"]
    assert summary == {"weeks": 1, "latest_bootstrap_resamples": 7}
    assert duration["status"] == "ok"
    assert duration["completed_spells"] == 5
    assert duration["elapsed_weeks"] == 3


def test_v5_duration_audit_rejects_tampered_point_estimate() -> None:
    payload, membership, execution = _v5_duration_audit_fixture()
    duration = payload["weekly"][0]["duration_context"]
    duration["conditional_survival"]["4w"] = 0.75
    duration["departure_probability"]["4w"] = 0.25

    with pytest.raises(audit_outputs.AuditFailure, match="conditional_survival.4w"):
        audit_outputs._audit_v5_duration(payload, membership, execution)


def test_v5_duration_audit_links_latest_resamples_to_execution_parameters() -> None:
    payload, membership, execution = _v5_duration_audit_fixture()
    duration = payload["weekly"][0]["duration_context"]
    duration["bootstrap"]["resamples"] = 8

    with pytest.raises(audit_outputs.AuditFailure, match="resamples/execution"):
        audit_outputs._audit_v5_duration(payload, membership, execution)


def test_v5_duration_audit_rejects_ci_without_enough_valid_resamples() -> None:
    payload, membership, execution = _v5_duration_audit_fixture()
    duration = payload["weekly"][0]["duration_context"]
    duration["bootstrap"]["valid_resamples"] = 7

    with pytest.raises(audit_outputs.AuditFailure, match="valid-resample linkage"):
        audit_outputs._audit_v5_duration(payload, membership, execution)


def test_v5_duration_audit_honors_exact_weekly_as_of_timestamp() -> None:
    payload, membership, execution = _v5_duration_audit_fixture()
    latest = pd.to_datetime(membership.iloc[-1]["date"], utc=True)
    payload["weekly"][0]["data_as_of"] = (latest - timedelta(hours=1)).isoformat()

    with pytest.raises(audit_outputs.AuditFailure, match="as-of evidence"):
        audit_outputs._audit_v5_duration(payload, membership, execution)


def _v5_fx_audit_fixture(
    rows: int,
) -> tuple[dict, dict[str, pd.DataFrame], pd.DataFrame]:
    from regime_lab.analysis.fx import FXFeatureResult
    from regime_lab.analysis.fx_ablation import (
        FX_ABLATION_OOS_COLUMNS,
        run_fx_shadow_ablation,
    )

    observation_weeks = pd.date_range(
        "2020-01-03", periods=rows, freq="W-FRI"
    )
    cutoffs = (
        (observation_weeks + pd.offsets.Hour(16))
        .tz_localize("America/New_York")
        .tz_convert("UTC")
    )
    position = np.arange(rows, dtype=float)
    latent = np.sin(position / 5.0) + 0.35 * np.cos(position / 11.0)
    states = pd.Series(
        np.select(
            [latent > 0.35, latent < -0.35],
            ["risk_on", "risk_off"],
            default="transition",
        ),
        index=cutoffs,
        dtype="object",
    )
    core = pd.DataFrame(
        {
            "core__trend": np.roll(latent, 2),
            "core__stress": np.cos(position / 7.0),
        },
        index=cutoffs,
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
        index=observation_weeks,
        dtype=float,
    )
    availability = (
        observation_weeks
        + pd.offsets.Day(3)
        + pd.offsets.Hour(16)
        + pd.offsets.Minute(15)
    ).tz_localize("America/New_York").tz_convert("UTC")
    coverage = pd.DataFrame(
        {
            "feature_available_at": availability,
            "bilateral_level_count": 9,
            "core_level_count": 3,
            "archive_correction_quarantined": False,
            "archive_correction_available_at": pd.NaT,
            "archive_correction_quarantine_until_week": pd.NaT,
        },
        index=observation_weeks,
    )
    empty = pd.DataFrame(index=observation_weeks)
    result = FXFeatureResult(
        features=features,
        weekly_usd_log_levels=empty,
        weekly_availability=empty,
        coverage=coverage,
        status=empty,
    )
    evidence: list[pd.DataFrame] = []
    ablation = run_fx_shadow_ablation(
        core,
        states,
        result,
        cutoffs,
        bootstrap_resamples=19,
        evidence_sink=evidence.append,
    )
    ablation_oos = (
        evidence[0]
        if evidence
        else pd.DataFrame(columns=FX_ABLATION_OOS_COLUMNS)
    )
    feature_sidecar = features.reset_index(names="observation_week")
    coverage_sidecar = coverage.reset_index(names="observation_week")
    membership = pd.DataFrame({"date": cutoffs, "state": states.to_numpy()})
    return (
        {"model": {"fx_ablation": ablation}},
        {
            "fx-features.csv": feature_sidecar,
            "fx-coverage.csv": coverage_sidecar,
            "fx-ablation-oos.csv": ablation_oos,
        },
        membership,
    )


@pytest.mark.parametrize(
    ("rows", "expected_status"),
    [(156, "insufficient_history"), (157, "evaluated")],
)
def test_v5_fx_audit_accepts_complete_readiness_contract(
    rows: int,
    expected_status: str,
) -> None:
    payload, frames, membership = _v5_fx_audit_fixture(rows)

    summary = audit_outputs._audit_v5_fx(payload, frames, membership)

    assert summary["status"] == expected_status
    assert summary["eligible_common_weeks"] == rows - 1


def test_v5_fx_audit_binds_model_and_gate_to_preregistration() -> None:
    payload, _, _ = _v5_fx_audit_fixture(156)
    payload["meta"] = {"mode": "demo"}
    payload["model"]["fx_ablation"]["gate"]["bootstrap_resamples"] = 1_999
    preregistration = json.loads(
        (ROOT / "config" / "structural_v5.json").read_text(encoding="utf-8")
    )

    audit_outputs._audit_v5_fx_provenance(payload, preregistration)

    payload["model"]["fx_ablation"]["gate"]["alpha"] = 0.10
    with pytest.raises(audit_outputs.AuditFailure, match="preregistration"):
        audit_outputs._audit_v5_fx_provenance(payload, preregistration)


def test_v5_fx_audit_rejects_self_consistent_forged_common_origin() -> None:
    payload, frames, membership = _v5_fx_audit_fixture(157)
    origins = payload["model"]["fx_ablation"]["common_evaluation_origins"]
    origins["rows"][0]["train_size"] += 1

    with pytest.raises(
        audit_outputs.AuditFailure,
        match="common evaluation origins",
    ):
        audit_outputs._audit_v5_fx(payload, frames, membership)


def test_v5_fx_audit_rejects_forged_holm_adjustment() -> None:
    payload, frames, membership = _v5_fx_audit_fixture(157)
    comparison = payload["model"]["fx_ablation"]["gate"]["comparisons"][0]
    comparison["holm_adjusted_p_value"] += 0.001

    with pytest.raises(audit_outputs.AuditFailure, match="Holm adjustment"):
        audit_outputs._audit_v5_fx(payload, frames, membership)


def test_v5_fx_audit_rejects_oos_probability_forgery() -> None:
    payload, frames, membership = _v5_fx_audit_fixture(157)
    evidence = frames["fx-ablation-oos.csv"]
    evidence.loc[0, "p_risk_on"] += 0.01
    evidence.loc[0, "p_transition"] -= 0.01

    with pytest.raises(audit_outputs.AuditFailure, match="metrics disagree"):
        audit_outputs._audit_v5_fx(payload, frames, membership)


def test_v5_fx_audit_rejects_oos_actual_forgery() -> None:
    payload, frames, membership = _v5_fx_audit_fixture(157)
    evidence = frames["fx-ablation-oos.csv"]
    evidence.loc[0, "actual"] = (
        "risk_off" if evidence.loc[0, "actual"] != "risk_off" else "risk_on"
    )

    with pytest.raises(audit_outputs.AuditFailure, match="membership evidence"):
        audit_outputs._audit_v5_fx(payload, frames, membership)


def test_v5_fx_audit_rejects_oos_purge_or_common_hash_forgery() -> None:
    payload, frames, membership = _v5_fx_audit_fixture(157)
    evidence = frames["fx-ablation-oos.csv"]
    evidence.loc[0, "last_train_target"] = evidence.loc[0, "origin_date"]

    with pytest.raises(audit_outputs.AuditFailure, match="training target"):
        audit_outputs._audit_v5_fx(payload, frames, membership)

    payload, frames, membership = _v5_fx_audit_fixture(157)
    frames["fx-ablation-oos.csv"]["common_origins_sha256"] = "0" * 64
    with pytest.raises(audit_outputs.AuditFailure, match="common-origin hash"):
        audit_outputs._audit_v5_fx(payload, frames, membership)


def test_v5_fx_audit_rejects_forged_raw_bootstrap_p_value() -> None:
    payload, frames, membership = _v5_fx_audit_fixture(157)
    comparison = payload["model"]["fx_ablation"]["gate"]["comparisons"][0]
    comparison["raw_p_value"] += 0.001

    with pytest.raises(audit_outputs.AuditFailure, match="raw bootstrap"):
        audit_outputs._audit_v5_fx(payload, frames, membership)


def test_v5_fx_audit_binds_variant_counts_to_core_feature_manifest() -> None:
    payload, frames, membership = _v5_fx_audit_fixture(157)

    with pytest.raises(audit_outputs.AuditFailure, match="feature manifest"):
        audit_outputs._audit_v5_fx(
            payload,
            frames,
            membership,
            core_feature_count=3,
        )


def _v5_fx_quarantine_fixture() -> tuple[pd.DataFrame, pd.Timestamp]:
    weeks = pd.date_range("2023-01-06", periods=35, freq="W-FRI")
    correction = pd.Timestamp("2023-01-09T05:00:00Z")
    quarantined = pd.Series(False, index=weeks)
    quarantined.iloc[:27] = True
    available = pd.Series(
        pd.NaT,
        index=weeks,
        dtype="datetime64[ns, UTC]",
    )
    available.loc[quarantined] = correction
    until = pd.Series(pd.NaT, index=weeks, dtype="datetime64[ns]")
    until.loc[quarantined] = pd.Timestamp("2023-07-14")
    coverage = pd.DataFrame(
        {
            "archive_correction_quarantined": quarantined,
            "archive_correction_available_at": available,
            "archive_correction_quarantine_until_week": until,
            "feature_status": np.where(
                quarantined,
                "correction_quarantine",
                "ok",
            ),
        },
        index=weeks,
    )
    return coverage, correction


def test_v5_fx_audit_rebuilds_exact_27_origin_correction_quarantine() -> None:
    coverage, correction = _v5_fx_quarantine_fixture()

    summary = audit_outputs._audit_v5_fx_correction_quarantine(
        coverage,
        correction_events=[correction],
    )

    assert summary == {
        "correction_events": 1,
        "visible_correction_events": 1,
        "quarantined_weeks": 27,
        "first_quarantined_cutoff": "2023-01-13",
        "last_quarantined_cutoff": "2023-07-14",
    }


def test_v5_fx_audit_rejects_gap_in_correction_quarantine_union() -> None:
    coverage, correction = _v5_fx_quarantine_fixture()
    week = coverage.index[10]
    coverage.loc[week, "archive_correction_quarantined"] = False
    coverage.loc[week, "archive_correction_available_at"] = pd.NaT
    coverage.loc[week, "archive_correction_quarantine_until_week"] = pd.NaT
    coverage.loc[week, "feature_status"] = "ok"

    with pytest.raises(audit_outputs.AuditFailure, match="mask disagrees"):
        audit_outputs._audit_v5_fx_correction_quarantine(
            coverage,
            correction_events=[correction],
        )


def test_v5_fx_audit_rejects_non_27_week_correction_end() -> None:
    coverage, correction = _v5_fx_quarantine_fixture()
    coverage.loc[
        coverage.index[0],
        "archive_correction_quarantine_until_week",
    ] = pd.Timestamp("2023-07-21")

    with pytest.raises(audit_outputs.AuditFailure, match="exactly 27 origins"):
        audit_outputs._audit_v5_fx_correction_quarantine(
            coverage,
            correction_events=[correction],
        )


def test_v5_execution_parameters_reject_rehashed_override_tamper() -> None:
    parameters = _v5_execution_parameters()
    parameters["duration_bootstrap_resamples"] = 1_999
    unhashed = {key: value for key, value in parameters.items() if key != "sha256"}
    parameters["sha256"] = audit_outputs.canonical_json_sha256(unhashed)
    payload = {"model": {"profile": "quick", "execution_parameters": parameters}}

    with pytest.raises(audit_outputs.AuditFailure, match="override linkage"):
        audit_outputs._audit_v5_execution_parameters(payload)


def test_v5_execution_parameters_reject_sha_tamper() -> None:
    parameters = _v5_execution_parameters()
    parameters["sha256"] = "0" * 64
    payload = {"model": {"profile": "quick", "execution_parameters": parameters}}

    with pytest.raises(audit_outputs.AuditFailure, match="SHA-256"):
        audit_outputs._audit_v5_execution_parameters(payload)


def test_v5_file_contract_recomputes_hash_and_rows(tmp_path: Path) -> None:
    path = tmp_path / "directional-oos-predictions.csv"
    pd.DataFrame({"value": [1, 2]}).to_csv(path, index=False)
    contract = {
        "directional_oos_predictions": {
            "path": path.name,
            "row_count": 2,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    }

    frames = audit_outputs._audit_v5_file_contracts(
        contract,
        tmp_path,
        context="fixture",
    )

    assert list(frames) == [path.name]
    path.write_text("value\n3\n", encoding="utf-8")
    with pytest.raises(audit_outputs.AuditFailure, match="SHA-256"):
        audit_outputs._audit_v5_file_contracts(
            contract,
            tmp_path,
            context="fixture",
        )


def test_v5_audit_runs_all_inherited_structural_audits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions = pd.DataFrame({"model": ["markov"]})
    payload = {"model": {"champion": "markov"}}
    feature_manifest = {"group_features": {"legacy_v3": ["signal"]}}
    calls: list[str] = []

    def fake_read_csv(path: Path, date_columns: tuple[str, ...]) -> pd.DataFrame:
        assert path == tmp_path / "oos-predictions.csv"
        assert date_columns == ("origin_date", "target_date")
        return predictions

    def fake_transition(
        actual_payload: dict,
        artifacts: Path,
        **kwargs: object,
    ) -> dict[str, bool]:
        calls.append("transition")
        assert actual_payload is payload
        assert artifacts == tmp_path
        assert kwargs == {
            "main_predictions": predictions,
            "main_champion": "markov",
            "main_published_split": "holdout",
        }
        return {"verified": True}

    def fake_ablation(
        actual_payload: dict,
        artifacts: Path,
        **kwargs: object,
    ) -> dict[str, bool]:
        calls.append("feature_ablation")
        assert actual_payload is payload
        assert artifacts == tmp_path
        assert kwargs == {
            "main_predictions": predictions,
            "feature_manifest": feature_manifest,
        }
        return {"verified": True}

    def fake_joint_survival(artifacts: Path) -> dict[str, bool]:
        calls.append("joint_survival")
        assert artifacts == tmp_path
        return {"verified": True}

    monkeypatch.setattr(audit_outputs, "read_csv", fake_read_csv)
    monkeypatch.setattr(
        audit_outputs,
        "audit_transition_outputs",
        fake_transition,
    )
    monkeypatch.setattr(
        audit_outputs,
        "audit_feature_ablation",
        fake_ablation,
    )
    monkeypatch.setattr(
        audit_outputs,
        "audit_joint_survival_forecasts",
        fake_joint_survival,
    )

    result = audit_outputs._audit_v5_inherited_structural_outputs(
        payload,
        tmp_path,
        feature_manifest=feature_manifest,
    )

    assert calls == ["transition", "feature_ablation", "joint_survival"]
    assert result == {
        "transition": {"verified": True},
        "feature_ablation": {"verified": True},
        "joint_survival": {"verified": True},
    }


def test_v5_core_audit_recomputes_champion_and_metrics(tmp_path: Path) -> None:
    payload, artifacts = _self_contained_v4_core_fixture(tmp_path)

    summary = audit_outputs._audit_v5_core_model(payload, artifacts)

    assert summary == {
        "profile": "quick",
        "champion": "markov",
        "models": 16,
        "selection_origins": 3,
        "holdout_origins": 3,
        "fallback_rows": 0,
        "stacking": {
            "origins": 6,
            "experts": ["markov", "xgboost", "xgb_hazard_destination"],
            "rows": 18,
        },
        "multiscale": None,
        "core_artifacts": None,
    }


def test_v5_core_audit_rejects_forged_embedded_leaderboard(
    tmp_path: Path,
) -> None:
    source, artifacts = _self_contained_v4_core_fixture(tmp_path)
    payload = deepcopy(source)
    payload["model"]["leaderboard"][0]["log_loss"] += 0.1

    with pytest.raises(audit_outputs.AuditFailure, match="embedded leaderboard"):
        audit_outputs._audit_v5_core_model(payload, artifacts)


def test_v5_core_audit_rejects_forged_stacking_weight(
    tmp_path: Path,
) -> None:
    payload, artifacts = _self_contained_v4_core_fixture(tmp_path)
    mutated = tmp_path / "mutated"
    mutated.mkdir()
    for name in (
        "candidate-manifest.json",
        "oos-predictions.csv",
        "walk-forward-splits.csv",
        "model-leaderboard.csv",
        "selection-diagnostics.csv",
        "stacking-weights.csv",
    ):
        (mutated / name).write_bytes((artifacts / name).read_bytes())
    stacking = pd.read_csv(mutated / "stacking-weights.csv")
    stacking.loc[0, "weight"] = float(stacking.loc[0, "weight"]) + 0.1
    stacking.to_csv(mutated / "stacking-weights.csv", index=False)

    with pytest.raises(audit_outputs.AuditFailure, match="weight mismatch"):
        audit_outputs._audit_v5_core_model(payload, mutated)


def _v5_model_forecast_record(
    *,
    model: str,
    target: str,
    probability: tuple[float, float, float],
    fallback: bool,
    fallback_reason: str,
) -> dict[str, object]:
    state = audit_outputs.STATE_ORDER[int(np.argmax(probability))]
    probabilities = {
        name: round(float(probability[position]), 8)
        for position, name in enumerate(audit_outputs.STATE_ORDER)
    }
    entropy = -sum(
        value * np.log(value) for value in probabilities.values() if value > 0
    ) / np.log(len(audit_outputs.STATE_ORDER))
    return {
        "state": state,
        "probabilities": probabilities,
        "confidence": round(probabilities[state], 8),
        "entropy": round(float(entropy), 8),
        "date": target,
        "method": audit_outputs.V5_FORECAST_COMPARISON_METHOD,
        "model": model,
        "fallback": fallback,
        "fallback_reason": fallback_reason,
    }


def _v5_model_forecast_audit_fixture(tmp_path: Path) -> dict[str, object]:
    models = audit_outputs.V5_FORECAST_COMPARISON_MODELS
    probability_by_model = {
        "markov": (0.6, 0.3, 0.1),
        "xgboost": (0.2, 0.7, 0.1),
        "xgb_hazard_destination": (0.1, 0.2, 0.7),
        "causal_dynamic_ensemble": (0.5, 0.4, 0.1),
        "causal_multiscale_ensemble": (0.4, 0.5, 0.1),
    }
    source_frames: list[pd.DataFrame] = []
    published_weeks: list[dict[str, object]] = []
    for week_position, (origin, target) in enumerate(
        (
            ("2026-08-14", "2026-08-21"),
            ("2026-08-21", "2026-08-28"),
        )
    ):
        source_rows: list[dict[str, object]] = []
        published_rows: list[dict[str, object]] = []
        for model_position, model in enumerate(models):
            probability = probability_by_model.get(
                model,
                (0.3, 0.4, 0.3),
            )
            fallback = model_position == week_position + 2
            fallback_reason = "fixture_fallback" if fallback else ""
            predicted = audit_outputs.STATE_ORDER[int(np.argmax(probability))]
            source_rows.append(
                {
                    "origin_date": f"{origin}T20:00:00Z",
                    "target_date": f"{target}T20:00:00Z",
                    "model": model,
                    "predicted": predicted,
                    "p_risk_on": probability[0],
                    "p_transition": probability[1],
                    "p_risk_off": probability[2],
                    "fallback": fallback,
                    "fallback_reason": fallback_reason,
                }
            )
            published_rows.append(
                _v5_model_forecast_record(
                    model=model,
                    target=target,
                    probability=probability,
                    fallback=fallback,
                    fallback_reason=fallback_reason,
                )
            )
        source_frames.append(pd.DataFrame(source_rows))
        published_weeks.append(
            {"date": origin, "model_forecasts": published_rows}
        )

    oos_path = tmp_path / "oos-predictions.csv"
    source_frames[0].to_csv(oos_path, index=False)
    source_frames[1].to_csv(tmp_path / "structural-forecasts.csv", index=False)
    return {
        "model": {
            "leaderboard": [{"name": model} for model in models],
            "forecast_comparison": {
                "role": "research_comparison",
                "horizon_weeks": 1,
                "models": list(models),
            },
            "core_artifacts": {
                "oos_predictions": {
                    "path": oos_path.name,
                    "row_count": len(source_frames[0]),
                    "sha256": hashlib.sha256(oos_path.read_bytes()).hexdigest(),
                }
            },
        },
        "weekly": published_weeks,
    }


def test_v5_model_forecast_audit_binds_historical_and_latest_sources(
    tmp_path: Path,
) -> None:
    payload = _v5_model_forecast_audit_fixture(tmp_path)

    summary = audit_outputs._audit_v5_model_forecasts(payload, tmp_path)

    assert summary == {
        "status": "verified",
        "models": len(audit_outputs.V5_FORECAST_COMPARISON_MODELS),
        "weeks": 2,
        "historical_rows": len(audit_outputs.V5_FORECAST_COMPARISON_MODELS),
        "latest_rows": len(audit_outputs.V5_FORECAST_COMPARISON_MODELS),
    }


def test_v5_model_forecast_audit_rejects_absent_contract() -> None:
    payload = {"model": {}, "weekly": [{"date": "2026-08-21"}]}

    with pytest.raises(audit_outputs.AuditFailure, match="metadata is required"):
        audit_outputs._audit_v5_model_forecasts(payload, Path("unused"))


def test_v5_model_forecast_audit_rejects_orphan_rows_without_metadata() -> None:
    payload = {
        "model": {},
        "weekly": [{"date": "2026-08-21", "model_forecasts": []}],
    }

    with pytest.raises(audit_outputs.AuditFailure, match="require"):
        audit_outputs._audit_v5_model_forecasts(payload, Path("unused"))


def test_v5_model_forecast_audit_rejects_model_order_tamper(
    tmp_path: Path,
) -> None:
    payload = _v5_model_forecast_audit_fixture(tmp_path)
    payload["model"]["forecast_comparison"]["models"][0:2] = [
        "xgboost",
        "markov",
    ]

    with pytest.raises(audit_outputs.AuditFailure, match="model order"):
        audit_outputs._audit_v5_model_forecasts(payload, tmp_path)


@pytest.mark.parametrize(
    ("week_position", "model_position", "field", "replacement", "message"),
    (
        (0, 0, "probability", 0.51, "probability/source"),
        (1, 1, "date", "2026-09-04", "target date/source"),
        (1, 3, "fallback_reason", "tampered", "fallback reason/source"),
    ),
)
def test_v5_model_forecast_audit_rejects_source_parity_tamper(
    tmp_path: Path,
    week_position: int,
    model_position: int,
    field: str,
    replacement: object,
    message: str,
) -> None:
    payload = _v5_model_forecast_audit_fixture(tmp_path)
    row = payload["weekly"][week_position]["model_forecasts"][model_position]
    if field == "probability":
        row["probabilities"]["risk_on"] = replacement
    else:
        row[field] = replacement

    with pytest.raises(audit_outputs.AuditFailure, match=message):
        audit_outputs._audit_v5_model_forecasts(payload, tmp_path)


def _v5_multiscale_audit_fixture(
    tmp_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from regime_lab.analysis.structural_models import (
        DEFAULT_ENSEMBLE_EXPERTS,
        PROBABILITY_COLUMNS,
        causal_multiscale_ensemble,
        forecast_structural_probabilities,
    )

    origins = pd.date_range("2021-01-01", periods=36, freq="W-FRI", tz="UTC")
    pattern = np.asarray(
        ["risk_on", "transition", "risk_off", "transition"], dtype=object
    )
    current = np.resize(pattern, len(origins))
    actual = np.roll(current, -1)
    confidence = {
        "markov": 0.58,
        "xgboost": 0.67,
        "xgb_hazard_destination": 0.72,
    }
    expert_rows: list[dict[str, object]] = []
    for position, origin in enumerate(origins):
        for name in DEFAULT_ENSEMBLE_EXPERTS:
            probability = np.full(3, (1.0 - confidence[name]) / 2.0)
            probability[audit_outputs.STATE_ORDER.index(str(actual[position]))] = (
                confidence[name]
            )
            expert_rows.append(
                {
                    "origin_date": origin,
                    "target_date": origin + pd.offsets.Week(1),
                    "model": name,
                    "evaluation_split": (
                        "selection" if position < 30 else "holdout"
                    ),
                    "current_state": str(current[position]),
                    "actual": str(actual[position]),
                    "predicted": str(actual[position]),
                    **{
                        column: float(probability[index])
                        for index, column in enumerate(PROBABILITY_COLUMNS)
                    },
                    "fallback": False,
                    "fallback_reason": "",
                }
            )
    experts = pd.DataFrame(expert_rows)
    multiscale = causal_multiscale_ensemble(experts)
    latest_origin = origins[-1] + pd.offsets.Week(1)
    latest = forecast_structural_probabilities(
        origin_date=latest_origin,
        current_state="transition",
        markov_probability=(0.3, 0.5, 0.2),
        xgboost_probability=(0.25, 0.55, 0.2),
        binary_xgboost_p_change=0.2,
        historical_oos_predictions=experts,
        current_duration_weeks=3,
        include_multiscale=True,
    )
    structural = latest.probabilities.copy()
    structural["target_date"] = latest_origin + pd.offsets.Week(1)
    structural.to_csv(tmp_path / "structural-forecasts.csv", index=False)
    assert latest.multiscale_scale_predictions is not None
    scale_predictions = pd.concat(
        [multiscale.scale_predictions, latest.multiscale_scale_predictions],
        ignore_index=True,
    )
    latest_weights = latest.stacking_weights.loc[
        latest.stacking_weights["ensemble_model"].astype(str).eq(
            "causal_multiscale_ensemble"
        )
    ]
    stacking = pd.concat(
        [multiscale.weights, latest_weights], ignore_index=True, sort=False
    )
    predictions = pd.concat(
        [experts, multiscale.predictions], ignore_index=True, sort=False
    )
    return predictions, stacking, scale_predictions


def test_v5_multiscale_audit_rebuilds_all_scales_and_equal_average(
    tmp_path: Path,
) -> None:
    predictions, stacking, scales = _v5_multiscale_audit_fixture(tmp_path)

    summary = audit_outputs.audit_v5_multiscale_ensemble(
        predictions,
        stacking,
        scales,
        tmp_path,
    )

    assert summary == {
        "oos_origins": 36,
        "latest_origins": 1,
        "scale_rows": 111,
        "stacking_rows": 333,
        "scales": [26, 52, 104],
        "outer_scale_weights": [1.0 / 3.0] * 3,
    }


def test_v5_multiscale_audit_rejects_forged_scale_probability(
    tmp_path: Path,
) -> None:
    predictions, stacking, scales = _v5_multiscale_audit_fixture(tmp_path)
    scales.loc[0, "p_risk_on"] += 0.01
    scales.loc[0, "p_risk_off"] -= 0.01

    with pytest.raises(audit_outputs.AuditFailure, match="scale probability"):
        audit_outputs.audit_v5_multiscale_ensemble(
            predictions,
            stacking,
            scales,
            tmp_path,
        )


def test_audit_dispatches_v5_before_loading_v4_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload_path = tmp_path / "regime-results.json"
    payload_path.write_text(
        json.dumps({"meta": {"result_version": "weekly-regime-result-v5"}}),
        encoding="utf-8",
    )
    expected = {"ok": True, "contract": "v5"}
    monkeypatch.setattr(audit_outputs, "audit_v5", lambda *args: expected)

    assert audit_outputs.audit(payload_path, tmp_path / "missing", "auto") == expected


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

    _, diagnostics = audit_outputs.choose_selection_champion(
        metrics,
        predictions,
        minimum_log_loss_improvement=0.01,
    )

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
        load_config(default_config_path()),
        profile_name="quick",
        contract_version="v4",
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
