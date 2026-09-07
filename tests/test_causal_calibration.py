"""Causal and probability contracts for completed-OOS calibration."""

import numpy as np
import pandas as pd
import pytest

from regime_lab.analysis.causal_calibration import (
    BASELINE_MODEL,
    CANDIDATE_MODELS,
    EXPERT_MODELS,
    PROBABILITY_COLUMNS,
    _prepare,
    _route_departure,
    build_causal_calibration,
)
from regime_lab.analysis.labels import STATE_ORDER


def source_frame(n=112):
    rng = np.random.default_rng(481)
    dates = pd.date_range("2019-01-04", periods=n + 1, freq="7D", tz="UTC")
    states = np.asarray(STATE_ORDER)[(np.arange(n + 1) // 5) % 3]
    rows = []
    for expert_index, expert in enumerate(EXPERT_MODELS):
        for i in range(n):
            current = list(STATE_ORDER).index(states[i])
            probability = rng.uniform(0.05, 0.2, 3)
            probability[current] += 0.5 + 0.05 * expert_index
            probability /= probability.sum()
            rows.append({
                "origin_date": dates[i], "target_date": dates[i + 1],
                "model": expert, "evaluation_split": "selection",
                "current_state": states[i], "actual": states[i + 1],
                "fallback": False,
                **dict(zip(PROBABILITY_COLUMNS, probability, strict=True)),
            })
    return pd.DataFrame(rows)


def test_calibration_uses_only_strictly_completed_targets_and_normalized_probabilities():
    result = build_causal_calibration(source_frame())
    output = result.predictions
    assert len(output) == 112 * len(CANDIDATE_MODELS)
    assert not output.duplicated(["model", "origin_date"]).any()
    trained = output.loc[~output.fallback]
    assert (trained.train_size >= 104).all()
    assert (trained.last_train_target < trained.origin_date).all()
    assert np.allclose(output[list(PROBABILITY_COLUMNS)].sum(axis=1), 1, atol=1e-12)
    assert np.isfinite(output[list(PROBABILITY_COLUMNS)]).all().all()
    first = output.loc[output.model == CANDIDATE_MODELS[0]]
    assert first.iloc[104].fallback
    assert not first.iloc[105].fallback
    assert first.iloc[105].train_size == 104


def test_early_forecasts_do_not_change_after_future_outcomes_are_modified():
    source = source_frame()
    origin = sorted(source.origin_date.unique())[108]
    original = build_causal_calibration(source).predictions
    changed = source.copy()
    # Equal-target rows are deliberately included: the purge must exclude them.
    future = changed.target_date >= origin
    changed.loc[future, "actual"] = changed.loc[future, "actual"].map({
        "risk_on": "risk_off", "transition": "risk_on", "risk_off": "transition",
    })
    other = build_causal_calibration(changed).predictions
    columns = [*PROBABILITY_COLUMNS, "train_size", "last_train_target", "fallback"]
    pd.testing.assert_frame_equal(
        original.loc[original.origin_date <= origin, columns],
        other.loc[other.origin_date <= origin, columns],
        check_exact=True,
    )


def test_warmup_uses_named_baseline_and_full_origin_coverage():
    source = source_frame(n=20)
    output = build_causal_calibration(source).predictions
    assert output.fallback.all()
    baseline = source.loc[source.model == BASELINE_MODEL].sort_values("origin_date")
    for _, part in output.groupby("model"):
        np.testing.assert_array_equal(
            part[list(PROBABILITY_COLUMNS)].to_numpy(),
            baseline[list(PROBABILITY_COLUMNS)].to_numpy(),
        )


def test_candidate_selection_cannot_see_diagnostic_outcomes():
    source = source_frame(n=116)
    for field in ("origin_date", "target_date"):
        shifted_ns = source[field].astype("int64").to_numpy() + 100 * 7 * 86400 * 1_000_000_000
        source[field] = pd.to_datetime(shifted_ns, unit="ns", utc=True)
    source = source.loc[~((source.origin_date.dt.year < 2023) & (source.target_date.dt.year >= 2023))].copy()
    source["evaluation_split"] = np.where(source.target_date.dt.year < 2023, "selection", "holdout")
    original = build_causal_calibration(source)
    assert original.selection
    changed = source.copy()
    diagnostic = changed.evaluation_split == "holdout"
    assert diagnostic.any()
    changed.loc[diagnostic, "actual"] = changed.loc[diagnostic, "actual"].map({
        "risk_on": "risk_off", "transition": "risk_on", "risk_off": "transition",
    })
    other = build_causal_calibration(changed)
    assert original.selection == other.selection
    selection = original.predictions.evaluation_split == "selection"
    pd.testing.assert_frame_equal(original.predictions.loc[selection], other.predictions.loc[selection], check_exact=True)


@pytest.mark.parametrize("alteration", ["duplicate", "missing", "label", "probability", "split"])
def test_invalid_expert_contracts_fail_instead_of_silently_realigning(alteration):
    source = source_frame(n=5)
    if alteration == "duplicate":
        source = pd.concat([source, source.iloc[[0]]], ignore_index=True)
    elif alteration == "missing":
        source = source.iloc[1:]
    elif alteration == "label":
        source.loc[0, "actual"] = "risk_off"
    elif alteration == "probability":
        source.loc[0, PROBABILITY_COLUMNS[0]] = 0.99
    else:
        source.loc[0, "evaluation_split"] = "holdout"
    with pytest.raises(ValueError):
        _prepare(source)


def test_failed_expert_probability_is_replaced_with_available_baseline():
    source = source_frame(n=5)
    failed = (source.model == "xgb_hazard_destination") & (source.origin_date == source.origin_date.min())
    source.loc[failed, "fallback"] = True
    source.loc[failed, list(PROBABILITY_COLUMNS)] = [1 / 3] * 3
    _, probability = _prepare(source)
    np.testing.assert_array_equal(probability[0, -1], probability[0, EXPERT_MODELS.index(BASELINE_MODEL)])


def test_destination_routing_preserves_departure_probability_and_excludes_staying():
    for state in range(3):
        result = _route_departure(0.3, np.array([0.2, 0.3, 0.5]), state)
        assert result[state] == pytest.approx(0.7)
        assert sum(result) == pytest.approx(1)
        assert np.min(result) > 0
