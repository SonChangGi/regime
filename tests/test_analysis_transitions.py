from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from regime_lab.analysis.transitions import (
    DISCRIMINATIVE_PROBABILITY_COLUMNS,
    DurationAwareTVTPHurdleClassifier,
    MarkovDiscriminativeBlendClassifier,
    causal_state_durations,
    derive_causal_transition_features,
)
from regime_lab.schema import STATE_ORDER


def _transition_data(rows: int = 72) -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2021-01-08", periods=rows, freq="W-FRI")
    pattern = np.asarray(
        [
            *("risk_on",) * 6,
            *("transition",) * 3,
            *("risk_off",) * 5,
            *("transition",) * 3,
        ],
        dtype=object,
    )
    states = np.resize(pattern, rows + 1)
    position = np.arange(rows, dtype=float)
    features = pd.DataFrame(
        {
            "current_state": states[:-1],
            "risk_score": np.sin(position / 7.0),
            "breadth": np.cos(position / 5.0),
            "credit": np.where(states[:-1] == "risk_off", 1.0, -0.25),
        },
        index=index,
    )
    features.loc[index[::13], "breadth"] = np.nan
    target = pd.Series(states[1:], index=index, name="next_state", dtype="object")
    return features, target


def _blend_frame(
    current: list[str], probabilities: list[tuple[float, float, float]]
) -> pd.DataFrame:
    frame = pd.DataFrame({"current_state": current})
    for column, values in zip(
        DISCRIMINATIVE_PROBABILITY_COLUMNS,
        np.asarray(probabilities, dtype=float).T,
        strict=True,
    ):
        frame[column] = values
    return frame


def test_causal_duration_and_path_features_do_not_change_when_future_is_appended() -> None:
    features, _ = _transition_data(rows=30)
    threshold_args = {"lower_threshold": -0.5, "upper_threshold": 0.5}
    prefix = derive_causal_transition_features(features.iloc[:20], **threshold_args)
    full = derive_causal_transition_features(features, **threshold_args)

    pd.testing.assert_frame_equal(prefix, full.iloc[:20])
    assert full["state_duration_weeks"].iloc[0] == 1
    assert full["risk_score_delta_1w"].iloc[0] != full["risk_score_delta_1w"].iloc[0]
    assert {
        "risk_score_delta_1w",
        "risk_score_acceleration_1w",
        "risk_score_nearest_boundary",
        "risk_score_state_boundary_distance",
    } < set(full.columns)

    continued = causal_state_durations(
        pd.Series(["risk_on", "risk_on", "transition"]),
        initial_state="risk_on",
        initial_duration=8,
    )
    assert continued.tolist() == [9, 10, 1]


def test_hurdle_probability_contract_and_adjacent_transition_graph() -> None:
    features, target = _transition_data()
    estimator = DurationAwareTVTPHurdleClassifier(
        adjacent_only=True,
        random_state=41,
    ).fit(features.iloc[:-8], target.iloc[:-8])
    forecast = features.iloc[-8:].copy()
    forecast.iloc[0, forecast.columns.get_loc("current_state")] = "risk_on"
    forecast.iloc[1, forecast.columns.get_loc("current_state")] = "risk_off"
    probabilities = estimator.predict_proba(forecast)

    assert tuple(estimator.classes_) == STATE_ORDER
    assert probabilities.shape == (8, 3)
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0.0).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
    assert probabilities[0, STATE_ORDER.index("risk_off")] == 0.0
    assert probabilities[1, STATE_ORDER.index("risk_on")] == 0.0


def test_hurdle_missing_classes_use_stable_surfaced_fallbacks() -> None:
    rows = 16
    features = pd.DataFrame(
        {
            "current_state": ["risk_on"] * rows,
            "signal": np.linspace(-1.0, 1.0, rows),
        }
    )
    target = pd.Series(["risk_on"] * rows)
    estimator = DurationAwareTVTPHurdleClassifier(random_state=7).fit(
        features, target
    )
    probabilities = estimator.predict_proba(
        pd.DataFrame(
            {
                "current_state": ["risk_on", "transition", "risk_off"],
                "signal": [0.2, 0.3, 0.4],
            }
        )
    )

    assert estimator.used_fallback_ is True
    assert set(estimator.missing_target_classes_) == {"transition", "risk_off"}
    assert any("hazard_class_missing" in reason for reason in estimator.fallback_reasons_)
    assert any("destination_fallback" in reason for reason in estimator.fallback_reasons_)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
    assert probabilities[0, STATE_ORDER.index("risk_off")] == 0.0
    assert probabilities[2, STATE_ORDER.index("risk_on")] == 0.0


def test_hurdle_surfaces_observed_direct_jumps_and_routes_them_adjacent() -> None:
    features = pd.DataFrame(
        {
            "current_state": ["risk_on", "risk_on", "risk_off", "risk_off"] * 4,
            "signal": np.linspace(-1.0, 1.0, 16),
        }
    )
    target = pd.Series(
        ["risk_off", "risk_on", "risk_on", "risk_off"] * 4,
        dtype="object",
    )
    estimator = DurationAwareTVTPHurdleClassifier(adjacent_only=True).fit(
        features, target
    )
    probabilities = estimator.predict_proba(features.iloc[:4])

    assert estimator.forbidden_transition_count_ == 8
    assert any(
        reason == "forbidden_transitions_routed_adjacent:8"
        for reason in estimator.fallback_reasons_
    )
    assert estimator.used_fallback_ is True
    assert estimator.fit_diagnostics_["degraded_reasons"] == (
        "target_classes_missing:transition",
        "forbidden_transitions_routed_adjacent:8",
    )
    assert (probabilities[:2, STATE_ORDER.index("risk_off")] == 0.0).all()
    assert (probabilities[2:, STATE_ORDER.index("risk_on")] == 0.0).all()


def test_hurdle_is_cloneable_and_deterministic() -> None:
    features, target = _transition_data()
    prototype = DurationAwareTVTPHurdleClassifier(
        hazard_C=0.2,
        destination_C=0.3,
        random_state=53,
    )
    cloned = clone(prototype)
    assert cloned.get_params() == prototype.get_params()

    outputs: list[np.ndarray] = []
    for _ in range(2):
        estimator = clone(prototype).fit(features.iloc[:-10], target.iloc[:-10])
        outputs.append(estimator.predict_proba(features.iloc[-10:]))
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=0.0, atol=1e-12)


def test_blend_fits_weight_on_calibration_rows_and_emits_fixed_state_order() -> None:
    calibration = _blend_frame(
        ["risk_on", "risk_on", "transition", "transition", "risk_off", "risk_off"],
        [
            (0.90, 0.08, 0.02),
            (0.75, 0.20, 0.05),
            (0.20, 0.70, 0.10),
            (0.10, 0.60, 0.30),
            (0.03, 0.17, 0.80),
            (0.02, 0.08, 0.90),
        ],
    )
    target = pd.Series(
        ["risk_on", "transition", "transition", "risk_off", "risk_off", "risk_off"]
    )
    markov_current = [
        "risk_on",
        "risk_on",
        "risk_on",
        "transition",
        "transition",
        "risk_off",
        "risk_off",
    ]
    markov_next = [
        "risk_on",
        "risk_on",
        "transition",
        "risk_on",
        "risk_off",
        "risk_off",
        "transition",
    ]
    estimator = MarkovDiscriminativeBlendClassifier(weight_grid_size=101).fit(
        calibration,
        target,
        markov_current_states=markov_current,
        markov_next_states=markov_next,
    )
    weight_before_prediction = estimator.markov_weight_
    forecast = _blend_frame(
        ["risk_on", "transition", "risk_off"],
        [(0.8, 0.15, 0.05), (0.25, 0.5, 0.25), (0.05, 0.2, 0.75)],
    )
    probabilities = estimator.predict_proba(forecast)

    assert tuple(estimator.classes_) == STATE_ORDER
    assert 0.0 <= estimator.markov_weight_ <= 1.0
    assert estimator.markov_weight_ == weight_before_prediction
    assert estimator.fit_diagnostics_["markov_training_scope"] == (
        "separate_training_inputs"
    )
    assert list(inspect.signature(estimator.predict_proba).parameters) == ["features"]
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)


def test_blend_missing_classes_is_deterministic_cloneable_and_surfaced() -> None:
    calibration = _blend_frame(
        ["risk_on", "risk_on", "risk_on"],
        [(0.9, 0.08, 0.02), (0.8, 0.15, 0.05), (0.7, 0.2, 0.1)],
    )
    target = pd.Series(["risk_on", "risk_on", "risk_on"])
    prototype = MarkovDiscriminativeBlendClassifier(weight_grid_size=51)
    assert clone(prototype).get_params() == prototype.get_params()

    outputs: list[np.ndarray] = []
    weights: list[float] = []
    for _ in range(2):
        estimator = clone(prototype).fit(calibration, target)
        outputs.append(estimator.predict_proba(calibration))
        weights.append(estimator.markov_weight_)
        assert estimator.used_fallback_ is True
        assert set(estimator.missing_target_classes_) == {"transition", "risk_off"}
        assert set(estimator.unobserved_markov_source_states_) == {
            "transition",
            "risk_off",
        }
    assert weights[0] == weights[1]
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=0.0, atol=1e-12)


def test_blend_rejects_malformed_discriminative_probabilities() -> None:
    frame = _blend_frame(
        ["risk_on", "transition"],
        [(0.8, 0.1, 0.1), (0.2, 0.2, 0.2)],
    )
    with pytest.raises(ValueError, match="sum to one"):
        MarkovDiscriminativeBlendClassifier().fit(
            frame, pd.Series(["risk_on", "risk_off"])
        )
