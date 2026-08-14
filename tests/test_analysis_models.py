from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from regime_lab.analysis.models import MODEL_NAMES, MODEL_REGISTRY
from regime_lab.analysis.models import BenchmarkProfile, StateEncodedXGBoostClassifier
from regime_lab.analysis.models import align_probabilities, augment_with_current_state
from regime_lab.analysis.models import build_model, model_complexity_rank
from regime_lab.analysis.models import model_manifest, model_manifest_sha256
from regime_lab.analysis.models import serialize_model_manifest
from regime_lab.schema import STATE_ORDER


NEW_MODELS = (
    "ridge_logistic",
    "transition_logistic",
    "duration_tvtp_hurdle",
    "shrinkage_lda",
    "spline_logistic",
    "xgboost",
)


def _xgboost_runtime_available() -> bool:
    try:
        from xgboost import XGBClassifier  # noqa: F401
    except Exception:
        return False
    return True


XGBOOST_RUNTIME_AVAILABLE = _xgboost_runtime_available()


def _model_data(rows: int = 120) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(481)
    index = pd.date_range("2018-01-05", periods=rows, freq="W-FRI")
    position = np.arange(rows)
    states = np.resize(np.asarray(STATE_ORDER, dtype=object), rows)
    features = pd.DataFrame(
        {
            "trend": np.sin(position / 6.0) + rng.normal(0.0, 0.05, rows),
            "stress": np.cos(position / 4.0) + rng.normal(0.0, 0.05, rows),
            "macro": (states == "risk_on").astype(float) + rng.normal(0.0, 0.1, rows),
            "credit": (states == "risk_off").astype(float) + rng.normal(0.0, 0.1, rows),
            "breadth": rng.normal(size=rows),
            "liquidity": rng.normal(size=rows),
        },
        index=index,
    )
    features.loc[index[::19], "macro"] = np.nan
    target = pd.Series(states, index=index, name="next_regime")
    current = target.shift(1).fillna("transition").astype(str)
    return features, target, current


def _small_profile() -> BenchmarkProfile:
    return replace(
        BenchmarkProfile.quick(),
        xgboost_trees=8,
        spline_pca_components=4,
    )


def _matrix_for(
    name: str,
    features: pd.DataFrame,
    current: pd.Series,
) -> pd.DataFrame:
    if name == "transition_logistic":
        return augment_with_current_state(features, current)
    return features


MODEL_PARAMETERS = [
    "ridge_logistic",
    "transition_logistic",
    "shrinkage_lda",
    "spline_logistic",
    pytest.param(
        "xgboost",
        marks=pytest.mark.skipif(
            not XGBOOST_RUNTIME_AVAILABLE,
            reason="xgboost native runtime is unavailable",
        ),
    ),
]


def test_registry_keeps_existing_eight_and_adds_versioned_models() -> None:
    assert MODEL_NAMES == (
        "majority",
        "persistence",
        "markov",
        "elastic_net_logistic",
        "calibrated_linear_svm",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        *NEW_MODELS,
    )
    assert set(MODEL_NAMES) < set(MODEL_REGISTRY)
    assert "gaussian_hmm" in MODEL_REGISTRY
    assert MODEL_REGISTRY["transition_logistic"].requires_current_state is True
    assert MODEL_REGISTRY["transition_logistic"].feature_design == (
        "pooled_current_state_dummies_shared_feature_slopes"
    )
    assert MODEL_REGISTRY["duration_tvtp_hurdle"].requires_current_state is True
    assert "stay_switch_hurdle" in (
        MODEL_REGISTRY["duration_tvtp_hurdle"].feature_design
    )
    assert model_complexity_rank("ridge_logistic") < model_complexity_rank(
        "xgboost"
    )


def test_model_manifest_is_json_stable_and_hashes_exact_serialization() -> None:
    serialized = serialize_model_manifest(_small_profile(), random_state=31)
    parsed = json.loads(serialized)

    assert parsed == model_manifest(_small_profile(), random_state=31)
    assert parsed["profile"] == "quick"
    assert [row["name"] for row in parsed["models"]] == list(MODEL_NAMES)
    transition = next(
        row for row in parsed["models"] if row["name"] == "transition_logistic"
    )
    assert transition["feature_design"] == (
        "pooled_current_state_dummies_shared_feature_slopes"
    )
    hurdle = next(
        row for row in parsed["models"] if row["name"] == "duration_tvtp_hurdle"
    )
    assert hurdle["task"] == "multiclass_next_state"
    assert hurdle["horizons_weeks"] == [1]
    assert hurdle["search_space"]["hazard_C"] == [0.05, 0.1]
    assert parsed["transition_research"]["horizons_weeks"] == [1, 4, 13]
    assert model_manifest_sha256(_small_profile(), random_state=31) == hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()
    assert serialize_model_manifest(_small_profile(), random_state=31) == serialized


def test_transition_augmentation_defaults_to_pooled_shared_slopes() -> None:
    features, _, current = _model_data(rows=12)
    augmented = augment_with_current_state(features, current)

    assert augmented.index.equals(features.index)
    assert augmented.columns[: len(features.columns)].tolist() == list(features.columns)
    assert len(augmented.columns) == len(features.columns) + len(STATE_ORDER)
    assert not any(name.startswith("state_interaction__") for name in augmented)
    np.testing.assert_allclose(
        augmented[[f"current_state__{state}" for state in STATE_ORDER]].sum(axis=1),
        1.0,
    )

    interacted = augment_with_current_state(features, current, interactions=True)
    assert "state_interaction__risk_on__trend" in interacted
    expected = features["trend"] * (current == "risk_on").astype(float)
    np.testing.assert_allclose(
        interacted["state_interaction__risk_on__trend"], expected, equal_nan=True
    )


@pytest.mark.parametrize("name", MODEL_PARAMETERS)
def test_new_estimators_are_cloneable_and_emit_aligned_probabilities(name: str) -> None:
    features, target, current = _model_data()
    matrix = _matrix_for(name, features, current)
    estimator = clone(build_model(name, _small_profile(), random_state=23))

    estimator.fit(matrix.iloc[:-8], target.iloc[:-8])
    raw = estimator.predict_proba(matrix.iloc[-8:])
    aligned = align_probabilities(
        raw,
        estimator.classes_,
        expected_rows=8,
    )

    assert set(map(str, estimator.classes_)) == set(STATE_ORDER)
    assert aligned.shape == (8, len(STATE_ORDER))
    assert np.isfinite(aligned).all()
    assert (aligned >= 0.0).all()
    np.testing.assert_allclose(aligned.sum(axis=1), 1.0, atol=1e-12)


@pytest.mark.parametrize("name", MODEL_PARAMETERS)
def test_new_estimators_are_deterministic_for_a_fixed_seed(name: str) -> None:
    features, target, current = _model_data()
    matrix = _matrix_for(name, features, current)
    profile = _small_profile()
    probabilities: list[np.ndarray] = []

    for _ in range(2):
        estimator = clone(build_model(name, profile, random_state=71))
        estimator.fit(matrix.iloc[:-8], target.iloc[:-8])
        probabilities.append(
            align_probabilities(
                estimator.predict_proba(matrix.iloc[-8:]),
                estimator.classes_,
                expected_rows=8,
            )
        )

    np.testing.assert_allclose(probabilities[0], probabilities[1], rtol=0, atol=1e-12)


def test_xgboost_adapter_raises_clear_import_error_when_dependency_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    original_import = builtins.__import__

    def reject_xgboost(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "xgboost":
            raise ImportError("simulated missing dependency")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_xgboost)
    estimator = StateEncodedXGBoostClassifier(n_estimators=2)
    with pytest.raises(ImportError, match="requires dependency 'xgboost'"):
        estimator.fit(np.arange(24, dtype=float).reshape(12, 2), np.resize(STATE_ORDER, 12))


@pytest.mark.skipif(
    not XGBOOST_RUNTIME_AVAILABLE,
    reason="xgboost native runtime is unavailable",
)
def test_xgboost_adapter_preserves_supported_order_when_a_class_is_missing() -> None:
    features, target, _ = _model_data(rows=90)
    features = features.fillna(features.median())
    target = target.replace("transition", "risk_off")
    estimator = StateEncodedXGBoostClassifier(n_estimators=6, random_state=53)
    estimator.fit(features, target)

    assert tuple(estimator.classes_) == ("risk_on", "risk_off")
    aligned = align_probabilities(
        estimator.predict_proba(features.iloc[-3:]),
        estimator.classes_,
        expected_rows=3,
    )
    assert (aligned[:, STATE_ORDER.index("transition")] > 0.0).all()
    np.testing.assert_allclose(aligned.sum(axis=1), 1.0, atol=1e-12)


def test_align_probabilities_floors_only_a_missing_supported_class() -> None:
    aligned = align_probabilities(
        np.asarray([[0.75, 0.25], [0.0, 1.0]]),
        ("risk_on", "risk_off"),
        expected_rows=2,
    )

    assert aligned.shape == (2, 3)
    assert (aligned[:, STATE_ORDER.index("transition")] > 0.0).all()
    assert aligned[1, STATE_ORDER.index("risk_on")] == 0.0
    np.testing.assert_allclose(aligned.sum(axis=1), 1.0, atol=1e-12)


@pytest.mark.parametrize(
    ("probabilities", "classes", "kwargs", "message"),
    [
        ([[np.nan, 0.5, 0.5]], STATE_ORDER, {}, "NaN or infinite"),
        ([[np.inf, 0.0, 0.0]], STATE_ORDER, {}, "NaN or infinite"),
        ([[-0.1, 0.5, 0.6]], STATE_ORDER, {}, "in \\[0, 1\\]"),
        ([[0.0, 0.0, 0.0]], STATE_ORDER, {}, "positive sum"),
        ([[0.6, 0.6, 0.6]], STATE_ORDER, {}, "sum to one"),
        ([[0.5, 0.5]], ("risk_on", "risk_on"), {}, "duplicates"),
        ([[0.5, 0.5]], ("risk_on", "unknown"), {}, "unsupported states"),
        ([[0.2, 0.3, 0.5]], STATE_ORDER, {"expected_rows": 2}, "expected_rows"),
    ],
)
def test_align_probabilities_rejects_invalid_estimator_output(
    probabilities,
    classes,
    kwargs,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        align_probabilities(np.asarray(probabilities, dtype=float), classes, **kwargs)
