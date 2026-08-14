"""Model factory and probability-safe baselines for weekly regime forecasts."""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Callable, Iterable, Literal, Mapping

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from .labels import STATE_ORDER


@dataclass(frozen=True)
class BenchmarkProfile:
    """Computational budget for a deterministic benchmark run."""

    name: str
    max_origins: int | None
    minimum_train_weeks: int
    random_forest_trees: int
    extra_trees: int
    hist_gradient_iterations: int
    svm_calibration_splits: int
    hmm_iterations: int
    # Defaults preserve compatibility with callers that construct the historical
    # standard profile directly.  Quick/full override them explicitly below.
    xgboost_trees: int = 160
    spline_pca_components: int = 8

    @classmethod
    def quick(cls) -> "BenchmarkProfile":
        # Ten most recent weekly origins keeps interactive runs in seconds.
        return cls(
            name="quick",
            max_origins=10,
            minimum_train_weeks=104,
            random_forest_trees=24,
            extra_trees=24,
            hist_gradient_iterations=50,
            svm_calibration_splits=2,
            hmm_iterations=80,
            xgboost_trees=40,
            spline_pca_components=6,
        )

    @classmethod
    def full(cls) -> "BenchmarkProfile":
        return cls(
            name="full",
            max_origins=None,
            minimum_train_weeks=520,
            random_forest_trees=350,
            extra_trees=350,
            hist_gradient_iterations=250,
            svm_calibration_splits=4,
            hmm_iterations=300,
            xgboost_trees=300,
            spline_pca_components=12,
        )

    def with_overrides(
        self,
        *,
        max_origins: int | None = None,
        minimum_train_weeks: int | None = None,
    ) -> "BenchmarkProfile":
        values: dict[str, Any] = {}
        if max_origins is not None:
            values["max_origins"] = max_origins
        if minimum_train_weeks is not None:
            values["minimum_train_weeks"] = minimum_train_weeks
        return replace(self, **values)


ModelKind = Literal["baseline", "learned", "latent", "synthetic"]
ModelFactory = Callable[[BenchmarkProfile, int], Any]


@dataclass(frozen=True)
class ModelSpec:
    """One versioned source of model identity and selection complexity."""

    name: str
    kind: ModelKind
    complexity_rank: int
    profiles: tuple[str, ...]
    default: bool = True
    requires_current_state: bool = False
    dependency: str | None = None
    feature_design: str = "shared_features"
    task: str = "multiclass_next_state"
    horizons_weeks: tuple[int, ...] = (1,)
    search_space: Mapping[str, tuple[Any, ...]] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    factory: ModelFactory | None = field(default=None, repr=False, compare=False)


def resolve_profile(profile: str | BenchmarkProfile) -> BenchmarkProfile:
    if isinstance(profile, BenchmarkProfile):
        return profile
    if profile == "quick":
        return BenchmarkProfile.quick()
    if profile == "full":
        return BenchmarkProfile.full()
    raise ValueError("profile must be 'quick', 'full', or BenchmarkProfile")


def align_probabilities(
    probabilities: np.ndarray,
    observed_classes: Iterable[str],
    *,
    state_order: tuple[str, ...] = STATE_ORDER,
    floor: float = 1e-9,
    expected_rows: int | None = None,
) -> np.ndarray:
    """Align estimator probabilities to the stable public state order.

    A training fold can legitimately lack a rare supported class.  Only such
    missing columns receive a tiny floor before each row is normalised.  Invalid
    estimator output is rejected so the caller can record an explicit fallback.
    """

    raw = np.asarray(probabilities, dtype=float)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.ndim != 2:
        raise ValueError("probabilities must be a one- or two-dimensional array")
    if expected_rows is not None and raw.shape[0] != int(expected_rows):
        raise ValueError(
            "probability rows do not match expected_rows: "
            f"{raw.shape[0]} != {int(expected_rows)}"
        )
    classes = [str(value) for value in observed_classes]
    if not classes:
        raise ValueError("observed_classes must not be empty")
    if len(classes) != len(set(classes)):
        raise ValueError("observed_classes contains duplicates")
    if len(state_order) != len(set(state_order)):
        raise ValueError("state_order contains duplicates")
    unknown = sorted(set(classes).difference(state_order))
    if unknown:
        raise ValueError(f"observed_classes contains unsupported states: {unknown}")
    if raw.shape[1] != len(classes):
        raise ValueError("probability columns do not match observed_classes")
    if not np.isfinite(raw).all():
        raise ValueError("probabilities contain NaN or infinite values")
    if (raw < 0.0).any() or (raw > 1.0).any():
        raise ValueError("probabilities must be in [0, 1]")
    row_sums = raw.sum(axis=1)
    if (row_sums <= 0.0).any():
        raise ValueError("probability rows must have a positive sum")
    # Tree boosters commonly emit float32 softmax rows; permit only their
    # expected rounding noise before the final stable normalisation.
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("probability rows must sum to one")
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("floor must be a positive finite number")
    aligned = np.full((raw.shape[0], len(state_order)), floor, dtype=float)
    positions = {state: index for index, state in enumerate(state_order)}
    for source_index, state in enumerate(classes):
        aligned[:, positions[state]] = raw[:, source_index]
    aligned /= aligned.sum(axis=1, keepdims=True)
    return aligned


def class_prior_probabilities(
    target: pd.Series | np.ndarray,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    values = np.asarray(target, dtype=object)
    counts = np.array([(values == state).sum() for state in STATE_ORDER], dtype=float)
    return (counts + alpha) / (counts.sum() + alpha * len(STATE_ORDER))


def majority_probabilities(target: pd.Series | np.ndarray) -> np.ndarray:
    """A stable one-hot majority baseline (ties follow ``STATE_ORDER``)."""

    values = np.asarray(target, dtype=object)
    counts = np.array([(values == state).sum() for state in STATE_ORDER], dtype=int)
    output = np.full(len(STATE_ORDER), 1e-9, dtype=float)
    output[int(np.argmax(counts))] = 1.0
    return output / output.sum()


def persistence_probabilities(current_state: str) -> np.ndarray:
    output = np.full(len(STATE_ORDER), 1e-9, dtype=float)
    try:
        output[STATE_ORDER.index(str(current_state))] = 1.0
    except ValueError:
        output[:] = 1.0 / len(STATE_ORDER)
    return output / output.sum()


class SmoothedMarkovClassifier:
    """First-order transition baseline with Laplace smoothing."""

    def __init__(self, alpha: float = 1.0) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = float(alpha)
        self.transition_matrix_: np.ndarray | None = None

    def fit(
        self,
        current_states: pd.Series | np.ndarray,
        next_states: pd.Series | np.ndarray,
    ) -> "SmoothedMarkovClassifier":
        current = np.asarray(current_states, dtype=object)
        target = np.asarray(next_states, dtype=object)
        if len(current) != len(target):
            raise ValueError("current_states and next_states must have equal length")
        counts = np.full(
            (len(STATE_ORDER), len(STATE_ORDER)), self.alpha, dtype=float
        )
        positions = {state: index for index, state in enumerate(STATE_ORDER)}
        for source, destination in zip(current, target, strict=False):
            if source in positions and destination in positions:
                counts[positions[source], positions[destination]] += 1.0
        self.transition_matrix_ = counts / counts.sum(axis=1, keepdims=True)
        return self

    def predict_proba(self, current_states: Iterable[str]) -> np.ndarray:
        if self.transition_matrix_ is None:
            raise RuntimeError("SmoothedMarkovClassifier must be fit first")
        prior = self.transition_matrix_.mean(axis=0)
        output: list[np.ndarray] = []
        for state in current_states:
            try:
                output.append(self.transition_matrix_[STATE_ORDER.index(str(state))])
            except ValueError:
                output.append(prior)
        return np.vstack(output)


def augment_with_current_state(
    features: pd.DataFrame,
    current_states: pd.Series | np.ndarray | Iterable[str],
    *,
    interactions: bool = False,
) -> pd.DataFrame:
    """Add stable current-state dummies and optional state interactions.

    The helper is deliberately separate from the estimator so the validation
    layer can create train and test matrices from the state known at each origin
    without allowing a transformer to infer it from future labels.  The default
    pooled design keeps shared feature slopes; full state-by-feature interactions
    are opt-in because the transition class is sparse relative to feature count.
    """

    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    if features.columns.has_duplicates:
        raise ValueError("feature columns must be unique")
    if isinstance(current_states, pd.Series):
        if not current_states.index.equals(features.index):
            raise ValueError("current_states index must exactly match features")
        states = current_states.astype(str).to_numpy(dtype=object)
    else:
        states = np.asarray(list(current_states), dtype=object).reshape(-1)
        states = np.asarray([str(value) for value in states], dtype=object)
    if len(states) != len(features):
        raise ValueError("current_states length must match features")
    invalid = sorted(set(states).difference(STATE_ORDER))
    if invalid:
        raise ValueError(f"current_states contains unsupported labels: {invalid}")

    generated_names: list[str] = []
    for state in STATE_ORDER:
        generated_names.append(f"current_state__{state}")
        if interactions:
            generated_names.extend(
                f"state_interaction__{state}__{column}"
                for column in features.columns
            )
    collisions = sorted(set(generated_names).intersection(map(str, features.columns)))
    if collisions:
        raise ValueError(f"generated current-state columns collide: {collisions}")

    generated: dict[str, pd.Series] = {}
    for state in STATE_ORDER:
        indicator = pd.Series(
            (states == state).astype(float), index=features.index, dtype=float
        )
        generated[f"current_state__{state}"] = indicator
        if interactions:
            for column in features.columns:
                generated[f"state_interaction__{state}__{column}"] = (
                    features[column].astype(float) * indicator
                )
    return pd.concat([features.copy(), pd.DataFrame(generated)], axis=1)


class AdaptivePCA(TransformerMixin, BaseEstimator):
    """Fold-local PCA with a hard component cap for safe spline expansion."""

    def __init__(self, max_components: int = 8, random_state: int = 17) -> None:
        self.max_components = max_components
        self.random_state = random_state

    def fit(self, features: Any, target: Any | None = None) -> "AdaptivePCA":
        del target
        matrix = check_array(features, ensure_min_samples=2, ensure_min_features=1)
        if int(self.max_components) < 1:
            raise ValueError("max_components must be positive")
        component_count = min(
            int(self.max_components), matrix.shape[1], max(1, matrix.shape[0] - 1)
        )
        self.pca_ = PCA(
            n_components=component_count,
            svd_solver="randomized",
            iterated_power=3,
            random_state=self.random_state,
        )
        self.pca_.fit(matrix)
        self.n_features_in_ = int(matrix.shape[1])
        self.n_components_ = int(component_count)
        return self

    def transform(self, features: Any) -> np.ndarray:
        check_is_fitted(self, "pca_")
        matrix = check_array(features, ensure_min_features=self.n_features_in_)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                "feature count changed after AdaptivePCA fit: "
                f"{matrix.shape[1]} != {self.n_features_in_}"
            )
        return self.pca_.transform(matrix)


class StateEncodedXGBoostClassifier(ClassifierMixin, BaseEstimator):
    """Cloneable XGBoost adapter with stable three-state label semantics."""

    def __init__(
        self,
        *,
        n_estimators: int = 160,
        learning_rate: float = 0.04,
        max_depth: int = 2,
        min_child_weight: float = 8.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_lambda: float = 12.0,
        reg_alpha: float = 0.5,
        gamma: float = 0.1,
        random_state: int = 17,
        n_jobs: int = 1,
    ) -> None:
        # sklearn clone requires constructor parameters to be stored unchanged.
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, features: Any, target: Any) -> "StateEncodedXGBoostClassifier":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "xgboost challenger requires dependency 'xgboost'"
            ) from exc
        except Exception as exc:  # pragma: no cover - native runtime dependent
            raise RuntimeError(
                "xgboost is installed but its native runtime could not be loaded"
            ) from exc

        matrix, labels = check_X_y(features, target, dtype=float)
        labels = np.asarray([str(value) for value in labels], dtype=object)
        invalid = sorted(set(labels).difference(STATE_ORDER))
        if invalid:
            raise ValueError(f"target contains unsupported states: {invalid}")
        observed = tuple(state for state in STATE_ORDER if state in set(labels))
        if len(observed) < 2:
            raise ValueError("xgboost requires at least two observed target classes")
        positions = {state: index for index, state in enumerate(observed)}
        encoded = np.asarray([positions[str(value)] for value in labels], dtype=int)
        objective = "binary:logistic" if len(observed) == 2 else "multi:softprob"
        metric = "logloss" if len(observed) == 2 else "mlogloss"
        parameters: dict[str, Any] = {
            "objective": objective,
            "eval_metric": metric,
            "n_estimators": int(self.n_estimators),
            "learning_rate": float(self.learning_rate),
            "max_depth": int(self.max_depth),
            "min_child_weight": float(self.min_child_weight),
            "subsample": float(self.subsample),
            "colsample_bytree": float(self.colsample_bytree),
            "reg_lambda": float(self.reg_lambda),
            "reg_alpha": float(self.reg_alpha),
            "gamma": float(self.gamma),
            "tree_method": "hist",
            "n_jobs": int(self.n_jobs),
            "random_state": int(self.random_state),
            "verbosity": 0,
        }
        if len(observed) > 2:
            parameters["num_class"] = len(observed)
        self.estimator_ = XGBClassifier(**parameters)
        self.estimator_.fit(matrix, encoded, verbose=False)
        self.classes_ = np.asarray(observed, dtype=object)
        self.n_features_in_ = int(matrix.shape[1])
        return self

    def predict_proba(self, features: Any) -> np.ndarray:
        check_is_fitted(self, ("estimator_", "classes_"))
        matrix = check_array(features, dtype=float)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                "feature count changed after xgboost fit: "
                f"{matrix.shape[1]} != {self.n_features_in_}"
            )
        probability = np.asarray(self.estimator_.predict_proba(matrix), dtype=float)
        if probability.ndim == 1:
            probability = np.column_stack([1.0 - probability, probability])
        return probability

    def predict(self, features: Any) -> np.ndarray:
        probability = self.predict_proba(features)
        return self.classes_[probability.argmax(axis=1)]


def _scaled_linear_pipeline(classifier: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def _construct_model(
    name: str,
    cfg: BenchmarkProfile,
    random_state: int,
) -> Any:
    if name == "elastic_net_logistic":
        return _scaled_linear_pipeline(
            LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=0.25,
                C=0.5,
                class_weight="balanced",
                max_iter=2_000,
                tol=1e-4,
                random_state=random_state,
            )
        )
    if name == "ridge_logistic":
        return _scaled_linear_pipeline(
            LogisticRegression(
                solver="lbfgs",
                C=0.10,
                class_weight=None,
                max_iter=2_000,
                tol=1e-6,
                random_state=random_state,
            )
        )
    if name == "transition_logistic":
        return _scaled_linear_pipeline(
            LogisticRegression(
                solver="lbfgs",
                C=0.05,
                class_weight=None,
                max_iter=2_000,
                tol=1e-6,
                random_state=random_state,
            )
        )
    if name == "shrinkage_lda":
        return _scaled_linear_pipeline(
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        )
    if name == "spline_logistic":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                (
                    "pca",
                    AdaptivePCA(
                        max_components=cfg.spline_pca_components,
                        random_state=random_state,
                    ),
                ),
                (
                    "spline",
                    SplineTransformer(
                        n_knots=3,
                        degree=2,
                        knots="quantile",
                        extrapolation="linear",
                        include_bias=False,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        solver="lbfgs",
                        C=0.10,
                        class_weight=None,
                        max_iter=2_000,
                        tol=1e-6,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if name == "calibrated_linear_svm":
        base = _scaled_linear_pipeline(
            LinearSVC(
                C=0.5,
                class_weight="balanced",
                max_iter=10_000,
                random_state=random_state,
            )
        )
        return CalibratedClassifierCV(
            estimator=base,
            method="sigmoid",
            cv=TimeSeriesSplit(n_splits=cfg.svm_calibration_splits, gap=1),
            ensemble=True,
        )
    if name == "random_forest":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=cfg.random_forest_trees,
                        max_depth=6,
                        min_samples_leaf=4,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=1,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if name == "extra_trees":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "classifier",
                    ExtraTreesClassifier(
                        n_estimators=cfg.extra_trees,
                        max_depth=7,
                        min_samples_leaf=4,
                        max_features="sqrt",
                        class_weight="balanced",
                        n_jobs=1,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if name == "hist_gradient_boosting":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=cfg.hist_gradient_iterations,
                        max_leaf_nodes=15,
                        min_samples_leaf=10,
                        l2_regularization=1.0,
                        early_stopping=False,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if name == "xgboost":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "classifier",
                    StateEncodedXGBoostClassifier(
                        n_estimators=cfg.xgboost_trees,
                        learning_rate=0.04,
                        max_depth=2,
                        min_child_weight=8.0,
                        subsample=1.0,
                        colsample_bytree=1.0,
                        reg_lambda=12.0,
                        reg_alpha=0.5,
                        gamma=0.1,
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    raise KeyError(f"unknown learned model: {name}")


def hmm_available() -> bool:
    try:
        import hmmlearn  # noqa: F401
    except ImportError:
        return False
    return True


class GaussianHMMChallenger:
    """Optional unsupervised Gaussian HMM mapped to supervised next states.

    ``hmmlearn`` is intentionally optional.  Instantiation raises a clear
    ``ImportError`` when it is absent, allowing the benchmark to mark the model
    unavailable without affecting the required suite.
    """

    def __init__(
        self,
        *,
        n_iter: int = 100,
        random_state: int = 17,
        n_components: int = 3,
    ) -> None:
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "Gaussian HMM challenger requires optional dependency 'hmmlearn'"
            ) from exc
        self._model_class = GaussianHMM
        self.n_iter = n_iter
        self.random_state = random_state
        self.n_components = n_components
        self.imputer_ = SimpleImputer(strategy="median", add_indicator=True)
        self.scaler_ = StandardScaler()
        self.pca_: PCA | None = None
        self.train_matrix_: np.ndarray | None = None
        self.model_: Any | None = None
        self.hidden_to_state_: np.ndarray | None = None
        self.classes_ = np.asarray(STATE_ORDER, dtype=object)

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "GaussianHMMChallenger":
        matrix = self.imputer_.fit_transform(features)
        matrix = self.scaler_.fit_transform(matrix)
        factor_count = max(1, min(8, matrix.shape[1], len(matrix) - 1))
        self.pca_ = PCA(n_components=factor_count, random_state=self.random_state)
        matrix = self.pca_.fit_transform(matrix)
        self.train_matrix_ = matrix.copy()
        components = min(self.n_components, max(2, len(matrix) // 20))
        self.model_ = self._model_class(
            n_components=components,
            covariance_type="diag",
            n_iter=self.n_iter,
            min_covar=1e-4,
            random_state=self.random_state,
        )
        self.model_.fit(matrix)
        posterior = self.model_.predict_proba(matrix)
        mapping = np.full((components, len(STATE_ORDER)), 0.25, dtype=float)
        target_array = np.asarray(target, dtype=object)
        for state_index, state in enumerate(STATE_ORDER):
            mask = target_array == state
            if mask.any():
                mapping[:, state_index] += posterior[mask].sum(axis=0)
        mapping /= mapping.sum(axis=1, keepdims=True)
        self.hidden_to_state_ = mapping
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if self.model_ is None or self.hidden_to_state_ is None:
            raise RuntimeError("GaussianHMMChallenger must be fit first")
        matrix = self.scaler_.transform(self.imputer_.transform(features))
        pca = getattr(self, "pca_", None)
        if pca is not None:
            matrix = pca.transform(matrix)
        # ``hidden_to_state_`` was learned against the supervised y_{t+1}
        # target.  Applying the HMM transition matrix here would therefore add
        # another horizon and silently turn this into a t+2 forecast.
        train_matrix = getattr(self, "train_matrix_", None)
        if train_matrix is None:
            hidden_now = self.model_.predict_proba(matrix)
            direct = hidden_now @ self.hidden_to_state_
            return direct / direct.sum(axis=1, keepdims=True)
        probabilities: list[np.ndarray] = []
        history = train_matrix.copy()
        for row in matrix:
            history = np.vstack([history, row])
            hidden_now = self.model_.predict_proba(history)[-1]
            mapped = hidden_now @ self.hidden_to_state_
            probabilities.append(mapped / mapped.sum())
        return np.vstack(probabilities)


def _factory_for(name: str) -> ModelFactory:
    def factory(profile: BenchmarkProfile, random_state: int) -> Any:
        return _construct_model(name, profile, random_state)

    return factory


_MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("majority", "baseline", 0, ("quick", "standard", "full")),
    ModelSpec("persistence", "baseline", 1, ("quick", "standard", "full")),
    ModelSpec("markov", "baseline", 2, ("quick", "standard", "full")),
    ModelSpec(
        "elastic_net_logistic",
        "learned",
        5,
        ("quick", "standard", "full"),
        factory=_factory_for("elastic_net_logistic"),
    ),
    ModelSpec(
        "calibrated_linear_svm",
        "learned",
        7,
        ("quick", "standard", "full"),
        factory=_factory_for("calibrated_linear_svm"),
    ),
    ModelSpec(
        "random_forest",
        "learned",
        11,
        ("quick", "standard", "full"),
        factory=_factory_for("random_forest"),
    ),
    ModelSpec(
        "extra_trees",
        "learned",
        12,
        ("quick", "standard", "full"),
        factory=_factory_for("extra_trees"),
    ),
    ModelSpec(
        "hist_gradient_boosting",
        "learned",
        10,
        ("quick", "standard", "full"),
        factory=_factory_for("hist_gradient_boosting"),
    ),
    ModelSpec(
        "ridge_logistic",
        "learned",
        3,
        ("quick", "standard", "full"),
        factory=_factory_for("ridge_logistic"),
    ),
    ModelSpec(
        "transition_logistic",
        "learned",
        6,
        ("quick", "standard", "full"),
        requires_current_state=True,
        feature_design="pooled_current_state_dummies_shared_feature_slopes",
        factory=_factory_for("transition_logistic"),
    ),
    ModelSpec(
        "duration_tvtp_hurdle",
        "learned",
        7,
        ("quick", "standard", "full"),
        requires_current_state=True,
        feature_design=(
            "duration_aware_stay_switch_hurdle_with_adjacent_destination_constraint"
        ),
        search_space=MappingProxyType(
            {
                "hazard_C": (0.05, 0.10),
                "destination_C": (0.05, 0.10),
                "smoothing": (1.0,),
            }
        ),
        # Fitting is handled by the validation layer because the estimator
        # receives the known current state and an origin-specific causal
        # duration rather than an ordinary shared feature matrix.
        factory=None,
    ),
    ModelSpec(
        "shrinkage_lda",
        "learned",
        4,
        ("quick", "standard", "full"),
        factory=_factory_for("shrinkage_lda"),
    ),
    ModelSpec(
        "spline_logistic",
        "learned",
        8,
        ("quick", "standard", "full"),
        factory=_factory_for("spline_logistic"),
    ),
    ModelSpec(
        "xgboost",
        "learned",
        13,
        ("quick", "standard", "full"),
        dependency="xgboost",
        factory=_factory_for("xgboost"),
    ),
    ModelSpec(
        "xgb_hazard_destination",
        "synthetic",
        14,
        ("quick", "standard", "full"),
        default=False,
        requires_current_state=True,
        feature_design=(
            "binary_xgboost_departure_hazard_with_xgboost_conditional_destination"
        ),
        task="synthetic_multiclass_next_state",
        search_space=MappingProxyType({"direct_jump_floor": (1e-6,)}),
    ),
    ModelSpec(
        "causal_dynamic_ensemble",
        "synthetic",
        15,
        ("quick", "standard", "full"),
        default=False,
        feature_design=(
            "causal_discounted_log_loss_weights_markov_xgboost_joint"
        ),
        task="synthetic_multiclass_next_state",
        search_space=MappingProxyType(
            {"half_life_weeks": (52,), "minimum_history_rows": (26,)}
        ),
    ),
    ModelSpec(
        "joint_survival_hazard",
        "synthetic",
        8,
        ("quick", "standard", "full"),
        default=False,
        requires_current_state=True,
        feature_design=(
            "one_week_causal_hazard_origin_covariates_frozen_duration_incremented"
        ),
        task="synthetic_multihorizon_departure_survival",
        horizons_weeks=(1, 4, 13),
    ),
    ModelSpec(
        "gaussian_hmm",
        "latent",
        9,
        ("full",),
        default=False,
        dependency="hmmlearn",
    ),
)

MODEL_REGISTRY: Mapping[str, ModelSpec] = MappingProxyType(
    {spec.name: spec for spec in _MODEL_SPECS}
)
MODEL_NAMES: tuple[str, ...] = tuple(
    spec.name for spec in _MODEL_SPECS if spec.default
)


def build_model(
    name: str,
    profile: str | BenchmarkProfile = "quick",
    *,
    random_state: int = 17,
) -> Any:
    """Construct a fresh non-DL challenger from the central model registry."""

    spec = MODEL_REGISTRY.get(str(name))
    if spec is None or spec.factory is None:
        raise KeyError(f"unknown learned model: {name}")
    cfg = resolve_profile(profile)
    if cfg.name not in spec.profiles:
        raise ValueError(f"model {name} is not enabled for profile {cfg.name}")
    return spec.factory(cfg, int(random_state))


def model_complexity_rank(name: str) -> int:
    spec = MODEL_REGISTRY.get(str(name))
    return spec.complexity_rank if spec is not None else 99


def require_model_dependencies(names: Iterable[str]) -> None:
    """Fail before walk-forward work when a requested runtime is unavailable."""

    checked: set[str] = set()
    for name in names:
        spec = MODEL_REGISTRY.get(str(name))
        if spec is None:
            raise ValueError(f"unknown model dependency request: {name}")
        dependency = spec.dependency
        if dependency is None or dependency in checked:
            continue
        checked.add(dependency)
        try:
            importlib.import_module(dependency)
        except Exception as exc:
            raise RuntimeError(
                f"model {spec.name} requires a working {dependency} runtime"
            ) from exc


def _manifest_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            if np.isnan(value):
                return "NaN"
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, np.generic):
        return _manifest_value(value.item())
    if isinstance(value, Mapping):
        return {
            str(key): _manifest_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_manifest_value(item) for item in value]
    if isinstance(value, BaseEstimator):
        return {
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "parameters": _manifest_value(value.get_params(deep=False)),
        }
    return {
        "class": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
    }


def model_manifest(
    profile: str | BenchmarkProfile = "quick",
    *,
    random_state: int = 17,
    names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return a JSON-safe, deterministic model-suite manifest."""

    cfg = resolve_profile(profile)
    selected = list(MODEL_NAMES if names is None else names)
    selected = list(dict.fromkeys(str(name) for name in selected))
    unknown = sorted(set(selected).difference(MODEL_REGISTRY))
    if unknown:
        raise ValueError(f"unknown manifest models: {unknown}")
    rows: list[dict[str, Any]] = []
    for name in selected:
        spec = MODEL_REGISTRY[name]
        estimator = (
            spec.factory(cfg, int(random_state)) if spec.factory is not None else None
        )
        rows.append(
            {
                "name": spec.name,
                "kind": spec.kind,
                "complexity_rank": spec.complexity_rank,
                "profiles": list(spec.profiles),
                "default": spec.default,
                "requires_current_state": spec.requires_current_state,
                "dependency": spec.dependency,
                "feature_design": spec.feature_design,
                "task": spec.task,
                "horizons_weeks": list(spec.horizons_weeks),
                "search_space": _manifest_value(spec.search_space),
                "estimator": _manifest_value(estimator),
            }
        )
    return {
        "schema_version": "1.0.0",
        "profile": cfg.name,
        "random_state": int(random_state),
        "profile_budget": {
            "max_origins": cfg.max_origins,
            "minimum_train_weeks": cfg.minimum_train_weeks,
            "random_forest_trees": cfg.random_forest_trees,
            "extra_trees": cfg.extra_trees,
            "hist_gradient_iterations": cfg.hist_gradient_iterations,
            "svm_calibration_splits": cfg.svm_calibration_splits,
            "hmm_iterations": cfg.hmm_iterations,
            "xgboost_trees": cfg.xgboost_trees,
            "spline_pca_components": cfg.spline_pca_components,
        },
        "models": rows,
        "transition_research": {
            "task": "any_departure_from_current_state",
            "horizons_weeks": [1, 4, 13],
            "outer_split": "expanding_walk_forward_target_end_purged",
            "inner_split": "selection_only_or_prequential_past_oos",
            "calibration_method": "probability_first_no_post_selection_refit",
            "threshold_policy": "selection_only_balanced_accuracy_with_0.5_fallback",
            "feature_set_version": "weekly-pit-structural-v4",
        },
    }


def serialize_model_manifest(
    profile: str | BenchmarkProfile = "quick",
    *,
    random_state: int = 17,
    names: Iterable[str] | None = None,
) -> str:
    return json.dumps(
        model_manifest(profile, random_state=random_state, names=names),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def model_manifest_sha256(
    profile: str | BenchmarkProfile = "quick",
    *,
    random_state: int = 17,
    names: Iterable[str] | None = None,
) -> str:
    serialized = serialize_model_manifest(
        profile, random_state=random_state, names=names
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


__all__ = [
    "MODEL_NAMES",
    "MODEL_REGISTRY",
    "AdaptivePCA",
    "BenchmarkProfile",
    "GaussianHMMChallenger",
    "ModelSpec",
    "SmoothedMarkovClassifier",
    "StateEncodedXGBoostClassifier",
    "align_probabilities",
    "augment_with_current_state",
    "build_model",
    "class_prior_probabilities",
    "hmm_available",
    "majority_probabilities",
    "model_complexity_rank",
    "model_manifest",
    "model_manifest_sha256",
    "persistence_probabilities",
    "require_model_dependencies",
    "resolve_profile",
    "serialize_model_manifest",
]
