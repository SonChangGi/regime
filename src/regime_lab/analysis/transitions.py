"""Causal, duration-aware transition models for the fixed three-state regime.

The estimators in this module deliberately consume the state known at the
forecast origin as an input column.  A row therefore represents
``(X_t, S_t) -> S_{t+1}``; neither estimator infers ``S_t`` from the forecast
target.  Probability columns always follow :data:`STATE_ORDER`.

The convex blend expects *out-of-sample* discriminative probabilities in its
input frame.  Producing those probabilities without look-ahead remains the
caller's responsibility.  Its weight is learned during ``fit`` only and
``predict_proba`` has no target argument, which makes accidental target-time
weight fitting impossible through the public prediction API.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .labels import STATE_ORDER


CURRENT_STATE_COLUMN = "current_state"
DURATION_COLUMN = "state_duration_weeks"
DISCRIMINATIVE_PROBABILITY_COLUMNS = tuple(
    f"discriminative_probability__{state}" for state in STATE_ORDER
)

_STATE_POSITION = {state: position for position, state in enumerate(STATE_ORDER)}
_ADJACENT_DESTINATIONS: dict[str, tuple[str, ...]] = {
    "risk_on": ("transition",),
    "transition": ("risk_on", "risk_off"),
    "risk_off": ("transition",),
}


def _as_supported_states(
    values: pd.Series | Iterable[str],
    *,
    expected_length: int | None = None,
    context: str,
) -> np.ndarray:
    if isinstance(values, pd.Series):
        raw = values.to_numpy(dtype=object)
    else:
        raw = np.asarray(list(values), dtype=object).reshape(-1)
    if expected_length is not None and len(raw) != expected_length:
        raise ValueError(
            f"{context} length must be {expected_length}, received {len(raw)}"
        )
    if pd.isna(raw).any():
        raise ValueError(f"{context} contains missing states")
    states = np.asarray([str(value) for value in raw], dtype=object)
    invalid = sorted(set(states).difference(STATE_ORDER))
    if invalid:
        raise ValueError(f"{context} contains unsupported states: {invalid}")
    return states


def _target_array(target: Any, frame: pd.DataFrame) -> np.ndarray:
    if isinstance(target, pd.Series) and not target.index.equals(frame.index):
        raise ValueError("target index must exactly match the feature frame")
    raw = np.asarray(target, dtype=object).reshape(-1)
    return _as_supported_states(
        raw,
        expected_length=len(frame),
        context="target",
    )


def _validate_frame(frame: Any, *, current_state_col: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    if frame.columns.has_duplicates:
        raise ValueError("feature columns must be unique")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("feature index must be unique and increasing")
    if not all(isinstance(column, str) for column in frame.columns):
        raise TypeError("feature column names must be strings")
    if current_state_col not in frame.columns:
        raise KeyError(f"missing current-state column: {current_state_col}")
    if frame.empty:
        raise ValueError("features must contain at least one row")
    return frame


def causal_state_durations(
    current_states: pd.Series | Iterable[str],
    *,
    initial_state: str | None = None,
    initial_duration: int = 0,
    name: str = DURATION_COLUMN,
) -> pd.Series:
    """Return the causal run length of each known state.

    ``initial_state`` and ``initial_duration`` let a forecast batch continue a
    training history without mutating estimator state.  Appending later states
    can never change an earlier duration.
    """

    if initial_state is not None and initial_state not in STATE_ORDER:
        raise ValueError(f"initial_state must be one of {STATE_ORDER}")
    if isinstance(initial_duration, bool) or int(initial_duration) != initial_duration:
        raise ValueError("initial_duration must be a non-negative integer")
    if int(initial_duration) < 0:
        raise ValueError("initial_duration must be a non-negative integer")
    if initial_state is None and int(initial_duration) != 0:
        raise ValueError("initial_duration requires initial_state")

    index = current_states.index if isinstance(current_states, pd.Series) else None
    states = _as_supported_states(current_states, context="current_states")
    previous = initial_state
    run_length = int(initial_duration)
    output = np.empty(len(states), dtype=np.int64)
    for position, state in enumerate(states):
        if state == previous:
            run_length += 1
        else:
            previous = str(state)
            run_length = 1
        output[position] = run_length
    return pd.Series(output, index=index, name=name, dtype="int64")


def derive_causal_transition_features(
    frame: pd.DataFrame,
    current_states: pd.Series | Iterable[str] | None = None,
    *,
    current_state_col: str = CURRENT_STATE_COLUMN,
    duration_col: str = DURATION_COLUMN,
    risk_score_col: str = "risk_score",
    lower_threshold: float | None = None,
    upper_threshold: float | None = None,
    initial_state: str | None = None,
    initial_duration: int = 0,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Add causal duration, score-path, and optional boundary features.

    All path features use backward differences only.  Boundary distances are
    added only when both train-fitted thresholds are supplied.  Existing
    generated columns are preserved unless ``overwrite=True`` so an explicitly
    supplied duration for a discontinuous validation slice is never replaced.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if frame.columns.has_duplicates:
        raise ValueError("frame columns must be unique")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("frame index must be unique and increasing")
    output = frame.copy()

    if current_states is None:
        if current_state_col not in output:
            raise KeyError(
                f"current_states was not supplied and {current_state_col!r} is absent"
            )
        states = _as_supported_states(
            output[current_state_col],
            expected_length=len(output),
            context="current_states",
        )
    else:
        if isinstance(current_states, pd.Series) and not current_states.index.equals(
            output.index
        ):
            raise ValueError("current_states index must exactly match frame")
        states = _as_supported_states(
            current_states,
            expected_length=len(output),
            context="current_states",
        )
        if current_state_col not in output:
            output[current_state_col] = states

    if overwrite or duration_col not in output:
        state_series = pd.Series(states, index=output.index, dtype="object")
        output[duration_col] = causal_state_durations(
            state_series,
            initial_state=initial_state,
            initial_duration=initial_duration,
            name=duration_col,
        )
    else:
        duration = pd.to_numeric(output[duration_col], errors="coerce")
        if (
            duration.isna().any()
            or (~np.isfinite(duration)).any()
            or (duration < 1).any()
            or not np.allclose(duration, np.round(duration), rtol=0.0, atol=1e-12)
        ):
            raise ValueError(f"{duration_col} must contain positive integer values")

    if risk_score_col not in output:
        return output
    score = pd.to_numeric(output[risk_score_col], errors="coerce").astype(float)

    generated: dict[str, pd.Series] = {
        "risk_score_delta_1w": score.diff(1),
        "risk_score_delta_2w": score.diff(2),
        "risk_score_delta_4w": score.diff(4),
        "risk_score_acceleration_1w": score.diff(1).diff(1),
    }
    if (lower_threshold is None) != (upper_threshold is None):
        raise ValueError("lower_threshold and upper_threshold must be supplied together")
    if lower_threshold is not None and upper_threshold is not None:
        lower = float(lower_threshold)
        upper = float(upper_threshold)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError("thresholds must be finite and lower_threshold < upper_threshold")
        width = upper - lower
        lower_distance = (score - lower) / width
        upper_distance = (upper - score) / width
        state_boundary_distance = np.select(
            [states == "risk_on", states == "transition", states == "risk_off"],
            [
                (score - upper) / width,
                np.minimum(lower_distance, upper_distance),
                (lower - score) / width,
            ],
            default=np.nan,
        )
        generated.update(
            {
                "risk_score_distance_above_lower": lower_distance,
                "risk_score_distance_below_upper": upper_distance,
                "risk_score_nearest_boundary": pd.concat(
                    [lower_distance.abs(), upper_distance.abs()], axis=1
                ).min(axis=1),
                "risk_score_state_boundary_distance": pd.Series(
                    state_boundary_distance,
                    index=output.index,
                    dtype=float,
                ),
            }
        )
    for column, values in generated.items():
        if overwrite or column not in output:
            output[column] = values
    return output.replace([np.inf, -np.inf], np.nan)


def _allowed_destinations(state: str, adjacent_only: bool) -> tuple[str, ...]:
    if adjacent_only:
        return _ADJACENT_DESTINATIONS[state]
    return tuple(candidate for candidate in STATE_ORDER if candidate != state)


def _logistic_pipeline(*, C: float, random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    solver="lbfgs",
                    max_iter=2_000,
                    tol=1e-7,
                    random_state=random_state,
                ),
            ),
        ]
    )


class DurationAwareTVTPHurdleClassifier(ClassifierMixin, BaseEstimator):
    """Time-varying transition-probability hurdle classifier.

    The first logistic model estimates ``P(S_{t+1} != S_t)``.  Conditional on
    switching, the second model estimates the destination.  With
    ``adjacent_only=True`` a direct ``risk_on <-> risk_off`` probability is
    exactly zero and any observed direct jump is routed through ``transition``.
    Missing hazard or destination classes use smoothed, source-state priors and
    are reported through ``fallback_reasons_`` and ``fit_diagnostics_``.
    """

    def __init__(
        self,
        *,
        current_state_col: str = CURRENT_STATE_COLUMN,
        duration_col: str = DURATION_COLUMN,
        derive_duration: bool = True,
        adjacent_only: bool = True,
        hazard_C: float = 0.10,
        destination_C: float = 0.10,
        smoothing: float = 1.0,
        random_state: int = 17,
    ) -> None:
        self.current_state_col = current_state_col
        self.duration_col = duration_col
        self.derive_duration = derive_duration
        self.adjacent_only = adjacent_only
        self.hazard_C = hazard_C
        self.destination_C = destination_C
        self.smoothing = smoothing
        self.random_state = random_state

    def _model_frame(
        self,
        frame: pd.DataFrame,
        *,
        fitting: bool,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        checked = _validate_frame(frame, current_state_col=self.current_state_col)
        if fitting:
            self.feature_names_in_ = np.asarray(checked.columns, dtype=object)
            self.n_features_in_ = len(checked.columns)
        else:
            expected = list(self.feature_names_in_)
            if list(checked.columns) != expected:
                raise ValueError(
                    "feature columns changed after fit: "
                    f"expected {expected}, received {list(checked.columns)}"
                )
        states = _as_supported_states(
            checked[self.current_state_col],
            expected_length=len(checked),
            context=self.current_state_col,
        )
        prepared = checked.copy()
        if self.duration_col not in prepared:
            if not self.derive_duration:
                raise KeyError(f"missing duration column: {self.duration_col}")
            initial_state = None if fitting else self.last_training_state_
            initial_duration = 0 if fitting else self.last_training_duration_
            prepared[self.duration_col] = causal_state_durations(
                pd.Series(states, index=prepared.index, dtype="object"),
                initial_state=initial_state,
                initial_duration=initial_duration,
                name=self.duration_col,
            )
        else:
            duration = pd.to_numeric(prepared[self.duration_col], errors="coerce")
            if (
                duration.isna().any()
                or (~np.isfinite(duration)).any()
                or (duration < 1).any()
                or not np.allclose(
                    duration, np.round(duration), rtol=0.0, atol=1e-12
                )
            ):
                raise ValueError(
                    f"{self.duration_col} must contain positive integer values"
                )
            prepared[self.duration_col] = duration.astype(float)

        numeric = prepared.drop(columns=[self.current_state_col]).copy()
        for column in numeric:
            converted = pd.to_numeric(numeric[column], errors="coerce")
            invalid = numeric[column].notna() & converted.isna()
            if invalid.any():
                raise TypeError(f"feature column {column!r} must be numeric")
            numeric[column] = converted.astype(float)
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        for state in STATE_ORDER:
            numeric[f"current_state__{state}"] = (states == state).astype(float)
        if fitting:
            self.model_feature_names_ = tuple(numeric.columns)
        elif tuple(numeric.columns) != self.model_feature_names_:
            raise ValueError("derived model feature columns changed after fit")
        return numeric, states

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series | Sequence[str],
    ) -> "DurationAwareTVTPHurdleClassifier":
        if not isinstance(self.adjacent_only, (bool, np.bool_)):
            raise TypeError("adjacent_only must be boolean")
        if not isinstance(self.derive_duration, (bool, np.bool_)):
            raise TypeError("derive_duration must be boolean")
        if not np.isfinite(self.smoothing) or float(self.smoothing) <= 0.0:
            raise ValueError("smoothing must be positive and finite")
        if not np.isfinite(self.hazard_C) or float(self.hazard_C) <= 0.0:
            raise ValueError("hazard_C must be positive and finite")
        if not np.isfinite(self.destination_C) or float(self.destination_C) <= 0.0:
            raise ValueError("destination_C must be positive and finite")

        matrix, current = self._model_frame(features, fitting=True)
        labels = _target_array(target, features)
        switch = labels != current
        self.classes_ = np.asarray(STATE_ORDER, dtype=object)
        self.last_training_state_ = str(current[-1])
        duration_values = matrix[self.duration_col].to_numpy(dtype=float)
        self.last_training_duration_ = int(round(float(duration_values[-1])))

        reasons: list[str] = []
        degraded_reasons: list[str] = []
        missing_target = tuple(state for state in STATE_ORDER if state not in set(labels))
        self.missing_target_classes_ = missing_target
        if missing_target:
            degraded_reasons.append(
                "target_classes_missing:" + ",".join(missing_target)
            )

        smoothing = float(self.smoothing)
        hazard_counts = np.full((len(STATE_ORDER), 2), smoothing, dtype=float)
        for source, changed in zip(current, switch, strict=True):
            hazard_counts[_STATE_POSITION[str(source)], int(changed)] += 1.0
        self.hazard_prior_by_state_ = hazard_counts[:, 1] / hazard_counts.sum(axis=1)
        self.hazard_estimator_: Pipeline | None = None
        if len(np.unique(switch)) == 2:
            self.hazard_estimator_ = _logistic_pipeline(
                C=float(self.hazard_C), random_state=int(self.random_state)
            )
            self.hazard_estimator_.fit(matrix, switch.astype(int))
        else:
            observed = "switch" if bool(switch[0]) else "stay"
            reasons.append(f"hazard_class_missing:only_{observed}_observed")

        destination_counts = np.zeros(
            (len(STATE_ORDER), len(STATE_ORDER)), dtype=float
        )
        for source in STATE_ORDER:
            source_position = _STATE_POSITION[source]
            for destination in _allowed_destinations(
                source, bool(self.adjacent_only)
            ):
                destination_counts[source_position, _STATE_POSITION[destination]] = (
                    smoothing
                )

        valid_switch = np.zeros(len(labels), dtype=bool)
        forbidden_count = 0
        for position, (source, destination, changed) in enumerate(
            zip(current, labels, switch, strict=True)
        ):
            if not changed:
                continue
            if str(destination) not in _allowed_destinations(
                str(source), bool(self.adjacent_only)
            ):
                forbidden_count += 1
                continue
            valid_switch[position] = True
            destination_counts[
                _STATE_POSITION[str(source)], _STATE_POSITION[str(destination)]
            ] += 1.0
        self.forbidden_transition_count_ = int(forbidden_count)
        if forbidden_count:
            degraded_reasons.append(
                f"forbidden_transitions_routed_adjacent:{forbidden_count}"
            )
        self.conditional_destination_prior_ = destination_counts.copy()
        self.conditional_destination_prior_ /= self.conditional_destination_prior_.sum(
            axis=1, keepdims=True
        )

        observed_destinations = tuple(
            state for state in STATE_ORDER if state in set(labels[valid_switch])
        )
        self.observed_destination_classes_ = observed_destinations
        self.destination_estimator_: Pipeline | None = None
        if valid_switch.sum() > 0 and len(observed_destinations) >= 2:
            self.destination_estimator_ = _logistic_pipeline(
                C=float(self.destination_C), random_state=int(self.random_state)
            )
            self.destination_estimator_.fit(matrix.loc[valid_switch], labels[valid_switch])
        else:
            if valid_switch.sum() == 0:
                reasons.append("destination_fallback:no_allowed_switch_observations")
            else:
                reasons.append(
                    "destination_class_missing:only_"
                    + observed_destinations[0]
                    + "_observed"
                )

        self.fallback_reasons_ = tuple((*degraded_reasons, *reasons))
        self.used_fallback_ = bool(reasons)
        self.fit_diagnostics_ = {
            "training_rows": int(len(matrix)),
            "switch_rows": int(switch.sum()),
            "allowed_switch_rows": int(valid_switch.sum()),
            "forbidden_transition_rows": int(forbidden_count),
            "missing_target_classes": missing_target,
            "fallback_reasons": self.fallback_reasons_,
            "degraded_reasons": tuple(degraded_reasons),
            "duration_source": (
                "derived_causally" if self.duration_col not in features else "provided"
            ),
        }
        return self

    def _hazard_probability(
        self, matrix: pd.DataFrame, current: np.ndarray
    ) -> np.ndarray:
        if self.hazard_estimator_ is None:
            return np.asarray(
                [self.hazard_prior_by_state_[_STATE_POSITION[str(state)]] for state in current],
                dtype=float,
            )
        raw = np.asarray(self.hazard_estimator_.predict_proba(matrix), dtype=float)
        observed = list(self.hazard_estimator_.classes_)
        return raw[:, observed.index(1)]

    def _destination_probability(
        self, matrix: pd.DataFrame, current: np.ndarray
    ) -> np.ndarray:
        if self.destination_estimator_ is None:
            raw_full = np.vstack(
                [
                    self.conditional_destination_prior_[_STATE_POSITION[str(state)]]
                    for state in current
                ]
            )
        else:
            raw = np.asarray(
                self.destination_estimator_.predict_proba(matrix), dtype=float
            )
            raw_full = np.zeros((len(matrix), len(STATE_ORDER)), dtype=float)
            for source_column, state in enumerate(self.destination_estimator_.classes_):
                raw_full[:, _STATE_POSITION[str(state)]] = raw[:, source_column]

        constrained = np.zeros_like(raw_full)
        for row, source in enumerate(current):
            allowed = _allowed_destinations(str(source), bool(self.adjacent_only))
            positions = [_STATE_POSITION[state] for state in allowed]
            mass = float(raw_full[row, positions].sum())
            if not np.isfinite(mass) or mass <= 0.0:
                constrained[row] = self.conditional_destination_prior_[
                    _STATE_POSITION[str(source)]
                ]
            else:
                constrained[row, positions] = raw_full[row, positions] / mass
        return constrained

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        check_is_fitted(
            self,
            (
                "classes_",
                "feature_names_in_",
                "hazard_prior_by_state_",
                "conditional_destination_prior_",
                "last_training_state_",
            ),
        )
        matrix, current = self._model_frame(features, fitting=False)
        hazard = np.asarray(self._hazard_probability(matrix, current), dtype=float)
        if not np.isfinite(hazard).all():
            raise RuntimeError("hazard estimator emitted non-finite probabilities")
        hazard = np.clip(hazard, 0.0, 1.0)
        destination = self._destination_probability(matrix, current)
        output = hazard[:, None] * destination
        for row, state in enumerate(current):
            output[row, _STATE_POSITION[str(state)]] += 1.0 - hazard[row]
        if (
            not np.isfinite(output).all()
            or (output < -1e-12).any()
            or (output > 1.0 + 1e-12).any()
        ):
            raise RuntimeError("hurdle estimator emitted invalid probabilities")
        output = np.clip(output, 0.0, 1.0)
        output /= output.sum(axis=1, keepdims=True)
        return output

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        probability = self.predict_proba(features)
        return self.classes_[probability.argmax(axis=1)]


def _strict_probability_frame(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    tolerance: float,
) -> np.ndarray:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise KeyError(f"missing probability columns: {missing}")
    try:
        probabilities = frame.loc[:, list(columns)].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("probability columns must be numeric") from exc
    if not np.isfinite(probabilities).all():
        raise ValueError("probability columns contain NaN or infinite values")
    if (probabilities < 0.0).any() or (probabilities > 1.0).any():
        raise ValueError("probability columns must be in [0, 1]")
    sums = probabilities.sum(axis=1)
    if not np.allclose(sums, 1.0, rtol=0.0, atol=tolerance):
        raise ValueError("probability columns must sum to one")
    return probabilities / sums[:, None]


class MarkovDiscriminativeBlendClassifier(ClassifierMixin, BaseEstimator):
    """Convex Markov/discriminative probability blend fit on calibration rows.

    ``features`` must carry a known current state and state-order-aligned,
    preferably OOS, discriminative probabilities.  Optional separate Markov
    transitions can be passed to ``fit`` so that its transition matrix is fit
    on an earlier training window while the convex weight is selected only on
    the supplied calibration frame.
    """

    def __init__(
        self,
        *,
        current_state_col: str = CURRENT_STATE_COLUMN,
        probability_columns: tuple[str, ...] = DISCRIMINATIVE_PROBABILITY_COLUMNS,
        markov_alpha: float = 1.0,
        weight_grid_size: int = 1001,
        probability_tolerance: float = 1e-6,
    ) -> None:
        self.current_state_col = current_state_col
        self.probability_columns = probability_columns
        self.markov_alpha = markov_alpha
        self.weight_grid_size = weight_grid_size
        self.probability_tolerance = probability_tolerance

    def _transition_matrix(
        self, current: np.ndarray, target: np.ndarray
    ) -> np.ndarray:
        counts = np.full(
            (len(STATE_ORDER), len(STATE_ORDER)),
            float(self.markov_alpha),
            dtype=float,
        )
        for source, destination in zip(current, target, strict=True):
            counts[_STATE_POSITION[str(source)], _STATE_POSITION[str(destination)]] += 1.0
        return counts / counts.sum(axis=1, keepdims=True)

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series | Sequence[str],
        *,
        markov_current_states: pd.Series | Iterable[str] | None = None,
        markov_next_states: pd.Series | Iterable[str] | None = None,
    ) -> "MarkovDiscriminativeBlendClassifier":
        checked = _validate_frame(features, current_state_col=self.current_state_col)
        if not np.isfinite(self.markov_alpha) or float(self.markov_alpha) <= 0.0:
            raise ValueError("markov_alpha must be positive and finite")
        if isinstance(self.weight_grid_size, bool) or int(self.weight_grid_size) < 2:
            raise ValueError("weight_grid_size must be an integer of at least two")
        if int(self.weight_grid_size) != self.weight_grid_size:
            raise ValueError("weight_grid_size must be an integer of at least two")
        if (
            not np.isfinite(self.probability_tolerance)
            or float(self.probability_tolerance) <= 0.0
        ):
            raise ValueError("probability_tolerance must be positive and finite")
        probability_columns = tuple(self.probability_columns)
        if len(probability_columns) != len(STATE_ORDER):
            raise ValueError(
                f"probability_columns must contain exactly {len(STATE_ORDER)} names"
            )
        if len(set(probability_columns)) != len(probability_columns):
            raise ValueError("probability_columns contains duplicates")

        calibration_current = _as_supported_states(
            checked[self.current_state_col],
            expected_length=len(checked),
            context=self.current_state_col,
        )
        calibration_target = _target_array(target, checked)
        discriminative = _strict_probability_frame(
            checked,
            probability_columns,
            tolerance=float(self.probability_tolerance),
        )

        separate_markov = markov_current_states is not None or markov_next_states is not None
        if separate_markov:
            if markov_current_states is None or markov_next_states is None:
                raise ValueError(
                    "markov_current_states and markov_next_states must be supplied together"
                )
            markov_current = _as_supported_states(
                markov_current_states,
                context="markov_current_states",
            )
            markov_target = _as_supported_states(
                markov_next_states,
                expected_length=len(markov_current),
                context="markov_next_states",
            )
        else:
            markov_current = calibration_current
            markov_target = calibration_target

        self.transition_matrix_ = self._transition_matrix(markov_current, markov_target)
        markov = np.vstack(
            [self.transition_matrix_[_STATE_POSITION[str(state)]] for state in calibration_current]
        )
        truth_position = np.asarray(
            [_STATE_POSITION[str(state)] for state in calibration_target], dtype=int
        )
        weights = np.linspace(0.0, 1.0, int(self.weight_grid_size), dtype=float)
        losses = np.empty(len(weights), dtype=float)
        for position, markov_weight in enumerate(weights):
            blended = markov_weight * markov + (1.0 - markov_weight) * discriminative
            actual_probability = np.clip(
                blended[np.arange(len(blended)), truth_position], 1e-15, 1.0
            )
            losses[position] = -float(np.log(actual_probability).mean())
        minimum = float(losses.min())
        tied = np.flatnonzero(np.isclose(losses, minimum, rtol=0.0, atol=1e-12))
        # An exact tie (for example identical component probabilities) has no
        # empirical preference.  Choose the most neutral deterministic weight.
        chosen = int(tied[np.argmin(np.abs(weights[tied] - 0.5))])

        self.markov_weight_ = float(weights[chosen])
        self.discriminative_weight_ = 1.0 - self.markov_weight_
        self.calibration_log_loss_ = float(losses[chosen])
        self.component_log_loss_ = {
            "discriminative": float(losses[0]),
            "markov": float(losses[-1]),
            "blend": float(losses[chosen]),
        }
        self.classes_ = np.asarray(STATE_ORDER, dtype=object)
        self.feature_names_in_ = np.asarray(
            (self.current_state_col, *probability_columns), dtype=object
        )
        self.n_features_in_ = len(self.feature_names_in_)
        missing_target = tuple(
            state for state in STATE_ORDER if state not in set(calibration_target)
        )
        unobserved_source = tuple(
            state for state in STATE_ORDER if state not in set(markov_current)
        )
        reasons: list[str] = []
        if missing_target:
            reasons.append("calibration_target_classes_missing:" + ",".join(missing_target))
        if unobserved_source:
            reasons.append("markov_source_states_missing:" + ",".join(unobserved_source))
        self.missing_target_classes_ = missing_target
        self.unobserved_markov_source_states_ = unobserved_source
        self.fallback_reasons_ = tuple(reasons)
        self.used_fallback_ = bool(reasons)
        self.fit_diagnostics_ = {
            "calibration_rows": int(len(checked)),
            "markov_training_rows": int(len(markov_current)),
            "markov_training_scope": (
                "separate_training_inputs" if separate_markov else "calibration_inputs"
            ),
            "markov_weight": self.markov_weight_,
            "fallback_reasons": self.fallback_reasons_,
        }
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        check_is_fitted(
            self,
            (
                "classes_",
                "transition_matrix_",
                "markov_weight_",
                "feature_names_in_",
            ),
        )
        checked = _validate_frame(features, current_state_col=self.current_state_col)
        current = _as_supported_states(
            checked[self.current_state_col],
            expected_length=len(checked),
            context=self.current_state_col,
        )
        discriminative = _strict_probability_frame(
            checked,
            tuple(self.probability_columns),
            tolerance=float(self.probability_tolerance),
        )
        markov = np.vstack(
            [self.transition_matrix_[_STATE_POSITION[str(state)]] for state in current]
        )
        output = self.markov_weight_ * markov + self.discriminative_weight_ * discriminative
        if not np.isfinite(output).all() or (output < 0.0).any():
            raise RuntimeError("blend emitted invalid probabilities")
        output /= output.sum(axis=1, keepdims=True)
        return output

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        probability = self.predict_proba(features)
        return self.classes_[probability.argmax(axis=1)]


__all__ = [
    "CURRENT_STATE_COLUMN",
    "DURATION_COLUMN",
    "DISCRIMINATIVE_PROBABILITY_COLUMNS",
    "DurationAwareTVTPHurdleClassifier",
    "MarkovDiscriminativeBlendClassifier",
    "causal_state_durations",
    "derive_causal_transition_features",
]
