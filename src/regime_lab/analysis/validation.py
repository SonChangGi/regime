"""Leakage-resistant expanding walk-forward evaluation and next-week forecast."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
import importlib
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .labels import STATE_ORDER
from .models import MODEL_NAMES, BenchmarkProfile, GaussianHMMChallenger
from .models import SmoothedMarkovClassifier, align_probabilities
from .models import augment_with_current_state, build_model
from .models import class_prior_probabilities, majority_probabilities
from .models import model_complexity_rank, persistence_probabilities
from .models import require_model_dependencies, resolve_profile
from .transitions import CURRENT_STATE_COLUMN, DurationAwareTVTPHurdleClassifier
from .transitions import derive_causal_transition_features


PROBABILITY_COLUMNS: tuple[str, ...] = tuple(
    f"p_{state}" for state in STATE_ORDER
)
BASELINE_MODELS: tuple[str, ...] = ("majority", "persistence", "markov")
SELECTION_BOOTSTRAP_BLOCK_WEEKS = 13
SELECTION_BOOTSTRAP_RESAMPLES = 1_999
SELECTION_BOOTSTRAP_SEED = 17
SELECTION_ALPHA = 0.05
SELECTION_MINIMUM_LOG_LOSS_IMPROVEMENT = 0.05
SELECTION_BRIER_TOLERANCE = 0.01
TRANSITION_MODELS: tuple[str, ...] = (
    "empirical_hazard",
    "markov_hazard",
    "duration_tvtp_hurdle",
    "regularized_logistic",
)
_TRANSITION_OPTIONAL_MODELS: tuple[str, ...] = ("binary_xgboost",)
_TRANSITION_EPSILON = 1e-6


@dataclass(frozen=True)
class BenchmarkResult:
    """Material outputs of the OOS benchmark."""

    leaderboard: pd.DataFrame
    champion: str
    predictions: pd.DataFrame
    split_audit: pd.DataFrame
    profile: BenchmarkProfile
    state_order: tuple[str, ...] = STATE_ORDER
    selection_end: pd.Timestamp | None = None
    selection_leaderboard: pd.DataFrame | None = None
    holdout_leaderboard: pd.DataFrame | None = None
    selection_diagnostics: pd.DataFrame | None = None
    stacking_weights: pd.DataFrame | None = None
    multiscale_scale_forecasts: pd.DataFrame | None = None
    state_label_history: pd.DataFrame | None = None
    weekly_state_forecasts: pd.DataFrame | None = None

    def predictions_for_split(
        self,
        split: Literal["selection", "holdout", "all"] = "all",
    ) -> pd.DataFrame:
        """Return OOS forecasts for an explicit evaluation segment."""

        if split == "all":
            return self.predictions.copy().reset_index(drop=True)
        if self.selection_end is None:
            raise ValueError(
                "selection/holdout predictions require run_benchmark(selection_end=...)"
            )
        return self.predictions.loc[
            self.predictions["evaluation_split"] == split
        ].reset_index(drop=True)

    def champion_predictions(
        self,
        split: Literal["selection", "holdout", "all"] | None = None,
    ) -> pd.DataFrame:
        """Return frozen-family champion forecasts.

        When a time split exists, the safe default is holdout only.  Callers
        must request ``split="all"`` explicitly to obtain retrospective
        selection-period tracks.
        """

        resolved_split = (
            "holdout" if split is None and self.selection_end is not None else split
        )
        resolved_split = "all" if resolved_split is None else resolved_split
        predictions = self.predictions_for_split(resolved_split)
        return predictions.loc[
            predictions["model"] == self.champion
        ].reset_index(drop=True)

    def champion_holdout_predictions(self) -> pd.DataFrame:
        if self.selection_end is None:
            raise ValueError("benchmark has no frozen holdout split")
        return self.champion_predictions(split="holdout")


@dataclass(frozen=True)
class TransitionBenchmarkResult:
    """Leakage-auditable multi-horizon departure-risk benchmark outputs.

    ``predictions`` contains only origins whose event outcome is known.  The
    private forecast frame is exposed through :meth:`latest_forecasts` so an
    unlabelled prospective row cannot be accidentally mixed into evaluation.
    Model-family, calibration, and decision-threshold choices use selection
    OOS rows only; post-cutoff rows are retrospective diagnostics.
    """

    leaderboard: pd.DataFrame
    predictions: pd.DataFrame
    split_audit: pd.DataFrame
    nested_selection: pd.DataFrame
    champions_by_horizon: Mapping[int, str]
    profile: BenchmarkProfile
    selection_end: pd.Timestamp
    candidate_status: pd.DataFrame
    _latest_forecast_rows: pd.DataFrame
    _latest_candidate_forecast_rows: pd.DataFrame

    def latest_forecasts(self, horizon: int | None = None) -> pd.DataFrame:
        """Return every unresolved prospective origin.

        A horizon ``h`` has exactly its final ``h`` origins unresolved because
        their ``t+h`` event labels extend beyond the observed weekly state
        history.  These rows remain physically separate from evaluation.
        """

        frame = self._latest_forecast_rows.copy()
        if horizon is None:
            return frame.reset_index(drop=True)
        requested = int(horizon)
        available = {int(value) for value in frame["horizon"].unique()}
        if requested not in available:
            raise KeyError(
                f"horizon {requested} is unavailable; expected one of {sorted(available)}"
            )
        return frame.loc[frame["horizon"] == requested].reset_index(drop=True)

    def latest_candidate_forecasts(
        self,
        *,
        horizon: int | None = None,
        model: str | None = None,
    ) -> pd.DataFrame:
        """Return unresolved forecasts for every successfully enabled family.

        The dashboard consumes only :meth:`latest_forecasts`, whose rows are
        frozen selection champions.  This broader frame exists solely so a
        preregistered structural composition can use (and independently audit)
        a non-champion expert such as the one-week binary XGBoost hazard.
        """

        frame = self._latest_candidate_forecast_rows.copy()
        if horizon is not None:
            frame = frame.loc[frame["horizon"].eq(int(horizon))]
        if model is not None:
            frame = frame.loc[frame["model"].astype(str).eq(str(model))]
        if frame.empty:
            requested = []
            if horizon is not None:
                requested.append(f"horizon={int(horizon)}")
            if model is not None:
                requested.append(f"model={str(model)!r}")
            raise KeyError(
                "transition candidate forecast is unavailable: "
                + ", ".join(requested or ["empty candidate frame"])
            )
        return frame.reset_index(drop=True)


def _validate_inputs(features: pd.DataFrame, states: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    if not isinstance(states, pd.Series):
        raise TypeError("states must be a pandas Series")
    if not isinstance(features.index, pd.DatetimeIndex):
        raise TypeError("features must use a DatetimeIndex")
    if features.index.has_duplicates or not features.index.is_monotonic_increasing:
        raise ValueError("feature index must be unique and increasing")
    if not features.index.equals(states.index):
        raise ValueError("features and states must have exactly the same index")
    if len(features) < 3:
        raise ValueError("at least three weekly observations are required")
    invalid = sorted(set(states.dropna().astype(str)).difference(STATE_ORDER))
    if invalid:
        raise ValueError(f"states contain unsupported labels: {invalid}")
    if states.isna().any():
        raise ValueError("states must be complete before walk-forward evaluation")
    non_numeric = [
        str(column)
        for column in features.columns
        if not pd.api.types.is_numeric_dtype(features[column])
    ]
    if non_numeric:
        raise TypeError(f"all feature columns must be numeric: {non_numeric}")
    if features.shape[1] == 0:
        raise ValueError("features must contain at least one column")
    return features.astype(float), states.astype(str)


def _model_names(
    models: Iterable[str] | None,
    *,
    include_hmm: bool,
) -> tuple[str, ...]:
    names = list(MODEL_NAMES if models is None else models)
    if include_hmm and "gaussian_hmm" not in names:
        names.append("gaussian_hmm")
    supported = set(MODEL_NAMES).union({"gaussian_hmm"})
    unknown = sorted(set(names).difference(supported))
    if unknown:
        raise ValueError(f"unknown benchmark models: {unknown}")
    # Preserve order while removing accidental duplicates.
    return tuple(dict.fromkeys(names))


def _predict_learned_model(
    name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    *,
    current_train: pd.Series | None = None,
    current_test: str | None = None,
    transition_train_frame: pd.DataFrame | None = None,
    transition_test_frame: pd.DataFrame | None = None,
    profile: BenchmarkProfile,
    random_state: int,
) -> np.ndarray:
    if name == "gaussian_hmm":
        estimator = GaussianHMMChallenger(
            n_iter=profile.hmm_iterations,
            random_state=random_state,
        )
    elif name == "duration_tvtp_hurdle":
        estimator = DurationAwareTVTPHurdleClassifier(
            derive_duration=False,
            adjacent_only=True,
            hazard_C=0.05,
            destination_C=0.05,
            smoothing=1.0,
            random_state=random_state,
        )
    else:
        estimator = build_model(name, profile, random_state=random_state)
    model_train = x_train
    model_test = x_test
    if name == "duration_tvtp_hurdle":
        if transition_train_frame is None or transition_test_frame is None:
            raise ValueError(
                "duration_tvtp_hurdle requires causal transition feature frames"
            )
        model_train = transition_train_frame
        model_test = transition_test_frame
    elif name == "transition_logistic":
        if current_train is None or current_test is None:
            raise ValueError(
                "transition_logistic requires current state at train and test origins"
            )
        model_train = augment_with_current_state(x_train, current_train)
        model_test = augment_with_current_state(x_test, [current_test])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.filterwarnings(
            "ignore", message="Could not find the number of physical cores"
        )
        warnings.filterwarnings(
            "ignore", message="Skipping features without any observed values"
        )
        estimator.fit(model_train, y_train)
        raw = estimator.predict_proba(model_test)
    if name == "duration_tvtp_hurdle" and bool(
        getattr(estimator, "used_fallback_", False)
    ):
        reasons = ";".join(getattr(estimator, "fallback_reasons_", ()))
        raise ValueError(f"duration_tvtp_hurdle fit fallback: {reasons}")
    classes = getattr(estimator, "classes_", None)
    if classes is None:
        raise RuntimeError(f"{name} did not expose fitted classes_")
    return align_probabilities(raw, classes, expected_rows=len(model_test))


def _probability_prediction(probability: np.ndarray) -> str:
    return STATE_ORDER[int(np.argmax(probability))]


def multiclass_brier_score(actual: Iterable[str], probabilities: np.ndarray) -> float:
    actual_values = np.asarray(list(actual), dtype=object)
    one_hot = np.zeros((len(actual_values), len(STATE_ORDER)), dtype=float)
    positions = {state: index for index, state in enumerate(STATE_ORDER)}
    for row, value in enumerate(actual_values):
        one_hot[row, positions[str(value)]] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    actual: Iterable[str],
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """Top-label expected calibration error for a multiclass forecast."""

    actual_values = np.asarray(list(actual), dtype=object)
    confidence = probabilities.max(axis=1)
    predicted = np.asarray(
        [STATE_ORDER[index] for index in probabilities.argmax(axis=1)], dtype=object
    )
    correct = (predicted == actual_values).astype(float)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        if index == 0:
            mask = (confidence >= boundaries[index]) & (
                confidence <= boundaries[index + 1]
            )
        else:
            mask = (confidence > boundaries[index]) & (
                confidence <= boundaries[index + 1]
            )
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return float(ece)


def evaluate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Build a probability-first leaderboard from long-form OOS forecasts."""

    rows: list[dict[str, object]] = []
    for model, group in predictions.groupby("model", sort=False):
        actual = group["actual"].astype(str).to_numpy()
        probability = group[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
        probability = np.clip(probability, 1e-9, 1.0)
        probability /= probability.sum(axis=1, keepdims=True)
        predicted = np.asarray(
            [STATE_ORDER[index] for index in probability.argmax(axis=1)], dtype=object
        )
        transition_state_actual = actual == "transition"
        transition_state_predicted = predicted == "transition"
        if "current_state" in group:
            current = group["current_state"].astype(str).to_numpy()
            # Dashboard transition_probability means any change away from the
            # current regime, not specifically entry into the middle state.
            transition_actual = actual != current
            transition_predicted = predicted != current
        else:
            # Backward-compatible evaluation for externally supplied tables.
            transition_actual = transition_state_actual
            transition_predicted = transition_state_predicted
        positions = {state: index for index, state in enumerate(STATE_ORDER)}
        actual_probability = np.asarray(
            [probability[row, positions[state]] for row, state in enumerate(actual)]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            balanced = balanced_accuracy_score(actual, predicted)
        rows.append(
            {
                "model": model,
                # sklearn's newer log_loss implementation sorts labels
                # lexicographically; compute directly to preserve the shared
                # dashboard order (risk_on, transition, risk_off).
                "log_loss": float(-np.log(actual_probability).mean()),
                "brier": multiclass_brier_score(actual, probability),
                "accuracy": float(accuracy_score(actual, predicted)),
                "balanced_accuracy": float(balanced),
                "macro_f1": float(
                    f1_score(
                        actual,
                        predicted,
                        labels=STATE_ORDER,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "transition_precision": float(
                    precision_score(
                        transition_actual,
                        transition_predicted,
                        zero_division=0,
                    )
                ),
                "transition_recall": float(
                    recall_score(
                        transition_actual,
                        transition_predicted,
                        zero_division=0,
                    )
                ),
                "transition_state_precision": float(
                    precision_score(
                        transition_state_actual,
                        transition_state_predicted,
                        zero_division=0,
                    )
                ),
                "transition_state_recall": float(
                    recall_score(
                        transition_state_actual,
                        transition_state_predicted,
                        zero_division=0,
                    )
                ),
                "calibration_error": expected_calibration_error(actual, probability),
                "n_predictions": int(len(group)),
                "fallback_count": int(group["fallback"].sum()),
            }
        )
    leaderboard = pd.DataFrame(rows)
    return leaderboard.sort_values(
        ["log_loss", "calibration_error", "model"], ignore_index=True
    )


def select_champion(
    leaderboard: pd.DataFrame,
    *,
    minimum_log_loss_improvement: float = 0.001,
    calibration_tolerance: float = 0.02,
    simplicity_tolerance: float = 0.01,
) -> str:
    """Prefer a calibrated, simpler challenger only after beating a baseline."""

    if leaderboard.empty:
        raise ValueError("cannot select a champion from an empty leaderboard")
    indexed = leaderboard.set_index("model", drop=False)
    baseline_names = [name for name in BASELINE_MODELS if name in indexed.index]
    if not baseline_names:
        raise ValueError("leaderboard must contain at least one baseline")
    baseline_table = indexed.loc[baseline_names].sort_values(
        ["log_loss", "calibration_error"]
    )
    baseline = baseline_table.iloc[0]
    baseline_name = str(baseline["model"])
    baseline_loss = float(baseline["log_loss"])
    baseline_calibration = float(baseline["calibration_error"])

    challengers = leaderboard.loc[
        ~leaderboard["model"].isin(BASELINE_MODELS)
        & (leaderboard["fallback_count"] == 0)
        & (
            leaderboard["log_loss"]
            <= baseline_loss - minimum_log_loss_improvement
        )
        & (
            leaderboard["calibration_error"]
            <= baseline_calibration + calibration_tolerance
        )
    ].copy()
    if challengers.empty:
        return baseline_name
    best_loss = float(challengers["log_loss"].min())
    near_best = challengers.loc[
        challengers["log_loss"] <= best_loss + simplicity_tolerance
    ].copy()
    near_best["complexity_rank"] = near_best["model"].map(model_complexity_rank)
    near_best = near_best.sort_values(
        ["complexity_rank", "calibration_error", "log_loss", "model"]
    )
    return str(near_best.iloc[0]["model"])


def _strict_comparable_predictions(
    predictions: pd.DataFrame,
    leaderboard: pd.DataFrame,
) -> pd.DataFrame:
    """Validate that every model was scored on the same causal OOS rows.

    Model comparison is paired by target week.  Silently accepting a missing
    origin, a duplicated target, or an invalid probability would invalidate
    both the paired bootstrap and the aggregate leaderboard, so comparison
    fails loudly before any champion decision is made.
    """

    required = {
        "origin_date",
        "target_date",
        "model",
        "evaluation_split",
        "actual",
        "fallback",
        *PROBABILITY_COLUMNS,
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"selection predictions missing required columns: {missing}")
    if predictions.empty:
        raise ValueError("selection predictions must not be empty")
    if leaderboard.empty or "model" not in leaderboard:
        raise ValueError("selection leaderboard must contain models")
    if leaderboard["model"].astype(str).duplicated().any():
        raise ValueError("selection leaderboard models must be unique")

    frame = predictions.copy()
    frame["model"] = frame["model"].astype(str)
    frame["origin_date"] = pd.to_datetime(frame["origin_date"], utc=True, errors="raise")
    frame["target_date"] = pd.to_datetime(frame["target_date"], utc=True, errors="raise")
    if frame.duplicated(["model", "target_date"]).any():
        raise ValueError("selection predictions contain duplicate model/target rows")

    prediction_models = set(frame["model"])
    leaderboard_models = set(leaderboard["model"].astype(str))
    if prediction_models != leaderboard_models:
        raise ValueError(
            "selection predictions and leaderboard must contain identical models"
        )
    if not frame["actual"].astype(str).isin(STATE_ORDER).all():
        raise ValueError("selection predictions contain unsupported actual states")

    probability = frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(probability).all():
        raise ValueError("selection predictions contain non-finite probabilities")
    if ((probability < 0.0) | (probability > 1.0)).any():
        raise ValueError("selection probabilities must be within [0, 1]")
    if not np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("selection probabilities must sum to one")
    valid_fallback = frame["fallback"].map(
        lambda value: isinstance(value, (bool, np.bool_))
        or (isinstance(value, (int, np.integer)) and int(value) in (0, 1))
    )
    if not valid_fallback.all():
        raise ValueError("selection fallback flags must be boolean")
    frame["fallback"] = frame["fallback"].astype(bool)

    ordered_models = leaderboard["model"].astype(str).tolist()
    comparison_columns = ["origin_date", "actual"]
    if "current_state" in frame:
        comparison_columns.append("current_state")
    reference = (
        frame.loc[frame["model"] == ordered_models[0]]
        .sort_values("target_date")
        .set_index("target_date")[comparison_columns]
    )
    for model in ordered_models[1:]:
        candidate = (
            frame.loc[frame["model"] == model]
            .sort_values("target_date")
            .set_index("target_date")[comparison_columns]
        )
        if not candidate.index.equals(reference.index):
            raise ValueError(
                f"selection model {model!r} does not share identical target origins"
            )
        if not candidate.equals(reference):
            raise ValueError(
                f"selection model {model!r} does not share identical OOS outcomes"
            )

    recomputed = evaluate_predictions(frame).set_index("model")
    supplied = leaderboard.assign(model=leaderboard["model"].astype(str)).set_index(
        "model"
    )
    for model in ordered_models:
        for metric in ("log_loss", "brier"):
            if metric not in supplied:
                raise ValueError(f"selection leaderboard missing {metric}")
            if not np.isclose(
                float(supplied.loc[model, metric]),
                float(recomputed.loc[model, metric]),
                rtol=1e-8,
                atol=1e-10,
            ):
                raise ValueError(
                    f"selection leaderboard {metric} does not match predictions for {model}"
                )
        for metric in ("n_predictions", "fallback_count"):
            if metric not in supplied:
                raise ValueError(f"selection leaderboard missing {metric}")
            if int(supplied.loc[model, metric]) != int(recomputed.loc[model, metric]):
                raise ValueError(
                    f"selection leaderboard {metric} does not match predictions for {model}"
                )
    return frame.sort_values(["target_date", "model"], ignore_index=True)


def _holm_adjusted_pvalues(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Return deterministic Holm family-wise adjusted p-values."""

    if not pvalues:
        return {}
    ordered = sorted(pvalues.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running_maximum = 0.0
    hypothesis_count = len(ordered)
    for rank, (model, pvalue) in enumerate(ordered):
        candidate = min(1.0, float(pvalue) * (hypothesis_count - rank))
        running_maximum = max(running_maximum, candidate)
        adjusted[model] = running_maximum
    return adjusted


def _moving_block_bootstrap_pvalues(
    improvements: Mapping[str, np.ndarray],
    *,
    block_length: int,
    resamples: int,
    random_state: int,
) -> tuple[dict[str, float], int]:
    """Test positive paired mean-loss improvements with common block draws."""

    if not improvements:
        return {}, 0
    if block_length < 1 or resamples < 1:
        raise ValueError("bootstrap block_length and resamples must be positive")
    lengths = {len(np.asarray(values)) for values in improvements.values()}
    if len(lengths) != 1:
        raise ValueError("paired loss improvements must have equal lengths")
    observation_count = lengths.pop()
    if observation_count < 1:
        raise ValueError("paired loss improvements must not be empty")

    # The nominal production block is 13 weeks.  Tiny quick-profile smoke tests
    # use a shorter effective block so a circular block is not always the entire
    # sample and the bootstrap remains a meaningful deterministic diagnostic.
    effective_block = min(block_length, max(1, observation_count // 2))
    blocks_per_sample = int(np.ceil(observation_count / effective_block))
    generator = np.random.default_rng(random_state)
    starts = generator.integers(
        0,
        observation_count,
        size=(resamples, blocks_per_sample),
    )
    offsets = np.arange(effective_block)
    indices = (starts[..., np.newaxis] + offsets) % observation_count
    indices = indices.reshape(resamples, -1)[:, :observation_count]

    pvalues: dict[str, float] = {}
    for model, values in improvements.items():
        differential = np.asarray(values, dtype=float)
        if not np.isfinite(differential).all():
            raise ValueError(f"non-finite paired losses for model {model}")
        observed = float(differential.mean())
        centred = differential - observed
        null_means = centred[indices].mean(axis=1)
        pvalues[model] = float(
            (1 + np.count_nonzero(null_means >= observed)) / (resamples + 1)
        )
    return pvalues, effective_block


def select_champion_with_diagnostics(
    leaderboard: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    minimum_log_loss_improvement: float = SELECTION_MINIMUM_LOG_LOSS_IMPROVEMENT,
    brier_tolerance: float = SELECTION_BRIER_TOLERANCE,
    alpha: float = SELECTION_ALPHA,
    bootstrap_block_weeks: int = SELECTION_BOOTSTRAP_BLOCK_WEEKS,
    bootstrap_resamples: int = SELECTION_BOOTSTRAP_RESAMPLES,
    random_state: int = SELECTION_BOOTSTRAP_SEED,
    simplicity_tolerance: float = 0.01,
) -> tuple[str, pd.DataFrame]:
    """Select a conservative pre-holdout champion and retain gate evidence.

    Every learned challenger is compared with the best probability baseline on
    identical selection weeks.  The gate requires material mean log-loss
    improvement, a Holm-adjusted one-sided paired moving-block-bootstrap test,
    no fallback forecasts, and no material Brier-score degradation.  No
    post-selection/holdout row is accepted by this function.
    """

    if minimum_log_loss_improvement < 0 or brier_tolerance < 0:
        raise ValueError("selection materiality thresholds must be non-negative")
    if not 0 < alpha < 1:
        raise ValueError("selection alpha must be between zero and one")
    frame = _strict_comparable_predictions(predictions, leaderboard)
    if "evaluation_split" in frame and not frame["evaluation_split"].eq(
        "selection"
    ).all():
        raise ValueError("champion selection accepts selection predictions only")

    table = leaderboard.assign(model=leaderboard["model"].astype(str)).copy()
    indexed = table.set_index("model", drop=False)
    baseline_names = [name for name in BASELINE_MODELS if name in indexed.index]
    if not baseline_names:
        raise ValueError("selection leaderboard must contain at least one baseline")
    reference_row = (
        indexed.loc[baseline_names]
        .reset_index(drop=True)
        .sort_values(["log_loss", "calibration_error", "model"])
        .iloc[0]
    )
    reference_model = str(reference_row["model"])
    reference_log_loss = float(reference_row["log_loss"])
    reference_brier = float(reference_row["brier"])

    loss_by_model: dict[str, np.ndarray] = {}
    for model, group in frame.groupby("model", sort=False):
        ordered = group.sort_values("target_date")
        actual = ordered["actual"].astype(str).to_numpy()
        probability = ordered[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
        positions = {state: index for index, state in enumerate(STATE_ORDER)}
        actual_probability = np.asarray(
            [probability[row, positions[state]] for row, state in enumerate(actual)]
        )
        loss_by_model[str(model)] = -np.log(actual_probability)

    challenger_names = [
        model for model in table["model"].tolist() if model not in BASELINE_MODELS
    ]
    paired_improvements = {
        model: loss_by_model[reference_model] - loss_by_model[model]
        for model in challenger_names
    }
    raw_pvalues, effective_block = _moving_block_bootstrap_pvalues(
        paired_improvements,
        block_length=bootstrap_block_weeks,
        resamples=bootstrap_resamples,
        random_state=random_state,
    )
    adjusted_pvalues = _holm_adjusted_pvalues(raw_pvalues)

    diagnostic_rows: list[dict[str, object]] = []
    passing_challengers: list[str] = []
    for model in table["model"].tolist():
        row = indexed.loc[model]
        improvement = reference_log_loss - float(row["log_loss"])
        brier_difference = float(row["brier"]) - reference_brier
        fallback_count = int(row["fallback_count"])
        is_reference = model == reference_model
        is_challenger = model not in BASELINE_MODELS
        failures: list[str] = []
        if is_challenger:
            if fallback_count != 0:
                failures.append("fallback_present")
            if improvement + 1e-12 < minimum_log_loss_improvement:
                failures.append("insufficient_log_loss_improvement")
            if adjusted_pvalues[model] > alpha:
                failures.append("holm_not_significant")
            if brier_difference > brier_tolerance + 1e-12:
                failures.append("brier_degradation")
            if not failures:
                passing_challengers.append(model)
        elif not is_reference:
            failures.append("non_reference_baseline")

        diagnostic_rows.append(
            {
                "model": model,
                "reference_model": reference_model,
                "is_reference": is_reference,
                "selected": False,
                "gate_passed": bool(is_reference or (is_challenger and not failures)),
                "gate_reason": "passed" if not failures else ";".join(failures),
                "log_loss": float(row["log_loss"]),
                "reference_log_loss": reference_log_loss,
                "absolute_log_loss_improvement": float(improvement),
                "brier": float(row["brier"]),
                "reference_brier": reference_brier,
                "brier_difference": float(brier_difference),
                "fallback_count": fallback_count,
                "raw_p_value": raw_pvalues.get(model, np.nan),
                "holm_adjusted_p_value": adjusted_pvalues.get(model, np.nan),
                "n_predictions": int(row["n_predictions"]),
                "bootstrap_block_weeks": int(bootstrap_block_weeks),
                "bootstrap_effective_block_weeks": int(effective_block),
                "bootstrap_resamples": int(bootstrap_resamples),
                "bootstrap_seed": int(random_state),
                "alpha": float(alpha),
                "minimum_log_loss_improvement": float(
                    minimum_log_loss_improvement
                ),
                "brier_tolerance": float(brier_tolerance),
            }
        )

    champion = reference_model
    if passing_challengers:
        passing = indexed.loc[passing_challengers].copy()
        best_loss = float(passing["log_loss"].min())
        near_best = passing.loc[
            passing["log_loss"] <= best_loss + simplicity_tolerance
        ].copy()
        near_best["complexity_rank"] = near_best["model"].map(
            model_complexity_rank
        )
        near_best = near_best.reset_index(drop=True).sort_values(
            ["complexity_rank", "calibration_error", "log_loss", "model"]
        )
        champion = str(near_best.iloc[0]["model"])

    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics["selected"] = diagnostics["model"].eq(champion)
    return champion, diagnostics


def _coerce_selection_end(
    value: str | pd.Timestamp,
    index: pd.DatetimeIndex,
) -> pd.Timestamp:
    cutoff = pd.Timestamp(value)
    if index.tz is None:
        if cutoff.tzinfo is not None:
            cutoff = cutoff.tz_localize(None)
    elif cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize(index.tz)
    else:
        cutoff = cutoff.tz_convert(index.tz)
    return cutoff


def _time_split_test_positions(
    all_positions: list[int],
    index: pd.DatetimeIndex,
    *,
    selection_end: pd.Timestamp,
    maximum_origins: int | None,
    selection_max_origins: int | None,
    minimum_selection_predictions: int,
    minimum_holdout_predictions: int,
) -> tuple[list[int], set[int], set[int]]:
    """Allocate independent selection and holdout origin budgets."""

    if minimum_selection_predictions < 1 or minimum_holdout_predictions < 1:
        raise ValueError("minimum split prediction counts must be positive")
    if selection_max_origins is not None and selection_max_origins < 1:
        raise ValueError("selection_max_origins must be positive or None")
    selection_available = [
        position
        for position in all_positions
        if pd.Timestamp(index[position + 1]) < selection_end
    ]
    holdout_available = [
        position
        for position in all_positions
        if pd.Timestamp(index[position + 1]) >= selection_end
    ]
    if len(selection_available) < minimum_selection_predictions:
        raise ValueError(
            "insufficient selection OOS predictions before selection_end: "
            f"{len(selection_available)} < {minimum_selection_predictions}"
        )
    if len(holdout_available) < minimum_holdout_predictions:
        raise ValueError(
            "insufficient holdout OOS predictions at/after selection_end: "
            f"{len(holdout_available)} < {minimum_holdout_predictions}"
        )

    if selection_max_origins is None:
        selected_positions = selection_available
    else:
        selection_budget = max(minimum_selection_predictions, selection_max_origins)
        selection_budget = min(selection_budget, len(selection_available))
        selected_positions = selection_available[-selection_budget:]

    if maximum_origins is None:
        holdout_positions = holdout_available
    else:
        if maximum_origins < minimum_holdout_predictions:
            raise ValueError(
                "profile max_origins cannot cover the holdout minimum: "
                f"{maximum_origins} < {minimum_holdout_predictions}"
            )
        holdout_budget = min(len(holdout_available), maximum_origins)
        holdout_positions = holdout_available[-holdout_budget:]

    combined = sorted([*selected_positions, *holdout_positions])
    return combined, set(selected_positions), set(holdout_positions)


def run_benchmark(
    features: pd.DataFrame,
    states: pd.Series,
    *,
    profile: str | BenchmarkProfile = "quick",
    models: Iterable[str] | None = None,
    include_hmm: bool = False,
    gap: int = 1,
    minimum_train_weeks: int | None = None,
    random_state: int = 17,
    selection_end: str | pd.Timestamp | None = None,
    selection_max_origins: int | None = None,
    model_workers: int = 1,
    minimum_selection_predictions: int = 12,
    minimum_holdout_predictions: int = 12,
    progress: Callable[[str], None] | None = None,
    checkpoint_directory: str | Path | None = None,
    source_fingerprint_sha256: str | None = None,
) -> BenchmarkResult:
    """Compare non-DL models with an expanding, one-week-purged walk-forward.

    Each row at origin ``t`` predicts the label at ``t+1``.  With the default
    ``gap=1`` the origin immediately preceding the test origin is excluded from
    fitting.  Every imputer, scaler, calibration model, and estimator is newly
    fit inside each outer training fold.

    When ``selection_end`` is supplied, only targets strictly before that date
    select the champion.  Targets on/after the date are excluded from every
    selection calculation and evaluated separately as diagnostics.  For
    bounded profiles ``max_origins`` cap diagnostic rows independently.  By
    default standard/full profiles retain every available pre-cutoff selection
    origin.  Quick profiles keep a three-origin smoke reserve (or the requested
    minimum, when larger) unless ``selection_max_origins`` is explicit.
    """

    features, states = _validate_inputs(features, states)
    if checkpoint_directory is None and source_fingerprint_sha256 is not None:
        raise ValueError(
            "source_fingerprint_sha256 requires checkpoint_directory"
        )
    if gap < 0:
        raise ValueError("gap must be non-negative")
    if model_workers < 1:
        raise ValueError("model_workers must be positive")
    cfg = resolve_profile(profile)
    if minimum_train_weeks is not None:
        if minimum_train_weeks < 12:
            raise ValueError("minimum_train_weeks must be at least 12")
        cfg = cfg.with_overrides(minimum_train_weeks=minimum_train_weeks)
    names = _model_names(models, include_hmm=include_hmm)
    require_model_dependencies(names)
    if not any(name in BASELINE_MODELS for name in names):
        raise ValueError("models must include at least one baseline")
    transition_features: pd.DataFrame | None = None
    if "duration_tvtp_hurdle" in names:
        transition_features = derive_causal_transition_features(features, states)

    # There are len(features)-1 supervised origins because the final state's
    # t+1 label is not yet known.
    supervised_count = len(features) - 1
    first_test_position = cfg.minimum_train_weeks + gap
    if first_test_position >= supervised_count:
        raise ValueError(
            "not enough observations for requested training window and gap: "
            f"need > {first_test_position + 1}, got {len(features)}"
        )
    all_test_positions = list(range(first_test_position, supervised_count))
    cutoff: pd.Timestamp | None = None
    selection_positions: set[int] = set()
    holdout_positions: set[int] = set()
    if selection_end is None:
        test_positions = all_test_positions
        if cfg.max_origins is not None:
            test_positions = test_positions[-cfg.max_origins :]
    else:
        cutoff = _coerce_selection_end(selection_end, features.index)
        resolved_selection_max_origins = selection_max_origins
        if resolved_selection_max_origins is None and cfg.name == "quick":
            resolved_selection_max_origins = max(3, minimum_selection_predictions)
        (
            test_positions,
            selection_positions,
            holdout_positions,
        ) = _time_split_test_positions(
            all_test_positions,
            features.index,
            selection_end=cutoff,
            maximum_origins=cfg.max_origins,
            selection_max_origins=resolved_selection_max_origins,
            minimum_selection_predictions=minimum_selection_predictions,
            minimum_holdout_predictions=minimum_holdout_predictions,
        )

    next_states = states.shift(-1)
    prediction_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    total_origins = len(test_positions)
    checkpoint = None
    cached_origins: dict[int, Any] = {}
    if checkpoint_directory is not None:
        # Kept lazy so the default V4 path does not import or initialize the
        # private V5 checkpoint subsystem.
        from regime_lab.walkforward_checkpoint import (
            BenchmarkCheckpointIdentity,
            ResolvedBenchmarkParameters,
            WalkForwardCheckpoint,
        )

        checkpoint_parameters = ResolvedBenchmarkParameters.from_arguments(
            profile=cfg,
            models=names,
            include_hmm=include_hmm,
            gap=gap,
            random_state=random_state,
            selection_end=cutoff,
            selection_max_origins=selection_max_origins,
            model_workers=model_workers,
            minimum_selection_predictions=minimum_selection_predictions,
            minimum_holdout_predictions=minimum_holdout_predictions,
        )
        checkpoint_identity = BenchmarkCheckpointIdentity.build(
            features,
            states,
            checkpoint_parameters,
            source_fingerprint_sha256=source_fingerprint_sha256,
        )
        expected_origin_dates = tuple(
            pd.Timestamp(features.index[position]) for position in test_positions
        )
        checkpoint_origin_dates = tuple(
            origin.origin_date for origin in checkpoint_identity.origins
        )
        if checkpoint_origin_dates != expected_origin_dates:
            raise RuntimeError(
                "checkpoint origin resolver differs from run_benchmark"
            )
        checkpoint = WalkForwardCheckpoint.open(
            checkpoint_directory,
            checkpoint_identity,
        )
        cached_origins = {
            record.origin.sequence: record
            for record in checkpoint.load_completed_origins()
        }
    checkpoint_count = min(21, total_origins)
    progress_checkpoints = set(
        np.rint(np.linspace(1, total_origins, checkpoint_count)).astype(int)
    )
    for origin_number, test_position in enumerate(test_positions, start=1):
        if progress is not None and origin_number in progress_checkpoints:
            progress_origin = pd.Timestamp(features.index[test_position])
            progress_target = pd.Timestamp(features.index[test_position + 1])
            progress(
                f"walk-forward {origin_number}/{total_origins}: "
                f"{progress_origin.date().isoformat()} → "
                f"{progress_target.date().isoformat()}"
            )
        cached = cached_origins.get(origin_number)
        if cached is not None:
            cached_split = dict(cached.split_audit)
            if cached_split["first_purged_origin"] is None:
                cached_split["first_purged_origin"] = pd.NaT
            split_rows.append(cached_split)
            prediction_rows.extend(
                dict(row) for row in cached.prediction_rows
            )
            continue
        train_stop = test_position - gap
        x_train = features.iloc[:train_stop]
        y_train = next_states.iloc[:train_stop].astype(str)
        current_train = states.iloc[:train_stop].astype(str)
        x_test = features.iloc[[test_position]]
        transition_train_frame = (
            None
            if transition_features is None
            else transition_features.iloc[:train_stop]
        )
        transition_test_frame = (
            None
            if transition_features is None
            else transition_features.iloc[[test_position]]
        )
        current_test = str(states.iloc[test_position])
        actual = str(states.iloc[test_position + 1])
        origin_date = pd.Timestamp(features.index[test_position])
        target_date = pd.Timestamp(features.index[test_position + 1])
        evaluation_split = (
            "legacy"
            if cutoff is None
            else ("selection" if test_position in selection_positions else "holdout")
        )

        purged = features.index[train_stop:test_position]
        split_row = {
            "origin_date": origin_date,
            "target_date": target_date,
            "train_size": int(train_stop),
            "train_start": pd.Timestamp(features.index[0]),
            "last_train_origin": pd.Timestamp(features.index[train_stop - 1]),
            "last_train_target": pd.Timestamp(features.index[train_stop]),
            "purged_origin_count": int(len(purged)),
            "first_purged_origin": (
                pd.Timestamp(purged[0]) if len(purged) else pd.NaT
            ),
            "gap": int(gap),
            "evaluation_split": evaluation_split,
        }

        def evaluate_name(name: str) -> dict[str, object]:
            fallback = False
            fallback_reason = ""
            try:
                if name == "majority":
                    probability = majority_probabilities(y_train)
                elif name == "persistence":
                    probability = persistence_probabilities(current_test)
                elif name == "markov":
                    probability = (
                        SmoothedMarkovClassifier(alpha=1.0)
                        .fit(current_train, y_train)
                        .predict_proba([current_test])[0]
                    )
                else:
                    probability = _predict_learned_model(
                        name,
                        x_train,
                        y_train,
                        x_test,
                        current_train=current_train,
                        current_test=current_test,
                        transition_train_frame=transition_train_frame,
                        transition_test_frame=transition_test_frame,
                        profile=cfg,
                        random_state=random_state,
                    )[0]
            except (ImportError, ValueError, RuntimeError, FloatingPointError) as exc:
                # A rare class can make a fold-specific calibration impossible;
                # the benchmark remains complete and makes degradation explicit.
                probability = class_prior_probabilities(y_train)
                fallback = True
                fallback_reason = f"{type(exc).__name__}: {exc}"
            probability = np.asarray(probability, dtype=float)
            probability = np.clip(probability, 1e-9, 1.0)
            probability /= probability.sum()
            return {
                    "origin_date": origin_date,
                    "target_date": target_date,
                    "model": name,
                    "evaluation_split": evaluation_split,
                    "current_state": current_test,
                    "actual": actual,
                    "predicted": _probability_prediction(probability),
                    **{
                        column: float(probability[index])
                        for index, column in enumerate(PROBABILITY_COLUMNS)
                    },
                    "train_size": int(train_stop),
                    "gap": int(gap),
                    "fallback": fallback,
                    "fallback_reason": fallback_reason,
                }

        if model_workers == 1 or len(names) == 1:
            origin_predictions = [evaluate_name(name) for name in names]
        else:
            with ThreadPoolExecutor(
                max_workers=min(model_workers, len(names)),
                thread_name_prefix="regime-model",
            ) as executor:
                # executor.map preserves registry order, keeping artifact row
                # ordering deterministic while model fits remain independent.
                origin_predictions = list(executor.map(evaluate_name, names))
        if checkpoint is not None:
            checkpoint.save_origin(
                origin_number,
                origin_predictions,
                split_row,
            )
        split_rows.append(split_row)
        prediction_rows.extend(origin_predictions)

    predictions = pd.DataFrame(prediction_rows)
    split_audit = pd.DataFrame(split_rows)
    selection_leaderboard: pd.DataFrame | None = None
    holdout_leaderboard: pd.DataFrame | None = None
    selection_diagnostics: pd.DataFrame | None = None
    if cutoff is None:
        leaderboard = evaluate_predictions(predictions)
        champion = select_champion(leaderboard)
        leaderboard.insert(
            1, "selected", leaderboard["model"].astype(str).eq(champion)
        )
    else:
        selection_predictions = predictions.loc[
            predictions["evaluation_split"] == "selection"
        ]
        holdout_predictions = predictions.loc[
            predictions["evaluation_split"] == "holdout"
        ]
        selection_leaderboard = evaluate_predictions(selection_predictions)
        champion, selection_diagnostics = select_champion_with_diagnostics(
            selection_leaderboard,
            selection_predictions,
            random_state=random_state,
        )
        selection_leaderboard.insert(
            1,
            "selected",
            selection_leaderboard["model"].astype(str).eq(champion),
        )
        holdout_leaderboard = evaluate_predictions(holdout_predictions)
        holdout_leaderboard.insert(
            1,
            "selected",
            holdout_leaderboard["model"].astype(str).eq(champion),
        )

        selection_metrics = selection_leaderboard.drop(columns=["selected"]).rename(
            columns={
                column: f"selection_{column}"
                for column in selection_leaderboard.columns
                if column not in {"model", "selected"}
            }
        )
        leaderboard = holdout_leaderboard.merge(
            selection_metrics, on="model", how="left", validate="one_to_one"
        )
        leaderboard.insert(2, "evaluation_split", "holdout")
    return BenchmarkResult(
        leaderboard=leaderboard,
        champion=champion,
        predictions=predictions,
        split_audit=split_audit,
        profile=cfg,
        selection_end=cutoff,
        selection_leaderboard=selection_leaderboard,
        holdout_leaderboard=holdout_leaderboard,
        selection_diagnostics=selection_diagnostics,
    )


def forecast_next_regime(
    features: pd.DataFrame,
    states: pd.Series,
    *,
    champion_name: str = "elastic_net_logistic",
    as_of: str | pd.Timestamp | None = None,
    profile: str | BenchmarkProfile = "quick",
    gap: int = 1,
    minimum_train_weeks: int | None = None,
    random_state: int = 17,
) -> pd.Series:
    """Fit through an as-of date and return state-order-aligned t+1 chances.

    An arbitrary calendar date resolves to the latest completed weekly row at
    or before it.  The returned Series always has exactly three values ordered
    ``risk_on, transition, risk_off`` and summing to one.
    """

    features, states = _validate_inputs(features, states)
    if gap < 0:
        raise ValueError("gap must be non-negative")
    cutoff = features.index[-1] if as_of is None else pd.Timestamp(as_of)
    eligible = features.index[features.index <= cutoff]
    if eligible.empty:
        raise KeyError("as_of precedes the first completed weekly observation")
    origin_date = pd.Timestamp(eligible[-1])
    test_position = int(features.index.get_loc(origin_date))
    train_stop = test_position - gap
    cfg = resolve_profile(profile)
    required_history = (
        cfg.minimum_train_weeks
        if minimum_train_weeks is None
        else int(minimum_train_weeks)
    )
    if required_history < 12:
        raise ValueError("minimum_train_weeks must be at least 12")
    if train_stop < required_history:
        raise ValueError(
            "not enough history to forecast at as_of: "
            f"{train_stop} < {required_history}"
        )
    next_states = states.shift(-1)
    x_train = features.iloc[:train_stop]
    y_train = next_states.iloc[:train_stop].astype(str)
    current_train = states.iloc[:train_stop].astype(str)
    current_test = str(states.iloc[test_position])
    x_test = features.iloc[[test_position]]
    transition_features = (
        derive_causal_transition_features(features, states)
        if champion_name == "duration_tvtp_hurdle"
        else None
    )
    fallback = False
    fallback_reason = ""

    if champion_name == "majority":
        probability = majority_probabilities(y_train)
    elif champion_name == "persistence":
        probability = persistence_probabilities(current_test)
    elif champion_name == "markov":
        probability = (
            SmoothedMarkovClassifier()
            .fit(current_train, y_train)
            .predict_proba([current_test])[0]
        )
    else:
        try:
            probability = _predict_learned_model(
                champion_name,
                x_train,
                y_train,
                x_test,
                current_train=current_train,
                current_test=current_test,
                transition_train_frame=(
                    None
                    if transition_features is None
                    else transition_features.iloc[:train_stop]
                ),
                transition_test_frame=(
                    None
                    if transition_features is None
                    else transition_features.iloc[[test_position]]
                ),
                profile=cfg,
                random_state=random_state,
            )[0]
        except (ImportError, ValueError, RuntimeError, FloatingPointError) as exc:
            probability = class_prior_probabilities(y_train)
            fallback = True
            fallback_reason = f"{type(exc).__name__}: {exc}"
    probability = np.asarray(probability, dtype=float)
    probability = np.clip(probability, 1e-9, 1.0)
    probability /= probability.sum()
    result = pd.Series(probability, index=STATE_ORDER, name="next_week_probability")
    result.attrs.update(
        {
            "as_of": origin_date.isoformat(),
            "champion": champion_name,
            "gap": gap,
            "state_order": STATE_ORDER,
            "fallback": fallback,
            "fallback_reason": fallback_reason,
        }
    )
    return result


def _transition_targets(
    states: pd.Series,
    horizon: int,
) -> tuple[pd.Series, pd.Series]:
    """Build causal departure-event and first-destination targets.

    At origin ``t`` the event is true when any state in ``t+1 .. t+h``
    differs from ``S_t``.  The pseudo destination is the first departed state,
    or ``S_t`` when no departure occurs.  The latter lets the multiclass TVTP
    hurdle primitive estimate the exact binary event probability as
    ``1 - P(pseudo_destination == S_t)`` without replacing a leave-and-return
    event by the terminal state.
    """

    if isinstance(horizon, bool) or int(horizon) != horizon or int(horizon) < 1:
        raise ValueError("transition horizons must be positive integers")
    resolved = int(horizon)
    if resolved >= len(states):
        raise ValueError("transition horizon must be shorter than the state history")
    values = states.to_numpy(dtype=object)
    event = np.zeros(len(states) - resolved, dtype=bool)
    destination = np.empty(len(event), dtype=object)
    for origin in range(len(event)):
        current = str(values[origin])
        future = values[origin + 1 : origin + resolved + 1]
        departed = np.flatnonzero(future != current)
        event[origin] = bool(len(departed))
        destination[origin] = (
            str(future[int(departed[0])]) if len(departed) else current
        )
    index = states.index[: len(event)]
    return (
        pd.Series(event, index=index, name="actual_change", dtype=bool),
        pd.Series(destination, index=index, name="first_departure_state", dtype="object"),
    )


def _transition_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.drop(columns=[CURRENT_STATE_COLUMN]).copy()
    current = frame[CURRENT_STATE_COLUMN].astype(str)
    for state in STATE_ORDER:
        output[f"current_state__{state}"] = current.eq(state).astype(float)
    return output.astype(float).replace([np.inf, -np.inf], np.nan)


def _transition_candidate_names(
    models: Iterable[str] | None,
    *,
    include_xgboost: bool,
    include_joint_survival: bool,
) -> tuple[tuple[str, ...], pd.DataFrame]:
    names = list(TRANSITION_MODELS if models is None else models)
    if include_xgboost and "binary_xgboost" not in names:
        names.append("binary_xgboost")
    if include_joint_survival and "joint_survival_hazard" not in names:
        names.append("joint_survival_hazard")
    names = list(dict.fromkeys(str(name) for name in names))
    supported = set(TRANSITION_MODELS).union(
        _TRANSITION_OPTIONAL_MODELS, {"joint_survival_hazard"}
    )
    unknown = sorted(set(names).difference(supported))
    if unknown:
        raise ValueError(f"unknown transition benchmark models: {unknown}")

    rows: list[dict[str, object]] = []
    available: list[str] = []
    for name in names:
        reason = ""
        usable = True
        if name in {"binary_xgboost", "joint_survival_hazard"}:
            try:
                importlib.import_module("xgboost")
            except (ImportError, OSError) as exc:  # pragma: no cover - runtime dependent
                usable = False
                reason = f"{type(exc).__name__}: {exc}"
        if usable:
            available.append(name)
        rows.append(
            {
                "model": name,
                "requested": True,
                "available": usable,
                "published": usable,
                "selection_eligible": name != "joint_survival_hazard",
                "role": (
                    "shadow_coherence_benchmark"
                    if name == "joint_survival_hazard"
                    else "candidate"
                ),
                "reason": reason,
            }
        )
    if not available:
        raise ImportError("no requested transition benchmark model is available")
    return tuple(available), pd.DataFrame(rows)


def _regularized_binary_pipeline(*, random_state: int) -> Pipeline:
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
                    C=0.025,
                    solver="lbfgs",
                    max_iter=2_000,
                    tol=1e-7,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _binary_xgboost_pipeline(
    profile: BenchmarkProfile,
    *,
    random_state: int,
) -> Pipeline:
    from xgboost import XGBClassifier

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
            (
                "classifier",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_estimators=min(int(profile.xgboost_trees), 120),
                    max_depth=3,
                    learning_rate=0.04,
                    min_child_weight=8.0,
                    subsample=0.85,
                    colsample_bytree=0.65,
                    reg_alpha=1.0,
                    reg_lambda=8.0,
                    tree_method="hist",
                    n_jobs=1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _smoothed_binary_probability(target: pd.Series) -> float:
    return float((target.astype(int).sum() + 1.0) / (len(target) + 2.0))


def _fit_transition_candidate(
    name: str,
    augmented: pd.DataFrame,
    numeric: pd.DataFrame,
    event_target: pd.Series,
    one_week_event_target: pd.Series,
    destination_target: pd.Series,
    *,
    horizon: int,
    train_stop: int,
    test_position: int,
    profile: BenchmarkProfile,
    random_state: int,
) -> tuple[float, bool, str, float | None]:
    x_train = numeric.iloc[:train_stop]
    y_train = event_target.iloc[:train_stop]
    current_train = augmented[CURRENT_STATE_COLUMN].iloc[:train_stop].astype(str)
    current_test = str(augmented[CURRENT_STATE_COLUMN].iloc[test_position])
    fallback_probability = _smoothed_binary_probability(y_train)
    fallback = False
    fallback_reason = ""
    one_week_hazard: float | None = None
    try:
        if name == "empirical_hazard":
            probability = fallback_probability
        elif name == "markov_hazard":
            one_step_target = (
                augmented[CURRENT_STATE_COLUMN].shift(-1).iloc[:train_stop].astype(str)
            )
            markov_probability = (
                SmoothedMarkovClassifier(alpha=1.0)
                .fit(current_train, one_step_target)
                .predict_proba([current_test])[0]
            )
            stay_probability = float(
                markov_probability[STATE_ORDER.index(current_test)]
            )
            # Under the fitted homogeneous Markov baseline, avoiding every
            # departure for h steps requires h consecutive self transitions.
            probability = 1.0 - stay_probability ** int(horizon)
        elif name == "duration_tvtp_hurdle":
            estimator = DurationAwareTVTPHurdleClassifier(
                derive_duration=False,
                adjacent_only=True,
                hazard_C=0.05,
                destination_C=0.05,
                smoothing=1.0,
                random_state=random_state,
            )
            train_frame = augmented.iloc[:train_stop]
            estimator.fit(train_frame, destination_target.iloc[:train_stop])
            probability_vector = estimator.predict_proba(
                augmented.iloc[[test_position]]
            )[0]
            probability = 1.0 - float(
                probability_vector[STATE_ORDER.index(current_test)]
            )
            reasons = tuple(getattr(estimator, "fallback_reasons_", ()))
            fallback = bool(getattr(estimator, "used_fallback_", False))
            fallback_reason = ";".join(reasons)
        elif name == "regularized_logistic":
            if y_train.nunique() < 2:
                raise ValueError("binary target has only one observed class")
            estimator = _regularized_binary_pipeline(random_state=random_state)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                estimator.fit(x_train, y_train.astype(int))
            probability = float(
                estimator.predict_proba(numeric.iloc[[test_position]])[0, 1]
            )
        elif name == "binary_xgboost":
            if y_train.nunique() < 2:
                raise ValueError("binary target has only one observed class")
            estimator = _binary_xgboost_pipeline(profile, random_state=random_state)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                estimator.fit(x_train, y_train.astype(int))
            probability = float(
                estimator.predict_proba(numeric.iloc[[test_position]])[0, 1]
            )
        elif name == "joint_survival_hazard":
            one_week_train = one_week_event_target.iloc[:train_stop]
            if one_week_train.nunique() < 2:
                raise ValueError("one-week binary target has only one observed class")
            estimator = _binary_xgboost_pipeline(profile, random_state=random_state)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                estimator.fit(x_train, one_week_train.astype(int))
            one_week_hazard = float(
                estimator.predict_proba(numeric.iloc[[test_position]])[0, 1]
            )
            probability = 1.0 - (1.0 - one_week_hazard) ** int(horizon)
        else:  # pragma: no cover - guarded by candidate resolution
            raise ValueError(f"unknown transition model: {name}")
    except (ImportError, OSError, ValueError, RuntimeError, FloatingPointError) as exc:
        probability = fallback_probability
        fallback = True
        fallback_reason = f"{type(exc).__name__}: {exc}"
    probability = float(np.clip(probability, _TRANSITION_EPSILON, 1.0 - _TRANSITION_EPSILON))
    if one_week_hazard is not None:
        one_week_hazard = float(
            np.clip(one_week_hazard, _TRANSITION_EPSILON, 1.0 - _TRANSITION_EPSILON)
        )
    return probability, fallback, fallback_reason, one_week_hazard


def _calibrate_transition_probability(
    raw_probability: float,
    history: pd.DataFrame,
    *,
    minimum_rows: int,
    random_state: int,
) -> tuple[float, str, bool, str]:
    if len(history) < minimum_rows:
        return (
            raw_probability,
            "identity",
            True,
            f"insufficient_prequential_rows:{len(history)}<{minimum_rows}",
        )
    target = history["actual_change"].astype(int)
    if target.nunique() < 2 or int(target.sum()) < 3 or int((1 - target).sum()) < 3:
        return raw_probability, "identity", True, "insufficient_event_classes"
    raw = np.clip(
        history["raw_p_change"].to_numpy(dtype=float),
        _TRANSITION_EPSILON,
        1.0 - _TRANSITION_EPSILON,
    )
    design = np.log(raw / (1.0 - raw)).reshape(-1, 1)
    estimator = LogisticRegression(
        C=0.10,
        solver="lbfgs",
        max_iter=1_000,
        random_state=random_state,
    )
    try:
        estimator.fit(design, target)
        clipped = float(
            np.clip(raw_probability, _TRANSITION_EPSILON, 1.0 - _TRANSITION_EPSILON)
        )
        value = float(
            estimator.predict_proba(
                np.asarray([[np.log(clipped / (1.0 - clipped))]], dtype=float)
            )[0, 1]
        )
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        return raw_probability, "identity", True, f"{type(exc).__name__}: {exc}"
    return (
        float(np.clip(value, _TRANSITION_EPSILON, 1.0 - _TRANSITION_EPSILON)),
        "prequential_platt_logit",
        False,
        "",
    )


def _transition_threshold(
    history: pd.DataFrame,
    *,
    minimum_rows: int,
) -> tuple[float, str]:
    if len(history) < minimum_rows:
        return 0.5, f"fallback_0.5:insufficient_rows:{len(history)}<{minimum_rows}"
    actual = history["actual_change"].to_numpy(dtype=bool)
    if not actual.any() or actual.all():
        return 0.5, "fallback_0.5:insufficient_event_classes"
    probability = history["p_change"].to_numpy(dtype=float)
    scored: list[tuple[float, float]] = []
    for threshold in np.linspace(0.05, 0.95, 91):
        predicted = probability >= threshold
        true_positive_rate = float(predicted[actual].mean())
        true_negative_rate = float((~predicted[~actual]).mean())
        scored.append(((true_positive_rate + true_negative_rate) / 2.0, float(threshold)))
    best_score = max(score for score, _ in scored)
    tied = [threshold for score, threshold in scored if np.isclose(score, best_score)]
    chosen = min(tied, key=lambda value: (abs(value - 0.5), -value))
    return float(chosen), "prequential_balanced_accuracy"


def evaluate_transition_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Evaluate binary departure probabilities by horizon and explicit split."""

    required = {
        "horizon",
        "model",
        "evaluation_split",
        "actual_change",
        "p_change",
        "predicted_change",
        "fallback",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"transition predictions missing columns: {missing}")
    rows: list[dict[str, object]] = []
    grouped = predictions.groupby(
        ["horizon", "evaluation_split", "model"], sort=False
    )
    for (horizon, split, model), group in grouped:
        actual = group["actual_change"].to_numpy(dtype=bool)
        probability = group["p_change"].to_numpy(dtype=float)
        if (
            not np.isfinite(probability).all()
            or (probability <= 0.0).any()
            or (probability >= 1.0).any()
        ):
            raise ValueError("transition probabilities must be finite and strictly in (0, 1)")
        predicted = group["predicted_change"].to_numpy(dtype=bool)
        log_loss = -np.mean(
            actual * np.log(probability) + (~actual) * np.log(1.0 - probability)
        )
        brier = np.mean((probability - actual.astype(float)) ** 2)
        average_precision: float | None = None
        if actual.any():
            average_precision = float(average_precision_score(actual, probability))
        false_positives = int(np.count_nonzero(predicted & ~actual))
        years = max(float(len(group)) / 52.1775, 1.0 / 52.1775)
        rows.append(
            {
                "horizon": int(horizon),
                "evaluation_split": str(split),
                "model": str(model),
                "log_loss": float(log_loss),
                "brier": float(brier),
                "average_precision": average_precision,
                "precision": float(precision_score(actual, predicted, zero_division=0)),
                "recall": float(recall_score(actual, predicted, zero_division=0)),
                "false_alarms_per_year": float(false_positives / years),
                "n_predictions": int(len(group)),
                "event_count": int(actual.sum()),
                "non_event_count": int((~actual).sum()),
                "fallback_count": int(group["fallback"].astype(bool).sum()),
                "calibration_fallback_count": int(
                    group.get("calibration_fallback", pd.Series(False, index=group.index))
                    .astype(bool)
                    .sum()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["horizon", "evaluation_split", "log_loss", "brier", "model"],
        ignore_index=True,
    )


def _select_transition_champion(
    predictions: pd.DataFrame,
    candidates: tuple[str, ...],
) -> tuple[str, str]:
    if predictions.empty:
        fallback = "markov_hazard" if "markov_hazard" in candidates else candidates[0]
        return fallback, "insufficient_prequential_selection_history"
    eligible_candidates = tuple(
        name for name in candidates if name != "joint_survival_hazard"
    )
    if not eligible_candidates:
        raise ValueError("transition selection requires a non-shadow candidate")
    table = evaluate_transition_predictions(predictions)
    table = table.loc[table["model"].isin(eligible_candidates)]
    table = table.sort_values(["log_loss", "brier", "fallback_count", "model"])
    baselines = table.loc[
        table["model"].isin(("empirical_hazard", "markov_hazard"))
    ]
    if baselines.empty:
        chosen = str(table.iloc[0]["model"])
        return chosen, "lowest_prequential_log_loss:no_baseline_candidate"
    baseline = baselines.sort_values(["log_loss", "brier", "model"]).iloc[0]
    eligible = table.loc[
        ~table["model"].isin(("empirical_hazard", "markov_hazard"))
        & table["fallback_count"].eq(0)
        & (table["log_loss"] <= float(baseline["log_loss"]) - 0.005)
        & (table["brier"] <= float(baseline["brier"]) + 0.005)
    ]
    if eligible.empty:
        return str(baseline["model"]), "conservative_probability_baseline_gate"
    complexity = {
        "regularized_logistic": 0,
        "duration_tvtp_hurdle": 1,
        "binary_xgboost": 2,
        "joint_survival_hazard": 3,
    }
    eligible = eligible.assign(
        _complexity=eligible["model"].map(complexity).fillna(99)
    ).sort_values(["log_loss", "brier", "_complexity", "model"])
    return str(eligible.iloc[0]["model"]), "passed_probability_baseline_gate"


def _transition_test_positions(
    index: pd.DatetimeIndex,
    *,
    horizon: int,
    minimum_train_weeks: int,
    selection_end: pd.Timestamp,
    maximum_diagnostic_origins: int | None,
    selection_max_origins: int | None,
    minimum_selection_predictions: int,
    minimum_diagnostic_predictions: int,
) -> tuple[list[int], set[int], set[int]]:
    first_origin = minimum_train_weeks + horizon
    last_known_origin = len(index) - horizon - 1
    if first_origin > last_known_origin:
        raise ValueError(
            f"not enough observations for horizon {horizon}: "
            f"first origin {first_origin}, last known origin {last_known_origin}"
        )
    available = list(range(first_origin, last_known_origin + 1))
    selection = [
        position
        for position in available
        if pd.Timestamp(index[position + horizon]) < selection_end
    ]
    diagnostic = [
        position
        for position in available
        if pd.Timestamp(index[position]) >= selection_end
    ]
    if len(selection) < minimum_selection_predictions:
        raise ValueError(
            f"insufficient horizon-{horizon} selection predictions: "
            f"{len(selection)} < {minimum_selection_predictions}"
        )
    if len(diagnostic) < minimum_diagnostic_predictions:
        raise ValueError(
            f"insufficient horizon-{horizon} retrospective diagnostics: "
            f"{len(diagnostic)} < {minimum_diagnostic_predictions}"
        )
    if selection_max_origins is not None:
        budget = max(selection_max_origins, minimum_selection_predictions)
        selection = selection[-min(len(selection), budget) :]
    if maximum_diagnostic_origins is not None:
        diagnostic = diagnostic[-min(len(diagnostic), maximum_diagnostic_origins) :]
    combined = sorted([*selection, *diagnostic])
    return combined, set(selection), set(diagnostic)


def run_transition_benchmark(
    features: pd.DataFrame,
    states: pd.Series,
    *,
    horizons: Iterable[int] = (1, 4, 13),
    profile: str | BenchmarkProfile = "quick",
    models: Iterable[str] | None = None,
    include_xgboost: bool = False,
    include_joint_survival: bool = False,
    minimum_train_weeks: int | None = None,
    selection_end: str | pd.Timestamp = "2023-01-01",
    selection_max_origins: int | None = None,
    minimum_selection_predictions: int = 12,
    minimum_diagnostic_predictions: int = 12,
    minimum_inner_predictions: int = 12,
    random_state: int = 17,
    progress: Callable[[str], None] | None = None,
) -> TransitionBenchmarkResult:
    """Benchmark 1/4/13-week departure risk with strict horizon purging.

    A horizon event means at least one departure from the origin state during
    ``t+1 .. t+h``.  Every training label satisfies ``target_end < origin``;
    consequently the recorded purge/gap is exactly ``h``.  Model family,
    Platt-logit calibration, and thresholds are learned prequentially from
    earlier selection OOS rows only.  Outcomes on/after ``selection_end`` are
    named ``retrospective_diagnostic`` and never enter those choices.
    """

    features, states = _validate_inputs(features, states)
    raw_horizons = tuple(horizons)
    try:
        invalid_horizon = any(
            isinstance(value, (bool, np.bool_))
            or int(value) != value
            or int(value) < 1
            for value in raw_horizons
        )
    except (TypeError, ValueError, OverflowError):
        invalid_horizon = True
    if not raw_horizons or invalid_horizon:
        raise ValueError("horizons must contain positive integers")
    resolved_horizons = tuple(
        dict.fromkeys(int(value) for value in raw_horizons)
    )
    cfg = resolve_profile(profile)
    required_history = (
        cfg.minimum_train_weeks
        if minimum_train_weeks is None
        else int(minimum_train_weeks)
    )
    if required_history < 12:
        raise ValueError("minimum_train_weeks must be at least 12")
    if min(
        minimum_selection_predictions,
        minimum_diagnostic_predictions,
        minimum_inner_predictions,
    ) < 1:
        raise ValueError("transition minimum prediction counts must be positive")
    if selection_max_origins is not None and selection_max_origins < 1:
        raise ValueError("selection_max_origins must be positive or None")
    cutoff = _coerce_selection_end(selection_end, features.index)
    names, candidate_status = _transition_candidate_names(
        models,
        include_xgboost=include_xgboost,
        include_joint_survival=include_joint_survival,
    )
    if include_joint_survival and 1 not in resolved_horizons:
        raise ValueError("joint_survival_hazard requires the one-week horizon")
    augmented = derive_causal_transition_features(features, states)
    numeric = _transition_numeric_frame(augmented)

    raw_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    target_cache: dict[int, tuple[pd.Series, pd.Series]] = {
        horizon: _transition_targets(states, horizon)
        for horizon in resolved_horizons
    }
    one_week_event_target = (
        target_cache[1][0]
        if 1 in target_cache
        else _transition_targets(states, 1)[0]
    )
    for horizon in resolved_horizons:
        event_target, destination_target = target_cache[horizon]
        resolved_selection_max = selection_max_origins
        if resolved_selection_max is None and cfg.name == "quick":
            resolved_selection_max = max(3, minimum_selection_predictions)
        positions, selection_positions, _ = _transition_test_positions(
            features.index,
            horizon=horizon,
            minimum_train_weeks=required_history,
            selection_end=cutoff,
            maximum_diagnostic_origins=cfg.max_origins,
            selection_max_origins=resolved_selection_max,
            minimum_selection_predictions=minimum_selection_predictions,
            minimum_diagnostic_predictions=minimum_diagnostic_predictions,
        )
        checkpoints = set(
            np.rint(np.linspace(1, len(positions), min(12, len(positions)))).astype(int)
        )
        for ordinal, origin_position in enumerate(positions, start=1):
            if progress is not None and ordinal in checkpoints:
                progress(
                    f"transition horizon {horizon}w {ordinal}/{len(positions)}: "
                    f"{features.index[origin_position].date().isoformat()}"
                )
            train_stop = origin_position - horizon
            origin_date = pd.Timestamp(features.index[origin_position])
            target_start = pd.Timestamp(features.index[origin_position + 1])
            target_end = pd.Timestamp(features.index[origin_position + horizon])
            evaluation_split = (
                "selection"
                if origin_position in selection_positions
                else "retrospective_diagnostic"
            )
            last_train_origin_position = train_stop - 1
            last_train_target_position = last_train_origin_position + horizon
            if last_train_target_position >= origin_position:
                raise RuntimeError("transition training target is not strictly pre-origin")
            split_rows.append(
                {
                    "origin_date": origin_date,
                    "target_start": target_start,
                    "target_end": target_end,
                    "horizon": int(horizon),
                    "evaluation_split": evaluation_split,
                    "train_size": int(train_stop),
                    "train_start": pd.Timestamp(features.index[0]),
                    "last_train_origin": pd.Timestamp(
                        features.index[last_train_origin_position]
                    ),
                    "last_train_target_end": pd.Timestamp(
                        features.index[last_train_target_position]
                    ),
                    "purged_origin_count": int(horizon),
                    "gap": int(horizon),
                }
            )
            for name in names:
                (
                    raw_probability,
                    fallback,
                    fallback_reason,
                    one_week_hazard,
                ) = _fit_transition_candidate(
                    name,
                    augmented,
                    numeric,
                    event_target,
                    one_week_event_target,
                    destination_target,
                    horizon=horizon,
                    train_stop=train_stop,
                    test_position=origin_position,
                    profile=cfg,
                    random_state=random_state,
                )
                raw_rows.append(
                    {
                        "origin_date": origin_date,
                        "target_start": target_start,
                        "target_end": target_end,
                        "horizon": int(horizon),
                        "model": name,
                        "evaluation_split": evaluation_split,
                        "current_state": str(states.iloc[origin_position]),
                        "actual_change": bool(event_target.iloc[origin_position]),
                        "raw_p_change": raw_probability,
                        "one_week_hazard": one_week_hazard,
                        "train_size": int(train_stop),
                        "gap": int(horizon),
                        "fallback": fallback,
                        "fallback_reason": fallback_reason,
                    }
                )

    predictions = pd.DataFrame(raw_rows).sort_values(
        ["horizon", "origin_date", "model"], ignore_index=True
    )
    calibrated_rows: list[dict[str, object]] = []
    for row in predictions.to_dict(orient="records"):
        origin = pd.Timestamp(row["origin_date"])
        history = predictions.loc[
            predictions["horizon"].eq(int(row["horizon"]))
            & predictions["model"].eq(str(row["model"]))
            & predictions["evaluation_split"].eq("selection")
            & (predictions["target_end"] < origin)
        ]
        probability, calibration_method, calibration_fallback, calibration_reason = (
            _calibrate_transition_probability(
                float(row["raw_p_change"]),
                history,
                minimum_rows=minimum_inner_predictions,
                random_state=random_state,
            )
        )
        threshold_history = pd.DataFrame(calibrated_rows)
        if not threshold_history.empty:
            threshold_history = threshold_history.loc[
                threshold_history["horizon"].eq(int(row["horizon"]))
                & threshold_history["model"].eq(str(row["model"]))
                & threshold_history["evaluation_split"].eq("selection")
                & (threshold_history["target_end"] < origin)
            ]
        threshold, threshold_method = _transition_threshold(
            threshold_history,
            minimum_rows=minimum_inner_predictions,
        )
        row.update(
            {
                "p_change": probability,
                "calibration_method": calibration_method,
                "calibration_fallback": calibration_fallback,
                "calibration_fallback_reason": calibration_reason,
                "threshold": threshold,
                "threshold_method": threshold_method,
                "predicted_change": bool(probability >= threshold),
            }
        )
        calibrated_rows.append(row)
    predictions = pd.DataFrame(calibrated_rows).sort_values(
        ["horizon", "origin_date", "model"], ignore_index=True
    )

    nested_rows: list[dict[str, object]] = []
    for (horizon, origin), current_rows in predictions.groupby(
        ["horizon", "origin_date"], sort=True
    ):
        history = predictions.loc[
            predictions["horizon"].eq(int(horizon))
            & predictions["evaluation_split"].eq("selection")
            & (predictions["target_end"] < pd.Timestamp(origin))
        ]
        history_origins = int(history["origin_date"].nunique())
        if history_origins < minimum_inner_predictions:
            selected = "markov_hazard" if "markov_hazard" in names else names[0]
            reason = (
                f"fallback_family:insufficient_prequential_origins:"
                f"{history_origins}<{minimum_inner_predictions}"
            )
        else:
            selected, reason = _select_transition_champion(history, names)
        selected_row = current_rows.loc[current_rows["model"].eq(selected)].iloc[0]
        split = str(selected_row["evaluation_split"])
        nested_rows.append(
            {
                "origin_date": pd.Timestamp(origin),
                "horizon": int(horizon),
                "evaluation_split": split,
                "selected_model": selected,
                "selection_reason": reason,
                "threshold": float(selected_row["threshold"]),
                "threshold_method": str(selected_row["threshold_method"]),
                "selection_history_origins": history_origins,
                "selection_scope": "earlier_selection_oos_only",
                "selection_locked": split == "retrospective_diagnostic",
            }
        )
    nested_selection = pd.DataFrame(nested_rows).sort_values(
        ["horizon", "origin_date"], ignore_index=True
    )

    leaderboard = evaluate_transition_predictions(predictions)
    champions: dict[int, str] = {}
    for horizon in resolved_horizons:
        selection_predictions = predictions.loc[
            predictions["horizon"].eq(horizon)
            & predictions["evaluation_split"].eq("selection")
        ]
        champion, _ = _select_transition_champion(selection_predictions, names)
        champions[int(horizon)] = champion
    leaderboard.insert(
        3,
        "selected",
        [
            str(model) == champions[int(horizon)]
            for horizon, model in zip(
                leaderboard["horizon"], leaderboard["model"], strict=True
            )
        ],
    )

    latest_rows: list[dict[str, object]] = []
    latest_candidate_rows: list[dict[str, object]] = []

    def prospective_target_date(position: int) -> pd.Timestamp:
        if position < len(features.index):
            return pd.Timestamp(features.index[position])
        weeks_beyond_last = position - (len(features.index) - 1)
        return pd.Timestamp(features.index[-1]) + timedelta(
            days=7 * int(weeks_beyond_last)
        )

    for horizon in resolved_horizons:
        event_target, destination_target = target_cache[horizon]
        champion = champions[horizon]
        for candidate_name in names:
            raw_history = predictions.loc[
                predictions["horizon"].eq(horizon)
                & predictions["model"].eq(candidate_name)
                & predictions["evaluation_split"].eq("selection")
            ]
            threshold, threshold_method = _transition_threshold(
                raw_history,
                minimum_rows=minimum_inner_predictions,
            )
            # Exactly the final h origins have no fully observed t+1..t+h
            # target.  Each candidate is regenerated at the historical origin
            # with the same purged fit and selection-only calibration contract.
            for forecast_position in range(len(features) - horizon, len(features)):
                forecast_origin = pd.Timestamp(features.index[forecast_position])
                train_stop = forecast_position - horizon
                if train_stop < required_history:
                    raise ValueError(
                        f"not enough completed targets for horizon-{horizon} "
                        f"forecast at {forecast_origin.date().isoformat()}"
                    )
                last_train_origin_position = train_stop - 1
                last_train_target_position = last_train_origin_position + horizon
                if last_train_target_position >= forecast_position:
                    raise RuntimeError(
                        "prospective transition training target is not strictly pre-origin"
                    )
                (
                    raw_probability,
                    fallback,
                    fallback_reason,
                    one_week_hazard,
                ) = _fit_transition_candidate(
                    candidate_name,
                    augmented,
                    numeric,
                    event_target,
                    one_week_event_target,
                    destination_target,
                    horizon=horizon,
                    train_stop=train_stop,
                    test_position=forecast_position,
                    profile=cfg,
                    random_state=random_state,
                )
                (
                    probability,
                    calibration_method,
                    calibration_fallback,
                    calibration_reason,
                ) = _calibrate_transition_probability(
                    raw_probability,
                    raw_history,
                    minimum_rows=minimum_inner_predictions,
                    random_state=random_state,
                )
                result_row = {
                    "origin_date": forecast_origin,
                    "target_start": prospective_target_date(forecast_position + 1),
                    "target_end": prospective_target_date(
                        forecast_position + horizon
                    ),
                    "horizon": int(horizon),
                    "model": candidate_name,
                    "evaluation_split": "prospective",
                    "current_state": str(states.iloc[forecast_position]),
                    "actual_change": pd.NA,
                    "raw_p_change": raw_probability,
                    "one_week_hazard": one_week_hazard,
                    "p_change": probability,
                    "threshold": threshold,
                    "predicted_change": bool(probability >= threshold),
                    "train_size": int(train_stop),
                    "last_train_origin": pd.Timestamp(
                        features.index[last_train_origin_position]
                    ),
                    "last_train_target_end": pd.Timestamp(
                        features.index[last_train_target_position]
                    ),
                    "gap": int(horizon),
                    "fallback": fallback,
                    "fallback_reason": fallback_reason,
                    "calibration_method": calibration_method,
                    "calibration_fallback": calibration_fallback,
                    "calibration_fallback_reason": calibration_reason,
                    "threshold_method": threshold_method,
                    "selection_scope": "selection_oos_only",
                    "selection_locked": True,
                }
                latest_candidate_rows.append(result_row)
                if candidate_name == champion:
                    latest_rows.append(result_row.copy())

    return TransitionBenchmarkResult(
        leaderboard=leaderboard,
        predictions=predictions,
        split_audit=pd.DataFrame(split_rows).sort_values(
            ["horizon", "origin_date"], ignore_index=True
        ),
        nested_selection=nested_selection,
        champions_by_horizon=champions,
        profile=cfg,
        selection_end=cutoff,
        candidate_status=candidate_status,
        _latest_forecast_rows=pd.DataFrame(latest_rows).sort_values(
            ["horizon", "origin_date"], ignore_index=True
        ),
        _latest_candidate_forecast_rows=pd.DataFrame(
            latest_candidate_rows
        ).sort_values(["horizon", "origin_date", "model"], ignore_index=True),
    )


__all__ = [
    "BASELINE_MODELS",
    "PROBABILITY_COLUMNS",
    "TRANSITION_MODELS",
    "BenchmarkResult",
    "TransitionBenchmarkResult",
    "evaluate_predictions",
    "evaluate_transition_predictions",
    "expected_calibration_error",
    "forecast_next_regime",
    "multiclass_brier_score",
    "run_benchmark",
    "run_transition_benchmark",
    "select_champion",
    "select_champion_with_diagnostics",
]

# Structural composition lives in a separate module so the v3 walk-forward
# code above remains untouched until a pipeline explicitly opts in.  Re-export
# the public APIs here for callers that treat validation as the analysis entry
# point.  The structural module imports this module only inside its augmentation
# function, after module initialisation, so this does not create a runtime cycle.
from .structural_models import augment_benchmark_with_structural_models
from .structural_models import build_causal_dynamic_ensemble_oos
from .structural_models import build_xgb_hazard_destination_oos
from .structural_models import causal_dynamic_ensemble
from .structural_models import causal_multiscale_ensemble
from .structural_models import forecast_structural_probabilities
from .structural_models import project_joint_survival_hazard

__all__.extend(
    [
        "augment_benchmark_with_structural_models",
        "build_causal_dynamic_ensemble_oos",
        "build_xgb_hazard_destination_oos",
        "causal_dynamic_ensemble",
        "causal_multiscale_ensemble",
        "forecast_structural_probabilities",
        "project_joint_survival_hazard",
    ]
)
