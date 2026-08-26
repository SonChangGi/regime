"""Deterministic, matched-origin Model Confidence Set diagnostics.

This module implements the range-statistic elimination procedure from Hansen,
Lunde, and Nason (2011) with a circular moving-block bootstrap.  It is a
supplemental all-model comparison: it does not replace the project's frozen
materiality, Holm, Brier, and fallback promotion gates.

The input is deliberately a wide loss matrix.  Every row is one ordered OOS
origin and every column is one model evaluated at that exact origin.  Missing,
duplicated, or re-ordered origins are rejected instead of silently intersected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


MCS_METHOD = "hansen_lunde_nason_range_circular_moving_block_bootstrap"
MCS_DEFAULT_ALPHA = 0.10
MCS_DEFAULT_BLOCK_WEEKS = 13
MCS_DEFAULT_RESAMPLES = 1_999
MCS_DEFAULT_SEED = 17


@dataclass(frozen=True)
class ModelConfidenceSetStep:
    """One equal-predictive-ability test and any resulting elimination."""

    step: int
    active_models: tuple[str, ...]
    test_statistic: float
    bootstrap_p_value: float
    rejected: bool
    eliminated_model: str | None
    elimination_score: float | None
    remaining_models: tuple[str, ...]


@dataclass(frozen=True)
class ModelConfidenceSetResult:
    """Auditable retained set and ordered elimination evidence."""

    method: str
    alpha: float
    observation_count: int
    nominal_block_length: int
    effective_block_length: int
    bootstrap_resamples: int
    bootstrap_seed: int
    retained_models: tuple[str, ...]
    eliminated_models: tuple[str, ...]
    elimination_path: tuple[ModelConfidenceSetStep, ...]
    termination_reason: str


def validate_matched_loss_matrix(losses: pd.DataFrame) -> pd.DataFrame:
    """Return a numeric copy after enforcing the exact paired-origin contract.

    The ordered index is the OOS origin key.  A complete wide matrix is the
    explicit proof that every model has one loss for every same origin.  The
    function never fills missing values, sorts rows, or takes an intersection.
    """

    if not isinstance(losses, pd.DataFrame):
        raise TypeError("losses must be a pandas DataFrame")
    if losses.empty:
        raise ValueError("loss matrix must not be empty")
    if len(losses) < 3:
        raise ValueError("loss matrix needs at least three ordered origins")
    if losses.shape[1] < 2:
        raise ValueError("loss matrix needs at least two models")
    if losses.index.has_duplicates:
        raise ValueError("loss matrix origins must be unique")
    if losses.index.hasnans:
        raise ValueError("loss matrix origins must not be missing")
    if not losses.index.is_monotonic_increasing:
        raise ValueError("loss matrix origins must be increasing")
    if losses.columns.has_duplicates:
        raise ValueError("loss matrix model names must be unique")

    model_names = tuple(losses.columns)
    if any(not isinstance(model, str) or not model.strip() for model in model_names):
        raise ValueError("loss matrix model names must be non-empty strings")
    if any(pd.api.types.is_bool_dtype(dtype) for dtype in losses.dtypes):
        raise ValueError("loss matrix values must be numeric losses, not booleans")
    if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in losses.dtypes):
        raise ValueError("loss matrix values must be numeric")

    numeric = losses.astype(float, copy=True)
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(
            "loss matrix must be complete and finite at every matched origin"
        )
    return numeric


def _validate_configuration(
    *,
    alpha: float,
    block_length: int,
    resamples: int,
    random_state: int,
) -> None:
    if isinstance(alpha, bool) or not np.isfinite(float(alpha)) or not 0 < alpha < 1:
        raise ValueError("MCS alpha must be between zero and one")
    for name, value in (
        ("block_length", block_length),
        ("resamples", resamples),
        ("random_state", random_state),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"MCS {name} must be an integer")
    if int(block_length) < 1:
        raise ValueError("MCS block_length must be positive")
    # Fewer draws provide such coarse p-values that an alpha=10% MCS is not a
    # useful statistical diagnostic.  Production defaults remain much larger.
    if int(resamples) < 99:
        raise ValueError("MCS resamples must be at least 99")
    if int(random_state) < 0:
        raise ValueError("MCS random_state must be non-negative")


def _circular_block_indices(
    observation_count: int,
    *,
    block_length: int,
    resamples: int,
    random_state: int,
) -> tuple[np.ndarray, int]:
    # For short smoke samples, prevent a circular block from becoming the
    # entire series, which would make every resampled mean identical.
    effective_block = min(block_length, max(1, observation_count // 2))
    blocks_per_sample = int(np.ceil(observation_count / effective_block))
    generator = np.random.default_rng(random_state)
    starts = generator.integers(
        0,
        observation_count,
        size=(resamples, blocks_per_sample),
    )
    offsets = np.arange(effective_block, dtype=int)
    indices = (starts[..., np.newaxis] + offsets) % observation_count
    return (
        indices.reshape(resamples, -1)[:, :observation_count],
        effective_block,
    )

def _studentized_pairwise_differentials(
    loss_values: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return observed and centered-null bootstrap pairwise t statistics."""

    model_count = loss_values.shape[1]
    resamples = bootstrap_indices.shape[0]
    observed = np.zeros((model_count, model_count), dtype=float)
    bootstrapped = np.zeros((resamples, model_count, model_count), dtype=float)

    for left in range(model_count):
        for right in range(left + 1, model_count):
            differential = loss_values[:, left] - loss_values[:, right]
            mean_difference = float(np.mean(differential))
            centred = differential - mean_difference
            null_means = centred[bootstrap_indices].mean(axis=1)
            standard_error = float(np.std(null_means, ddof=1))
            scale = max(1.0, float(np.max(np.abs(differential))))
            numerical_zero = np.finfo(float).eps * 64.0 * scale

            if standard_error <= numerical_zero:
                statistic = (
                    0.0
                    if abs(mean_difference) <= numerical_zero
                    else float(np.copysign(np.inf, mean_difference))
                )
                bootstrap_statistics = np.zeros(resamples, dtype=float)
            else:
                statistic = mean_difference / standard_error
                bootstrap_statistics = null_means / standard_error

            observed[left, right] = statistic
            observed[right, left] = -statistic
            bootstrapped[:, left, right] = bootstrap_statistics
            bootstrapped[:, right, left] = -bootstrap_statistics
    return observed, bootstrapped


def model_confidence_set(
    losses: pd.DataFrame,
    *,
    alpha: float = MCS_DEFAULT_ALPHA,
    block_length: int = MCS_DEFAULT_BLOCK_WEEKS,
    resamples: int = MCS_DEFAULT_RESAMPLES,
    random_state: int = MCS_DEFAULT_SEED,
) -> ModelConfidenceSetResult:
    """Compute the range-statistic Model Confidence Set.

    The same block draws are shared by every pair and every sequential step.
    Bootstrap loss differentials are centered under equal predictive ability,
    and p-values use the finite-resample ``+1`` correction.  When the null is
    rejected, the model with the largest worst-case pairwise t statistic is
    removed; exact ties are resolved by model name.
    """

    _validate_configuration(
        alpha=alpha,
        block_length=block_length,
        resamples=resamples,
        random_state=random_state,
    )
    matrix = validate_matched_loss_matrix(losses)
    model_names = tuple(str(model) for model in matrix.columns)
    bootstrap_indices, effective_block = _circular_block_indices(
        len(matrix),
        block_length=int(block_length),
        resamples=int(resamples),
        random_state=int(random_state),
    )
    observed_all, bootstrap_all = _studentized_pairwise_differentials(
        matrix.to_numpy(dtype=float), bootstrap_indices
    )

    active = list(range(len(model_names)))
    eliminated: list[str] = []
    path: list[ModelConfidenceSetStep] = []
    termination_reason = "singleton"

    while len(active) > 1:
        observed = observed_all[np.ix_(active, active)]
        bootstrapped = bootstrap_all[:, active][:, :, active]
        test_statistic = float(np.max(np.abs(observed)))
        null_statistics = np.max(np.abs(bootstrapped), axis=(1, 2))
        bootstrap_p_value = float(
            (1 + np.count_nonzero(null_statistics >= test_statistic))
            / (int(resamples) + 1)
        )
        rejected = bool(bootstrap_p_value <= float(alpha))
        active_models = tuple(model_names[index] for index in active)

        eliminated_model: str | None = None
        elimination_score: float | None = None
        if rejected:
            scores = np.max(observed, axis=1)
            worst_score = float(np.max(scores))
            tied_positions = np.flatnonzero(
                np.isclose(scores, worst_score, rtol=1e-12, atol=1e-14)
            )
            eliminated_position = min(
                tied_positions,
                key=lambda position: model_names[active[int(position)]],
            )
            eliminated_index = active.pop(int(eliminated_position))
            eliminated_model = model_names[eliminated_index]
            elimination_score = worst_score
            eliminated.append(eliminated_model)
        else:
            termination_reason = "equal_predictive_ability_not_rejected"

        remaining_models = tuple(model_names[index] for index in active)
        path.append(
            ModelConfidenceSetStep(
                step=len(path) + 1,
                active_models=active_models,
                test_statistic=test_statistic,
                bootstrap_p_value=bootstrap_p_value,
                rejected=rejected,
                eliminated_model=eliminated_model,
                elimination_score=elimination_score,
                remaining_models=remaining_models,
            )
        )
        if not rejected:
            break

    return ModelConfidenceSetResult(
        method=MCS_METHOD,
        alpha=float(alpha),
        observation_count=int(len(matrix)),
        nominal_block_length=int(block_length),
        effective_block_length=int(effective_block),
        bootstrap_resamples=int(resamples),
        bootstrap_seed=int(random_state),
        retained_models=tuple(model_names[index] for index in active),
        eliminated_models=tuple(eliminated),
        elimination_path=tuple(path),
        termination_reason=termination_reason,
    )
