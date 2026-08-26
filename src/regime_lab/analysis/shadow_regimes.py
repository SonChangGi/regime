"""Prospective-only causal shadow regime filters.

The canonical supervised labels are never read or rewritten here.  Both
algorithms consume observations in timestamp order and expose filtered output
only; there is no backward smoothing pass.  They are research shadows, not a
new ground truth and not automatically eligible for champion selection.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import exp, log, pi, sqrt
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .labels import STATE_ORDER


SHADOW_REGIME_SCHEMA_VERSION = "regime-causal-shadow/1"


def _canonical_sha256(value: Mapping[str, Any] | Sequence[Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class DirectJumpHSMMConfig:
    """Fixed parameters for a truncated explicit-duration forward filter."""

    minimum_duration_weeks: tuple[int, int, int] = (2, 1, 2)
    base_exit_hazards: tuple[float, float, float] = (0.04, 0.18, 0.04)
    duration_hazard_slopes: tuple[float, float, float] = (0.01, 0.02, 0.01)
    maximum_exit_hazard: float = 0.55
    maximum_duration_weeks: int = 52
    # Rows are origin states and columns are destinations in STATE_ORDER.
    # The risk_on <-> risk_off cells are positive by design: direct jumps are
    # supported rather than routed through an invented transition week.
    destination_weights: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.75, 0.25),
        (0.5, 0.0, 0.5),
        (0.25, 0.75, 0.0),
    )
    emission_floor: float = 1e-12

    def __post_init__(self) -> None:
        state_count = len(STATE_ORDER)
        for field_name in (
            "minimum_duration_weeks",
            "base_exit_hazards",
            "duration_hazard_slopes",
        ):
            if len(getattr(self, field_name)) != state_count:
                raise ValueError(f"{field_name} must follow STATE_ORDER")
        if any(
            isinstance(value, bool) or int(value) != value or int(value) < 1
            for value in self.minimum_duration_weeks
        ):
            raise ValueError("minimum durations must be positive integers")
        if self.maximum_duration_weeks < max(self.minimum_duration_weeks):
            raise ValueError("maximum duration is below a minimum duration")
        if any(not 0.0 <= float(value) < 1.0 for value in self.base_exit_hazards):
            raise ValueError("base exit hazards must be in [0, 1)")
        if any(float(value) < 0.0 for value in self.duration_hazard_slopes):
            raise ValueError("duration hazard slopes must be non-negative")
        if not 0.0 < float(self.maximum_exit_hazard) < 1.0:
            raise ValueError("maximum_exit_hazard must be in (0, 1)")
        if self.maximum_exit_hazard < max(self.base_exit_hazards):
            raise ValueError("maximum exit hazard is below a base hazard")
        weights = np.asarray(self.destination_weights, dtype=float)
        if weights.shape != (state_count, state_count):
            raise ValueError("destination_weights must be a square STATE_ORDER matrix")
        if not np.isfinite(weights).all() or (weights < 0.0).any():
            raise ValueError("destination weights must be finite and non-negative")
        if not np.allclose(np.diag(weights), 0.0, rtol=0.0, atol=0.0):
            raise ValueError("destination weights cannot transition to the same state")
        if not np.allclose(weights.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("each destination row must sum to one")
        if weights[0, 2] <= 0.0 or weights[2, 0] <= 0.0:
            raise ValueError("direct risk_on/risk_off jumps must have positive weight")
        if not 0.0 < float(self.emission_floor) < 1.0 / state_count:
            raise ValueError("emission_floor must be small and positive")

    def manifest(self) -> dict[str, Any]:
        return {
            "minimum_duration_weeks": list(self.minimum_duration_weeks),
            "base_exit_hazards": list(self.base_exit_hazards),
            "duration_hazard_slopes": list(self.duration_hazard_slopes),
            "maximum_exit_hazard": float(self.maximum_exit_hazard),
            "maximum_duration_weeks": int(self.maximum_duration_weeks),
            "destination_weights": [list(row) for row in self.destination_weights],
            "emission_floor": float(self.emission_floor),
        }


@dataclass(frozen=True)
class FilteredHSMMShadowResult:
    probabilities: pd.DataFrame
    states: pd.Series
    diagnostics: pd.DataFrame
    configuration_sha256: str
    method: str = "causal_forward_explicit_duration_hsmm_direct_jump"
    role: str = "prospective_shadow_only"
    canonical_target: bool = False


def _validate_emissions(emissions: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(emissions, pd.DataFrame):
        raise TypeError("emission_probabilities must be a DataFrame")
    if tuple(str(column) for column in emissions.columns) != STATE_ORDER:
        raise ValueError(f"emission columns must be exactly {STATE_ORDER}")
    if not isinstance(emissions.index, pd.DatetimeIndex):
        raise TypeError("emissions must use a DatetimeIndex")
    if emissions.empty:
        raise ValueError("emissions must not be empty")
    if emissions.index.has_duplicates or not emissions.index.is_monotonic_increasing:
        raise ValueError("emission index must be unique and increasing")
    numeric = emissions.astype(float)
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("emission probabilities must be finite")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("emission probabilities must be in [0, 1]")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("each emission row must sum to one")
    return numeric


def _initial_prior(value: Mapping[str, float] | Sequence[float] | None) -> np.ndarray:
    if value is None:
        return np.full(len(STATE_ORDER), 1.0 / len(STATE_ORDER), dtype=float)
    if isinstance(value, Mapping):
        if set(value) != set(STATE_ORDER):
            raise ValueError("initial_prior keys must equal STATE_ORDER")
        raw = np.asarray([value[state] for state in STATE_ORDER], dtype=float)
    else:
        raw = np.asarray(value, dtype=float)
    if raw.shape != (len(STATE_ORDER),):
        raise ValueError("initial_prior has the wrong shape")
    if not np.isfinite(raw).all() or (raw < 0.0).any() or raw.sum() <= 0.0:
        raise ValueError("initial_prior must be finite, non-negative, and nonzero")
    return raw / raw.sum()


def _hsmm_exit_hazard(
    state_position: int,
    duration_weeks: int,
    config: DirectJumpHSMMConfig,
) -> float:
    minimum = config.minimum_duration_weeks[state_position]
    if duration_weeks < minimum:
        return 0.0
    older_by = duration_weeks - minimum
    return min(
        config.maximum_exit_hazard,
        config.base_exit_hazards[state_position]
        + config.duration_hazard_slopes[state_position] * older_by,
    )


def filter_direct_jump_hsmm_shadow(
    emission_probabilities: pd.DataFrame,
    *,
    config: DirectJumpHSMMConfig | None = None,
    initial_prior: Mapping[str, float] | Sequence[float] | None = None,
) -> FilteredHSMMShadowResult:
    """Causally forward-filter soft evidence with explicit durations.

    ``emission_probabilities.iloc[:t+1]`` fully determines row ``t``.  The
    function performs neither parameter fitting nor backward smoothing, and it
    never consumes the supervised next-week state.
    """

    emissions = _validate_emissions(emission_probabilities)
    settings = config or DirectJumpHSMMConfig()
    prior = _initial_prior(initial_prior)
    state_count = len(STATE_ORDER)
    maximum_duration = settings.maximum_duration_weeks
    weights = np.asarray(settings.destination_weights, dtype=float)
    posterior = np.zeros((state_count, maximum_duration), dtype=float)

    probability_rows: list[np.ndarray] = []
    state_rows: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    previous_map_state: str | None = None

    for row_number, (at, raw_emission) in enumerate(emissions.iterrows()):
        emission = np.maximum(
            raw_emission.to_numpy(dtype=float), settings.emission_floor
        )
        emission /= emission.sum()
        switched = np.zeros_like(posterior)
        direct_switched = np.zeros_like(posterior)

        if row_number == 0:
            predictive = np.zeros_like(posterior)
            predictive[:, 0] = prior
        else:
            predictive = np.zeros_like(posterior)
            for origin in range(state_count):
                for duration_position in range(maximum_duration):
                    mass = float(posterior[origin, duration_position])
                    if mass <= 0.0:
                        continue
                    duration = duration_position + 1
                    hazard = _hsmm_exit_hazard(origin, duration, settings)
                    stay_position = min(duration_position + 1, maximum_duration - 1)
                    predictive[origin, stay_position] += mass * (1.0 - hazard)
                    for destination in range(state_count):
                        switch_mass = mass * hazard * weights[origin, destination]
                        if switch_mass <= 0.0:
                            continue
                        predictive[destination, 0] += switch_mass
                        switched[destination, 0] += switch_mass
                        if {origin, destination} == {0, 2}:
                            direct_switched[destination, 0] += switch_mass

        unnormalised = predictive * emission[:, None]
        normalizer = float(unnormalised.sum())
        if not np.isfinite(normalizer) or normalizer <= 0.0:
            raise FloatingPointError(f"HSMM posterior collapsed at {at}")
        posterior = unnormalised / normalizer
        state_probability = posterior.sum(axis=1)
        map_position = int(np.argmax(state_probability))
        map_state = STATE_ORDER[map_position]
        duration_grid = np.arange(1, maximum_duration + 1, dtype=float)
        expected_duration = float((posterior * duration_grid[None, :]).sum())
        map_duration = int(np.argmax(posterior[map_position]) + 1)
        switch_probability = float((switched * emission[:, None]).sum() / normalizer)
        direct_probability = float(
            (direct_switched * emission[:, None]).sum() / normalizer
        )
        hard_direct_jump = (
            previous_map_state is not None
            and {previous_map_state, map_state} == {"risk_on", "risk_off"}
        )

        probability_rows.append(state_probability)
        state_rows.append(map_state)
        diagnostics.append(
            {
                "as_of": pd.Timestamp(at),
                "filtered_switch_probability": switch_probability,
                "filtered_direct_jump_probability": direct_probability,
                "map_duration_weeks": map_duration,
                "posterior_expected_duration_weeks": expected_duration,
                "map_direct_jump": bool(hard_direct_jump),
                "uses_backward_smoothing": False,
                "uses_supervised_target": False,
            }
        )
        previous_map_state = map_state

    return FilteredHSMMShadowResult(
        probabilities=pd.DataFrame(
            probability_rows, index=emissions.index, columns=STATE_ORDER, dtype=float
        ),
        states=pd.Series(state_rows, index=emissions.index, name="shadow_state"),
        diagnostics=pd.DataFrame(diagnostics).set_index("as_of"),
        configuration_sha256=_canonical_sha256(settings.manifest()),
    )


@dataclass(frozen=True)
class BOCPDConfig:
    """Normal-mean Bayesian online changepoint configuration.

    A changepoint is defined as occurring immediately before the current
    observation.  The new-run branch therefore uses the prior predictive,
    while growth branches use each existing run's posterior predictive.
    """

    constant_hazard: float = 1.0 / 26.0
    prior_mean: float = 0.0
    prior_mean_variance: float = 4.0
    observation_variance: float = 1.0
    maximum_run_length: int = 520
    density_floor: float = 1e-300

    def __post_init__(self) -> None:
        if not 0.0 < float(self.constant_hazard) < 1.0:
            raise ValueError("constant_hazard must be in (0, 1)")
        for name in ("prior_mean", "prior_mean_variance", "observation_variance"):
            if not np.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.prior_mean_variance <= 0.0 or self.observation_variance <= 0.0:
            raise ValueError("BOCPD variances must be positive")
        if isinstance(self.maximum_run_length, bool) or self.maximum_run_length < 1:
            raise ValueError("maximum_run_length must be a positive integer")
        if not 0.0 < float(self.density_floor) < 1e-20:
            raise ValueError("density_floor must be small and positive")

    def manifest(self) -> dict[str, Any]:
        return {
            "constant_hazard": float(self.constant_hazard),
            "prior_mean": float(self.prior_mean),
            "prior_mean_variance": float(self.prior_mean_variance),
            "observation_variance": float(self.observation_variance),
            "maximum_run_length": int(self.maximum_run_length),
            "density_floor": float(self.density_floor),
            "changepoint_timing": "immediately_before_current_observation",
        }


@dataclass(frozen=True)
class BOCPDShadowResult:
    diagnostics: pd.DataFrame
    final_run_length_posterior: pd.Series
    configuration_sha256: str
    method: str = "bayesian_online_changepoint_normal_mean"
    role: str = "prospective_transition_alert_shadow_only"
    canonical_target: bool = False


def _normal_density(value: float, mean: float, variance: float, floor: float) -> float:
    if variance <= 0.0 or not np.isfinite(variance):
        raise FloatingPointError("predictive variance must be positive and finite")
    exponent = -0.5 * ((value - mean) ** 2) / variance
    density = exp(max(-745.0, exponent)) / sqrt(2.0 * pi * variance)
    return max(float(floor), float(density))


def _normal_mean_update(
    prior_mean: float,
    prior_variance: float,
    observation: float,
    observation_variance: float,
) -> tuple[float, float]:
    precision = 1.0 / prior_variance + 1.0 / observation_variance
    variance = 1.0 / precision
    mean = variance * (
        prior_mean / prior_variance + observation / observation_variance
    )
    return float(mean), float(variance)


def _logsumexp(values: np.ndarray) -> float:
    """Stable log(sum(exp(values))) over finite log masses."""

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("-inf")
    maximum = float(finite.max())
    return maximum + log(float(np.exp(finite - maximum).sum()))


def _validate_signal(signal: pd.Series) -> pd.Series:
    if not isinstance(signal, pd.Series):
        raise TypeError("signal must be a pandas Series")
    if not isinstance(signal.index, pd.DatetimeIndex):
        raise TypeError("signal must use a DatetimeIndex")
    if signal.empty:
        raise ValueError("signal must not be empty")
    if signal.index.has_duplicates or not signal.index.is_monotonic_increasing:
        raise ValueError("signal index must be unique and increasing")
    numeric = pd.to_numeric(signal, errors="coerce").astype(float)
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("signal must be finite and complete")
    return numeric


def bayesian_online_changepoint_shadow(
    signal: pd.Series,
    *,
    config: BOCPDConfig | None = None,
) -> BOCPDShadowResult:
    """Return a causal run-length posterior and transition-alert probability."""

    observations = _validate_signal(signal)
    settings = config or BOCPDConfig()
    maximum = int(settings.maximum_run_length)

    run_probability = np.asarray([1.0], dtype=float)
    means = np.asarray([settings.prior_mean], dtype=float)
    variances = np.asarray([settings.prior_mean_variance], dtype=float)
    rows: list[dict[str, Any]] = []

    for position, (at, raw_value) in enumerate(observations.items()):
        value = float(raw_value)
        if position == 0:
            updated_mean, updated_variance = _normal_mean_update(
                settings.prior_mean,
                settings.prior_mean_variance,
                value,
                settings.observation_variance,
            )
            evidence = _normal_density(
                value,
                settings.prior_mean,
                settings.prior_mean_variance + settings.observation_variance,
                settings.density_floor,
            )
            log_evidence = log(evidence)
            run_probability = np.asarray([1.0], dtype=float)
            means = np.asarray([updated_mean], dtype=float)
            variances = np.asarray([updated_variance], dtype=float)
            changepoint_probability = 1.0
        else:
            prior_predictive = _normal_density(
                value,
                settings.prior_mean,
                settings.prior_mean_variance + settings.observation_variance,
                settings.density_floor,
            )
            predictive = np.asarray(
                [
                    _normal_density(
                        value,
                        float(mean),
                        float(variance) + settings.observation_variance,
                        settings.density_floor,
                    )
                    for mean, variance in zip(means, variances, strict=True)
                ],
                dtype=float,
            )
            new_length = min(maximum + 1, len(run_probability) + 1)
            log_joint = np.full(new_length, float("-inf"), dtype=float)
            log_joint[0] = log(settings.constant_hazard) + log(prior_predictive)

            contributor_log_weights: list[list[float]] = [
                [] for _ in range(new_length)
            ]
            contributor_means: list[list[float]] = [[] for _ in range(new_length)]
            contributor_variances: list[list[float]] = [[] for _ in range(new_length)]
            reset_mean, reset_variance = _normal_mean_update(
                settings.prior_mean,
                settings.prior_mean_variance,
                value,
                settings.observation_variance,
            )
            contributor_log_weights[0].append(float(log_joint[0]))
            contributor_means[0].append(reset_mean)
            contributor_variances[0].append(reset_variance)

            for previous_run_length, previous_probability in enumerate(run_probability):
                destination = min(previous_run_length + 1, maximum)
                if previous_probability <= 0.0:
                    continue
                log_mass = (
                    log(float(previous_probability))
                    + log(1.0 - settings.constant_hazard)
                    + log(float(predictive[previous_run_length]))
                )
                log_joint[destination] = float(
                    np.logaddexp(log_joint[destination], log_mass)
                )
                updated_mean, updated_variance = _normal_mean_update(
                    float(means[previous_run_length]),
                    float(variances[previous_run_length]),
                    value,
                    settings.observation_variance,
                )
                contributor_log_weights[destination].append(log_mass)
                contributor_means[destination].append(updated_mean)
                contributor_variances[destination].append(updated_variance)

            log_evidence = _logsumexp(log_joint)
            if not np.isfinite(log_evidence):
                raise FloatingPointError(f"BOCPD posterior collapsed at {at}")
            run_probability = np.exp(log_joint - log_evidence)
            run_probability /= run_probability.sum()
            changepoint_probability = float(run_probability[0])
            next_means = np.empty(new_length, dtype=float)
            next_variances = np.empty(new_length, dtype=float)
            for run_length in range(new_length):
                log_weights = np.asarray(
                    contributor_log_weights[run_length], dtype=float
                )
                if log_weights.size == 0:
                    # A numerically zero posterior branch cannot influence any
                    # future posterior.  Keep a finite placeholder sufficient
                    # statistic instead of requiring fictitious positive mass.
                    next_means[run_length] = reset_mean
                    next_variances[run_length] = reset_variance
                    continue
                weights = np.exp(log_weights - _logsumexp(log_weights))
                branch_means = np.asarray(contributor_means[run_length], dtype=float)
                branch_variances = np.asarray(
                    contributor_variances[run_length], dtype=float
                )
                mixed_mean = float(np.dot(weights, branch_means))
                mixed_second = float(
                    np.dot(weights, branch_variances + branch_means**2)
                )
                next_means[run_length] = mixed_mean
                next_variances[run_length] = max(
                    settings.density_floor, mixed_second - mixed_mean**2
                )
            means = next_means
            variances = next_variances

        run_grid = np.arange(len(run_probability), dtype=float)
        rows.append(
            {
                "as_of": pd.Timestamp(at),
                "observation": value,
                "changepoint_probability": changepoint_probability,
                "map_run_length": int(np.argmax(run_probability)),
                "expected_run_length": float(np.dot(run_probability, run_grid)),
                "posterior_segment_mean": float(np.dot(run_probability, means)),
                "negative_log_predictive_density": float(
                    -log_evidence
                ),
                "uses_future_observation": False,
                "uses_supervised_target": False,
            }
        )

    return BOCPDShadowResult(
        diagnostics=pd.DataFrame(rows).set_index("as_of"),
        final_run_length_posterior=pd.Series(
            run_probability,
            index=pd.RangeIndex(len(run_probability), name="run_length"),
            name="probability",
        ),
        configuration_sha256=_canonical_sha256(settings.manifest()),
    )


def shadow_model_registry_document() -> dict[str, Any]:
    """Describe implemented code without asserting that a real-data run exists."""

    body = {
        "schema_version": SHADOW_REGIME_SCHEMA_VERSION,
        "canonical_target": False,
        "automatic_promotion_eligible": False,
        "models": [
            {
                "id": "filtered_hsmm",
                "aliases": ["hsmm_explicit_duration"],
                "implementation": "causal_forward_explicit_duration_hsmm_direct_jump",
                "status": "implemented_unrun",
                "result": None,
                "uses_backward_smoothing": False,
                "uses_supervised_target": False,
            },
            {
                "id": "bayesian_online_changepoint",
                "implementation": "normal_mean_run_length_filter",
                "status": "implemented_unrun",
                "result": None,
                "uses_future_observation": False,
                "uses_supervised_target": False,
            },
            {
                "id": "dynamic_factor_tvtp",
                "implementation": "expanding_prefix_pca_direct_jump_tvtp",
                "status": "implemented_unrun",
                "result": None,
                "causality_scope": "structural_row_prefix_only",
                "vintage_safety": (
                    "not_established_without_origin_snapshot_vintages"
                ),
                "operational_oos_eligible": False,
                "uses_supervised_target_at_prediction": False,
                "gap_weeks": 1,
            },
        ],
    }
    return {**body, "sha256": _canonical_sha256(body)}


__all__ = [
    "BOCPDConfig",
    "BOCPDShadowResult",
    "DirectJumpHSMMConfig",
    "FilteredHSMMShadowResult",
    "SHADOW_REGIME_SCHEMA_VERSION",
    "bayesian_online_changepoint_shadow",
    "filter_direct_jump_hsmm_shadow",
    "shadow_model_registry_document",
]
