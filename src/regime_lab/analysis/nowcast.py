"""Causal shadow nowcast smoothing for three-state regime probabilities.

This module is deliberately separate from the canonical label generator.  It
provides a sensitivity view: soft evidence that is already available at week
``t`` is passed through a fixed, explicit-duration state filter.  There is no
``fit`` method and therefore no opportunity for a future holdout period to
influence the filter parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from regime_lab.schema import STATE_ORDER


_STATE_POSITION = {state: position for position, state in enumerate(STATE_ORDER)}
_OPPOSITE_RISK_STATE = {"risk_on": "risk_off", "risk_off": "risk_on"}


@dataclass(frozen=True)
class ShadowNowcastConfig:
    """Small, fixed parameter set for the duration-aware shadow filter.

    ``risk_state_base_hazard`` and ``transition_base_hazard`` are weekly exit
    hazards after ``minimum_duration_weeks``.  The shared duration slope lets
    old states become gradually easier to leave, while ``maximum_exit_hazard``
    prevents duration alone from forcing a switch.
    """

    minimum_duration_weeks: int = 2
    maximum_duration_weeks: int = 26
    risk_state_base_hazard: float = 0.04
    transition_base_hazard: float = 0.18
    duration_hazard_slope: float = 0.01
    maximum_exit_hazard: float = 0.45
    emission_floor: float = 1e-9

    def __post_init__(self) -> None:
        if self.minimum_duration_weeks < 1:
            raise ValueError("minimum_duration_weeks must be at least one")
        if self.maximum_duration_weeks < self.minimum_duration_weeks:
            raise ValueError(
                "maximum_duration_weeks must be at least minimum_duration_weeks"
            )
        for name in ("risk_state_base_hazard", "transition_base_hazard"):
            value = float(getattr(self, name))
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.duration_hazard_slope < 0.0:
            raise ValueError("duration_hazard_slope must be non-negative")
        if not 0.0 < self.maximum_exit_hazard < 1.0:
            raise ValueError("maximum_exit_hazard must be in (0, 1)")
        if self.maximum_exit_hazard < max(
            self.risk_state_base_hazard, self.transition_base_hazard
        ):
            raise ValueError("maximum_exit_hazard cannot be below a base hazard")
        if not 0.0 < self.emission_floor < 1.0 / len(STATE_ORDER):
            raise ValueError("emission_floor must be small and positive")


@dataclass(frozen=True)
class ShadowNowcastResult:
    """Filtered probabilities, routed display states, and causal diagnostics."""

    probabilities: pd.DataFrame
    states: pd.Series
    diagnostics: pd.DataFrame

    def summary(self) -> dict[str, float | int]:
        """Return compact flip and duration diagnostics for publication."""

        if self.diagnostics.empty:
            return {
                "observations": 0,
                "state_changes": 0,
                "routed_direct_jumps": 0,
                "mean_completed_duration_weeks": 0.0,
                "latest_duration_weeks": 0,
            }
        changed = self.diagnostics["state_changed"].astype(bool)
        completed = self.diagnostics.loc[changed, "previous_duration_weeks"]
        completed = pd.to_numeric(completed, errors="coerce").dropna()
        return {
            "observations": int(len(self.diagnostics)),
            "state_changes": int(changed.sum()),
            "routed_direct_jumps": int(
                self.diagnostics["transition_routed"].astype(bool).sum()
            ),
            "mean_completed_duration_weeks": (
                float(completed.mean()) if not completed.empty else 0.0
            ),
            "latest_duration_weeks": int(
                self.diagnostics["duration_weeks"].iloc[-1]
            ),
        }


def _validate_emissions(emissions: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(emissions, pd.DataFrame):
        raise TypeError("emission_probabilities must be a pandas DataFrame")
    if tuple(emissions.columns) != STATE_ORDER:
        raise ValueError(f"emission columns must be exactly ordered as {STATE_ORDER}")
    if not isinstance(emissions.index, pd.DatetimeIndex):
        raise TypeError("emission_probabilities must use a DatetimeIndex")
    if not emissions.index.is_monotonic_increasing or emissions.index.has_duplicates:
        raise ValueError("emission index must be unique and increasing")
    numeric = emissions.astype(float)
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("emission probabilities must be finite")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("emission probabilities must be in [0, 1]")
    totals = values.sum(axis=1)
    if not np.allclose(totals, 1.0, atol=1e-8, rtol=0.0):
        raise ValueError("each emission row must sum to one")
    return numeric


def _initial_prior(
    value: Mapping[str, float] | Sequence[float] | None,
) -> np.ndarray:
    if value is None:
        return np.full(len(STATE_ORDER), 1.0 / len(STATE_ORDER), dtype=float)
    if isinstance(value, Mapping):
        if set(value) != set(STATE_ORDER):
            raise ValueError(f"initial_prior keys must be exactly {STATE_ORDER}")
        prior = np.asarray([value[state] for state in STATE_ORDER], dtype=float)
    else:
        prior = np.asarray(value, dtype=float)
    if prior.shape != (len(STATE_ORDER),):
        raise ValueError(f"initial_prior must have {len(STATE_ORDER)} values")
    if not np.isfinite(prior).all() or (prior < 0.0).any():
        raise ValueError("initial_prior must be finite and non-negative")
    total = float(prior.sum())
    if total <= 0.0:
        raise ValueError("initial_prior must have positive mass")
    return prior / total


def _exit_hazard(
    state: str, duration_weeks: int, config: ShadowNowcastConfig
) -> float:
    if duration_weeks < config.minimum_duration_weeks:
        return 0.0
    base = (
        config.transition_base_hazard
        if state == "transition"
        else config.risk_state_base_hazard
    )
    older_by = duration_weeks - config.minimum_duration_weeks
    return min(
        config.maximum_exit_hazard,
        base + config.duration_hazard_slope * float(older_by),
    )


def _destinations(state: str) -> tuple[str, ...]:
    # A single weekly transition can never jump directly between risk_on/off.
    if state == "transition":
        return ("risk_on", "risk_off")
    return ("transition",)


def _entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0.0]
    if len(positive) == 0:
        return 0.0
    return float(-np.sum(positive * np.log(positive)) / log(len(STATE_ORDER)))


def filter_shadow_nowcast(
    emission_probabilities: pd.DataFrame,
    *,
    config: ShadowNowcastConfig | None = None,
    initial_prior: Mapping[str, float] | Sequence[float] | None = None,
) -> ShadowNowcastResult:
    """Filter state evidence sequentially with fixed explicit durations.

    Only row ``t`` and the previous posterior are used to produce the output at
    ``t``.  The returned hard state is a display-oriented MAP path with an
    additional causal routing rule: an apparent direct ``risk_on``/``risk_off``
    jump is shown as ``transition`` first.  Canonical supervised labels are not
    changed by this function.
    """

    emissions = _validate_emissions(emission_probabilities)
    settings = config or ShadowNowcastConfig()
    state_count = len(STATE_ORDER)
    max_duration = settings.maximum_duration_weeks
    posterior = np.zeros((state_count, max_duration), dtype=float)
    prior = _initial_prior(initial_prior)

    probability_rows: list[np.ndarray] = []
    diagnostic_rows: list[dict[str, object]] = []
    routed_states: list[str] = []
    previous_routed: str | None = None
    routed_duration = 0

    for row_number, (_, emission_row) in enumerate(emissions.iterrows()):
        emission = np.maximum(emission_row.to_numpy(dtype=float), settings.emission_floor)
        emission /= emission.sum()
        posterior_switch_probability = 0.0

        if row_number == 0:
            unnormalised = np.zeros_like(posterior)
            unnormalised[:, 0] = prior * emission
        else:
            predictive = np.zeros_like(posterior)
            switched_predictive = np.zeros_like(posterior)
            for state_position, state in enumerate(STATE_ORDER):
                for duration_position in range(max_duration):
                    mass = float(posterior[state_position, duration_position])
                    if mass <= 0.0:
                        continue
                    duration = duration_position + 1
                    exit_hazard = _exit_hazard(state, duration, settings)
                    next_duration_position = min(duration_position + 1, max_duration - 1)
                    predictive[state_position, next_duration_position] += mass * (
                        1.0 - exit_hazard
                    )
                    destinations = _destinations(state)
                    switched_mass = mass * exit_hazard / len(destinations)
                    for destination in destinations:
                        destination_position = _STATE_POSITION[destination]
                        predictive[destination_position, 0] += switched_mass
                        switched_predictive[destination_position, 0] += switched_mass
            unnormalised = predictive * emission[:, None]
            normaliser = float(unnormalised.sum())
            if normaliser > 0.0:
                posterior_switch_probability = float(
                    (switched_predictive * emission[:, None]).sum() / normaliser
                )

        normaliser = float(unnormalised.sum())
        if not np.isfinite(normaliser) or normaliser <= 0.0:
            raise RuntimeError("shadow filter lost all posterior probability mass")
        posterior = unnormalised / normaliser
        state_probabilities = posterior.sum(axis=1)
        state_probabilities /= state_probabilities.sum()

        raw_state = STATE_ORDER[int(np.argmax(state_probabilities))]
        transition_routed = (
            previous_routed in _OPPOSITE_RISK_STATE
            and raw_state == _OPPOSITE_RISK_STATE[previous_routed]
        )
        routed_state = "transition" if transition_routed else raw_state
        state_changed = previous_routed is not None and routed_state != previous_routed
        previous_duration = routed_duration if state_changed else np.nan
        routed_duration = 1 if state_changed or previous_routed is None else routed_duration + 1

        probability_rows.append(state_probabilities.copy())
        routed_states.append(routed_state)
        diagnostic_rows.append(
            {
                "raw_state": raw_state,
                "state": routed_state,
                "duration_weeks": routed_duration,
                "previous_duration_weeks": previous_duration,
                "state_changed": bool(state_changed),
                "transition_routed": bool(transition_routed),
                "confidence": float(np.max(state_probabilities)),
                "normalised_entropy": _entropy(state_probabilities),
                "posterior_switch_probability": posterior_switch_probability,
            }
        )
        previous_routed = routed_state

    probabilities = pd.DataFrame(
        probability_rows,
        index=emissions.index,
        columns=STATE_ORDER,
        dtype=float,
    )
    states = pd.Series(
        routed_states, index=emissions.index, name="shadow_regime", dtype="object"
    )
    diagnostics = pd.DataFrame(diagnostic_rows, index=emissions.index)
    return ShadowNowcastResult(
        probabilities=probabilities,
        states=states,
        diagnostics=diagnostics,
    )


__all__ = [
    "ShadowNowcastConfig",
    "ShadowNowcastResult",
    "filter_shadow_nowcast",
]
