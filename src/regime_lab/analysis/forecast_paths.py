"""Research-only competing risks and exact state/duration path probabilities.

All horizons are marginals of the same path law. Economic covariates stay at the
origin; state and spell age evolve. No operational forecast or label is replaced.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from regime_lab.analysis.boundary_forecast import BoundaryInputs
from regime_lab.schema import STATE_ORDER

PATH_MODEL = "directional_duration_hazard"
MARKOV_PATH_MODEL = "markov_duration_path_baseline"
FIRST_DEPARTURE_ORDER = ("no_departure", *STATE_ORDER)
HORIZONS = (1, 4, 13)


def _horizons(values: Sequence[int]) -> tuple[int, ...]:
    if not values or any(isinstance(h, bool) or not isinstance(h, int) or h < 1 for h in values):
        raise ValueError("horizons must be positive integers")
    if tuple(values) != tuple(sorted(set(values))):
        raise ValueError("horizons must be unique and increasing")
    return tuple(values)


def _probability(values: np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("invalid next-state probability")
    if not np.isclose(p.sum(), 1, atol=1e-10, rtol=0):
        raise ValueError("next-state probabilities must sum to one")
    return p


def project_multistate_paths(
    transition: Callable[[str, int], np.ndarray], current_state: str,
    current_duration_weeks: int, *, horizons: Sequence[int] = HORIZONS,
) -> list[dict]:
    """Enumerate a compressed path distribution without Monte Carlo error.

    A risk-off origin has entry probability zero until it *leaves and returns*.
    Occupancy counts any future risk-off week (but not the observed origin).
    First departure is absorbing bookkeeping, while the market state can return.
    """
    horizons = _horizons(horizons)
    if current_state not in STATE_ORDER:
        raise ValueError("unknown current state")
    if isinstance(current_duration_weeks, bool) or not isinstance(current_duration_weeks, int) or current_duration_weeks < 1:
        raise ValueError("duration must be a positive integer")
    start = STATE_ORDER.index(current_state)
    # state, age, first destination (-1 = no departure), new entry, occupancy
    mass = {(start, current_duration_weeks, -1, False, False): 1.0}
    cache: dict[tuple[int, int], np.ndarray] = {}
    results = []
    for step in range(1, max(horizons) + 1):
        following: defaultdict[tuple, float] = defaultdict(float)
        for (state, age, first, entered, occupied), weight in mass.items():
            if (state, age) not in cache:
                cache[state, age] = _probability(transition(STATE_ORDER[state], age))
            for destination, probability in enumerate(cache[state, age]):
                if probability == 0:
                    continue
                key = (destination, age + 1 if state == destination else 1,
                       destination if first == -1 and destination != start else first,
                       entered or (state != 2 and destination == 2),
                       occupied or destination == 2)
                following[key] += weight * probability
        mass = dict(following)
        if step not in horizons:
            continue
        endpoint = np.zeros(3)
        departure = np.zeros(4)
        entry = occupancy = 0.
        for (state, _, first, entered, occupied), weight in mass.items():
            endpoint[state] += weight
            departure[first + 1] += weight
            entry += weight * entered
            occupancy += weight * occupied
        # Tiny summation drift is allowed, invalid model outputs are not repaired.
        if not np.isclose(endpoint.sum(), 1, atol=1e-10, rtol=0):
            raise RuntimeError("path mass was not conserved")
        results.append({"horizon_weeks": step,
                        "first_departure": dict(zip(FIRST_DEPARTURE_ORDER, map(float, departure))),
                        "endpoint": dict(zip(STATE_ORDER, map(float, endpoint))),
                        "any_risk_off_entry": float(np.clip(entry, 0, 1)),
                        "any_risk_off_occupancy": float(np.clip(occupancy, 0, 1))})
    return results


def path_outcomes(states: pd.Series, horizons: Sequence[int] = HORIZONS) -> pd.DataFrame:
    """Matured outcomes, using observed future paths only for evaluation."""
    horizons = _horizons(horizons)
    if not isinstance(states.index, pd.DatetimeIndex) or states.index.has_duplicates or not states.index.is_monotonic_increasing:
        raise ValueError("states require a unique chronological date index")
    if states.empty or not states.isin(STATE_ORDER).all():
        raise ValueError("states must contain only official states")
    values = states.to_numpy()
    rows = []
    for horizon in horizons:
        for origin in range(len(states) - horizon):
            current = values[origin]
            path = values[origin + 1:origin + horizon + 1]
            changed = np.flatnonzero(path != current)
            previous = values[origin:origin + horizon]
            rows.append({"origin_date": states.index[origin],
                         "target_date": states.index[origin + horizon],
                         "horizon_weeks": horizon, "current_state": current,
                         "first_departure": str(path[changed[0]]) if len(changed) else "no_departure",
                         "endpoint": str(path[-1]),
                         "any_risk_off_entry": bool(((previous != "risk_off") & (path == "risk_off")).any()),
                         "any_risk_off_occupancy": bool((path == "risk_off").any())})
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class DirectionHazardConfig:
    version: str = "direction-hazard-v1"
    train_window_weeks: int = 520
    logistic_c: float = .1
    minimum_rows: int = 52
    minimum_events: int = 4
    prior_count: float = 1.

    def __post_init__(self):
        if self.version != "direction-hazard-v1":
            raise ValueError("unsupported hazard version")
        for key in ("train_window_weeks", "minimum_rows", "minimum_events"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not np.isfinite([self.logistic_c, self.prior_count]).all() or min(self.logistic_c, self.prior_count) <= 0:
            raise ValueError("hazard regularization and prior must be positive")


def hazard_features(inputs: BoundaryInputs) -> pd.DataFrame:
    feature = inputs.features
    # Current standardized shock is observed; future shocks never enter features.
    shock = (feature.return_1w / feature.volatility.shift(1)).clip(-8, 8)
    return pd.DataFrame({"state_transition": inputs.states.eq("transition").astype(float),
                         "state_risk_off": inputs.states.eq("risk_off").astype(float),
                         "distance_lower": feature.distance_lower,
                         "distance_upper": feature.distance_upper,
                         "log_state_age": feature.log_state_age,
                         "negative_shock": -shock.clip(upper=0),
                         "positive_shock": shock.clip(lower=0)}, index=feature.index)


@dataclass
class FittedDirectionHazard:
    features_at_origin: np.ndarray
    estimators: dict[str, object | None]
    prior_odds: dict[str, np.ndarray]
    routes: dict[str, np.ndarray]
    audit: dict

    def probability_grid(self, ages: Sequence[int]) -> dict[tuple[str, int], np.ndarray]:
        """Batch sklearn inference for all reachable state/age pairs."""
        keys = [(state, age) for state in STATE_ORDER for age in sorted(set(ages))]
        matrix = np.tile(self.features_at_origin, (len(keys), 1))
        for row, (state, age) in enumerate(keys):
            matrix[row, :2] = [state == "transition", state == "risk_off"]
            matrix[row, 4] = np.log1p(age)
        odds = {}
        for direction, estimator in self.estimators.items():
            if estimator is None:
                value = np.asarray([self.prior_odds[direction][STATE_ORDER.index(state)] for state, _ in keys])
            else:
                # exp(log odds) supplies competing event odds relative to stay.
                value = np.exp(np.clip(estimator.decision_function(matrix), -20, 20))
            impossible = np.asarray([state == ("risk_off" if direction == "worsening" else "risk_on") for state, _ in keys])
            value[impossible] = 0
            odds[direction] = value
        output = {}
        for row, (state, age) in enumerate(keys):
            current = STATE_ORDER.index(state)
            total = 1 + odds["worsening"][row] + odds["recovery"][row]
            probability = np.zeros(3)
            probability[current] = 1 / total
            for direction in ("worsening", "recovery"):
                probability += odds[direction][row] / total * self.routes[direction][current]
            output[state, age] = _probability(probability)
        return output

    def paths(self, current_state: str, age: int, horizons: Sequence[int] = HORIZONS) -> list[dict]:
        horizons = _horizons(horizons)
        ages = list(range(1, max(horizons) + 1)) + list(range(age, age + max(horizons)))
        grid = self.probability_grid(ages)
        return project_multistate_paths(lambda state, duration: grid[state, duration], current_state, age, horizons=horizons)


def fit_direction_hazard(inputs: BoundaryInputs, position: int, *,
                         config: DirectionHazardConfig = DirectionHazardConfig(),
                         features: pd.DataFrame | None = None) -> FittedDirectionHazard:
    if isinstance(position, bool) or position < 521 or position >= len(inputs.states):
        raise ValueError("origin must follow frozen label fit period")
    feature = hazard_features(inputs) if features is None else features
    if not feature.index.equals(inputs.states.index):
        raise ValueError("hazard features and official state indexes differ")
    train = np.arange(max(52, position - config.train_window_weeks - 1), position - 1)
    current = np.asarray([STATE_ORDER.index(s) for s in inputs.states.iloc[train]])
    following = np.asarray([STATE_ORDER.index(s) for s in inputs.states.iloc[train + 1]])
    estimators, priors, routes, audits = {}, {}, {}, {}
    for direction, sign in (("worsening", 1), ("recovery", -1)):
        delta = (following - current) * sign
        eligible = (delta >= 0) & (current != (2 if sign == 1 else 0))
        x, y = feature.iloc[train].to_numpy()[eligible], (delta[eligible] > 0).astype(int)
        estimator = None
        reason = None
        if len(y) < config.minimum_rows or min(int(y.sum()), int((y == 0).sum())) < config.minimum_events:
            reason = "insufficient_completed_direction_events"
        else:
            estimator = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                      StandardScaler(), LogisticRegression(C=config.logistic_c, max_iter=1000, tol=1e-6))
            estimator.fit(x, y)
        prior_odds, routing = np.zeros(3), np.zeros((3, 3))
        for state in range(3):
            destinations = np.flatnonzero((np.arange(3) - state) * sign > 0)
            if not len(destinations):
                continue
            stays = int(((current == state) & (following == state)).sum())
            events = int(((current == state) & ((following - state) * sign > 0)).sum())
            prior_odds[state] = (events + config.prior_count) / (stays + config.prior_count)
            counts = np.asarray([int(((current == state) & (following == target)).sum()) + config.prior_count for target in destinations])
            routing[state, destinations] = counts / counts.sum()
        estimators[direction], priors[direction], routes[direction] = estimator, prior_odds, routing
        audits[direction] = {"training_rows": len(y), "events": int(y.sum()),
                             "fallback": reason is not None, "fallback_reason": reason}
    return FittedDirectionHazard(feature.iloc[position].to_numpy(float), estimators, priors, routes,
                                 {"last_train_target": inputs.states.index[train[-1] + 1].isoformat(),
                                  "train_size": len(train), "directions": audits})


def fit_markov_path(states: pd.Series, position: int, *, train_window_weeks: int = 520) -> np.ndarray:
    """Add-one Markov baseline using the same strictly completed training window."""
    if position < 2 or position >= len(states) or train_window_weeks < 1:
        raise ValueError("invalid Markov training window/origin")
    matrix = np.ones((3, 3))
    for origin in range(max(0, position - train_window_weeks - 1), position - 1):
        matrix[STATE_ORDER.index(states.iloc[origin]), STATE_ORDER.index(states.iloc[origin + 1])] += 1
    return matrix / matrix.sum(axis=1, keepdims=True)
