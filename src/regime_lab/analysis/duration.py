"""Causal regime-duration research primitives for the v5 draft.

The module is deliberately standalone.  It consumes only the regime history
available at ``as_of``; the last spell in that truncated history is always
right-censored.  Kaplan-Meier estimates and bootstrap intervals therefore do
not use the eventual end of the current spell.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd


STATE_ORDER: tuple[str, ...] = ("risk_on", "transition", "risk_off")
SPELL_COLUMNS: tuple[str, ...] = (
    "episode_id",
    "state",
    "start_date",
    "end_date",
    "departure_date",
    "duration_weeks",
    "event_observed",
    "is_current",
)
KM_COLUMNS: tuple[str, ...] = (
    "duration_weeks",
    "at_risk",
    "events",
    "censored",
    "survival",
)


def _positive_integer(value: object, *, name: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if resolved != value:
        raise ValueError(f"{name} must be an integer")
    minimum = 0 if allow_zero else 1
    if resolved < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return resolved


def _validate_weekly_states(
    states: pd.Series,
    *,
    as_of: object | None = None,
) -> pd.Series:
    if not isinstance(states, pd.Series):
        raise TypeError("states must be a pandas Series")
    if not isinstance(states.index, pd.DatetimeIndex):
        raise TypeError("states must use a DatetimeIndex")
    if states.empty:
        raise ValueError("states must not be empty")
    if states.index.has_duplicates or not states.index.is_monotonic_increasing:
        raise ValueError("state dates must be unique and increasing")
    if len(states) > 1:
        calendar = states.index.tz_localize(None).normalize()
        deltas = calendar[1:] - calendar[:-1]
        if not bool((deltas == np.timedelta64(7, "D")).all()):
            raise ValueError("states must contain consecutive weekly observations")
    if states.isna().any():
        raise ValueError("states must not contain missing values")
    values = states.astype(str)
    unsupported = sorted(set(values).difference(STATE_ORDER))
    if unsupported:
        raise ValueError(f"unsupported states: {unsupported}")

    if as_of is None:
        return values.copy()
    cutoff = pd.Timestamp(as_of)
    if states.index.tz is None and cutoff.tzinfo is not None:
        cutoff = cutoff.tz_localize(None)
    elif states.index.tz is not None and cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize(states.index.tz)
    elif states.index.tz is not None and cutoff.tzinfo is not None:
        cutoff = cutoff.tz_convert(states.index.tz)
    truncated = values.loc[values.index <= cutoff]
    if truncated.empty:
        raise ValueError("as_of precedes the first state observation")
    return truncated


def causal_spell_table(
    states: pd.Series,
    *,
    as_of: object | None = None,
) -> pd.DataFrame:
    """Build sequential spells using only observations available at ``as_of``.

    A completed spell has ``event_observed=True`` and a departure date equal to
    the first weekly observation of the next state.  The final spell is always
    right-censored, even when later states exist in the caller's full series.
    """

    history = _validate_weekly_states(states, as_of=as_of)
    values = history.to_numpy(dtype=object)
    rows: list[dict[str, Any]] = []
    start = 0
    for position in range(1, len(history) + 1):
        departed = position < len(history) and values[position] != values[start]
        final = position == len(history)
        if not departed and not final:
            continue
        stop = position - 1
        event_observed = bool(departed)
        rows.append(
            {
                "episode_id": len(rows),
                "state": str(values[start]),
                "start_date": history.index[start],
                "end_date": history.index[stop],
                "departure_date": history.index[position]
                if event_observed
                else pd.NaT,
                "duration_weeks": int(position - start),
                "event_observed": event_observed,
                "is_current": not event_observed,
            }
        )
        start = position

    result = pd.DataFrame(rows, columns=SPELL_COLUMNS)
    result["episode_id"] = result["episode_id"].astype("int64")
    result["duration_weeks"] = result["duration_weeks"].astype("int64")
    result["event_observed"] = result["event_observed"].astype(bool)
    result["is_current"] = result["is_current"].astype(bool)
    return result


def _validate_spells(spells: pd.DataFrame, *, state: str) -> pd.DataFrame:
    if not isinstance(spells, pd.DataFrame):
        raise TypeError("spells must be a pandas DataFrame")
    missing = sorted(set(SPELL_COLUMNS).difference(spells.columns))
    if missing:
        raise ValueError(f"spell table is missing columns: {missing}")
    if state not in STATE_ORDER:
        raise ValueError(f"state must be one of {STATE_ORDER}")
    selected = spells.loc[spells["state"].astype(str).eq(state)].copy()
    if selected.empty:
        raise ValueError(f"no spells are available for state {state}")
    durations = pd.to_numeric(selected["duration_weeks"], errors="coerce")
    if durations.isna().any() or (durations < 1).any() or not np.allclose(
        durations, np.floor(durations)
    ):
        raise ValueError("spell durations must be positive integers")
    selected["duration_weeks"] = durations.astype("int64")
    selected["event_observed"] = selected["event_observed"].astype(bool)
    return selected.reset_index(drop=True)


def kaplan_meier_table(spells: pd.DataFrame, *, state: str) -> pd.DataFrame:
    """Return a discrete-time, right-censored Kaplan-Meier table."""

    selected = _validate_spells(spells, state=state)
    durations = selected["duration_weeks"].to_numpy(dtype=int)
    observed = selected["event_observed"].to_numpy(dtype=bool)
    survival = 1.0
    rows: list[dict[str, Any]] = []
    for duration in sorted(set(int(value) for value in durations)):
        at_risk = int(np.count_nonzero(durations >= duration))
        at_time = durations == duration
        events = int(np.count_nonzero(at_time & observed))
        censored = int(np.count_nonzero(at_time & ~observed))
        if events:
            survival *= 1.0 - events / at_risk
        rows.append(
            {
                "duration_weeks": duration,
                "at_risk": at_risk,
                "events": events,
                "censored": censored,
                "survival": float(survival),
            }
        )
    return pd.DataFrame(rows, columns=KM_COLUMNS)


def survival_at(km: pd.DataFrame, elapsed_weeks: int) -> float:
    """Evaluate the KM step function ``S(t)=P(T>t)`` at integer week ``t``."""

    elapsed = _positive_integer(
        elapsed_weeks, name="elapsed_weeks", allow_zero=True
    )
    if not isinstance(km, pd.DataFrame) or not set(KM_COLUMNS).issubset(km.columns):
        raise ValueError("km table has an invalid schema")
    eligible = km.loc[km["duration_weeks"].astype(int) <= elapsed]
    if eligible.empty:
        return 1.0
    return float(eligible.iloc[-1]["survival"])


def _resolve_horizons(horizons: Iterable[int]) -> tuple[int, ...]:
    resolved = tuple(
        _positive_integer(value, name="horizon") for value in horizons
    )
    if not resolved:
        raise ValueError("horizons must not be empty")
    if len(resolved) != len(set(resolved)):
        raise ValueError("horizons must not contain duplicates")
    return resolved


def conditional_duration_estimates(
    km: pd.DataFrame,
    *,
    elapsed_weeks: int,
    horizons: Sequence[int] = (4, 13),
    restriction_weeks: int = 52,
) -> dict[str, Any] | None:
    """Compute survival, departure risk, median, and remaining-life RMST.

    ``elapsed_weeks`` counts observed in-state weeks, while the KM event time is
    the number of weekly intervals from spell start to the first observation of
    the next state.  A spell observed in-state for ``d`` weeks is therefore
    conditioned just before KM time ``d`` (at ``d - 1``), so spells departing
    on the next observation remain in the risk set.

    For remaining duration ``R`` and restriction ``H``, the discrete RMST is
    ``sum(P(R > k), k=0..H-1)``.  It is therefore 52 when no departure has been
    observed beyond the current elapsed duration.
    """

    elapsed = _positive_integer(elapsed_weeks, name="elapsed_weeks")
    resolved_horizons = _resolve_horizons(horizons)
    restriction = _positive_integer(
        restriction_weeks, name="restriction_weeks"
    )
    conditioning_time = elapsed - 1
    denominator = survival_at(km, conditioning_time)
    if not np.isfinite(denominator) or denominator <= 0.0:
        return None

    conditional_survival = {
        f"{horizon}w": float(
            np.clip(
                survival_at(km, conditioning_time + horizon) / denominator,
                0.0,
                1.0,
            )
        )
        for horizon in resolved_horizons
    }
    departure_probability = {
        key: float(1.0 - value) for key, value in conditional_survival.items()
    }

    event_times = km.loc[km["events"].astype(int) > 0, "duration_weeks"]
    maximum_search = (
        max(0, int(event_times.max()) - conditioning_time)
        if not event_times.empty
        else 0
    )
    median_remaining: int | None = None
    for remaining in range(1, maximum_search + 1):
        ratio = survival_at(km, conditioning_time + remaining) / denominator
        if ratio <= 0.5 + 1e-12:
            median_remaining = remaining
            break

    rmst = sum(
        float(
            np.clip(
                survival_at(km, conditioning_time + remaining) / denominator,
                0.0,
                1.0,
            )
        )
        for remaining in range(restriction)
    )
    return {
        "conditional_survival": conditional_survival,
        "departure_probability": departure_probability,
        "median_remaining_weeks": median_remaining,
        "restricted_mean_remaining_weeks": float(rmst),
        "restriction_weeks": restriction,
    }


def _percentile_interval(
    values: Sequence[float],
    *,
    minimum_valid: int,
) -> dict[str, float] | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < minimum_valid:
        return None
    lower, upper = np.quantile(finite, [0.025, 0.975])
    return {"lower": float(lower), "upper": float(upper)}


def _episode_bootstrap_intervals(
    spells: pd.DataFrame,
    *,
    state: str,
    elapsed_weeks: int,
    horizons: tuple[int, ...],
    restriction_weeks: int,
    resamples: int,
    seed: int,
) -> tuple[dict[str, Any], int]:
    if resamples == 0:
        return {
            "conditional_survival": {f"{h}w": None for h in horizons},
            "departure_probability": {f"{h}w": None for h in horizons},
            "median_remaining_weeks": None,
            "restricted_mean_remaining_weeks": None,
        }, 0

    selected = _validate_spells(spells, state=state)
    generator = np.random.default_rng(seed)
    draws: list[dict[str, Any]] = []
    for _ in range(resamples):
        positions = generator.integers(0, len(selected), size=len(selected))
        sample = selected.iloc[positions].reset_index(drop=True)
        estimate = conditional_duration_estimates(
            kaplan_meier_table(sample, state=state),
            elapsed_weeks=elapsed_weeks,
            horizons=horizons,
            restriction_weeks=restriction_weeks,
        )
        if estimate is not None:
            draws.append(estimate)

    minimum_valid = max(1, int(np.ceil(resamples * 0.8)))
    conditional_ci: dict[str, Any] = {}
    departure_ci: dict[str, Any] = {}
    for horizon in horizons:
        key = f"{horizon}w"
        conditional_ci[key] = _percentile_interval(
            [draw["conditional_survival"][key] for draw in draws],
            minimum_valid=minimum_valid,
        )
        departure_ci[key] = _percentile_interval(
            [draw["departure_probability"][key] for draw in draws],
            minimum_valid=minimum_valid,
        )
    median_ci = _percentile_interval(
        [
            float(draw["median_remaining_weeks"])
            for draw in draws
            if draw["median_remaining_weeks"] is not None
        ],
        minimum_valid=minimum_valid,
    )
    rmst_ci = _percentile_interval(
        [draw["restricted_mean_remaining_weeks"] for draw in draws],
        minimum_valid=minimum_valid,
    )
    return (
        {
            "conditional_survival": conditional_ci,
            "departure_probability": departure_ci,
            "median_remaining_weeks": median_ci,
            "restricted_mean_remaining_weeks": rmst_ci,
        },
        len(draws),
    )


def conditional_duration_summary(
    spells: pd.DataFrame,
    *,
    state: str,
    elapsed_weeks: int,
    horizons: Sequence[int] = (4, 13),
    restriction_weeks: int = 52,
    min_completed_spells: int = 5,
    bootstrap_resamples: int = 1_999,
    bootstrap_seed: int = 17,
) -> dict[str, Any]:
    """Summarize one state's conditional duration with episode-bootstrap CIs."""

    elapsed = _positive_integer(elapsed_weeks, name="elapsed_weeks")
    resolved_horizons = _resolve_horizons(horizons)
    restriction = _positive_integer(
        restriction_weeks, name="restriction_weeks"
    )
    minimum_completed = _positive_integer(
        min_completed_spells, name="min_completed_spells"
    )
    resamples = _positive_integer(
        bootstrap_resamples,
        name="bootstrap_resamples",
        allow_zero=True,
    )
    seed = _positive_integer(
        bootstrap_seed, name="bootstrap_seed", allow_zero=True
    )
    selected = _validate_spells(spells, state=state)
    completed = int(selected["event_observed"].sum())
    censored = int((~selected["event_observed"]).sum())
    base: dict[str, Any] = {
        "method": "state_specific_kaplan_meier",
        "state": state,
        "elapsed_weeks": elapsed,
        "episodes": int(len(selected)),
        "completed_spells": completed,
        "censored_spells": censored,
        "minimum_completed_spells": minimum_completed,
        "bootstrap": {
            "unit": "episode",
            "resamples": resamples,
            "valid_resamples": 0,
            "seed": seed,
            "interval": 0.95,
        },
    }
    if completed < minimum_completed:
        return {
            **base,
            "status": "insufficient_history",
            "conditional_survival": {f"{h}w": None for h in resolved_horizons},
            "departure_probability": {f"{h}w": None for h in resolved_horizons},
            "median_remaining_weeks": None,
            "restricted_mean_remaining_weeks": None,
            "restriction_weeks": restriction,
            "ci95": None,
        }

    km = kaplan_meier_table(selected, state=state)
    point = conditional_duration_estimates(
        km,
        elapsed_weeks=elapsed,
        horizons=resolved_horizons,
        restriction_weeks=restriction,
    )
    if point is None:
        return {
            **base,
            "status": "unavailable",
            "conditional_survival": {f"{h}w": None for h in resolved_horizons},
            "departure_probability": {f"{h}w": None for h in resolved_horizons},
            "median_remaining_weeks": None,
            "restricted_mean_remaining_weeks": None,
            "restriction_weeks": restriction,
            "ci95": None,
        }

    intervals, valid_resamples = _episode_bootstrap_intervals(
        selected,
        state=state,
        elapsed_weeks=elapsed,
        horizons=resolved_horizons,
        restriction_weeks=restriction,
        resamples=resamples,
        seed=seed,
    )
    base["bootstrap"]["valid_resamples"] = valid_resamples
    return {
        **base,
        "status": "ok",
        **point,
        "ci95": intervals,
    }


def duration_context(
    states: pd.Series,
    *,
    as_of: object | None = None,
    horizons: Sequence[int] = (4, 13),
    restriction_weeks: int = 52,
    min_completed_spells: int = 5,
    bootstrap_resamples: int = 1_999,
    bootstrap_seed: int = 17,
) -> dict[str, Any]:
    """Build the causal duration context for the current state at ``as_of``."""

    history = _validate_weekly_states(states, as_of=as_of)
    spells = causal_spell_table(history)
    current = spells.loc[spells["is_current"]].iloc[-1]
    result = conditional_duration_summary(
        spells,
        state=str(current["state"]),
        elapsed_weeks=int(current["duration_weeks"]),
        horizons=horizons,
        restriction_weeks=restriction_weeks,
        min_completed_spells=min_completed_spells,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    return {
        "as_of": history.index[-1].date().isoformat(),
        **result,
    }


__all__ = [
    "KM_COLUMNS",
    "SPELL_COLUMNS",
    "STATE_ORDER",
    "causal_spell_table",
    "conditional_duration_estimates",
    "conditional_duration_summary",
    "duration_context",
    "kaplan_meier_table",
    "survival_at",
]
