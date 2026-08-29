"""Conditional asset-outcome research for the v5 draft.

This module reports historical state-conditioned outcomes only.  It does not
construct allocations, portfolio weights, or trading recommendations.  A
signal observed at origin ``t`` enters at the adjusted open of week ``t+1``;
an ``h``-week outcome exits at the adjusted close of week ``t+h``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


ASSETS: tuple[str, ...] = ("SPY", "QQQ", "IWM", "TLT", "HYG", "UUP")
HORIZONS: tuple[int, ...] = (1, 4, 13)
STATE_ORDER: tuple[str, ...] = ("risk_on", "transition", "risk_off")
DEFAULT_ASSET_COLUMNS: dict[str, str] = {
    asset: f"{asset.lower()}_close" for asset in ASSETS
}
DEFAULT_ASSET_OPEN_COLUMNS: dict[str, str] = {
    asset: f"{asset.lower()}_adjusted_open" for asset in ASSETS
}
OUTCOME_COLUMNS: tuple[str, ...] = (
    "origin_position",
    "origin_date",
    "entry_date",
    "exit_date",
    "state",
    "episode_id",
    "asset",
    "horizon_weeks",
    "execution_lag_weeks",
    "return_currency",
    "forward_return",
    "max_drawdown",
)
POINT_METRICS: tuple[str, ...] = (
    "mean_return",
    "median_return",
    "positive_rate",
    "annualized_volatility",
    "downside_volatility",
    "cvar_5",
    "mean_max_drawdown",
)
DECISION_USEFULNESS_METRICS: tuple[str, ...] = (
    "unconditional_benchmark_mean_return",
    "excess_mean_return",
    "episode_equal_mean_return",
    "episode_equal_unconditional_benchmark_mean_return",
    "episode_equal_excess_return",
)


@dataclass(frozen=True)
class ConditionalOutcomeResult:
    """Fully auditable origin-level outcomes and their grouped statistics."""

    outcomes: pd.DataFrame
    statistics: pd.DataFrame


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


def _validate_weekly_index(index: pd.Index, *, context: str) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"{context} must use a DatetimeIndex")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError(f"{context} dates must be unique and increasing")
    if len(index) > 1:
        calendar = index.tz_localize(None).normalize()
        deltas = calendar[1:] - calendar[:-1]
        if not bool((deltas == np.timedelta64(7, "D")).all()):
            raise ValueError(f"{context} must contain consecutive weekly observations")
    return index


def _validate_states(states: pd.Series) -> pd.Series:
    if not isinstance(states, pd.Series):
        raise TypeError("states must be a pandas Series")
    if states.empty:
        raise ValueError("states must not be empty")
    _validate_weekly_index(states.index, context="states")
    if states.isna().any():
        raise ValueError("states must not contain missing values")
    values = states.astype(str)
    unsupported = sorted(set(values).difference(STATE_ORDER))
    if unsupported:
        raise ValueError(f"unsupported states: {unsupported}")
    return values


def _resolve_prices(
    prices: pd.DataFrame,
    states: pd.Series,
    *,
    asset_columns: Mapping[str, str] | None,
    asset_open_columns: Mapping[str, str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")
    _validate_weekly_index(prices.index, context="prices")
    state_positions = prices.index.get_indexer(states.index)
    if bool((state_positions < 0).any()):
        raise ValueError("states dates must be a subset of the prices weekly index")
    if len(state_positions) > 1 and not bool((np.diff(state_positions) == 1).all()):
        raise ValueError("states dates must be a consecutive subset of prices")

    configured_closes = dict(
        DEFAULT_ASSET_COLUMNS if asset_columns is None else asset_columns
    )
    if set(configured_closes) != set(ASSETS):
        raise ValueError(f"asset_columns keys must be exactly {ASSETS}")
    configured_opens = dict(
        DEFAULT_ASSET_OPEN_COLUMNS
        if asset_open_columns is None
        else asset_open_columns
    )
    if set(configured_opens) != set(ASSETS):
        raise ValueError(f"asset_open_columns keys must be exactly {ASSETS}")

    resolved_closes: dict[str, pd.Series] = {}
    resolved_opens: dict[str, pd.Series] = {}
    for asset in ASSETS:
        requested_close = str(configured_closes[asset])
        close_candidates = tuple(
            dict.fromkeys((requested_close, asset, asset.lower()))
        )
        close_column = next(
            (name for name in close_candidates if name in prices.columns), None
        )
        if close_column is None:
            raise KeyError(
                f"missing adjusted-close column for {asset}: {requested_close}"
            )

        requested_open = str(configured_opens[asset])
        open_column = requested_open if requested_open in prices.columns else None
        if open_column is None:
            raise KeyError(
                f"missing adjusted-open column for {asset}: {requested_open}"
            )

        for price_kind, column, resolved in (
            ("close", close_column, resolved_closes),
            ("open", open_column, resolved_opens),
        ):
            values = pd.to_numeric(prices[column], errors="coerce").astype(float)
            finite = values[np.isfinite(values)]
            if (finite <= 0.0).any():
                raise ValueError(
                    f"{asset} adjusted {price_kind} prices must be positive where observed"
                )
            if np.isinf(values.to_numpy(dtype=float)).any():
                raise ValueError(
                    f"{asset} adjusted {price_kind} prices must not contain infinities"
                )
            resolved[asset] = values
    return (
        pd.DataFrame(resolved_closes, index=prices.index, dtype=float),
        pd.DataFrame(resolved_opens, index=prices.index, dtype=float),
        state_positions,
    )


def _state_episode_ids(states: pd.Series) -> pd.Series:
    changed = states.ne(states.shift(1))
    return (changed.cumsum() - 1).astype("int64").rename("episode_id")


def _within_window_max_drawdown(path: np.ndarray) -> float:
    running_peak = np.maximum.accumulate(path)
    drawdowns = path / running_peak - 1.0
    return float(np.min(drawdowns))


def build_forward_outcomes(
    prices: pd.DataFrame,
    states: pd.Series,
    *,
    asset_columns: Mapping[str, str] | None = None,
    asset_open_columns: Mapping[str, str] | None = None,
    horizons: Sequence[int] = HORIZONS,
    execution_lag_weeks: int = 1,
    return_currency: str = "USD",
) -> pd.DataFrame:
    """Materialize next-open, weekly-close forward outcomes.

    ``states`` may cover any consecutive subset of the full ``prices`` index.
    This lets reconstructed OOS origins use subsequent price rows without
    pretending those later rows were part of the signal sample.
    """

    labels = _validate_states(states)
    normalized_closes, normalized_opens, state_positions = _resolve_prices(
        prices,
        labels,
        asset_columns=asset_columns,
        asset_open_columns=asset_open_columns,
    )
    lag = _positive_integer(execution_lag_weeks, name="execution_lag_weeks")
    resolved_horizons = tuple(
        _positive_integer(value, name="horizon") for value in horizons
    )
    if not resolved_horizons or len(resolved_horizons) != len(
        set(resolved_horizons)
    ):
        raise ValueError("horizons must be non-empty and unique")
    currency = str(return_currency).strip().upper()
    if not currency:
        raise ValueError("return_currency must not be empty")

    episodes = _state_episode_ids(labels)
    rows: list[dict[str, Any]] = []
    for origin_position, origin_date in enumerate(labels.index):
        origin_price_position = int(state_positions[origin_position])
        entry_position = origin_price_position + lag
        if entry_position >= len(prices.index):
            continue
        for horizon in resolved_horizons:
            exit_position = entry_position + horizon - 1
            if exit_position >= len(prices.index):
                continue
            entry_date = prices.index[entry_position]
            exit_date = prices.index[exit_position]
            for asset in ASSETS:
                entry_open = float(normalized_opens[asset].iloc[entry_position])
                close_path = normalized_closes[asset].iloc[
                    entry_position : exit_position + 1
                ].to_numpy(dtype=float)
                path = np.concatenate(([entry_open], close_path))
                if len(close_path) != horizon or not np.isfinite(path).all():
                    continue
                rows.append(
                    {
                        "origin_position": origin_position,
                        "origin_date": origin_date,
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "state": str(labels.iloc[origin_position]),
                        "episode_id": int(episodes.iloc[origin_position]),
                        "asset": asset,
                        "horizon_weeks": horizon,
                        "execution_lag_weeks": lag,
                        "return_currency": currency,
                        "forward_return": float(close_path[-1] / entry_open - 1.0),
                        "max_drawdown": _within_window_max_drawdown(path),
                    }
                )
    result = pd.DataFrame(rows, columns=OUTCOME_COLUMNS)
    if result.empty:
        return result
    for column in (
        "origin_position",
        "episode_id",
        "horizon_weeks",
        "execution_lag_weeks",
    ):
        result[column] = result[column].astype("int64")
    return result.sort_values(
        ["origin_position", "horizon_weeks", "asset"], ignore_index=True
    )


def _historical_cvar_5(returns: np.ndarray) -> float:
    tail_count = max(1, int(np.ceil(0.05 * len(returns))))
    return float(np.mean(np.sort(returns)[:tail_count]))


def _point_metrics(frame: pd.DataFrame, *, horizon_weeks: int) -> dict[str, float]:
    returns = frame["forward_return"].to_numpy(dtype=float)
    drawdowns = frame["max_drawdown"].to_numpy(dtype=float)
    annualization = np.sqrt(52.0 / horizon_weeks)
    volatility = (
        float(np.std(returns, ddof=1) * annualization)
        if len(returns) > 1
        else float("nan")
    )
    downside = float(
        np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2)) * annualization
    )
    return {
        "mean_return": float(np.mean(returns)),
        "median_return": float(np.median(returns)),
        "positive_rate": float(np.mean(returns > 0.0)),
        "annualized_volatility": volatility,
        "downside_volatility": downside,
        "cvar_5": _historical_cvar_5(returns),
        "mean_max_drawdown": float(np.mean(drawdowns)),
    }


def _episode_bounded_circular_blocks(
    frame: pd.DataFrame,
    *,
    block_length: int,
) -> list[pd.DataFrame]:
    """Return fixed-length blocks with uniform weekly-origin marginals.

    Every origin in an episode-contiguous run is used once as a block start.
    Wrapping stays inside that run.  Consequently, at every position in the
    block, each origin appears in exactly one of the pooled blocks.  Uniformly
    sampling these blocks therefore targets the same weekly-origin weighting
    as :func:`_point_metrics`, including when episode lengths differ.
    """

    blocks: list[pd.DataFrame] = []
    ordered = frame.sort_values("origin_position").reset_index(drop=True)
    for _, episode in ordered.groupby("episode_id", sort=False):
        episode = episode.sort_values("origin_position").reset_index(drop=True)
        gap_group = episode["origin_position"].diff().ne(1).cumsum()
        for _, contiguous in episode.groupby(gap_group, sort=False):
            contiguous = contiguous.reset_index(drop=True)
            offsets = np.arange(block_length)
            for start in range(len(contiguous)):
                positions = (start + offsets) % len(contiguous)
                blocks.append(contiguous.iloc[positions].reset_index(drop=True))
    return blocks


def _percentile_interval(values: Sequence[float]) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return float("nan"), float("nan")
    lower, upper = np.quantile(finite, [0.025, 0.975])
    return float(lower), float(upper)


def _block_bootstrap_intervals(
    frame: pd.DataFrame,
    *,
    horizon_weeks: int,
    block_length: int,
    resamples: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    if resamples == 0:
        return {metric: (float("nan"), float("nan")) for metric in POINT_METRICS}
    blocks = _episode_bounded_circular_blocks(frame, block_length=block_length)
    if not blocks:
        return {metric: (float("nan"), float("nan")) for metric in POINT_METRICS}
    generator = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {metric: [] for metric in POINT_METRICS}
    target_rows = len(frame)
    for _ in range(resamples):
        sampled: list[pd.DataFrame] = []
        rows = 0
        while rows < target_rows:
            block = blocks[int(generator.integers(0, len(blocks)))]
            sampled.append(block)
            rows += len(block)
        replicate = pd.concat(sampled, ignore_index=True).iloc[:target_rows]
        metrics = _point_metrics(replicate, horizon_weeks=horizon_weeks)
        for metric, value in metrics.items():
            draws[metric].append(value)
    return {
        metric: _percentile_interval(values) for metric, values in draws.items()
    }


def _episode_equal_mean(frame: pd.DataFrame) -> float:
    """Give each contiguous regime episode one vote, irrespective of length."""

    if frame.empty:
        return float("nan")
    episode_means = frame.groupby("episode_id", sort=False)["forward_return"].mean()
    return float(episode_means.mean())


def _greedy_non_overlapping_count(frame: pd.DataFrame) -> int:
    """Count a deterministic maximum set of non-overlapping holding windows.

    Entry is at the weekly open and exit is at the weekly close, so two windows
    sharing the same date still overlap intraperiod.  The next selected entry
    must therefore be strictly after the previous selected exit.
    """

    if frame.empty:
        return 0
    intervals = frame.assign(
        _entry=pd.to_datetime(frame["entry_date"], utc=True),
        _exit=pd.to_datetime(frame["exit_date"], utc=True),
    ).sort_values(["_entry", "_exit"], kind="mergesort")
    selected = 0
    previous_exit: pd.Timestamp | None = None
    for entry_value, exit_value in zip(
        intervals["_entry"], intervals["_exit"], strict=True
    ):
        entry = pd.Timestamp(entry_value)
        exit_date = pd.Timestamp(exit_value)
        if previous_exit is None or entry > previous_exit:
            selected += 1
            previous_exit = exit_date
    return selected


def _whole_episode_bootstrap_interval(
    frame: pd.DataFrame,
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap whole episodes and retain episode-equal weighting.

    Rows are never split across bootstrap units.  This complements the legacy
    weekly-origin block bootstrap rather than changing its certified estimates.
    """

    if resamples == 0 or frame.empty:
        return float("nan"), float("nan")
    episode_means = (
        frame.groupby("episode_id", sort=False)["forward_return"]
        .mean()
        .to_numpy(dtype=float)
    )
    if len(episode_means) == 0:
        return float("nan"), float("nan")
    generator = np.random.default_rng(seed)
    draws = [
        float(np.mean(episode_means[generator.integers(0, len(episode_means), len(episode_means))]))
        for _ in range(resamples)
    ]
    return _percentile_interval(draws)


def summarize_conditional_outcomes(
    outcomes: pd.DataFrame,
    *,
    assets: Sequence[str] = ASSETS,
    horizons: Sequence[int] = HORIZONS,
    states: Sequence[str] = STATE_ORDER,
    min_observations: int = 20,
    min_unique_episodes: int = 5,
    min_non_overlapping_observations: int = 5,
    bootstrap_block_weeks: int = 13,
    bootstrap_resamples: int = 1_999,
    bootstrap_seed: int = 17,
) -> pd.DataFrame:
    """Aggregate outcomes and attach episode-bounded circular-block CIs."""

    if not isinstance(outcomes, pd.DataFrame):
        raise TypeError("outcomes must be a pandas DataFrame")
    missing = sorted(set(OUTCOME_COLUMNS).difference(outcomes.columns))
    if missing:
        raise ValueError(f"outcomes are missing columns: {missing}")
    resolved_assets = tuple(str(asset).upper() for asset in assets)
    if set(resolved_assets) != set(ASSETS):
        raise ValueError(f"assets must be exactly {ASSETS}")
    resolved_horizons = tuple(
        _positive_integer(value, name="horizon") for value in horizons
    )
    if set(resolved_horizons) != set(HORIZONS):
        raise ValueError(f"horizons must be exactly {HORIZONS}")
    resolved_states = tuple(str(state) for state in states)
    if set(resolved_states) != set(STATE_ORDER):
        raise ValueError(f"states must be exactly {STATE_ORDER}")
    minimum_rows = _positive_integer(min_observations, name="min_observations")
    minimum_episodes = _positive_integer(
        min_unique_episodes, name="min_unique_episodes"
    )
    minimum_non_overlapping = _positive_integer(
        min_non_overlapping_observations,
        name="min_non_overlapping_observations",
    )
    block_length = _positive_integer(
        bootstrap_block_weeks, name="bootstrap_block_weeks"
    )
    resamples = _positive_integer(
        bootstrap_resamples,
        name="bootstrap_resamples",
        allow_zero=True,
    )
    seed = _positive_integer(
        bootstrap_seed, name="bootstrap_seed", allow_zero=True
    )

    rows: list[dict[str, Any]] = []
    unconditional = {
        (str(asset), int(horizon)): group.sort_values("origin_position")
        for (asset, horizon), group in outcomes.groupby(
            ["asset", "horizon_weeks"], sort=False
        )
    }
    for state_index, state in enumerate(resolved_states):
        for asset_index, asset in enumerate(resolved_assets):
            for horizon in resolved_horizons:
                group = outcomes.loc[
                    outcomes["state"].astype(str).eq(state)
                    & outcomes["asset"].astype(str).eq(asset)
                    & outcomes["horizon_weeks"].astype(int).eq(horizon)
                ].sort_values("origin_position")
                count = int(len(group))
                unique_episodes = int(group["episode_id"].nunique())
                non_overlapping_count = _greedy_non_overlapping_count(group)
                supported = (
                    count >= minimum_rows
                    and unique_episodes >= minimum_episodes
                    and non_overlapping_count >= minimum_non_overlapping
                )
                metrics = (
                    _point_metrics(group, horizon_weeks=horizon)
                    if count
                    else {metric: float("nan") for metric in POINT_METRICS}
                )
                group_seed = seed + state_index * 10_000 + asset_index * 100 + horizon
                intervals = (
                    _block_bootstrap_intervals(
                        group,
                        horizon_weeks=horizon,
                        block_length=block_length,
                        resamples=resamples,
                        seed=group_seed,
                    )
                    if supported
                    else {
                        metric: (float("nan"), float("nan"))
                        for metric in POINT_METRICS
                    }
                )
                benchmark = unconditional.get((asset, horizon), outcomes.iloc[0:0])
                benchmark_mean = (
                    float(benchmark["forward_return"].mean())
                    if len(benchmark)
                    else float("nan")
                )
                episode_equal_mean = _episode_equal_mean(group)
                episode_equal_benchmark_mean = _episode_equal_mean(benchmark)
                episode_equal_benchmark_episodes = int(
                    benchmark["episode_id"].nunique()
                )
                episode_interval = (
                    _whole_episode_bootstrap_interval(
                        group,
                        resamples=resamples,
                        seed=group_seed + 1_000_000,
                    )
                    if supported
                    else (float("nan"), float("nan"))
                )
                row: dict[str, Any] = {
                    "state": state,
                    "asset": asset,
                    "horizon_weeks": horizon,
                    "execution_lag_weeks": int(group["execution_lag_weeks"].iloc[0])
                    if count
                    else 1,
                    "return_currency": str(group["return_currency"].iloc[0])
                    if count
                    else "USD",
                    "sample_start": pd.Timestamp(group["origin_date"].min()).date().isoformat()
                    if count
                    else None,
                    "sample_end": pd.Timestamp(group["origin_date"].max()).date().isoformat()
                    if count
                    else None,
                    "n": count,
                    "non_overlapping_n": non_overlapping_count,
                    "unique_episodes": unique_episodes,
                    "status": "ok" if supported else "insufficient_support",
                    "minimum_observations": minimum_rows,
                    "minimum_unique_episodes": minimum_episodes,
                    "minimum_non_overlapping_observations": (
                        minimum_non_overlapping
                    ),
                    "bootstrap_method": "episode_bounded_circular_block",
                    "bootstrap_block_weeks": block_length,
                    "bootstrap_resamples": resamples,
                    "bootstrap_seed": group_seed,
                    "unconditional_benchmark_method": (
                        "same_asset_horizon_all_origins_mean"
                    ),
                    "unconditional_benchmark_n": int(len(benchmark)),
                    "unconditional_benchmark_mean_return": benchmark_mean,
                    "excess_mean_return": (
                        float(metrics["mean_return"] - benchmark_mean)
                        if count and np.isfinite(benchmark_mean)
                        else float("nan")
                    ),
                    "episode_equal_mean_return": episode_equal_mean,
                    "episode_equal_unconditional_benchmark_method": (
                        "same_asset_horizon_all_state_episodes_equal_weight"
                    ),
                    "episode_equal_unconditional_benchmark_episode_n": (
                        episode_equal_benchmark_episodes
                    ),
                    "episode_equal_unconditional_benchmark_mean_return": (
                        episode_equal_benchmark_mean
                    ),
                    "episode_equal_excess_return": (
                        float(
                            episode_equal_mean
                            - episode_equal_benchmark_mean
                        )
                        if np.isfinite(episode_equal_mean)
                        and np.isfinite(episode_equal_benchmark_mean)
                        else float("nan")
                    ),
                    "episode_bootstrap_method": "whole_episode_resampling",
                    "episode_bootstrap_resamples": resamples,
                    "episode_bootstrap_seed": group_seed + 1_000_000,
                    "episode_equal_mean_return_ci95_lower": episode_interval[0],
                    "episode_equal_mean_return_ci95_upper": episode_interval[1],
                    **metrics,
                }
                for metric, (lower, upper) in intervals.items():
                    row[f"{metric}_ci95_lower"] = lower
                    row[f"{metric}_ci95_upper"] = upper
                rows.append(row)
    return pd.DataFrame(rows)


def build_conditional_asset_statistics(
    prices: pd.DataFrame,
    states: pd.Series,
    *,
    asset_columns: Mapping[str, str] | None = None,
    asset_open_columns: Mapping[str, str] | None = None,
    min_observations: int = 20,
    min_unique_episodes: int = 5,
    min_non_overlapping_observations: int = 5,
    bootstrap_block_weeks: int = 13,
    bootstrap_resamples: int = 1_999,
    bootstrap_seed: int = 17,
) -> ConditionalOutcomeResult:
    """Build the fixed six-asset, 1/4/13-week conditional outcome study."""

    outcomes = build_forward_outcomes(
        prices,
        states,
        asset_columns=asset_columns,
        asset_open_columns=asset_open_columns,
        horizons=HORIZONS,
        execution_lag_weeks=1,
        return_currency="USD",
    )
    statistics = summarize_conditional_outcomes(
        outcomes,
        min_observations=min_observations,
        min_unique_episodes=min_unique_episodes,
        min_non_overlapping_observations=min_non_overlapping_observations,
        bootstrap_block_weeks=bootstrap_block_weeks,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    return ConditionalOutcomeResult(outcomes=outcomes, statistics=statistics)


__all__ = [
    "ASSETS",
    "ConditionalOutcomeResult",
    "DEFAULT_ASSET_COLUMNS",
    "DEFAULT_ASSET_OPEN_COLUMNS",
    "DECISION_USEFULNESS_METRICS",
    "HORIZONS",
    "OUTCOME_COLUMNS",
    "POINT_METRICS",
    "STATE_ORDER",
    "build_conditional_asset_statistics",
    "build_forward_outcomes",
    "summarize_conditional_outcomes",
]
