"""Causal feature engineering for canonical weekly regime data.

The functions in this module deliberately avoid interpolation, backfilling, and
centred windows.  Availability/release-time filtering belongs in the data
layer; once a canonical as-of frame reaches this module every transformation
uses only the current row and rows to its left.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from regime_lab.analysis.structural_features import build_market_group_features


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for :func:`build_weekly_features`.

    ``price_columns=None`` enables a conservative name-based discovery.  A
    production pipeline should pass the columns explicitly so a macro level is
    never accidentally interpreted as a traded price.
    """

    price_columns: tuple[str, ...] | None = None
    price_groups: Mapping[str, tuple[str, ...]] | None = None
    benchmark_column: str = "spy_close"
    volume_columns: tuple[str, ...] | None = None
    return_lookbacks: tuple[int, ...] = (1, 4, 13, 26, 52)
    volatility_windows: tuple[int, ...] = (4, 13, 26)
    drawdown_windows: tuple[int, ...] = (13, 52)
    generic_change_lookbacks: tuple[int, ...] = (1, 4, 13)
    generic_z_windows: tuple[int, ...] = (13, 52)
    relative_lookbacks: tuple[int, ...] = (4, 13, 26)
    breadth_return_lookbacks: tuple[int, ...] = (1, 4)
    breadth_trend_windows: tuple[int, ...] = (13, 26)
    dispersion_lookbacks: tuple[int, ...] = (1, 4)
    correlation_windows: tuple[int, ...] = (13, 26)
    spread_lookbacks: tuple[int, ...] = (4, 13)
    generic_change_z_pairs: tuple[tuple[int, int], ...] = ((4, 52),)
    include_generic_levels: bool = True


# These liquid bond, commodity, and currency ETFs are deliberately excluded
# from equity breadth.  Unknown configured price columns remain eligible so the
# feature builder also works with a custom US-equity universe.
_NON_EQUITY_PRICE_STEMS = frozenset(
    {"shy", "ief", "tlt", "hyg", "lqd", "gld", "uup"}
)

# A compact set of economically interpretable relative-price signals.  Missing
# legs are skipped rather than synthesized or imputed.
_MARKET_SPREAD_PAIRS = (
    ("cyclical_defensive", "xly_close", "xlp_close"),
    ("growth_defensive", "xlk_close", "xlu_close"),
    ("high_yield_investment_grade", "hyg_close", "lqd_close"),
    ("credit_treasury", "hyg_close", "ief_close"),
    ("long_short_treasury", "tlt_close", "shy_close"),
)

_CROSS_ASSET_CORRELATION_PAIRS = (
    ("equity_duration", "spy_close", "tlt_close"),
    ("equity_credit", "spy_close", "hyg_close"),
    ("equity_usd", "spy_close", "uup_close"),
    ("credit_duration", "hyg_close", "tlt_close"),
)


def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("frame must use a DatetimeIndex")
    if frame.index.has_duplicates:
        raise ValueError("frame index must be unique")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("frame index must be sorted in increasing time order")
    return frame


def _discover_price_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    discovered: list[str] = []
    for column in frame.columns:
        name = str(column).lower()
        if name.endswith("_close") or name.endswith("_price"):
            if pd.api.types.is_numeric_dtype(frame[column]):
                discovered.append(str(column))
    return tuple(discovered)


def _discover_volume_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        str(column)
        for column in frame.columns
        if str(column).lower().endswith("_volume")
        and pd.api.types.is_numeric_dtype(frame[column])
    )


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], kind: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise KeyError(f"missing configured {kind} columns: {missing}")


def _positive_log(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    return np.log(numeric.where(numeric > 0.0))


def _rolling_min_periods(window: int) -> int:
    # Short windows should not emit a one-observation volatility of zero, while
    # long windows can become usable before an entire year has accumulated.
    return max(2, min(window, (window + 1) // 2))


def _price_stem(column: str) -> str:
    name = str(column).lower()
    for suffix in ("_close", "_price"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _row_mean_with_minimum(values: pd.DataFrame, minimum: int) -> pd.Series:
    """Return a row mean only when enough actually observed values exist."""

    counts = values.notna().sum(axis=1)
    return values.mean(axis=1, skipna=True).where(counts >= minimum)


def _average_pairwise_correlation(
    returns: pd.DataFrame,
    window: int,
) -> pd.Series:
    """Past-only mean of available pairwise rolling correlations."""

    correlations = [
        returns[left]
        .rolling(window, min_periods=_rolling_min_periods(window))
        .corr(returns[right])
        for left, right in combinations(returns.columns, 2)
    ]
    if not correlations:
        return pd.Series(np.nan, index=returns.index, dtype=float)
    pairwise = pd.concat(correlations, axis=1)
    return _row_mean_with_minimum(pairwise, 1).clip(-1.0, 1.0)


def build_cross_asset_correlation_features(
    frame: pd.DataFrame,
    *,
    windows: Iterable[int] = (13, 26),
) -> pd.DataFrame:
    """Build fixed, scale-free correlations from already-approved ETF inputs."""

    validated = _validate_frame(frame)
    raw_windows = tuple(windows)
    if (
        not raw_windows
        or any(
            isinstance(window, (bool, np.bool_))
            or not isinstance(window, (int, np.integer))
            or int(window) < 2
            for window in raw_windows
        )
    ):
        raise ValueError("correlation windows must contain integers of at least two")
    resolved_windows = tuple(dict.fromkeys(int(window) for window in raw_windows))
    output: dict[str, pd.Series] = {}
    for label, left_name, right_name in _CROSS_ASSET_CORRELATION_PAIRS:
        if left_name not in validated or right_name not in validated:
            continue
        left_return = _positive_log(validated[left_name]).diff(1)
        right_return = _positive_log(validated[right_name]).diff(1)
        for window in resolved_windows:
            output[f"cross_asset__{label}__correlation_{window}w"] = (
                left_return.rolling(
                    window,
                    min_periods=_rolling_min_periods(window),
                ).corr(right_return).clip(-1.0, 1.0)
            )
    return pd.DataFrame(output, index=validated.index, dtype=float)


def build_weekly_features(
    frame: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Build a deterministic, past-only weekly feature matrix.

    Parameters
    ----------
    frame:
        Canonical as-of data indexed by completed US trading week.  Values must
        already respect their release timestamps.  Missing values are retained
        for fold-local imputation by the model pipeline.
    config:
        Feature selection and lookbacks.

    Returns
    -------
    pandas.DataFrame
        Numeric feature matrix on exactly the input index.  The output contains
        no infinities and performs no forward-looking fill.
    """

    frame = _validate_frame(frame)
    cfg = config or FeatureConfig()
    price_columns = (
        _discover_price_columns(frame)
        if cfg.price_columns is None
        else tuple(cfg.price_columns)
    )
    volume_columns = (
        _discover_volume_columns(frame)
        if cfg.volume_columns is None
        else tuple(cfg.volume_columns)
    )
    _require_columns(frame, price_columns, "price")
    _require_columns(frame, volume_columns, "volume")

    features: dict[str, pd.Series] = {}
    price_set = set(price_columns)
    volume_set = set(volume_columns)
    log_prices = {column: _positive_log(frame[column]) for column in price_columns}
    one_week_returns = {column: series.diff(1) for column, series in log_prices.items()}

    for column in price_columns:
        log_price = log_prices[column]
        one_week_return = one_week_returns[column]

        for lookback in cfg.return_lookbacks:
            ret = log_price.diff(lookback)
            features[f"{column}__log_return_{lookback}w"] = ret
            if lookback > 1:
                trailing_vol = one_week_return.rolling(
                    lookback, min_periods=_rolling_min_periods(lookback)
                ).std(ddof=0)
                denominator = trailing_vol * np.sqrt(float(lookback))
                features[f"{column}__risk_adjusted_trend_{lookback}w"] = (
                    ret / denominator.replace(0.0, np.nan)
                )

        for window in cfg.volatility_windows:
            min_periods = _rolling_min_periods(window)
            realised = one_week_return.rolling(
                window, min_periods=min_periods
            ).std(ddof=0) * np.sqrt(52.0)
            downside = one_week_return.clip(upper=0.0).pow(2).rolling(
                window, min_periods=min_periods
            ).mean().pow(0.5) * np.sqrt(52.0)
            features[f"{column}__realized_vol_{window}w"] = realised
            features[f"{column}__downside_vol_{window}w"] = downside

        numeric_price = pd.to_numeric(frame[column], errors="coerce").astype(float)
        for window in cfg.drawdown_windows:
            min_periods = _rolling_min_periods(window)
            trailing_peak = numeric_price.rolling(
                window, min_periods=min_periods
            ).max()
            trailing_floor = numeric_price.rolling(
                window, min_periods=min_periods
            ).min()
            features[f"{column}__drawdown_{window}w"] = (
                numeric_price / trailing_peak - 1.0
            )
            features[f"{column}__recovery_from_low_{window}w"] = (
                numeric_price / trailing_floor - 1.0
            )
        features[f"{column}__missing"] = numeric_price.isna().astype(float)

    # Relative performance captures breadth and cross-asset risk appetite while
    # remaining scale-free.  The benchmark itself is skipped.
    if cfg.benchmark_column in price_set:
        benchmark_log = log_prices[cfg.benchmark_column]
        for column in price_columns:
            if column == cfg.benchmark_column:
                continue
            relative_log = _positive_log(frame[column]) - benchmark_log
            for lookback in cfg.relative_lookbacks:
                features[
                    f"{column}_vs_{cfg.benchmark_column}__relative_return_{lookback}w"
                ] = relative_log.diff(lookback)

    # Market internals compress the equity cross-section into a deliberately
    # small set of breadth, dispersion, and synchronization signals.  At least
    # two observed ETFs are required on a row so a single surviving series is
    # never mislabeled as market breadth.
    equity_columns = tuple(
        column
        for column in price_columns
        if _price_stem(column) not in _NON_EQUITY_PRICE_STEMS
    )
    if len(equity_columns) >= 2:
        equity_log_prices = pd.DataFrame(
            {column: log_prices[column] for column in equity_columns},
            index=frame.index,
        )
        for lookback in cfg.breadth_return_lookbacks:
            equity_returns = equity_log_prices.diff(lookback)
            positive = equity_returns.gt(0.0).where(equity_returns.notna())
            features[
                f"market_internal__positive_return_share_{lookback}w"
            ] = _row_mean_with_minimum(positive.astype(float), 2)

        for window in cfg.breadth_trend_windows:
            trailing_mean = equity_log_prices.rolling(
                window, min_periods=_rolling_min_periods(window)
            ).mean()
            observed = equity_log_prices.notna() & trailing_mean.notna()
            above_trend = equity_log_prices.gt(trailing_mean).where(observed)
            features[
                f"market_internal__above_trailing_mean_share_{window}w"
            ] = _row_mean_with_minimum(above_trend.astype(float), 2)

        for lookback in cfg.dispersion_lookbacks:
            equity_returns = equity_log_prices.diff(lookback)
            dispersion = equity_returns.std(axis=1, skipna=True, ddof=0)
            features[
                f"market_internal__log_return_dispersion_{lookback}w"
            ] = dispersion.where(equity_returns.notna().sum(axis=1) >= 2)

        equity_one_week_returns = equity_log_prices.diff(1)
        directions = np.sign(equity_one_week_returns).where(
            equity_one_week_returns.notna()
        )
        synchronization = _row_mean_with_minimum(directions, 2).abs()
        features["market_internal__directional_synchronization_1w"] = synchronization

        for window in cfg.correlation_windows:
            features[
                f"market_internal__average_pairwise_correlation_{window}w"
            ] = _average_pairwise_correlation(equity_one_week_returns, window)

    price_name_map = {str(column).lower(): column for column in price_columns}
    for label, numerator_name, denominator_name in _MARKET_SPREAD_PAIRS:
        numerator = price_name_map.get(numerator_name)
        denominator = price_name_map.get(denominator_name)
        if numerator is None or denominator is None:
            continue
        relative_log = log_prices[numerator] - log_prices[denominator]
        for lookback in cfg.spread_lookbacks:
            features[
                f"market_spread__{label}__relative_return_{lookback}w"
            ] = relative_log.diff(lookback)

    log_volume_changes: dict[str, pd.Series] = {}
    for column in volume_columns:
        volume = pd.to_numeric(frame[column], errors="coerce").astype(float)
        log_volume = np.log1p(volume.where(volume >= 0.0))
        for window in cfg.generic_z_windows:
            min_periods = _rolling_min_periods(window)
            mean = log_volume.rolling(window, min_periods=min_periods).mean()
            std = log_volume.rolling(window, min_periods=min_periods).std(ddof=0)
            features[f"{column}__log_z_{window}w"] = (
                (log_volume - mean) / std.replace(0.0, np.nan)
            )
        log_volume_changes[column] = log_volume.diff(1)
        features[f"{column}__log_change_1w"] = log_volume_changes[column]
        features[f"{column}__missing"] = volume.isna().astype(float)

    if len(volume_columns) >= 2:
        volume_changes = pd.DataFrame(log_volume_changes, index=frame.index)
        rising = volume_changes.gt(0.0).where(volume_changes.notna())
        features["volume_internal__rising_volume_share_1w"] = (
            _row_mean_with_minimum(rising.astype(float), 2)
        )

        matched_pressure: dict[str, pd.Series] = {}
        for volume_column, volume_change in log_volume_changes.items():
            stem = str(volume_column).lower()
            if not stem.endswith("_volume"):
                continue
            price_column = price_name_map.get(f"{stem[:-len('_volume')]}_close")
            if price_column is None:
                continue
            price_return = one_week_returns[price_column]
            valid = price_return.notna() & volume_change.notna()
            # Positive values mean rising volume accompanied advancing prices;
            # negative values mean rising volume accompanied falling prices.
            pressure = np.sign(price_return).where(volume_change > 0.0, 0.0)
            matched_pressure[volume_column] = pressure.where(valid)
        if len(matched_pressure) >= 2:
            features["volume_internal__net_price_volume_confirmation_1w"] = (
                _row_mean_with_minimum(
                    pd.DataFrame(matched_pressure, index=frame.index), 2
                )
            )

    numeric_columns = [
        str(column)
        for column in frame.select_dtypes(include=[np.number, "bool"]).columns
        if str(column) not in price_set and str(column) not in volume_set
    ]
    for column in numeric_columns:
        series = pd.to_numeric(frame[column], errors="coerce").astype(float)
        if cfg.include_generic_levels:
            features[f"{column}__level"] = series
        for lookback in cfg.generic_change_lookbacks:
            features[f"{column}__change_{lookback}w"] = series.diff(lookback)
        for window in cfg.generic_z_windows:
            min_periods = _rolling_min_periods(window)
            mean = series.rolling(window, min_periods=min_periods).mean()
            std = series.rolling(window, min_periods=min_periods).std(ddof=0)
            features[f"{column}__z_{window}w"] = (
                (series - mean) / std.replace(0.0, np.nan)
            )
        for change_lookback, z_window in cfg.generic_change_z_pairs:
            change = series.diff(change_lookback)
            min_periods = _rolling_min_periods(z_window)
            mean = change.rolling(z_window, min_periods=min_periods).mean()
            std = change.rolling(z_window, min_periods=min_periods).std(ddof=0)
            features[
                f"{column}__change_{change_lookback}w_z_{z_window}w"
            ] = (change - mean) / std.replace(0.0, np.nan)
        features[f"{column}__missing"] = series.isna().astype(float)

    # Explicit groups prevent the sector, broad-market, and cross-asset ETF
    # universes from being blended into a single breadth statistic.  This is
    # additive: callers without ``price_groups`` retain the exact v3 output.
    if cfg.price_groups:
        grouped = build_market_group_features(
            frame,
            cfg.price_groups,
            return_lookbacks=cfg.breadth_return_lookbacks,
        )
        features.update({str(column): grouped[column] for column in grouped})

    result = pd.DataFrame(features, index=frame.index, dtype=float)
    # Keep NaNs so every train fold owns its imputation statistics.
    return result.replace([np.inf, -np.inf], np.nan)


__all__ = [
    "FeatureConfig",
    "build_cross_asset_correlation_features",
    "build_weekly_features",
]
