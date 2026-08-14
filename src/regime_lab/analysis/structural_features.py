"""Causal structural feature blocks for the weekly regime dataset.

Every transformation in this module is deterministic and uses only the
current row and rows to its left.  Missing observations remain missing so
fold-local model preprocessing, rather than feature engineering, owns any
imputation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


_ANFCI_FEATURES = frozenset(
    {
        "anfci__level",
        "anfci__change_1w",
        "anfci__change_4w",
        "anfci__z_52w",
    }
)

_STRUCTURAL_GROUPS = (
    (
        "sector_breadth",
        "GICS sector breadth, downside, dispersion, and leadership",
        ("market_group__gics_sector__",),
    ),
    (
        "broad_size_style_breadth",
        "Broad-market, size, and style breadth",
        ("market_group__broad_size_style__",),
    ),
    (
        "cross_asset_breadth",
        "Treasury, credit, commodity, and currency ETF breadth",
        ("market_group__cross_asset__",),
    ),
    (
        "treasury_curve",
        "Nelson-Siegel Treasury curve factors",
        ("treasury_curve__",),
    ),
    (
        "bank_credit",
        "Bank credit growth and funding composition",
        ("bank_credit__",),
    ),
    (
        "financial_conditions",
        "Adjusted National Financial Conditions Index",
        ("anfci__",),
    ),
    (
        "release_innovation",
        "Point-in-time macro release and revision innovations",
        ("release_innovation__",),
    ),
)


def _validate_index(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("frame must use a DatetimeIndex")
    if frame.index.has_duplicates:
        raise ValueError("frame index must be unique")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("frame index must be sorted in increasing time order")


def _resolve_column(frame: pd.DataFrame, configured: str, *, price: bool = False) -> str | None:
    raw = str(configured).strip()
    candidates = [raw, raw.lower()]
    if price and not raw.lower().endswith(("_close", "_price")):
        candidates.append(f"{raw.lower()}_close")
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _positive_log(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    return np.log(numeric.where(numeric > 0.0))


def _row_stat_with_minimum(
    values: pd.DataFrame,
    statistic: str,
    minimum: int,
) -> pd.Series:
    counts = values.notna().sum(axis=1)
    if statistic == "mean":
        output = values.mean(axis=1, skipna=True)
    elif statistic == "median":
        output = values.median(axis=1, skipna=True)
    else:  # pragma: no cover - internal callers use a fixed vocabulary.
        raise ValueError(f"unsupported row statistic: {statistic}")
    return output.where(counts >= minimum)


def _median_absolute_deviation(values: pd.DataFrame, minimum: int) -> pd.Series:
    counts = values.notna().sum(axis=1)
    medians = values.median(axis=1, skipna=True)
    deviations = values.sub(medians, axis=0).abs()
    return deviations.median(axis=1, skipna=True).where(counts >= minimum)


def _positive_leadership_hhi(values: pd.DataFrame, minimum: int) -> pd.Series:
    """Return HHI of positive-return magnitudes within a cross-section."""

    observed = values.notna().sum(axis=1)
    positive = values.clip(lower=0.0).where(values.notna())
    total = positive.sum(axis=1, min_count=1)
    weights = positive.div(total.replace(0.0, np.nan), axis=0)
    hhi = weights.pow(2).sum(axis=1, min_count=1)
    return hhi.where((observed >= minimum) & (total > 0.0))


def build_market_group_features(
    frame: pd.DataFrame,
    price_groups: Mapping[str, Sequence[str]],
    *,
    return_lookbacks: Sequence[int] = (1, 4),
    enhanced_group: str = "gics_sector",
    minimum_observed: int = 2,
) -> pd.DataFrame:
    """Build separately identified breadth blocks for configured ETF groups.

    Group coverage divides the number of usable returns by the full configured
    membership, so a temporarily missing ETF cannot silently redefine the
    cross-section.  The sector group additionally exposes downside breadth,
    robust median/MAD dispersion, positive-leader concentration, and changes
    in one-week breadth.
    """

    _validate_index(frame)
    if minimum_observed < 2:
        raise ValueError("minimum_observed must be at least 2")
    lookbacks = tuple(dict.fromkeys(int(item) for item in return_lookbacks))
    if not lookbacks or any(item <= 0 for item in lookbacks):
        raise ValueError("return_lookbacks must contain positive integers")

    normalized_groups: dict[str, tuple[str, ...]] = {}
    configured_members: dict[str, tuple[str, ...]] = {}
    seen_members: dict[str, str] = {}
    for raw_name, raw_members in price_groups.items():
        name = str(raw_name).strip().lower()
        if not name or "__" in name:
            raise ValueError(f"invalid price group name: {raw_name!r}")
        members = tuple(str(item).strip() for item in raw_members if str(item).strip())
        normalized_member_names = tuple(
            item.lower()
            if item.lower().endswith(("_close", "_price"))
            else f"{item.lower()}_close"
            for item in members
        )
        if len(set(normalized_member_names)) != len(normalized_member_names):
            raise ValueError(f"duplicate members in price group {name!r}")
        for member in normalized_member_names:
            owner = seen_members.get(member)
            if owner is not None and owner != name:
                raise ValueError(
                    f"price group member {member!r} appears in both {owner!r} and {name!r}"
                )
            seen_members[member] = name
        resolved = tuple(
            column
            for item in members
            if (column := _resolve_column(frame, item, price=True)) is not None
        )
        normalized_groups[name] = resolved
        configured_members[name] = normalized_member_names

    output: dict[str, pd.Series] = {}
    for group, columns in normalized_groups.items():
        configured_count = len(configured_members[group])
        if configured_count == 0:
            continue
        log_prices = pd.DataFrame(
            {column: _positive_log(frame[column]) for column in columns},
            index=frame.index,
        )
        for lookback in lookbacks:
            returns = log_prices.diff(lookback)
            observed = returns.notna().sum(axis=1)
            positive = returns.gt(0.0).where(returns.notna()).astype(float)
            prefix = f"market_group__{group}"
            positive_share = _row_stat_with_minimum(
                positive, "mean", minimum_observed
            )
            output[f"{prefix}__positive_return_share_{lookback}w"] = positive_share
            output[f"{prefix}__coverage_{lookback}w"] = (
                observed.astype(float) / float(configured_count)
            ).clip(0.0, 1.0)

            if group == enhanced_group:
                downside = returns.lt(0.0).where(returns.notna()).astype(float)
                output[f"{prefix}__downside_share_{lookback}w"] = (
                    _row_stat_with_minimum(downside, "mean", minimum_observed)
                )
                output[f"{prefix}__median_log_return_{lookback}w"] = (
                    _row_stat_with_minimum(returns, "median", minimum_observed)
                )
                output[f"{prefix}__mad_dispersion_{lookback}w"] = (
                    _median_absolute_deviation(returns, minimum_observed)
                )
                output[f"{prefix}__leadership_concentration_{lookback}w"] = (
                    _positive_leadership_hhi(returns, minimum_observed)
                )

        one_week_name = f"market_group__{group}__positive_return_share_1w"
        if group == enhanced_group and one_week_name in output:
            output[f"market_group__{group}__breadth_acceleration_1w"] = output[
                one_week_name
            ].diff(1)
            output[f"market_group__{group}__breadth_acceleration_4w"] = output[
                one_week_name
            ].diff(4)

    result = pd.DataFrame(output, index=frame.index, dtype=float)
    return result.replace([np.inf, -np.inf], np.nan)


def build_nelson_siegel_features(
    frame: pd.DataFrame,
    series_months: Mapping[str, float],
    *,
    lambda_per_month: float = 0.0609,
    minimum_maturities: int = 4,
) -> pd.DataFrame:
    """Fit fixed-loading Nelson-Siegel factors independently at each cutoff."""

    _validate_index(frame)
    if not np.isfinite(lambda_per_month) or lambda_per_month <= 0.0:
        raise ValueError("lambda_per_month must be positive")
    if minimum_maturities < 3:
        raise ValueError("minimum_maturities must be at least 3")

    maturities: list[float] = []
    values: list[pd.Series] = []
    for series_id, months in series_months.items():
        maturity = float(months)
        if not np.isfinite(maturity) or maturity <= 0.0:
            raise ValueError(f"invalid maturity for {series_id!r}: {months!r}")
        column = _resolve_column(frame, str(series_id))
        if column is None:
            continue
        maturities.append(maturity)
        values.append(pd.to_numeric(frame[column], errors="coerce").astype(float))
    if len(set(maturities)) != len(maturities):
        raise ValueError("Nelson-Siegel maturities must be unique")

    columns = (
        "treasury_curve__nelson_siegel_level",
        "treasury_curve__nelson_siegel_slope",
        "treasury_curve__nelson_siegel_curvature",
        "treasury_curve__coverage",
    )
    result = pd.DataFrame(np.nan, index=frame.index, columns=columns, dtype=float)
    configured_count = len(series_months)
    if not values or configured_count == 0:
        return result

    observations = pd.concat(values, axis=1)
    tau = np.asarray(maturities, dtype=float)
    scaled = lambda_per_month * tau
    slope_loading = -np.expm1(-scaled) / scaled
    curvature_loading = slope_loading - np.exp(-scaled)
    design = np.column_stack(
        [np.ones_like(tau), slope_loading, curvature_loading]
    )

    finite_counts = observations.notna().sum(axis=1)
    result["treasury_curve__coverage"] = (
        finite_counts.astype(float) / float(configured_count)
    ).clip(0.0, 1.0)
    for position, (_, row) in enumerate(observations.iterrows()):
        y = row.to_numpy(dtype=float)
        valid = np.isfinite(y)
        if int(valid.sum()) < minimum_maturities:
            continue
        coefficients, _, rank, _ = np.linalg.lstsq(design[valid], y[valid], rcond=None)
        if rank < 3:
            continue
        result.iloc[position, :3] = coefficients
    return result.replace([np.inf, -np.inf], np.nan)


def build_bank_credit_features(
    frame: pd.DataFrame,
    *,
    total_credit: str = "TOTBKCR",
    commercial_industrial: str = "TOTCI",
    deposits: str = "DPSACBW027SBOG",
    borrowings_millions: str = "H8B3094NCBA",
) -> pd.DataFrame:
    """Build scale-aware bank-credit growth and funding-ratio features."""

    _validate_index(frame)
    configured = {
        "total": total_credit,
        "ci": commercial_industrial,
        "deposits": deposits,
        "borrowings": borrowings_millions,
    }
    resolved = {
        name: _resolve_column(frame, series_id)
        for name, series_id in configured.items()
    }
    if all(column is None for column in resolved.values()):
        return pd.DataFrame(index=frame.index, dtype=float)

    data = {
        name: (
            pd.to_numeric(frame[column], errors="coerce").astype(float)
            if column is not None
            else pd.Series(np.nan, index=frame.index, dtype=float)
        )
        for name, column in resolved.items()
    }
    total_log = np.log(data["total"].where(data["total"] > 0.0))
    ci_log = np.log(data["ci"].where(data["ci"] > 0.0))
    total_denominator = data["total"].where(data["total"] > 0.0)
    borrowings_billions = data["borrowings"] / 1000.0

    output: dict[str, pd.Series] = {}
    for lookback in (4, 13):
        output[f"bank_credit__log_growth_{lookback}w"] = total_log.diff(lookback)
        output[f"bank_credit__ci_log_growth_{lookback}w"] = ci_log.diff(lookback)
    output["bank_credit__ci_share"] = data["ci"] / total_denominator
    output["bank_credit__deposit_funding_ratio"] = (
        data["deposits"] / total_denominator
    )
    output["bank_credit__borrowing_ratio"] = borrowings_billions / total_denominator
    output["bank_credit__coverage"] = pd.concat(data, axis=1).notna().mean(axis=1)
    result = pd.DataFrame(output, index=frame.index, dtype=float)
    return result.replace([np.inf, -np.inf], np.nan)


def build_release_innovation_features(
    values: pd.DataFrame,
    observed_periods: pd.DataFrame,
    *,
    series: Sequence[str],
    revision_sequences: pd.DataFrame | None = None,
    prior_release_window: int = 12,
    minimum_prior_releases: int = 4,
) -> pd.DataFrame:
    """Encode low-frequency releases and selected-period revisions causally.

    A period change is a new selected observation period.  A revision event is
    a value change while the selected period stays unchanged.  At each event,
    expected delta and MAD scale are computed from *prior* event deltas only;
    the current delta is added to history afterwards.  Non-event signal values
    are exactly zero, while coverage records whether enough prior events exist.
    ``revision_sequences`` is accepted for lineage alignment but value changes,
    not provider sequence numbers alone, define a revision signal.
    """

    _validate_index(values)
    _validate_index(observed_periods)
    if not values.index.equals(observed_periods.index):
        raise ValueError("values and observed_periods must use the same index")
    if revision_sequences is not None:
        _validate_index(revision_sequences)
        if not values.index.equals(revision_sequences.index):
            raise ValueError("revision_sequences must use the same index")
    if prior_release_window <= 0:
        raise ValueError("prior_release_window must be positive")
    if minimum_prior_releases < 1:
        raise ValueError("minimum_prior_releases must be positive")

    output: dict[str, pd.Series] = {}
    for configured in series:
        series_id = str(configured).strip().lower()
        value_column = _resolve_column(values, configured)
        period_column = _resolve_column(observed_periods, configured)
        if value_column is None or period_column is None:
            continue
        numeric = pd.to_numeric(values[value_column], errors="coerce").astype(float)
        periods = observed_periods[period_column]

        previous_numeric = numeric.shift(1)
        previous_periods = periods.shift(1)
        comparable = numeric.notna() & previous_numeric.notna()
        has_period_pair = periods.notna() & previous_periods.notna()
        period_change = has_period_pair & periods.ne(previous_periods)
        value_change = comparable & numeric.ne(previous_numeric)
        revision_event = has_period_pair & periods.eq(previous_periods) & value_change
        event = period_change | revision_event
        delta = (numeric - previous_numeric).where(event)

        event_values = event.to_numpy(dtype=bool)
        deltas = delta.to_numpy(dtype=float)
        expected = np.zeros(len(values), dtype=float)
        mad_scale = np.zeros(len(values), dtype=float)
        standardized = np.zeros(len(values), dtype=float)
        coverage = np.zeros(len(values), dtype=float)
        prior_count = np.zeros(len(values), dtype=float)
        history: list[float] = []
        for position in range(len(values)):
            available_history = history[-prior_release_window:]
            count = len(available_history)
            prior_count[position] = float(count)
            coverage[position] = min(
                float(count) / float(minimum_prior_releases), 1.0
            )
            if not event_values[position]:
                continue
            if count < minimum_prior_releases:
                expected[position] = np.nan
                mad_scale[position] = np.nan
                standardized[position] = np.nan
            else:
                prior = np.asarray(available_history, dtype=float)
                median = float(np.median(prior))
                scale = float(1.4826 * np.median(np.abs(prior - median)))
                expected[position] = median
                mad_scale[position] = scale
                if np.isfinite(deltas[position]) and scale > 0.0:
                    standardized[position] = (deltas[position] - median) / scale
                else:
                    standardized[position] = np.nan
            if np.isfinite(deltas[position]):
                history.append(float(deltas[position]))

        prefix = f"release_innovation__{series_id}"
        output[f"{prefix}__event"] = event.astype(float)
        output[f"{prefix}__period_change"] = period_change.astype(float)
        output[f"{prefix}__revision_event"] = revision_event.astype(float)
        output[f"{prefix}__delta"] = delta.where(event, 0.0)
        output[f"{prefix}__expected_delta"] = pd.Series(
            expected, index=values.index, dtype=float
        )
        output[f"{prefix}__mad_scale"] = pd.Series(
            mad_scale, index=values.index, dtype=float
        )
        output[f"{prefix}__standardized"] = pd.Series(
            standardized, index=values.index, dtype=float
        )
        output[f"{prefix}__coverage"] = pd.Series(
            coverage, index=values.index, dtype=float
        )
        output[f"{prefix}__prior_event_count"] = pd.Series(
            prior_count, index=values.index, dtype=float
        )

    result = pd.DataFrame(output, index=values.index, dtype=float)
    return result.replace([np.inf, -np.inf], np.nan)


def build_structural_feature_manifest(
    columns: Sequence[object],
) -> tuple[dict[str, Any], ...]:
    """Map every model feature to exactly one stable feature-family record."""

    ordered = tuple(dict.fromkeys(str(column) for column in columns))
    assigned: set[str] = set()
    manifest: list[dict[str, Any]] = []
    for group_id, description, prefixes in _STRUCTURAL_GROUPS:
        matches = tuple(
            column
            for column in ordered
            if column not in assigned
            and column.startswith(prefixes)
            and (group_id != "financial_conditions" or column in _ANFCI_FEATURES)
        )
        if not matches:
            continue
        assigned.update(matches)
        manifest.append(
            {
                "id": group_id,
                "description": description,
                "feature_count": len(matches),
                "features": matches,
            }
        )
    remaining = tuple(column for column in ordered if column not in assigned)
    if remaining:
        manifest.append(
            {
                "id": "legacy_v3",
                "description": "Protected v3 market, macro, OHLC, and availability features",
                "feature_count": len(remaining),
                "features": remaining,
            }
        )
    return tuple(manifest)


__all__ = [
    "build_bank_credit_features",
    "build_market_group_features",
    "build_nelson_siegel_features",
    "build_release_innovation_features",
    "build_structural_feature_manifest",
]
