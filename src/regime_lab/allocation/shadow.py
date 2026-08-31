"""Selection-frozen allocation and sector-rotation shadow.

The module is downstream of the operating forecast.  All fitted payoff,
confidence, and momentum choices end before the model selection boundary;
the post-selection path only applies those frozen choices.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, timedelta
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from regime_lab.analysis.decision_shadow import (
    STATE_ORDER,
    _scheduled_nyse_entry_at,
    _selected_model_forecast,
)
from regime_lab.integrity import canonical_json_sha256_v1


ALLOCATION_SPEC_SCHEMA_VERSION = "regime-allocation-shadow-spec/1"
ALLOCATION_RESULT_SCHEMA_VERSION = "regime-allocation-shadow-candidate/1"
PORTFOLIO_INTENT_SCHEMA_VERSION = "regime-portfolio-intent/1"


def default_allocation_shadow_spec_path() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "allocation-shadow-v1.json"


def load_allocation_shadow_spec(path: str | Path | None = None) -> dict[str, Any]:
    selected = default_allocation_shadow_spec_path() if path is None else Path(path)
    document = json.loads(selected.read_text(encoding="utf-8"))
    if document.get("schema_version") != ALLOCATION_SPEC_SCHEMA_VERSION:
        raise ValueError("allocation-shadow spec schema is invalid")
    if document.get("role") != "research_only_no_forecast_or_champion_effect":
        raise ValueError("allocation-shadow role is invalid")
    assets = document.get("assets")
    if not isinstance(assets, Mapping):
        raise ValueError("allocation-shadow assets are invalid")
    if assets.get("anchor") != ["SPY", "TLT"]:
        raise ValueError("allocation-shadow anchor assets are invalid")
    sectors = assets.get("sectors")
    if (
        not isinstance(sectors, list)
        or len(sectors) != 11
        or len(set(sectors)) != len(sectors)
    ):
        raise ValueError("allocation-shadow sector universe is invalid")
    anchor = document.get("anchor_weights")
    if not isinstance(anchor, Mapping) or set(anchor) != {"SPY", "TLT"}:
        raise ValueError("allocation-shadow anchor weights are invalid")
    if not math.isclose(sum(float(anchor[a]) for a in anchor), 1.0, abs_tol=1e-12):
        raise ValueError("allocation-shadow anchor weights must sum to one")
    execution = document.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("allocation-shadow execution is invalid")
    for field in ("no_trade_one_way_band", "partial_adjustment", "one_way_trade_cap"):
        value = float(execution.get(field, float("nan")))
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"allocation-shadow {field} is invalid")
    if execution.get("late_signal_policy") != "no_trade":
        raise ValueError("allocation-shadow late signal policy is invalid")
    sector = document.get("sector_rotation")
    if not isinstance(sector, Mapping):
        raise ValueError("allocation-shadow sector policy is invalid")
    if set(sector.get("inception_dates", {})) != set(sectors):
        raise ValueError("allocation-shadow sector inception map is invalid")
    if int(sector.get("top_n", 0)) < 1:
        raise ValueError("allocation-shadow sector top_n is invalid")
    return document


def split_safe_asset_return_frames(
    prices: pd.DataFrame,
    assets: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return split-safe prior-close/open and open/close price relatives."""

    normalized_assets = tuple(str(asset).upper() for asset in assets)
    if not normalized_assets or len(set(normalized_assets)) != len(normalized_assets):
        raise ValueError("allocation-shadow assets must be unique and non-empty")
    split_candidates = np.asarray(
        sorted(
            {
                numerator / denominator
                for numerator in range(1, 21)
                for denominator in range(1, 21)
            }
        ),
        dtype=float,
    )
    gaps: dict[str, pd.Series] = {}
    intraday: dict[str, pd.Series] = {}
    for asset in normalized_assets:
        prefix = asset.lower()
        columns = {
            "adjusted_close": f"{prefix}_close",
            "raw_open": f"{prefix}_raw_open",
            "raw_close": f"{prefix}_raw_close",
            "dividend": f"{prefix}_dividend_amount",
        }
        missing = [column for column in columns.values() if column not in prices]
        if missing:
            raise ValueError(
                f"allocation-shadow {asset} prices are missing columns: {missing}"
            )
        adjusted = pd.to_numeric(
            prices[columns["adjusted_close"]], errors="coerce"
        ).astype(float)
        raw_open = pd.to_numeric(prices[columns["raw_open"]], errors="coerce").astype(
            float
        )
        raw_close = pd.to_numeric(
            prices[columns["raw_close"]], errors="coerce"
        ).astype(float)
        dividend = pd.to_numeric(
            prices[columns["dividend"]], errors="coerce"
        ).astype(float)
        if bool((dividend[np.isfinite(dividend)] < 0.0).any()):
            raise ValueError("allocation-shadow dividends must be non-negative")
        inferred_split = (
            adjusted.div(adjusted.shift(1)) * raw_close.shift(1) / (raw_close + dividend)
        )
        finite = inferred_split[np.isfinite(inferred_split)]
        if bool((finite <= 0.0).any()):
            raise ValueError("allocation-shadow inferred split must be positive")
        if not finite.empty:
            values = finite.to_numpy(dtype=float)
            errors = np.min(
                np.abs(values[:, None] / split_candidates[None, :] - 1.0), axis=1
            )
            if bool((errors > 0.02).any()):
                raise ValueError(
                    "allocation-shadow inferred split is outside sanity tolerance"
                )
        gaps[asset] = inferred_split * raw_open / raw_close.shift(1)
        intraday[asset] = raw_close / raw_open
    return (
        pd.DataFrame(gaps, index=prices.index, dtype=float),
        pd.DataFrame(intraday, index=prices.index, dtype=float),
    )


def _normalized_probabilities(forecast: Mapping[str, Any]) -> dict[str, float]:
    raw = forecast.get("probabilities")
    if not isinstance(raw, Mapping) or set(raw) != set(STATE_ORDER):
        raise ValueError("allocation-shadow forecast probabilities are invalid")
    values = np.asarray([float(raw[state]) for state in STATE_ORDER], dtype=float)
    if (
        not np.isfinite(values).all()
        or (values < 0.0).any()
        or not math.isclose(float(values.sum()), 1.0, abs_tol=1e-6)
    ):
        raise ValueError("allocation-shadow probabilities must sum to one")
    values /= float(values.sum())
    return {state: float(values[index]) for index, state in enumerate(STATE_ORDER)}


def _date_lookup(index: pd.DatetimeIndex) -> dict[str, pd.Timestamp]:
    result: dict[str, pd.Timestamp] = {}
    for raw in index:
        timestamp = pd.Timestamp(raw)
        key = timestamp.date().isoformat()
        if key in result:
            raise ValueError("allocation-shadow price dates must be unique")
        result[key] = timestamp
    return result


def _selection_origins(
    index: pd.DatetimeIndex,
    *,
    selection_end: date,
    lookback_weeks: int,
) -> pd.DatetimeIndex:
    eligible = [
        index[position]
        for position in range(len(index) - 1)
        if index[position + 1].date() <= selection_end
        and index[position + 1].date() - index[position].date() == timedelta(days=7)
    ]
    return pd.DatetimeIndex(eligible[-lookback_weeks:])


def _shrunk_state_payoffs(
    origin_states: pd.Series,
    future_relative_returns: pd.DataFrame,
    origins: pd.DatetimeIndex,
    *,
    pseudo_weeks: float,
    prior: float,
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    result: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for asset in future_relative_returns.columns:
        by_state: dict[str, dict[str, float | int | None]] = {}
        values = future_relative_returns[asset].reindex(origins)
        for state in STATE_ORDER:
            sample = values.loc[origin_states.reindex(origins).eq(state)].dropna()
            n = int(len(sample))
            mean = float(sample.mean()) if n else prior
            weight = n / (n + pseudo_weeks) if n else 0.0
            estimate = prior + weight * (mean - prior)
            standard_error = (
                float(sample.std(ddof=1) / math.sqrt(n)) if n > 1 else None
            )
            half_width = (
                None if standard_error is None else 1.96 * weight * standard_error
            )
            by_state[state] = {
                "observations": n,
                "raw_mean": mean if n else None,
                "shrinkage_weight": float(weight),
                "estimate": float(estimate),
                "ci_lower": None if half_width is None else float(estimate - half_width),
                "ci_upper": None if half_width is None else float(estimate + half_width),
            }
        result[str(asset)] = by_state
    return result


def _expected_payoff(
    estimates: Mapping[str, Mapping[str, Mapping[str, Any]]],
    asset: str,
    probabilities: Mapping[str, float],
    *,
    field: str = "estimate",
) -> float:
    return float(
        sum(
            float(probabilities[state]) * float(estimates[asset][state][field] or 0.0)
            for state in STATE_ORDER
        )
    )


def _expected_interval(
    estimates: Mapping[str, Mapping[str, Mapping[str, Any]]],
    asset: str,
    probabilities: Mapping[str, float],
) -> tuple[float | None, float | None]:
    lower = [estimates[asset][state]["ci_lower"] for state in STATE_ORDER]
    upper = [estimates[asset][state]["ci_upper"] for state in STATE_ORDER]
    if any(value is None for value in (*lower, *upper)):
        return None, None
    return (
        float(sum(probabilities[state] * float(lower[i]) for i, state in enumerate(STATE_ORDER))),
        float(sum(probabilities[state] * float(upper[i]) for i, state in enumerate(STATE_ORDER))),
    )


def _cross_section_z(values: Mapping[str, float]) -> dict[str, float]:
    if not values:
        return {}
    raw = np.asarray(list(values.values()), dtype=float)
    scale = float(raw.std(ddof=0))
    if not np.isfinite(scale) or scale <= 1e-12:
        return {key: 0.0 for key in values}
    mean = float(raw.mean())
    return {key: float((value - mean) / scale) for key, value in values.items()}


def _momentum_at(
    prices: pd.DataFrame,
    origin: pd.Timestamp,
    sectors: Sequence[str],
    *,
    start_lag: int,
    end_lag: int,
    inceptions: Mapping[str, str],
    seasoning_weeks: int,
) -> dict[str, float]:
    position = int(prices.index.get_loc(origin))
    if position < start_lag:
        return {}
    spy_start = float(prices["spy_close"].iloc[position - start_lag])
    spy_end = float(prices["spy_close"].iloc[position - end_lag])
    if not np.isfinite([spy_start, spy_end]).all() or min(spy_start, spy_end) <= 0.0:
        return {}
    spy_return = spy_end / spy_start - 1.0
    result: dict[str, float] = {}
    for asset in sectors:
        seasoned_at = date.fromisoformat(str(inceptions[asset])) + timedelta(
            weeks=seasoning_weeks
        )
        if origin.date() < seasoned_at:
            continue
        column = f"{asset.lower()}_close"
        start = float(prices[column].iloc[position - start_lag])
        end = float(prices[column].iloc[position - end_lag])
        if np.isfinite([start, end]).all() and min(start, end) > 0.0:
            result[asset] = float(end / start - 1.0 - spy_return)
    return result


def _composite_momentum_at(
    prices: pd.DataFrame,
    origin: pd.Timestamp,
    sectors: Sequence[str],
    sector_spec: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    components: dict[str, dict[str, float]] = {}
    for candidate in sector_spec["momentum_candidates"]:
        raw = _momentum_at(
            prices,
            origin,
            sectors,
            start_lag=int(candidate["start_lag_weeks"]),
            end_lag=int(candidate["end_lag_weeks"]),
            inceptions=sector_spec["inception_dates"],
            seasoning_weeks=int(sector_spec["seasoning_weeks"]),
        )
        components[str(candidate["id"])] = _cross_section_z(raw)
    common = set(sectors)
    for values in components.values():
        common &= set(values)
    weights = sector_spec["momentum_combination"]
    combined = {
        asset: float(
            sum(float(weights[name]) * components[name][asset] for name in weights)
        )
        for asset in common
    }
    return combined, components


def _month_changed(previous: pd.Timestamp | None, current: pd.Timestamp) -> bool:
    return previous is None or (previous.year, previous.month) != (current.year, current.month)


def _turnover(
    previous_assets: np.ndarray,
    previous_cash: float,
    target_assets: np.ndarray,
    target_cash: float,
) -> tuple[float, float]:
    full_l1_assets = float(np.abs(target_assets - previous_assets).sum())
    one_way = 0.5 * (
        full_l1_assets + abs(float(target_cash) - float(previous_cash))
    )
    return one_way, full_l1_assets


@dataclass(frozen=True)
class _TradeDecision:
    assets: np.ndarray
    cash: float
    one_way: float
    full_l1: float
    action: str
    expected_benefit: float | None


def _trade_decision(
    pretrade_assets: np.ndarray,
    pretrade_cash: float,
    aim_assets: np.ndarray,
    *,
    execution: Mapping[str, Any],
    cost_rate: float,
    expected_returns: np.ndarray | None,
    initial: bool,
    force_no_trade: bool = False,
) -> _TradeDecision:
    aim_cash = max(0.0, 1.0 - float(aim_assets.sum()))
    if force_no_trade:
        return _TradeDecision(
            pretrade_assets.copy(), float(pretrade_cash), 0.0, 0.0, "no_trade", None
        )
    desired_one_way, _ = _turnover(
        pretrade_assets, pretrade_cash, aim_assets, aim_cash
    )
    if not initial and desired_one_way < float(execution["no_trade_one_way_band"]):
        return _TradeDecision(
            pretrade_assets.copy(), float(pretrade_cash), 0.0, 0.0, "band_hold", None
        )
    if initial:
        tentative = aim_assets.copy()
    else:
        alpha = float(execution["partial_adjustment"])
        tentative = pretrade_assets + alpha * (aim_assets - pretrade_assets)
    tentative_cash = max(0.0, 1.0 - float(tentative.sum()))
    one_way, _ = _turnover(
        pretrade_assets, pretrade_cash, tentative, tentative_cash
    )
    cap = float(execution["one_way_trade_cap"])
    if not (initial and bool(execution["initial_allocation_exempt_from_trade_cap"])):
        if one_way > cap and one_way > 0.0:
            scale = cap / one_way
            tentative = pretrade_assets + scale * (tentative - pretrade_assets)
            tentative_cash = max(0.0, 1.0 - float(tentative.sum()))
    one_way, full_l1 = _turnover(
        pretrade_assets, pretrade_cash, tentative, tentative_cash
    )
    expected_benefit: float | None = None
    if expected_returns is not None:
        expected_benefit = float(
            np.dot(tentative - pretrade_assets, expected_returns)
        )
        hurdle = (
            float(execution["minimum_expected_benefit_cost_multiple"])
            * cost_rate
            * full_l1
        )
        if not initial and expected_benefit <= hurdle:
            return _TradeDecision(
                pretrade_assets.copy(),
                float(pretrade_cash),
                0.0,
                0.0,
                "economic_hold",
                expected_benefit,
            )
    action = "initial_allocate" if initial else "rebalance"
    return _TradeDecision(
        np.maximum(tentative, 0.0),
        float(tentative_cash),
        float(one_way),
        float(full_l1),
        action,
        expected_benefit,
    )


def _run_strategy(
    aims: pd.DataFrame,
    expected_returns: pd.DataFrame | None,
    gaps: pd.DataFrame,
    intraday: pd.DataFrame,
    cash_factors: pd.Series,
    index: pd.DatetimeIndex,
    *,
    assets: Sequence[str],
    execution: Mapping[str, Any],
    cost_bps: float,
) -> dict[str, Any]:
    asset_list = list(assets)
    cost_rate = float(cost_bps) / 10_000.0
    prior_assets = np.zeros(len(asset_list), dtype=float)
    prior_cash = 1.0
    rows: list[dict[str, Any]] = []
    gross_wealth = 1.0
    net_wealth = 1.0
    peak = 1.0
    for offset, timestamp in enumerate(index):
        gap_values = gaps.loc[timestamp, asset_list].to_numpy(dtype=float)
        intraday_values = intraday.loc[timestamp, asset_list].to_numpy(dtype=float)
        held = prior_assets > 1e-12
        if bool((~np.isfinite(intraday_values[held])).any()) or bool(
            (intraday_values[held] <= 0.0).any()
        ):
            raise ValueError("allocation-shadow held return is unavailable")
        if offset == 0:
            gap_factor = 1.0
            pretrade_assets = prior_assets.copy()
            pretrade_cash = prior_cash
        else:
            if bool((~np.isfinite(gap_values[held])).any()) or bool(
                (gap_values[held] <= 0.0).any()
            ):
                raise ValueError("allocation-shadow held gap is unavailable")
            safe_gaps = np.where(np.isfinite(gap_values), gap_values, 1.0)
            gap_factor = float(prior_cash + np.dot(prior_assets, safe_gaps))
            if not np.isfinite(gap_factor) or gap_factor <= 0.0:
                raise ValueError("allocation-shadow gap wealth is invalid")
            pretrade_assets = prior_assets * safe_gaps / gap_factor
            pretrade_cash = prior_cash / gap_factor
        aim = aims.loc[timestamp, asset_list].to_numpy(dtype=float)
        if not np.isfinite(aim).all():
            aim = pretrade_assets.copy()
        if (aim < -1e-12).any() or float(aim.sum()) > 1.0 + 1e-10:
            raise ValueError("allocation-shadow aim must be long-only with gross at most one")
        unavailable = ~np.isfinite(intraday_values) | (intraday_values <= 0.0)
        if bool((aim[unavailable] > 1e-12).any()):
            raise ValueError("allocation-shadow aim uses an unavailable asset")
        expected = (
            None
            if expected_returns is None
            else expected_returns.loc[timestamp, asset_list].to_numpy(dtype=float)
        )
        if expected is not None and not np.isfinite(expected).all():
            expected = None
        trade = _trade_decision(
            pretrade_assets,
            pretrade_cash,
            np.maximum(aim, 0.0),
            execution=execution,
            cost_rate=cost_rate,
            expected_returns=expected,
            initial=offset == 0,
        )
        transaction_cost = cost_rate * trade.full_l1
        if transaction_cost >= 1.0:
            raise ValueError("allocation-shadow transaction cost exhausts wealth")
        safe_intraday = np.where(np.isfinite(intraday_values), intraday_values, 1.0)
        cash_factor = float(cash_factors.loc[timestamp])
        if not np.isfinite(cash_factor) or cash_factor <= 0.0:
            raise ValueError("allocation-shadow cash factor is invalid")
        holding_factor = float(
            np.dot(trade.assets, safe_intraday) + trade.cash * cash_factor
        )
        gross_return = gap_factor * holding_factor - 1.0
        net_return = gap_factor * (1.0 - transaction_cost) * holding_factor - 1.0
        gross_wealth *= 1.0 + gross_return
        net_wealth *= 1.0 + net_return
        peak = max(peak, net_wealth)
        rows.append(
            {
                "gross_return": float(gross_return),
                "net_return": float(net_return),
                "gross_wealth": float(gross_wealth),
                "net_wealth": float(net_wealth),
                "drawdown": float(net_wealth / peak - 1.0),
                "one_way_turnover": trade.one_way,
                "full_l1_turnover": trade.full_l1,
                "transaction_cost_rate": float(transaction_cost),
                "action": trade.action,
                "expected_benefit": trade.expected_benefit,
                "pretrade_weights": pretrade_assets.copy(),
                "target_weights": trade.assets.copy(),
            }
        )
        prior_assets = trade.assets * safe_intraday / holding_factor
        prior_cash = trade.cash * cash_factor / holding_factor
    return {
        "rows": rows,
        "close_weights": prior_assets,
        "close_cash": float(prior_cash),
    }


def _metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    annualization: float,
    gamma: float,
) -> dict[str, Any]:
    if not rows:
        return {
            "weeks": 0,
            "gross_cumulative_return": None,
            "cumulative_return": None,
            "annualized_return": None,
            "annualized_volatility": None,
            "sharpe": None,
            "certainty_equivalent_return": None,
            "maximum_drawdown": None,
            "one_way_turnover_sum": 0.0,
            "full_l1_turnover_sum": 0.0,
            "annualized_one_way_turnover": None,
            "annualized_full_l1_turnover": None,
            "annualized_one_way_turnover_including_initial": None,
            "annualized_full_l1_turnover_including_initial": None,
            "initial_one_way_turnover": 0.0,
            "initial_full_l1_turnover": 0.0,
            "transaction_cost_rate_sum": 0.0,
        }
    net = np.asarray([float(row["net_return"]) for row in rows], dtype=float)
    gross_wealth = float(rows[-1]["gross_wealth"])
    net_wealth = float(rows[-1]["net_wealth"])
    years = len(rows) / annualization
    annual_return = net_wealth ** (1.0 / years) - 1.0
    annual_vol = float(net.std(ddof=1) * math.sqrt(annualization)) if len(net) > 1 else 0.0
    sharpe = (
        float(net.mean() / net.std(ddof=1) * math.sqrt(annualization))
        if len(net) > 1 and net.std(ddof=1) > 0.0
        else None
    )
    one_way = float(sum(float(row["one_way_turnover"]) for row in rows))
    full_l1 = float(sum(float(row["full_l1_turnover"]) for row in rows))
    initial_one_way = float(rows[0]["one_way_turnover"])
    initial_full_l1 = float(rows[0]["full_l1_turnover"])
    recurring_one_way = max(0.0, one_way - initial_one_way)
    recurring_full_l1 = max(0.0, full_l1 - initial_full_l1)
    return {
        "weeks": len(rows),
        "gross_cumulative_return": gross_wealth - 1.0,
        "cumulative_return": net_wealth - 1.0,
        "annualized_return": float(annual_return),
        "annualized_volatility": annual_vol,
        "sharpe": sharpe,
        "certainty_equivalent_return": float(annual_return - 0.5 * gamma * annual_vol**2),
        "maximum_drawdown": float(min(float(row["drawdown"]) for row in rows)),
        "one_way_turnover_sum": one_way,
        "full_l1_turnover_sum": full_l1,
        "annualized_one_way_turnover": float(recurring_one_way / years),
        "annualized_full_l1_turnover": float(recurring_full_l1 / years),
        "annualized_one_way_turnover_including_initial": float(one_way / years),
        "annualized_full_l1_turnover_including_initial": float(full_l1 / years),
        "initial_one_way_turnover": initial_one_way,
        "initial_full_l1_turnover": initial_full_l1,
        "transaction_cost_rate_sum": float(
            sum(float(row["transaction_cost_rate"]) for row in rows)
        ),
    }


def _selection_skill(
    evidence: Mapping[str, Any] | None,
    *,
    minimum_predictions: int,
) -> tuple[bool, dict[str, Any]]:
    if not isinstance(evidence, Mapping):
        return False, {"status": "unavailable", "reason": "selection_evidence_missing"}
    try:
        count = int(evidence["selection_n_predictions"])
        model_loss = float(evidence["selection_log_loss"])
        benchmark_loss = float(evidence["benchmark_selection_log_loss"])
    except (KeyError, TypeError, ValueError):
        return False, {"status": "unavailable", "reason": "selection_evidence_invalid"}
    passed = count >= minimum_predictions and model_loss < benchmark_loss
    return passed, {
        "status": "passed" if passed else "failed",
        "selection_n_predictions": count,
        "selection_log_loss": model_loss,
        "benchmark_selection_log_loss": benchmark_loss,
    }


def allocation_calibration_evidence(
    model: Mapping[str, Any],
    *,
    forecast_model: str,
) -> dict[str, Any]:
    """Extract the frozen selector-period skill gate from the public leaderboard."""

    rows = model.get("leaderboard")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("allocation-shadow model leaderboard is invalid")
    by_name = {
        str(row.get("name")): row for row in rows if isinstance(row, Mapping)
    }
    selected = by_name.get(forecast_model)
    benchmark = by_name.get("majority")
    if not isinstance(selected, Mapping) or not isinstance(benchmark, Mapping):
        raise ValueError("allocation-shadow calibration rows are missing")
    return {
        "selection_n_predictions": int(selected["selection_n_predictions"]),
        "selection_log_loss": float(selected["selection_log_loss"]),
        "benchmark_selection_log_loss": float(benchmark["selection_log_loss"]),
    }


def _momentum_selection(
    prices: pd.DataFrame,
    origins: pd.DatetimeIndex,
    sectors: Sequence[str],
    future_relative: pd.DataFrame,
    sector_spec: Mapping[str, Any],
) -> tuple[dict[str, Any], float]:
    outcomes: list[float] = []
    regress_x: list[float] = []
    regress_y: list[float] = []
    prior_month: pd.Timestamp | None = None
    for origin in origins:
        position = int(prices.index.get_loc(origin))
        target = pd.Timestamp(prices.index[position + 1])
        entry = _scheduled_nyse_entry_at(target.date().isoformat())
        if not _month_changed(prior_month, entry):
            continue
        prior_month = entry
        composite, _ = _composite_momentum_at(
            prices, origin, sectors, sector_spec
        )
        available = {
            asset: score
            for asset, score in composite.items()
            if np.isfinite(future_relative.at[origin, asset])
        }
        ranked = sorted(available, key=available.get, reverse=True)
        chosen = ranked[: int(sector_spec["top_n"])]
        if chosen:
            outcomes.append(float(future_relative.loc[origin, chosen].mean()))
        for asset, score in available.items():
            regress_x.append(float(score))
            regress_y.append(float(future_relative.at[origin, asset]))
    values = np.asarray(outcomes, dtype=float)
    x = np.asarray(regress_x, dtype=float)
    y = np.asarray(regress_y, dtype=float)
    denominator = (
        float(np.dot(x - x.mean(), x - x.mean())) if len(x) > 1 else 0.0
    )
    slope = (
        float(np.dot(x - x.mean(), y - y.mean()) / denominator)
        if denominator > 0.0
        else 0.0
    )
    slope_weight = len(values) / (len(values) + 52.0)
    return (
        {
            "status": "frozen_pre_2023_selection_only",
            "selected_id": "equal_weight_26_to_4_and_52_to_4_week",
            "component_weights": dict(sector_spec["momentum_combination"]),
            "selection_rebalances": int(len(values)),
            "selection_mean_relative_return": (
                float(values.mean()) if len(values) else None
            ),
            "selection_skill_passed": bool(
                len(values) and float(values.mean()) > 0.0
            ),
        },
        float(slope * slope_weight),
    )


def _signal_frame(
    weekly: Sequence[Mapping[str, Any]],
    prices: pd.DataFrame,
    *,
    forecast_model: str,
) -> list[dict[str, Any]]:
    lookup = _date_lookup(pd.DatetimeIndex(prices.index))
    result: list[dict[str, Any]] = []
    for source in weekly:
        origin_date = str(source.get("date"))
        origin = lookup.get(origin_date)
        if origin is None:
            raise ValueError("allocation-shadow origin is absent from price panel")
        forecast = _selected_model_forecast(source, forecast_model=forecast_model)
        probabilities = _normalized_probabilities(forecast)
        target_date = str(forecast["date"])
        result.append(
            {
                "origin": origin,
                "origin_date": origin_date,
                "target_date": target_date,
                "target": lookup.get(target_date),
                "probabilities": probabilities,
            }
        )
    return result


def _weights_dict(values: np.ndarray, assets: Sequence[str]) -> dict[str, float]:
    return {
        asset: float(values[index])
        for index, asset in enumerate(assets)
        if float(values[index]) > 1e-12 or asset in {"SPY", "TLT"}
    }


def _delta_dict(values: np.ndarray, assets: Sequence[str]) -> dict[str, float]:
    return {
        asset: float(values[index])
        for index, asset in enumerate(assets)
        if abs(float(values[index])) > 1e-12 or asset in {"SPY", "TLT"}
    }


def build_allocation_shadow_candidate(
    weekly: Sequence[Mapping[str, Any]],
    prices: pd.DataFrame,
    states: pd.Series,
    *,
    forecast_model: str,
    selection_end: str | date,
    current_signal: Mapping[str, Any],
    calibration_evidence: Mapping[str, Any] | None = None,
    spec_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build one bounded, fully self-financing post-selection research path."""

    if not weekly:
        raise ValueError("allocation-shadow requires weekly forecasts")
    if not isinstance(prices.index, pd.DatetimeIndex) or not prices.index.is_monotonic_increasing:
        raise ValueError("allocation-shadow prices require a sorted DatetimeIndex")
    spec = load_allocation_shadow_spec(spec_path)
    selection_end_date = (
        selection_end if isinstance(selection_end, date) else date.fromisoformat(str(selection_end))
    )
    anchors = tuple(spec["assets"]["anchor"])
    sectors = tuple(spec["assets"]["sectors"])
    assets = (*anchors, *sectors)
    gaps, intraday = split_safe_asset_return_frames(prices, assets)
    future_intraday = (intraday - 1.0).shift(-1)
    origins = _selection_origins(
        pd.DatetimeIndex(prices.index),
        selection_end=selection_end_date,
        lookback_weeks=int(spec["payoff_decoder"]["selection_lookback_weeks"]),
    )
    minimum = int(spec["payoff_decoder"]["minimum_selection_weeks"])
    if len(origins) < minimum:
        raise ValueError("allocation-shadow selection history is insufficient")
    # The operating forecast is P(S[t+1] | information at t), so its payoff
    # decoder must estimate E[R[t+1] | S[t+1]].  Conditioning on S[t] would
    # mix a target-state probability with an origin-state payoff table.
    target_states_at_origin = states.reindex(prices.index).shift(-1)
    anchor_relative = pd.DataFrame(
        {"SPY_MINUS_TLT": future_intraday["SPY"] - future_intraday["TLT"]},
        index=prices.index,
    )
    sector_relative = pd.DataFrame(
        {
            asset: future_intraday[asset] - future_intraday["SPY"]
            for asset in sectors
        },
        index=prices.index,
    )
    pseudo = float(spec["payoff_decoder"]["shrinkage_pseudo_weeks"])
    prior = float(spec["payoff_decoder"]["prior_relative_payoff"])
    anchor_payoffs = _shrunk_state_payoffs(
        target_states_at_origin,
        anchor_relative,
        origins,
        pseudo_weeks=pseudo,
        prior=prior,
    )
    sector_payoffs = _shrunk_state_payoffs(
        target_states_at_origin,
        sector_relative,
        origins,
        pseudo_weeks=pseudo,
        prior=prior,
    )
    relative_sample = anchor_relative["SPY_MINUS_TLT"].reindex(origins).dropna()
    relative_volatility = float(relative_sample.std(ddof=1))
    if not np.isfinite(relative_volatility) or relative_volatility <= 0.0:
        raise ValueError("allocation-shadow selection relative volatility is invalid")
    confidence = float(spec["payoff_decoder"]["default_confidence_multiplier"])
    confidence_candidates = [
        {
            "confidence_multiplier": float(value),
            "status": "not_evaluated_no_pre_2023_forecast_probabilities",
        }
        for value in spec["payoff_decoder"]["confidence_multipliers"]
    ]
    skill_passed, skill = _selection_skill(
        calibration_evidence,
        minimum_predictions=int(spec["selection_gate"]["minimum_calibration_predictions"]),
    )
    if not skill_passed:
        confidence = 0.0
    momentum_selection, momentum_slope = _momentum_selection(
        prices,
        origins,
        sectors,
        sector_relative,
        spec["sector_rotation"],
    )
    momentum_skill_passed = bool(momentum_selection["selection_skill_passed"])
    signal_rows = _signal_frame(weekly, prices, forecast_model=forecast_model)
    realized = [row for row in signal_rows if row["target"] is not None]
    if not realized:
        raise ValueError("allocation-shadow has no realized OOS targets")
    evaluation_index = pd.DatetimeIndex([row["target"] for row in realized])
    if evaluation_index.has_duplicates:
        raise ValueError("allocation-shadow target weeks must be unique")
    anchor = spec["anchor_weights"]
    decoder = spec["payoff_decoder"]
    sector_spec = spec["sector_rotation"]
    aims = {
        name: pd.DataFrame(0.0, index=evaluation_index, columns=assets, dtype=float)
        for name in (
            "realistic_60_40",
            "spy_buy_and_hold",
            "regime_only",
            "momentum_only",
            "combined",
        )
    }
    expected_frames = {
        name: pd.DataFrame(0.0, index=evaluation_index, columns=assets, dtype=float)
        for name in ("regime_only", "momentum_only", "combined")
    }
    previous_month: pd.Timestamp | None = None
    momentum_selection_assets: list[str] = []
    combined_selection_assets: list[str] = []
    latest_ranking: list[dict[str, Any]] = []
    latest_regime_expected = np.zeros(len(assets), dtype=float)
    latest_combined_expected = np.zeros(len(assets), dtype=float)
    for row in signal_rows:
        origin = row["origin"]
        probabilities = row["probabilities"]
        relative_payoff = _expected_payoff(
            anchor_payoffs, "SPY_MINUS_TLT", probabilities
        )
        risk_scale = min(
            1.0,
            float(decoder["target_weekly_relative_volatility"]) / relative_volatility,
        )
        tilt = confidence * float(decoder["maximum_spy_tilt"]) * math.tanh(
            relative_payoff
            / max(relative_volatility * float(decoder["signal_scale"]), 1e-12)
        ) * risk_scale
        equity_weight = float(np.clip(float(anchor["SPY"]) + tilt, 0.0, 1.0))
        momentum_z, _ = _composite_momentum_at(
            prices, origin, sectors, sector_spec
        )
        regime_relative = {
            asset: _expected_payoff(sector_payoffs, asset, probabilities)
            for asset in momentum_z
        }
        momentum_expected = {
            asset: momentum_slope * momentum_z[asset] for asset in momentum_z
        }
        regime_z = _cross_section_z(regime_relative)
        score_weights = sector_spec["combined_score_weights"]
        combined_score = {
            asset: float(score_weights["momentum"]) * momentum_z.get(asset, 0.0)
            + float(score_weights["regime"]) * regime_z.get(asset, 0.0)
            for asset in momentum_z
        }
        entry_month = _scheduled_nyse_entry_at(row["target_date"])
        if _month_changed(previous_month, entry_month):
            if momentum_skill_passed:
                momentum_selection_assets = sorted(
                    momentum_z, key=momentum_z.get, reverse=True
                )[: int(sector_spec["top_n"])]
                combined_selection_assets = sorted(
                    combined_score, key=combined_score.get, reverse=True
                )[: int(sector_spec["top_n"])]
            else:
                momentum_selection_assets = []
                combined_selection_assets = []
            previous_month = entry_month
        ranking: list[dict[str, Any]] = []
        ranking_sleeve = min(
            float(sector_spec["maximum_total_asset_weight"]),
            equity_weight * float(sector_spec["maximum_equity_fraction"]),
            len(combined_selection_assets)
            * float(sector_spec["maximum_symbol_weight"]),
        )
        ranking_symbol_weight = (
            min(
                float(sector_spec["maximum_symbol_weight"]),
                ranking_sleeve / len(combined_selection_assets),
            )
            if combined_selection_assets
            else 0.0
        )
        for asset in sorted(combined_score, key=combined_score.get, reverse=True):
            ci_lower, ci_upper = _expected_interval(
                sector_payoffs, asset, probabilities
            )
            ranking.append(
                {
                    "symbol": asset,
                    "score": float(combined_score[asset]),
                    "relative_momentum": float(momentum_z[asset]),
                    "regime_relative_payoff": float(regime_relative[asset]),
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "target_weight": (
                        float(ranking_symbol_weight)
                        if asset in combined_selection_assets
                        else 0.0
                    ),
                    "selected": asset in combined_selection_assets,
                }
            )
        latest_ranking = ranking
        regime_expected_vector = np.zeros(len(assets), dtype=float)
        regime_expected_vector[0] = 0.5 * relative_payoff
        regime_expected_vector[1] = -0.5 * relative_payoff
        momentum_vector = np.zeros(len(assets), dtype=float)
        for asset, value in momentum_expected.items():
            momentum_vector[assets.index(asset)] = value
        combined_vector = regime_expected_vector.copy()
        for asset in combined_score:
            combined_vector[assets.index(asset)] = (
                0.5 * relative_payoff
                + float(score_weights["momentum"]) * momentum_expected[asset]
                + float(score_weights["regime"]) * regime_relative[asset]
            )
        latest_regime_expected = regime_expected_vector
        latest_combined_expected = combined_vector
        if row["target"] is None:
            continue
        target = row["target"]
        base = np.zeros(len(assets), dtype=float)
        base[0], base[1] = float(anchor["SPY"]), float(anchor["TLT"])
        aims["realistic_60_40"].loc[target] = base
        spy_only = np.zeros(len(assets), dtype=float)
        spy_only[0] = 1.0
        aims["spy_buy_and_hold"].loc[target] = spy_only
        regime_weights = np.zeros(len(assets), dtype=float)
        regime_weights[0], regime_weights[1] = equity_weight, 1.0 - equity_weight
        aims["regime_only"].loc[target] = regime_weights
        momentum_weights = base.copy()
        momentum_sleeve = min(
            float(sector_spec["maximum_total_asset_weight"]),
            float(anchor["SPY"]) * float(sector_spec["maximum_equity_fraction"]),
            len(momentum_selection_assets) * float(sector_spec["maximum_symbol_weight"]),
        )
        per_symbol = (
            min(
                float(sector_spec["maximum_symbol_weight"]),
                momentum_sleeve / len(momentum_selection_assets),
            )
            if momentum_selection_assets
            else 0.0
        )
        momentum_weights[0] -= per_symbol * len(momentum_selection_assets)
        for asset in momentum_selection_assets:
            momentum_weights[assets.index(asset)] = per_symbol
        aims["momentum_only"].loc[target] = momentum_weights
        combined_weights = regime_weights.copy()
        combined_sleeve = min(
            float(sector_spec["maximum_total_asset_weight"]),
            equity_weight * float(sector_spec["maximum_equity_fraction"]),
            len(combined_selection_assets) * float(sector_spec["maximum_symbol_weight"]),
        )
        combined_per_symbol = (
            min(
                float(sector_spec["maximum_symbol_weight"]),
                combined_sleeve / len(combined_selection_assets),
            )
            if combined_selection_assets
            else 0.0
        )
        combined_weights[0] -= combined_per_symbol * len(combined_selection_assets)
        for asset in combined_selection_assets:
            combined_weights[assets.index(asset)] = combined_per_symbol
        aims["combined"].loc[target] = combined_weights
        expected_frames["regime_only"].loc[target] = regime_expected_vector
        expected_frames["momentum_only"].loc[target] = momentum_vector
        expected_frames["combined"].loc[target] = combined_vector
    dgs3mo = pd.to_numeric(prices.get("dgs3mo"), errors="coerce").astype(float)
    cash_factors = (1.0 + dgs3mo.shift(1) / 100.0 / 52.0).reindex(evaluation_index)
    if cash_factors.isna().any() or bool((cash_factors <= 0.0).any()):
        raise ValueError("allocation-shadow DGS3MO cash proxy is unavailable")
    annualization = float(spec["metrics"]["annualization_weeks"])
    gamma = float(spec["metrics"]["certainty_equivalent_risk_aversion"])
    runs_by_cost: dict[str, dict[str, Any]] = {}
    metrics_by_cost: dict[str, dict[str, Any]] = {}
    for cost_bps in spec["cost"]["sensitivity_bps_per_traded_notional"]:
        key = f"{float(cost_bps):g}bps"
        runs_by_cost[key] = {}
        metrics_by_cost[key] = {}
        for name in aims:
            run = _run_strategy(
                aims[name],
                expected_frames.get(name),
                gaps,
                intraday,
                cash_factors,
                evaluation_index,
                assets=assets,
                execution=spec["execution"],
                cost_bps=float(cost_bps),
            )
            runs_by_cost[key][name] = run
            metrics_by_cost[key][name] = _metrics(
                run["rows"], annualization=annualization, gamma=gamma
            )
    primary_key = f"{float(spec['cost']['primary_bps_per_traded_notional']):g}bps"
    primary_runs = runs_by_cost[primary_key]
    primary_metrics = metrics_by_cost[primary_key]
    active_metrics = primary_metrics["combined"]
    baseline_metrics = primary_metrics["realistic_60_40"]
    required = tuple(spec["selection_gate"]["candidate_required_to_improve"])
    primary_improved = all(
        float(active_metrics[field]) > float(baseline_metrics[field]) for field in required
    )
    sensitivity_20 = metrics_by_cost.get("20bps", {})
    sensitivity_improved = (
        not bool(spec["selection_gate"]["require_20bps_certainty_equivalent_improvement"])
        or float(sensitivity_20["combined"]["certainty_equivalent_return"])
        > float(sensitivity_20["realistic_60_40"]["certainty_equivalent_return"])
    )
    turnover_passed = float(active_metrics["annualized_one_way_turnover"]) <= float(
        spec["selection_gate"]["maximum_annualized_one_way_turnover"]
    )
    drawdown_passed = float(active_metrics["maximum_drawdown"]) >= float(
        baseline_metrics["maximum_drawdown"]
    ) - float(spec["selection_gate"]["maximum_drawdown_shortfall"])
    recurring_rebalances = sum(
        row["action"] == "rebalance" for row in primary_runs["combined"]["rows"][1:]
    )
    activity_passed = recurring_rebalances >= int(
        spec["selection_gate"]["minimum_recurring_executed_rebalances"]
    )
    ablation_improved = all(
        all(
            float(active_metrics[field]) > float(primary_metrics[name][field])
            for field in required
        )
        for name in spec["selection_gate"]["require_ablation_improvement"]
    )
    improved = (
        skill_passed
        and momentum_skill_passed
        and primary_improved
        and sensitivity_improved
        and turnover_passed
        and drawdown_passed
        and activity_passed
        and ablation_improved
    )
    policy_status = "candidate_selected" if improved else "baseline_preferred"
    recommended_policy = "combined" if improved else "realistic_60_40"
    performance_path: list[dict[str, Any]] = []
    for position, timestamp in enumerate(evaluation_index):
        strategy_rows = {
            name: {
                key: value
                for key, value in primary_runs[name]["rows"][position].items()
                if key
                in {
                    "gross_return",
                    "net_return",
                    "gross_wealth",
                    "net_wealth",
                    "drawdown",
                    "one_way_turnover",
                    "full_l1_turnover",
                    "transaction_cost_rate",
                    "action",
                }
            }
            for name in aims
        }
        performance_path.append(
            {
                "week": timestamp.date().isoformat(),
                "date": timestamp.date().isoformat(),
                "strategies": strategy_rows,
            }
        )
    latest = signal_rows[-1]
    latest_probabilities = latest["probabilities"]
    latest_target_date = latest["target_date"]
    latest_aims = {name: aims[name].iloc[-1].to_numpy(dtype=float) for name in aims}
    # If the latest forecast targets an unrealized week, rebuild its aims from the
    # current ranking and decoder rather than copying the prior realized week.
    if latest["target"] is None:
        relative_payoff = _expected_payoff(
            anchor_payoffs, "SPY_MINUS_TLT", latest_probabilities
        )
        risk_scale = min(
            1.0,
            float(decoder["target_weekly_relative_volatility"]) / relative_volatility,
        )
        equity = float(anchor["SPY"]) + confidence * float(decoder["maximum_spy_tilt"]) * math.tanh(
            relative_payoff / max(relative_volatility * float(decoder["signal_scale"]), 1e-12)
        ) * risk_scale
        equity = float(np.clip(equity, 0.0, 1.0))
        baseline_aim = np.zeros(len(assets), dtype=float)
        baseline_aim[0], baseline_aim[1] = float(anchor["SPY"]), float(anchor["TLT"])
        regime_aim = np.zeros(len(assets), dtype=float)
        regime_aim[0], regime_aim[1] = equity, 1.0 - equity
        combined_aim = regime_aim.copy()
        selected_now = [row["symbol"] for row in latest_ranking if row["selected"]]
        sleeve = min(
            float(sector_spec["maximum_total_asset_weight"]),
            equity * float(sector_spec["maximum_equity_fraction"]),
            len(selected_now) * float(sector_spec["maximum_symbol_weight"]),
        )
        per_symbol = sleeve / len(selected_now) if selected_now else 0.0
        combined_aim[0] -= per_symbol * len(selected_now)
        for asset in selected_now:
            combined_aim[assets.index(asset)] = per_symbol
        latest_aims["realistic_60_40"] = baseline_aim
        spy_only = np.zeros(len(assets), dtype=float)
        spy_only[0] = 1.0
        latest_aims["spy_buy_and_hold"] = spy_only
        latest_aims["regime_only"] = regime_aim
        latest_aims["combined"] = combined_aim
    recommended_run = primary_runs[recommended_policy]
    shadow_run = primary_runs["combined"]
    recommended_prior = np.asarray(recommended_run["close_weights"], dtype=float)
    recommended_cash = float(recommended_run["close_cash"])
    shadow_prior = np.asarray(shadow_run["close_weights"], dtype=float)
    shadow_cash = float(shadow_run["close_cash"])
    timing_action = str(current_signal.get("action"))
    recommended_expected = (
        latest_combined_expected if recommended_policy == "combined" else None
    )
    recommended_trade = _trade_decision(
        recommended_prior,
        recommended_cash,
        latest_aims[recommended_policy],
        execution=spec["execution"],
        cost_rate=float(spec["cost"]["primary_bps_per_traded_notional"]) / 10_000.0,
        expected_returns=recommended_expected,
        initial=False,
        force_no_trade=timing_action == "no_trade",
    )
    shadow_trade = _trade_decision(
        shadow_prior,
        shadow_cash,
        latest_aims["combined"],
        execution=spec["execution"],
        cost_rate=float(spec["cost"]["primary_bps_per_traded_notional"]) / 10_000.0,
        expected_returns=latest_combined_expected,
        initial=False,
        force_no_trade=timing_action == "no_trade",
    )
    delta = recommended_trade.assets - recommended_prior
    current_intent = {
        "schema_version": PORTFOLIO_INTENT_SCHEMA_VERSION,
        "forecast": {
            "origin_date": latest["origin_date"],
            "target_week": latest_target_date,
            "model": forecast_model,
            "probabilities": latest_probabilities,
        },
        "prior": {
            "basis": "reconstructed_strategy_close",
            "weights": _weights_dict(recommended_prior, assets),
            "cash": recommended_cash,
        },
        "aim": {
            "policy": recommended_policy,
            "weights": _weights_dict(latest_aims[recommended_policy], assets),
        },
        "shadow_aim": {
            "policy": "combined",
            "weights": _weights_dict(latest_aims["combined"], assets),
        },
        "recommended": {
            "policy": recommended_policy,
            "weights": _weights_dict(recommended_trade.assets, assets),
            "cash": recommended_trade.cash,
            "action": recommended_trade.action,
        },
        "target": {
            "weights": _weights_dict(recommended_trade.assets, assets),
            "cash": recommended_trade.cash,
        },
        "shadow_target": {
            "policy": "combined",
            "weights": _weights_dict(shadow_trade.assets, assets),
            "cash": shadow_trade.cash,
            "action": shadow_trade.action,
        },
        "order_delta": {
            "weights": _delta_dict(delta, assets),
            "one_way_turnover": recommended_trade.one_way,
            "full_l1_turnover": recommended_trade.full_l1,
        },
        "cost": {
            "bps_per_traded_notional": float(
                spec["cost"]["primary_bps_per_traded_notional"]
            ),
            "estimated_rate": (
                float(spec["cost"]["primary_bps_per_traded_notional"])
                / 10_000.0
                * recommended_trade.full_l1
            ),
        },
        "expected_returns": {
            "basis": "selection_frozen_weekly_relative_payoff",
            "recommended": (
                None
                if recommended_expected is None
                else _delta_dict(recommended_expected, assets)
            ),
            "shadow": _delta_dict(latest_combined_expected, assets),
        },
        "cash_accrual": {
            "source": "DGS3MO",
            "as_of": latest["origin_date"],
            "annual_yield_percent": float(prices.at[latest["origin"], "dgs3mo"]),
            "weekly_factor": float(
                1.0 + float(prices.at[latest["origin"], "dgs3mo"]) / 100.0 / 52.0
            ),
        },
        "timing": dict(current_signal),
    }
    body = {
        "schema_version": ALLOCATION_RESULT_SCHEMA_VERSION,
        "role": spec["role"],
        "policy_status": policy_status,
        "recommended_target": recommended_policy,
        "spec": {
            "path": "config/allocation-shadow-v1.json",
            "sha256": canonical_json_sha256_v1(spec),
            "spec_id": spec["spec_id"],
        },
        "selection": {
            "end": selection_end_date.isoformat(),
            "start": origins.min().date().isoformat(),
            "weeks": len(origins),
            "payoff_method": (
                "target_state_conditioned_next_open_to_close_relative_return"
            ),
            "shrinkage_pseudo_weeks": pseudo,
            "relative_volatility": relative_volatility,
            "calibration_skill": skill,
            "confidence_multiplier": confidence,
            "confidence_candidates": confidence_candidates,
            "target_state_payoffs": anchor_payoffs["SPY_MINUS_TLT"],
        },
        "execution_contract": dict(spec["execution"]),
        "return_accounting": dict(spec["return_accounting"]),
        "current_intent": current_intent,
        "performance": {
            "evaluation_start_week": evaluation_index.min().date().isoformat(),
            "evaluation_end_week": evaluation_index.max().date().isoformat(),
            "strategies": primary_metrics,
            "cost_sensitivity": metrics_by_cost,
            "economic_gate": {
                "calibration_skill_passed": skill_passed,
                "primary_return_and_ce_improved": primary_improved,
                "20bps_ce_improved": sensitivity_improved,
                "annualized_one_way_turnover_passed": turnover_passed,
                "maximum_drawdown_passed": drawdown_passed,
                "positive_selection_momentum_passed": momentum_skill_passed,
                "recurring_executed_rebalances": recurring_rebalances,
                "minimum_recurring_activity_passed": activity_passed,
                "ablation_improvement_passed": ablation_improved,
            },
        },
        "performance_path": performance_path,
        "sector_rotation": {
            "policy_status": policy_status,
            "selected_strategy": recommended_policy,
            "momentum_selection": momentum_selection,
            "ranking": latest_ranking,
        },
        "affects_official_forecast": False,
        "affects_champion_selection": False,
    }
    return body


def rebase_allocation_candidate_intent(
    candidate: Mapping[str, Any],
    *,
    prior_weights: Mapping[str, float] | None,
    prior_cash: float,
    prior_basis: str,
) -> dict[str, Any]:
    """Freeze an intent against the prospective ledger's actual closing state."""

    result = deepcopy(dict(candidate))
    if result.get("schema_version") != ALLOCATION_RESULT_SCHEMA_VERSION:
        raise ValueError("allocation-shadow candidate schema is invalid")
    spec = load_allocation_shadow_spec()
    identity = result.get("spec")
    if (
        not isinstance(identity, Mapping)
        or identity.get("sha256") != canonical_json_sha256_v1(spec)
        or identity.get("spec_id") != spec.get("spec_id")
    ):
        raise ValueError("allocation-shadow candidate spec is inconsistent")
    intent = result.get("current_intent")
    if not isinstance(intent, dict):
        raise ValueError("allocation-shadow current intent is invalid")
    assets = tuple((*spec["assets"]["anchor"], *spec["assets"]["sectors"]))
    prior = np.asarray(
        [float((prior_weights or {}).get(asset, 0.0)) for asset in assets],
        dtype=float,
    )
    cash = float(prior_cash)
    if (
        not np.isfinite(prior).all()
        or not np.isfinite(cash)
        or (prior < -1e-12).any()
        or cash < -1e-12
        or not math.isclose(float(prior.sum()) + cash, 1.0, abs_tol=1e-8)
    ):
        raise ValueError("allocation-shadow prospective prior is invalid")

    def vector(block_name: str) -> np.ndarray:
        block = intent.get(block_name)
        weights = block.get("weights") if isinstance(block, Mapping) else None
        if not isinstance(weights, Mapping):
            raise ValueError(f"allocation-shadow {block_name} weights are invalid")
        return np.asarray([float(weights.get(asset, 0.0)) for asset in assets])

    expected_block = intent.get("expected_returns")
    if not isinstance(expected_block, Mapping):
        raise ValueError("allocation-shadow expected returns are invalid")

    def expected_vector(field: str) -> np.ndarray | None:
        raw = expected_block.get(field)
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise ValueError("allocation-shadow expected returns are invalid")
        return np.asarray([float(raw.get(asset, 0.0)) for asset in assets])

    initial = bool(float(prior.sum()) <= 1e-12 and cash >= 1.0 - 1e-12)
    force_no_trade = intent.get("timing", {}).get("action") == "no_trade"
    cost_rate = float(spec["cost"]["primary_bps_per_traded_notional"]) / 10_000.0
    recommended_trade = _trade_decision(
        prior,
        cash,
        vector("aim"),
        execution=spec["execution"],
        cost_rate=cost_rate,
        expected_returns=expected_vector("recommended"),
        initial=initial,
        force_no_trade=force_no_trade,
    )
    shadow_trade = _trade_decision(
        prior,
        cash,
        vector("shadow_aim"),
        execution=spec["execution"],
        cost_rate=cost_rate,
        expected_returns=expected_vector("shadow"),
        initial=initial,
        force_no_trade=force_no_trade,
    )
    delta = recommended_trade.assets - prior
    intent["prior"] = {
        "basis": str(prior_basis),
        "weights": _weights_dict(prior, assets),
        "cash": cash,
    }
    recommended_policy = str(intent["aim"]["policy"])
    intent["recommended"] = {
        "policy": recommended_policy,
        "weights": _weights_dict(recommended_trade.assets, assets),
        "cash": recommended_trade.cash,
        "action": recommended_trade.action,
    }
    intent["target"] = {
        "weights": _weights_dict(recommended_trade.assets, assets),
        "cash": recommended_trade.cash,
    }
    intent["shadow_target"] = {
        "policy": "combined",
        "weights": _weights_dict(shadow_trade.assets, assets),
        "cash": shadow_trade.cash,
        "action": shadow_trade.action,
    }
    intent["order_delta"] = {
        "weights": _delta_dict(delta, assets),
        "one_way_turnover": recommended_trade.one_way,
        "full_l1_turnover": recommended_trade.full_l1,
    }
    intent["cost"]["estimated_rate"] = cost_rate * recommended_trade.full_l1
    return result


__all__ = [
    "ALLOCATION_RESULT_SCHEMA_VERSION",
    "allocation_calibration_evidence",
    "build_allocation_shadow_candidate",
    "default_allocation_shadow_spec_path",
    "load_allocation_shadow_spec",
    "rebase_allocation_candidate_intent",
    "split_safe_asset_return_frames",
]
