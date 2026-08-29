"""Preregistered probability-to-exposure decision shadow.

This module is intentionally downstream of the official probability forecast.
It cannot select a model, alter a probability, or promote a champion.  Its
historical reconstruction and prospective ledger evaluation are separately
labelled evidence tracks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    MO,
    USLaborDay,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)
from pandas.tseries.offsets import DateOffset

from regime_lab.integrity import canonical_json_sha256_v1


STATE_ORDER = ("risk_on", "transition", "risk_off")
ASSETS = ("SPY", "TLT")
SPEC_SCHEMA_VERSION = "regime-decision-shadow-spec/1"
RESULT_SCHEMA_VERSION = "regime-prospective-decision-shadow/2"
EXECUTION_CONTRACT = {
    "signal_origin": "completed_weekly_close",
    "first_tradable_point": "next_week_adjusted_open",
    "target_return_window": "next_week_open_to_close",
    "rebalance_frequency": "weekly",
    "late_signal_policy": "no_trade",
    "holding_period_weeks": 1,
}
TURNOVER_DEFINITION = "sum_absolute_target_minus_drifted_pretrade_weight"
RETURN_ACCOUNTING_CONTRACT = {
    "method": "split_safe_price_only_weekly",
    "total_factor_source": "provider_adjusted_close_to_close",
    "split_inference": (
        "total_factor_x_prior_raw_close_over_current_raw_close_plus_dividend"
    ),
    "gap_window": "prior_raw_close_to_target_raw_open_after_inferred_split",
    "holding_window": "target_raw_open_to_raw_close",
    "distribution_policy": "excluded_without_ex_date_for_all_strategies",
}


class _NYSEStandardHolidayCalendar(AbstractHolidayCalendar):
    """Recurring full-day NYSE holidays used to locate the weekly entry open."""

    rules = [
        Holiday(
            "New Year's Day",
            month=1,
            day=1,
            observance=sunday_to_monday,
        ),
        Holiday(
            "Martin Luther King Jr. Day",
            month=1,
            day=1,
            start_date=pd.Timestamp("1998-01-01"),
            offset=DateOffset(weekday=MO(3)),
        ),
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday(
            "Juneteenth National Independence Day",
            month=6,
            day=19,
            start_date=pd.Timestamp("2022-01-01"),
            observance=nearest_workday,
        ),
        Holiday(
            "Independence Day",
            month=7,
            day=4,
            observance=nearest_workday,
        ),
        USLaborDay,
        USThanksgivingDay,
        Holiday(
            "Christmas Day",
            month=12,
            day=25,
            observance=nearest_workday,
        ),
    ]


def _iso_market_date(value: object, *, field: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"decision-shadow {field} must be an ISO date") from exc


def _aware_timestamp(value: object, *, field: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"decision-shadow {field} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _scheduled_nyse_entry_at(target_week: object) -> pd.Timestamp:
    """Return the first regular NYSE session open in the target market week."""

    target = _iso_market_date(target_week, field="target_week")
    week_start = target - timedelta(days=target.weekday())
    week_end = week_start + timedelta(days=6)
    holidays = {
        pd.Timestamp(value).date()
        for value in _NYSEStandardHolidayCalendar().holidays(
            start=pd.Timestamp(week_start),
            end=pd.Timestamp(week_end),
        )
    }
    holidays.update(
        {
            date(2007, 1, 2),
            date(2012, 10, 29),
            date(2012, 10, 30),
        }
    )
    session = week_start
    while session <= week_end:
        if session.weekday() < 5 and session not in holidays:
            return pd.Timestamp(
                datetime.combine(session, time(9, 30)),
                tz="America/New_York",
            )
        session += timedelta(days=1)
    raise ValueError("decision-shadow target week has no standard NYSE session")


def _current_signal_contract(
    latest_week: Mapping[str, Any],
    *,
    decision_at: object | None,
    forecast_model: str,
) -> dict[str, Any]:
    origin_date = _iso_market_date(latest_week.get("date"), field="origin_date")
    forecast = _selected_model_forecast(
        latest_week,
        forecast_model=forecast_model,
    )
    target_week = _iso_market_date(forecast.get("date"), field="target_week")
    scheduled_entry = _scheduled_nyse_entry_at(target_week)
    if decision_at is None:
        decision = pd.Timestamp(
            datetime.combine(origin_date, time(16, 0)),
            tz="America/New_York",
        ).tz_convert("UTC")
    else:
        decision = _aware_timestamp(decision_at, field="decision_at")
    scheduled = decision < scheduled_entry.tz_convert("UTC")
    return {
        "origin_date": origin_date.isoformat(),
        "target_week": target_week.isoformat(),
        "scheduled_entry_at": scheduled_entry.isoformat(),
        "decision_at": decision.isoformat(),
        "forecast_model": forecast_model,
        "status": "scheduled" if scheduled else "missed_entry",
        "action": "trade_at_scheduled_open" if scheduled else "no_trade",
    }


def default_decision_shadow_spec_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "config"
        / "decision-shadow-v2.json"
    )


def load_decision_shadow_spec(path: str | Path | None = None) -> dict[str, Any]:
    selected = default_decision_shadow_spec_path() if path is None else Path(path)
    document = json.loads(selected.read_text(encoding="utf-8"))
    if document.get("schema_version") != SPEC_SCHEMA_VERSION:
        raise ValueError("decision-shadow spec schema is invalid")
    if document.get("role") != "research_only_no_forecast_or_champion_effect":
        raise ValueError("decision-shadow role is invalid")
    if document.get("assets") != list(ASSETS):
        raise ValueError("decision-shadow assets are invalid")
    if document.get("execution") != EXECUTION_CONTRACT:
        raise ValueError("decision-shadow execution contract is invalid")
    if document.get("return_accounting") != RETURN_ACCOUNTING_CONTRACT:
        raise ValueError("decision-shadow return accounting contract is invalid")
    cost = document.get("cost")
    if (
        not isinstance(cost, Mapping)
        or cost.get("turnover_definition") != TURNOVER_DEFINITION
        or not isinstance(cost.get("initial_allocation_costed_from_cash"), bool)
    ):
        raise ValueError("decision-shadow cost contract is invalid")
    cost_bps = float(cost.get("one_way_turnover_bps", float("nan")))
    if not np.isfinite(cost_bps) or cost_bps < 0.0:
        raise ValueError("decision-shadow transaction cost must be non-negative")
    mapping = document.get("probability_weight_mapping")
    if not isinstance(mapping, Mapping) or set(mapping) != set(STATE_ORDER):
        raise ValueError("decision-shadow probability mapping is invalid")
    for state in STATE_ORDER:
        weights = mapping[state]
        if not isinstance(weights, Mapping) or set(weights) != set(ASSETS):
            raise ValueError("decision-shadow state weights are invalid")
        values = [float(weights[asset]) for asset in ASSETS]
        if any(value < 0.0 or value > 1.0 for value in values) or not math.isclose(
            sum(values), 1.0, abs_tol=1e-12
        ):
            raise ValueError("decision-shadow state weights must be long-only and sum to one")
    return document


def _selected_model_forecast(
    week: Mapping[str, Any],
    *,
    forecast_model: str,
) -> Mapping[str, Any]:
    if not isinstance(forecast_model, str) or not forecast_model:
        raise ValueError("decision-shadow forecast_model must be non-empty")
    raw_forecasts = week.get("model_forecasts")
    if not isinstance(raw_forecasts, Sequence) or isinstance(
        raw_forecasts,
        (str, bytes),
    ):
        raise ValueError("decision-shadow model_forecasts must be a sequence")
    matches = [
        row
        for row in raw_forecasts
        if isinstance(row, Mapping) and row.get("model") == forecast_model
    ]
    if len(matches) != 1:
        raise ValueError(
            "decision-shadow requires exactly one forecast for forecast_model"
        )
    origin_date = _iso_market_date(week.get("date"), field="origin_date")
    target_week = _iso_market_date(matches[0].get("date"), field="target_week")
    if target_week != origin_date + timedelta(days=7):
        raise ValueError(
            "decision-shadow model forecast target must be origin plus 7 days"
        )
    return matches[0]


def split_safe_price_only_return_frames(
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = {
        asset: {
            "adjusted_close": f"{asset.lower()}_close",
            "raw_open": f"{asset.lower()}_raw_open",
            "raw_close": f"{asset.lower()}_raw_close",
            "dividend": f"{asset.lower()}_dividend_amount",
        }
        for asset in ASSETS
    }
    missing = [
        name
        for asset in ASSETS
        for name in columns[asset].values()
        if name not in prices
    ]
    if missing:
        raise ValueError(f"decision-shadow prices are missing columns: {missing}")
    gap_relatives: dict[str, pd.Series] = {}
    open_to_close_relatives: dict[str, pd.Series] = {}
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
    for asset in ASSETS:
        adjusted_close = pd.to_numeric(
            prices[columns[asset]["adjusted_close"]], errors="coerce"
        ).astype(float)
        raw_open = pd.to_numeric(
            prices[columns[asset]["raw_open"]], errors="coerce"
        ).astype(float)
        raw_close = pd.to_numeric(
            prices[columns[asset]["raw_close"]], errors="coerce"
        ).astype(float)
        dividend = pd.to_numeric(
            prices[columns[asset]["dividend"]], errors="coerce"
        ).astype(float)
        finite_dividends = dividend[np.isfinite(dividend)]
        if (finite_dividends < 0.0).any():
            raise ValueError("decision-shadow dividends must be non-negative")
        prior_adjusted_close = adjusted_close.shift(1)
        prior_raw_close = raw_close.shift(1)
        total_factor = adjusted_close / prior_adjusted_close
        inferred_split = (
            total_factor * prior_raw_close / (raw_close + dividend)
        )
        finite_splits = inferred_split[np.isfinite(inferred_split)]
        if (finite_splits <= 0.0).any():
            raise ValueError("decision-shadow inferred split must be positive")
        if not finite_splits.empty:
            split_values = finite_splits.to_numpy(dtype=float)
            relative_errors = np.min(
                np.abs(split_values[:, None] / split_candidates[None, :] - 1.0),
                axis=1,
            )
            if bool((relative_errors > 0.02).any()):
                raise ValueError(
                    "decision-shadow inferred split is outside sanity tolerance"
                )
        gap_relatives[asset] = inferred_split * raw_open / prior_raw_close
        open_to_close_relatives[asset] = raw_close / raw_open
    return (
        pd.DataFrame(gap_relatives, index=prices.index, dtype=float),
        pd.DataFrame(open_to_close_relatives, index=prices.index, dtype=float),
    )


def _run_self_financing_strategy(
    target_weights: pd.DataFrame,
    gap_relatives: pd.DataFrame,
    open_to_close_relatives: pd.DataFrame,
    evaluation_index: pd.DatetimeIndex,
    *,
    cost_rate: float,
    initial_allocation_costed_from_cash: bool,
    late_signal_policy: str,
) -> dict[str, pd.Series | pd.DataFrame]:
    """Run one weekly-open strategy with drift-aware, self-financing accounting."""

    if late_signal_policy != "no_trade":
        raise ValueError("unsupported late-signal policy")
    if not np.isfinite(cost_rate) or cost_rate < 0.0:
        raise ValueError("cost_rate must be non-negative and finite")
    index = pd.DatetimeIndex(evaluation_index)
    targets = target_weights.reindex(index).loc[:, list(ASSETS)].astype(float)
    gaps = gap_relatives.reindex(index).loc[:, list(ASSETS)].astype(float)
    intraday = (
        open_to_close_relatives.reindex(index).loc[:, list(ASSETS)].astype(float)
    )

    net_rows: list[float] = []
    gross_rows: list[float] = []
    turnover_rows: list[float] = []
    transaction_cost_rows: list[float] = []
    applied_targets: list[np.ndarray] = []
    pretrade_rows: list[np.ndarray] = []
    prior_close_weights: np.ndarray | None = None
    prior_close_cash = 1.0

    for timestamp in index:
        gap_values = gaps.loc[timestamp].to_numpy(dtype=float)
        open_to_close = intraday.loc[timestamp].to_numpy(dtype=float)
        if not np.isfinite(open_to_close).all() or (open_to_close <= 0.0).any():
            raise ValueError(
                "decision-shadow open-to-close relatives must be positive and finite"
            )

        if prior_close_weights is None:
            gap_factor = 1.0
            pretrade_weights = np.zeros(len(ASSETS), dtype=float)
            pretrade_cash = 1.0
        else:
            if not np.isfinite(gap_values).all() or (gap_values <= 0.0).any():
                raise ValueError(
                    "decision-shadow gap relatives must be positive and finite"
                )
            gap_factor = float(
                prior_close_cash
                + np.dot(prior_close_weights, gap_values)
            )
            if not np.isfinite(gap_factor) or gap_factor <= 0.0:
                raise ValueError("decision-shadow close-to-open wealth must be positive")
            pretrade_weights = prior_close_weights * gap_values / gap_factor
            pretrade_cash = prior_close_cash / gap_factor

        requested = targets.loc[timestamp].to_numpy(dtype=float)
        if np.isfinite(requested).all():
            if (requested < -1e-12).any() or requested.sum() > 1.0 + 1e-12:
                raise ValueError(
                    "decision-shadow target weights must be long-only with gross at most one"
                )
            applied = np.maximum(requested, 0.0)
            target_cash = max(0.0, 1.0 - float(applied.sum()))
            if prior_close_weights is None:
                turnover = (
                    float(np.abs(applied - pretrade_weights).sum())
                    if initial_allocation_costed_from_cash
                    else 0.0
                )
            else:
                turnover = float(np.abs(applied - pretrade_weights).sum())
        else:
            applied = pretrade_weights.copy()
            target_cash = float(pretrade_cash)
            turnover = 0.0

        transaction_cost = cost_rate * turnover
        if transaction_cost >= 1.0:
            raise ValueError("decision-shadow transaction cost exhausts portfolio wealth")
        intraday_factor = float(target_cash + np.dot(applied, open_to_close))
        if not np.isfinite(intraday_factor) or intraday_factor <= 0.0:
            raise ValueError("decision-shadow open-to-close wealth must be positive")

        gross_factor = gap_factor * intraday_factor
        net_factor = gap_factor * (1.0 - transaction_cost) * intraday_factor
        gross_rows.append(gross_factor - 1.0)
        net_rows.append(net_factor - 1.0)
        turnover_rows.append(turnover)
        transaction_cost_rows.append(transaction_cost)
        applied_targets.append(applied.copy())
        pretrade_rows.append(pretrade_weights.copy())

        prior_close_weights = applied * open_to_close / intraday_factor
        prior_close_cash = target_cash / intraday_factor

    return {
        "net_returns": pd.Series(net_rows, index=index, dtype=float),
        "gross_returns": pd.Series(gross_rows, index=index, dtype=float),
        "turnover": pd.Series(turnover_rows, index=index, dtype=float),
        "transaction_cost": pd.Series(
            transaction_cost_rows, index=index, dtype=float
        ),
        "target_weights": pd.DataFrame(
            applied_targets, index=index, columns=ASSETS, dtype=float
        ),
        "pretrade_weights": pd.DataFrame(
            pretrade_rows, index=index, columns=ASSETS, dtype=float
        ),
    }


def _metrics(
    returns: pd.Series,
    turnover: pd.Series,
    *,
    annualization: float,
    risk_aversion: float,
) -> dict[str, Any]:
    clean = returns.dropna().astype(float)
    if clean.empty:
        return {
            "weeks": 0,
            "cumulative_return": None,
            "annualized_return": None,
            "annualized_volatility": None,
            "sharpe": None,
            "certainty_equivalent_return": None,
            "maximum_drawdown": None,
            "annualized_turnover": None,
        }
    wealth = (1.0 + clean).cumprod()
    years = len(clean) / annualization
    annualized_return = float(wealth.iloc[-1] ** (1.0 / years) - 1.0)
    annualized_volatility = (
        float(clean.std(ddof=1) * math.sqrt(annualization))
        if len(clean) > 1
        else 0.0
    )
    sharpe = (
        float(clean.mean() / clean.std(ddof=1) * math.sqrt(annualization))
        if len(clean) > 1 and clean.std(ddof=1) > 0.0
        else None
    )
    wealth_with_initial = np.concatenate(([1.0], wealth.to_numpy(dtype=float)))
    maximum_drawdown = float(
        (wealth_with_initial / np.maximum.accumulate(wealth_with_initial) - 1.0).min()
    )
    return {
        "weeks": int(len(clean)),
        "cumulative_return": float(wealth.iloc[-1] - 1.0),
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "certainty_equivalent_return": float(
            annualized_return - 0.5 * risk_aversion * annualized_volatility**2
        ),
        "maximum_drawdown": maximum_drawdown,
        "annualized_turnover": float(
            turnover.reindex(clean.index).fillna(0.0).mean() * annualization
        ),
    }


def prospective_ledger_shadow_contract(
    summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project the private ledger summary onto the public decision track."""

    empty_performance = {
        "status": "pending",
        "weeks": 0,
        "gross_cumulative_return": None,
        "net_cumulative_return": None,
        "turnover_sum": None,
        "transaction_cost_rate_sum": None,
        "transaction_cost_bps": None,
        "forecast_hit_count": None,
        "forecast_accuracy": None,
        "actual_state_counts": None,
    }
    if summary is None:
        ledger_status = "empty"
        entry_count = 0
        pending_count = 0
        unresolved_count = 0
        realized_count = 0
        partial_count = 0
        evaluation_manifest_sha256 = canonical_json_sha256_v1([])
        performance = empty_performance
    else:
        if summary.get("schema_version") != "regime-prospective-ledger-summary/2":
            raise ValueError("decision-shadow prospective ledger summary is invalid")
        ledger_status = str(summary.get("status", ""))
        if ledger_status not in {"empty", "pending", "completed", "partial"}:
            raise ValueError("decision-shadow prospective ledger status is invalid")
        count_fields = (
            "entry_count",
            "pending_evaluation_count",
            "unresolved_due_evaluation_count",
            "realized_evaluation_count",
            "partial_evaluation_count",
        )
        counts = {field: summary.get(field) for field in count_fields}
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts.values()
        ):
            raise ValueError("decision-shadow prospective ledger counts are invalid")
        entry_count = int(counts["entry_count"])
        pending_count = int(counts["pending_evaluation_count"])
        unresolved_count = int(counts["unresolved_due_evaluation_count"])
        realized_count = int(counts["realized_evaluation_count"])
        partial_count = int(counts["partial_evaluation_count"])
        evaluation_manifest_sha256 = str(
            summary.get("evaluation_manifest_sha256", "")
        )
        if len(evaluation_manifest_sha256) != 64:
            raise ValueError(
                "decision-shadow prospective evaluation manifest hash is invalid"
            )
        raw_performance = summary.get("performance")
        if not isinstance(raw_performance, Mapping):
            raise ValueError("decision-shadow prospective performance is invalid")
        performance = dict(raw_performance)
    return {
        "status": "pending" if ledger_status == "empty" else ledger_status,
        "evidence_track": "operational_oos",
        "ledger_entry_count": entry_count,
        "pending_evaluation_count": pending_count,
        "unresolved_due_evaluation_count": unresolved_count,
        "realized_evaluation_count": realized_count,
        "partial_evaluation_count": partial_count,
        "evaluation_manifest_sha256": evaluation_manifest_sha256,
        "performance": performance,
        "affects_official_forecast": False,
        "affects_champion_selection": False,
    }


def build_decision_shadow(
    weekly: Sequence[Mapping[str, Any]],
    prices: pd.DataFrame,
    *,
    forecast_model: str,
    spec_path: str | Path | None = None,
    prospective_ledger_summary: Mapping[str, Any] | None = None,
    decision_at: object | None = None,
) -> dict[str, Any]:
    """Evaluate reconstructed forecasts and reserve a prospective ledger track."""

    if not weekly:
        raise ValueError("decision-shadow requires at least one weekly forecast")
    spec = load_decision_shadow_spec(spec_path)
    gap_relatives, open_to_close_relatives = (
        split_safe_price_only_return_frames(prices)
    )
    close_to_close_returns = gap_relatives * open_to_close_relatives - 1.0
    signal_rows: list[dict[str, Any]] = []
    mapping = spec["probability_weight_mapping"]
    for source in weekly:
        forecast = _selected_model_forecast(
            source,
            forecast_model=forecast_model,
        )
        probabilities = forecast.get("probabilities", {})
        if not isinstance(probabilities, Mapping) or set(probabilities) != set(STATE_ORDER):
            raise ValueError("decision-shadow forecast probabilities are invalid")
        probability_values = np.asarray(
            [float(probabilities[state]) for state in STATE_ORDER], dtype=float
        )
        if (
            not np.isfinite(probability_values).all()
            or (probability_values < 0.0).any()
            or not math.isclose(
                float(probability_values.sum()),
                1.0,
                abs_tol=1e-6,
                rel_tol=0.0,
            )
        ):
            raise ValueError("decision-shadow probabilities must be finite and sum to one")
        probability_values = probability_values / float(probability_values.sum())
        normalized_probabilities = {
            state: float(probability_values[index])
            for index, state in enumerate(STATE_ORDER)
        }
        weights = {
            asset: sum(
                normalized_probabilities[state] * float(mapping[state][asset])
                for state in STATE_ORDER
            )
            for asset in ASSETS
        }
        signal_rows.append({"origin_date": str(source["date"]), **weights})
    signals = pd.DataFrame(signal_rows)
    if signals.empty:
        aligned = pd.DataFrame(columns=ASSETS, index=prices.index, dtype=float)
    else:
        price_index_by_date = {
            pd.Timestamp(index).date().isoformat(): index for index in prices.index
        }
        resolved_index = [
            price_index_by_date.get(value) for value in signals.pop("origin_date")
        ]
        if any(value is None for value in resolved_index):
            raise ValueError(
                "decision-shadow forecast origin is absent from the price panel"
            )
        signals.index = pd.DatetimeIndex(resolved_index)
        if signals.index.has_duplicates:
            raise ValueError("decision-shadow signal origins must be unique")
        aligned = signals.reindex(prices.index)

    # A signal finalized at weekly close t sets the target for week t+1's open.
    # Because the provider has no ex-date, weekly dividends are removed from the
    # adjusted total factor before it is split into the prior-position gap and
    # target-position open-to-close legs.  Every strategy therefore shares the
    # same split-safe, price-only accounting contract.
    dynamic_weights = aligned.shift(1)
    cost_rate = float(spec["cost"]["one_way_turnover_bps"]) / 10_000.0

    buy_hold_row = {
        asset: float(spec["benchmarks"]["buy_and_hold"][asset]) for asset in ASSETS
    }
    static_row = {
        asset: float(spec["benchmarks"]["static_60_40"][asset]) for asset in ASSETS
    }
    static_weights = pd.DataFrame(
        {asset: static_row[asset] for asset in ASSETS},
        index=prices.index,
        dtype=float,
    )
    buy_hold_weights = pd.DataFrame(
        {asset: buy_hold_row[asset] for asset in ASSETS},
        index=prices.index,
        dtype=float,
    )
    base_returns = (close_to_close_returns * static_weights).sum(
        axis=1, min_count=len(ASSETS)
    )
    vol_spec = spec["benchmarks"]["vol_target_60_40"]
    trailing_vol = (
        base_returns.rolling(
            int(vol_spec["lookback_weeks"]),
            min_periods=int(vol_spec["minimum_observations"]),
        )
        .std(ddof=1)
        * math.sqrt(float(spec["metrics"]["annualization_weeks"]))
    )
    scale = (
        float(vol_spec["annual_target"]) / trailing_vol.replace(0.0, np.nan)
    ).clip(upper=float(vol_spec["maximum_gross_exposure"]))
    vol_target_weights = static_weights.mul(scale, axis=0).shift(1)
    strategies = {
        "probability_shadow": dynamic_weights,
        "spy_buy_and_hold": buy_hold_weights,
        "static_60_40": static_weights,
        "vol_target_60_40": vol_target_weights,
    }
    dynamic_valid = dynamic_weights.dropna(how="any")
    vol_target_valid = vol_target_weights.dropna(how="any")
    if dynamic_valid.empty or vol_target_valid.empty:
        common_index = prices.index[:0]
    else:
        first_evaluation = max(dynamic_valid.index.min(), vol_target_valid.index.min())
        last_evaluation = min(dynamic_valid.index.max(), vol_target_valid.index.max())
        if first_evaluation > last_evaluation:
            common_index = prices.index[:0]
        else:
            common_index = prices.index[
                (prices.index >= first_evaluation) & (prices.index <= last_evaluation)
            ]
            valid_relatives = (
                np.isfinite(gap_relatives.reindex(common_index)).all(axis=1)
                & np.isfinite(
                    open_to_close_relatives.reindex(common_index)
                ).all(axis=1)
                & gap_relatives.reindex(common_index).gt(0.0).all(axis=1)
                & open_to_close_relatives.reindex(common_index)
                .gt(0.0)
                .all(axis=1)
            )
            if not bool(valid_relatives.all()):
                raise ValueError(
                    "decision-shadow price-only return decomposition is incomplete "
                    "in the evaluation window"
                )
    summaries: dict[str, Any] = {}
    annualization = float(spec["metrics"]["annualization_weeks"])
    risk_aversion = float(spec["metrics"]["certainty_equivalent_risk_aversion"])
    first_tradable_week = (
        None
        if dynamic_valid.empty
        else pd.Timestamp(dynamic_valid.index.min()).date().isoformat()
    )
    evaluation_start_week = (
        None
        if common_index.empty
        else pd.Timestamp(common_index.min()).date().isoformat()
    )
    evaluation_end_week = (
        None
        if common_index.empty
        else pd.Timestamp(common_index.max()).date().isoformat()
    )
    for name, weights in strategies.items():
        path = _run_self_financing_strategy(
            weights,
            gap_relatives,
            open_to_close_relatives,
            common_index,
            cost_rate=cost_rate,
            initial_allocation_costed_from_cash=bool(
                spec["cost"]["initial_allocation_costed_from_cash"]
            ),
            late_signal_policy=str(spec["execution"]["late_signal_policy"]),
        )
        net = path["net_returns"]
        gross = path["gross_returns"]
        turnover = path["turnover"]
        transaction_cost = path["transaction_cost"]
        assert isinstance(net, pd.Series)
        assert isinstance(gross, pd.Series)
        assert isinstance(turnover, pd.Series)
        assert isinstance(transaction_cost, pd.Series)
        net_metrics = _metrics(
            net,
            turnover,
            annualization=annualization,
            risk_aversion=risk_aversion,
        )
        gross_metrics = _metrics(
            gross,
            turnover,
            annualization=annualization,
            risk_aversion=risk_aversion,
        )
        summaries[name] = {
            **net_metrics,
            "gross_cumulative_return": gross_metrics["cumulative_return"],
            "transaction_cost_rate_sum": float(
                transaction_cost.reindex(net.dropna().index).sum()
            ),
            "transaction_cost_bps": float(spec["cost"]["one_way_turnover_bps"]),
        }
    latest_signal_origin = str(signal_rows[-1]["origin_date"])
    latest_target_weights = {
        asset: float(signal_rows[-1][asset]) for asset in ASSETS
    }
    minimum_weeks = int(spec["metrics"]["minimum_evaluation_weeks"])
    shadow_weeks = int(summaries["probability_shadow"]["weeks"])
    historical_status = "completed" if shadow_weeks >= minimum_weeks else "insufficient_history"
    current_signal = _current_signal_contract(
        weekly[-1],
        decision_at=decision_at,
        forecast_model=forecast_model,
    )
    body = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "role": "research_only_no_forecast_or_champion_effect",
        "spec": {
            "path": "config/decision-shadow-v2.json",
            "sha256": canonical_json_sha256_v1(spec),
            "spec_id": spec["spec_id"],
        },
        "execution_contract": dict(spec["execution"]),
        "current_signal": current_signal,
        "historical_reconstructed_shadow": {
            "status": historical_status,
            "evidence_track": "reconstructed_oos",
            "evidence_status": "historical_reconstructed_shadow",
            "first_tradable_week": first_tradable_week,
            "evaluation_start_week": evaluation_start_week,
            "evaluation_end_week": evaluation_end_week,
            "minimum_evaluation_weeks": minimum_weeks,
            "latest_target_weights": latest_target_weights,
            "allocation_policy": {
                "method": "probability_weighted_state_portfolios",
                "assets": list(ASSETS),
                "forecast_model": forecast_model,
                "latest_signal_origin": latest_signal_origin,
                "latest_target_weights": latest_target_weights,
            },
            "strategies": summaries,
        },
        "prospective_ledger": prospective_ledger_shadow_contract(
            prospective_ledger_summary
        ),
    }
    return body


__all__ = [
    "RESULT_SCHEMA_VERSION",
    "build_decision_shadow",
    "default_decision_shadow_spec_path",
    "load_decision_shadow_spec",
    "prospective_ledger_shadow_contract",
    "split_safe_price_only_return_frames",
]
