"""Preregistered probability-to-exposure decision shadow.

This module is intentionally downstream of the official probability forecast.
It cannot select a model, alter a probability, or promote a champion.  Its
historical reconstruction and prospective ledger evaluation are separately
labelled evidence tracks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from regime_lab.integrity import canonical_json_sha256_v1


STATE_ORDER = ("risk_on", "transition", "risk_off")
SPEC_SCHEMA_VERSION = "regime-decision-shadow-spec/1"
RESULT_SCHEMA_VERSION = "regime-prospective-decision-shadow/1"


def default_decision_shadow_spec_path() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "decision-shadow.json"


def load_decision_shadow_spec(path: str | Path | None = None) -> dict[str, Any]:
    selected = default_decision_shadow_spec_path() if path is None else Path(path)
    document = json.loads(selected.read_text(encoding="utf-8"))
    if document.get("schema_version") != SPEC_SCHEMA_VERSION:
        raise ValueError("decision-shadow spec schema is invalid")
    if document.get("role") != "research_only_no_forecast_or_champion_effect":
        raise ValueError("decision-shadow role is invalid")
    if document.get("assets") != ["SPY", "TLT"]:
        raise ValueError("decision-shadow assets are invalid")
    mapping = document.get("probability_weight_mapping")
    if not isinstance(mapping, Mapping) or set(mapping) != set(STATE_ORDER):
        raise ValueError("decision-shadow probability mapping is invalid")
    for state in STATE_ORDER:
        weights = mapping[state]
        if not isinstance(weights, Mapping) or set(weights) != {"SPY", "TLT"}:
            raise ValueError("decision-shadow state weights are invalid")
        values = [float(weights[asset]) for asset in ("SPY", "TLT")]
        if any(value < 0.0 or value > 1.0 for value in values) or not math.isclose(
            sum(values), 1.0, abs_tol=1e-12
        ):
            raise ValueError("decision-shadow state weights must be long-only and sum to one")
    return document


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
    maximum_drawdown = float((wealth / wealth.cummax() - 1.0).min())
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
        "annualized_turnover": float(turnover.reindex(clean.index).fillna(0.0).mean() * annualization),
    }


def build_decision_shadow(
    weekly: Sequence[Mapping[str, Any]],
    prices: pd.DataFrame,
    *,
    spec_path: str | Path | None = None,
    prospective_ledger_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate reconstructed forecasts and reserve a prospective ledger track."""

    spec = load_decision_shadow_spec(spec_path)
    price_columns = {asset: f"{asset.lower()}_close" for asset in ("SPY", "TLT")}
    missing = [column for column in price_columns.values() if column not in prices]
    if missing:
        raise ValueError(f"decision-shadow prices are missing columns: {missing}")
    returns = prices[[price_columns["SPY"], price_columns["TLT"]]].pct_change()
    returns.columns = ["SPY", "TLT"]
    signal_rows: list[dict[str, Any]] = []
    mapping = spec["probability_weight_mapping"]
    for source in weekly:
        probabilities = source.get("next_week", {}).get("probabilities", {})
        if not isinstance(probabilities, Mapping) or set(probabilities) != set(STATE_ORDER):
            continue
        weights = {
            asset: sum(
                float(probabilities[state]) * float(mapping[state][asset])
                for state in STATE_ORDER
            )
            for asset in ("SPY", "TLT")
        }
        signal_rows.append({"origin_date": str(source["date"]), **weights})
    signals = pd.DataFrame(signal_rows)
    if signals.empty:
        aligned = pd.DataFrame(columns=["SPY", "TLT"], index=prices.index)
    else:
        price_index_by_date = {
            pd.Timestamp(index).date().isoformat(): index for index in prices.index
        }
        resolved_index = [
            price_index_by_date.get(value) for value in signals.pop("origin_date")
        ]
        signals.index = pd.DatetimeIndex(resolved_index)
        signals = signals.loc[signals.index.notna()]
        aligned = signals.reindex(prices.index)

    # A signal observed at t is first executable at t+1 close and earns the
    # subsequent t+1 -> t+2 weekly return.  This avoids same-close execution.
    dynamic_weights = aligned.shift(2)
    cost_rate = float(spec["cost"]["one_way_turnover_bps"]) / 10_000.0

    static_weights = pd.DataFrame(
        {"SPY": 0.6, "TLT": 0.4}, index=prices.index, dtype=float
    )
    buy_hold_weights = pd.DataFrame(
        {"SPY": 1.0, "TLT": 0.0}, index=prices.index, dtype=float
    )
    base_returns = (returns * static_weights).sum(axis=1, min_count=2)
    vol_spec = spec["benchmarks"]["vol_target_60_40"]
    trailing_vol = (
        base_returns.rolling(
            int(vol_spec["lookback_weeks"]),
            min_periods=int(vol_spec["minimum_observations"]),
        )
        .std(ddof=1)
        .shift(1)
        * math.sqrt(float(spec["metrics"]["annualization_weeks"]))
    )
    scale = (
        float(vol_spec["annual_target"]) / trailing_vol.replace(0.0, np.nan)
    ).clip(upper=float(vol_spec["maximum_gross_exposure"]))
    vol_target_weights = static_weights.mul(scale, axis=0)
    strategies = {
        "probability_shadow": dynamic_weights,
        "spy_buy_and_hold": buy_hold_weights,
        "static_60_40": static_weights,
        "vol_target_60_40": vol_target_weights,
    }
    common_index = prices.index[
        returns.notna().all(axis=1)
        & dynamic_weights.notna().all(axis=1)
        & vol_target_weights.notna().all(axis=1)
    ]
    summaries: dict[str, Any] = {}
    annualization = float(spec["metrics"]["annualization_weeks"])
    risk_aversion = float(spec["metrics"]["certainty_equivalent_risk_aversion"])
    first_trade_at: str | None = None
    for name, weights in strategies.items():
        weights = weights.reindex(common_index)
        tradable = weights.dropna(how="any")
        turnover = weights.diff().abs().sum(axis=1, min_count=1)
        if bool(spec["cost"]["initial_allocation_costed_from_cash"]):
            first_valid = tradable.index.min() if not tradable.empty else None
            if first_valid is not None:
                turnover.loc[first_valid] = float(weights.loc[first_valid].abs().sum())
        gross = (returns * weights).sum(axis=1, min_count=2)
        transaction_cost = turnover.fillna(0.0) * cost_rate
        net = gross - transaction_cost
        if name == "probability_shadow" and not aligned.dropna(how="any").empty:
            first_signal_position = prices.index.get_loc(
                aligned.dropna(how="any").index.min()
            )
            if first_signal_position + 1 < len(prices.index):
                first_trade_at = pd.Timestamp(
                    prices.index[first_signal_position + 1]
                ).isoformat()
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
            "total_transaction_cost": float(
                transaction_cost.reindex(net.dropna().index).sum()
            ),
            "transaction_cost_bps": float(spec["cost"]["one_way_turnover_bps"]),
        }
    minimum_weeks = int(spec["metrics"]["minimum_evaluation_weeks"])
    shadow_weeks = int(summaries["probability_shadow"]["weeks"])
    historical_status = "completed" if shadow_weeks >= minimum_weeks else "insufficient_history"
    ledger = dict(prospective_ledger_summary or {})
    prospective_count = ledger.get("entry_count")
    prospective_status = (
        "awaiting_realized_targets"
        if not isinstance(prospective_count, int) or prospective_count == 0
        else "ledger_recorded_outcomes_pending"
    )
    body = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "role": "research_only_no_forecast_or_champion_effect",
        "spec": {
            "path": "config/decision-shadow.json",
            "sha256": canonical_json_sha256_v1(spec),
            "spec_id": spec["spec_id"],
        },
        "execution_contract": dict(spec["execution"]),
        "historical_reconstructed_shadow": {
            "status": historical_status,
            "evidence_track": "reconstructed_oos",
            "evidence_status": "historical_reconstructed_shadow",
            "first_tradable_at": first_trade_at,
            "minimum_evaluation_weeks": minimum_weeks,
            "strategies": summaries,
        },
        "prospective_ledger": {
            "status": prospective_status,
            "evidence_track": "operational_oos",
            "ledger_entry_count": prospective_count if isinstance(prospective_count, int) else 0,
            "realized_evaluation_count": 0,
            "affects_official_forecast": False,
            "affects_champion_selection": False,
        },
    }
    return body


__all__ = [
    "RESULT_SCHEMA_VERSION",
    "build_decision_shadow",
    "default_decision_shadow_spec_path",
    "load_decision_shadow_spec",
]
