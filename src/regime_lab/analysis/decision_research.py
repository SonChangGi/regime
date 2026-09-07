"""Matched OOS alert budgets and economic confusion matrices for local review."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


def _threshold(
    scores: np.ndarray, events: np.ndarray, annual_budget: float
) -> float | None:
    """Maximise detected events under a frozen historical false-alarm budget."""
    maximum_false = annual_budget * len(scores) / 52.1775
    choices = []
    for value in np.unique(scores):
        alert = scores >= value
        false = int((alert & ~events).sum())
        if false <= maximum_false:
            choices.append((int((alert & events).sum()), -false, float(value)))
    if not choices or max(x[0] for x in choices) == 0:
        return None
    return max(choices)[2]


def _alert_metrics(
    events: np.ndarray, alerts: np.ndarray, returns: np.ndarray
) -> dict[str, Any]:
    n = len(events)
    true = int((events & alerts).sum())
    false = int((~events & alerts).sum())
    missed_loss = float(-np.minimum(returns[events & ~alerts], 0).sum())
    # This is a matched decision-loss score, not a self-financing NAV path.
    # Each alert is a one-week SPY-to-cash decision with two 10bp orders.
    utility = -alerts.astype(float) * returns - alerts.astype(float) * 0.002
    return {
        "n": n,
        "events": int(events.sum()),
        "alerts": int(alerts.sum()),
        "true_positive": true,
        "false_positive": false,
        "precision": true / int(alerts.sum()) if alerts.any() else None,
        "recall": true / int(events.sum()) if events.any() else None,
        "false_alarms_per_year": false / (n / 52.1775) if n else None,
        "missed_event_downside_sum": missed_loss,
        "mean_decision_utility_after_20bp_round_trip": (
            float(utility.mean()) if n else None
        ),
        "utility_basis": "isolated_one_week_price_only_SPY_to_zero_return_cash_decisions_not_portfolio_NAV",
    }


def build_decision_research_v2(
    predictions: pd.DataFrame,
    transition_predictions: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    forecast_model: str,
    selection_end: str,
    outcome_rows: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Use stored OOS predictions; never fit to post-selection outcomes.

    Alert threshold estimation is prequential inside selection and then frozen
    at selection_end. The two simple baselines only use completed origin prices.
    Optional outcome_rows are already-derived model-conditioned asset outcomes.
    """
    sample = (
        predictions.loc[predictions.model.eq(forecast_model)]
        .drop(columns=["p_change"], errors="ignore")
        .copy()
    )
    sample["origin_key"] = pd.to_datetime(sample.origin_date, utc=True).dt.strftime(
        "%Y-%m-%d"
    )
    sample["target_key"] = pd.to_datetime(sample.target_date, utc=True).dt.strftime(
        "%Y-%m-%d"
    )
    if sample.origin_key.duplicated().any():
        raise ValueError("decision research requires unique model origins")
    hazard = transition_predictions.loc[
        transition_predictions.model.eq("binary_xgboost")
        & transition_predictions.horizon.eq(1)
    ].copy()
    hazard["origin_key"] = pd.to_datetime(hazard.origin_date, utc=True).dt.strftime(
        "%Y-%m-%d"
    )
    hazard["target_key"] = pd.to_datetime(hazard.target_end, utc=True).dt.strftime(
        "%Y-%m-%d"
    )
    frame = sample.merge(
        hazard[["origin_key", "target_key", "p_change"]],
        on=["origin_key", "target_key"],
        validate="one_to_one",
    )
    if len(frame) != len(sample):
        raise ValueError("decision alert candidates must share exactly matched origins")
    index_dates = pd.Index([t.date().isoformat() for t in prices.index])
    if index_dates.has_duplicates:
        raise ValueError("decision research canonical dates are not unique")
    close = pd.Series(pd.to_numeric(prices.spy_close).to_numpy(), index=index_dates)
    volatility = (
        close.pct_change(fill_method=None).rolling(13, min_periods=13).std(ddof=1)
    )
    drawdown = 1 - close / close.rolling(26, min_periods=26).max()
    returns = pd.Series(
        (
            pd.to_numeric(prices.spy_raw_close) / pd.to_numeric(prices.spy_raw_open) - 1
        ).to_numpy(),
        index=index_dates,
    )
    frame["volatility"] = frame.origin_key.map(volatility)
    frame["drawdown"] = frame.origin_key.map(drawdown)
    frame["spy_return"] = frame.target_key.map(returns)
    required = ["p_change", "volatility", "drawdown", "spy_return"]
    if not np.isfinite(frame[required].to_numpy(float)).all():
        raise ValueError("decision research matched scores/returns are unavailable")
    rank = {"risk_on": 0, "transition": 1, "risk_off": 2}
    frame["all_departure"] = frame.actual.ne(frame.current_state)
    frame["risk_worsening"] = frame.actual.map(rank) > frame.current_state.map(rank)
    selection = frame.loc[frame.evaluation_split.eq("selection")].sort_values(
        "origin_key"
    )
    diagnostic = frame.loc[frame.evaluation_split.eq("holdout")].sort_values(
        "origin_key"
    )
    cutoff = pd.Timestamp(selection_end).date().isoformat()
    if (
        selection.empty
        or diagnostic.empty
        or selection.target_key.max() > cutoff
        or diagnostic.origin_key.min() <= cutoff
    ):
        raise ValueError("decision research split crosses selection boundary")
    rows = []
    for target in ("all_departure", "risk_worsening"):
        for name, column in (
            ("binary_xgboost", "p_change"),
            ("trailing_13w_volatility", "volatility"),
            ("trailing_26w_drawdown", "drawdown"),
        ):
            for budget in (4, 8, 12):
                train_scores = selection[column].to_numpy(float)
                test_scores = diagnostic[column].to_numpy(float)
                # No worsening is possible from the highest-risk state.
                if target == "risk_worsening":
                    train_scores = np.where(
                        selection.current_state.eq("risk_off"), -1, train_scores
                    )
                    test_scores = np.where(
                        diagnostic.current_state.eq("risk_off"), -1, test_scores
                    )
                events = selection[target].to_numpy(bool)
                threshold = _threshold(train_scores, events, budget)
                alerts = (
                    test_scores >= threshold
                    if threshold is not None
                    else np.zeros(len(test_scores), bool)
                )
                prequential_alerts = []
                prequential_events = []
                prequential_returns = []
                for k, row in enumerate(selection.itertuples(index=False)):
                    history = selection.target_key.to_numpy() < row.origin_key
                    if int(history.sum()) < 52:
                        continue
                    local = _threshold(train_scores[history], events[history], budget)
                    prequential_alerts.append(
                        local is not None and train_scores[k] >= local
                    )
                    prequential_events.append(bool(getattr(row, target)))
                    prequential_returns.append(float(row.spy_return))
                rows.append(
                    {
                        "target": target,
                        "score": name,
                        "annual_false_alarm_budget": budget,
                        "frozen_threshold": threshold,
                        "threshold_status": (
                            "no_historically_useful_threshold"
                            if threshold is None
                            else "frozen_selection_only"
                        ),
                        "selection_n": len(selection),
                        "selection_false_alarms_per_year": (
                            float(
                                ((train_scores >= threshold) & ~events).sum()
                                / (len(events) / 52.1775)
                            )
                            if threshold is not None
                            else 0.0
                        ),
                        "prequential_selection": _alert_metrics(
                            np.asarray(prequential_events, bool),
                            np.asarray(prequential_alerts, bool),
                            np.asarray(prequential_returns, float),
                        ),
                        "retrospective_diagnostic": _alert_metrics(
                            diagnostic[target].to_numpy(bool),
                            alerts,
                            diagnostic.spy_return.to_numpy(float),
                        ),
                        "weekly_alerts": [
                            {"origin": str(origin), "alert": bool(value)}
                            for origin, value in zip(diagnostic.origin_key, alerts)
                        ],
                    }
                )
    matrix = []
    diagnostic = diagnostic.copy()
    if outcome_rows is not None:
        outcomes = outcome_rows.loc[
            outcome_rows.conditioning_model.eq(forecast_model)
            & outcome_rows.asset.eq("SPY")
            & outcome_rows.horizon_weeks.eq(1)
        ].copy()
        outcomes["origin_key"] = pd.to_datetime(
            outcomes.origin_date, utc=True
        ).dt.strftime("%Y-%m-%d")
        diagnostic = diagnostic.merge(
            outcomes[["origin_key", "forward_return"]],
            on="origin_key",
            validate="one_to_one",
        )
        value_column = "forward_return"
        if len(diagnostic) != len(frame.loc[frame.evaluation_split.eq("holdout")]):
            raise ValueError("economic loss matrix lost matched diagnostic origins")
    else:
        value_column = "spy_return"
    for (predicted, actual), group in diagnostic.groupby(["predicted", "actual"]):
        values = group[value_column].to_numpy(float)
        matrix.append(
            {
                "predicted": str(predicted),
                "actual": str(actual),
                "n": len(group),
                "mean_return": float(values.mean()),
                "negative_return_sum": float(-np.minimum(values, 0).sum()),
                "positive_rate": float((values > 0).mean()),
                "worst_return": float(values.min()),
                "worsening_events": int(group.risk_worsening.sum()),
            }
        )
    return {
        "schema_version": "regime-decision-research/2",
        "role": "research_only_no_forecast_effect",
        "evidence_track": "reconstructed_oos",
        "selection_end": selection_end,
        "diagnostic_origins": len(diagnostic),
        "threshold_protocol": "prequential_selection_then_frozen_threshold_holdout",
        "comparison_contract": "same_origin_target_week_open_to_close",
        "alert_budgets": rows,
        "forecast_actual_loss_matrix": {
            "asset": "SPY",
            "holding_weeks": 1,
            "return_basis": (
                "adjusted_derived_outcome"
                if outcome_rows is not None
                else "price_only_raw_open_to_close"
            ),
            "rows": matrix,
        },
        "automatic_promotion_eligible": False,
    }
