import numpy as np
import pandas as pd

from regime_lab.analysis.decision_research import _threshold, build_decision_research_v2


def _frames():
    dates = pd.date_range("2017-01-06", periods=390, freq="W-FRI", tz="UTC")
    k = np.arange(len(dates))
    close = 100 * np.exp(0.002 * k + 0.02 * np.sin(k / 4))
    prices = pd.DataFrame(
        {
            "spy_close": close,
            "spy_raw_open": close / np.exp(0.001 + 0.01 * np.sin(k / 3)),
            "spy_raw_close": close,
        },
        index=dates,
    )
    rows = []
    hazards = []
    for i in range(30, len(dates) - 1):
        actual = "transition" if i % 7 == 0 else "risk_on"
        split = (
            "selection"
            if dates[i + 1] < pd.Timestamp("2023-01-01", tz="UTC")
            else "holdout"
        )
        rows.append(
            {
                "origin_date": dates[i],
                "target_date": dates[i + 1],
                "model": "model",
                "actual": actual,
                "current_state": "risk_on",
                "predicted": "risk_on",
                "evaluation_split": split,
                "p_change": 0.1,
            }
        )
        hazards.append(
            {
                "origin_date": dates[i],
                "target_end": dates[i + 1],
                "model": "binary_xgboost",
                "horizon": 1,
                "p_change": 0.8 if i % 7 == 0 else 0.2,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(hazards), prices


def test_threshold_obeys_selection_budget():
    scores = np.arange(104) / 104
    events = np.arange(104) % 4 == 0
    threshold = _threshold(scores, events, 4)
    assert (
        threshold is None
        or ((scores >= threshold) & ~events).sum() / (104 / 52.1775) <= 4
    )


def test_future_actuals_cannot_change_frozen_alert_thresholds():
    pred, hazard, prices = _frames()
    # First post-2023 origin is deliberately omitted because its origin still
    # belongs to 2022; match the project's purged selection/holdout boundary.
    pred = pred.loc[
        ~(
            pred.evaluation_split.eq("holdout")
            & (pred.origin_date < pd.Timestamp("2023-01-01", tz="UTC"))
        )
    ]
    a = build_decision_research_v2(
        pred, hazard, prices, forecast_model="model", selection_end="2023-01-01"
    )
    changed = pred.copy()
    changed.loc[changed.evaluation_split.eq("holdout"), "actual"] = "risk_off"
    b = build_decision_research_v2(
        changed, hazard, prices, forecast_model="model", selection_end="2023-01-01"
    )
    assert [r["frozen_threshold"] for r in a["alert_budgets"]] == [
        r["frozen_threshold"] for r in b["alert_budgets"]
    ]
    assert len(a["alert_budgets"]) == 18
    assert {r["target"] for r in a["alert_budgets"]} == {
        "all_departure",
        "risk_worsening",
    }
    assert (
        sum(r["n"] for r in a["forecast_actual_loss_matrix"]["rows"])
        == a["diagnostic_origins"]
    )
