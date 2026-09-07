import numpy as np
import pandas as pd

from regime_lab.analysis.label_sensitivity import run_label_sensitivity


def _prices():
    rng = np.random.default_rng(730)
    return pd.DataFrame(
        {"spy_close": 100 * np.exp(np.cumsum(rng.normal(0.002, 0.025, 150)))},
        index=pd.date_range("2006-01-06", periods=150, freq="W-FRI"),
    )


def test_entire_grid_executes_and_post_selection_changes_cannot_affect_results():
    prices = _prices()
    cutoff = prices.index[125]
    result = run_label_sensitivity(
        prices, fit_weeks=80, selection_end=cutoff, evaluate_model_ranks=False
    )
    changed = prices.copy()
    changed.loc[changed.index >= cutoff, "spy_close"] *= np.arange(1, 26)
    repeated = run_label_sensitivity(
        changed, fit_weeks=80, selection_end=cutoff, evaluate_model_ranks=False
    )
    assert result.summary == repeated.summary
    assert result.summary["execution_summary"]["evaluated_spec_count"] == 243
    assert len(result.spec_metrics) == 244
    assert (
        result.spec_metrics[
            ["occupancy_risk_on", "occupancy_transition", "occupancy_risk_off"]
        ]
        .sum(axis=1)
        .sub(1)
        .abs()
        .max()
        < 1e-10
    )
    assert (
        result.summary["execution_summary"]["model_rank_robustness"]["status"]
        == "not_executed"
    )


def test_fixed_representatives_use_common_origins_and_purged_targets():
    prices = _prices()
    result = run_label_sensitivity(
        prices, fit_weeks=80, selection_end=prices.index[105], evaluate_model_ranks=True
    )
    rows = result.representative_oos
    assert rows["spec_id"].nunique() == 4
    assert rows["model"].nunique() == 3
    assert (
        pd.to_datetime(rows["last_train_target"]) < pd.to_datetime(rows["origin_date"])
    ).all()
    assert rows.groupby(["spec_id", "model"])["origin_date"].apply(tuple).nunique() == 1
    assert result.summary["status"] == "evaluated"
