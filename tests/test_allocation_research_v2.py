from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from regime_lab.allocation.research_v2 import (
    _fit_decoder,
    _forward_open_return,
    _probability_diagnostics,
    _selection_sample,
    _simulate,
    build_allocation_shadow_v2,
)
from regime_lab.allocation.shadow import split_safe_asset_return_frames
from test_allocation_shadow import _fixture


def _inputs():
    weekly, prices, states, signal = _fixture()
    names = ("causal_dynamic_ensemble", "markov")
    rows = []
    for k in range(300, 676):
        actual = states.iloc[k + 1]
        for model in names:
            p = {s: 0.1 for s in ("risk_on", "transition", "risk_off")}
            p[actual] = 0.8 if model == names[0] else 0.6
            total = sum(p.values())
            p = {s: v / total for s, v in p.items()}
            rows.append(
                {
                    "origin_date": prices.index[k],
                    "target_date": prices.index[k + 1],
                    "actual": actual,
                    "model": model,
                    "evaluation_split": "selection",
                    **{f"p_{s}": v for s, v in p.items()},
                }
            )
    return weekly, prices, states, signal, pd.DataFrame(rows)


def _spec():
    path = Path(__file__).resolve().parents[1] / "config/allocation-shadow-v2.json"
    return {**json.loads(path.read_text()), "bootstrap_resamples": 19}


def test_open_to_open_payoff_includes_exit_gap_not_exit_intraday():
    gaps = pd.DataFrame({"SPY": [1, 1.1, 1.2], "TLT": [1, 1, 1]})
    intra = pd.DataFrame({"SPY": [1.03, 1.04, 99.0], "TLT": [1, 1, 1]})
    got = _forward_open_return(gaps, intra, 0, 2)
    assert got[0] == pytest.approx(1.03 * 1.1 * 1.04 * 1.2 - 1)
    assert got[1] == 0


def test_decoder_never_uses_unmatured_or_postselection_returns():
    _, prices, _, _, pred = _inputs()
    spec = _spec()
    cutoff = pd.Timestamp("2022-12-31", tz="UTC")
    sample = _selection_sample(pred, "causal_dynamic_ensemble", cutoff)
    gap, intra = split_safe_asset_return_frames(prices, ("SPY", "TLT"))
    doc, beta, draws = _fit_decoder(sample, prices, gap, intra, cutoff, spec)
    changed_gap = gap.copy()
    changed_intra = intra.copy()
    changed_gap.loc[changed_gap.index > cutoff] = 11
    changed_intra.loc[changed_intra.index > cutoff] = 0.1
    other, other_beta, other_draws = _fit_decoder(
        sample, prices, changed_gap, changed_intra, cutoff, spec
    )
    np.testing.assert_array_equal(beta, other_beta)
    np.testing.assert_array_equal(draws, other_draws)
    assert doc == other
    assert all(
        pd.Timestamp(r["last_training_exit"]) < pd.Timestamp(r["origin"])
        for r in doc["validation"]["rows"]
    )


def test_core_risk_rebalances_even_when_every_alpha_is_rejected():
    index = pd.date_range("2024-01-05", periods=12, freq="W-FRI", tz="UTC")
    gap = pd.DataFrame({"SPY": 1.0, "TLT": 1.0}, index=index)
    intra = pd.DataFrame({"SPY": 1.1, "TLT": 0.99}, index=index)
    cash = pd.Series(1.0, index=index)
    signals = [
        {
            "eligible_tilt": 0.0,
            "expected_relative_return": 0.0,
            "alpha_screen": "mean_payoff_below_cost",
            "risk_match_scale": 1.0,
        }
        for _ in index
    ]
    core = _simulate(index, gap, intra, cash, signals, _spec(), 10, "core_60_40")
    alpha = _simulate(index, gap, intra, cash, signals, _spec(), 10, "regime_alpha")
    assert any(r["action"] == "core_risk_rebalance" for r in core)
    assert [r["net_return"] for r in core] == [r["net_return"] for r in alpha]
    assert max(r["one_way_turnover"] for r in alpha[1:]) <= 0.1 + 1e-12


def test_probability_skill_is_separate_from_calibration():
    *_, pred = _inputs()
    cutoff = pd.Timestamp("2022-12-31", tz="UTC")
    a = _selection_sample(pred, "causal_dynamic_ensemble", cutoff)
    b = _selection_sample(pred, "markov", cutoff)
    result = _probability_diagnostics(a, b, _spec())
    assert result["benchmark"] == "markov"
    assert result["skill_gate_passed"]
    assert result["calibration"]["status"] == "diagnostic_not_certification"
    assert sum(r["n"] for r in result["calibration"]["bins"]) == 3 * len(a)


def test_versioned_policy_runs_without_altering_v1_inputs(tmp_path):
    weekly, prices, states, signal, pred = _inputs()
    before = deepcopy(weekly)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(_spec()))
    result = build_allocation_shadow_v2(
        weekly,
        prices,
        pred,
        forecast_model="causal_dynamic_ensemble",
        selection_end="2022-12-31",
        current_signal=signal,
        spec_path=path,
    )
    assert weekly == before
    assert result["affects_issued_ledger"] is False
    assert result["performance"]["automatic_promotion_eligible"] is False
    assert result["performance"]["weeks"] == 99
    assert result["execution_contract"]["alpha_cost_threshold_relative_return"] == 0.004
    assert all(3 <= r["holding_weeks"] <= 5 for r in result["sector_rotation"]["rows"])
    sector = result["sector_rotation"]
    assert all(
        pd.Timestamp(r["exit_week"]) <= pd.Timestamp("2022-12-31")
        for r in sector["rows"]
    )
    assert sector["holdout"]["n_months"] > 0
    assert all(
        pd.Timestamp(r["origin"]) > pd.Timestamp("2022-12-31")
        for r in sector["holdout"]["rows"]
    )
    assert all(3 <= r["holding_weeks"] <= 5 for r in sector["holdout"]["rows"])
    json.dumps(result, allow_nan=False)
