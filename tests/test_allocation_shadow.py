from __future__ import annotations

from copy import deepcopy
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from regime_lab.allocation.shadow import (
    _scheduled_nyse_entry_at,
    _trade_decision,
    _turnover,
    build_allocation_shadow_candidate,
    load_allocation_shadow_spec,
    rebase_allocation_candidate_intent,
)
from regime_lab.contract_v5 import V5ContractError, _validate_allocation_candidate


FORECAST_MODEL = "causal_dynamic_ensemble"
SECTORS = ("XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY")


def _fixture() -> tuple[list[dict[str, object]], pd.DataFrame, pd.Series, dict[str, object]]:
    index = pd.date_range("2010-01-08", periods=780, freq="W-FRI", tz="UTC")
    step = np.arange(len(index), dtype=float)
    prices = pd.DataFrame(index=index)
    asset_names = ("SPY", "TLT", *SECTORS)
    for offset, asset in enumerate(asset_names):
        weekly_return = (
            0.0012
            + 0.00008 * offset
            + 0.006 * np.sin(step / (11.0 + offset / 4.0) + offset)
        )
        close = 100.0 * np.exp(np.cumsum(weekly_return))
        open_price = close / np.exp(
            0.0004 + 0.002 * np.sin(step / 8.0 + offset)
        )
        prefix = asset.lower()
        prices[f"{prefix}_close"] = close
        prices[f"{prefix}_raw_open"] = open_price
        prices[f"{prefix}_raw_close"] = close
        prices[f"{prefix}_dividend_amount"] = 0.0
    prices["dgs3mo"] = 3.0 + 0.5 * np.sin(step / 40.0)
    states = pd.Series(
        np.asarray(("risk_on", "transition", "risk_off"), dtype=object)[
            (step.astype(int) // 17) % 3
        ],
        index=index,
        dtype=object,
    )
    weekly: list[dict[str, object]] = []
    for position in range(680, len(index)):
        origin = index[position]
        phase = position % 3
        probabilities = (
            {"risk_on": 0.72, "transition": 0.2, "risk_off": 0.08}
            if phase == 0
            else {"risk_on": 0.12, "transition": 0.76, "risk_off": 0.12}
            if phase == 1
            else {"risk_on": 0.08, "transition": 0.2, "risk_off": 0.72}
        )
        weekly.append(
            {
                "date": origin.date().isoformat(),
                "model_forecasts": [
                    {
                        "model": FORECAST_MODEL,
                        "date": (origin + timedelta(days=7)).date().isoformat(),
                        "probabilities": probabilities,
                    }
                ],
            }
        )
    last = weekly[-1]
    signal = {
        "origin_date": last["date"],
        "target_week": last["model_forecasts"][0]["date"],
        "scheduled_entry_at": _scheduled_nyse_entry_at(
            last["model_forecasts"][0]["date"]
        ).isoformat(),
        "decision_at": f"{last['date']}T20:00:00+00:00",
        "forecast_model": FORECAST_MODEL,
        "status": "scheduled",
        "action": "trade_at_scheduled_open",
    }
    return weekly, prices, states, signal


def _build() -> dict[str, object]:
    weekly, prices, states, signal = _fixture()
    return build_allocation_shadow_candidate(
        weekly,
        prices,
        states,
        forecast_model=FORECAST_MODEL,
        selection_end="2022-12-31",
        current_signal=signal,
        calibration_evidence={
            "selection_n_predictions": 365,
            "selection_log_loss": 0.5,
            "benchmark_selection_log_loss": 0.8,
        },
    )


def test_allocation_candidate_freezes_selection_decoder_and_bounded_paths() -> None:
    candidate = _build()

    _validate_allocation_candidate(
        candidate,
        context="payload.research.prospective_decision_shadow.allocation_candidate",
    )

    assert candidate["schema_version"] == "regime-allocation-shadow-candidate/1"
    assert candidate["selection"]["end"] == "2022-12-31"
    assert candidate["selection"]["confidence_multiplier"] == 0.5
    assert {
        row["status"] for row in candidate["selection"]["confidence_candidates"]
    } == {"not_evaluated_no_pre_2023_forecast_probabilities"}
    assert candidate["sector_rotation"]["momentum_selection"]["selected_id"] == (
        "equal_weight_26_to_4_and_52_to_4_week"
    )
    assert len(candidate["performance_path"]) == 99
    assert set(candidate["performance"]["strategies"]) == {
        "realistic_60_40",
        "spy_buy_and_hold",
        "regime_only",
        "momentum_only",
        "combined",
    }
    for row in candidate["performance_path"]:
        assert set(row["strategies"]) == set(candidate["performance"]["strategies"])
        for values in row["strategies"].values():
            assert values["gross_wealth"] > 0.0
            assert values["net_wealth"] > 0.0
            assert values["one_way_turnover"] <= 1.0 + 1e-12
    recurring = candidate["performance_path"][1:]
    assert all(
        values["one_way_turnover"] <= 0.1 + 1e-12
        for row in recurring
        for values in row["strategies"].values()
    )


def test_allocation_contract_binds_policy_to_every_economic_gate() -> None:
    candidate = _build()
    gate = candidate["performance"]["economic_gate"]
    assert "calibration_skill_passed" in gate

    selected = deepcopy(candidate)
    selected_gate = selected["performance"]["economic_gate"]
    for field, value in selected_gate.items():
        if isinstance(value, bool):
            selected_gate[field] = True
    selected["policy_status"] = "candidate_selected"
    selected["recommended_target"] = "combined"
    selected["current_intent"]["recommended"]["policy"] = "combined"
    selected["sector_rotation"]["policy_status"] = "candidate_selected"
    selected["sector_rotation"]["selected_strategy"] = "combined"
    _validate_allocation_candidate(selected, context="candidate")

    selected_gate["20bps_ce_improved"] = False
    with pytest.raises(V5ContractError, match="differs from economic gate"):
        _validate_allocation_candidate(selected, context="candidate")


def test_allocation_contract_rejects_path_accounting_tampering() -> None:
    candidate = _build()

    broken = deepcopy(candidate)
    broken["performance_path"][0]["strategies"]["combined"]["gross_wealth"] += 0.01
    with pytest.raises(V5ContractError, match="wealth recursion is inconsistent"):
        _validate_allocation_candidate(broken, context="candidate")

    broken = deepcopy(candidate)
    broken["performance_path"][0]["strategies"]["combined"]["net_return"] += 0.01
    with pytest.raises(V5ContractError, match="net return is inconsistent"):
        _validate_allocation_candidate(broken, context="candidate")

    broken = deepcopy(candidate)
    broken["performance_path"][0]["strategies"]["combined"]["drawdown"] -= 0.01
    with pytest.raises(V5ContractError, match="drawdown is inconsistent"):
        _validate_allocation_candidate(broken, context="candidate")

    broken = deepcopy(candidate)
    broken["performance_path"][0]["strategies"]["combined"][
        "transaction_cost_rate"
    ] += 0.001
    with pytest.raises(V5ContractError, match="transaction cost is inconsistent"):
        _validate_allocation_candidate(broken, context="candidate")

    broken = deepcopy(candidate)
    broken["current_intent"]["cost"]["bps_per_traded_notional"] = 20.0
    with pytest.raises(V5ContractError, match="primary cost is invalid"):
        _validate_allocation_candidate(broken, context="candidate")


def test_allocation_contract_rejects_trading_hold_and_sector_policy_mismatch() -> None:
    candidate = _build()

    broken = deepcopy(candidate)
    broken["performance_path"][0]["strategies"]["combined"]["action"] = "no_trade"
    with pytest.raises(V5ContractError, match="hold has a trade"):
        _validate_allocation_candidate(broken, context="candidate")

    broken = deepcopy(candidate)
    broken["sector_rotation"]["policy_status"] = (
        "candidate_selected"
        if candidate["policy_status"] == "baseline_preferred"
        else "baseline_preferred"
    )
    with pytest.raises(V5ContractError, match="sector_rotation policy is inconsistent"):
        _validate_allocation_candidate(broken, context="candidate")


def test_sector_contract_enforces_seasoning_caps_and_composite_score() -> None:
    spec = load_allocation_shadow_spec()
    assert spec["sector_rotation"]["seasoning_weeks"] == 104
    assert spec["sector_rotation"]["top_n"] == 3
    assert spec["sector_rotation"]["maximum_total_asset_weight"] == 0.15
    assert spec["sector_rotation"]["maximum_equity_fraction"] == 0.25
    assert spec["sector_rotation"]["maximum_symbol_weight"] == 0.05
    assert spec["sector_rotation"]["momentum_combination"] == {
        "26_to_4_week": 0.5,
        "52_to_4_week": 0.5,
    }
    assert spec["sector_rotation"]["combined_score_weights"] == {
        "momentum": 0.8,
        "regime": 0.2,
    }

    candidate = _build()
    ranking = candidate["sector_rotation"]["ranking"]
    selected = [row for row in ranking if row["selected"]]
    assert len(selected) <= 3
    assert sum(row["target_weight"] for row in selected) <= 0.15 + 1e-12
    assert all(row["target_weight"] <= 0.05 + 1e-12 for row in selected)


def test_current_intent_is_self_financing_and_uses_same_execution_gate() -> None:
    candidate = _build()
    intent = candidate["current_intent"]

    assert intent["prior"]["basis"] == "reconstructed_strategy_close"
    for field in ("prior", "recommended", "target", "shadow_target"):
        block = intent[field]
        assert np.isclose(sum(block["weights"].values()) + block["cash"], 1.0)
    assert intent["forecast"]["target_week"] == intent["timing"]["target_week"]
    assert intent["order_delta"]["one_way_turnover"] <= 0.1 + 1e-12
    assert intent["cost"]["estimated_rate"] == (
        intent["cost"]["bps_per_traded_notional"]
        / 10_000.0
        * intent["order_delta"]["full_l1_turnover"]
    )


def test_current_intent_rebases_to_prospective_cash_genesis_before_freeze() -> None:
    candidate = _build()
    reconstructed = candidate["current_intent"]
    assert reconstructed["prior"]["basis"] == "reconstructed_strategy_close"

    rebased = rebase_allocation_candidate_intent(
        candidate,
        prior_weights=None,
        prior_cash=1.0,
        prior_basis="prospective_cash_genesis",
    )
    intent = rebased["current_intent"]
    assert intent["prior"] == {
        "basis": "prospective_cash_genesis",
        "weights": {"SPY": 0.0, "TLT": 0.0},
        "cash": 1.0,
    }
    assert intent["recommended"]["action"] == "initial_allocate"
    assert intent["order_delta"]["one_way_turnover"] == pytest.approx(1.0)
    assert intent["target"]["weights"] != reconstructed["target"]["weights"]


def test_turnover_definitions_cost_risky_orders_without_treating_cash_as_order() -> None:
    prior = np.asarray([0.0, 0.0])
    target = np.asarray([0.6, 0.4])
    one_way, full_l1 = _turnover(prior, 1.0, target, 0.0)
    assert one_way == 1.0
    assert full_l1 == 1.0

    execution = load_allocation_shadow_spec()["execution"]
    decision = _trade_decision(
        np.asarray([0.6, 0.4]),
        0.0,
        np.asarray([0.3, 0.7]),
        execution=execution,
        cost_rate=0.001,
        expected_returns=np.asarray([-0.02, 0.02]),
        initial=False,
    )
    assert decision.one_way <= 0.1 + 1e-12
    assert decision.full_l1 <= 0.2 + 1e-12


def test_monthly_sector_clock_uses_target_week_entry_month() -> None:
    # The 2026-07-31 signal targets the week ending Aug 7 and enters Aug 3.
    entry = _scheduled_nyse_entry_at("2026-08-07")
    assert (entry.year, entry.month, entry.day) == (2026, 8, 3)


def test_payoff_decoder_conditions_on_target_state_not_origin_state() -> None:
    weekly, prices, _, signal = _fixture()
    alternating = np.asarray(
        ["risk_on" if index % 2 == 0 else "risk_off" for index in range(len(prices))],
        dtype=object,
    )
    states = pd.Series(alternating, index=prices.index, dtype=object)
    target_return = np.where(alternating == "risk_on", 0.02, -0.02)
    prices["spy_raw_open"] = prices["spy_raw_close"] / (1.0 + target_return)
    prices["tlt_raw_open"] = prices["tlt_raw_close"]

    candidate = build_allocation_shadow_candidate(
        weekly,
        prices,
        states,
        forecast_model=FORECAST_MODEL,
        selection_end="2022-12-31",
        current_signal=signal,
        calibration_evidence={
            "selection_n_predictions": 365,
            "selection_log_loss": 0.5,
            "benchmark_selection_log_loss": 0.8,
        },
    )
    payoffs = candidate["selection"]["target_state_payoffs"]
    assert payoffs["risk_on"]["estimate"] > 0.0
    assert payoffs["risk_off"]["estimate"] < 0.0
