from __future__ import annotations

import numpy as np
import pandas as pd

from regime_lab.analysis.decision_shadow import build_decision_shadow


def test_decision_shadow_has_matched_costed_benchmarks_and_separate_evidence_tracks() -> None:
    index = pd.date_range("2020-01-03", periods=120, freq="W-FRI", tz="UTC")
    position = np.arange(len(index), dtype=float)
    prices = pd.DataFrame(
        {
            "spy_close": 100.0 * np.exp(np.cumsum(0.002 + 0.01 * np.sin(position / 9))),
            "tlt_close": 100.0 * np.exp(np.cumsum(0.001 - 0.006 * np.sin(position / 9))),
        },
        index=index,
    )
    weekly = []
    for offset, origin in enumerate(index[20:-2]):
        if offset % 2:
            probabilities = {"risk_on": 0.7, "transition": 0.2, "risk_off": 0.1}
        else:
            probabilities = {"risk_on": 0.1, "transition": 0.2, "risk_off": 0.7}
        weekly.append(
            {
                "date": origin.date().isoformat(),
                "next_week": {"probabilities": probabilities},
            }
        )

    result = build_decision_shadow(weekly, prices)

    historical = result["historical_reconstructed_shadow"]
    prospective = result["prospective_ledger"]
    strategies = historical["strategies"]
    assert historical["evidence_track"] == "reconstructed_oos"
    assert prospective["evidence_track"] == "operational_oos"
    assert prospective["affects_official_forecast"] is False
    assert prospective["affects_champion_selection"] is False
    assert set(strategies) == {
        "probability_shadow",
        "spy_buy_and_hold",
        "static_60_40",
        "vol_target_60_40",
    }
    assert {row["weeks"] for row in strategies.values()} == {
        strategies["probability_shadow"]["weeks"]
    }
    assert historical["first_tradable_at"] == index[21].isoformat()
    shadow = strategies["probability_shadow"]
    assert shadow["annualized_turnover"] > 0.0
    assert shadow["total_transaction_cost"] > 0.0
    assert shadow["cumulative_return"] <= shadow["gross_cumulative_return"]
    for field in (
        "sharpe",
        "certainty_equivalent_return",
        "maximum_drawdown",
    ):
        assert shadow[field] is not None
