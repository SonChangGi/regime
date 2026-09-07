from copy import deepcopy
from dataclasses import replace
from datetime import date, timedelta

import pytest

from regime_lab.forecast_ledger import (
    ForecastLedger,
    _evaluate_completed_week,
    build_operational_diagnostics,
    read_operational_diagnostics,
)
from test_forecast_ledger import _v2_entry


def _completed(entry):
    return _evaluate_completed_week(
        entry,
        entry.forecast["decision_shadow"],
        target_week=entry.target_at.date(),
        actual_next_state="risk_on",
        gap_relatives={"SPY": 1.0, "TLT": 1.0},
        open_to_close_relatives={"SPY": 1.01, "TLT": 1.0},
        price_observations={
            "SPY": {"open": 100.0, "close": 101.0},
            "TLT": {"open": 100.0, "close": 100.0},
        },
        evaluated_at=entry.target_at,
        prior_evaluation=None,
        genesis_source="cash_genesis",
    )


def test_operational_scores_use_frozen_baseline_and_do_not_change_hashes():
    entry = _v2_entry()
    forecast = deepcopy(entry.forecast)
    forecast["model_forecasts"].append(
        {
            "model": "markov",
            "date": entry.target_at.date().isoformat(),
            "state": "risk_on",
            "probabilities": {"risk_on": 0.4, "transition": 0.3, "risk_off": 0.3},
        }
    )
    forecast["local_publication_at"] = entry.decision_at.isoformat()
    entry = replace(entry, forecast=forecast, inserted_at=entry.decision_at)
    evaluation = _completed(entry)
    hashes = entry.forecast_sha256, evaluation.evaluation_sha256
    result = build_operational_diagnostics([entry], [evaluation], as_of=entry.target_at)
    assert (entry.forecast_sha256, evaluation.evaluation_sha256) == hashes
    scores = result["probability_scores"]
    assert scores["completed_weeks"] == 1
    assert scores["benchmarks"]["markov"]["matched_n"] == 1
    assert scores["benchmarks"]["markov"]["log_loss_improvement"] > 0
    assert result["timing"]["on_time_rate"] == 1
    assert result["continuous_segments"][0]["benchmarks"]["weekly_60_40"][
        "net_cumulative_return"
    ] == pytest.approx(1.006 * 0.999 - 1)


def test_late_signal_benchmarks_keep_cash():
    entry = _v2_entry()
    forecast = deepcopy(entry.forecast)
    signal = forecast["decision_shadow"]["current_signal"]
    from datetime import datetime

    decision = datetime.fromisoformat(signal["scheduled_entry_at"]) + timedelta(hours=1)
    signal.update(
        decision_at=decision.isoformat(), status="missed_entry", action="no_trade"
    )
    entry = replace(entry, decision_at=decision, forecast=forecast)
    result = build_operational_diagnostics(
        [entry], [_completed(entry)], as_of=entry.target_at
    )
    assert result["timing"]["on_time_rate"] == 0
    assert all(
        b["net_cumulative_return"] == 0
        for b in result["continuous_segments"][0]["benchmarks"].values()
    )


def test_read_only_diagnostics_never_rewrites_issued_database(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    entry = _v2_entry()
    evaluation = _completed(entry)
    with ForecastLedger(path) as ledger:
        ledger.append(entry)
        ledger.append_evaluation(evaluation)
    before = path.read_bytes()
    result = read_operational_diagnostics(path, as_of=entry.target_at)
    assert path.read_bytes() == before
    assert result["issued_entries_unchanged"]
    assert result["cross_segment_cumulative_return"] is None
