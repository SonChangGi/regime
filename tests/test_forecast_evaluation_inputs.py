"""Evaluate view inputs against completed forecasts, including the Python scorer."""
from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

from regime_lab.analysis.validation import evaluate_predictions


ROOT = Path(__file__).resolve().parents[1]
STATES = ("risk_on", "transition", "risk_off")
RESEARCH = ("boundary_filtered_history", "boundary_student_t")


def run_evaluation(payload, *options):
    program = """
const api=require('./web/insights.js'),fs=require('fs');
const {payload,options}=JSON.parse(fs.readFileSync(0,'utf8'));
const before=JSON.stringify(payload);
const results=options.map(option=>api.forecastEvaluation(payload,option));
console.log(JSON.stringify({results,unchanged:before===JSON.stringify(payload),
 quality:results.map(result=>api.modelQuality({leaderboard:result.leaderboard},result.leaderboard[0]?.name))}));
"""
    result = subprocess.run(["node", "-e", program], cwd=ROOT, text=True,
                            input=json.dumps({"payload": payload, "options": options or [{}]}),
                            capture_output=True, check=True)
    return json.loads(result.stdout)


@pytest.fixture
def example():
    dates = [(date(2023, 1, 6) + timedelta(weeks=i)).isoformat() for i in range(121)]
    models = ["model_a", "model_b", *RESEARCH]
    history = {name: [] for name in RESEARCH}
    weekly = []
    for index, origin in enumerate(dates[:-1]):
        current = STATES[(index // 3) % 3]
        forecasts = []
        for number, name in enumerate(models):
            predicted = (index // (number + 2)) % 3
            p = np.full(3, .1 + number * .01)
            p[predicted] = 1 - 2 * p[0] if predicted != 0 else 1 - 2 * p[1]
            probabilities = dict(zip(STATES, p.tolist(), strict=True))
            if name in RESEARCH:
                history[name].append({"origin_date": origin + "T21:00:00Z", "target_date": dates[index + 1] + "T21:00:00Z",
                                      "current_state": "risk_off", "actual": "wrong_source_label",
                                      "probabilities": probabilities})
            else:
                forecasts.append({"model": name, "date": dates[index + 1], "state": "wrong_argmax",
                                  "probabilities": probabilities, "fallback": index == 104})
        weekly.append({"date": origin, "current": {"state": current}, "model_forecasts": forecasts})
    return {"model": {"champion": "model_a", "forecast_comparison": {"models": models[:2]},
                       "leaderboard": [{"name": name, "rank": 9 - i, "log_loss": 987,
                                        "selection_log_loss": .123 + i} for i, name in enumerate(models[:2])]},
            "weekly": weekly,
            "research": {"forecast_improvement": {"schema_version": "regime-forecast-improvement/1",
                "models": [{"id": name, "history": history[name][:-1], "latest": history[name][-1],
                            "metrics": {"holdout": {"log_loss": 999}, "selection": {"log_loss": .222}}} for name in RESEARCH]}}}


def expected_frame(payload, scope):
    actual = {week["date"]: week["current"]["state"] for week in payload["weekly"]}
    research = {model["id"]: {row["origin_date"][:10]: row for row in [*model["history"], model["latest"]]}
                for model in payload.get("research", {}).get("forecast_improvement", {}).get("models", [])}
    names = [*payload["model"]["forecast_comparison"]["models"], *research]
    records = []
    for week in payload["weekly"]:
        if not scope["start"] <= week["date"] <= scope["end"]:
            continue
        for name in names:
            source = research[name][week["date"]] if name in research else next(row for row in week["model_forecasts"] if row["model"] == name)
            target = source.get("target_date", source.get("date"))[:10]
            if target > scope["asOf"]:
                continue
            records.append({"model": name, "origin_date": week["date"], "actual": actual[target],
                            "current_state": actual[week["date"]], "fallback": source.get("fallback", False),
                            **{"p_" + state: source["probabilities"][state] for state in STATES}})
    return pd.DataFrame(records)


def assert_python_metrics(payload, result):
    expected = evaluate_predictions(expected_frame(payload, result["scope"])).set_index("model")
    for row in result["leaderboard"]:
        for metric, value in expected.loc[row["name"]].items():
            if pd.isna(value):
                assert row[metric] is None, (row["name"], metric)
            else:
                assert row[metric] == pytest.approx(value, abs=1e-12), (row["name"], metric)


def test_default_and_each_window_use_origins_and_count_only_completed_targets(example):
    output = run_evaluation(example, {}, {"window": 26}, {"window": 104}, {"window": "all"})
    assert output["unchanged"]
    for result, count in zip(output["results"], (52, 26, 104, 120), strict=True):
        scope = result["scope"]
        assert scope["originCount"] == count
        assert scope["completedCount"] == count - 1
        assert scope["pendingCount"] == 1
        assert scope["excludedCount"] == 0
        assert scope["completedEnd"] == scope["asOf"]
        assert scope["completedStart"] > scope["start"]
        assert_python_metrics(example, result)
    assert output["results"][0]["leaderboard"][0]["log_loss"] != output["results"][1]["leaderboard"][0]["log_loss"]


def test_historical_asof_excludes_later_truth_and_preserves_inputs(example):
    as_of = example["weekly"][70]["date"]
    original = run_evaluation(example, {"asOf": as_of, "window": 52})["results"][0]
    mutated = deepcopy(example)
    for week in mutated["weekly"][71:]:
        week["current"]["state"] = "risk_on"
        week["model_forecasts"] = []
    changed = run_evaluation(mutated, {"asOf": as_of, "window": 52})["results"][0]
    assert changed == original
    assert original["scope"]["completedCount"] == 51
    assert original["scope"]["end"] == as_of
    assert_python_metrics(example, original)


def test_authoritative_actual_and_argmax_override_stored_research_labels(example):
    result = run_evaluation(example, {"window": "all"})["results"][0]
    assert_python_metrics(example, result)
    assert len(result["leaderboard"]) == 4
    assert all(row["n_predictions"] == 119 for row in result["leaderboard"])
    for row in result["leaderboard"]:
        frame = expected_frame(example, result["scope"])
        frame = frame.loc[frame.model.eq(row["name"])]
        actual = frame.actual.map(STATES.index).to_numpy()
        current = frame.current_state.map(STATES.index).to_numpy()
        predicted = frame[["p_" + state for state in STATES]].to_numpy().argmax(axis=1)
        assert row["worsening_event_count"] == int((actual > current).sum())
        assert row["on_time_worsening_count"] == int(((actual > current) & (predicted > current)).sum())
        assert row["recovery_event_count"] == int((actual < current).sum())
        assert row["on_time_recovery_count"] == int(((actual < current) & (predicted < current)).sum())


@pytest.mark.parametrize("failure", ["missing_research", "missing_official", "invalid_probability", "mismatched_target"])
def test_incomplete_model_forecast_excludes_origin_for_every_model(example, failure):
    before = run_evaluation(example)["results"][0]
    if failure == "missing_research":
        example["research"]["forecast_improvement"]["models"][0]["history"].pop(100)
    elif failure == "missing_official":
        example["weekly"][100]["model_forecasts"].pop()
    elif failure == "invalid_probability":
        example["weekly"][100]["model_forecasts"][0]["probabilities"]["risk_on"] = -.1
    else:
        example["weekly"][100]["model_forecasts"][0]["date"] = example["weekly"][102]["date"]
    after = run_evaluation(example)["results"][0]
    assert after["scope"]["excludedCount"] == 1
    assert after["scope"]["pendingCount"] == 1
    assert all(row["n_predictions"] == before["scope"]["completedCount"] - 1 for row in after["leaderboard"])


def test_empty_window_returns_missing_metrics_without_stale_headline(example):
    result = run_evaluation(example, {"asOf": "2000-01-01"})
    assert result["results"][0]["scope"]["completedCount"] == 0
    assert all(row["log_loss"] is None and row["scope_rank"] is None for row in result["results"][0]["leaderboard"])
    assert result["quality"][0]["logLoss"] is None
    assert result["quality"][0]["weeks"] == 0


def test_selection_metrics_and_original_rank_stay_fixed_while_view_rank_is_computed(example):
    result = run_evaluation(example)["results"][0]
    for row in result["leaderboard"][:2]:
        source = next(item for item in example["model"]["leaderboard"] if item["name"] == row["name"])
        assert row["rank"] == source["rank"]
        assert row["selection_log_loss"] == source["selection_log_loss"]
    assert sorted(row["scope_rank"] for row in result["leaderboard"]) == [1, 2, 3, 4]
    assert min(result["leaderboard"], key=lambda row: row["log_loss"])["scope_rank"] == 1


def test_comparisons_distinguish_probability_difference_from_identical_decisions(example):
    example["selection"] = {"operating_champion": {"name": "model_b"}}
    for week in example["weekly"]:
        first, second = week["model_forecasts"]
        first["probabilities"] = {"risk_on": .70, "transition": .20, "risk_off": .10}
        second["probabilities"] = {"risk_on": .69, "transition": .21, "risk_off": .10}
    result = run_evaluation(example)["results"][0]
    difference = result["comparisons"]["model_a"]
    assert difference["samePredictions"] == difference["comparedWeeks"] == 51
    assert difference["meanProbabilityDifference"] == pytest.approx(.02 / 3)
    rows = {row["name"]: row for row in result["leaderboard"]}
    assert difference["logLossDifference"] == rows["model_a"]["log_loss"] - rows["model_b"]["log_loss"]
    assert result["comparisons"]["model_b"]["meanProbabilityDifference"] == 0


def test_python_hard_probability_clamping_and_calibration_bins(example):
    for index, week in enumerate(example["weekly"]):
        values = ([1, 0, 0], [.6, .3, .1], [.4, .4, .2], [.2, .2, .6])[index % 4]
        for row in week["model_forecasts"]:
            row["probabilities"] = dict(zip(STATES, values, strict=True))
    result = run_evaluation(example, {"window": "all"})["results"][0]
    assert_python_metrics(example, result)


def test_published_thirteen_models_match_python_scorer_for_full_and_selected_scope():
    payload = json.loads((ROOT / "publication/live/regime-results.json").read_text())
    output = run_evaluation(payload, {"window": "all"}, {"window": 52}, {"asOf": "2025-12-26", "window": 52})
    assert output["unchanged"]
    for result, n in zip(output["results"], (191, 51, 51), strict=True):
        assert len(result["leaderboard"]) == 13
        assert result["scope"]["completedCount"] == n
        assert result["scope"]["excludedCount"] == 0
        assert_python_metrics(payload, result)
    full = output["results"][0]
    dynamic = next(row for row in full["leaderboard"] if row["name"] == "causal_dynamic_ensemble")
    multiscale = next(row for row in full["leaderboard"] if row["name"] == "causal_multiscale_ensemble")
    assert dynamic["log_loss"] != multiscale["log_loss"]
    assert full["comparisons"]["causal_multiscale_ensemble"]["samePredictions"] == 191
    assert full["comparisons"]["causal_multiscale_ensemble"]["meanProbabilityDifference"] > 0
