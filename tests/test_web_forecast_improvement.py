"""Research model selection must change real views without borrowing evidence."""

import copy
import json
import math
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def run_js(program, payload):
    output = subprocess.run(
        ["node", "-e", "const api=require('./web/app.js'), insights=require('./web/insights.js');const input=JSON.parse(require('fs').readFileSync(0,'utf8'));" + program],
        input=json.dumps(payload), text=True, capture_output=True, check=True, cwd=ROOT,
    )
    return json.loads(output.stdout)


def metrics_for(rows):
    order = ("risk_on", "transition", "risk_off")
    scored = []
    for row in rows:
        actual, current = order.index(row["actual"]), order.index(row["current_state"])
        p = [row["probabilities"][state] for state in order]
        predicted = max(range(3), key=lambda position: p[position])
        event, alert = actual != current, predicted != current
        scored.append({"loss": -math.log(p[actual]), "brier": sum((value - int(position == actual)) ** 2 for position, value in enumerate(p)),
            "event": event, "hit": event and alert, "false": not event and alert,
            "worsening": actual > current, "worsening_hit": actual > current and predicted > current,
            "recovery": actual < current, "recovery_hit": actual < current and predicted < current})
    total = lambda key: sum(row[key] for row in scored)
    return {"n_predictions": len(rows), "log_loss": total("loss") / len(rows), "brier": total("brier") / len(rows),
        "transition_recall": total("hit") / total("event") if total("event") else 0,
        "on_time_departure_count": total("hit"), "transition_event_count": total("event"), "false_alarm_count": total("false"),
        "false_alarms_per_year": total("false") / len(rows) * 52.1775,
        "worsening_event_count": total("worsening"), "on_time_worsening_count": total("worsening_hit"),
        "recovery_event_count": total("recovery"), "on_time_recovery_count": total("recovery_hit")}


@pytest.fixture
def payload():
    result = json.loads((ROOT / "publication/live/regime-results.json").read_text())
    weeks = result["weekly"][-3:]
    latest_date = result["meta"]["data_as_of"][:10]
    cutoff_clock = result["meta"]["data_as_of"][11:19]
    models = []
    for index, model in enumerate(("boundary_filtered_history", "boundary_student_t")):
        probabilities = dict(zip(("risk_on", "transition", "risk_off"), ([.8, .15, .05] if index == 0 else [.2, .7, .1]), strict=True))
        history = []
        for week, target in zip(weeks[:-1], weeks[1:], strict=True):
            history.append({"origin_date": week["date"] + "T" + cutoff_clock + "Z", "target_date": target["date"] + "T" + cutoff_clock + "Z",
                "current_state": week["current"]["state"], "actual": target["current"]["state"],
                "probabilities": probabilities, "raw_probabilities": probabilities, "predicted": max(probabilities, key=probabilities.get),
                "evaluation_split": "holdout", "calibration": {"temperature": 1, "rows": 156, "last_train_target": "2020-01-03T20:00:00Z"}})
        latest = {**history[-1], "origin_date": result["meta"]["data_as_of"], "target_date": weeks[-1]["next_week"]["date"] + "T" + cutoff_clock + "Z",
                  "current_state": weeks[-1]["current"]["state"], "actual": None, "evaluation_split": "unobserved"}
        selection_history = [{**copy.deepcopy(history[0]), "origin_date": f"2022-12-{day:02}T20:00:00Z", "target_date": f"2022-12-{day + 7:02}T20:00:00Z", "evaluation_split": "selection"} for day in (2, 9)]
        models.append({"id": model, "label": model, "metrics": {"selection": metrics_for(selection_history), "holdout": metrics_for(history)}, "history": selection_history + history, "latest": latest})
    result["research"]["forecast_improvement"] = {
        "schema_version": "regime-forecast-improvement/1", "selected_model": models[0]["id"],
        "data_as_of": result["meta"]["data_as_of"], "evidence_track": "reconstructed_market",
        "selection": {"frozen_model": models[0]["id"], "selection_end": "2022-12-31"},
        "models": models, "baselines": [],
        "provenance": {key: "a" * 64 for key in ("input_sha256", "code_sha256", "baseline_oos_sha256", "cache_key")},
    }
    return result


def test_research_models_are_optional_and_preserve_official_identity(payload):
    original = copy.deepcopy(payload)
    result = run_js("""
const before=JSON.stringify(input), official=input.model.champion;
const all=api.forecastComparisonModels(input), model=insights.forecastComparisonModel(input);
const without=structuredClone(input);delete without.research.forecast_improvement;
console.log(JSON.stringify({all,original:api.forecastComparisonModels(without),champion:model.champion,
official,unchanged:before===JSON.stringify(input),valid:insights.validateForecastImprovement(input)}));
""", payload)
    assert result["all"] == result["original"] + [model["id"] for model in payload["research"]["forecast_improvement"]["models"]]
    assert result["champion"] == result["official"]
    assert result["unchanged"] and payload == original
    assert result["valid"] == []


def test_selection_changes_probabilities_summary_chart_and_history(payload):
    result = run_js("""
const model=insights.forecastComparisonModel(input), week=input.weekly.at(-1);
const rows=insights.forecastImprovementModels(input).map(row=>({
name:row.id,quality:insights.modelQuality(model,row.id),forecast:api.forecastForWeek(week,row.id,input),
history:api.forecastForWeek(input.weekly.at(-2),row.id,input),
chart:api.modelLossComparisonRows(model.leaderboard,row.id,input.model.champion).map(row=>row.name)}));
console.log(JSON.stringify(rows));
""", payload)
    for actual, expected in zip(result, payload["research"]["forecast_improvement"]["models"], strict=True):
        assert actual["quality"]["model"] == expected["id"]
        assert actual["quality"]["logLoss"] == expected["metrics"]["holdout"]["log_loss"]
        assert actual["quality"]["researchCandidate"]
        assert actual["quality"]["worseningCaptured"] == 0
        assert actual["forecast"]["probabilities"] == expected["latest"]["probabilities"]
        assert actual["history"]["probabilities"] == expected["history"][-1]["probabilities"]
        assert expected["id"] in actual["chart"]
        assert payload["model"]["champion"] in actual["chart"]
    assert result[0]["forecast"]["state"] != result[1]["forecast"]["state"]


def test_missing_research_history_and_asset_returns_never_borrow_other_models(payload):
    result = run_js("""
const name='boundary_filtered_history';
console.log(JSON.stringify({past:api.forecastForWeek(input.weekly[0],name,input),
returns:api.conditionalStatsRowsForBasis(input,'forecast',name),complete:api.modelConditionedAssetRowsComplete(input,name),
view:api.parseViewState('?model='+name+'&basis=forecast',input,input.weekly)}));
""", payload)
    assert result["past"] is None
    assert result["returns"] == []
    assert not result["complete"]
    assert result["view"]["basis"] == "forecast"
    assert result["view"]["model"] == "boundary_filtered_history"


def test_research_deep_link_survives_pending_sidecar_and_legacy_remains_unchanged(payload):
    result = run_js("""
const core=structuredClone(input);delete core.research;
const query='?model=boundary_student_t&basis=forecast';
console.log(JSON.stringify({pending:api.parseViewState(query,core,core.weekly,{researchPending:true}),
ready:api.parseViewState(query,input,input.weekly),unavailable:api.parseViewState(query,core,core.weekly),
legacy:insights.validateForecastImprovement(core)}));
""", payload)
    assert result["pending"]["model"] == result["ready"]["model"] == "boundary_student_t"
    assert result["pending"]["basis"] == result["ready"]["basis"] == "forecast"
    assert result["unavailable"]["model"] == payload["model"]["champion"]
    assert result["legacy"] == []


@pytest.mark.parametrize("change", ["duplicate", "future_target", "unpurged", "probability", "latest_actual", "metrics", "state"])
def test_optional_block_rejects_corrupt_or_misaligned_evidence(payload, change):
    block = payload["research"]["forecast_improvement"]
    model = block["models"][0]
    if change == "duplicate":
        model["history"][1]["origin_date"] = model["history"][0]["origin_date"]
    elif change == "future_target":
        model["history"][0]["target_date"] = "2099-01-01"
    elif change == "unpurged":
        model["latest"]["calibration"]["last_train_target"] = model["latest"]["origin_date"]
    elif change == "probability":
        model["latest"]["probabilities"]["risk_on"] = -1
    elif change == "latest_actual":
        model["latest"]["actual"] = "risk_on"
    elif change == "metrics":
        model["metrics"]["holdout"]["n_predictions"] += 1
    else:
        current = model["history"][-1]["current_state"]
        model["history"][-1]["current_state"] = "risk_off" if current != "risk_off" else "risk_on"
    errors = run_js("console.log(JSON.stringify(insights.validateForecastImprovement(input)))", payload)
    assert errors


@pytest.mark.parametrize("key", ["transition_recall", "transition_event_count", "on_time_departure_count", "false_alarm_count", "false_alarms_per_year", "worsening_event_count", "on_time_worsening_count", "recovery_event_count", "on_time_recovery_count", "log_loss", "brier"])
def test_metric_only_tampering_is_rejected_against_unchanged_history(payload, key):
    model = payload["research"]["forecast_improvement"]["models"][0]
    before = copy.deepcopy(model["history"])
    model["metrics"]["holdout"][key] += .1 if key in {"transition_recall", "log_loss", "brier", "false_alarms_per_year"} else 1
    result = run_js("console.log(JSON.stringify(insights.validateForecastImprovement(input)))", payload)
    assert result and model["history"] == before
    assert any(key in message for message in result)


@pytest.mark.parametrize("change", ["same_day_cutoff", "same_day_latest", "no_timezone", "two_week_history", "two_week_latest"])
def test_exact_cutoff_and_adjacent_week_are_required(payload, change):
    from datetime import datetime, timedelta
    block = payload["research"]["forecast_improvement"]
    model = block["models"][0]
    if change == "same_day_cutoff":
        block["data_as_of"] = (datetime.fromisoformat(block["data_as_of"]) - timedelta(hours=1)).isoformat()
    elif change == "same_day_latest":
        model["latest"]["origin_date"] = (datetime.fromisoformat(model["latest"]["origin_date"]) - timedelta(hours=1)).isoformat()
    elif change == "no_timezone":
        model["history"][0]["origin_date"] = model["history"][0]["origin_date"].removesuffix("Z")
    elif change == "two_week_history":
        model["history"][0]["target_date"] = "2022-12-16T20:00:00Z"
    else:
        model["latest"]["target_date"] = (datetime.fromisoformat(model["latest"]["target_date"]) + timedelta(days=7)).isoformat()
    assert run_js("console.log(JSON.stringify(insights.validateForecastImprovement(input)))", payload)


@pytest.fixture
def payload_with_assets(payload):
    # Reuse the established return-statistics schema as synthetic routing data;
    # the real producer's outputs are independently checked in the release drill.
    original = payload["research"]["model_conditioned_asset_stats"]
    assets = {key: copy.deepcopy(value) for key, value in original.items() if key not in {"models", "rows"}}
    mapping = {"recency_weighted_xgboost_208w": "boundary_filtered_history", "xgboost": "boundary_student_t"}
    assets["models"] = list(mapping.values())
    assets["rows"] = [{**copy.deepcopy(row), "conditioning_model": mapping[row["conditioning_model"]]} for row in original["rows"] if row["conditioning_model"] in mapping]
    payload["research"]["forecast_improvement"]["asset_statistics"] = assets
    return payload


def test_research_asset_selection_routes_all_assets_horizons_and_weighting(payload_with_assets):
    result = run_js("""
const ids=insights.FORECAST_RESEARCH_IDS, before=JSON.stringify(input);
const results=ids.map(id=>({id,complete:api.modelConditionedAssetRowsComplete(input,id),
rows:api.modelConditionedAssetRows(input,id),benchmarks:[1,4,13].map(horizon=>
 ['episode','weekly'].map(weighting=>api.conditionalBenchmarkForAsset(input,'forecast',id,'SPY',horizon,null,weighting)))}));
console.log(JSON.stringify({results,unchanged:before===JSON.stringify(input),errors:api.validatePayload(input).errors}));
""", payload_with_assets)
    assert result["unchanged"] and result["errors"] == []
    expected = payload_with_assets["research"]["forecast_improvement"]["asset_statistics"]["rows"]
    for actual in result["results"]:
        assert actual["complete"]
        assert actual["rows"] == [row for row in expected if row["conditioning_model"] == actual["id"]]
        assert len(actual["rows"]) == 54
        for horizon, benchmarks in zip((1, 4, 13), actual["benchmarks"], strict=True):
            row = next(row for row in actual["rows"] if row["asset"] == "SPY" and row["horizon_weeks"] == horizon)
            assert benchmarks[0]["value"] == row["episode_equal_unconditional_benchmark_mean_return"]
            assert benchmarks[1]["value"] == row["unconditional_benchmark_mean_return"]


@pytest.mark.parametrize("change", ["duplicate", "wrong_model", "execution_lag", "benchmark", "support"])
def test_research_assets_reuse_full_existing_return_contract(payload_with_assets, change):
    assets = payload_with_assets["research"]["forecast_improvement"]["asset_statistics"]
    if change == "duplicate":
        assets["rows"][1] = copy.deepcopy(assets["rows"][0])
    elif change == "wrong_model":
        assets["rows"][0]["conditioning_model"] = "causal_dynamic_ensemble"
    elif change == "execution_lag":
        assets["execution_lag_weeks"] = 0
    elif change == "benchmark":
        row = next(row for row in assets["rows"] if row["unconditional_benchmark_mean_return"] is not None)
        row["unconditional_benchmark_mean_return"] += .5
    else:
        row = next(row for row in assets["rows"] if row["status"] == "ok")
        row["n"] = 0
    errors = run_js("console.log(JSON.stringify(api.validatePayload(input).errors))", payload_with_assets)
    assert errors
