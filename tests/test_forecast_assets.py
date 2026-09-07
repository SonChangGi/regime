"""Improved forecasts must change classification while preserving trade timing."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from regime_lab.analysis.outcomes import ASSETS
from regime_lab.research import forecast_assets
from regime_lab.research.forecast_improvement import _metric_table, RESEARCH_MODELS
from regime_lab.schema import STATE_ORDER


@pytest.fixture(scope="module")
def fixture():
    index = pd.date_range("2022-09-02T21:00:00Z", periods=42, freq="W-FRI")
    canonical = pd.DataFrame(index=index)
    for asset in ASSETS:
        canonical[f"{asset.lower()}_adjusted_open"] = 200 + np.arange(len(index)) * 2.
        canonical[f"{asset.lower()}_close"] = canonical[f"{asset.lower()}_adjusted_open"] * 1.1
    states = np.asarray([STATE_ORDER[i % 3] for i in range(len(index))])
    models = []
    for model_number, model in enumerate(RESEARCH_MODELS):
        chosen = STATE_ORDER[model_number * 2]
        probabilities = {state: .8 if state == chosen else .1 for state in STATE_ORDER}
        history = []
        metrics_rows = []
        for position, origin in enumerate(index[:-1]):
            if origin.year < 2023 <= index[position + 1].year:
                continue
            row = {"origin_date": origin.isoformat(), "target_date": index[position + 1].isoformat(),
                   "evaluation_split": "selection" if index[position + 1].year < 2023 else "holdout",
                   "current_state": states[position], "actual": states[position+1], "predicted": chosen,
                   "probabilities": probabilities.copy(), "raw_probabilities": probabilities.copy(),
                   "calibration": {"temperature": 1., "rows": max(0,len(history)-1), "last_train_target": (origin-pd.DateOffset(weeks=1)).isoformat()}}
            history.append(row)
            metrics_rows.append({key: value for key, value in row.items() if key not in {"probabilities", "raw_probabilities", "calibration"}} | {"model": model, "fallback": False, **{f"p_{state}": value for state,value in probabilities.items()}})
        latest = {**deepcopy(history[-1]), "origin_date": index[-1].isoformat(), "target_date": (index[-1]+pd.DateOffset(weeks=1)).isoformat(),
                  "current_state": states[-1], "actual": None, "evaluation_split": "unobserved",
                  "calibration": {"temperature": 1., "rows": len(history)-1, "last_train_target": index[-2].isoformat()}}
        models.append({"id": model, "history": history, "latest": latest, "metrics": _metric_table(pd.DataFrame(metrics_rows))})
    block = {"schema_version": "regime-forecast-improvement/1", "selected_model": RESEARCH_MODELS[0], "evidence_track": "reconstructed_market", "data_as_of": index[-1].isoformat(),
             "models": models, "provenance": {key: "a"*64 for key in ("input_sha256","code_sha256","baseline_oos_sha256","cache_key")}}
    return canonical, block


def test_model_classification_changes_asset_rows_and_uses_next_open(fixture, tmp_path):
    canonical, block = fixture
    original = deepcopy(block)
    result = forecast_assets.build_forecast_asset_statistics(canonical, block, bootstrap_resamples=0, cache_directory=tmp_path)
    assert len(result["rows"]) == 2 * 6 * 3 * 3
    assert tuple(result["models"]) == RESEARCH_MODELS
    rows = pd.DataFrame(result["rows"])
    for model, expected in zip(RESEARCH_MODELS, ("risk_on", "risk_off")):
        chosen = rows.loc[rows.conditioning_model.eq(model)]
        assert set(chosen.loc[chosen.n.gt(0), "state"]) == {expected}
    cached = next(tmp_path.iterdir())
    outcomes = pd.read_csv(cached / "outcomes.csv")
    one_week = outcomes.loc[outcomes.horizon_weeks.eq(1)]
    np.testing.assert_allclose(one_week.forward_return, .1, rtol=0, atol=1e-14)
    assert (pd.to_datetime(one_week.entry_date) == pd.to_datetime(one_week.exit_date)).all()
    assert (pd.to_datetime(one_week.entry_date)-pd.to_datetime(one_week.origin_date)).dt.days.eq(7).all()
    assert not pd.to_datetime(outcomes.origin_date).eq(canonical.index[-1]).any()
    assert block == original


def test_cache_reuses_exact_inputs_but_rebuilds_for_bootstrap_setting(fixture, tmp_path, monkeypatch):
    first = forecast_assets.build_forecast_asset_statistics(*fixture, bootstrap_resamples=0, cache_directory=tmp_path)
    actual = forecast_assets.v5._model_conditioned_research
    calls = []
    def traced(*args, **kwargs):
        calls.append(kwargs["bootstrap_resamples"])
        return actual(*args, **kwargs)
    monkeypatch.setattr(forecast_assets.v5, "_model_conditioned_research", traced)
    assert forecast_assets.build_forecast_asset_statistics(*fixture, bootstrap_resamples=0, cache_directory=tmp_path) == first
    assert calls == []
    forecast_assets.build_forecast_asset_statistics(*fixture, bootstrap_resamples=1, cache_directory=tmp_path)
    assert calls == [1]
    assert len(list(tmp_path.iterdir())) == 2


def test_cache_rejects_corrupted_asset_results_without_replacing_them(fixture, tmp_path):
    forecast_assets.build_forecast_asset_statistics(*fixture, bootstrap_resamples=0, cache_directory=tmp_path)
    folder = next(tmp_path.iterdir())
    result = folder / "asset-statistics.json"
    result.write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        forecast_assets.build_forecast_asset_statistics(*fixture, bootstrap_resamples=0, cache_directory=tmp_path)
    assert result.read_text() == "{}"


def test_failed_outcome_calculation_returns_no_partial_cache(fixture, tmp_path, monkeypatch):
    def failure(*args, **kwargs):
        raise RuntimeError("outcome calculation interrupted")
    monkeypatch.setattr(forecast_assets.v5, "_model_conditioned_research", failure)
    with pytest.raises(RuntimeError, match="interrupted"):
        forecast_assets.build_forecast_asset_statistics(*fixture, bootstrap_resamples=0, cache_directory=tmp_path)
    assert list(tmp_path.iterdir()) == []
