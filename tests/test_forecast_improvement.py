"""Research composition must keep actual model history and validated caches."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from regime_lab.analysis.labels import CausalRegimeLabeler, RegimeLabelConfig
from regime_lab.research import forecast_improvement as research
from regime_lab.schema import STATE_ORDER


@pytest.fixture(scope="module")
def frames():
    rng = np.random.default_rng(43)
    index = pd.date_range("2012-01-06", periods=610, freq="W-FRI", tz="UTC")
    canonical = pd.DataFrame({"spy_close": 100 * np.exp((rng.standard_t(5, len(index))*.015+.001).cumsum())}, index=index)
    labeler = CausalRegimeLabeler(RegimeLabelConfig(minimum_fit_observations=260)).fit(canonical.iloc[:520])
    states = labeler.transform(canonical)
    rows = []
    for position in range(521, len(states)-1):
        if index[position].year < 2023 <= index[position+1].year:
            continue
        current = states.iloc[position]
        rows.append({"origin_date": index[position].isoformat(), "target_date": index[position+1].isoformat(),
                     "model": "causal_dynamic_ensemble", "current_state": current, "actual": states.iloc[position+1],
                     "evaluation_split": "selection" if index[position+1].year < 2023 else "holdout",
                     **{f"p_{state}": .8 if state == current else .1 for state in STATE_ORDER}, "fallback": False})
    return canonical, states, pd.DataFrame(rows)


def test_complete_block_has_real_model_history_and_does_not_mutate_inputs(frames):
    canonical, states, baseline = frames
    originals = (canonical.copy(deep=True), states.copy(deep=True), baseline.copy(deep=True))
    block = research.build_forecast_improvement(canonical, states, baseline)
    assert block["selected_model"] == "boundary_filtered_history"
    assert block["selection"]["weekly_reselection"] is False
    assert [row["id"] for row in block["models"]] == list(research.RESEARCH_MODELS)
    holdout_count = int(baseline.evaluation_split.eq("holdout").sum())
    for model in block["models"]:
        assert len(model["history"]) == len(baseline)
        assert sum(row["evaluation_split"] == "holdout" for row in model["history"]) == holdout_count
        assert model["metrics"]["holdout"]["n_predictions"] == holdout_count
        assert model["latest"]["actual"] is None
        for row in model["history"]:
            assert row["actual"] == states.loc[pd.Timestamp(row["target_date"])]
            assert row["current_state"] == states.loc[pd.Timestamp(row["origin_date"])]
            assert row["predicted"] == max(row["probabilities"], key=row["probabilities"].get)
    pd.testing.assert_frame_equal(canonical, originals[0])
    pd.testing.assert_series_equal(states, originals[1])
    pd.testing.assert_frame_equal(baseline, originals[2])
    json.dumps(block, allow_nan=False)


def test_cache_reuses_matching_code_and_inputs_without_model_fitting(frames, tmp_path, monkeypatch):
    first = research.build_forecast_improvement(*frames, cache_directory=tmp_path)
    def unexpected(*args, **kwargs):
        raise AssertionError("cache should avoid model execution")
    monkeypatch.setattr(research, "run_boundary_walk_forward", unexpected)
    second = research.build_forecast_improvement(*frames, cache_directory=tmp_path)
    assert first == second


def test_corrupted_cache_fails_without_overwriting_accepted_files(frames, tmp_path):
    block = research.build_forecast_improvement(*frames, cache_directory=tmp_path)
    folder = tmp_path / block["provenance"]["cache_key"]
    output = folder / "block.json"
    output.write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        research.build_forecast_improvement(*frames, cache_directory=tmp_path)
    assert output.read_text() == "{}"


def test_cache_key_changes_with_input_revision(frames, tmp_path, monkeypatch):
    first = research.build_forecast_improvement(*frames, cache_directory=tmp_path)
    revised = frames[0].copy()
    revised["extra_known_feature"] = 1.
    calls = []
    actual = research.run_boundary_walk_forward
    def traced(*args, **kwargs):
        calls.append(True)
        return actual(*args, **kwargs)
    monkeypatch.setattr(research, "run_boundary_walk_forward", traced)
    second = research.build_forecast_improvement(revised, *frames[1:], cache_directory=tmp_path)
    assert len(calls) == 1
    assert first["provenance"]["cache_key"] != second["provenance"]["cache_key"]


def test_missing_origin_is_not_silently_scored_on_smaller_sample(frames):
    with pytest.raises(ValueError, match="every official OOS"):
        research.build_forecast_improvement(*frames[:2], frames[2].iloc[:-1])


def test_source_state_mismatch_prevents_partial_results(frames):
    corrupted = frames[2].copy()
    corrupted.loc[0, "actual"] = next(state for state in STATE_ORDER if state != corrupted.loc[0, "actual"])
    with pytest.raises(ValueError, match="official states"):
        research.build_forecast_improvement(*frames[:2], corrupted)


def test_failed_model_execution_leaves_no_cache_result(frames, tmp_path, monkeypatch):
    def failed(*args, **kwargs):
        raise RuntimeError("planned model failure")
    monkeypatch.setattr(research, "run_boundary_walk_forward", failed)
    with pytest.raises(RuntimeError, match="planned"):
        research.build_forecast_improvement(*frames, cache_directory=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_snapshot_attachment_is_deep_copy_and_cutoff_locked(frames):
    block = research.build_forecast_improvement(*frames)
    snapshot = {"meta": {"data_as_of": block["data_as_of"]}, "weekly": [{"preserved": True}], "research": {"existing": {"value": 1}}}
    source = deepcopy(snapshot)
    attached = research.with_forecast_improvement(snapshot, block)
    attached["research"]["forecast_improvement"]["models"][0]["latest"]["actual"] = "tampered"
    assert snapshot == source
    assert block["models"][0]["latest"]["actual"] is None
    wrong = deepcopy(snapshot)
    wrong["meta"]["data_as_of"] = "2000-01-01"
    with pytest.raises(ValueError, match="cutoff"):
        research.with_forecast_improvement(wrong, block)
