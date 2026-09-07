"""Selected-model summaries must retain their own retrospective evidence."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Versioned real payload keeps the test runnable without an ignored preview build.
# Its operating leaderboard is also the source of the comprehensive preview.
ACTUAL_PAYLOAD = ROOT / "publication/live/regime-results.json"
METRICS = {
    "logLoss": "log_loss",
    "brier": "brier",
    "calibration": "calibration_error",
    "selectionCalibration": "selection_calibration_error",
    "recall": "transition_recall",
    "precision": "transition_precision",
    "captured": "on_time_departure_count",
    "events": "transition_event_count",
    "falseAlarms": "false_alarms_per_year",
    "falseAlarmCount": "false_alarm_count",
    "delay": "mean_detection_delay_forecast_weeks",
    "weeks": "n_predictions",
    "fallbackCount": "fallback_count",
}


def _run_js(program, data):
    prelude = """
const fs = require('fs'), api = require('./web/insights.js');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
"""
    process = subprocess.run(
        ["node", "-e", prelude + program],
        input=json.dumps(data, allow_nan=False),
        text=True,
        capture_output=True,
        check=True,
        cwd=ROOT,
    )
    return json.loads(process.stdout)


@pytest.fixture
def actual_model():
    return json.loads(ACTUAL_PAYLOAD.read_text())["model"]


def test_all_eleven_real_models_follow_repeated_selection_without_mutation(
    actual_model,
):
    rows = {row["name"]: row for row in actual_model["leaderboard"]}
    assert len(rows) == len(actual_model["leaderboard"]) == 11
    names = list(rows)
    sequence = names + list(reversed(names)) + [actual_model["champion"], names[0]]
    result = _run_js(
        """
const before = JSON.stringify(input.model);
const selected = input.sequence.map(name => api.modelQuality(input.model, name));
console.log(JSON.stringify({selected, unchanged: before === JSON.stringify(input.model)}));
""",
        {"model": actual_model, "sequence": sequence},
    )
    assert result["unchanged"]
    for name, summary in zip(sequence, result["selected"], strict=True):
        row = rows[name]
        assert summary["model"] == name
        assert summary["available"] is True
        for public, source in METRICS.items():
            assert summary[public] == row[source], (name, public)
        assert summary["calibrationDrift"] == pytest.approx(
            row["calibration_error"] - row["selection_calibration_error"]
        )


def test_unknown_missing_duplicate_and_selection_only_never_borrow_champion(
    actual_model,
):
    selected = next(
        row["name"]
        for row in actual_model["leaderboard"]
        if row["name"] != actual_model["champion"]
    )
    result = _run_js(
        """
const selected = input.selected, original = input.model;
const picked = original.leaderboard.find(row => row.name === selected);
const without = {...original, leaderboard: original.leaderboard.filter(row => row.name !== selected)};
const noLeaderboard = {...original}; delete noLeaderboard.leaderboard;
const cases = {
  unknown: api.modelQuality(original, 'unknown-model'),
  missing: api.modelQuality(without, selected),
  missingLeaderboard: api.modelQuality(noLeaderboard, selected),
  emptyLeaderboard: api.modelQuality({...original, leaderboard: []}, selected),
  championMissing: api.modelQuality({...original, leaderboard: original.leaderboard.filter(row => row.name !== original.champion)}, original.champion),
  defaultWithoutRow: api.modelQuality({...original, leaderboard: []}),
  duplicate: api.modelQuality({...original, leaderboard: [...original.leaderboard, {...picked}]}, selected),
  selectionOnly: api.modelQuality({...without, leaderboard: [...without.leaderboard, {...picked, evaluation_split: 'selection'}]}, selected),
  cleared: api.modelQuality(original, '')
};
console.log(JSON.stringify(cases));
""",
        {"model": actual_model, "selected": selected},
    )
    for name, summary in result.items():
        assert summary["available"] is False, name
        for field in [*METRICS, "calibrationDrift"]:
            assert summary[field] is None, (name, field)


def test_selection_duplicate_is_excluded_from_unique_retrospective_row(actual_model):
    selected = actual_model["champion"]
    result = _run_js(
        """
const row = input.leaderboard.find(row => row.name === input.champion);
const duplicate = {...row, evaluation_split: 'selection', log_loss: 777, on_time_departure_count: 888};
const model = {...input, leaderboard: [duplicate, ...input.leaderboard]};
console.log(JSON.stringify(api.modelQuality(model, input.champion)));
""",
        actual_model,
    )
    row = next(row for row in actual_model["leaderboard"] if row["name"] == selected)
    assert result["available"] is True
    assert result["logLoss"] == row["log_loss"]
    assert result["captured"] == row["on_time_departure_count"]


def test_null_and_missing_fields_stay_unknown_even_with_champion_health(actual_model):
    result = _run_js(
        """
const name = input.leaderboard.find(row => row.name !== input.champion).name;
const model = structuredClone(input), row = model.leaderboard.find(row => row.name === name);
row.log_loss = null;
row.transition_recall = null;
delete row.on_time_departure_count;
delete row.false_alarms_per_year;
row.metrics = {log_loss: 999}; // Explicit top-level null must remain unknown.
console.log(JSON.stringify({name, summary: api.modelQuality(model, name), row}));
""",
        actual_model,
    )
    summary = result["summary"]
    assert summary["model"] == result["name"]
    assert summary["available"] is True
    for field in ("logLoss", "recall", "captured", "falseAlarms"):
        assert summary[field] is None
    for field in ("brier", "events", "falseAlarmCount", "weeks"):
        assert summary[field] == result["row"][METRICS[field]]


def test_zero_capture_and_no_event_are_distinct_from_missing(actual_model):
    result = _run_js(
        """
const row = {...input.leaderboard[0], name: 'zero-case', on_time_departure_count: 0,
  transition_event_count: 39, transition_recall: 0, false_alarm_count: 0, false_alarms_per_year: 0};
const model = {...input, leaderboard: [row]};
const zeroCapture = api.modelQuality(model, row.name);
row.transition_event_count = 0;
const noEvent = api.modelQuality(model, row.name);
row.transition_recall = null;
const noEventUndefinedRecall = api.modelQuality(model, row.name);
console.log(JSON.stringify({zeroCapture, noEvent, noEventUndefinedRecall}));
""",
        actual_model,
    )
    zero = result["zeroCapture"]
    assert (zero["captured"], zero["events"], zero["recall"]) == (0, 39, 0)
    assert zero["falseAlarms"] == zero["falseAlarmCount"] == 0
    # The renderer uses events === 0 for '평가 전환 없음'; the API preserves
    # the supplied zero count without replacing it with champion evidence.
    assert result["noEvent"]["captured"] == result["noEvent"]["events"] == 0
    assert result["noEvent"]["recall"] == 0
    assert result["noEventUndefinedRecall"]["events"] == 0
    assert result["noEventUndefinedRecall"]["recall"] is None


def test_nonfinite_or_non_numeric_metrics_are_actual_null_before_json_encoding(
    actual_model,
):
    result = _run_js(
        """
const mappings = input.metrics;
const invalid = [NaN, Infinity, -Infinity, '0.3', undefined, null];
const checks = invalid.map(value => {
  const row = {...input.model.leaderboard[0]};
  for (const field of Object.values(mappings)) row[field] = value;
  const summary = api.modelQuality({...input.model, leaderboard: [row]}, row.name);
  // JSON.stringify silently turns NaN/Infinity into null, so inspect before it.
  return Object.keys(mappings).every(key => summary[key] === null)
    && summary.calibrationDrift === null;
});
console.log(JSON.stringify(checks));
""",
        {"model": actual_model, "metrics": METRICS},
    )
    assert result == [True] * 6
