from copy import deepcopy
from datetime import date, datetime, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import pytest
from regime_lab.dashboard_split import (
    build_dashboard_split,
    build_history_chunks,
    validate_history_chunks,
    CORE_WEEK_COUNT,
)
from regime_lab.publication_contract import PublicContractError


def _shift_synthetic_dates(value, delta):
    """Keep relative weekly forecast dates consistent in the capacity fixture."""
    if isinstance(value, dict):
        return {key: _shift_synthetic_dates(item, delta) for key, item in value.items()}
    if isinstance(value, list):
        return [_shift_synthetic_dates(item, delta) for item in value]
    if isinstance(value, str):
        try:
            parsed = (
                date.fromisoformat(value)
                if len(value) == 10
                else datetime.fromisoformat(value)
            )
        except ValueError:
            return value
        return (parsed + delta).isoformat()
    return value


def test_five_year_growth_is_bounded_and_exactly_reconstructable():
    source = json.loads(
        (Path(__file__).parents[1] / "publication/live/regime-results.json").read_text()
    )
    # Add five years of synthetic older rows to exercise capacity and chronology.
    # Repeated financial values are never used as evidence or model evaluations.
    original = deepcopy(source["weekly"])
    first_actual = date.fromisoformat(original[0]["date"])
    added_weeks = 5 * 52
    synthetic = []
    for position in range(added_weeks):
        template = original[position % len(original)]
        week = first_actual - timedelta(weeks=added_weeks - position)
        synthetic.append(
            _shift_synthetic_dates(
                template, week - date.fromisoformat(template["date"])
            )
        )
    source["weekly"] = synthetic + original
    source["meta"]["generation_id"] = "synthetic-five-year-history-capacity-test"
    source["model"]["evidence_artifacts"]["weekly_state_forecasts"]["row_count"] = len(
        source["weekly"]
    )
    dates = [date.fromisoformat(row["date"]) for row in source["weekly"]]
    assert len(set(dates)) == len(dates)
    assert all(
        right - left == timedelta(weeks=1) for left, right in zip(dates, dates[1:])
    )
    assert source["weekly"][-len(original) :] == original
    raw = json.dumps(source, ensure_ascii=False).encode()
    core_raw, research_raw = build_dashboard_split(source, payload_raw=raw)
    core = json.loads(core_raw)
    files, bindings = build_history_chunks(source, payload_raw=raw)
    history, documents = [], []
    expected_generation = source["meta"]["generation_id"]
    expected_hash = hashlib.sha256(raw).hexdigest()
    assert core["history_sidecars"] == bindings
    assert core["generation_id"] == expected_generation
    assert core["source_payload_sha256"] == expected_hash
    research = json.loads(research_raw)
    assert research["generation_id"] == expected_generation
    assert research["source_payload_sha256"] == expected_hash
    for binding in bindings:
        chunk = files["data/" + binding["path"]]
        assert hashlib.sha256(chunk).hexdigest() == binding["sha256"]
        document = json.loads(chunk)
        assert document["generation_id"] == expected_generation
        assert document["source_payload_sha256"] == expected_hash
        assert document["weekly"][0]["date"] == binding["start"]
        assert document["weekly"][-1]["date"] == binding["end"]
        assert len(document["weekly"]) == binding["row_count"]
        history.extend(document["weekly"])
        documents.append(document)
    assert history + core["payload"]["weekly"] == source["weekly"]
    assert len(core["payload"]["weekly"]) == CORE_WEEK_COUNT
    assert len(core_raw) < 1_000_000
    assert max(len(x) for x in files.values()) < 1_500_000
    # Exercise the real browser history contract, not only Python serialisation.
    program = """
const fs = require('fs'), api = require('./web/app.js');
const fixture = JSON.parse(fs.readFileSync(0, 'utf8'));
const source = api.validateCoreEnvelope(fixture.core);
if (!source) throw new Error('synthetic capacity core rejected');
const rows = api.mergeHistoryParts(source, fixture.documents);
const rejects = (documents) => { try { api.mergeHistoryParts(source, documents); return false; } catch { return true; } };
const foreign = structuredClone(fixture.documents); foreign[0].generation_id = 'foreign-generation';
const duplicate = structuredClone(fixture.documents); duplicate[0].weekly[1].date = duplicate[0].weekly[0].date;
const differentHash = structuredClone(fixture.documents); differentHash[0].source_payload_sha256 = '0'.repeat(64);
console.log(JSON.stringify({
  exact: require('util').isDeepStrictEqual(rows, fixture.expected),
  count: rows.length,
  uniqueDates: new Set(rows.map(row => row.date)).size,
  foreignRejected: rejects(foreign),
  duplicateRejected: rejects(duplicate),
  sourceMismatchRejected: rejects(differentHash)
}));
"""
    process = subprocess.run(
        ["node", "-e", program],
        input=json.dumps(
            {"core": core, "documents": documents, "expected": source["weekly"]}
        ),
        text=True,
        capture_output=True,
        check=True,
        cwd=Path(__file__).parents[1],
    )
    browser = json.loads(process.stdout)
    assert browser["exact"]
    assert browser["count"] == browser["uniqueDates"] == len(source["weekly"])
    assert (
        browser["foreignRejected"]
        and browser["duplicateRejected"]
        and browser["sourceMismatchRejected"]
    )
    validate_history_chunks(files, payload=source, payload_raw=raw)
    tampered = dict(files)
    tampered[next(iter(tampered))] = b"{}"
    with pytest.raises(PublicContractError, match="history"):
        validate_history_chunks(tampered, payload=source, payload_raw=raw)
