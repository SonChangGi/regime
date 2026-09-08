from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import hashlib
import json
import sqlite3

import pytest

from regime_lab.forecast_ledger import ForecastLedger
from regime_lab.integrity import canonical_json_sha256_v1
from regime_lab.operational_forecast import frame_sha256
from regime_lab import local_forecast_workflow as workflow
from test_forecast_probability import _issued
from test_forecast_ledger import _states


@pytest.fixture
def local_copy(tmp_path):
    source = tmp_path / 'source.sqlite3'
    entry = _issued()
    with ForecastLedger(source, clock=lambda: entry.decision_at) as ledger:
        ledger.append(entry)
    before = source.read_bytes()
    directory = tmp_path / 'preview'
    receipt = workflow.initialize_local_copy(source, directory)
    return source, before, directory, entry, receipt


def test_copy_score_revision_and_summary_never_change_source(local_copy, monkeypatch):
    source, before, directory, entry, receipt = local_copy
    monkeypatch.setattr(workflow, '_now', lambda: entry.target_at + timedelta(days=1))
    states = _states(entry.origin_week, entry.target_at.date())
    manifest = {'frames':{'states':frame_sha256(states)}}
    result = workflow.score_local_copy(directory, states=states, state_manifest=manifest,
        label_spec_sha256=entry.label_spec_sha256)
    assert result['operational_diagnostics']['probability_maturity']['coverage']['completed_entries'] == 1
    states.iloc[-1] = 'risk_off'
    manifest['frames']['states'] = frame_sha256(states)
    result = workflow.score_local_copy(directory, states=states, state_manifest=manifest,
        label_spec_sha256=entry.label_spec_sha256)
    assert result['operational_diagnostics']['probability_maturity']['revision_conflicts']
    summary = workflow.local_summary(directory)
    assert summary['operational_diagnostics']['probability_maturity']['coverage']['status'] == 'needs_attention'
    assert source.read_bytes() == before
    assert receipt['source_sha256'] == hashlib.sha256(before).hexdigest()
    with sqlite3.connect(source) as db:
        assert db.execute('SELECT count(*) FROM forecast_probability_evaluations').fetchone()[0] == 0


def test_unmarked_or_redirected_workspace_is_rejected(local_copy, tmp_path):
    source, before, directory, entry, receipt = local_copy
    with pytest.raises(FileNotFoundError):
        workflow.local_summary(tmp_path)
    receipt['ledger'] = str(source)
    (directory / 'local-workspace.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='identity'):
        workflow.local_summary(directory)
    assert source.read_bytes() == before


def test_local_issue_rejects_replay_and_freezes_only_in_copy(local_copy, monkeypatch):
    source, before, directory, entry, receipt = local_copy
    now = entry.decision_at + timedelta(hours=1)
    monkeypatch.setattr(workflow, '_now', lambda: now)
    monkeypatch.setattr(workflow, 'validate_preparation_recipe', lambda *args: None)
    lock = {'sha256':'d'*64}
    payload = {'model':{'candidate_manifest_sha256':entry.model_manifest_sha256}, 'label':{'spec_sha256':entry.label_spec_sha256}}
    prepared = {'key_sha256':'e'*64, 'research_replay':False, 'local_issue_eligible':True,
        'status':'prepared_for_review', 'recipe_lock_sha256':lock['sha256'], 'prepared_at':now.isoformat(),
        'origin_at':entry.decision_at.isoformat(), 'target_at':entry.target_at.isoformat(),
        'issue_deadline_at':(entry.decision_at+timedelta(hours=60)).isoformat(),
        'input_cutoff_at':entry.decision_at.isoformat(), 'input_hashes':{},
        'champion':entry.forecast['selection']['operating_champion'], 'forecast':entry.forecast['model_forecasts'][0],
        'model_forecasts':entry.forecast['model_forecasts']}
    prepared['document_sha256'] = canonical_json_sha256_v1(prepared)
    replay = deepcopy(prepared)
    replay['research_replay'] = True
    replay['document_sha256'] = canonical_json_sha256_v1({k:v for k,v in replay.items() if k != 'document_sha256'})
    with pytest.raises(ValueError, match='fresh'):
        workflow.issue_local_preparation(directory, prepared=replay, recipe_lock=lock, locked_payload=payload)
    issued = workflow.issue_local_preparation(directory, prepared=prepared, recipe_lock=lock, locked_payload=payload)
    assert issued['status'] == 'issued_local_preview'
    assert workflow.issue_local_preparation(directory, prepared=prepared, recipe_lock=lock, locked_payload=payload)['status'] == 'already_issued'
    with ForecastLedger(directory/workflow.LEDGER_NAME) as ledger:
        assert len(ledger.list_entries()) == 2
        assert ledger.list_entries()[-1].forecast['evidence_track'] == 'local_preview'
    assert source.read_bytes() == before


def test_score_requires_an_independent_state_hash(local_copy):
    _, _, directory, entry, _ = local_copy
    with pytest.raises(ValueError, match='independent'):
        workflow.score_local_copy(directory, states=_states(entry.target_at.date()),
            state_manifest={'frames':{'states':'0'*64}}, label_spec_sha256=entry.label_spec_sha256)
