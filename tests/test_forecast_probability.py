from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import sqlite3

import pytest

from regime_lab.forecast_ledger import (ForecastLedger, ForecastLedgerError,
    ForecastEvaluationEntry, mature_forecast_evaluations, build_operational_diagnostics)
from regime_lab.forecast_probability import (append_probability_evaluation,
    mature_probability_evaluations, make_probability_evaluation, probability_summary)
from test_forecast_ledger import _v2_entry, _price_panel, _states
from test_operational_diagnostics import _completed


def _issued():
    entry = _v2_entry()
    document = deepcopy(entry.forecast)
    document['local_publication_at'] = entry.decision_at.isoformat()
    document['evidence_track'] = 'operational_oos'
    return replace(entry, forecast=document)


@pytest.mark.parametrize('execution', ['complete', 'missing_distribution', 'legacy'])
def test_probability_matures_without_execution_dependency_and_remains_immutable(execution):
    entry = _issued()
    if execution == 'legacy':
        document = deepcopy(entry.forecast)
        document.pop('decision_shadow')
        entry = replace(entry, forecast=document)
    prices = _price_panel(entry.origin_week, entry.target_at.date())
    if execution == 'missing_distribution':
        prices = prices.drop(columns=['tlt_dividend_amount'])
    states = _states(entry.origin_week, entry.target_at.date())
    with ForecastLedger(':memory:', clock=lambda: entry.decision_at) as ledger:
        ledger.append(entry)
        mature_forecast_evaluations(ledger, canonical=prices, states=states, evaluated_at=entry.target_at)
        results = ledger.list_probability_evaluations()
        assert len(results) == 1
        scores = probability_summary(results)
        assert scores['completed_weeks'] == scores['prospective_completed_weeks'] == 1
        assert scores['log_loss'] == pytest.approx(-__import__('math').log(.6))
        assert scores['brier'] == pytest.approx(.26)
        assert len(ledger.list_evaluations()) == (0 if execution == 'missing_distribution' else 1)
        mature_forecast_evaluations(ledger, canonical=prices, states=states, evaluated_at=entry.target_at + timedelta(days=1))
        assert ledger.list_probability_evaluations() == results
        with pytest.raises(sqlite3.IntegrityError, match='append-only'):
            ledger._connection.execute('DELETE FROM forecast_probability_evaluations')
        ledger._connection.rollback()


def test_late_storage_and_missing_receipts_are_not_prospective():
    entry = _issued()
    with ForecastLedger(':memory:', clock=lambda: entry.target_at + timedelta(days=1)) as ledger:
        ledger.append(entry)
        mature_probability_evaluations(ledger, states=_states(entry.target_at.date()), evaluated_at=entry.target_at + timedelta(days=1))
        results = ledger.list_probability_evaluations()
        summary = probability_summary(results)
        assert summary['completed_weeks'] == 1
        assert summary['prospective_completed_weeks'] == 0
        assert summary['log_loss'] is None
        diag = build_operational_diagnostics(ledger.list_entries(), [], probability_evaluations=results)
        assert diag['timing']['on_time_entries'] == 0
        assert 'deadline_missed' in diag['timing']['rows'][0]['issue_evidence']['reasons']
    diag = build_operational_diagnostics([entry], [])
    assert diag['timing']['on_time_entries'] == 0
    assert 'missing_storage_receipt' in diag['timing']['rows'][0]['issue_evidence']['reasons']


def test_reference_hash_must_match_before_terminal_insert():
    entry = _issued()
    evaluation = _completed(entry)
    document = deepcopy(evaluation.evaluation)
    document['forecast_sha256'] = '0' * 64
    invalid = ForecastEvaluationEntry(entry.key, evaluation.evaluated_at, 'completed', document)
    with ForecastLedger(':memory:', clock=lambda: entry.decision_at) as ledger:
        ledger.append(entry)
        with pytest.raises(ForecastLedgerError, match='reference hash'):
            ledger.append_evaluation(invalid)
        ledger.append_evaluation(evaluation)
        assert ledger.read_evaluation(entry.key) == evaluation
        original = ledger.read(entry.key)
        probability = make_probability_evaluation(original, actual='risk_on', evaluated_at=entry.target_at)
        probability['forecast_sha256'] = '0' * 64
        with pytest.raises(ForecastLedgerError, match='reference hash'):
            append_probability_evaluation(ledger, probability)
        assert not ledger.list_probability_evaluations()


def test_probability_maturity_respects_target_time_and_label_version():
    entry = _issued()
    with ForecastLedger(':memory:', clock=lambda: entry.decision_at) as ledger:
        ledger.append(entry)
        states = _states(entry.target_at.date())
        before = mature_probability_evaluations(ledger, states=states, evaluated_at=entry.target_at-timedelta(seconds=1))
        assert before['pending_count'] == 1
        assert not ledger.list_probability_evaluations()
        mismatch = mature_probability_evaluations(ledger, states=states, evaluated_at=entry.target_at, label_spec_sha256='0'*64)
        assert mismatch['unresolved'][0]['reason'] == 'label_spec_mismatch'
        assert not ledger.list_probability_evaluations()


def test_republication_cannot_inflate_week_count_or_select_better_realized_score():
    first = _issued()
    second_document = deepcopy(first.forecast)
    second_document['model_forecasts'][0]['probabilities'] = {'risk_on':.99,'transition':.005,'risk_off':.005}
    second_document['local_publication_at'] = (first.decision_at + timedelta(hours=1)).isoformat()
    second = replace(first, decision_at=first.decision_at+timedelta(hours=1),
                     input_snapshot_sha256='e'*64, forecast=second_document)
    docs=[]
    for entry in (first,second):
        stored=replace(entry, inserted_at=entry.decision_at)
        docs.append(make_probability_evaluation(stored, actual='risk_on', evaluated_at=entry.target_at))
    scores=probability_summary(list(reversed(docs)))
    assert scores['completed_entries']==2
    assert scores['completed_weeks']==scores['prospective_completed_weeks']==1
    assert scores['duplicate_target_entries']==1
    assert scores['log_loss']==pytest.approx(-__import__('math').log(.6))
    # Deduplicating a week must not conceal a different definition of its label.
    alternate_label = deepcopy(docs[1])
    alternate_label['label_spec_sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='one label specification'):
        probability_summary([docs[0], alternate_label])


def test_revised_actual_labels_are_reported_without_rewriting_first_scores():
    entry = _issued()
    states = _states(entry.origin_week, entry.target_at.date())
    with ForecastLedger(':memory:', clock=lambda: entry.decision_at) as ledger:
        ledger.append(entry)
        first = mature_probability_evaluations(ledger, states=states, evaluated_at=entry.target_at,
            label_spec_sha256=entry.label_spec_sha256)
        frozen = ledger.list_probability_evaluations()
        assert frozen[0]['schema_version'] == 'regime-probability-evaluation/2'
        assert frozen[0]['label_snapshot']['sha256'] == first['label_snapshot']['sha256']
        states.iloc[-1] = 'risk_off'
        for _ in range(2):
            result = mature_probability_evaluations(ledger, states=states, evaluated_at=entry.target_at,
                label_spec_sha256=entry.label_spec_sha256)
            assert result['coverage']['status'] == 'needs_attention'
            assert result['unresolved'][0]['reason'] == 'official_label_revision'
            assert result['revision_conflicts'][0]['first_scores_preserved']
        assert ledger.list_probability_evaluations() == frozen
        assert len(ledger.list_probability_revisions()) == 1
        with pytest.raises(sqlite3.IntegrityError, match='append-only'):
            ledger._connection.execute('DELETE FROM forecast_probability_label_revisions')
        ledger._connection.rollback()


def test_probability_failure_propagates_even_when_investment_evaluation_completes():
    entry = _issued()
    document = deepcopy(entry.forecast)
    document['model_forecasts'].append({'model':'invalid_extra_baseline', 'date':entry.target_at.date().isoformat(),
        'probabilities':{'risk_on':-1,'transition':1,'risk_off':1}})
    entry = replace(entry, forecast=document)
    with ForecastLedger(':memory:', clock=lambda: entry.decision_at) as ledger:
        ledger.append(entry)
        report = mature_forecast_evaluations(ledger, canonical=_price_panel(entry.origin_week, entry.target_at.date()),
            states=_states(entry.origin_week, entry.target_at.date()), evaluated_at=entry.target_at)
        assert report.appended[0].status == 'completed'
        assert report.probability_maturity['coverage']['missing_entries'] == 1
        assert report.probability_maturity['unresolved'][0]['reason'].startswith('probability_contract:')
        diagnostics = build_operational_diagnostics(ledger.list_probability_forecasts(), ledger.list_evaluations(),
            probability_evaluations=[], probability_maturity=report.probability_maturity, as_of=entry.target_at)
        assert diagnostics['probability_maturity']['coverage']['status'] == 'needs_attention'
        assert diagnostics['probability_scores']['completed_weeks'] == 0


def test_fixed_deadline_and_legacy_lead_cohorts_are_separate():
    from regime_lab.forecast_probability import issue_evidence
    entry = _issued()
    deadline = entry.decision_at + timedelta(hours=60)
    document = deepcopy(entry.forecast)
    document['issue_deadline_at'] = deadline.isoformat()
    document['input_cutoff_at'] = entry.decision_at.isoformat()
    fixed = replace(entry, forecast=document, inserted_at=entry.decision_at)
    snapshot = {'sha256':'f'*64, 'available_at':entry.target_at.isoformat(), 'label_spec_sha256':entry.label_spec_sha256}
    score = make_probability_evaluation(fixed, actual='risk_on', evaluated_at=entry.target_at, label_snapshot=snapshot)
    assert probability_summary([score])['issuance_cohorts']['fixed_deadline']['weeks'] == 1
    assert score['issuance']['remaining_lead_hours'] == 168
    assert not issue_evidence(replace(fixed, inserted_at=deadline))['eligible']
    legacy = replace(entry, inserted_at=entry.target_at-timedelta(seconds=1))
    legacy_score = make_probability_evaluation(legacy, actual='risk_on', evaluated_at=entry.target_at)
    summary = probability_summary([legacy_score])
    assert summary['issuance_cohorts']['legacy_target_deadline']['weeks'] == 1
    assert summary['rows'][0]['remaining_lead_hours'] == pytest.approx(1/3600)


def test_legacy_immutable_evaluation_stays_byte_equivalent():
    entry = _issued()
    with ForecastLedger(':memory:', clock=lambda: entry.decision_at) as ledger:
        ledger.append(entry)
        legacy = make_probability_evaluation(ledger.read(entry.key), actual='risk_on', evaluated_at=entry.target_at)
        append_probability_evaluation(ledger, legacy)
        mature_probability_evaluations(ledger, states=_states(entry.target_at.date()), evaluated_at=entry.target_at)
        assert ledger.list_probability_evaluations() == (legacy,)
