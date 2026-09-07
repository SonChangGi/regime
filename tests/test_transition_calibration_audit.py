"""Independent v2 generation audit, version compatibility and replay safety."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from regime_lab.analysis.causal_calibration import TransitionCalibrator
from regime_lab.analysis.models import BenchmarkProfile
from regime_lab.analysis.validation import run_transition_benchmark

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load_script('audit_outputs')
replay = load_script('replay_transition_calibration_audit')


def source(raw=.8, horizon=13):
    origin = pd.date_range('2016-01-08', periods=390, freq='7D', tz='UTC')
    data = pd.DataFrame({'origin_date': origin, 'target_end': origin + pd.Timedelta(7 * horizon, unit='D'),
                         'raw_p_change': raw, 'actual_change': np.arange(390) % 5 == 0,
                         'horizon': horizon, 'model': 'fixture'})
    cutoff = pd.Timestamp('2023-01-01', tz='UTC')
    data['evaluation_split'] = np.where(data.target_end < cutoff, 'selection', 'retrospective_diagnostic')
    return data.loc[~((data.origin_date < cutoff) & (data.target_end >= cutoff))].copy()


def issued_row(data, origin='2026-09-04'):
    fit = TransitionCalibrator(data).fit(origin)
    probability, method, fallback, reason = fit.apply(.7)
    return pd.Series({'origin_date': pd.Timestamp(origin, tz='UTC'), 'raw_p_change': .7,
                      'p_change': probability, 'calibration_method': method,
                      'calibration_fallback': fallback, 'calibration_fallback_reason': reason,
                      **fit.metadata})


@pytest.mark.parametrize('raw', [.2, .8])
@pytest.mark.parametrize('origin', ['2016-07-01', '2018-06-08', '2022-12-30', '2026-09-04'])
def test_independent_auditor_matches_past_and_frozen_fits(raw, origin):
    data = source(raw=raw)
    row = issued_row(data, origin)
    # Auditor has its own fit and scoring code, not a call to this producer.
    audit.audit_transition_calibration_row(row, data, selection_end='2023-01-01', minimum_rows=12)


def test_csv_metadata_roundtrip_and_shrink_branch(tmp_path):
    data = source()
    recent = data.origin_date.ge(pd.Timestamp('2021-06-01', tz='UTC'))
    data.loc[recent, 'actual_change'] = np.arange(int(recent.sum())) % 5 < 3
    row = issued_row(data)
    assert row.calibration_method == 'prequential_shrunk_platt_logit'
    path = tmp_path / 'rows.csv'
    pd.DataFrame([row]).to_csv(path, index=False)
    readback = pd.read_csv(path).iloc[0]
    audit.audit_transition_calibration_row(readback, data, selection_end='2023-01-01', minimum_rows=12)


@pytest.mark.parametrize('field,value', [
    ('p_change', .01), ('calibration_selection_as_of', '2026-01-01'),
    ('calibration_selection_end_exclusive', '2024-01-01'),
    ('calibration_fit_last_target', '2026-09-04'),
    ('calibration_validation_last_target', '2023-01-01'),
    ('calibration_validation_rows', 999), ('calibration_platt_log_loss', .01),
    ('calibration_shrink_weight', .1), ('calibration_version', 'transition-calibration/99'),
])
def test_auditor_rejects_probability_causal_metadata_and_version_tampering(field, value):
    data = source()
    row = issued_row(data)
    row[field] = value
    with pytest.raises(audit.AuditFailure, match='calibration'):
        audit.audit_transition_calibration_row(row, data, selection_end='2023-01-01', minimum_rows=12)


def test_missing_version_cannot_silently_choose_legacy_replay():
    row = issued_row(source()).drop('calibration_version')
    with pytest.raises(audit.AuditFailure, match='version missing'):
        audit.audit_transition_calibration_row(row, source(), selection_end='2023-01-01', minimum_rows=12)


def test_v2_replay_does_not_call_the_producer(monkeypatch):
    data = source()
    row = issued_row(data)
    def forbidden(*args, **kwargs):
        raise AssertionError('audit reused the production calibrator')
    monkeypatch.setattr(TransitionCalibrator, 'fit', forbidden)
    audit.audit_transition_calibration_row(row, data, selection_end='2023-01-01', minimum_rows=12)


def test_auditor_cache_does_not_hide_changed_evidence():
    data = source()
    row = issued_row(data)
    cache = {}
    audit.audit_transition_calibration_row(row, data, selection_end='2023-01-01', minimum_rows=12, cache=cache)
    changed = data.copy()
    selection = changed.evaluation_split.eq('selection')
    changed.loc[selection, 'actual_change'] = ~changed.loc[selection, 'actual_change']
    with pytest.raises(audit.AuditFailure, match='calibration'):
        audit.audit_transition_calibration_row(row, changed, selection_end='2023-01-01', minimum_rows=12, cache=cache)
    diagnostic = data.evaluation_split.eq('retrospective_diagnostic')
    data.loc[diagnostic, 'actual_change'] = ~data.loc[diagnostic, 'actual_change']
    audit.audit_transition_calibration_row(row, data, selection_end='2023-01-01', minimum_rows=12, cache=cache)


@pytest.fixture(scope='module')
def benchmark():
    dates = pd.date_range('2018-01-05', periods=270, freq='W-FRI', tz='UTC')
    features = pd.DataFrame({'score': np.sin(np.arange(len(dates)) / 9)}, index=dates)
    states = pd.Series(np.asarray(['risk_on', 'transition', 'risk_off'])[(np.arange(len(dates)) // 11) % 3], index=dates)
    return run_transition_benchmark(features, states, models=('empirical_hazard', 'markov_hazard'),
                                    minimum_train_weeks=52, profile=BenchmarkProfile(name='test', max_origins=8, minimum_train_weeks=52, random_forest_trees=8, extra_trees=8, hist_gradient_iterations=12, svm_calibration_splits=2, hmm_iterations=10),
                                    selection_end='2022-01-01', selection_max_origins=8,
                                    minimum_selection_predictions=5, minimum_diagnostic_predictions=5,
                                    minimum_inner_predictions=5)


def test_real_benchmark_candidate_csv_passes_versioned_generation_audit(tmp_path, benchmark):
    predictions = benchmark.predictions
    for _, row in predictions.iterrows():
        history = predictions.loc[predictions.horizon.eq(row.horizon) & predictions.model.eq(row.model)]
        audit.audit_transition_calibration_row(row, history, selection_end='2022-01-01', minimum_rows=5)
    candidates = benchmark.latest_candidate_forecasts()
    candidates.to_csv(tmp_path / 'transition-candidate-forecasts.csv', index=False)
    result = audit.audit_transition_candidate_forecasts(
        {'model': {'transition_selection_end': '2022-01-01'}}, tmp_path,
        evaluated_predictions=predictions, candidate_models={'empirical_hazard', 'markov_hazard'},
        minimum_inner_predictions=5,
    )
    assert result == {'models': 2, 'rows': 36}
    candidates.loc[0, 'calibration_selection_as_of'] = '2026-01-01'
    candidates.to_csv(tmp_path / 'transition-candidate-forecasts.csv', index=False)
    with pytest.raises(audit.AuditFailure, match='metadata mismatch'):
        audit.audit_transition_candidate_forecasts(
            {'model': {'transition_selection_end': '2022-01-01'}}, tmp_path,
            evaluated_predictions=predictions, candidate_models={'empirical_hazard', 'markov_hazard'},
            minimum_inner_predictions=5,
        )


def test_atomic_report_preserves_last_valid_output_on_serialization_or_replace_failure(tmp_path, monkeypatch):
    target = tmp_path / 'audit.json'
    target.write_text('{"old":true}')
    with pytest.raises(ValueError):
        replay.write_atomic_report({'bad': float('nan')}, target)
    assert json.loads(target.read_text()) == {'old': True}
    original = Path.replace
    def fail_replace(self, destination):
        raise OSError('injected replacement failure')
    monkeypatch.setattr(Path, 'replace', fail_replace)
    with pytest.raises(OSError):
        replay.write_atomic_report({'new': True}, target)
    assert json.loads(target.read_text()) == {'old': True}
    assert not list(tmp_path.glob('*.tmp'))
    monkeypatch.setattr(Path, 'replace', original)
    replay.write_atomic_report({'new': True}, target)
    assert json.loads(target.read_text()) == {'new': True}


def test_replay_refuses_source_aliases_and_live_output(tmp_path):
    source_file = tmp_path / 'source.json'
    source_file.write_text('{}')
    link = tmp_path / 'linked.json'
    link.hardlink_to(source_file)
    for output in [source_file, link, ROOT / 'publication/live/other.json']:
        with pytest.raises(ValueError, match='cannot'):
            replay.validate_output_path(output, [source_file])
