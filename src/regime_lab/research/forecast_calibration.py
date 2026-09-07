"""Calibration comparison from one generation's in-memory derived forecasts."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from regime_lab.analysis.causal_calibration import (
    TransitionCalibrator, TRANSITION_CALIBRATION_VERSION,
    TRANSITION_CALIBRATION_BLOCK_WEEKS, TRANSITION_CALIBRATION_MAX_BLOCKS,
    TRANSITION_CALIBRATION_MIN_TRAIN_ROWS, TRANSITION_CALIBRATION_SHRINK_WEIGHT,
)
from regime_lab.v5 import _anchored_isotonic_transition_risk


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def build_calibration_audit_from_frames(payload: dict, data: pd.DataFrame,
                                       future: pd.DataFrame, *,
                                       selection_end: str | None = None,
                                       sources: list[dict] | None = None) -> dict:
    payload, data, future = deepcopy(payload), data.copy(deep=True), future.copy(deep=True)
    selection_end = selection_end or payload['model'].get('transition_selection_end', '2023-01-01')
    cutoff = pd.to_datetime(selection_end, utc=True)
    for frame in (data, future):
        for column in ('origin_date', 'target_end'):
            frame[column] = pd.to_datetime(frame[column], utc=True)
    latest_origin = pd.Timestamp(payload['meta']['data_as_of'])
    calibrators = {}
    choices = {}
    for key, group in data.groupby(['horizon', 'model']):
        calibrators[key] = TransitionCalibrator(group, selection_end=cutoff)
        fit = calibrators[key].fit()
        choices[key] = {'method': fit.method, 'fallback': fit.fallback,
                        'fallback_reason': fit.reason, **fit.metadata}
    all_rows = pd.concat([data, future], ignore_index=True)
    require(not all_rows.duplicated(['horizon', 'model', 'origin_date']).any(), 'duplicate derived forecast origin')
    all_rows['newly_selected'] = [
        calibrators[(r.horizon, r.model)].fit(r.origin_date).apply(r.raw_p_change)[0]
        for r in all_rows.itertuples(index=False)
    ]

    def scores(y, p):
        y = np.asarray(y, dtype=float)
        p = np.asarray(p, dtype=float)
        require(len(p) == len(y) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'invalid scoring probabilities or mismatched origins')
        clipped = np.clip(p, 1e-6, 1 - 1e-6)
        return {'log_loss': float(-(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)).mean()),
                'brier': float(((p - y) ** 2).mean())}

    def period(part, variants, actual='actual_change'):
        return {'n_origins': len(part), 'event_count': int(part[actual].astype(bool).sum()),
                'first_origin': pd.Timestamp(part.origin_date.min()).isoformat(),
                'last_origin': pd.Timestamp(part.origin_date.max()).isoformat(),
                'last_target': pd.Timestamp(part.target_end.max()).isoformat(),
                **{name: scores(part[actual], part[column]) for name, column in variants.items()}}

    model_results = []
    latest_week = payload['weekly'][-1]
    for (horizon, model), group in all_rows.groupby(['horizon', 'model'], sort=True):
        current = group.loc[group.origin_date.eq(latest_origin)]
        require(len(current) == 1, 'each model/horizon must have exactly one latest forecast')
        current = current.iloc[0]
        archived = latest_week['transition_risk'][f'{horizon}w']
        recorded_final = float(archived['probability']) if archived['model'] == model else None
        historical = {}
        for split, part in group.loc[group.evaluation_split.ne('prospective')].groupby('evaluation_split'):
            historical[split] = period(part, {'stored_calibrated': 'p_change',
                                              'identity': 'raw_p_change',
                                              'newly_selected': 'newly_selected'})
        model_results.append({
            'model': model, 'horizon_weeks': int(horizon),
            'calibration_choice': choices[(horizon, model)],
            'historical': historical,
            'latest': {'origin_date': current.origin_date.isoformat(),
                       'target_end': current.target_end.isoformat(),
                       'actual_change': None, 'evaluation_split': 'prospective',
                       'stored_calibrated': float(current.p_change),
                       'identity': float(current.raw_p_change),
                       'newly_selected': float(current.newly_selected),
                       'stored_final': recorded_final,
                       'stored_final_available_for_this_model': recorded_final is not None},
        })

    index = all_rows.set_index(['horizon', 'model', 'origin_date'])
    comparison_rows = []
    coherence_violations = 0
    for week in payload['weekly']:
        origin = pd.Timestamp(week['data_as_of'])
        original = week['transition_risk']
        stored = {key: dict(value) for key, value in original.items()}
        variants = {name: {key: dict(value) for key, value in original.items()}
                    for name in ('identity', 'newly_selected')}
        observations = {}
        for horizon in (4, 13):
            model = original[f'{horizon}w']['model']
            row = index.loc[(horizon, model, origin)]
            require(row.target_end.date().isoformat() == original[f'{horizon}w']['target_end'],
                    'source/OOS target differs from archived issued target')
            stored[f'{horizon}w']['probability'] = float(row.p_change)
            observations[horizon] = row
            variants['identity'][f'{horizon}w']['probability'] = float(row.raw_p_change)
            variants['newly_selected'][f'{horizon}w']['probability'] = float(row.newly_selected)
        stored_projected, _ = _anchored_isotonic_transition_risk(stored)
        require(all(np.isclose(stored_projected[key]['probability'], original[key]['probability'],
                               atol=1e-7, rtol=0) for key in ('1w', '4w', '13w')),
                'source/OOS probabilities do not reproduce the archived issued generation')
        projected = {}
        for name, risk in variants.items():
            projected[name], _ = _anchored_isotonic_transition_risk(risk)
            p = [projected[name][f'{h}w']['probability'] for h in (1, 4, 13)]
            coherence_violations += int(not p[0] <= p[1] <= p[2])
            require(p[0] == original['1w']['probability'], 'official one-week anchor changed')
        for horizon in (4, 13):
            key = f'{horizon}w'
            row = observations[horizon]
            comparison_rows.append({
                'origin_date': origin.isoformat(), 'target_end': row.target_end.isoformat(),
                'horizon_weeks': horizon, 'model': original[key]['model'],
                'actual_change': None if pd.isna(row.actual_change) else bool(row.actual_change),
                'stored_final': float(original[key]['probability']),
                'identity': float(projected['identity'][key]['probability']),
                'newly_selected': float(projected['newly_selected'][key]['probability']),
                'identity_before_projection': float(variants['identity'][key]['probability']),
                'newly_selected_before_projection': float(variants['newly_selected'][key]['probability']),
            })
    require(coherence_violations == 0, 'coherence projection failed')
    comparison = pd.DataFrame(comparison_rows)
    published = []
    for horizon, group in comparison.groupby('horizon_weeks'):
        matured = group.loc[group.actual_change.notna()]
        published.append({
            'horizon_weeks': int(horizon), 'models': sorted(group.model.unique()),
            'historical': period(matured, {name: name for name in ('stored_final', 'identity', 'newly_selected')}),
            'latest': group.iloc[-1].to_dict(),
            'history': group.to_dict('records'),
        })
    # A compact UI contract keeps OLD issued values distinct from NEW calibration.
    flat_rows = []
    latest_rows = []
    for model_result in model_results:
        horizon, model = model_result['horizon_weeks'], model_result['model']
        public = next((item for item in published if item['horizon_weeks'] == horizon
                       and item['models'] == [model]), None)
        for split, metrics in model_result['historical'].items():
            final = public['historical'] if public is not None and split == 'retrospective_diagnostic' else None
            if final is not None:
                require(final['n_origins'] == metrics['n_origins'], 'final comparison origin count differs')
                require(final['first_origin'] == metrics['first_origin'] and final['last_origin'] == metrics['last_origin'], 'final comparison origin range differs')
            flat_rows.append({
                'model': model, 'horizon_weeks': horizon, 'evaluation_split': split,
                'n_predictions': metrics['n_origins'],
                'raw_log_loss': metrics['identity']['log_loss'],
                'raw_brier': metrics['identity']['brier'],
                'calibrated_log_loss': metrics['newly_selected']['log_loss'],
                'calibrated_brier': metrics['newly_selected']['brier'],
                'final_log_loss': None if final is None else final['newly_selected']['log_loss'],
                'final_brier': None if final is None else final['newly_selected']['brier'],
                'previous_calibrated_log_loss': metrics['stored_calibrated']['log_loss'],
                'previous_calibrated_brier': metrics['stored_calibrated']['brier'],
                'previous_published_log_loss': None if final is None else final['stored_final']['log_loss'],
                'previous_published_brier': None if final is None else final['stored_final']['brier'],
                'identity_final_log_loss': None if final is None else final['identity']['log_loss'],
                'identity_final_brier': None if final is None else final['identity']['brier'],
                'selected_method': model_result['calibration_choice']['method'] if split == 'retrospective_diagnostic' else 'past_block_selection',
                'latest_selected_method': model_result['calibration_choice']['method'],
                'final_available': final is not None,
            })
        latest = model_result['latest']
        latest_rows.append({
            'model': model, 'horizon_weeks': horizon, 'origin_date': latest['origin_date'],
            'target_end': latest['target_end'], 'selected_method': model_result['calibration_choice']['method'],
            'raw_probability': latest['identity'],
            'calibrated_probability': latest['newly_selected'],
            'final_probability': None if public is None else public['latest']['newly_selected'],
            'previous_calibrated_probability': latest['stored_calibrated'],
            'previous_published_probability': latest['stored_final'],
            'identity_final_probability': None if public is None else public['latest']['identity'],
        })
    report = {
        'schema_version': 'regime-calibration-audit/1',
        'calibration_version': TRANSITION_CALIBRATION_VERSION,
        'status': 'completed', 'role': 'research_preview_not_issued_forecast',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'data_as_of': payload['meta']['data_as_of'],
        'source_generation_id': payload['meta']['generation_id'],
        'selection_end_exclusive': cutoff.isoformat(),
        'evidence_scope': 'Fixed-policy replay on already inspected diagnostic data; no new holdout claim.',
        'policy': {
            'candidates': {'identity': 0.0, 'prequential_shrunk_platt_logit': TRANSITION_CALIBRATION_SHRINK_WEIGHT,
                           'prequential_platt_logit': 1.0},
            'platt_C': .1, 'platt_input': 'logit_raw_probability',
            'selection_metric': 'binary_log_loss', 'clip_epsilon': 1e-6,
            'block_weeks': TRANSITION_CALIBRATION_BLOCK_WEEKS,
            'block_anchor': '2000-01-07T00:00:00+00:00',
            'maximum_validation_blocks': TRANSITION_CALIBRATION_MAX_BLOCKS,
            'minimum_fit_rows': TRANSITION_CALIBRATION_MIN_TRAIN_ROWS,
            'minimum_validation_rows': 26, 'minimum_events_and_nonevents': 3,
            'purge': 'inner_training_target_end < validation_block_start; choice_target_end < selection_as_of',
            'diagnostic_use_for_selection_or_fit': False,
            'post_cutoff_policy': 'choice_and_coefficients_frozen',
        },
        'comparison_definitions': {
            'model_horizon_results': 'Same origins for raw identity, stored calibrated (before coherence), and prequentially selected calibration.',
            'published_horizon_comparison': 'Same matured origins; archived model identities and official one-week anchor fixed; identity and newly_selected each apply existing 1w-anchored coherence jointly to 4w and 13w.',
            'stored_final': 'Actual archived issued probability; null for model/horizon pairs never issued in the archived latest week.',
            'latest': 'Prospective probability, without an actual outcome or performance score.',
        },
        'selection_protocol': f'고정 identity·Platt·25% 축소 후보를 이전 최대 3개 26주 블록의 Log loss로 선택한다. 각 내부 적합은 해당 검증 블록 시작보다 앞선 target만 사용하며, 선택 및 계수는 {cutoff.date().isoformat()}에 동결한다. 진단 구간은 선택에 사용하지 않는다.',
        'rows': flat_rows,
        'latest_rows': latest_rows,
        'model_horizon_results': model_results,
        'published_horizon_comparison': published,
        'verification': {'derived_oos_rows': len(data), 'prospective_candidate_rows': len(future),
                         'model_horizon_pairs': len(model_results),
                         'weekly_coherence_origins': len(payload['weekly']),
                         'coherence_violations': coherence_violations,
                         'one_week_anchor_unchanged': True,
                         'source_bytes_unchanged': True,
                         'base_models_refit': False, 'live_publication_updated': False},
        'sources': sources or [],
    }
    return report
