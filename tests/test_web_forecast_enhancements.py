"""Forecast exploration keeps target, origin, and evidence scope aligned."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def run_js(source):
    result = subprocess.run(["node", "-"], input=source, text=True, capture_output=True, cwd=ROOT, check=True)
    return json.loads(result.stdout)


FIXTURE = r"""
const api = require('./web/forecast-enhancements.js');
const prediction = (model, origin, horizon, probabilities) => ({ model, origin_date:origin,
  target_date:new Date(Date.parse(origin)+horizon*7*86400000).toISOString(),horizon_weeks:horizon,
  probabilities,current_state:'risk_on',worsening_probability:1-probabilities.risk_on,recovery_probability:0 });
const a = prediction('a','2026-09-04T20:00:00Z',1,{risk_on:.8,transition:.15,risk_off:.05});
const b = prediction('b','2026-09-04T20:00:00Z',1,{risk_on:.2,transition:.3,risk_off:.5});
const long = prediction('a','2026-09-04T20:00:00Z',4,{risk_on:.4,transition:.35,risk_off:.25});
const old = prediction('a','2026-08-28T20:00:00Z',1,{risk_on:.6,transition:.3,risk_off:.1});
const payload={meta:{generation_id:'generation-a',data_as_of:'2026-09-04T20:00:00Z'}};
const data={schema_version:'regime-forecast-enhancements/1',source_generation_id:'generation-a',
 data_as_of:payload.meta.data_as_of,protocol:{display_model:'a'},latest:[a,b,long],history:[old],
 model_metrics:[{model:'a',horizon_weeks:1,n_predictions:51,log_loss:.5,evaluation_split:'holdout'},
 {model:'a',horizon_weeks:4,n_predictions:48,log_loss:.8,evaluation_split:'holdout'}],
 alerts:{history:[{model:'a',origin_date:old.origin_date,horizon_weeks:1,policy:'test',alert:false}],
 latest:[{model:'a',origin_date:a.origin_date,horizon_weeks:1,policy:'test',alert:true}]}};
"""


def test_model_horizon_and_origin_select_distinct_forecasts_and_matched_evidence():
    value = run_js(FIXTURE + """
const one=api.view(data,{week:'2026-09-04',model:'a',horizon:1});
const four=api.view(data,{week:'2026-09-04',model:'a',horizon:4});
const second=api.view(data,{week:'2026-09-04',model:'b',horizon:1});
const older=api.view(data,{week:'2026-08-28',model:'a',horizon:1});
const missing=api.view(data,{week:'2026-08-21',model:'a',horizon:1});
console.log(JSON.stringify({one,four,second,old:older,missing,validation:api.validate(data,payload)}));
""")
    assert value['validation'] == []
    assert value['one']['forecast']['probabilities']['risk_on'] == .8
    assert value['four']['forecast']['probabilities']['risk_on'] == .4
    assert value['four']['metrics'][0]['n_predictions'] == 48
    assert value['second']['forecast']['probabilities']['risk_off'] == .5
    assert value['old']['forecast']['probabilities']['risk_on'] == .6
    assert value['old']['alertForecasts'][0]['alert'] is False
    assert value['one']['alertForecasts'][0]['alert'] is True
    assert value['missing']['forecast'] is None
    assert value['missing']['alertForecasts'] == []


def test_foreign_generation_cutoff_bad_probabilities_and_misaligned_horizon_are_rejected():
    value = run_js(FIXTURE + """
const changes=[d=>d.source_generation_id='foreign',d=>d.data_as_of='2026-08-28T20:00:00Z',
 d=>d.latest[0].probabilities.risk_on=.9,d=>d.latest[0].target_date='2026-09-18T20:00:00Z',
 d=>d.latest[0].worsening_probability=-.1];
console.log(JSON.stringify(changes.map(change=>{const copy=structuredClone(data);change(copy);return api.validate(copy,payload)})));
""")
    assert all(value)


def test_verified_operational_sidecar_overrides_both_displays_without_mutating_payload():
    value = run_js(FIXTURE + """
globalThis.REGIME_ENHANCEMENTS=api;
const dashboard=require('./web/app.js');
payload.research={operational_diagnostics:{probability_scores:{prospective_completed_weeks:0},source:'original'}};
data.operational={probability_scores:{prospective_completed_weeks:2},source:'verified-sidecar'};
const before=JSON.stringify(payload);
const unloaded=dashboard.operationalDiagnosticsForPayload(payload);
globalThis.fetch=async()=>({status:200,ok:true,json:async()=>data});
(async()=>{
  let notified=0;
  await api.update({payload,onDataReady:()=>{notified++;}});
  await new Promise(resolve=>setImmediate(resolve));
  const loaded=dashboard.operationalDiagnosticsForPayload(payload);
  const foreign=dashboard.operationalDiagnosticsForPayload({...payload,meta:{...payload.meta,generation_id:'foreign'}});
  const stale=dashboard.operationalDiagnosticsForPayload({...payload,meta:{...payload.meta,data_as_of:'2026-08-28T20:00:00Z'}});
  console.log(JSON.stringify({unloaded,loaded,foreign,stale,notified,unchanged:JSON.stringify(payload)===before}));
})();
""")
    assert value['unloaded']['source'] == 'original'
    assert value['loaded']['probability_scores']['prospective_completed_weeks'] == 2
    assert value['foreign']['source'] == value['stale']['source'] == 'original'
    assert value['notified'] == 1 and value['unchanged']
    app = (ROOT/'web/app.js').read_text()
    assert 'const issued = operationalDiagnosticsForPayload(state.raw)?.probability_scores;' in app
    assert 'if (operational) renderOperationalDiagnostics(operational, container);' in app
    assert 'onDataReady: () => { renderSelectedWeek(); renderResearchUpgrades(); }' in app


def test_absent_operational_extension_keeps_original_diagnostics():
    value = run_js(FIXTURE + """
globalThis.REGIME_ENHANCEMENTS=api;
const dashboard=require('./web/app.js');
payload.research={operational_diagnostics:{source:'original'}};
globalThis.fetch=async()=>({status:200,ok:true,json:async()=>data});
(async()=>{
  await api.update({payload});
  await new Promise(resolve=>setImmediate(resolve));
  console.log(JSON.stringify({loaded:!!api.getData(payload),value:dashboard.operationalDiagnosticsForPayload(payload)}));
})();
""")
    assert value['loaded'] and value['value']['source'] == 'original'


def test_new_and_long_horizon_models_do_not_inherit_one_week_selection_metrics():
    value = run_js(FIXTURE + """
const insights=require('./web/insights.js');
payload.model={leaderboard:[{name:'a',rank:7,selection_log_loss:.14,selection_calibration_error:.02}],
 forecast_comparison:{models:['a']}};
payload.forecast_enhancements=data;
payload.weekly=[{date:'2026-09-04',current:{state:'risk_on'}}];
console.log(JSON.stringify({one:insights.forecastEvaluation(payload,{horizon:1}).leaderboard,
 four:insights.forecastEvaluation(payload,{horizon:4}).leaderboard}));
""")
    one = {row['name']:row for row in value['one']}
    assert one['a']['rank'] == 7 and one['a']['selection_log_loss'] == .14
    assert one['b']['selection_log_loss'] is None
    assert value['four'][0]['selection_log_loss'] is None
    assert value['four'][0]['selection_calibration_error'] is None


def test_terminal_probability_does_not_invent_path_entry_probability():
    value = run_js(FIXTURE + """
console.log(JSON.stringify(api.view(data,{week:'2026-09-04',model:'a',horizon:4}).forecast));
""")
    assert 'first_departure_probability' not in value
    assert 'risk_off_entry_probability' not in value


def test_available_km_probability_intervals_follow_the_selected_week():
    value = run_js("""
const api=require('./web/insights.js');
const v=api.transitionOutlook({duration_context:{departure_probability:{'4w':.4},
ci95:{departure_probability:{'4w':{lower:.2,upper:.6}}}}});
console.log(JSON.stringify(v));
""")
    assert value[0]['baselineLower'] == .2
    assert value[0]['baselineUpper'] == .6
    assert value[1]['baselineLower'] is None


def test_operating_departure_calibration_is_separate_from_endpoint_model_selection():
    value = run_js(FIXTURE + """
data.calibration={metrics:[{model:'adaptive_shrink_after_coherence',horizon_weeks:4,n_predictions:48}],
 history:[{model:'adaptive_shrink_after_coherence',origin_date:a.origin_date,horizon_weeks:4,probability:.42}]};
console.log(JSON.stringify(api.view(data,{week:'2026-09-04',model:'a',horizon:4})));
""")
    assert value['calibration'][0]['model'] == 'adaptive_shrink_after_coherence'
    assert value['calibrationForecast'][0]['probability'] == .42
    assert value['alerts'] == []  # This policy has a one-week worsening target.


def test_duplicate_histories_are_rejected_but_latest_can_repeat_the_last_history():
    value = run_js(FIXTURE + """
data.history.push(a);
const valid=api.validate(data,payload);
data.history.push(a);
console.log(JSON.stringify({valid,duplicate:api.validate(data,payload)}));
""")
    assert value['valid'] == []
    assert value['duplicate']


def test_event_calibration_and_economic_diagnostics_keep_their_own_targets():
    value = run_js(FIXTURE + """
data.event_metrics=[{model:'a',horizon_weeks:1,target:'first_departure'},
 {model:'b',horizon_weeks:1,target:'first_departure'},
 {model:'a',horizon_weeks:4,target:'endpoint_risk_off'}];
data.calibration_health=[{model:'a',horizon_weeks:4,target:'worsening',current_state:'risk_on'},
 {model:'b',horizon_weeks:4,target:'worsening',current_state:'risk_on'}];
data.economics={rows:[{model:'a',horizon_weeks:4,stratum:'risk_on'}],
 incremental:{metrics:[{model:'downside_state_only',horizon_weeks:4},
 {model:'downside_state_plus_a',horizon_weeks:4},{model:'downside_state_only',horizon_weeks:13}]}};
const one=api.view(data,{week:'2026-09-04',model:'a',horizon:1});
const four=api.view(data,{week:'2026-09-04',model:'a',horizon:4});
console.log(JSON.stringify({one,four}));
""")
    assert [row['target'] for row in value['one']['eventMetrics']] == ['first_departure']
    assert [row['target'] for row in value['four']['eventMetrics']] == ['endpoint_risk_off']
    assert [row['model'] for row in value['four']['calibrationHealth']] == ['a']
    assert [row['model'] for row in value['four']['economicIncremental']] == [
        'downside_state_only', 'downside_state_plus_a',
    ]
    assert value['one']['economicHorizon'] == 4
    assert value['four']['economics'][0]['stratum'] == 'risk_on'


def test_selected_date_evaluation_excludes_not_yet_mature_targets_and_pairs_baseline():
    value = run_js(FIXTURE + """
const make=(model,origin,actual,p)=>({...prediction(model,origin,1,p),actual,evaluation_split:'retrospective_diagnostic'});
data.history=[make('a','2026-08-07','risk_on',{risk_on:.8,transition:.1,risk_off:.1}),
 make('markov_endpoint','2026-08-07','risk_on',{risk_on:.5,transition:.3,risk_off:.2}),
 make('a','2026-08-14','risk_off',{risk_on:.8,transition:.1,risk_off:.1}),
 make('markov_endpoint','2026-08-14','risk_off',{risk_on:.2,transition:.3,risk_off:.5}),
 make('a','2026-08-21','transition',{risk_on:.2,transition:.7,risk_off:.1})];
const selected=api.evaluate(data,{week:'2026-08-14',horizon:1,scope:'selected'});
const all=api.evaluate(data,{week:'2026-08-14',horizon:1,scope:'all'});
console.log(JSON.stringify({selected:selected.find(r=>r.model==='a'),all:all.find(r=>r.model==='a')}));
""")
    assert value['selected']['n_predictions'] == 1
    assert value['selected']['evaluation_target_end'] == '2026-08-14'
    assert value['all']['n_predictions'] == 2  # Unmatched candidate origin is excluded.
    assert value['all']['evaluation_target_end'] == '2026-08-21'
    assert value['selected']['delta_log_loss'] < 0 < value['all']['delta_log_loss']


def test_applied_departure_and_candidate_calibration_stay_on_the_same_origin_and_event():
    value = run_js(FIXTURE + """
data.calibration={latest:[{model:'adaptive_shrink_after_coherence',origin_date:'2026-09-04',horizon_weeks:4,probability:.68}],
history:[{model:'adaptive_shrink_after_coherence',origin_date:'2026-08-28',horizon_weeks:4,probability:.42}]};
console.log(JSON.stringify(api.appliedComparison(data,{date:'2026-09-04',transition_risk:{'4w':{probability:.5,target_end:'2026-10-02'}}})));
""")
    assert value[0]['applied'] == .5
    assert value[0]['candidate'] == .68
    assert abs(value[0]['delta'] - .18) < 1e-12
    assert 'candidate' not in value[1]


def test_alert_model_and_origin_are_independent_of_endpoint_horizon_and_keep_policy_rows():
    value = run_js(FIXTURE + """
data.alerts={metrics:[{model:'a',policy:'weekly_threshold',evaluation_split:'retrospective_diagnostic',n_predictions:51},
{model:'a',policy:'bounded_episode_budget',evaluation_split:'retrospective_diagnostic',n_predictions:51},
{model:'b',policy:'bounded_episode_budget',evaluation_split:'retrospective_diagnostic',n_predictions:50}],
latest:[{model:'a',policy:'bounded_episode_budget',origin_date:'2026-09-04',horizon_weeks:1,policy_status:'budget_limited'}],
history:[{model:'a',policy:'bounded_episode_budget',origin_date:'2026-08-28',horizon_weeks:1,policy_status:'issued'}]};
console.log(JSON.stringify({latest:api.alertView(data,{week:'2026-09-04',model:'a'}),old:api.alertView(data,{week:'2026-08-28',model:'a'})}));
""")
    assert [row['policy'] for row in value['latest']['metrics']] == ['bounded_episode_budget', 'weekly_threshold']
    assert value['latest']['forecasts'][0]['policy_status'] == 'budget_limited'
    assert value['old']['forecasts'][0]['policy_status'] == 'issued'


def test_weekly_candidate_summary_must_match_the_outer_generation_and_cutoff():
    value = run_js(FIXTURE + """
data.weekly_candidates={schema_version:'regime-weekly-candidates-summary/1',source_generation_id:'generation-a',data_as_of:data.data_as_of};
const valid=api.validate(data,payload);data.weekly_candidates.source_generation_id='older-generation';
console.log(JSON.stringify({valid,invalid:api.validate(data,payload)}));
""")
    assert value['valid'] == []
    assert value['invalid']


def test_alert_expiry_does_not_assign_a_one_week_deadline_to_an_open_episode():
    value = run_js(FIXTURE + """
console.log(JSON.stringify([
api.alertExpiry({policy:'episode_hysteresis_cooldown',target_date:'2026-09-11'}),
api.alertExpiry({policy:'bounded_episode_budget',valid_until:'2026-09-11',target_date:'2026-09-18'}),
api.alertExpiry({policy:'weekly_threshold',target_date:'2026-09-18'}),
api.alertExpiry({policy:'bounded_episode_budget',target_date:'2026-09-18'})]));
""")
    assert value == ['고정 만료 없음', '2026-09-11', '2026-09-18', '—']


def test_every_forecast_selection_survives_a_shared_link():
    value = run_js(FIXTURE + """
const selected={model:'evolving_boundary_gjr_skewt',horizon:13,alertModel:'evolving_boundary_ewma',
 healthScope:'risk_off',healthTarget:'endpoint_risk_off',labelHorizon:4};
const url=api.selectionUrl('http://localhost/?week=2026-08-07&window=52#history',selected);
console.log(JSON.stringify({restored:api.readSelections(url.search),url:String(url)}));
""")
    assert value['restored'] == {
        'model': 'evolving_boundary_gjr_skewt', 'horizon': 13,
        'alertModel': 'evolving_boundary_ewma', 'healthScope': 'risk_off',
        'healthTarget': 'endpoint_risk_off', 'labelHorizon': 4,
        'evaluationScope': 'selected',
    }
    assert 'week=2026-08-07' in value['url']
    assert 'window=52' in value['url']
    assert value['url'].endswith('#history')


def test_label_sensitivity_never_pools_different_horizons():
    value = run_js(FIXTURE + """
const robustness={label_sensitivity:{rows:[1,4,13].flatMap(horizon_weeks=>['ridge','gjr'].map(model=>({model,horizon_weeks,log_loss:horizon_weeks/10})))}};
console.log(JSON.stringify([1,4,13].map(h=>api.labelSensitivityRows(robustness,h))));
""")
    for horizon, rows in zip([1, 4, 13], value, strict=True):
        assert len(rows) == 2
        assert {row['horizon_weeks'] for row in rows} == {horizon}


COMMON_HISTORY_FIXTURE = r"""
const insights=require('./web/insights.js'), app=require('./web/app.js');
const dates=Array.from({length:70},(_,i)=>new Date(Date.parse('2025-01-03')+i*7*86400000).toISOString().slice(0,10));
const codes=['risk_on','transition','risk_off'];
const forecast=(model,i,h)=>({model,origin_date:dates[i],target_date:new Date(Date.parse(dates[i])+h*7*86400000).toISOString().slice(0,10),horizon_weeks:h,current_state:codes[i%3],probabilities:{risk_on:.3+(i%2)*.2,transition:.3,risk_off:.4-(i%2)*.2},evaluation_split:'retrospective_diagnostic'});
const data={schema_version:'regime-forecast-enhancements/1',source_generation_id:'test',data_as_of:dates.at(-1),model_metrics:[],latest:[],history:[]};
for(let i=0;i<70;i++)for(const h of [1,4,13])for(const model of ['evolving_boundary_ewma','evolving_boundary_gjr_skewt','markov_endpoint',...(h>1?['direct_endpoint_ridge','direct_endpoint_xgboost']:[])]) data.history.push(forecast(model,i,h));
const payload={meta:{generation_id:'test',data_as_of:dates.at(-1)},model:{forecast_comparison:{models:['official']},leaderboard:[{name:'official'}],champion:'official'},selection:{operating_champion:'official'},forecast_enhancements:data,
weekly:dates.map((date,i)=>({date,current:{state:codes[i%3]},model_forecasts:[{...forecast('official',i,1),date:forecast('official',i,1).target_date}],next_week:{...forecast('official',i,1),date:forecast('official',i,1).target_date}}))};
"""


def test_common_registry_and_history_use_same_horizon_mature_targets_and_real_window():
    value = run_js(COMMON_HISTORY_FIXTURE + """
const recent=insights.forecastEvaluation(payload,{asOf:dates.at(-1),window:52,horizon:4});
const all=insights.forecastEvaluation(payload,{asOf:dates.at(-1),window:'all',horizon:4});
const earlier=insights.forecastEvaluation(payload,{asOf:dates[35],window:'all',horizon:13});
const manual=data.history.filter(r=>r.model==='evolving_boundary_gjr_skewt'&&r.horizon_weeks===4&&r.origin_date>=dates[18]&&r.target_date<=dates.at(-1));
const expected=manual.reduce((sum,r)=>sum-Math.log(r.probabilities[payload.weekly.find(w=>w.date===r.target_date).current.state]),0)/manual.length;
const selectedForecast=app.forecastForWeek(payload.weekly[60],'evolving_boundary_gjr_skewt',payload,4);
console.log(JSON.stringify({one:insights.forecastModelIds(payload,1),four:insights.forecastModelIds(payload,4),recent,all,earlier,expected,forecast:selectedForecast,
 missing:app.forecastForWeek({date:'2024-01-05',next_week:payload.weekly[0].next_week},'evolving_boundary_gjr_skewt',payload,1),
 foreign:insights.forecastModelIds({...payload,meta:{...payload.meta,generation_id:'foreign'}},4)}));
""")
    assert 'official' in value['one']
    assert 'evolving_boundary_gjr_skewt' in value['one']
    assert 'direct_endpoint_ridge' not in value['one']
    assert 'official' not in value['four']
    assert {'direct_endpoint_ridge', 'direct_endpoint_xgboost'} <= set(value['four'])
    assert value['recent']['scope']['completedCount'] == 48
    assert value['recent']['scope']['pendingCount'] == 4
    assert value['all']['scope']['completedCount'] == 66
    assert value['earlier']['scope']['completedCount'] == 23
    assert value['earlier']['scope']['completedEnd'] <= value['earlier']['scope']['asOf']
    row = next(row for row in value['recent']['leaderboard'] if row['name'] == 'evolving_boundary_gjr_skewt')
    assert abs(row['log_loss'] - value['expected']) < 1e-12
    assert row['mean_detection_delay_forecast_weeks'] is None
    assert value['forecast']['horizon_weeks'] == 4
    assert value['missing'] is None
    assert value['foreign'] == []
