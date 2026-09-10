"""Exercise the actual model renderers and their date/window event handlers."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]

HARNESS = r"""
const fs=require('fs'),vm=require('vm'),path=require('path');
const source=fs.readFileSync('web/app.js','utf8');
const html=fs.readFileSync('web/index.html','utf8');
const htmlIds=new Set([...html.matchAll(/\bid="([^"]+)"/g)].map(match=>match[1]));
const full=JSON.parse(fs.readFileSync('publication/live/regime-results.json'));
function matches(node,selector){const [tag,...classes]=selector.split('.');
 return (!tag||node.tag===tag)&&classes.every(name=>node.className.split(' ').includes(name));}
function descendants(node,selector){return node.children.flatMap(child=>[
 ...(matches(child,selector)?[child]:[]),...descendants(child,selector)]);}
function node(tag='div'){
 const n={tag,children:[],attrs:{},dataset:{},style:{setProperty(){}},listeners:{},className:'',hidden:false,disabled:false,_text:'',_value:'',
 append(...children){for(const child of children.filter(Boolean)){child.parent=n;n.children.push(child)}},
 replaceChildren(...children){n.children=[];n._text='';n.append(...children)},
 setAttribute(key,value){n.attrs[key]=String(value)},removeAttribute(key){delete n.attrs[key]},
 getAttribute(key){return n.attrs[key]??null},addEventListener(event,handler){n.listeners[event]=handler},
 querySelector(selector){return descendants(n,selector)[0]||null},querySelectorAll(selector){return descendants(n,selector)},
 remove(){if(n.parent)n.parent.children=n.parent.children.filter(child=>child!==n)},
 get options(){return n.children},get value(){return n._value},set value(value){n._value=String(value)},
 get textContent(){return n._text+n.children.map(child=>child.textContent).join(' ')},
 set textContent(value){n._text=String(value);n.children=[]}};
 n.classList={add(...values){n.className=[...new Set([...n.className.split(' ').filter(Boolean),...values])].join(' ')},
 remove(...values){n.className=n.className.split(' ').filter(value=>!values.includes(value)).join(' ')},
 toggle(value,on){const active=on??!n.className.split(' ').includes(value);if(active)this.add(value);else this.remove(value)}};
 return n;
}
const context=vm.createContext({module:{exports:{}},require:require('module').createRequire(path.resolve('web/app.js')),
 URL,URLSearchParams,Intl,Date,setTimeout,clearTimeout});
const program=source.replace('const dashboardApi = Object.freeze({',`const dashboardApi = Object.freeze({
 test:{state,dom,bindEvents,selectWeek,renderModel,forecastComparisonForView,
 renderContractOverview,renderTransitionHorizons,renderDurationContext,renderNextForecastSurface,renderTimeline,renderOperationalDiagnostics,renderDecisionResearch,renderForecastAuditPanels,renderMultistateForecast,applyDashboardView,
 configure(){
  renderContractOverview=()=>{};renderRegime=()=>{};renderNextForecastSurface=week=>week.next_week;
  renderTransition=()=>{};renderSemanticLabels=()=>{};renderHeaderDataAsOf=()=>{};
  renderHistory=()=>{};renderTimeline=()=>{};renderFactors=()=>{};renderContextExtremes=()=>{};
  renderDrivers=()=>{};renderMarket=()=>{};renderDurationContext=()=>{};renderFxContext=()=>{};
  renderDecisionShadowCurrentSummary=()=>{};renderExecutionBrief=()=>{};syncHoldingsCalculator=()=>{};
  applyExpiredForecastDomState=()=>{};renderTransitionModels=()=>{};renderConditionalStats=()=>{};
 },setHistoryLoader(loader){ensureHistory=loader},bumpLoad(){loadSequence+=1}},`);
vm.runInContext(program,context);
context.document={createElement:node,createTextNode(text){const n=node('text');n.textContent=text;return n},
 getElementById(id){return htmlIds.has(id)?api.dom[id]||null:null}};
let current=new URL('http://localhost/?model=causal_dynamic_ensemble&window=52#history');
context.window={addEventListener(){},get location(){return current},history:{replaceState(_a,_b,value){current=new URL(value,current)}}};
const api=context.module.exports.test;api.configure();
for(const id of new Set([
 ...[...source.matchAll(/dom\["([^"]+)"\]/g)].map(match=>match[1]),
 ...htmlIds]))api.dom[id]=node();
api.dom.dashboard=node();
for(const id of ['history-window','model-evaluation-window'])for(const value of ['26','52','104','all']){
 const option=node('option');option.value=value;api.dom[id].append(option);
}
api.state.raw=full;api.state.weekly=full.weekly;api.state.selectedIndex=full.weekly.length-1;
api.state.comparisonModel='causal_dynamic_ensemble';api.state.historyAvailability='ready';
api.state.sidecarAvailability.research='ready';api.bindEvents();
function snapshot(){return {
 week:api.state.weekly[api.state.selectedIndex]?.date,model:api.state.comparisonModel,
 quality:api.dom['model-quality-brief'].textContent,qualityModel:api.dom['model-quality-brief'].dataset.model,
 summary:api.dom['champion-summary'].textContent,comparison:api.dom['model-input-comparison'].textContent,
 comparisonHidden:api.dom['model-input-comparison'].hidden,caption:api.dom['model-caption'].textContent,
 note:api.dom['model-evaluation-note'].textContent,rank:api.dom['model-forecast-rank'].textContent,
 metricValues:['model-forecast-rank','model-forecast-log-loss','model-forecast-brier','model-forecast-calibration'].map(id=>api.dom[id].textContent),
 metricsHidden:api.dom['model-forecast-metrics'].hidden,
 table:api.dom['leaderboard-body'].textContent,chart:api.dom['model-loss-chart'].textContent,
 selectedRows:api.dom['leaderboard-body'].children.filter(row=>row.attrs['aria-current']==='true').map(row=>row.dataset.model),
 windows:['history-window','model-evaluation-window'].map(id=>api.dom[id].value),
 query:Object.fromEntries(current.searchParams),scope:api.forecastComparisonForView().evaluationScope};}
async function chooseWindow(id,value){const control=api.dom[id];control.value=value;await control.listeners.change({target:control})}
function chooseModel(value){const control=api.dom['model-forecast-select'];control.value=value;control.listeners.change()}
async function chooseDate(value){api.dom['analysis-date'].value=value;await api.dom['analysis-date'].listeners.change()}
function deferred(){let resolve;const promise=new Promise(r=>resolve=r);return {promise,resolve}}
function delayHistory(){const response=deferred();let pending;
 api.state.weekly=full.weekly.slice(-26);api.state.selectedIndex=25;api.state.historyAvailability='deferred';
 api.setHistoryLoader(()=>pending??=(async()=>{await response.promise;
  const date=api.state.pendingHistoryWeek||api.state.weekly[api.state.selectedIndex].date;
  api.state.pendingHistoryWeek=null;api.state.weekly=full.weekly;api.state.historyAvailability='ready';
  api.selectWeek(Math.max(0,full.weekly.findIndex(week=>week.date===date)),false);return true;})());return response;
}
"""


def run_js(scenario):
    result = subprocess.run(["node", "-"], input=HARNESS + "\n" + scenario,
                            text=True, capture_output=True, check=True, cwd=ROOT)
    return json.loads(result.stdout)


def test_model_selection_updates_actual_headline_table_and_identical_decision_explanation():
    result = run_js("""
api.selectWeek(full.weekly.length-1,false);const dynamic=snapshot();
chooseModel('causal_multiscale_ensemble');const multi=snapshot();
chooseModel('boundary_filtered_history');console.log(JSON.stringify({dynamic,multi,boundary:snapshot()}));
""")
    assert "0.6345" in result["dynamic"]["quality"]
    assert "0.6349" in result["multi"]["quality"]
    assert "0/14회" in result["multi"]["quality"]
    assert "51/51주" in result["multi"]["comparison"]
    assert "예측 국면 일치 51/51주" in result["multi"]["comparison"]
    assert "선택 모델 멀티스케일 앙상블" in result["multi"]["summary"]
    assert result["multi"]["selectedRows"] == ["causal_multiscale_ensemble"]
    assert result["multi"]["qualityModel"] == "causal_multiscale_ensemble"
    assert "0.4953" in result["boundary"]["quality"]
    assert "5/14회" in result["boundary"]["quality"]
    assert result["boundary"]["selectedRows"] == ["boundary_filtered_history"]
    assert result["boundary"]["query"]["model"] == "boundary_filtered_history"


@pytest.mark.parametrize("control", ["history-window", "model-evaluation-window"])
def test_each_window_control_updates_both_selects_and_real_model_outputs(control):
    result = run_js(f"""
(async()=>{{api.selectWeek(full.weekly.length-1,false);const recent=snapshot();
await chooseWindow('{control}','all');console.log(JSON.stringify({{recent,all:snapshot()}}));}})();
""")
    assert result["recent"]["scope"]["completedCount"] == 51
    assert result["all"]["windows"] == ["all", "all"]
    assert result["all"]["scope"]["completedCount"] == 191
    assert "0.495 Log loss" in result["all"]["quality"]
    assert "3/40회" in result["all"]["quality"]
    assert "평가 191주" in result["all"]["caption"]
    assert result["all"]["table"] != result["recent"]["table"]
    assert result["all"]["chart"] != result["recent"]["chart"]
    assert result["all"]["query"]["window"] == "all"


def test_date_input_recalculates_completed_scope_and_starting_week_has_no_rank():
    result = run_js("""
(async()=>{api.selectWeek(full.weekly.length-1,false);await chooseDate('2025-12-26');const historical=snapshot();
await chooseDate(full.weekly[0].date);console.log(JSON.stringify({historical,start:snapshot()}));})();
""")
    assert result["historical"]["week"] == "2025-12-26"
    assert "0.4081" in result["historical"]["quality"]
    assert "1/7회" in result["historical"]["quality"]
    assert result["historical"]["scope"]["completedEnd"] == "2025-12-26"
    assert result["start"]["scope"]["completedCount"] == 0
    assert result["start"]["rank"] == "—"
    assert "완료된 예측 없음" in result["start"]["caption"]
    assert "평가 0주" in result["start"]["caption"]
    assert "확률 오차 —" in result["start"]["quality"]
    assert result["start"]["comparisonHidden"]


def test_delayed_date_request_cannot_replace_a_later_week_choice():
    result = run_js("""
(async()=>{const response=delayHistory();api.selectWeek(25,false);
const older=chooseDate('2025-01-03');api.dom['week-select'].value='2026-08-28';api.dom['week-select'].listeners.change();
const chosen=snapshot();response.resolve();await older;console.log(JSON.stringify({chosen,ready:snapshot()}));})();
""")
    assert result["chosen"]["week"] == result["ready"]["week"] == "2026-08-28"
    assert result["ready"]["query"]["week"] == "2026-08-28"
    assert result["ready"]["scope"]["completedEnd"] == "2026-08-28"


def test_latest_of_two_pending_date_inputs_wins_after_history_arrives():
    result = run_js("""
(async()=>{const response=delayHistory();api.selectWeek(25,false);
const older=chooseDate('2025-01-03'),newer=chooseDate('2025-12-26');response.resolve();await Promise.all([older,newer]);
console.log(JSON.stringify(snapshot()));})();
""")
    assert result["week"] == "2025-12-26"
    assert "0.4081" in result["quality"]
    assert result["scope"]["completedEnd"] == "2025-12-26"


def test_longer_window_loads_deferred_history_and_recalculates_without_changing_selected_week():
    result = run_js("""
(async()=>{const response=delayHistory();api.selectWeek(25,false);const pending=chooseWindow('model-evaluation-window','104');
const during={enabled:!api.dom['model-evaluation-window'].options.find(option=>option.value==='104').disabled};
response.resolve();await pending;console.log(JSON.stringify({during,ready:snapshot()}));})();
""")
    assert result["during"]["enabled"]
    assert result["ready"]["week"] == "2026-09-04"
    assert result["ready"]["scope"]["completedCount"] == 103
    assert result["ready"]["windows"] == ["104", "104"]
    assert result["ready"]["query"]["window"] == "104"


def test_legacy_without_per_model_history_keeps_published_metrics_and_hides_forecast_selector():
    result = run_js("""
api.selectWeek(full.weekly.length-1,false);chooseModel('causal_multiscale_ensemble');const previous=snapshot();
const legacy=structuredClone(full);legacy.meta.result_version='weekly-regime-result-v4';delete legacy.research;
for(const week of legacy.weekly)delete week.model_forecasts;
api.state.raw=legacy;api.state.weekly=legacy.weekly;api.state.sidecarAvailability.research='not_applicable';
api.state.comparisonModel='causal_dynamic_ensemble';
api.selectWeek(legacy.weekly.length-1,false);
const comparison=api.forecastComparisonForView();
console.log(JSON.stringify({previous,view:snapshot(),scopeAbsent:comparison.evaluationScope===undefined,
 originalMetrics:JSON.stringify(comparison.leaderboard)===JSON.stringify(legacy.model.leaderboard),
 selectorHidden:api.dom['model-forecast-field'].hidden,forecastHidden:api.dom['model-forecast-explorer'].hidden,
 evaluationHidden:api.dom['model-evaluation-field'].hidden,
 scopeHidden:api.dom['model-evaluation-note'].hidden&&api.dom['model-input-comparison'].hidden}));
""")
    assert result["scopeAbsent"] and result["originalMetrics"]
    assert result["selectorHidden"] and result["forecastHidden"]
    assert result["evaluationHidden"] and result["scopeHidden"]
    assert not result["previous"]["metricsHidden"]
    assert all(value != "—" for value in result["previous"]["metricValues"])
    assert result["view"]["metricsHidden"]
    assert result["view"]["metricValues"] == ["—"] * 4
    assert "191주" in result["view"]["caption"]
    assert "Log loss" in result["view"]["quality"]
    assert "3/40회" in result["view"]["quality"]


def test_pending_research_model_clears_previous_metrics_then_shows_its_own_after_loading():
    result = run_js("""
api.selectWeek(full.weekly.length-1,false);chooseModel('causal_multiscale_ensemble');const previous=snapshot();
const core=structuredClone(full);delete core.research;
api.state.raw=core;api.state.sidecarAvailability.research='pending';
api.state.comparisonModel='boundary_filtered_history';api.renderModel();const pending=snapshot();
const pendingCaption=api.dom['model-detail-caption'].textContent;
api.state.raw=full;api.state.sidecarAvailability.research='ready';api.renderModel();
console.log(JSON.stringify({previous,pending,pendingCaption,loaded:snapshot()}));
""")
    assert not result["previous"]["metricsHidden"]
    assert all(value != "—" for value in result["previous"]["metricValues"])
    assert "경계 · 과거 잔차" in result["pendingCaption"]
    assert result["pending"]["model"] == "boundary_filtered_history"
    assert result["pending"]["metricsHidden"]
    assert result["pending"]["metricValues"] == ["—"] * 4
    assert result["loaded"]["model"] == "boundary_filtered_history"
    assert not result["loaded"]["metricsHidden"]
    assert all(value != "—" for value in result["loaded"]["metricValues"])
    assert result["loaded"]["metricValues"] != result["previous"]["metricValues"]
    assert result["loaded"]["rank"] == "1 / 13"
    assert "0.4953" in result["loaded"]["metricValues"][1]


def test_historical_timing_uses_selected_origin_and_target_with_latest_publication_separate():
    result = run_js("""
api.selectWeek(full.weekly.length-2,false); api.renderContractOverview();
const old={origin:api.dom['forecast-origin-at'].textContent,target:api.dom['forecast-target-at'].textContent,
 issued:api.dom['forecast-decision-at'].textContent,latest:api.dom['latest-publication-info'].textContent,latestHidden:api.dom['latest-publication-info'].hidden,
 summary:api.dom['forecast-window-summary'].textContent};
api.dom['latest-week'].listeners.click();api.renderContractOverview();
console.log(JSON.stringify({old,current:{origin:api.dom['forecast-origin-at'].textContent,target:api.dom['forecast-target-at'].textContent,
 issued:api.dom['forecast-decision-at'].textContent,latestHidden:api.dom['latest-publication-info'].hidden,disabled:api.dom['latest-week'].disabled,
 summary:api.dom['forecast-window-summary'].textContent}}));
""")
    assert "8월 28일" in result["old"]["origin"]
    assert "9월 4일" in result["old"]["target"]
    assert "과거 재구성" in result["old"]["issued"]
    assert "9월 6일" in result["old"]["latest"] and "9월 11일" in result["old"]["latest"]
    assert not result["old"]["latestHidden"] and result["current"]["latestHidden"]
    assert "9월 4일" in result["current"]["origin"]
    assert "9월 11일" in result["current"]["target"] and result["current"]["disabled"]
    assert "2026.08.28 → 2026.09.04 · 과거" in result["old"]["summary"]
    assert "2026.09.04 → 2026.09.11" in result["current"]["summary"]


def test_period_predictions_directions_and_duration_render_real_values_for_each_origin():
    result = run_js("""
function view(index){api.selectWeek(index,false);const week=full.weekly[index];
 api.renderTransitionHorizons(week);api.renderDurationContext(week.duration_context);api.renderNextForecastSurface(week);
 return {horizon:api.dom['transition-horizon-bars'].textContent,duration:api.dom['duration-context'].textContent,
 direction:api.dom['next-direction-summary'].textContent,
 research:api.dom['multistate-forecast'].textContent};}
console.log(JSON.stringify({latest:view(full.weekly.length-1),previous:view(full.weekly.length-2)}));
""")
    latest = result["latest"]
    for value in ("50.7%", "63.7%", "45.5%", "78.2%", "최초 이탈", "6.9%"):
        assert value in latest["horizon"]
    for value in ("3–8주", "46 / 1개", "46 / 47개"):
        assert value in latest["duration"]
    assert "추정치 95% 구간" in latest["duration"]
    assert "유지 80.7%" in latest["direction"] and "악화 19.3%" in latest["direction"]
    assert result["previous"]["horizon"] != latest["horizon"]
    assert result["previous"]["duration"] != latest["duration"]
    assert "국면 예측 연구" in latest["research"]
    assert "연구 비교 모델" in latest["research"]


def test_model_timeline_remains_navigable_after_past_click_and_shared_latest_returns():
    result = run_js("""
api.selectWeek(full.weekly.length-1,false);api.applyDashboardView('model');api.renderTimeline();
const old=api.dom['regime-timeline'].querySelectorAll('button.timeline-cell').find(item=>item.dataset.date==='2026-08-28');
old.focus=()=>{};old.listeners.click();api.renderTimeline();
const past={week:snapshot().week,end:api.dom['timeline-end'].textContent,view:api.dom.dashboard.dataset.activeView,
 next:!api.dom['next-week'].disabled,latest:!api.dom['latest-week'].disabled};
api.dom['next-week'].listeners.click();const next=snapshot().week;
api.dom['previous-week'].listeners.click();api.dom['latest-week'].listeners.click();
console.log(JSON.stringify({past,next,latest:snapshot().week}));
""")
    assert result["past"] == {"week": "2026-08-28", "end": "2026-09-04", "view": "model", "next": True, "latest": True}
    assert result["next"] == result["latest"] == "2026-09-04"


def test_detection_delay_is_displayed_with_recognized_and_unrecognized_denominators():
    result = run_js("""
(async()=>{api.selectWeek(full.weekly.length-1,false);await chooseWindow('model-evaluation-window','all');
chooseModel('boundary_filtered_history');console.log(JSON.stringify(api.dom['model-health-strip'].textContent));})();
""")
    assert "인식한 전환의 평균 지연 0.56주" in result
    assert "인식 36/40건 · 미인식 4건" in result


def test_operational_scores_distinguish_mature_labels_from_verified_prospective_denominator():
    result = run_js("""
const target=node(); api.renderOperationalDiagnostics({timing:{issued_entry_count:10,deadline_observed_entries:9,on_time_entries:7},
 probability_scores:{completed_weeks:8,prospective_completed_weeks:5,excluded_late_or_unverified_weeks:3,
 log_loss:.321,brier:.123,evaluation_basis:'independent_of_investment_execution'}},target);
console.log(JSON.stringify(target.textContent));
""")
    assert "라벨 확정 평가 전체" in result and "8주" in result
    assert "대상 시각 전 저장·발행 확인" in result and "5주" in result
    assert "매매 예정시각 전 발행" in result
    assert "지연·발행 미확인 제외" in result and "3주" in result
    assert "0.321" in result and "0.123" in result
    assert "라벨 확정 평가 전체" in result


def test_alert_controls_change_actual_policy_rows_without_reusing_current_week_scope():
    result = run_js("""
const target=node();api.renderDecisionResearch(full.research.decision_research_v2,target);
const panel=target.children[0],controls=panel.children.find(item=>item.className==='research-filter-controls');
const [event,budget]=controls.children,table=panel.children[3];
const before=table.textContent;event.value='all_departure';event.listeners.change();const changed=table.textContent;
budget.value='8';budget.listeners.change();console.log(JSON.stringify({before,changed,otherBudget:table.textContent,
 caption:panel.children[1].textContent}));
""")
    assert result["before"] != result["changed"]
    assert result["changed"] != result["otherBudget"]
    assert "2023년 이후 진단" in result["caption"]


def test_optional_audit_panels_distinguish_previous_publication_new_research_and_empty_prospective_information():
    result = run_js("""
const target=node();api.renderForecastAuditPanels({calibration_audit:{schema_version:'regime-calibration-audit/1',data_as_of:full.meta.data_as_of,
 rows:[{model:'markov_hazard',horizon_weeks:13,evaluation_split:'retrospective_diagnostic',n_predictions:179,
 previous_published_log_loss:.4377,raw_log_loss:.3196,calibrated_log_loss:.442,final_log_loss:null,raw_brier:.089,final_brier:null,
 previous_published_brier:.1393,selected_method:'past_block_selection'}],
 artifacts:[{label:'전체 JSON',url:'./data/calibration.json'},{label:'bad',url:'javascript:alert(1)'}]},
 forecast_information:{schema_version:'regime-forecast-information/1',data_as_of:full.meta.data_as_of,rows:[],
 notes:['일정 정보는 전향적 축적 중입니다.']}},target);
console.log(JSON.stringify({text:target.textContent,methods:api.dom['forecast-method-notes'].textContent,links:target.querySelectorAll('a').map(item=>({href:item.href,text:item.textContent}))}));
""")
    assert "기존 발행 Log loss" in result["text"]
    assert "새 보정 연구 Log loss" in result["text"]
    assert "새 최종 연구 Log loss" in result["text"]
    assert "0.4377" in result["text"] and "0.3196" in result["text"] and "—" in result["text"]
    assert "운영 반영 여부는 별도" not in result["text"]
    assert "전향적 축적" not in result["text"] and "전향적 축적" in result["methods"]
    assert "비교 가능한 완료 표본이 없습니다" in result["text"]
    assert result["links"] == [{"href": "./data/calibration.json", "text": "전체 JSON"}]


def test_operational_duplicate_entries_do_not_inflate_displayed_week_count():
    result = run_js("""
const target=node();api.renderOperationalDiagnostics({probability_scores:{completed_weeks:2,completed_entries:4,
 prospective_completed_weeks:2,duplicate_target_entries:2,excluded_late_or_unverified_weeks:0,
 evaluation_basis:'independent_of_investment_execution',log_loss:1.3539324,brier:.8309954}},target);
console.log(JSON.stringify(target.textContent));
""")
    assert "2주 / 4건" in result
    assert "대상 시각 전 저장·발행 확인" in result and "2주" in result
    assert "동일 대상 중복 2건 제외" in result
    assert "1.3539" in result and "0.831" in result


def test_information_comparison_preserves_small_deltas_and_prospective_evidence_labels():
    result = run_js("""
const target=node();api.renderForecastAuditPanels({forecast_information:{
 schema_version:'regime-forecast-information/1',data_as_of:full.meta.data_as_of,
 rows:[{feature_block:'vix3m_term_structure',model:'boundary_vix3m_augmented',
 baseline_model:'boundary_existing_vol_control',evaluation_split:'holdout',matched_n:191,
 baseline_log_loss:.3731897933,candidate_log_loss:.3738624464,delta_log_loss:.0006726532,
 delta_brier:.0004161311,delta_worsening_brier:.0000063045474}],
 notes:['board_ebp: prospective_only — revised monthly history not backdated']
}},target);console.log(JSON.stringify({text:target.textContent,methods:api.dom['forecast-method-notes'].textContent}));
""")
    assert "191주" in result["text"] and "과거 진단" in result["text"]
    assert "VIX3M 기간구조" in result["text"] and "경계 + VIX3M" in result["text"]
    assert "0.3732" in result["text"] and "0.3739" in result["text"]
    assert "6.30e-6" in result["text"]
    assert "EBP: 실제 최초 확보 시점" in result["methods"]
    assert "최초 확보" not in result["text"]
    assert "prospective_only" not in result["methods"]


def test_optional_method_notes_are_grouped_without_repetition_and_clear_on_rerender():
    result = run_js("""
const target=node();api.renderForecastAuditPanels({forecast_information:{schema_version:'regime-forecast-information/1',
 notes:['fomc: prospective_only — schedule','bls: prospective_only — schedule','fomc: prospective_only — schedule',
 '모든 비교는 같은 origin/target에서 재계산한 기준선과 후보를 사용합니다.']}},target);
const notes=api.dom['forecast-method-notes'].textContent;
api.renderForecastAuditPanels({},target);console.log(JSON.stringify({notes,cleared:api.dom['forecast-method-notes'].textContent}));
""")
    assert result["notes"].count("실제 최초 확보 시점") == 1
    assert "FOMC 일정 · CPI·고용 일정" in result["notes"]
    assert "모든 비교는" not in result["notes"]
    assert result["cleared"] == ""


RESEARCH_RENDER_FIXTURE = r"""
const origin=full.meta.data_as_of;
const horizons=Object.fromEntries([1,4,13].map(h=>[`${h}w`,{horizon_weeks:h,target_date:new Date(Date.parse(origin)+h*7*86400000).toISOString(),
 endpoint:h===1?{risk_on:.8,transition:.15,risk_off:.05}:h===4?{risk_on:.6,transition:.25,risk_off:.15}:{risk_on:.4,transition:.35,risk_off:.25},
 first_departure:h===1?{no_departure:.8,risk_on:0,transition:.15,risk_off:.05}:h===4?{no_departure:.5,risk_on:0,transition:.4,risk_off:.1}:{no_departure:.3,risk_on:0,transition:.5,risk_off:.2},
 any_risk_off_entry:h===1?.05:h===4?.2:.4,any_risk_off_occupancy:h===1?.05:h===4?.2:.4}]));
const block={schema_version:'regime-forecast-research/1',data_as_of:origin,selected_model:'paths',automatic_promotion:false,
 models:[{id:'paths',label:'방향 위험률',role:'challenger',history:[],latest:{origin_date:origin,current_state:'risk_on',next_state:horizons['1w'].endpoint,horizons}},
 {id:'asymmetric',label:'비대칭 경계',role:'challenger',history:[],latest:{origin_date:origin,current_state:'risk_on',next_state:{risk_on:.3,transition:.5,risk_off:.2},horizons:{}}}],
 metrics:[{model:'paths',horizon_weeks:1,target:'next_state',period:'retrospective_2023_2026',weeks:191,log_loss:.54,brier:.33,worsening_brier:.083,worsening_events:19,worsening_hits:0,recovery_events:21,recovery_hits:0,departure_false_alarms_per_year:0},
 {model:'asymmetric',horizon_weeks:1,target:'next_state',period:'retrospective_2023_2026',weeks:191,log_loss:.36,brier:.22,worsening_brier:.076,worsening_events:19,worsening_hits:0,recovery_events:21,recovery_hits:16,departure_false_alarms_per_year:.82}]};
"""


def test_research_selector_changes_native_probabilities_without_replacing_official_model():
    result = run_js(RESEARCH_RENDER_FIXTURE + r"""
api.state.raw={...full,research:{...full.research,forecast_research:block}};
const official=JSON.stringify(api.state.raw.weekly.at(-1).next_week);
api.renderMultistateForecast(full.weekly.at(-1));const paths=api.dom['multistate-forecast'].textContent;
const control=api.dom['multistate-forecast'].querySelector('select');control.value='asymmetric';control.listeners.change();
console.log(JSON.stringify({paths,asymmetric:api.dom['multistate-forecast'].textContent,
 oneWeekNote:api.dom['multistate-forecast'].querySelector('.research-empty')?.textContent,
 officialUnchanged:official===JSON.stringify(api.state.raw.weekly.at(-1).next_week),query:Object.fromEntries(current.searchParams)}));
""")
    assert "다음 주 위험선호 80.0%" in result["paths"]
    assert "위험회피 진입" in result["paths"] and "40.0%" in result["paths"]
    assert "다음 주 위험선호 30.0%" in result["asymmetric"]
    assert result["oneWeekNote"] == "1주 예측"
    assert "위험회피 진입" not in result["asymmetric"]
    assert "191주" in result["paths"] and "후보와 기준선" in result["paths"]
    assert result["officialUnchanged"] and result["query"]["research_model"] == "asymmetric"


def test_pending_research_is_labeled_loading_then_renders_when_evidence_arrives():
    result = run_js(RESEARCH_RENDER_FIXTURE + r"""
api.state.raw={...full,research:{...full.research}};delete api.state.raw.research.forecast_research;
api.state.sidecarAvailability.research='pending';api.renderMultistateForecast(full.weekly.at(-1));
const pending={text:api.dom['multistate-forecast'].textContent,busy:api.dom['multistate-forecast'].attrs['aria-busy']};
api.state.raw={...full,research:{...full.research,forecast_research:block}};api.state.sidecarAvailability.research='ready';
api.renderTransitionHorizons(full.weekly.at(-1));console.log(JSON.stringify({pending,ready:api.dom['multistate-forecast'].textContent,
 busy:api.dom['multistate-forecast'].attrs['aria-busy']}));
""")
    assert "불러오는 중" in result["pending"]["text"]
    assert "결과가 아직 없습니다" not in result["pending"]["text"]
    assert result["pending"]["busy"] == "true" and result["busy"] == "false"
    assert "40.0%" in result["ready"]
