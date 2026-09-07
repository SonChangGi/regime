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
const full=JSON.parse(fs.readFileSync('publication/live/regime-results.json'));
function matches(node,selector){return selector.startsWith('.')?node.className.split(' ').includes(selector.slice(1)):node.tag===selector;}
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
 configure(){
  renderContractOverview=()=>{};renderRegime=()=>{};renderNextForecastSurface=week=>week.next_week;
  renderTransition=()=>{};renderSemanticLabels=()=>{};renderHeaderDataAsOf=()=>{};
  renderHistory=()=>{};renderTimeline=()=>{};renderFactors=()=>{};renderContextExtremes=()=>{};
  renderDrivers=()=>{};renderMarket=()=>{};renderDurationContext=()=>{};renderFxContext=()=>{};
  renderDecisionShadowCurrentSummary=()=>{};renderExecutionBrief=()=>{};syncHoldingsCalculator=()=>{};
  applyExpiredForecastDomState=()=>{};renderTransitionModels=()=>{};renderConditionalStats=()=>{};
 },setHistoryLoader(loader){ensureHistory=loader},bumpLoad(){loadSequence+=1}},`);
vm.runInContext(program,context);
context.document={createElement:node,createTextNode(text){const n=node('text');n.textContent=text;return n}};
let current=new URL('http://localhost/?model=causal_dynamic_ensemble&window=52#history');
context.window={addEventListener(){},get location(){return current},history:{replaceState(_a,_b,value){current=new URL(value,current)}}};
const api=context.module.exports.test;api.configure();
for(const id of [...source.matchAll(/dom\["([^"]+)"\]/g)].map(match=>match[1]))api.dom[id]=node();
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
    assert "포착·오경보 횟수가 같습니다" in result["multi"]["comparison"]
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
    assert "경계 전환 · 과거 충격" in result["pendingCaption"]
    assert result["pending"]["model"] == "boundary_filtered_history"
    assert result["pending"]["metricsHidden"]
    assert result["pending"]["metricValues"] == ["—"] * 4
    assert result["loaded"]["model"] == "boundary_filtered_history"
    assert not result["loaded"]["metricsHidden"]
    assert all(value != "—" for value in result["loaded"]["metricValues"])
    assert result["loaded"]["metricValues"] != result["previous"]["metricValues"]
    assert result["loaded"]["rank"] == "1 / 13"
    assert "0.4953" in result["loaded"]["metricValues"][1]
