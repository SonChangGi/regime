"""Copied view links preserve alert filters and the visible performance series."""
import json
from pathlib import Path
import subprocess

from test_dashboard_async_view_intent import HARNESS as LOADER_HARNESS


ROOT = Path(__file__).resolve().parents[1]


def run_js(program):
    result = subprocess.run(["node", "-"], input=program, text=True,
                            capture_output=True, check=True, cwd=ROOT)
    return json.loads(result.stdout)


def test_valid_new_options_keep_existing_seven_query_meanings():
    result = run_js("""
const api=require('./web/app.js'),fs=require('fs');
const p=JSON.parse(fs.readFileSync('publication/live/regime-results.json'));
const query='?week='+p.weekly.at(-2).date+'&model=xgboost&window=104&basis=forecast&weighting=weekly&horizon=4&assets=TLT';
const legacy=api.parseViewState(query,p,p.weekly);
const full=api.parseViewState(query+'&alert=all_departure&budget=12&strategies=spy_buy_and_hold,realistic_60_40,spy_buy_and_hold,invalid',p,p.weekly);
console.log(JSON.stringify({legacy,full}));
""")
    assert set(result["legacy"]) == {"week", "model", "window", "basis", "weighting", "horizon", "asset", "modelHorizon"}
    assert result["legacy"]["modelHorizon"] == 1
    assert result["legacy"]["horizon"] == 4
    assert {key: result["full"][key] for key in result["legacy"]} == result["legacy"]
    assert result["full"]["alert"] == "all_departure"
    assert result["full"]["budget"] == 12
    assert result["full"]["strategies"] == ["realistic_60_40", "spy_buy_and_hold"]


def test_invalid_and_unavailable_inputs_fall_back_without_empty_charts():
    result = run_js("""
const api=require('./web/app.js'),fs=require('fs');
const p=JSON.parse(fs.readFileSync('publication/live/regime-results.json'));
const invalid=api.parseViewState('?alert=__proto__&budget=12.0&strategies=unknown,__proto__',p,p.weekly);
const core=structuredClone(p);delete core.research;
const query='?alert=all_departure&budget=8&strategies=spy_buy_and_hold,realistic_60_40';
console.log(JSON.stringify({invalid,pending:api.parseViewState(query,core,core.weekly,{researchPending:true}),
 ready:api.parseViewState(query,p,p.weekly),legacy:api.parseViewState(query,core,core.weekly)}));
""")
    assert result["invalid"]["alert"] == "risk_worsening"
    assert result["invalid"]["budget"] == 4
    assert result["invalid"]["strategies"] is None
    assert set(result["pending"]["strategies"]) == set(result["ready"]["strategies"])
    assert result["legacy"]["strategies"] is None
    assert result["legacy"]["alert"] == "all_departure"
    assert result["legacy"]["budget"] == 8


CONTROL_HARNESS = r"""
const fs=require('fs'),vm=require('vm'),path=require('path');
const realInsights=require('./web/insights.js');let chartKeys=[],copied=null;
function descendants(parent,match){return parent.children.flatMap(child=>[
 ...(match(child)?[child]:[]),...descendants(child,match)]);}
function node(tag){const n={tag,children:[],attrs:{},listeners:{},className:'',hidden:false,_text:'',_value:null,
 append(...children){n.children.push(...children.filter(Boolean))},
 replaceChildren(...children){n.children=children.filter(Boolean);n._text=''},
 setAttribute(key,value){n.attrs[key]=String(value)},addEventListener(event,handler){n.listeners[event]=handler},
 querySelector(selector){return descendants(n,child=>child.tag===selector)[0]||null},
 querySelectorAll(selector){return descendants(n,child=>selector.startsWith('.')?child.className.split(' ').includes(selector.slice(1)):child.tag===selector)},
 focus(){},get textContent(){return n._text+n.children.map(child=>child.textContent).join(' ')},
 set textContent(value){n._text=String(value);n.children=[]},
 get value(){return n._value??(tag==='select'?n.children[0]?.value:'')??''},
 set value(value){n._value=String(value)}};return n;}
const context=vm.createContext({module:{exports:{}},require:require('module').createRequire(path.resolve('web/app.js')),
 URL,URLSearchParams,Intl,Date,REGIME_INSIGHTS:{...realInsights,createPerformanceRenderers:()=>({
 renderPerformanceLineChart(_rows,keys){chartKeys=[...keys];return node('div')},
 renderPerformanceDrawdownChart:()=>node('div'),renderPerformanceFallback:()=>node('div'),
 renderPerformanceBridge:()=>node('div'),renderPerformanceTurnover:()=>node('div'),
 renderPerformanceDetailTable:()=>node('div'),turnoverValues:()=>[]})}});
let program=fs.readFileSync('web/app.js','utf8').replace('const dashboardApi = Object.freeze({',`const dashboardApi = Object.freeze({
 testView:{state,dom,renderResearchUpgrades,renderDecisionShadow,syncViewUrl,copyCurrentViewLink,
 configure(){renderDecisionShadowCurrentSummary=()=>{};renderDecisionShadowEconomicSummary=()=>{};
 clearDecisionShadowEntryTimer=()=>{};scheduleDecisionShadowEntryRerender=()=>{};
 decisionShadowTimingPolicy=()=>null;renderHoldingsInputs=()=>{};}},`);
vm.runInContext(program,context);const exported=context.module.exports,api=exported.testView;api.configure();
context.document={createElement:node};let current=new URL('http://localhost/?unrelated=keep#performance');
context.window={get location(){return current},history:{replaceState(_a,_b,value){current=new URL(value,current)}},setTimeout:()=>0};
context.navigator={clipboard:{async writeText(value){copied=value}}};
for(const key of ['research-upgrades','forecast-alert-research','decision-shadow-block','decision-shadow-nav','decision-shadow-grid','decision-shadow-caption','screen-reader-status','copy-view-link'])api.dom[key]=node('div');
const strategies={probability_shadow:{weeks:2},static_60_40:{weeks:2},spy_buy_and_hold:{weeks:2},vol_target_60_40:{weeks:2}};
const payload={meta:{data_as_of:'2026-09-04'},model:{champion:'xgboost',forecast_comparison:{models:['xgboost']}},weekly:[{date:'2026-09-04'}],research:{
 decision_research_v2:{alert_budgets:[{target:'risk_worsening',annual_false_alarm_budget:4,score:'binary_xgboost',retrospective_diagnostic:{recall:.11}},
 {target:'all_departure',annual_false_alarm_budget:12,score:'binary_xgboost',retrospective_diagnostic:{recall:.93}}]},
 prospective_decision_shadow:{schema_version:'regime-prospective-decision-shadow/1',historical_reconstructed_shadow:{strategies,status:'completed',minimum_evaluation_weeks:1,
 evaluation_start_week:'2023-01-06',evaluation_end_week:'2026-09-04',series:[{date:'2026-08-28'},{date:'2026-09-04'}]}}}};
api.state.raw=payload;api.state.weekly=payload.weekly;api.state.selectedIndex=0;api.state.comparisonModel='xgboost';api.state.sidecarAvailability.research='ready';
function controls(){return descendants(api.dom['forecast-alert-research'],child=>child.tag==='select')}
function choose(index,value){const control=controls()[index];control.value=value;control.listeners.change()}
function toggle(label){const control=api.dom['decision-shadow-grid'].querySelectorAll('.performance-switch').find(child=>child.textContent===label);control.listeners.click()}
function snapshot(){return {alert:api.state.decisionResearchTarget,budget:api.state.decisionResearchBudget,
 strategies:api.state.performanceVisible,visible:[...chartKeys],query:Object.fromEntries(current.searchParams),
 selectedControls:controls().map(child=>child.value),text:api.dom['forecast-alert-research'].textContent}}
"""


def test_real_filter_and_strategy_handlers_copy_the_same_new_tab_view():
    result = run_js(CONTROL_HARNESS + """
(async()=>{api.renderDecisionShadow();choose(0,'all_departure');choose(1,'12');toggle('위험 국면 전략');toggle('변동 60/40');
const chosen=snapshot();await api.copyCurrentViewLink();const restored=exported.parseViewState(new URL(copied).search,payload,payload.weekly);
console.log(JSON.stringify({chosen,copied,restored}));})();
""")
    chosen = result["chosen"]
    assert chosen["selectedControls"] == ["all_departure", "12"]
    assert "93.0%" in chosen["text"] and "11.0%" not in chosen["text"]
    assert chosen["visible"] == ["static_60_40", "spy_buy_and_hold"]
    assert result["restored"]["strategies"] == chosen["visible"]
    assert result["restored"]["alert"] == chosen["alert"] == "all_departure"
    assert result["restored"]["budget"] == chosen["budget"] == 12
    assert chosen["query"]["unrelated"] == "keep"
    assert result["copied"].endswith("#performance")


def test_last_strategy_stays_visible_and_default_alert_keys_are_removed():
    result = run_js(CONTROL_HARNESS + """
api.renderDecisionShadow();choose(0,'all_departure');choose(1,'12');
choose(0,'risk_worsening');choose(1,'4');
toggle('위험 국면 전략');toggle('60/40');toggle('변동 60/40');toggle('SPY');
console.log(JSON.stringify(snapshot()));
""")
    assert result["visible"] == ["spy_buy_and_hold"]
    assert result["query"]["strategies"] == "spy_buy_and_hold"
    assert "alert" not in result["query"] and "budget" not in result["query"]


PERSISTENT_LOADER = LOADER_HARNESS.replace(
    "query:Object.fromEntries(current.searchParams)",
    "alert:loader.state.decisionResearchTarget,budget:loader.state.decisionResearchBudget,strategies:loader.state.performanceVisible,query:Object.fromEntries(current.searchParams)",
)


def test_real_loader_restores_choices_through_delayed_research():
    result = run_js(PERSISTENT_LOADER + """
(async()=>{current.searchParams.set('alert','all_departure');current.searchParams.set('budget','12');
current.searchParams.set('strategies','realistic_60_40,spy_buy_and_hold');
const response=deferred(),flow=await start(response);loader.syncViewUrl();const pending=snapshot();
response.resolve(full.research);await flow.completion;console.log(JSON.stringify({pending,ready:snapshot()}));})();
""")
    for phase in ("pending", "ready"):
        assert result[phase]["alert"] == "all_departure"
        assert result[phase]["budget"] == 12
        assert set(result[phase]["strategies"]) == {"realistic_60_40", "spy_buy_and_hold"}
        assert result[phase]["query"]["budget"] == "12"


def test_user_choices_while_loading_override_old_link_values():
    result = run_js(PERSISTENT_LOADER + """
(async()=>{current.searchParams.set('alert','all_departure');current.searchParams.set('budget','12');
current.searchParams.set('strategies','realistic_60_40,spy_buy_and_hold');
const response=deferred(),flow=await start(response);
loader.state.decisionResearchTarget='risk_worsening';loader.state.decisionResearchBudget=8;
loader.state.performanceVisible=['spy_buy_and_hold'];loader.syncViewUrl();const chosen=snapshot();
response.resolve(full.research);await flow.completion;console.log(JSON.stringify({chosen,ready:snapshot()}));})();
""")
    assert result["ready"] == result["chosen"]
    assert result["ready"]["budget"] == 8
    assert result["ready"]["strategies"] == ["spy_buy_and_hold"]
