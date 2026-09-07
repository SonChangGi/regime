"""Run the real staged loader while delaying its research response."""
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


HARNESS = r"""
const fs=require('fs'),vm=require('vm'),path=require('path');
const requireApp=require('module').createRequire(path.resolve('web/app.js'));
const full=JSON.parse(fs.readFileSync('publication/live/regime-results.json'));
const core=structuredClone(full);delete core.research;core.weekly=core.weekly.slice(-26);
const source={source:'core',payload:core,expectedWeeklyRows:full.weekly.length,historySidecars:[{}]};
const context=vm.createContext({module:{exports:{}},require:requireApp,URL,URLSearchParams,Intl,Date,setTimeout,clearTimeout});
let program=fs.readFileSync('web/app.js','utf8');
program=program.replace('const dashboardApi = Object.freeze({', `const dashboardApi = Object.freeze({
  testLoader:{state,dom,loadData,selectWeek,syncViewUrl,syncConditionalBasisControl,
    configure(source,response){
      loadDashboardSource=async()=>source;loadResearchSidecar=()=>response;
      loadV5ComparisonSummary=async()=>null;loadSelectionFamilyAudit=async()=>null;
      showAppState=(phase,...details)=>{if(phase==='error')throw new Error(details.join(' '))};setText=()=>{};applyPayloadStateTheme=()=>{};
      populateDateControls=()=>{};showDashboard=()=>{};restoreFragmentAfterRender=()=>{};
      renderStaticSections=()=>syncConditionalBasisControl();renderConditionalStats=()=>syncConditionalBasisControl();
      renderSelectedWeek=()=>{};renderAnalysisCoverage=()=>{};renderModel=()=>{};renderDecisionShadow=()=>{};
      renderTransitionHorizons=()=>{};
      syncHistoryWindowControl=()=>{};setSnapNote=()=>{};ensureHistory=async()=>false;
    }
  },`);
vm.runInContext(program,context);
const api=context.module.exports,loader=api.testLoader;
context.document={documentElement:{dataset:{}}};
let current=new URL('http://localhost/?week='+core.weekly.at(-1).date+'&model=xgboost&window=52&basis=forecast&horizon=1&assets=SPY#history');
context.window={get location(){return current},history:{replaceState(_a,_b,value){current=new URL(value,current)}}};
const node=()=>({value:'',hidden:false,disabled:false,setAttribute(){}});
for(const name of ['analysis-date','week-select','previous-week','next-week','latest-week','conditional-basis-field','screen-reader-status'])loader.dom[name]=node();
loader.dom['conditional-basis-select']={...node(),options:[{value:'forecast'},{value:'observed'}]};
function deferred(){let resolve;const promise=new Promise(r=>resolve=r);return {promise,resolve};}
async function start(response){loader.configure(source,response.promise);const completion=loader.loadData();
  for(let i=0;i<8;i++)await Promise.resolve();return {completion};}
function snapshot(){return {model:loader.state.comparisonModel,basis:loader.state.outcomeBasis,
 week:loader.state.weekly[loader.state.selectedIndex]?.date,pendingWeek:loader.state.pendingHistoryWeek,
 query:Object.fromEntries(current.searchParams)};}
"""


def run_js(scenario):
    result = subprocess.run(
        ["node", "-"], input=HARNESS + "\n" + scenario,
        text=True, capture_output=True, check=True, cwd=ROOT,
    )
    return json.loads(result.stdout)


def test_forecast_basis_survives_core_render_url_sync_and_research_arrival():
    result = run_js("""
(async()=>{const response=deferred(),flow=await start(response);
 loader.syncConditionalBasisControl();loader.syncViewUrl();const pending=snapshot();
 response.resolve(full.research);await flow.completion;
 console.log(JSON.stringify({pending,ready:snapshot()}));})();
""")
    for phase in ("pending", "ready"):
        assert result[phase]["basis"] == "forecast"
        assert result[phase]["query"]["basis"] == "forecast"
        assert result[phase]["model"] == "xgboost"


def test_choices_made_while_research_loads_override_initial_view_intent():
    result = run_js("""
(async()=>{current.searchParams.set('week',full.weekly[0].date);
 const response=deferred(),flow=await start(response);loader.syncViewUrl();
 const pendingWeek=current.searchParams.get('week');
 loader.state.comparisonModel='markov';loader.state.outcomeBasis='observed';
 loader.selectWeek(core.weekly.length-2,true);loader.syncViewUrl();const chosen=snapshot();
 response.resolve(full.research);await flow.completion;
 console.log(JSON.stringify({requestedWeek:full.weekly[0].date,pendingWeek,chosen,ready:snapshot()}));})();
""")
    assert result["pendingWeek"] == result["requestedWeek"]
    assert result["chosen"]["pendingWeek"] is None
    assert result["ready"] == result["chosen"]
    assert result["ready"]["model"] == "markov"
    assert result["ready"]["basis"] == "observed"


def test_a_previous_load_cannot_overwrite_a_more_recent_reload():
    result = run_js("""
(async()=>{const older=deferred(),first=await start(older);
 current.searchParams.set('model','markov');current.searchParams.set('basis','observed');
 const newer=deferred(),second=await start(newer);newer.resolve(full.research);await second.completion;
 const ready=snapshot();older.resolve(full.research);await first.completion;
 console.log(JSON.stringify({ready,afterOlder:snapshot()}));})();
""")
    assert result["ready"] == result["afterOlder"]
    assert result["ready"]["model"] == "markov"
    assert result["ready"]["basis"] == "observed"


def test_forecast_basis_falls_back_only_after_unavailable_research_is_known():
    result = run_js("""
(async()=>{const response=deferred(),flow=await start(response);const pending=snapshot();
 response.resolve(null);await flow.completion;
 console.log(JSON.stringify({pending,ready:snapshot()}));})();
""")
    assert result["pending"]["basis"] == "forecast"
    assert result["ready"]["basis"] == "observed"
    assert result["ready"]["model"] == "xgboost"
