"""Research rerenders retain filter intent independently of API availability."""
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]

HARNESS = r"""
const fs=require('fs'),vm=require('vm'),path=require('path');
const context=vm.createContext({module:{exports:{}},
  require:require('module').createRequire(path.resolve('web/app.js')),Intl,Date});
const program=fs.readFileSync('web/app.js','utf8').replace(
  'const dashboardApi = Object.freeze({',
  'const dashboardApi = Object.freeze({testResearch:{state,dom,renderResearchUpgrades},');
vm.runInContext(program,context);
const api=context.module.exports.testResearch;
function node(tag){
  const n={tag,children:[],attrs:{},listeners:{},className:'',hidden:false,_text:'',_value:null,
    append(...children){n.children.push(...children)},
    replaceChildren(...children){n.children=[...children];n._text=''},
    setAttribute(key,value){n.attrs[key]=String(value)},
    addEventListener(event,handler){n.listeners[event]=handler},
    get textContent(){return n._text+n.children.map(child=>child.textContent).join(' ')},
    set textContent(value){n._text=String(value);n.children=[]},
    get value(){return n._value??(tag==='select'?n.children[0]?.value:'')??''},
    set value(value){n._value=String(value)},
  };return n;
}
context.document={createElement:node};
const container=node('div');api.dom['research-upgrades']=container;
const rows=[
 {target:'risk_worsening',annual_false_alarm_budget:4,score:'binary_xgboost',
  retrospective_diagnostic:{recall:.11}},
 {target:'all_departure',annual_false_alarm_budget:8,score:'binary_xgboost',
  retrospective_diagnostic:{recall:.66}},
 {target:'all_departure',annual_false_alarm_budget:12,score:'binary_xgboost',
  retrospective_diagnostic:{recall:.93}},
];
function descendants(parent,tag){return parent.children.flatMap(child=>[
  ...(child.tag===tag?[child]:[]),...descendants(child,tag)]);}
function show(research={alert_budgets:rows}){
  api.state.raw={meta:{data_as_of:'2026-09-04'},research:research===null?{}:{decision_research_v2:research}};
  api.renderResearchUpgrades();
}
function choose(index,value){const select=descendants(container,'select')[index];
  select.value=value;select.listeners.change();}
function snapshot(){return {hidden:container.hidden,selection:descendants(container,'select').map(n=>n.value),
  intent:[api.state.decisionResearchTarget,api.state.decisionResearchBudget],text:container.textContent};}
"""


def run_js(scenario):
    result = subprocess.run(
        ["node", "-"], input=HARNESS + "\n" + scenario,
        text=True, capture_output=True, check=True, cwd=ROOT,
    )
    return json.loads(result.stdout)


def test_changed_filters_keep_the_same_rows_after_whole_research_rerender():
    result = run_js("""
show();const initial=snapshot();choose(0,'all_departure');choose(1,'12');
const chosen=snapshot();show();console.log(JSON.stringify({initial,chosen,rerendered:snapshot()}));
""")
    assert "11.0%" in result["initial"]["text"]
    assert "93.0%" in result["chosen"]["text"]
    assert "11.0%" not in result["chosen"]["text"]
    assert result["chosen"]["selection"] == ["all_departure", "12"]
    assert result["chosen"]["intent"] == ["all_departure", 12]
    assert result["rerendered"] == result["chosen"]


def test_missing_rows_and_hidden_research_do_not_replace_filter_intent():
    result = run_js("""
show();choose(0,'all_departure');choose(1,'12');const chosen=snapshot();
show({});const missingRows=snapshot();show(null);const hidden=snapshot();
show({alert_budgets:[rows[0]]});const partial=snapshot();show();
console.log(JSON.stringify({chosen,missingRows,hidden,partial,restored:snapshot()}));
""")
    for phase in ("missingRows", "partial"):
        assert result[phase]["selection"] == ["all_departure", "12"]
        assert "선택 조건의 경보 진단이 없습니다." in result[phase]["text"]
        assert "11.0%" not in result[phase]["text"]
        assert "93.0%" not in result[phase]["text"]
    assert result["hidden"]["hidden"] is True
    assert result["hidden"]["selection"] == []
    assert result["hidden"]["intent"] == ["all_departure", 12]
    assert result["restored"] == result["chosen"]


def test_invalid_options_fall_back_but_valid_new_choices_survive_data_changes():
    result = run_js("""
api.state.decisionResearchTarget='removed_target';api.state.decisionResearchBudget=99;
show();const normalized=snapshot();choose(0,'all_departure');choose(1,'8');
show({alert_budgets:[rows[0]]});const missing=snapshot();
choose(1,'12');show();const changed=snapshot();
console.log(JSON.stringify({normalized,missing,changed}));
""")
    assert result["normalized"]["selection"] == ["risk_worsening", "4"]
    assert result["normalized"]["intent"] == ["risk_worsening", 4]
    assert "11.0%" in result["normalized"]["text"]
    assert result["missing"]["selection"] == ["all_departure", "8"]
    assert result["missing"]["intent"] == ["all_departure", 8]
    assert "선택 조건의 경보 진단이 없습니다." in result["missing"]["text"]
    assert result["changed"]["selection"] == ["all_departure", "12"]
    assert result["changed"]["intent"] == ["all_departure", 12]
    assert "93.0%" in result["changed"]["text"]
