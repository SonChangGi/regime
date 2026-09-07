"""Exercise the actual holdings controls across input and observation changes."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]

HARNESS = r"""
const fs = require('fs'), vm = require('vm'), {createRequire} = require('module');
class Element {
  constructor(tag='div') { this.tagName=tag; this.children=[]; this.dataset={}; this.listeners={}; this.value=''; this.hidden=false; this._text=''; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children=[]; this._text=''; this.append(...nodes); }
  setAttribute(name,value) { this[name]=String(value); }
  addEventListener(name,callback) { (this.listeners[name] ||= []).push(callback); }
  fire(name,target=this) { for (const callback of this.listeners[name] || []) callback({target,preventDefault(){}}); }
  querySelectorAll(selector) { return this.children.flatMap(node => node instanceof Element ? [...(node.tagName===selector ? [node] : []), ...node.querySelectorAll(selector)] : []); }
  get textContent() { return this._text + this.children.map(node=>node.textContent ?? node).join(''); }
  set textContent(value) { this.children=[]; this._text=String(value); }
}
const nodes={};
const document={createElement:tag=>new Element(tag),createTextNode:text=>({textContent:text}),getElementById:id=>nodes[id] ||= new Element()};
let source=fs.readFileSync('web/app.js','utf8');
const marker='const dashboardApi = Object.freeze({';
if (!source.includes(marker)) throw Error('dashboard export boundary not found');
source=source.replace(marker, marker+' _holdingsTest: {state, dom, initializeDom, bindEvents, calculatorPortfolio, syncHoldingsCalculator, renderHoldingsResult},');
const context={module:{exports:{}},require:createRequire(process.cwd()+'/web/app.js'),document,console};
vm.runInNewContext(source,context,{filename:'web/app.js'});
const test=context.module.exports._holdingsTest;
context.window={addEventListener(){}};
test.initializeDom(); test.bindEvents();
const payload=JSON.parse(fs.readFileSync('publication/live/regime-results.json','utf8'));
test.state.raw=payload; test.state.weekly=payload.weekly; test.state.selectedIndex=payload.weekly.length-1;
nodes['holdings-total'].value='10000'; nodes['holdings-target-basis'].value='policy';
const form=nodes['holdings-form'], result=nodes['holdings-result'];
const inputs=()=>nodes['holdings-inputs'].querySelectorAll('input');
const input=asset=>inputs().find(node=>node.dataset.asset===asset);
const intent=payload.research.prospective_decision_shadow.allocation_candidate.current_intent;
const rows=()=>Object.fromEntries(result.querySelectorAll('tbody').flatMap(body=>body.children).map(row=>{
  const values=row.children.map(cell=>cell.textContent);
  return [values[0],values.slice(1).map(value=>Number(value.replace(/[^0-9.+-]/g,'')))];
}));
test.syncHoldingsCalculator();
"""


def run_js(program: str):
    completed = subprocess.run(
        ["node", "-"], input=HARNESS + program, cwd=ROOT,
        text=True, capture_output=True, check=True,
    )
    return json.loads(completed.stdout)


def test_past_observation_hides_current_targets_and_return_restores_draft_and_result():
    result = run_js(r"""
input('CASH').value='20'; input('SPY').value='80';
form.fire('submit'); const before=rows();
test.state.selectedIndex-=8; test.syncHoldingsCalculator();
const past={hidden:nodes['holdings-calculator'].hidden,empty:result.children.length===0,portfolio:test.calculatorPortfolio()};
test.state.selectedIndex=payload.weekly.length-1; test.syncHoldingsCalculator();
console.log(JSON.stringify({before,past,after:rows(),hidden:nodes['holdings-calculator'].hidden,draft:input('SPY').value,caption:result.querySelectorAll('caption')[0].textContent}));
""")
    assert result["past"] == {"hidden": True, "empty": True, "portfolio": None}
    assert not result["hidden"] and result["draft"] == "80"
    assert result["before"] == result["after"]
    assert result["after"]["SPY"][3] == -2000
    assert "공식 실행정책" in result["caption"]


def test_submitted_calculation_updates_on_value_holdings_and_target_changes():
    result = run_js(r"""
form.fire('submit'); const initial=rows();
nodes['holdings-total'].value='20000'; form.fire('input',nodes['holdings-total']); const amount=rows();
input('CASH').value='0'; input('SPY').value='80'; input('TLT').value='20';
form.fire('input',input('TLT')); const rebalance=rows();
intent.target={weights:{SPY:.4,TLT:.4},cash:.2};
nodes['holdings-target-basis'].value='execution'; form.fire('change',nodes['holdings-target-basis']);
console.log(JSON.stringify({initial,amount,rebalance,execution:rows(),caption:result.querySelectorAll('caption')[0].textContent}));
""")
    assert result["initial"]["SPY"][3] == 6000
    assert result["amount"]["SPY"][3] == 12000
    assert result["rebalance"]["SPY"][3] == -4000
    assert result["execution"]["SPY"][3] == -8000
    assert result["execution"]["CASH"][3] == 4000
    assert "이번 실행 목표" in result["caption"]


def test_invalid_edits_replace_old_amounts_and_recover_without_another_submit():
    result = run_js(r"""
form.fire('input',input('CASH')); const pristine=result.children.length===0;
form.fire('submit'); input('SPY').value='80'; form.fire('input',input('SPY'));
const invalid={tables:result.querySelectorAll('table').length,text:result.textContent};
input('CASH').value='20'; form.fire('input',input('CASH')); const recovered=rows();
nodes['holdings-total'].value=''; form.fire('input',nodes['holdings-total']);
const blank={tables:result.querySelectorAll('table').length,text:result.textContent};
nodes['holdings-total'].value='0'; form.fire('input',nodes['holdings-total']);
console.log(JSON.stringify({pristine,invalid,recovered,blank,zero:rows()}));
""")
    assert result["pristine"]
    assert result["invalid"]["tables"] == 0 and "100%" in result["invalid"]["text"]
    assert result["recovered"]["SPY"][3] == -2000
    assert result["blank"]["tables"] == 0 and "평가금액" in result["blank"]["text"]
    assert all(row[3] == 0 for row in result["zero"].values())


def test_comparison_model_never_changes_official_target_and_stale_signal_is_hidden():
    result = run_js(r"""
form.fire('submit'); const before=rows();
test.state.comparisonModel='boundary_student_t'; test.syncHoldingsCalculator(); const comparison=rows();
payload.research.prospective_decision_shadow.current_signal.origin_date=payload.weekly.at(-2).date;
test.syncHoldingsCalculator();
console.log(JSON.stringify({before,comparison,hidden:nodes['holdings-calculator'].hidden,empty:result.children.length===0}));
""")
    assert result["comparison"] == result["before"]
    assert result["hidden"] and result["empty"]


def test_target_asset_changes_preserve_user_holdings_and_recalculate_on_sync():
    result = run_js(r"""
intent.target={weights:{SPY:.4,TLT:.4,XLK:.2},cash:0};
test.syncHoldingsCalculator();
input('CASH').value='0'; input('SPY').value='60'; input('XLK').value='40';
form.fire('submit'); const policy=rows();
nodes['holdings-target-basis'].value='execution'; form.fire('change',nodes['holdings-target-basis']); const execution=rows();
intent.target={weights:{SPY:.3,TLT:.3},cash:.4}; test.syncHoldingsCalculator();
console.log(JSON.stringify({policy,execution,updated:rows(),draft:input('XLK').value}));
""")
    assert result["draft"] == "40"
    assert result["policy"]["XLK"][3] == -4000
    assert result["execution"]["XLK"][3] == -2000
    assert result["updated"]["XLK"][3] == -4000
    assert result["updated"]["CASH"][3] == 4000
