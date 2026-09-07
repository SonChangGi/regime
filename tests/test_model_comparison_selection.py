"""The selected comparison remains visible without replacing operating identity."""
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def run_js(program):
    result = subprocess.run(["node", "-"], input=program, text=True,
                            capture_output=True, check=True, cwd=ROOT)
    return json.loads(result.stdout)


def test_comparison_chart_keeps_selected_and_operating_models_below_top_six():
    result = run_js("""
const api=require('./web/app.js');
const rows=Array.from({length:11},(_,i)=>({name:`m${i+1}`,rank:i+1,log_loss:(i+1)/10,selection_log_loss:i/10}));
const before=JSON.stringify(rows);
console.log(JSON.stringify({both:api.modelLossComparisonRows(rows,'m11','m10').map(r=>r.name),
 same:api.modelLossComparisonRows(rows,'m10','m10').map(r=>r.name),unchanged:before===JSON.stringify(rows)}));
""")
    assert result["both"] == ["m1", "m2", "m3", "m4", "m10", "m11"]
    assert result["same"] == ["m1", "m2", "m3", "m4", "m5", "m10"]
    assert result["unchanged"] is True


def test_comparison_chart_uses_available_scores_and_original_values():
    result = run_js("""
const api=require('./web/app.js');
console.log(JSON.stringify(api.modelLossComparisonRows([
 {name:'missing-score',rank:1,log_loss:null},
 {name:'selected',rank:3,log_loss:.7,selection_log_loss:.2},
 {name:'operating',rank:2,log_loss:.3,selection_log_loss:.1}], 'selected','operating')));
""")
    assert [row["name"] for row in result] == ["operating", "selected"]
    assert result[1]["holdout"] == .7
    assert result[1]["selection"] == .2


def test_table_selection_moves_without_duplicate_badges_or_changing_champion():
    result = run_js("""
function node(className='') {
 let classes=new Set(className.split(' ').filter(Boolean));
 const n={children:[],dataset:{},attrs:{},textContent:'',
  get className(){return [...classes].join(' ')},set className(value){classes=new Set(value.split(' ').filter(Boolean))},
  classList:{toggle(k,on){if(on)classes.add(k);else classes.delete(k)},contains(k){return classes.has(k)}},
  append(...children){for(const child of children){child.parent=n;n.children.push(child)}},
  setAttribute(k,v){n.attrs[k]=v},removeAttribute(k){delete n.attrs[k]},
  querySelector(selector){for(const child of n.children){if(child.classList.contains(selector.slice(1)))return child;const found=child.querySelector(selector);if(found)return found}return null},
  remove(){n.parent.children=n.parent.children.filter(child=>child!==n)}
 };return n;
}
global.document={createElement:()=>node()};
const api=require('./web/app.js');const body=node();
for(const model of ['operating','selected']){const row=node(model==='operating'?'is-champion':'');row.dataset.model=model;row.append(node('model-name-cell'));body.append(row)}
api.markSelectedComparisonRows(body,'selected');api.markSelectedComparisonRows(body,'selected');
const first=body.children.map(row=>({current:row.attrs['aria-current']||null,badges:row.children[0].children.length}));
api.markSelectedComparisonRows(body,'operating');
console.log(JSON.stringify({first,second:body.children.map(row=>({current:row.attrs['aria-current']||null,badges:row.children[0].children.length,champion:row.classList.contains('is-champion')}))}));
""")
    assert result["first"] == [{"current": None, "badges": 0}, {"current": "true", "badges": 1}]
    assert result["second"] == [
        {"current": "true", "badges": 1, "champion": True},
        {"current": None, "badges": 0, "champion": False},
    ]
