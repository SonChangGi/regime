"""Performance chart geometry preserves values, gaps, and comparison semantics."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = r"""
const {createPerformanceRenderers}=require('./web/insights.js');
function node(tag,className,text){return {tag,className:className||'',textContent:text||'',attrs:{},style:{},children:[],append(...nodes){this.children.push(...nodes)},setAttribute(key,value){this.attrs[key]=value}}}
function svg(tag,attrs={}){const n=node(tag,attrs.class);n.attrs=attrs;return n}
const finite=v=>typeof v==='number'&&Number.isFinite(v)?v:null;
const api=createPerformanceRenderers({createElement:node,createSvg:svg,finiteNumber:finite,
 firstValue:(r,ks)=>ks.map(k=>r[k]).find(v=>v!==undefined),
 formatNumber:(v,d=2)=>Number(v.toFixed(d)).toString(),
 formatSignedPercent:(v,d=1)=>finite(v)===null?'—':`${Number((v*100).toFixed(d))}%`,
 formatDate:d=>d,performanceRowValue:(r,k,m)=>finite(r.strategies[k]?.[m]),performanceRowDate:r=>r.date,
 strategySummaryRow:(s,k)=>s[k]||{}});
const all=n=>[n,...n.children.flatMap(all)];
const has=(n,c)=>n.className.split(' ').includes(c);
const find=(n,c)=>all(n).filter(x=>has(x,c));
const keys=['combined','spy_buy_and_hold','realistic_60_40'];
const labels={combined:'실행 기준 · 결합',spy_buy_and_hold:'SPY',realistic_60_40:'60/40'};
const rows=Array.from({length:8},(_,i)=>({date:`2026-01-${String(i+1).padStart(2,'0')}`,strategies:Object.fromEntries(keys.map((k,j)=>[k,{wealth:i===3?null:1+i*.01+j*.0001,drawdown:i===3?null:-i*.01-j*.0001}]))}));
"""


def run_js(program: str):
    result = subprocess.run(["node", "-"], input=HARNESS + program, cwd=ROOT, text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def test_time_charts_retain_missing_spans_and_regular_axes_without_mutating_rows():
    result = run_js(r"""
const before=JSON.stringify(rows);
const wealth=api.renderPerformanceLineChart(rows,keys,labels,'2026-01', 'combined');
const drawdown=api.renderPerformanceDrawdownChart(rows,keys,labels,'2026-01','combined');
const wealthSvg=find(wealth,'performance-chart')[0],ddSvg=find(drawdown,'performance-chart')[0];
const paths=chart=>find(chart,'performance-series').filter(n=>n.tag==='path').map(n=>n.attrs.d);
console.log(JSON.stringify({unchanged:before===JSON.stringify(rows),wealth:paths(wealthSvg),drawdown:paths(ddSvg),
 areas:find(ddSvg,'performance-drawdown-area').map(n=>n.attrs.d),
 ticks:find(wealthSvg,'performance-axis-label').filter(n=>n.attrs['text-anchor']==='end'&&n.attrs.x===51).map(n=>Number(n.textContent)),
 zero:find(ddSvg,'performance-zero-line').length,reference:find(wealthSvg,'performance-reference-line').length,
 dates:find(wealthSvg,'performance-axis-tick').map(n=>n.attrs.x),ddDates:find(ddSvg,'performance-axis-tick').map(n=>n.attrs.x),
 benchmarks:find(wealthSvg,'is-benchmark').length,legend:find(wealth,'performance-legend-swatch').length,
 short:api.renderPerformanceLineChart(rows.slice(0,7),keys,labels,'short','combined')}));
""")
    assert result["unchanged"]
    assert len(result["wealth"]) == len(result["drawdown"]) == 6
    assert all(path.count("M") == 1 for path in result["wealth"] + result["drawdown"])
    assert len(result["areas"]) == 2
    assert all(path.endswith(" Z") for path in result["areas"])
    assert result["zero"] == result["reference"] == 1
    assert len(result["dates"]) == 5 and result["dates"] == result["ddDates"]
    differences = [b-a for a, b in zip(result["ticks"], result["ticks"][1:])]
    assert all(step == pytest.approx(differences[0]) for step in differences)
    assert result["benchmarks"] == 4 and result["legend"] == 3
    assert result["short"] is None


def test_crowded_endpoints_stay_in_bounds_with_separate_exact_values():
    result = run_js(r"""
const ks=Array.from({length:8},(_,i)=>'s'+i),ls=Object.fromEntries(ks.map(k=>[k,k]));
const rs=rows.map(r=>({...r,strategies:Object.fromEntries(ks.map(k=>[k,{wealth:1.1,drawdown:-.1}]))}));
const chart=api.renderPerformanceDrawdownChart(rs,ks,ls,'all','s0');
console.log(JSON.stringify({ys:find(chart,'performance-endpoint-label').map(n=>n.attrs.y),
 values:find(chart,'performance-endpoint-value').map(n=>n.textContent),
 guides:find(chart,'performance-endpoint-guide').map(n=>[n.attrs.y1,n.attrs.y2])}));
""")
    ys = result["ys"]
    assert min(ys) >= 24 and max(ys) <= 188
    assert all(b-a >= 19.99 for a, b in zip(ys, ys[1:]))
    assert result["values"] == ["-10%"] * 8
    assert len(set(pair[0] for pair in result["guides"])) == 1


def test_date_ticks_are_compact_with_full_dates_preserved():
    result = run_js(r"""
const tickLabels=rs=>find(api.renderPerformanceLineChart(rs,keys,labels,'range','combined'),'performance-axis-label')
 .filter(n=>n.children.some(c=>c.tag==='title')).map(n=>({label:n.textContent,title:n.children[0].textContent}));
const longer=Array.from({length:27},(_,i)=>({...rows[i%8],date:new Date(Date.UTC(2025,0,3+i*7)).toISOString().slice(0,10)}));
console.log(JSON.stringify({short:tickLabels(rows),long:tickLabels(longer)}));
""")
    assert len(result["short"]) == len(result["long"]) == 5
    for tick in result["short"]:
        assert tick["label"] == tick["title"][5:10].replace("-", ".")
    for tick in result["long"]:
        assert tick["label"] == tick["title"][:7].replace("-", ".")
    assert result["short"][0]["title"] == "2026-01-01"
    assert result["long"][-1]["title"] == "2025-07-04"


def test_summary_bars_use_signed_zero_and_turnover_does_not_invent_activity():
    result = run_js(r"""
const s={combined:{cumulative_return:-.1,annualized_one_way_turnover:0,annualized_full_l1_turnover:0},
 spy_buy_and_hold:{cumulative_return:.2,annualized_one_way_turnover:.4,annualized_full_l1_turnover:.8},
 realistic_60_40:{cumulative_return:0,annualized_full_l1_turnover:.4}};
const before=JSON.stringify(s);
const summary=api.renderPerformanceFallback(s,keys,labels,'range',4,'combined');
const turnover=api.renderPerformanceTurnover(s,keys,labels,'range',4,'combined');
console.log(JSON.stringify({unchanged:before===JSON.stringify(s),
 bars:find(summary,'performance-summary-track').map(n=>n.children.map(c=>c.style)),
 turnover:find(turnover,'performance-turnover-track').map(n=>n.children[0].style.width),
 values:find(turnover,'performance-turnover-row').map(n=>n.children[2].textContent),
 axes:find(summary,'performance-bar-tick').map(n=>n.textContent)}));
""")
    assert result["unchanged"]
    negative, positive, zero = result["bars"]
    assert float(negative[0]["left"].rstrip("%")) < float(negative[1]["left"].rstrip("%"))
    assert positive[0]["left"] == positive[1]["left"]
    assert zero[0]["width"] == "0%"
    assert result["turnover"] == ["0%", "100%", "50%"]
    assert result["values"] == ["0%", "40%", "20%"]
    assert "0%" in result["axes"]


@pytest.mark.parametrize("gross,net", [(0.12, 0.1), (-0.1, -0.12), (0, 0)])
def test_cost_waterfall_uses_same_scale_and_exact_gross_cost_net(gross: float, net: float):
    result = run_js(f"""
const metrics={{gross_cumulative_return:{gross},cumulative_return:{net}}},before=JSON.stringify(metrics);
const chart=api.renderPerformanceBridge(metrics,'range',20);
console.log(JSON.stringify({{unchanged:before===JSON.stringify(metrics),
 bars:find(chart,'performance-bridge-bar').map(n=>n.attrs),
 values:find(chart,'performance-bridge-value').map(n=>n.textContent),
 connectors:find(chart,'performance-bridge-connector').map(n=>n.attrs),
 zero:find(chart,'performance-zero-line').length,missing:api.renderPerformanceBridge({{}},'range',20)}}));
""")
    assert result["unchanged"] and result["missing"] is None
    bars = result["bars"]
    assert len(bars) == 3 and result["zero"] == 1
    assert all(bar["height"] >= 0 for bar in bars)
    assert len(result["connectors"]) == 2
    assert result["values"][1].endswith("%p")
    if gross or net:
        assert bars[1]["height"] / bars[0]["height"] == pytest.approx(abs(net-gross) / abs(gross))
    else:
        assert all(bar["height"] == 0 for bar in bars)
