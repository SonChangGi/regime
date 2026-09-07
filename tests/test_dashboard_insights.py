"""Behavioral contracts for portfolio, uncertainty, and deferred history views."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def run_js(program: str):
    completed = subprocess.run(
        ["node", "-"], input=program, text=True, capture_output=True, check=True,
        cwd=ROOT,
    )
    return json.loads(completed.stdout)


def test_weighting_switch_changes_estimate_interval_and_comparable_benchmark_together():
    result = run_js("""
const api = require('./web/insights.js');
const row = {mean_return: .02,mean_return_ci95_lower: -.01,mean_return_ci95_upper: .04,
 excess_mean_return: .01,unconditional_benchmark_mean_return:.01,
 episode_equal_mean_return:-.03,episode_equal_mean_return_ci95_lower:-.05,
 episode_equal_mean_return_ci95_upper:-.01,episode_equal_excess_return:-.02,
 episode_equal_unconditional_benchmark_mean_return:-.01,n:80,unique_episodes:8,non_overlapping_n:20};
console.log(JSON.stringify({weekly:api.conditionalMetrics(row,'weekly'),episode:api.conditionalMetrics(row,'episode'),missing:api.conditionalMetrics(null)}));
""")
    assert result["weekly"]["mean"] == 0.02
    assert result["weekly"]["lower"] == -0.01
    assert result["weekly"]["benchmark"] == 0.01
    assert result["episode"]["mean"] == -0.03
    assert result["episode"]["upper"] == -0.01
    assert result["episode"]["benchmark"] == -0.01
    assert result["episode"]["episodes"] == 8
    assert result["episode"]["sample"] == 80
    assert result["episode"]["nonOverlapping"] == 20
    assert result["missing"]["mean"] is None


def test_portfolio_adjustment_distinguishes_cash_entry_rebalance_and_invalid_holdings():
    result = run_js("""
const api=require('./web/insights.js');
console.log(JSON.stringify({
 entry:api.portfolioAdjustment({CASH:1},{SPY:.6,TLT:.4},10000),
 rebalance:api.portfolioAdjustment({SPY:.8,TLT:.2},{SPY:.6,TLT:.4},10000),
 invalid:api.portfolioAdjustment({SPY:.8,TLT:.4},{SPY:.6,TLT:.4}),
 blank:api.portfolioAdjustment({SPY:NaN,TLT:0,CASH:1},{SPY:.6,TLT:.4}),
 missing:api.portfolioAdjustment({CASH:1},null)
}));
""")
    assert result["entry"]["oneWay"] == 1
    assert result["entry"]["fullL1"] == 1
    assert abs(result["rebalance"]["oneWay"] - .2) < 1e-10
    assert abs(result["rebalance"]["fullL1"] - .4) < 1e-10
    spy = next(row for row in result["rebalance"]["rows"] if row["asset"] == "SPY")
    assert abs(spy["amount"] + 2000) < 1e-8
    assert result["invalid"]["error"]
    assert result["blank"]["error"]
    assert result["missing"]["error"]


def test_deadline_is_not_reused_as_a_future_issue_and_stability_axis_is_inverted():
    result = run_js("""
const api=require('./web/insights.js');
console.log(JSON.stringify({
 timing:api.forecastTiming({status:'active',target_at:'2026-09-04T20:00:00Z'},{},'2026-09-08',Date.parse('2026-09-06T00:00:00Z')),
 point:api.contextPosition({trend:.5,stress:.8}),
 missing:api.contextPosition({trend:.5,stress:null})
}));
""")
    assert result["timing"]["elapsed"] is True
    assert result["timing"]["nextIssueAt"] is None
    assert result["timing"]["confirmedEntry"] is False
    assert result["point"]["stress"] == -.8
    assert result["point"]["x"] > 50
    assert result["point"]["y"] > 50
    assert result["missing"]["stress"] is None


def test_history_chunks_require_matching_generation_order_and_full_cardinality():
    result = run_js("""
const fs=require('fs');const api=require('./web/app.js');
const payload=JSON.parse(fs.readFileSync('publication/live/regime-results.json'));
const weekly=payload.weekly;delete payload.research;
const fullCount=weekly.length;payload.weekly=weekly.slice(-26);
const old=weekly.slice(0,-26),parts=[];
for(let i=0;i<old.length;i+=104)parts.push(old.slice(i,i+104));
const envelope={schema_version:'regime-dashboard-core/2',generation_id:payload.meta.generation_id,
 source_payload_sha256:'a'.repeat(64),payload,research_sidecar:{path:'regime-research.json',sha256:'b'.repeat(64)},
 history_sidecars:parts.map((rows,i)=>({path:`regime-history-${String(i).padStart(3,'0')}.json`,sha256:'c'.repeat(64),row_count:rows.length,start:rows[0].date,end:rows.at(-1).date}))};
const source=api.validateCoreEnvelope(envelope);
const documents=parts.map(rows=>({schema_version:'regime-dashboard-history/1',generation_id:source.generationId,source_payload_sha256:source.sourcePayloadSha256,weekly:rows}));
const throws=(fn)=>{try{fn();return false;}catch{return true;}};
const foreign=structuredClone(documents);foreign[0].generation_id='another';
const duplicate=structuredClone(documents);duplicate[0].weekly[1]=duplicate[0].weekly[0];
const reverse=structuredClone(documents);[reverse[0].weekly[1],reverse[0].weekly[2]]=[reverse[0].weekly[2],reverse[0].weekly[1]];
const pathTraversal=structuredClone(envelope);pathTraversal.history_sidecars[0].path='../regime-history-000.json';
console.log(JSON.stringify({count:api.mergeHistoryParts(source,documents).length,fullCount,
 partialErrors:api.validatePayload(payload,{expectedWeeklyRows:fullCount}).errors,
 wronglyFullErrors:api.validatePayload(payload).errors,
 foreign:throws(()=>api.mergeHistoryParts(source,foreign)),duplicate:throws(()=>api.mergeHistoryParts(source,duplicate)),
 reverse:throws(()=>api.mergeHistoryParts(source,reverse)),traversal:api.validateCoreEnvelope(pathTraversal)}));
""")
    assert result["count"] == result["fullCount"]
    assert result["partialErrors"] == []
    assert result["wronglyFullErrors"]
    assert result["foreign"] and result["duplicate"] and result["reverse"]
    assert result["traversal"] is None


def test_weighting_is_preserved_in_view_state_without_mutating_payload():
    result = run_js("""
const fs=require('fs'),api=require('./web/app.js');
const p=JSON.parse(fs.readFileSync('publication/live/regime-results.json'));const before=JSON.stringify(p);
const view=api.parseViewState('?basis=forecast&weighting=weekly&horizon=13',p,p.weekly);
console.log(JSON.stringify({view,unchanged:before===JSON.stringify(p)}));
""")
    assert result["view"]["weighting"] == "weekly"
    assert result["view"]["horizon"] == 13
    assert result["unchanged"] is True


def test_research_scope_uses_fixed_research_dates_not_the_selected_week():
    result = run_js("""
const api=require('./web/insights.js');
const p={meta:{data_as_of:'2026-08-28T20:00:00Z'},weekly:[{date:'2026-01-23'}],research:{extensions:{downside:{as_of:'2026-08-21T20:00:00Z',latest:[{horizon_weeks:4,origin_date:'2026-08-14T20:00:00Z'},{horizon_weeks:13}]}}}};
const earlier=api.researchScope(p);p.weekly[0].date='2026-02-13';
console.log(JSON.stringify({earlier,later:api.researchScope(p),missing:api.researchScope({research:{extensions:{downside:{latest:[{horizon_weeks:4}]}}}})}));
""")
    assert result["earlier"] == result["later"]
    assert result["earlier"]["asOf"] == "2026-08-28T20:00:00Z"
    assert result["earlier"]["downside"][0]["origin"] == "2026-08-14T20:00:00Z"
    assert result["earlier"]["downside"][1]["origin"] == "2026-08-21T20:00:00Z"
    assert result["missing"]["downside"][0]["origin"] is None


def test_sensitivity_ranges_keep_control_separate_from_evaluated_grid():
    result = run_js("""
const api=require('./web/insights.js');
console.log(JSON.stringify(api.sensitivityRanges({weekly_flip_rate:[{spec_id:'operating_control',value:.9},{spec_id:'grid-a',value:.1},{spec_id:'grid-b',value:.3}],transition_jaccard:[{spec_id:'operating_control',value:1},{spec_id:'grid-a',value:null}]})));
""")
    assert result[0]["control"] == .9
    assert result[0]["lower"] == .1
    assert result[0]["upper"] == .3
    assert result[1]["control"] == 1
    assert result[1]["lower"] is None
    assert result[2]["control"] is None
