"""Exercise the real chart renderer and navigation without a browser dependency."""
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
HARNESS = r"""
const fs=require('fs'),vm=require('vm'),path=require('path');
const context=vm.createContext({module:{exports:{}},
 require:require('module').createRequire(path.resolve('web/app.js')),
 URL,URLSearchParams,Intl,Date,requestAnimationFrame:fn=>fn()});
let program=fs.readFileSync('web/app.js','utf8');
program=program.replace('const dashboardApi = Object.freeze({', `
const originalChartSemantics={observedHistoryMeasure,forecastHistoryMeasure,historyStateForWeek,
 forecastForWeek,actualNextWeekForWeek,forecastEntropyForWeek};
const dashboardApi = Object.freeze({
 testChart:{state,dom,renderHistory,renderTimeline,previewChartDateFromPointer,
  handleChartKeydown,handleTimelineKeydown,resetChartPreview,
  restoreSemantics(payload){
   ({observedHistoryMeasure,forecastHistoryMeasure,historyStateForWeek,forecastForWeek,
     actualNextWeekForWeek,forecastEntropyForWeek}=originalChartSemantics);
   state.raw=payload;state.weekly=payload.weekly;state.selectedIndex=payload.weekly.length-1;
  },
  configure(rows,windowSize){
   state.raw={meta:{result_version:'weekly-regime-result-v4'},immutable:'fixture'};
   state.weekly=rows;state.selectedIndex=Math.min(1,rows.length-1);
   state.historyWindow=windowSize;state.comparisonModel='test';state.modelForecastHorizon=4;
   state.chartPinnedDate=null;state.chartPreviewDate=null;
   selectedHistory=()=>state.weekly.slice(-state.historyWindow);
   historyComparisonMeta=()=>({observedMeasure:'관측 점수',model:state.comparisonModel});
   observedHistoryMeasure=(week,code)=>week?.observed?.[code]??null;
   forecastHistoryMeasure=(week,code)=>week?.predicted?.[code]??null;
   forecastForWeek=week=>week?.predicted?{state:'risk_on',probabilities:week.predicted}:null;
   actualNextWeekForWeek=week=>week.actual;
   forecastEntropyForWeek=week=>week?.predicted?.risk_on==null?null:.5;
   historyStateForWeek=week=>week.currentState;
   modelForecastLabel=()=> 'test';
   selectWeek=index=>{state.selectedIndex=index};
  }
 },`);
vm.runInContext(program,context);
const api=context.module.exports,chart=api.testChart;
let activeNode=null;
class Node {
 constructor(tag='div') {
  this.tag=tag;this.className='';this.attrs={};this.dataset={};this.children=[];this.events={};
  this.hidden=false;this.tabIndex=-1;this._text='';this.scrollLeft=0;
  this.style={setProperty:(key,value)=>{this.style[key]=value}};
  this.classList={contains:name=>this.className.split(/\s+/).includes(name),
   toggle:(name,on)=>{const s=new Set(this.className.split(/\s+/).filter(Boolean));
    (on??!s.has(name))?s.add(name):s.delete(name);this.className=[...s].join(' ')},
   add:(...names)=>names.forEach(name=>this.classList.toggle(name,true)),
   remove:(...names)=>names.forEach(name=>this.classList.toggle(name,false))};
 }
 set textContent(value){this._text=String(value);this.children=[]}
 get textContent(){return this._text+this.children.map(n=>n.textContent).join('')}
 setAttribute(key,value){this.attrs[key]=String(value);if(key==='class')this.className=String(value);
  if(key.startsWith('data-'))this.dataset[key.slice(5).replace(/-([a-z])/g,(_,c)=>c.toUpperCase())]=String(value)}
 getAttribute(key){return this.attrs[key]??null}
 append(...nodes){nodes.forEach(n=>{n.parentNode=this;this.children.push(n)})}
 replaceChildren(...nodes){this.children=[];this._text='';this.append(...nodes)}
 querySelectorAll(selector){return this.children.flatMap(n=>[...(matches(n,selector)?[n]:[]),...n.querySelectorAll(selector)])}
 querySelector(selector){return this.querySelectorAll(selector)[0]??null}
 getBoundingClientRect(){return this.rect||{left:0,top:0,width:this.clientWidth||0,height:560}}
 addEventListener(type,fn){this.events[type]=fn}
 focus(){activeNode=this}
 scrollIntoView(){this.scrolled=true}
 scrollTo(options){this.scrollLeft=options.left}
 click(){this.events.click?.({currentTarget:this})}
}
function matches(node,selector){
 const attr=selector.match(/^\[([^=]+)="([^"]+)"\]$/);
 if(attr)return node.getAttribute(attr[1])===attr[2];
 const [tag,...classes]=selector.split('.');
 return (!tag||node.tag===tag)&&classes.every(c=>node.classList.contains(c));
}
context.document={createElement:tag=>new Node(tag),createElementNS:(_ns,tag)=>new Node(tag)};
context.window={location:new URL('http://localhost/?week=2026-01-02&model=test&window=26#history')};
for(const [,id] of fs.readFileSync('web/index.html','utf8').matchAll(/\bid="([^"]+)"/g))chart.dom[id]=new Node();
const svg=chart.dom['probability-chart'],wrap=chart.dom['probability-chart-wrap'];
function weeks(count){return Array.from({length:count},(_,i)=>{
 const date=new Date(Date.UTC(2025,11,19+i*7)),target=new Date(date);target.setUTCDate(target.getUTCDate()+28);
 return {date:date.toISOString().slice(0,10),currentState:['risk_on','transition','risk_off'][i%3],
  observed:{risk_on:.6,transition:.3,risk_off:.1},predicted:{risk_on:.7,transition:.2,risk_off:.1},
  actual:{date:target.toISOString().slice(0,10),state:null,status:'pending'}};
})}
function semanticFixture(count=8){
 const rows=weeks(count),codes=['risk_on','risk_on',null,'transition','transition','risk_off','risk_off','risk_on'];
 const future=(date,h)=>{const d=new Date(date+'T20:00:00Z');d.setUTCDate(d.getUTCDate()+7*h);return d.toISOString()};
 const vector=state=>Object.fromEntries(api.STATE_ORDER.map(s=>[s,s===state? .8:.1]));
 for(const [i,row] of rows.entries()){
  row.current={state:codes[i%codes.length],memberships:{risk_on:.01,transition:.1,risk_off:.89}};
  row.model_forecasts=[['markov','risk_off'],['xgboost','transition']].map(([model,state])=>({
   model,state,date:future(row.date,1).slice(0,10),probabilities:vector(state)}));
 }
 const asof=rows.at(-1).date+'T20:00:00Z';
 const payload={meta:{result_version:'weekly-regime-result-v5',generation_id:'fixture',data_as_of:asof},
  weekly:rows,model:{champion:'markov'}};
 payload.forecast_enhancements={source_generation_id:'fixture',data_as_of:asof,history:rows.flatMap(row=>
  [['evolving_boundary_ewma',4,'risk_on'],['evolving_boundary_gjr_skewt',4,'risk_off'],
   ['evolving_boundary_gjr_skewt',13,'transition']].map(([model,h,state])=>({
    model,horizon_weeks:h,origin_date:row.date+'T20:00:00Z',target_date:future(row.date,h),
    probabilities:vector(state),actual:'risk_off'})))};
 return payload;
}
function render(rows,width=720,windowSize=rows.length){
 context.window.matchMedia=()=>({matches:width<=760});wrap.clientWidth=width;
 chart.configure(rows,windowSize);chart.renderHistory();
 const canvasWidth=Number(svg.attrs.viewBox.split(' ')[2]);
 svg.scrollWidth=canvasWidth;wrap.scrollWidth=canvasWidth;svg.rect={left:17,top:25,width:canvasWidth,height:560};
 return canvasWidth;
}
const find=(selector)=>svg.querySelectorAll(selector);
const key=(value,target)=>({key:value,currentTarget:target,preventDefault(){this.prevented=true}});
const dateText=date=>date.replaceAll('-','.');
function unchangedState(){return JSON.stringify({raw:chart.state.raw,weekly:chart.state.weekly,
 selectedIndex:chart.state.selectedIndex,model:chart.state.comparisonModel,horizon:chart.state.modelForecastHorizon,
 window:chart.state.historyWindow,url:context.window.location.href})}
function inspection(){return {pinned:chart.state.chartPinnedDate,preview:chart.state.chartPreviewDate,
 active:find('.chart-point.is-active').map(n=>n.dataset.date),
 cursor:Number(svg.querySelector('[data-chart-cursor="true"]')?.attrs.x1),
 readout:chart.dom['chart-readout-date'].textContent,
 selectedLabel:svg.querySelector('.chart-selected-label')?.querySelector('text').textContent}}
"""


def run_js(program):
    completed = subprocess.run(
        ["node", "-"], input=HARNESS + program, cwd=ROOT,
        text=True, capture_output=True, check=True, timeout=20,
    )
    return json.loads(completed.stdout)


@pytest.mark.parametrize("width,count", [(360, 26), (720, 26), (1200, 26), (390, 52)])
def test_responsive_rendered_positions_map_back_to_the_same_dates(width, count):
    result = run_js(f"""
const rows=weeks({count}),canvas=render(rows,{width}),before=unchangedState();
// A shifted/scaled SVG rectangle also covers a horizontally scrolled canvas.
svg.rect={{left:-143,top:25,width:canvas*.75,height:420}};
const points=find('.chart-point.risk_on.observed'),hits=[];
for(const i of [0,Math.floor((rows.length-1)/2),rows.length-1]){{
 const x=Number(points[i].attrs.cx);
 chart.previewChartDateFromPointer({{clientX:svg.rect.left+x/canvas*svg.rect.width,clientY:90}},true);
 hits.push({{...inspection(),expected:rows[i].date,expectedText:dateText(rows[i].date),x,
  endpointDates:find('.chart-endpoint-label').map(n=>n.dataset.date)}});
}}
console.log(JSON.stringify({{canvas,scroll:wrap.classList.contains('is-scroll-mode'),hits,
 unchanged:before===unchangedState(),bounds:[Number(points[0].attrs.cx),Number(points.at(-1).attrs.cx)],
 ticks:find('.chart-date-label').map(n=>({{x:Number(n.attrs.x),date:n.querySelector('title').textContent}})),
 axes:find('.chart-axis-label').map(n=>n.textContent)}}));
""")
    assert result["canvas"] == (728 if count == 52 else width)
    assert result["scroll"] == (count == 52)
    assert result["unchanged"]
    left, right = (42, 66) if result["canvas"] <= 560 else (52, 142)
    assert result["bounds"] == [left, result["canvas"] - right]
    assert result["axes"] == ["0%", "25%", "50%", "75%", "100%"] * 2
    assert 2 <= len(result["ticks"]) <= (4 if width == 360 else 9)
    assert result["ticks"][0]["x"] == left
    assert result["ticks"][-1]["x"] == result["canvas"] - right
    for hit in result["hits"]:
        assert hit["pinned"] == hit["expected"] and hit["preview"] is None
        assert hit["active"] == [hit["expected"]] * 6
        assert hit["readout"] == hit["expectedText"]
        assert hit["selectedLabel"] == hit["expected"][5:].replace("-", ".")
        assert hit["cursor"] == pytest.approx(hit["x"], abs=0.01)
        assert hit["endpointDates"] == [result["ticks"][-1]["date"]] * 6


def test_missing_spans_and_last_origin_control_end_labels_and_selected_dots():
    result = run_js(r"""
const rows=weeks(5);rows[2].predicted=null;rows[4].predicted=null;
render(rows);const before=unchangedState();
const linePath=find('.chart-series.risk_on.forecast')[0].attrs.d;
const endpoints=find('.chart-endpoint-label').map(n=>({date:n.dataset.date,title:n.querySelector('title').textContent}));
chart.handleChartKeydown(key('Home'));chart.handleChartKeydown(key('ArrowRight'));chart.handleChartKeydown(key('ArrowRight'));
console.log(JSON.stringify({path:linePath,endpoints,active:find('.chart-point.is-active').map(n=>n.className),
 missingReadout:chart.dom['chart-readout-forecast-risk-on'].textContent,
 unchanged:before===unchangedState()}));
""")
    assert result["path"].count("M") == 2 and result["path"].count("L") == 1
    assert len(result["endpoints"]) == 3  # Only the observed panel has last-origin values.
    assert all(row["date"] == "2026-01-16" for row in result["endpoints"])
    assert all("관측 점수" in row["title"] and "2026-01-16" in row["title"] for row in result["endpoints"])
    assert len(result["active"]) == 3 and all("observed" in c for c in result["active"])
    assert result["missingReadout"] == "—" and result["unchanged"]


def test_keyboard_and_hover_change_only_chart_inspection_and_restore_pinned_date():
    result = run_js(r"""
const rows=weeks(5);render(rows);const before=unchangedState(),steps=[];
for(const k of ['Home','ArrowLeft','ArrowRight','End','ArrowRight','Escape']){
 const event=key(k);chart.handleChartKeydown(event);steps.push({...inspection(),prevented:event.prevented});
}
const points=find('.chart-point.risk_on.observed');
chart.previewChartDateFromPointer({clientX:17+Number(points[3].attrs.cx),clientY:90},false);
const hover=inspection();chart.resetChartPreview();const restored=inspection();
const other=key('a');chart.handleChartKeydown(other);
console.log(JSON.stringify({steps,hover,restored,otherPrevented:!!other.prevented,unchanged:before===unchangedState()}));
""")
    expected = ["2025-12-19", "2025-12-19", "2025-12-26", "2026-01-16", "2026-01-16", "2025-12-26"]
    assert [step["pinned"] for step in result["steps"]] == expected
    assert all(step["prevented"] and step["active"] == [day] * 6 for step, day in zip(result["steps"], expected))
    assert result["hover"]["pinned"] == "2025-12-26"
    assert result["hover"]["preview"] == "2026-01-09"
    assert result["hover"]["active"] == ["2026-01-09"] * 6
    assert result["restored"]["pinned"] == "2025-12-26"
    assert result["restored"]["preview"] is None
    assert result["restored"]["active"] == ["2025-12-26"] * 6
    assert result["unchanged"] and not result["otherPrevented"]


def test_year_wrappers_keep_chronological_keyboard_order_and_full_history():
    result = run_js(r"""
const rows=weeks(30);render(rows,360,26);chart.renderTimeline();
const timeline=chart.dom['regime-timeline'],buttons=timeline.querySelectorAll('button.timeline-cell');
const groups=timeline.querySelectorAll('.timeline-year-group').map(n=>({year:n.dataset.year,dates:n.querySelectorAll('button.timeline-cell').map(b=>b.dataset.date)}));
const before=chart.state.selectedIndex;
chart.handleTimelineKeydown(key('ArrowRight',buttons[1]));
const crossed=activeNode.dataset.date,afterMove=chart.state.selectedIndex;
chart.handleTimelineKeydown(key('End',activeNode));const end=activeNode.dataset.date;
chart.handleTimelineKeydown(key('Home',activeNode));const home=activeNode.dataset.date;
chart.handleTimelineKeydown(key('Enter',activeNode));const entered=chart.state.selectedIndex;
chart.handleTimelineKeydown(key('ArrowRight',activeNode));chart.handleTimelineKeydown(key(' ',activeNode));
const spaced=chart.state.selectedIndex;
chart.handleTimelineKeydown(key('End',activeNode));chart.handleTimelineKeydown(key('Escape',activeNode));
console.log(JSON.stringify({groups,count:buttons.length,chartCount:chart.state.chartHistory.length,
 crossed,before,afterMove,end,home,entered,spaced,escaped:activeNode.dataset.date,
 tabs:buttons.filter(b=>b.tabIndex===0).map(b=>b.dataset.date),
 labels:buttons.every(b=>b.attrs['aria-label'].includes('관측 국면'))}));
""")
    assert [group["year"] for group in result["groups"]] == ["2025", "2026"]
    dates = [day for group in result["groups"] for day in group["dates"]]
    assert dates == sorted(set(dates)) and result["count"] == 30
    assert result["chartCount"] == 26
    assert result["crossed"] == "2026-01-02"
    assert result["before"] == result["afterMove"] == 1
    assert result["home"] == dates[0] and result["end"] == dates[-1]
    assert result["entered"] == 0 and result["spaced"] == 1
    assert result["tabs"] == [result["escaped"]] == ["2025-12-26"]
    assert result["labels"]


def test_endpoint_layout_preserves_values_and_single_or_empty_history_is_finite():
    result = run_js(r"""
const input=[{y:190,value:.1},{y:189,value:.11},{y:190,value:.12}],before=JSON.stringify(input);
const labels=api.chartEndpointPositions(input,40,200);
render(weeks(1),360);const single={points:find('.chart-point').map(n=>[Number(n.attrs.cx),Number(n.attrs.cy)]),
 labels:find('.chart-endpoint-label').length,bands:find('.chart-regime-band').map(n=>[Number(n.attrs.x),Number(n.attrs.width)])};
chart.previewChartDateFromPointer({clientX:10000,clientY:80},true);single.date=chart.state.chartPinnedDate;
render([],360);chart.handleChartKeydown(key('Home'));chart.previewChartDateFromPointer({clientX:0},true);
console.log(JSON.stringify({labels,unchanged:before===JSON.stringify(input),single,
 empty:{paths:find('.chart-series').length,endpoints:find('.chart-endpoint-label').length,bands:find('.chart-regime-band').length,text:svg.textContent,readout:chart.dom['chart-readout-date'].textContent}}));
""")
    assert result["unchanged"]
    assert sorted(row["value"] for row in result["labels"]) == [0.1, 0.11, 0.12]
    ys = [row["labelY"] for row in result["labels"]]
    assert min(ys) >= 40 and max(ys) <= 200
    assert all(right - left >= 20 for left, right in zip(ys, ys[1:]))
    assert len(result["single"]["points"]) == result["single"]["labels"] == 6
    assert all(x == 168 and 0 <= y <= 560 for x, y in result["single"]["points"])
    assert result["single"]["date"] == "2025-12-19"
    assert result["single"]["bands"] == [[42, 252], [42, 252]]
    assert result["empty"]["paths"] == result["empty"]["endpoints"] == result["empty"]["bands"] == 0
    assert "없습니다" in result["empty"]["text"] and result["empty"]["readout"] == "—"


def test_regime_meaning_uses_published_current_state_and_selected_model_horizon():
    result = run_js(r"""
const payload=semanticFixture(),before=JSON.stringify(payload);
render(payload.weekly);chart.restoreSemantics(payload);
const origin=payload.weekly.at(-1),cases=[];
for(const [model,h] of [['markov',1],['xgboost',1],['evolving_boundary_ewma',4],
 ['evolving_boundary_gjr_skewt',4],['evolving_boundary_gjr_skewt',13],['evolving_boundary_ewma',13]]){
 chart.state.comparisonModel=model;chart.state.modelForecastHorizon=h;
 cases.push({observed:api.historyStateForWeek(origin,'observed',model,payload),
  predicted:api.historyStateForWeek(origin,'forecast',model,payload),
  actual:api.actualNextWeekForWeek(origin,model,payload)});
}
console.log(JSON.stringify({cases,missingCurrent:api.historyStateForWeek(payload.weekly[2],'observed','markov',payload),
 unchanged:before===JSON.stringify(payload)}));
""")
    assert result["unchanged"] and result["missingCurrent"] is None
    assert [row["observed"] for row in result["cases"]] == ["risk_on"] * 6
    assert [row["predicted"] for row in result["cases"]] == [
        "risk_off", "transition", "risk_on", "risk_off", "transition", None,
    ]
    assert [row["actual"]["status"] for row in result["cases"]] == ["pending"] * 5 + ["unavailable"]
    assert all(row["actual"]["state"] is None for row in result["cases"])


def test_regime_bands_merge_only_adjacent_known_states_at_week_midpoints():
    result = run_js(r"""
const states=['risk_on','risk_on',null,'transition','transition','risk_off','risk_off','risk_on'];
const before=JSON.stringify(states);
console.log(JSON.stringify({bands:api.chartRegimeBands(states,50,750),unchanged:before===JSON.stringify(states),
 single:api.chartRegimeBands(['transition'],42,294),
 gap:api.chartRegimeBands(['risk_on','unknown','risk_on'],0,200),
 empty:[[],[null],['unknown']].map(s=>api.chartRegimeBands(s,42,294)),
 invalidBounds:[api.chartRegimeBands(['risk_on'],42,42),api.chartRegimeBands(['risk_on'],43,42)]}));
""")
    assert result["unchanged"]
    assert result["bands"] == [
        {"state": "risk_on", "startIndex": 0, "endIndex": 1, "left": 50, "right": 200},
        {"state": "transition", "startIndex": 3, "endIndex": 4, "left": 300, "right": 500},
        {"state": "risk_off", "startIndex": 5, "endIndex": 6, "left": 500, "right": 700},
        {"state": "risk_on", "startIndex": 7, "endIndex": 7, "left": 700, "right": 750},
    ]
    assert result["single"] == [
        {"state": "transition", "startIndex": 0, "endIndex": 0, "left": 42, "right": 294},
    ]
    assert [(b["left"], b["right"]) for b in result["gap"]] == [(0, 50), (150, 200)]
    assert result["empty"] == [[], [], []] and result["invalidBounds"] == [[], []]


def test_rendered_bands_follow_model_horizon_and_keep_pending_actuals_neutral():
    result = run_js(r"""
const payload=semanticFixture(),before=JSON.stringify(payload);
render(payload.weekly);chart.restoreSemantics(payload);
const cases=[];
for(const [model,h] of [['markov',1],['xgboost',1],['evolving_boundary_ewma',4],
 ['evolving_boundary_gjr_skewt',4],['evolving_boundary_gjr_skewt',13],['evolving_boundary_ewma',13]]){
 chart.state.comparisonModel=model;chart.state.modelForecastHorizon=h;chart.renderHistory();
 chart.handleChartKeydown(key('End'));
 cases.push({h,bands:find('.chart-regime-band').map(n=>({panel:n.dataset.chartPanel,state:n.dataset.state,
  start:n.dataset.startDate,end:n.dataset.endDate,x:Number(n.attrs.x),width:Number(n.attrs.width),
  title:n.querySelector('title').textContent})),
  lastActual:find('.actual-outcome-marker').at(-1).className,
  readout:chart.dom['chart-readout-actual'].textContent,
  target:chart.dom['chart-readout-target-date'].textContent,
  aria:chart.dom['chart-selection-readout'].attrs['aria-label']});
}
// Selecting a historical cutoff must not reveal outcomes already known later in the payload.
chart.state.comparisonModel='evolving_boundary_ewma';chart.state.modelForecastHorizon=4;
chart.state.selectedIndex=1;chart.renderHistory();
const historicalActuals=find('.actual-outcome-marker').map(n=>n.className);
console.log(JSON.stringify({cases,historicalActuals,unchanged:before===JSON.stringify(payload)}));
""")
    assert result["unchanged"]
    expected_observed = [
        ("risk_on", "2025-12-19", "2025-12-26"),
        ("transition", "2026-01-09", "2026-01-16"),
        ("risk_off", "2026-01-23", "2026-01-30"),
        ("risk_on", "2026-02-06", "2026-02-06"),
    ]
    for row, predicted in zip(result["cases"], ["risk_off", "transition", "risk_on", "risk_off", "transition", None]):
        observed = [b for b in row["bands"] if b["panel"] == "observed"]
        assert [(b["state"], b["start"], b["end"]) for b in observed] == expected_observed
        assert all("관측 국면" in b["title"] and b["start"] in b["title"] and b["end"] in b["title"] for b in observed)
        forecast = [b for b in row["bands"] if b["panel"] == "forecast"]
        if predicted:
            assert len(forecast) == 1
            assert forecast[0]["state"] == predicted
            assert (forecast[0]["start"], forecast[0]["end"]) == ("2025-12-19", "2026-02-06")
            assert (forecast[0]["x"], forecast[0]["width"]) == (52, 526)
            assert f'{row["h"]}주 예측 국면' in forecast[0]["title"]
            assert "pending" in row["lastActual"] and row["readout"] == "결과 대기"
            assert row["target"].startswith("→ ")
            assert f'실제 {row["h"]}주 후 결과 대기' in row["aria"]
        else:
            assert forecast == [] and "unavailable" in row["lastActual"]
            assert row["readout"] == "—" and row["target"] == "대상 —"
        assert not any(code in row["lastActual"] for code in ("risk_on", "transition", "risk_off"))
    assert all("pending" in classes for classes in result["historicalActuals"])


@pytest.mark.parametrize("width", [411, 412])
def test_regime_background_geometry_precedes_lines_without_interior_labels(width):
    result = run_js(f"""
const payload=semanticFixture(3);
payload.weekly.forEach((row,i)=>row.current.state=api.STATE_ORDER[i]);
render(payload.weekly,{width});chart.restoreSemantics(payload);
chart.state.comparisonModel='markov';chart.state.modelForecastHorizon=1;chart.renderHistory();
const panel=find('.chart-panel-bg.observed')[0];
const bands=find('.chart-regime-band').filter(n=>n.dataset.chartPanel==='observed');
console.log(JSON.stringify({{bands:bands.map(n=>[Number(n.attrs.x),Number(n.attrs.width)]),
 interiorLabels:find('.chart-regime-band-label').length,
 paintOrder:bands.every(n=>svg.children.indexOf(panel)<svg.children.indexOf(n)
  &&svg.children.indexOf(n)<svg.children.indexOf(find('.chart-series')[0]))}}));
""")
    assert len(result["bands"]) == 3 and result["paintOrder"]
    assert result["interiorLabels"] == 0
    assert result["bands"] == (
        [[42, 75.75], [117.75, 151.5], [269.25, 75.75]]
        if width == 411 else [[42, 76], [118, 152], [270, 76]]
    )
