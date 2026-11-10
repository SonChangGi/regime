"""Responsive plotting keeps published values, gaps and probability domains."""
import json
from pathlib import Path
import subprocess

def test_plot_options_preserve_probabilities_returns_and_gaps():
    source = Path(__file__).resolve().parents[1] / "web/chart-engine.js"
    script = """
globalThis.document={documentElement:{dataset:{theme:'dark'}},addEventListener(){}};
require(process.argv[1]);
const spec={dates:['2026-01-02','2026-01-09','2026-01-16'],panels:[{title:'관측',probability:true,percent:true,series:[{label:'Risk-on',values:[0.1,null,0.8],color:'#8eb1ff'}]},{title:'예측',probability:true,percent:true,series:[{label:'Risk-on',values:[0.2,0.3,0.7],color:'#8eb1ff'}]}]};
const o=RegimeCharts.trend(spec);
console.log(JSON.stringify({grids:o.grid.map(g=>[g.left,g.right,g.containLabel]),axes:o.yAxis.map(a=>[a.min,a.max,a.interval]),data:o.series.map(s=>s.data),gaps:o.series.map(s=>s.connectNulls),zoom:o.dataZoom[0].xAxisIndex,changes:[RegimeCharts.rangeChange(1,1.2,"relative"),RegimeCharts.rangeChange(-.1,-.2,"pp"),RegimeCharts.rangeChange(null,1,"relative")],formatted:o.yAxis[0].axisLabel.formatter(.25)}));
"""
    result = json.loads(subprocess.check_output(["node", "-e", script, str(source)], text=True))
    assert result["axes"] == [[0, 1, .25], [0, 1, .25]]
    assert result["data"] == [[.1, None, .8], [.2, .3, .7]]
    assert result["gaps"] == [False, False]
    assert result["zoom"] == [0, 1]
    assert result["formatted"] == "25%"

    assert abs(result["changes"][0] - 20) < 1e-9
    assert result["changes"][1:] == [-10, None]

    assert result["grids"] == [[52, 22, False], [52, 22, False]]


def test_keyboard_navigation_is_single_step_and_retains_local_selection():
    source = Path(__file__).resolve().parents[1] / "web/chart-engine.js"
    script = """
const calls=[],actions=[];let option;
globalThis.document={documentElement:{dataset:{theme:'dark'}},addEventListener(){},querySelectorAll(){return []}};
globalThis.ResizeObserver=globalThis.MutationObserver=class{observe(){}disconnect(){}};
const chart={dispose(){},getOption(){return option},setOption(o){option=o},on(){},getZr(){return{on(){}}},dispatchAction(a){actions.push(a)}};globalThis.echarts={init(){return chart}};
const reset={hidden:true},host={style:{},dataset:{},classList:{add(){}},replaceChildren(){},setAttribute(){},isConnected:true};
const wrap={classList:{add(){}},querySelector(s){return s==='.qe-regime-history'?host:s.includes('.qe-reset')?reset:null}};host.parentElement=wrap;
require(process.argv[1]);
RegimeCharts.history(wrap,{dates:['a','b','c','d'],selection:{initialIndex:2,resetIndex:3},panels:[{series:[{label:'risk',values:[.1,.2,.3,.4]}]}]},i=>calls.push(i));
let parentCalls=0;function press(key){let stopped=false;host.onkeydown({key,preventDefault(){},stopPropagation(){stopped=true}});if(!stopped)parentCalls++;}
press('ArrowLeft');press('ArrowRight');press('Home');press('End');press('ArrowUp');press('ArrowDown');press('Escape');
console.log(JSON.stringify({calls,parentCalls,tooltipIndices:actions.filter(a=>a.type==='showTip').map(a=>a.dataIndex),lastAction:actions.at(-1).type,wrapperTabIndex:wrap.tabIndex}));
"""
    result = json.loads(subprocess.check_output(["node", "-e", script, str(source)], text=True))
    assert result["calls"] == [1, 2, 0, 3, 2, 3, 3]
    assert result["tooltipIndices"] == result["calls"]
    assert result["parentCalls"] == 0
    assert result["lastAction"] == "hideTip"
    assert result["wrapperTabIndex"] == -1
