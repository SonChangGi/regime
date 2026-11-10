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
console.log(JSON.stringify({axes:o.yAxis.map(a=>[a.min,a.max,a.interval]),data:o.series.map(s=>s.data),gaps:o.series.map(s=>s.connectNulls),zoom:o.dataZoom[0].xAxisIndex,formatted:o.yAxis[0].axisLabel.formatter(.25)}));
"""
    result = json.loads(subprocess.check_output(["node", "-e", script, str(source)], text=True))
    assert result["axes"] == [[0, 1, .25], [0, 1, .25]]
    assert result["data"] == [[.1, None, .8], [.2, .3, .7]]
    assert result["gaps"] == [False, False]
    assert result["zoom"] == [0, 1]
    assert result["formatted"] == "25%"
