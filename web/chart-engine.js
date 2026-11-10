/* Shared 0.4.2 chart grammar, with Regime probabilities and meaning colours preserved. */
(function(root){
 'use strict';
 const colors={risk_on:'#8eb1ff',transition:'#f3c46f',risk_off:'#ff9abd'};
 const mounted=new Set();
 const palette=['#63cbd4','#b4a1ed','#e8bb62','#83bce8'];
 function tokens(){
  const style=root.getComputedStyle?.(document.documentElement),light=document.documentElement.dataset.theme==='light';
  const token=(key,fallback)=>style?.getPropertyValue('--qs-'+key).trim()||fallback;
  return {text:token('text',light?'#18253a':'#edf2fb'),muted:token('muted',light?'#526480':'#a8b5cc'),grid:token('grid',light?'#dde5f0':'#2b3850'),surface:token('raised',light?'#e9eff7':'#222d42')};
 }
// Direct range interaction. Native vertical touch scrolling remains available.
function bindDirectRange(host, chart, spec, onRange, onClear, onInspect){
 if(!host.addEventListener)return {clear(){},dispose(){}};
 const doc=host.ownerDocument,output=doc.createElement('div'),band=doc.createElement('div');
 output.className='qe-direct-range';output.setAttribute('aria-live','polite');output.hidden=true;
 output.dataset.quantChartReadout='';
 host.after(output);
 const presentation=globalThis.QuantRangeReadout;
 const presenter=presentation?.createRangeReadout(host,output);
 Object.assign(band.style,{position:'absolute',inset:'0',pointerEvents:'none',zIndex:'2',overflow:'hidden'});band.hidden=true;
 host.style.position='relative';host.style.touchAction='pan-y';host.style.userSelect='none';host.append(band);
 let drag=null,range=null,suppress=false;
 const index=x=>Math.max(0,Math.min(spec.dates.length-1,Math.round(chart.convertFromPixel({xAxisIndex:0},x))));
 const local=e=>{const rect=host.getBoundingClientRect();return [e.clientX-rect.left,e.clientY-rect.top];};
 const hit=(point,clamp)=>clamp||chart.containPixel({gridIndex:'all'},point)?index(point[0]):null;
 const visible=()=>spec.series.filter(s=>chart.getOption().legend?.[0]?.selected?.[s.legendName||s.name]!==false);
 const pixels=axis=>range.map(i=>chart.convertToPixel({xAxisIndex:axis},i));
 const geometry=()=>presentation?.echartsGeometry(chart,pixels(0),visible().filter(s=>!s.panel).flatMap(s=>range.flatMap(i=>
   Number.isFinite(s.values[i])?[chart.convertToPixel({gridIndex:0},[i,s.values[i]])]:[])));
 const draw=()=>{
   if(!range)return;
   band.replaceChildren();
   const grids=chart.getOption().grid||[];
   for(let i=0;i<grids.length;i++){
     const box=presentation?.echartsGeometry(chart,null,null,i)?.bounds;
     if(!box)continue;
     const [a,b]=pixels(i),left=Math.max(box.left,a),right=Math.min(box.left+box.width,b);
     if(right<left)continue;
     const mark=doc.createElement('span');
     Object.assign(mark.style,{position:'absolute',left:left+'px',top:box.top+'px',width:Math.max(1,right-left)+'px',height:box.height+'px',
       background:'color-mix(in srgb,var(--qs-accent,#78d9e1) 13%,transparent)',borderInline:'1px solid color-mix(in srgb,var(--qs-accent,#78d9e1) 70%,transparent)'});
     band.append(mark);
   }
   presenter?.place(geometry());
 };
 const format=(v,s)=>Number.isFinite(v)?(v*(s.scale||1)).toLocaleString('ko-KR',{maximumFractionDigits:2})+(s.unit||''):'관측 없음';
 const clear=()=>{if(range)chart.setOption({tooltip:{showContent:true}});range=null;band.hidden=true;presenter?.clear();output.hidden=true;onClear?.();};
 const present=(a,b)=>{
   if(!Number.isFinite(a)||!Number.isFinite(b))return;
   if(!range)chart.setOption({tooltip:{showContent:false}});chart.dispatchAction({type:'hideTip'});
   range=[Math.min(a,b),Math.max(a,b)].map(i=>Math.max(0,Math.min(spec.dates.length-1,Math.round(i))));
   const [lo,hi]=range;
   const items=visible().map(s=>{
     const x=s.values[lo],y=s.values[hi],valid=Number.isFinite(x)&&Number.isFinite(y);
     const useRatio=s.measure!=='pp'&&s.measure!=='absolute';
     const relative=useRatio&&x>0,delta=!valid?null:relative?(y/x-1)*100:(y-x)*(s.scale||1);
     const unit=s.measure==='pp'?'%p':relative?'%':s.unit||'';
     const change=presentation?.signedChange(delta,unit)??(delta==null?'—':delta.toLocaleString('ko-KR',{maximumFractionDigits:2})+unit);
     return {name:s.name,change:valid&&useRatio&&!relative?'Δ '+change:change,
       detail:format(x,s)+' → '+format(y,s)+(valid&&useRatio&&!relative?' · 상대 변화율 미제공':''),color:s.color};
   });
   const start=spec.dates[lo],end=spec.dates[hi];
   band.hidden=false;draw();
   if(presenter)presenter.show({start,end,items,dense:new Set(visible().map(s=>s.panel||0)).size>1},geometry());
   else{output.textContent=start+' → '+end+' · '+items.map(i=>i.name+' '+i.detail+' · '+i.change).join(' · ');output.hidden=false;}
   onRange?.(lo,hi);
 };
 const refresh=()=>{if(range){chart.setOption({tooltip:{showContent:false}});draw();}};
 const bindTouch=root.QuantTouchChart;
 const touchCleanup=bindTouch?.bindTouchChart(host,{
   hit,inspect:i=>{if(onInspect)onInspect(i);else chart.dispatchAction({type:'showTip',seriesIndex:0,dataIndex:i});},
   compare:present,clear,rangeReadout:()=>output,navigate:(a,b)=>bindTouch.navigateECharts(chart,a,b)
 });
 const down=e=>{if(e.button!==0||e.pointerType==='touch')return;suppress=false;if(e.altKey)return;const value=hit(local(e));if(value!=null)drag={id:e.pointerId,x:e.clientX,index:value,active:false};};
 const move=e=>{if(!drag||drag.id!==e.pointerId||(!drag.active&&Math.abs(e.clientX-drag.x)<6))return;if(!drag.active){drag.active=true;try{host.setPointerCapture?.(e.pointerId);}catch{}}e.preventDefault();e.stopImmediatePropagation();present(drag.index,index(local(e)[0]));};
 const up=e=>{if(drag?.id!==e.pointerId)return;suppress=drag.active;drag=null;};
 const click=e=>{if(suppress){e.preventDefault();e.stopImmediatePropagation();suppress=false;}else if(hit(local(e))!=null)clear();};
 const key=e=>{if(e.key==='Escape')clear();};
 host.addEventListener('pointerdown',down,true);host.addEventListener('pointermove',move,{capture:true,passive:false});host.addEventListener('pointerup',up,true);host.addEventListener('pointercancel',up,true);host.addEventListener('click',click,true);host.addEventListener('keydown',key,true);
 chart.on('datazoom',draw);for(const event of ['legendselectchanged','legendselected','legendunselected'])chart.on(event,()=>{if(range)present(...range);});
 const resizeRange=typeof ResizeObserver!=='undefined'?new ResizeObserver(draw):null;resizeRange?.observe(host);
 return {clear,present,refresh,dispose(){touchCleanup?.();resizeRange?.disconnect();for(const [type,fn]of [['pointerdown',down],['pointermove',move],['pointerup',up],['pointercancel',up],['click',click],['keydown',key]])host.removeEventListener(type,fn,true);band.remove();presenter?.destroy();output.remove();}};
}

 function mount(host,build,height,onSelect,selection={}){
  if(!root.echarts)return;
  const previous=host._qe?.getOption();host._qe?.dispose();host.replaceChildren();host.classList.add('qe-responsive-chart');host.style.height=height+'px';host.tabIndex=0;
  const chart=root.echarts.init(host,null,{renderer:'svg'});host._qe=chart;mounted.add(host);
  const option=build();if(previous?.legend?.[0]?.selected&&option.legend)option.legend.selected=previous.legend[0].selected;
  chart.setOption(option);
  const direct=bindDirectRange(host,chart,{dates:option.xAxis[0].data,series:option.series.filter(s=>s.type==='line').map(s=>({name:(option.title?.[s.xAxisIndex]?.text?option.title[s.xAxisIndex].text+' · ':'')+s.name,legendName:s.name,values:s.data,panel:s.xAxisIndex||0,color:s.lineStyle?.color,measure:s.rangePercent?'pp':'relative',scale:s.rangePercent?100:1,unit:s.rangePercent?'%':''}))},undefined,undefined,i=>choose(i));
  let reset=host.parentElement.querySelector(':scope > .qe-reset');if(!reset){reset=document.createElement('button');reset.type='button';reset.className='qe-reset';reset.textContent='전체 기간';reset.dataset.quantResetView='';host.after(reset);}reset.hidden=true;
  reset.onclick=()=>chart.dispatchAction({type:'dataZoom',start:0,end:100});host.ondblclick=reset.onclick;chart.on('datazoom',()=>{const z=chart.getOption().dataZoom?.[0];reset.hidden=!z||(z.start===0&&z.end===100);});
  const lastIndex=option.xAxis?.[0]?.data?.length-1||0;let selected=Number.isInteger(selection.initialIndex)&&selection.initialIndex>=0?selection.initialIndex:lastIndex;
  const choose=i=>{const count=chart.getOption().xAxis?.[0]?.data?.length||0;selected=Math.max(0,Math.min(count-1,i));onSelect?.(selected);chart.dispatchAction({type:'showTip',seriesIndex:0,dataIndex:selected});};
  chart.getZr().on('click',e=>{const x=chart.convertFromPixel({xAxisIndex:0},e.offsetX);if(Number.isFinite(x))choose(Math.round(x));});
  let anchor=null;host.onkeydown=e=>{if(e.key==='0'){e.preventDefault();reset.click();return;}if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home','End','Escape'].includes(e.key)){e.preventDefault();e.stopPropagation();if(e.key==='Escape'){anchor=null;choose(Number.isInteger(selection.resetIndex)&&selection.resetIndex>=0?selection.resetIndex:lastIndex);chart.dispatchAction({type:'hideTip'});}else{if(e.shiftKey)anchor??=selected;else{anchor=null;direct.clear();}choose(e.key==='Home'?0:e.key==='End'?lastIndex:selected+(['ArrowLeft','ArrowUp'].includes(e.key)?-1:1));if(e.shiftKey)direct.present(anchor,selected);}}};
  const resize=new ResizeObserver(()=>{if(host.clientWidth>0){chart.resize();direct.refresh();}});resize.observe(host);
  const theme=new MutationObserver(()=>{const current=chart.getOption(),next=build();if(next.legend)next.legend.selected=current.legend?.[0]?.selected;next.dataZoom=current.dataZoom;chart.setOption(next,true);direct.refresh();});theme.observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
  const dispose=chart.dispose.bind(chart);chart.dispose=()=>{resize.disconnect();theme.disconnect();mounted.delete(host);direct.dispose();dispose();};host.dataset.renderer='echarts';
 }
 function trend(spec){const t=tokens(),panels=spec.panels||[{title:'',series:spec.series}];
  return {animation:false,textStyle:{color:t.text,fontFamily:'Pretendard, sans-serif'},grid:panels.map((_,i)=>({left:52,right:22,top:46+i*270,height:205,containLabel:false})),
   title:panels.map((p,i)=>({text:p.title,left:12,top:14+i*270,textStyle:{color:t.text,fontSize:13,fontWeight:600}})),
   legend:{type:'scroll',top:panels.length*270+2,textStyle:{color:t.muted},pageTextStyle:{color:t.muted}},
   tooltip:{trigger:'axis',confine:true,hideDelay:0,transitionDuration:0,className:'qe-point-tooltip',backgroundColor:t.surface,borderColor:t.grid,textStyle:{color:t.text,fontSize:12},axisPointer:{type:'line'}},axisPointer:{link:[{xAxisIndex:'all'}]},
   xAxis:panels.map((_,i)=>({type:'category',gridIndex:i,data:spec.dates,boundaryGap:false,axisLabel:{color:t.muted,hideOverlap:true,fontSize:11,formatter:v=>v.slice(2,7)},axisTick:{show:false},axisLine:{lineStyle:{color:t.grid}}})),
   yAxis:panels.map((p,i)=>({type:'value',gridIndex:i,min:p.probability?0:undefined,max:p.probability?1:undefined,interval:p.probability?.25:undefined,scale:true,splitNumber:4,axisLabel:{color:t.muted,fontSize:11,formatter:v=>p.percent?`${+(v*100).toFixed(1)}%`:`${+v.toFixed(2)}`},splitLine:{lineStyle:{color:t.grid,type:'dashed'}}})),
   dataZoom:[{type:'inside',xAxisIndex:panels.map((_,i)=>i),filterMode:'none',preventDefaultMouseMove:false,moveOnMouseMove:'alt'}],
   series:panels.flatMap((p,i)=>p.series.map((s,j)=>({name:s.label,rangePercent:p.percent,type:'line',xAxisIndex:i,yAxisIndex:i,data:s.values,showSymbol:false,connectNulls:false,lineStyle:{width:2.2,color:s.color||palette[j%palette.length],type:s.dashed?'dashed':'solid'},itemStyle:{color:s.color||palette[j%palette.length]},tooltip:{valueFormatter:v=>v==null?'—':p.percent?`${(v*100).toFixed(2)}%`:`${+v.toFixed(3)}`},emphasis:{focus:'series'},...(s.bands?{markArea:{silent:true,label:{show:false},data:s.bands}}:{})}))) };
 }
 function history(wrap,spec,onSelect){let host=wrap.querySelector('.qe-regime-history');if(!host){host=document.createElement('div');host.className='qe-regime-history';wrap.append(host);}wrap.classList.add('qe-history-wrap');wrap.tabIndex=-1;host.setAttribute('role','img');host.setAttribute('aria-label','관측·예측 확률 날짜 탐색');mount(host,()=>{const themed={...spec,panels:spec.panels.map(p=>({...p,series:p.series.map(s=>({...s,color:s.code?getComputedStyle(document.documentElement).getPropertyValue('--'+s.code.replace('_','-')).trim():s.color,bands:s.bands?.map(b=>[{...b[0],itemStyle:{color:getComputedStyle(document.documentElement).getPropertyValue('--'+String(b[0].state||'unknown').replace('_','-')).trim(),opacity:.045}},b[1]])}))}))};const option=trend(themed);option.legend.show=false;return option;},spec.panels.length*270+12,onSelect,spec.selection);
  const labels={risk_on:'위험선호',transition:'전환',risk_off:'위험회피'};for(const button of document.querySelectorAll('#history-chart-legend button')){const code=button.dataset.qrSeriesId?.replace('observed','');if(!labels[code])continue;button.setAttribute('aria-pressed',String(host._qe.getOption().legend?.[0]?.selected?.[labels[code]]!==false));button.onclick=()=>{const visible=button.getAttribute('aria-pressed')==='true';const buttons=[...document.querySelectorAll('#history-chart-legend button')];if(visible&&buttons.filter(b=>b.getAttribute('aria-pressed')==='true').length<=1)return;host._qe.dispatchAction({type:visible?'legendUnSelect':'legendSelect',name:labels[code]});button.setAttribute('aria-pressed',String(!visible));};}
  wrap.querySelector('.qe-outcomes')?.remove();
  if(spec.outcomes){const outcomes=document.createElement('div');outcomes.className='qe-outcomes';const label=document.createElement('span');label.textContent=spec.outcomeTitle;outcomes.append(label);const strip=document.createElement('div');strip.className='qe-outcomes-strip';spec.outcomes.forEach((o,i)=>{const item=document.createElement('span');item.style.background=colors[o.state]||'transparent';item.style.borderColor=colors[o.state]||'var(--line)';item.title=spec.dates[i]+' · '+({risk_on:'위험선호',transition:'전환',risk_off:'위험회피'}[o.state]||o.status);strip.append(item);});outcomes.append(strip);wrap.append(outcomes);}}

 function upgrade(){
  if(!root.echarts)return;
  for(const host of mounted)if(!host.isConnected)host._qe?.dispose();
  for(const svg of document.querySelectorAll('svg[data-qr-spec]')){
   if(svg.dataset.qeConverted){const owner=svg.closest('.performance-visual');owner?.querySelectorAll('.qr-measure,.performance-legend').forEach(e=>e.hidden=true);continue;}
   const spec=JSON.parse(decodeURIComponent(svg.dataset.qrSpec));if(!spec.series||spec.kind==='category')continue;
   const host=document.createElement('div');host.className='qe-performance';host.setAttribute('role','img');host.setAttribute('aria-label',svg.getAttribute('aria-label')||'성과 추이');svg.after(host);svg.hidden=true;svg.style.display='none';svg.dataset.qeConverted='true';svg.removeAttribute('data-qr-spec');svg.closest('.performance-visual')?.classList.add('qe-converted-performance');
   const lines=spec.series.map((s,i)=>({label:s.label,color:getComputedStyle(svg.querySelector('[data-qr-series-id=\"'+s.id+'\"]')||svg).stroke||palette[i%palette.length],values:spec.domain.map(d=>s.points.find(p=>p.key===d.key)?.value??null)}));
   mount(host,()=>{const percent=spec.series[0]?.unit==='%';const option=trend({dates:spec.domain.map(d=>d.label),panels:[{title:'',percent,series:lines.map((s,i)=>({...s,color:getComputedStyle(svg.querySelector('[data-qr-series-id=\"'+spec.series[i].id+'\"]')||svg).stroke||s.color}))}]});if(percent)option.yAxis[0].max=0;option.series[0].markLine={silent:true,symbol:'none',label:{show:false},lineStyle:{color:tokens().muted,type:'dashed',width:1},data:[{yAxis:percent?0:1}]};return option;},320);
   const details=document.createElement('details');details.className='qe-performance-range';const summary=document.createElement('summary');summary.textContent='구간 비교';details.append(summary);
   const start=document.createElement('input'),end=document.createElement('input');start.type=end.type='date';start.setAttribute('aria-label','구간 시작');end.setAttribute('aria-label','구간 종료');start.min=end.min=spec.domain[0].label;start.max=end.max=spec.domain.at(-1).label;start.value=start.min;end.value=end.max;const apply=document.createElement('button');apply.type='button';apply.textContent='비교';const output=document.createElement('div');output.setAttribute('aria-live','polite');apply.onclick=()=>{const first=spec.domain.findIndex(d=>d.label>=start.value),last=spec.domain.findLastIndex(d=>d.label<=end.value);output.replaceChildren();if(first<0||last<=first){output.textContent='비교할 두 관측일을 선택하세요.';return;}for(const series of spec.series){const a=series.points.find(p=>p.key===spec.domain[first].key)?.value,b=series.points.find(p=>p.key===spec.domain[last].key)?.value;const row=document.createElement('p');const value=rangeChange(a,b,series.measure);row.textContent=series.label+' · '+(value===null?'관측 없음':value.toLocaleString('ko-KR',{maximumFractionDigits:2})+(series.measure==='pp'?'%p':'%'));output.append(row);}};details.append(start,end,apply,output);host.parentElement.after(details);
   const guide=svg.closest('.card')?.querySelector('.performance-scroll-guide');if(guide)guide.hidden=true;
  }
 }
 function rangeChange(a,b,measure){if(typeof a!=='number'||typeof b!=='number'||!Number.isFinite(a)||!Number.isFinite(b))return null;return measure==='pp'?(b-a)*100:a>0?(b/a-1)*100:null;}
 root.RegimeCharts={history,upgrade,colors,trend,rangeChange};
 if(typeof document!=='undefined')document.addEventListener('DOMContentLoaded',()=>{let pending=false;new MutationObserver(()=>{if(pending)return;pending=true;requestAnimationFrame(()=>{pending=false;upgrade();});}).observe(document.body,{childList:true,subtree:true});upgrade();});
})(globalThis);
