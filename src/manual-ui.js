const manualToken=document.currentScript.dataset.token;
let manualReport=null,manualLatest=null,manualDownload=null,manualRevision=0;
const manualFields={},manualQuoteRows=new Map();
function manualElement(tag,text){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;return e}
function manualControl(key,label,options,value){
 const box=manualElement('div'),lab=manualElement('label',label);lab.htmlFor='manual-'+key;
 const input=manualElement(options?'select':'input');input.id=lab.htmlFor;
 if(options)for(const [v,t] of options){const o=manualElement('option',t);o.value=v;input.append(o)}
 else{input.type='number';input.step='any';input.style.width='100px'}
 input.value=value;box.append(lab,input);$('manual-controls').append(box);manualFields[key]=input;
 input.oninput=()=>{manualRevision++;manualLatest=null;$('manual-output').replaceChildren();$('manual-status').textContent='Filters changed. Apply to recalculate.'};return input;
}
manualControl('strategy','Base strategy',[], '');
manualControl('edge','Minimum expected return (%)',null,2);
manualControl('min','Minimum decimal odds',null,4);
manualControl('max','Maximum decimal odds',null,1000);
manualControl('outcome','Outcome filter',[['all','All (including combined)'],['home','Home only'],['draw','Draw only'],['away','Away only']],'all');
manualControl('sizing','Staking',[['levels','Five levels'],['flat','Flat level-1 amount']],'levels');
manualControl('width','Kelly band width (%)',null,1);
for(let i=1;i<=5;i++)manualControl('amount'+i,'Level '+i+' amount',null,i);
const manualApply=manualElement('button','Apply filters & recalculate');$('manual-controls').append(manualApply);
manualApply.onclick=manualEvaluate;
manualFields.source={value:'bet365'};
async function manualPost(path,body){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json','X-App-Token':manualToken},body:JSON.stringify(body)});const data=await r.json();if(!r.ok)throw Error(data.error||'Request failed');return data}
function manualKey(m){return [m.date,m.home,m.away].join('|')}
function manualSetReport(r){
 manualRevision++;
 manualReport=r;const selected=manualFields.strategy.value||'Legacy variance score';manualFields.strategy.replaceChildren();
 for(const name of r.strategy_names.filter(n=>!n.includes('filtered search candidate'))){const o=manualElement('option',name);o.value=name;manualFields.strategy.append(o)}
 manualFields.strategy.value=selected;manualEvaluate();
}
async function manualEvaluate(){
 const revision=manualRevision;
 if(!manualReport)return;manualApply.disabled=true;$('manual-status').textContent='Recalculating live choices and historical filters…';$('manual-output').replaceChildren();
 const f=manualFields,quotes={};
 for(const [key,state] of manualQuoteRows){quotes[key]=f.source.value==='winner'?{home:Number(state.home.value),draw:Number(state.draw.value),away:Number(state.away.value),confirmed:state.confirm.checked,observed_at:state.observed_at}: {snapshot:state.snapshot}}
 try{
  const result=await manualPost('/api/manual',{strategy:f.strategy.value,source:f.source.value,min_edge:Number(f.edge.value)/100,min_odds:Number(f.min.value),max_odds:Number(f.max.value),outcome:f.outcome.value,width:Number(f.width.value)/100,sizing:f.sizing.value,amounts:[1,2,3,4,5].map(i=>Number(f['amount'+i].value)),quotes});
  if(revision!==manualRevision)throw Error('Filters or predictions changed. Apply again to recalculate.');
  manualLatest=result;manualRender(result);$('manual-status').textContent='Applied '+result.settings.strategy+' · '+result.settings.source+' · edge > '+(100*result.settings.min_edge)+'% · odds '+result.settings.min_odds+'–'+result.settings.max_odds+'.';
 }catch(e){$('manual-status').textContent=e.message}finally{manualApply.disabled=false}
}
function manualTable(parent,headers){const scroll=manualElement('div');scroll.className='scroll';const table=manualElement('table'),head=manualElement('thead'),tr=manualElement('tr'),body=manualElement('tbody');for(const h of headers)tr.append(manualElement('th',h));head.append(tr);table.append(head,body);scroll.append(table);parent.append(scroll);return body}
function manualCurve(parent,curve){
 parent.append(manualElement('h3',curve.split+' · Cumulative profit'));
 const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 800 260');svg.style.width='100%';svg.setAttribute('role','img');svg.setAttribute('aria-label',curve.split+' manual-filter cumulative profit');
 const node=(tag,a,t)=>{const n=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(a))n.setAttribute(k,v);if(t)n.textContent=t;svg.append(n)};
 const points=curve.points,values=points.map(p=>p.net);let lo=Math.min(...values),hi=Math.max(...values);const pad=Math.max(5,(hi-lo)*.12);lo-=pad;hi+=pad;
 const x=i=>85+i/Math.max(1,points.length-1)*650,y=v=>210-(v-lo)/(hi-lo)*180;
 for(let i=0;i<5;i++){const v=lo+(hi-lo)*i/4;node('line',{x1:85,x2:735,y1:y(v),y2:y(v),stroke:'#40546b'});node('text',{x:75,y:y(v)+4,'text-anchor':'end',fill:'#aebed0','font-size':13},'$'+v.toFixed(0))}
 node('line',{x1:85,x2:735,y1:y(0),y2:y(0),stroke:'#aebed0'});node('polyline',{points:points.map((p,i)=>x(i)+','+y(p.net)).join(' '),fill:'none',stroke:curve.split==='Test'?'#77e2bc':'#80acff','stroke-width':2.5});
 for(let i=0;i<=4;i++){const j=Math.round((points.length-1)*i/4);node('text',{x:x(j),y:242,fill:'#aebed0','font-size':12,'text-anchor':'middle'},points[j].label)}parent.append(svg);
}
function manualRender(r){
 const parent=$('manual-output');parent.replaceChildren();parent.append(manualElement('p',r.note));
 const body=manualTable(parent,['Match','Decision / selection','Odds','Expected return','Level / amount','Split','Source / observed','Reason']);
 for(const p of r.predictions){const b=p.bet,tr=manualElement('tr'),currency=p.quote.currency==='ILS'?'₪':'$';
  for(const v of [p.date+' · '+p.home+' vs '+p.away,b.decision+' · '+b.selection,b.decimal_odds?.toFixed(3)??'—',b.expected_net_per_unit===null?'—':(100*b.expected_net_per_unit).toFixed(2)+'%',b.level+' / '+currency+b.amount.toFixed(2),['home','draw','away'].filter(s=>b[s+'_amount']>0).map(s=>s+': '+currency+b[s+'_amount'].toFixed(2)).join(' + '),p.quote.source+' · '+(p.quote.fetched_at??'unavailable'),b.reason])cell(tr,v);
  body.append(tr);
 }
 const download=manualElement('button','Download these filtered selections');download.onclick=()=>{
  const headers=['date','home','away','strategy','source','currency','decision','selection','odds','edge','level','amount','home_amount','draw_amount','away_amount','observed_at','min_edge','min_odds','max_odds','outcome_filter','kelly_width','sizing','reason'];
  const rows=r.predictions.map(p=>[p.date,p.home,p.away,r.settings.strategy,r.settings.source,p.quote.currency,p.bet.decision,p.bet.selection,p.bet.decimal_odds,p.bet.expected_net_per_unit,p.bet.level,p.bet.amount,p.bet.home_amount,p.bet.draw_amount,p.bet.away_amount,p.quote.fetched_at,r.settings.min_edge,r.settings.min_odds,r.settings.max_odds,r.settings.outcome,r.settings.width,r.settings.sizing,p.bet.reason]);
  const esc=v=>'"'+String(typeof v==='string'&&/^[=+@-]/.test(v)?"'"+v:v??'').replaceAll('"','""')+'"';
  if(manualDownload)URL.revokeObjectURL(manualDownload);manualDownload=URL.createObjectURL(new Blob([[headers,...rows].map(row=>row.map(esc).join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'}));const a=manualElement('a');a.href=manualDownload;a.download='manual-strategy-selections.csv';a.click();
 };parent.append(download,manualElement('h3','Historical results for these filters'),manualElement('p',r.history.note));
 if(r.history.rows.length){const tb=manualTable(parent,['Split','Season','Bets','Staked','Gross returns','Net profit','ROI']);for(const s of r.history.rows){const tr=manualElement('tr');for(const v of [s.split,s.season,s.bets,money(s.staked),money(s.gross),money(s.net),s.roi===null?'—':(100*s.roi).toFixed(2)+'%'])cell(tr,v);tb.append(tr)}for(const curve of r.history.curves)manualCurve(parent,curve)}
}
if(currentReport)manualSetReport(currentReport);
