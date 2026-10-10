const stakeAmounts=[1,2,3,4,5];
let stakeHistory=null;
let stakeCurves=null;
const money=n=>n.toLocaleString('en-US',{style:'currency',currency:'USD'});
const stakeTotals=document.createElement('div');stakeTotals.className='metrics';
const methodDetails=document.createElement('details'),methodSummary=document.createElement('summary');
methodSummary.textContent='Backtest method and limitations';
$('stake-method').before(stakeTotals,methodDetails);methodDetails.append(methodSummary,$('stake-method'));
const historyChart=document.createElement('div');historyChart.className='scroll';
methodDetails.after(historyChart);
const searchLink=document.createElement('p'),searchAnchor=document.createElement('a');
searchAnchor.href='/strategy-search';searchAnchor.target='_blank';searchAnchor.rel='noopener';
searchAnchor.textContent='Explore strategy search: edge, odds, outcomes, calibration and stakes';searchLink.append(searchAnchor);methodDetails.after(searchLink);
const cumulativeChart=document.createElement('div');historyChart.before(cumulativeChart);
function drawCumulative(){
 cumulativeChart.replaceChildren();if(!stakeCurves)return;
 const selected=$('strategy').value;
 const heading=document.createElement('h3');heading.textContent='Cumulative profit';cumulativeChart.append(heading);
 if(selected==='all'){const note=document.createElement('p');note.textContent='Select one strategy above to see its cumulative training and test profit curves.';cumulativeChart.append(note);return;}
 const note=document.createElement('p');note.className='muted';note.textContent=stakeCurves.ordering+' Each split starts at $0; panels use separate vertical scales.';cumulativeChart.append(note);
 for(const split of ['Train (OOB)','Test']){
  const curve=stakeCurves.curves.find(c=>c.strategy===selected&&c.split===split);if(!curve)continue;
  const values=[0];for(const p of curve.points)values.push(values.at(-1)+p.levels.reduce((sum,v,i)=>sum+stakeAmounts[i]*v[1],0));
  const title=document.createElement('p');title.textContent=split+' · Final net profit '+money(values.at(-1));cumulativeChart.append(title);
  const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 780 270');svg.style.width='100%';
  svg.setAttribute('role','img');svg.setAttribute('aria-label',split+' cumulative profit, ending at '+money(values.at(-1)));
  const node=(tag,attrs,content)=>{const e=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(content)e.textContent=content;svg.append(e)};
  let low=Math.min(...values),high=Math.max(...values);const pad=Math.max(5,(high-low)*.12);low-=pad;high+=pad;
  const x=i=>80+650*i/curve.points.length,y=v=>215-180*(v-low)/(high-low),color=split==='Test'?'#77e2bc':'#80acff';
  for(let i=0;i<5;i++){const v=low+(high-low)*i/4;node('line',{x1:80,x2:730,y1:y(v),y2:y(v),stroke:'#29394d'});node('text',{x:72,y:y(v)+4,'text-anchor':'end',fill:'#aebed0','font-size':13},'$'+v.toFixed(0))}
  node('line',{x1:80,x2:730,y1:y(0),y2:y(0),stroke:'#aebed0'});
  node('polyline',{points:values.map((v,i)=>x(i)+','+y(v)).join(' '),stroke:color,'stroke-width':2.5,fill:'none'});
  for(let i=0;i<=4;i++){const position=Math.round(i*curve.points.length/4),p=curve.points[Math.max(0,position-1)];node('text',{x:x(position),y:241,'text-anchor':'middle',fill:'#aebed0','font-size':12},position===0?'Start':p.season+' W'+p.week)}
  cumulativeChart.append(svg);
 }
}
fetch('/api/stake-curves').then(r=>{if(!r.ok)throw Error('Cumulative backtest data unavailable.');return r.json()}).then(r=>{stakeCurves=r;drawCumulative()}).catch(e=>{cumulativeChart.textContent=e.message});
function drawHistoryChart(items,all){
 historyChart.replaceChildren();
 const title=document.createElement('h3');title.textContent=all?'Train vs test ROI by strategy':'Net profit by season';historyChart.append(title);
 if(!items.length)return;
 const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');
 const width=all?1000:800,left=all?400:220,height=items.length*34+65;
 svg.setAttribute('viewBox',`0 0 ${width} ${height}`);svg.style.width='100%';svg.style.minWidth=all?'900px':'650px';
 svg.setAttribute('role','img');svg.setAttribute('aria-label',title.textContent+'. Blue: training OOB; green: held-out test. Exact figures in the tables below.');
 const node=(tag,attrs,content)=>{const e=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(content)e.textContent=content;svg.append(e)};
 const values=items.map(t=>all?(t.stake?100*(t.gross/t.stake-1):0):t.gross-t.stake);
 let low=Math.min(0,...values),high=Math.max(0,...values);const pad=Math.max(1,(high-low)*.20);low-=pad;high+=pad;
 const x=v=>left+(v-low)/(high-low)*(width-left-50);
 node('text',{x:8,y:18,fill:'#aebed0','font-size':12},'Blue: Train (OOB) · Green: Test · '+(all?'ROI (%)':'Net profit ($)'));
 node('line',{x1:x(0),x2:x(0),y1:30,y2:height-10,stroke:'#aebed0'});
 items.forEach((t,i)=>{const y=50+i*34,value=values[i],color=t.split==='Test'?'#77e2bc':'#80acff';
  node('text',{x:left-12,y:y+4,'text-anchor':'end',fill:'#e5edf6','font-size':12},all?t.strategy+' · '+t.split:t.season+' · '+t.split);
  node('rect',{x:Math.min(x(0),x(value)),y:y-9,width:Math.max(1,Math.abs(x(value)-x(0))),height:18,fill:color});
  node('text',{x:x(value)+(value>=0?6:-6),y:y+4,'text-anchor':value>=0?'start':'end',fill:'#e5edf6','font-size':12},all?value.toFixed(2)+'%':money(value));
 });historyChart.append(svg);
}
for(let i=0;i<5;i++){
 const wrap=document.createElement('div'),label=document.createElement('label'),input=document.createElement('input');
 label.htmlFor='stake-'+i;label.textContent='Level '+(i+1)+' ($)';input.id=label.htmlFor;
 input.type='number';input.min='0.01';input.step='0.01';input.value=stakeAmounts[i];
 input.oninput=input.onchange=()=>{
  const values=Array.from(document.querySelectorAll('#stake-inputs input'),x=>Number(x.value));
  if(values.some((x,j)=>!Number.isFinite(x)||x<=0||x>100000||Math.abs(x*100-Math.round(x*100))>1e-6||(j&&x<values[j-1]))){
   $('stake-message').textContent='Enter positive amounts in cents, ascending from level 1 to 5 (maximum $100,000). Previous valid amounts remain applied.';return;
  }
  stakeAmounts.splice(0,5,...values);$('stake-message').textContent='Custom amounts apply to this view. CSV exports retain $1–$5. Synthetic split amounts are displayed rounded to cents.';
  renderBets();renderStakeHistory();
 };
 wrap.append(label,input);$('stake-inputs').append(wrap);
}
function renderStakeHistory(){
 if(!stakeHistory)return;
 drawCumulative();
 const selected=$('strategy').value,all=selected==='all';
 const rows=stakeHistory.rows.filter(s=>all||s.strategy===selected);
 const seasons=new Map();
 for(const s of rows){
  const key=JSON.stringify([s.strategy,s.split,s.season]);
  if(!seasons.has(key))seasons.set(key,{strategy:s.strategy,split:s.split,season:s.season,bets:0,matches:0,unavailable:0,stake:0,gross:0,flat:0});
  const t=seasons.get(key),amount=s.level?stakeAmounts[s.level-1]:0;
  t.bets+=s.bets;t.matches+=s.matches;t.unavailable+=s.unavailable;t.stake+=s.bets*amount;t.gross+=s.unit_returns*amount;
  t.flat+=s.original_gross_return-s.original_bets;
 }
 $('stake-seasons').replaceChildren();$('stake-levels').replaceChildren();
 drawHistoryChart(Array.from(seasons.values()).filter(t=>all?t.season==='All seasons':t.season!=='All seasons').sort((a,b)=>all?a.strategy.localeCompare(b.strategy)||b.split.localeCompare(a.split):a.season.localeCompare(b.season)),all);
 stakeTotals.replaceChildren();
 if(!all)for(const t of seasons.values())if(t.season==='All seasons'){
  const box=document.createElement('div'),title=document.createElement('b'),net=document.createElement('div'),description=document.createElement('p');
  box.className='metric';title.textContent=t.split;net.textContent=money(t.gross-t.stake)+' net profit';
  net.style.fontSize='22px';net.style.color=t.gross>=t.stake?'#77e2bc':'#ffd391';
  description.className='muted';description.textContent=`${t.bets} bets · ${money(t.stake)} staked · ${t.stake?((t.gross/t.stake-1)*100).toFixed(2):'0.00'}% ROI`;
  box.append(title,net,description);stakeTotals.append(box);
 }
 const numbers=(tr,stake,gross)=>{cell(tr,money(stake));cell(tr,money(gross));cell(tr,money(gross-stake),(gross-stake)>=0?'prob':'');cell(tr,stake?((gross/stake-1)*100).toFixed(2)+'%':'—')};
 for(const t of seasons.values()){
  const tr=document.createElement('tr');cell(tr,t.strategy);cell(tr,t.split);cell(tr,t.season);cell(tr,`${t.bets} / ${t.matches-t.bets-t.unavailable} / ${t.unavailable}`);
  numbers(tr,t.stake,t.gross);cell(tr,money(t.flat));$('stake-seasons').append(tr);
 }
 for(const s of rows.filter(s=>s.season==='All seasons'&&s.level>0)){
  const tr=document.createElement('tr'),amount=stakeAmounts[s.level-1];cell(tr,s.strategy);cell(tr,s.split);cell(tr,`${s.level} · ${money(amount)}`);cell(tr,s.bets);
  numbers(tr,s.bets*amount,s.unit_returns*amount);$('stake-levels').append(tr);
 }
}
fetch('/api/stake-results').then(async r=>{if(!r.ok)throw Error('Historical results unavailable; run python scripts/backtest_stake_levels.py.');return r.json()})
 .then(data=>{stakeHistory=data;$('stake-method').textContent=data.methodology;renderStakeHistory()})
 .catch(e=>{$('stake-method').textContent=e.message});
