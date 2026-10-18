<script>
(()=>{
const data=JSON.parse(document.getElementById('tg-data').textContent);
const selector=document.getElementById('tg-strategy'),models=document.getElementById('tg-models');
const palette=['#b65a15','#9652b0','#2161b6','#c34269','#237e5a','#746327','#148997','#72549c','#111827'];
const money=v=>(v<0?'−':'')+'$'+Math.abs(v).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
for(const s of [...new Set(data.curves.map(c=>c.strategy))]){const o=document.createElement('option');o.value=s;o.textContent=s;selector.appendChild(o)}
selector.value=data.default_strategy;
function draw(){
 const allowed=n=>n==='Expanded RF'||models.value==='all'||(models.value==='teams'?n.endsWith('__teams'):n.endsWith('__no_teams'));
 const selected=data.curves.filter(c=>c.strategy===selector.value&&allowed(c.model));
 const root=document.getElementById('tg-charts');root.replaceChildren();
 for(const split of ['Train (in-sample)','Train (OOB)','Test']){
  const curves=selected.filter(c=>c.split===split);if(!curves.length)continue;
  const h=document.createElement('h3');h.textContent=split+(split==='Test'?' · 2022–23':' · all 2008–22 training seasons');root.appendChild(h);
  const vals=curves.flatMap(c=>c.points.map(p=>p.net)),low=Math.min(0,...vals),high=Math.max(0,...vals),pad=Math.max(10,(high-low)*.08),lo=low-pad,hi=high+pad;
  const labels=curves[0].points.map(p=>p.label),count=labels.length-1;
  const x=i=>90+i/count*950,y=v=>320-(v-lo)/(hi-lo)*270;
  let svg='<svg style="width:100%;max-width:1100px;display:block" viewBox="0 0 1100 370" role="img" aria-label="'+split+' cumulative net profit">';
  for(let j=0;j<5;j++){const v=lo+j*(hi-lo)/4;svg+=`<line x1="90" x2="1040" y1="${y(v)}" y2="${y(v)}" stroke="#dce2e8"/><text x="2" y="${y(v)}" font-size="12">${money(v)}</text>`}
  svg+=`<line x1="90" x2="1040" y1="${y(0)}" y2="${y(0)}" stroke="#8b95a2"/>`;
  for(const c of curves){const color=palette[data.models.indexOf(c.model)];svg+=`<polyline fill="none" stroke="${color}" stroke-width="2.3" points="${c.points.map((p,i)=>x(i)+','+y(p.net)).join(' ')}"><title>${c.model}: ${money(c.points.at(-1).net)}</title></polyline>`}
  for(let j=0;j<5;j++){const i=Math.round(j*count/4);svg+=`<text x="${x(i)}" y="350" text-anchor="${j===0?'start':j===4?'end':'middle'}" font-size="11">${labels[i]}</text>`}
  const panel=document.createElement('div');panel.innerHTML=svg+'</svg>';root.appendChild(panel);
  const legend=document.createElement('div');legend.style.cssText='display:flex;flex-wrap:wrap;gap:16px;margin-bottom:25px';
  for(const c of curves){const el=document.createElement('span');el.style.color=palette[data.models.indexOf(c.model)];el.textContent=c.model.replace('__',' · ')+': '+money(c.points.at(-1).net);legend.appendChild(el)}root.appendChild(legend);
 }
 const rows=data.summary.filter(r=>r.strategy===selector.value&&allowed(r.model));
 const table=document.createElement('table');table.innerHTML='<thead><tr><th>Model</th><th>Split</th><th>Matches</th><th>Bets</th><th>Staked</th><th>Net profit</th><th>ROI</th><th>Flat $1 net</th></tr></thead>';
 const tbody=document.createElement('tbody');for(const r of rows){const tr=document.createElement('tr');for(const v of [r.model,r.split,r.matches,r.bets,money(r.staked),money(r.net),(100*r.roi).toFixed(2)+'%',money(r.flat_net)]){const td=document.createElement('td');td.textContent=v;tr.appendChild(td)}tbody.appendChild(tr)}table.appendChild(tbody);document.getElementById('tg-table').replaceChildren(table);
}
selector.addEventListener('change',draw);models.addEventListener('change',draw);draw();
})();
</script>
