"""Graphs and a browser report for the expected-return sensitivity experiment."""
import base64
import json
from compare_return_cutoffs import DEST, CUTOFFS, pd
from bankroll_backtest import METHODS
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics import renderPDF
from reportlab.lib.colors import HexColor
import pypdfium2 as pdfium

# Strategy colors are shared by every static and interactive chart.
COLORS=['#2563eb','#d97706','#16a34a']


def panel(drawing, title, series, y, date_axis=False):
    drawing.add(String(65,y+285,title,fontName='Helvetica-Bold',fontSize=17))
    chart=LinePlot()
    chart.x,chart.y,chart.width,chart.height=80,y,955,220
    chart.data=[points for _,points in series]
    values=[v for _,points in series for _,v in points]
    lo,hi=min(values+[1000. if date_axis else 0.]),max(values+[1000. if date_axis else 0.])
    pad=max(10.,(hi-lo)*.09)
    chart.yValueAxis.valueMin,chart.yValueAxis.valueMax=lo-pad,hi+pad
    chart.yValueAxis.labelTextFormat='$%0.0f'
    chart.yValueAxis.visibleGrid=True
    chart.yValueAxis.gridStrokeColor=HexColor('#e2e8f0')
    if date_axis:
        origin=pd.Timestamp('2022-08-01')
        chart.xValueAxis.valueMin,chart.xValueAxis.valueMax=0,310
        chart.xValueAxis.valueSteps=[0,61,122,184,243,304]
        chart.xValueAxis.labelTextFormat=lambda x:(origin+pd.Timedelta(days=x)).strftime('%b %Y')
    else:
        chart.xValueAxis.valueMin,chart.xValueAxis.valueMax=0,10
        chart.xValueAxis.valueSteps=[x*100 for x in CUTOFFS]
        chart.xValueAxis.labelTextFormat='%0.0f%%'
        drawing.add(String(425,y-42,'Minimum expected net return cutoff',fontSize=12))
    for i,(label,_) in enumerate(series):
        chart.lines[i].strokeColor=HexColor(COLORS[i])
        chart.lines[i].strokeWidth=2
        x=80+(i%3)*320
        ly=y+260-(i//3)*18
        drawing.add(Line(x,ly,x+20,ly,strokeColor=HexColor(COLORS[i]),strokeWidth=2))
        drawing.add(String(x+26,ly-3,label,fontSize=11))
    drawing.add(chart)


def canvas():
    d=Drawing(1100,830)
    d.add(Rect(0,0,1100,830,fillColor=HexColor('#ffffff'),strokeColor=None))
    return d


def save(d,name):
    with pdfium.PdfDocument(renderPDF.drawToString(d)) as doc:
        page=doc[0]
        bitmap=page.render(scale=1.5)
        bitmap.to_pil().save(DEST/name)
        bitmap.close()
        page.close()


def main():
    summary=pd.read_csv(DEST/'summary.csv')
    curves=pd.read_csv(DEST/'curves.csv')
    forward=DEST.parents[1]/'season_2024/return_cutoffs'
    has_forward=(forward/'verification.json').exists()
    full_reference=None
    if has_forward:
        extra=pd.read_csv(forward/'summary.csv')
        full_reference=extra[(extra.family=='RF_classifier_odds_full')&(extra.cutoff==.02)]
        summary=pd.concat([summary,extra[extra.family!='RF_classifier_odds_full']],ignore_index=True)
        extra_curves=pd.read_csv(forward/'curves.csv')
        curves=pd.concat([curves,extra_curves[extra_curves.family!='RF_classifier_odds_full']],ignore_index=True)
    latest='forward_2024' if has_forward else 'test_2022'
    latest_label='2024-25 forward test' if has_forward else '2022-23 test'
    rf=summary[(summary.family=='RF_classifier')&(summary.variant=='minimum_cv_form_allowed')]
    seasons=[('test_2022','2022-23 test: net profit by return cutoff',480),
             ('audit_2021','2021-22 audit: net profit by return cutoff',75)]
    if has_forward:
        seasons.insert(0,('forward_2024','2024-25 forward test (768 eligible matches): net profit by cutoff',885))
        d=Drawing(1100,1235)
        d.add(Rect(0,0,1100,1235,fillColor=HexColor('#ffffff'),strokeColor=None))
    else:d=canvas()
    for split,title,y in seasons:
        series=[]
        for method in METHODS:
            a=rf[(rf.split==split)&(rf.method==method)].sort_values('cutoff')
            series.append((method,list(zip(a.cutoff*100,a.net))))
        panel(d,title+' - RF classifier with form',series,y)
    save(d,'cutoff_profit.png')
    d=Drawing(1200,1110)
    d.add(Rect(0,0,1200,1110,fillColor=HexColor('#ffffff'),strokeColor=None))
    d.add(String(60,1070,'RF classifier with form: strategies at each return cutoff',
                 fontName='Helvetica-Bold',fontSize=22))
    count=int(rf[rf.split==latest].matches.iloc[0])
    d.add(String(60,1044,f'{latest_label} | {count:,} matches | Each scenario starts at $1,000 | Same axes',
                 fontSize=13))
    for i,method in enumerate(METHODS):
        x=60+i*370
        d.add(Line(x,1018,x+28,1018,strokeColor=HexColor(COLORS[i]),strokeWidth=2))
        d.add(String(x+36,1014,method,fontSize=13))
    rf_curves=curves[(curves.family=='RF_classifier')&
                     (curves.variant=='minimum_cv_form_allowed')&(curves.split==latest)]
    values=[1000.,*(1000+rf_curves.net).tolist()]
    pad=max(10.,(max(values)-min(values))*.09)
    origin=pd.Timestamp('2024-08-01' if has_forward else '2022-08-01')
    for i,cutoff in enumerate(CUTOFFS):
        left=85+(i%2)*570
        y=730-(i//2)*315
        d.add(String(left,y+225,f'Expected net return > {cutoff*100:g}%',
                     fontName='Helvetica-Bold',fontSize=16))
        chart=LinePlot()
        chart.x,chart.y,chart.width,chart.height=left,y,460,200
        chart.data=[]
        for j,method in enumerate(METHODS):
            a=rf_curves[(rf_curves.method==method)&(rf_curves.cutoff==cutoff)&
                        (rf_curves.date!='Start')].sort_values('date')
            points=[(0.,1000.)]+[(float((pd.Timestamp(r.date)-origin).days),
                                 float(1000+r.net)) for r in a.itertuples()]
            chart.data.append(points)
            chart.lines[j].strokeColor=HexColor(COLORS[j])
            chart.lines[j].strokeWidth=2
        chart.yValueAxis.valueMin,chart.yValueAxis.valueMax=min(values)-pad,max(values)+pad
        chart.yValueAxis.labelTextFormat='$%0.0f'
        chart.yValueAxis.visibleGrid=True
        chart.yValueAxis.gridStrokeColor=HexColor('#e2e8f0')
        chart.xValueAxis.valueMin,chart.xValueAxis.valueMax=0,310
        chart.xValueAxis.valueSteps=[0,122,243,304]
        chart.xValueAxis.labelTextFormat=lambda x:(origin+pd.Timedelta(days=x)).strftime('%b %Y')
        d.add(chart)
    save(d,'cutoff_bankrolls.png')
    html="""<!doctype html><meta charset="utf-8"><title>Expected-return cutoff comparison</title>
<style>body{font:15px system-ui;max-width:1400px;margin:30px auto;padding:0 22px;color:#172536}
table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd}
th{background:#eef2f5}img{max-width:100%}.scroll{overflow:auto}label{margin-right:20px}
select{padding:8px}svg{width:100%;max-width:1100px}
#charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:24px}
.chart-card{min-width:0}</style>
<h1>Minimum expected-return cutoffs</h1>
<p>Compare 0%, 1%, 2%, 3%, 5% and 10%, with <b>no quoted-odds restriction</b>.
Choose the highest nominal expected-return outcome, then require probability × odds − 1
to exceed the cutoff. Each simulation starts independently at $1,000, with 2% per-match
and 10% daily exposure caps. Robust quarter Kelly subtracts two percentage points from
the probability after the nominal cutoff is applied.</p>
<p>2024–25 uses the saved 2022–23-test models, trained through 22 May 2022.
Earlier seasons retain their original frozen models. This is a sensitivity
comparison, not a cutoff selected on unseen data. Your existing 2% setting is unchanged.
Historical Bet365 simulations, without fees or rounding. The two earlier seasons have previously
been inspected; the highest realized profit does not establish a future optimum.</p>
<p><a href="../filtered_bankroll.html">Current 2% comparison</a> ·
<a href="summary.csv">2021–22 and 2022–23 results CSV</a> · <a href="verification.json">Earlier verification</a></p>
<h2>Explore the model and season</h2>
<label>Model <select id="family"></select></label>
<label>Features <select id="variant"><option value="minimum_cv_form_allowed">Form allowed</option>
<option value="minimum_cv_without_form">Without new form</option></select></label>
<label>Season <select id="split"><option value="test_2022">2022–23 test</option>
<option value="audit_2021">2021–22 audit</option>FORWARD_OPTION</select></label>
<p id="coverage"></p>
<div id="table" class="scroll"></div>
<p>ROI = net profit / total stakes; drawdown is the largest decline from a previous
daily closing bankroll peak. Daily volatility is the sample standard deviation of net profit / opening
bankroll across all eligible match dates, including dates with no bet; it is not annualized.
Curves use weekly closing snapshots.</p>
<p><b>Color always identifies the strategy:</b> blue = flat $1, orange = quarter Kelly,
green = robust quarter Kelly. Each panel shows one return cutoff. All panels use the
same bankroll scale for the selected model and season.</p>
<div id="charts"></div>
<h2>RF classifier with form: compare seasons</h2>"""
    html=html.replace('FORWARD_OPTION','<option value="forward_2024" selected>2024–25 forward test</option>' if has_forward else '')
    if has_forward:
        notes='''<aside style="background:#fff5dc;padding:18px;margin:20px 0">
<b>2024–25 coverage: 768 of 1,446 fixtures (53.1%).</b> England 166/380, Germany 210/306,
Spain 134/380, Italy 258/380. Both feature variants use this same subset. All 1,446 games feed
past-match form, including games excluded from betting. Each side needs at least 10 dated starter ratings.
Only cached FC23/FC24 snapshots dated before the season are used; these ratings are old.
Undated FC25 ratings are excluded. Actual starting lineups are used, so this assumes lineups were
available before placing bets. Cached Bet365 prices are not timestamp-aligned to lineup publication.
This is a historical simulation, not an executable price guarantee. The eligible subset can differ
systematically from the excluded games. Feature selection, model parameters and the six cutoffs stay
fixed; nothing is refitted or selected on 2024–25.</aside>
<p><a href="../../season_2024/return_cutoffs/summary.csv">2024–25 results CSV</a> ·
<a href="../../season_2024/return_cutoffs/verification.json">2024–25 settlement checks</a> ·
<a href="../../season_2024/input_verification.json">Input checks</a> ·
<a href="../../season_2024/return_cutoffs/plan.json">Frozen-model audit</a></p>'''
        html=html.replace('<h2>Explore the model and season</h2>',notes+'<h2>Explore the model and season</h2>')
    for name in ['cutoff_profit.png','cutoff_bankrolls.png']:
        image=base64.b64encode((DEST/name).read_bytes()).decode()
        html+='<img alt="'+name.replace('_',' ')+'" src="data:image/png;base64,'+image+'">'
    if has_forward:
        html+='<h2>Separate full-season reference: odds-only RF classifier</h2><p>This different saved model uses only the three bookmaker odds and covers all 1,446 fixtures. It is not the form model, and its full-season result is not a matched-sample estimate of the value of form. Results below use the unchanged 2% cutoff.</p>'
        table=full_reference[['method','bets','staked','net','roi','max_drawdown','daily_return_sd_pct']].copy()
        table['roi']*=100;table['max_drawdown']*=100
        table.columns=['Strategy','Bets','Staked ($)','Net ($)','ROI (%)','Max drawdown (%)','Daily volatility (%)']
        html+=table.to_html(index=False,float_format=lambda v:f'{v:,.2f}',border=0)
    # Older experiments did not store volatility. JSON null keeps their display explicit.
    data=dict(summary=json.loads(summary.to_json(orient='records',double_precision=15)),curves=curves.to_dict('records'))
    html+='<script id="data" type="application/json">'+json.dumps(data,allow_nan=False).replace('<','\\u003c')+'</script>'
    html+=r"""<script>
const data=JSON.parse(document.getElementById('data').textContent);
const $=id=>document.getElementById(id),methods=['Flat $1','Quarter Kelly','Robust quarter Kelly'];
const cutoffs=[0,.01,.02,.03,.05,.1],colors=['#2563eb','#d97706','#16a34a'];
for(const f of [...new Set(data.summary.map(r=>r.family))]) $('family').add(new Option(f,f));
$('family').value='RF_classifier';
const money=n=>n.toLocaleString('en-US',{style:'currency',currency:'USD'});
function selected(r){return r.family===$('family').value&&r.variant===$('variant').value&&r.split===$('split').value}
function render(){
 const rows=data.summary.filter(selected).sort((a,b)=>methods.indexOf(a.method)-methods.indexOf(b.method)||a.cutoff-b.cutoff);
 const t=document.createElement('table'),head=t.createTHead().insertRow();
 $('coverage').textContent=$('split').value==='forward_2024'?'2024–25: 768 eligible fixtures out of 1,446. Old dated ratings; see coverage limitations above.':'1,445 fixtures in this earlier season.';
 for(const v of ['Staking','Cutoff','Bets','Total staked','Net profit','Ending balance','ROI','Max drawdown','Daily volatility']){
  const th=document.createElement('th');th.textContent=v;head.append(th)}
 for(const r of rows){const tr=t.insertRow();for(const v of [r.method,(r.cutoff*100)+'%',r.bets,money(r.staked),money(r.net),money(r.ending_bankroll),(r.roi*100).toFixed(2)+'%',(r.max_drawdown*100).toFixed(2)+'%',r.daily_return_sd_pct==null?'—':r.daily_return_sd_pct.toFixed(3)+'%'])tr.insertCell().textContent=v}
 $('table').replaceChildren(t);$('charts').replaceChildren();
 const allPoints=data.curves.filter(r=>selected(r)&&r.date!=='Start');
 const dates=allPoints.map(r=>Date.parse(r.date)),start=Math.min(...dates),end=Math.max(...dates);
 const values=[1000,...allPoints.map(r=>1000+r.net)],low=Math.min(...values),high=Math.max(...values),pad=Math.max(10,(high-low)*.1);
 const lo=low-pad,hi=high+pad,x=d=>85+(d-start)/(end-start)*590,y=v=>290-(v-lo)/(hi-lo)*220;
 for(const cutoff of cutoffs){
  const points=allPoints.filter(r=>r.cutoff===cutoff);
  const box=document.createElement('div');box.className='chart-card';
  const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 700 350');svg.setAttribute('role','img');svg.setAttribute('aria-label','Strategy bankrolls at return cutoff '+cutoff*100+'%');
  const add=(tag,attrs,text)=>{const n=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);if(text)n.textContent=text;svg.append(n);return n};
  const heading=document.createElement('h3');heading.textContent='Expected net return > '+cutoff*100+'%';box.append(heading);
  methods.forEach((method,i)=>add('text',{x:85+i*195,y:30,fill:colors[i],'font-size':14},method));
  for(let i=0;i<5;i++){let v=lo+(hi-lo)*i/4;add('line',{x1:85,x2:675,y1:y(v),y2:y(v),stroke:'#ddd'});add('text',{x:0,y:y(v)+4,'font-size':14},money(v))}
  methods.forEach((method,i)=>{let a=points.filter(r=>r.method===method).sort((a,b)=>a.date.localeCompare(b.date));const path=[[85,y(1000)],...a.map(r=>[x(Date.parse(r.date)),y(1000+r.net)])];add('polyline',{points:path.map(p=>p.join(',')).join(' '),fill:'none',stroke:colors[i],'stroke-width':2.5})});
  for(let i=0;i<4;i++){const date=start+(end-start)*i/3;add('text',{x:x(date),y:325,'text-anchor':'middle','font-size':14},new Date(date).toISOString().slice(0,7))}
  box.append(svg);$('charts').append(box);
 }
}
for(const id of ['family','variant','split'])$(id).onchange=render;render();
</script>"""
    (DEST/'report.html').write_text(html,encoding='utf-8')
    print(rf[['split','method','cutoff','bets','net','max_drawdown']].round(4).to_string(index=False))


if __name__=='__main__':
    main()
