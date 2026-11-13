"""Compare the chosen 2% expected-return filter with the original Kelly replay."""
import base64
from backtest_match_form import OUT, pd, np
from bankroll_backtest import METHODS
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics import renderPDF
from reportlab.lib.colors import HexColor
import pypdfium2 as pdfium

PROFILES = {'baseline':'Baseline: expected net return > 0%',
            'edge2':'Selected: expected net return > 2%'}


def main():
    tables, curves = [], []
    for profile in PROFILES:
        folder = OUT if profile == 'baseline' else OUT/profile
        data = pd.read_csv(folder/'bankroll_summary.csv')
        data = data[data.variant.isin(['minimum_cv_form_allowed','minimum_cv_without_form'])]
        tables.append(data.assign(profile=profile))
        data = pd.read_csv(folder/'bankroll_curves.csv')
        curves.append(data.assign(profile=profile))
    summary = pd.concat(tables,ignore_index=True)
    curve = pd.concat(curves,ignore_index=True)
    summary.to_csv(OUT/'filtered_bankroll_comparison.csv',index=False)
    # A positive robust Kelly stake already requires nominal EV > .02 * odds,
    # so the nominal 2% gate alone cannot remove any such stake.
    robust = summary[summary.method=='Robust quarter Kelly']
    idx = ['family','variant','split']
    base = robust[robust.profile=='baseline'].set_index(idx)
    edge = robust[robust.profile=='edge2'].set_index(idx)
    for col in ['net','staked','bets','max_drawdown']:
        np.testing.assert_allclose(base[col].sort_index(),edge[col].sort_index(),atol=1e-8)
    drawing = Drawing(1100,810)
    drawing.add(Rect(0,0,1100,810,fillColor=HexColor('#ffffff'),strokeColor=None))
    origin = pd.Timestamp('2022-08-01')
    colors = ['#64748b','#2563eb']
    selected = curve[(curve.family=='RF_classifier') &
                     (curve.variant=='minimum_cv_form_allowed') &
                     (curve.split=='test_2022') & (curve.date!='Start')]
    for method,y in [('Quarter Kelly',465),('Robust quarter Kelly',75)]:
        drawing.add(String(65,y+280,method+' - RF classifier with form, $1,000 starting balance',
                           fontName='Helvetica-Bold',fontSize=17))
        chart=LinePlot()
        chart.x,chart.y,chart.width,chart.height=75,y,965,220
        chart.data=[]
        for i,(profile,label) in enumerate(PROFILES.items()):
            data=selected[(selected.method==method)&(selected.profile==profile)].sort_values('date')
            chart.data.append([(0.,1000.)]+[
                (float((pd.Timestamp(row.date)-origin).days),float(1000+row.net))
                for row in data.itertuples()])
            chart.lines[i].strokeColor=HexColor(colors[i])
            chart.lines[i].strokeWidth=2
            if i==0:
                chart.lines[i].strokeDashArray=[5,3]
            drawing.add(Line(75,y+258-i*15,98,y+258-i*15,strokeColor=HexColor(colors[i]),strokeWidth=2))
            drawing.add(String(105,y+255-i*15,label,fontSize=10))
        values=[v for line in chart.data for _,v in line]
        pad=max(20,(max(values)-min(values))*.08)
        chart.yValueAxis.valueMin=min(values)-pad
        chart.yValueAxis.valueMax=max(values)+pad
        chart.xValueAxis.valueMin,chart.xValueAxis.valueMax=0,310
        chart.xValueAxis.valueSteps=[0,61,122,184,243,304]
        chart.xValueAxis.labelTextFormat=lambda x:(origin+pd.Timedelta(days=x)).strftime('%b %Y')
        chart.yValueAxis.labelTextFormat='$%0.0f'
        chart.yValueAxis.visibleGrid=True
        chart.yValueAxis.gridStrokeColor=HexColor('#e2e8f0')
        drawing.add(chart)
    with pdfium.PdfDocument(renderPDF.drawToString(drawing)) as doc:
        page=doc[0]
        bitmap=page.render(scale=1.5)
        bitmap.to_pil().save(OUT/'filtered_kelly_curves.png')
        bitmap.close()
        page.close()
    html = """<!doctype html><meta charset="utf-8"><title>Kelly with 2% minimum expected return</title>
<style>body{font:15px system-ui;max-width:1400px;margin:30px auto;padding:0 20px;color:#172536}
table{border-collapse:collapse;font-size:13px}th,td{padding:7px;border-bottom:1px solid #ddd}
th{background:#eef2f5}.scroll{overflow:auto}img{max-width:100%}</style>
<h1>Kelly: 2% minimum expected return</h1>
<p><b>Expected gross return = model probability × quoted decimal odds.</b>
The selected filter requires this to exceed 1.02: expected net return above 2%.
All valid quoted decimal odds are eligible; no additional quoted-odds floor or cap is applied.
The original positive-return baseline is shown for comparison.</p>
<p>Select the highest nominal expected-return H/D/A outcome, then apply the return filter.
If that selection fails, skip the fixture. All methods use the same nominal return gate.
Robust Kelly subsequently subtracts two percentage points from the selected probability.
All runs start at $1,000, cap each match at 2% and each day at 10% of opening cash,
and reserve stakes before same-date settlement.</p>
<p>Seven model families; the same frozen minimum-CV choices with and without form.
1,445 fixtures per season. No refitting or threshold search.
Historical Bet365 simulations without fees or rounding; previously inspected seasons.
Each result is a separate alternative, not additive or a guaranteed return.</p>
<p><a href="overview.html">Original comparison</a> ·
<a href="filtered_bankroll_comparison.csv">Download all filtered results</a></p>
<p>The return-only filter leaves robust quarter Kelly unchanged: a positive robust stake
already requires nominal expected net return above 0.02 × odds, which exceeds 2%.</p>"""
    for split,title in [('test_2022','2022–23 test'),('audit_2021','2021–22 audit')]:
        html+='<h2>'+title+'</h2>'
        for family in ['RF_classifier',*sorted(set(summary.family)-{'RF_classifier'})]:
            data=summary[(summary.family==family)&(summary.split==split)]
            no=data[data.variant=='minimum_cv_without_form'].set_index(['profile','method'])
            yes=data[data.variant=='minimum_cv_form_allowed'].set_index(['profile','method'])
            display=yes[['bets','staked','net','ending_bankroll','roi','max_drawdown']].join(
                no[['net']],rsuffix='_without_form').reset_index()
            display['profile']=display.profile.map(PROFILES)
            display['roi']*=100
            display['max_drawdown']*=100
            display.columns=['Filter','Staking','Bets','Total staked ($)','Net with form ($)',
                             'Ending bankroll ($)','ROI (%)','Max drawdown (%)','Net without form ($)']
            text=display.to_html(index=False,float_format=lambda x:f'{x:,.2f}')
            html+=('<h3>RF classifier</h3><div class="scroll">'+text+'</div>' if family=='RF_classifier'
                   else '<details><summary>'+family+'</summary><div class="scroll">'+text+'</div></details>')
    image=base64.b64encode((OUT/'filtered_kelly_curves.png').read_bytes()).decode()
    html+='<h2>RF classifier: 2022–23 bankroll paths</h2><p>Baseline and return-only robust Kelly curves overlap exactly.</p><img alt="Quarter Kelly and robust quarter Kelly with filters" src="data:image/png;base64,'+image+'">'
    (OUT/'filtered_bankroll.html').write_text(html,encoding='utf-8')
    print(summary[(summary.family=='RF_classifier')&(summary.variant=='minimum_cv_form_allowed')][
        ['split','profile','method','bets','net','ending_bankroll','max_drawdown']].to_string(index=False))


if __name__=='__main__':
    main()
