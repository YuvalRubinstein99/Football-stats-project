"""Plot daily cumulative home-only profit for the selected binary forest."""
import math
from select_expanded_features import ROOT,pd,np
from reportlab.graphics.shapes import Drawing,Rect,Line,String,PolyLine,Circle
from reportlab.graphics import renderSVG,renderPDF
from reportlab.lib.colors import HexColor
import pypdfium2 as pdfium

OUT=ROOT/'results/lineup_improvements/binary_retrained/betting_no_draw'
RULES=[('Home only: positive expected return','Expected return > 0%','#087f8c'),
       ('Home only: expected return >2%','Expected return > 2%','#8252bd')]


def text(d,x,y,value,size=11,color='#263548',bold=False,**kwargs):
    d.add(String(x,y,value,fontName='Helvetica-Bold' if bold else 'Helvetica',fontSize=size,
                 fillColor=HexColor(color),**kwargs))


def main():
    ledger=pd.read_csv(OUT/'selected_ledger.csv.gz')
    frame=ledger[(ledger.family=='RF_binary')&(ledger.split=='Test 2022-23')
                 &ledger.strategy.isin([r[0] for r in RULES])].copy()
    frame['fixture_date']=pd.to_datetime(frame.fixture_date)
    summary=pd.read_csv(OUT/'selected_results.csv')
    summary=summary[(summary.family=='RF_binary')&(summary.split=='Test 2022-23')]
    first,last=frame.fixture_date.min(),frame.fixture_date.max()
    start=first-pd.Timedelta(days=1)
    dates=pd.DatetimeIndex([start,*sorted(frame.fixture_date.unique())])
    width,height=1200,850
    d=Drawing(width,height)
    d.add(Rect(0,0,width,height,fillColor=HexColor('#ffffff'),strokeColor=None))
    text(d,52,808,'Home-only bets: cumulative profit',26,bold=True)
    text(d,52,779,'Selected binary random forest  |  2022-23 test season  |  1,065 fixtures',13)
    text(d,52,756,'Bet when P(home win) x home odds - 1 clears the threshold; otherwise skip.',12,color='#627185')
    exports=[]
    left,right=95,1040
    x=lambda day:left+(pd.Timestamp(day)-start).days/(last-start).days*(right-left)
    for staking,top,bottom,step,title in [
        ('flat',688,450,10,'Flat stakes: 1 unit per bet'),
        ('tiers',337,99,50,'Variable stakes: existing 1-5-unit tiers')]:
        curves=[]
        for strategy,label,color in RULES:
            rows=frame[(frame.strategy==strategy)&(frame.staking==staking)]
            assert len(rows)==1065 and rows.source_row.nunique()==1065
            daily=rows.groupby('fixture_date').net.sum().reindex(dates,fill_value=0.)
            curve=daily.cumsum()
            saved=summary[(summary.strategy==strategy)&(summary.staking==staking)].iloc[0]
            assert np.isclose(curve.iloc[-1],saved.net)
            assert np.isclose(rows.stake.sum(),saved.staked)
            assert int(rows.bet.sum())==saved.bets
            curves.append((curve,saved,label,color))
            exports.extend(dict(date=str(day.date()),staking=staking,strategy=strategy,
                daily_net=float(value),cumulative_net=float(curve.loc[day])) for day,value in daily.items())
        allvalues=np.concatenate([c.to_numpy() for c,_,_,_ in curves])
        lo=math.floor(min(0.,float(allvalues.min()))/step)*step
        hi=math.ceil(float(allvalues.max())/step)*step
        if hi==lo:hi=lo+step
        y=lambda value:bottom+(value-lo)/(hi-lo)*(top-bottom)
        text(d,52,top+39,title,17,bold=True)
        for j,(curve,saved,label,color) in enumerate(curves):
            xpos=95+j*530
            d.add(Line(xpos,top+19,xpos+22,top+19,strokeColor=HexColor(color),strokeWidth=3))
            text(d,xpos+30,top+15,f'{label}: {int(saved.bets)} bets | {saved.staked:,.0f} units staked | ROI {saved.roi:.1%}',11,color=color)
        for value in np.arange(lo,hi+step/2,step):
            d.add(Line(left,y(value),right,y(value),strokeColor=HexColor('#e5eaf0'),strokeWidth=.6))
            text(d,left-13,y(value)-4,f'{value:+.0f}' if value else '0',11,color='#627185',textAnchor='end')
        d.add(Line(left,y(0),right,y(0),strokeColor=HexColor('#8c9aaa'),strokeWidth=1))
        for tick in pd.to_datetime(['2022-08-01','2022-10-01','2022-12-01','2023-02-01','2023-04-01','2023-06-01']):
            if start<=tick<=last:
                d.add(Line(x(tick),bottom,x(tick),bottom-5,strokeColor=HexColor('#8c9aaa')))
                text(d,x(tick),bottom-20,tick.strftime('%b %Y'),10,color='#627185',textAnchor='middle')
        text(d,52,bottom+110,'Units',10,color='#627185')
        for curve,saved,label,color in curves:
            points=[]
            # Daily settlement steps avoid implying an ordering within the day.
            previous=0.
            for day,value in curve.items():
                points.extend([x(day),y(previous),x(day),y(float(value))])
                previous=float(value)
            d.add(PolyLine(points,strokeColor=HexColor(color),strokeWidth=2.1,fillColor=None))
            d.add(Circle(x(last),y(previous),3,fillColor=HexColor(color),strokeColor=None))
            text(d,right+12,y(previous)-4,f'{previous:+.2f}',13,color=color,bold=True)
    text(d,52,49,'Net profit after stakes; both curves start at zero. Daily settlement includes drawn matches as losses.',11,color='#627185')
    text(d,52,29,'Different vertical scales. No compounding, fees or bankroll limits. Previously inspected historical test data.',10,color='#627185')
    stem=OUT/'home_only_cumulative'
    renderSVG.drawToFile(d,str(stem.with_suffix('.svg')))
    with pdfium.PdfDocument(renderPDF.drawToString(d)) as document:
        page=document[0]
        bitmap=page.render(scale=1.5)
        bitmap.to_pil().save(stem.with_suffix('.png'))
        bitmap.close();page.close()
    pd.DataFrame(exports).to_csv(OUT/'home_only_daily_curves.csv',index=False)
    print('Saved PNG, SVG and daily curve data. All four endpoints and stake totals reconcile.')


if __name__=='__main__':
    main()
