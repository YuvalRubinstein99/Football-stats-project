"""Render training-period forward-test profit curves, with daily settlement."""
import math
from select_expanded_features import np,pd
from backtest_binary_training_oos import OUT
from plot_binary_home_filters import (RULES,text,Drawing,Rect,Line,PolyLine,Circle,
                                     HexColor,renderSVG,renderPDF,pdfium)


def render(frame,summary,protocol,*,out=OUT,heading=None,subtitle=None,note=None,footer=None):
    frame=frame.copy()
    frame['fixture_date']=pd.to_datetime(frame.fixture_date)
    start=frame.fixture_date.min()-pd.Timedelta(days=1)
    last=frame.fixture_date.max()
    dates=pd.DatetimeIndex([start,*sorted(frame.fixture_date.unique())])
    matches=frame.source_row.nunique()
    d=Drawing(1200,850)
    d.add(Rect(0,0,1200,850,fillColor=HexColor('#ffffff'),strokeColor=None))
    text(d,52,808,heading or 'Training seasons: out-of-sample cumulative profit',25,bold=True)
    subtitle=subtitle or ('Same RF feature set; annual forward refits | 2015-16 to 2021-22' if protocol=='fixed'
              else 'RF with past-only feature selection | 2017-18 to 2021-22')
    text(d,52,779,f'{subtitle} | {matches:,} fixtures',13)
    note=note or ('The final feature choice is retrospective; every prediction uses a model fitted on earlier seasons.'
          if protocol=='fixed' else 'Each season chooses features using the preceding two validation seasons, then refits on all earlier data.')
    text(d,52,756,note,11,color='#627185')
    exports=[]
    left,right=95,1040
    x=lambda day:left+(pd.Timestamp(day)-start).days/(last-start).days*(right-left)
    for staking,top,bottom,title in [('flat',688,450,'Flat stakes: 1 unit per bet'),
                                    ('tiers',337,99,'Variable stakes: existing 1-5-unit tiers')]:
        curves=[]
        for strategy,label,color in RULES:
            rows=frame[(frame.strategy==strategy)&(frame.staking==staking)]
            assert len(rows)==matches and rows.source_row.nunique()==matches
            daily=rows.groupby('fixture_date').net.sum().reindex(dates,fill_value=0.)
            curve=daily.cumsum()
            saved=summary[(summary.strategy==strategy)&(summary.staking==staking)].iloc[0]
            assert np.isclose(curve.iloc[-1],saved.net)
            assert np.isclose(rows.stake.sum(),saved.staked) and int(rows.bet.sum())==saved.bets
            curves.append((curve,saved,label,color))
            exports.extend(dict(protocol=protocol,date=str(day.date()),staking=staking,strategy=strategy,
                daily_net=float(value),cumulative_net=float(curve.loc[day])) for day,value in daily.items())
        values=np.concatenate([c.to_numpy() for c,_,_,_ in curves])
        spread=max(float(values.max()),0)-min(float(values.min()),0)
        raw=max(spread/5,1)
        scale=10**math.floor(math.log10(raw))
        step=next(v*scale for v in [1,2,2.5,5,10] if v*scale>=raw)
        lo=math.floor(min(0.,float(values.min()))/step)*step
        hi=math.ceil(max(0.,float(values.max()))/step)*step
        if lo==hi:hi+=step
        y=lambda value:bottom+(value-lo)/(hi-lo)*(top-bottom)
        text(d,52,top+39,title,17,bold=True)
        for j,(curve,saved,label,color) in enumerate(curves):
            xpos=95+j*530
            d.add(Line(xpos,top+19,xpos+22,top+19,strokeColor=HexColor(color),strokeWidth=3))
            text(d,xpos+30,top+15,f'{label}: {int(saved.bets):,} bets | {saved.staked:,.0f} staked | ROI {saved.roi:.1%}',11,color=color)
        for value in np.arange(lo,hi+step/2,step):
            d.add(Line(left,y(value),right,y(value),strokeColor=HexColor('#e5eaf0'),strokeWidth=.6))
            text(d,left-13,y(value)-4,f'{value:+g}' if value else '0',11,color='#627185',textAnchor='end')
        d.add(Line(left,y(0),right,y(0),strokeColor=HexColor('#8c9aaa'),strokeWidth=1))
        first_year=int(frame.season.min())
        short_period=(last-start).days<=400
        first_label=start.strftime('%b %Y') if short_period else f'{first_year}-{str(first_year+1)[-2:]}'
        text(d,left,bottom-20,first_label,10,color='#627185',textAnchor='middle')
        ticks=(pd.date_range(start+pd.Timedelta(days=30),last,freq='2MS') if short_period
               else [pd.Timestamp(year=year,month=8,day=1) for year in range(start.year,last.year+1)])
        for tick in ticks:
            if start<=tick<=last:
                d.add(Line(x(tick),bottom,x(tick),bottom-5,strokeColor=HexColor('#8c9aaa')))
                label=tick.strftime('%b %Y') if short_period else f'{tick.year}-{str(tick.year+1)[-2:]}'
                text(d,x(tick),bottom-20,label,10,color='#627185',textAnchor='middle')
        text(d,52,top+4,'Units',10,color='#627185')
        previous_label=None
        for curve,saved,label,color in curves:
            points=[];previous=0.
            for day,value in curve.items():
                points.extend([x(day),y(previous),x(day),y(float(value))]);previous=float(value)
            d.add(PolyLine(points,strokeColor=HexColor(color),strokeWidth=2.1,fillColor=None))
            d.add(Circle(x(last),y(previous),3,fillColor=HexColor(color),strokeColor=None))
            label_y=y(previous)-4
            if previous_label is not None and abs(label_y-previous_label)<19:
                label_y=previous_label+19 if label_y>=previous_label else previous_label-19
                d.add(Line(right+3,y(previous),right+10,label_y+4,strokeColor=HexColor(color)))
            text(d,right+12,label_y,f'{previous:+.2f}',13,color=color,bold=True)
            previous_label=label_y
    text(d,52,49,'Home bets only. Draws count as losses. Daily settlement; no compounding, fees or bankroll limits.',11,color='#627185')
    text(d,52,29,footer or 'Different vertical scales. Historical research sample; rating publication timestamps remain unverified.',10,color='#627185')
    stem=out/f'{protocol}_cumulative'
    renderSVG.drawToFile(d,str(stem.with_suffix('.svg')))
    with pdfium.PdfDocument(renderPDF.drawToString(d)) as document:
        page=document[0];bitmap=page.render(scale=1.5)
        bitmap.to_pil().save(stem.with_suffix('.png'))
        bitmap.close();page.close()
    return exports


def main():
    ledger=pd.read_csv(OUT/'ledger.csv.gz')
    summaries=pd.read_csv(OUT/'summary.csv')
    exports=[]
    for protocol,frame in ledger.groupby('protocol'):
        exports.extend(render(frame,summaries[summaries.protocol==protocol],protocol))
    pd.DataFrame(exports).to_csv(OUT/'daily_curves.csv',index=False)
    print('Saved fixed and nested PNG/SVG plots; all eight endpoints reconcile.')


if __name__=='__main__':
    main()
