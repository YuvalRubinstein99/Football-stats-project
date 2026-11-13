"""Side-by-side old/new cumulative returns on identical training and test fixtures."""
import math
import html
import json
from select_expanded_features import np,pd
from compare_binary_same_split import OUT
from plot_binary_home_filters import (RULES,text,Drawing,Rect,Line,PolyLine,Circle,
                                     HexColor,renderSVG,renderPDF,pdfium)

COLORS={'old':'#b85252','new':'#087f8c'}


def draw(ledger,summary,staking):
    d=Drawing(1440,920)
    d.add(Rect(0,0,1440,920,fillColor=HexColor('#ffffff'),strokeColor=None))
    text(d,50,878,'Old vs new: identical training and test data',27,bold=True)
    kind='Flat stakes: 1 unit per bet' if staking=='flat' else 'Variable stakes: existing 1-5-unit tiers'
    text(d,50,849,kind,16)
    text(d,50,825,'Both models refitted on the same 8,521 rows; both evaluated on the same 1,065 held-out rows.',12,color='#627185')
    for i,name in enumerate(['old','new']):
        x=50+i*530
        d.add(Line(x,799,x+26,799,strokeColor=HexColor(COLORS[name]),strokeWidth=3))
        label='Old: depth 5, full feature set' if name=='old' else 'New: depth 2, minimum leaf 100, 12 features'
        text(d,x+35,795,label,13,color=COLORS[name],bold=True)
    exports=[]
    for split,left,right in [('train',80,625),('test',800,1345)]:
        frame=ledger[(ledger.split==split)&(ledger.staking==staking)].copy()
        frame['fixture_date']=pd.to_datetime(frame.fixture_date)
        start=frame.fixture_date.min()-pd.Timedelta(days=1);last=frame.fixture_date.max()
        dates=pd.DatetimeIndex([start,*sorted(frame.fixture_date.unique())])
        title='IN-SAMPLE | 2014-15 to 2021-22' if split=='train' else 'TEST | 2022-23'
        text(d,left,756,title,18,bold=True)
        x=lambda day:left+(pd.Timestamp(day)-start).days/(last-start).days*(right-left)
        for (strategy,label,_),top,bottom in zip(RULES,[675,335],[435,95]):
            text(d,left,top+49,label,15,bold=True)
            curves=[]
            for j,name in enumerate(['old','new']):
                rows=frame[(frame.model==name)&(frame.strategy==strategy)]
                assert rows.match_id.is_unique and len(rows)==(8521 if split=='train' else 1065)
                daily=rows.groupby('fixture_date').net.sum().reindex(dates,fill_value=0.)
                cumulative=daily.cumsum()
                saved=summary[(summary.model==name)&(summary.split==split)&(summary.strategy==strategy)&(summary.staking==staking)].iloc[0]
                assert np.isclose(cumulative.iloc[-1],saved.net)
                assert np.isclose(rows.stake.sum(),saved.staked) and rows.bet.sum()==saved.bets
                curves.append((name,cumulative))
                text(d,left+j*285,top+25,f'{name.capitalize()}: {int(saved.bets):,} bets | ROI {saved.roi:+.1%}',11,color=COLORS[name])
                exports.extend(dict(model=name,split=split,staking=staking,strategy=strategy,
                    date=str(day.date()),daily_net=float(value),cumulative_net=float(cumulative.loc[day]))
                    for day,value in daily.items())
            values=np.concatenate([c.to_numpy() for _,c in curves])
            span=max(float(values.max()),0)-min(float(values.min()),0)
            raw=max(span/5,1);scale=10**math.floor(math.log10(raw))
            step=next(v*scale for v in [1,2,2.5,5,10] if v*scale>=raw)
            lo=math.floor(min(0.,float(values.min()))/step)*step
            hi=math.ceil(max(0.,float(values.max()))/step)*step
            if hi==lo:hi+=step
            y=lambda value:bottom+(value-lo)/(hi-lo)*(top-bottom)
            for value in np.arange(lo,hi+step/2,step):
                d.add(Line(left,y(value),right,y(value),strokeColor=HexColor('#e5eaf0'),strokeWidth=.6))
                text(d,left-9,y(value)-4,f'{value:+g}' if value else '0',10,color='#627185',textAnchor='end')
            d.add(Line(left,y(0),right,y(0),strokeColor=HexColor('#8c9aaa'),strokeWidth=1))
            if split=='test':ticks=[start,*pd.date_range(start+pd.Timedelta(days=30),last,freq='2MS')]
            else:ticks=[start,*[pd.Timestamp(year=v,month=8,day=1) for v in [2016,2018,2020,2021]]]
            for tick in ticks:
                if start<=tick<=last:
                    label=tick.strftime('%b %y') if split=='test' else str(tick.year)
                    text(d,x(tick),bottom-19,label,10,color='#627185',textAnchor='middle')
            previous_label=None
            for name,curve in curves:
                points=[];previous=0.
                for day,value in curve.items():
                    points.extend([x(day),y(previous),x(day),y(float(value))]);previous=float(value)
                color=COLORS[name]
                d.add(PolyLine(points,strokeColor=HexColor(color),strokeWidth=2,fillColor=None))
                d.add(Circle(x(last),y(previous),2.5,fillColor=HexColor(color),strokeColor=None))
                label_y=y(previous)-4
                if previous_label is not None and abs(label_y-previous_label)<17:
                    label_y=previous_label+17 if label_y>=previous_label else previous_label-17
                    d.add(Line(right+2,y(previous),right+7,label_y+4,strokeColor=HexColor(color)))
                text(d,right+9,label_y,f'{previous:+.2f}',11,color=color,bold=True)
                previous_label=label_y
    text(d,50,46,'Net profit in units. Each panel has its own scale. Home bets only; draws lose. No compounding, fees or bankroll limits.',11,color='#627185')
    text(d,50,25,'Retrospective diagnostic: the 2022-23 test season influenced the new specification. No settings or thresholds changed in this comparison.',10,color='#627185')
    stem=OUT/f'cumulative_{staking}'
    renderSVG.drawToFile(d,str(stem.with_suffix('.svg')))
    with pdfium.PdfDocument(renderPDF.drawToString(d)) as document:
        page=document[0];bitmap=page.render(scale=1.5)
        bitmap.to_pil().save(stem.with_suffix('.png'));bitmap.close();page.close()
    return exports


def main():
    ledger=pd.read_csv(OUT/'ledger.csv.gz')
    summary=pd.read_csv(OUT/'returns.csv')
    exports=[]
    for staking in ['flat','tiers']:exports.extend(draw(ledger,summary,staking))
    pd.DataFrame(exports).to_csv(OUT/'daily_curves.csv',index=False)
    plan=json.loads((OUT/'plan.json').read_text())
    gaps=pd.read_csv(OUT/'probability_gaps.csv')
    body='<h1>Old and new models on exactly the same split</h1>'
    body+='<p>Both fits: 8,521 matches, 2014-15 through 2021-22. Both tests: 1,065 matches in 2022-23.</p>'
    body+='<p>'+html.escape(plan['controls'])+'</p><p>'+html.escape(plan['caveat'])+'</p>'
    body+='<p>'+html.escape(plan['reason_for_split'])+'</p>'
    body+='<h2>Probability generalization gaps</h2>'+gaps.to_html(index=False,float_format=lambda v:f'{v:.6f}')
    body+='<img src="cumulative_flat.svg" alt="Old versus new, flat stakes">'
    body+='<img src="cumulative_tiers.svg" alt="Old versus new, variable stakes">'
    cols=['model','strategy','staking','bets_train','net_train','roi_train','bets_test','net_test','roi_test','roi_gap_percentage_points']
    body+='<h2>Betting generalization gaps</h2>'+pd.read_csv(OUT/'betting_gaps.csv')[cols].to_html(index=False,float_format=lambda v:f'{v:.5f}')
    body+='<h2>Paired test comparison</h2><pre>'+html.escape((OUT/'paired_test_difference.json').read_text())+'</pre>'
    body+='<h2>Verification</h2><pre>'+html.escape((OUT/'verification.json').read_text())+'</pre>'
    (OUT/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Matched model comparison</title>'
        '<style>body{font:15px system-ui;margin:30px;color:#263548}img{width:100%;max-width:1440px}'
        'table{border-collapse:collapse;font-size:12px}td,th{padding:7px;border-bottom:1px solid #ddd}</style>'+body,encoding='utf-8')
    print('Saved two comparison figures, report and daily curves; all 16 endpoints reconcile.')


if __name__=='__main__':main()
