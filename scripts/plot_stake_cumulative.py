"""Cumulative dollar profit ordered by season/matchweek (not exact kickoff dates)."""
from pathlib import Path
import csv
import json
from collections import defaultdict
from reportlab.graphics.shapes import Drawing, Rect, Line, PolyLine
from reportlab.lib.colors import HexColor
from plot_stake_history import text, save, TRAIN, TEST

ROOT=Path(__file__).resolve().parents[1]


def main():
    weeks={}
    for split,filename in [('Test','test.csv')]:
        with (ROOT/'data'/filename).open(encoding='utf-8-sig',newline='') as f:
            weeks[split]=[int(row['Matchweek'].split()[-1]) for row in csv.DictReader(f)]
    buckets=defaultdict(lambda:[[0,0.] for _ in range(5)])
    with (ROOT/'results/stake_level_bet_ledger.csv').open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            if row['split']!='Test':continue
            week=weeks[row['split']][int(row['source_row'])]
            key=(row['strategy'],row['split'],row['season'],week)
            levels=buckets[key];level=int(row['level'])
            if level:
                levels[level-1][0]+=1
                levels[level-1][1]+=float(row['unit_return'])-1
    curves=defaultdict(list)
    for (strategy,split,season,week),levels in sorted(buckets.items()):
        curves[strategy,split].append(dict(season=season,week=week,levels=levels))
    result=dict(ordering='Season, then matchweek. Each point settles all bets in that matchweek. Exact kickoff dates are unavailable; this is not a daily bankroll curve.',
                curves=[dict(strategy=k[0],split=k[1],points=v) for k,v in curves.items()])
    (ROOT/'results/stake_cumulative.json').write_text(json.dumps(result,separators=(',',':')),encoding='utf-8')
    summaries=json.loads((ROOT/'results/stake_level_results.json').read_text(encoding='utf-8'))['rows']
    for (strategy,split),points in curves.items():
        actual=sum(sum((i+1)*v[1] for i,v in enumerate(p['levels'])) for p in points)
        expected=sum(s['net_profit_dollars'] for s in summaries if s['strategy']==strategy and s['split']==split and s['season']=='All seasons')
        assert abs(actual-expected)<1e-7,(strategy,split,actual,expected)
    chosen='Square-root profit weighting'
    d=Drawing(1150,460);d.add(Rect(0,0,1150,460,fillColor=HexColor('#ffffff'),strokeColor=None))
    text(d,35,425,'Cumulative profit as bets are settled',26)
    text(d,35,398,chosen+' | Five-level $1-$5 stakes | Each split starts at $0',13)
    for split,top,color in [('Test',351,TEST)]:
        points=curves[chosen,split];values=[0.]
        for p in points:values.append(values[-1]+sum((i+1)*v[1] for i,v in enumerate(p['levels'])))
        left,right,bottom=90,1070,top-205
        lo,hi=min(values),max(values);pad=max(10,(hi-lo)*.13);lo-=pad;hi+=pad
        x=lambda i:left+i/len(points)*(right-left)
        y=lambda v:bottom+(v-lo)/(hi-lo)*(top-35-bottom)
        text(d,35,top,split+' — '+f'${values[-1]:,.2f} final net profit',17,color)
        for i in range(5):
            v=lo+(hi-lo)*i/4
            d.add(Line(left,y(v),right,y(v),strokeColor=HexColor('#e1e6ed')))
            text(d,left-9,y(v)-4,f'${v:,.0f}',10,textAnchor='end')
        d.add(Line(left,y(0),right,y(0),strokeColor=HexColor('#8293a7')))
        d.add(PolyLine([c for i,v in enumerate(values) for c in (x(i),y(v))],strokeColor=HexColor(color),strokeWidth=2,fillColor=None))
        starts=[i+1 for i,p in enumerate(points) if i==0 or p['season']!=points[i-1]['season']]
        if split=='Test': starts=[1]+list(range(10,len(points)+1,10))+[len(points)]
        for i in starts:
            label=points[i-1]['season'].replace('–','-') if split!='Test' else 'Week '+str(points[i-1]['week'])
            text(d,x(i),bottom-18,label,9,textAnchor='middle')
        text(d,580,bottom-39,'Season / matchweek order' if split!='Test' else '2022-2023 matchweek',10,textAnchor='middle')
    text(d,35,73,'Points aggregate settled bets by matchweek; no exact-date ordering is available. Held-out test only.',10)
    text(d,35,51,'Training-period profit is unavailable pending a chronological backtest.',10)
    text(d,35,29,'Net profit subtracts stakes. No starting bankroll, compounding, fees or synthetic-leg cent rounding. Historical results only.',10)
    save(d,'stake_cumulative_profit')
    print(f'Saved cumulative curves; all {len(curves)} endpoints reconcile with historical summaries.')


if __name__=='__main__':main()
