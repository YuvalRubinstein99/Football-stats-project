import sys,csv,json
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from historical_leagues import row_leagues

def main():
    leagues=row_leagues();buckets=defaultdict(lambda:[[0,0.] for _ in range(5)])
    with (ROOT/'results/stake_level_bet_ledger.csv').open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            key=r['strategy'],r['split'],leagues[r['split'],int(r['source_row'])],r['season'],int(r['week'])
            values=buckets[key];level=int(r['level'])
            if level:values[level-1][0]+=1;values[level-1][1]+=float(r['unit_return'])-1
    curves=defaultdict(list)
    for (strategy,split,league,season,week),levels in sorted(buckets.items()):
        curves[strategy,split,league].append(dict(season=season,week=week,levels=levels))
    path=ROOT/'results/stake_cumulative.json';report=json.loads(path.read_text(encoding='utf-8'))
    report['league_curves']=[dict(strategy=k[0],split=k[1],league=k[2],points=v) for k,v in curves.items()]
    for c in report['curves']:
        overall=sum(sum((i+1)*x[1] for i,x in enumerate(p['levels'])) for p in c['points'])
        parts=sum(sum((i+1)*x[1] for i,x in enumerate(p['levels'])) for k,v in curves.items() if k[:2]==(c['strategy'],c['split']) for p in v)
        assert abs(overall-parts)<1e-7
    path.write_text(json.dumps(report,separators=(',',':')),encoding='utf-8')
    print(f'{len(curves)} league curves; every strategy/split reconciles.')
if __name__=='__main__':main()
