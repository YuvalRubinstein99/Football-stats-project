"""Audit frozen weights against source summaries and independently rebuilt curves."""
import csv
import json
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]/'results/bankroll_backtest'
with (BASE/'summary.csv').open(newline='',encoding='utf-8') as stream:
    summaries={(r['model'],r['league'],r['split'],r['method']):r for r in csv.DictReader(stream)}
count=0
for path in (BASE/'daily').glob('*.json'):
    source=json.loads(path.read_text())
    if 'portfolios' not in source:continue
    for chosen in source['optimization']:
        weights=chosen['weights']
        assert abs(sum(weights)-1)<1e-12 and min(weights)>=0
        assert all(chosen['train']['score']+1e-9>=b['train']['score'] for b in chosen['benchmarks'])
        for split,key in [('Train (in-sample)','train'),('Test','test')]:
            days=next(p['days'] for p in source['portfolios'] if p['league']==chosen['league'] and p['split']==split)
            balance=peak=1000.;dd=0.
            for day in days:
                balance+=sum(w*n for w,n in zip(weights,day[1:]));peak=max(peak,balance);dd=max(dd,1-balance/peak)
            expected=sum(w*float(summaries[source['model'],chosen['league'],split,m]['net']) for w,m in zip(weights,source['methods']))
            assert abs(balance-1000-expected)<max(1e-5,abs(expected)*1e-10)
            assert abs(dd-chosen[key]['max_drawdown'])<1e-9
            assert abs(balance-1000-chosen[key]['net'])<max(1e-5,abs(expected)*1e-10)
        count+=1
assert count==220
print(f'Passed {count} frozen allocations: normalized weights, training benchmark scores, train/test profits and daily drawdowns.')
