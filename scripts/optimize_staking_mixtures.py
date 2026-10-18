"""Select fixed sleeve allocations on train only; evaluate frozen weights on test."""
import csv
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.lineup_deps'))
import numpy as np

BASE = ROOT / 'results/bankroll_backtest'
WEIGHTS = np.array([[a/20, b/20, (20-a-b)/20] for a in range(21) for b in range(21-a)] + [[1/3]*3])


def metrics(days, weights):
    profits = np.asarray([d[1:] for d in days], float)
    equity = np.vstack([np.full((1, len(weights)), 1000.), 1000 + np.cumsum(profits @ weights.T, axis=0)])
    peaks = np.maximum.accumulate(equity, axis=0)
    dd = np.max(1-equity/peaks, axis=0)
    years = max((date.fromisoformat(days[-1][0][:10])-date.fromisoformat(days[0][0][:10])).days/365.25, 1/365.25)
    cagr = (equity[-1]/1000)**(1/years)-1
    return [dict(net=float(equity[-1, i]-1000), max_drawdown=float(dd[i]),
                 cagr=float(cagr[i]), score=float(cagr[i]/max(dd[i], .01))) for i in range(len(weights))]


def select(train_days):
    scores = metrics(train_days, WEIGHTS)
    # Stable deterministic ties favor lower drawdown, then balanced allocations.
    index = max(range(len(scores)), key=lambda i: (scores[i]['score'], -scores[i]['max_drawdown'], -float(np.sum(WEIGHTS[i]**2))))
    return WEIGHTS[index], scores[index]


def main():
    rows = []
    for path in sorted((BASE / 'daily').glob('*.json')):
        source = json.loads(path.read_text())
        if 'portfolios' not in source:continue
        output = []
        for league in ['All', 'E0', 'SP1', 'D1', 'I1']:
            train = next(p['days'] for p in source['portfolios'] if p['league']==league and p['split']=='Train (in-sample)')
            weights, train_score = select(train)
            # Only access test after weights have been selected and frozen.
            test = next(p['days'] for p in source['portfolios'] if p['league']==league and p['split']=='Test')
            test_score = metrics(test, np.array([weights]))[0]
            benchmarks = []
            for label, w in [('Equal thirds', [1/3]*3), *zip(source['methods'], np.eye(3).tolist())]:
                benchmarks.append(dict(method=label, train=metrics(train, np.array([w]))[0], test=metrics(test, np.array([w]))[0]))
            result = dict(league=league, weights=weights.tolist(), train=train_score, test=test_score, benchmarks=benchmarks)
            output.append(result)
            rows.append(dict(model=source['model'], league=league, flat=weights[0], quarter=weights[1], robust=weights[2],
                             train_net=train_score['net'], train_drawdown=train_score['max_drawdown'], train_score=train_score['score'],
                             test_net=test_score['net'], test_drawdown=test_score['max_drawdown'], equal_test_net=benchmarks[0]['test']['net']))
        source['optimization'] = output
        path.write_text(json.dumps(source, separators=(',', ':')), encoding='utf-8')
    with (BASE / 'train_selected_mixtures.csv').open('w', newline='', encoding='utf-8') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(configurations=len(rows), positive_test=sum(r['test_net']>0 for r in rows),
                         beats_equal=sum(r['test_net']>r['equal_test_net']+1e-7 for r in rows),
                         expanded_rf=[r for r in rows if r['model']=='Expanded RF']), indent=2))


if __name__ == '__main__':
    main()
