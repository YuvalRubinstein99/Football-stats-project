"""Reconstruct fixed equal-capital strategy sleeves from daily backtest ledgers."""
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results' / 'bankroll_backtest'
METHODS = ['Flat $1', 'Quarter Kelly', 'Robust quarter Kelly']


def main():
    daily = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    spent = defaultdict(float)
    with gzip.open(BASE / 'ledger.csv.gz', 'rt', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if row['model'] != 'Expanded RF':
                continue
            key = row['split'], row['portfolio_league'], row['method']
            daily[key[:2]][row['date']][row['method']] += float(row['net'])
            spent[key] += float(row['stake'])
    with (BASE / 'summary.csv').open(newline='', encoding='utf-8') as stream:
        reference = {(r['split'], r['league'], r['method']): r
                     for r in csv.DictReader(stream) if r['model'] == 'Expanded RF'}
    results = []
    for (split, league), dates in daily.items():
        balances = {m: 1000. for m in METHODS}
        balance = peak = 1000.
        dd = dd_dollars = 0.
        points = []
        for date, nets in sorted(dates.items()):
            for m in METHODS:
                balances[m] += nets[m]
            balance = sum(balances.values()) / 3
            peak = max(peak, balance)
            dd = max(dd, 1 - balance / peak)
            dd_dollars = max(dd_dollars, peak - balance)
            points.append({'date': date, 'net': balance - 1000})
        for m in METHODS:
            assert abs(balances[m] - float(reference[split, league, m]['ending_bankroll'])) < 1e-5
        total_staked = sum(spent[split, league, m] for m in METHODS) / 3
        result = dict(split=split, league=league, net=balance-1000,
                      staked=total_staked, roi=(balance-1000)/total_staked,
                      max_drawdown=dd, max_drawdown_dollars=dd_dollars)
        print(json.dumps(result))
        results.append(dict(**result, points=points))
    output = dict(model='Expanded RF', initial_bankroll=1000,
                  allocation={m: 1/3 for m in METHODS}, rebalanced=False,
                  note='Illustrative weights, not optimized. Flat sleeve stake is $1/3. Historical Bet365 odds; train is in-sample. Each league is an independent $1000 portfolio.',
                  results=results)
    (BASE / 'staking_mixture.json').write_text(json.dumps(output), encoding='utf-8')


if __name__ == '__main__':
    main()
