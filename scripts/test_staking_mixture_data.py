"""Ensure single-strategy mixtures recover the original daily risk statistics."""
import csv
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / 'results/bankroll_backtest'
with (BASE / 'summary.csv').open(newline='', encoding='utf-8') as stream:
    reference = {(r['model'], r['split'], r['league'], r['method']): r for r in csv.DictReader(stream)}
checked = 0
for path in (BASE / 'daily').glob('*.json'):
    source = json.loads(path.read_text())
    if 'portfolios' not in source:continue
    for p in source['portfolios']:
        for i, method in enumerate(source['methods']):
            balance = peak = 1000.
            drawdown = 0.
            for day in p['days']:
                balance += day[i+1]
                peak = max(peak, balance)
                drawdown = max(drawdown, 1-balance/peak)
            expected = reference[source['model'], p['split'], p['league'], method]
            assert abs(drawdown-float(expected['max_drawdown'])) < 1e-9
            assert abs(balance-float(expected['ending_bankroll'])) < max(1e-5, abs(balance)*1e-10)
            checked += 1
assert checked == len(reference)
print(f'Passed: {checked} single-strategy daily paths recover original profit and drawdown.')
