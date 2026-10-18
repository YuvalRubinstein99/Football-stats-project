"""Export daily sleeve profits for interactive mixtures, checking source totals."""
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / 'results/bankroll_backtest'
METHODS = ['Flat $1', 'Quarter Kelly', 'Robust quarter Kelly']


def main():
    daily = defaultdict(lambda: defaultdict(lambda: [0., 0., 0.]))
    with gzip.open(BASE / 'ledger.csv.gz', 'rt', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            key = row['model'], row['split'], row['portfolio_league']
            daily[key][row['date']][METHODS.index(row['method'])] += float(row['net'])
    with (BASE / 'summary.csv').open(encoding='utf-8', newline='') as stream:
        summaries = list(csv.DictReader(stream))
    for row in summaries:
        values = daily[row['model'], row['split'], row['league']]
        total = sum(v[METHODS.index(row['method'])] for v in values.values())
        assert abs(total-float(row['net'])) <= max(1e-5, abs(total)*1e-10)
    target = BASE / 'daily'
    target.mkdir(exist_ok=True)
    for model in dict.fromkeys(row['model'] for row in summaries):
        portfolios = [dict(split=split, league=league,
                           days=[[date, *net] for date, net in sorted(values.items())])
                      for (name, split, league), values in daily.items() if name == model]
        filename = hashlib.sha256(model.encode()).hexdigest() + '.json'
        (target / filename).write_text(json.dumps(dict(model=model, methods=METHODS, portfolios=portfolios), separators=(',', ':')), encoding='utf-8')
    print(f'Validated {len(summaries)} daily strategy totals and exported all models.')
    from optimize_staking_mixtures import main as optimize
    optimize()


if __name__ == '__main__':
    main()
