"""Rebuild five-level dollar results from vector RF OOB/train and held-out test predictions."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.lineup_deps'))
sys.path.insert(0, str(ROOT / 'src'))
import csv
import hashlib
import json
from collections import defaultdict
import numpy as np
import pandas as pd
from lineup_model import LineupForest, clean
from live_predictions import probabilities
from live_bets import recommendations
from stake_levels import size_bet


KEY = ['home_team_name', 'away_team_name', 'home_score', 'away_score', 'Matchweek',
       'B365A', 'B365D', 'B365H', 'home_GD_prior', 'away_GD_prior',
       'home_Points_prior', 'away_Points_prior']


def identity(row):
    return tuple(str(row[k]) if k in KEY[:2] else
                 int(str(row[k]).split()[-1]) if k == 'Matchweek' else float(row[k]) for k in KEY)


def main():
    model = LineupForest()  # Checks training hash, feature order and sklearn version.
    details = json.loads((ROOT/'results/run_details.json').read_text())
    test_path = ROOT/'data/test.csv'
    if hashlib.sha256(test_path.read_bytes()).hexdigest() != details['input_sha256']['test.csv']:
        raise ValueError('Test input hash differs from the saved model run')
    season_map = defaultdict(set)
    for path in sorted((ROOT/'data/season_data').glob('*_proccessed.csv')):
        code = path.name[6:10]
        season = f'20{code[:2]}–20{code[2:]}'
        for row in csv.DictReader(path.read_text(encoding='utf-8-sig').splitlines()):
            try:
                season_map[identity(row)].add(season)
            except (ValueError, KeyError):
                continue  # Incomplete source rows cannot match a complete model row.
    train = clean(pd.read_csv(ROOT/'data/train.csv')).replace([np.inf,-np.inf],np.nan).dropna()
    oob = model.forest.oob_prediction_
    if oob.shape != (len(train), 2) or not np.isfinite(oob).all() or (oob < 0).any():
        raise ValueError('Invalid or misaligned vector OOB predictions')
    tests = pd.read_csv(ROOT/'results/match_predictions.csv')
    raw_test = pd.read_csv(test_path)
    if len(tests) != details['test_rows'] or tests.source_test_row.duplicated().any():
        raise ValueError('Test predictions are incomplete or duplicated')
    summaries = {}
    ledger_path = ROOT/'results/stake_level_bet_ledger.csv'
    fields = ['split','season','source_row','home','away','strategy','level','action',
              'kelly_score','stake_dollars','gross_return_dollars','net_profit_dollars',
              'unit_return','original_bet','original_unit_return']
    with ledger_path.open('w',encoding='utf-8',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        for split, frame in [('Train (OOB)',train),('Test',tests)]:
            print(f'Evaluating {split}: {len(frame)} matches',flush=True)
            for position, (index,row) in enumerate(frame.iterrows()):
                if split == 'Train (OOB)':
                    original = row
                    p, _ = probabilities(*oob[position])
                    source_row = int(index)
                else:
                    source_row = int(row.source_test_row)
                    original = raw_test.iloc[source_row]
                    if identity(original)[:4] != identity({**original.to_dict(), **row.to_dict()})[:4]:
                        raise ValueError('Test prediction identity mismatch')
                    for k in ['B365A','B365D','B365H']:
                        if float(original[k]) != float(row[k]):
                            raise ValueError('Test odds mismatch')
                    p = [float(row['RF_vector_'+k]) for k in ('Home','Draw','Away')]
                seasons = season_map.get(identity(original),set())
                if len(seasons) != 1:
                    raise ValueError(f'Ambiguous/missing season: {split} row {source_row}: {seasons}')
                season = next(iter(seasons))
                match = dict(home=original.home_team_name,away=original.away_team_name,
                             p_home=p[0],p_draw=p[1],p_away=p[2],
                             **{k:float(original[k]) for k in ['B365A','B365D','B365H']})
                outcome = 'home' if original.home_score > original.away_score else 'away' if original.home_score < original.away_score else 'draw'
                odds = float(original[{'home':'B365H','draw':'B365D','away':'B365A'}[outcome]])
                for bet in recommendations(match):
                    sizing = size_bet(bet); level = sizing['stake_level']
                    unit_return = bet[outcome+'_stake'] * odds
                    stake = sizing['stake_dollars']
                    gross = stake * unit_return
                    writer.writerow(dict(split=split,season=season,source_row=source_row,
                        home=match['home'],away=match['away'],strategy=bet['strategy'],level=level,
                        action=sizing['staking_action'],kelly_score=sizing['kelly_score'],
                        stake_dollars=stake,gross_return_dollars=gross,net_profit_dollars=gross-stake,
                        unit_return=unit_return,original_bet=int(bet['action']=='Bet'),original_unit_return=unit_return))
                    for group in (season,'All seasons'):
                        key = (split,group,bet['strategy'],level)
                        if key not in summaries:
                            summaries[key] = dict(split=split,season=group,strategy=bet['strategy'],level=level,
                                matches=0,bets=0,wins=0,unit_returns=0.,stake_dollars=0.,gross_return_dollars=0.,
                                original_bets=0,original_gross_return=0.,unavailable=0)
                        s = summaries[key]; s['matches']+=1; s['bets']+=int(level>0)
                        s['wins']+=int(level>0 and unit_return>0)
                        s['unit_returns']+=unit_return if level else 0
                        s['stake_dollars']+=stake; s['gross_return_dollars']+=gross
                        s['original_bets']+=int(bet['action']=='Bet'); s['original_gross_return']+=unit_return
                        s['unavailable']+=int(bet['action']=='Unavailable')
    rows = sorted(summaries.values(),key=lambda s:(s['strategy'],s['split'],s['season'],s['level']))
    for s in rows:
        s['net_profit_dollars']=s['gross_return_dollars']-s['stake_dollars']
        s['roi']=s['net_profit_dollars']/s['stake_dollars'] if s['stake_dollars'] else None
    methodology = ('Saved vector RF only. Train uses out-of-bag goal predictions; test uses saved held-out RF_vector probabilities. '
        'OOB excludes the predicted row from its trees but is not a chronological walk-forward backtest. '
        'Historical actual-lineup inputs differ from live expected lineups. Each original strategy chooses its selection first; '
        'the new gate skips nonpositive model EV. Kelly score = (probability × decimal odds − 1) / (decimal odds − 1). '
        'Levels: (0,1%], (1%,2%], (2%,3%], (3%,4%], >4%, for $1–$5. Fixed thresholds; no tuning to train or test profits. '
        'Gross returns include returned stakes; net profit subtracts all stakes. Flat $1 comparison uses original rules without the new gate. '
        'No compounding, starting-bankroll constraint, fees, or cent-rounding of synthetic legs. Season labels are exact joins to historical season files. '
        'These are hypothetical historical returns, not guaranteed future winnings; the test set has already been examined in earlier project analyses.')
    report = dict(model='RF_vector',amounts=[1,2,3,4,5],methodology=methodology,rows=rows,
                  train_matches=len(train),test_matches=len(tests))
    (ROOT/'results/stake_level_results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    with (ROOT/'results/stake_level_results.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Wrote season summaries and the auditable bet ledger.',flush=True)


if __name__ == '__main__':
    main()
