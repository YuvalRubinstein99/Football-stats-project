"""Rebuild five-level dollar results from held-out vector RF test predictions only."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.lineup_deps'))
sys.path.insert(0, str(ROOT / 'src'))
import csv
import hashlib
import json
from collections import defaultdict
import pandas as pd
from live_bets import integrated_recommendations
from stake_levels import size_bet


KEY = ['home_team_name', 'away_team_name', 'home_score', 'away_score', 'Matchweek',
       'B365A', 'B365D', 'B365H', 'home_GD_prior', 'away_GD_prior',
       'home_Points_prior', 'away_Points_prior']


def identity(row):
    return tuple(str(row[k]) if k in KEY[:2] else
                 int(str(row[k]).split()[-1]) if k == 'Matchweek' else float(row[k]) for k in KEY)


def main():
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
    tests = pd.read_csv(ROOT/'results/match_predictions.csv')
    raw_test = pd.read_csv(test_path)
    if len(tests) != details['test_rows'] or tests.source_test_row.duplicated().any():
        raise ValueError('Test predictions are incomplete or duplicated')
    summaries = {}
    ledger_path = ROOT/'results/stake_level_bet_ledger.csv'
    fields = ['split','season','source_row','home','away','strategy','level','action',
              'kelly_score','stake_dollars','gross_return_dollars','net_profit_dollars',
              'unit_return','original_bet','original_unit_return','decimal_odds','expected_net_per_unit','selected_outcome','week']
    with ledger_path.open('w',encoding='utf-8',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        for split, frame in [('Test',tests)]:
            print(f'Evaluating {split}: {len(frame)} matches',flush=True)
            for position, (index,row) in enumerate(frame.iterrows()):
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
                for bet in integrated_recommendations(match):
                    sizing = size_bet(bet); level = sizing['stake_level']
                    unit_return = bet[outcome+'_stake'] * odds
                    stake = sizing['stake_dollars']
                    gross = stake * unit_return
                    writer.writerow(dict(split=split,season=season,source_row=source_row,
                        home=match['home'],away=match['away'],strategy=bet['strategy'],level=level,
                        action=sizing['staking_action'],kelly_score=sizing['kelly_score'],
                        stake_dollars=stake,gross_return_dollars=gross,net_profit_dollars=gross-stake,
                        unit_return=unit_return,original_bet=int(bet['action']=='Bet'),original_unit_return=unit_return,
                        decimal_odds=bet['decimal_odds'],expected_net_per_unit=bet['expected_net_per_unit'],
                        selected_outcome='combined' if bet['synthetic'] else next((side for side in ('home','draw','away') if bet[side+'_stake']>0),'none'),
                        week=int(str(original.Matchweek).split()[-1])))
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
    methodology = 'Held-out test predictions only. Training profit is unavailable pending a chronological backtest. Historical actual lineups differ from live projected lineups. Gross returns include stakes; net profit subtracts stakes. No compounding, bankroll limit or fees. The test set has already been inspected; filtered presets are exploratory, not independently validated.'
    report = dict(model='RF_vector',amounts=[1,2,3,4,5],methodology=methodology,rows=rows,
                  train_matches=0,test_matches=len(tests))
    (ROOT/'results/stake_level_results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    with (ROOT/'results/stake_level_results.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Wrote season summaries and the auditable bet ledger.',flush=True)


if __name__ == '__main__':
    main()
