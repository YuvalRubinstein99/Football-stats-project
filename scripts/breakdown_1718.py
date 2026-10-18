"""Audit 2017-18 default filtered Legacy variance bets without fitting new rules."""
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def main():
    train = read(ROOT/'data/train.csv')
    leagues = defaultdict(set)
    paths=list((ROOT/'data').glob('*_data*/*1718_form_bets.csv'))+[ROOT/'data/epl_data(fully_proccessed)/epl1718_proccessed_with_form.csv']
    for path in paths:
        for r in read(path):
            leagues[r['home_team_name'], r['away_team_name'], float(r['home_score']), float(r['away_score'])].add(path.parent.name)
    bets = []
    for r in read(ROOT/'results/stake_level_bet_ledger.csv'):
        if r['season'] != '2017–2018' or r['split'] != 'Train (OOB)' or r['strategy'] != 'Legacy variance — filtered search candidate' or int(r['level']) == 0:
            continue
        t = train[int(r['source_row'])]
        assert (r['home'],r['away']) == (t['home_team_name'],t['away_team_name'])
        found = leagues[r['home'],r['away'],float(t['home_score']),float(t['away_score'])]
        assert len(found) == 1, (r['home'],r['away'],found)
        odds = float(r['decimal_odds'])
        r.update(league=next(iter(found)), odds_band='4–<5' if odds<5 else '5–<7' if odds<7 else '7–<10' if odds<10 else '10+',
                 week_band='01–10' if int(r['week'])<=10 else '11–20' if int(r['week'])<=20 else '21–30' if int(r['week'])<=30 else '31–38')
        bets.append(r)
    output=[]
    for dimension in ['total','league','selected_outcome','odds_band','level','week_band']:
        groups=defaultdict(list)
        for r in bets:groups['All' if dimension=='total' else r[dimension]].append(r)
        for name,rows in sorted(groups.items()):
            stake=sum(float(r['stake_dollars']) for r in rows)
            net=sum(float(r['net_profit_dollars']) for r in rows)
            expected=sum(float(r['stake_dollars'])*float(r['expected_net_per_unit']) for r in rows)
            output.append(dict(dimension=dimension,group=name,bets=len(rows),wins=sum(float(r['unit_return'])>0 for r in rows),staked=round(stake,2),net=round(net,2),roi_pct=round(100*net/stake,2),flat_net=round(sum(float(r['unit_return'])-1 for r in rows),2),model_expected_net=round(expected,2)))
        assert abs(sum(s['net'] for s in output if s['dimension']==dimension)+189)<.011
        assert sum(s['bets'] for s in output if s['dimension']==dimension)==578
    dest=ROOT/'results/season_1718_breakdown.csv'
    with dest.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    print(json.dumps(output,ensure_ascii=True,indent=2))

if __name__=='__main__':main()
