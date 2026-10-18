"""Build expanded RF OOB/test ledgers, curves and cached-input live preview."""
import sys,csv,json
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.lineup_deps'));sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
import numpy as np
from expanded_model import ExpandedForest
from lineup_model import clean
from live_predictions import probabilities,atomic_write
from live_bets import integrated_recommendations,save
from stake_levels import size_bet
from historical_leagues import row_leagues
from backtest_stake_levels import identity
OUT=ROOT/'results/expanded_interface'

def preview_csv():
    report=json.loads((OUT/'live/predictions.json').read_text(encoding='utf-8'))
    fields=['league','date','time','home','away','expected_home_goals','expected_away_goals','p_home','p_draw','p_away','pick','modal_score','lineup_status','ratings_edition','matchweek','notes']
    with (OUT/'live/predictions.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(report['predictions'])

def main():
    OUT.mkdir(exist_ok=True);model=ExpandedForest()
    train=pd.read_csv(ROOT/'data/expanded_2008/train.csv');test=pd.read_csv(ROOT/'data/expanded_2008/test.csv')
    added=pd.read_csv(ROOT/'data/expanded_2008/added_row_provenance.csv')
    original=clean(pd.read_csv(ROOT/'data/train.csv')).replace([np.inf,-np.inf],np.nan).dropna();leagues=row_leagues()
    oldrows=list(csv.DictReader((ROOT/'results/stake_level_bet_ledger.csv').open(encoding='utf-8')))
    seasons={(r['split'],int(r['source_row'])):r['season'] for r in oldrows};del oldrows
    comparisons=pd.read_csv(ROOT/'results/expanded_comparison/match_predictions.csv')
    summaries={};buckets=defaultdict(lambda:[[0,0.] for _ in range(5)])
    fields=['split','season','source_row','home','away','strategy','level','action','kelly_score','stake_dollars','gross_return_dollars','net_profit_dollars','unit_return','original_bet','original_unit_return','decimal_odds','expected_net_per_unit','selected_outcome','week','league']
    with (OUT/'stake_level_bet_ledger.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for split,frame in [('Train (OOB)',train),('Test',test)]:
            print(f'Expanded {split}: {len(frame)} matches',flush=True)
            for i,r in frame.iterrows():
                if split=='Train (OOB)':
                    p,_=probabilities(*model.forest.oob_prediction_[i])
                    if i<len(added):
                        season=added.iloc[i].season.replace('/','–');league={'E0':'Premier League','D1':'Bundesliga','I1':'Serie A','SP1':'La Liga'}[added.iloc[i].league]
                    else:
                        source=int(original.index[i-len(added)]);season=seasons[split,source];league=leagues[split,source]
                else:
                    season=seasons[split,i];league=leagues[split,i]
                    p=[comparisons.iloc[i]['expanded__RF_vector__p_'+s] for s in ('home','draw','away')]
                match=dict(home=r.home_team_name,away=r.away_team_name,p_home=p[0],p_draw=p[1],p_away=p[2],**{k:float(r[k]) for k in ['B365H','B365D','B365A']})
                side='home' if r.home_score>r.away_score else 'away' if r.home_score<r.away_score else 'draw';odds=r[{'home':'B365H','draw':'B365D','away':'B365A'}[side]]
                for bet in integrated_recommendations(match):
                    sizing=size_bet(bet);level=sizing['stake_level'];stake=sizing['stake_dollars'];ret=bet[side+'_stake']*odds;gross=stake*ret
                    writer.writerow(dict(split=split,season=season,source_row=i,home=r.home_team_name,away=r.away_team_name,strategy=bet['strategy'],level=level,action=sizing['staking_action'],kelly_score=sizing['kelly_score'],stake_dollars=stake,gross_return_dollars=gross,net_profit_dollars=gross-stake,unit_return=ret,original_bet=int(bet['action']=='Bet'),original_unit_return=ret,decimal_odds=bet['decimal_odds'],expected_net_per_unit=bet['expected_net_per_unit'],selected_outcome='combined' if bet['synthetic'] else next((s for s in ('home','draw','away') if bet[s+'_stake']>0),'none'),week=int(r.Matchweek),league=league))
                    for group in (season,'All seasons'):
                        key=split,group,bet['strategy'],level
                        s=summaries.setdefault(key,dict(split=split,season=group,strategy=bet['strategy'],level=level,matches=0,bets=0,wins=0,unit_returns=0.,stake_dollars=0.,gross_return_dollars=0.,original_bets=0,original_gross_return=0.,unavailable=0))
                        s['matches']+=1;s['bets']+=int(level>0);s['wins']+=int(level>0 and ret>0);s['unit_returns']+=ret if level else 0;s['stake_dollars']+=stake;s['gross_return_dollars']+=gross;s['original_bets']+=int(bet['action']=='Bet');s['original_gross_return']+=ret;s['unavailable']+=int(bet['action']=='Unavailable')
                    for lg in ('All',league):
                        b=buckets[bet['strategy'],split,lg,season,int(r.Matchweek)]
                        if level:b[level-1][0]+=1;b[level-1][1]+=ret-1
    rows=list(summaries.values())
    for r in rows:r['net_profit_dollars']=r['gross_return_dollars']-r['stake_dollars'];r['roi']=r['net_profit_dollars']/r['stake_dollars'] if r['stake_dollars'] else None
    methodology='Expanded 2008–22 RF (19,263 training matches). Train (OOB) predicts each row only with trees omitting that row, but can use later seasons; not chronological validation. Test: 2022–23, excluded from fitting but already inspected. Actual historical lineups and dated FIFA ratings. Net profit subtracts stakes; no compounding, fees or bankroll limit.'
    atomic_write(OUT/'stake_level_results.json',json.dumps(dict(model='Expanded RF',amounts=[1,2,3,4,5],methodology=methodology,rows=rows,train_matches=len(train),test_matches=len(test)),allow_nan=False))
    with (OUT/'stake_level_results.csv').open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    curves=defaultdict(list)
    for (strategy,split,league,season,week),levels in sorted(buckets.items()):curves[strategy,split,league].append(dict(season=season,week=week,levels=levels))
    groups=[dict(strategy=k[0],split=k[1],league=k[2],points=v) for k,v in curves.items()]
    atomic_write(OUT/'stake_cumulative.json',json.dumps(dict(ordering='Season / matchweek order, not exact dates. Expanded RF; each split starts at $0.',curves=[r for r in groups if r['league']=='All'],league_curves=[r for r in groups if r['league']!='All'])))
    reference=pd.read_csv(ROOT/'results/expanded_comparison/strategy_results.csv')
    for name in reference[reference.model=='expanded__RF_vector'].itertuples():
        assert abs(sum(r['net_profit_dollars'] for r in rows if r['strategy']==name.strategy and r['split']=='Test' and r['season']=='All seasons')-name.net)<1e-7
    # Recompute saved inputs, explicitly preserving their original collection time.
    live=json.loads((ROOT/'results/live/predictions.json').read_text(encoding='utf-8'))
    audits={a['fixture']:a['features'] for a in live['feature_audit']}
    for m in live['predictions']:
        h,a=model.predict(audits['|'.join(m[k] for k in ['league','date','home','away'])]);p,score=probabilities(h,a)
        m.update(expected_home_goals=h,expected_away_goals=a,p_home=p[0],p_draw=p[1],p_away=p[2],modal_score=score,pick=['Home','Draw','Away'][np.argmax(p)])
        m.pop('strategies',None)
    live.update(model=model.metadata['name'],model_metadata=model.metadata)
    live['warnings'].insert(0,'Expanded RF preview recomputed from previously saved inputs; collection timestamp unchanged. Refresh for current lineups and odds.')
    save(live,OUT/'live');atomic_write(OUT/'live/predictions.json',json.dumps(live,allow_nan=False))
    preview_csv()
    print('Expanded history reconciled with comparison; cached-input preview generated.',flush=True)

if __name__=='__main__':
    if '--preview-csv-only' in sys.argv:preview_csv()
    else:main()
