"""Separate chronological RF experiments on newly collected older seasons.

No active model/report is overwritten. No strategy search or threshold fitting.
"""
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.lineup_deps'))
sys.path.insert(0,str(ROOT/'src'))
import csv
import hashlib
import json
from collections import defaultdict
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss
from live_predictions import probabilities
from live_bets import integrated_recommendations
from stake_levels import size_bet
from scrape_older_seasons import OUT, write_csv, atomic_json, stamp

DEST=ROOT/'results/older_seasons'

def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    status=json.loads((OUT/'status.json').read_text())
    if status['state'] not in ('completed','completed_with_gaps'):raise ValueError('Wait for collection to finish')
    DEST.mkdir(parents=True,exist_ok=True)
    coverage={(r['league'],r['season']):r for r in read(OUT/'results_coverage.csv')}
    complete=set();audit=[]
    for r in read(OUT/'lineup_coverage.csv'):
        key=(r['league'],r['season']);reference=coverage.get(key)
        valid=bool(reference and int(reference['matches'])==int(r['matches']) and int(reference['invalid_rows'])==0)
        audit.append(dict(**r,reference_matches=reference['matches'] if reference else '',season_count_check='pass' if valid else 'excluded'))
        if valid:complete.add(key)
    write_csv(DEST/'season_audit.csv',audit)
    rows=[r for r in read(OUT/'lineup_features.csv') if (r['league'],r['season']) in complete]
    for r in rows:
        r['Home_min_max']=float(r['HomePlayer_Overall_min'])*float(r['HomePlayer_Overall_max'])
        r['Away_min_max']=float(r['AwayPlayer_Overall_min'])*float(r['AwayPlayer_Overall_max'])
    # Names retained in the archive's stable team vocabulary, independently of active model.
    categorical=['home_team_name','away_team_name','league']
    numeric=[k for k in rows[0] if k.startswith(('HomePlayer_','AwayPlayer_','home_GD','away_GD','home_Points','away_Points'))]+['Matchweek','B365H','B365D','B365A','Home_min_max','Away_min_max']
    seasons=sorted({r['season'] for r in rows});summaries=[];all_predictions=[]
    config=dict(n_estimators=2000,max_depth=5,max_features=.3,max_samples=.5,bootstrap=True,random_state=20261011,n_jobs=4)
    manifest=dict(state='running',started_at=stamp(),config=config,input_sha256=hashlib.sha256((OUT/'lineup_features.csv').read_bytes()).hexdigest(),folds=[],limitations='Chronological expanding windows within newly sourced 2008–14 data only. Actual lineups; dated rating snapshots may be stale. Entire league-seasons failing reference match-count checks excluded; counts alone do not verify fixture identity. Fixed strategies, no tuning. Different training data from active model, not comparable as a direct model upgrade.')
    atomic_json(DEST/'status.json',manifest)
    for season in seasons[1:]:
        train=[r for r in rows if r['season']<season];test=[r for r in rows if r['season']==season]
        assert max(r['date'] for r in train)<min(r['date'] for r in test)
        print(f'{season}: {len(train)} earlier training matches, {len(test)} test matches',flush=True)
        encoder=OneHotEncoder(sparse_output=False,handle_unknown='ignore')
        def cats(rs):return [[r[k] for k in categorical] for r in rs]
        def nums(rs):return np.array([[float(r[k]) for k in numeric] for r in rs])
        xtrain=np.column_stack((encoder.fit_transform(cats(train)),nums(train)))
        xtest=np.column_stack((encoder.transform(cats(test)),nums(test)))
        model=RandomForestRegressor(**config).fit(xtrain,[[int(r['home_score']),int(r['away_score'])] for r in train])
        means=model.predict(xtest);predictions=[];aggregates=defaultdict(lambda:dict(bets=0,staked=0.,net=0.,flat_net=0.))
        probs=[];labels=[];market=[]
        for r,mean in zip(test,means):
            p,_=probabilities(*mean);probs.append(p)
            outcome=0 if int(r['home_score'])>int(r['away_score']) else 2 if int(r['home_score'])<int(r['away_score']) else 1
            labels.append(outcome);side=('home','draw','away')[outcome]
            odds=[float(r[k]) for k in ('B365H','B365D','B365A')];q=1/np.array(odds);market.append(q/q.sum())
            match=dict(home=r['home_team_name'],away=r['away_team_name'],p_home=p[0],p_draw=p[1],p_away=p[2],B365H=odds[0],B365D=odds[1],B365A=odds[2])
            predictions.append(dict(match_id=r['match_id'],season=season,date=r['date'],league=r['league'],**match,actual_outcome=side))
            for bet in integrated_recommendations(match):
                sizing=size_bet(bet);stake=sizing['stake_dollars']
                if not stake:continue
                ret=bet[side+'_stake']*odds[outcome]
                for group in ('All',r['league']):
                    a=aggregates[bet['strategy'],group];a['bets']+=1;a['staked']+=stake;a['net']+=stake*(ret-1);a['flat_net']+=ret-1
        for (strategy,league),a in aggregates.items():summaries.append(dict(season=season,strategy=strategy,league=league,**a,roi=a['net']/a['staked']))
        all_predictions.extend(predictions)
        manifest['folds'].append(dict(season=season,train_matches=len(train),test_matches=len(test),log_loss=log_loss(labels,probs,labels=[0,1,2]),market_log_loss=log_loss(labels,market,labels=[0,1,2])))
        write_csv(DEST/'chronological_predictions.csv',all_predictions);write_csv(DEST/'strategy_results.csv',summaries);atomic_json(DEST/'status.json',manifest)
    manifest.update(state='completed',finished_at=stamp());atomic_json(DEST/'status.json',manifest)
    print('Completed separate chronological experiments.',flush=True)

if __name__=='__main__':
    DEST.mkdir(parents=True,exist_ok=True)
    lock=DEST/'run.lock'
    try:lock.mkdir()
    except FileExistsError:raise SystemExit('Evaluation already running, or stale run.lock needs process verification')
    try:
        if '--wait' in sys.argv:
            print('Waiting for historical collection to finish (up to two hours).',flush=True)
            deadline=time.monotonic()+7200
            while json.loads((OUT/'status.json').read_text())['state']=='running':
                if time.monotonic()>deadline:raise TimeoutError('Collection still running after two hours')
                time.sleep(30)
        main()
    finally:lock.rmdir()
