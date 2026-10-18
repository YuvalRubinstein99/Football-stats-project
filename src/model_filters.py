"""Identical manual odds/edge/outcome/stake gates for live and historical selections."""
import json
from functools import lru_cache
from collections import defaultdict
from pathlib import Path
import numpy as np
from manual_strategies import settings,gate
from live_bets import integrated_recommendations,SEARCH_CANDIDATE
LEAGUES={'E0':'Premier League','SP1':'La Liga','D1':'Bundesliga','I1':'Serie A'}

def config(strategy,raw=None):
    default=dict(min_edge=.02 if strategy==SEARCH_CANDIDATE else 0,min_odds=4 if strategy==SEARCH_CANDIDATE else 1.000001)
    league=(raw or {}).get('league','all')
    if league not in ['all',*LEAGUES]:raise ValueError('Unknown league')
    return {**settings({**default,**(raw or {}),'strategy':'Legacy variance score' if strategy==SEARCH_CANDIDATE else strategy}),'league':league}

def apply(bet,cfg):
    outcome='combined' if bet['synthetic'] else next((s for s in ['home','draw','away'] if bet[s+'_stake']>0),'none')
    level,reason=gate(bet['decimal_odds'],bet['expected_net_per_unit'],outcome,cfg)
    amount=cfg['amounts'][level-1] if level else 0
    return {**bet,'stake_level':level,'stake_dollars':amount,'staking_action':'Unavailable' if bet['action']=='Unavailable' else 'Bet' if level else 'Skip','staking_reason':reason,
        **{s+'_dollars':amount*bet[s+'_stake'] for s in ['home','draw','away']}}

@lru_cache(maxsize=3)
def ledger(name):
    from model_strategy_app import ROOT,HISTORY,BASE,RECENCY,CLASSIFIER,model,model_path,pd,joblib,transform,predict
    from select_expanded_features import season_years
    from compare_expanded_models import outcome_probabilities
    from probability_calibration import SUFFIX,calibrate,artifacts
    base=name.removesuffix(SUFFIX);calibrated=name.endswith(SUFFIX)
    recent=base.endswith('__recent')
    classifier=base.startswith('RF classifier')
    data=CLASSIFIER if classifier else ROOT/'data/expanded_2008'
    train=pd.read_csv(data/'train.csv');test=pd.read_csv(data/'test.csv')
    cache=ROOT/'results/model_strategy/filter_cache';cache.mkdir(parents=True,exist_ok=True)
    # Version cache by exact model and data contents, not a UI model label.
    import hashlib
    path=model_path(base)
    calibration_key=json.dumps(artifacts(recent,classifier)['models'][base],sort_keys=True).encode() if calibrated else b''
    digest=hashlib.sha256(path.read_bytes()+(data/'train.csv').read_bytes()+(data/'test.csv').read_bytes()+calibration_key+(b'recent-v1' if recent else b'')).hexdigest()
    target=cache/(digest+'.joblib')
    if target.exists():return joblib.load(target)
    years=train.season.to_numpy() if classifier else season_years(train);fitted=model(name)
    if classifier:
        saved_train=pd.read_csv(CLASSIFIER/'train_predictions.csv');p=saved_train[[base+'__p_'+s for s in ['home','draw','away']]].to_numpy()
    elif base.startswith('Expanded RF'):
        x=np.column_stack([fitted.encoder.transform(train[fitted.categorical]),train[fitted.numeric].to_numpy()]);p=outcome_probabilities(fitted.forest.predict(x))
    else:
        x,xo=transform(fitted,train);p,_=predict(fitted['fit'],x,xo)
    saved=pd.read_csv((CLASSIFIER if classifier else RECENCY if recent else HISTORY)/'match_predictions.csv')
    pt=saved[[base+'__p_'+s for s in ['home','draw','away']]].to_numpy()
    if calibrated:p=calibrate(name,p);pt=calibrate(name,pt)
    sets=[('Train (in-sample)',train,p,years),('Test',test,pt,np.full(len(test),2022))]
    # Calibrator saw outcomes from later training seasons, so do not label its
    # transformed OOB predictions as independently held-out.
    if base=='Expanded RF' and not calibrated:sets.append(('Train (OOB)',train,outcome_probabilities(fitted.forest.oob_prediction_),years))
    result={}
    for split,frame,prob,season in sets:
        store=defaultdict(list)
        for i,r in frame.iterrows():
            match=dict(home=r.home_team_name,away=r.away_team_name,p_home=prob[i,0],p_draw=prob[i,1],p_away=prob[i,2],B365H=r.B365H,B365D=r.B365D,B365A=r.B365A)
            winner='home' if r.home_score>r.away_score else 'away' if r.home_score<r.away_score else 'draw'
            odds=r[{'home':'B365H','draw':'B365D','away':'B365A'}[winner]]
            for bet in integrated_recommendations(match):
                if bet['strategy']==SEARCH_CANDIDATE:continue
                outcome='combined' if bet['synthetic'] else next((s for s in ['home','draw','away'] if bet[s+'_stake']>0),'none')
                store[bet['strategy']].append((int(season[i]),int(r.Matchweek),bet['decimal_odds'],bet['expected_net_per_unit'],outcome,float(bet[winner+'_stake']*odds)))
        result[split]=dict(store)
    joblib.dump(result,target,compress=3)
    return result

@lru_cache(maxsize=1)
def league_rows():
    from model_strategy_app import ROOT,pd
    from select_expanded_features import clean
    from historical_leagues import row_leagues
    mapping=row_leagues();codes={v:k for k,v in LEAGUES.items()}
    old=pd.read_csv(ROOT/'data/expanded_2008/added_row_provenance.csv')
    original=clean(pd.read_csv(ROOT/'data/train.csv')).replace([np.inf,-np.inf],np.nan).dropna()
    test=clean(pd.read_csv(ROOT/'data/test.csv')).replace([np.inf,-np.inf],np.nan).dropna()
    train_codes=list(old.league)+[codes[mapping['Train (OOB)',int(i)]] for i in original.index]
    test_codes=[codes[mapping['Test',int(i)]] for i in test.index]
    for file,expected in [('train',train_codes),('test',test_codes)]:
        frame=pd.read_csv(ROOT/f'data/expanded_2008/{file}.csv')
        if len(frame)!=len(expected) or not set(expected)<=set(LEAGUES):raise ValueError('Historical league alignment failed')
        source=pd.concat([old,original],ignore_index=True) if file=='train' else test.reset_index(drop=True)
        for column in ['home_team_name','away_team_name']:
            if not np.array_equal(frame[column],source[column]):raise ValueError('Historical league fixture alignment failed')
    return {'Train (in-sample)':train_codes,'Train (OOB)':train_codes,'Test':test_codes}


def historical(name,cfg,by_league=False):
    curves=[];summary=[];league_curves=[];league_summary=[]
    for split,strategies in ledger(name).items():
        rows=strategies[cfg['strategy']];codes=league_rows()[split]
        if len(rows)!=len(codes):raise ValueError('Historical league row count mismatch')
        selected=[r for r,code in zip(rows,codes) if cfg['league']=='all' or code==cfg['league']]
        c,s=aggregate(name,cfg,split,selected);curves.append(c);summary.append(s)
        for code,label in LEAGUES.items():
            if cfg['league'] not in ['all',code]:continue
            subset=[r for r,league in zip(rows,codes) if league==code]
            c,s=aggregate(name,cfg,split,subset)
            league_curves.append({**c,'league':code,'league_name':label})
            league_summary.append({**s,'league':code,'league_name':label})
    return (curves,summary,league_curves,league_summary) if by_league else (curves,summary)


def aggregate(name,cfg,split,rows):
    weekly=defaultdict(lambda:[0.,0.]);bets=0;staked=0.;net=0.;flat=0.
    for season,week,odds,edge,outcome,ret in rows:
        level,_=gate(odds,edge,outcome,cfg);amount=cfg['amounts'][level-1] if level else 0
        profit=amount*(ret-1);weekly[season,week][0]+=profit;weekly[season,week][1]+=amount
        bets+=int(level>0);staked+=amount;net+=profit;flat+=(ret-1) if level else 0
    points=[dict(label='Start',net=0.,staked=0.)];total=0.;spent=0.
    for (season,week),(profit,amount) in sorted(weekly.items()):
        total+=profit;spent+=amount;points.append(dict(label=f'{season}–{season+1} W{week}',net=total,staked=spent))
    return (dict(split=split,points=points,model=name,strategy=cfg['strategy']),
        dict(split=split,matches=len(rows),bets=bets,staked=staked,net=net,flat_net=flat,roi=net/staked if staked else 0))
