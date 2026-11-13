"""Compact binary models and chronological calibration for the overfit audit."""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from home_win_model import target

ODDS=['market_logit','market_overround']
CORE=ODDS+['overall_gap','gd_form_gap','points_form_gap','matchweek']
POSITIONS=CORE+['home_attack_defense','away_attack_defense','mid_passing_gap',
                'mid_defending_gap','keeper_gap','minimum_position_coverage']


def market_probability(frame):
    prices=frame[['B365H','B365D','B365A']].to_numpy(float)
    if not np.isfinite(prices).all() or (prices<=1).any():
        raise ValueError('Valid decimal odds required')
    inverse=1/prices
    return inverse[:,0]/inverse.sum(axis=1)


def features(frame,columns):
    ph=market_probability(frame)
    result=pd.DataFrame({'market_logit':np.log(ph/(1-ph)),
                        'market_overround':(1/frame[['B365H','B365D','B365A']]).sum(axis=1).to_numpy()-1},index=frame.index)
    if any(c in columns for c in CORE[2:]):
        result['overall_gap']=frame.HomePlayer_Overall_mean-frame.AwayPlayer_Overall_mean
        result['gd_form_gap']=frame.home_GD_form-frame.away_GD_form
        result['points_form_gap']=frame.home_Points_form-frame.away_Points_form
        result['matchweek']=frame.Matchweek
    if any(c in columns for c in POSITIONS[len(CORE):]):
        result['home_attack_defense']=frame.home_attack_shooting_mean-frame.away_defense_defending_mean
        result['away_attack_defense']=frame.away_attack_shooting_mean-frame.home_defense_defending_mean
        result['mid_passing_gap']=frame.home_midfield_passing_mean-frame.away_midfield_passing_mean
        result['mid_defending_gap']=frame.home_midfield_defending_mean-frame.away_midfield_defending_mean
        result['keeper_gap']=frame.home_goalkeeper_overall_mean-frame.away_goalkeeper_overall_mean
        result['minimum_position_coverage']=frame[['home_position_matched_count','away_position_matched_count']].min(axis=1)
    if not set(columns)<=set(result):
        raise ValueError('Unknown feature column')
    return result[columns]


def configs():
    grid={'market':dict(family='market',columns=[]),
          'logit_odds':dict(family='logistic',columns=ODDS,C=.1)}
    for name,columns in [('core',CORE),('positions',POSITIONS)]:
        for c in [.01,.1]:
            grid[f'logit_{name}_C{c}']=dict(family='logistic',columns=columns,C=c)
        for depth,leaf in [(2,100),(3,50)]:
            grid[f'rf_{name}_d{depth}_leaf{leaf}']=dict(family='forest',columns=columns,
                                                     max_depth=depth,min_samples_leaf=leaf)
    return grid


def fit_model(frame,config):
    if config['family']=='market':
        return dict(config=config,pipeline=None)
    steps=[('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True))]
    if config['family']=='logistic':
        steps += [('scaler',StandardScaler()),('model',LogisticRegression(C=config['C'],max_iter=4000,tol=1e-7))]
    else:
        steps += [('model',RandomForestClassifier(n_estimators=500,criterion='log_loss',
            max_depth=config['max_depth'],min_samples_leaf=config['min_samples_leaf'],
            max_features=.7,max_samples=.5,n_jobs=4,random_state=20261020))]
    pipeline=Pipeline(steps).fit(features(frame,config['columns']),target(frame))
    return dict(config=config,pipeline=pipeline)


def predict_raw(bundle,frame):
    if bundle['pipeline'] is None:return market_probability(frame)
    return bundle['pipeline'].predict_proba(features(frame,bundle['config']['columns']))[:,1]


def logits(probability):
    p=np.asarray(probability,float)
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid probabilities')
    p=np.clip(p,1e-6,1-1e-6)
    return np.log(p/(1-p)).reshape(-1,1)


def fit_calibrator(past,cutoff):
    if past.empty or pd.to_datetime(past.fixture_date).max()>=pd.Timestamp(cutoff):
        raise ValueError('Calibration must use strictly earlier predictions')
    if not (pd.to_datetime(past.trained_through)<pd.to_datetime(past.fixture_date)).all():
        raise ValueError('Calibration inputs must be out-of-sample predictions')
    return LogisticRegression(C=1.,max_iter=2000,tol=1e-8).fit(logits(past.raw_probability),past.target)


def apply_calibrator(calibrator,p):
    return np.asarray(p,float) if calibrator is None else calibrator.predict_proba(logits(p))[:,1]


def predict_bundle(bundle,frame):
    return apply_calibrator(bundle.get('calibrator'),predict_raw(bundle,frame))


def select_model(validation,year):
    past=validation[validation.season.isin([year-2,year-1])]
    if past.empty or past.groupby('model').season.nunique().min()!=2:
        raise ValueError('Two earlier complete validation seasons required')
    means=past.groupby('model').log_loss.mean()
    return str(means.sort_values(kind='stable').index[0]),float(means.min())


def choose_threshold(predictions,year):
    past=predictions[predictions.season.isin([year-2,year-1])]
    if set(past.season)!={year-2,year-1}:raise ValueError('Two threshold-selection seasons required')
    records=[]
    for threshold in [0.,.02,.05,.10]:
        active=past.p_home_win*past.B365H-1>threshold
        profit=np.where(active,np.where(past.target==1,past.B365H-1,-1.),0.)
        by_season=pd.Series(profit,index=past.index).groupby(past.season).sum()
        eligible=int(active.sum())>=100 and bool((by_season>0).all())
        records.append(dict(threshold=threshold,bets=int(active.sum()),net=float(profit.sum()),
                            positive_both_seasons=bool((by_season>0).all()),eligible=eligible))
    eligible=[r for r in records if r['eligible']]
    best=max(eligible,key=lambda r:(r['net'],-r['threshold'])) if eligible else None
    return (best['threshold'] if best else None),records
