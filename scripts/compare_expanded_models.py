"""Versioned older-season expansion and matched-test goal-model comparison."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.lineup_deps'));sys.path.insert(0,str(ROOT/'src'))
import ast
import html
import json
import hashlib
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, PoissonRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import skellam, poisson
from lineup_model import clean
from lineup_sources import team_key
from live_bets import integrated_recommendations
from stake_levels import size_bet
from scrape_older_seasons import atomic_json, stamp

OUT=ROOT/'results/expanded_comparison'
DATA=ROOT/'data/expanded_2008'
SEED=20261007

# Reviewed archive spellings absent from the existing live alias table.
ALIASES={'Athletic Club de Bilbao':'Athletic Club','FC Bayern Munich':'Bayern Munich',
 '1. FC Nürnberg':'Nürnberg','DSC Arminia Bielefeld':'Arminia','Chievo Verona':'Chievo',
 'Fortuna Düsseldorf':'Düsseldorf','Granada CF':'Granada','Hertha BSC Berlin':'Hertha BSC',
 'RC Deportivo de La Coruña':'La Coruña','RCD Mallorca':'Mallorca','Real Betis Balompié':'Betis',
 'Real Madrid CF':'Real Madrid','Real Sporting de Gijón':'Sporting Gijón','Real Valladolid':'Valladolid',
 'SpVgg Greuther Fürth':'Greuther Fürth','VfL Bochum':'Bochum','VfL Wolfsburg':'Wolfsburg',
 'UD Almería':'Almería'}

def prepare():
    DATA.mkdir(parents=True,exist_ok=True)
    base=clean(pd.read_csv(ROOT/'data/train.csv')).replace([np.inf,-np.inf],np.nan).dropna()
    test=clean(pd.read_csv(ROOT/'data/test.csv')).replace([np.inf,-np.inf],np.nan).dropna()
    old=pd.read_csv(ROOT/'data/older_seasons/lineup_features.csv')
    # Recheck complete seasons directly, ignoring empty trailing CSV records.
    from scrape_older_seasons import csv_rows,date_value
    coverage=pd.read_csv(ROOT/'data/older_seasons/lineup_coverage.csv')
    checks=[]
    for r in coverage.itertuples():
        code=r.season[2:4]+r.season[7:9]
        reference=csv_rows(ROOT/f'data/older_seasons/raw/{r.league}_{code}.csv')
        keys=set()
        for q in reference:
            int(q['FTHG']);int(q['FTAG'])
            keys.add((date_value(q['Date']),q['HomeTeam'],q['AwayTeam']))
        checks.append('pass' if len(keys)==len(reference)==r.matches else 'excluded')
    coverage['season_count_check']=checks
    coverage.to_csv(DATA/'season_audit.csv',index=False)
    good={(r.league,r.season) for r in coverage.itertuples() if r.season_count_check=='pass' and r.league!='F1'}
    old=old[[ (r.league,r.season) in good for r in old.itertuples()]].copy()
    # Verify rating transforms against the user's actual notebook functions.
    nb=json.loads((ROOT/'notebooks/legacy/second_processing_pipeline.ipynb').read_text(encoding='utf-8'))
    source='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    tree=ast.parse(source)
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('third_largest','third_smallest','get_data_cols')]
    scope={'np':np,'pd':pd};exec(compile(ast.Module(body=funcs,type_ignores=[]),'<notebook feature functions>','exec'),scope)
    audit=pd.read_csv(ROOT/'data/older_seasons/player_rating_audit.csv')
    audit=audit[audit.match_id.isin(old.match_id)]
    assert (audit.rating_date<audit.date).all()
    assert (audit.groupby('match_id').size()==22).all()
    rebuilt=pd.DataFrame(index=old.match_id)
    for side,prefix in [('home','HomePlayer'),('away','AwayPlayer')]:
        wide=audit[audit.side==side].pivot(index='match_id',columns='position',values='rating').reindex(old.match_id)
        for i in range(1,12):rebuilt[f'{prefix}{i}_Overall']=wide[i]
        scope['get_data_cols'](rebuilt,'Overall',prefix)
    for col in base:
        if col.startswith(('HomePlayer_','AwayPlayer_')):
            assert np.allclose(old[col],rebuilt[col],atol=1e-10),col
            old[col]=rebuilt[col].to_numpy()
    known={team_key(n):n for n in set(base.home_team_name)|set(base.away_team_name)}
    mapping=[]
    for name in sorted(set(old.home_team_name)|set(old.away_team_name)):
        resolved=ALIASES.get(name,name)
        # Archive contains a soft-hyphen spelling of Almeria.
        if name.startswith('UD Almer'):resolved='Almería'
        resolved=known.get(team_key(resolved),resolved)
        mapping.append(dict(source=name,canonical=resolved,seen_in_original=resolved in known.values()))
    names={r['source']:r['canonical'] for r in mapping}
    old['home_team_name']=old.home_team_name.map(names);old['away_team_name']=old.away_team_name.map(names)
    old['Home_min_max']=old.HomePlayer_Overall_min*old.HomePlayer_Overall_max
    old['Away_min_max']=old.AwayPlayer_Overall_min*old.AwayPlayer_Overall_max
    added=old[list(base.columns)].copy()
    assert not added.isna().any().any()
    assert old.season.str[:4].astype(int).between(2008,2013).all()
    assert not old.duplicated(['league','date','home_team_name','away_team_name']).any()
    expanded=pd.concat([added,base],ignore_index=True)
    expanded.to_csv(DATA/'train.csv',index=False);test.to_csv(DATA/'test.csv',index=False)
    old[['match_id','season','date','league','home_team_name','away_team_name','latest_rating_date']].to_csv(DATA/'added_row_provenance.csv',index=False)
    pd.DataFrame(mapping).to_csv(DATA/'team_aliases.csv',index=False)
    summary=dict(added_matches=len(added),original_train=len(base),expanded_train=len(expanded),test_matches=len(test),added_by_season=old.groupby('season').size().to_dict(),notebook_rating_features_verified=True,excluded_leagues=['F1'],excluded_incomplete_seasons=coverage[(coverage.league!='F1')&(coverage.season_count_check!='pass')][['league','season']].to_dict('records'))
    atomic_json(DATA/'manifest.json',summary)
    return base,expanded,test,summary

def outcome_probabilities(means):
    safe=np.maximum(means,1e-8)
    p=np.column_stack([skellam.sf(0,safe[:,0],safe[:,1]),skellam.pmf(0,safe[:,0],safe[:,1]),skellam.cdf(-1,safe[:,0],safe[:,1])])
    if not np.isfinite(p).all():raise ValueError('Invalid outcome probabilities')
    return p/p.sum(axis=1,keepdims=True)

def report(metrics,strategies,manifest):
    m=pd.DataFrame(metrics);s=pd.DataFrame(strategies)
    m.to_csv(OUT/'model_metrics.csv',index=False);s.to_csv(OUT/'strategy_results.csv',index=False)
    notes='''Same 2022–23 test matches for every model; this test has already been examined, so comparisons are exploratory.
Original training: 2014–22. Expanded training: eligible 2008–14 matches plus original training.
Four original leagues only. Incomplete source league-seasons excluded. FIFA summaries are checked against notebook functions.
Earlier ratings use strictly pre-match snapshots; older and existing rating sources have different collection conventions.
Earlier form uses prior calendar dates and last five matches, resetting each season; original processed rows are preserved.
All models use the same predictors including Bet365 odds, with training-only one-hot encoding and scaling.
Vector OLS fits both goal targets but is mathematically equivalent to separate OLS equations; it does not model goal dependence.
Poisson regression fits two separate log-link goal equations; no shared-goal latent component is assumed.
Vector RF uses 2,000 trees; separate Poisson-loss RF uses 200 trees per target, matching existing family settings.
OLS/Ridge raw predictions are used for goal errors; negative predictions are floored only for Poisson probability conversion.
Ridge alpha=1 and Poisson alpha=1 are fixed controls, not test-tuned. A Poisson alpha=0.01 sensitivity is also shown.
Outcome probabilities for all goal models use independent Poisson conversion; this is distinct from Poisson regression training.
Fixed strategies and $1–$5 stakes, no compounding or bankroll limits. No live-model promotion. No claim that added data must improve results.'''
    body='<h1>Adding 2008–14: model comparison</h1><p>'+html.escape(json.dumps(manifest['data']))+'</p>'
    body+='<h2>Identical test matches</h2>'+m.to_html(index=False,float_format=lambda x:f'{x:.5f}')
    body+='<h2>Fixed Legacy variance filter</h2>'+s[s.strategy.str.contains('filtered search candidate')].to_html(index=False,float_format=lambda x:f'{x:.2f}')
    body+='<h2>All fixed strategies</h2>'+s.to_html(index=False,float_format=lambda x:f'{x:.3f}')
    body+='<h2>Method and limitations</h2><pre>'+html.escape(notes)+'</pre>'
    (OUT/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Expanded FIFA model comparison</title><style>body{font:15px system-ui;margin:30px;color:#182532}table{border-collapse:collapse;font-size:12px}td,th{padding:8px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap;line-height:1.6}</style>'+body,encoding='utf-8')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    base,expanded,test,counts=prepare()
    ytest=test[['home_score','away_score']].to_numpy(float)
    labels=np.where(ytest[:,0]>ytest[:,1],0,np.where(ytest[:,0]<ytest[:,1],2,1))
    odds=test[['B365H','B365D','B365A']].to_numpy(float)
    metrics=[];strategies=[];preds=test[['home_team_name','away_team_name','home_score','away_score','B365H','B365D','B365A']].copy()
    manifest=dict(state='running',started_at=stamp(),data=counts,completed=[],input_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'data/train.csv',ROOT/'data/test.csv',DATA/'train.csv']})
    atomic_json(OUT/'status.json',manifest)
    cats=['home_team_name','away_team_name'];nums=[c for c in base if c not in cats+['home_score','away_score']]
    for period,frame in [('original',base),('expanded',expanded)]:
        enc=OneHotEncoder(sparse_output=False,handle_unknown='ignore');scale=StandardScaler()
        train_cat=enc.fit_transform(frame[cats]);test_cat=enc.transform(test[cats])
        x=np.column_stack([train_cat,scale.fit_transform(frame[nums])])
        xt=np.column_stack([test_cat,scale.transform(test[nums])]);y=frame[['home_score','away_score']].to_numpy(float)
        xr=np.column_stack([train_cat,frame[nums].to_numpy(float)]);xtr=np.column_stack([test_cat,test[nums].to_numpy(float)])
        models={
          'RF_vector':RandomForestRegressor(n_estimators=2000,max_depth=5,max_features=.3,max_samples=.5,random_state=SEED,n_jobs=4,oob_score=True),
          'RF_separate':None,
          'Linear_vector':LinearRegression(),
          'Ridge_vector':Ridge(alpha=1),
          'Poisson_alpha1':MultiOutputRegressor(PoissonRegressor(alpha=1,max_iter=3000,tol=1e-7)),
          'Poisson_alpha001':MultiOutputRegressor(PoissonRegressor(alpha=.01,max_iter=3000,tol=1e-7))}
        for family,model in models.items():
            name=period+'__'+family;print('Fitting '+name,flush=True)
            with warnings.catch_warnings(record=True) as messages:
                warnings.simplefilter('always')
                if family=='RF_separate':
                    model=[RandomForestRegressor(n_estimators=200,criterion='poisson',max_depth=5,max_features=.3,max_samples=.5 if j==0 else .25,random_state=SEED+j,n_jobs=4).fit(xr,y[:,j]) for j in range(2)]
                else:model.fit(xr if family.startswith('RF') else x,y)
            if any(issubclass(w.category,ConvergenceWarning) for w in messages):raise RuntimeError(name+' did not converge')
            means=np.column_stack([m.predict(xtr) for m in model]) if family=='RF_separate' else model.predict(xtr if family.startswith('RF') else xt)
            if not np.isfinite(means).all():raise ValueError(name+' has nonfinite goals')
            p=outcome_probabilities(means)
            row=dict(model=name,train_matches=len(frame),test_matches=len(test),goal_mse=float(np.mean((means-ytest)**2)),goal_mae=float(np.mean(abs(means-ytest))),log_loss=log_loss(labels,p,labels=[0,1,2]),auc_macro=roc_auc_score(labels,p,multi_class='ovr',average='macro'),brier=float(np.mean(np.sum((p-np.eye(3)[labels])**2,axis=1))),accuracy=float(np.mean(p.argmax(axis=1)==labels)),negative_goal_predictions=int((means<0).sum()),score_nll=float(-poisson.logpmf(ytest,np.maximum(means,1e-8)).sum(axis=1).mean()))
            metrics.append(row)
            totals={}
            for i,(_,r) in enumerate(test.iterrows()):
                match=dict(home=r.home_team_name,away=r.away_team_name,p_home=p[i,0],p_draw=p[i,1],p_away=p[i,2],B365H=odds[i,0],B365D=odds[i,1],B365A=odds[i,2])
                for bet in integrated_recommendations(match):
                    stake=size_bet(bet)['stake_dollars'];ret=bet[('home','draw','away')[labels[i]]+'_stake']*odds[i,labels[i]]
                    a=totals.setdefault(bet['strategy'],dict(bets=0,staked=0.,net=0.,flat_net=0.))
                    if stake:a['bets']+=1;a['staked']+=stake;a['net']+=stake*(ret-1);a['flat_net']+=ret-1
            for strategy,a in totals.items():strategies.append(dict(model=name,strategy=strategy,**a,roi=a['net']/a['staked'] if a['staked'] else 0))
            for j,side in enumerate(('home','draw','away')):preds[name+'__p_'+side]=p[:,j]
            for j,side in enumerate(('home','away')):preds[name+'__goals_'+side]=means[:,j]
            joblib.dump(dict(model=model,encoder=enc,scaler=None if family.startswith('RF') else scale,numeric=nums,categorical=cats),OUT/(name+'.joblib'))
            manifest['completed'].append(name);atomic_json(OUT/'status.json',manifest)
            preds.to_csv(OUT/'match_predictions.csv',index=False);report(metrics,strategies,manifest)
            print(json.dumps(row),flush=True)
    market=1/odds;market/=market.sum(axis=1,keepdims=True)
    metrics.append(dict(model='Market',test_matches=len(test),log_loss=log_loss(labels,market,labels=[0,1,2]),auc_macro=roc_auc_score(labels,market,multi_class='ovr',average='macro'),brier=float(np.mean(np.sum((market-np.eye(3)[labels])**2,axis=1))),accuracy=float(np.mean(market.argmax(axis=1)==labels))))
    manifest.update(state='completed',finished_at=stamp());atomic_json(OUT/'status.json',manifest);report(metrics,strategies,manifest)

if __name__=='__main__':main()
