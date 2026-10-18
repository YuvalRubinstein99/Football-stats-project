"""Training-only seasonal feature-group selection for expanded regressions.

Nine prespecified interpretable feature sets, three forward validation seasons,
and the one-standard-error simplicity rule. The 2022-23 test is opened only
after the choices have been written to disk. No betting-profit optimization.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import sys,json,csv,hashlib,warnings,html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.lineup_deps'));sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize
from scipy.stats import logistic,norm
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LinearRegression,PoissonRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits
from lineup_model import clean
from compare_expanded_models import outcome_probabilities
from compare_ordered_models import objective,probability,metrics,bets,SEARCH_CANDIDATE

OUT=ROOT/'results/feature_selection'
BASE=ROOT/'results/expanded_comparison'
FAMILIES=['Linear','Poisson','Ordered_logit','Ordered_probit']
VALIDATION_YEARS=[2019,2020,2021]


def season_years(train):
    old=pd.read_csv(ROOT/'data/expanded_2008/added_row_provenance.csv')
    original=clean(pd.read_csv(ROOT/'data/train.csv')).replace([np.inf,-np.inf],np.nan).dropna()
    lookup={}
    with (ROOT/'results/stake_level_bet_ledger.csv').open(encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['split']=='Train (OOB)':lookup[int(r['source_row'])]=int(r['season'][:4])
    years=np.r_[old.season.str[:4].astype(int),[lookup[int(i)] for i in original.index]]
    assert len(years)==len(train) and years.min()==2008 and years.max()==2021
    assert np.array_equal(train.iloc[len(old):].home_team_name,original.home_team_name)
    assert np.allclose(train.iloc[len(old):].home_score,original.home_score)
    return years


def candidates(train):
    allcols=[c for c in train if c not in ['home_score','away_score']]
    odds=['B365H','B365D','B365A']
    fifa=['HomePlayer_Overall_mean','AwayPlayer_Overall_mean']
    form=[c for c in allcols if c.endswith(('_form','_prior'))]
    teams=['home_team_name','away_team_name']
    week=['Matchweek']
    return dict(full=allcols,
        no_teams=[c for c in allcols if c not in teams],
        compact_fifa_form_odds=fifa+form+odds+week,
        compact_fifa_odds=fifa+odds+week,
        form_odds=form+odds+week,
        odds_only=odds,
        fifa_form_no_odds=fifa+form+week,
        fifa_only=fifa,
        compact_with_teams=fifa+form+odds+week+teams)


def design(train,valid,cols):
    cats=[c for c in cols if c.endswith('_team_name')]
    nums=[c for c in cols if c not in cats]
    enc=OneHotEncoder(sparse_output=False,handle_unknown='ignore') if cats else None
    scaler=StandardScaler()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='Found unknown categories')
        xcat=enc.fit_transform(train[cats]) if cats else np.empty((len(train),0))
        vcat=enc.transform(valid[cats]) if cats else np.empty((len(valid),0))
    x=np.column_stack([xcat,scaler.fit_transform(train[nums])])
    xv=np.column_stack([vcat,scaler.transform(valid[nums])])
    center=x.mean(axis=0)
    _,singular,vh=np.linalg.svd(x-center,full_matrices=False)
    keep=singular>singular[0]*1e-10
    basis=vh[keep].T/(singular[keep]/np.sqrt(len(train)))
    artifact=dict(columns=cols,categorical=cats,numeric=nums,encoder=enc,scaler=scaler,center=center,basis=basis)
    return x,xv,(x-center)@basis,(xv-center)@basis,artifact


def fit(family,x,xo,goals,twice=False,sample_weight=None):
    if family in ['Linear','Poisson']:
        model=LinearRegression() if family=='Linear' else MultiOutputRegressor(PoissonRegressor(alpha=.01,max_iter=4000,tol=1e-7))
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            model.fit(x,goals,sample_weight=sample_weight)
        return dict(family=family,model=model),dict(converged=True)
    link=family.split('_')[1];dist=logistic if link=='logit' else norm
    y=np.sign(goals[:,0]-goals[:,1]).astype(int)+1
    cuts=dist.ppf(np.cumsum(np.bincount(y)/len(y))[:2])
    initial=np.r_[np.zeros(xo.shape[1]),cuts[0],np.log(cuts[1]-cuts[0])]
    starts=[initial]
    if twice:starts.append(initial+np.random.default_rng(914).normal(0,.02,len(initial)))
    results=[]
    for start in starts:
        result=minimize(objective,start,args=(xo,y,link,sample_weight),jac=True,method='L-BFGS-B',
            bounds=[(None,None)]*(len(initial)-1)+[(-12,8)],
            options=dict(maxiter=2000,ftol=1e-13,gtol=1e-7,maxls=50))
        assert result.success and np.max(abs(result.jac))<2e-5,(family,result.message,np.max(abs(result.jac)))
        results.append(result)
    if twice:assert abs(results[0].fun-results[1].fun)<1e-7
    result=min(results,key=lambda r:r.fun)
    return dict(family=family,theta=result.x,link=link),dict(converged=True,iterations=result.nit,gradient_max=float(np.max(abs(result.jac))))


def predict(model,x,xo):
    if model['family'] in ['Linear','Poisson']:
        goals=model['model'].predict(x)
        assert np.isfinite(goals).all()
        return outcome_probabilities(goals),goals
    return probability(model['theta'],xo,model['link']),None


def transform(artifact,frame):
    cats=artifact['categorical'];nums=artifact['numeric'];enc=artifact['encoder']
    xc=enc.transform(frame[cats]) if cats else np.empty((len(frame),0))
    x=np.column_stack([xc,artifact['scaler'].transform(frame[nums])])
    return x,(x-artifact['center'])@artifact['basis']


def report(m,s,selection,cv,paired,ledger):
    def table(frame,digits=5):return frame.to_html(index=False,float_format=lambda x:f'{x:,.{digits}f}')
    body='<h1>Feature selection: expanded regressions vs RF</h1><p>Same 19,263 training matches (2008–22) and 1,447 test matches (2022–23). FS = training-only feature selection.</p>'
    body+='<h2>Chosen features</h2>'+table(pd.DataFrame([dict(model=k,feature_set=v['candidate'],columns=', '.join(v['columns']),cv_log_loss=v['mean_log_loss'],selection_threshold=v['threshold']) for k,v in selection.items()]))
    body+='<p>Nine feature sets were declared before fitting. Each is validated on 2019–20, 2020–21 and 2021–22 using only earlier seasons. Choose the fewest encoded parameters within one standard error of the lowest mean validation log loss; ties favor fewer raw columns, then lower loss. The standard error is a simplicity heuristic from only three correlated seasons, not a confidence interval. Selection reduces complexity but cannot guarantee less overfitting or higher profit.</p>'
    body+='<h2>Test prediction quality</h2><p>Lower log loss, Brier and goal MSE is better; higher AUC and accuracy is better. Ordered models predict H/D/A directly and have no goal MSE.</p>'+table(m)
    body+='<h2>Fixed filtered Legacy variance</h2><p>Same odds ≥ 4 and expected return &gt; 2% filter, $1–$5 levels. Flat net is $1 on those exact selected bets; ROI = net / staked.</p>'+table(s[s.strategy==SEARCH_CANDIDATE],3)
    body+='<h2>Cumulative test profit</h2><p>Filtered Legacy variance, RF and the four selected models; no compounding.</p>'
    selected=ledger[(ledger.strategy==SEARCH_CANDIDATE)&(ledger.model.str.endswith('_FS')|(ledger.model=='expanded__RF_vector'))]
    weekly=selected.groupby(['week','model']).net.sum().unstack(fill_value=0).sort_index()
    weekly.to_csv(OUT/'weekly_profit.csv')
    curves={k:np.r_[0,weekly[k].cumsum()] for k in weekly}
    lo=min(0,min(v.min() for v in curves.values()))-20;hi=max(0,max(v.max() for v in curves.values()))+20
    yp=lambda v:310-(v-lo)/(hi-lo)*270
    svg='<svg viewBox="0 0 1100 450" role="img" aria-label="Cumulative test profit for selected models and random forest">'
    for v in np.linspace(lo,hi,5):svg+=f'<line x1="80" x2="1050" y1="{yp(v)}" y2="{yp(v)}" stroke="#ced6df"/><text x="5" y="{yp(v)}">${v:.0f}</text>'
    for j,(name,values) in enumerate(curves.items()):
        color=['#c56c22','#9360ba','#355eaa','#ca4963','#168568'][j]
        pts=' '.join(f'{80+i/len(weekly)*970:.2f},{yp(v):.2f}' for i,v in enumerate(values))
        svg+=f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5"><title>{name}: ${values[-1]:.2f}</title></polyline>'
        svg+=f'<text x="{30+(j%3)*350}" y="{380+(j//3)*30}" fill="{color}">{name.replace("expanded__","")}: ${values[-1]:.2f}</text>'
    body+=svg+'<text x="80" y="340">Start</text><text x="930" y="340">Matchweek 38</text></svg>'
    body+='<h2>All 15 strategies: test net dollars</h2>'+table(s.pivot(index='strategy',columns='model',values='net').reset_index(),2)
    body+='<details><summary>Full strategy accounting</summary>'+table(s,3)+'</details>'
    body+='<h2>Paired test log-loss differences</h2><p>Negative favors the first model. Approximate 95% intervals: 2,000 paired matchweek resamples, not accounting for all temporal/team dependence or earlier test inspection.</p>'+table(paired)
    body+='<details><summary>All training-only validation results</summary>'+table(cv)+'</details>'
    body+='<h2>Method</h2><p>OLS predicts both goal targets. Poisson uses two log-link regressions with fixed alpha=0.01, unchanged from the prior baseline. Both use independent-Poisson conversion to H/D/A probabilities. Ordered logit/probit fit away win &lt; draw &lt; home win with two cutpoints and shared slopes. Scaling, team encoding and removal of non-identifiable directions are fitted separately inside each training fold. The full-data ordered models retain all predictors except exact collinearity. Final feature choices are refit on all 19,263 training rows. No OOB claim for regressions; in-sample training profit is omitted. Betting rules and stake thresholds are unchanged and were not optimized here. Test results are exploratory because this season has been examined previously. Actual historical lineups and recorded odds; no fees, bankroll limits or compounding.</p>'
    body+='<p>References: <a href="https://scikit-learn.org/stable/modules/cross_validation.html">Training-only preprocessing and validation</a>; <a href="https://www.statsmodels.org/stable/examples/notebooks/generated/ordinal_regression.html">ordered logit/probit parameterization</a>.</p>'
    doc='<!doctype html><meta charset="utf-8"><title>Feature-selected models vs expanded RF</title><style>body{font:15px system-ui;margin:30px;color:#182532}table{border-collapse:collapse;font-size:12px}td,th{padding:8px;border-bottom:1px solid #ddd}svg{width:100%;max-width:1100px}details{margin:20px 0}</style>'+body
    (OUT/'report.html').write_text(doc,encoding='utf-8')
    path=BASE/'report.html'
    prior=path.read_text(encoding='utf-8').split('<!-- feature-selection -->')[0]
    path.write_text(prior+'<!-- feature-selection --><hr><section id="feature-selection">'+body+'</section>',encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=True)
    original_status=json.loads((BASE/'status.json').read_text())
    for name,digest in original_status['input_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    train=pd.read_csv(ROOT/'data/expanded_2008/train.csv');years=season_years(train)
    sets=candidates(train)
    plan=dict(candidates=sets,validation_years=VALIDATION_YEARS,score='mean seasonal log loss',rule='one-standard-error, then smallest average encoded dimension',input_hashes=original_status['input_hashes'])
    (OUT/'selection_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    records=[]
    for year in VALIDATION_YEARS:
        tr=train.loc[years<year];va=train.loc[years==year]
        assert len(tr)>0 and len(va)>0 and years[years<year].max()<year
        goals=tr[['home_score','away_score']].to_numpy()
        labels=(1-np.sign(va.home_score-va.away_score)).to_numpy(int)
        for candidate,cols in sets.items():
            x,xv,xo,xov,artifact=design(tr,va,cols)
            for family in FAMILIES:
                model,diag=fit(family,x,xo,goals)
                p,_=predict(model,xv,xov)
                loss=log_loss(labels,p,labels=[0,1,2])
                records.append(dict(family=family,candidate=candidate,validation_year=year,train_matches=len(tr),validation_matches=len(va),log_loss=loss,encoded_features=x.shape[1],rank=xo.shape[1],raw_columns=len(cols)))
                print(f'{year} {family} {candidate}: {loss:.6f}',flush=True)
            pd.DataFrame(records).to_csv(OUT/'validation_folds.csv',index=False)
    folds=pd.DataFrame(records)
    cv=folds.groupby(['family','candidate']).agg(mean_log_loss=('log_loss','mean'),std_log_loss=('log_loss','std'),encoded_features=('encoded_features','mean'),raw_columns=('raw_columns','first')).reset_index()
    cv['se']=cv.std_log_loss/np.sqrt(len(VALIDATION_YEARS))
    selection={}
    for family in FAMILIES:
        choices=cv[cv.family==family];best=choices.loc[choices.mean_log_loss.idxmin()]
        threshold=float(best.mean_log_loss+best.se)
        chosen=choices[choices.mean_log_loss<=threshold].sort_values(['encoded_features','raw_columns','mean_log_loss','candidate']).iloc[0]
        selection[family]=dict(candidate=chosen.candidate,columns=sets[chosen.candidate],mean_log_loss=float(chosen.mean_log_loss),threshold=threshold,best_candidate=best.candidate)
    # Freeze choices before opening test data or computing any test predictions.
    (OUT/'selected_features.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    cv.to_csv(OUT/'validation_summary.csv',index=False)
    print('FROZEN FEATURE CHOICES: '+json.dumps(selection),flush=True)
    test=pd.read_csv(ROOT/'data/expanded_2008/test.csv')
    labels=(1-np.sign(test.home_score-test.away_score)).to_numpy(int)
    goals=train[['home_score','away_score']].to_numpy()
    true_goals=test[['home_score','away_score']].to_numpy()
    probs={};rows=[];diagnostics={}
    for family,choice in selection.items():
        x,xt,xo,xot,artifact=design(train,test,choice['columns'])
        model,diag=fit(family,x,xo,goals,twice=True)
        name='expanded__'+family+'_FS'
        p,g=predict(model,xt,xot);probs[name]=p
        row=metrics(name,p,labels)
        if g is not None:row.update(goal_mse=float(np.mean((g-true_goals)**2)),goal_mae=float(np.mean(abs(g-true_goals))))
        rows.append(row);diagnostics[name]=diag
        artifact['fit']=model
        joblib.dump(artifact,OUT/(name+'.joblib'))
        restored=joblib.load(OUT/(name+'.joblib'));rt,rto=transform(restored,test)
        np.testing.assert_allclose(predict(restored['fit'],rt,rto)[0],p,atol=1e-12)
    baseline=pd.read_csv(BASE/'match_predictions.csv')
    ordered=pd.read_csv(ROOT/'results/ordered_comparison/match_predictions.csv')
    baseline_metrics=pd.read_csv(BASE/'model_metrics.csv')
    for data,names in [(baseline,['RF_vector','Linear_vector','Poisson_alpha001']),(ordered,['Ordered_logit','Ordered_probit'])]:
        for c in ['home_team_name','away_team_name','home_score','away_score','B365H','B365D','B365A']:assert np.array_equal(data[c],test[c]),c
        for family in names:
            name='expanded__'+family;p=data[[name+'__p_'+side for side in ['home','draw','away']]].to_numpy();probs[name]=p
            row=metrics(name,p,labels)
            old=baseline_metrics[baseline_metrics.model==name]
            if len(old):row.update(goal_mse=float(old.iloc[0].goal_mse),goal_mae=float(old.iloc[0].goal_mae))
            rows.append(row)
    market=1/test[['B365H','B365D','B365A']].to_numpy();market/=market.sum(axis=1,keepdims=True)
    rows.append(metrics('Market',market,labels));m=pd.DataFrame(rows)
    strategy=[];ledger=[]
    for name,p in probs.items():
        assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(axis=1),1)
        a,b=bets(name,p,test,labels);strategy.extend(a);ledger.extend(b)
    s=pd.DataFrame(strategy);ledger=pd.DataFrame(ledger)
    previous=pd.concat([pd.read_csv(BASE/'strategy_results.csv'),pd.read_csv(ROOT/'results/ordered_comparison/strategy_results.csv')]).drop_duplicates(['model','strategy'])
    for name in probs:
        if name.endswith('_FS'):continue
        joined=s[s.model==name].merge(previous[previous.model==name],on='strategy',suffixes=('_new','_old'))
        assert len(joined)==15
        for col in ['net','staked','bets','flat_net']:np.testing.assert_allclose(joined[col+'_new'],joined[col+'_old'],atol=1e-7)
    weeks=test.Matchweek.to_numpy(int);unique=np.unique(weeks);rng=np.random.default_rng(20261013)
    samples=rng.integers(0,len(unique),size=(2000,len(unique)));counts=np.array([(weeks==w).sum() for w in unique])
    full_names=dict(Linear='Linear_vector',Poisson='Poisson_alpha001',Ordered_logit='Ordered_logit',Ordered_probit='Ordered_probit')
    paired=[]
    for family in FAMILIES:
        name='expanded__'+family+'_FS'
        for reference in ['expanded__RF_vector','expanded__'+full_names[family]]:
            loss=lambda key:-np.log(np.maximum(probs[key][np.arange(len(test)),labels],1e-15))
            delta=loss(name)-loss(reference);sums=np.array([delta[weeks==w].sum() for w in unique])
            draws=sums[samples].sum(axis=1)/counts[samples].sum(axis=1)
            paired.append(dict(model=name,reference=reference,difference=delta.mean(),ci_low=np.quantile(draws,.025),ci_high=np.quantile(draws,.975)))
    paired=pd.DataFrame(paired)
    pred=test[['home_team_name','away_team_name','home_score','away_score','B365H','B365D','B365A']].copy()
    for name,p in probs.items():
        for j,side in enumerate(['home','draw','away']):pred[name+'__p_'+side]=p[:,j]
    pred.to_csv(OUT/'match_predictions.csv',index=False)
    restored=pd.read_csv(OUT/'match_predictions.csv')
    for name,p in probs.items():np.testing.assert_allclose(restored[[name+'__p_'+side for side in ['home','draw','away']]],p,atol=1e-14)
    m.to_csv(OUT/'model_metrics.csv',index=False);s.to_csv(OUT/'strategy_results.csv',index=False)
    ledger.to_csv(OUT/'test_bet_ledger.csv',index=False);paired.to_csv(OUT/'paired_differences.csv',index=False)
    (OUT/'status.json').write_text(json.dumps(dict(state='completed',fits=diagnostics,checks=['forward season splits','selection frozen before test load','convergence','serialized inference parity','baseline strategy reconciliation','probability and export parity']),indent=2),encoding='utf-8')
    report(m,s,selection,cv,paired,ledger)
    print(m.to_string(index=False));print(s[s.strategy==SEARCH_CANDIDATE].to_string(index=False));print(paired.to_string(index=False))


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
