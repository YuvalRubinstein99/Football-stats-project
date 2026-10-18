"""Matched team-identity ablation, retaining FIFA, all form/prior fields and odds."""
import json,html,hashlib
from select_expanded_features import (
    ROOT,BASE,np,pd,joblib,threadpool_limits,candidates,design,fit,predict,
    transform,metrics,bets,FAMILIES,SEARCH_CANDIDATE)

OUT=ROOT/'results/team_identity'


def main():
    OUT.mkdir(exist_ok=True)
    plan=json.loads((ROOT/'results/feature_selection/selection_plan.json').read_text())
    for path,digest in plan['input_hashes'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    train=pd.read_csv(ROOT/'data/expanded_2008/train.csv')
    test=pd.read_csv(ROOT/'data/expanded_2008/test.csv')
    cv=pd.read_csv(ROOT/'results/feature_selection/validation_summary.csv')
    folds=pd.read_csv(ROOT/'results/feature_selection/validation_folds.csv')
    sets=candidates(train)
    names=['compact_fifa_form_odds','compact_with_teams']
    assert set(sets[names[1]])-set(sets[names[0]])=={'home_team_name','away_team_name'}
    for n in names:assert sets[n]==plan['candidates'][n]
    goals=train[['home_score','away_score']].to_numpy()
    actual=test[['home_score','away_score']].to_numpy()
    labels=(1-np.sign(test.home_score-test.away_score)).to_numpy(int)
    probs={};rows=[];diagnostics={};strategies=[];ledger=[]
    for feature_set in names:
        x,xt,xo,xot,artifact=design(train,test,sets[feature_set])
        for family in FAMILIES:
            name=family+('__teams' if feature_set==names[1] else '__no_teams')
            print('Fitting '+name,flush=True)
            model,diag=fit(family,x,xo,goals,twice=True)
            p,g=predict(model,xt,xot);probs[name]=p
            row=metrics(name,p,labels)
            row.update(family=family,team_identity=feature_set==names[1],raw_features=len(sets[feature_set]),encoded_features=x.shape[1],rank=xo.shape[1],cv_log_loss=float(cv[(cv.family==family)&(cv.candidate==feature_set)].iloc[0].mean_log_loss))
            if g is not None:row['goal_mse']=float(np.mean((g-actual)**2))
            rows.append(row);diagnostics[name]=diag
            bundle=dict(**artifact,fit=model)
            joblib.dump(bundle,OUT/(name+'.joblib'))
            restored=joblib.load(OUT/(name+'.joblib'));a,b=transform(restored,test)
            np.testing.assert_allclose(predict(restored['fit'],a,b)[0],p,atol=1e-12)
            print(json.dumps(row),flush=True)
    saved=pd.read_csv(BASE/'match_predictions.csv')
    for col in ['home_team_name','away_team_name','home_score','away_score','B365H','B365D','B365A']:
        assert np.array_equal(saved[col],test[col]),col
    probs['Expanded RF']=saved[['expanded__RF_vector__p_'+s for s in ['home','draw','away']]].to_numpy()
    rows.append(metrics('Expanded RF',probs['Expanded RF'],labels))
    for name,p in probs.items():
        assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(axis=1),1)
        a,b=bets(name,p,test,labels);strategies.extend(a);ledger.extend(b)
    m=pd.DataFrame(rows);s=pd.DataFrame(strategies);ledger=pd.DataFrame(ledger)
    previous=pd.read_csv(BASE/'strategy_results.csv')
    joined=s[s.model=='Expanded RF'].merge(previous[previous.model=='expanded__RF_vector'],on='strategy',suffixes=('_new','_old'))
    assert len(joined)==15
    for c in ['net','staked','bets','flat_net']:np.testing.assert_allclose(joined[c+'_new'],joined[c+'_old'],atol=1e-7)
    weeks=test.Matchweek.to_numpy(int);unique=np.unique(weeks)
    rng=np.random.default_rng(20261014);samples=rng.integers(0,len(unique),size=(2000,len(unique)))
    counts=np.array([(weeks==w).sum() for w in unique]);paired=[]
    summary=[]
    for family in FAMILIES:
        without=family+'__no_teams';withteams=family+'__teams'
        loss=lambda name:-np.log(np.maximum(probs[name][np.arange(len(test)),labels],1e-15))
        delta=loss(withteams)-loss(without)
        sums=np.array([delta[weeks==w].sum() for w in unique]);draws=sums[samples].sum(axis=1)/counts[samples].sum(axis=1)
        paired.append(dict(family=family,log_loss_change=delta.mean(),ci_low=np.quantile(draws,.025),ci_high=np.quantile(draws,.975)))
        a=m[m.model==without].iloc[0];b=m[m.model==withteams].iloc[0]
        profit=lambda name:float(s[(s.model==name)&(s.strategy==SEARCH_CANDIDATE)].iloc[0].net)
        summary.append(dict(model=family,cv_without_teams=a.cv_log_loss,cv_with_teams=b.cv_log_loss,test_without_teams=a.log_loss,test_with_teams=b.log_loss,profit_without_teams=profit(without),profit_with_teams=profit(withteams)))
    summary=pd.DataFrame(summary);paired=pd.DataFrame(paired)
    pred=test[['home_team_name','away_team_name','home_score','away_score','B365H','B365D','B365A']].copy()
    for name,p in probs.items():
        for j,side in enumerate(['home','draw','away']):pred[name+'__p_'+side]=p[:,j]
    pred.to_csv(OUT/'match_predictions.csv',index=False)
    recovered=pd.read_csv(OUT/'match_predictions.csv')
    for name,p in probs.items():np.testing.assert_allclose(recovered[[name+'__p_'+side for side in ['home','draw','away']]],p,atol=1e-14)
    for name,frame in [('model_metrics',m),('strategy_results',s),('paired_differences',paired),('summary',summary),('test_bet_ledger',ledger)]:frame.to_csv(OUT/(name+'.csv'),index=False)
    relevant=folds[folds.candidate.isin(names)]
    assert len(relevant)==24 and set(relevant.validation_year)=={2019,2020,2021}
    relevant.to_csv(OUT/'validation_folds.csv',index=False)
    status=dict(state='completed',train_matches=len(train),test_matches=len(test),features={n:sets[n] for n in names},fits=diagnostics,input_hashes=plan['input_hashes'],checks=['input hashes','same predictors except team identity','saved forward-validation scores reused','convergence','serialized inference parity','test alignment','probabilities and export parity','RF strategy reconciliation'])
    (OUT/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    table=lambda frame:frame.to_html(index=False,float_format=lambda v:f'{v:,.5f}')
    body='<h1>Do home and away team identities help?</h1><p>All eight regressions retain both mean FIFA ratings, all eight form/prior variables, three Bet365 odds and matchweek. The only change within each pair is adding separate home-team and away-team categorical effects.</p>'
    body+='<h2>Matched comparison</h2><p>Lower log loss is better. CV averages forward validation on 2019–20, 2020–21 and 2021–22, trained only on earlier seasons. Test is the same 1,447 matches from 2022–23. Profit uses filtered Legacy variance and unchanged $1–$5 levels.</p>'+table(summary)
    body+='<h2>Test metrics</h2>'+table(m)
    body+='<h2>Default strategy accounting</h2>'+table(s[s.strategy==SEARCH_CANDIDATE])
    body+='<h2>All 15 strategies: test net dollars</h2>'+table(s.pivot(index='strategy',columns='model',values='net').reset_index())
    body+='<h2>Team effect on test log loss</h2><p>With teams minus without teams: negative favors team identity. Approximate 95% intervals use 2,000 paired matchweek resamples; these do not account for every time/team dependency or earlier test inspection.</p>'+table(paired)
    body+='<details><summary>Per-season forward validation</summary>'+table(relevant)+'</details>'
    body+='<details><summary>Full strategy accounting</summary>'+table(s)+'</details>'
    body+='<h2>Interpretation and method</h2><p>Home/away rating and form values already distinguish the two sides. Adding team identity additionally estimates persistent effects for individual clubs, separately by home/away role. These fixed effects are additive, not pairwise matchup interactions. Team encoding is fitted on training rows only; unseen clubs have zero dummy values. The 16 raw inputs with teams expand to '+str(int(m[m.team_identity==True].iloc[0].encoded_features))+' encoded columns, versus 14 without teams. Ordered models remove non-identifiable directions before unpenalized likelihood fitting. OLS is unpenalized; Poisson retains fixed alpha=0.01. Team effects are not separately tuned or shrunk in this experiment.</p>'
    body+='<p>Both specifications are retained explicitly; no simplicity rule removes requested features. The previously computed CV scores are reused because the feature sets and input hashes match exactly. These test comparisons are exploratory on a previously inspected season. No live RF replacement. Historical actual lineups and odds; no fees, compounding or bankroll limits.</p>'
    doc='<!doctype html><meta charset="utf-8"><title>Team identities: matched comparison</title><style>body{font:15px system-ui;margin:30px;color:#182532}table{border-collapse:collapse;font-size:12px}td,th{padding:8px;border-bottom:1px solid #ddd}details{margin:20px 0}</style>'+body
    (OUT/'report.html').write_text(doc,encoding='utf-8')
    path=BASE/'report.html';prior=path.read_text(encoding='utf-8').split('<!-- team-identity -->')[0]
    prior=prior.replace('<section id="feature-selection">','<section id="feature-selection"><p><a href="#team-identity">NEW: FIFA + form + odds, with and without home/away team identities</a></p>') if 'href="#team-identity"' not in prior else prior
    path.write_text(prior+'<!-- team-identity --><hr><section id="team-identity">'+body+'</section>',encoding='utf-8')
    print(summary.to_string(index=False));print(paired.to_string(index=False))


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
