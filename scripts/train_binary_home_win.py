"""Retrain home-win/not-home-win models, selecting only on earlier seasons."""
import argparse
import hashlib
import json
import html
from pathlib import Path
from select_expanded_features import ROOT, np, pd, joblib, threadpool_limits
from sklearn.metrics import log_loss,roc_auc_score
from compare_lineup_improvements import add_controls,candidates
from compare_positional_matchups import chronological_split,check_dates
from home_win_model import fit,predict,target

DATA = ROOT/'results/lineup_improvements'
OUT = DATA/'binary_retrained'


def metrics(probabilities, frame):
    p = np.asarray(probabilities,float)
    y = target(frame)
    picks = p[:,1]>=.5
    return dict(matches=len(y),correct=int((picks==y).sum()),accuracy=float(np.mean(picks==y)),
        log_loss=float(log_loss(y,p,labels=[0,1])),brier=float(np.mean((p[:,1]-y)**2)),
        auc=float(roc_auc_score(y,p[:,1])),home_picks=int(picks.sum()),no_home_picks=int((~picks).sum()),
        home_wins=int(y.sum()),no_home_wins=int((1-y).sum()))


def paired(a,b,frame):
    y=target(frame)
    rows=np.arange(len(y))
    delta=-np.log(np.maximum(a[rows,y],1e-15))+np.log(np.maximum(b[rows,y],1e-15))
    weeks=pd.to_datetime(frame.fixture_date).dt.to_period('W').astype(str).to_numpy()
    unique=np.unique(weeks)
    sums=np.array([delta[weeks==w].sum() for w in unique])
    counts=np.array([(weeks==w).sum() for w in unique])
    indices=np.random.default_rng(20261110).integers(0,len(unique),size=(2000,len(unique)))
    boot=sums[indices].sum(axis=1)/counts[indices].sum(axis=1)
    return dict(log_loss_difference=float(delta.mean()),lower_95=float(np.quantile(boot,.025)),
                upper_95=float(np.quantile(boot,.975)),correct_difference=int(
                    ((a[:,1]>=.5)==y).sum()-((b[:,1]>=.5)==y).sum()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trees',type=int,default=500)
    args=parser.parse_args()
    if args.trees<100:
        parser.error('At least 100 trees required')
    OUT.mkdir(parents=True,exist_ok=True)
    # Refuse stale feature/code provenance rather than silently using mixed runs.
    prior_plan=json.loads((DATA/'plan.json').read_text())
    for name,digest in prior_plan['input_hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Earlier experiment provenance changed: {name}')
    train=add_controls(pd.read_csv(DATA/'train_features.csv'),'train')
    base=joblib.load(ROOT/'results/expanded_comparison/expanded__RF_vector.joblib')
    sets=candidates(base['categorical']+base['numeric'])
    paths=[DATA/'train_features.csv',DATA/'test_features.csv',DATA/'predictions.csv',DATA/'plan.json',
           ROOT/'src/home_win_model.py',Path(__file__),ROOT/'scripts/compare_lineup_improvements.py']
    plan=dict(target='1: home win; 0: draw or away win',probability_order=['no_home_win','home_win'],
        candidates=sets,families=['Logistic','RF_binary'],logistic_c=[.01,.1,1.,10.],trees=args.trees,
        validation_years=[2019,2020],audit_year=2021,test_year=2022,threshold=.5,
        selection='Minimum mean seasonal binary log loss. Logistic C selected per candidate, then '
            'candidate per family. Fixed 0.5 threshold; ties predict home win. Choices frozen before evaluation.',
        training='Direct binary logistic likelihood / random-forest log-loss classification. '
            'No goal-count fitting, no Poisson probability conversion, no class reweighting. '
            'Preprocessing fitted independently in each training fold.',
        caveat=prior_plan['caveat'],evaluation=prior_plan['evaluation'],
        input_hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    cache=OUT/'cache';cache.mkdir(exist_ok=True)
    def run(tr,va,family,candidate,c,split,save=False):
        check_dates(tr,va)
        signature=dict(plan=plan,family=family,candidate=candidate,c=c,split=split)
        key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
        path=cache/(key+'.npz')
        if path.exists() and not save:
            with np.load(path) as saved:
                return saved['p'],None
        print(f'{split}: {family} / {candidate} / C={c}',flush=True)
        bundle=fit(tr,sets[candidate],family,args.trees,c)
        p=predict(bundle,va)
        np.savez_compressed(path,p=p)
        if save:
            bundle.update(experiment_plan=plan,selection=selection[family],
                trained_through=str(tr.fixture_date.max()),training_matches=len(tr))
            model_path=OUT/f'{family}__{candidate}.joblib'
            joblib.dump(bundle,model_path)
            if not np.allclose(predict(joblib.load(model_path),va.head(12)),p[:12]):
                raise ValueError('Reloaded classifier probabilities differ')
            if candidate==selection[family]['candidate']:
                joblib.dump(bundle,OUT/f'{family}_selected.joblib')
        return p,bundle
    validation,selection,parameters=[],{},{}
    for family in plan['families']:
        parameters[family]={}
        for candidate in sets:
            for c in (plan['logistic_c'] if family=='Logistic' else [1.]):
                for year in plan['validation_years']:
                    tr,va=chronological_split(train,year)
                    p,_=run(tr,va,family,candidate,c,str(year))
                    validation.append(dict(family=family,candidate=candidate,C=c,season=year,**metrics(p,va)))
            table=pd.DataFrame(validation)
            mean=table[(table.family==family)&(table.candidate==candidate)].groupby('C').log_loss.mean()
            chosen=float(mean.idxmin())
            parameters[family][candidate]=dict(C=chosen,mean_log_loss=float(mean.loc[chosen]))
            pd.DataFrame(validation).to_csv(OUT/'validation.csv',index=False)
        chosen=min(parameters[family],key=lambda k:parameters[family][k]['mean_log_loss'])
        selection[family]=dict(candidate=chosen,**parameters[family][chosen])
        print('Selected',family,selection[family],flush=True)
    (OUT/'selection.json').write_text(json.dumps(dict(selection=selection,parameters=parameters),indent=2),encoding='utf-8')
    test=add_controls(pd.read_csv(DATA/'test_features.csv'),'test')
    previous=pd.read_csv(DATA/'predictions.csv')
    records,exports,differences,weights=[],[],[],[]
    for split,tr,va in [('Audit 2021-22',*chronological_split(train,2021)),('Test 2022-23',train,test)]:
        probability_map={}
        for family in plan['families']:
            for candidate in sets:
                c=parameters[family][candidate]['C']
                p,bundle=run(tr,va,family,candidate,c,split,save=split.startswith('Test'))
                probability_map[family,candidate]=p
                records.append(dict(split=split,origin='direct_binary',family=family,candidate=candidate,C=c,
                    selected=candidate==selection[family]['candidate'],**metrics(p,va)))
                for i,r in enumerate(va.itertuples()):
                    exports.append(dict(split=split,family=family,candidate=candidate,source_row=r.source_row,
                        date=r.fixture_date,league=r.league,home_team=r.home_team_name,away_team=r.away_team_name,
                        home_score=r.home_score,away_score=r.away_score,target=int(r.home_score>r.away_score),
                        p_home_win=float(p[i,1]),p_no_home_win=float(p[i,0]),prediction=int(p[i,1]>=.5)))
                if split.startswith('Test'):
                    values=bundle['model'].coef_[0] if family=='Logistic' else bundle['model'].feature_importances_
                    weights.extend(dict(family=family,candidate=candidate,feature=name,value=float(value),
                        type='coefficient' if family=='Logistic' else 'impurity_importance')
                        for name,value in zip(bundle['feature_names'],values))
        for (family,candidate),g in previous[previous.split==split].groupby(['family','candidate']):
            aligned=va[['source_row','home_score','away_score']].merge(g,on='source_row',validate='one_to_one',
                how='left',suffixes=('_source',''))
            if aligned.p_home.isna().any() or not np.array_equal(aligned.home_score_source,aligned.home_score) or not np.array_equal(aligned.away_score_source,aligned.away_score):
                raise ValueError('Prior goal-model sample mismatch')
            ph=aligned.p_home.to_numpy(float)
            p=np.column_stack([1-ph,ph])
            probability_map[family,candidate]=p
            records.append(dict(split=split,origin='goal_conversion',family=family,candidate=candidate,C=np.nan,
                                selected=candidate=='baseline',**metrics(p,va)))
        odds=1/va[['B365H','B365D','B365A']].to_numpy(float)
        ph=odds[:,0]/odds.sum(axis=1)
        records.append(dict(split=split,origin='market',family='Market',candidate='normalized_odds',C=np.nan,
                            selected=False,**metrics(np.column_stack([1-ph,ph]),va)))
        for family,comparator in [('Logistic','Poisson'),('RF_binary','RF_goals')]:
            for candidate in sets:
                differences.append(dict(split=split,family=family,candidate=candidate,comparator=comparator,
                    comparator_candidate=candidate,**paired(probability_map[family,candidate],probability_map[comparator,candidate],va)))
            chosen=selection[family]['candidate']
            differences.append(dict(split=split,family=family,candidate=chosen,comparator=comparator,
                comparator_candidate='validation_selected_baseline',
                **paired(probability_map[family,chosen],probability_map[comparator,'baseline'],va)))
        pd.DataFrame(records).to_csv(OUT/'metrics.csv',index=False)
    for name,rows in [('predictions',exports),('paired_differences',differences),('feature_weights',weights)]:
        pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
    (OUT/'summary.json').write_text(json.dumps(dict(selection=selection,metrics=records),indent=2),encoding='utf-8')
    table=pd.DataFrame(records)
    body='<h1>Direct home-win binary classifiers</h1>'
    for key in ['target','training','selection','evaluation','caveat']:
        body+='<p>'+html.escape(plan[key])+'</p>'
    body+='<p>The no-home-win class includes both draws and away wins. This model does not separately '
    body+='estimate draw or away-win probabilities. Accuracy uses a fixed 50% threshold; no betting '
    body+='weighting, skipping, or threshold tuning is applied.</p>'
    body+='<h2>Validation-selected models and market</h2>'+table[table.selected|table.family.eq('Market')].to_html(index=False,float_format=lambda v:f'{v:.5f}')
    body+='<h2>All candidates</h2>'+table.to_html(index=False,float_format=lambda v:f'{v:.5f}')
    body+='<h2>Direct binary versus goal conversion</h2><p>Negative log-loss differences favor the binary '
    body+='classifier. Approximate 95% intervals use 2,000 paired calendar-week bootstrap samples.</p>'
    body+=pd.DataFrame(differences).to_html(index=False,float_format=lambda v:f'{v:.5f}')
    body+='<h2>Validation folds</h2>'+pd.DataFrame(validation).to_html(index=False,float_format=lambda v:f'{v:.5f}')
    (OUT/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Binary home-win retraining</title>'
        '<style>body{font:15px system-ui;margin:30px}table{border-collapse:collapse;font-size:12px}'
        'td,th{padding:7px;border-bottom:1px solid #ddd}</style>'+body,encoding='utf-8')
    print(table[(table.split=='Test 2022-23')&(table.origin=='direct_binary')].to_string(index=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):
        main()
