"""Chronological temperature calibration for all nine expanded model variants."""
import json,hashlib
from select_expanded_features import ROOT,np,pd,joblib,season_years,design,fit,predict,transform,threadpool_limits
from compare_expanded_models import outcome_probabilities
from sklearn.base import clone
from sklearn.metrics import log_loss,roc_auc_score
from probability_calibration import scale,fit_temperature

OUT=ROOT/'results/calibration'

def scores(p,y):
    return dict(log_loss=float(log_loss(y,p,labels=[0,1,2])),brier=float(np.mean(np.sum((p-np.eye(3)[y])**2,axis=1))),auc=float(roc_auc_score(y,p,multi_class='ovr')))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(ROOT/'data/expanded_2008/train.csv');years=season_years(train)
    names=['Expanded RF']+[f+'__'+t for f in ['Linear','Poisson','Ordered_logit','Ordered_probit'] for t in ['no_teams','teams']]
    paths={n:ROOT/('results/expanded_comparison/expanded__RF_vector.joblib' if n=='Expanded RF' else 'results/team_identity/'+n+'.joblib') for n in names}
    inputs=[ROOT/'data/expanded_2008/train.csv',ROOT/'data/expanded_2008/test.csv',*paths.values()]
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    plan=dict(method='Temperature scaling; fixed bounds 0.25–4; no method selection',calibration_seasons=[2019,2020],audit_season=2021,final_calibration_seasons=[2019,2020,2021],test_season=2022,input_hashes=hashes)
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    models={};records=[];fold_records=[]
    for name in names:
        bundle=joblib.load(paths[name]);folds=[]
        for year in [2019,2020,2021]:
            tr=train.loc[years<year];va=train.loc[years==year]
            key=hashlib.sha256((hashes[paths[name].relative_to(ROOT).as_posix()]+hashes['data/expanded_2008/train.csv']+str(year)).encode()).hexdigest()
            cache=OUT/(key+'.npz')
            if cache.exists():p=np.load(cache)['p']
            else:
                print(f'Forward fit: {name}, predict {year}–{year+1}, train ends {year-1}',flush=True)
                goals=tr[['home_score','away_score']].to_numpy()
                if name=='Expanded RF':
                    enc=clone(bundle['encoder']);cats=bundle['categorical'];nums=bundle['numeric']
                    x=np.column_stack([enc.fit_transform(tr[cats]),tr[nums].to_numpy()]);xv=np.column_stack([enc.transform(va[cats]),va[nums].to_numpy()])
                    estimator=clone(bundle['model']);estimator.set_params(n_jobs=4,oob_score=False);estimator.fit(x,goals)
                    p=outcome_probabilities(estimator.predict(xv))
                else:
                    x,xv,xo,xov,_=design(tr,va,bundle['columns'])
                    fitted,_=fit(name.split('__')[0],x,xo,goals);p,_=predict(fitted,xv,xov)
                np.savez_compressed(cache,p=p)
            y=(1-np.sign(va.home_score-va.away_score)).to_numpy(int)
            folds.append((p,y))
            fold_records.extend(dict(model=name,source_row=int(idx),season=year,p_home=float(pp[0]),p_draw=float(pp[1]),p_away=float(pp[2]),outcome=int(yy)) for idx,pp,yy in zip(va.index,p,y))
        cp=np.concatenate([p for p,y in folds[:2]]);cy=np.concatenate([y for p,y in folds[:2]])
        audit_t=fit_temperature(cp,cy);ap,ay=folds[2]
        for variant,prob in [('Raw',ap),('Calibrated',scale(ap,audit_t))]:records.append(dict(model=name,split='Forward audit 2021–22',variant=variant,**scores(prob,ay)))
        temperature=fit_temperature(np.concatenate([p for p,y in folds]),np.concatenate([y for p,y in folds]))
        models[name]=dict(temperature=temperature,audit_temperature=audit_t,calibration_matches=sum(len(y) for p,y in folds),audit_raw=scores(ap,ay),audit_calibrated=scores(scale(ap,audit_t),ay))
        print(f'{name}: final T={temperature:.5f}; audit log loss {models[name]["audit_raw"]["log_loss"]:.5f} -> {models[name]["audit_calibrated"]["log_loss"]:.5f}',flush=True)
    # Freeze all parameters before accessing test labels or reporting test quality.
    artifact=dict(**plan,models=models)
    (OUT/'calibrators.json').write_text(json.dumps(artifact,indent=2))
    pd.DataFrame(fold_records).to_csv(OUT/'forward_predictions.csv',index=False)
    test=pd.read_csv(ROOT/'data/expanded_2008/test.csv');saved=pd.read_csv(ROOT/'results/team_identity/match_predictions.csv')
    for col in ['home_team_name','away_team_name','home_score','away_score']:assert np.array_equal(test[col],saved[col])
    y=(1-np.sign(test.home_score-test.away_score)).to_numpy(int)
    bins=[]
    for name in names:
        raw=saved[[name+'__p_'+s for s in ['home','draw','away']]].to_numpy();cal=scale(raw,models[name]['temperature'])
        for variant,p in [('Raw',raw),('Calibrated',cal)]:
            records.append(dict(model=name,split='Test 2022–23 (previously inspected)',variant=variant,**scores(p,y)))
            for j,outcome in enumerate(['home','draw','away']):
                for lo in np.arange(0,1,.1):
                    mask=(p[:,j]>=lo)&(p[:,j]<(lo+.1) if lo<.9 else p[:,j]<=1)
                    if mask.any():bins.append(dict(model=name,variant=variant,outcome=outcome,lower=float(lo),count=int(mask.sum()),predicted=float(p[mask,j].mean()),observed=float((y[mask]==j).mean())))
        models[name]['test_raw']=scores(raw,y);models[name]['test_calibrated']=scores(cal,y)
    (OUT/'calibrators.json').write_text(json.dumps(artifact,indent=2))
    pd.DataFrame(records).to_csv(OUT/'metrics.csv',index=False);pd.DataFrame(bins).to_csv(OUT/'reliability.csv',index=False)
    print('Calibration complete. Results: results/calibration/metrics.csv',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
