"""Select exponential season weights on 2019–21; audit 2021–22; test 2022–23."""
import json,hashlib
from select_expanded_features import ROOT,np,pd,joblib,season_years,design,fit,predict,transform,threadpool_limits
from sklearn.base import clone
from compare_expanded_models import outcome_probabilities
from calibrate_models import scores
from probability_calibration import scale,fit_temperature

OUT=ROOT/'results/recency'
RECENT='__recent'

def weights(years,half_life):
    years=np.asarray(years)
    if half_life is None:return np.ones(len(years))
    if half_life<=0:raise ValueError('Half-life must be positive')
    w=2.**(-(years.max()-years)/half_life)
    return w/w.mean()

def train_predict(name,bundle,tr,va,w):
    goals=tr[['home_score','away_score']].to_numpy()
    if name=='Expanded RF':
        enc=clone(bundle['encoder']);cats=bundle['categorical'];nums=bundle['numeric']
        x=np.column_stack([enc.fit_transform(tr[cats]),tr[nums].to_numpy()]);xv=np.column_stack([enc.transform(va[cats]),va[nums].to_numpy()])
        estimator=clone(bundle['model']);estimator.set_params(n_jobs=4,oob_score=False)
        estimator.fit(x,goals,sample_weight=w)
        fitted=dict(model=estimator,encoder=enc,numeric=nums,categorical=cats,scaler=None)
        return outcome_probabilities(estimator.predict(xv)),fitted
    x,xv,xo,xov,artifact=design(tr,va,bundle['columns'])
    model,_=fit(name.split('__')[0],x,xo,goals,sample_weight=w)
    return predict(model,xv,xov)[0],dict(**artifact,fit=model)

def bundle_predict(name,bundle,frame):
    if name=='Expanded RF':
        x=np.column_stack([bundle['encoder'].transform(frame[bundle['categorical']]),frame[bundle['numeric']].to_numpy()])
        return outcome_probabilities(bundle['model'].predict(x))
    x,xo=transform(bundle,frame);return predict(bundle['fit'],x,xo)[0]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(ROOT/'data/expanded_2008/train.csv');years=season_years(train)
    prior=json.loads((ROOT/'results/calibration/calibrators.json').read_text())
    for path,digest in prior['input_hashes'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    forward=pd.read_csv(ROOT/'results/calibration/forward_predictions.csv')
    names=list(prior['models']);choices=[None,3,6]
    plan=dict(candidates_half_life_seasons=choices,selection_seasons=[2019,2020],audit_season=2021,test_season=2022,selection_metric='Mean seasonal raw log loss; ties favor equal weights',weight_formula='2 ** (-(latest_training_season - season) / half_life), normalized to mean 1',input_hashes=prior['input_hashes'])
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    selected={};records=[];fold_rows=[];pending={};calibration={}
    for name in names:
        path=ROOT/('results/expanded_comparison/expanded__RF_vector.joblib' if name=='Expanded RF' else 'results/team_identity/'+name+'.joblib')
        bundle=joblib.load(path);preds={}
        for half in choices:
            for year in [2019,2020]:
                va=train.loc[years==year];tr=train.loc[years<year]
                if half is None:
                    rows=forward[(forward.model==name)&(forward.season==year)].sort_values('source_row')
                    assert np.array_equal(rows.source_row,va.index)
                    p=rows[['p_home','p_draw','p_away']].to_numpy()
                else:
                    key=hashlib.sha256((path.read_bytes()+str((half,year)).encode()+(ROOT/'data/expanded_2008/train.csv').read_bytes())).hexdigest()
                    cache=OUT/(key+'.npz')
                    if cache.exists():p=np.load(cache)['p']
                    else:
                        print(f'{name}: half-life {half}, predict {year}',flush=True)
                        p,_=train_predict(name,bundle,tr,va,weights(years[years<year],half));np.savez_compressed(cache,p=p)
                y=(1-np.sign(va.home_score-va.away_score)).to_numpy(int);preds[half,year]=(p,y)
                fold_rows.append(dict(model=name,half_life='equal' if half is None else half,season=year,**scores(p,y)))
        means={half:np.mean([scores(*preds[half,year])['log_loss'] for year in [2019,2020]]) for half in choices}
        best=min(choices,key=lambda half:means[half]);selected[name]=dict(half_life=best,cv_log_loss=float(means[best]),equal_cv_log_loss=float(means[None]))
        # Selection is frozen before the audit season is predicted or inspected.
        (OUT/'selected.json').write_text(json.dumps(selected,indent=2))
        print(f'SELECTED {name}: {best or "equal"}, CV {means[None]:.6f} -> {means[best]:.6f}',flush=True)
        if best is None:
            rows=forward[(forward.model==name)&(forward.season==2021)].sort_values('source_row')
            ap=rows[['p_home','p_draw','p_away']].to_numpy()
        else:ap,_=train_predict(name,bundle,train.loc[years<2021],train.loc[years==2021],weights(years[years<2021],best))
        ay=(1-np.sign(train.loc[years==2021].home_score-train.loc[years==2021].away_score)).to_numpy(int)
        cp=np.concatenate([preds[best,year][0] for year in [2019,2020]]);cy=np.concatenate([preds[best,year][1] for year in [2019,2020]])
        audit_t=fit_temperature(cp,cy);final_t=fit_temperature(np.concatenate([cp,ap]),np.concatenate([cy,ay]))
        calibration[name+RECENT]=dict(temperature=final_t,audit_temperature=audit_t,calibration_matches=len(cy)+len(ay),audit_raw=scores(ap,ay),audit_calibrated=scores(scale(ap,audit_t),ay))
        equal=prior['models'][name]['audit_raw']
        records.extend([dict(model=name,split='Forward audit 2021–22',variant='Equal weights',**equal),dict(model=name,split='Forward audit 2021–22',variant='Selected weights',**scores(ap,ay))])
        # Full training fit has no access to test data or outcomes.
        if best is None:fitted=bundle
        else:
            print(f'Full fit {name}: half-life {best}',flush=True)
            _,fitted=train_predict(name,bundle,train,train.iloc[:1],weights(years,best))
        joblib.dump(fitted,OUT/(name+RECENT+'.joblib'))
        pending[name]=dict(train=scores(bundle_predict(name,fitted,train),(1-np.sign(train.home_score-train.away_score)).to_numpy(int)))
        selected[name]['effective_matches']=float(weights(years,best).sum()**2/np.sum(weights(years,best)**2))
    # Freeze all choices and calibration before test evaluation.
    (OUT/'selected.json').write_text(json.dumps(selected,indent=2))
    pd.DataFrame(fold_rows).to_csv(OUT/'validation.csv',index=False)
    test=pd.read_csv(ROOT/'data/expanded_2008/test.csv');saved=pd.read_csv(ROOT/'results/team_identity/match_predictions.csv')
    for col in ['home_team_name','away_team_name','home_score','away_score']:assert np.array_equal(test[col],saved[col])
    output=test.copy();y=(1-np.sign(test.home_score-test.away_score)).to_numpy(int)
    diagnostics=[]
    for name in names:
        fitted=joblib.load(OUT/(name+RECENT+'.joblib'));p=bundle_predict(name,fitted,test)
        raw=saved[[name+'__p_'+s for s in ['home','draw','away']]].to_numpy()
        for variant,prob in [('Equal weights',raw),('Selected weights',p)]:records.append(dict(model=name,split='Test 2022–23 (previously inspected)',variant=variant,**scores(prob,y)))
        calibration[name+RECENT]['test_raw']=scores(p,y);calibration[name+RECENT]['test_calibrated']=scores(scale(p,calibration[name+RECENT]['temperature']),y)
        for j,s in enumerate(['home','draw','away']):output[name+RECENT+'__p_'+s]=p[:,j]
        diagnostics.append(dict(model=name+RECENT,train_log_loss=pending[name]['train']['log_loss'],forward_cv_log_loss=selected[name]['cv_log_loss'],test_log_loss=scores(p,y)['log_loss']))
    output.to_csv(OUT/'match_predictions.csv',index=False);pd.DataFrame(diagnostics).to_csv(OUT/'diagnostics.csv',index=False);pd.DataFrame(records).to_csv(OUT/'metrics.csv',index=False)
    hashes={**prior['input_hashes'],**{(OUT/(n+RECENT+'.joblib')).relative_to(ROOT).as_posix():hashlib.sha256((OUT/(n+RECENT+'.joblib')).read_bytes()).hexdigest() for n in names}}
    (OUT/'calibrators.json').write_text(json.dumps(dict(models=calibration,input_hashes=hashes),indent=2))
    (OUT/'summary.json').write_text(json.dumps(dict(plan=plan,models=selected,metrics=records),indent=2))
    print('Recent-data weighting completed.',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
