"""Four fixed RF classifier ablations; same rows, chronological folds and forest size."""
import json,hashlib
from select_expanded_features import ROOT,pd,np,joblib,threadpool_limits
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from classifier_features import FORM,RICH
from calibrate_models import scores
from probability_calibration import fit_temperature,scale
OUT=ROOT/'results/classifier'
VARIANTS={'RF classifier':[],'RF classifier + adjusted form':FORM,'RF classifier + lineup':RICH,'RF classifier + both':FORM+RICH}

def matrix(bundle,frame):return np.column_stack([bundle['encoder'].transform(frame[bundle['categorical']]),frame[bundle['numeric']].to_numpy(float)])
def fit_model(base,frame,extra):
    cats=base['categorical'];nums=base['numeric']+extra;enc=clone(base['encoder']).fit(frame[cats])
    valid=RandomForestClassifier().get_params();params={k:v for k,v in base['model'].get_params().items() if k in valid}
    params.update(criterion='log_loss',oob_score=False,n_jobs=4)
    bundle=dict(encoder=enc,categorical=cats,numeric=nums,forest=RandomForestClassifier(**params))
    y=(1-np.sign(frame.home_score-frame.away_score)).to_numpy(int)
    bundle['forest'].fit(matrix(bundle,frame),y)
    assert np.array_equal(bundle['forest'].classes_,[0,1,2])
    return bundle

def main():
    train=pd.read_csv(OUT/'train.csv');base=joblib.load(ROOT/'results/expanded_comparison/expanded__RF_vector.joblib')
    inputs=[OUT/'train.csv',OUT/'test.csv',ROOT/'results/expanded_comparison/expanded__RF_vector.joblib']
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    plan=dict(variants=VARIANTS,validation_seasons=[2019,2020],audit_season=2021,test_season=2022,method='Direct H/D/A RF, same 2000 trees/depth/features/bootstrap settings as goal RF; log-loss split criterion; equal sample weights. Fixed ablations, no test-based feature selection.',input_hashes=hashes)
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    models={};records=[];diagnostics=[];trainpred=train.copy()
    for name,extra in VARIANTS.items():
        folds=[]
        for year in [2019,2020,2021]:
            tr=train[train.season<year];va=train[train.season==year]
            key=hashlib.sha256((json.dumps(hashes,sort_keys=True)+name+str(year)).encode()).hexdigest();cache=OUT/(key+'.npz')
            if cache.exists():p=np.load(cache)['p']
            else:
                print(f'Forward fit {name}: predict {year}',flush=True)
                fitted=fit_model(base,tr,extra);p=fitted['forest'].predict_proba(matrix(fitted,va));np.savez_compressed(cache,p=p)
            y=(1-np.sign(va.home_score-va.away_score)).to_numpy(int);folds.append((p,y))
            records.append(dict(model=name,split=f'Forward {year}–{year+1}',**scores(p,y)))
        cp=np.concatenate([p for p,y in folds[:2]]);cy=np.concatenate([y for p,y in folds[:2]])
        t=fit_temperature(cp,cy);ap,ay=folds[2];final_t=fit_temperature(np.concatenate([p for p,y in folds]),np.concatenate([y for p,y in folds]))
        models[name]=dict(temperature=final_t,audit_temperature=t,audit_raw=scores(ap,ay),audit_calibrated=scores(scale(ap,t),ay),calibration_matches=sum(len(y) for p,y in folds))
        print(f'Full fit {name}; audit log loss {scores(ap,ay)["log_loss"]:.6f}',flush=True)
        fitted=fit_model(base,train,extra);joblib.dump(fitted,OUT/(name+'.joblib'))
        p=fitted['forest'].predict_proba(matrix(fitted,train));y=(1-np.sign(train.home_score-train.away_score)).to_numpy(int)
        for j,s in enumerate(['home','draw','away']):trainpred[name+'__p_'+s]=p[:,j]
        diagnostics.append(dict(model=name,train_log_loss=scores(p,y)['log_loss'],forward_cv_log_loss=float(np.mean([scores(*v)['log_loss'] for v in folds[:2]]))))
    # All models and temperatures frozen before evaluating test labels.
    (OUT/'calibrators.json').write_text(json.dumps(dict(models=models,input_hashes=hashes),indent=2))
    test=pd.read_csv(OUT/'test.csv');testpred=test.copy();y=(1-np.sign(test.home_score-test.away_score)).to_numpy(int)
    for name in VARIANTS:
        fitted=joblib.load(OUT/(name+'.joblib'));p=fitted['forest'].predict_proba(matrix(fitted,test))
        for j,s in enumerate(['home','draw','away']):testpred[name+'__p_'+s]=p[:,j]
        m=scores(p,y);models[name].update(test_raw=m,test_calibrated=scores(scale(p,models[name]['temperature']),y))
        records.append(dict(model=name,split='Test 2022–23 (previously inspected)',**m))
        next(r for r in diagnostics if r['model']==name)['test_log_loss']=m['log_loss']
    old=pd.read_csv(ROOT/'results/calibration/forward_predictions.csv')
    for year in [2019,2020,2021]:
        r=old[(old.model=='Expanded RF')&(old.season==year)];records.append(dict(model='Expanded RF (goal vector)',split=f'Forward {year}–{year+1}',**scores(r[['p_home','p_draw','p_away']].to_numpy(),r.outcome.to_numpy(int))))
    previous=json.loads((ROOT/'results/calibration/calibrators.json').read_text())['models']['Expanded RF']['test_raw']
    records.append(dict(model='Expanded RF (goal vector)',split='Test 2022–23 (previously inspected)',**previous))
    for name in VARIANTS:hashes[(OUT/(name+'.joblib')).relative_to(ROOT).as_posix()]=hashlib.sha256((OUT/(name+'.joblib')).read_bytes()).hexdigest()
    trainpred.to_csv(OUT/'train_predictions.csv',index=False);testpred.to_csv(OUT/'match_predictions.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'diagnostics.csv',index=False);pd.DataFrame(records).to_csv(OUT/'metrics.csv',index=False)
    (OUT/'calibrators.json').write_text(json.dumps(dict(models=models,input_hashes=hashes),indent=2))
    (OUT/'summary.json').write_text(json.dumps(dict(plan=plan,metrics=records),indent=2))
    print('Classifier and feature experiments complete.',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
