"""Strict forward model/mixture selection with prequential calibration."""
import sys, json, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from select_expanded_features import np,pd,joblib,threadpool_limits
from train_recent_models import train_predict,weights,bundle_predict
from train_classifier_models import fit_model,matrix,VARIANTS
from probability_calibration import scale,fit_temperature
from bankroll_backtest import METHODS,simulate
from optimize_staking_mixtures import select,metrics

OUT=ROOT/'results/chronological_selection'
LEAGUES=['All','E0','SP1','D1','I1']


def daily(frame,prob):
    summaries=[];streams=[]
    for method in METHODS:
        summary,ledger=simulate(frame,prob,method)
        summaries.append(summary)
        stream={}
        for r in ledger:stream[r['date']]=stream.get(r['date'],0.)+r['net']
        streams.append(stream)
    return [[d,*[s[d] for s in streams]] for d in sorted(streams[0])],summaries


def curve(days,w):
    result=metrics(days,np.array([w]))[0];net=0.;points=[dict(date='Start',net=0.)]
    for d,*v in days:
        net+=sum(a*b for a,b in zip(v,w));points.append(dict(date=d,net=net))
    return dict(**result,points=points)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(ROOT/'results/classifier/train.csv')
    basepath=ROOT/'results/expanded_comparison/expanded__RF_vector.joblib'
    base=joblib.load(basepath)
    forms=[c for c in train if c.endswith(('_form','_prior'))]
    compact=['HomePlayer_Overall_mean','AwayPlayer_Overall_mean',*forms,'B365H','B365D','B365A','Matchweek']
    specs={'Expanded RF':dict(kind='goal',half=None,bundle=base)}
    for family in ['Linear','Poisson','Ordered_logit','Ordered_probit']:
        for teams in [False,True]:
            name=family+('__teams' if teams else '__no_teams')
            specs[name]=dict(kind='regression',half=None,bundle=dict(columns=compact+(['home_team_name','away_team_name'] if teams else [])))
    for name,spec in list(specs.items()):specs[name+'__half3']={**spec,'half':3}
    for name,extra in VARIANTS.items():specs[name]=dict(kind='classifier',extra=extra,half=None)
    plan=dict(folds=[2019,2020,2021],selection_seasons=[2020,2021],calibration_seed=2019,
              candidates=44,recency='Fixed three-season half-life, not full-training-selected half-life',
              regression_features=compact,teams='Two fixed variants for each regression family, with and without one-hot team identity',
              objective='OOF CAGR / max(daily maximum drawdown, 0.01); 5% weight grid plus equal thirds',
              calibration='Each OOF season uses only earlier OOF seasons for temperature; 2019 seeds calibration and is excluded from selection for every candidate',
              limits='Previously examined test season; previously developed fixed forest settings/features. Historical actual lineups and Bet365 odds, no fees. This is a two-season OOF selection experiment, not untouched prospective evidence.',
              train_sha256=hashlib.sha256((ROOT/'results/classifier/train.csv').read_bytes()).hexdigest(),
              code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    oldforward=pd.read_csv(ROOT/'results/calibration/forward_predictions.csv')
    classifier_plan=json.loads((ROOT/'results/classifier/plan.json').read_text())
    oof={};temperatures={};audit=[]
    for name,spec in specs.items():
        previous_p=[];previous_y=[];raw_parts=[];cal_parts=[];indexes=[]
        for year in plan['folds']:
            va=train[train.season==year]
            tr=train[(train.season<year)&(train.fixture_date<va.fixture_date.min())]
            assert tr.fixture_date.max()<va.fixture_date.min()
            fingerprint=hashlib.sha256((plan['train_sha256']+plan['code_sha256']+name+str(year)).encode()).hexdigest()
            cache=OUT/(fingerprint+'.npz')
            if cache.exists():p=np.load(cache)['p']
            elif name=='Expanded RF' and len(tr)==sum(train.season<year):
                saved=oldforward[(oldforward.model==name)&(oldforward.season==year)].sort_values('source_row')
                assert np.array_equal(saved.source_row,va.index)
                p=saved[['p_home','p_draw','p_away']].to_numpy()
            elif spec['kind']=='classifier' and len(tr)==sum(train.season<year):
                key=hashlib.sha256((json.dumps(classifier_plan['input_hashes'],sort_keys=True)+name+str(year)).encode()).hexdigest()
                oldcache=ROOT/'results/classifier'/(key+'.npz')
                if oldcache.exists():p=np.load(oldcache)['p']
                else:
                    print(f'Fit {name}, predict {year}',flush=True)
                    fitted=fit_model(base,tr,spec['extra']);p=fitted['forest'].predict_proba(matrix(fitted,va))
            else:
                print(f'Fit {name}, predict {year}',flush=True)
                p,_=train_predict(name.split('__half3')[0],spec['bundle'],tr,va,weights(tr.season.to_numpy(),spec['half']))
            assert p.shape==(len(va),3) and np.isfinite(p).all() and np.allclose(p.sum(1),1)
            np.savez_compressed(cache,p=p)
            y=(1-np.sign(va.home_score-va.away_score)).to_numpy(int)
            temperature=fit_temperature(np.concatenate(previous_p),np.concatenate(previous_y)) if previous_p else 1.
            if year in plan['selection_seasons']:
                raw_parts.append(p);cal_parts.append(scale(p,temperature));indexes.extend(va.index)
            audit.append(dict(model=name,season=year,train_last_date=tr.fixture_date.max(),validation_first_date=va.fixture_date.min(),temperature=temperature,calibration_seasons=[v for v in plan['folds'] if v<year]))
            previous_p.append(p);previous_y.append(y)
        temperatures[name]=fit_temperature(np.concatenate(previous_p),np.concatenate(previous_y))
        for label,parts in [(name,raw_parts),(name+'__calibrated',cal_parts)]:
            oof[label]=(train.loc[indexes].reset_index(drop=True),np.concatenate(parts))
        print('OOF complete: '+name,flush=True)
    (OUT/'fold_audit.json').write_text(json.dumps(audit,indent=2))
    candidates=[];histories={}
    for name,(frame,p) in oof.items():
        for league in LEAGUES:
            mask=np.ones(len(frame),bool) if league=='All' else frame.league.eq(league).to_numpy()
            days,_=daily(frame.loc[mask],p[mask]);w,score=select(days)
            candidates.append(dict(model=name,league=league,weights=w.tolist(),oof=score,robust_oof=metrics(days,np.array([[0,0,1]]))[0]))
            histories[name,league]=days
    chosen=[]
    for league in LEAGUES:
        eligible=[r for r in candidates if r['league']==league]
        winner=max(eligible,key=lambda r:(r['oof']['score'],-r['oof']['max_drawdown'],r['model']))
        robust=max(eligible,key=lambda r:(r['robust_oof']['score'],-r['robust_oof']['max_drawdown'],r['model']))
        chosen.append({**winner,'robust_selected_model':robust['model']})
    # Freeze model choices, weights and final calibration before opening test data.
    frozen=dict(plan=plan,choices=chosen,temperatures=temperatures,candidates=candidates)
    (OUT/'frozen_choices.json').write_text(json.dumps(frozen,indent=2))
    print('FROZEN '+json.dumps(chosen),flush=True)
    test=pd.read_csv(ROOT/'results/classifier/test.csv')
    assert train.fixture_date.max()<test.fixture_date.min()
    predictions={}
    for label in sorted({r['model'] for r in chosen}|{r['robust_selected_model'] for r in chosen}):
        name=label.removesuffix('__calibrated')
        if name not in predictions:
            spec=specs[name];print('Final fit: '+name,flush=True)
            if spec['kind']=='classifier':
                fitted=fit_model(base,train,spec['extra']);p=fitted['forest'].predict_proba(matrix(fitted,test))
            else:p,fitted=train_predict(name.split('__half3')[0],spec['bundle'],train,test,weights(train.season.to_numpy(),spec['half']))
            joblib.dump(fitted,OUT/(name+'.joblib'));predictions[name]=p
        if label.endswith('__calibrated'):predictions[label]=scale(predictions[name],temperatures[name])
    results=[]
    for choice in chosen:
        name=choice['model'];league=choice['league'];mask=np.ones(len(test),bool) if league=='All' else test.league.eq(league).to_numpy()
        days,_=daily(test.loc[mask],predictions[name][mask]);other=choice['robust_selected_model']
        robust_days,_=daily(test.loc[mask],predictions[other][mask])
        results.append(dict(**choice,oof_mixture=curve(histories[name,league],choice['weights']),
                            oof_robust=curve(histories[name,league],[0,0,1]),
                            test_mixture=curve(days,choice['weights']),test_robust=curve(days,[0,0,1]),
                            test_robust_selected=curve(robust_days,[0,0,1])))
    report=dict(plan=plan,results=results)
    (OUT/'report.json').write_text(json.dumps(report,separators=(',',':')))
    destination=ROOT/'results/bankroll_backtest/daily'/(hashlib.sha256(b'OOF selection report').hexdigest()+'.json')
    destination.write_text(json.dumps(report,separators=(',',':')))
    print(json.dumps([{k:v for k,v in r.items() if not k.startswith(('oof_','test_'))}|{k:{a:b for a,b in v.items() if a!='points'} for k,v in r.items() if k.startswith('test_')} for r in results],indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
