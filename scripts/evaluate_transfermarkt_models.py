"""Paired all-row Transfermarkt ablation with chronological validation."""
import json,hashlib,copy
from select_expanded_features import ROOT,pd,np,joblib,threadpool_limits
from classifier_features import club,FORM,RICH
from train_recent_models import train_predict,bundle_predict
from train_classifier_models import fit_model,matrix
from calibrate_models import scores
from bankroll_backtest import simulate,METHODS

OUT=ROOT/'results/transfermarkt_models'
MODELS=['Expanded RF','RF classifier + both','Linear','Poisson','Ordered_logit','Ordered_probit']
EXTRA=['tm_home_log_value','tm_away_log_value','tm_home_missing','tm_away_missing']
TM_ALIASES={'1. FC Nuremberg':'Nürnberg','Arminia Bielefeld':'Arminia','AC Carpi':'Carpi','AC Monza':'Monza','ACF Fiorentina':'Fiorentina','Associazione Sportiva Roma':'Roma','Atalanta BC':'Atalanta','Benevento Calcio':'Benevento','Bologna Football Club 1909':'Bologna','Brescia Calcio':'Brescia','Cádiz CF':'Cádiz','Cagliari Calcio':'Cagliari','Catania FC':'Catania','CD Leganés':'Leganés','Celta de Vigo':'Celta Vigo','Cesena FC':'Cesena','Córdoba CF':'Córdoba','Delfino Pescara 1936':'Pescara','Deportivo de La Coruña':'La Coruña','FC Crotone':'Crotone','FC Empoli':'Empoli','FC Ingolstadt 04':'Ingolstadt 04','Frosinone Calcio':'Frosinone','Genoa CFC':'Genoa','Juventus FC':'Juventus','Palermo FC':'Palermo','Parma Calcio 1913':'Parma','RCD Espanyol Barcelona':'Espanyol','Real Valladolid CF':'Valladolid','SD Eibar':'Eibar','SD Huesca':'Huesca','Siena FC':'Siena','Società Sportiva Lazio S.p.A.':'Lazio','Spezia Calcio':'Spezia','Sunderland AFC':'Sunderland','SV Darmstadt 98':'Darmstadt 98','Torino FC':'Torino','UC Sampdoria':'Sampdoria','UD Las Palmas':'Las Palmas','Udinese Calcio':'Udinese','US Cremonese':'Cremonese','US Lecce':'Lecce','US Livorno 1915':'Livorno','US Salernitana 1919':'Salernitana','US Sassuolo':'Sassuolo','Venezia FC':'Venezia'}
TM_ALIASES.update({n+' FC':n for n in ['Arsenal','Brentford','Burnley','Chelsea','Everton','Fulham','Girona','Liverpool','Middlesbrough','Reading','Southampton','Watford']})
TM_KEYS={club(a):club(b) for a,b in TM_ALIASES.items()}
def tm_club(name):return TM_KEYS.get(club(name),club(name))

def prepare():
    tm=pd.read_csv(ROOT/'data/transfermarkt_values/match_features.csv')
    lookup={}
    for r in tm.to_dict('records'):
        key=(r['fixture_date'],r['league'],tm_club(r['home_team_name']),tm_club(r['away_team_name']))
        assert key not in lookup,key
        lookup[key]=r
    frames={};coverage=[];unmatched=[]
    for split in ['train','test']:
        frame=pd.read_csv(ROOT/f'results/classifier/{split}.csv');frame['source_row']=np.arange(len(frame))
        added=[]
        for r in frame.to_dict('records'):
            key=(r['fixture_date'],r['league'],club(r['home_team_name']),club(r['away_team_name']))
            hit=lookup.get(key);item={}
            if hit is None and r['season']>=2013:unmatched.append(dict(split=split,**{k:r[k] for k in ['source_row','fixture_date','league','home_team_name','away_team_name']}))
            for side in ['home','away']:
                value=hit[side+'_xi_value_eur'] if hit else np.nan
                item[f'tm_{side}_log_value']=np.log1p(value) if pd.notna(value) else np.nan
                item[f'tm_{side}_missing']=int(pd.isna(value))
            added.append(item)
        frames[split]=pd.concat([frame,pd.DataFrame(added)],axis=1)
        frames[split].to_csv(OUT/f'{split}.csv',index=False)
        for (year,league),g in frames[split].groupby(['season','league']):
            coverage.append(dict(split=split,season=int(year),league=league,matches=len(g),both_values=int(((g.tm_home_missing==0)&(g.tm_away_missing==0)).sum())))
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
    pd.DataFrame(unmatched).to_csv(OUT/'unmatched.csv',index=False)
    print('Coverage',json.dumps({s:dict(matches=len(f),both_values=int(((f.tm_home_missing==0)&(f.tm_away_missing==0)).sum())) for s,f in frames.items()}),flush=True)
    return frames

def impute(tr,va):
    tr=tr.copy();va=va.copy();medians={}
    for c in EXTRA[:2]:
        median=float(tr[c].median());assert np.isfinite(median)
        medians[c]=median;tr[c]=tr[c].fillna(median);va[c]=va[c].fillna(median)
    return tr,va,medians

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    frames=prepare();base=joblib.load(ROOT/'results/expanded_comparison/expanded__RF_vector.joblib')
    compact=['HomePlayer_Overall_mean','AwayPlayer_Overall_mean',*[c for c in frames['train'] if c.endswith(('_form','_prior'))],'B365H','B365D','B365A','Matchweek','home_team_name','away_team_name']
    plan=dict(models=MODELS,extra_features=EXTRA,regression_features=compact,folds=[2020,2021],test_season=2022,
              policy='All original 19,263 train and 1,447 test rows preserved, including original test duplicate. Missing XI values imputed to training-fold median with missing indicators; no future valuations. Fixed existing parameters; equal sample weights; raw probabilities; no test selection.',
              hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'results/classifier/train.csv',ROOT/'results/classifier/test.csv',ROOT/'data/transfermarkt_values/match_features.csv',ROOT/'results/expanded_comparison/expanded__RF_vector.joblib',__file__ and __import__('pathlib').Path(__file__)]})
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    metrics=[];bank=[];curves=[];audit=[];preds={}
    def record(split,model,variant,f,p):
        assert len(f)==len(p) and np.isfinite(p).all() and np.allclose(p.sum(1),1)
        preds[split,model,variant]=(f,p)
        for league in ['All','E0','SP1','D1','I1']:
            mask=np.ones(len(f),bool) if league=='All' else f.league.eq(league).to_numpy()
            g=f.loc[mask];q=p[mask];y=(1-np.sign(g.home_score-g.away_score)).to_numpy(int)
            metrics.append(dict(split=split,model=model,variant=variant,league=league,matches=len(g),**scores(q,y)))
            for method in METHODS:
                stats,_=simulate(g,q,method);pts=stats.pop('points')
                bank.append(dict(split=split,model=model,variant=variant,league=league,method=method,**stats))
                curves.append(dict(split=split,model=model,variant=variant,league=league,method=method,points=pts))
        pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False);pd.DataFrame(bank).to_csv(OUT/'bankroll.csv',index=False)
    for year in [2020,2021,2022]:
        va=frames['test'] if year==2022 else frames['train'][frames['train'].season==year]
        tr=frames['train'] if year==2022 else frames['train'][(frames['train'].season<year)&(frames['train'].fixture_date<va.fixture_date.min())]
        assert tr.fixture_date.max()<va.fixture_date.min()
        tr,va,medians=impute(tr,va)
        audit.append(dict(year=year,train_rows=len(tr),evaluation_rows=len(va),train_last=tr.fixture_date.max(),evaluation_first=va.fixture_date.min(),imputation_medians=medians))
        split='Test' if year==2022 else f'Forward {year}'
        for model in MODELS:
            for variant in ['Baseline','With values']:
                extra=EXTRA if variant=='With values' else []
                key=hashlib.sha256((json.dumps(plan,sort_keys=True)+str((year,model,variant))).encode()).hexdigest();cache=OUT/(key+'.npz')
                if cache.exists():
                    saved=np.load(cache);p=saved['p'];pt=saved['pt'] if year==2022 else None
                else:
                    print('Fit',year,model,variant,len(tr),len(va),flush=True)
                    if model=='RF classifier + both':
                        fitted=fit_model(base,tr,FORM+RICH+extra);p=fitted['forest'].predict_proba(matrix(fitted,va))
                        pt=fitted['forest'].predict_proba(matrix(fitted,tr)) if year==2022 else None
                    else:
                        spec={**base,'numeric':base['numeric']+extra} if model=='Expanded RF' else dict(columns=compact+extra)
                        p,fitted=train_predict(model,spec,tr,va,np.ones(len(tr)))
                        pt=bundle_predict(model,fitted,tr) if year==2022 else None
                    np.savez_compressed(cache,p=p,**({'pt':pt} if pt is not None else {}))
                    if year==2022:joblib.dump(dict(model=model,variant=variant,bundle=fitted,imputation_medians=medians,extra=extra),OUT/(model.replace(' ','_')+'_'+variant.replace(' ','_')+'.joblib'))
                record(split,model,variant,va,p)
                if year==2022:record('Train in-sample',model,variant,tr,pt)
    for model in MODELS:
        for variant in ['Baseline','With values']:
            pairs=[preds[f'Forward {year}',model,variant] for year in [2020,2021]]
            record('Train chronological OOF',model,variant,pd.concat([a for a,b in pairs]),np.concatenate([b for a,b in pairs]))
    differences=[];rng=np.random.default_rng(914)
    for split in ['Train chronological OOF','Test']:
        for model in MODELS:
            f,a=preds[split,model,'With values'];_,b=preds[split,model,'Baseline']
            y=(1-np.sign(f.home_score-f.away_score)).to_numpy(int);ix=np.arange(len(y))
            diff=-np.log(np.clip(a[ix,y],1e-15,1))+np.log(np.clip(b[ix,y],1e-15,1))
            weeks=pd.to_datetime(f.fixture_date).dt.to_period('W').astype(str).to_numpy();groups=[np.flatnonzero(weeks==w) for w in np.unique(weeks)]
            sums=np.array([diff[g].sum() for g in groups]);counts=np.array([len(g) for g in groups]);draw=rng.integers(0,len(groups),(2000,len(groups)))
            lo,hi=np.quantile(sums[draw].sum(1)/counts[draw].sum(1),[.025,.975])
            differences.append(dict(split=split,model=model,log_loss_change=float(diff.mean()),ci_low=float(lo),ci_high=float(hi)))
    pd.DataFrame(differences).to_csv(OUT/'paired_differences.csv',index=False)
    (OUT/'fold_audit.json').write_text(json.dumps(audit,indent=2));(OUT/'curves.json').write_text(json.dumps(curves))
    from report_transfermarkt_models import render
    render(metrics,bank,curves,differences)
    print('COMPLETE',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
