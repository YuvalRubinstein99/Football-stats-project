"""Forward test saved pre-2022 models on 2024-25; no fitting or cutoff selection."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
from select_expanded_features import ROOT,pd,np,joblib,threadpool_limits
from compare_match_form import FAMILIES,configs,infer,evaluate_scores
from prepare_form_2024 import DEST
from bankroll_backtest import simulate,METHODS
from compare_return_cutoffs import CUTOFFS

BASE=ROOT/'results/match_form'
OUT=DEST/'return_cutoffs'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(DEST/'matches.csv')
    selections=json.loads((BASE/'selection.json').read_text())
    plan=json.loads((BASE/'selection_plan.json').read_text())
    old_test=pd.read_csv(BASE/'test.csv')
    old_predictions=pd.read_csv(BASE/'predictions.csv')
    train=pd.read_csv(BASE/'train.csv')
    assert train.fixture_date.max()<frame.fixture_date.min()
    assert hashlib.sha256((BASE/'train.csv').read_bytes()).hexdigest()==plan['data_audit']['output_hashes']['train']
    jobs=[]
    for family in FAMILIES:
        for variant,selection in [('minimum_cv_without_form','selected_without_form'),
                                  ('minimum_cv_form_allowed','selected_form_allowed')]:
            choice=configs(family,plan['candidates'])[selections[family][selection]['best_mean_config']]
            path=BASE/(family+'_minimum_cv_without_form.joblib') if variant.endswith('without_form') else BASE/(
                family+'_test_2022_'+choice['candidate']+'_'+str(choice['alpha'])+'.joblib')
            jobs.append((family,variant,path,frame,plan['candidates'][choice['candidate']]))
    # The saved conservative odds-only classifier can cover the entire season,
    # without pretending that it is the selected classifier with form.
    all_frame=pd.read_csv(DEST/'all_matches.csv')
    jobs.append(('RF_classifier_odds_full','selected_odds_only',
                 BASE/'RF_classifier_test_2022_odds_None.joblib',all_frame,['B365H','B365D','B365A']))
    manifest=dict(season='2024-25',model_training_last_date=train.fixture_date.max(),
        cutoff_primary=.02,cutoffs=CUTOFFS,initial_bankroll=1000.,
        policy='Use saved 2022-test model artifacts unchanged. No training or feature/threshold '
               'selection on this season. Predictive features only use strictly prior dates. '
               'No quoted-odds floor/cap beyond valid prices. Each simulation independently starts at $1000.',
        coverage=json.loads((DEST/'data_audit.json').read_text()),
        full_sample_reference='RF_classifier_odds_full is the separate saved conservative odds-only '
                              'classifier, evaluated on all 1446 fixtures. Other models share the rated subset.',
        hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                for p in {Path(__file__),DEST/'matches.csv',DEST/'all_matches.csv',BASE/'selection.json',
                          ROOT/'src/bankroll_backtest.py',*[j[2] for j in jobs]}})
    (OUT/'plan.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    summaries=[];curves=[];predictions=[];metrics=[];count=0
    with gzip.open(OUT/'ledger.csv.gz','wt',encoding='utf-8',newline='') as handle:
        writer=None
        for family,variant,path,f,cols in jobs:
            bundle=joblib.load(path)
            assert bundle['columns']==cols
            p,goals=infer(bundle,f)
            np.testing.assert_allclose(p.sum(1),1,atol=1e-10)
            assert np.isfinite(p).all() and (p>=0).all()
            # Confirm the stored artifact still reproduces the earlier reported predictions.
            old_variant='selected_form_allowed' if family=='RF_classifier_odds_full' else variant
            old_family='RF_classifier' if family=='RF_classifier_odds_full' else family
            ref=old_predictions[(old_predictions.family==old_family)&
                (old_predictions.variant==old_variant)&(old_predictions.split=='test_2022')].head(20)
            subset=old_test.set_index('source_row').loc[ref.source_row]
            np.testing.assert_allclose(infer(bundle,subset)[0],ref[['p_home','p_draw','p_away']],atol=1e-9)
            blind=f.head(20).copy();blind[['home_score','away_score']]=999
            np.testing.assert_allclose(infer(bundle,blind)[0],p[:20],atol=1e-10)
            for league in ['ALL',*sorted(f.league.unique())]:
                mask=np.ones(len(f),bool) if league=='ALL' else (f.league==league).to_numpy()
                metrics.append(dict(family=family,variant=variant,split='forward_2024',league=league,
                    matches=int(mask.sum()),**evaluate_scores(p[mask],goals[mask] if goals is not None else None,f.loc[mask])))
            outcomes=(1-np.sign(f.home_score-f.away_score)).to_numpy(int)
            ids=f.source_row.to_numpy(int)
            for i,r in enumerate(f.itertuples()):
                predictions.append(dict(family=family,variant=variant,split='forward_2024',
                    source_row=int(r.source_row),date=r.fixture_date,league=r.league,outcome=int(outcomes[i]),
                    p_home=p[i,0],p_draw=p[i,1],p_away=p[i,2]))
            for method in METHODS:
                previous=len(f)+1
                for cutoff in CUTOFFS:
                    stat,ledger=simulate(f,p,method,initial=1000.,min_edge=cutoff)
                    points=stat.pop('points')
                    meta=dict(family=family,variant=variant,split='forward_2024',method=method,cutoff=cutoff)
                    detail=pd.DataFrame(ledger)
                    sides=detail.side.map({'home':0,'draw':1,'away':2}).to_numpy()
                    net=detail.stake*(np.where(sides==outcomes[detail.source_row.to_numpy(int)],detail.odds,0)-1)
                    np.testing.assert_allclose(net,detail.net,atol=1e-9)
                    daily=detail.groupby('date',sort=True)[['stake','net']].sum()
                    opening=1000+daily.net.cumsum().shift(1,fill_value=0)
                    np.testing.assert_allclose(detail.opening_bankroll,detail.date.map(opening),atol=1e-9)
                    assert (daily.stake<=opening*.1+1e-9).all()
                    assert (detail.stake<=detail.opening_bankroll*.02+1e-9).all()
                    placed=detail[detail.stake>0]
                    assert len(placed)==stat['bets']<=previous
                    previous=stat['bets']
                    assert (placed.probability*placed.odds>1+cutoff).all()
                    np.testing.assert_allclose(1000+net.sum(),stat['ending_bankroll'],atol=1e-9)
                    balances=np.r_[1000.,1000+daily.net.cumsum().to_numpy()]
                    dd=(1-balances/np.maximum.accumulate(balances)).max()
                    np.testing.assert_allclose(dd,stat['max_drawdown'],atol=1e-10)
                    summaries.append(dict(**meta,**stat,daily_return_sd_pct=float((daily.net/opening).std()*100),
                        daily_profit_sd_dollars=float(daily.net.std()),gross_return=float(detail.stake.sum()+net.sum()),
                        match_dates=len(daily)))
                    curves.extend(dict(**meta,**point) for point in points)
                    for entry in ledger:
                        entry['source_row']=int(ids[entry['source_row']])
                        record=dict(**meta,**entry)
                        if writer is None:writer=csv.DictWriter(handle,fieldnames=list(record));writer.writeheader()
                        writer.writerow(record)
                    count+=len(ledger)
            print(f'{family} / {variant}: {len(f)} matches verified',flush=True)
    pd.DataFrame(predictions).to_csv(DEST/'predictions.csv',index=False)
    pd.DataFrame(metrics).to_csv(DEST/'metrics.csv',index=False)
    pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'curves.csv',index=False)
    (OUT/'verification.json').write_text(json.dumps(dict(simulations=len(summaries),ledger_rows=count,
        saved_model_reproduction='passed',current_target_blind_inference='passed',
        settlement_and_cash_caps='passed',drawdown_reconciliation='passed'),indent=2),encoding='utf-8')
    result=pd.DataFrame(summaries)
    print(result[(result.family=='RF_classifier')&(result.variant=='minimum_cv_form_allowed')&(result.cutoff==.02)].to_string(index=False))


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
