"""Freeze a compact, calibrated home-win model before a 2025-26 holdout evaluation."""
import hashlib
import json
from pathlib import Path
from select_expanded_features import ROOT,np,pd,joblib,threadpool_limits
from regularized_home_win import (configs,fit_model,predict_raw,fit_calibrator,apply_calibrator,
                                  select_model,choose_threshold,predict_bundle,market_probability)
from home_win_model import target
from train_binary_home_win import metrics
from backtest_binary_training_oos import portfolios

DATA=ROOT/'results/lineup_improvements'
OUT=DATA/'binary_retrained/regularized'


def binary_metrics(ph,frame):
    return metrics(np.column_stack([1-ph,ph]),frame)


def export(frame,raw,model,cutoff,ph=None):
    cols=['source_row','season','fixture_date','league','home_team_name','away_team_name',
          'home_score','away_score','B365H','B365D','B365A']
    result=frame[cols].copy()
    result['target']=target(frame)
    result['trained_through']=cutoff
    result['raw_probability']=raw
    result['p_home_win']=raw if ph is None else ph
    result['model']=model
    result['match_id']=result.season.astype(str)+':'+result.source_row.astype(str)
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cache=OUT/'cache';cache.mkdir(exist_ok=True)
    paths=[DATA/'train_features.csv',DATA/'test_features.csv',ROOT/'src/regularized_home_win.py',Path(__file__)]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    grid=configs()
    plan=dict(grid=grid,probability_variants=['raw','sigmoid'],
        calibration='Sigmoid on logits; C=1. Fit on preceding two seasons of raw forward predictions '
                    '(one season for 2016 initialization). Never fit on in-sample probabilities.',
        selection='Minimum mean binary log loss over the previous two forward validation seasons; '
                  'include raw bookmaker probabilities. Outer seasons 2018 through 2022.',
        threshold='Choose among 0%, 2%, 5%, 10% using previous two seasons only. Require >=100 bets '
                  'combined and positive flat net in BOTH seasons; maximize total flat net among eligible '
                  'thresholds, ties favor lower cutoff. Otherwise select no betting.',
        history='2014-2022 already inspected in previous experiments; historical nested replay is not untouched data.',
        final='Freeze model, calibration, threshold and all preprocessing on 2014-2022 before opening 2025-26 outcomes.',
        holdout=dict(season=2025,leagues=['D1','I1','SP1'],minimum_rated_starters=10,
            policy='All valid-odds fixtures get the market benchmark. Model comparison uses complete unique '
                   'lineups with >=10 dated ratings per side; same eligible rows for both. '
                   'Use only FC23/FC24 snapshots dated before season start. Existing snapshots are stale. '
                   'No model or threshold revisions after evaluation. Actual historical starters, not forecast lineups.'),
        input_hashes=hashes)
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    train=pd.read_csv(DATA/'train_features.csv')
    old_test=pd.read_csv(DATA/'test_features.csv')
    frame=pd.concat([train,old_test],ignore_index=True)
    raw_history=[];predictions=[];validation=[];decisions=[];outer=[]
    for year in range(2015,2023):
        tr,va=frame[frame.season<year],frame[frame.season==year]
        assert tr.fixture_date.max()<va.fixture_date.min()
        if year>=2018:
            chosen,score=select_model(pd.DataFrame(validation),year)
            past=pd.concat(predictions,ignore_index=True)
            cutoff,search=choose_threshold(past[past.model==chosen],year)
            decisions.append(dict(season=year,model=chosen,validation_log_loss=score,threshold=cutoff,
                                  selection_seasons=[year-2,year-1],threshold_search=search))
            (OUT/'decisions.json').write_text(json.dumps(decisions,indent=2),encoding='utf-8')
            print(f'Predict {year}: selected {chosen}; betting cutoff={cutoff}',flush=True)
        else:chosen=None
        print(f'  Fit {len(grid)-1} compact models: {len(tr)} earlier matches',flush=True)
        for name,config in grid.items():
            key=hashlib.sha256(json.dumps(dict(hashes=hashes,year=year,name=name,config=config),sort_keys=True).encode()).hexdigest()
            path=cache/(key+'.npz')
            if path.exists():
                with np.load(path) as saved:
                    assert np.array_equal(saved['source_row'],va.source_row.to_numpy())
                    raw=saved['raw']
            else:
                bundle=fit_model(tr,config);raw=predict_raw(bundle,va)
                np.savez_compressed(path,raw=raw,source_row=va.source_row.to_numpy())
            raw_frame=export(va,raw,name,tr.fixture_date.max())
            variants={'raw':raw}
            if year>=2016:
                past=pd.concat(raw_history,ignore_index=True)
                past=past[(past.model==name)&past.season.isin([year-2,year-1])]
                calibrator=fit_calibrator(past,va.fixture_date.min())
                variants['sigmoid']=apply_calibrator(calibrator,raw)
            for variant,ph in variants.items():
                model=name+'__'+variant
                record=export(va,raw,model,tr.fixture_date.max(),ph)
                record['calibration_last_date']=(past.fixture_date.max() if variant=='sigmoid' else '')
                predictions.append(record)
                validation.append(dict(season=year,model=model,**binary_metrics(ph,va)))
                if model==chosen:outer.append(record.assign(chosen_threshold=cutoff))
            raw_history.append(raw_frame)
        pd.DataFrame(validation).to_csv(OUT/'validation.csv',index=False)
    all_predictions=pd.concat(predictions,ignore_index=True)
    all_predictions.to_csv(OUT/'candidate_predictions.csv.gz',index=False)
    selected=pd.concat(outer,ignore_index=True)
    selected.to_csv(OUT/'selected_forward_predictions.csv',index=False)
    # Comparisons on exactly the same replay fixtures.
    comparisons=[]
    for name,g in [('selected',selected),('market',all_predictions[(all_predictions.model=='market__raw')&(all_predictions.season>=2018)])]:
        comparisons.append(dict(model=name,**binary_metrics(g.p_home_win.to_numpy(),g)))
    old=pd.read_csv(DATA/'binary_retrained/training_oos/predictions.csv')
    old=old[(old.protocol=='fixed')&(old.season>=2018)].copy()
    test_predictions=pd.read_csv(DATA/'binary_retrained/predictions.csv')
    ref=test_predictions[(test_predictions.split=='Test 2022-23')&(test_predictions.family=='RF_binary')&(test_predictions.candidate=='fifa_prior')]
    joined=old_test.merge(ref[['source_row','p_home_win']],on='source_row',validate='one_to_one')
    old=pd.concat([old,joined],ignore_index=True)
    old['match_id']=old.season.astype(str)+':'+old.source_row.astype(str)
    old=old.set_index('match_id').loc[selected.match_id].reset_index()
    assert np.array_equal(old.home_score.to_numpy(),selected.home_score.to_numpy())
    assert np.array_equal(old.away_score.to_numpy(),selected.away_score.to_numpy())
    comparisons.append(dict(model='original_RF',**binary_metrics(old.p_home_win.to_numpy(),old)))
    pd.DataFrame(comparisons).to_csv(OUT/'replay_metrics.csv',index=False)
    ledgers=[];summaries=[]
    for name,g in [('selected',selected),('original_RF',old),('market',all_predictions[(all_predictions.model=='market__raw')&(all_predictions.season>=2018)])]:
        ledger,records=portfolios(g,name);ledgers.append(ledger);summaries.extend(records)
    pd.concat(ledgers,ignore_index=True).to_csv(OUT/'replay_ledger.csv.gz',index=False)
    pd.DataFrame(summaries).to_csv(OUT/'replay_summary.csv',index=False)
    # Final choice uses 2021 and 2022 only, with no 2025 labels or inputs opened.
    chosen,score=select_model(pd.DataFrame(validation),2023)
    chosen_predictions=all_predictions[all_predictions.model==chosen]
    cutoff,search=choose_threshold(chosen_predictions,2023)
    name,variant=chosen.split('__')
    print(f'FREEZE for holdout: {chosen}; cutoff={cutoff}',flush=True)
    bundle=fit_model(frame,grid[name])
    raw_past=pd.concat(raw_history,ignore_index=True)
    raw_past=raw_past[(raw_past.model==name)&raw_past.season.isin([2021,2022])]
    bundle['calibrator']=fit_calibrator(raw_past,'2023-08-01') if variant=='sigmoid' else None
    bundle.update(model_name=chosen,threshold=cutoff,trained_through=frame.fixture_date.max(),
                  training_matches=len(frame),selection_seasons=[2021,2022],plan=plan)
    path=OUT/'frozen_model.joblib';joblib.dump(bundle,path)
    np.testing.assert_allclose(predict_bundle(joblib.load(path),frame.head(30)),predict_bundle(bundle,frame.head(30)))
    fitted=predict_bundle(bundle,frame)
    in_sample=export(frame,fitted,chosen,frame.fixture_date.max())
    in_sample.to_csv(OUT/'in_sample_predictions.csv',index=False)
    ledger,records=portfolios(in_sample,'in_sample')
    ledger.to_csv(OUT/'in_sample_ledger.csv.gz',index=False)
    pd.DataFrame(records).to_csv(OUT/'in_sample_summary.csv',index=False)
    frozen=dict(model=chosen,threshold=cutoff,threshold_search=search,validation_log_loss=score,
        training_matches=len(frame),training_last_date=frame.fixture_date.max(),selection_seasons=[2021,2022],
        calibration_seasons=[2021,2022] if variant=='sigmoid' else [],
        in_sample_metrics=binary_metrics(fitted,frame),
        model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        plan_sha256=hashlib.sha256((OUT/'plan.json').read_bytes()).hexdigest(),
        holdout_status='Not evaluated; 2025-26 reserved for one frozen evaluation.')
    (OUT/'frozen_selection.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    print(pd.DataFrame(comparisons).to_string(index=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
