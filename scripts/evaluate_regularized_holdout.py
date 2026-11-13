"""Evaluate the frozen candidate and betting policy once, without refitting."""
import hashlib
import json
from select_expanded_features import ROOT,np,pd,joblib,threadpool_limits
from regularized_home_win import predict_bundle,predict_raw,market_probability
from train_regularized_home_win import OUT,export,binary_metrics
from prepare_regularized_holdout import DEST
from backtest_binary_training_oos import portfolios
from binary_betting import settle,summary
from plot_binary_training_oos import render


def policy_returns(frame,cutoffs,label):
    """NaN means the prespecified no-bet choice, never a default zero cutoff."""
    ph=frame.p_home_win.to_numpy()
    odds=frame[['B365H','B365D','B365A']].to_numpy(float)
    cutoff=np.broadcast_to(np.asarray(cutoffs,float),(len(frame),))
    active=np.isfinite(cutoff)&(ph*odds[:,0]-1>cutoff)
    outcomes=np.where(frame.home_score>frame.away_score,0,np.where(frame.home_score==frame.away_score,1,2))
    ledgers=[];records=[]
    for staking in ['flat','tiers']:
        result=settle(ph,odds,outcomes,dict(pick=np.zeros(len(frame),int),active=active),staking)
        record=dict(period=label,staking=staking,**summary(result,outcomes));records.append(record)
        rows=frame.copy();rows['staking']=staking;rows['stake']=result['stakes'];rows['net']=result['profit']
        rows['cutoff']=cutoff;rows['period']=label;ledgers.append(rows)
    return pd.concat(ledgers,ignore_index=True),records


def interval(rows):
    rows=rows.copy();rows['fixture_date']=pd.to_datetime(rows.fixture_date)
    weeks=pd.period_range(rows.fixture_date.min().to_period('W'),rows.fixture_date.max().to_period('W'),freq='W')
    weekly=rows.groupby(rows.fixture_date.dt.to_period('W'))[['net','stake']].sum().reindex(weeks,fill_value=0.).to_numpy()
    starts=np.random.default_rng(20261110).integers(0,len(weeks),(5000,int(np.ceil(len(weeks)/4))))
    ids=((starts[:,:,None]+np.arange(4))%len(weeks)).reshape(5000,-1)[:,:len(weeks)]
    total=weekly[ids].sum(axis=1);valid=total[:,1]>0
    roi=total[valid,0]/total[valid,1]
    return dict(roi_lower_95=float(np.quantile(roi,.025)),roi_upper_95=float(np.quantile(roi,.975))) if len(roi) else {}


def main():
    frozen=json.loads((OUT/'frozen_selection.json').read_text())
    assert hashlib.sha256((OUT/'frozen_model.joblib').read_bytes()).hexdigest()==frozen['model_sha256']
    assert hashlib.sha256((OUT/'plan.json').read_bytes()).hexdigest()==frozen['plan_sha256']
    audit=json.loads((DEST/'data_audit.json').read_text())
    assert audit['frozen_model_sha256']==frozen['model_sha256']
    assert hashlib.sha256((DEST/'matches.csv').read_bytes()).hexdigest()==audit['output_sha256']
    bundle=joblib.load(OUT/'frozen_model.joblib')
    frame=pd.read_csv(DEST/'matches.csv').sort_values(['fixture_date','league','source_row']).reset_index(drop=True)
    assert frame.fixture_date.min()>bundle['trained_through']
    p=predict_bundle(bundle,frame)
    np.testing.assert_allclose(p,predict_bundle(bundle,frame.drop(columns=['home_score','away_score'])))
    predictions=export(frame,predict_raw(bundle,frame),frozen['model'],bundle['trained_through'],p)
    predictions.to_csv(DEST/'predictions.csv',index=False)
    metrics=[dict(sample='same eligible matches',model='frozen_model',**binary_metrics(p,frame)),
             dict(sample='same eligible matches',model='market',**binary_metrics(market_probability(frame),frame))]
    all_matches=pd.read_csv(DEST/'all_matches.csv')
    valid=np.isfinite(all_matches[['B365H','B365D','B365A']]).all(axis=1)&(all_matches[['B365H','B365D','B365A']]>1).all(axis=1)
    full=all_matches[valid]
    metrics.append(dict(sample='all valid-odds matches',model='market',**binary_metrics(market_probability(full),full)))
    pd.DataFrame(metrics).to_csv(DEST/'metrics.csv',index=False)
    ledger,records=portfolios(predictions,'holdout_2025')
    totals=pd.DataFrame(records)
    for i,row in totals.iterrows():
        ci=interval(ledger[(ledger.strategy==row.strategy)&(ledger.staking==row.staking)])
        for name,value in ci.items():totals.loc[i,name]=value
    ledger.to_csv(DEST/'ledger.csv.gz',index=False)
    totals.to_csv(DEST/'summary.csv',index=False)
    cutoff=np.nan if frozen['threshold'] is None else frozen['threshold']
    policy_ledger,policy_summary=policy_returns(predictions,cutoff,'holdout_2025')
    replay=pd.read_csv(OUT/'selected_forward_predictions.csv').sort_values(['fixture_date','league','source_row']).reset_index(drop=True)
    replay_policy,replay_summary=policy_returns(replay,replay.chosen_threshold.to_numpy(float),'historical_replay')
    pd.concat([replay_policy,policy_ledger],ignore_index=True).to_csv(OUT/'threshold_policy_ledger.csv.gz',index=False)
    pd.DataFrame([*replay_summary,*policy_summary]).to_csv(OUT/'threshold_policy_summary.csv',index=False)
    curves=render(ledger,totals,'holdout_2025',out=DEST,
        heading='Frozen regularized model: 2025-26 holdout',
        subtitle='Depth-2 forest, minimum 100 samples per leaf | Model and thresholds frozen before evaluation',
        note='Home-only EV >0% (teal, selected cutoff) and >2% (purple, fixed comparison).',
        footer='Held out from this experiment. Dated FC23/FC24 ratings are stale; eligible lineup subset. Different vertical scales.')
    pd.DataFrame(curves).to_csv(DEST/'daily_curves.csv',index=False)
    # Label the adaptive history separately from the final frozen candidate.
    replay_ledger=pd.read_csv(OUT/'replay_ledger.csv.gz')
    replay_ledger=replay_ledger[replay_ledger.protocol=='selected'].copy()
    replay_ledger['source_row']=replay_ledger.match_id
    replay_totals=pd.read_csv(OUT/'replay_summary.csv')
    replay_totals=replay_totals[replay_totals.protocol=='selected']
    render(replay_ledger,replay_totals,'historical_replay',out=OUT,
        heading='Regularized selection: historical forward predictions',
        subtitle='2018-19 to 2022-23 | Model and calibration selected using only earlier seasons',
        note='Fixed EV filters shown. The separate threshold-selection policy abstained in all five replay seasons.',
        footer='Previously inspected historical data. Candidate grid developed after earlier experiments. Different vertical scales.')
    receipt=dict(status='evaluated_frozen_without_refitting',model=frozen['model'],threshold=frozen['threshold'],
        model_sha256=frozen['model_sha256'],plan_sha256=frozen['plan_sha256'],
        sample_matches=len(frame),all_matches=len(all_matches),metrics=metrics,
        caveats=audit['limitation'],uncertainty='5,000 circular four-week block bootstrap samples; '
                 'descriptive conditional ROI intervals, excluding fitting and selection uncertainty.')
    (DEST/'evaluation.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(pd.DataFrame(metrics).to_string(index=False))
    print(totals.to_string(index=False))
    print(pd.DataFrame([*replay_summary,*policy_summary]).to_string(index=False))


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
