"""Refit old and new fixed model specifications on exactly the same train/test split."""
import hashlib
import json
from pathlib import Path
from select_expanded_features import ROOT,np,pd,joblib,threadpool_limits
from home_win_model import fit as fit_old,predict as predict_old,target
from regularized_home_win import fit_model,predict_raw,market_probability
from train_binary_home_win import metrics
from backtest_binary_training_oos import portfolios

DATA=ROOT/'results/lineup_improvements'
BASE=DATA/'binary_retrained'
OUT=BASE/'matched_comparison'


def score(p,frame):
    return metrics(np.column_stack([1-p,p]),frame)


def verify_settlements(ledger):
    edge=ledger.p_home_win*ledger.B365H-1
    cutoff=np.where(ledger.strategy.str.endswith('>2%'),.02,0.)
    active=edge>cutoff
    tier=np.clip(np.ceil(np.round(edge/(ledger.B365H-1)*100,10)),1,5)
    stakes=np.where(active,np.where(ledger.staking.eq('flat'),1,tier),0.)
    expected=stakes*np.where(ledger.home_score>ledger.away_score,ledger.B365H-1,-1.)
    np.testing.assert_array_equal(stakes,ledger.stake)
    np.testing.assert_allclose(expected,ledger.net,atol=1e-12)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(DATA/'train_features.csv')
    test=pd.read_csv(DATA/'test_features.csv')
    assert len(train)==8521 and len(test)==1065
    assert train.fixture_date.max()<test.fixture_date.min()
    assert train.source_row.is_unique and test.source_row.is_unique
    old_reference=joblib.load(BASE/'RF_binary_selected.joblib')
    new_reference=joblib.load(BASE/'regularized/frozen_model.joblib')
    assert old_reference['selection']['candidate']=='fifa_prior'
    assert new_reference['model_name']=='rf_positions_d2_leaf100__raw'
    assert new_reference['calibrator'] is None
    paths=[DATA/'train_features.csv',DATA/'test_features.csv',BASE/'RF_binary_selected.joblib',
           BASE/'regularized/frozen_model.joblib',ROOT/'src/home_win_model.py',
           ROOT/'src/regularized_home_win.py',Path(__file__)]
    plan=dict(training_matches=len(train),training_seasons='2014-15 through 2021-22',
        training_last_date=train.fixture_date.max(),test_matches=len(test),test_season='2022-23',
        test_first_date=test.fixture_date.min(),test_last_date=test.fixture_date.max(),
        old=dict(candidate='fifa_prior',columns=old_reference['columns'],trees=500,max_depth=5,
                 min_samples_leaf=1,max_features=.3,max_samples=.5),
        new=dict(config=new_reference['config'],trees=500,max_features=.7,max_samples=.5),
        controls='Identical ordered training rows, test rows, labels, odds, random seed (20261020), '
                 '500 trees, and home-only betting rules. Fit all preprocessing on the training rows only. '
                 'Ordinary fitted predictions for training; no test rows used in either fit. No retuning.',
        betting='EV >0% and >2%; identical flat and existing 1-5-unit stake rules. Draws lose. '
                'The models may select different bets from the same fixtures.',
        caveat='2022-23 is previously inspected and influenced the new model specification. This is a '
               'controlled retrospective diagnostic, not an untouched evaluation of model selection. '
               'Both features and forest regularization differ, so their individual effects are not isolated. '
               'Historical rating publication timestamps remain unverified.',
        reason_for_split='The 2025-26 prepared table lacks the old specification\'s history and rating inputs. '
                         'Use the common complete historical split rather than fabricate missing old inputs.',
        input_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    print('Refit OLD on the common 8,521 training matches',flush=True)
    old=fit_old(train,old_reference['columns'],'RF_binary',trees=500)
    print('Refit NEW on the same 8,521 training matches',flush=True)
    new=fit_model(train,new_reference['config'])
    predictions=[];quality=[];ledgers=[];returns=[];probabilities={}
    for name,bundle,infer in [('old',old,lambda b,f:predict_old(b,f)[:,1]),('new',new,predict_raw)]:
        joblib.dump(bundle,OUT/f'{name}_same_split.joblib')
        for split,frame in [('train',train),('test',test)]:
            ph=infer(bundle,frame)
            np.testing.assert_allclose(ph[:20],infer(bundle,frame.head(20).drop(columns=['home_score','away_score'])))
            np.testing.assert_allclose(ph[:20],infer(joblib.load(OUT/f'{name}_same_split.joblib'),frame.head(20)))
            probabilities[name,split]=ph
            quality.append(dict(model=name,split=split,**score(ph,frame)))
            cols=['source_row','season','fixture_date','league','home_team_name','away_team_name',
                  'home_score','away_score','B365H','B365D','B365A']
            rows=frame[cols].copy()
            rows['p_home_win']=ph;rows['model']=name;rows['split']=split
            rows['match_id']=rows.season.astype(str)+':'+rows.source_row.astype(str)
            rows['trained_through']=train.fixture_date.max()
            predictions.append(rows)
            ledger,summary=portfolios(rows,name)
            verify_settlements(ledger)
            ledgers.append(ledger)
            returns.extend(dict(model=name,split=split,**r) for r in summary)
    # The old refit must reproduce the originally reported saved model exactly.
    for split,frame in [('train',train),('test',test)]:
        np.testing.assert_allclose(probabilities['old',split],predict_old(old_reference,frame)[:,1],atol=1e-12)
        quality.append(dict(model='market',split=split,**score(market_probability(frame),frame)))
    # This new refit must reproduce its earlier pre-2022 forward fit, not its 2025 artifact.
    earlier=pd.read_csv(BASE/'regularized/candidate_predictions.csv.gz')
    earlier=earlier[(earlier.season==2022)&(earlier.model=='rf_positions_d2_leaf100__raw')].set_index('source_row')
    np.testing.assert_allclose(probabilities['new','test'],earlier.loc[test.source_row,'p_home_win'],atol=1e-12)
    forecasts=pd.concat(predictions,ignore_index=True)
    for split in ['train','test']:
        left=forecasts[(forecasts.model=='old')&(forecasts.split==split)].reset_index(drop=True)
        right=forecasts[(forecasts.model=='new')&(forecasts.split==split)].reset_index(drop=True)
        pd.testing.assert_frame_equal(left.drop(columns=['model','p_home_win']),right.drop(columns=['model','p_home_win']))
    forecasts.to_csv(OUT/'predictions.csv',index=False)
    ledger=pd.concat(ledgers,ignore_index=True);ledger.to_csv(OUT/'ledger.csv.gz',index=False)
    returns=pd.DataFrame(returns);returns.to_csv(OUT/'returns.csv',index=False)
    quality=pd.DataFrame(quality);quality.to_csv(OUT/'metrics.csv',index=False)
    gaps=[]
    for name in ['old','new','market']:
        tr=quality[(quality.model==name)&(quality.split=='train')].iloc[0]
        te=quality[(quality.model==name)&(quality.split=='test')].iloc[0]
        gaps.append(dict(model=name,train_log_loss=tr.log_loss,test_log_loss=te.log_loss,
            log_loss_gap=te.log_loss-tr.log_loss,train_brier=tr.brier,test_brier=te.brier,
            brier_gap=te.brier-tr.brier,train_accuracy=tr.accuracy,test_accuracy=te.accuracy,
            accuracy_gap=tr.accuracy-te.accuracy))
    pd.DataFrame(gaps).to_csv(OUT/'probability_gaps.csv',index=False)
    matched=returns[returns.split=='train'].merge(returns[returns.split=='test'],
        on=['model','strategy','staking'],suffixes=('_train','_test'),validate='one_to_one')
    matched['roi_gap_percentage_points']=(matched.roi_train-matched.roi_test)*100
    matched.to_csv(OUT/'betting_gaps.csv',index=False)
    # Paired four-week block bootstrap of the test log-loss difference, new minus old.
    y=target(test)
    losses={name:-(y*np.log(probabilities[name,'test'])+(1-y)*np.log1p(-probabilities[name,'test'])) for name in ['old','new']}
    dates=pd.to_datetime(test.fixture_date)
    weeks=pd.period_range(dates.min().to_period('W'),dates.max().to_period('W'),freq='W')
    rows=pd.DataFrame(dict(week=dates.dt.to_period('W'),difference=losses['new']-losses['old'],count=1))
    weekly=rows.groupby('week')[['difference','count']].sum().reindex(weeks,fill_value=0).to_numpy()
    rng=np.random.default_rng(20261110)
    starts=rng.integers(0,len(weeks),(5000,int(np.ceil(len(weeks)/4))))
    ids=((starts[:,:,None]+np.arange(4))%len(weeks)).reshape(5000,-1)[:,:len(weeks)]
    total=weekly[ids].sum(axis=1);boot=total[:,0]/total[:,1]
    uncertainty=dict(metric='Test binary log loss: new minus old; negative favors new',
        difference=float(rows.difference.mean()),lower_95=float(np.quantile(boot,.025)),upper_95=float(np.quantile(boot,.975)),
        limitation='Descriptive paired interval; does not account for model-selection exposure to this test season.')
    (OUT/'paired_test_difference.json').write_text(json.dumps(uncertainty,indent=2),encoding='utf-8')
    (OUT/'verification.json').write_text(json.dumps(dict(training_rows_identical=True,test_rows_identical=True,
        old_refit_reproduces_original=True,new_refit_reproduces_earlier_forward_fit=True,
        outcome_free_inference=True,saved_model_roundtrip=True,settlements_checked=len(ledger)),indent=2),encoding='utf-8')
    print(pd.DataFrame(gaps).to_string(index=False))
    print(matched[['model','strategy','staking','bets_train','net_train','roi_train','bets_test','net_test','roi_test','roi_gap_percentage_points']].to_string(index=False))
    print(json.dumps(uncertainty,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
