"""Independently verify temporal selection, frozen inference and all bet settlements."""
import hashlib
import json
from select_expanded_features import ROOT,np,pd,joblib
from train_regularized_home_win import OUT
from prepare_regularized_holdout import DEST
from regularized_home_win import select_model,choose_threshold,predict_bundle


def verify_ledger(frame,policy=False):
    edge=frame.p_home_win*frame.B365H-1
    cutoff=frame.cutoff.to_numpy(float) if policy else np.where(frame.strategy.str.endswith('>2%'),.02,0.)
    active=np.isfinite(cutoff)&(edge>cutoff)
    tier=np.clip(np.ceil(np.round(edge/(frame.B365H-1)*100,10)),1,5)
    stake=np.where(active,np.where(frame.staking.eq('flat'),1.,tier),0.)
    net=stake*np.where(frame.home_score>frame.away_score,frame.B365H-1,-1.)
    np.testing.assert_array_equal(stake,frame.stake)
    np.testing.assert_allclose(net,frame.net,atol=1e-11)


def main():
    plan=json.loads((OUT/'plan.json').read_text())
    frozen=json.loads((OUT/'frozen_selection.json').read_text())
    for path,digest in plan['input_hashes'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    assert hashlib.sha256((OUT/'frozen_model.joblib').read_bytes()).hexdigest()==frozen['model_sha256']
    assert hashlib.sha256((OUT/'plan.json').read_bytes()).hexdigest()==frozen['plan_sha256']
    predictions=pd.read_csv(OUT/'candidate_predictions.csv.gz')
    assert not predictions.duplicated(['model','match_id']).any()
    assert (pd.to_datetime(predictions.trained_through)<pd.to_datetime(predictions.fixture_date)).all()
    calibrated=predictions[predictions.model.str.endswith('__sigmoid')]
    assert (pd.to_datetime(calibrated.calibration_last_date)<pd.to_datetime(calibrated.fixture_date)).all()
    validation=pd.read_csv(OUT/'validation.csv')
    decisions=json.loads((OUT/'decisions.json').read_text())
    selected=pd.read_csv(OUT/'selected_forward_predictions.csv')
    for decision in decisions:
        year=decision['season'];name,loss=select_model(validation,year)
        assert name==decision['model'] and np.isclose(loss,decision['validation_log_loss'])
        cutoff,_=choose_threshold(predictions[predictions.model==name],year)
        assert cutoff==decision['threshold']
        expected=predictions[(predictions.model==name)&(predictions.season==year)].set_index('match_id')
        actual=selected[selected.season==year].set_index('match_id')
        assert set(expected.index)==set(actual.index)
        np.testing.assert_allclose(expected.loc[actual.index,'p_home_win'],actual.p_home_win)
    name,loss=select_model(validation,2023)
    assert name==frozen['model'] and np.isclose(loss,frozen['validation_log_loss'])
    cutoff,_=choose_threshold(predictions[predictions.model==name],2023)
    assert cutoff==frozen['threshold']
    holdout=pd.read_csv(DEST/'matches.csv')
    forecast=pd.read_csv(DEST/'predictions.csv').set_index('source_row')
    model=joblib.load(OUT/'frozen_model.joblib')
    np.testing.assert_allclose(predict_bundle(model,holdout),forecast.loc[holdout.source_row,'p_home_win'],atol=1e-12)
    assert model['trained_through']<holdout.fixture_date.min()
    audit=pd.read_csv(DEST/'rating_audit.csv')
    dates=audit.rating_date.dropna()
    assert (pd.to_datetime(dates)<pd.to_datetime(holdout.fixture_date).min()).all()
    receipts=json.loads((DEST/'data_audit.json').read_text())
    for path,digest in receipts['input_hashes'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    total=0
    for directory,prefix in [(OUT,'replay_'),(OUT,'in_sample_'),(DEST,'')]:
        ledger=pd.read_csv(directory/f'{prefix}ledger.csv.gz',low_memory=False)
        verify_ledger(ledger);total+=len(ledger)
        summaries=pd.read_csv(directory/f'{prefix}summary.csv')
        for row in summaries.itertuples():
            group=ledger[(ledger.protocol==row.protocol)&(ledger.strategy==row.strategy)&(ledger.staking==row.staking)]
            assert len(group)==row.matches
            assert np.isclose(group.net.sum(),row.net)
            assert np.isclose(group.stake.sum(),row.staked)
            assert int(group.stake.gt(0).sum())==row.bets
    policy=pd.read_csv(OUT/'threshold_policy_ledger.csv.gz');verify_ledger(policy,policy=True);total+=len(policy)
    assert (policy[policy.period=='historical_replay'].stake==0).all()
    curves=pd.read_csv(DEST/'daily_curves.csv')
    totals=pd.read_csv(DEST/'summary.csv')
    for row in totals.itertuples():
        c=curves[(curves.strategy==row.strategy)&(curves.staking==row.staking)].sort_values('date')
        assert np.isclose(c.cumulative_net.iloc[-1],row.net)
    print(f'Passed: {len(predictions):,} candidate forecasts, chronological choices, calibration dates, '
          f'frozen holdout inference, rating dates, {total:,} settlements and graph endpoints.')


if __name__=='__main__':main()
