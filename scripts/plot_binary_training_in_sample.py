"""Apply the saved final binary RF to its own training rows and plot betting profit."""
import hashlib
import json
from select_expanded_features import ROOT,np,pd,joblib,threadpool_limits
from compare_lineup_improvements import add_controls
from home_win_model import predict
from train_binary_home_win import metrics
from backtest_binary_training_oos import portfolios
from plot_binary_training_oos import render

DATA=ROOT/'results/lineup_improvements'
OUT=DATA/'binary_retrained/training_in_sample'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    model_path=DATA/'binary_retrained/RF_binary_selected.joblib'
    bundle=joblib.load(model_path)
    # Check the exact original experiment inputs before claiming training-set fit.
    for name,digest in bundle['experiment_plan']['input_hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Saved model input provenance changed: {name}')
    train=add_controls(pd.read_csv(DATA/'train_features.csv'),'train')
    assert len(train)==bundle['training_matches']==8521
    assert str(train.fixture_date.max())==bundle['trained_through']
    assert train.source_row.is_unique and set(train.season)==set(range(2014,2022))
    assert bundle['family']=='RF_binary' and bundle['selection']['candidate']=='fifa_prior'
    # Standard predict_proba, deliberately not OOB or cross-validated predictions.
    p=predict(bundle,train)
    cols=['source_row','season','fixture_date','league','home_team_name','away_team_name',
          'home_score','away_score','B365H','B365D','B365A']
    predictions=train[cols].copy()
    predictions['p_home_win']=p[:,1]
    predictions['p_no_home_win']=p[:,0]
    predictions['candidate']=bundle['selection']['candidate']
    predictions['protocol']='in_sample'
    predictions.to_csv(OUT/'predictions.csv',index=False)
    ledger,records=portfolios(predictions,'in_sample')
    totals=pd.DataFrame(records)
    # Independent reconciliation of all four filters/stake combinations.
    edge=ledger.p_home_win.to_numpy()*ledger.B365H.to_numpy()-1
    cutoff=np.where(ledger.strategy.str.endswith('>2%'),.02,0.)
    active=edge>cutoff
    tiers=np.clip(np.ceil(np.round(edge/(ledger.B365H.to_numpy()-1)*100,10)),1,5)
    stake=np.where(active,np.where(ledger.staking.eq('flat'),1.,tiers),0.)
    net=stake*np.where(ledger.home_score>ledger.away_score,ledger.B365H-1,-1.)
    np.testing.assert_array_equal(stake,ledger.stake)
    np.testing.assert_allclose(net,ledger.net,atol=1e-12)
    for row in totals.itertuples():
        mask=(ledger.strategy==row.strategy)&(ledger.staking==row.staking)
        assert int(mask.sum())==len(train)
        assert np.isclose(net[mask].sum(),row.net)
        assert np.isclose(stake[mask].sum(),row.staked)
        assert int((stake[mask]>0).sum())==row.bets
    ledger.to_csv(OUT/'ledger.csv.gz',index=False)
    totals.to_csv(OUT/'summary.csv',index=False)
    plan=dict(evaluation='IN-SAMPLE: saved final model predicts the exact rows it was fitted on. '
              'No refitting, no OOB probabilities, no held-out or forward predictions.',
              model=str(model_path.relative_to(ROOT)),model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
              candidate=bundle['selection']['candidate'],training_matches=len(train),
              trained_through=bundle['trained_through'],metrics=metrics(p,train),
              filters=['P(home win)*home odds - 1 > 0','P(home win)*home odds - 1 > 0.02'],
              settlement='Home only; draws and away wins lose. Existing flat and 1-5-unit stakes. '
                         'Daily chronological settlement, no compounding, fees or bankroll limits.',
              caveat='Descriptive training fit; optimistic as a measure of future betting performance.')
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    curves=render(ledger,totals,'in_sample',out=OUT,
        heading='Training data: in-sample cumulative profit',
        subtitle='Saved final binary RF predicts its own training matches | 2014-15 to 2021-22',
        note='Home-only filters: P(home win) x home odds - 1 > 0% (teal) or > 2% (purple).',
        footer='IN-SAMPLE: these results measure training fit, not unseen-match performance. Different vertical scales.')
    pd.DataFrame(curves).to_csv(OUT/'daily_curves.csv',index=False)
    print(totals.to_string(index=False))
    print(f'Checked {len(ledger):,} settlements and all four graph endpoints. Saved PNG, SVG and tables.')


if __name__=='__main__':
    with threadpool_limits(limits=4):
        main()
