"""Check original-row preservation, fold boundaries and bankroll accounting."""
import json
from select_expanded_features import ROOT,pd,np
from evaluate_transfermarkt_models import OUT,EXTRA,impute

for split in ['train','test']:
    original=pd.read_csv(ROOT/f'results/classifier/{split}.csv')
    prepared=pd.read_csv(OUT/f'{split}.csv')
    assert len(original)==len(prepared)
    pd.testing.assert_frame_equal(original,prepared[original.columns])
    assert np.array_equal(prepared.source_row,np.arange(len(original)))
for row in json.loads((OUT/'fold_audit.json').read_text()):
    assert row['train_last']<row['evaluation_first']
    train=pd.read_csv(OUT/'train.csv')
    if row['year']<2022:train=train[(train.season<row['year'])&(train.fixture_date<row['evaluation_first'])]
    for col,median in row['imputation_medians'].items():assert np.isclose(train[col].median(),median)
bank=pd.read_csv(OUT/'bankroll.csv');metrics=pd.read_csv(OUT/'metrics.csv')
keys=['split','model','variant','league','method']
assert not bank.duplicated(keys).any()
for _,g in bank.groupby(['split','model','league','method']):
    assert set(g.variant)=={'Baseline','With values'} and g.matches.nunique()==1
assert (bank.ending_bankroll>=0).all() and bank.max_drawdown.between(0,1).all()
assert np.allclose(bank.net+1000,bank.ending_bankroll)
curves=json.loads((OUT/'curves.json').read_text())
assert len(curves)==len(bank)==900
indexed={tuple(r[k] for k in keys):r for r in bank.to_dict('records')}
for r in curves:
    summary=indexed[tuple(r[k] for k in keys)]
    assert r['points'][0]==dict(date='Start',net=0.)
    assert np.isclose(r['points'][-1]['net'],summary['net'])
assert len(metrics)==300 and np.isfinite(metrics[['log_loss','brier','auc']]).all().all()
print('Passed: original rows and features preserved; imputation is train-only; strict forward folds; 900 paired bankroll curves reconcile.')
