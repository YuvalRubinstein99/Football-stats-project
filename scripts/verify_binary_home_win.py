"""Reconcile direct binary exports, temporal selection and saved classifiers."""
import json
import hashlib
from select_expanded_features import ROOT,np,pd,joblib,threadpool_limits
from compare_lineup_improvements import add_controls
from home_win_model import predict,target
from train_binary_home_win import metrics,OUT,DATA


def main():
    plan=json.loads((OUT/'plan.json').read_text())
    for name,digest in plan['input_hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    p=pd.read_csv(OUT/'predictions.csv')
    table=pd.read_csv(OUT/'metrics.csv')
    assert len(p)==34080 and not p.duplicated(['split','family','candidate','source_row']).any()
    for (split,family,candidate),g in p.groupby(['split','family','candidate']):
        probs=g[['p_no_home_win','p_home_win']].to_numpy()
        assert len(g)==1065 and np.isfinite(probs).all() and (probs>=0).all()
        assert np.allclose(probs.sum(axis=1),1)
        assert np.array_equal(g.target,target(g))
        assert np.array_equal(g.prediction,(g.p_home_win>=.5).astype(int))
        row=table[(table.split==split)&(table.family==family)&(table.candidate==candidate)].iloc[0]
        for key,value in metrics(probs,g).items():
            assert np.isclose(row[key],value),(split,family,candidate,key)
    selected=json.loads((OUT/'selection.json').read_text())
    cv=pd.read_csv(OUT/'validation.csv')
    assert set(cv.season)=={2019,2020}
    test=add_controls(pd.read_csv(DATA/'test_features.csv'),'test')
    for family,choice in selected['selection'].items():
        mean=cv[cv.family==family].groupby(['candidate','C']).log_loss.mean()
        assert mean.idxmin()==(choice['candidate'],choice['C'])
        bundle=joblib.load(OUT/f'{family}_selected.joblib')
        assert bundle['model'].__class__.__name__ in ['LogisticRegression','RandomForestClassifier']
        assert bundle['training_matches']==8521
        assert pd.Timestamp(bundle['trained_through'])<pd.to_datetime(test.fixture_date).min()
        assert bundle['probability_columns']==['p_no_home_win','p_home_win']
        predicted=predict(bundle,test)
        saved=p[(p.split=='Test 2022-23')&(p.family==family)&(p.candidate==choice['candidate'])]
        aligned=test[['source_row']].merge(saved,on='source_row',validate='one_to_one',how='left')
        np.testing.assert_allclose(predicted,aligned[['p_no_home_win','p_home_win']].to_numpy())
    print('Verified 34,080 direct binary forecasts, metrics, validation-only selection and selected-model reload parity.')


if __name__=='__main__':
    with threadpool_limits(limits=4):
        main()
