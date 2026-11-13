"""Tests for preprocessing isolation, temporal calibration and selection."""
import unittest
from select_expanded_features import np,pd,threadpool_limits
from regularized_home_win import (configs,features,fit_model,predict_raw,fit_calibrator,
    apply_calibrator,select_model,choose_threshold)


class RegularizationTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(dict(B365H=[2.,3.,1.5]*100,B365D=[3.,3.5,4.]*100,B365A=[3.,2.,6.]*100,
            HomePlayer_Overall_mean=[70.,80.,75.]*100,AwayPlayer_Overall_mean=[72.,74.,79.]*100,
            home_GD_form=[1,2,3]*100,away_GD_form=[3,2,1]*100,home_Points_form=[5,7,9]*100,
            away_Points_form=[9,5,7]*100,Matchweek=[4,5,6]*100,home_score=[2,0,1]*100,away_score=[0,1,1]*100))

    def test_features_and_predictions_do_not_read_scores(self):
        f=self.frame();cfg=configs()['logit_core_C0.01'];model=fit_model(f,cfg)
        p=predict_raw(model,f.head())
        np.testing.assert_allclose(p,predict_raw(model,f.head().drop(columns=['home_score','away_score'])))
        medians=model['pipeline']['imputer'].statistics_.copy()
        changed=f.head().copy();changed['HomePlayer_Overall_mean']=9999
        predict_raw(model,changed)
        np.testing.assert_array_equal(medians,model['pipeline']['imputer'].statistics_)

    def test_forest_is_strongly_regularized(self):
        model=fit_model(self.frame(),configs()['rf_core_d2_leaf100'])
        forest=model['pipeline']['model']
        self.assertEqual(forest.max_depth,2);self.assertEqual(forest.min_samples_leaf,100)
        self.assertTrue(all(tree.tree_.max_depth<=2 for tree in forest.estimators_))

    def test_calibration_rejects_current_or_in_sample_predictions(self):
        rows=pd.DataFrame(dict(raw_probability=[.2,.4,.6,.8],target=[0,1,0,1],
            fixture_date=['2020-02-01']*4,trained_through=['2019-06-01']*4))
        cal=fit_calibrator(rows,'2021-01-01')
        p=apply_calibrator(cal,np.array([.1,.9]));self.assertTrue(((p>0)&(p<1)).all())
        with self.assertRaises(ValueError):fit_calibrator(rows,'2020-02-01')
        rows['trained_through']='2020-03-01'
        with self.assertRaises(ValueError):fit_calibrator(rows,'2021-01-01')

    def test_future_scores_cannot_change_selection(self):
        records=[dict(season=y,model=m,log_loss=v) for y in [2016,2017] for m,v in [('a',.6),('b',.7)]]
        before=select_model(pd.DataFrame(records),2018)
        records += [dict(season=2018,model='a',log_loss=10),dict(season=2018,model='b',log_loss=0)]
        self.assertEqual(before,select_model(pd.DataFrame(records),2018))

    def test_threshold_abstains_without_historical_support(self):
        f=pd.DataFrame(dict(season=[2020]*100+[2021]*100,p_home_win=[.6]*200,B365H=[2.]*200,target=[0]*200))
        self.assertIsNone(choose_threshold(f,2022)[0])
        f['target']=1
        threshold,_=choose_threshold(f,2022)
        self.assertEqual(threshold,0.)
        f=pd.concat([f,pd.DataFrame(dict(season=[2022]*100,p_home_win=[.99]*100,B365H=[10.]*100,target=[0]*100))])
        self.assertEqual(choose_threshold(f,2022)[0],threshold)


if __name__=='__main__':
    with threadpool_limits(limits=4):unittest.main()
