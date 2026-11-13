"""Target semantics, preprocessing isolation and direct binary inference checks."""
import tempfile
import unittest
from pathlib import Path
from select_expanded_features import np,pd,joblib,threadpool_limits
from home_win_model import target,fit,predict
from train_binary_home_win import metrics


class BinaryHomeWinTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(dict(home_team_name=['A','B','C']*10,away_team_name=['D','E','F']*10,
            home_score=[2,1,0]*10,away_score=[0,1,2]*10,gap=[1.,np.nan,3.]*10))

    def test_draw_and_away_are_both_nonwins(self):
        np.testing.assert_array_equal(target(self.frame())[:3],[1,0,0])
        invalid=self.frame();invalid.loc[0,'home_score']=np.nan
        with self.assertRaises(ValueError):
            target(invalid)

    def test_direct_binary_fit_and_missing_values(self):
        frame=self.frame()
        columns=['home_team_name','away_team_name','gap']
        for family in ['Logistic','RF_binary']:
            bundle=fit(frame,columns,family,trees=10,regularization_c=.1)
            np.testing.assert_array_equal(bundle['model'].classes_,[0,1])
            self.assertEqual(bundle['imputer'].statistics_[0],2.)
            query=frame.head(3).copy()
            query['gap']=[np.nan,5.,6.]
            query['home_team_name']='Unseen'
            p=predict(bundle,query)
            self.assertEqual(p.shape,(3,2))
            self.assertTrue(np.isfinite(p).all() and np.allclose(p.sum(axis=1),1))
            self.assertEqual(bundle['imputer'].statistics_[0],2.)
            # Inference must not read final scores.
            np.testing.assert_allclose(p,predict(bundle,query.drop(columns=['home_score','away_score'])))
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'model.joblib';joblib.dump(bundle,path)
                np.testing.assert_allclose(p,predict(joblib.load(path),query))

    def test_target_columns_are_rejected(self):
        with self.assertRaises(ValueError):
            fit(self.frame(),['gap','home_score'],'Logistic')
        with self.assertRaises(ValueError):
            fit(self.frame(),['gap','gap'],'Logistic')

    def test_metrics_use_home_probability_and_home_on_ties(self):
        frame=self.frame().head(3)
        p=np.array([[.5,.5],[.8,.2],[.6,.4]])
        m=metrics(p,frame)
        self.assertEqual(m['correct'],3)
        self.assertEqual(m['home_picks'],1)
        self.assertEqual(m['no_home_picks'],2)


if __name__=='__main__':
    with threadpool_limits(limits=4):
        unittest.main(verbosity=2)
