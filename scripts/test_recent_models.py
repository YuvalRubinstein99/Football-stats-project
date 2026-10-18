import sys,unittest,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.lineup_deps'));sys.path.insert(0,str(ROOT/'src'))
from train_recent_models import weights,np,pd
from compare_ordered_models import objective
from scipy.optimize import check_grad

class RecentModels(unittest.TestCase):
    def test_decay(self):
        y=np.array([2008,2015,2018,2021]);w=weights(y,3)
        self.assertAlmostEqual(w.mean(),1)
        self.assertAlmostEqual(w[-2]/w[-1],.5)
        self.assertAlmostEqual(w[-3]/w[-1],.25)
        self.assertTrue(np.all(np.diff(w)>0))
        np.testing.assert_array_equal(weights(y,None),np.ones(4))
        with self.assertRaises(ValueError):weights(y,0)
    def test_ordered_weighted_gradient(self):
        rng=np.random.default_rng(26);x=rng.normal(size=(30,4));y=np.arange(30)%3
        theta=np.r_[rng.normal(0,.1,4),-.5,.2];w=np.linspace(.2,2,30)
        for link in ['logit','probit']:
            err=check_grad(lambda t:objective(t,x,y,link,w)[0],lambda t:objective(t,x,y,link,w)[1],theta)
            self.assertLess(err,1e-6)
            a=objective(theta,x,y,link);b=objective(theta,x,y,link,np.ones(30))
            self.assertAlmostEqual(a[0],b[0]);np.testing.assert_allclose(a[1],b[1])
    def test_artifact_selection_and_inference(self):
        from model_strategy_app import infer,view
        from probability_calibration import calibrate
        from live_bets import SEARCH_CANDIDATE
        summary=json.loads((ROOT/'results/recency/summary.json').read_text())
        cv=pd.read_csv(ROOT/'results/recency/validation.csv');expected=pd.read_csv(ROOT/'results/recency/match_predictions.csv')
        self.assertEqual(set(cv.season),{2019,2020})
        for name,info in summary['models'].items():
            losses=cv[cv.model==name].groupby('half_life').log_loss.mean()
            self.assertAlmostEqual(info['cv_log_loss'],losses.min())
            recent=name+'__recent';p,g=infer(recent,expected.iloc[173].to_dict())
            np.testing.assert_allclose(p,expected.iloc[173][[recent+'__p_'+s for s in ['home','draw','away']]].to_numpy(float),atol=1e-8)
            pc,_=infer(recent+'__calibrated',expected.iloc[173].to_dict())
            np.testing.assert_allclose(pc,calibrate(recent,p),atol=1e-8)
        result=view('rf',True,SEARCH_CANDIDATE,dict(league='D1'),recent=True)
        self.assertTrue(result['recent']);self.assertIn('home_GD_form',result['features'])
        self.assertEqual({r['split'] for r in result['summary']},{'Train (in-sample)','Test'})
        self.assertTrue(all(r['league']=='D1' for r in result['matches']))
        for curve,total in zip(result['curves'],result['summary']):self.assertAlmostEqual(curve['points'][-1]['net'],total['net'])

if __name__=='__main__':unittest.main()
