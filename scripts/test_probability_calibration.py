import sys,unittest,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.lineup_deps'));sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from probability_calibration import scale,fit_temperature,artifacts,calibrate,SUFFIX

class CalibrationTests(unittest.TestCase):
    def test_temperature_and_normalization(self):
        p=np.array([[.7,.2,.1],[0,1,0],[.1,.3,.6]])
        np.testing.assert_allclose(scale(p,1),p,atol=1e-14)
        self.assertLess(scale(p,2)[0,0],p[0,0])
        self.assertGreater(scale(p,.5)[0,0],p[0,0])
        np.testing.assert_allclose(scale(p,2).sum(axis=1),1)
        with self.assertRaises(ValueError):scale(p,0)
        with self.assertRaises(ValueError):scale([[.2,.2,.2]],1)
    def test_recovers_known_temperature(self):
        rng=np.random.default_rng(47);p=rng.dirichlet([1,1,1],20000)
        truth=scale(p,1.7);y=np.array([rng.choice(3,p=row) for row in truth])
        self.assertAlmostEqual(fit_temperature(p,y),1.7,delta=.08)
    def test_saved_forward_fit_and_integration(self):
        from model_strategy_app import pd,infer,view
        from model_filters import config,historical
        from live_bets import SEARCH_CANDIDATE
        bundle=artifacts();forward=pd.read_csv(ROOT/'results/calibration/forward_predictions.csv')
        test=pd.read_csv(ROOT/'data/expanded_2008/test.csv')
        for name,info in bundle['models'].items():
            rows=forward[forward.model==name]
            self.assertEqual(set(rows.season),{2019,2020,2021})
            p=rows[['p_home','p_draw','p_away']].to_numpy();y=rows.outcome.to_numpy()
            self.assertAlmostEqual(fit_temperature(p,y),info['temperature'],places=6)
            mask=rows.season<2021
            self.assertAlmostEqual(fit_temperature(p[mask],y[mask]),info['audit_temperature'],places=6)
            raw,g=infer(name,test.iloc[0].to_dict());cal,gc=infer(name+SUFFIX,test.iloc[0].to_dict())
            np.testing.assert_allclose(cal,calibrate(name,raw))
            if g is None:self.assertIsNone(gc)
            else:np.testing.assert_allclose(g,gc,atol=1e-12)
        result=view('rf',True,SEARCH_CANDIDATE,dict(league='E0'),calibrated=True)
        self.assertTrue(result['calibrated'])
        self.assertEqual({r['split'] for r in result['summary']},{'Train (in-sample)','Test'})
        for curve,total in zip(result['curves'],result['summary']):
            self.assertAlmostEqual(curve['points'][-1]['net'],total['net'])
        for match in result['matches']:
            if 'probabilities' in match:self.assertAlmostEqual(sum(match['probabilities']),1)

if __name__=='__main__':unittest.main()
