import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import model_strategy_app
from bankroll_backtest import stakes,simulate,eligible,np,METHODS
import pandas as pd

class Bankroll(unittest.TestCase):
    def test_formula_and_robust_abstention(self):
        p=[[.6,.2,.2]];o=[[2,3,3]]
        self.assertAlmostEqual(stakes(p,o,'Quarter Kelly',1000)[1][0],20) # 2% cap
        p=[[.51,.245,.245]]
        self.assertAlmostEqual(stakes(p,o,'Quarter Kelly',1000)[1][0],5)
        self.assertEqual(stakes(p,o,'Robust quarter Kelly',1000)[1][0],0)
        self.assertEqual(stakes([[.4,.3,.3]],[[2,3,3]],'Flat $1')[1][0],0)
    def test_same_day_exposure_and_settlement(self):
        f=pd.DataFrame([dict(fixture_date='2020-01-01',home_team_name='A',away_team_name='B',league='E0',B365H=2,B365D=3,B365A=3,home_score=1,away_score=0)]*10)
        p=np.tile([.6,.2,.2],(10,1))
        result,rows=simulate(f,p,'Quarter Kelly')
        self.assertAlmostEqual(sum(r['stake'] for r in rows),100)
        self.assertTrue(all(r['opening_bankroll']==1000 for r in rows))
        self.assertAlmostEqual(result['net'],100)
        self.assertAlmostEqual(result['points'][-1]['net'],result['net'])
        f.home_score=0;f.away_score=1
        loss,lrows=simulate(f,p,'Quarter Kelly')
        self.assertEqual([r['stake'] for r in rows],[r['stake'] for r in lrows])
        self.assertAlmostEqual(loss['max_drawdown'],.1)
    def test_robust_never_exceeds_quarter(self):
        rng=np.random.default_rng(42);p=rng.dirichlet([2,2,2],100);o=rng.uniform(1.1,12,(100,3))
        # Without a shared-day cap binding, robust stakes cannot exceed nominal stakes.
        for a,b in zip(p,o):self.assertLessEqual(stakes([a],[b],METHODS[2])[1][0],stakes([a],[b],METHODS[1])[1][0]+1e-10)
    def test_expected_return_and_decimal_odds_are_separate_filters(self):
        np.testing.assert_array_equal(
            eligible(np.array([.51,.51001,.26,.26]),np.array([2.,2.,4.,4.]),.02,1),
            [False,True,True,True])
        np.testing.assert_array_equal(
            eligible(np.array([.6,.26,.26]),np.array([2.,4.,4.1]),.02,4.,4.),
            [False,True,False])
        # The nominal 4% edge passes the return filter, but robust Kelly abstains.
        p=[[.26,.37,.37]];o=[[4.,2.,2.]]
        self.assertGreater(stakes(p,o,'Quarter Kelly',min_edge=.02,min_odds=4)[1][0],0)
        self.assertEqual(stakes(p,o,'Robust quarter Kelly',min_edge=.02,min_odds=4)[1][0],0)

    def test_filter_excludes_selection_without_substitution(self):
        # Home has the highest EV but fails minimum odds. Away passes both
        # thresholds; the browser's filter semantics still skip the fixture.
        for method in METHODS:
            side,amount=stakes([[.6,.1,.3]],[[2.,3.,4.]],method,min_edge=.02,min_odds=4.)
            self.assertEqual(side[0],0)
            self.assertEqual(amount[0],0)

    def test_filtered_simulation_gates_before_daily_exposure(self):
        f=pd.DataFrame([dict(fixture_date='2020-01-01',home_team_name='A',away_team_name='B',
                            league='E0',B365H=2,B365D=3,B365A=3,home_score=1,away_score=0)]*20)
        p=np.array([[.51,.245,.245]]*10+[[.6,.2,.2]]*10)
        for method in METHODS:
            _,expected=stakes(p,f[['B365H','B365D','B365A']],method,min_edge=.02)
            result,rows=simulate(f,p,method,min_edge=.02)
            np.testing.assert_allclose([r['stake'] for r in rows],expected)
            np.testing.assert_array_equal(expected[:10],0)
            self.assertEqual(result['bets'],10)
            self.assertAlmostEqual(result['net'],expected.sum())

if __name__=='__main__':unittest.main()
