import unittest
from select_expanded_features import np
from binary_betting import selections,settle,summary
from stake_levels import size_bet


class StrategyTests(unittest.TestCase):
    def test_draws_are_retained_and_lose(self):
        ph=np.array([.7,.3,.3]);odds=np.tile([2.,3.,4.],(3,1))
        choice=selections(ph,odds)['Most likely side']
        r=settle(ph,odds,[0,1,2],choice)
        np.testing.assert_array_equal(r['pick'],[0,2,2])
        np.testing.assert_allclose(r['profit'],[1,-1,3])
        m=summary(r,[0,1,2])
        self.assertEqual(m['bets'],3);self.assertEqual(m['wins'],2);self.assertEqual(m['draw_losses'],1)

    def test_home_edge_rule_skips_before_results(self):
        ph=np.array([.6,.3,.5]);odds=np.tile([2.,3.,4.],(3,1))
        choice=selections(ph,odds)['Home only: positive expected return']
        np.testing.assert_array_equal(choice['active'],[True,False,False])
        self.assertFalse(choice['proxy'])
        r=settle(ph,odds,[1,0,0],choice)
        np.testing.assert_allclose(r['profit'],[-1,0,0])
        self.assertEqual(summary(r,[1,0,0])['skips'],2)

    def test_no_draw_selection_and_legacy_tier_parity(self):
        ph=np.array([.504,.511,.53,.7,.1]);odds=np.tile([2.,3.,4.],(5,1))
        for strategy,choice in selections(ph,odds).items():
            self.assertTrue(np.isin(choice['pick'],[0,2]).all())
            r=settle(ph,odds,[0,1,2,0,1],choice,'tiers')
            for i in range(len(ph)):
                pick=choice['pick'][i]
                old=size_bet(dict(action='Bet' if choice['active'][i] else 'Skip',decimal_odds=r['price'][i],
                    expected_net_per_unit=r['edge'][i],home_stake=float(pick==0),away_stake=float(pick==2),
                    draw_stake=0.,note='test'))
                self.assertEqual(r['stakes'][i],old['stake_dollars'],strategy)
            self.assertTrue(np.all(r['profit'][~r['bet']]==0))

    def test_empty_strategy_has_zero_stake_and_no_roi(self):
        ph=np.array([.1]);odds=np.array([[2.,3.,4.]])
        choice=selections(ph,odds)['Home only: positive expected return']
        m=summary(settle(ph,odds,[1],choice),[1])
        self.assertEqual(m['net'],0);self.assertEqual(m['bets'],0);self.assertTrue(np.isnan(m['roi']))


if __name__=='__main__':
    unittest.main(verbosity=2)
