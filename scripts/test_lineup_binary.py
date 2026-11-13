import unittest
from evaluate_lineup_binary import settle,np


class BinarySettlementTests(unittest.TestCase):
    def test_draw_is_covered_only_by_hedge(self):
        p=np.tile([.4,.3,.3],(3,1))
        odds=np.tile([2.,3.,6.],(3,1))
        result=settle(p,odds,[0,1,2])
        hedge,straight=result['away_or_draw'],result['straight_away']
        np.testing.assert_allclose(hedge['stakes'],np.tile([0,2/3,1/3],(3,1)))
        np.testing.assert_allclose(hedge['profit'],[-1,1,1])
        np.testing.assert_allclose(straight['profit'],[-1,-1,5])
        np.testing.assert_array_equal(hedge['correct'],[False,True,True])
        np.testing.assert_array_equal(straight['won'],[False,False,True])

    def test_home_tie_and_all_three_outcomes(self):
        result=settle(np.tile([.5,.2,.3],(3,1)),np.tile([2.,3.,6.],(3,1)),[0,1,2])
        for r in result.values():
            self.assertTrue(r['home'].all())
            np.testing.assert_allclose(r['profit'],[1,-1,-1])

    def test_invalid_probabilities_or_odds_rejected(self):
        with self.assertRaises(ValueError):
            settle([[.6,.3,.3]],[[2,3,4]],[0])
        with self.assertRaises(ValueError):
            settle([[.4,.3,.3]],[[2,np.nan,4]],[0])


if __name__=='__main__':
    unittest.main()
