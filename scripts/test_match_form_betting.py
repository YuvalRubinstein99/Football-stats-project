"""Hand-calculated settlement checks, independent of historical profitability."""
import unittest
from backtest_match_form import np, pd, RULES, settle_arrays, summarize


class BettingTests(unittest.TestCase):
    def fixtures(self):
        return pd.DataFrame([
            dict(home_team_name='Home', away_team_name='Away',
                 home_score=h, away_score=a, B365H=2., B365D=4., B365A=4.)
            for h, a in [(1, 0), (0, 0), (0, 1)]
        ])

    def test_single_outcome_stakes_and_net(self):
        frame = self.fixtures()
        result = settle_arrays(frame, [[.6, .2, .2]] * 3)
        j = RULES.index('Most likely outcome')
        np.testing.assert_array_equal(result['unit_return'][:, j], [2, 0, 0])
        np.testing.assert_array_equal(result['tier_stake'][:, j], [5, 5, 5])
        stat, _ = summarize(['a', 'b', 'c'], result['tier_stake'][:, j],
                            result['unit_return'][:, j])
        self.assertEqual(stat['total_staked'], 15)
        self.assertEqual(stat['gross_return'], 10)
        self.assertEqual(stat['net_profit'], -5)
        self.assertAlmostEqual(stat['roi'], -1 / 3)

    def test_synthetic_double_chance_is_one_total_stake(self):
        result = settle_arrays(self.fixtures(), [[.6, .2, .2]] * 3)
        j = RULES.index('Home loses vs not (synthetic double chance)')
        np.testing.assert_allclose(result['home_weight'][:, j], 2 / 3)
        np.testing.assert_allclose(result['draw_weight'][:, j], 1 / 3)
        np.testing.assert_array_equal(result['away_weight'][:, j], 0)
        np.testing.assert_allclose(result['unit_return'][:, j], [4 / 3, 4 / 3, 0])
        stat, _ = summarize(['a', 'b', 'c'], result['original_stake'][:, j],
                            result['unit_return'][:, j])
        self.assertEqual(stat['total_staked'], 3)
        self.assertAlmostEqual(stat['net_profit'], -1 / 3)

    def test_positive_edge_gate_does_not_change_original_rule(self):
        frame = self.fixtures()
        frame['B365H'] = 1.8
        result = settle_arrays(frame, [[.5, .25, .25]] * 3)
        j = RULES.index('Most likely outcome')
        np.testing.assert_array_equal(result['original_stake'][:, j], 1)
        np.testing.assert_array_equal(result['eligible_stake'][:, j], 0)
        np.testing.assert_array_equal(result['tier_stake'][:, j], 0)

    def test_drawdown_includes_starting_zero_and_nets_same_day(self):
        stat, curve = summarize(['a', 'a', 'b', 'c'], [1, 1, 2, 1], [0, 0, 3, 0])
        np.testing.assert_array_equal(curve.cumulative_net, [-2, 2, 1])
        self.assertEqual(stat['max_daily_drawdown_dollars'], 2)
        self.assertEqual(stat['max_daily_exposure'], 2)
        self.assertEqual(stat['net_profit'], 1)

    def test_invalid_probabilities_are_rejected(self):
        with self.assertRaises(ValueError):
            settle_arrays(self.fixtures(), [[.6, .3, .3]] * 3)


if __name__ == '__main__':
    unittest.main()
