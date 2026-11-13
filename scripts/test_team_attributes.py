import unittest
import numpy as np
import pandas as pd
from prepare_team_attributes import ATTRIBUTES, PlayerIndex, aggregate_side, comparisons
from compare_team_attributes import candidates, choose, fit_model


class TeamAttributeTests(unittest.TestCase):
    def test_identity_ambiguity_and_no_fuzzy_match(self):
        index = PlayerIndex(pd.DataFrame([
            dict(long_name='John Smith', short_name='J. Smith', club_name='Alpha'),
            dict(long_name='James Smith', short_name='J. Smith', club_name='Beta'),
        ]))
        self.assertEqual(index.match('J. Smith', 'Alpha')[0], 0)
        self.assertIsNone(index.match('J. Smith', 'Unknown')[0])
        self.assertIsNone(index.match('Jon Smith', 'Alpha')[0])

    def test_goalkeeper_excluded_missing_not_zero_and_signed_comparisons(self):
        players = [dict(player_positions='ST', **{a: 70 for a in ATTRIBUTES}) for _ in range(10)]
        players.append(dict(player_positions='GK', **{a: 99 for a in ATTRIBUTES}))
        home = aggregate_side(players, 'home')
        self.assertEqual(home['home_pace_mean'], 70)
        self.assertEqual(home['home_pace_count'], 10)
        self.assertTrue(np.isnan(aggregate_side(players[:7], 'home')['home_pace_mean']))
        away = aggregate_side([dict(player_positions='CB', **{a: 60 for a in ATTRIBUTES}) for _ in range(10)], 'away')
        out = comparisons({**home, **away})
        self.assertEqual(out['pace_advantage'], 10)
        self.assertEqual(out['home_shooting_vs_defending'], 10)
        self.assertEqual(out['away_shooting_vs_defending'], -10)

    def test_one_se_can_select_baseline(self):
        sets = {'baseline': ['a'], 'pace': ['a', 'b']}
        records = [dict(candidate='baseline', log_loss=x) for x in (.98, .98)]
        records += [dict(candidate='pace', log_loss=x) for x in (.96, .99)]
        self.assertEqual(choose(records, sets)['candidate'], 'baseline')
        for cols in candidates(['B365H', 'B365D', 'B365A']).values():
            self.assertFalse(set(cols) & {'home_score', 'away_score', 'fixture_date', 'source_row'})

    def test_imputation_is_fitted_only_on_training_rows(self):
        frame = pd.DataFrame(dict(
            home_team_name=['A'] * 6, away_team_name=['B'] * 6,
            home_score=[2, 1, 0, 3, 1, 0], away_score=[0, 1, 2, 0, 1, 3],
            pace_advantage=[1., 2., np.nan, 3., np.nan, 4.]))
        fitted = fit_model(frame, ['home_team_name', 'away_team_name', 'pace_advantage'],
                           'RF_classifier', 5)
        self.assertEqual(fitted['imputer'].statistics_[0], 2.5)
        transformed = fitted['imputer'].transform(pd.DataFrame({'pace_advantage': [np.nan, 1000.]}))
        self.assertEqual(transformed[0, 0], 2.5)
        self.assertEqual(fitted['imputer'].statistics_[0], 2.5)


if __name__ == '__main__':
    unittest.main()
