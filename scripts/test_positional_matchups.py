"""Checks for role assignment, matchup directions, missingness, and date isolation."""
import unittest
from select_expanded_features import np, pd, threadpool_limits
from positional_features import aggregate_units, matchup_features, primary_position, unit, feature_columns
from compare_positional_matchups import candidates, chronological_split, select, fit_model, predict


def player(position, rating):
    return dict(player_positions=position, overall=rating, shooting=rating, passing=rating,
                dribbling=rating, defending=rating, pace=rating, physicality=rating)


class PositionalTests(unittest.TestCase):
    def test_primary_position_counts_player_once(self):
        self.assertEqual(primary_position('ST, CAM'), 'ST')
        self.assertEqual(unit('RW, RM'), 'attack')
        self.assertEqual(unit('LM, LW'), 'midfield')
        self.assertEqual(unit('LWB'), 'defense')
        self.assertEqual(unit(np.nan), 'unknown')
        f = aggregate_units([player('ST, CM', 80)], 'home', min_matched=1)
        self.assertEqual(f['home_attack_count'], 1)
        self.assertEqual(f['home_midfield_count'], 0)

    def test_opposing_units_and_missingness(self):
        home = [player('ST', 90)] + [player('CM', 80)] * 3 + [player('CB', 70)] * 4 + [player('GK', 85)]
        away = [player('ST', 60)] + [player('CM', 75)] * 3 + [player('CB', 65)] * 4 + [player('GK', 82)]
        f = {**aggregate_units(home, 'home'), **aggregate_units(away, 'away')}
        gaps = matchup_features(f)
        self.assertEqual(gaps['home_attack_shooting_vs_defense_defending'], 25)
        self.assertEqual(gaps['away_attack_shooting_vs_defense_defending'], -10)
        self.assertEqual(gaps['midfield_passing_advantage'], 5)
        self.assertEqual(gaps['home_attack_overall_vs_goalkeeper_overall'], 8)
        swapped = matchup_features({**aggregate_units(away, 'home'), **aggregate_units(home, 'away')})
        self.assertEqual(swapped['home_attack_shooting_vs_defense_defending'], -10)
        self.assertEqual(swapped['midfield_passing_advantage'], -5)
        incomplete = aggregate_units(home[:7], 'home')
        self.assertTrue(np.isnan(incomplete['home_attack_overall_mean']))
        self.assertEqual(incomplete['home_attack_count'], 1)
        invalid = [player('GK', 99)] + [player('CB', 70)] * 7
        missing = {**aggregate_units(invalid, 'home'), **aggregate_units(away, 'away')}
        self.assertTrue(np.isnan(matchup_features(missing)['home_attack_shooting_vs_defense_defending']))
        self.assertEqual(missing['home_defense_overall_mean'], 70)
        self.assertEqual(set(f) | set(gaps), set(feature_columns()))

    def test_no_targets_or_dates_in_features(self):
        for cols in candidates(['home_team_name', 'away_team_name', 'B365H']).values():
            self.assertEqual(len(cols), len(set(cols)))
            self.assertFalse(set(cols) & {'home_score', 'away_score', 'fixture_date', 'season', 'source_row'})

    def test_chronology_and_selection(self):
        frame = pd.DataFrame(dict(season=[2018, 2019], fixture_date=['2018-09-01', '2019-09-01']))
        tr, va = chronological_split(frame, 2019)
        self.assertEqual(len(tr), 1)
        frame.loc[0, 'fixture_date'] = '2019-09-01'
        with self.assertRaises(ValueError):
            chronological_split(frame, 2019)
        records = [dict(family='Poisson', candidate=c, season=y, log_loss=v)
                   for c, vals in [('baseline', [1., 1.]), ('positional', [.9, .95])]
                   for y, v in zip([2019, 2020], vals)]
        self.assertEqual(select(records)['Poisson']['candidate'], 'positional')

    def test_fit_predict_training_only_imputation(self):
        frame = pd.DataFrame(dict(home_team_name=['A'] * 9, away_team_name=['B'] * 9,
            home_score=[2, 1, 0] * 3, away_score=[0, 1, 2] * 3,
            gap=[1., 2., np.nan, 3., 4., 5., 6., 7., 8.]))
        columns = ['home_team_name', 'away_team_name', 'gap']
        for family in ['Poisson', 'RF_goals']:
            model = fit_model(frame, columns, family, 5)
            prep = model['fitted']['preprocessing'] if family == 'Poisson' else model['fitted']
            self.assertEqual(prep['imputer'].statistics_[0], 4.5)
            valid = frame.iloc[:3].copy()
            valid['gap'] = [np.nan, 10., 12.]
            p, goals = predict(model, valid)
            self.assertTrue(np.allclose(p.sum(axis=1), 1))
            self.assertEqual(goals.shape, (3, 2))
            self.assertEqual(prep['imputer'].statistics_[0], 4.5)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        unittest.main(verbosity=2)
