import unittest
import numpy as np
import pandas as pd
from compare_attribute_regressions import candidates, choose, design, transform, fit_model, probabilities, FAMILIES


class AttributeRegressionTests(unittest.TestCase):
    def test_candidates_can_remove_original_predictors(self):
        base = ['home_team_name', 'away_team_name', 'B365H', 'B365D', 'B365A',
                'HomePlayer_Overall_mean', 'AwayPlayer_Overall_mean', 'Matchweek',
                'home_GD_form', 'away_GD_form']
        sets = candidates(base)
        self.assertEqual(sets['odds_only'], ['B365H', 'B365D', 'B365A'])
        self.assertEqual(len(sets['odds_pace']), 4)
        self.assertNotIn('home_team_name', sets['compact_original'])
        for cols in sets.values():
            self.assertEqual(len(cols), len(set(cols)))
            self.assertFalse(set(cols) & {'home_score', 'away_score', 'source_row', 'fixture_date'})

    def test_selection_accounts_for_encoded_complexity(self):
        sets = {'teams': ['home_team_name', 'away_team_name'], 'small': ['a', 'b', 'c']}
        records = [dict(candidate=n, log_loss=x, encoded_features=d)
                   for n, d in [('teams', 100), ('small', 3)] for x in [.97, .99]]
        self.assertEqual(choose(records, sets)['candidate'], 'small')

    def test_fold_only_imputation_and_unknown_team(self):
        tr = pd.DataFrame(dict(home_team_name=['A', 'B', 'A'], value=[1., np.nan, 3.]))
        artifact, _, _ = design(tr, ['home_team_name', 'value'])
        before = artifact['imputer'].statistics_.copy()
        x, xo = transform(artifact, pd.DataFrame(dict(home_team_name=['new'], value=[np.nan])))
        self.assertTrue(np.array_equal(before, artifact['imputer'].statistics_))
        self.assertEqual(before[0], 2.)
        self.assertTrue(np.isfinite(x).all() and np.isfinite(xo).all())

    def test_all_families_produce_valid_probabilities(self):
        rng = np.random.default_rng(50)
        n = 150
        tr = pd.DataFrame(dict(skill=rng.normal(size=n), home_score=rng.poisson(1.5, n),
                               away_score=rng.poisson(1.1, n)))
        tr.loc[::7, 'skill'] = np.nan
        for family in FAMILIES:
            fitted, diag = fit_model(family, tr, ['skill'])
            p, goals = probabilities(fitted, tr)
            self.assertTrue(diag['converged'])
            self.assertEqual(p.shape, (n, 3))
            self.assertTrue(np.isfinite(p).all() and (p >= 0).all())
            self.assertTrue(np.allclose(p.sum(axis=1), 1))
            self.assertEqual(goals is None, family.startswith('Ordered'))


if __name__ == '__main__':
    unittest.main()
