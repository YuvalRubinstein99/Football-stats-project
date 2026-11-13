import unittest
from datetime import date, timedelta
from select_expanded_features import np, pd, threadpool_limits
from match_form import build_form, fixture_key, columns
from compare_match_form import train_model, infer, candidates, select, YEARS, FAMILIES
from compare_attribute_regressions import transform


class MatchFormTests(unittest.TestCase):
    def game(self, day, **kwargs):
        return dict(league='E0', season=2019, date=date(2020, 6, 1)+timedelta(days=day),
                    home='Arsenal', away='Chelsea', hg=2, ag=1, hst=6, ast=3, **kwargs)

    def test_current_and_future_results_cannot_change_form(self):
        game = self.game(0)
        query = self.game(7)
        a = build_form([game], [query])[fixture_key(query)]
        b = build_form([game, {**query, 'hg':99}, self.game(8)], [query])[fixture_key(query)]
        self.assertEqual(a, b)
        self.assertEqual(a['home_recent_goals_for_5'], 2)
        self.assertEqual(a['away_recent_goals_against_5'], 2)
        self.assertEqual(a['recent_sot_for_5_gap'], 3)

    def test_missing_shots_are_not_zero_and_counts_are_reported(self):
        games = [self.game(0), {**self.game(1), 'hst':np.nan, 'ast':np.nan}]
        query = self.game(4)
        result = build_form(games, [query])[fixture_key(query)]
        self.assertEqual(result['home_recent_sot_for_5'], 6)
        self.assertEqual(result['home_recent_sot_matches_5'], 1)
        self.assertEqual(result['home_recent_matches_5'], 2)

    def test_venue_reversal_and_explicit_covid_season(self):
        game = self.game(0)
        query = {**self.game(40), 'home':'Chelsea', 'away':'Arsenal'}
        result = build_form([game], [query])[fixture_key(query)]
        self.assertEqual(result['home_recent_goals_for_3'], 1)
        self.assertEqual(result['home_recent_sot_against_3'], 6)
        new = {**query, 'season':2020}
        result = build_form([game], [new])[fixture_key(new)]
        self.assertTrue(np.isnan(result['home_recent_goals_for_3']))
        self.assertEqual(result['home_recent_matches_3'], 0)

    def test_windows_and_exponential_weighting(self):
        games = [{**self.game(i), 'hg':i} for i in range(11)]
        query = self.game(12)
        result = build_form(games, [query])[fixture_key(query)]
        self.assertEqual(result['home_recent_goals_for_3'], 9)
        self.assertEqual(result['home_recent_goals_for_5'], 8)
        self.assertEqual(result['home_recent_goals_for_10'], 5.5)
        self.assertGreater(result['home_recent_goals_for_ewm5'], 5)
        self.assertEqual(len(columns('5')), len(set(columns('5'))))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            build_form([games[0], games[0]], [query])

    def test_selection_can_reject_form_and_reduce_baseline(self):
        sets = {'odds':['B365H'], 'full':['B365H','home_recent_goals_for_5']}
        cfgs = {n:dict(candidate=n, alpha=None) for n in sets}
        records = [dict(config=n, season=y, log_loss=.97 + .01*i,
                        active_features=len(sets[n])) for n in sets for i,y in enumerate(YEARS)]
        self.assertEqual(select(records, cfgs, sets, True)['candidate'], 'odds')
        self.assertEqual(select(records, cfgs, sets, False)['candidate'], 'odds')
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            select(records[:-1], cfgs, sets, True)

    def test_every_model_preprocesses_training_only_and_ignores_targets_at_inference(self):
        rng = np.random.default_rng(40)
        frame = pd.DataFrame(dict(skill=rng.normal(size=180), home_score=rng.poisson(1.6,180),
                                  away_score=rng.poisson(1.2,180)))
        frame.loc[::9, 'skill'] = np.nan
        for family in FAMILIES:
            fitted, diag = train_model(family, frame, ['skill'], alpha=.03, trees=100)
            p, _ = infer(fitted, frame.iloc[:10])
            changed = frame.iloc[:10].copy()
            changed[['home_score', 'away_score']] = 99
            pp, _ = infer(fitted, changed)
            np.testing.assert_allclose(p, pp)
            self.assertTrue(np.isfinite(p).all())
            np.testing.assert_allclose(p.sum(axis=1), 1)
            self.assertTrue(diag['converged'])
            median = fitted['preprocessing']['imputer'].statistics_.copy()
            transform(fitted['preprocessing'], pd.DataFrame(dict(skill=[10000., np.nan])))
            np.testing.assert_array_equal(median, fitted['preprocessing']['imputer'].statistics_)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        unittest.main()
