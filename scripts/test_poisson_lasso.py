import unittest
from select_expanded_features import np, pd
from sklearn.linear_model import PoissonRegressor
from poisson_lasso import fit_poisson_lasso, predict_poisson_lasso, kkt_error
from poisson_lasso_cv import select_penalties, design, transform


class PoissonLassoTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        self.x = rng.normal(size=(300, 4))
        self.y = rng.poisson(np.exp(.3 + self.x @ np.array([.4, -.2, 0., 0.])))

    def test_zero_penalty_matches_poisson_mle(self):
        fitted = fit_poisson_lasso(self.x, self.y, 0)
        reference = PoissonRegressor(alpha=0, tol=1e-10, max_iter=5000).fit(self.x, self.y)
        self.assertTrue(np.allclose(predict_poisson_lasso(fitted, self.x),
                                   reference.predict(self.x), rtol=2e-5, atol=2e-5))

    def test_strong_lasso_zeroes_slopes_but_not_intercept(self):
        fitted = fit_poisson_lasso(self.x, self.y, 10)
        self.assertEqual(fitted['nonzero'], 0)
        self.assertAlmostEqual(fitted['intercept'], np.log(self.y.mean()), places=7)
        self.assertLess(fitted['kkt_error'], 1e-5)

    def test_collinear_features_and_warm_start_kkt(self):
        x = np.column_stack([self.x, self.x[:, 0], self.x[:, 1] - self.x[:, 2]])
        strong = fit_poisson_lasso(x, self.y, .1)
        weak = fit_poisson_lasso(x, self.y, .01, warm=strong)
        self.assertLess(kkt_error(x, self.y, weak['intercept'], weak['coef'], .01), 1e-5)
        self.assertTrue((predict_poisson_lasso(weak, x) > 0).all())

    def test_one_se_chooses_stronger_penalty(self):
        records = [dict(target=side, alpha=alpha, poisson_deviance=value)
                   for side in ('home', 'away') for alpha, values in [(.01, [1., 1.2]), (.1, [1.1, 1.15])]
                   for value in values]
        result = select_penalties(records)
        self.assertEqual(result['home']['minimum_deviance_alpha'], .01)
        self.assertEqual(result['home']['one_se_alpha'], .1)

    def test_preprocessing_is_fold_only_and_preserves_column_names(self):
        tr = pd.DataFrame(dict(home_team_name=['A', 'A', 'B', 'B'],
                               away_team_name=['B', 'B', 'A', 'A'], skill=[1., np.nan, 3., 5.]))
        prep, x = design(tr, list(tr.columns))
        before = prep['scaler'].mean_.copy()
        new = transform(prep, pd.DataFrame(dict(home_team_name=['new'], away_team_name=['A'], skill=[np.nan])))
        self.assertTrue(np.array_equal(before, prep['scaler'].mean_))
        self.assertTrue(np.isfinite(new).all())
        self.assertEqual(len(prep['feature_names']), x.shape[1])
        self.assertEqual(len(prep['feature_sources']), x.shape[1])
        self.assertAlmostEqual(prep['imputer'].statistics_[0], 3.)


if __name__ == '__main__':
    unittest.main()
