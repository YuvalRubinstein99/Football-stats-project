"""Reproducible joint-score experiment derived from prediction.ipynb.

Run with Python; optional local dependencies live in .prediction_deps.
Outputs are isolated in results/. Original notebooks/models are untouched.
"""
from pathlib import Path
import sys

from project_paths import ROOT, DATA, OUT
if (ROOT / '.prediction_deps').exists():
    sys.path.insert(0, str(ROOT / '.prediction_deps'))

import json
import platform
import hashlib
import html
import numpy as np
import pandas as pd
import scipy
import sklearn
import joblib
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp, xlogy
from scipy.stats import skellam, poisson
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss, roc_auc_score, confusion_matrix, mean_squared_error

SEED = 20261007
SIMULATIONS = 50_000

LABELS = ['Away', 'Draw', 'Home']


def prepare():
    raw_train = pd.read_csv(DATA / 'train.csv')
    raw_test = pd.read_csv(DATA / 'test.csv')
    frames, counts = [], {}
    for name, raw in [('train', raw_train), ('test', raw_test)]:
        frame = raw.copy()
        frame['Matchweek'] = frame['Matchweek'].astype(str).str.split().str[-1].astype(int)
        frame['Home_min_max'] = frame.HomePlayer_Overall_max * frame.HomePlayer_Overall_min
        frame['Away_min_max'] = frame.AwayPlayer_Overall_max * frame.AwayPlayer_Overall_min
        frame = frame.drop(columns=[c for c in frame if c.startswith('Unnamed:') or 'bench' in c or 'points_to' in c])
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        # Keep row identifiers for score/odds alignment after filtering.
        counts[name + '_dropped'] = len(raw) - len(frame)
        frames.append(frame)
    train, test = frames
    target = ['home_score', 'away_score']
    categorical = [c for c in ['home_team_name', 'away_team_name', 'league'] if c in train]
    numeric = [c for c in train if c not in categorical + target]
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    train_cat = encoder.fit_transform(train[categorical])
    test_cat = encoder.transform(test[categorical])
    xtrain = np.column_stack([train_cat, train[numeric].to_numpy(float)])
    xtest = np.column_stack([test_cat, test[numeric].to_numpy(float)])
    ytrain = train[target].to_numpy(int)
    ytest = test[target].to_numpy(int)
    odds = test[['B365A', 'B365D', 'B365H']].to_numpy(float)
    assert np.isfinite(xtrain).all() and np.isfinite(xtest).all()
    assert (ytrain >= 0).all() and (ytest >= 0).all() and (odds > 1).all()
    return xtrain, xtest, ytrain, ytest, odds, test, counts, encoder.get_feature_names_out(categorical).tolist() + numeric


def joint_logpmf(h, a, u, v, c, return_shared=False):
    """Stable bivariate Poisson likelihood and posterior shared count E[W|H,A]."""
    k = np.arange(int(np.minimum(h, a).max()) + 1)[None, :]
    hh, aa = h[:, None] - k, a[:, None] - k
    valid = (hh >= 0) & (aa >= 0)
    hh, aa = np.maximum(hh, 0), np.maximum(aa, 0)
    terms = (xlogy(hh, u[:, None]) - gammaln(hh + 1)
             + xlogy(aa, v[:, None]) - gammaln(aa + 1)
             + xlogy(k, c[:, None]) - gammaln(k + 1))
    terms = np.where(valid, terms, -np.inf)
    normalizer = logsumexp(terms, axis=1)
    lp = normalizer - u - v - c
    if return_shared:
        ew = (np.exp(terms - normalizer[:, None]) * k).sum(axis=1)
        return lp, ew
    return lp


def rates(theta, means):
    """RF means are offsets; three rates are conditional on each match's inputs."""
    z = np.column_stack([np.ones(len(means)), np.log(means)])
    u = means[:, 0] * np.exp(theta[0])
    v = means[:, 1] * np.exp(theta[1])
    c = np.exp(z @ theta[2:])
    return u, v, c, z


def fit_joint(means, scores):
    # All means here are out-of-fold TRAIN predictions, never test predictions.
    independent = np.log(scores.sum(axis=0) / means.sum(axis=0))

    def objective(theta):
        u, v, c, z = rates(theta, means)
        lp, ew = joint_logpmf(scores[:, 0], scores[:, 1], u, v, c, True)
        grad = np.r_[np.mean(u - scores[:, 0] + ew),
                     np.mean(v - scores[:, 1] + ew), z.T @ (c - ew) / len(means)]
        return -lp.mean(), grad

    starts = []
    for shared in [0.03, 0.15, 0.4]:
        init = np.r_[independent, np.log(shared), 0., 0.]
        result = minimize(objective, init, jac=True, method='L-BFGS-B',
                          bounds=[(-2, 2), (-2, 2), (-16, 2), (-2, 2), (-2, 2)],
                          options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-7})
        starts.append(result)
    converged = [r for r in starts if r.success and np.isfinite(r.fun)]
    if not converged:
        raise RuntimeError('Joint optimizer failed: ' + str([r.message for r in starts]))
    best = min(converged, key=lambda r: r.fun)
    ind_means = means * np.exp(independent)
    independent_nll = -poisson.logpmf(scores, ind_means).sum(axis=1).mean()
    use_boundary = best.fun >= independent_nll - 1e-8
    diagnostics = {'parameters': best.x.tolist(), 'independent_log_scales': independent.tolist(),
                   'oof_joint_nll': float(min(best.fun, independent_nll)),
                   'oof_independent_nll': float(independent_nll),
                   'zero_covariance_boundary': bool(use_boundary),
                   'optimizer_runs': [{'success': bool(r.success), 'nll': float(r.fun),
                                       'message': str(r.message)} for r in starts]}
    return best.x, independent, use_boundary, diagnostics


def exact_outcomes(u, v):
    p = np.column_stack([skellam.cdf(-1, u, v), skellam.pmf(0, u, v), skellam.sf(0, u, v)])
    assert np.isfinite(p).all() and (p >= 0).all()
    assert np.allclose(p.sum(axis=1), 1, atol=1e-10)
    return p


def simulate_outcomes(u, v, c, n=SIMULATIONS):
    rng = np.random.default_rng(SEED)
    result = np.zeros((len(u), 3))
    # Simulate both full scores, including the same shared count in each pair.
    for i in range(len(u)):
        shared = rng.poisson(c[i], n)
        home = rng.poisson(u[i], n) + shared
        away = rng.poisson(v[i], n) + shared
        result[i] = [(home < away).mean(), (home == away).mean(), (home > away).mean()]
    return result


from betting_rules import strategy_scores


def evaluate(name, p, labels, odds, market, scores, model_rates=None):
    chosen = p.argmax(axis=1)
    cm = confusion_matrix(labels, chosen, labels=[0, 1, 2])
    metrics = {'model': name, 'matches': len(labels), 'correct': int((chosen == labels).sum()),
               'accuracy': float((chosen == labels).mean()), 'log_loss': float(log_loss(labels, p, labels=[0,1,2])),
               'brier': float(np.mean(np.sum((p - np.eye(3)[labels])**2, axis=1))),
               'auc_ovr': float(roc_auc_score(labels, p, multi_class='ovr')),
               'predicted_draws': int((chosen == 1).sum()), 'correct_draws': int(cm[1, 1]),
               'mean_draw_probability': float(p[:, 1].mean()), 'actual_draw_rate': float((labels == 1).mean())}
    if model_rates is not None:
        u, v, c = model_rates
        metrics['joint_score_nll'] = float(-joint_logpmf(scores[:,0], scores[:,1], u, v, c).mean())
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(OUT / (name + '_confusion.csv'))
    rows, ledgers = [], []
    for strategy, (values, threshold) in strategy_scores(p, odds, market).items():
        pick = values.argmax(axis=1)
        bet = values[np.arange(len(p)), pick] > threshold
        win = (pick == labels) & bet
        selected_odds = odds[np.arange(len(p)), pick]
        profit = np.where(bet, np.where(win, selected_odds - 1, -1), 0)
        count = int(bet.sum())
        rows.append({'model': name, 'strategy': strategy, 'bets': count, 'wins': int(win.sum()),
                     'losses': int((bet & ~win).sum()), 'win_rate': float(win.sum()/count) if count else 0,
                     'net_profit_units': float(profit.sum()), 'roi': float(profit.sum()/count) if count else 0})
        ledgers.append(pd.DataFrame({'model': name, 'strategy': strategy, 'test_position': np.arange(len(p)),
                                    'pick': pick, 'bet': bet, 'win': win, 'decimal_odds': selected_odds,
                                    'net_profit_units': profit}))
    return metrics, rows, ledgers


def self_checks():
    # Verify independence limit, joint normalization, and simulation vs exact outcomes.
    h, a = np.meshgrid(np.arange(30), np.arange(30), indexing='ij')
    h, a = h.ravel(), a.ravel()
    u, v, c = [np.full(len(h), value) for value in [1.3, 0.9, 0.2]]
    joint = np.exp(joint_logpmf(h, a, u, v, c))
    assert abs(joint.sum() - 1) < 1e-10
    assert np.allclose(joint_logpmf(h, a, u, v, c*0), poisson.logpmf(h,u)+poisson.logpmf(a,v))
    grid_outcomes = np.array([joint[h<a].sum(), joint[h==a].sum(), joint[h>a].sum()])
    exact = exact_outcomes(np.array([1.3]), np.array([0.9]))[0]
    assert np.allclose(grid_outcomes, exact)
    simulated = simulate_outcomes(np.array([1.3]), np.array([0.9]), np.array([0.2]), n=200_000)[0]
    assert np.max(np.abs(simulated-exact)) < 0.005


def fit_vector_forest(xtrain, xtest, ytrain):
    """One native multi-output forest, with both goals influencing each split."""
    forest = RandomForestRegressor(n_estimators=2000, criterion='squared_error',
                                   max_depth=5, max_features=0.3, max_samples=0.5,
                                   oob_score=mean_squared_error,
                                   random_state=SEED, n_jobs=4, verbose=1)
    # Hold the previous train-CV-selected settings fixed to isolate the loss change.
    # OOB MSE evaluates training rows using only trees that did not sample them.
    forest.fit(xtrain, ytrain)  # ytrain has TWO columns: [home_goals, away_goals].
    vectors = forest.predict(xtest)  # (n_matches, 2), with shared tree structure.
    assert forest.n_outputs_ == 2 and len(forest.estimators_) == 2000
    assert all(tree.n_outputs_ == 2 for tree in forest.estimators_)
    assert vectors.shape == (len(xtest), 2) and np.isfinite(vectors).all()
    assert (vectors >= 0).all()
    joblib.dump(forest, OUT / 'vector_forest.joblib')
    params = {'max_features': 0.3, 'max_samples': 0.5,
              'oob_mse': float(forest.oob_score_), 'selection': 'fixed_previous_CV_settings'}
    old_cv = OUT / 'vector_cv_results.csv'
    if old_cv.exists():
        old_cv.rename(OUT / 'vector_previous_poisson_cv_results.csv')
    pd.DataFrame([params]).to_csv(OUT / 'vector_fit_details.csv', index=False)
    return vectors, params


def main():
    OUT.mkdir(exist_ok=True)
    self_checks()
    xtrain, xtest, ytrain, ytest, odds, test, counts, features = prepare()
    print(f'Train: {len(xtrain)}; test: {len(xtest)}; features: {len(features)}', flush=True)
    means_test, means_oof, best_params = [], [], {}
    folds = KFold(n_splits=3, shuffle=False)  # Same regression CV partitions as original cv=3.
    for j, side in enumerate(['home', 'away']):
        forest = RandomForestRegressor(n_estimators=200, criterion='poisson', max_depth=5,
                                       max_samples=0.5 if j == 0 else 0.25,
                                       random_state=SEED+j, n_jobs=4)
        search = GridSearchCV(forest, {'max_features': [0.02, 0.07, 0.15, 0.25, 0.3]},
                              scoring='neg_mean_squared_error', cv=folds, n_jobs=1)
        print(f'Fitting {side} RF grid from original notebook...', flush=True)
        search.fit(xtrain, ytrain[:, j])
        best_params[side] = search.best_params_
        means_test.append(search.predict(xtest))
        print(f'{side}: {search.best_params_}; generating training-only OOF predictions...', flush=True)
        means_oof.append(cross_val_predict(search.best_estimator_, xtrain, ytrain[:, j], cv=folds, n_jobs=1))
    means_test = np.maximum(np.column_stack(means_test), 1e-8)
    means_oof = np.maximum(np.column_stack(means_oof), 1e-8)
    print('Fitting one multi-output RF on [home_goals, away_goals]...', flush=True)
    vector_means, vector_params = fit_vector_forest(xtrain, xtest, ytrain)
    best_params['vector'] = vector_params
    best_params['vector']['n_estimators'] = 2000
    best_params['vector']['criterion'] = 'squared_error'
    best_params['vector']['scoring'] = 'out_of_bag_mean_squared_error'
    print(f'Vector RF: {vector_params}; output shape: {vector_means.shape}', flush=True)
    print('Fitting joint likelihood to OOF training scores...', flush=True)
    theta, independent, boundary, diagnostics = fit_joint(means_oof, ytrain)
    u, v, c, _ = rates(theta, means_test)
    ind = means_test * np.exp(independent)
    if boundary:
        u, v, c = ind[:, 0], ind[:, 1], np.zeros(len(ind))
    market = 1 / odds
    market /= market.sum(axis=1, keepdims=True)
    joint_exact = exact_outcomes(u, v)
    print(f'Simulating {SIMULATIONS:,} joint score pairs per test match...', flush=True)
    joint_sim = simulate_outcomes(u, v, c)
    probabilities = {'RF_independent': exact_outcomes(means_test[:,0], means_test[:,1]),
                     'RF_independent_calibrated': exact_outcomes(ind[:,0], ind[:,1]),
                     'RF_bivariate_exact': joint_exact, 'RF_bivariate_simulated': joint_sim,
                     'RF_vector': exact_outcomes(np.maximum(vector_means[:,0], 1e-8), np.maximum(vector_means[:,1], 1e-8)),
                     'Market': market}
    zero = np.zeros(len(ytest))
    all_rates = {'RF_independent': (means_test[:,0], means_test[:,1], zero),
                 'RF_independent_calibrated': (ind[:,0], ind[:,1], zero),
                 'RF_bivariate_exact': (u,v,c), 'RF_bivariate_simulated': (u,v,c),
                 'RF_vector': (np.maximum(vector_means[:,0], 1e-8), np.maximum(vector_means[:,1], 1e-8), zero)}
    labels = np.sign(ytest[:,0]-ytest[:,1]).astype(int) + 1
    metric_rows, strategy_rows, ledgers = [], [], []
    per_match = test[['home_team_name','away_team_name','home_score','away_score','B365A','B365D','B365H']].copy()
    per_match.insert(0, 'source_test_row', test.index)
    per_match['lambda_home_unshared'] = u
    per_match['lambda_away_unshared'] = v
    per_match['lambda_shared'] = c
    per_match['joint_expected_home_goals'] = u+c
    per_match['joint_expected_away_goals'] = v+c
    per_match['vector_expected_home_goals'] = vector_means[:,0]
    per_match['vector_expected_away_goals'] = vector_means[:,1]
    per_match['independent_expected_home_goals'] = means_test[:,0]
    per_match['independent_expected_away_goals'] = means_test[:,1]
    per_match['calibrated_expected_home_goals'] = ind[:,0]
    per_match['calibrated_expected_away_goals'] = ind[:,1]
    for name, p in probabilities.items():
        m, s, ledger = evaluate(name, p, labels, odds, market, ytest, all_rates.get(name))
        if name in all_rates:
            ru, rv, rc = all_rates[name]
            if name == 'RF_vector':
                ru, rv = vector_means[:,0], vector_means[:,1]
            m['home_goal_rmse'] = float(np.sqrt(np.mean((ru+rc-ytest[:,0])**2)))
            m['away_goal_rmse'] = float(np.sqrt(np.mean((rv+rc-ytest[:,1])**2)))
            m['home_goal_mse'] = float(np.mean((ru+rc-ytest[:,0])**2))
            m['away_goal_mse'] = float(np.mean((rv+rc-ytest[:,1])**2))
            m['mean_goal_mse'] = (m['home_goal_mse'] + m['away_goal_mse']) / 2
            m['home_goal_mae'] = float(np.mean(np.abs(ru+rc-ytest[:,0])))
            m['away_goal_mae'] = float(np.mean(np.abs(rv+rc-ytest[:,1])))
            m['mean_goal_mae'] = (m['home_goal_mae'] + m['away_goal_mae']) / 2
        metric_rows.append(m)
        strategy_rows.extend(s)
        ledgers.extend(ledger)
        for j, outcome in enumerate(LABELS):
            per_match[name + '_' + outcome] = p[:,j]
    metrics = pd.DataFrame(metric_rows)
    strategies = pd.DataFrame(strategy_rows)
    metrics.to_csv(OUT / 'model_metrics.csv', index=False)
    strategies.to_csv(OUT / 'strategy_results.csv', index=False)
    per_match.to_csv(OUT / 'match_predictions.csv', index=False)
    pd.concat(ledgers, ignore_index=True).to_csv(OUT / 'bet_ledger.csv', index=False)
    pd.DataFrame({'home_oof_mean': means_oof[:,0], 'away_oof_mean': means_oof[:,1],
                  'home_score': ytrain[:,0], 'away_score': ytrain[:,1]}).to_csv(OUT / 'training_oof_predictions.csv', index=False)
    diagnostics.update({'seed': SEED, 'simulations_per_match': SIMULATIONS, 'best_rf_parameters': best_params,
                        'train_rows': len(ytrain), 'test_rows': len(ytest), 'filter_counts': counts,
                        'features': features, 'mean_shared_rate': float(c.mean()),
                        'shared_rate_range': [float(c.min()), float(c.max())],
                        'simulation_mean_absolute_error': float(np.abs(joint_sim-joint_exact).mean()),
                        'simulation_max_absolute_error': float(np.abs(joint_sim-joint_exact).max()),
                        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                                     'pandas': pd.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__},
                        'input_sha256': {f: hashlib.sha256((DATA/f).read_bytes()).hexdigest() for f in ['train.csv','test.csv']}})
    (OUT / 'run_details.json').write_text(json.dumps(diagnostics, indent=2), encoding='utf-8')
    notes = '''This experiment adapts the random-forest branch of prediction.ipynb. It uses the same
features, hyperparameter grid and train.csv/test.csv split, with fixed random seeds and correct
row alignment. The absent league column is omitted. Betting odds remain input features, as in
the source notebook. Seven incomplete training rows are dropped; no test labels train the models.
The baseline is a reproducible refit, not the old pickle. Only the RF branch is compared here.

RF_vector is ONE native multi-output RandomForestRegressor fitted to the two-column
target [home_score, away_score]. All 2,000 trees share splits across both targets; it is
not a MultiOutputRegressor wrapper. Depth=5, max_features=0.3, and max_samples=0.5
are held at the previous training-CV-selected values to isolate the loss change.
The vector model uses criterion='squared_error' and training out-of-bag MSE scoring.
No new hyperparameter search is performed; the previous settings are held fixed.
Its predictions average leaf means across trees to estimate conditional means.
For the RF_vector outcome/betting comparison, these point predictions feed the original independent
Poisson/Skellam conversion. The separate-forest baselines retain 200 trees each, so the
comparison also differs in tree count. Shared-tree prediction does NOT remove the
conditional independence assumption in the outcome conversion. Squared-error training estimates
goal means but does not establish a Poisson distribution. Rates are floored at 1e-8 only for
probability conversion; MSE, MAE, and RMSE use raw predictions and are all reported.

The joint model retains the RF predictors as offsets: u=RF_home*exp(b_home),
v=RF_away*exp(b_away), c=exp(g0+g_home*log(RF_home)+g_away*log(RF_away)).
Five global coefficients are fitted by joint score likelihood on three-fold out-of-fold TRAIN
predictions. Each match receives three rates. H=Poisson(u)+W and A=Poisson(v)+W,
where W=Poisson(c) is shared. Dependence is constrained to be nonnegative.
This is a joint calibration layer, not end-to-end joint training of the forests. A separately
calibrated independent control estimates only the two mean-scale coefficients using the same OOF data.
Hyperparameter selection uses all training folds; this is not nested or chronological validation.
The CSVs contain no dates, so a clean temporal split or upstream feature leakage cannot be verified.

Simulation uses 50,000 score pairs per match and a fixed seed. Exact probabilities are also reported
because the shared count cancels from H-A, which is Skellam(u,v). Use exact results to distinguish
model changes from Monte Carlo variation. Joint score NLL assesses the full 2D distribution.

Each bet stakes one unit. A win earns decimal_odds-1; a loss costs one unit. Profit excludes the
original notebook's arbitrary initial one-unit balance. ROI=net_profit/number_of_bets.
The nine original decision rules are reproduced, including legacy heuristics. Moving draw
probability into the home column is only a selection heuristic: a real draw still loses a home-win
bet; these are not double-chance bets. The original 'Sharpe' expression is labeled a legacy score,
not a Sharpe ratio. Market power weighting uses only market probabilities and must match across models.
No strategy or threshold is selected using these test results. Test profit is a retrospective result.
'''
    (OUT / 'methodology.txt').write_text(notes, encoding='utf-8')
    main_strategies = strategies[strategies.model != 'Market']
    report = '<!doctype html><html><head><meta charset="utf-8"><title>Joint goal prediction comparison</title><style>body{font:15px system-ui;max-width:1400px;margin:32px auto;padding:0 20px;color:#182532}table{border-collapse:collapse;width:100%;margin:20px 0;font-size:13px}th,td{border-bottom:1px solid #ddd;padding:8px;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#edf3f8}pre{white-space:pre-wrap;line-height:1.6;background:#f4f6f8;padding:20px}</style></head><body>'
    report += '<h1>Joint goal prediction: held-out results</h1><p>11,561 training matches; 1,447 test matches. One unit per bet. Lower log loss and Brier score are better.</p>'
    report += '<h2>Predictive performance</h2>' + metrics.to_html(index=False, float_format=lambda x: f'{x:.5f}')
    report += '<h2>Wins and profit by strategy</h2>' + main_strategies.to_html(index=False, float_format=lambda x: f'{x:.4f}')
    report += '<h2>Method and limitations</h2><pre>' + html.escape(notes) + '</pre></body></html>'
    (OUT / 'report.html').write_text(report, encoding='utf-8')
    from plot_bivariate_results import main as make_plots
    make_plots()
    from analyze_prediction_errors import main as analyze_errors
    analyze_errors()
    from plot_probability_gap import main as plot_gap
    plot_gap()
    from evaluate_home_loss import main as evaluate_binary
    evaluate_binary()
    evaluate_binary('sqrt_profit')
    from evaluate_home_win import main as evaluate_binary_win
    evaluate_binary_win()
    evaluate_binary_win('sqrt_profit')
    print(metrics.to_string(index=False), flush=True)
    print(strategies[strategies.model.isin(['RF_independent','RF_bivariate_exact','RF_bivariate_simulated'])].to_string(index=False), flush=True)
    print('Shared rate:', diagnostics['mean_shared_rate'], 'Simulation MAE:', diagnostics['simulation_mean_absolute_error'], flush=True)
    print('Results saved to', OUT, flush=True)


if __name__ == '__main__':
    main()
