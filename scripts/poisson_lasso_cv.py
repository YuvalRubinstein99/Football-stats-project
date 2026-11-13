"""Four chronological folds select separate home/away Poisson lasso penalties."""
import hashlib
import json
from pathlib import Path
from select_expanded_features import ROOT, pd, np, joblib, threadpool_limits
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_poisson_deviance
from compare_attribute_regressions import candidates
from compare_expanded_models import outcome_probabilities
from calibrate_models import scores
from compare_team_attributes import labels
from poisson_lasso import fit_poisson_lasso, predict_poisson_lasso

DATA = ROOT / 'results/team_attributes'
OUT = DATA / 'poisson_lasso'
ALPHAS = np.logspace(0, -4, 13).tolist()
YEARS = [2017, 2018, 2019, 2020]


def design(frame, columns):
    cats = [c for c in columns if c.endswith('_team_name')]
    nums = [c for c in columns if c not in cats]
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    imputer = SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)
    category = encoder.fit_transform(frame[cats])
    numeric = imputer.fit_transform(frame[nums])
    scaler = StandardScaler()
    x = scaler.fit_transform(np.column_stack([category, numeric]))
    names = list(encoder.get_feature_names_out(cats)) + list(imputer.get_feature_names_out(nums))
    sources = [name for name, values in zip(cats, encoder.categories_) for _ in values]
    sources += [name.removeprefix('missingindicator_') for name in imputer.get_feature_names_out(nums)]
    return dict(columns=columns, categorical=cats, numeric=nums, encoder=encoder,
                imputer=imputer, scaler=scaler, feature_names=names, feature_sources=sources), x


def transform(prep, frame):
    return prep['scaler'].transform(np.column_stack([
        prep['encoder'].transform(frame[prep['categorical']]),
        prep['imputer'].transform(frame[prep['numeric']])]))


def select_penalties(records):
    choices = {}
    for side in ('home', 'away'):
        table = pd.DataFrame([r for r in records if r['target'] == side])
        grouped = table.groupby('alpha').poisson_deviance.agg(['mean', 'std', 'count'])
        best = float(grouped['mean'].idxmin())
        threshold = float(grouped.loc[best, 'mean'] + grouped.loc[best, 'std'] / np.sqrt(grouped.loc[best, 'count']))
        sparse = float(grouped[grouped['mean'] <= threshold].index.max())
        choices[side] = dict(minimum_deviance_alpha=best, one_se_alpha=sparse,
                            minimum_cv_deviance=float(grouped.loc[best, 'mean']),
                            one_se_cv_deviance=float(grouped.loc[sparse, 'mean']),
                            one_se_threshold=threshold)
    return choices


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / 'cache'
    cache.mkdir(exist_ok=True)
    train = pd.read_csv(DATA / 'train.csv')
    original = joblib.load(ROOT / 'results/expanded_comparison/expanded__RF_vector.joblib')
    columns = candidates(original['categorical'] + original['numeric'])['all_attributes']
    inputs = [DATA / 'train.csv', DATA / 'test.csv', DATA / 'feature_audit.json',
              Path(__file__), ROOT / 'src/poisson_lasso.py', ROOT / 'scripts/compare_attribute_regressions.py',
              ROOT / 'scripts/compare_expanded_models.py', DATA / 'regressions/metrics.csv',
              DATA / 'regressions/selection_plan.json']
    plan = dict(columns=columns, alphas=ALPHAS, selection_seasons=YEARS,
        audit_season=2021, test_season=2022,
        objective='mean(exp(intercept + X beta) - y*(intercept + X beta)) + alpha*sum(abs(beta)); unpenalized intercept',
        selection='Separate home/away penalties minimizing mean seasonal Poisson deviance. Also strongest alpha '
                  'within one standard error of the best for sparsity; four correlated seasons, not a formal interval.',
        preprocessing='All encoded predictors standardized using training data only, including team dummies. '
                      'Fold-only median imputation and missing indicators. No SVD rotation before L1.',
        caveat='Same matched three-league sample as the earlier experiment; partial attribute coverage, '
               'unverified historical FIFA publication dates, previously inspected audit/test seasons.',
        input_hashes={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (OUT / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    prior_plan = json.loads((DATA / 'regressions/selection_plan.json').read_text())
    for path in (DATA / 'train.csv', DATA / 'test.csv'):
        relative = path.relative_to(ROOT).as_posix()
        if prior_plan['input_hashes'][relative] != plan['input_hashes'][relative]:
            raise ValueError('Previous Poisson comparison used different input rows; rebuild it first')
    records = []
    for year in YEARS:
        tr, va = train[train.season < year], train[train.season == year]
        if tr.empty or va.empty or tr.fixture_date.max() >= va.fixture_date.min():
            raise ValueError('Chronological fold overlap')
        prep, x = design(tr, columns)
        xv = transform(prep, va)
        for side in ('home', 'away'):
            warm = None
            for alpha in ALPHAS:
                key = hashlib.sha256(json.dumps(dict(plan=plan, year=year, side=side, alpha=alpha),
                                               sort_keys=True).encode()).hexdigest()
                path = cache / (key + '.joblib')
                if path.exists():
                    model = joblib.load(path)
                else:
                    print(f'CV {year} {side}: alpha={alpha:.6g}', flush=True)
                    model = fit_poisson_lasso(x, tr[side + '_score'].to_numpy(), alpha, warm=warm)
                    joblib.dump(model, path)
                warm = model
                mu = predict_poisson_lasso(model, xv)
                records.append(dict(season=year, target=side, alpha=alpha, train_matches=len(tr),
                    validation_matches=len(va), encoded_features=x.shape[1], nonzero=model['nonzero'],
                    kkt_error=model['kkt_error'], iterations=model['iterations'],
                    poisson_deviance=float(mean_poisson_deviance(va[side + '_score'], mu))))
            pd.DataFrame(records).to_csv(OUT / 'cv_scores.csv', index=False)
    selection = select_penalties(records)
    (OUT / 'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    # Freeze all penalties before audit/test comparisons.
    test = pd.read_csv(DATA / 'test.csv')
    metrics, predictions, coefficients = [], [], []
    for split, tr, va in [('Forward audit 2021-22', train[train.season < 2021], train[train.season == 2021]),
                          ('Test 2022-23 (previously inspected)', train, test)]:
        if tr.fixture_date.max() >= va.fixture_date.min():
            raise ValueError('Chronological evaluation overlap')
        prep, x = design(tr, columns)
        xv = transform(prep, va)
        for variant, key in [('Lasso minimum CV deviance', 'minimum_deviance_alpha'), ('Lasso one SE', 'one_se_alpha')]:
            goals, models, active = [], {}, set()
            for side in ('home', 'away'):
                alpha = selection[side][key]
                print(f'{split} / {variant} / {side}: alpha={alpha:.6g}', flush=True)
                model = fit_poisson_lasso(x, tr[side + '_score'].to_numpy(), alpha)
                models[side] = model
                goals.append(predict_poisson_lasso(model, xv))
                active.update(np.flatnonzero(model['coef']))
                if split.startswith('Test'):
                    coefficients.extend(dict(variant=variant, target=side, alpha=alpha,
                        feature=name, source_feature=source, standardized_coefficient=float(coef),
                        nonzero=bool(coef != 0)) for name, source, coef in
                        zip(prep['feature_names'], prep['feature_sources'], model['coef']))
            goals = np.column_stack(goals)
            p = outcome_probabilities(goals)
            actual = va[['home_score', 'away_score']].to_numpy()
            metrics.append(dict(model=variant, split=split, matches=len(va), **scores(p, labels(va)),
                goal_mse=float(np.mean((goals - actual)**2)), goal_mae=float(np.mean(np.abs(goals - actual))),
                mean_poisson_deviance=float(np.mean([mean_poisson_deviance(actual[:, j], goals[:, j]) for j in range(2)])),
                home_alpha=models['home']['alpha'], away_alpha=models['away']['alpha'],
                home_nonzero=models['home']['nonzero'], away_nonzero=models['away']['nonzero'],
                active_encoded_union=len(active), active_raw_union=len({prep['feature_sources'][i] for i in active}),
                total_encoded=x.shape[1], total_raw=len(columns),
                max_kkt_error=max(m['kkt_error'] for m in models.values())))
            predictions.extend(dict(model=variant, split=split, source_row=int(row.source_row),
                outcome=int(y), p_home=float(pp[0]), p_draw=float(pp[1]), p_away=float(pp[2]),
                home_mean=float(mu[0]), away_mean=float(mu[1]))
                for (_, row), y, pp, mu in zip(va.iterrows(), labels(va), p, goals))
            if split.startswith('Test'):
                joblib.dump(dict(preprocessing=prep, models=models, plan=plan, selection=selection),
                            OUT / ('minimum_cv.joblib' if key == 'minimum_deviance_alpha' else 'one_se.joblib'))
    previous = pd.read_csv(DATA / 'regressions/metrics.csv')
    for name in ['original', 'all_attributes', 'odds_overall_form']:
        rows = previous[(previous.family == 'Poisson') & (previous.candidate == name)]
        for _, row in rows.iterrows():
            metrics.append(dict(model='Previous Poisson: ' + name, split=row.split, matches=int(row.matches),
                                log_loss=row.log_loss, brier=row.brier, auc=row.auc,
                                goal_mse=row.goal_mse, goal_mae=row.goal_mae,
                                total_raw=row.raw_features, total_encoded=row.encoded_features))
    table, coef = pd.DataFrame(metrics), pd.DataFrame(coefficients)
    table.to_csv(OUT / 'metrics.csv', index=False)
    coef.to_csv(OUT / 'coefficients.csv', index=False)
    pd.DataFrame(predictions).to_csv(OUT / 'predictions.csv', index=False)
    active = coef[coef.nonzero].sort_values(['variant', 'target', 'standardized_coefficient'])
    active.to_csv(OUT / 'retained_coefficients.csv', index=False)
    (OUT / 'summary.json').write_text(json.dumps(dict(plan=plan, selection=selection,
        results=[{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in metrics]), indent=2), encoding='utf-8')
    body = '<h1>Poisson lasso with chronological cross-validation</h1><p>' + plan['caveat'] + '</p>'
    body += '<p>' + plan['selection'] + '</p><p>' + plan['preprocessing'] + '</p>'
    body += '<h2>Selected penalties</h2>' + pd.DataFrame(selection).T.to_html()
    body += '<h2>Audit and test comparison</h2>' + table.to_html(index=False, float_format=lambda v: f'{v:.6f}')
    body += '<h2>Nonzero coefficients</h2><p>Coefficients are per training standard deviation. '
    body += 'Correlated inputs can substitute for each other; selection is not evidence of causation.</p>' + active.to_html(index=False)
    body += '<details><summary>All CV scores and KKT checks</summary>' + pd.DataFrame(records).to_html(index=False) + '</details>'
    body += '<p>Objective follows the <a href="https://www.statsmodels.org/v0.14.4/generated/statsmodels.genmod.generalized_linear_model.GLM.fit_regularized.html">Poisson GLM L1 formulation</a>. '
    body += 'Solved with nonnegative coefficient splitting and <a href="https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html">SciPy L-BFGS-B</a>; intercept unpenalized.</p>'
    (OUT / 'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Poisson lasso CV</title>'
        '<style>body{font:15px system-ui;margin:32px;color:#182532}table{border-collapse:collapse;font-size:12px}'
        'td,th{padding:7px;border-bottom:1px solid #ddd}</style>' + body, encoding='utf-8')
    print(json.dumps(selection, indent=2), flush=True)
    print(table[table.split.str.startswith('Test')].to_string(index=False), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
