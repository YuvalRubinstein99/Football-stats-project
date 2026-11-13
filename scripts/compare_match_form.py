"""Chronological feature/window selection and matched form ablations, offline."""
import argparse
import hashlib
import html
import json
from pathlib import Path
from select_expanded_features import ROOT, pd, np, joblib, threadpool_limits, fit, predict
from compare_attribute_regressions import design, transform
from compare_team_attributes import labels
from compare_expanded_models import outcome_probabilities
from calibrate_models import scores
from match_form import columns, WINDOWS
from poisson_lasso import fit_poisson_lasso, predict_poisson_lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

OUT = ROOT / 'results/match_form'
FAMILIES = ['RF_classifier', 'RF_goals', 'Linear', 'Poisson',
            'Ordered_logit', 'Ordered_probit', 'Poisson_lasso']
YEARS = [2018, 2019, 2020]
ALPHAS = [1., .3, .1, .03, .01, .003, .001]


def candidates(base):
    odds = ['B365H', 'B365D', 'B365A']
    ability = odds + ['HomePlayer_Overall_mean', 'AwayPlayer_Overall_mean', 'Matchweek']
    compact = ability + [c for c in base if c.endswith(('_prior', '_form'))]
    sets = {'odds': odds, 'ability': ability, 'compact': compact, 'original': base}
    for window in WINDOWS:
        for kind in ('all', 'gaps'):
            sets[f'compact_{kind}_{window}'] = compact + columns(window, kind)
    for kind in ('goals', 'shots'):
        sets['compact_' + kind + '_5'] = compact + columns('5', kind)
    sets['original_all_5'] = base + columns('5')
    return sets


def configs(family, sets):
    if family == 'Poisson_lasso':
        names = ['odds', 'ability', 'compact', 'original',
                 *['compact_all_' + w for w in WINDOWS], 'original_all_5']
        return {f'{n}@{a:g}': dict(candidate=n, alpha=a) for n in names for a in ALPHAS}
    return {n: dict(candidate=n, alpha=None) for n in sets}


def train_model(family, frame, cols, alpha=None, trees=300):
    prep, x, xo = design(frame, cols)
    goals = frame[['home_score', 'away_score']].to_numpy()
    bundle = dict(family=family, preprocessing=prep, columns=cols)
    diag = dict(converged=True, raw_features=len(cols), encoded_features=x.shape[1])
    if family.startswith('RF_'):
        cls = RandomForestClassifier if family == 'RF_classifier' else RandomForestRegressor
        model = cls(n_estimators=trees, max_depth=5, max_features=.3, max_samples=.5,
                    n_jobs=4, random_state=20261020,
                    criterion='log_loss' if family == 'RF_classifier' else 'squared_error')
        model.fit(x, labels(frame) if family == 'RF_classifier' else goals)
        bundle['model'] = model
    elif family == 'Poisson_lasso':
        scale = StandardScaler().fit(x)  # Also scale team dummies for equal L1 treatment.
        xs = scale.transform(x)
        models = [fit_poisson_lasso(xs, goals[:, j], alpha) for j in range(2)]
        bundle.update(models=models, scale=scale, alpha=alpha)
        cats = prep['categorical']
        names = (list(prep['encoder'].get_feature_names_out(cats)) if cats else [])
        names += list(prep['imputer'].get_feature_names_out(prep['numeric']))
        sources = ([c for c, values in zip(cats, prep['encoder'].categories_) for _ in values]
                   if cats else [])
        sources += [n.removeprefix('missingindicator_') for n in
                    prep['imputer'].get_feature_names_out(prep['numeric'])]
        active = np.any(np.column_stack([m['coef'] for m in models]) != 0, axis=1)
        bundle.update(feature_names=names, feature_sources=sources)
        diag.update(active_features=int(active.sum()),
                    active_raw_features=len({s for s, a in zip(sources, active) if a}),
                    max_kkt_error=max(m['kkt_error'] for m in models))
    else:
        bundle['model'], diagnostic = fit(family, x, xo, goals)
        diag.update(diagnostic)
    diag.setdefault('active_features', x.shape[1])
    return bundle, diag


def infer(bundle, frame):
    x, xo = transform(bundle['preprocessing'], frame)
    family = bundle['family']
    if family == 'RF_classifier':
        if not np.array_equal(bundle['model'].classes_, [0, 1, 2]):
            raise ValueError('Unexpected outcome order')
        p, goals = bundle['model'].predict_proba(x), None
    elif family == 'RF_goals':
        goals = bundle['model'].predict(x)
        p = outcome_probabilities(goals)
    elif family == 'Poisson_lasso':
        goals = np.column_stack([predict_poisson_lasso(m, bundle['scale'].transform(x))
                                 for m in bundle['models']])
        p = outcome_probabilities(goals)
    else:
        p, goals = predict(bundle['model'], x, xo)
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError('Invalid outcome probabilities')
    return p, goals


def select(records, cfgs, sets, allow_form):
    allowed = {name for name, c in cfgs.items()
               if allow_form or not any('_recent_' in col or col.startswith('recent_')
                                       for col in sets[c['candidate']])}
    table = pd.DataFrame(records)
    grouped = table[table.config.isin(allowed)].groupby('config').agg(
        mean=('log_loss', 'mean'), std=('log_loss', 'std'), count=('log_loss', 'size'),
        complexity=('active_features', 'mean'))
    if not (grouped['count'] == len(YEARS)).all():
        raise ValueError('Incomplete chronological folds')
    best = grouped['mean'].idxmin()
    threshold = float(grouped.loc[best, 'mean'] + grouped.loc[best, 'std'] / np.sqrt(len(YEARS)))
    eligible = grouped[grouped['mean'] <= threshold].index
    chosen = min(eligible, key=lambda n: (grouped.loc[n, 'complexity'],
        len(sets[cfgs[n]['candidate']]), grouped.loc[n, 'mean'], n))
    return dict(config=chosen, **cfgs[chosen], columns=sets[cfgs[chosen]['candidate']],
                best_mean_config=best, cv_log_loss=float(grouped.loc[chosen, 'mean']),
                best_mean_log_loss=float(grouped.loc[best, 'mean']), threshold=threshold)


def evaluate_scores(p, g, frame):
    result = scores(p, labels(frame))
    result['accuracy'] = float((p.argmax(axis=1) == labels(frame)).mean())
    if g is not None:
        result['goal_mse'] = float(np.mean((g - frame[['home_score', 'away_score']].to_numpy())**2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trees', type=int, default=300)
    args = parser.parse_args()
    if args.trees < 100:
        parser.error('At least 100 trees required')
    audit = json.loads((OUT / 'data_audit.json').read_text())
    for split, digest in audit['output_hashes'].items():
        if hashlib.sha256((OUT / (split+'.csv')).read_bytes()).hexdigest() != digest:
            raise ValueError('Prepared data changed; rebuild audit')
    train = pd.read_csv(OUT / 'train.csv')
    base_model = joblib.load(ROOT / 'results/expanded_comparison/expanded__RF_vector.joblib')
    sets = candidates(base_model['categorical'] + base_model['numeric'])
    source_files = [Path(__file__), ROOT/'src/match_form.py', ROOT/'src/poisson_lasso.py',
                    ROOT/'scripts/compare_attribute_regressions.py',
                    ROOT/'scripts/select_expanded_features.py', ROOT/'scripts/compare_ordered_models.py',
                    ROOT/'scripts/compare_expanded_models.py']
    plan = dict(families=FAMILIES, candidates=sets, validation_seasons=YEARS,
        audit_season=2021, test_season=2022, trees=args.trees, alphas=ALPHAS,
        data_audit=audit, code_hashes={p.relative_to(ROOT).as_posix():
                    hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        selection='Minimum mean seasonal H/D/A log loss plus one-standard-error simplicity rule. '
                  'Without-form and form-allowed selections both can remove original features. '
                  'Form-allowed selection may reject every new feature. '
                  'Poisson lasso uses one shared alpha across two separately fitted goal equations, '
                  'jointly selected by outcome log loss; not a dependent bivariate Poisson model. '
                  'Three-fold standard error is a simplicity heuristic.',
        evaluation='Fixed seasonal model; pre-match features update using completed prior fixtures. '
                   'Matched rows for every model/variant. Pool four leagues; report each separately. '
                   'All preprocessing fitted only on training fold. No active model promotion.')
    (OUT / 'selection_plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    cache = OUT / 'cache'
    cache.mkdir(exist_ok=True)
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()

    def evaluate(family, config, tr, va, split, save=None):
        if tr.empty or va.empty or tr.fixture_date.max() >= va.fixture_date.min():
            raise ValueError('Invalid chronological split')
        key = hashlib.sha256((signature + family + json.dumps(config, sort_keys=True) + split).encode()).hexdigest()
        path = cache / (key + '.npz')
        meta = cache / (key + '.json')
        if save is None and path.exists() and meta.exists():
            with np.load(path) as f:
                p, g = f['p'], f['g']
            return p, g if g.size else None, json.loads(meta.read_text())
        print(f'{family} / {config} / {split}: {len(tr)} -> {len(va)}', flush=True)
        bundle, diag = train_model(family, tr, sets[config['candidate']], config['alpha'], args.trees)
        p, g = infer(bundle, va)
        if save is not None:
            bundle['experiment_signature'] = signature
            joblib.dump(bundle, save)
            # Verify persisted models, including preprocessing, reproduce predictions.
            loaded_p, loaded_g = infer(joblib.load(save), va.iloc[:20])
            np.testing.assert_allclose(loaded_p, p[:20], atol=1e-10)
            if g is not None:
                np.testing.assert_allclose(loaded_g, g[:20], atol=1e-10)
        np.savez_compressed(path, p=p, g=g if g is not None else np.array([]))
        meta.write_text(json.dumps(diag), encoding='utf-8')
        return p, g, diag

    records, selections = [], {}
    for family in FAMILIES:
        cfgs = configs(family, sets)
        family_rows = []
        for name, config in cfgs.items():
            for year in YEARS:
                tr, va = train[train.season < year], train[train.season == year]
                p, g, diag = evaluate(family, config, tr, va, str(year))
                row = dict(family=family, config=name, season=year, **diag, **evaluate_scores(p, g, va))
                family_rows.append(row)
                records.append(row)
        selections[family] = {
            'selected_without_form': select(family_rows, cfgs, sets, False),
            'selected_form_allowed': select(family_rows, cfgs, sets, True)}
        pd.DataFrame(records).to_csv(OUT / 'cv_scores.csv', index=False)
        (OUT / 'selection.json').write_text(json.dumps(selections, indent=2), encoding='utf-8')
        print('Selected ' + family + ': ' + json.dumps(selections[family]), flush=True)
    # No audit/test metric participates in the frozen selection.
    test = pd.read_csv(OUT / 'test.csv')
    metrics, predictions, coefficients = [], [], []
    for split, tr, va in [('audit_2021', train[train.season < 2021], train[train.season == 2021]),
                          ('test_2022', train, test)]:
        for family in FAMILIES:
            choices = selections[family]
            variants = {k: dict(candidate=v['candidate'], alpha=v['alpha']) for k, v in choices.items()}
            variants['minimum_cv_form_allowed'] = configs(family, sets)[choices['selected_form_allowed']['best_mean_config']]
            # A fixed 5-match ablation complements feature selection.
            if family != 'Poisson_lasso':
                variants.update(original=dict(candidate='original', alpha=None),
                                original_plus_form5=dict(candidate='original_all_5', alpha=None))
            evaluated = {}
            for variant, config in variants.items():
                token = json.dumps(config, sort_keys=True)
                if token not in evaluated:
                    dest = OUT / (family + '_' + split + '_' + config['candidate'] + '_' +
                                  str(config['alpha']) + '.joblib') if split == 'test_2022' else None
                    evaluated[token] = evaluate(family, config, tr, va, split, dest)
                p, g, diag = evaluated[token]
                for league in ['ALL', *sorted(va.league.unique())]:
                    mask = np.ones(len(va), bool) if league == 'ALL' else (va.league == league).to_numpy()
                    metrics.append(dict(family=family, variant=variant, split=split, league=league,
                        candidate=config['candidate'], alpha=config['alpha'], matches=int(mask.sum()),
                        **diag, **evaluate_scores(p[mask], g[mask] if g is not None else None, va.loc[mask])))
                outcomes = labels(va)
                for i, row in enumerate(va.itertuples()):
                    predictions.append(dict(family=family, variant=variant, split=split,
                        source_row=int(row.source_row), date=row.fixture_date, league=row.league,
                        outcome=int(outcomes[i]), p_home=p[i, 0], p_draw=p[i, 1], p_away=p[i, 2]))
                if split == 'test_2022' and family == 'Poisson_lasso':
                    dest = OUT / (family + '_' + split + '_' + config['candidate'] + '_' + str(config['alpha']) + '.joblib')
                    bundle = joblib.load(dest)
                    for side, model in zip(['home', 'away'], bundle['models']):
                        coefficients.extend(dict(variant=variant, target=side, feature=n, source_feature=s,
                            coefficient=float(c)) for n, s, c in zip(
                            bundle['feature_names'], bundle['feature_sources'], model['coef']) if c != 0)
        pd.DataFrame(metrics).to_csv(OUT / 'metrics.csv', index=False)
    pred = pd.DataFrame(predictions)
    pred.to_csv(OUT / 'predictions.csv', index=False)
    pd.DataFrame(coefficients).to_csv(OUT / 'lasso_retained_coefficients.csv', index=False)
    table = pd.DataFrame(metrics)
    rows = []
    for family in FAMILIES:
        m = table[(table.family == family) & (table.split == 'test_2022') & (table.league == 'ALL')].set_index('variant')
        old, new = m.loc['selected_without_form'], m.loc['selected_form_allowed']
        choice = selections[family]['selected_form_allowed']
        rows.append(dict(model=family, without_form_log_loss=old.log_loss,
            form_allowed_log_loss=new.log_loss, difference=new.log_loss-old.log_loss,
            minimum_cv_form_allowed_log_loss=m.loc['minimum_cv_form_allowed'].log_loss,
            selected_without=old.candidate, selected_with=new.candidate,
            without_features=int(old.active_features), with_features=int(new.active_features),
            selected_columns=', '.join(choice['columns'])))
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / 'comparison.csv', index=False)
    # Paired calendar-date bootstrap, preserving within-day fixture clustering.
    paired = []
    for family in FAMILIES:
        subset = pred[(pred.family == family) & (pred.split == 'test_2022')]
        aa = subset[subset.variant == 'selected_without_form'].sort_values('source_row')
        bb = subset[subset.variant == 'selected_form_allowed'].sort_values('source_row')
        assert aa.source_row.tolist() == bb.source_row.tolist()
        y = aa.outcome.to_numpy()
        loss = lambda f: -np.log(np.clip(f[['p_home','p_draw','p_away']].to_numpy()[np.arange(len(y)), y], 1e-15, 1))
        delta = pd.DataFrame(dict(date=aa.date.to_numpy(), difference=loss(bb)-loss(aa)))
        days = delta.groupby('date').difference.agg(['sum','size']).to_numpy()
        rng = np.random.default_rng(20261020)
        picks = rng.integers(len(days), size=(2000, len(days)))
        sampled = days[picks]
        diffs = sampled[:, :, 0].sum(axis=1) / sampled[:, :, 1].sum(axis=1)
        paired.append(dict(model=family, difference=float(delta.difference.mean()),
            ci_low=float(np.quantile(diffs, .025)), ci_high=float(np.quantile(diffs, .975))))
    pd.DataFrame(paired).to_csv(OUT / 'paired_differences.csv', index=False)
    body = '<h1>Attacking, defensive and shots-on-target form</h1>'
    body += '<p>' + html.escape(audit['rules']) + '</p><p>' + html.escape(audit['limitations']) + '</p>'
    body += '<p>' + html.escape(plan['selection']) + '</p><p>' + html.escape(plan['evaluation']) + '</p>'
    body += f'<p>{len(train):,} training and {len(test):,} matched test fixtures. Validation: 2018–19, 2019–20, 2020–21. Audit: 2021–22. Test: 2022–23.</p>'
    body += '<h2>Selected models: test log loss (lower is better)</h2>' + comparison.drop(columns='selected_columns').to_html(index=False)
    body += '<p>Feature counts are encoded dimensions; lasso counts the union of nonzero coefficients across its two goal equations. '
    body += 'The minimum-CV column is selected on validation, never on test. All model settings are fixed except the declared feature/window and lasso-alpha search.</p>'
    body += '<h2>Paired differences</h2><p>Form allowed minus without form; 2,000 date-cluster resamples. Intervals do not account for repeated test inspection or all team/time dependence.</p>'
    body += pd.DataFrame(paired).to_html(index=False)
    body += '<h2>Chosen feature sets</h2>' + comparison[['model','selected_without','selected_with','selected_columns']].to_html(index=False)
    body += '<h2>Audit, fixed ablations and league breakdown</h2>' + table.to_html(index=False)
    (OUT / 'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Past-match form comparison</title>'
        '<style>body{font:15px system-ui;margin:30px;color:#182532}table{border-collapse:collapse;font-size:12px}'
        'td,th{padding:7px;border-bottom:1px solid #ddd;max-width:650px}h2{margin-top:32px}</style>'+body, encoding='utf-8')
    (OUT / 'summary.json').write_text(json.dumps(dict(comparison=rows, paired_differences=paired,
        selection=selections, experiment_signature=signature), indent=2), encoding='utf-8')
    print(comparison.drop(columns='selected_columns').to_string(index=False), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
