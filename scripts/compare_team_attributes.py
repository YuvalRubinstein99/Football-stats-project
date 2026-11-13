"""Chronological group selection for team skill comparisons; frozen before audit/test."""
import argparse
import hashlib
import html
import json
import time
from pathlib import Path
from select_expanded_features import ROOT, pd, np, joblib, threadpool_limits
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from compare_expanded_models import outcome_probabilities
from calibrate_models import scores
from prepare_team_attributes import OUT, ATTRIBUTES


def candidates(base):
    groups = {att: [f'{side}_{att}_{suffix}' for side in ('home', 'away')
                   for suffix in ('mean', 'count')] + [att + '_advantage']
              for att in ATTRIBUTES}
    matchups = [f'{side}_{att}_vs_defending' for side in ('home', 'away')
                for att in ('shooting', 'dribbling', 'pace')]
    all_attributes = sum(groups.values(), [])
    return {'baseline': base, **{att: base + cols for att, cols in groups.items()},
            'all_attributes': base + all_attributes,
            'matchups': base + groups['pace'] + groups['shooting'] + groups['dribbling']
                       + groups['defending'] + matchups,
            'all_attributes_and_matchups': base + all_attributes + matchups}


def fit_model(frame, columns, family, trees):
    cats = [c for c in columns if c.endswith('_team_name')]
    nums = [c for c in columns if c not in cats]
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    imputer = SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)
    x = np.column_stack([encoder.fit_transform(frame[cats]), imputer.fit_transform(frame[nums])])
    params = dict(n_estimators=trees, max_depth=5, max_features=.3, max_samples=.5,
                  random_state=20261020, n_jobs=4)
    goals = frame[['home_score', 'away_score']].to_numpy()
    if family == 'RF_classifier':
        estimator = RandomForestClassifier(criterion='log_loss', **params)
        y = labels(frame)
    else:
        estimator = RandomForestRegressor(criterion='squared_error', **params)
        y = goals
    estimator.fit(x, y)
    return dict(family=family, columns=columns, categorical=cats, numeric=nums,
                encoder=encoder, imputer=imputer, model=estimator)


def predict(bundle, frame):
    x = np.column_stack([bundle['encoder'].transform(frame[bundle['categorical']]),
                         bundle['imputer'].transform(frame[bundle['numeric']])])
    model = bundle['model']
    if bundle['family'] == 'RF_classifier':
        if not np.array_equal(model.classes_, [0, 1, 2]):
            raise ValueError('Unexpected class order')
        return model.predict_proba(x)
    return outcome_probabilities(model.predict(x))


def labels(frame):
    return (1 - np.sign(frame.home_score - frame.away_score)).to_numpy(int)


def choose(records, sets):
    table = pd.DataFrame(records)
    grouped = table.groupby('candidate').log_loss.agg(['mean', 'std', 'count'])
    best = grouped['mean'].idxmin()
    threshold = float(grouped.loc[best, 'mean'] + grouped.loc[best, 'std'] / np.sqrt(grouped.loc[best, 'count']))
    eligible = grouped[grouped['mean'] <= threshold].index
    chosen = min(eligible, key=lambda n: (len(sets[n]), grouped.loc[n, 'mean'], n))
    return dict(candidate=chosen, columns=sets[chosen], best_mean_candidate=best,
                mean_log_loss=float(grouped.loc[chosen, 'mean']), threshold=threshold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trees', type=int, default=500)
    args = parser.parse_args()
    if args.trees < 100:
        parser.error('Use at least 100 trees for the comparison')
    started = time.monotonic()
    train_path, test_path = OUT / 'train.csv', OUT / 'test.csv'
    train = pd.read_csv(train_path)
    baseline_bundle = joblib.load(ROOT / 'results/expanded_comparison/expanded__RF_vector.joblib')
    base = baseline_bundle['categorical'] + baseline_bundle['numeric']
    sets = candidates(base)
    codepaths = [Path(__file__), ROOT / 'scripts/prepare_team_attributes.py',
                 ROOT / 'scripts/compare_expanded_models.py']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [train_path, test_path, OUT / 'feature_audit.json', *codepaths]}
    plan = dict(candidates=sets, selection_years=[2019, 2020], audit_year=2021,
                test_year=2022, trees=args.trees, input_hashes=hashes,
                families=['RF_classifier', 'RF_goals'],
                rule='Fewest raw columns within one standard error of best mean seasonal log loss. '
                     'Two seasonal folds make this a simplicity heuristic, not a confidence interval.',
                sample='Identical named-lineup rows for all models, 2014 onward, Spain/Germany/Italy. '
                       'Missing attributes retained; medians/indicators/encoders fitted inside each fold.',
                test_caveat='Previously inspected 2022-23 season. FIFA edition publication dates unverified.')
    (OUT / 'selection_plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    records, selection = [], {}
    cache_dir = OUT / 'cache'
    cache_dir.mkdir(exist_ok=True)

    def forward(family, candidate, tr, va, split):
        key = hashlib.sha256(json.dumps(dict(plan=plan, family=family, candidate=candidate,
                             train=tr.index.tolist(), valid=va.index.tolist(), split=split),
                             sort_keys=True).encode()).hexdigest()
        cache = cache_dir / (key + '.npz')
        if cache.exists():
            with np.load(cache) as saved:
                return saved['p']
        print(f'{family} / {candidate} / {split}: {len(tr)} -> {len(va)} matches', flush=True)
        fitted = fit_model(tr, sets[candidate], family, args.trees)
        p = predict(fitted, va)
        np.savez_compressed(cache, p=p)
        return p

    for family in plan['families']:
        family_records = []
        for candidate in sets:
            for year in plan['selection_years']:
                tr, va = train[train.season < year], train[train.season == year]
                if tr.empty or va.empty or tr.fixture_date.max() >= va.fixture_date.min():
                    raise ValueError('Invalid chronological fold')
                p = forward(family, candidate, tr, va, str(year))
                record = dict(family=family, candidate=candidate, season=year,
                              train_matches=len(tr), validation_matches=len(va),
                              log_loss=float(log_loss(labels(va), p, labels=[0, 1, 2])))
                family_records.append(record)
                records.append(record)
        selection[family] = choose(family_records, sets)
    # Persist choices before reading audit/test labels for model comparisons.
    (OUT / 'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    pd.DataFrame(records).to_csv(OUT / 'selection_scores.csv', index=False)
    test = pd.read_csv(test_path)
    metrics, predictions = [], []
    for split, tr, va in [
        ('Forward audit 2021-22', train[train.season < 2021], train[train.season == 2021]),
        ('Test 2022-23 (previously inspected)', train, test),
    ]:
        if tr.fixture_date.max() >= va.fixture_date.min():
            raise ValueError('Training overlaps evaluation dates')
        for family in plan['families']:
            chosen = selection[family]['candidate']
            for candidate in dict.fromkeys(['baseline', 'all_attributes_and_matchups', chosen]):
                p = forward(family, candidate, tr, va, split)
                metrics.append(dict(family=family, candidate=candidate, split=split,
                                    selected=candidate == chosen, matches=len(va), **scores(p, labels(va))))
                predictions.extend(dict(family=family, candidate=candidate, split=split,
                    source_row=int(row.source_row), date=row.fixture_date, league=row.league,
                    outcome=int(y), p_home=float(pp[0]), p_draw=float(pp[1]), p_away=float(pp[2]))
                    for (_, row), y, pp in zip(va.iterrows(), labels(va), p))
        # A matched-sample market reference, not a different dataset's baseline.
        market = 1 / va[['B365H', 'B365D', 'B365A']].to_numpy(float)
        market /= market.sum(axis=1, keepdims=True)
        metrics.append(dict(family='Market', candidate='normalized_odds', split=split,
                            selected=False, matches=len(va), **scores(market, labels(va))))
    for family, selected in selection.items():
        fitted = fit_model(train, selected['columns'], family, args.trees)
        fitted['selection'] = selected
        fitted['experiment_plan'] = plan
        joblib.dump(fitted, OUT / (family + '_selected.joblib'))
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(OUT / 'metrics.csv', index=False)
    pd.DataFrame(predictions).to_csv(OUT / 'predictions.csv', index=False)
    summary = dict(plan=plan, selection=selection, metrics=metrics.to_dict('records'),
                   elapsed_seconds=time.monotonic() - started)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    coverage = pd.read_csv(OUT / 'coverage.csv')
    body = '<h1>Team attributes and chronological feature selection</h1>'
    body += '<p>Same matches, two forest families, fixed settings. Lower log loss and Brier are better.</p>'
    body += '<p>' + html.escape(plan['test_caveat']) + ' Missing attributes use training-only imputation.</p>'
    body += '<h2>Selected groups</h2>' + pd.DataFrame([
        dict(model=k, selected=v['candidate'], best_mean=v['best_mean_candidate'],
             selection_log_loss=v['mean_log_loss']) for k, v in selection.items()
    ]).to_html(index=False)
    body += '<h2>Audit and test</h2>' + metrics.to_html(index=False, float_format=lambda x: f'{x:.6f}')
    body += '<h2>Attribute coverage</h2>' + coverage.to_html(index=False)
    body += '<h2>Selection folds</h2>' + pd.DataFrame(records).to_html(index=False)
    (OUT / 'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Team attributes</title>'
        '<style>body{font:15px system-ui;max-width:1200px;margin:32px auto;color:#182532}'
        'td,th{padding:8px;border-bottom:1px solid #ddd}table{border-collapse:collapse;font-size:13px}</style>'
        + body, encoding='utf-8')
    print(metrics.to_string(index=False), flush=True)
    print('Selected:', {k: v['candidate'] for k, v in selection.items()}, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
