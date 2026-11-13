"""Train and evaluate positional-unit matchups on identical chronological samples."""
import argparse
import hashlib
import html
import json
from pathlib import Path
from select_expanded_features import ROOT, np, pd, joblib, threadpool_limits
from compare_team_attributes import candidates as attribute_candidates, labels, fit_model as fit_forest
from compare_attribute_regressions import fit_model as fit_regression, probabilities
from compare_expanded_models import outcome_probabilities
from calibrate_models import scores
from positional_features import feature_columns

OUT = ROOT / 'results/positional_matchups'
FAMILIES = ['Poisson', 'RF_goals']


def candidates(base):
    attrs = attribute_candidates(base)['all_attributes_and_matchups']
    positions = feature_columns()
    return dict(baseline=base, team_attributes=attrs,
                positional=base + positions, attributes_and_positions=attrs + positions)


def fit_model(frame, columns, family, trees):
    if family == 'Poisson':
        model, diagnostics = fit_regression('Poisson', frame, columns)
    else:
        model, diagnostics = fit_forest(frame, columns, family, trees), {}
    return dict(family=family, fitted=model, columns=columns, diagnostics=diagnostics)


def predict(bundle, frame):
    fitted = bundle['fitted']
    if bundle['family'] == 'Poisson':
        p, goals = probabilities(fitted, frame)
    else:
        x = np.column_stack([fitted['encoder'].transform(frame[fitted['categorical']]),
                             fitted['imputer'].transform(frame[fitted['numeric']])])
        goals = fitted['model'].predict(x)
        p = outcome_probabilities(goals)
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError('Invalid predicted outcome probabilities')
    return p, goals


def chronological_split(frame, year):
    train, valid = frame[frame.season < year], frame[frame.season == year]
    check_dates(train, valid)
    return train, valid


def check_dates(train, valid):
    if train.empty or valid.empty or pd.to_datetime(train.fixture_date).max() >= pd.to_datetime(valid.fixture_date).min():
        raise ValueError('Invalid chronological split')


def select(records):
    table = pd.DataFrame(records)
    selected = {}
    for family, g in table.groupby('family'):
        means = g.groupby('candidate').log_loss.mean().sort_values()
        selected[family] = dict(candidate=means.index[0], mean_log_loss=float(means.iloc[0]))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trees', type=int, default=500)
    args = parser.parse_args()
    if args.trees < 100:
        parser.error('Use at least 100 trees')
    train = pd.read_csv(OUT / 'train.csv')
    original = joblib.load(ROOT / 'results/expanded_comparison/expanded__RF_vector.joblib')
    sets = candidates(original['categorical'] + original['numeric'])
    feature_audit = json.loads((OUT / 'feature_audit.json').read_text())
    paths = [OUT / 'train.csv', OUT / 'test.csv', OUT / 'feature_audit.json', Path(__file__),
        ROOT / 'src/positional_features.py', ROOT / 'scripts/compare_team_attributes.py',
        ROOT / 'scripts/compare_attribute_regressions.py', ROOT / 'scripts/select_expanded_features.py',
        ROOT / 'scripts/compare_expanded_models.py']
    plan = dict(candidates=sets, families=FAMILIES, trees=args.trees,
        selection_seasons=[2019, 2020], audit_season=2021, test_season=2022,
        selection_rule='Minimum unweighted mean seasonal outcome log loss on 2019 and 2020 only; '
            'all four predeclared candidates reported on audit/test, with no subsequent retuning.',
        preprocessing='Fold-only median imputation and missing indicators, team encoding; '
            'Poisson alpha=0.01 with scaled numeric features; RF depth=5, max_features=0.3, '
            'max_samples=0.5. Independent Poisson conversion to home/draw/away probabilities.',
        position_rule=feature_audit['position_rule'], missing_rule=feature_audit['missing_rule'],
        caveat=feature_audit['caveat'] + ' Audit and test seasons have been inspected previously.',
        input_hashes={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (OUT / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    records = []
    for family in FAMILIES:
        for candidate, columns in sets.items():
            for year in plan['selection_seasons']:
                tr, va = chronological_split(train, year)
                print(f'Validation {year}: {family} / {candidate}', flush=True)
                fitted = fit_model(tr, columns, family, args.trees)
                p, _ = predict(fitted, va)
                records.append(dict(family=family, candidate=candidate, season=year,
                    train_matches=len(tr), matches=len(va), **scores(p, labels(va))))
    selected = select(records)
    (OUT / 'selection.json').write_text(json.dumps(selected, indent=2), encoding='utf-8')
    pd.DataFrame(records).to_csv(OUT / 'selection_scores.csv', index=False)
    # Choices are frozen before evaluating these previously inspected periods.
    test = pd.read_csv(OUT / 'test.csv')
    metrics, predictions, paired = [], [], []
    for split, tr, va in [('Forward audit 2021-22', *chronological_split(train, 2021)),
                          ('Test 2022-23 (previously inspected)', train, test)]:
        check_dates(tr, va)
        y = labels(va)
        for family in FAMILIES:
            losses = {}
            for candidate, columns in sets.items():
                print(f'{split}: {family} / {candidate}', flush=True)
                bundle = fit_model(tr, columns, family, args.trees)
                p, goals = predict(bundle, va)
                losses[candidate] = -np.log(np.maximum(p[np.arange(len(y)), y], 1e-15))
                metrics.append(dict(family=family, candidate=candidate, split=split,
                    selected=candidate == selected[family]['candidate'], matches=len(va),
                    accuracy=float(np.mean(p.argmax(axis=1) == y)), **scores(p, y),
                    goal_mae=float(np.abs(goals - va[['home_score', 'away_score']].to_numpy()).mean())))
                for i, row in enumerate(va.itertuples()):
                    predictions.append(dict(family=family, candidate=candidate, split=split,
                        source_row=row.source_row, date=row.fixture_date, league=row.league,
                        home_team=row.home_team_name, away_team=row.away_team_name,
                        home_score=row.home_score, away_score=row.away_score, outcome=int(y[i]),
                        p_home=float(p[i, 0]), p_draw=float(p[i, 1]), p_away=float(p[i, 2]),
                        expected_home_goals=float(goals[i, 0]), expected_away_goals=float(goals[i, 1])))
                if split.startswith('Test'):
                    bundle.update(experiment_plan=plan, selection=selected[family])
                    path = OUT / f'{family}__{candidate}.joblib'
                    joblib.dump(bundle, path)
                    restored = joblib.load(path)
                    rp, rg = predict(restored, va.head(12))
                    if not np.allclose(rp, p[:12]) or not np.allclose(rg, goals[:12]):
                        raise ValueError('Saved-model prediction mismatch')
            for candidate, baseline in [('positional', 'baseline'),
                                         ('attributes_and_positions', 'team_attributes')]:
                delta = losses[candidate] - losses[baseline]
                # Resample calendar-week blocks, preserving pairing across models.
                groups = pd.to_datetime(va.fixture_date).dt.to_period('W').astype(str).to_numpy()
                blocks = [delta[groups == week] for week in np.unique(groups)]
                sums, counts = np.array([b.sum() for b in blocks]), np.array([len(b) for b in blocks])
                rng = np.random.default_rng(20261110)
                sample = rng.integers(0, len(blocks), size=(2000, len(blocks)))
                boot = sums[sample].sum(axis=1) / counts[sample].sum(axis=1)
                paired.append(dict(family=family, candidate=candidate, comparator=baseline, split=split,
                    log_loss_difference=float(delta.mean()), lower_95=float(np.quantile(boot, .025)),
                    upper_95=float(np.quantile(boot, .975)), week_blocks=len(blocks)))
        market = 1 / va[['B365H', 'B365D', 'B365A']].to_numpy(float)
        market /= market.sum(axis=1, keepdims=True)
        metrics.append(dict(family='Market', candidate='normalized_odds', split=split,
            selected=False, matches=len(va), accuracy=float(np.mean(market.argmax(axis=1) == y)),
            **scores(market, y)))
        pd.DataFrame(metrics).to_csv(OUT / 'metrics.csv', index=False)
    pd.DataFrame(predictions).to_csv(OUT / 'predictions.csv', index=False)
    pd.DataFrame(paired).to_csv(OUT / 'paired_differences.csv', index=False)
    summary = dict(selection=selected, metrics=metrics, paired_differences=paired)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    body = '<h1>Preferred-position matchups</h1><p>Attack versus opposing defense and goalkeeper; '
    body += 'midfield versus midfield. Positive gaps favor the named side; midfield gaps favor home.</p>'
    for key in ['position_rule', 'missing_rule', 'selection_rule', 'caveat']:
        body += '<p>' + html.escape(plan[key]) + '</p>'
    body += '<h2>Prediction quality</h2><p>Lower log loss and goal MAE are better. '
    body += 'All candidates retain the original team, form, odds and overall-rating inputs.</p>'
    body += pd.DataFrame(metrics).to_html(index=False, float_format=lambda x: f'{x:.5f}')
    body += '<h2>Incremental effect of positions</h2><p>Negative differences favor positions. '
    body += 'Approximate 95% intervals resample calendar weeks; they do not account for all '
    body += 'team dependence, historical experiment selection or repeated test inspection.</p>'
    body += pd.DataFrame(paired).to_html(index=False, float_format=lambda x: f'{x:.5f}')
    body += '<h2>Coverage</h2>' + pd.read_csv(OUT / 'coverage.csv').to_html(index=False)
    body += '<h2>Training-only selection</h2>' + pd.DataFrame(records).to_html(index=False)
    (OUT / 'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Position matchups</title>'
        '<style>body{font:15px system-ui;margin:32px;color:#182532}table{border-collapse:collapse;'
        'font-size:12px}td,th{padding:8px;border-bottom:1px solid #ddd}</style>' + body, encoding='utf-8')
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)
    print('Validation-selected candidates:', selected, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
