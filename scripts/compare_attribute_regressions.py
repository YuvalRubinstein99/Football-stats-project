"""Team subratings and genuinely reduced predictor sets for four regression families."""
import hashlib
import html
import json
from pathlib import Path
from select_expanded_features import ROOT, np, pd, joblib, fit, predict, threadpool_limits
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from calibrate_models import scores
from compare_team_attributes import candidates as forest_candidates, labels
from prepare_team_attributes import ATTRIBUTES

DATA = ROOT / 'results/team_attributes'
OUT = DATA / 'regressions'
FAMILIES = ['Linear', 'Poisson', 'Ordered_logit', 'Ordered_probit']


def candidates(base):
    odds = ['B365H', 'B365D', 'B365A']
    overall = ['HomePlayer_Overall_mean', 'AwayPlayer_Overall_mean']
    form = [c for c in base if c.endswith(('_form', '_prior'))]
    core = odds + overall + form + ['Matchweek']
    gaps = [a + '_advantage' for a in ATTRIBUTES]
    means = [f'{side}_{a}_mean' for side in ('home', 'away') for a in ATTRIBUTES]
    matchups = [f'{side}_{a}_vs_defending' for side in ('home', 'away')
                for a in ('shooting', 'dribbling', 'pace')]
    expanded = forest_candidates(base)['all_attributes_and_matchups']
    sets = dict(
        original=base,
        all_attributes=expanded,
        all_without_teams=[c for c in expanded if not c.endswith('_team_name')],
        original_without_teams=[c for c in base if not c.endswith('_team_name')],
        odds_only=odds,
        odds_overall=odds + overall,
        odds_form=odds + form,
        odds_overall_form=odds + overall + form,
        compact_original=core,
        odds_skill_gaps=odds + gaps,
        odds_skill_means=odds + means,
        odds_matchups=odds + matchups,
        compact_skill_gaps=core + gaps,
        compact_skill_means=core + means,
        compact_matchups=core + matchups,
        compact_all=core + gaps + matchups,
    )
    # Single-attribute candidates can replace most original predictors, rather
    # than only append another variable to the complete original model.
    for attribute in ATTRIBUTES:
        sets['odds_' + attribute] = odds + [attribute + '_advantage']
    return sets


def design(train, columns):
    cats = [c for c in columns if c.endswith('_team_name')]
    nums = [c for c in columns if c not in cats]
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) if cats else None
    imputer = SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)
    scaler = StandardScaler()
    numeric = scaler.fit_transform(imputer.fit_transform(train[nums]))
    categorical = encoder.fit_transform(train[cats]) if cats else np.empty((len(train), 0))
    x = np.column_stack([categorical, numeric])
    center = x.mean(axis=0)
    _, singular, vh = np.linalg.svd(x - center, full_matrices=False)
    keep = singular > (singular[0] * 1e-10 if len(singular) else 0)
    basis = vh[keep].T / (singular[keep] / np.sqrt(len(train)))
    artifact = dict(columns=columns, categorical=cats, numeric=nums, encoder=encoder,
                    imputer=imputer, scaler=scaler, center=center, basis=basis)
    return artifact, x, (x - center) @ basis


def transform(artifact, frame):
    numeric = artifact['scaler'].transform(artifact['imputer'].transform(frame[artifact['numeric']]))
    categorical = (artifact['encoder'].transform(frame[artifact['categorical']])
                   if artifact['categorical'] else np.empty((len(frame), 0)))
    x = np.column_stack([categorical, numeric])
    return x, (x - artifact['center']) @ artifact['basis']


def fit_model(family, frame, columns):
    artifact, x, xo = design(frame, columns)
    model, diagnostics = fit(family, x, xo, frame[['home_score', 'away_score']].to_numpy())
    return dict(model=model, preprocessing=artifact), dict(
        **diagnostics, raw_features=len(columns), encoded_features=x.shape[1], rank=xo.shape[1])


def probabilities(bundle, frame):
    x, xo = transform(bundle['preprocessing'], frame)
    return predict(bundle['model'], x, xo)


def choose(records, sets):
    table = pd.DataFrame(records)
    grouped = table.groupby('candidate').agg(mean=('log_loss', 'mean'), std=('log_loss', 'std'),
        count=('log_loss', 'count'), encoded=('encoded_features', 'mean'))
    best = grouped['mean'].idxmin()
    threshold = float(grouped.loc[best, 'mean'] + grouped.loc[best, 'std'] / np.sqrt(grouped.loc[best, 'count']))
    eligible = grouped[grouped['mean'] <= threshold].index
    selected = min(eligible, key=lambda n: (grouped.loc[n, 'encoded'], len(sets[n]), grouped.loc[n, 'mean'], n))
    return dict(candidate=selected, columns=sets[selected], raw_features=len(sets[selected]),
                mean_encoded_features=float(grouped.loc[selected, 'encoded']),
                best_mean_candidate=best, mean_log_loss=float(grouped.loc[selected, 'mean']),
                best_mean_log_loss=float(grouped.loc[best, 'mean']), threshold=threshold)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / 'cache'
    cache.mkdir(exist_ok=True)
    train = pd.read_csv(DATA / 'train.csv')
    baseline = joblib.load(ROOT / 'results/expanded_comparison/expanded__RF_vector.joblib')
    base = baseline['categorical'] + baseline['numeric']
    sets = candidates(base)
    paths = [DATA / 'train.csv', DATA / 'test.csv', DATA / 'feature_audit.json',
             Path(__file__), ROOT / 'scripts/select_expanded_features.py',
             ROOT / 'scripts/compare_ordered_models.py', ROOT / 'scripts/compare_expanded_models.py',
             ROOT / 'scripts/compare_team_attributes.py']
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    plan = dict(families=FAMILIES, candidates=sets, selection_seasons=[2019, 2020],
                audit_season=2021, test_season=2022, input_hashes=hashes,
                method='Predeclared reduced feature sets; smallest encoded dimension within one '
                       'standard error of minimum mean seasonal log loss, then raw column count. '
                       'The two-fold standard error is a simplicity heuristic, not a confidence interval.',
                preprocessing='Fold-only median imputation with indicators, scaling, team encoding '
                              'and SVD for ordered-model identifiability. Poisson alpha=0.01.',
                caveat='Same 8521 training and 1065 test matches as the attribute forest experiment. '
                       'Three leagues, partial attribute coverage; unverified FIFA snapshot publication '
                       'dates. The audit and test periods have been inspected before; results are exploratory.')
    (OUT / 'selection_plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')

    def evaluate(family, candidate, tr, va, split, save_model=False):
        if tr.empty or va.empty or tr.fixture_date.max() >= va.fixture_date.min():
            raise ValueError('Invalid chronological split')
        signature = dict(plan=plan, family=family, candidate=candidate, split=split,
                         train_rows=tr.source_row.tolist(), validation_rows=va.source_row.tolist())
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        path, meta = cache / (key + '.npz'), cache / (key + '.json')
        if not save_model and path.exists() and meta.exists():
            with np.load(path) as saved:
                p, goals = saved['p'], saved['goals']
            return p, goals if goals.size else None, json.loads(meta.read_text())
        print(f'{family} / {candidate} / {split}: {len(tr)} -> {len(va)}', flush=True)
        bundle, diagnostics = fit_model(family, tr, sets[candidate])
        p, goals = probabilities(bundle, va)
        np.savez_compressed(path, p=p, goals=goals if goals is not None else np.array([]))
        meta.write_text(json.dumps(diagnostics), encoding='utf-8')
        if save_model:
            bundle.update(selection=selection[family], experiment_plan=plan)
            joblib.dump(bundle, OUT / (family + '_selected.joblib'))
        return p, goals, diagnostics

    records, selection = [], {}
    for family in FAMILIES:
        family_records = []
        for name in sets:
            for year in plan['selection_seasons']:
                tr, va = train[train.season < year], train[train.season == year]
                p, _, diag = evaluate(family, name, tr, va, str(year))
                row = dict(family=family, candidate=name, season=year, train_matches=len(tr),
                           validation_matches=len(va), **diag, **scores(p, labels(va)))
                family_records.append(row)
                records.append(row)
        selection[family] = choose(family_records, sets)
        pd.DataFrame(records).to_csv(OUT / 'selection_scores.csv', index=False)
        print(f'Selected {family}: {selection[family]["candidate"]} '
              f'({selection[family]["raw_features"]} raw features)', flush=True)
    (OUT / 'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    # No audit/test comparison contributes to the saved choices.
    test = pd.read_csv(DATA / 'test.csv')
    metrics, predictions = [], []
    for split, tr, va in [
        ('Forward audit 2021-22', train[train.season < 2021], train[train.season == 2021]),
        ('Test 2022-23 (previously inspected)', train, test),
    ]:
        for family in FAMILIES:
            chosen = selection[family]
            names = dict.fromkeys(['original', 'all_attributes', chosen['best_mean_candidate'], chosen['candidate']])
            for name in names:
                selected = name == chosen['candidate']
                p, goals, diagnostics = evaluate(family, name, tr, va, split,
                                                save_model=split.startswith('Test') and selected)
                metric = dict(family=family, candidate=name, split=split, selected=selected,
                    matches=len(va), **diagnostics, **scores(p, labels(va)))
                if goals is not None:
                    actual = va[['home_score', 'away_score']].to_numpy()
                    metric.update(goal_mse=float(np.mean((goals-actual)**2)),
                                  goal_mae=float(np.mean(np.abs(goals-actual))))
                metrics.append(metric)
                predictions.extend(dict(family=family, candidate=name, split=split,
                    source_row=int(row.source_row), date=row.fixture_date, outcome=int(y),
                    p_home=float(pp[0]), p_draw=float(pp[1]), p_away=float(pp[2]))
                    for (_, row), y, pp in zip(va.iterrows(), labels(va), p))
        pd.DataFrame(metrics).to_csv(OUT / 'metrics.csv', index=False)
    pd.DataFrame(predictions).to_csv(OUT / 'predictions.csv', index=False)
    comparison = []
    table = pd.DataFrame(metrics)
    for family, chosen in selection.items():
        test_metrics = table[(table.family == family) & table.split.str.startswith('Test')].set_index('candidate')
        comparison.append(dict(model=family, selected=chosen['candidate'], original_features=len(base),
            all_features=len(sets['all_attributes']), selected_features=chosen['raw_features'],
            selected_encoded_features=int(test_metrics.loc[chosen['candidate'], 'encoded_features']),
            original_test_log_loss=test_metrics.loc['original', 'log_loss'],
            all_attributes_test_log_loss=test_metrics.loc['all_attributes', 'log_loss'],
            selected_test_log_loss=test_metrics.loc[chosen['candidate'], 'log_loss'],
            retained_features=', '.join(chosen['columns'])))
    comparison = pd.DataFrame(comparison)
    comparison.to_csv(OUT / 'comparison.csv', index=False)
    (OUT / 'summary.json').write_text(json.dumps(dict(plan=plan, selection=selection,
        comparison=comparison.to_dict('records')), indent=2), encoding='utf-8')
    body = '<h1>Regression attributes and feature reduction</h1><p>' + html.escape(plan['caveat']) + '</p>'
    body += '<p>' + html.escape(plan['method']) + '</p>'
    body += '<p>Raw feature counts include team names as individual columns. Encoded counts include '
    body += 'team dummies and missing-value indicators. Ordered models additionally remove exact collinearity.</p>'
    body += comparison.to_html(index=False, float_format=lambda x: f'{x:.6f}')
    body += '<h2>Audit and test metrics</h2>' + table.to_html(index=False, float_format=lambda x: f'{x:.6f}')
    body += '<details><summary>Selection folds and convergence</summary>' + pd.DataFrame(records).to_html(index=False) + '</details>'
    (OUT / 'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Regression feature reduction</title>'
        '<style>body{font:15px system-ui;margin:32px;color:#182532}table{border-collapse:collapse;font-size:12px}'
        'td,th{padding:8px;border-bottom:1px solid #ddd}td:last-child{max-width:450px}</style>' + body, encoding='utf-8')
    print(comparison.to_string(index=False), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
