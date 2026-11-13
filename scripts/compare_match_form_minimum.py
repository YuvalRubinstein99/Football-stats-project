"""Evaluate already-frozen minimum-CV choices on both sides of the form ablation.

No new search: both config identifiers were saved before audit/test evaluation.
Run after compare_match_form.py, then verify_match_form_results.py.
"""
import hashlib
import json
from pathlib import Path
from select_expanded_features import pd, np, joblib, threadpool_limits
from compare_match_form import OUT, FAMILIES, configs, train_model, infer, evaluate_scores, labels


def main():
    plan = json.loads((OUT / 'selection_plan.json').read_text())
    selections = json.loads((OUT / 'selection.json').read_text())
    metrics = pd.read_csv(OUT / 'metrics.csv')
    predictions = pd.read_csv(OUT / 'predictions.csv')
    # Idempotent: retain outputs from the original experiment, replace this supplement.
    metrics = metrics[metrics.variant != 'minimum_cv_without_form'].copy()
    predictions = predictions[predictions.variant != 'minimum_cv_without_form'].copy()
    train, test = [pd.read_csv(OUT / (s + '.csv')) for s in ['train', 'test']]
    for s in ['train', 'test']:
        assert hashlib.sha256((OUT / (s+'.csv')).read_bytes()).hexdigest() == plan['data_audit']['output_hashes'][s]
    new_metrics, new_predictions = [], []
    choices = []
    for family in FAMILIES:
        choice = selections[family]['selected_without_form']['best_mean_config']
        config = configs(family, plan['candidates'])[choice]
        cols = plan['candidates'][config['candidate']]
        assert not any('_recent_' in c or c.startswith('recent_') for c in cols)
        choices.append(dict(family=family, config=choice, **config))
        for split, tr, va in [('audit_2021', train[train.season < 2021], train[train.season == 2021]),
                              ('test_2022', train, test)]:
            print(f'Frozen minimum-CV baseline {family} / {choice} / {split}', flush=True)
            bundle, diag = train_model(family, tr, cols, config['alpha'], plan['trees'])
            p, g = infer(bundle, va)
            if split == 'test_2022':
                path = OUT / (family + '_minimum_cv_without_form.joblib')
                joblib.dump(bundle, path)
                np.testing.assert_allclose(infer(joblib.load(path), va.iloc[:20])[0], p[:20], atol=1e-10)
            for league in ['ALL', *sorted(va.league.unique())]:
                mask = np.ones(len(va), bool) if league == 'ALL' else (va.league == league).to_numpy()
                new_metrics.append(dict(family=family, variant='minimum_cv_without_form',
                    split=split, league=league, candidate=config['candidate'], alpha=config['alpha'],
                    matches=int(mask.sum()), **diag,
                    **evaluate_scores(p[mask], g[mask] if g is not None else None, va.loc[mask])))
            outcomes = labels(va)
            new_predictions.extend(dict(family=family, variant='minimum_cv_without_form', split=split,
                source_row=int(r.source_row), date=r.fixture_date, league=r.league, outcome=int(outcomes[i]),
                p_home=p[i,0], p_draw=p[i,1], p_away=p[i,2]) for i,r in enumerate(va.itertuples()))
    metrics = pd.concat([metrics, pd.DataFrame(new_metrics)], ignore_index=True)
    predictions = pd.concat([predictions, pd.DataFrame(new_predictions)], ignore_index=True)
    metrics.to_csv(OUT / 'metrics.csv', index=False)
    predictions.to_csv(OUT / 'predictions.csv', index=False)
    rows = []
    for family in FAMILIES:
        m = metrics[(metrics.family == family) & (metrics.split == 'test_2022') & (metrics.league == 'ALL')].set_index('variant')
        old, new = m.loc['minimum_cv_without_form'], m.loc['minimum_cv_form_allowed']
        sample = predictions[(predictions.family == family) & (predictions.split == 'test_2022')]
        a = sample[sample.variant == 'minimum_cv_without_form'].sort_values('source_row')
        b = sample[sample.variant == 'minimum_cv_form_allowed'].sort_values('source_row')
        assert a.source_row.tolist() == b.source_row.tolist()
        y = a.outcome.to_numpy()
        loss = lambda f: -np.log(np.clip(f[['p_home','p_draw','p_away']].to_numpy()[np.arange(len(y)), y], 1e-15, 1))
        differences = loss(b)-loss(a)
        # CSV round trips must not turn identical predictions into a spurious
        # interval that excludes zero by approximately machine precision.
        differences[np.abs(differences) < 1e-12] = 0
        delta = pd.DataFrame(dict(date=a.date.to_numpy(), difference=differences))
        days = delta.groupby('date').difference.agg(['sum','size']).to_numpy()
        sampled = days[np.random.default_rng(20261020).integers(len(days), size=(2000, len(days)))]
        diffs = sampled[:,:,0].sum(axis=1) / sampled[:,:,1].sum(axis=1)
        rows.append(dict(model=family, without_form_log_loss=old.log_loss, form_allowed_log_loss=new.log_loss,
            difference=new.log_loss-old.log_loss, ci_low=np.quantile(diffs,.025), ci_high=np.quantile(diffs,.975),
            without_candidate=old.candidate, form_candidate=new.candidate,
            without_raw_features=int(old.raw_features), form_raw_features=int(new.raw_features),
            without_active_features=int(old.active_features), form_active_features=int(new.active_features)))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'minimum_cv_comparison.csv', index=False)
    receipt = dict(method='Both minimum-mean configurations were frozen in selection.json before '
                         'audit/test. This supplement evaluates the missing without-form comparator; no new tuning.',
                   choices=choices, selection_sha256=hashlib.sha256((OUT/'selection.json').read_bytes()).hexdigest(),
                   code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT / 'minimum_cv_audit.json').write_text(json.dumps(receipt, indent=2))
    path = OUT / 'report.html'
    body = path.read_text(encoding='utf-8').split('<!-- minimum-cv-comparison -->')[0]
    body += '<!-- minimum-cv-comparison --><h2>Matched minimum-CV comparison</h2><p>' + receipt['method'] + '</p>'
    body += '<p>Form allowed minus without form; negative favors form. Intervals use 2,000 date-cluster bootstrap resamples. '
    body += 'They do not account for repeated test inspection or all temporal/team dependence.</p>' + table.to_html(index=False)
    path.write_text(body, encoding='utf-8')
    print(table.to_string(index=False), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
