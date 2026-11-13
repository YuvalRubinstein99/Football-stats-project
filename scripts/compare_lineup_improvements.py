"""Chronological ablations for identity, lineup change, learned weights and player updates."""
import argparse
import hashlib
import html
import json
from pathlib import Path
from select_expanded_features import ROOT, np, pd, joblib, threadpool_limits
from sklearn.linear_model import PoissonRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.exceptions import ConvergenceWarning
import warnings
from compare_attribute_regressions import design
from compare_positional_matchups import fit_model as original_fit, predict, check_dates, chronological_split
from compare_team_attributes import labels
from calibrate_models import scores
from positional_features import feature_columns
from lineup_history import (LEVEL_COLUMNS, COVERAGE_COLUMNS, CHANGE_COLUMNS, HISTORY_COLUMNS,
                            PRIOR_COLUMNS, EXPERIENCE_COLUMNS, DYNAMIC_COLUMNS)

OUT = ROOT / 'results/lineup_improvements'
PAIRS = [('improved_positions','old_positions'), ('learned_weights','compact_gaps'),
         ('lineup_changes','learned_weights'), ('full_dynamic','fifa_prior'), ('full_dynamic','baseline')]


def add_controls(frame, split):
    old = pd.read_csv(ROOT / f'results/positional_matchups/{split}.csv')
    keys = ['source_row','fixture_date','league','home_team_name','away_team_name']
    if not frame[keys].equals(old[keys]):
        raise ValueError('Old and new comparison samples differ')
    extra = {f'legacy_{c}': old[c].to_numpy() for c in feature_columns()}
    extra.update(compact_attack_defense=frame.home_attack_shooting_mean-frame.away_defense_defending_mean,
        compact_away_attack_defense=frame.away_attack_shooting_mean-frame.home_defense_defending_mean,
        compact_home_dribble_defense=frame.home_attack_dribbling_mean-frame.away_defense_defending_mean,
        compact_away_dribble_defense=frame.away_attack_dribbling_mean-frame.home_defense_defending_mean,
        compact_home_pace=frame.home_attack_pace_mean-frame.away_defense_pace_mean,
        compact_away_pace=frame.away_attack_pace_mean-frame.home_defense_pace_mean,
        compact_midfield_passing=frame.home_midfield_passing_mean-frame.away_midfield_passing_mean,
        compact_midfield_defending=frame.home_midfield_defending_mean-frame.away_midfield_defending_mean,
        compact_goalkeeper=frame.home_goalkeeper_overall_mean-frame.away_goalkeeper_overall_mean)
    return pd.concat([frame, pd.DataFrame(extra, index=frame.index)], axis=1)


def candidates(base):
    gaps = ['compact_attack_defense','compact_away_attack_defense', 'compact_home_dribble_defense',
            'compact_away_dribble_defense','compact_home_pace','compact_away_pace',
            'compact_midfield_passing','compact_midfield_defending','compact_goalkeeper']
    learned = base + LEVEL_COLUMNS + COVERAGE_COLUMNS
    changes = learned + CHANGE_COLUMNS + HISTORY_COLUMNS
    prior = changes + PRIOR_COLUMNS + EXPERIENCE_COLUMNS
    return dict(baseline=base, old_positions=base+[f'legacy_{c}' for c in feature_columns()],
                improved_positions=base+feature_columns(), compact_gaps=base+gaps+COVERAGE_COLUMNS,
                learned_weights=learned, lineup_changes=changes,
                fifa_prior=prior, full_dynamic=prior+DYNAMIC_COLUMNS)


def fit(frame, columns, family, trees, alpha):
    if family != 'Poisson':
        return original_fit(frame, columns, family, trees)
    prep, x, _ = design(frame, columns)
    model = MultiOutputRegressor(PoissonRegressor(alpha=alpha, max_iter=4000, tol=1e-7))
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(x, frame[['home_score','away_score']].to_numpy())
    fitted = dict(preprocessing=prep, model=dict(family='Poisson', model=model))
    return dict(family=family, fitted=fitted, columns=columns, alpha=alpha)


def metric(p, goals, frame):
    y = labels(frame)
    return dict(**scores(p,y), accuracy=float(np.mean(p.argmax(axis=1)==y)),
        goal_mae=float(np.mean(np.abs(goals-frame[['home_score','away_score']].to_numpy()))))


def paired_differences(predictions, frame, family, split):
    y = labels(frame)
    groups = pd.to_datetime(frame.fixture_date).dt.to_period('W').astype(str).to_numpy()
    weeks = np.unique(groups)
    rng = np.random.default_rng(20261110)
    indices = rng.integers(0,len(weeks),size=(2000,len(weeks)))
    results = []
    for candidate, comparator in PAIRS:
        a, b = predictions[candidate], predictions[comparator]
        delta = -np.log(np.maximum(a[np.arange(len(y)),y],1e-15)) + np.log(np.maximum(b[np.arange(len(y)),y],1e-15))
        sums = np.array([delta[groups==week].sum() for week in weeks])
        counts = np.array([(groups==week).sum() for week in weeks])
        boot = sums[indices].sum(axis=1)/counts[indices].sum(axis=1)
        results.append(dict(family=family, split=split, candidate=candidate, comparator=comparator,
            difference=float(delta.mean()), lower_95=float(np.quantile(boot,.025)), upper_95=float(np.quantile(boot,.975))))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trees', type=int, default=500)
    args = parser.parse_args()
    if args.trees < 100:
        parser.error('At least 100 trees required')
    train = add_controls(pd.read_csv(OUT / 'train_features.csv'), 'train')
    baseline = joblib.load(ROOT / 'results/expanded_comparison/expanded__RF_vector.joblib')
    sets = candidates(baseline['categorical']+baseline['numeric'])
    families = ['Poisson','RF_goals']
    paths = [OUT / 'train_features.csv', OUT / 'test_features.csv', OUT / 'identity_manifest.json',
             OUT / 'dynamic_manifest.json', Path(__file__), ROOT / 'src/lineup_history.py',
             ROOT / 'src/player_identity.py', ROOT / 'scripts/compare_attribute_regressions.py',
             ROOT / 'scripts/compare_positional_matchups.py', ROOT / 'scripts/compare_team_attributes.py',
             ROOT / 'scripts/compare_expanded_models.py', ROOT / 'scripts/select_expanded_features.py']
    plan = dict(candidates=sets, families=families, trees=args.trees, selection_years=[2019,2020],
        poisson_alphas=[.01,.1,1.], audit_year=2021, test_year=2022,
        selection='Select Poisson penalty per feature set, then feature set per family, by minimum '
            'mean seasonal log loss in 2019 and 2020. RF settings fixed. Freeze choices before evaluation.',
        evaluation='Sequential pre-match features. Monthly player updates can use earlier evaluation '
            'outcomes; downstream prediction weights remain fixed throughout each evaluation season. '
            'All candidates retain the same original baseline inputs and use identical matches.',
        caveat='Previously inspected audit/test seasons; unverified FIFA snapshot publication dates. '
            'Starter-based player MAP approximation with no substitution exposure or in-match segments. '
            'Partial lineups and preferred positions. Bootstrap intervals omit experiment-selection uncertainty.',
        input_hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (OUT / 'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    cache = OUT / 'cache'
    cache.mkdir(exist_ok=True)
    def evaluate(tr, va, family, candidate, alpha, split, save=False):
        check_dates(tr,va)
        key = hashlib.sha256(json.dumps(dict(plan=plan,family=family,candidate=candidate,
            alpha=alpha,split=split),sort_keys=True).encode()).hexdigest()
        path = cache / (key+'.npz')
        if path.exists() and not save:
            with np.load(path) as data:
                return data['p'],data['goals'],None
        print(f'{split}: {family} / {candidate} / alpha={alpha}',flush=True)
        bundle = fit(tr,sets[candidate],family,args.trees,alpha)
        p,goals = predict(bundle,va)
        np.savez_compressed(path,p=p,goals=goals)
        if save:
            bundle.update(experiment_plan=plan,selection=selection[family])
            model_path = OUT / f'{family}__{candidate}.joblib'
            joblib.dump(bundle,model_path)
            rp,rg = predict(joblib.load(model_path),va.head(10))
            if not np.allclose(rp,p[:10]) or not np.allclose(rg,goals[:10]):
                raise ValueError('Saved model parity failure')
        return p,goals,bundle
    records, penalties, selection = [], {}, {}
    for family in families:
        penalties[family] = {}
        for candidate in sets:
            for alpha in (plan['poisson_alphas'] if family=='Poisson' else [0.]):
                for year in plan['selection_years']:
                    tr,va = chronological_split(train,year)
                    p,g,_ = evaluate(tr,va,family,candidate,alpha,str(year))
                    records.append(dict(family=family,candidate=candidate,alpha=alpha,season=year,
                                        matches=len(va),**metric(p,g,va)))
            scores_df = pd.DataFrame(records)
            means = scores_df[(scores_df.family==family)&(scores_df.candidate==candidate)].groupby('alpha').log_loss.mean()
            best = float(means.idxmin())
            penalties[family][candidate] = dict(alpha=best,mean_log_loss=float(means.loc[best]))
            pd.DataFrame(records).to_csv(OUT / 'selection_scores.csv',index=False)
        chosen = min(penalties[family],key=lambda c:penalties[family][c]['mean_log_loss'])
        selection[family] = dict(candidate=chosen,**penalties[family][chosen])
        print('Selected:',family,selection[family],flush=True)
    (OUT / 'selection.json').write_text(json.dumps(dict(selection=selection,penalties=penalties),indent=2),encoding='utf-8')
    test = add_controls(pd.read_csv(OUT / 'test_features.csv'),'test')
    metrics, predictions, paired, coefficients = [], [], [], []
    for split,tr,va in [('Audit 2021-22',*chronological_split(train,2021)),('Test 2022-23',train,test)]:
        for family in families:
            all_p = {}
            for candidate in sets:
                alpha = penalties[family][candidate]['alpha']
                p,g,bundle = evaluate(tr,va,family,candidate,alpha,split,save=split.startswith('Test'))
                all_p[candidate] = p
                metrics.append(dict(split=split,family=family,candidate=candidate,alpha=alpha,
                    selected=candidate==selection[family]['candidate'],matches=len(va),**metric(p,g,va)))
                for i,r in enumerate(va.itertuples()):
                    predictions.append(dict(split=split,family=family,candidate=candidate,source_row=r.source_row,
                        date=r.fixture_date,league=r.league,home_team=r.home_team_name,away_team=r.away_team_name,
                        home_score=r.home_score,away_score=r.away_score,p_home=p[i,0],p_draw=p[i,1],p_away=p[i,2],
                        expected_home_goals=g[i,0],expected_away_goals=g[i,1]))
                if split.startswith('Test') and family=='Poisson':
                    prep = bundle['fitted']['preprocessing']
                    names = list(prep['encoder'].get_feature_names_out(prep['categorical'])) + list(prep['imputer'].get_feature_names_out(prep['numeric']))
                    for side,estimator in zip(['home','away'],bundle['fitted']['model']['model'].estimators_):
                        coefficients.extend(dict(candidate=candidate,target=side,feature=name,coefficient=float(value))
                                            for name,value in zip(names,estimator.coef_))
            paired.extend(paired_differences(all_p,va,family,split))
        market = 1/va[['B365H','B365D','B365A']].to_numpy(float)
        market /= market.sum(axis=1,keepdims=True)
        metrics.append(dict(split=split,family='Market',candidate='normalized_odds',alpha=np.nan,selected=False,
            matches=len(va),**scores(market,labels(va)),accuracy=float(np.mean(market.argmax(axis=1)==labels(va)))))
        pd.DataFrame(metrics).to_csv(OUT / 'metrics.csv',index=False)
    for name,values in [('predictions',predictions),('paired_differences',paired),('coefficients',coefficients)]:
        pd.DataFrame(values).to_csv(OUT / f'{name}.csv',index=False)
    (OUT / 'summary.json').write_text(json.dumps(dict(selection=selection,metrics=metrics,paired=paired),indent=2),encoding='utf-8')
    body = '<h1>Four lineup improvements</h1>'
    for key in ['selection','evaluation','caveat']:
        body += '<p>'+html.escape(plan[key])+'</p>'
    body += ('<p>baseline: original inputs; old_positions: prior matching; improved_positions: rematched '
        'identities; compact_gaps: fixed differences; learned_weights: separate unit strengths; '
        'lineup_changes: prior-five lineup deviations; fifa_prior: matched FIFA priors and experience; '
        'full_dynamic: additionally learned attacking/defensive player corrections.</p>')
    body += '<h2>Prediction quality</h2><p>Lower log loss/MAE is better; higher accuracy is better.</p>'
    body += pd.DataFrame(metrics).to_html(index=False,float_format=lambda x:f'{x:.5f}')
    body += '<h2>Paired incremental differences</h2><p>Negative favors the new feature set. '
    body += 'Approximate 95% intervals from 2,000 paired calendar-week bootstrap samples.</p>'
    body += pd.DataFrame(paired).to_html(index=False,float_format=lambda x:f'{x:.5f}')
    body += '<h2>Identity coverage</h2>'+pd.read_csv(OUT / 'coverage.csv').to_html(index=False)
    body += '<h2>Validation selection</h2>'+pd.DataFrame(records).to_html(index=False)
    (OUT / 'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Lineup improvements</title>'
        '<style>body{font:15px system-ui;margin:30px}table{border-collapse:collapse;font-size:12px}'
        'td,th{padding:7px;border-bottom:1px solid #ddd}</style>'+body,encoding='utf-8')
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):
        main()
