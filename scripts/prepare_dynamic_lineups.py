"""Build past-only lineup deviations and monthly FIFA-prior player estimates."""
import json
import hashlib
from select_expanded_features import ROOT, pd, joblib, threadpool_limits
from lineup_history import lineup_changes, fit_player_effects, player_features, lineup_key

OUT = ROOT / 'results/lineup_improvements'


def main():
    train, test = [pd.read_csv(OUT / f'{s}.csv').assign(data_split=s) for s in ['train', 'test']]
    frame = pd.concat([train, test], ignore_index=True)
    lineups = json.loads((OUT / 'lineups.json').read_text())
    changes, history = lineup_changes(frame)
    features, diagnostics = [], []
    warm = None
    for month, batch in frame.groupby(pd.to_datetime(frame.fixture_date).dt.to_period('M'), sort=True):
        cutoff = month.start_time
        past = frame[pd.to_datetime(frame.fixture_date) < cutoff]
        state = fit_player_effects(past, lineups, cutoff, warm=warm)
        warm = state
        for idx, r in batch.iterrows():
            features.append(dict(row=idx, **player_features(state, lineups[lineup_key(r)])))
        diagnostics.append({k: state[k] for k in ['cutoff', 'last_result_date', 'training_matches',
                                                'alpha', 'half_life', 'iterations', 'max_gradient']})
        print(f'Player update {month}: {state["training_matches"]} prior matches, '
              f'{len(state["ids"])} players', flush=True)
    dynamic = pd.DataFrame(features).set_index('row').reindex(frame.index)
    frame = pd.concat([frame, changes, dynamic], axis=1)
    for split in ['train', 'test']:
        frame[frame.data_split == split].to_csv(OUT / f'{split}_features.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(OUT / 'dynamic_fit_audit.csv', index=False)
    # The saved state is the actual last prediction month's state, with no later labels.
    joblib.dump(dict(player_state=warm, lineup_history=history), OUT / 'history_state.joblib')
    paths = [OUT / 'train.csv', OUT / 'test.csv', OUT / 'lineups.json',
             ROOT / 'src/lineup_history.py', ROOT / 'scripts/prepare_dynamic_lineups.py']
    (OUT / 'dynamic_manifest.json').write_text(json.dumps(dict(
        input_hashes={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        lineup_change_rule='Current unit strengths minus means from at most five prior club lineups '
            'within 365 days, at least three valid observations. Same-day rows never update one another.',
        dynamic_rule='Monthly penalized Poisson player attack/defense corrections to FIFA priors, '
            'using only earlier calendar months. Starter averages, >=8 matched per team. FIFA prior '
            '0.02*(overall-70), prior corrections centered at zero with L2 alpha=1.0. Results weighted '
            'with 365-day half-life. League-specific home/away intercepts. New players have zero correction.',
        evaluation='Sequential pre-match evaluation: earlier audit/test outcomes may update later-month '
            'player features. Outcome prediction model weights are fixed for each evaluation season.',
        limitation='Starter-based approximation; no substitution minutes, red-card timing or on-pitch '
            'segments. Not a reproduction of published adjusted plus-minus. FIFA snapshot dates unverified.'
    ), indent=2), encoding='utf-8')


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
