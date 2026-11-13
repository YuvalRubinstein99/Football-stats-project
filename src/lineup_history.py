"""Strictly past lineup changes and starter-based, FIFA-prior player effects."""
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.optimize import minimize
from classifier_features import club

STRENGTHS = ['attack_shooting', 'attack_dribbling', 'attack_pace',
             'midfield_passing', 'midfield_defending', 'defense_defending',
             'defense_pace', 'goalkeeper_overall']
LEVEL_COLUMNS = [f'{s}_{v}_mean' for s in ['home', 'away'] for v in STRENGTHS]
COVERAGE_COLUMNS = [f'{s}_{u}_count' for s in ['home', 'away']
                    for u in ['attack', 'midfield', 'defense', 'goalkeeper']]
CHANGE_COLUMNS = [f'{s}_{v}_change' for s in ['home', 'away'] for v in STRENGTHS]
HISTORY_COLUMNS = [f'{s}_{v}_history_count' for s in ['home', 'away'] for v in STRENGTHS]
PRIOR_COLUMNS = [f'{s}_fifa_prior' for s in ['home', 'away']]
EXPERIENCE_COLUMNS = [f'{s}_player_history_mean' for s in ['home', 'away']]
DYNAMIC_COLUMNS = [f'{s}_learned_{kind}' for s in ['home', 'away'] for kind in ['attack', 'defense']]


def lineup_changes(frame, window=5, max_days=365, min_history=3):
    """Same-date rows all read state before any row on that date is added."""
    result = pd.DataFrame(index=frame.index, columns=CHANGE_COLUMNS + HISTORY_COLUMNS, dtype=float)
    state = defaultdict(lambda: deque(maxlen=window))
    for date, batch in frame.sort_values(['fixture_date', 'league', 'source_row']).groupby('fixture_date', sort=True):
        day = pd.Timestamp(date)
        for idx, row in batch.iterrows():
            for side in ['home', 'away']:
                history = state[row.league, club(row[f'{side}_team_name'])]
                recent = [values for prior_day, values in history if (day-prior_day).days <= max_days]
                for j, value in enumerate(STRENGTHS):
                    valid = [values[j] for values in recent if np.isfinite(values[j])]
                    result.loc[idx, f'{side}_{value}_history_count'] = len(valid)
                    result.loc[idx, f'{side}_{value}_change'] = (
                        row[f'{side}_{value}_mean'] - np.mean(valid) if len(valid) >= min_history else np.nan)
        for _, row in batch.iterrows():
            for side in ['home', 'away']:
                state[row.league, club(row[f'{side}_team_name'])].append(
                    (day, [row[f'{side}_{v}_mean'] for v in STRENGTHS]))
    # Return a plain serializable state for subsequent pre-match queries.
    return result, dict(state)


def lineup_key(row):
    return f'{row["data_split"]}:{int(row["source_row"])}'


def eligible(players):
    return len(players['home']) >= 8 and len(players['away']) >= 8


def prior(players):
    """Fixed prior: a 10-point FIFA advantage gives a 0.2 log-rate difference."""
    return .02 * (np.mean([float(p['overall']) for p in players]) - 70) if len(players) >= 8 else np.nan


def poisson_objective(theta, x, offset, goals, weights, penalty):
    eta = offset + x @ theta
    # Reject non-finite trials rather than silently changing the Poisson objective.
    if not np.isfinite(eta).all() or np.max(eta) > 50:
        return np.inf, np.zeros_like(theta)
    mu = np.exp(eta)
    loss = np.sum(weights * (mu - goals * eta)) + .5 * np.dot(penalty, theta ** 2)
    gradient = np.asarray(x.T @ (weights * (mu-goals))).ravel() + penalty * theta
    return float(loss), gradient


def fit_player_effects(history, lineups, cutoff, alpha=1., half_life=365., warm=None):
    """Penalized Poisson MAP offsets for attack/defense, refitted before each month.

    Observations are whole matches with matched starters, not on-pitch segments.
    FIFA sets the prior log-strength; all learned corrections are shrunk to zero.
    """
    cutoff = pd.Timestamp(cutoff)
    if len(history) and pd.to_datetime(history.fixture_date).max() >= cutoff:
        raise ValueError('Player history reaches the prediction cutoff')
    rows = [(r, lineups[lineup_key(r)]) for r in history.to_dict('records')]
    rows = [(r, p) for r, p in rows if eligible(p)]
    ids = sorted({p['player_id'] for _, sides in rows for players in sides.values() for p in players})
    lookup = {pid: i for i, pid in enumerate(ids)}
    leagues = sorted({r['league'] for r, _ in rows})
    league_ids = {league: i for i, league in enumerate(leagues)}
    counts = defaultdict(int)
    for _, sides in rows:
        for players in sides.values():
            for p in players:
                counts[p['player_id']] += 1
    state = dict(ids=ids, lookup=lookup, leagues=leagues, counts=dict(counts), cutoff=cutoff.isoformat(),
                 last_result_date=str(history.fixture_date.max()) if len(history) else None,
                 alpha=alpha, half_life=half_life, training_matches=len(rows), iterations=0,
                 max_gradient=0., success=True, theta=np.zeros(2*len(ids)+2*len(leagues)))
    if len(rows) < 100:
        return state
    row_idx, col_idx, values, goals, offsets, weights = [], [], [], [], [], []
    n = len(ids)
    for k, (r, sides) in enumerate(rows):
        weight = 2. ** (-(cutoff-pd.Timestamp(r['fixture_date'])).days / half_life)
        for side_number, (side, other) in enumerate([('home','away'), ('away','home')]):
            obs = 2*k + side_number
            for p in sides[side]:
                row_idx.append(obs); col_idx.append(lookup[p['player_id']]); values.append(1/len(sides[side]))
            for p in sides[other]:
                row_idx.append(obs); col_idx.append(n+lookup[p['player_id']]); values.append(-1/len(sides[other]))
            row_idx.append(obs); col_idx.append(2*n+2*league_ids[r['league']]+side_number); values.append(1.)
            goals.append(r[f'{side}_score'])
            offsets.append(prior(sides[side])-prior(sides[other]))
            weights.append(weight)
    x = coo_matrix((values, (row_idx, col_idx)), shape=(2*len(rows), len(state['theta']))).tocsr()
    penalty = np.r_[np.full(2*n, alpha), np.full(2*len(leagues), .001)]
    initial = state['theta'].copy()
    initial[2*n:] = np.tile(np.log([1.5, 1.2]), len(leagues))
    if warm:
        for pid, i in lookup.items():
            old = warm['lookup'].get(pid)
            if old is not None:
                initial[i] = warm['theta'][old]
                initial[n+i] = warm['theta'][len(warm['ids'])+old]
        for league, j in league_ids.items():
            if league in warm['leagues']:
                old = warm['leagues'].index(league)
                initial[2*n+2*j:2*n+2*j+2] = warm['theta'][2*len(warm['ids'])+2*old:2*len(warm['ids'])+2*old+2]
    result = minimize(poisson_objective, initial, args=(x, np.asarray(offsets), np.asarray(goals),
                      np.asarray(weights), penalty), jac=True, method='L-BFGS-B',
                      options=dict(maxiter=1000, ftol=1e-11, gtol=1e-5, maxls=40))
    if not result.success or np.max(np.abs(result.jac)) > .02:
        raise ValueError(f'Player MAP fit failed: {result.message}, gradient={np.max(np.abs(result.jac))}')
    state.update(theta=result.x, iterations=int(result.nit), max_gradient=float(np.max(np.abs(result.jac))))
    return state


def player_features(state, players):
    features = {}
    n = len(state['ids'])
    for side in ['home', 'away']:
        members = players[side]
        features[f'{side}_fifa_prior'] = prior(members)
        features[f'{side}_player_history_mean'] = (float(np.mean([state['counts'].get(p['player_id'], 0)
            for p in members])) if len(members) >= 8 else np.nan)
        for kind, start in [('attack', 0), ('defense', n)]:
            corrections = [state['theta'][start+state['lookup'][p['player_id']]]
                           if p['player_id'] in state['lookup'] else 0. for p in members]
            features[f'{side}_learned_{kind}'] = float(np.mean(corrections)) if len(members) >= 8 else np.nan
    return features
