"""Replay the flat filtered-variance strategy on upcoming lineup feature audits."""
from datetime import date, datetime, timezone
from functools import lru_cache

import joblib
from xgboost import XGBRegressor

from compare_expanded_models import outcome_probabilities
from lineup_model import LineupForest
from lineup_sources import team_key
from live_bets import qualifies_filtered_variance, recommendations
from live_predictions import ROOT

MODEL_DIR = ROOT / 'results/xgboost_goal_vector'
LEAGUES = {'E0', 'SP1', 'D1', 'I1'}


class LineupXGBoost(LineupForest):
    def __init__(self):
        self.encoder, self.numeric = joblib.load(MODEL_DIR / 'preprocessing.joblib')
        self.categorical = list(self.encoder.feature_names_in_)
        self.teams = {team_key(name): name for group in self.encoder.categories_ for name in group}
        self.estimator = XGBRegressor(n_jobs=1)
        self.estimator.load_model(MODEL_DIR / 'goal_vector.ubj')

    def predict(self, row):
        goals = self.estimator.predict(self.feature_matrix(row)).astype(float)
        return goals[0].tolist(), outcome_probabilities(goals)[0].tolist()


@lru_cache(maxsize=1)
def fitted():
    return LineupXGBoost()


def board(report, now=None):
    now = now or datetime.now(timezone.utc)
    audits = {r['fixture']: r['features'] for r in report.get('feature_audit', [])}
    fixtures = []
    for m in [*report.get('predictions', []), *report.get('skipped', []), *report.get('lineups_awaiting_prices', [])]:
        if m['league'] not in LEAGUES:
            continue
        day = date.fromisoformat(m['date'])
        kickoff = m.get('kickoff_utc')
        if (datetime.fromisoformat(kickoff) <= now if kickoff else day < now.date()):
            continue
        card = {k: m.get(k) for k in ('league', 'date', 'kickoff_utc', 'time', 'home', 'away',
                                       'lineup_status', 'home_lineup', 'away_lineup', 'ratings_edition')}
        fixtures.append(card)
        card.update(decision='Waiting', reason=m.get('reason') or '', stake=0)
        if not kickoff and day == now.date():
            card['reason'] = 'Refresh to verify today’s kickoff time.'
        if card['reason']:
            continue
        key = '|'.join(m[k] for k in ('league', 'date', 'home', 'away'))
        if key not in audits or any(len(m.get(side+'_lineup') or []) != 11 for side in ('home','away')):
            card['reason'] = 'Both starting XIs and their model inputs are required.'
            continue
        try:
            goals, probabilities = fitted().predict(audits[key])
            match = dict(m, **dict(zip(('p_home', 'p_draw', 'p_away'), probabilities)))
            pick = next(b for b in recommendations(match) if b['strategy'] == 'Legacy variance score')
            if pick['action'] != 'Bet':
                raise ValueError(pick['note'])
            qualifies = qualifies_filtered_variance(pick)
            price, edge = pick['decimal_odds'], pick['expected_net_per_unit']
            reasons = []
            if not 4 <= price <= 1000:
                reasons.append('Selected odds fall outside 4–1000.')
            if edge <= .02:
                reasons.append('Expected return must exceed 2%.')
            card.update(goals=goals, probabilities=probabilities, selection=pick['selection'],
                        selected_probability=pick['probability'],
                        odds=price, edge=edge, decision='Qualifies' if qualifies else 'Skip',
                        stake=int(qualifies), reason=' '.join(reasons))
        except (KeyError, ValueError, TypeError) as exc:
            card['reason'] = str(exc)
    fixtures.sort(key=lambda m: (m['date'], m['kickoff_utc'] or m['date']+'T'+(m['time'] or '23:59'), m['home']))
    return dict(generated_at=report['generated_at'], fixtures=fixtures,
                warnings=[w for w in report.get('warnings', []) if 'lineups unavailable' in w or 'older than' in w])
