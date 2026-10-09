"""Reconstruct the saved vector forest's exact feature order and encoder."""
import hashlib
import json
import math
import statistics

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from lineup_sources import team_key
from live_predictions import ROOT


def rating_features(players, prefix):
    if len(players) != 11 or len({p['ea_id'] for p in players}) != 11:
        raise ValueError(f'{prefix}: exactly 11 distinct rated starters required')
    ratings = [float(p['overall']) for p in players]
    if not all(math.isfinite(v) and 1 <= v <= 99 for v in ratings):
        raise ValueError('Player overall ratings must be in [1, 99]')
    return {prefix + '_Overall_max': max(ratings), prefix + '_Overall_min': min(ratings),
            prefix + '_Overall_sd': statistics.stdev(ratings),
            prefix + '_Overall_mean': statistics.mean(ratings),
            prefix + '_Overall_mean_ln': statistics.mean(math.log(v) for v in ratings),
            prefix + '_Overall_mean_sqrt': statistics.mean(math.sqrt(v) for v in ratings),
            prefix + '_3rd_best': sorted(ratings, reverse=True)[2],
            prefix + '_3rd_worst': sorted(ratings)[2]}


def clean(frame):
    frame = frame.copy()
    frame['Matchweek'] = frame['Matchweek'].astype(str).str.split().str[-1].astype(int)
    frame['Home_min_max'] = frame.HomePlayer_Overall_max * frame.HomePlayer_Overall_min
    frame['Away_min_max'] = frame.AwayPlayer_Overall_max * frame.AwayPlayer_Overall_min
    return frame.drop(columns=[c for c in frame if c.startswith('Unnamed:') or 'bench' in c or 'points_to' in c])


class LineupForest:
    def __init__(self):
        model_path = ROOT / 'results/vector_forest.joblib'
        details_path = ROOT / 'results/run_details.json'
        if not model_path.exists() or not details_path.exists():
            raise ValueError('Saved vector forest or its metadata is missing. Run python run.py train first.')
        details = json.loads(details_path.read_text(encoding='utf-8'))
        train_path = ROOT / 'data/train.csv'
        if hashlib.sha256(train_path.read_bytes()).hexdigest() != details['input_sha256']['train.csv']:
            raise ValueError('Training data differs from saved model metadata; retrain before live inference')
        import sklearn
        if details['versions']['sklearn'] != sklearn.__version__:
            raise ValueError('Install the scikit-learn version in requirements.txt or retrain the model')
        train = clean(pd.read_csv(train_path)).replace([np.inf, -np.inf], np.nan).dropna()
        self.categorical = [c for c in ['home_team_name', 'away_team_name', 'league'] if c in train]
        self.numeric = [c for c in train if c not in self.categorical + ['home_score', 'away_score']]
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(train[self.categorical])
        features = self.encoder.get_feature_names_out(self.categorical).tolist() + self.numeric
        if features != details['features']:
            raise ValueError('Reconstructed feature order differs from saved model metadata')
        self.forest = joblib.load(model_path)
        self.forest.verbose = 0
        if self.forest.n_features_in_ != len(features) or self.forest.n_outputs_ != 2:
            raise ValueError('Saved forest has an incompatible input/output schema')
        self.teams = {team_key(n): n for c in ['home_team_name', 'away_team_name'] for n in train[c].unique()}
        self.metadata = {'name': 'Saved 2,000-tree lineup vector random forest',
                         'path': str(model_path), 'sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
                         'train_sha256': details['input_sha256']['train.csv'], 'feature_count': len(features)}

    def predict(self, row):
        frame = clean(pd.DataFrame([row]))
        missing = [c for c in self.numeric + self.categorical if c not in frame]
        if missing:
            raise ValueError('Missing model features: ' + ', '.join(missing))
        if not all(math.isfinite(float(frame.iloc[0][c])) for c in self.numeric):
            raise ValueError('Model inputs contain missing or nonfinite values')
        for field in ('B365H', 'B365D', 'B365A'):
            if float(row[field]) <= 1:
                raise ValueError('Missing/invalid Bet365 odds: ' + field)
        for c in ('home_team_name', 'away_team_name'):
            frame[c] = frame[c].map(lambda n: self.teams.get(team_key(n), n))
        matrix = np.column_stack([self.encoder.transform(frame[self.categorical]), frame[self.numeric].to_numpy(float)])
        means = self.forest.predict(matrix)[0]
        if not np.isfinite(means).all() or (means < 0).any():
            raise ValueError('Forest returned invalid goal predictions')
        return [float(v) for v in means]


def form_features(history, fixture, today):
    """Season-to-date totals and last five completed league matches, before today."""
    start_year = today.year if today.month >= 7 else today.year - 1
    from datetime import date
    season_start = date(start_year, 7, 1)
    current = [m for m in history if m['league'] == fixture['league'] and season_start <= m['date'] < today]
    result, counts = {}, {}
    for side in ('home', 'away'):
        team = team_key(fixture[side])
        games = sorted([m for m in current if team in (team_key(m['home']), team_key(m['away']))], key=lambda m: m['date'])
        gd = [(m['hg']-m['ag']) * (1 if team_key(m['home']) == team else -1) for m in games]
        points = [3 if g > 0 else 1 if g == 0 else 0 for g in gd]
        result.update({side+'_GD_prior': sum(gd), side+'_Points_prior': sum(points),
                       side+'_GD_form': sum(gd[-5:]), side+'_Points_form': sum(points[-5:])})
        counts[side] = len(games)
    return result, counts
