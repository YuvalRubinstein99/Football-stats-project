"""Past-only attacking/defensive form, shared by offline and future inference."""
from collections import defaultdict
from itertools import groupby
import numpy as np
from classifier_features import club

METRICS = ('goals_for', 'goals_against', 'sot_for', 'sot_against')
WINDOWS = ('3', '5', '10', 'ewm5')


def fixture_key(row):
    return (row['league'], row['date'], club(row['home']), club(row['away']))


def columns(window, kind='all'):
    metrics = METRICS if kind in ('all', 'gaps') else (
        METRICS[:2] if kind == 'goals' else METRICS[2:])
    names = ([] if kind == 'gaps' else
             [f'{s}_recent_{m}_{window}' for s in ('home', 'away') for m in metrics])
    names += [f'recent_{m}_{window}_gap' for m in metrics]
    names += [f'{s}_recent_{c}_{window}' for s in ('home', 'away')
              for c in ('matches', 'sot_matches')]
    return names


def build_form(games, queries):
    """Read all queries on a day before adding that day's completed results.

    Seasons are explicit so postponed summer games keep their real season.
    Missing shots are never converted to zero. EWM half-life is five matches.
    """
    state = defaultdict(list)
    keys = [fixture_key(g) for g in games]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate historical fixture')
    output = {}
    entries = [(g['date'], False, g) for g in games] + [(q['date'], True, q) for q in queries]
    for day, group in groupby(sorted(entries, key=lambda x: x[0]), key=lambda x: x[0]):
        batch = list(group)
        for _, is_query, row in batch:
            if not is_query:
                continue
            values = {}
            for side in ('home', 'away'):
                past = state[row['league'], row['season'], club(row[side])]
                values[side + '_recent_latest_date'] = past[-1][0].isoformat() if past else ''
                for window in WINDOWS:
                    part = past if window == 'ewm5' else past[-int(window):]
                    values[f'{side}_recent_matches_{window}'] = len(part)
                    values[f'{side}_recent_sot_matches_{window}'] = sum(
                        np.isfinite(p[1][2:]).all() for p in part)
                    for j, metric in enumerate(METRICS):
                        a = np.array([p[1][j] for p in part], float)
                        weights = (2. ** (-np.arange(len(a)-1, -1, -1) / 5.)
                                   if window == 'ewm5' else np.ones(len(a)))
                        valid = np.isfinite(a)
                        values[f'{side}_recent_{metric}_{window}'] = (
                            float(np.average(a[valid], weights=weights[valid])) if valid.any() else np.nan)
            for window in WINDOWS:
                for metric in METRICS:
                    suffix = f'recent_{metric}_{window}'
                    values[suffix + '_gap'] = values['home_' + suffix] - values['away_' + suffix]
            output[fixture_key(row)] = values
        for _, is_query, row in batch:
            if is_query:
                continue
            for side, gf, ga, sf, sa in [
                ('home', row['hg'], row['ag'], row.get('hst', np.nan), row.get('ast', np.nan)),
                ('away', row['ag'], row['hg'], row.get('ast', np.nan), row.get('hst', np.nan)),
            ]:
                state[row['league'], row['season'], club(row[side])].append((day, (gf, ga, sf, sa)))
    return output
