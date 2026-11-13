"""Build league-specific fantasy form from strictly earlier, available team matches.

FPL adapter reads cached player-match CSVs. Other providers use the documented
team-match import contract; no league is silently given another league's points.
"""
import argparse
import hashlib
import io
import json
from select_expanded_features import ROOT, pd, np
from lineup_sources import team_key
from collect_fantasy_history import fetch

OUT = ROOT / 'results/fantasy'
IMPORTS = ROOT / 'data/fantasy/imports'
METRICS = ['points_mean', 'rating_mean', 'bps_per90',
           'influence_per90', 'creativity_per90', 'threat_per90']
KEYS = ['source', 'league', 'season', 'fixture_id', 'team']
REQUIRED = KEYS + ['opponent', 'kickoff', 'available_at', 'source_url']
LEAGUES = {'E0': 'Premier League / FPL', 'SP1': 'La Liga / fantasy provider export',
           'I1': 'Serie A / fantasy provider export', 'D1': 'Bundesliga / fantasy provider export',
           'F1': 'Ligue 1 / fantasy provider export'}


def validate_history(frame):
    missing = set(REQUIRED) - set(frame)
    if missing:
        raise ValueError('Missing history fields: ' + ', '.join(sorted(missing)))
    frame = frame.copy()
    if frame[REQUIRED].isna().any().any():
        raise ValueError('History identity/provenance fields cannot be missing')
    for col in ['kickoff', 'available_at']:
        frame[col] = pd.to_datetime(frame[col], utc=True, errors='raise')
    if (frame.available_at < frame.kickoff + pd.Timedelta(hours=2)).any():
        raise ValueError('Ratings cannot be available before a match has finished')
    frame['season'] = pd.to_numeric(frame.season, errors='raise').astype(int)
    for col in ['team', 'opponent']:
        frame[col] = frame[col].map(team_key)
    if (frame.team == frame.opponent).any() or not set(frame.league) <= set(LEAGUES):
        raise ValueError('Invalid team pairing or league')
    for metric in METRICS:
        frame[metric] = pd.to_numeric(frame.get(metric, np.nan), errors='raise')
    if np.isinf(frame[METRICS].to_numpy()).any():
        raise ValueError('Infinite fantasy value')
    if frame.points_mean.isna().any():
        raise ValueError('Every imported team match needs points_mean')
    frame = frame.drop_duplicates()
    if frame.duplicated(KEYS).any():
        raise ValueError('Conflicting duplicate team-fixture records')
    # A common model must not mix changing publishers within a league.
    if frame.groupby('league').source.nunique().gt(1).any():
        raise ValueError('Use one fantasy provider per league per experiment')
    for _, match in frame.groupby(['source', 'league', 'season', 'fixture_id']):
        if len(match) != 2 or set(match.team) != set(match.opponent) or match.kickoff.nunique() != 1:
            raise ValueError('Each fantasy fixture must have two reciprocal team records')
    return frame.sort_values('kickoff').reset_index(drop=True)


def fpl_history(players, teams, year, url):
    """Use fixture-specific opponents to recover historical clubs after transfers."""
    players = players.copy()
    mapping = dict(zip(teams.id.astype(int), teams.name))
    rows = []
    for fixture, match in players.groupby('fixture'):
        home_flag = match.was_home.astype(str).str.lower().map({'true': True, 'false': False,
                                                               '1': True, '0': False})
        if home_flag.isna().any():
            raise ValueError('Invalid FPL was_home flag')
        if match.kickoff_time.dropna().nunique() != 1:
            raise ValueError('Ambiguous FPL fixture kickoff')
        if match[['team_h_score', 'team_a_score']].isna().any().any():
            continue  # Unplayed fixture, not a rating observation.
        kickoff = pd.to_datetime(match.kickoff_time.iloc[0], utc=True)
        for is_home in [True, False]:
            side, other = match.loc[home_flag == is_home], match.loc[home_flag != is_home]
            opponents = side.opponent_team.dropna().unique()
            own = other.opponent_team.dropna().unique()
            if len(opponents) != 1 or len(own) != 1:
                raise ValueError('Cannot identify FPL fixture clubs')
            own_id, opponent_id = int(own[0]), int(opponents[0])
            played = side[pd.to_numeric(side.minutes, errors='raise') > 0].copy()
            identity = 'element' if 'element' in played else 'name'
            if played[identity].duplicated().any():
                raise ValueError('Duplicate FPL player-fixture record')
            minutes = pd.to_numeric(played.minutes, errors='raise').sum()
            if len(played) < 11 or minutes < 800:
                raise ValueError('Incomplete FPL team appearance coverage')
            row = dict(source='FPL', league='E0', season=year, fixture_id=str(fixture),
                team=mapping[own_id], opponent=mapping[opponent_id], kickoff=kickoff,
                available_at=kickoff + pd.Timedelta(hours=72), source_url=url,
                availability_basis='assumed kickoff + 72h; historical publication not archived',
                points_mean=pd.to_numeric(played.total_points, errors='raise').mean())
            for metric in ['bps', 'influence', 'creativity', 'threat']:
                values = pd.to_numeric(played[metric], errors='coerce')
                row[metric + '_per90'] = values.sum() / minutes * 90 if values.notna().all() else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def attach_history(matches, history, min_history=3):
    """The target date's UTC midnight is the cutoff (local inputs have no times).

    Never uses target-match fantasy rows, lineups, player minutes, or target
    coverage to decide eligibility. Late-published games are excluded explicitly.
    """
    if min_history < 1:
        raise ValueError('min_history must be positive')
    matches = matches.copy()
    groups = {(league, int(season), team): g for (league, season, team), g in
              history.groupby(['league', 'season', 'team'])}
    result = []
    for row in matches.to_dict('records'):
        cutoff = pd.to_datetime(row['fixture_date'], utc=True).normalize()
        row['fantasy_cutoff'] = cutoff.isoformat()
        for side in ['home', 'away']:
            key = (row['league'], int(row['season']), team_key(row[side + '_team_name']))
            past = groups.get(key, history.iloc[:0])
            past = past[(past.kickoff < cutoff) & (past.available_at < cutoff)].sort_values('kickoff')
            row[side + '_fantasy_history_n'] = len(past)
            row[side + '_fantasy_latest_kickoff'] = past.kickoff.max() if len(past) else None
            row[side + '_fantasy_latest_available'] = past.available_at.max() if len(past) else None
            for metric in METRICS:
                for window, part in [('last5', past.tail(5)), ('season', past)]:
                    values = part[metric].dropna()
                    row[f'{side}_fantasy_{metric}_{window}'] = values.mean() if len(values) >= min_history else np.nan
        for metric in METRICS:
            for window in ['last5', 'season']:
                suffix = f'fantasy_{metric}_{window}'
                row[suffix + '_gap'] = row['home_' + suffix] - row['away_' + suffix]
        row['fantasy_eligible'] = min(row[s + '_fantasy_history_n'] for s in ['home', 'away']) >= min_history
        result.append(row)
    return pd.DataFrame(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=int, default=2016)
    parser.add_argument('--end', type=int, default=2022)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    histories, inputs, issues = [], {}, []
    for year in range(args.start, args.end + 1):
        base = ('https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/'
                f'{year}-{str(year+1)[-2:]}/')
        try:
            blobs = [fetch(base + file, offline=True) for file in ['gws/merged_gw.csv', 'teams.csv']]
        except FileNotFoundError:
            issues.append(dict(league='E0', season=year, status='missing_cached_FPL_history'))
            continue
        for file, blob in zip(['gws/merged_gw.csv', 'teams.csv'], blobs):
            inputs[base + file] = hashlib.sha256(blob).hexdigest()
        histories.append(fpl_history(*(pd.read_csv(io.BytesIO(b)) for b in blobs), year, base + 'gws/merged_gw.csv'))
    for path in sorted(IMPORTS.glob('*.csv')):
        inputs[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        histories.append(pd.read_csv(path))
    history = (validate_history(pd.concat(histories, ignore_index=True)) if histories
               else validate_history(pd.DataFrame(columns=REQUIRED + METRICS)))
    matches = []
    for split in ['train', 'test']:
        path = ROOT / f'results/classifier/{split}.csv'
        inputs[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        frame = pd.read_csv(path)
        frame['split'] = split
        frame['source_row'] = np.arange(len(frame))
        matches.append(frame)
    matches = pd.concat(matches, ignore_index=True)
    matches = matches[matches.season.between(args.start, args.end)]
    joined = attach_history(matches, history)
    joined.to_csv(OUT / 'matches.csv', index=False)
    coverage = joined.groupby(['league', 'season', 'split']).agg(
        matches=('fantasy_eligible', 'size'), eligible=('fantasy_eligible', 'sum')).reset_index()
    coverage.to_csv(OUT / 'coverage.csv', index=False)
    audit_cols = ['split', 'source_row', 'league', 'season', 'fixture_date', 'fantasy_cutoff',
                  'fantasy_eligible'] + [s + '_fantasy_' + c for s in ['home', 'away']
                  for c in ['history_n', 'latest_kickoff', 'latest_available']]
    joined[audit_cols].to_csv(OUT / 'temporal_audit.csv', index=False)
    status = [dict(league=league, provider=provider, fantasy_team_matches=int((history.league == league).sum()),
                   base_matches=int((matches.league == league).sum()),
                   status=('ready_for_coverage_checks' if (history.league == league).any()
                           else 'not_collected; no real model comparison'))
              for league, provider in LEAGUES.items()]
    manifest = dict(status=status, input_hashes=inputs, issues=issues,
        prepared_matches_sha256=hashlib.sha256((OUT / 'matches.csv').read_bytes()).hexdigest(),
        method='Team performance ratings from prior matches; no target-match lineup inference. '
               'Three prior observations required; within-season last-five and expanding means. '
               'Strict availability before target UTC date midnight.',
        limitations='FPL archived ratings are final revised values, not timestamped historical snapshots. '
                    '72-hour availability is an assumption, not verified publication timing. '
                    'Existing EA features and odds retain earlier provenance limitations. '
                    'Test season 2022-23 was examined previously.')
    (OUT / 'data_audit.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(coverage.to_string(index=False))
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
