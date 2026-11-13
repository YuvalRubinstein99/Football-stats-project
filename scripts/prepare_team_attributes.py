"""Recover season-specific FIFA subratings from existing named starting lineups."""
import json
import hashlib
import re
from collections import defaultdict, Counter
from functools import lru_cache
from select_expanded_features import ROOT, pd, np
from classifier_features import club
from lineup_sources import normalize

OUT = ROOT / 'results/team_attributes'
ATTRIBUTES = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physicality']
RENAMES = {'Known As': 'short_name', 'Full Name': 'long_name', 'Overall': 'overall',
           'Club Name': 'club_name', 'Positions Played': 'player_positions',
           'Pace Total': 'pace', 'Shooting Total': 'shooting', 'Passing Total': 'passing',
           'Dribbling Total': 'dribbling', 'Defending Total': 'defending',
           'Physicality Total': 'physicality', 'physic': 'physicality'}


def name_key(value):
    return normalize(str(value)) if pd.notna(value) else ''


class PlayerIndex:
    def __init__(self, frame):
        self.frame = frame.rename(columns=RENAMES).reset_index(drop=True)
        self.records = self.frame.to_dict('records')
        self.names, self.clubs = defaultdict(set), defaultdict(list)
        self.tokens = {}
        for i, row in enumerate(self.records):
            for field in ('long_name', 'short_name'):
                key = name_key(row[field])
                if key:
                    self.names[key].add(i)
            self.clubs[club(str(row['club_name']))].append(i)
            self.tokens[i] = [set(name_key(row[field]).split()) for field in ('long_name', 'short_name')]

    @lru_cache(maxsize=None)
    def match(self, name, team):
        key = name_key(name)
        if not key or '\ufffd' in str(name):
            return None, 'missing_or_damaged_name'
        exact = self.names.get(key, set())
        local = exact.intersection(self.clubs.get(club(team), []))
        if len(local) == 1:
            return next(iter(local)), 'exact_name_and_club'
        if len(exact) == 1:
            return next(iter(exact)), 'unique_exact_name'
        if exact:
            return None, 'ambiguous_exact_name'
        # No edit-distance matching. Require all supplied name tokens and one
        # unique player within the club. Single-token nicknames require exact names.
        tokens = set(key.split())
        candidates = [i for i in self.clubs.get(club(team), [])
                      if len(tokens) >= 2 and any(tokens <= t for t in self.tokens[i])]
        if len(candidates) == 1:
            return candidates[0], 'unique_name_tokens_and_club'
        return None, 'ambiguous_or_unmatched_name'


def aggregate_side(players, prefix, min_players=8):
    """Missing subratings, especially goalkeeper values, are never replaced by zero."""
    output = {prefix + '_matched_starters': len(players)}
    outfield = [p for p in players if 'GK' not in [s.strip() for s in str(p['player_positions']).split(',')]]
    output[prefix + '_matched_outfield'] = len(outfield)
    for att in ATTRIBUTES:
        values = [float(p[att]) for p in outfield
                  if pd.notna(p[att]) and 1 <= float(p[att]) <= 99]
        output[f'{prefix}_{att}_count'] = len(values)
        output[f'{prefix}_{att}_mean'] = float(np.mean(values)) if len(values) >= min_players else np.nan
    return output


def comparisons(features):
    for att in ATTRIBUTES:
        features[att + '_advantage'] = features['home_' + att + '_mean'] - features['away_' + att + '_mean']
    for side, opponent in [('home', 'away'), ('away', 'home')]:
        for attack in ['shooting', 'dribbling', 'pace']:
            features[f'{side}_{attack}_vs_defending'] = (
                features[f'{side}_{attack}_mean'] - features[f'{opponent}_defending_mean'])
    return features


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hashes, features, players_audit, exclusions = {}, {}, [], []
    for year in range(2014, 2023):
        paths = [ROOT / f'data/players_{(year+1)%100:02d}.csv',
                 ROOT / f'data/season_data/season{year%100:02d}{(year+1)%100:02d}.csv']
        for path in paths:
            hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        index = PlayerIndex(pd.read_csv(paths[0], low_memory=False))
        matches = pd.read_csv(paths[1], low_memory=False)
        print(f'Join attributes for {year}: {len(matches)} named lineups', flush=True)
        ambiguous = set()
        for source_row, row in enumerate(matches.to_dict('records')):
            key = year, club(row['home_team_name']), club(row['away_team_name'])
            if key in ambiguous:
                continue
            if key in features:
                del features[key]
                ambiguous.add(key)
                exclusions.append(dict(split='source', source_row=source_row, season=year,
                                       league='', reason='Ambiguous duplicate named fixture'))
                continue
            feat = {}
            all_ids = []
            for side, prefix in [('home', 'HomePlayer'), ('away', 'AwayPlayer')]:
                resolved, accepted = [], []
                for slot in range(1, 12):
                    name = row.get(f'{prefix}{slot}')
                    pid, method = index.match(str(name), str(row[side + '_team_name']))
                    player = index.records[pid] if pid is not None else None
                    # Compare against the exact Overall used by the original pipeline;
                    # mismatches are excluded, not used to guess identity.
                    original = row.get(f'{prefix}{slot}_Overall', np.nan)
                    compatible = player is not None and pd.notna(original) and abs(float(player['overall']) - float(original)) < 1e-8
                    players_audit.append(dict(season=year, source_row=source_row, side=side,
                        slot=slot, source_name=name, rating_row=pid,
                        matched_name=player['long_name'] if player is not None else '',
                        method=method, original_overall=original,
                        rating_overall=player['overall'] if player is not None else np.nan,
                        accepted=compatible))
                    if compatible:
                        resolved.append(pid)
                        accepted.append(player)
                if len(resolved) != len(set(resolved)):
                    accepted = []
                all_ids.extend(resolved)
                feat.update(aggregate_side(accepted, side))
            if len(all_ids) != len(set(all_ids)):
                # A player cannot appear twice or on both teams.
                feat = {**aggregate_side([], 'home'), **aggregate_side([], 'away')}
            features[key] = comparisons(feat)
    coverage = []
    for split in ('train', 'test'):
        path = ROOT / f'results/classifier/{split}.csv'
        hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        frame = pd.read_csv(path)
        rows = []
        for i, row in frame.iterrows():
            key = int(row.season), club(row.home_team_name), club(row.away_team_name)
            # Restrict to the named-lineup domain; all variants get these same rows.
            if key not in features or row.league not in ['SP1', 'D1', 'I1']:
                exclusions.append(dict(split=split, source_row=i, season=int(row.season),
                                       league=row.league, reason='No named source lineup'))
                continue
            rows.append({**row.to_dict(), **features[key], 'source_row': i})
        joined = pd.DataFrame(rows)
        # Known duplicate test fixtures must not count twice.
        dup = joined.duplicated(['fixture_date', 'league', 'home_team_name', 'away_team_name'], keep=False)
        for _, row in joined[dup].iterrows():
            exclusions.append(dict(split=split, source_row=int(row.source_row), season=int(row.season),
                                   league=row.league, reason='Ambiguous duplicate fixture'))
        joined = joined[~dup].copy()
        joined.to_csv(OUT / f'{split}.csv', index=False)
        means = [f'{side}_{att}_mean' for side in ('home', 'away') for att in ATTRIBUTES]
        for (season, league), group in joined.groupby(['season', 'league']):
            coverage.append(dict(split=split, season=int(season), league=league, matches=len(group),
                all_attributes_available=int(group[means].notna().all(axis=1).sum()),
                average_matched_starters=float(group[['home_matched_starters','away_matched_starters']].to_numpy().mean())))
    pd.DataFrame(players_audit).to_csv(OUT / 'player_matching_audit.csv', index=False)
    pd.DataFrame(exclusions).to_csv(OUT / 'excluded_rows.csv', index=False)
    pd.DataFrame(coverage).to_csv(OUT / 'coverage.csv', index=False)
    (OUT / 'feature_audit.json').write_text(json.dumps(dict(
        input_hashes=hashes, attributes=ATTRIBUTES, coverage=coverage,
        feature_rule='Outfield starter means with >=8 matched valid players per attribute; otherwise missing. '
                     'Home-away gaps and shooting/dribbling/pace vs opposing defending. '
                     'Unique names, no fuzzy matching; original overall must agree. Fold-only imputation.',
        limitation='Season-edition FIFA files have no verified publication timestamps. This inherits '
                   'the historical pipeline snapshot limitation and is an exploratory experiment, not a '
                   'strict as-of live backtest. No named EPL lineups or pre-2014 subratings used.'
    ), indent=2), encoding='utf-8')
    print(pd.DataFrame(coverage).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
