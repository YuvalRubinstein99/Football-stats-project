"""Rebuild identities and positional features on the unchanged comparison sample."""
import json
import hashlib
from collections import defaultdict
from select_expanded_features import ROOT, np, pd
from prepare_team_attributes import RENAMES
from classifier_features import club
from player_identity import IdentityIndex, stable_id, name_key, canonical_club
from positional_features import aggregate_units, matchup_features, unit

OUT = ROOT / 'results/lineup_improvements'
OLD = ROOT / 'results/positional_matchups'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'results/team_attributes/feature_audit.json').read_text())
    for name, expected in manifest['input_hashes'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Source changed since original sample: {name}')
    previous_names = defaultdict(set)
    fixtures, audit = {}, []
    for year in range(2014, 2023):
        rating_path = ROOT / f'data/players_{(year+1)%100:02d}.csv'
        rating_frame = pd.read_csv(rating_path, low_memory=False).drop_duplicates().rename(columns=RENAMES)
        rating_frame['_source_rating_row'] = rating_frame.index
        players = rating_frame.to_dict('records')
        index = IdentityIndex(players, previous_names)
        source = pd.read_csv(ROOT / f'data/season_data/season{year%100:02d}{(year+1)%100:02d}.csv')
        keys = [(year, club(r.home_team_name), club(r.away_team_name)) for r in source.itertuples()]
        duplicates = pd.Series(keys).duplicated(keep=False).to_numpy()
        for row_id, r in enumerate(source.to_dict('records')):
            if duplicates[row_id]:
                continue
            resolved, records = {'home': [], 'away': []}, []
            for side, prefix in [('home', 'HomePlayer'), ('away', 'AwayPlayer')]:
                for slot in range(1, 12):
                    name = r.get(f'{prefix}{slot}')
                    pid, method = index.match(str(name), r[f'{side}_team_name'])
                    p = players[pid] if pid is not None else None
                    identifier = stable_id(p) if p else None
                    legacy = r.get(f'{prefix}{slot}_Overall', np.nan)
                    accepted = p is not None and identifier is not None
                    entry = dict(season=year, lineup_source_row=row_id, side=side, slot=slot,
                        team=r[f'{side}_team_name'], source_name=name,
                        matched_name=p['long_name'] if p else '', player_id=identifier,
                        rating_row=p['_source_rating_row'] if p else None, method=method, accepted=accepted,
                        club_agrees=bool(p and canonical_club(p.get('club_name')) == canonical_club(r[f'{side}_team_name'])),
                        original_overall=legacy, rating_overall=p['overall'] if p else np.nan,
                        overall_agrees=bool(p and pd.notna(legacy) and float(p['overall']) == float(legacy)),
                        duplicate_fixture=False)
                    records.append(entry)
                    if accepted:
                        resolved[side].append(dict(player_id=identifier,
                            **{k: p.get(k) for k in ['player_positions', 'overall', 'shooting', 'passing',
                                                     'defending', 'dribbling', 'pace', 'physicality']}))
            ids = [p['player_id'] for side in resolved.values() for p in side]
            if len(ids) != len(set(ids)):
                resolved = {'home': [], 'away': []}
                for entry in records:
                    entry.update(accepted=False, duplicate_fixture=True)
            audit.extend(records)
            feat = {**aggregate_units(resolved['home'], 'home'), **aggregate_units(resolved['away'], 'away')}
            fixtures[keys[row_id]] = dict(features={**feat, **matchup_features(feat)}, players=resolved)
        # Only earlier editions supply alternate names to later editions.
        for p in players:
            pid = stable_id(p)
            if pid:
                for field in ['short_name', 'long_name']:
                    key = name_key(p.get(field))
                    if key:
                        previous_names[key].add(pid)
        print(f'Rematched {year}', flush=True)
    lineups, coverage = {}, []
    for split in ['train', 'test']:
        frame = pd.read_csv(OLD / f'{split}.csv')
        fresh = []
        for r in frame.itertuples():
            result = fixtures[int(r.season), club(r.home_team_name), club(r.away_team_name)]
            fresh.append(result['features'])
            lineups[f'{split}:{r.source_row}'] = result['players']
        fresh = pd.DataFrame(fresh)
        improved = frame.copy()
        for column in fresh:
            improved[column] = fresh[column].to_numpy()
        improved.to_csv(OUT / f'{split}.csv', index=False)
        for version, data in [('old', frame), ('improved', improved)]:
            for season, g in data.groupby('season'):
                cols = [c for c in fresh if '_vs_' in c or c.startswith('midfield_') or c == 'goalkeeper_overall_advantage']
                coverage.append(dict(split=split, season=int(season), version=version, matches=len(g),
                    matched_players=int(g[['home_position_matched_count','away_position_matched_count']].sum().sum()),
                    all_matchups=int(g[cols].notna().all(axis=1).sum()),
                    complete_22=int(((g.home_position_matched_count == 11) & (g.away_position_matched_count == 11)
                        & (g.home_position_unknown_count == 0) & (g.away_position_unknown_count == 0)).sum())))
    pd.DataFrame(audit).to_csv(OUT / 'identity_audit.csv', index=False)
    pd.DataFrame(coverage).to_csv(OUT / 'coverage.csv', index=False)
    pd.DataFrame(audit).query('accepted == False').groupby(['source_name','team','method'], dropna=False).size().sort_values(
        ascending=False).rename('appearances').reset_index().to_csv(OUT / 'unresolved_names.csv', index=False)
    (OUT / 'lineups.json').write_text(json.dumps(lineups, allow_nan=True), encoding='utf-8')
    paths = [ROOT / 'src/player_identity.py', ROOT / 'src/positional_features.py',
             ROOT / 'scripts/prepare_lineup_improvements.py', OLD / 'train.csv', OLD / 'test.csv']
    (OUT / 'identity_manifest.json').write_text(json.dumps(dict(
        source_hashes=manifest['input_hashes'],
        input_hashes={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        rule='Identity determined by unique names, explicit club aliases, club-constrained token/prefix '
             'matches and names from earlier FIFA editions linked by stable EA ID. No fuzzy edit-distance '
             'matching. Legacy overall agreement is an audit field, not an identity gate. Duplicate IDs '
             'invalidate a fixture. Identical snapshot rows are deduplicated before matching; '
             'conflicting duplicate FIFA IDs raise an error. Preferred positions, not observed formation roles.',
        caveat=manifest['limitation']), indent=2), encoding='utf-8')
    print(pd.DataFrame(coverage).groupby(['split','version']).sum(numeric_only=True).drop(columns='season').to_string())


if __name__ == '__main__':
    main()
