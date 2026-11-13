"""Recover preferred positions from the verified existing player matching audit."""
import hashlib
import json
from select_expanded_features import ROOT, pd
from prepare_team_attributes import RENAMES
from classifier_features import club
from positional_features import aggregate_units, matchup_features, primary_position, unit, feature_columns

DATA = ROOT / 'results/team_attributes'
OUT = ROOT / 'results/positional_matchups'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    original = json.loads((DATA / 'feature_audit.json').read_text())
    # Audit row IDs are meaningful only for the exact source snapshots.
    for name, expected in original['input_hashes'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Stale matching audit: rebuild team attributes first ({name})')
    audit = pd.read_csv(DATA / 'player_matching_audit.csv')
    features, position_audit = {}, []
    for year in range(2014, 2023):
        players = pd.read_csv(ROOT / f'data/players_{(year+1)%100:02d}.csv',
                              low_memory=False).rename(columns=RENAMES).to_dict('records')
        source = pd.read_csv(ROOT / f'data/season_data/season{year%100:02d}{(year+1)%100:02d}.csv',
                             low_memory=False)
        keys = [(year, club(r.home_team_name), club(r.away_team_name)) for r in source.itertuples()]
        duplicates = pd.Series(keys).duplicated(keep=False).to_numpy()
        for row_id, group in audit[audit.season == year].groupby('source_row', sort=False):
            key = keys[int(row_id)]
            if duplicates[int(row_id)]:
                continue
            accepted = group[group.accepted == True]
            duplicate_players = accepted.rating_row.duplicated().any()
            resolved = {'home': [], 'away': []}
            for r in group.itertuples():
                compatible = bool(r.accepted) and not duplicate_players
                player = players[int(r.rating_row)] if pd.notna(r.rating_row) else None
                if bool(r.accepted) and (player is None or float(player['overall']) != r.original_overall
                                        or player['long_name'] != r.matched_name):
                    raise ValueError('Matching audit is inconsistent with player snapshot')
                if compatible:
                    resolved[r.side].append(player)
                position_audit.append(dict(season=year, lineup_source_row=int(row_id), side=r.side,
                    slot=r.slot, source_name=r.source_name, matched_name=r.matched_name,
                    rating_row=r.rating_row, match_method=r.method, accepted=compatible,
                    duplicate_player_fixture=bool(duplicate_players),
                    positions=player.get('player_positions', '') if player else '',
                    primary_position=primary_position(player.get('player_positions')) if compatible else '',
                    unit=unit(player.get('player_positions')) if compatible else 'unmatched'))
            feat = {**aggregate_units(resolved['home'], 'home'),
                    **aggregate_units(resolved['away'], 'away')}
            features[key] = {**feat, **matchup_features(feat)}
        print(f'Prepared preferred-position features: {year}', flush=True)
    coverage = []
    for split in ['train', 'test']:
        frame = pd.read_csv(DATA / f'{split}.csv')
        rows = []
        for r in frame.itertuples():
            key = int(r.season), club(r.home_team_name), club(r.away_team_name)
            if key not in features:
                raise ValueError(f'No unambiguous positional source: {key}')
            rows.append(features[key])
        joined = pd.concat([frame.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
        if joined.duplicated(['fixture_date', 'league', 'home_team_name', 'away_team_name']).any():
            raise ValueError('Duplicate evaluation fixtures')
        joined.to_csv(OUT / f'{split}.csv', index=False)
        means = [f'{s}_{u}_overall_mean' for s in ['home', 'away']
                 for u in ['attack', 'midfield', 'defense', 'goalkeeper']]
        gaps = list(matchup_features(features[next(iter(features))]))
        for (season, league), g in joined.groupby(['season', 'league']):
            coverage.append(dict(split=split, season=int(season), league=league, matches=len(g),
                all_units_available=int(g[means].notna().all(axis=1).sum()),
                all_matchups_available=int(g[gaps].notna().all(axis=1).sum()),
                complete_22_positions=int(((g.home_position_matched_count == 11)
                    & (g.away_position_matched_count == 11)
                    & (g.home_position_unknown_count == 0) & (g.away_position_unknown_count == 0)).sum())))
    pd.DataFrame(position_audit).to_csv(OUT / 'position_audit.csv', index=False)
    pd.DataFrame(coverage).to_csv(OUT / 'coverage.csv', index=False)
    paths = [DATA / 'player_matching_audit.csv', DATA / 'feature_audit.json',
             DATA / 'train.csv', DATA / 'test.csv', ROOT / 'src/positional_features.py',
             ROOT / 'scripts/prepare_positional_features.py']
    metadata = dict(columns=feature_columns(), source_hashes=original['input_hashes'],
        input_hashes={p.relative_to(ROOT).as_posix(): digest(p) for p in paths}, coverage=coverage,
        position_rule='First listed FIFA preferred position; one unit per player. LW/RW attack, '
            'LM/RM/CAM/CDM midfield, LWB/RWB defense. Not observed match-day positions.',
        missing_rule='Means need >=8 accepted starters per side, >=1 attacker/goalkeeper or >=2 '
            'midfielders/defenders with valid ratings. Counts retained. Missing values never zero-filled '
            'before subtraction; model imputation uses training folds only. Partial units are estimates.',
        caveat=original['limitation'])
    (OUT / 'feature_audit.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(pd.DataFrame(coverage).groupby('split').sum(numeric_only=True).drop(columns='season').to_string())


if __name__ == '__main__':
    main()
