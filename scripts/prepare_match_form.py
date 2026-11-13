"""Rebuild form offline using complete cached league histories, before filtering."""
import csv
import hashlib
import json
from pathlib import Path
from select_expanded_features import ROOT, pd, np
from live_predictions import parse_csv
from match_form import build_form, fixture_key

OUT = ROOT / 'results/match_form'


def load_history():
    games, inputs, coverage = [], {}, []
    for year in range(2008, 2023):
        season = f'{year%100:02d}{(year+1)%100:02d}'
        for league in ['E0', 'D1', 'SP1', 'I1']:
            paths = [ROOT / f'data/older_seasons/raw/{league}_{season}.csv',
                     ROOT / f'results/classifier/cache/mmz4281_{season}_{league}.csv']
            path = next((p for p in paths if p.exists()), None)
            if path is None:
                raise FileNotFoundError(f'Missing cached history: {league} {year}')
            payload = path.read_bytes()
            inputs[path.relative_to(ROOT).as_posix()] = hashlib.sha256(payload).hexdigest()
            parsed = parse_csv(payload, results=True)
            # Retain the same completed rows as parse_csv, including games with no odds.
            raw = [r for r in csv.DictReader(payload.decode('utf-8-sig', errors='replace').splitlines())
                   if r.get('Date') and r.get('HomeTeam') and r.get('FTHG') and r.get('FTAG')]
            if len(raw) != len(parsed):
                raise ValueError('Cached source row alignment failed')
            for r, original in zip(parsed, raw):
                if r['league'] != league:
                    raise ValueError('Wrong cached league')
                r['season'] = year
                for key, field in [('hst', 'HST'), ('ast', 'AST')]:
                    text = original.get(field, '')
                    value = float(text) if text.strip() else np.nan
                    if np.isfinite(value) and (value < 0 or not value.is_integer()):
                        raise ValueError('Invalid shots-on-target observation')
                    r[key] = value
                games.append(r)
            coverage.append(dict(league=league, season=year, matches=len(parsed),
                shots_complete=sum(np.isfinite([r['hst'], r['ast']]).all() for r in parsed)))
    return games, inputs, coverage


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    games, inputs, coverage = load_history()
    lookup = {fixture_key(g): g for g in games}
    if len(lookup) != len(games):
        raise ValueError('Duplicate cached fixture')
    frames, queries, excluded = {}, [], []
    for split in ('train', 'test'):
        path = ROOT / f'results/classifier/{split}.csv'
        inputs[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        frame = pd.read_csv(path)
        frame['source_row'] = np.arange(len(frame))
        duplicate = frame.duplicated(['league', 'fixture_date', 'home_team_name', 'away_team_name'], keep=False)
        keep = []
        for i, row in frame.iterrows():
            query = dict(league=row.league, season=int(row.season),
                         date=pd.Timestamp(row.fixture_date).date(),
                         home=row.home_team_name, away=row.away_team_name)
            actual = lookup.get(fixture_key(query))
            reason = ('ambiguous_duplicate_model_fixture' if duplicate.iloc[i] else
                      'fixture_not_in_cached_history' if actual is None else
                      'target_score_disagrees_with_dated_history' if
                      (actual['hg'], actual['ag']) != (row.home_score, row.away_score) else '')
            if reason:
                excluded.append(dict(split=split, source_row=i, reason=reason))
                continue
            if actual['season'] != query['season']:
                raise ValueError('Season mismatch')
            keep.append(i)
            queries.append(query)
        frames[split] = frame.loc[keep].copy()
    features = build_form(games, queries)
    for split, frame in frames.items():
        extra = [features[fixture_key(dict(league=r.league, date=pd.Timestamp(r.fixture_date).date(),
                 home=r.home_team_name, away=r.away_team_name))] for r in frame.itertuples()]
        joined = pd.concat([frame.reset_index(drop=True), pd.DataFrame(extra)], axis=1)
        for side in ('home', 'away'):
            latest = pd.to_datetime(joined[side + '_recent_latest_date'])
            if not (latest.isna() | (latest < pd.to_datetime(joined.fixture_date))).all():
                raise ValueError('Form contains same-day or future observations')
        joined.to_csv(OUT / (split + '.csv'), index=False)
        joined[['source_row', 'league', 'season', 'fixture_date',
                'home_recent_latest_date', 'away_recent_latest_date',
                'home_recent_matches_5', 'away_recent_matches_5',
                'home_recent_sot_matches_5', 'away_recent_sot_matches_5']].to_csv(
                    OUT / (split + '_temporal_audit.csv'), index=False)
    pd.DataFrame(excluded).to_csv(OUT / 'excluded_rows.csv', index=False)
    pd.DataFrame(coverage).to_csv(OUT / 'source_coverage.csv', index=False)
    inputs['src/match_form.py'] = hashlib.sha256((ROOT / 'src/match_form.py').read_bytes()).hexdigest()
    inputs['scripts/prepare_match_form.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    audit = dict(input_hashes=inputs, history_matches=len(games),
        train_matches=len(frames['train']), test_matches=len(frames['test']), exclusions=excluded,
        output_hashes={s: hashlib.sha256((OUT / (s+'.csv')).read_bytes()).hexdigest() for s in frames},
        rules='All cached completed games feed history, including games excluded from model fitting. '
              'Strictly prior dates, explicit source seasons (including COVID postponements), '
              'within-season windows 3/5/10 and exponential half-life 5 matches; '
              'missing shots remain missing, with coverage counts and fold-fitted imputation.',
        limitations='Existing EA snapshot dates and odds timing retain previous provenance limitations. '
                    '2021-22 audit and 2022-23 test have been inspected previously. '
                    'History updates after completed matches within validation/test seasons; '
                    'model parameters stay fixed for each season.')
    (OUT / 'data_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('input_hashes', 'output_hashes')}, indent=2))


if __name__ == '__main__':
    main()
