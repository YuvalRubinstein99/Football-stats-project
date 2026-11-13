"""Extend seasons using the existing Football-Data/Transfermarkt source collectors."""
import argparse
import csv
import gzip
import io
import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from scrape_older_seasons import atomic_json, csv_rows, date_value, digest, fetch, odds_ok, stamp, write_csv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/recent_seasons'
TM = 'https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/'
LEAGUES = {'GB1': 'E0', 'ES1': 'SP1', 'L1': 'D1', 'IT1': 'I1'}
RATINGS_URL = ('https://www.kaggle.com/api/v1/datasets/download/'
               'stefanoleone992/ea-sports-fc-24-complete-player-dataset')
RESULT_FIELDS = ['season', 'league', 'fixture_date', 'home_team_name', 'away_team_name',
                 'home_score', 'away_score', 'B365H', 'B365D', 'B365A', 'valid_bet365', 'source_file']


def valid_gzip(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        if not next(csv.DictReader(stream), None):
            raise ValueError('Empty compressed CSV')


def valid_ratings(path):
    with zipfile.ZipFile(path) as archive:
        if not any(n.endswith('.csv') for n in archive.namelist()) or archive.testzip():
            raise ValueError('Invalid ratings archive')


def stream_csv(path, rows, fields):
    """Atomic, constant-memory writing; empty selections clear stale rows."""
    temp = path.with_suffix('.tmp')
    count = 0
    try:
        with temp.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
                count += 1
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return count


def result_rows(path, year, league, as_of):
    matches, seen = [], set()
    for row in csv_rows(path):
        hg, ag = str(row.get('FTHG', '')).strip(), str(row.get('FTAG', '')).strip()
        if not hg.isdigit() or not ag.isdigit():
            continue
        day = date_value(row['Date'])
        if day > as_of:
            continue
        key = day, row['HomeTeam'], row['AwayTeam']
        if not all(key) or key in seen:
            raise ValueError('Invalid or duplicate fixture ' + str(key))
        seen.add(key)
        matches.append(dict(season=year, league=league, fixture_date=day,
            home_team_name=row['HomeTeam'], away_team_name=row['AwayTeam'],
            home_score=int(hg), away_score=int(ag), B365H=row.get('B365H', ''),
            B365D=row.get('B365D', ''), B365A=row.get('B365A', ''),
            valid_bet365=odds_ok(row), source_file=path.name))
    return matches


def extraction_cached(receipt, signature, out, names):
    if receipt.exists():
        prior = json.loads(receipt.read_text(encoding='utf-8'))
        if (prior.get('signature') == signature and set(prior.get('outputs', {})) == set(names)
                and all((out / name).exists() and digest(out / name) == sha
                        for name, sha in prior['outputs'].items())):
            return prior['summary']
    return None


def save_receipt(path, signature, summary, out, names):
    atomic_json(path, dict(signature=signature, summary=summary,
                          outputs={name: digest(out / name) for name in names}))


def extract_history(raw, out, years, leagues, as_of, sources):
    names = ['games.csv', 'lineups.csv']
    receipt = out / 'history_extraction.json'
    signature = dict(version=1, years=years, leagues=leagues, as_of=as_of,
                     sources={n: sources[n]['sha256'] for n in ('games', 'game_lineups')})
    prior = extraction_cached(receipt, signature, out, names)
    if prior is not None:
        return {**prior, 'reused_extraction': True}
    competitions = {key for key, league in LEAGUES.items() if league in leagues}
    year_set, games = set(map(str, years)), {}
    with gzip.open(raw / 'games.csv.gz', 'rt', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        for row in reader:
            if (row['competition_id'] not in competitions or row['season'] not in year_set
                    or not row.get('date') or row['date'][:10] > as_of):
                continue
            if not all(str(row.get(side + '_club_goals', '')).isdigit() for side in ('home', 'away')):
                continue
            if row['game_id'] in games:
                raise ValueError('Duplicate game ID: ' + row['game_id'])
            games[row['game_id']] = row
    # Stage both outputs; a failed archive scan leaves the previous pair intact.
    staged_games, staged_lineups = out / 'games.staged.csv', out / 'lineups.staged.csv'
    stream_csv(staged_games, games.values(), fields)
    starters = {}
    try:
        with gzip.open(raw / 'game_lineups.csv.gz', 'rt', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            def selected():
                for row in reader:
                    if row['game_id'] not in games:
                        continue
                    if row.get('type') == 'starting_lineup':
                        starters.setdefault((row['game_id'], row['club_id']), []).append(row['player_id'])
                    yield row
            rows = stream_csv(staged_lineups, selected(), reader.fieldnames)
        full = 0
        for gid, game in games.items():
            home = starters.get((gid, game['home_club_id']), [])
            away = starters.get((gid, game['away_club_id']), [])
            full += len(home) == len(away) == 11 and len(set(home + away)) == 22 and '' not in home + away
        staged_games.replace(out / 'games.csv')
        staged_lineups.replace(out / 'lineups.csv')
    finally:
        staged_games.unlink(missing_ok=True)
        staged_lineups.unlink(missing_ok=True)
    summary = dict(rows=rows, games=len(games), games_with_22_starters=full,
                   latest_source_match=max((r['date'] for r in games.values()), default=None))
    save_receipt(receipt, signature, summary, out, names)
    return {**summary, 'reused_extraction': False}


def extract_ratings(raw, out, source):
    receipt, names = out / 'ratings_extraction.json', ['ratings_fc23_fc24.csv']
    signature = dict(version=1, sha256=source['sha256'])
    prior = extraction_cached(receipt, signature, out, names)
    if prior is not None:
        return prior
    versions, dates = Counter(), set()
    with zipfile.ZipFile(raw / 'fc24.zip') as archive:
        members = [n for n in archive.namelist()
                   if n.endswith('male_players.csv') and not n.endswith('female_players.csv')]
        if len(members) != 1:
            raise ValueError('Expected male_players.csv not uniquely identified')
        with archive.open(members[0]) as stream:
            reader = csv.DictReader(io.TextIOWrapper(stream, encoding='utf-8-sig'))
            def selected():
                for row in reader:
                    if float(row.get('fifa_version', '0')) >= 23:
                        versions[row['fifa_version']] += 1
                        dates.add(row.get('fifa_update_date') or row.get('update_as_of', ''))
                        yield row
            count = stream_csv(out / names[0], selected(), reader.fieldnames)
    summary = dict(rows=count, versions=dict(versions), dates=sorted(dates))
    save_receipt(receipt, signature, summary, out, names)
    return summary


def main(argv=None):
    today = datetime.now(timezone.utc).date()
    current = today.year if today.month >= 7 else today.year - 1
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--earliest', type=int, default=2023, help='Season starting year')
    parser.add_argument('--latest', type=int, default=current, help='Inclusive season starting year')
    parser.add_argument('--leagues', nargs='+', choices=list(LEAGUES.values()), default=list(LEAGUES.values()))
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--delay', type=float, default=3)
    parser.add_argument('--offline', action='store_true', help='Verified caches only')
    parser.add_argument('--refresh-current', action='store_true', help='Refresh current-season results/odds')
    parser.add_argument('--refresh-archives', action='store_true', help='Refresh shared games/lineups/players')
    args = parser.parse_args(argv)
    if not 2014 <= args.earliest <= args.latest <= current or args.delay < 1:
        parser.error('Use season starting years 2014 through current season, delay >= 1')
    if args.offline and (args.refresh_current or args.refresh_archives):
        parser.error('Offline mode cannot refresh sources')
    out, years = args.output.resolve(), list(range(args.earliest, args.latest + 1))
    leagues = list(dict.fromkeys(args.leagues))
    raw = out / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    lock = out / 'run.lock'
    try:
        lock.mkdir()
    except FileExistsError:
        parser.error('Collection already running or stale run.lock exists; verify the process first')
    status = dict(started_at=stamp(), state='running', seasons=years, leagues=leagues,
                  sources=[], errors=[], model_ready=False)
    try:
        rows, coverage = [], []
        for year in years:
            code = f'{year%100:02d}{(year+1)%100:02d}'
            for league in leagues:
                label, path = f'{code}/{league}', raw / f'{code}_{league}.csv'
                entry = dict(season=year, league=league, matches=0,
                    expected=306 if league == 'D1' else 380, complete=False, with_bet365=0,
                    season_status='in_progress' if year == current else 'historical',
                    source_status='unavailable')
                try:
                    print('Results ' + label, flush=True)
                    source = fetch(f'https://www.football-data.co.uk/mmz4281/{label}.csv',
                        path, csv_rows, args.delay, offline=args.offline,
                        refresh=args.refresh_current and year == current)
                    status['sources'].append(source)
                    matches = result_rows(path, year, league, today.isoformat())
                    rows.extend(matches)
                    entry.update(matches=len(matches), complete=len(matches) == entry['expected'],
                        with_bet365=sum(r['valid_bet365'] for r in matches), source_status='available')
                except Exception as exc:
                    status['errors'].append(dict(source=label, error=str(exc)))
                coverage.append(entry)
                atomic_json(out / 'status.json', status)
        # Retain the previous complete result table if any requested partition failed.
        status['results_published'] = not status['errors']
        if status['results_published']:
            stream_csv(out / 'results_odds.csv', rows, RESULT_FIELDS)
            write_csv(out / 'coverage.csv', coverage)
        else:
            write_csv(out / 'attempted_coverage.csv', coverage)
        sources = {}
        for table in ('games', 'game_lineups', 'players'):
            path = raw / (table + '.csv.gz')
            try:
                print('Historical ' + table, flush=True)
                sources[table] = fetch(TM + path.name, path, valid_gzip, args.delay,
                                      offline=args.offline, refresh=args.refresh_archives)
                status['sources'].append(sources[table])
            except Exception as exc:
                status['errors'].append(dict(source=table, error=str(exc)))
            atomic_json(out / 'status.json', status)
        if {'games', 'game_lineups'} <= sources.keys():
            status['lineups'] = extract_history(raw, out, years, leagues, today.isoformat(), sources)
        try:
            print('Historical FC23/24 ratings', flush=True)
            source = fetch(RATINGS_URL, raw / 'fc24.zip', valid_ratings, args.delay, offline=args.offline)
            status['sources'].append(source)
            status['ratings'] = extract_ratings(raw, out, source)
        except Exception as exc:
            status['errors'].append(dict(source='FC24 ratings', error=str(exc)))
        status.update(finished_at=stamp(), results_matches=len(rows), coverage=coverage,
            state='completed_with_gaps' if status['errors'] else 'completed',
            note='Sources collected separately; identities and dated ratings still require joining '
                 'before model use. See collection_audit.csv for requested seasons and source gaps.')
        atomic_json(out / 'status.json', status)
        from audit_recent_seasons import audit
        audit(out, status)
        print(json.dumps({k: v for k, v in status.items() if k not in ('sources', 'coverage')}, indent=2), flush=True)
        return 1 if status['errors'] else 0
    except BaseException:
        status.update(state='interrupted_or_failed', finished_at=stamp())
        atomic_json(out / 'status.json', status)
        raise
    finally:
        lock.rmdir()


if __name__ == '__main__':
    raise SystemExit(main())
