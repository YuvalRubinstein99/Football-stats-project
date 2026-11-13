"""Audit every requested season, including missing and partially collected seasons."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from scrape_recent_seasons import OUT, LEAGUES, write_csv


def read(out, name):
    path = out / name
    if path.exists():
        with path.open(encoding='utf-8-sig', newline='') as stream:
            yield from csv.DictReader(stream)


def audit(out=OUT, status=None):
    status = status or json.loads((out / 'status.json').read_text(encoding='utf-8'))
    counts, odds, seen = Counter(), Counter(), set()
    for row in read(out, 'results_odds.csv'):
        key = tuple(row[k] for k in ('season', 'league', 'fixture_date', 'home_team_name', 'away_team_name'))
        if key in seen:
            raise ValueError('Duplicate fixture: ' + str(key))
        seen.add(key)
        counts[key[:2]] += 1
        odds[key[:2]] += row['valid_bet365'] == 'True'
    games = defaultdict(list)
    for row in read(out, 'games.csv'):
        games[row['season'], LEAGUES[row['competition_id']]].append(row)
    starters = defaultdict(list)
    for row in read(out, 'lineups.csv'):
        if row['type'] == 'starting_lineup':
            starters[row['game_id'], row['club_id']].append(row['player_id'])
    years = sorted(map(str, status.get('seasons') or {k[0] for k in counts.keys() | games.keys()}))
    audits = []
    for year in years:
        for league in status.get('leagues', list(LEAGUES.values())):
            key = year, league
            full = 0
            for row in games[key]:
                home = starters[row['game_id'], row['home_club_id']]
                away = starters[row['game_id'], row['away_club_id']]
                full += len(home) == len(away) == 11 and len(set(home + away)) == 22 and '' not in home + away
            audits.append(dict(season=year, league=league, results_matches=counts[key],
                bet365_matches=odds[key], lineup_source_matches=len(games[key]),
                complete_unique_starter_sets=full, model_ready_matches=0,
                results_from_successful_run=status.get('results_published', True),
                note='Not joined/rated' if counts[key] and games[key] else 'Missing results or lineup source'))
    write_csv(out / 'collection_audit.csv', audits)
    return audits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    print(json.dumps(audit(args.output), indent=2))


if __name__ == '__main__':
    main()
