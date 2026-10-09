"""Expected lineups + current EA ratings + form + odds -> saved random forest."""
import csv
from datetime import datetime, timezone
import io
import json
import sys

from live_predictions import ROOT, LIVE, LEAGUES, download, deduplicate, atomic_write, probabilities


def run(leagues=None, seasons=3, days=14, offline=False, progress=print, output=LIVE, **unused):
    # Optional project-local dependencies used by the desktop development setup.
    if (ROOT / '.lineup_deps').exists():
        sys.path.insert(0, str(ROOT / '.lineup_deps'))
    try:
        from lineup_sources import RW_URL, RW_LEAGUES, Ratings, fetch, parse_lineups, match_player, team_key
        from lineup_model import LineupForest, form_features, rating_features
    except ImportError as exc:
        raise ValueError('Install lineup dependencies: python -m pip install -r requirements-live.txt') from exc
    leagues = list(dict.fromkeys(leagues or LEAGUES))
    if any(l not in LEAGUES for l in leagues) or not 1 <= seasons <= 10 or not 1 <= days <= 90:
        raise ValueError('Invalid leagues or date/history window')
    now = datetime.now(timezone.utc)
    today = now.date()
    year = today.year if today.month >= 7 else today.year-1
    progress('Loading the saved lineup random forest and verifying its feature schema…')
    forest = LineupForest()
    overrides_path = ROOT / 'data/live_overrides.json'
    overrides = json.loads(overrides_path.read_text(encoding='utf-8')) if overrides_path.exists() else {}
    predictions, skipped, sources, warnings, history, lineups = [], [], [], [], [], []
    for league in leagues:
        progress(f'Downloading current-season results for {LEAGUES[league]}…')
        rows, source = download(f'mmz4281/{year%100:02d}{(year+1)%100:02d}/{league}.csv', output/'cache', offline, True)
        if not rows or any(r['league'] != league for r in rows):
            raise ValueError('Empty or wrong-league results: ' + league)
        history.extend(rows)
        sources.append(source)
        progress(f'Downloading expected lineups for {LEAGUES[league]}…')
        try:
            html, source = fetch(RW_URL + RW_LEAGUES[league], output/'source_cache', offline)
            source['kind'] = 'lineups'
            sources.append(source)
            if (now - datetime.fromisoformat(source['fetched_at'])).total_seconds() > 86400:
                warnings.append(f'{league}: cached lineups are older than 24 hours; predictions will be withheld.')
            else:
                lineups.extend(parse_lineups(html, league, today))
        except (OSError, ValueError, KeyError) as exc:
            warnings.append(f'{league} lineups unavailable: {exc}')
    fixtures, source = download('fixtures.csv', output/'cache', offline)
    sources.append(source)
    history = deduplicate(history)
    completed = {tuple(m[k] for k in ('league', 'date', 'home', 'away')) for m in history}
    fixtures = [m for m in deduplicate(fixtures) if m['league'] in leagues and 0 <= (m['date']-today).days <= days
                and tuple(m[k] for k in ('league', 'date', 'home', 'away')) not in completed]
    progress('Loading the current EA ratings catalogue…')
    ratings = Ratings(output/'source_cache', offline)
    lineup_index = {}
    for item in lineups:
        key = (item['league'], item['date'], team_key(item['home']), team_key(item['away']))
        lineup_index.setdefault(key, []).append(item)
    audits = []
    for fixture in fixtures:
        f = {**fixture, 'date': fixture['date'].isoformat()}
        label = f'{f["home"]} vs {f["away"]}'
        progress('Matching lineups and ratings: ' + label)
        try:
            key = (f['league'], f['date'], team_key(f['home']), team_key(f['away']))
            candidates = lineup_index.get(key, [])
            if len(candidates) != 1:
                raise ValueError('No unique expected lineup for this fixture and date')
            lineup = candidates[0]
            if lineup.get('kickoff_utc'):
                if datetime.fromisoformat(lineup['kickoff_utc']) <= now:
                    raise ValueError('Kickoff has passed; live/in-play prediction is not supported')
            elif fixture['date'] == today:
                raise ValueError('Cannot verify kickoff time for today\'s fixture')
            form, counts = form_features(history, fixture, today)
            fixture_key = '|'.join([f['league'], f['date'], f['home'], f['away']])
            if any(g['league'] == f['league'] and g['date'] < fixture['date'] and
                   ({team_key(g['home']), team_key(g['away'])} & {team_key(f['home']), team_key(f['away'])})
                   for g in fixtures):
                raise ValueError('An intervening fixture changes form; refresh after that match is completed')
            matchweek = overrides.get('matchweeks', {}).get(fixture_key)
            if matchweek is None:
                # Explicitly labeled estimate. Do not infer through unequal games played or intervening fixtures.
                if counts['home'] != counts['away']:
                    raise ValueError('Unequal games played: supply the official matchweek in data/live_overrides.json')
                matchweek = counts['home'] + 1
                week_source = 'estimated from equal games played; verify against official schedule'
            else:
                week_source = 'user-supplied official matchweek'
            if type(matchweek) is not int or not 1 <= matchweek <= 50:
                raise ValueError('Invalid matchweek override')
            row = {**form, 'Matchweek': matchweek, 'home_team_name': f['home'], 'away_team_name': f['away']}
            for field in ('B365H', 'B365D', 'B365A'):
                row[field] = f.get(field)
                if row[field] is None:
                    raise ValueError('Missing Bet365 odds: ' + field)
            matched = {}
            for side, prefix in [('home', 'HomePlayer'), ('away', 'AwayPlayer')]:
                players = lineup[side+'_players']
                if lineup[side+'_status'] not in ('predicted', 'confirmed') or len(players) != 11:
                    raise ValueError(f'{side}: complete expected/confirmed starting XI unavailable')
                squad = ratings.squad(lineup[side])
                mapped, errors = [], []
                for player in players:
                    try:
                        mapped.append(match_player(player, squad, overrides.get('player_ids', {})))
                    except ValueError as exc:
                        if str(exc).startswith('Rating missing:') and str(player['source_id']) not in overrides.get('player_ids', {}):
                            try:
                                mapped.append(ratings.search_player(player))
                            except (ValueError, OSError, KeyError) as search_exc:
                                errors.append(str(search_exc))
                        else:
                            errors.append(str(exc))
                if errors:
                    raise ValueError('; '.join(errors))
                row.update(rating_features(mapped, prefix))
                matched[side] = mapped
            h, a = forest.predict(row)
            p, score = probabilities(max(h, 1e-8), max(a, 1e-8))
            unknown = [f[s] for s in ('home', 'away') if team_key(f[s]) not in forest.teams]
            notes = []
            if unknown:
                notes.append('Team unseen in training: ' + ', '.join(unknown))
            if week_source.startswith('estimated'):
                notes.append('Matchweek estimated from games played')
            mismatch = [x['lineup_name'] for side in matched.values() for x in side
                        if x['match_method'] == 'global_exact_name_club_mismatch']
            if mismatch:
                notes.append('EA club differs from lineup (exact-name match): ' + ', '.join(mismatch))
            predictions.append({**f, 'expected_home_goals': h, 'expected_away_goals': a,
                'p_home': p[0], 'p_draw': p[1], 'p_away': p[2], 'modal_score': score,
                'pick': ['Home', 'Draw', 'Away'][max(range(3), key=p.__getitem__)],
                'limited_history': bool(unknown), 'lineup_status': lineup['home_status']+'/'+lineup['away_status'],
                'ratings_edition': ratings.edition, 'matchweek': matchweek, 'notes': '; '.join(notes),
                'home_lineup': matched['home'], 'away_lineup': matched['away']})
            audits.append({'fixture': fixture_key, 'features': row, 'matchweek_source': week_source})
        except (ValueError, OSError, KeyError) as exc:
            skipped.append({**f, 'reason': str(exc)})
    sources.extend({**s, 'kind': 'ratings'} for s in ratings.sources)
    warnings.append('Lineup RF uses the existing historical model; future accuracy is unverified. EA launch ratings may differ from historical career-mode ratings.')
    if any(p['notes'].find('Matchweek estimated') >= 0 for p in predictions):
        warnings.append('Some matchweeks are estimates. Check the matchweek in each lineup audit, especially after postponements.')
    if skipped:
        warnings.append(f'{len(skipped)} fixtures withheld because required inputs could not be verified. No Poisson-baseline fallback was used.')
    if not fixtures:
        warnings.append('No fixtures in the selected window.')
    report = {'generated_at': now.isoformat(), 'as_of': today.isoformat(), 'model': forest.metadata['name'],
        'model_kind': 'lineup_rf', 'model_metadata': forest.metadata, 'ratings_edition': ratings.edition,
        'leagues': leagues, 'days': days, 'seasons': 1, 'sources': sources, 'warnings': warnings,
        'validation': {}, 'predictions': predictions, 'skipped': skipped, 'feature_audit': audits}
    from live_bets import save as save_strategies
    save_strategies(report, output)
    atomic_write(output/'predictions.json', json.dumps(report, indent=2, allow_nan=False))
    fields = ['league','date','time','home','away','expected_home_goals','expected_away_goals','p_home','p_draw','p_away',
              'pick','modal_score','lineup_status','ratings_edition','matchweek','notes']
    buf=io.StringIO(); writer=csv.DictWriter(buf, fieldnames=fields, extrasaction='ignore')
    writer.writeheader(); writer.writerows(predictions)
    atomic_write(output/'predictions.csv', buf.getvalue())
    progress(f'Done: {len(predictions)} lineup RF predictions; {len(skipped)} fixtures need attention.')
    return report
