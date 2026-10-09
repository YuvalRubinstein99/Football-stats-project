"""Download football-data.co.uk CSVs and fit a pre-match Poisson baseline.

Standard library only. This is independent of the legacy player-rating forest.
"""
import argparse
import csv
from collections import defaultdict
from datetime import date, datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / 'results' / 'live'
LEAGUES = {'E0': 'Premier League', 'SP1': 'La Liga', 'D1': 'Bundesliga',
           'I1': 'Serie A', 'F1': 'Ligue 1'}
BASE = 'https://www.football-data.co.uk/'
MODEL = 'Recency-weighted, shrunk home/away attack-defence Poisson baseline v1'


def parse_date(value):
    for fmt in ('%d/%m/%Y', '%d/%m/%y', '%Y-%m-%d'):
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            pass
    raise ValueError('Invalid match date: ' + value)


def parse_csv(payload, results=False):
    text = payload.decode('utf-8-sig', errors='replace')
    reader = csv.DictReader(io.StringIO(text))
    required = {'Div', 'Date', 'HomeTeam', 'AwayTeam'}
    if results:
        required |= {'FTHG', 'FTAG'}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError('Source did not return the expected football CSV columns')
    rows = []
    for row in reader:
        if not row.get('Date') or not row.get('HomeTeam'):
            continue
        if results and (not row.get('FTHG') or not row.get('FTAG')):
            continue
        if not row.get('AwayTeam'):
            raise ValueError('Match has no away team')
        match = {'league': row['Div'].strip(), 'date': parse_date(row['Date']),
                 'home': row['HomeTeam'].strip(), 'away': row['AwayTeam'].strip(),
                 'time': row.get('Time', '')}
        for field in ('B365H', 'B365D', 'B365A'):
            try:
                odds = float(row.get(field, ''))
                match[field] = odds if math.isfinite(odds) and odds > 1 else None
            except (ValueError, TypeError):
                match[field] = None
        if results:
            for target, source in [('hg', 'FTHG'), ('ag', 'FTAG')]:
                value = float(row[source])
                if not math.isfinite(value) or value < 0 or not value.is_integer():
                    raise ValueError('Invalid completed-match score')
                match[target] = int(value)
        rows.append(match)
    return rows


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(text, encoding='utf-8')
    temp.replace(path)


def download(relative, cache, offline=False, results=False):
    path = cache / relative.replace('/', '_')
    if offline:
        payload = path.read_bytes()
    else:
        for attempt in range(3):
            try:
                request = Request(BASE + relative, headers={'User-Agent': 'FootballStatsProject/1.0'})
                with urlopen(request, timeout=30) as response:
                    payload = response.read(10_000_001)
                if len(payload) > 10_000_000:
                    raise ValueError('Source response exceeds 10 MB')
                break
            except (HTTPError, URLError, TimeoutError) as exc:
                if isinstance(exc, HTTPError) and exc.code not in (429, 500, 502, 503, 504):
                    raise
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        parse_csv(payload, results=results)  # Never cache HTML/error pages.
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix('.tmp')
        temp.write_bytes(payload)
        temp.replace(path)
    rows = parse_csv(payload, results=results)
    return rows, {'url': BASE + relative, 'sha256': hashlib.sha256(payload).hexdigest(),
                  'fetched_at': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                  'cached': offline, 'rows': len(rows)}


def deduplicate(rows):
    unique = {}
    for row in rows:
        key = tuple(row[k] for k in ('league', 'date', 'home', 'away'))
        if key in unique and any(unique[key].get(k) != row.get(k) for k in ('hg', 'ag')):
            raise ValueError('Conflicting scores for ' + str(key))
        unique[key] = row
    return sorted(unique.values(), key=lambda r: (r['date'], r['league'], r['home']))


def fit(history, cutoff):
    """Fit only on earlier dates; same-day matches cannot leak into each other."""
    stats = defaultdict(lambda: [0., 0., 0.])
    total_h = total_a = total_w = 0.
    for m in history:
        if m['date'] >= cutoff:
            continue
        w = 0.5 ** ((cutoff - m['date']).days / 180)
        total_h += w * m['hg']
        total_a += w * m['ag']
        total_w += w
        for team, venue, gf, ga in [(m['home'], 'h', m['hg'], m['ag']),
                                    (m['away'], 'a', m['ag'], m['hg'])]:
            s = stats[team, venue]
            s[0] += w * gf
            s[1] += w * ga
            s[2] += w
    if total_w <= 0:
        raise ValueError('No completed matches before prediction date')
    return stats, max(total_h / total_w, .05), max(total_a / total_w, .05)


def goal_rates(model, home, away):
    stats, base_h, base_a = model
    h = stats.get((home, 'h'), [0., 0., 0.])
    a = stats.get((away, 'a'), [0., 0., 0.])
    prior = 8.  # Fixed pseudo-match strength, not tuned on validation matches.
    attack_h = (h[0] + prior * base_h) / (h[2] + prior) / base_h
    defence_h = (h[1] + prior * base_a) / (h[2] + prior) / base_a
    attack_a = (a[0] + prior * base_a) / (a[2] + prior) / base_a
    defence_a = (a[1] + prior * base_h) / (a[2] + prior) / base_h
    return (max(.05, min(8., base_h * attack_h * defence_a)),
            max(.05, min(8., base_a * attack_a * defence_h)), min(h[2], a[2]))


def probabilities(home_rate, away_rate):
    def pmfs(rate):
        p = [math.exp(-rate)]
        for n in range(1, 60):
            p.append(p[-1] * rate / n)
        return p
    hp, ap = pmfs(home_rate), pmfs(away_rate)
    outcomes = [0., 0., 0.]  # Home / draw / away.
    best = (-1., 0, 0)
    for h, ph in enumerate(hp):
        for a, pa in enumerate(ap):
            p = ph * pa
            outcomes[0 if h > a else 1 if h == a else 2] += p
            if p > best[0]:
                best = (p, h, a)
    total = sum(outcomes)
    return [p / total for p in outcomes], f'{best[1]}-{best[2]}'


def validate(history):
    """Last 60 eligible matches, walk-forward with a 100-match warm-up."""
    candidates = history[-60:]
    losses, baseline_losses, correct = [], [], 0
    for m in candidates:
        earlier = [r for r in history if r['date'] < m['date']]
        if len(earlier) < 100:
            continue
        model = fit(earlier, m['date'])
        h, a, _ = goal_rates(model, m['home'], m['away'])
        p, _ = probabilities(h, a)
        base, _ = probabilities(model[1], model[2])
        label = 0 if m['hg'] > m['ag'] else 1 if m['hg'] == m['ag'] else 2
        losses.append(-math.log(max(p[label], 1e-15)))
        baseline_losses.append(-math.log(max(base[label], 1e-15)))
        correct += max(range(3), key=p.__getitem__) == label
    n = len(losses)
    return {'matches': n, 'accuracy': correct / n if n else None,
            'log_loss': sum(losses) / n if n else None,
            'league_baseline_log_loss': sum(baseline_losses) / n if n else None}


def pipeline(leagues=None, seasons=3, days=14, offline=False, progress=print, output=LIVE, model='poisson'):
    if model == 'lineup_rf':
        from lineup_pipeline import run
        return run(leagues=leagues, seasons=seasons, days=days, offline=offline, progress=progress, output=output)
    if model != 'poisson':
        raise ValueError('Unknown prediction model')
    leagues = list(dict.fromkeys(leagues or LEAGUES))
    if not leagues or any(l not in LEAGUES for l in leagues):
        raise ValueError('Select supported league codes: ' + ', '.join(LEAGUES))
    if not 1 <= seasons <= 10 or not 1 <= days <= 90:
        raise ValueError('Seasons must be 1–10 and days 1–90')
    today = datetime.now(timezone.utc).date()
    season_start = today.year if today.month >= 7 else today.year - 1
    sources, history, warnings = [], [], []
    for league in leagues:
        for year in range(season_start - seasons + 1, season_start + 1):
            season = f'{year % 100:02d}{(year + 1) % 100:02d}'
            progress(f'Downloading {LEAGUES[league]} {year}/{year+1}…' if not offline
                     else f'Reading cached {league} {season}…')
            rows, source = download(f'mmz4281/{season}/{league}.csv', output / 'cache', offline, True)
            if not rows or any(r['league'] != league for r in rows):
                raise ValueError(f'Empty or wrong-league result file: {source["url"]}')
            history.extend(rows)
            sources.append(source)
    fixtures, source = download('fixtures.csv', output / 'cache', offline)
    sources.append(source)
    history = deduplicate(history)
    completed = {tuple(m[k] for k in ('league', 'date', 'home', 'away')) for m in history}
    fixtures = [f for f in deduplicate(fixtures) if f['league'] in leagues
                and 0 <= (f['date'] - today).days <= days
                and tuple(f[k] for k in ('league', 'date', 'home', 'away')) not in completed]
    predictions, metrics = [], {}
    for league in leagues:
        progress(f'Fitting and checking {LEAGUES[league]}…')
        matches = [m for m in history if m['league'] == league and m['date'] < today]
        if len(matches) < 100:
            raise ValueError(f'{league}: only {len(matches)} historical matches; need at least 100')
        latest = max(m['date'] for m in matches)
        metrics[league] = {**validate(matches), 'training_matches': len(matches),
                           'latest_result': latest.isoformat()}
        if (today - latest).days > 21:
            warnings.append(f'{LEAGUES[league]}: latest result is {latest}; history may be stale or league on break.')
        model = fit(matches, today)
        for f in fixtures:
            if f['league'] != league:
                continue
            h, a, support = goal_rates(model, f['home'], f['away'])
            p, score = probabilities(h, a)
            predictions.append({**f, 'date': f['date'].isoformat(),
                                'expected_home_goals': round(h, 3), 'expected_away_goals': round(a, 3),
                                'p_home': p[0], 'p_draw': p[1], 'p_away': p[2],
                                'pick': ['Home', 'Draw', 'Away'][max(range(3), key=p.__getitem__)],
                                'modal_score': score, 'limited_history': support < 3})
    predictions.sort(key=lambda p: (p['date'], p['time'], p['league'], p['home']))
    if not predictions:
        warnings.append('No fixtures in the selected window. The provider publishes a limited fixture list; try refreshing later.')
    if offline:
        warnings.append('Offline mode: cached source files may be out of date.')
    report = {'generated_at': datetime.now(timezone.utc).isoformat(), 'as_of': today.isoformat(),
              'model': MODEL, 'leagues': leagues, 'days': days, 'seasons': seasons,
              'warnings': warnings, 'sources': sources, 'validation': metrics, 'predictions': predictions}
    from live_bets import save as save_strategies
    save_strategies(report, output)
    atomic_write(output / 'predictions.json', json.dumps(report, indent=2, allow_nan=False))
    buf = io.StringIO()
    fields = ['league', 'date', 'time', 'home', 'away', 'expected_home_goals', 'expected_away_goals',
              'p_home', 'p_draw', 'p_away', 'pick', 'modal_score', 'limited_history']
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(predictions)
    atomic_write(output / 'predictions.csv', buf.getvalue())
    progress(f'Done: {len(predictions)} fixtures. Saved to {output}')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--leagues', nargs='+', choices=list(LEAGUES), default=list(LEAGUES))
    parser.add_argument('--seasons', type=int, default=3)
    parser.add_argument('--days', type=int, default=14)
    parser.add_argument('--offline', action='store_true', help='Use previously downloaded source files')
    parser.add_argument('--model', choices=['poisson', 'lineup_rf'], default='lineup_rf')
    args = parser.parse_args(argv)
    try:
        report = pipeline(**vars(args))
    except (OSError, ValueError) as exc:
        parser.exit(1, f'Prediction run failed: {exc}\n')
    for warning in report['warnings']:
        print('Warning:', warning)


if __name__ == '__main__':
    main()
