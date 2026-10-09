"""Public lineup and EA rating adapters; no guessed player ratings."""
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import unicodedata
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
from live_predictions import atomic_write

EA_URL = 'https://www.ea.com/games/ea-sports-fc/ratings'
RW_URL = 'https://www.rotowire.com/soccer/lineups.php?league='
RW_LEAGUES = {'E0': 'EPL', 'SP1': 'LIGA', 'D1': 'BUND', 'I1': 'SERI', 'F1': 'FRAN'}

# Explicit aliases across Football-Data, RotoWire, EA, and historical training names.
TEAM_GROUPS = [
    ['AFC Bournemouth', 'Bournemouth'], ['Brighton & Hove Albion', 'Brighton', 'Brighton and Hove Albion'],
    ['Manchester City', 'Man City'], ['Manchester United', 'Man United', 'Manchester Utd', 'Man Utd'],
    ['Leeds United', 'Leeds'], ['Leicester City', 'Leicester'], ['Newcastle United', 'Newcastle', 'Newcastle Utd'],
    ['Norwich City', 'Norwich'], ['Nottingham Forest', "Nott'm Forest", "Nott'm Forest FC", "Nott'm Forest"],
    ['Tottenham Hotspur', 'Tottenham', 'Spurs'], ['West Ham United', 'West Ham'],
    ['Wolverhampton Wanderers', 'Wolves'], ['Hull City', 'Hull'], ['Coventry City', 'Coventry'],
    ['Ipswich Town', 'Ipswich'], ['Sheffield United', 'Sheffield Utd'],
    ['Athletic Club', 'Ath Bilbao', 'Athletic Bilbao', 'Athletic'], ['Atlético Madrid', 'Ath Madrid', 'Atlético de Madrid', 'Atletico Madrid'],
    ['Betis', 'Real Betis'], ['Celta Vigo', 'Celta', 'RC Celta', 'RC Celta de Vigo'],
    ['Espanyol', 'Espanol', 'RCD Espanyol'], ['Alavés', 'Alaves', 'Deportivo Alavés', 'D. Alavés'],
    ['Barcelona', 'FC Barcelona'], ['Real Sociedad', 'Sociedad'], ['Rayo Vallecano', 'Vallecano'],
    ['La Coruña', 'La Coruna', 'Deportivo La Coruña', 'RC Deportivo', 'Deportivo'], ['Málaga', 'Malaga', 'Málaga CF'],
    ['Valencia', 'Valencia CF'], ['Elche', 'Elche CF'], ['Osasuna', 'CA Osasuna'],
    ['Sevilla', 'Sevilla FC'], ['Villarreal', 'Villarreal CF'], ['Levante', 'Levante UD'], ['Getafe', 'Getafe CF'],
    ['Santander', 'Racing Santander', 'Racing de Santander', 'R. Racing Club'],
    ['Dortmund', 'Borussia Dortmund', 'Borussia Dortmund'], ['Bayern Munich', 'Bayern München', 'FC Bayern München'],
    ['Eint Frankfurt', 'Ein Frankfurt', 'Eintracht Frankfurt', 'Frankfurt'], ['Gladbach', "M'gladbach", 'Borussia Mönchengladbach', 'Mönchengladbach'],
    ['Köln', 'FC Koln', '1. FC Köln', 'FC Köln', 'Koln'], ['Hamburger SV', 'Hamburg', 'Hamburger SV'],
    ['Mainz 05', 'Mainz', '1. FSV Mainz 05', 'FSV Mainz 05'], ['Paderborn 07', 'Paderborn', 'SC Paderborn 07', 'SC Paderborn'],
    ['Leverkusen', 'Bayer Leverkusen', 'Bayer 04 Leverkusen'], ['Stuttgart', 'VfB Stuttgart'],
    ['Freiburg', 'SC Freiburg'], ['Hoffenheim', 'TSG Hoffenheim', 'TSG 1899 Hoffenheim', '1899 Hoffenheim'],
    ['Augsburg', 'FC Augsburg'], ['Schalke 04', 'FC Schalke 04'], ['Werder Bremen', 'SV Werder Bremen'],
    ['Union Berlin', '1. FC Union Berlin'], ['Elversberg', 'SV Elversberg', 'SV 07 Elversberg'],
    ['Inter', 'Internazionale', 'Inter Milan', 'Lombardia FC'], ['Milan', 'AC Milan', 'Milano FC'],
    ['Roma', 'AS Roma'], ['Hellas Verona', 'Verona'], ['Napoli', 'SSC Napoli'], ['Lazio', 'Latium', 'SS Lazio'],
    ['Atalanta', 'Bergamo Calcio'],
    ['Paris SG', 'Paris Saint-Germain', 'Paris Saint Germain', 'PSG'], ['Lyon', 'Olympique Lyonnais', 'OL'],
    ['Marseille', 'Olympique de Marseille', 'OM'], ['Lille', 'LOSC Lille'], ['Monaco', 'AS Monaco'],
    ['Nice', 'OGC Nice'], ['Brest', 'Stade Brestois 29'], ['Rennes', 'Stade Rennais FC', 'Stade Rennais'],
    ['Lens', 'RC Lens'], ['Strasbourg', 'RC Strasbourg Alsace'], ['Auxerre', 'AJ Auxerre'],
    ['Angers', 'Angers SCO'], ['Lorient', 'FC Lorient'], ['Troyes', 'ESTAC Troyes'],
    ['Le Havre', 'Le Havre AC', 'Havre AC'], ['Toulouse', 'Toulouse FC'], ['Le Mans', 'Le Mans FC'],
]


def normalize(name):
    name = name.translate(str.maketrans({'ø':'o', 'Ø':'O', 'ł':'l', 'Ł':'L', 'ß':'ss',
                                       'ı':'i', 'đ':'dj', 'Đ':'Dj'}))
    name = ''.join(c for c in unicodedata.normalize('NFKD', name) if not unicodedata.combining(c))
    return ' '.join(re.sub(r'[^a-z0-9]+', ' ', name.lower()).split())


ALIASES = {normalize(alias): normalize(group[0]) for group in TEAM_GROUPS for alias in group}
# Reviewed spellings from the downloaded RotoWire and EA pages, not fuzzy matches.
PLAYER_NAMES = {'yehor yarmolyuk': 'Yehor Yarmoliuk', 'oliver mcburnie': 'Oli McBurnie',
                'savio': 'Savinho'}


def team_key(name):
    key = normalize(name)
    return ALIASES.get(key, key)


def fetch(url, cache, offline=False, ttl=0):
    """Bounded, cached public GET. No automatic stale fallback on HTTP failures."""
    cache = Path(cache)
    path = cache / (hashlib.sha256(url.encode()).hexdigest() + '.html')
    cached = offline or (path.exists() and time.time() - path.stat().st_mtime < ttl)
    if cached:
        payload = path.read_bytes()
    else:
        req = Request(url, headers={'User-Agent': 'FootballStatsProject/1.0'})
        with urlopen(req, timeout=30) as response:
            payload = response.read(8_000_001)
        if len(payload) > 8_000_000:
            raise ValueError('Source exceeds 8 MB: ' + url)
        # These providers return HTML; error/blocked pages fail during parsing.
        atomic_write(path, payload.decode('utf-8'))
        time.sleep(.2)
    return payload.decode('utf-8'), {'url': url, 'cached': cached,
        'fetched_at': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        'sha256': hashlib.sha256(payload).hexdigest()}


def parse_lineups(html, league, today):
    soup = BeautifulSoup(html, 'html.parser')
    matches = []
    for card in soup.select('.lineup.is-soccer'):
        names = [card.select_one('.lineup__mteam.' + side) for side in ('is-home', 'is-visit')]
        time_node = card.select_one('.lineup__time b')
        if not all(names) or not time_node:
            continue
        month_day = time_node.get_text(' ', strip=True)
        dates = []
        for year in (today.year-1, today.year, today.year+1):
            try:
                dates.append(datetime.strptime(f'{month_day} {year}', '%B %d %Y').date())
            except ValueError:
                pass
        if not dates:
            raise ValueError('Unrecognized lineup date: ' + month_day)
        match_date = min(dates, key=lambda d: abs((d-today).days))
        item = {'league': league, 'date': match_date.isoformat(),
                'home': names[0].get_text(' ', strip=True), 'away': names[1].get_text(' ', strip=True)}
        clock = re.search(r'(\d{1,2}:\d{2}\s*[AP]M)\s+ET', time_node.parent.get_text(' ', strip=True))
        if clock:
            local = datetime.strptime(f'{match_date.isoformat()} {clock.group(1)}', '%Y-%m-%d %I:%M %p')
            item['kickoff_utc'] = local.replace(tzinfo=ZoneInfo('America/New_York')).astimezone(timezone.utc).isoformat()
        for side, css in [('home', 'is-home'), ('away', 'is-visit')]:
            listing = card.select_one('.lineup__list.' + css)
            players, status = [], 'unavailable'
            if listing:
                node = listing.select_one('.lineup__status')
                if node:
                    text = node.get_text(' ', strip=True).lower()
                    status = 'confirmed' if 'confirmed' in text else 'predicted' if ('predicted' in text or 'expected' in text) else 'unavailable'
                for child in listing.find_all('li', recursive=False):
                    if 'lineup__title' in child.get('class', []):
                        break  # Do not treat the injury list as starters.
                    if 'lineup__player' not in child.get('class', []):
                        continue
                    link = child.find('a', href=re.compile('/soccer/player/'))
                    if link:
                        players.append({'name': link.get('title') or link.get_text(' ', strip=True),
                                        'source_id': link['href'].rsplit('-', 1)[-1],
                                        'injury_flag': (child.select_one('.lineup__inj').get_text(strip=True)
                                                        if child.select_one('.lineup__inj') else '')})
            item[side + '_players'] = players
            item[side + '_status'] = status
        matches.append(item)
    if not matches and not re.search(r'no (games|matches|lineups)', soup.get_text(' '), re.I):
        raise ValueError('No recognizable RotoWire lineup cards; source may have changed or blocked access')
    return matches


def parse_ea(html):
    soup = BeautifulSoup(html, 'html.parser')
    node = soup.select_one('script#__NEXT_DATA__')
    if not node:
        raise ValueError('EA ratings page has no embedded data; source may have changed')
    p = json.loads(node.string)['props']['pageProps']
    if not isinstance(p.get('ratingDetails', {}).get('items'), list):
        raise ValueError('EA ratings schema changed')
    return p


class Ratings:
    def __init__(self, cache, offline=False):
        self.cache, self.offline = cache, offline
        html, source = fetch(EA_URL, cache, offline, ttl=7*86400)
        p = parse_ea(html)
        self.edition = p['gameDetails']['slug']
        self.sources = [source]
        self.teams = {}
        self.loaded = {}
        for group in p['ratingsFilters']['teamGroups']:
            if group.get('gender', {}).get('id') != 0:
                continue
            for team in group['teams']:
                self.teams.setdefault(team_key(team['label']), {})[team['id']] = team

    def squad(self, team):
        key = team_key(team)
        if key in self.loaded:
            return self.loaded[key]
        options = self.teams.get(key, {})
        if len(options) != 1:
            raise ValueError(f'EA team match missing/ambiguous: {team}; add an explicit team alias')
        team_id = next(iter(options))
        html, source = fetch(f'{EA_URL}?team={team_id}', self.cache, self.offline, ttl=7*86400)
        p = parse_ea(html)
        if p['gameDetails']['slug'] != self.edition:
            raise ValueError('EA edition changed during collection; refresh ratings cache')
        detail = p['ratingDetails']
        if len(detail['items']) < detail['totalItems']:
            raise ValueError(f'EA squad for {team} is paginated; refusing partial ratings')
        rows = []
        for r in detail['items']:
            if r['team']['id'] != team_id:
                raise ValueError('EA ignored the requested team filter')
            if r.get('gender', {}).get('id') != 0:
                continue
            rows.append({'ea_id': str(r['id']), 'name': ' '.join(filter(None, [r.get('firstName'), r.get('lastName')])),
                         'common_name': r.get('commonName') or '', 'overall': r['overallRating'],
                         'club': r['team']['label'], 'edition': self.edition})
        self.sources.append(source)
        self.loaded[key] = rows
        return rows

    def search_player(self, player):
        # EA launch rosters may lag transfers. Search by full name, requiring a
        # unique exact name match; never substitute a similar name globally.
        search_name = PLAYER_NAMES.get(normalize(player['name']), player['name'])
        html, source = fetch(EA_URL + '?' + urlencode({'search': search_name}),
                             self.cache, self.offline, ttl=7*86400)
        p = parse_ea(html)
        if p['gameDetails']['slug'] != self.edition:
            raise ValueError('EA edition mismatch in player search')
        details = p['ratingDetails']
        if details['totalItems'] > len(details['items']):
            raise ValueError('Player search is incomplete: ' + player['name'])
        key = normalize(search_name)
        rows = []
        for r in details['items']:
            name = ' '.join(filter(None, [r.get('firstName'), r.get('lastName')]))
            common = r.get('commonName') or ''
            if r.get('gender', {}).get('id') == 0 and key in (normalize(name), normalize(common)):
                rows.append({'ea_id': str(r['id']), 'name': name, 'common_name': common,
                             'overall': r['overallRating'], 'club': r['team']['label'], 'edition': self.edition})
        self.sources.append(source)
        if len(rows) != 1:
            raise ValueError('No unique exact EA player search match: ' + player['name'])
        result = match_player(player, rows)
        result['match_method'] = 'global_exact_name_club_mismatch'
        return result


def match_player(player, squad, overrides=None):
    """Exact normalized names, or unique full-name tokens within the same club.

    No fuzzy maximum or missing-rating imputation. Global exact search is a
    separate, explicitly flagged step.
    """
    overrides = overrides or {}
    explicit = overrides.get(str(player['source_id']))
    if explicit:
        matches = [r for r in squad if r['ea_id'] == str(explicit)]
        method = 'explicit_id'
    else:
        key = normalize(PLAYER_NAMES.get(normalize(player['name']), player['name']))
        matches = [r for r in squad if key in (normalize(r['name']), normalize(r['common_name']))]
        method = 'reviewed_name_alias' if normalize(player['name']) in PLAYER_NAMES else 'exact_name'
        if not matches and len(key.split()) >= 2:
            tokens = key.split()
            matches = [r for r in squad if any(set(tokens).issubset(set(normalize(n).split()))
                                              for n in (r['name'], r['common_name']))]
            method = 'unique_name_tokens'
    if len(matches) != 1:
        raise ValueError(f'Rating {"ambiguous" if matches else "missing"}: {player["name"]} (RotoWire {player["source_id"]})')
    r = matches[0]
    if not isinstance(r['overall'], (int, float)) or not 1 <= r['overall'] <= 99:
        raise ValueError('Invalid EA overall rating: ' + player['name'])
    return {**player, **r, 'lineup_name': player['name'], 'match_method': method}
