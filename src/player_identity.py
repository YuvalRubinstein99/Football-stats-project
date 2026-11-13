"""Conservative season-local identity matching, independent of legacy ratings."""
import re
from collections import defaultdict
from functools import lru_cache
from classifier_features import club
from lineup_sources import normalize

CLUB_ALIASES = {
    'Athletic Club de Bilbao': 'Athletic Club', 'Levante Unión Deportiva': 'Levante',
    'RCD Espanyol de Barcelona': 'Espanyol', 'Venezia FC': 'Venezia', 'Cádiz CF': 'Cádiz',
    'Torino F.C.': 'Torino', 'VfL Bochum 1848': 'Bochum', 'U.S. Sassuolo Calcio': 'Sassuolo',
    'US Salernitana 1919': 'Salernitana', 'Udinese Calcio': 'Udinese',
    'Sport-Club Freiburg': 'Freiburg', 'U.C. Sampdoria': 'Sampdoria',
    'AC Chievo Verona': 'Chievo', 'AC Cesena': 'Cesena', 'ACF Fiorentina': 'Fiorentina',
    'Bologna FC 1909': 'Bologna', 'Cagliari Calcio': 'Cagliari', 'Empoli FC': 'Empoli',
    'Genoa CFC': 'Genoa', 'Hellas Verona FC': 'Hellas Verona', 'Parma Calcio 1913': 'Parma',
    'US Città di Palermo': 'Palermo', 'US Sassuolo': 'Sassuolo', 'Spezia Calcio': 'Spezia',
    'FC Ingolstadt 04': 'Ingolstadt', 'SV Darmstadt 98': 'Darmstadt',
    'SD Eibar': 'Eibar', 'SD Huesca': 'Huesca', 'UD Las Palmas': 'Las Palmas',
    'Real Valladolid CF': 'Valladolid', 'Piemonte Calcio': 'Juventus',
}
CLUB_KEYS = {normalize(k): club(v) for k, v in CLUB_ALIASES.items()}


def canonical_club(value):
    name = str(value)
    return CLUB_KEYS.get(normalize(name), club(name))


def stable_id(player):
    value = player.get('sofifa_id')
    try:
        if value is not None and int(value) > 0:
            return str(int(value))
    except (TypeError, ValueError, OverflowError):
        pass
    match = re.search(r'/players/(\d+)/(\d+)/', str(player.get('Image Link', '')))
    if match:
        return str(int(match[1] + match[2]))
    return None


def name_key(value):
    return normalize(value) if isinstance(value, str) and '\ufffd' not in value else ''


class IdentityIndex:
    def __init__(self, records, earlier_names=None):
        self.records = records
        self.names, self.clubs = defaultdict(set), defaultdict(set)
        self.tokens, self.by_id = {}, {}
        for i, p in enumerate(records):
            pid = stable_id(p)
            if pid:
                if pid in self.by_id:
                    raise ValueError('Duplicate FIFA ID in snapshot')
                self.by_id[pid] = i
            self.clubs[canonical_club(p.get('club_name'))].add(i)
            self.tokens[i] = []
            for field in ['short_name', 'long_name']:
                key = name_key(p.get(field))
                if key:
                    self.names[key].add(i)
                    self.tokens[i].append(key.split())
        self.historical = defaultdict(set)
        for name, ids in (earlier_names or {}).items():
            for pid in ids:
                if pid in self.by_id:
                    self.historical[name].add(self.by_id[pid])

    @lru_cache(maxsize=None)
    def match(self, name, team):
        key = name_key(name)
        if not key:
            return None, 'missing_or_damaged_name'
        local_ids = self.clubs.get(canonical_club(team), set())
        exact = self.names.get(key, set())
        local = exact & local_ids
        if len(local) == 1:
            return next(iter(local)), 'exact_name_club'
        if len(exact) == 1:
            return next(iter(exact)), 'unique_exact_name'
        if exact:
            return None, 'ambiguous_exact_name'
        earlier = self.historical.get(key, set())
        local = earlier & local_ids
        if len(local) == 1:
            return next(iter(local)), 'earlier_name_same_id_club'
        if len(earlier) == 1:
            return next(iter(earlier)), 'earlier_name_same_id'
        tokens = key.split()
        if len(tokens) >= 2:
            candidates = {i for i in local_ids if any(set(tokens) <= set(t) for t in self.tokens[i])}
            if len(candidates) == 1:
                return next(iter(candidates)), 'unique_tokens_club'
            # Match explicit initials or substantial abbreviated given names.
            # At least one full surname must agree; arbitrary first-letter
            # matching of two different full given names is not allowed.
            candidates = set()
            for i in local_ids:
                for target in self.tokens[i]:
                    if len(target) < 2 or len(tokens[-1]) < 4 or tokens[-1] not in target[1:]:
                        continue
                    first = tokens[0]
                    prefix = (len(first) == 1 or len(first) >= 4) and target[0].startswith(first)
                    if prefix and all(t in target for t in tokens[1:]):
                        candidates.add(i)
            if len(candidates) == 1:
                return next(iter(candidates)), 'given_prefix_full_surname_club'
        return None, 'unresolved'
