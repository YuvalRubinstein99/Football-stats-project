"""Preferred-position unit strengths from already matched starting players.

Positions describe the FIFA snapshot, not observed match-day formations.
Each player belongs to exactly one unit, using the first listed position.
"""
import re
import numpy as np

POSITION_UNITS = {
    **dict.fromkeys(['ST', 'CF', 'LF', 'RF', 'LW', 'RW', 'LS', 'RS'], 'attack'),
    **dict.fromkeys(['CAM', 'CM', 'CDM', 'LM', 'RM', 'LAM', 'RAM',
                     'LCM', 'RCM', 'LDM', 'RDM'], 'midfield'),
    **dict.fromkeys(['CB', 'LB', 'RB', 'LWB', 'RWB', 'LCB', 'RCB'], 'defense'),
    'GK': 'goalkeeper',
}
UNIT_ATTRIBUTES = {
    'attack': ['overall', 'shooting', 'dribbling', 'pace', 'physicality'],
    'midfield': ['overall', 'passing', 'dribbling', 'defending', 'physicality'],
    'defense': ['overall', 'defending', 'pace', 'physicality'],
    'goalkeeper': ['overall'],
}
MIN_UNIT_PLAYERS = dict(attack=1, midfield=2, defense=2, goalkeeper=1)


def primary_position(value):
    if not isinstance(value, str) or not value.strip():
        return ''
    return re.split(r'[,;/\s]+', value.strip().upper())[0]


def unit(value):
    return POSITION_UNITS.get(primary_position(value), 'unknown')


def valid_rating(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return np.isfinite(value) and 1 <= value <= 99


def aggregate_units(players, side, min_matched=8):
    """Keep counts even when insufficient identity coverage suppresses means."""
    output = {f'{side}_position_matched_count': len(players),
              f'{side}_position_unknown_count': sum(unit(p.get('player_positions')) == 'unknown'
                                                    for p in players)}
    for group, attributes in UNIT_ATTRIBUTES.items():
        members = [p for p in players if unit(p.get('player_positions')) == group]
        output[f'{side}_{group}_count'] = len(members)
        for attribute in attributes:
            values = [float(p[attribute]) for p in members if valid_rating(p.get(attribute))]
            output[f'{side}_{group}_{attribute}_count'] = len(values)
            output[f'{side}_{group}_{attribute}_mean'] = (
                float(np.mean(values)) if len(players) >= min_matched
                and len(values) >= MIN_UNIT_PLAYERS[group] else np.nan)
    return output


def matchup_features(features):
    output = {}
    for side, opponent in [('home', 'away'), ('away', 'home')]:
        for attack, defense in [('overall', 'overall'), ('shooting', 'defending'),
                                ('dribbling', 'defending'), ('pace', 'pace'),
                                ('physicality', 'physicality')]:
            name = f'{side}_attack_{attack}_vs_defense_{defense}'
            output[name] = (features[f'{side}_attack_{attack}_mean']
                            - features[f'{opponent}_defense_{defense}_mean'])
        output[f'{side}_attack_overall_vs_goalkeeper_overall'] = (
            features[f'{side}_attack_overall_mean'] - features[f'{opponent}_goalkeeper_overall_mean'])
    for attribute in UNIT_ATTRIBUTES['midfield']:
        output[f'midfield_{attribute}_advantage'] = (
            features[f'home_midfield_{attribute}_mean'] - features[f'away_midfield_{attribute}_mean'])
    output['goalkeeper_overall_advantage'] = (
        features['home_goalkeeper_overall_mean'] - features['away_goalkeeper_overall_mean'])
    return output


def feature_columns():
    empty = {**aggregate_units([], 'home'), **aggregate_units([], 'away')}
    return list(empty) + list(matchup_features(empty))
