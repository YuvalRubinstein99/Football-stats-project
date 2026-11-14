"""Home goals minus away goals, including split-stake Asian handicap settlement."""
import numpy as np
from scipy.stats import skellam


def margin_bands(home, away):
    home, away = np.maximum(home, 1e-8), np.maximum(away, 1e-8)
    return {'away_2plus': skellam.cdf(-2, home, away),
            'away_1': skellam.pmf(-1, home, away), 'draw': skellam.pmf(0, home, away),
            'home_1': skellam.pmf(1, home, away), 'home_2plus': skellam.sf(1, home, away)}


def handicap_legs(line):
    # A quarter line splits evenly across its two adjacent half-goal lines.
    return np.floor(2 * line) / 2, np.ceil(2 * line) / 2


def handicap_edges(home, away, line, home_odds, away_odds):
    home, away = np.maximum(home, 1e-8), np.maximum(away, 1e-8)
    home_win = away_win = 0.
    for leg in handicap_legs(line):
        home_win += .5 * skellam.sf(np.floor(-leg), home, away)
        away_win += .5 * skellam.cdf(np.ceil(-leg) - 1, home, away)
    return home_win * (home_odds - 1) - away_win, away_win * (away_odds - 1) - home_win


def handicap_net(margin, line, odds):
    net = 0.
    for leg in handicap_legs(line):
        adjusted = margin + leg
        net += .5 * np.where(adjusted > 0, odds - 1, np.where(adjusted < 0, -1., 0.))
    return net
