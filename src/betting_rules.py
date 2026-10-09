"""Betting decision rules shared by training and report rebuilds."""
import numpy as np

HOME_OR_SKIP = 'Home win or skip (sqrt profit)'


def home_or_skip_scores(p, odds):
    """Keep home selections from the binary square-root-profit rule; skip others."""
    no_win_odds = 1 / (1 / odds[:, 0] + 1 / odds[:, 1])
    home_value = p[:, 2] * np.sqrt(odds[:, 2] - 1)
    no_win_value = (p[:, 0] + p[:, 1]) * np.sqrt(np.maximum(no_win_odds - 1, 0))
    scores = np.full_like(p, -np.inf, dtype=float)
    # The common evaluator bets iff the selected score is > 0.
    # Home wins ties, matching the existing binary rule.
    scores[:, 2] = (home_value >= no_win_value).astype(float)
    return scores


def strategy_scores(p, odds, market):
    shifted = p.copy()
    shifted[:, 2] += shifted[:, 1]
    shifted[:, 1] = 0
    return {
        'Most likely outcome': (p, 0),
        'Expected return > 1': (p * odds, 1),
        'Log odds weighting': (p * np.log(odds), 0),
        'Square-root odds weighting': (p * np.sqrt(odds), 0),
        'Square-root profit weighting': (p * np.sqrt(odds - 1), 0),
        'Legacy draw-to-home weighting': (shifted, 0),
        'Legacy draw-to-home return weighting': (shifted * odds, 0),
        'Legacy market power weighting': (np.power(odds, market), 1),
        'Legacy variance score': (p * (odds**2 - p * odds**2 + 2*p*odds - 1), 0),
        HOME_OR_SKIP: (home_or_skip_scores(p, odds), 0),
    }
