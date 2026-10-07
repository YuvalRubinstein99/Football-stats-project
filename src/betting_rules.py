"""Original nine fixed decision rules, shared by training and report rebuilds."""
import numpy as np


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
    }
