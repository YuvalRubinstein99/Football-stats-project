"""Full-time total goals from the two independent Poisson goal forecasts."""
import math


def over_under_25(home_goals, away_goals, over_odds=None, under_odds=None):
    mean = max(home_goals, 1e-8) + max(away_goals, 1e-8)
    under = math.exp(-mean) * (1 + mean + mean * mean / 2)
    market = {}
    for side, probability, price in [('over', 1 - under, over_odds), ('under', under, under_odds)]:
        available = price is not None and math.isfinite(price) and price > 1
        market[side] = dict(probability=probability, odds=price if available else None,
                            edge=probability * price - 1 if available else None)
    return market
