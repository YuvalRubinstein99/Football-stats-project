"""Fixed dollar tiers ranked by positive Kelly score; no outcome-based tuning."""
import math

AMOUNTS = [1, 2, 3, 4, 5]


def size_bet(bet):
    """Preserve the original selection, adding a separate staking gate.

    Kelly is a ranking score here, not a fraction of a supplied bankroll.
    Synthetic selections are equal-payout dutches, so the same formula applies.
    """
    odds = bet.get('decimal_odds')
    ev = bet.get('expected_net_per_unit')
    score = ev / (odds - 1) if (bet['action'] == 'Bet' and odds and odds > 1
                                              and ev is not None and ev > 0) else 0.
    level = min(5, max(1, math.ceil(round(score * 100, 10)))) if score > 0 else 0
    stake = AMOUNTS[level - 1] if level else 0
    return dict(stake_level=level, kelly_score=score, stake_dollars=stake,
                staking_action='Bet' if level else ('Unavailable' if bet['action']=='Unavailable' else 'Skip'),
                staking_reason='' if level else (bet['note'] if bet['action']!='Bet' else 'No positive model edge; five-level staking skips this selection.'),
                home_dollars=stake*bet['home_stake'], draw_dollars=stake*bet['draw_stake'],
                away_dollars=stake*bet['away_stake'])
