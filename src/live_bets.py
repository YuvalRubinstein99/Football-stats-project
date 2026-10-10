"""Apply the 14 existing backtest rules to saved/live predictions.

Pure-Python implementation, checked against betting_rules.strategy_scores.
All amounts are fractions of ONE total unit, not bankroll recommendations.
"""
import csv
import io
import math
import json
from pathlib import Path
from stake_levels import size_bet

STANDARD = ['Most likely outcome', 'Expected return > 1', 'Log odds weighting',
            'Square-root odds weighting', 'Square-root profit weighting',
            'Legacy draw-to-home weighting', 'Legacy draw-to-home return weighting',
            'Legacy market power weighting', 'Legacy variance score', 'Home win or skip (sqrt profit)']
BINARY = ['Home loses vs not (synthetic double chance)', 'Home loses vs not (sqrt profit)',
          'Home wins vs not (synthetic double chance)', 'Home wins vs not (sqrt profit)']
STRATEGIES = STANDARD + BINARY
SEARCH_CANDIDATE = 'Legacy variance — filtered search candidate'


def integrated_recommendations(match, model_kind='lineup_rf'):
    rows = recommendations(match)
    candidate = next(dict(b) for b in rows if b['strategy']=='Legacy variance score')
    candidate['strategy'] = SEARCH_CANDIDATE
    reason = ''
    if model_kind != 'lineup_rf':
        candidate['action'] = 'Unavailable'
        reason = 'This search candidate requires the saved lineup RF probabilities.'
    elif candidate['action'] == 'Bet' and not (
            4 <= candidate['decimal_odds'] <= 1000 and candidate['expected_net_per_unit'] > .02):
        candidate['action'] = 'Skip'
        reason = 'Selected outcome must have odds 4.00–1000 and model expected return strictly above 2%.'
    if candidate['action'] != 'Bet':
        candidate.update(selection=candidate['action'],probability=None,decimal_odds=None,
                         expected_net_per_unit=None,home_stake=0.,draw_stake=0.,away_stake=0.)
    candidate['note'] = (reason or candidate['note']) + ' Fixed exploratory search candidate: raw lineup RF, selected on 2018–20. Not a proven optimum.'
    return rows + [candidate]


def recommendations(match):
    def empty(name, action, reason):
        return dict(strategy=name, action=action, selection=action, probability=None,
                    decimal_odds=None, expected_net_per_unit=None, away_stake=0., draw_stake=0.,
                    home_stake=0., synthetic=False, note=reason)
    try:
        p = [float(match[k]) for k in ('p_away','p_draw','p_home')]
        o = [float(match[k]) for k in ('B365A','B365D','B365H')]
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in p) or abs(sum(p)-1) > 1e-8:
            raise ValueError('Invalid probabilities')
        if not all(math.isfinite(x) and x > 1 for x in o):
            raise ValueError('Invalid odds')
    except (KeyError, TypeError, ValueError):
        return [empty(n,'Unavailable','Valid probabilities and all three Bet365 prices are required.') for n in STRATEGIES]
    labels = [match['away']+' wins', 'Draw', match['home']+' wins']

    def bet(name, indices, synthetic=False):
        odds = 1 / sum(1/o[i] for i in indices) if synthetic else o[indices[0]]
        stakes = [odds/o[i] if i in indices else 0. for i in range(3)]
        prob = sum(p[i] for i in indices)
        label = ' + '.join(labels[i] for i in indices) if synthetic else labels[indices[0]]
        note = 'Synthetic split stake; not a quoted double-chance price.' if synthetic else ''
        if name.startswith('Legacy'):
            note = 'Legacy heuristic; selection scores are not calibrated probabilities.'
        if odds <= 1:
            note += ' Effective payout is at most the stake, even when correct.'
        return dict(strategy=name, action='Bet', selection=label, probability=prob, decimal_odds=odds,
                    expected_net_per_unit=prob*odds-1, away_stake=stakes[0], draw_stake=stakes[1],
                    home_stake=stakes[2], synthetic=synthetic, note=note)

    market = [1/x for x in o]; total = sum(market); market = [x/total for x in market]
    shifted = [p[0],0.,p[2]+p[1]]
    values = [p, [p[i]*o[i] for i in range(3)], [p[i]*math.log(o[i]) for i in range(3)],
              [p[i]*math.sqrt(o[i]) for i in range(3)], [p[i]*math.sqrt(o[i]-1) for i in range(3)],
              shifted, [shifted[i]*o[i] for i in range(3)], [o[i]**market[i] for i in range(3)],
              [p[i]*(o[i]**2-p[i]*o[i]**2+2*p[i]*o[i]-1) for i in range(3)]]
    result=[]
    for name, score in zip(STANDARD, values):
        chosen=max(range(3),key=score.__getitem__)  # Legacy A/D/H tie ordering.
        threshold=1 if name in ('Expected return > 1','Legacy market power weighting') else 0
        result.append(bet(name,[chosen]) if score[chosen]>threshold else empty(name,'Skip','No outcome clears this rule\'s threshold.'))
    x2 = 1/(1/o[0]+1/o[1])
    home_score=p[2]*math.sqrt(o[2]-1)
    no_home_score=(p[0]+p[1])*math.sqrt(max(x2-1,0))
    result.append(bet(STANDARD[-1],[2]) if home_score>=no_home_score else
                  empty(STANDARD[-1],'Skip','The binary square-root-profit rule prefers away/draw.'))
    for name in BINARY:
        single = 0 if name.startswith('Home loses') else 2
        others = [i for i in range(3) if i!=single]
        hedge = 1/sum(1/o[i] for i in others)
        left, right = p[single], sum(p[i] for i in others)
        if 'sqrt profit' in name:
            # The old home-win evaluator rejects synthetic odds below 1.
            if single==2 and hedge<1:
                result.append(empty(name,'Unavailable','Original home-win sqrt rule requires synthetic payout >= 1.'))
                continue
            left*=math.sqrt(o[single]-1)
            right*=math.sqrt(max(hedge-1,0))
        result.append(bet(name,[single]) if left>=right else bet(name,others,True))
    return result


def enrich(report):
    for match in report['predictions']:
        match['strategies']=integrated_recommendations(match,report.get('model_kind','lineup_rf'))
        for bet in match['strategies']:
            bet.update(size_bet(bet))
    report['strategy_names']=STRATEGIES + [SEARCH_CANDIDATE]
    return report


def export_csv(report):
    buf=io.StringIO()
    fields=['date','league','home','away','strategy','action','selection','probability','decimal_odds',
            'expected_net_per_unit','home_stake','draw_stake','away_stake','synthetic','note',
            'stake_level','kelly_score','stake_dollars','staking_action','staking_reason',
            'home_dollars','draw_dollars','away_dollars']
    writer=csv.DictWriter(buf,fieldnames=fields)
    writer.writeheader()
    for match in report['predictions']:
        for row in match['strategies']:
            writer.writerow({**{k:match[k] for k in ('date','league','home','away')}, **row})
    return buf.getvalue()


def save(report, output):
    from live_predictions import atomic_write
    enrich(report)
    atomic_write(output/'strategy_recommendations.csv',export_csv(report))


if __name__=='__main__':
    from live_predictions import LIVE, atomic_write
    path=LIVE/'predictions.json'
    report=json.loads(path.read_text(encoding='utf-8'))
    save(report,LIVE)
    atomic_write(path,json.dumps(report,indent=2,allow_nan=False))
    print(f'Added {len(report["strategy_names"])} strategies for {len(report["predictions"])} saved predictions.')
