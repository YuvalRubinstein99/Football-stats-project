"""User-controlled filters, applied identically to live bets and historical ledgers."""
import csv
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from live_bets import STRATEGIES, recommendations

ROOT=Path(__file__).resolve().parents[1]


def number(value,low,high):
    if isinstance(value,bool):raise ValueError('Numeric value required')
    n=float(value)
    if not math.isfinite(n) or not low<=n<=high:raise ValueError(f'Number must be between {low} and {high}')
    return n


def settings(raw):
    strategy=raw.get('strategy')
    if strategy not in STRATEGIES:raise ValueError('Choose an original strategy')
    outcome=raw.get('outcome','all')
    if outcome not in ('all','home','draw','away'):raise ValueError('Invalid outcome filter')
    source=raw.get('source','bet365')
    if source not in ('bet365','winner','polymarket'):raise ValueError('Invalid odds source')
    amounts=raw.get('amounts',[1,2,3,4,5])
    if not isinstance(amounts,list) or len(amounts)!=5:raise ValueError('Five stake amounts required')
    amounts=[number(x,.01,100000) for x in amounts]
    if any(abs(x*100-round(x*100))>1e-6 for x in amounts):raise ValueError('Stake amounts must be whole cents')
    if amounts!=sorted(amounts):raise ValueError('Stake amounts must ascend')
    result=dict(strategy=strategy,outcome=outcome,source=source,
        min_edge=number(raw.get('min_edge',.02),0,10),min_odds=number(raw.get('min_odds',4),1.000001,1000),
        max_odds=number(raw.get('max_odds',1000),1.000001,1000),
        width=number(raw.get('width',.01),.0001,1),amounts=amounts,sizing=raw.get('sizing','levels'))
    if result['sizing'] not in ('flat','levels'):raise ValueError('Invalid sizing')
    if result['min_odds']>result['max_odds']:raise ValueError('Minimum odds exceed maximum')
    return result


def gate(odds,edge,outcome,config):
    if odds is None or edge is None:return 0,'Missing prices or no original strategy bet'
    if not config['min_odds']<=odds<=config['max_odds']:return 0,'Outside odds range'
    if edge<=config['min_edge']:return 0,'Expected return does not exceed minimum'
    if config['outcome']!='all' and outcome!=config['outcome']:return 0,'Outcome filter excludes original selection'
    level=1 if config['sizing']=='flat' else min(5,max(1,math.ceil(round(edge/(odds-1)/config['width'],10))))
    return level,''


def evaluate_match(match,config,odds=None):
    row=dict(match)
    if odds is not None:
        row.update(B365H=odds.get('home'),B365D=odds.get('draw'),B365A=odds.get('away'))
    bet=next(b for b in recommendations(row) if b['strategy']==config['strategy'])
    outcome='combined' if bet['synthetic'] else next((s for s in ('home','draw','away') if bet[s+'_stake']>0),'none')
    level,reason=gate(bet['decimal_odds'],bet['expected_net_per_unit'],outcome,config)
    amount=config['amounts'][level-1] if level else 0
    return dict(bet,level=level,amount=amount,reason=reason,
                decision='Unavailable' if bet['action']=='Unavailable' else 'Bet' if level else 'Skip',
                **{s+'_amount':amount*bet[s+'_stake'] for s in ('home','draw','away')})


@lru_cache(maxsize=2)
def load_ledger(path,mtime):
    rows=defaultdict(list)
    with open(path,encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f)
        if 'decimal_odds' not in reader.fieldnames:raise ValueError('Rebuild the stake backtest for manual filters')
        for r in reader:
            if r['strategy'] not in STRATEGIES or r['split']!='Test':continue
            rows[r['strategy']].append((r['split'],r['season'],int(r['week']),float(r['decimal_odds']) if r['decimal_odds'] else None,
                float(r['expected_net_per_unit']) if r['expected_net_per_unit'] else None,r['selected_outcome'],float(r['unit_return'])))
    return rows


def historical(config):
    if config['source']!='bet365':
        return dict(rows=[],curves=[],note='No historical Winner/Polymarket quotes are available. Bet365 history is not reused for this source.')
    path=ROOT/'results/stake_level_bet_ledger.csv'
    rows=load_ledger(str(path),path.stat().st_mtime_ns)[config['strategy']]
    totals={};weekly=defaultdict(float)
    for split,season,week,odds,edge,outcome,ret in rows:
        level,_=gate(odds,edge,outcome,config);stake=config['amounts'][level-1] if level else 0
        profit=stake*(ret-1)
        weekly[split,season,week]+=profit
        for group in (season,'All seasons'):
            key=split,group
            if key not in totals:totals[key]=dict(split=split,season=group,bets=0,staked=0.,gross=0.,net=0.)
            t=totals[key];t['bets']+=int(level>0);t['staked']+=stake;t['gross']+=stake*ret;t['net']+=profit
    curves=[]
    for split in ('Test',):
        points=[dict(label='Start',net=0.)];net=0
        for (sp,season,week),profit in sorted(weekly.items()):
            if sp==split:net+=profit;points.append(dict(label=f'{season} W{week}',net=net))
        curves.append(dict(split=split,points=points))
    for t in totals.values():t['roi']=t['net']/t['staked'] if t['staked'] else None
    return dict(rows=list(totals.values()),curves=curves,
        note='Historical Bet365 prices and the saved lineup RF only, regardless of the current live model choice. Training-period results are unavailable pending a chronological backtest. Changes to filters use already-inspected historical data. Each curve starts at $0; points aggregate season/matchweek profit. No bankroll limits, compounding or fees.')


def evaluate(report,raw):
    from odds_sources import quote_for,fixture_key,check_execution
    config=settings(raw);quotes=raw.get('quotes',{})
    if not isinstance(quotes,dict):raise ValueError('Invalid quotes')
    results=[]
    for match in report['predictions']:
        meta=dict(source='Saved Bet365',fetched_at=report['generated_at'],currency='USD')
        try:
            odds=None
            if config['source']!='bet365':odds,meta=quote_for(config['source'],quotes.get(fixture_key(match)),match)
            bet=check_execution(evaluate_match(match,config,odds),meta)
        except (ValueError,TypeError,KeyError) as exc:
            bet=dict(decision='Unavailable',reason=str(exc),selection='—',decimal_odds=None,expected_net_per_unit=None,
                     probability=None,amount=0,level=0,home_amount=0,draw_amount=0,away_amount=0)
            meta=dict(source=config['source'],currency='ILS' if config['source']=='winner' else 'USD',fetched_at=None)
        results.append(dict(date=match['date'],home=match['home'],away=match['away'],bet=bet,quote=meta))
    return dict(settings=config,predictions=results,history=historical(config),
        note='Uses probabilities and Bet365 odds from the saved prediction run. Filters and stakes change selections and profit accounting; they do not retrain the model. Amounts are in dollars. Historical comparisons use the saved lineup RF.')
