"""H/A-only historical rules for a home-win binary model.

The away score uses P(no home win), not P(away win), and is explicitly a proxy.
Draws are retained and settle as losses. Home-only edge rules use genuine P(home).
"""
import numpy as np


def selections(p_home, odds):
    ph,o=np.asarray(p_home,float),np.asarray(odds,float)
    if ph.ndim!=1 or o.shape!=(len(ph),3) or not np.isfinite(ph).all() or ((ph<0)|(ph>1)).any():
        raise ValueError('Invalid home probabilities or odds shape')
    if not np.isfinite(o).all() or (o<=1).any():
        raise ValueError('Decimal H/D/A odds must be finite and exceed one')
    p=np.column_stack([ph,1-ph])
    prices=o[:,[0,2]]
    market=1/o;market/=market.sum(axis=1,keepdims=True)
    scores={
        'Most likely side':p,
        'Return-score filter':p*prices,
        'Log-odds weighting':p*np.log(prices),
        'Square-root odds weighting':p*np.sqrt(prices),
        'Square-root profit weighting':p*np.sqrt(prices-1),
        'Legacy market-power weighting':prices**market[:,[0,2]],
        'Legacy variance weighting':p*(prices**2-p*prices**2+2*p*prices-1),
    }
    output={}
    for name,score in scores.items():
        pick=np.where(score[:,0]>=score[:,1],0,2)
        idx=(pick==2).astype(int)
        chosen=score[np.arange(len(ph)),idx]
        active=chosen>(1 if name in ['Return-score filter','Legacy market-power weighting'] else 0)
        output[name]=dict(pick=pick,active=active,proxy=True)
    square=scores['Square-root profit weighting']
    output['Home only: sqrt-profit comparison']=dict(pick=np.zeros(len(ph),int),
        active=square[:,0]>=square[:,1],proxy=True)
    variance=output['Legacy variance weighting']
    pick=variance['pick']
    selected_odds=o[np.arange(len(ph)),pick]
    probability=np.where(pick==0,ph,1-ph)
    output['Legacy variance: odds >=4, score edge >2%']=dict(pick=pick,
        active=variance['active']&(selected_odds>=4)&(selected_odds<=1000)&(probability*selected_odds-1>.02),proxy=True)
    output['Home only: probability >=50%']=dict(pick=np.zeros(len(ph),int),active=ph>=.5,proxy=False)
    output['Home only: positive expected return']=dict(pick=np.zeros(len(ph),int),active=ph*o[:,0]>1,proxy=False)
    output['Home only: expected return >2%']=dict(pick=np.zeros(len(ph),int),active=ph*o[:,0]-1>.02,proxy=False)
    return output


def settle(p_home,odds,outcomes,selection,staking='flat'):
    ph,o,y=np.asarray(p_home,float),np.asarray(odds,float),np.asarray(outcomes,int)
    pick=np.asarray(selection['pick'],int)
    active=np.asarray(selection['active'],bool)
    if y.shape!=ph.shape or pick.shape!=ph.shape or active.shape!=ph.shape or not np.isin(y,[0,1,2]).all():
        raise ValueError('Invalid outcomes or selection shapes')
    if not np.isin(pick,[0,2]).all():
        raise ValueError('Only straight home and away bets are permitted')
    price=o[np.arange(len(ph)),pick]
    score_probability=np.where(pick==0,ph,1-ph)
    edge=score_probability*price-1
    if staking=='flat':
        stakes=active.astype(float)
    elif staking=='tiers':
        positive=active&(edge>0)
        ranking=np.where(positive,edge/(price-1),0)
        stakes=np.where(positive,np.clip(np.ceil(np.round(ranking*100,10)),1,5),0)
    else:
        raise ValueError('Unknown staking rule')
    bet=stakes>0
    win=bet&(pick==y)
    unit_profit=np.where(pick==y,price-1,-1.)
    profit=stakes*unit_profit
    return dict(pick=pick,bet=bet,win=win,stakes=stakes,profit=profit,price=price,
        score_probability=score_probability,edge= edge,matched_flat=np.where(bet,unit_profit,0.))


def summary(result,outcomes):
    r=result;y=np.asarray(outcomes,int)
    stake=float(r['stakes'].sum());bets=int(r['bet'].sum());wins=int(r['win'].sum())
    cumulative=np.r_[0.,np.cumsum(r['profit'])]
    return dict(matches=len(y),bets=bets,skips=len(y)-bets,wins=wins,losses=bets-wins,
        win_rate=wins/bets if bets else np.nan,staked=stake,net=float(r['profit'].sum()),
        roi=float(r['profit'].sum()/stake) if stake else np.nan,
        same_bets_flat_net=float(r['matched_flat'].sum()),
        home_bets=int((r['bet']&(r['pick']==0)).sum()),away_bets=int((r['bet']&(r['pick']==2)).sum()),
        draw_losses=int((r['bet']&(y==1)).sum()),
        max_drawdown=float((np.maximum.accumulate(cumulative)-cumulative).max()))
