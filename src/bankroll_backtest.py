"""Comparable cash-constrained daily-settlement staking experiments."""
import numpy as np

METHODS=['Flat $1','Quarter Kelly','Robust quarter Kelly']

def stakes(prob,odds,method,bankroll=1000.):
    p=np.asarray(prob,float);o=np.asarray(odds,float)
    if p.shape!=o.shape or p.ndim!=2 or p.shape[1]!=3:raise ValueError('Expected H/D/A matrices')
    if not np.isfinite(p).all() or not np.isfinite(o).all() or (o<=1).any() or (p<0).any() or not np.allclose(p.sum(1),1):raise ValueError('Invalid probabilities or odds')
    if method not in METHODS:raise ValueError('Unknown staking method')
    side=np.argmax(p*o-1,axis=1);i=np.arange(len(p));q=p[i,side];price=o[i,side]
    edge=q*price-1
    if method=='Flat $1':amount=np.where(edge>0,1.,0.)
    else:
        if method=='Robust quarter Kelly':q=np.maximum(0,q-.02)
        amount=bankroll*np.minimum(.02,.25*np.maximum(0,(q*price-1)/(price-1)))
    amount=np.minimum(amount,max(0,bankroll)*.02)
    # Reserve all stakes before any same-day outcomes become available.
    total=amount.sum();cap=max(0,bankroll)*.10
    if total>cap and total:amount*=cap/total
    return side,amount

def simulate(frame,prob,method,initial=1000.):
    if not np.isfinite(initial) or initial<=0:raise ValueError('Positive initial bankroll required')
    f=frame.reset_index(drop=True);p=np.asarray(prob)
    if len(f)!=len(p):raise ValueError('Row alignment mismatch')
    balance=initial;peak=initial;drawdown=0.;spent=0.;bets=0;records=[];points=[dict(date='Start',net=0.)]
    odds=f[['B365H','B365D','B365A']].to_numpy(float)
    selected,_=stakes(p,odds,method,initial) # validate inputs once
    ix=np.arange(len(f));prices=odds[ix,selected];q=p[ix,selected]
    positive=q*prices>1
    if method=='Robust quarter Kelly':q=np.maximum(0,q-.02)
    fractions=np.minimum(.02,.25*np.maximum(0,(q*prices-1)/(prices-1)))
    outcomes=(1-np.sign(f.home_score-f.away_score)).to_numpy(int)
    returns=np.where(selected==outcomes,prices,0.)
    dates=f.fixture_date.to_numpy();homes=f.home_team_name.to_numpy();aways=f.away_team_name.to_numpy();leagues=f.league.to_numpy()
    for date,ids in f.groupby('fixture_date',sort=True).indices.items():
        amount=np.where(positive[ids],1.,0.) if method=='Flat $1' else balance*fractions[ids]
        amount=np.minimum(amount,max(0,balance)*.02)
        total=amount.sum()
        if total>balance*.10 and total:amount*=max(0,balance)*.10/total
        profit=amount*(returns[ids]-1);opening=balance;balance+=profit.sum();spent+=amount.sum();bets+=int((amount>0).sum())
        peak=max(peak,balance);drawdown=max(drawdown,1-balance/peak)
        for j,i in enumerate(ids):
            records.append(dict(date=str(date),source_row=int(i),home=homes[i],away=aways[i],league=leagues[i],side=['home','draw','away'][selected[i]],probability=float(p[i,selected[i]]),odds=float(prices[i]),opening_bankroll=float(opening),stake=float(amount[j]),net=float(profit[j])))
        points.append(dict(date=str(date),net=float(balance-initial)))
    # Chart sampled at weekly closing balances; simulation and drawdown use every date.
    import pandas as pd
    weekly={}
    for pt in points[1:]:weekly[str(pd.Timestamp(pt['date']).to_period('W'))]=pt
    return dict(matches=len(f),bets=bets,staked=float(spent),net=float(balance-initial),ending_bankroll=float(balance),roi=float((balance-initial)/spent) if spent else 0.,max_drawdown=float(drawdown),points=[points[0],*weekly.values()]),records
