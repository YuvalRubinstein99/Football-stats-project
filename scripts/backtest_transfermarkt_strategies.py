"""Replay all existing selection rules on cached paired predictions, no refit."""
import json,hashlib
from collections import defaultdict
from select_expanded_features import ROOT,pd,np
from live_bets import recommendations,STRATEGIES,SEARCH_CANDIDATE
from stake_levels import size_bet

OUT=ROOT/'results/transfermarkt_models'
NAMES=STRATEGIES+[SEARCH_CANDIDATE]
SIZING=['Flat $1','Five levels $1–$5','Quarter Kelly','Robust quarter Kelly']

def selections(frame,p):
    arrays={n:np.zeros((len(frame),5)) for n in NAMES}
    for i,r in enumerate(frame.to_dict('records')):
        match=dict(home=r['home_team_name'],away=r['away_team_name'],p_home=p[i,0],p_draw=p[i,1],p_away=p[i,2],**{k:r[k] for k in ['B365H','B365D','B365A']})
        bets=recommendations(match)
        candidate=next(dict(b) for b in bets if b['strategy']=='Legacy variance score');candidate['strategy']=SEARCH_CANDIDATE
        if candidate['action']=='Bet' and not(4<=candidate['decimal_odds']<=1000 and candidate['expected_net_per_unit']>.02):candidate['action']='Skip'
        bets.append(candidate)
        outcome='home' if r['home_score']>r['away_score'] else 'away' if r['home_score']<r['away_score'] else 'draw'
        price=r[{'home':'B365H','away':'B365A','draw':'B365D'}[outcome]]
        for b in bets:
            if b['action']!='Bet':continue
            o=b['decimal_odds'];q=b['probability'];level=size_bet(b)['stake_dollars']
            arrays[b['strategy']][i]=[1,level,max(0,(q*o-1)/(o-1)) if o>1 else 0,max(0,((max(0,q-.02))*o-1)/(o-1)) if o>1 else 0,b[outcome+'_stake']*price-1]
    return arrays

def simulate(f,a,method):
    balance=peak=1000.;dd=dd_dollars=spent=0.;bets=0;points=[dict(date='Start',net=0.)];weekly={}
    if '_settlement_groups' not in f.attrs:
        f.attrs['_settlement_groups']=[(day,ids,str(pd.Timestamp(day).to_period('W'))) for day,ids in f.groupby('fixture_date',sort=True).indices.items()]
    for day,ids,week in f.attrs['_settlement_groups']:
        if method==SIZING[0]:amount=a[ids,0].copy()
        elif method==SIZING[1]:amount=a[ids,1].copy()
        else:amount=balance*.25*a[ids,2 if method==SIZING[2] else 3]
        amount=np.minimum(amount,max(0,balance)*.02)
        total=amount.sum()
        if total>max(0,balance)*.1 and total:amount*=max(0,balance)*.1/total
        balance+=float((amount*a[ids,4]).sum());spent+=float(amount.sum());bets+=int((amount>0).sum())
        peak=max(peak,balance);dd=max(dd,1-balance/peak);dd_dollars=max(dd_dollars,peak-balance)
        weekly[week]=dict(date=day,net=balance-1000.)
    return dict(matches=len(f),bets=bets,staked=spent,net=balance-1000.,max_drawdown=dd,max_drawdown_dollars=dd_dollars,points=points+list(weekly.values()))

def main():
    plan=json.loads((OUT/'plan.json').read_text());train=pd.read_csv(OUT/'train.csv');test=pd.read_csv(OUT/'test.csv')
    allsummary=[];checks=0
    for model in plan['models']:
        packs={s:[] for s in ['Test','Train in-sample','Train chronological OOF']}
        for variant in ['Baseline','With values']:
            ps=[];fs=[]
            for year in [2020,2021,2022]:
                key=hashlib.sha256((json.dumps(plan,sort_keys=True)+str((year,model,variant))).encode()).hexdigest()
                saved=np.load(OUT/(key+'.npz'))
                if year==2022:
                    pairs=[('Test',test,saved['p']),('Train in-sample',train,saved['pt'])]
                else:
                    fs.append(train[train.season==year]);ps.append(saved['p']);continue
            pairs.append(('Train chronological OOF',pd.concat(fs,ignore_index=True),np.concatenate(ps)))
            for split,frame,p in pairs:
                print(model,variant,split,flush=True)
                assert len(frame)==len(p)
                arrays=selections(frame,p)
                for league in ['All','E0','SP1','D1','I1']:
                    mask=np.ones(len(frame),bool) if league=='All' else frame.league.eq(league).to_numpy()
                    f=frame.loc[mask].reset_index(drop=True)
                    for strategy,a in arrays.items():
                        for method in SIZING:
                            stats=simulate(f,a[mask],method)
                            row=dict(model=model,variant=variant,split=split,league=league,strategy=strategy,method=method,**stats)
                            assert np.isclose(row['points'][-1]['net'],row['net']) and 0<=row['max_drawdown']<=1 and row['net']>=-1000-1e-7
                            packs[split].append(row);allsummary.append({k:v for k,v in row.items() if k!='points'});checks+=1
        for split,rows in packs.items():
            name='TM strategies '+model+' '+split
            path=ROOT/'results/bankroll_backtest/daily'/(hashlib.sha256(name.encode()).hexdigest()+'.json')
            path.write_text(json.dumps(dict(rows=rows)),encoding='utf-8')
    pd.DataFrame(allsummary).to_csv(OUT/'selection_strategies.csv',index=False)
    (OUT/'selection_strategy_policy.json').write_text(json.dumps(dict(strategies=NAMES,sizing=SIZING,checks=checks,
        policy='Same frozen predictions and original train/test rows. Flat $1 follows each rule even if estimated edge is negative; five levels and Kelly require positive edge. All methods capped at 2% per match, 10% per day from opening bankroll, initially $1000. Synthetic double chance splits one total stake at equal payout; not quoted double-chance odds. Fixed filtered legacy variance applies odds 4–1000 and expected return >2% to every model as an exploratory rule, not an optimized claim. No test tuning.'),indent=2))
    from report_transfermarkt_strategies import install
    install(plan['models'],NAMES,SIZING)
    print('Validated',checks,'strategy curves',flush=True)

if __name__=='__main__':main()
