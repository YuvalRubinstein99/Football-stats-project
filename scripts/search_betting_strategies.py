"""Fixed-model exploratory strategy search with disjoint calibration/search/validation periods."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.lineup_deps'),str(ROOT/'src')]
import csv,json,html
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import skellam
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,log_loss
from lineup_model import LineupForest,clean
from live_bets import recommendations,STRATEGIES


def stats(profit,stake,mask):
    p=profit[mask];s=stake[mask];curve=np.r_[0,np.cumsum(p)]
    return dict(bets=int((s>0).sum()),staked=float(s.sum()),net=float(p.sum()),
                roi=float(p.sum()/s.sum()) if s.sum() else 0.,
                drawdown=float(np.max(np.maximum.accumulate(curve)-curve)))


def main():
    out=ROOT/'results'
    model=LineupForest()
    train=clean(pd.read_csv(ROOT/'data/train.csv')).replace([np.inf,-np.inf],np.nan).dropna()
    test=pd.read_csv(out/'match_predictions.csv');rawtest=pd.read_csv(ROOT/'data/test.csv')
    labels={}
    with (out/'stake_level_bet_ledger.csv').open(encoding='utf-8') as f:
        for r in csv.DictReader(f):labels[(r['split'],int(r['source_row']))]=r['season']
    means=np.maximum(model.forest.oob_prediction_,1e-8)
    if means.shape!=(len(train),2):raise ValueError('OOB alignment mismatch')
    ptrain=np.column_stack([skellam.cdf(-1,means[:,0],means[:,1]),skellam.pmf(0,means[:,0],means[:,1]),skellam.sf(0,means[:,0],means[:,1])])
    ptest=test[['RF_vector_Away','RF_vector_Draw','RF_vector_Home']].to_numpy()
    rows=[];pp=[]
    for split,frame,probs in [('Train (OOB)',train,ptrain),('Test',test,ptest)]:
        for i,(idx,r) in enumerate(frame.iterrows()):
            source=int(idx) if split!='Test' else int(r.source_test_row)
            original=r if split!='Test' else rawtest.iloc[source]
            if split=='Test':
                for k in ['home_team_name','away_team_name','home_score','away_score','B365A','B365D','B365H']:
                    if original[k]!=r[k]:raise ValueError('Test alignment mismatch')
            rows.append(dict(season=labels[split,source],week=int(str(original.Matchweek).split()[-1]),source=source,
                split=split,home=original.home_team_name,away=original.away_team_name,
                odds=[float(original[k]) for k in ['B365A','B365D','B365H']],
                outcome=int(np.sign(original.home_score-original.away_score))+1))
            pp.append(probs[i])
    order=sorted(range(len(rows)),key=lambda i:(rows[i]['season'],rows[i]['week'],rows[i]['source']))
    rows=[rows[i] for i in order];p=np.array(pp)[order];p/=p.sum(axis=1,keepdims=True)
    odds=np.array([r['odds'] for r in rows]);y=np.array([r['outcome'] for r in rows]);season=np.array([r['season'] for r in rows])
    years=np.array([int(s[:4]) for s in season])
    masks=dict(calibration=years<2018,search=(years>=2018)&(years<2020),validation=(years>=2020)&(years<2022),test=years==2022)
    # Multinomial logistic calibration trained only on the earliest four seasons.
    calibrator=LogisticRegression(C=1.,max_iter=1000).fit(np.log(np.clip(p[masks['calibration']],1e-9,1)),y[masks['calibration']])
    calibrated=calibrator.predict_proba(np.log(np.clip(p,1e-9,1)))
    market=1/odds;market/=market.sum(axis=1,keepdims=True)
    diagnostics=[]
    for name,probs in [('Raw RF',p),('Calibrated RF',calibrated),('Market',market)]:
        for period,mask in masks.items():
            diagnostics.append(dict(model=name,period=period,auc=float(roc_auc_score(y[mask],probs[mask],multi_class='ovr',average='macro')),log_loss=float(log_loss(y[mask],probs[mask]))))
    base={}
    for name,probs in [('raw',p),('calibrated',calibrated)]:
        print('Building selections:',name,flush=True)
        packs=[[] for _ in STRATEGIES]
        for i,r in enumerate(rows):
            m=dict(home=r['home'],away=r['away'],p_away=probs[i,0],p_draw=probs[i,1],p_home=probs[i,2],B365A=odds[i,0],B365D=odds[i,1],B365H=odds[i,2])
            for j,b in enumerate(recommendations(m)):
                legs=np.array([b['away_stake'],b['draw_stake'],b['home_stake']])
                packs[j].append([b['action']=='Bet',b['decimal_odds'] or 1,b['expected_net_per_unit'] if b['expected_net_per_unit'] is not None else -1,
                    float(legs[y[i]]*odds[i,y[i]]-1),int(legs.argmax()) if not b['synthetic'] else 3])
        for j,strategy in enumerate(STRATEGIES):base[name,strategy]=np.array(packs[j])
    def evaluate(config):
        b=base[config['probabilities'],config['strategy']];o=b[:,1];edge=b[:,2]
        gate=(b[:,0]>0)&(edge>config['min_edge'])&(o>=config['min_odds'])&(o<=config['max_odds'])
        if config['outcome']!='all':gate&=b[:,4]=={'away':0,'draw':1,'home':2}[config['outcome']]
        stake=gate.astype(float)
        if config['sizing']!='flat':
            score=np.maximum(edge,0)/np.maximum(o-1,1e-8)
            stake*=np.minimum(5,np.maximum(1,np.ceil(np.round(score/config['sizing'],10))))
        return stake*b[:,3],stake
    candidates=[]
    for name,strategy in base:
        for edge in [0,.02,.05,.10,.20]:
            for low,high in [(1.01,1000),(1.01,2),(2,4),(4,1000)]:
                for outcome in ['all','home','draw','away']:
                    c=dict(probabilities=name,strategy=strategy,min_edge=edge,min_odds=low,max_odds=high,outcome=outcome,sizing='flat')
                    profit,stake=evaluate(c);s=stats(profit,stake,masks['search'])
                    annual=[stats(profit,stake,years==yr) for yr in [2018,2019]]
                    eligible=s['bets']>=100 and all(a['bets']>=30 for a in annual)
                    # Predeclared objective rewards profit and penalizes drawdown/losing seasons.
                    s['balanced']=s['net']-.5*s['drawdown']-sum(max(0,-a['net']) for a in annual)
                    candidates.append(dict(config=c,search=s,eligible=eligible))
    eligible=[c for c in candidates if c['eligible']]
    def rank(c,objective):
        s=c['search']
        if objective=='profit':return s['net']
        if objective=='drawdown':return -s['drawdown'] if s['net']>0 else -1e12+s['net']
        return s['balanced']
    # Shortlist three distinct flat-stake choices per original strategy, using search seasons ONLY.
    finalists=[];seen=set()
    for strategy in STRATEGIES:
        pool=[c for c in eligible if c['config']['strategy']==strategy]
        if not pool:continue
        for objective in ['profit','drawdown','balanced']:
            best=max(pool,key=lambda c:rank(c,objective));key=json.dumps(best['config'],sort_keys=True)
            if key in seen:continue
            seen.add(key)
            for sizing in ['flat',.01,.025,.05]:
                config={**best['config'],'sizing':sizing};profit,stake=evaluate(config)
                metrics={period:stats(profit,stake,mask) for period,mask in masks.items() if period!='calibration'}
                annual=[stats(profit,stake,years==yr) for yr in [2018,2019]]
                s=metrics['search'];s['balanced']=s['net']-.5*s['drawdown']-sum(max(0,-a['net']) for a in annual)
                finalists.append(dict(config=config,**metrics))
    chosen={objective:max(finalists,key=lambda c:rank(c,objective)) for objective in ['profit','drawdown','balanced']}
    # Validation and test never participate in the selection above.
    for c in finalists:
        profit,stake=evaluate(c['config']);c['seasons']=[dict(season=s,**stats(profit,stake,season==s)) for s in sorted(set(season)) if int(s[:4])>=2018]
    methodology=('Exploratory fixed-model study. Earliest 2014-18 OOB seasons fit logistic calibration; 2018-20 select strategies and stake bands; 2020-22 audit the frozen choices; 2022-23 is a previously inspected held-out comparison. '
        'Base training probabilities are OOB from a forest trained across 2014-22, so they can use later-season information. This is NOT a leakage-free walk-forward model backtest. '
        'One original strategy selection per match, filtered afterward. Synthetic selections qualify only for the all-outcomes filter. Odds bands overlap at 2 and 4 across alternative candidates. '
        'Flat search requires 100 total bets and 30 in each search season. Balanced score = search net profit minus half max drawdown minus losses in negative search seasons. '
        'Drawdown profile requires positive search profit when available. Five-level bands use Kelly score widths 1%, 2.5%, or 5%, capped at $5; flat stakes are $1. '
        'Drawdown uses season/matchweek/source-row order, not exact kickoff order. No fees, bankroll limits, or synthetic cent rounding. Selection across many candidates risks overfitting. No live settings changed.')
    report=dict(methodology=methodology,grid_count=len(candidates),eligible_count=len(eligible),finalist_count=len(finalists),diagnostics=diagnostics,chosen=chosen,finalists=finalists)
    (out/'strategy_search.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    for filename,items in [('strategy_search_grid.csv',candidates),('strategy_search_finalists.csv',finalists)]:
        records=[]
        for c in items:
            r=dict(c['config'])
            for period in ['search','validation','test']:
                if period in c:r.update({period+'_'+k:v for k,v in c[period].items()})
            if 'eligible' in c:r['eligible']=c['eligible']
            records.append(r)
        with (out/filename).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    parts=['<!doctype html><meta charset="utf-8"><title>Strategy search</title><style>body{font:16px system-ui;max-width:1250px;margin:40px auto;padding:24px;background:#0c1420;color:#e5edf6}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:12px;border-bottom:1px solid #40546b;text-align:left}p{line-height:1.6}pre{white-space:pre-wrap;color:#77e2bc}</style><h1>Fixed-model strategy search</h1>',f'<p>{html.escape(methodology)}</p>',f'<p>{len(candidates)} flat candidates; {len(eligible)} eligible; {len(finalists)} shortlisted rule/staking combinations.</p>']
    for objective,c in chosen.items():
        parts.extend([f'<h2>{objective.title()} profile — selected on 2018–20 only</h2>',f'<pre>{html.escape(json.dumps(c["config"],indent=2))}</pre>','<table><tr><th>Period</th><th>Bets</th><th>Staked</th><th>Net profit</th><th>ROI</th><th>Max drawdown</th></tr>'])
        for period in ['search','validation','test']:
            s=c[period];parts.append(f'<tr><td>{period}</td><td>{s["bets"]}</td><td>${s["staked"]:,.2f}</td><td>${s["net"]:,.2f}</td><td>{s["roi"]:.2%}</td><td>${s["drawdown"]:,.2f}</td></tr>')
        parts.append('</table>')
    parts.append('<h2>All shortlisted combinations</h2><table><tr><th>Strategy / probabilities / outcome</th><th>Edge / odds / stake width</th><th>Search net</th><th>Validation net</th><th>Test net</th></tr>')
    for c in finalists:
        f=c['config'];parts.append(f'<tr><td>{html.escape(f["strategy"])} / {f["probabilities"]} / {f["outcome"]}</td><td>{f["min_edge"]:.0%} / {f["min_odds"]}–{f["max_odds"]} / {f["sizing"]}</td><td>${c["search"]["net"]:.2f}</td><td>${c["validation"]["net"]:.2f}</td><td>${c["test"]["net"]:.2f}</td></tr>')
    parts.append('</table>');(out/'strategy_search.html').write_text('\n'.join(parts),encoding='utf-8')
    print(json.dumps(dict(grid=len(candidates),eligible=len(eligible),finalists=len(finalists),chosen=chosen),indent=2),flush=True)


if __name__=='__main__':main()
