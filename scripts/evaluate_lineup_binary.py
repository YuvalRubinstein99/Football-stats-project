"""Home win versus not-home-win: binary accuracy and two distinct settlements."""
import hashlib
import json
from select_expanded_features import ROOT, np, pd
from sklearn.metrics import log_loss, roc_auc_score

OUT = ROOT / 'results/lineup_improvements'
DEST = OUT / 'binary_home_win'


def settle(probabilities, odds, outcomes):
    """All arrays use H,D,A order. Exact probability ties choose home."""
    p, o, y = np.asarray(probabilities,float), np.asarray(odds,float), np.asarray(outcomes,int)
    if p.ndim != 2 or p.shape[1] != 3 or o.shape != p.shape or y.shape != (len(p),):
        raise ValueError('Invalid shapes')
    if not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(axis=1),1):
        raise ValueError('Invalid probabilities')
    if not np.isfinite(o).all() or (o<=1).any() or not np.isin(y,[0,1,2]).all():
        raise ValueError('Invalid odds or outcomes')
    home = p[:,0] >= .5
    correct = home == (y==0)
    double_odds = 1/(1/o[:,1]+1/o[:,2])
    hedge_stakes = np.column_stack([home.astype(float), np.where(home,0,double_odds/o[:,1]),
                                   np.where(home,0,double_odds/o[:,2])])
    straight_stakes = np.column_stack([home.astype(float), np.zeros(len(p)), (~home).astype(float)])
    result = {}
    for name, stakes in [('away_or_draw',hedge_stakes),('straight_away',straight_stakes)]:
        returns = stakes[np.arange(len(p)),y]*o[np.arange(len(p)),y]
        won = correct if name=='away_or_draw' else ((home & (y==0)) | (~home & (y==2)))
        assert np.allclose(stakes.sum(axis=1),1)
        if name=='away_or_draw':
            assert np.allclose(returns,np.where(correct,np.where(home,o[:,0],double_odds),0))
        result[name] = dict(home=home,correct=correct,won=won,stakes=stakes,
            profit=returns-1,odds=np.where(home,o[:,0],double_odds if name=='away_or_draw' else o[:,2]))
    return result


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    forecasts = pd.read_csv(OUT/'predictions.csv')
    source = {'Audit 2021-22':pd.read_csv(OUT/'train_features.csv').query('season == 2021'),
              'Test 2022-23':pd.read_csv(OUT/'test_features.csv')}
    metrics, betting, ledgers = [], [], []
    for split, data in source.items():
        data = data.sort_values(['fixture_date','league','source_row']).reset_index(drop=True)
        odds = data[['B365H','B365D','B365A']].to_numpy(float)
        outcomes = (1-np.sign(data.home_score-data.away_score)).to_numpy(int)
        groups = []
        for (family,candidate),g in forecasts[forecasts.split==split].groupby(['family','candidate']):
            joined = data[['source_row','home_score','away_score']].merge(g,on='source_row',how='left',
                validate='one_to_one',suffixes=('_source',''))
            assert len(joined)==1065 and joined.p_home.notna().all()
            assert np.array_equal(joined.home_score_source,joined.home_score)
            assert np.array_equal(joined.away_score_source,joined.away_score)
            groups.append((family,candidate,joined[['p_home','p_draw','p_away']].to_numpy(float)))
        market = 1/odds
        market /= market.sum(axis=1,keepdims=True)
        groups.append(('Market','normalized_odds',market))
        for family,candidate,p in groups:
            settled = settle(p,odds,outcomes)
            home = settled['away_or_draw']['home']
            actual_home = (outcomes==0).astype(int)
            metrics.append(dict(split=split,family=family,candidate=candidate,matches=len(data),
                correct=int(settled['away_or_draw']['correct'].sum()),accuracy=float(np.mean(home==actual_home)),
                binary_log_loss=float(log_loss(actual_home,np.column_stack([1-p[:,0],p[:,0]]),labels=[0,1])),
                binary_brier=float(np.mean((p[:,0]-actual_home)**2)),auc=float(roc_auc_score(actual_home,p[:,0])),
                home_picks=int(home.sum()),not_home_picks=int((~home).sum()),
                always_home_correct=int(actual_home.sum()),always_not_home_correct=int((1-actual_home).sum())))
            for settlement,r in settled.items():
                betting.append(dict(split=split,family=family,candidate=candidate,settlement=settlement,
                    bets=len(data),staked_units=float(len(data)),winning_bets=int(r['won'].sum()),
                    win_rate=float(r['won'].mean()),net_units=float(r['profit'].sum()),roi=float(r['profit'].mean()),
                    home_side_net=float(r['profit'][home].sum()),not_home_side_net=float(r['profit'][~home].sum()),
                    draws_on_not_home_picks=int(((~home)&(outcomes==1)).sum())))
                ledger=data[['source_row','fixture_date','league','home_team_name','away_team_name','home_score','away_score']].copy()
                ledger=ledger.assign(split=split,family=family,candidate=candidate,settlement=settlement,
                    chosen_side=np.where(home,'home','not_home'),p_home=p[:,0],p_not_home=p[:,1]+p[:,2],
                    binary_correct=r['correct'],winning_bet=r['won'],effective_odds=r['odds'],
                    home_stake=r['stakes'][:,0],draw_stake=r['stakes'][:,1],away_stake=r['stakes'][:,2],
                    net_units=r['profit'])
                ledgers.append(ledger)
    pd.DataFrame(metrics).to_csv(DEST/'accuracy.csv',index=False)
    pd.DataFrame(betting).to_csv(DEST/'returns.csv',index=False)
    pd.concat(ledgers,ignore_index=True).to_csv(DEST/'ledger.csv.gz',index=False,compression='gzip')
    method = dict(rule='Choose home when P(home)>=0.5, otherwise not-home. No new model training, '
        'threshold optimization, odds weighting or expected-return filter. Flat one unit per fixture.',
        away_or_draw='Synthetic double chance: odds=1/(1/draw_odds+1/away_odds); split the one-unit '
        'stake as effective_odds/draw_odds and effective_odds/away_odds. Either draw or away wins '
        'the covered outcome. These are not quoted double-chance prices. Assumes simultaneous '
        'recorded odds and fractional stakes, with no fees or rounding.',
        straight_away='Use the same binary selection but place a straight away-win bet on not-home '
        'picks. Draws lose. P(away)+P(draw) is not the probability that this straight bet wins.',
        caveat='Previously inspected historical audit/test matches. Binary correctness is distinct '
        'from settled bet wins and profit. No actual wagers placed.',
        input_hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      [OUT/'predictions.csv',OUT/'train_features.csv',OUT/'test_features.csv']})
    (DEST/'methodology.json').write_text(json.dumps(method,indent=2),encoding='utf-8')
    body='<h1>Home wins versus draw or away wins</h1>'
    body+=''.join('<p>'+method[key]+'</p>' for key in ['rule','away_or_draw','straight_away','caveat'])
    body+='<h2>Binary outcome accuracy</h2>'+pd.DataFrame(metrics).to_html(index=False,float_format=lambda v:f'{v:.5f}')
    body+='<h2>Flat one-unit betting returns</h2>'+pd.DataFrame(betting).to_html(index=False,float_format=lambda v:f'{v:.5f}')
    (DEST/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Home win binary evaluation</title>'
        '<style>body{font:15px system-ui;margin:30px}table{border-collapse:collapse;font-size:12px}'
        'td,th{padding:7px;border-bottom:1px solid #ddd}</style>'+body,encoding='utf-8')
    print(pd.DataFrame(metrics).query('split == "Test 2022-23"').to_string(index=False))
    print(pd.DataFrame(betting).query('split == "Test 2022-23"').to_string(index=False))


if __name__=='__main__':
    main()
