"""Summarize forward-test stability without optimizing the betting rules."""
import html
import json
from backtest_binary_training_oos import OUT,np,pd


def main():
    ledger=pd.read_csv(OUT/'ledger.csv.gz')
    ledger['fixture_date']=pd.to_datetime(ledger.fixture_date)
    records=[]
    for (protocol,strategy,staking),g in ledger.groupby(['protocol','strategy','staking']):
        # Include zero-bet weeks. Four-week circular blocks preserve local clusters.
        first=g.fixture_date.min().to_period('W')
        last=g.fixture_date.max().to_period('W')
        weeks=pd.period_range(first,last,freq='W')
        weekly=g.groupby(g.fixture_date.dt.to_period('W'))[['net','stake']].sum().reindex(weeks,fill_value=0.)
        values=weekly.to_numpy()
        rng=np.random.default_rng(20261110)
        starts=rng.integers(0,len(weeks),size=(5000,int(np.ceil(len(weeks)/4))))
        indices=((starts[:,:,None]+np.arange(4))%len(weeks)).reshape(5000,-1)[:,:len(weeks)]
        sums=values[indices].sum(axis=1)
        roi=sums[:,0]/sums[:,1]
        season=g.groupby('season').net.sum()
        active=g[g.bet].copy()
        best=active.nlargest(5,'net')
        records.append(dict(protocol=protocol,strategy=strategy,staking=staking,
            roi=float(g.net.sum()/g.stake.sum()),roi_lower_95=float(np.quantile(roi,.025)),
            roi_upper_95=float(np.quantile(roi,.975)),profitable_seasons=int((season>0).sum()),
            seasons=len(season),best_season_net=float(season.max()),worst_season_net=float(season.min()),
            predicted_return=float(g.expected_net.sum()/g.stake.sum()),
            net_without_best_five_bets=float(g.net.sum()-best.net.sum()),
            net_common_2017_2021=float(g.loc[g.season>=2017,'net'].sum()),
            stake_common_2017_2021=float(g.loc[g.season>=2017,'stake'].sum())))
    robust=pd.DataFrame(records)
    robust.to_csv(OUT/'robustness.csv',index=False)
    plan=json.loads((OUT/'plan.json').read_text())
    body='<h1>Training-period out-of-sample home betting</h1>'
    for field in ['fixed','nested','warmup','historical_features','betting','caveat']:
        body+='<p><strong>'+html.escape(field.replace('_',' ').capitalize())+': </strong>'+html.escape(plan[field])+'</p>'
    for protocol in ['fixed','nested']:
        body+=f'<h2>{protocol.capitalize()} forward predictions</h2><img src="{protocol}_cumulative.svg" alt="Cumulative profit">'
    for file,title in [('summary','Returns and drawdowns'),('by_season','Season stability'),
                       ('by_league','League breakdown'),('prediction_metrics','Probability accuracy'),
                       ('odds_diagnostics','Selected bets by odds'),('robustness','Robustness checks')]:
        body+='<h2>'+title+'</h2>'+pd.read_csv(OUT/f'{file}.csv').to_html(index=False,float_format=lambda v:f'{v:.4f}')
    body+='<p>ROI intervals are descriptive 95% percentile intervals from 5,000 circular four-week block resamples. '
    body+='They do not include model fitting or researcher-selection uncertainty. Removing the five best bets is '
    body+='a concentration diagnostic, not a proposed strategy. League and odds breakdowns are diagnostic, not '
    body+='permission to select profitable slices retrospectively.</p>'
    (OUT/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Training OOS betting</title>'
        '<style>body{font:15px system-ui;margin:32px;color:#263548}img{max-width:1200px;width:100%}'
        'table{border-collapse:collapse;font-size:12px}td,th{padding:7px;border-bottom:1px solid #ddd}</style>'+body,encoding='utf-8')
    print(robust.to_string(index=False))


if __name__=='__main__':
    main()
