"""Existing weighting/filter ideas applied to direct binary models, no draw bets."""
import hashlib
import json
import html
from pathlib import Path
from select_expanded_features import ROOT,np,pd
from binary_betting import selections,settle,summary

DATA=ROOT/'results/lineup_improvements'
MODELS=DATA/'binary_retrained'
OUT=MODELS/'betting_no_draw'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    predictions=pd.read_csv(MODELS/'predictions.csv')
    choices=json.loads((MODELS/'selection.json').read_text())['selection']
    sources={'Audit 2021-22':pd.read_csv(DATA/'train_features.csv').query('season==2021'),
             'Test 2022-23':pd.read_csv(DATA/'test_features.csv')}
    paths=[MODELS/'predictions.csv',MODELS/'selection.json',DATA/'train_features.csv',DATA/'test_features.csv',
           ROOT/'src/binary_betting.py',Path(__file__)]
    plan=dict(rule='Only straight home/away bets; draws remain in every evaluated sample and lose '
        'either bet. No outcome-based exclusions. One bet at most per fixture; ties choose home.',
        probability='The models predict P(home win) and P(draw or away win). H/A weighting assigns '
        'the latter to the away selection as a heuristic score. It is NOT a calibrated away-win '
        'probability. Away expected-return filters and stake scores are therefore proxies. '
        'Home-only expected-return rules use the genuine model probability of a paying outcome.',
        adaptations='Synthetic double-chance rules omitted. Draw-to-home rules duplicate probability '
        'and return-score rules when no draw selection exists. Home-only sqrt rule compares against '
        'straight-away proxy odds, not the old synthetic draw/away hedge. Legacy market power uses '
        'normalized three-way odds but only home/away can be selected. Fixed legacy odds>=4 and '
        'score-edge>2% filter retained as an exploratory heuristic, not claimed validated for binary models.',
        stakes='Flat: one unit per strategy selection. Tiers: existing positive-score gate, '
        'score=(probability_score*odds-1)/(odds-1); each 1 percentage point advances one tier, '
        'capped at five, stakes 1..5 units. Non-positive scores skip. Same-bets flat profit also shown.',
        selection='Model candidates selected on earlier validation seasons; no strategies or '
        'thresholds selected or optimized using these audit/test returns. All saved candidate '
        'summaries exported; primary table shows the two validation-selected models.',
        caveat='Previously inspected data, recorded Bet365 odds, no fees/slippage or stake rounding, '
        'no bankroll constraints or compounding. Historical simulation only.',
        input_hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    summaries,ledgers,curves=[],[],[]
    for (split,family,candidate),group in predictions.groupby(['split','family','candidate']):
        source=sources[split].sort_values(['fixture_date','league','source_row'])
        data=source[['source_row','fixture_date','league','home_team_name','away_team_name',
                     'home_score','away_score','B365H','B365D','B365A']].merge(group,on='source_row',
                         validate='one_to_one',how='left',suffixes=('_source',''))
        if len(data)!=1065 or data.p_home_win.isna().any():
            raise ValueError('Incomplete prediction/odds join')
        for field in ['home_score','away_score']:
            if not np.array_equal(data[field+'_source'],data[field]):
                raise ValueError('Outcome alignment failure')
        ph=data.p_home_win.to_numpy(float)
        odds=data[['B365H','B365D','B365A']].to_numpy(float)
        actual=(1-np.sign(data.home_score-data.away_score)).to_numpy(int)
        selected=candidate==choices[family]['candidate']
        for strategy,choice in selections(ph,odds).items():
            for staking in ['flat','tiers']:
                result=settle(ph,odds,actual,choice,staking)
                summaries.append(dict(split=split,family=family,candidate=candidate,selected=selected,
                    strategy=strategy,staking=staking,uses_away_proxy=choice['proxy'],**summary(result,actual)))
                if selected:
                    ledger=data[['source_row','fixture_date','league','home_team_name','away_team_name',
                        'home_score','away_score','B365H','B365D','B365A','p_home_win','p_no_home_win']].copy()
                    ledger=ledger.assign(split=split,family=family,candidate=candidate,strategy=strategy,staking=staking,
                        selection=np.where(result['pick']==0,'home','away'),bet=result['bet'],win=result['win'],
                        stake=result['stakes'],decimal_odds=result['price'],score_probability=result['score_probability'],
                        score_edge=result['edge'],net=result['profit'],same_bets_flat_net=result['matched_flat'],
                        cumulative_net=np.cumsum(result['profit']),draw=(actual==1))
                    ledgers.append(ledger)
                    weekly=ledger.groupby(pd.to_datetime(ledger.fixture_date).dt.to_period('W').astype(str)).agg(
                        bets=('bet','sum'),staked=('stake','sum'),net=('net','sum')).reset_index()
                    weekly['cumulative_net']=weekly.net.cumsum()
                    weekly=weekly.assign(split=split,family=family,candidate=candidate,strategy=strategy,staking=staking)
                    curves.append(weekly)
    table=pd.DataFrame(summaries)
    table.to_csv(OUT/'all_results.csv',index=False)
    selected=table[table.selected]
    selected.to_csv(OUT/'selected_results.csv',index=False)
    pd.concat(ledgers,ignore_index=True).to_csv(OUT/'selected_ledger.csv.gz',index=False,compression='gzip')
    pd.concat(curves,ignore_index=True).to_csv(OUT/'weekly_returns.csv',index=False)
    body='<h1>Binary models: weighting and skipping without draw bets</h1>'
    for key in ['rule','probability','adaptations','stakes','selection','caveat']:
        body+='<p>'+html.escape(plan[key])+'</p>'
    for split in ['Test 2022-23','Audit 2021-22']:
        for staking in ['flat','tiers']:
            body+=f'<h2>{split}: {staking}</h2>'
            show=selected[(selected.split==split)&(selected.staking==staking)]
            body+=show[['family','candidate','strategy','uses_away_proxy','bets','wins','win_rate','staked',
                'net','roi','same_bets_flat_net','draw_losses','max_drawdown']].to_html(index=False,float_format=lambda v:f'{v:.4f}')
    (OUT/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Binary model betting strategies</title>'
        '<style>body{font:15px system-ui;margin:30px}table{border-collapse:collapse;font-size:12px}'
        'td,th{padding:7px;border-bottom:1px solid #ddd}</style>'+body,encoding='utf-8')
    print(selected.query('split=="Test 2022-23"')[['family','strategy','staking','bets','staked','net','roi','same_bets_flat_net','draw_losses']].to_string(index=False),flush=True)


if __name__=='__main__':
    main()
