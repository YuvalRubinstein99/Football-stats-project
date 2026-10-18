"""Paired chronological evaluation; no test-selected features or strategies."""
import json,hashlib,html
from select_expanded_features import ROOT,pd,np,joblib,threadpool_limits
from train_recent_models import train_predict
from train_classifier_models import fit_model,matrix
from classifier_features import FORM,RICH
from calibrate_models import scores
from bankroll_backtest import simulate,METHODS

DATA=ROOT/'data/fifa_updates';OUT=ROOT/'results/fifa_updates'
VARIANTS=['original','season_frozen','updated']
MODELS=['Expanded RF','RF classifier + both','Linear','Poisson','Ordered_logit','Ordered_probit']

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    base=joblib.load(ROOT/'results/expanded_comparison/expanded__RF_vector.joblib')
    frames={v:pd.read_csv(DATA/f'train_{v}.csv') for v in VARIANTS}
    cols=['HomePlayer_Overall_mean','AwayPlayer_Overall_mean',*[c for c in frames['updated'] if c.endswith(('_form','_prior'))],'B365H','B365D','B365A','Matchweek','home_team_name','away_team_name']
    plan=dict(models=MODELS,variants=VARIANTS,validation_seasons=[2020,2021],test_season=2022,columns=cols,
              selection='No model, feature, stake or hyperparameter selection on test. Fixed existing forest settings and regression specifications.',
              primary_comparison='updated versus season_frozen; original ratings are descriptive and not guaranteed pre-match',
              limitations='Matched complete-XI subset, actual historical starters, Bet365 prices, previously inspected test season. Archive ends January 2023: later matches use last available prior snapshot. No Live Form.',
              hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in DATA.glob('*_*.csv')})
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    records=[];bankroll=[];curves=[];predictions={}
    for year in [2020,2021,2022]:
        # Test loaded only after the fixed plan is saved.
        valid={v:pd.read_csv(DATA/f'test_{v}.csv') if year==2022 else frames[v][frames[v].season==year] for v in VARIANTS}
        for v in VARIANTS:
            va=valid[v];tr=frames[v][(frames[v].season<year)&(frames[v].fixture_date<va.fixture_date.min())]
            assert tr.fixture_date.max()<va.fixture_date.min()
            assert np.array_equal(va.source_row,valid['updated'].source_row)
            split='Test 2022-23' if year==2022 else f'Forward {year}-{year+1}'
            for model in MODELS:
                fingerprint=hashlib.sha256((json.dumps(plan,sort_keys=True)+str((year,v,model))).encode()).hexdigest()
                cache=OUT/(fingerprint+'.npz')
                if cache.exists():p=np.load(cache)['p']
                else:
                    print('Fitting',year,v,model,'train',len(tr),'validation',len(va),flush=True)
                    if model=='RF classifier + both':
                        fitted=fit_model(base,tr,FORM+RICH);p=fitted['forest'].predict_proba(matrix(fitted,va))
                    else:p,_=train_predict(model,base if model=='Expanded RF' else dict(columns=cols),tr,va,np.ones(len(tr)))
                    assert np.isfinite(p).all() and np.allclose(p.sum(1),1)
                    np.savez_compressed(cache,p=p)
                predictions[year,v,model]=p
                for league in ['All','E0','SP1','D1','I1']:
                    mask=np.ones(len(va),bool) if league=='All' else va.league.eq(league).to_numpy()
                    f=va.loc[mask];q=p[mask];y=(1-np.sign(f.home_score-f.away_score)).to_numpy(int)
                    if not len(f):continue
                    row=dict(split=split,model=model,variant=v,league=league,matches=len(f),train_matches=len(tr),**scores(q,y))
                    records.append(row)
                    for method in METHODS:
                        stats,_=simulate(f,q,method);pts=stats.pop('points')
                        bankroll.append(dict(split=split,model=model,variant=v,league=league,method=method,**stats))
                        curves.append(dict(split=split,model=model,variant=v,league=league,method=method,points=pts))
                pd.DataFrame(records).to_csv(OUT/'metrics.csv',index=False)
                pd.DataFrame(bankroll).to_csv(OUT/'bankroll.csv',index=False)
    # Paired calendar-week block bootstrap: effect uncertainty, not a tuning score.
    differences=[];va=valid['updated'];weeks=pd.to_datetime(va.fixture_date).dt.to_period('W').astype(str).to_numpy()
    y=(1-np.sign(va.home_score-va.away_score)).to_numpy(int);rng=np.random.default_rng(914)
    for model in MODELS:
        a=predictions[2022,'updated',model];b=predictions[2022,'season_frozen',model]
        diff=-np.log(np.clip(a[np.arange(len(y)),y],1e-15,1))+np.log(np.clip(b[np.arange(len(y)),y],1e-15,1))
        groups=[np.flatnonzero(weeks==w) for w in np.unique(weeks)]
        samples=[np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]) for _ in range(2000)]
        lo,hi=np.quantile([diff[ix].mean() for ix in samples],[.025,.975])
        differences.append(dict(model=model,test_log_loss_change=float(diff.mean()),ci_low=float(lo),ci_high=float(hi)))
    pd.DataFrame(differences).to_csv(OUT/'paired_differences.csv',index=False)
    (OUT/'curves.json').write_text(json.dumps(curves))
    render(records,bankroll,curves,differences,plan)

def render(records,bankroll,curves,differences,plan):
    def table(rows):return pd.DataFrame(rows).to_html(index=False,float_format=lambda x:f'{x:.5f}')
    body='<h1>Historical FIFA updates: paired model comparison</h1><p>Negative log-loss change favors updated ratings. The primary baseline freezes ratings before the first league match; updated ratings use strictly earlier calendar dates. Original annual ratings are shown separately.</p>'
    body+='<h2>Test: updated minus season-frozen log loss</h2>'+table(differences)
    body+='<p>95% paired calendar-week bootstrap intervals; exploratory, previously examined test season. An interval crossing zero does not establish improvement.</p>'
    body+='<h2>Coverage</h2>'+pd.read_csv(DATA/'coverage.csv').to_html(index=False)
    body+='<h2>Cumulative profit</h2><div id="controls"></div><svg id="chart" viewBox="0 0 1000 390"></svg><div id="legend"></div>'
    body+='<h2>All-league predictive metrics</h2>'+table([r for r in records if r['league']=='All'])
    body+='<details><summary>Profits and daily maximum drawdowns by league</summary>'+table(bankroll)+'</details>'
    body+='<details><summary>Predictive metrics by league</summary>'+table(records)+'</details>'
    body+='<h2>Method and limits</h2><p>'+html.escape(plan['limitations'])+'</p><p>Each strategy starts with $1,000 per league and split; match cap 2%, daily cap 10%. No fees. Forward validation is out of sample, not in-sample training profit. Three rating treatments are refitted on identical matched training fixtures. Existing form and team identity are retained. No calibration was added or retuned in this isolated rating comparison.</p><p>Source: <a href="https://huggingface.co/datasets/jsulz/FIFA23">SoFIFA history mirror</a>. Dates are archive effective dates, not independently proven publication timestamps. No verifiable historical EA Live Form archive was found.</p>'
    js='const rows='+json.dumps(curves)+';'+"""
const fields=['split','model','league','method'];const choices={};
for(const k of fields){let s=document.createElement('select');s.setAttribute('aria-label',k);for(const v of new Set(rows.map(r=>r[k]))){s.add(new Option(v,v))}if(k==='split')s.value='Test 2022-23';s.onchange=draw;choices[k]=s;document.querySelector('#controls').append(s)}
function draw(){let lines=rows.filter(r=>fields.every(k=>r[k]===choices[k].value));let dates=[...new Set(lines.flatMap(r=>r.points.map(p=>p.date)).filter(d=>d!=='Start'))].sort();let values=lines.flatMap(r=>r.points.map(p=>p.net));let lo=Math.min(0,...values),hi=Math.max(0,...values);if(hi===lo)hi=lo+1;let y=v=>330-(v-lo)/(hi-lo)*290;let x=d=>d==='Start'?80:80+(dates.indexOf(d)+1)/Math.max(1,dates.length)*880;let svg='';for(let i=0;i<5;i++){let v=lo+(hi-lo)*i/4;svg+=`<line x1="80" x2="960" y1="${y(v)}" y2="${y(v)}" stroke="#ddd"/><text x="0" y="${y(v)}">$${v.toFixed(0)}</text>`}let colors=['#777','#d28b19','#006c80'];lines.forEach((r,i)=>{svg+=`<polyline fill="none" stroke="${colors[i]}" stroke-width="2" points="${r.points.map(p=>`${x(p.date)},${y(p.net)}`).join(' ')}"/>`});svg+=`<text x="80" y="365">${dates[0]||''}</text><text x="860" y="365">${dates.at(-1)||''}</text>`;document.querySelector('#chart').innerHTML=svg;document.querySelector('#legend').innerHTML=lines.map((r,i)=>`<span style="color:${colors[i]}">${r.variant}: $${r.points.at(-1).net.toFixed(2)}</span>`).join(' | ')}draw();
"""
    (OUT/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Historical FIFA updates</title><style>body{font:15px system-ui;max-width:1250px;margin:30px auto;padding:20px;color:#182532}table{border-collapse:collapse;font-size:12px}th,td{padding:7px;border-bottom:1px solid #ddd}details{overflow:auto;margin:20px 0}select{padding:10px;margin:5px}svg{width:100%}</style>'+body+'<script>'+js+'</script>',encoding='utf-8')

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
