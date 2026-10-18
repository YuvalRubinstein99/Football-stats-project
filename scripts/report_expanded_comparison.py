"""Validate matched experiment outputs and add uncertainty, calibration and plots."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.lineup_deps'));sys.path.insert(0,str(ROOT/'src'))
import hashlib
import json
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import log_loss
from scipy.stats import skellam
from live_bets import integrated_recommendations,SEARCH_CANDIDATE
from stake_levels import size_bet
from compare_expanded_models import OUT,DATA,report

def main():
    status=json.loads((OUT/'status.json').read_text())
    assert status['state']=='completed'
    for name,value in status['input_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value
    preds=pd.read_csv(OUT/'match_predictions.csv');metrics=pd.read_csv(OUT/'model_metrics.csv');strategies=pd.read_csv(OUT/'strategy_results.csv')
    test=pd.read_csv(ROOT/'data/test.csv');assert len(preds)==len(test)==1447
    for col in ['home_team_name','away_team_name','home_score','away_score','B365H','B365D','B365A']:assert (preds[col].to_numpy()==test[col].to_numpy()).all(),col
    y=preds[['home_score','away_score']].to_numpy();labels=np.where(y[:,0]>y[:,1],0,np.where(y[:,0]<y[:,1],2,1))
    weeks=test.Matchweek.astype(str).str.split().str[-1].astype(int).to_numpy();unique=np.unique(weeks)
    rng=np.random.default_rng(20261011);samples=rng.integers(0,len(unique),size=(2000,len(unique)))
    differences=[];losses={};calibration=[];weekly={}
    for name in status['completed']:
        p=preds[[name+'__p_'+s for s in ['home','draw','away']]].to_numpy()
        assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(axis=1),1)
        assert abs(log_loss(labels,p,labels=[0,1,2])-float(metrics[metrics.model==name].iloc[0].log_loss))<1e-10
        goals=preds[[name+'__goals_'+s for s in ['home','away']]].to_numpy()
        losses[name]={'log_loss':-np.log(np.maximum(p[np.arange(len(p)),labels],1e-15)),'goal_mse':np.mean((goals-y)**2,axis=1)}
        for j,side in enumerate(['home','draw','away']):
            for b in range(10):
                selected=(p[:,j]>=b/10)&(p[:,j]<(b+1)/10 if b<9 else p[:,j]<=1)
                if selected.any():calibration.append(dict(model=name,outcome=side,bin=b,count=int(selected.sum()),predicted=float(p[selected,j].mean()),observed=float((labels[selected]==j).mean())))
        profits=[]
        for i,r in preds.iterrows():
            match=dict(home=r.home_team_name,away=r.away_team_name,p_home=p[i,0],p_draw=p[i,1],p_away=p[i,2],B365H=r.B365H,B365D=r.B365D,B365A=r.B365A)
            bet=integrated_recommendations(match)[-1];stake=size_bet(bet)['stake_dollars'];side=['home','draw','away'][labels[i]]
            odds=[r.B365H,r.B365D,r.B365A][labels[i]];profits.append(stake*(bet[side+'_stake']*odds-1))
        expected=strategies[(strategies.model==name)&(strategies.strategy==SEARCH_CANDIDATE)].iloc[0].net
        assert abs(sum(profits)-expected)<1e-7
        weekly[name]=[sum(v for v,w in zip(profits,weeks) if w==week) for week in unique]
    comparisons=[('expanded__'+family,'original__'+family) for family in ['RF_vector','RF_separate','Linear_vector','Ridge_vector','Poisson_alpha1','Poisson_alpha001']]
    comparisons += [('expanded__RF_vector','expanded__'+f) for f in ['Linear_vector','Ridge_vector','Poisson_alpha1','Poisson_alpha001']]
    for a,b in comparisons:
        for metric in ['log_loss','goal_mse']:
            delta=losses[a][metric]-losses[b][metric]
            sums=np.array([delta[weeks==w].sum() for w in unique]);counts=np.array([(weeks==w).sum() for w in unique])
            draws=sums[samples].sum(axis=1)/counts[samples].sum(axis=1)
            differences.append(dict(model=a,reference=b,metric=metric,difference=float(delta.mean()),ci_low=float(np.quantile(draws,.025)),ci_high=float(np.quantile(draws,.975))))
    pd.DataFrame(differences).to_csv(OUT/'paired_differences.csv',index=False);pd.DataFrame(calibration).to_csv(OUT/'calibration.csv',index=False)
    pd.DataFrame({'week':unique,**weekly}).to_csv(OUT/'weekly_profit.csv',index=False)
    status['verification']=dict(test_alignment=True,probabilities_and_metrics=True,legacy_profit_reconciled=True,original_inputs_unchanged=True,versions=dict(numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__))
    (OUT/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    report(metrics.to_dict('records'),strategies.to_dict('records'),status)
    from reportlab.graphics.shapes import Drawing,Rect,Line,String,PolyLine
    from reportlab.graphics import renderPDF,renderSVG
    from reportlab.lib.colors import HexColor
    import pypdfium2 as pdfium
    d=Drawing(1100,750);d.add(Rect(0,0,1100,750,fillColor=HexColor('#ffffff'),strokeColor=None))
    def text(x,y,s,size=12):d.add(String(x,y,s,fontSize=size,fillColor=HexColor('#172a3a')))
    text(30,715,'Does adding 2008–14 improve the models?',25)
    text(30,688,'Same 1,447 test matches from 2022–23 | Lower log loss is better',13)
    colors={'original':'#97abc0','expanded':'#167e72'}
    families=['RF_vector','RF_separate','Linear_vector','Ridge_vector','Poisson_alpha1','Poisson_alpha001']
    for i,f in enumerate(families):
        yy=635-i*42;text(30,yy,f)
        for j,period in enumerate(colors):
            value=float(metrics[metrics.model==period+'__'+f].iloc[0].log_loss)
            width=max(0,(value-.95)*5000);yp=yy-j*14
            d.add(Rect(220,yp-4,width,10,fillColor=HexColor(colors[period]),strokeColor=None));text(225+width,yp-3,f'{value:.5f}',10)
    text(30,356,'Bars start at 0.95. Grey: original training; green: expanded training.',11)
    text(30,325,'Fixed filtered Legacy variance: cumulative test profit ($1–$5 stakes)',17)
    chosen=['original__RF_vector','expanded__RF_vector','expanded__Linear_vector','expanded__Poisson_alpha001']
    lines={k:np.r_[0,np.cumsum(weekly[k])] for k in chosen};lo=min(0,min(v.min() for v in lines.values()));hi=max(0,max(v.max() for v in lines.values()));pad=max(10,(hi-lo)*.1);lo-=pad;hi+=pad
    palette=['#97abc0','#167e72','#b45d30','#7553a3']
    x=lambda i:100+i/len(unique)*920;ypos=lambda v:100+(v-lo)/(hi-lo)*180
    for v in np.linspace(lo,hi,5):d.add(Line(100,ypos(v),1020,ypos(v),strokeColor=HexColor('#dde3e8')));text(35,ypos(v),f'${v:.0f}',10)
    for i,(name,vals) in enumerate(lines.items()):
        d.add(PolyLine([c for j,v in enumerate(vals) for c in (x(j),ypos(v))],strokeColor=HexColor(palette[i]),strokeWidth=2,fillColor=None));text(30+i*270,67,name,10)
        d.add(Rect(30+i*270,81,20,4,fillColor=HexColor(palette[i]),strokeColor=None))
    text(100,85,'Start',10);text(950,85,'Matchweek 38',10)
    text(30,32,'Historical exploratory comparison; independent Poisson outcome conversion. Test set already inspected.',11)
    renderSVG.drawToFile(d,str(OUT/'comparison.svg'))
    with pdfium.PdfDocument(renderPDF.drawToString(d)) as doc:
        page=doc[0];bitmap=page.render(scale=1.5);bitmap.to_pil().save(OUT/'comparison.png');bitmap.close();page.close()
    path=OUT/'report.html';body=path.read_text(encoding='utf-8')
    body+='<h2>Paired differences (negative is better)</h2><p>Approximate 95% intervals: 2,000 paired resamples of matchweek groups. These do not capture all across-week/team dependence or strategy-selection uncertainty.</p>'+pd.DataFrame(differences).to_html(index=False,float_format=lambda x:f'{x:.5f}')+'<img src="comparison.png" style="max-width:100%" alt="Model losses and cumulative test profits">'
    path.write_text(body,encoding='utf-8')
    print(pd.DataFrame(differences).to_string(index=False));print('All matched-test and profit checks passed.')

if __name__=='__main__':main()
