"""All-training fitted-value gains vs held-out test, with RF OOB kept separate."""
import json,hashlib,html
from select_expanded_features import ROOT,BASE,np,pd,joblib,threadpool_limits,transform,predict,metrics,bets,season_years,SEARCH_CANDIDATE
from compare_expanded_models import outcome_probabilities

OUT=ROOT/'results/team_identity'


def main():
    status=json.loads((OUT/'status.json').read_text())
    for path,digest in status['input_hashes'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    train=pd.read_csv(ROOT/'data/expanded_2008/train.csv');test=pd.read_csv(ROOT/'data/expanded_2008/test.csv')
    years=season_years(train)
    predictions=pd.read_csv(OUT/'match_predictions.csv')
    for c in predictions.columns[:7]:assert np.array_equal(predictions[c],test[c]),c
    cv=pd.read_csv(OUT/'model_metrics.csv').set_index('model')
    models=list(status['fits'])+['Expanded RF']
    curve_rows=[];summaries=[];quality=[]
    for name in models:
        print('Training gains: '+name,flush=True)
        if name=='Expanded RF':
            bundle=joblib.load(BASE/'expanded__RF_vector.joblib')
            x=np.column_stack([bundle['encoder'].transform(train[bundle['categorical']]),train[bundle['numeric']].to_numpy()])
            fitted=outcome_probabilities(bundle['model'].predict(x))
            oob=outcome_probabilities(bundle['model'].oob_prediction_)
        else:
            bundle=joblib.load(OUT/(name+'.joblib'));x,xo=transform(bundle,train)
            fitted,_=predict(bundle['fit'],x,xo)
        held=predictions[[name+'__p_'+s for s in ['home','draw','away']]].to_numpy()
        datasets=[('Train (in-sample)',train,fitted,years),('Test',test,held,np.full(len(test),2022))]
        if name=='Expanded RF':datasets.append(('Train (OOB)',train,oob,years))
        for split,frame,p,season in datasets:
            labels=(1-np.sign(frame.home_score-frame.away_score)).to_numpy(int)
            q=metrics(name,p,labels);q.update(split=split,matches=len(frame));quality.append(q)
            totals,ledger=bets(name,p,frame,labels)
            for row in totals:row.update(split=split,matches=len(frame))
            summaries.extend(totals)
            data=pd.DataFrame(ledger);del ledger
            data['season']=season[data['row'].to_numpy(int)]
            groups=data.groupby(['strategy','season','week'],sort=True)[['net','stake']].sum().reset_index()
            for strategy,part in groups.groupby('strategy',sort=False):
                points=[dict(label='Start',net=0.,staked=0.)]
                net=0.;staked=0.
                for r in part.itertuples():
                    net+=r.net;staked+=r.stake
                    points.append(dict(label=f'{r.season}–{r.season+1} W{r.week}',net=net,staked=staked))
                target=next(r for r in totals if r['strategy']==strategy)
                assert abs(net-target['net'])<1e-7 and abs(staked-target['staked'])<1e-7
                curve_rows.append(dict(model=name,split=split,strategy=strategy,points=points))
            del data
    summary=pd.DataFrame(summaries);q=pd.DataFrame(quality)
    previous=pd.read_csv(OUT/'strategy_results.csv')
    merged=summary[summary.split=='Test'].merge(previous,on=['model','strategy'],suffixes=('_new','_old'))
    assert len(merged)==len(previous)==135
    for c in ['net','staked','flat_net','bets']:np.testing.assert_allclose(merged[c+'_new'],merged[c+'_old'],atol=1e-7)
    # Independently reconcile the OOB baseline against the existing interface.
    existing=json.loads((ROOT/'results/expanded_interface/stake_cumulative.json').read_text())
    for curve in existing['curves']:
        if curve['split']!='Train (OOB)':continue
        expected=sum((i+1)*level[1] for point in curve['points'] for i,level in enumerate(point['levels']))
        actual_row=summary[(summary.model=='Expanded RF')&(summary.split=='Train (OOB)')&(summary.strategy==curve['strategy'])].iloc[0]
        assert abs(actual_row.net-expected)<1e-7
    diagnostics=[]
    for name in models:
        tr=q[(q.model==name)&(q.split=='Train (in-sample)')].iloc[0]
        te=q[(q.model==name)&(q.split=='Test')].iloc[0]
        diagnostics.append(dict(model=name,train_log_loss=tr.log_loss,forward_cv_log_loss=float(cv.loc[name].cv_log_loss) if name!='Expanded RF' else np.nan,test_log_loss=te.log_loss,test_minus_train=te.log_loss-tr.log_loss,train_auc=tr.auc_macro,test_auc=te.auc_macro))
    diag=pd.DataFrame(diagnostics)
    summary.to_csv(OUT/'train_test_strategy_results.csv',index=False)
    q.to_csv(OUT/'train_test_metrics.csv',index=False);diag.to_csv(OUT/'overfitting_diagnostics.csv',index=False)
    payload=dict(curves=curve_rows,summary=summaries,models=models,default_strategy=SEARCH_CANDIDATE)
    (OUT/'train_test_curves.json').write_text(json.dumps(payload),encoding='utf-8')
    table=lambda frame:frame.to_html(index=False,float_format=lambda x:f'{x:,.5f}')
    body='<h1>All-training cumulative profit and overfitting</h1><p>19,263 training matches, 2008–09 through 2021–22; 1,447 test matches, 2022–23. Every regression retains FIFA, form, prior performance, odds and matchweek. Compare models with and without club identity.</p>'
    body+='<p><strong>Train (in-sample) uses fitted predictions on the same rows used to train each model, including the RF.</strong> RF OOB is shown separately: each row uses trees that omitted it, but those trees can use later seasons. Neither training curve is a forward betting simulation. Test predictions are excluded from fitting; this test season has already been inspected. Cumulative dollars cover different numbers of matches, so compare ROI and prediction losses as well.</p>'
    body+='<h2>Prediction-loss diagnostics</h2><p>Lower loss is better. A large train-to-validation/test gap is consistent with overfitting, but season changes also contribute. The within-family change from adding club identity is the more controlled comparison.</p>'+table(diag)
    body+='<h2>Cumulative net profit</h2><label>Strategy <select id="tg-strategy"></select></label> <label>Models <select id="tg-models"><option value="teams">With teams + RF</option><option value="no_teams">Without teams + RF</option><option value="all">All models</option></select></label><div id="tg-charts"></div><h2>Accounting for selected strategy</h2><div id="tg-table"></div>'
    body+='<p>Each curve starts at $0 and carries across every included season in season/matchweek order. Results within the same week are aggregated. Same fixed $1–$5 stake rules; net returns subtract stakes. No fees, bankroll limits or compounding. No model was retrained for this diagnostic.</p>'
    body+='<details><summary>All split prediction metrics, including RF OOB</summary>'+table(q)+'</details>'
    body+='<script id="tg-data" type="application/json">'+json.dumps(payload).replace('<','\\u003c')+'</script>'
    body+=(ROOT/'src/training-gains-ui.js').read_text(encoding='utf-8')
    doc='<!doctype html><meta charset="utf-8"><title>Train and test cumulative profit</title><style>body{font:15px system-ui;margin:30px;color:#182532}table{border-collapse:collapse;font-size:12px}td,th{padding:8px;border-bottom:1px solid #ddd}svg{width:100%;max-width:1100px}select{padding:8px;margin:8px}details{margin:20px 0}</style>'+body
    (OUT/'training_gains.html').write_text(doc,encoding='utf-8')
    for path in [BASE/'report.html',OUT/'report.html']:
        prior=path.read_text(encoding='utf-8').split('<!-- training-gains -->')[0]
        if 'href="#training-gains"' not in prior:
            prior=prior.replace('<h1>Do home and away team identities help?</h1>','<h1>Do home and away team identities help?</h1><p><a href="#training-gains">NEW: all-training and test cumulative profit / overfitting</a></p>')
        path.write_text(prior+'<!-- training-gains --><hr><section id="training-gains">'+body+'</section>',encoding='utf-8')
    print(diag.to_string(index=False))
    print(summary[summary.strategy==SEARCH_CANDIDATE][['model','split','bets','staked','net','roi']].to_string(index=False))


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
