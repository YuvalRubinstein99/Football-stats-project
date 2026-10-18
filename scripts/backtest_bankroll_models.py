"""All saved model variants, fixed staking rules, actual-date settlement."""
import sys,json,hashlib,gzip,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from model_strategy_app import pd,np,joblib,model_path
from train_recent_models import bundle_predict
from probability_calibration import scale,artifacts
from bankroll_backtest import METHODS,simulate
from threadpoolctl import threadpool_limits
OUT=ROOT/'results/bankroll_backtest'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    frames={s:pd.read_csv(ROOT/f'results/classifier/{s}.csv') for s in ['train','test']}
    for split,f in frames.items():
        original=pd.read_csv(ROOT/f'data/expanded_2008/{split}.csv')
        for c in ['home_team_name','away_team_name','home_score','away_score','B365H','B365D','B365A']:assert np.array_equal(f[c],original[c]),c
        pd.to_datetime(f.fixture_date,errors='raise')
    plan=dict(initial_bankroll=1000,flat_stake=1,kelly_multiplier=.25,robust_probability_reduction=.02,per_match_cap=.02,daily_exposure_cap=.10,
        selection='Highest positive expected net return, one H/D/A selection per match; same selection across methods. Robust method may abstain.',
        robust='Quarter of worst-case binary Kelly with selected probability in [max(0,p-0.02),min(1,p+0.02)]. Fixed ambiguity radius; not an estimated confidence interval or tuned optimum.',
        settlement='All same-date stakes use opening daily bankroll. Scale proportionally to 10% total exposure. Settle together at day end. Fractional amounts without minimum stake/rounding. No fees, taxes or execution limits.',
        scope='Every saved equal/recency-weighted, raw/calibrated model; independent $1000 bankroll for each split and league. All-league portfolio simulated separately, not a sum of league profits. UI manual filters do not apply.',
        limitations='Train is in-sample, not evidence of future profitability. Test 2022–23 previously inspected. Historical actual lineups and Bet365 prices; not Winner returns. Original duplicate test fixture retained for matched comparisons. Daily drawdown excludes intraday swings.',
        input_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'results/classifier/train.csv',ROOT/'results/classifier/test.csv',Path(__file__),ROOT/'src/bankroll_backtest.py']})
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    results=[]
    with gzip.open(OUT/'ledger.csv.gz','wt',encoding='utf-8',newline='') as handle:
        writer=None
        for folder in ['calibration','recency','classifier']:
            calibration=artifacts(folder=='recency',folder=='classifier')
            saved=pd.read_csv(ROOT/f'results/{"team_identity" if folder=="calibration" else folder}/match_predictions.csv')
            for name,cal in calibration['models'].items():
                print('Backtesting '+name,flush=True)
                path=model_path(name);plan['input_hashes'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
                if folder=='classifier':
                    trainpred=pd.read_csv(ROOT/'results/classifier/train_predictions.csv');raw=trainpred[[name+'__p_'+s for s in ['home','draw','away']]].to_numpy()
                else:
                    bundle=joblib.load(path)
                    raw=bundle_predict(name.removesuffix('__recent'),bundle,frames['train'])
                    del bundle
                testp=saved[[name+'__p_'+s for s in ['home','draw','away']]].to_numpy()
                for calibrated in [False,True]:
                    label=name+('__calibrated' if calibrated else '')
                    for split,prob in [('train',raw),('test',testp)]:
                        f=frames[split];prob=scale(prob,cal['temperature']) if calibrated else prob
                        for league in ['All','E0','SP1','D1','I1']:
                            mask=np.ones(len(f),bool) if league=='All' else (f.league==league).to_numpy()
                            for method in METHODS:
                                result,rows=simulate(f.loc[mask],prob[mask],method)
                                meta=dict(model=label,split='Train (in-sample)' if split=='train' else 'Test',league=league,method=method)
                                results.append({**meta,**result})
                                for row in rows:
                                    row={**meta,**row,'portfolio_league':league}
                                    if writer is None:writer=csv.DictWriter(handle,fieldnames=list(row));writer.writeheader()
                                    writer.writerow(row)
    (OUT/'results.json').write_text(json.dumps(dict(plan=plan,results=results),separators=(',',':')))
    pd.DataFrame([{k:v for k,v in r.items() if k!='points'} for r in results]).to_csv(OUT/'summary.csv',index=False)
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    from build_staking_mixture_data import main as build_mixture_data
    build_mixture_data()
    print(f'Completed {len(results)} curves across {len(set(r["model"] for r in results))} model configurations.',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
