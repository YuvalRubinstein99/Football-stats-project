"""Forward predictions on training seasons, with fixed and nested RF selection."""
import hashlib
import json
from pathlib import Path
from select_expanded_features import ROOT, np, pd, joblib, threadpool_limits
from compare_lineup_improvements import add_controls, candidates
from compare_positional_matchups import chronological_split
from home_win_model import fit, predict, target
from train_binary_home_win import metrics
from binary_betting import selections, settle, summary
from plot_binary_home_filters import RULES

DATA = ROOT/'results/lineup_improvements'
OUT = DATA/'binary_retrained/training_oos'


def choose_candidate(records, year, names):
    """Choose on the two immediately preceding forward validation seasons only."""
    table = pd.DataFrame(records)
    years = [year-2, year-1]
    past = table[table.season.isin(years)]
    if len(past) != len(names)*2 or past.duplicated(['season','candidate']).any():
        raise ValueError('Two complete earlier validation seasons required')
    if set(past.candidate) != set(names):
        raise ValueError('Candidate grid differs')
    means = past.groupby('candidate').log_loss.mean()
    return min(names, key=lambda name: (means[name], name)), years


def portfolios(frame, protocol):
    frame = frame.sort_values(['fixture_date','league','source_row']).reset_index(drop=True)
    ph = frame.p_home_win.to_numpy()
    odds = frame[['B365H','B365D','B365A']].to_numpy(float)
    outcomes = np.where(frame.home_score>frame.away_score,0,
                        np.where(frame.home_score==frame.away_score,1,2))
    rules = selections(ph,odds)
    ledger, summaries = [], []
    for strategy, _, _ in RULES:
        for staking in ['flat','tiers']:
            result = settle(ph,odds,outcomes,rules[strategy],staking)
            rows = frame.copy()
            rows['protocol'],rows['strategy'],rows['staking'] = protocol,strategy,staking
            for dest,source in [('stake','stakes'),('net','profit'),('bet','bet'),('win','win'),('edge','edge')]:
                rows[dest] = result[source]
            rows['expected_net'] = rows.stake*rows.edge
            rows['cumulative_net'] = rows.net.cumsum()
            ledger.append(rows)
            record = dict(protocol=protocol,strategy=strategy,staking=staking,**summary(result,outcomes))
            daily = rows.groupby('fixture_date').net.sum().cumsum()
            daily = np.r_[0.,daily.to_numpy()]
            record['daily_max_drawdown'] = float((np.maximum.accumulate(daily)-daily).max())
            record['expected_net'] = float(rows.expected_net.sum())
            summaries.append(record)
    return pd.concat(ledger,ignore_index=True),summaries


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cache = OUT/'cache';cache.mkdir(exist_ok=True)
    prior = json.loads((DATA/'binary_retrained/plan.json').read_text())
    for path,digest in prior['input_hashes'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Prior binary experiment input changed: {path}')
    frame = add_controls(pd.read_csv(DATA/'train_features.csv'),'train')
    base = joblib.load(ROOT/'results/expanded_comparison/expanded__RF_vector.joblib')
    sets = candidates(base['categorical']+base['numeric'])
    original_selection = json.loads((DATA/'binary_retrained/selection.json').read_text())
    fixed = original_selection['selection']['RF_binary']['candidate']
    years = sorted(int(y) for y in frame.season.unique())
    assert years == list(range(2014,2022)) and frame.source_row.is_unique
    paths = [DATA/'train_features.csv',DATA/'train.csv',
             ROOT/'results/positional_matchups/train.csv',ROOT/'src/home_win_model.py',
             ROOT/'src/lineup_history.py',ROOT/'scripts/compare_lineup_improvements.py',
             ROOT/'src/binary_betting.py',Path(__file__)]
    hashes = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    plan = dict(family='RF_binary',trees=500,candidates=sets,fixed_candidate=fixed,
        prediction_seasons=years[1:],nested_evaluation_seasons=years[3:],
        fixed='Same final selected feature specification; preprocessing and RF refitted before each season. '
              'Feature choice is retrospective, so this is not an independent selection test.',
        nested='Before each season, select features by mean binary log loss in the previous two '
               'forward validation seasons; refit on all earlier seasons. No betting-return tuning.',
        warmup='2014-15 trains the first forecast. 2015-16 and 2016-17 initialize nested selection.',
        historical_features='Pre-match lineup history and monthly player effects may update during '
               'each evaluation season using only already completed matches; classifier refits annually.',
        caveat=prior['caveat']+' The candidate designs and RF hyperparameters were developed previously; '
               'nested evaluation removes future-season feature selection within this declared grid, '
               'but cannot undo prior researcher exposure to these historical seasons.',
        betting='Home only; fixed EV >0% and >2% filters; existing flat and 1-5-unit stakes. '
                'Draws count as losses. No compounding, fees or bankroll cap.',input_hashes=hashes)
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    validation, folds, forecasts = [], [], []
    for year in years[1:]:
        tr,va = chronological_split(frame,year)
        chosen,selection_years = (choose_candidate(validation,year,list(sets))
                                  if year>=2017 else (None,[]))
        print(f'Season {year}: train={len(tr)}, forecast={len(va)}, nested={chosen}',flush=True)
        fold = dict(season=year,train_matches=len(tr),forecast_matches=len(va),
                    trained_through=str(tr.fixture_date.max()),forecast_start=str(va.fixture_date.min()),
                    fixed_candidate=fixed,nested_candidate=chosen,selection_seasons=selection_years)
        folds.append(fold)
        # Record the decision before fitting/evaluating the current season.
        (OUT/'folds.json').write_text(json.dumps(folds,indent=2),encoding='utf-8')
        for candidate,columns in sets.items():
            signature = dict(hashes=hashes,season=year,candidate=candidate,columns=columns,trees=500)
            key = hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
            path = cache/(key+'.npz')
            if path.exists():
                with np.load(path) as saved:
                    assert np.array_equal(saved['source_row'],va.source_row.to_numpy())
                    p = saved['p']
            else:
                print(f'  Fitting {candidate}',flush=True)
                model = fit(tr,columns,'RF_binary',500)
                p = predict(model,va)
                np.savez_compressed(path,p=p,source_row=va.source_row.to_numpy())
            validation.append(dict(season=year,candidate=candidate,**metrics(p,va)))
            if candidate in [fixed,chosen]:
                cols = ['source_row','season','fixture_date','league','home_team_name','away_team_name',
                        'home_score','away_score','B365H','B365D','B365A']
                export = va[cols].copy()
                export['p_home_win'],export['candidate'] = p[:,1],candidate
                export['trained_through'] = tr.fixture_date.max()
                if candidate==fixed:
                    forecasts.append(export.assign(protocol='fixed'))
                if candidate==chosen:
                    forecasts.append(export.assign(protocol='nested'))
        pd.DataFrame(validation).to_csv(OUT/'validation.csv',index=False)
    predictions = pd.concat(forecasts,ignore_index=True)
    assert not predictions.duplicated(['protocol','source_row']).any()
    assert (pd.to_datetime(predictions.trained_through)<pd.to_datetime(predictions.fixture_date)).all()
    predictions.to_csv(OUT/'predictions.csv',index=False)
    ledgers,records,quality = [],[],[]
    for protocol,g in predictions.groupby('protocol'):
        ledger,totals = portfolios(g,protocol)
        ledgers.append(ledger);records.extend(totals)
        ph = g.p_home_win.to_numpy()
        market = 1/g[['B365H','B365D','B365A']].to_numpy(float)
        market = market[:,0]/market.sum(axis=1)
        for name,prob in [('RF',ph),('normalized_market',market)]:
            quality.append(dict(protocol=protocol,model=name,**metrics(np.column_stack([1-prob,prob]),g)))
    ledger = pd.concat(ledgers,ignore_index=True)
    ledger.to_csv(OUT/'ledger.csv.gz',index=False)
    pd.DataFrame(records).to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(quality).to_csv(OUT/'prediction_metrics.csv',index=False)
    for dimension in ['season','league']:
        grouped = ledger.groupby(['protocol','strategy','staking',dimension]).agg(
            matches=('net','size'),bets=('bet','sum'),wins=('win','sum'),staked=('stake','sum'),
            net=('net','sum'),expected_net=('expected_net','sum')).reset_index()
        grouped['roi'] = grouped.net/grouped.staked
        grouped.to_csv(OUT/f'by_{dimension}.csv',index=False)
    # Fixed bins expose where predicted edges and realized payoffs diverge.
    active = ledger[ledger.bet & ledger.staking.eq('flat')].copy()
    active['odds_band'] = pd.cut(active.B365H,[1,2,3,5,np.inf],right=False).astype(str)
    bins = active.groupby(['protocol','strategy','odds_band']).agg(
        bets=('bet','size'),predicted_home_rate=('p_home_win','mean'),win_rate=('win','mean'),
        mean_odds=('B365H','mean'),net=('net','sum'),expected_net=('expected_net','sum')).reset_index()
    bins['roi'] = bins.net/bins.bets
    bins.to_csv(OUT/'odds_diagnostics.csv',index=False)
    print(pd.DataFrame(records).to_string(index=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):
        main()
