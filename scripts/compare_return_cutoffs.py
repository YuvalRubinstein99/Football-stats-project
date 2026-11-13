"""Sensitivity analysis of fixed expected-return cutoffs on frozen probabilities."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
from backtest_match_form import ROOT, OUT, load_inputs, pd, np
from bankroll_backtest import simulate, METHODS

CUTOFFS = [0., .01, .02, .03, .05, .10]
DEST = OUT / 'return_cutoffs'


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    predictions, sources = load_inputs()
    predictions = predictions[predictions.variant.isin(
        ['minimum_cv_form_allowed', 'minimum_cv_without_form'])]
    plan = dict(cutoffs=CUTOFFS, initial_bankroll=1000., methods=METHODS,
        selection='Highest nominal expected-return H/D/A selection, then require '
                  'probability * decimal odds > 1 + cutoff. No quoted-odds restriction. '
                  'No substitution when the selected outcome fails.',
        sizing='Flat $1, quarter Kelly, robust quarter Kelly with a 0.02 probability reduction. '
               'Nominal return gate before robust sizing. 2% match / 10% daily cash caps; '
               'same-day stakes reserved before settlement. Independent $1000 per simulation.',
        interpretation='Sensitivity comparison only; does not select or change the active 2% cutoff. '
                       'Frozen chronological CV choices, same fixtures, no refitting. '
                       'Previously inspected 2021-22 audit and 2022-23 test. Historical Bet365 odds; '
                       'no fees or rounding. Best realized cutoff is a hindsight result.',
        hashes={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__), ROOT/'src/bankroll_backtest.py',
                          ROOT/'results/match_form/predictions.csv']})
    (DEST/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    rows, curves, ledger_count = [], [], 0
    with gzip.open(DEST/'ledger.csv.gz','wt',encoding='utf-8',newline='') as handle:
        writer = None
        for (family,variant,split), group in predictions.groupby(['family','variant','split']):
            expected = sources['test'] if split=='test_2022' else sources['train'].query('season == 2021')
            assert group.source_row.is_unique and set(group.source_row)==set(expected.index)
            frame = expected.loc[group.source_row].copy().reset_index()
            assert frame.fixture_date.tolist()==group.date.tolist()
            assert frame.league.tolist()==group.league.tolist()
            outcomes = (1-np.sign(frame.home_score-frame.away_score)).to_numpy(int)
            np.testing.assert_array_equal(outcomes,group.outcome)
            p = group[['p_home','p_draw','p_away']].to_numpy()
            source_ids = frame.source_row.to_numpy(int)
            for method in METHODS:
                previous_bets = len(frame)+1
                for cutoff in CUTOFFS:
                    result, ledger = simulate(frame,p,method,initial=1000.,min_edge=cutoff)
                    points = result.pop('points')
                    meta = dict(family=family,variant=variant,split=split,method=method,cutoff=cutoff)
                    detail = pd.DataFrame(ledger)
                    ids = detail.source_row.to_numpy(int)
                    sides = detail.side.map({'home':0,'draw':1,'away':2}).to_numpy()
                    net = detail.stake*(np.where(sides==outcomes[ids],detail.odds,0)-1)
                    np.testing.assert_allclose(net,detail.net,atol=1e-9)
                    daily = detail.groupby('date',sort=True)[['stake','net']].sum()
                    opening = 1000.+daily.net.cumsum().shift(1,fill_value=0)
                    np.testing.assert_allclose(detail.opening_bankroll,detail.date.map(opening),atol=1e-9)
                    assert (daily.stake<=opening*.10+1e-9).all()
                    assert (detail.stake<=detail.opening_bankroll*.02+1e-9).all()
                    placed = detail[detail.stake>0]
                    assert (placed.probability*placed.odds>1+cutoff).all()
                    assert result['bets']==len(placed)<=previous_bets
                    previous_bets=len(placed)
                    np.testing.assert_allclose(1000+net.sum(),result['ending_bankroll'],atol=1e-9)
                    balances = np.r_[1000.,1000.+daily.net.cumsum().to_numpy()]
                    dd = (1-balances/np.maximum.accumulate(balances)).max()
                    np.testing.assert_allclose(dd,result['max_drawdown'],atol=1e-10)
                    rows.append(dict(**meta,**result))
                    curves.extend(dict(**meta,**point) for point in points)
                    for entry in ledger:
                        entry['source_row']=int(source_ids[entry['source_row']])
                        record=dict(**meta,**entry)
                        if writer is None:
                            writer=csv.DictWriter(handle,fieldnames=list(record))
                            writer.writeheader()
                        writer.writerow(record)
                    ledger_count+=len(ledger)
            print(f'{family} / {variant} / {split}: six cutoffs verified',flush=True)
    table=pd.DataFrame(rows)
    # Reproduce the previously reported zero- and two-percent experiments.
    keys=['family','variant','split','method']
    checks=['bets','staked','net','ending_bankroll','roi','max_drawdown']
    for cutoff,folder in [(0.,OUT),(.02,OUT/'edge2')]:
        reference=pd.read_csv(folder/'bankroll_summary.csv').set_index(keys)
        current=table[table.cutoff==cutoff].set_index(keys)
        np.testing.assert_allclose(current[checks],reference.loc[current.index,checks],atol=1e-8)
    table.to_csv(DEST/'summary.csv',index=False)
    pd.DataFrame(curves).to_csv(DEST/'curves.csv',index=False)
    (DEST/'verification.json').write_text(json.dumps(dict(
        simulations=len(table),ledger_rows=ledger_count,
        checks=['fixture alignment','threshold eligibility','monotonic bet counts',
                'daily exposure','settlement accounting','drawdown',
                'saved 0% and 2% result reproduction']),indent=2),encoding='utf-8')
    print(f'Verified {len(table)} simulations / {ledger_count:,} settlements.')


if __name__=='__main__':
    main()
