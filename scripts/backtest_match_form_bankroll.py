"""Replay existing cash-constrained staking rules; no refitting or optimization."""
import json
import hashlib
import argparse
from backtest_match_form import ROOT, OUT, load_inputs, pd, np
from bankroll_backtest import simulate, METHODS


def main(profile='baseline'):
    policies = dict(baseline=dict(min_edge=0., min_odds=1., max_odds=None),
                    edge2=dict(min_edge=.02, min_odds=1., max_odds=None),
                    odds4_edge2=dict(min_edge=.02, min_odds=4., max_odds=1000.))
    policy = policies[profile]
    output = OUT if profile == 'baseline' else OUT / profile
    output.mkdir(parents=True, exist_ok=True)
    predictions, sources = load_inputs()
    if profile != 'baseline':
        predictions = predictions[predictions.variant.isin(
            ['minimum_cv_form_allowed', 'minimum_cv_without_form'])]
    plan = dict(initial_bankroll=1000., methods=METHODS, profile=profile, filters=policy,
        selection='Select highest nominal expected return among H/D/A, then filter that pick. '
                  'Expected gross payout probability * decimal odds must strictly exceed 1 + min_edge; '
                  'quoted odds bounds are inclusive. No replacement pick if the original fails. '
                  'The return filter uses nominal probabilities; robust sizing then subtracts 0.02.',
        sizing='Quarter Kelly capped at 2% per match and 10% per day of opening bankroll. '
               'Flat $1 respects the same caps. Reserve all stakes before daily settlement.',
        limits='Frozen earlier chronological CV model choices; no retraining or threshold optimization. '
               'Previously inspected audit/test seasons. Historical Bet365 quotes, no costs or rounding. '
               'Each simulation independently starts at $1000; profits are not additive.',
        hashes={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [ROOT/'src/bankroll_backtest.py', ROOT/'scripts/backtest_match_form_bankroll.py',
                 ROOT/'results/match_form/predictions.csv']})
    (output/'bankroll_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    rows, ledgers, curves = [], [], []
    for (family, variant, split), group in predictions.groupby(['family', 'variant', 'split']):
        expected = sources['test'] if split == 'test_2022' else sources['train'].query('season == 2021')
        assert group.source_row.is_unique and set(group.source_row) == set(expected.index)
        frame = expected.loc[group.source_row].copy().reset_index()
        assert frame.fixture_date.tolist() == group.date.tolist()
        np.testing.assert_array_equal(1 - np.sign(frame.home_score - frame.away_score), group.outcome)
        p = group[['p_home', 'p_draw', 'p_away']].to_numpy()
        for method in METHODS:
            result, ledger = simulate(frame, p, method, initial=1000., **policy)
            points = result.pop('points')
            meta = dict(family=family, variant=variant, split=split, method=method)
            # Reconcile every settlement day and enforce opening-cash exposure limits.
            detail = pd.DataFrame(ledger)
            ids = detail.source_row.to_numpy(int)
            sides = detail.side.map({'home':0, 'draw':1, 'away':2}).to_numpy()
            outcomes = group.outcome.to_numpy()[ids]
            net = detail.stake * (np.where(sides == outcomes, detail.odds, 0)-1)
            np.testing.assert_allclose(net, detail.net, atol=1e-9)
            daily = detail.groupby('date',sort=True)[['stake','net']].sum()
            opening = 1000.+daily.net.cumsum().shift(1,fill_value=0)
            np.testing.assert_allclose(detail.opening_bankroll,detail.date.map(opening),atol=1e-9)
            assert (daily.stake <= opening*.10+1e-9).all()
            assert (detail.stake <= detail.opening_bankroll*.02+1e-9).all()
            placed = detail[detail.stake > 0]
            assert (placed.probability*placed.odds > 1+policy['min_edge']).all()
            assert (placed.odds >= policy['min_odds']).all()
            if policy['max_odds'] is not None:
                assert (placed.odds <= policy['max_odds']).all()
            np.testing.assert_allclose(1000.+net.sum(), result['ending_bankroll'], atol=1e-9)
            np.testing.assert_allclose(detail.net.sum(), result['net'], atol=1e-9)
            rows.append(dict(**meta, initial_bankroll=1000., **result))
            original_ids = frame.source_row.to_numpy(int)
            for entry in ledger:
                entry['source_row'] = int(original_ids[entry['source_row']])
                ledgers.append(dict(**meta, **entry))
            curves.extend(dict(**meta, **point) for point in points)
        print(f'{profile}: {family} / {variant} / {split}', flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(output / 'bankroll_summary.csv', index=False)
    pd.DataFrame(ledgers).to_csv(output / 'bankroll_ledger.csv.gz', index=False, compression='gzip')
    pd.DataFrame(curves).to_csv(output / 'bankroll_curves.csv', index=False)
    selected = table[(table.variant == 'minimum_cv_form_allowed') & (table.split == 'test_2022')]
    print(selected.to_string(index=False))
    print(f'Verified {len(rows)} simulations and {len(ledgers):,} match settlements.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['baseline','edge2','odds4_edge2'], default='baseline')
    main(parser.parse_args().profile)
