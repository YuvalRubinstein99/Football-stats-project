"""Reconcile the saved bet ledger against all aggregate result tables."""
from backtest_match_form import OUT, MODES, np, pd


def main():
    keys = ['family', 'variant', 'split', 'strategy', 'league', 'staking']
    partials = []
    count = 0
    for chunk in pd.read_csv(OUT / 'ledger.csv.gz', chunksize=100000):
        weights = chunk[['home_weight', 'draw_weight', 'away_weight']].to_numpy()
        odds = chunk[['B365H', 'B365D', 'B365A']].to_numpy()
        np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-12)
        assert (weights >= 0).all()
        outcome = chunk.outcome.to_numpy(int)
        payout = (weights * odds)[np.arange(len(chunk)), outcome]
        np.testing.assert_allclose(payout, chunk.unit_return, atol=1e-12)
        np.testing.assert_array_equal(chunk.original_stake, 1)
        np.testing.assert_array_equal(chunk.eligible_stake, (chunk.tier_stake > 0).astype(int))
        assert chunk.tier_stake.isin(range(6)).all()
        for mode, prefix, col in zip(MODES, ['original', 'eligible', 'tier'],
                                     ['original_stake', 'eligible_stake', 'tier_stake']):
            stake = chunk[col].to_numpy()
            gross = stake * payout
            net = gross - stake
            np.testing.assert_allclose(net, chunk[prefix + '_net'], atol=1e-12)
            data = chunk[keys[:-1]].copy()
            data['staking'] = mode
            data['bets'] = (stake > 0).astype(int)
            data['wins'] = ((stake > 0) & (payout > 0)).astype(int)
            data['total_staked'] = stake
            data['gross_return'] = gross
            data['net_profit'] = net
            partials.append(data.groupby(keys).sum())
            data['league'] = 'ALL'
            partials.append(data.groupby(keys).sum())
        count += len(chunk)
    actual = pd.concat(partials).groupby(level=keys).sum()
    summary = pd.read_csv(OUT / 'summary.csv').set_index(keys)
    assert summary.index.is_unique
    assert len(actual.index.difference(summary.index)) == 0
    actual = actual.reindex(summary.index, fill_value=0)
    for col in actual.columns:
        np.testing.assert_allclose(actual[col], summary[col], atol=1e-8)
    roi = np.divide(actual.net_profit, actual.total_staked,
                    out=np.zeros(len(actual)), where=actual.total_staked > 0)
    np.testing.assert_allclose(roi, summary.roi, atol=1e-12)
    print(f'Reconciled {count:,} ledger rows against {len(summary):,} summaries.')


if __name__ == '__main__':
    main()
