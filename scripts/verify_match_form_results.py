"""Independently reconcile saved probabilities, metrics and temporal audits."""
import json
from select_expanded_features import ROOT, pd, np
from compare_team_attributes import labels
from calibrate_models import scores

OUT = ROOT / 'results/match_form'


def main():
    predictions = pd.read_csv(OUT / 'predictions.csv')
    metrics = pd.read_csv(OUT / 'metrics.csv')
    prepared = {s: pd.read_csv(OUT / (s + '.csv')) for s in ['train', 'test']}
    for frame in prepared.values():
        cutoff = pd.to_datetime(frame.fixture_date)
        for side in ['home', 'away']:
            latest = pd.to_datetime(frame[side + '_recent_latest_date'])
            assert (latest.isna() | (latest < cutoff)).all()
    checked = 0
    for (family, variant, split), group in predictions.groupby(['family', 'variant', 'split']):
        source = prepared['test'] if split == 'test_2022' else prepared['train'].query('season == 2021')
        expected = source.set_index('source_row').loc[group.source_row]
        assert group.source_row.is_unique
        assert set(group.source_row) == set(source.source_row)
        assert expected.fixture_date.tolist() == group.date.tolist()
        np.testing.assert_array_equal(labels(expected), group.outcome.to_numpy())
        p = group[['p_home', 'p_draw', 'p_away']].to_numpy()
        assert np.isfinite(p).all() and (p >= 0).all()
        np.testing.assert_allclose(p.sum(axis=1), 1)
        for league in ['ALL', *sorted(group.league.unique())]:
            mask = np.ones(len(group), bool) if league == 'ALL' else (group.league == league).to_numpy()
            recalculated = scores(p[mask], group.outcome.to_numpy()[mask])
            saved = metrics[(metrics.family == family) & (metrics.variant == variant)
                            & (metrics.split == split) & (metrics.league == league)]
            assert len(saved) == 1 and saved.iloc[0].matches == int(mask.sum())
            for name, value in recalculated.items():
                np.testing.assert_allclose(value, saved.iloc[0][name], atol=1e-12)
            checked += 1
    status = dict(passed=True, reconciled_metric_rows=checked,
                  probability_rows=len(predictions),
                  train_temporal_rows=len(prepared['train']), test_temporal_rows=len(prepared['test']))
    (OUT / 'verification.json').write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
