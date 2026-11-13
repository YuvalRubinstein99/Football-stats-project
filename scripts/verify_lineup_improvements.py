"""Reconcile exported forecasts, coverage, dates, source hashes and model selection."""
import hashlib
import json
from select_expanded_features import ROOT, pd, np
from calibrate_models import scores

OUT = ROOT / 'results/lineup_improvements'


def main():
    plan = json.loads((OUT / 'plan.json').read_text())
    for name,digest in plan['input_hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    forecasts = pd.read_csv(OUT/'predictions.csv')
    metrics = pd.read_csv(OUT/'metrics.csv')
    assert len(forecasts)==34080
    assert not forecasts.duplicated(['family','candidate','split','source_row']).any()
    for (family,candidate,split),g in forecasts.groupby(['family','candidate','split']):
        assert len(g)==1065
        p=g[['p_home','p_draw','p_away']].to_numpy()
        assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(axis=1),1)
        y=(1-np.sign(g.home_score-g.away_score)).to_numpy(int)
        row=metrics[(metrics.family==family)&(metrics.candidate==candidate)&(metrics.split==split)].iloc[0]
        for key,value in scores(p,y).items():
            assert np.isclose(value,row[key]),(family,candidate,split,key)
        assert np.isclose(row.accuracy,np.mean(p.argmax(axis=1)==y))
        actual=g[['home_score','away_score']].to_numpy()
        predicted=g[['expected_home_goals','expected_away_goals']].to_numpy()
        assert np.isclose(row.goal_mae,np.abs(predicted-actual).mean())
    audit=pd.read_csv(OUT/'dynamic_fit_audit.csv')
    dated=audit.dropna(subset=['last_result_date'])
    assert (pd.to_datetime(dated.last_result_date)<pd.to_datetime(dated.cutoff)).all()
    assert (audit.max_gradient<=.02).all()
    assert audit.training_matches.is_monotonic_increasing
    selected=json.loads((OUT/'selection.json').read_text())
    cv=pd.read_csv(OUT/'selection_scores.csv')
    assert set(cv.season)=={2019,2020}
    for family,choice in selected['selection'].items():
        group=cv[cv.family==family].groupby(['candidate','alpha']).log_loss.mean()
        assert group.idxmin()==(choice['candidate'],choice['alpha'])
    lineups=json.loads((OUT/'lineups.json').read_text())
    for split in ['train','test']:
        base=pd.read_csv(ROOT/f'results/positional_matchups/{split}.csv')
        frame=pd.read_csv(OUT/f'{split}_features.csv')
        cols=['source_row','fixture_date','league','home_team_name','away_team_name','home_score','away_score']
        pd.testing.assert_frame_equal(base[cols],frame[cols])
        for r in frame.itertuples():
            players=lineups[f'{split}:{r.source_row}']
            ids=[p['player_id'] for values in players.values() for p in values]
            assert len(ids)==len(set(ids))
            assert len(players['home'])==r.home_position_matched_count
            assert len(players['away'])==r.away_position_matched_count
    print(f'Verified {len(forecasts):,} predictions, all metrics, {len(audit)} monthly fits, '
          'selection, identical samples and unique starter identities.')


if __name__=='__main__':
    main()
