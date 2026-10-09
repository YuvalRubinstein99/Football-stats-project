"""Check source syntax and the saved experiment without fitting any models."""
from pathlib import Path
import json
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from project_paths import DATA, OUT
import numpy as np
import pandas as pd


def main():
    for p in [ROOT / 'run.py', *ROOT.glob('src/*.py'), *ROOT.glob('scripts/*.py')]:
        compile(p.read_text(encoding='utf-8'), str(p), 'exec')
    notebook = json.loads((ROOT / 'notebooks/prediction_bivariate.ipynb').read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), 'notebook', 'exec')
    metrics = pd.read_csv(OUT / 'model_metrics.csv').set_index('model')
    predictions = pd.read_csv(OUT / 'match_predictions.csv')
    labels = np.sign(predictions.home_score-predictions.away_score).astype(int).to_numpy()+1
    for model, row in metrics.iterrows():
        p = predictions[[model+'_'+s for s in ['Away','Draw','Home']]].to_numpy()
        assert np.isfinite(p).all() and (p >= 0).all()
        np.testing.assert_allclose(p.sum(axis=1), 1)
        assert int((p.argmax(axis=1)==labels).sum()) == int(row.correct)
        loss = -np.log(np.clip(p[np.arange(len(p)), labels], 1e-15, 1)).mean()
        np.testing.assert_allclose(loss, row.log_loss)
    details = json.loads((OUT / 'run_details.json').read_text(encoding='utf-8'))
    for name, expected in details['input_sha256'].items():
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest() == expected, name
    strategies = pd.read_csv(OUT / 'strategy_results.csv')
    assert not strategies.duplicated(['model','strategy']).any()
    assert (strategies.bets == strategies.wins+strategies.losses).all()
    expected_roi = np.divide(strategies.net_profit_units.to_numpy(), strategies.bets.to_numpy(),
                             out=np.zeros(len(strategies)), where=strategies.bets.to_numpy()!=0)
    np.testing.assert_allclose(strategies.roi, expected_roi)
    if (OUT / 'bet_ledger.csv').exists():
        ledger = pd.read_csv(OUT / 'bet_ledger.csv')
        assert not ledger.duplicated(['model','strategy','test_position']).any()
        summed = ledger.groupby(['model','strategy']).net_profit_units.sum().sort_index()
        np.testing.assert_allclose(strategies.set_index(['model','strategy']).net_profit_units.sort_index(), summed)
        from betting_rules import HOME_OR_SKIP
        selected = ledger[ledger.strategy == HOME_OR_SKIP]
        if not selected.empty:
            reference = pd.read_csv(OUT / 'home_win_sqrt_profit_bet_ledger.csv')
            for model, records in selected.groupby('model'):
                records = records.sort_values('test_position')
                old = reference[reference.model == model].sort_values('test_position')
                np.testing.assert_array_equal(records.bet, old.home_stake == 1)
                assert (records.pick == 2).all()
                expected = np.where(records.bet, np.where(labels == 2, predictions.B365H - 1, -1), 0)
                np.testing.assert_allclose(records.net_profit_units, expected)
    slices = pd.read_csv(OUT / 'error_slices.csv')
    weeks = pd.read_csv(OUT / 'matchweek_errors.csv')
    raw = pd.read_csv(DATA / 'test.csv').iloc[predictions.source_test_row.to_numpy()]
    week_numbers = raw.Matchweek.astype(str).str.split().str[-1].astype(int).to_numpy()
    assert not weeks.duplicated(['model', 'matchweek']).any()
    for model, records in weeks.groupby('model'):
        probabilities = predictions[[model+'_'+s for s in ['Away', 'Draw', 'Home']]].to_numpy()
        assert records.matches.sum() == len(predictions)
        for row in records.itertuples():
            mask = week_numbers == row.matchweek
            assert row.matches == mask.sum()
            assert row.errors == (probabilities[mask].argmax(axis=1) != labels[mask]).sum()
            np.testing.assert_allclose(row.error_rate, row.errors/row.matches)
            np.testing.assert_allclose(row.mean_log_loss, -np.log(np.clip(probabilities[mask, labels[mask]], 1e-15, 1)).mean())
    assert (slices[slices.area!='Overall'].groupby(['model','area']).matches.sum()==len(predictions)).all()
    for stem in ['home_win', 'home_loss', 'home_win_sqrt_profit', 'home_loss_sqrt_profit']:
        if not (OUT / (stem+'_bet_ledger.csv')).exists():
            continue
        ledger = pd.read_csv(OUT / (stem+'_bet_ledger.csv'))
        np.testing.assert_allclose(ledger[['away_stake','home_stake','draw_stake']].sum(axis=1),1)
        match = predictions.iloc[ledger.test_position.to_numpy()]
        actual = (ledger.away_stake.to_numpy()*match.B365A.to_numpy()*(match.away_score>match.home_score).to_numpy()
                  +ledger.home_stake.to_numpy()*match.B365H.to_numpy()*(match.home_score>match.away_score).to_numpy()
                  +ledger.draw_stake.to_numpy()*match.B365D.to_numpy()*(match.home_score==match.away_score).to_numpy()-1)
        np.testing.assert_allclose(actual,ledger.net_profit_units)
    print('PASS: source/notebook syntax, input hashes, probabilities, outcome metrics, strategy accounting, binary payouts, and diagnostic counts.')


if __name__ == '__main__':
    main()
