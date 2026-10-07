"""Rebuild disposable betting ledgers from committed per-match predictions."""
from project_paths import OUT
import numpy as np
import pandas as pd
from betting_rules import strategy_scores
from result_files import write_csv


def main():
    predictions = pd.read_csv(OUT / 'match_predictions.csv')
    models = pd.read_csv(OUT / 'model_metrics.csv').model.tolist()
    odds = predictions[['B365A', 'B365D', 'B365H']].to_numpy()
    market = predictions[['Market_Away', 'Market_Draw', 'Market_Home']].to_numpy()
    labels = np.sign(predictions.home_score - predictions.away_score).astype(int).to_numpy() + 1
    rows, ledgers = [], []
    for model in models:
        p = predictions[[model + '_' + outcome for outcome in ['Away', 'Draw', 'Home']]].to_numpy()
        for strategy, (values, threshold) in strategy_scores(p, odds, market).items():
            pick = values.argmax(axis=1)
            bet = values[np.arange(len(p)), pick] > threshold
            win = (pick == labels) & bet
            selected_odds = odds[np.arange(len(p)), pick]
            profit = np.where(bet, np.where(win, selected_odds - 1, -1), 0)
            count = int(bet.sum())
            rows.append({'model':model, 'strategy':strategy, 'bets':count, 'wins':int(win.sum()),
                         'losses':int((bet & ~win).sum()), 'win_rate':float(win.sum()/count) if count else 0,
                         'net_profit_units':float(profit.sum()), 'roi':float(profit.sum()/count) if count else 0})
            ledgers.append(pd.DataFrame({'model':model, 'strategy':strategy,
                                         'test_position':np.arange(len(p)), 'pick':pick, 'bet':bet,
                                         'win':win, 'decimal_odds':selected_odds, 'net_profit_units':profit}))
    write_csv(pd.DataFrame(rows), OUT / 'strategy_results.csv')
    write_csv(pd.concat(ledgers, ignore_index=True), OUT / 'bet_ledger.csv')


if __name__ == '__main__':
    main()
