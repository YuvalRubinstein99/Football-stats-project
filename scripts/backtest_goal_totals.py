"""Evaluate over/under 2.5 on expanding-season XGBoost goal forecasts."""
from backtest_xgboost_goal_vectors import (
    OUT, ROOT, SETTINGS, TARGETS, load_sources, matrices, np, pd, xgb,
)
from classifier_features import club
from goal_totals import over_under_25
from live_predictions import parse_csv
from sklearn.metrics import log_loss


def matched_prices(frame, year):
    season = f'{year%100:02d}{(year+1)%100:02d}'
    games = []
    for league in ('E0', 'D1', 'I1', 'SP1'):
        path = ROOT / f'results/classifier/cache/mmz4281_{season}_{league}.csv'
        games.extend(parse_csv(path.read_bytes(), results=True))
    quotes = pd.DataFrame(games)
    keys = ['home_key', 'away_key']
    for side in ('home', 'away'):
        frame[side+'_key'] = frame[side+'_team_name'].map(club)
        quotes[side+'_key'] = quotes[side].map(club)
    unique = frame.drop_duplicates(keys, keep=False)
    merged = unique.merge(quotes, on=keys, validate='one_to_one', suffixes=('', '_source'))
    return merged[(merged.home_score == merged.hg) & (merged.away_score == merged.ag)]


def forward_goal_forecasts():
    (train, test), _ = load_sources()
    validation = pd.read_csv(OUT / 'validation.csv')
    for year in (2020, 2021, 2022):
        past = train[train.season < year]
        frame = (test if year == 2022 else train[train.season == year]).copy()
        x, xt, _, _ = matrices(past, frame)
        depth, rounds = validation[validation.season < year].groupby(['depth', 'rounds']).log_loss.mean().idxmin()
        model = xgb.XGBRegressor(**SETTINGS, max_depth=int(depth), n_estimators=int(rounds))
        model.fit(x, past[TARGETS].to_numpy())
        means = model.predict(xt).astype(float)
        frame['mean_home'], frame['mean_away'] = means.T
        yield year, len(frame), matched_prices(frame, year).sort_values('date').copy()


def main():
    ledgers = []
    for year, n, matched in forward_goal_forecasts():
        paired = matched.dropna(subset=['B365>2.5','B365<2.5']).copy()
        paired['p_over'] = [over_under_25(h, a)['over']['probability'] for h, a in zip(paired.mean_home,paired.mean_away)]
        paired['actual_over'] = paired.hg + paired.ag >= 3
        over, under = paired['B365>2.5'], paired['B365<2.5']
        paired['market_over'] = under / (over + under)
        over_ev = paired.p_over * over - 1
        under_ev = (1 - paired.p_over) * under - 1
        paired['edge'] = np.maximum(over_ev, under_ev)
        paired['side'] = np.where(over_ev >= under_ev, 'Over', 'Under')
        wins = (paired.side == 'Over') == paired.actual_over
        paired['unit_net'] = np.where(wins, np.where(paired.side == 'Over', over, under), 0) - 1
        paired['correct'] = (paired.p_over >= .5) == paired.actual_over
        print(year, 'matched', len(paired), 'of', n, flush=True)
        ledgers.append(paired)
    ledger = pd.concat(ledgers, ignore_index=True)
    fields = ['season','date','league','home','away','hg','ag','p_over','market_over','B365>2.5','B365<2.5','edge','side','unit_net']
    ledger[fields].to_csv(OUT/'over_under_25.csv.gz', index=False)
    for period, rows in [*ledger.groupby('season'), ('Pooled',ledger)]:
        print(period, 'games', len(rows), 'accuracy', rows.correct.mean(), 'over rate', rows.actual_over.mean(),
              'mean forecast', rows.p_over.mean(), 'log loss model/market', log_loss(rows.actual_over,rows.p_over),
              log_loss(rows.actual_over,rows.market_over))
        for cutoff in (0,.02):
            taken = rows[rows.edge > cutoff]
            print(cutoff, len(taken), round(taken.unit_net.sum(),2), round(taken.unit_net.mean(),4))


if __name__ == '__main__':
    main()
