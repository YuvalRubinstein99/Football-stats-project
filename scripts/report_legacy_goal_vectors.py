"""Replay the legacy rules on frozen goal-vector probabilities for both models."""
from backtest_transfermarkt_strategies import selections
from backtest_xgboost_goal_vectors import OUT, RESULTS, COLORS
from live_bets import SEARCH_CANDIDATE
from plot_binary_home_filters import (
    Drawing, HexColor, Line, PolyLine, Rect, np, pd, pdfium, renderPDF, text,
)

LABELS = {
    'Legacy draw-to-home weighting': 'Draw to home',
    'Legacy draw-to-home return weighting': 'Draw to home, return weighted',
    'Legacy market power weighting': 'Market power',
    'Legacy variance score': 'Variance score',
    SEARCH_CANDIDATE: 'Filtered variance',
}


def plot(weekly, split):
    drawing = Drawing(1040, 1095)
    drawing.add(Rect(0, 0, 1040, 1095, fillColor=HexColor('#ffffff'), strokeColor=None))
    period = '2022–23 test' if split == 'test' else '2014–22 training (in-sample)'
    text(drawing, 32, 1055, f'Legacy bets   /   {period}', 24, bold=True)
    text(drawing, 32, 1028, 'Cumulative net profit in units by scheduled matchweek', 13)
    text(drawing, 690, 1028, 'Original forest', 13, color=COLORS['RF_vector'], bold=True)
    text(drawing, 850, 1028, 'XGBoost', 13, color=COLORS['XGBoost'], bold=True)
    for col, (sizing, title) in enumerate([('flat', '1 unit per bet'), ('tiers', '1–5 units, positive edge required')]):
        left = 64 + 500 * col
        right = left + 412
        text(drawing, left, 990, title, 16, bold=True)
        for row, (strategy, label) in enumerate(LABELS.items()):
            top = 935 - row * 190
            bottom = top - 107
            panel = weekly[(weekly.strategy == strategy) & (weekly.staking == sizing)]
            series = panel.pivot(index=['season', 'week'], columns='model', values='cumulative_net')
            assert series.notna().all().all()
            lo = min(0., series.min().min())
            hi = max(0., series.max().max())
            def height(value):
                return bottom + (top - bottom) * (value - lo) / (hi - lo or 1)
            text(drawing, left, top + 29, label, 14, bold=True)
            for value in sorted({lo, 0., hi}):
                drawing.add(Line(left, height(value), right, height(value), strokeColor=HexColor('#cad2db')))
                text(drawing, left - 6, height(value) - 3, f'{value:,.0f}', 9, textAnchor='end')
            for j, (model, color) in enumerate(COLORS.items()):
                points = [left, height(0)]
                values = series[model].to_numpy()
                for i, value in enumerate(values, 1):
                    points.extend([left + (right - left) * i / len(values), height(value)])
                drawing.add(PolyLine(points, strokeWidth=1.9, strokeColor=HexColor(color), fillColor=None))
                text(drawing, left + j * 165, top + 10, f'{values[-1]:+,.2f} units', 11, color=color)
            text(drawing, left, bottom - 15, 'Start', 9)
            text(drawing, right, bottom - 15, 'Week 38' if split == 'test' else 'End of 2021–22', 9, textAnchor='end')
    with pdfium.PdfDocument(renderPDF.drawToString(drawing)) as doc:
        page = doc[0]
        bitmap = page.render(scale=1.4)
        bitmap.to_pil().save(OUT / f'legacy_{split}.png')
        bitmap.close()
        page.close()


def main():
    predictions = pd.read_csv(OUT / 'predictions.csv.gz', float_precision='round_trip')
    rows, paths = [], []
    for (model, split), frame in predictions.groupby(['model', 'split']):
        assert frame.source_row.is_unique
        arrays = selections(frame, frame[['p_home', 'p_draw', 'p_away']].to_numpy())
        for strategy in LABELS:
            bets = arrays[strategy]
            for col, sizing in enumerate(['flat', 'tiers']):
                stake = bets[:, col]
                net = stake * bets[:, 4]
                info = dict(model=model, split=split, strategy=strategy, staking=sizing)
                weekly = frame.assign(net=net).groupby(['season', 'Matchweek']).net.sum().reset_index(name='net')
                weekly = weekly.rename(columns={'Matchweek': 'week'})
                weekly['cumulative_net'] = weekly.net.cumsum()
                assert np.isclose(weekly.cumulative_net.iat[-1], net.sum())
                paths.append(weekly.assign(**info))
                rows.append(dict(**info, bets=int((stake > 0).sum()), staked=stake.sum(),
                                 net=net.sum(), roi=net.sum()/stake.sum(),
                                 flat_net_on_same_bets=bets[stake > 0, 4].sum()))
    stats = pd.DataFrame(rows)
    keys = ['model', 'split', 'strategy', 'staking']
    cols = ['bets', 'staked', 'net', 'roi']
    previous = pd.read_csv(OUT / 'strategies.csv').set_index(keys)
    known = stats[stats.strategy != SEARCH_CANDIDATE].set_index(keys)
    np.testing.assert_allclose(known[cols], previous.loc[known.index, cols], atol=1e-8)
    original = pd.read_csv(RESULTS / 'expanded_comparison/strategy_results.csv')
    saved = original[(original.model == 'original__RF_vector') & (original.strategy == SEARCH_CANDIDATE)].iloc[0]
    replay = stats[(stats.model == 'RF_vector') & (stats.split == 'test') &
                   (stats.strategy == SEARCH_CANDIDATE)].set_index('staking')
    np.testing.assert_allclose(replay.loc['tiers', cols].to_numpy(float), saved[cols].to_numpy(float))
    np.testing.assert_allclose(replay.loc['flat', 'net'], saved.flat_net)
    stats.to_csv(OUT / 'legacy_results.csv', index=False)
    weekly = pd.concat(paths, ignore_index=True)
    for split, group in weekly.groupby('split'):
        plot(group, split)
    print(stats[(stats.strategy == SEARCH_CANDIDATE) | ((stats.split == 'test') & (stats.model == 'XGBoost'))].to_string(index=False))


if __name__ == '__main__':
    main()
