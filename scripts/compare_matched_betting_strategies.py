"""Compare the existing H/A betting rules on the matched old/new forecasts."""
import math
from select_expanded_features import ROOT, np, pd
from binary_betting import selections, settle, summary
from plot_binary_home_filters import (
    Drawing, Rect, Line, PolyLine, HexColor, text, renderPDF, pdfium,
)

BASE = ROOT / 'results/lineup_improvements/binary_retrained/matched_comparison'
OUT = BASE / 'strategies'
COLORS = {'old': '#ab4848', 'new': '#00777c'}
TITLES = {
    'Home only: positive expected return': 'Home · expected return > 0%',
    'Home only: expected return >2%': 'Home · expected return > 2%',
    'Home only: probability >=50%': 'Home · win probability ≥ 50%',
    'Home only: sqrt-profit comparison': 'Home · square-root profit comparison',
    'Most likely side': 'Most likely side',
    'Return-score filter': 'Return-score filter',
    'Log-odds weighting': 'Log-odds weighting',
    'Square-root odds weighting': 'Square-root odds weighting',
    'Square-root profit weighting': 'Square-root profit weighting',
    'Legacy market-power weighting': 'Market-power weighting',
    'Legacy variance weighting': 'Variance weighting',
    'Legacy variance: odds >=4, score edge >2%': 'Variance · odds ≥ 4 and score edge > 2%',
}


def chart(split, staking, curves, stats, *, colors=COLORS, out=OUT, period=None):
    drawing = Drawing(1440, 1110)
    drawing.add(Rect(0, 0, 1440, 1110, fillColor=HexColor('#ffffff'), strokeColor=None))
    period = period or ('2022–23 test' if split == 'test' else '2014–15 to 2021–22 training fit')
    stakes = '1 unit per bet' if staking == 'flat' else '1–5-unit stakes'
    text(drawing, 36, 1074, f'Cumulative betting profit  |  {period}', 24, bold=True)
    text(drawing, 36, 1048, stakes, 15)
    for index, (model, color) in enumerate(colors.items()):
        label = model.title()+' model' if model in ('old','new') else model
        text(drawing, 234+index*190, 1048, label, 14, color=color, bold=True)
    text(drawing, 36, 1025, 'Net profit in units. Each panel starts at zero and has its own vertical scale. End totals and bet counts appear above each curve.', 12)
    for i, (rule, title) in enumerate(TITLES.items()):
        row, col = divmod(i, 3)
        left = 66 + col * 473
        right = left + 375
        top = 945 - 237 * row
        bottom = top - 144
        data = curves[(curves.strategy == rule) & (curves.split == split) & (curves.staking == staking)]
        low, high = min(0, data.profit.min()), max(0, data.profit.max())
        rough = max((high-low)/3, 1)
        magnitude = 10**math.floor(math.log10(rough))
        step = next(n*magnitude for n in [1, 2, 2.5, 5, 10] if n*magnitude >= rough)
        lo, hi = math.floor(low/step)*step, math.ceil(high/step)*step
        hi = max(lo+step, hi)
        start, end = data.date.min(), data.date.max()
        def x(day):
            return left+(day-start).days/(end-start).days*(right-left)
        def y(value):
            return bottom+(value-lo)/(hi-lo)*(top-bottom)
        text(drawing, left-25, top+49, title, 14, bold=True)
        for value in np.arange(lo, hi+step/2, step):
            drawing.add(Line(left, y(value), right, y(value), strokeWidth=.7,
                             strokeColor=HexColor('#939eab' if value == 0 else '#e2e6eb')))
            text(drawing, left-6, y(value)-3, f'{value:g}', 9, textAnchor='end')
        for j, (model, color) in enumerate(colors.items()):
            curve = data[data.model == model].sort_values('date')
            result = stats.loc[(split, staking, rule, model)]
            assert np.isclose(curve.profit.iloc[-1], result.net)
            text(drawing, left-25+208*j, top+25,
                 f'{model if model not in ("old","new") else model.title()} {result.net:+,.2f}  ({int(result.bets):,} bets)', 11, color=color)
            points, previous = [], 0.
            for day, profit in zip(curve.date, curve.profit):
                points.extend([x(day), y(previous), x(day), y(profit)])
                previous = profit
            drawing.add(PolyLine(points, strokeColor=HexColor(color), strokeWidth=1.55, fillColor=None))
        for date, anchor in [(start,'start'),(end,'end')]:
            text(drawing, x(date), bottom-16, date.strftime('%b %Y'), 10, textAnchor=anchor)
    text(drawing, 36, 32, 'Draws lose either straight bet. Rules comparing home and away use P(draw or away win) as the away score. Tier staking also skips non-positive score edges.', 11)
    with pdfium.PdfDocument(renderPDF.drawToString(drawing)) as document:
        page = document[0]
        bitmap = page.render(scale=1.5)
        bitmap.to_pil().save(out / f'{split}_{staking}.png')
        bitmap.close()
        page.close()


def main():
    OUT.mkdir(exist_ok=True)
    forecasts = pd.read_csv(BASE/'predictions.csv', parse_dates=['fixture_date'])
    records, series = [], []
    for split, rows in forecasts.groupby('split'):
        keys = ['match_id','fixture_date','home_team_name','away_team_name','home_score','away_score','B365H','B365D','B365A']
        fixtures = [g[keys].sort_values('match_id').reset_index(drop=True) for _, g in rows.groupby('model')]
        pd.testing.assert_frame_equal(*fixtures)
        for model, group in rows.groupby('model'):
            group = group.sort_values(['fixture_date','league','source_row'])
            assert group.match_id.is_unique and len(group) == (1065 if split=='test' else 8521)
            odds = group[['B365H','B365D','B365A']].to_numpy()
            p = group.p_home_win.to_numpy()
            actual = (1-np.sign(group.home_score-group.away_score)).to_numpy(int)
            rules = selections(p, odds)
            assert set(rules)==set(TITLES)
            for rule, choice in rules.items():
                for staking in ['flat','tiers']:
                    bet = settle(p, odds, actual, choice, staking)
                    stats = summary(bet, actual)
                    meta = dict(split=split, model=model, strategy=rule, staking=staking)
                    records.append(dict(**meta, uses_away_proxy=choice['proxy'], **stats))
                    prices = odds[np.arange(len(group)), bet['pick']]
                    net = bet['stakes'] * (np.where(bet['pick']==actual, prices, 0.)-1)
                    np.testing.assert_allclose(net, bet['profit'], atol=1e-12)
                    daily = pd.Series(net, index=group.fixture_date).groupby(level=0).sum()
                    baseline = pd.Series([0.], index=[daily.index[0]-pd.Timedelta(days=1)])
                    curve = pd.concat([baseline,daily]).cumsum().rename('profit').rename_axis('date').reset_index()
                    series.append(curve.assign(**meta))
    table = pd.DataFrame(records)
    keys = ['split','staking','strategy','model']
    table.to_csv(OUT/'summary.csv',index=False)
    stats = table.set_index(keys)
    prior = pd.read_csv(BASE/'returns.csv').set_index(keys)
    for field in ['net','bets','staked','roi']:
        np.testing.assert_allclose(stats.loc[prior.index,field], prior[field])
    curves = pd.concat(series,ignore_index=True)
    curves.to_csv(OUT/'daily_curves.csv.gz',index=False)
    for split in ['test','train']:
        for staking in ['flat','tiers']:
            chart(split,staking,curves,stats)
    print(table.query("split=='test' and staking=='flat'").pivot(index='strategy',columns='model',values=['bets','net','roi']).round(4).to_string())


if __name__ == '__main__':
    main()
