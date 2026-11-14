"""Compare existing 1–5-unit tiers with flat stakes on frozen handicap selections."""
from plot_binary_home_filters import (
    ROOT, pd, np, Drawing, Rect, Line, PolyLine, HexColor, text, renderPDF, pdfium,
)
from goal_difference import handicap_net
from stake_levels import size_bet

OUT = ROOT / 'results/xgboost_goal_vector'
THRESHOLDS = [(0., 'Expected return above 0%'), (.02, 'Expected return above 2%')]
COLORS = {'flat': '#18766b', 'tiers': '#7953a4'}


def plot(curves, summary):
    drawing = Drawing(1120, 690)
    drawing.add(Rect(0, 0, 1120, 690, fillColor=HexColor('#ffffff'), strokeColor=None))
    text(drawing, 32, 651, 'Asian handicap · flat and tiered stakes', 25, bold=True)
    text(drawing, 32, 624, '2020–21 through 2022–23 · forward predictions · same selections within each threshold', 13)
    text(drawing, 32, 602, 'Cumulative net profit in units. Each row shares a vertical scale. Stakes are fixed units with no compounding.', 12)
    stats = summary[summary.period == 'Pooled'].set_index(['threshold', 'staking'])
    for row, (cutoff, label) in enumerate(THRESHOLDS):
        panel = curves[curves.threshold == cutoff]
        low = np.floor(min(0., panel.cumulative_net.min()) / 25) * 25
        high = np.ceil(max(0., panel.cumulative_net.max()) / 25) * 25
        top = 492 - row * 264
        bottom = top - 155
        for col, (sizing, color) in enumerate(COLORS.items()):
            left, right = 65 + col * 550, 510 + col * 550
            selected = panel[panel.staking == sizing]
            dates = pd.to_datetime(selected.date)
            start = dates.min() - pd.Timedelta(days=1)
            x = lambda date: left + (right-left)*(date-start).days/(dates.max()-start).days
            y = lambda value: bottom + (top-bottom)*(value-low)/(high-low or 1)
            title = 'Flat · 1 unit' if sizing == 'flat' else 'Tiered · 1–5 units'
            entry = stats.loc[(cutoff, sizing)]
            text(drawing, left-25, top+66, label+'   /   '+title, 15, bold=True, color=color)
            text(drawing, left-25, top+43, f'{entry.net:+,.2f} units   |   ROI {entry.roi:+.2%}   |   {int(entry.bets):,} bets', 12, color=color)
            text(drawing, left-25, top+23, f'{entry.staked:,.0f} units staked   |   max daily drawdown {entry.drawdown:,.2f} units', 11)
            for value in sorted({low, 0., high}):
                drawing.add(Line(left, y(value), right, y(value), strokeColor=HexColor('#d3dbdb'), strokeWidth=.7))
                text(drawing, left-7, y(value)-3, f'{value:.0f}', 10, textAnchor='end')
            points = [left, y(0)]
            for date, value in zip(dates, selected.cumulative_net):
                points.extend([x(date), y(value)])
            drawing.add(PolyLine(points, strokeWidth=1.7, strokeColor=HexColor(color), fillColor=None))
            for date in [dates.min(), pd.Timestamp('2021-08-01'), pd.Timestamp('2022-08-01'), dates.max()]:
                text(drawing, x(date), bottom-21, date.strftime('%b %Y'), 10, textAnchor='middle')
    with pdfium.PdfDocument(renderPDF.drawToString(drawing)) as doc:
        page = doc[0]
        bitmap = page.render(scale=1.5)
        bitmap.to_pil().save(OUT / 'goal_difference_stakes.png')
        bitmap.close()
        page.close()


def main():
    frame = pd.read_csv(OUT / 'goal_difference.csv.gz', float_precision='round_trip')
    valid = frame.side.isin(['Home', 'Away']) & frame.edge.notna() & frame.unit_net.notna()
    odds = np.where(frame.side == 'Home', frame.B365AHH, frame.B365AHA)
    home = frame.side == 'Home'
    settled = handicap_net(np.where(home, frame.actual_margin, -frame.actual_margin),
                           np.where(home, frame.AHh, -frame.AHh), odds)
    np.testing.assert_allclose(settled[valid], frame.loc[valid, 'unit_net'], atol=1e-12)
    levels = []
    for price, edge, usable in zip(odds, frame.edge, valid):
        bet = dict(action='Bet' if usable else 'Unavailable', decimal_odds=price,
                   expected_net_per_unit=edge, home_stake=1., draw_stake=0., away_stake=0., note='')
        levels.append(size_bet(bet)['stake_dollars'])
    frame['tier_stake'] = levels
    summaries, paths, ledgers = [], [], []
    for cutoff, _ in THRESHOLDS:
        eligible = valid & (frame.edge > cutoff)
        for staking in COLORS:
            stakes = np.where(eligible, 1 if staking == 'flat' else frame.tier_stake, 0)
            assert (stakes > 0).sum() == eligible.sum()
            record = frame[['season', 'date', 'league', 'home', 'away', 'side', 'edge', 'unit_net']].copy()
            record = record.assign(threshold=cutoff, staking=staking, stake=stakes,
                                   net=stakes * frame.unit_net.fillna(0))
            ledgers.append(record)
            for period, group in [*record.groupby('season'), ('Pooled', record)]:
                daily = group.groupby('date').net.sum().cumsum()
                values = np.r_[0., daily.to_numpy()]
                summaries.append(dict(period=str(period), threshold=cutoff, staking=staking,
                                      bets=int((group.stake > 0).sum()), staked=group.stake.sum(),
                                      net=group.net.sum(), roi=group.net.sum()/group.stake.sum(),
                                      drawdown=float((np.maximum.accumulate(values)-values).max())))
                if period == 'Pooled':
                    assert np.isclose(daily.iat[-1], group.net.sum())
                    paths.append(daily.rename('cumulative_net').reset_index().assign(threshold=cutoff, staking=staking))
    summary = pd.DataFrame(summaries)
    original = summary[(summary.period == 'Pooled') & (summary.staking == 'flat')].sort_values('threshold')
    np.testing.assert_array_equal(original.bets, [2635, 1479])
    np.testing.assert_allclose(original.net, [-27.59, 8.69], atol=1e-8)
    curves = pd.concat(paths, ignore_index=True)
    summary.to_csv(OUT / 'goal_difference_stakes_summary.csv', index=False)
    pd.concat(ledgers, ignore_index=True).to_csv(OUT / 'goal_difference_stakes_ledger.csv.gz', index=False)
    curves.to_csv(OUT / 'goal_difference_stakes_curves.csv', index=False)
    plot(curves, summary)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
