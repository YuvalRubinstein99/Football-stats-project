"""Report home-only bets selected by the existing binary sqrt-profit rule."""
import re
import numpy as np
import pandas as pd
from reportlab.graphics.shapes import Drawing, Rect, Line
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.lib import colors
from project_paths import OUT
from betting_rules import HOME_OR_SKIP
from plot_bivariate_results import label, save, MODELS
from result_files import write_csv


def main():
    all_results = pd.read_csv(OUT / 'strategy_results.csv')
    summary = all_results[all_results.strategy == HOME_OR_SKIP].copy()
    ledger = pd.read_csv(OUT / 'bet_ledger.csv')
    ledger = ledger[ledger.strategy == HOME_OR_SKIP].copy()
    matches = ledger.test_position.nunique()
    summary['skipped'] = matches - summary.bets
    binary = pd.read_csv(OUT / 'home_win_sqrt_profit_strategy.csv').set_index('model')
    for row in summary.itertuples():
        assert row.bets == binary.loc[row.model, 'home_win_bets']
        np.testing.assert_allclose(row.net_profit_units, binary.loc[row.model, 'win_side_profit'])
    write_csv(summary, OUT / 'home_win_or_skip_strategy.csv')
    shown = MODELS + [('Market', 'Market probabilities', '#657386')]
    curves = [[(0, 0.)] + list(enumerate(
        ledger[ledger.model == model].sort_values('test_position').net_profit_units.cumsum(), 1))
        for model, _, _ in shown]
    values = [v for curve in curves for _, v in curve]
    fig = Drawing(1200, 660)
    fig.add(Rect(0, 0, 1200, 660, fillColor=colors.white, strokeColor=None))
    label(fig, 40, 618, 'Home win or skip: square-root-profit weighting', 24)
    label(fig, 40, 590, 'Compare home win with away-or-draw using probability x sqrt(decimal odds - 1).', 12)
    label(fig, 40, 568, 'Bet one unit only if home wins the comparison (including ties); otherwise stake zero.', 12)
    plot = LinePlot()
    plot.x, plot.y, plot.width, plot.height = 85, 110, 850, 410
    plot.data = curves
    plot.xValueAxis.valueMin, plot.xValueAxis.valueMax = 0, matches
    plot.yValueAxis.valueMin = 25 * np.floor(min(values) / 25)
    plot.yValueAxis.valueMax = max(25, 25 * np.ceil(max(values) / 25))
    plot.yValueAxis.visibleGrid = True
    plot.yValueAxis.gridStrokeColor = colors.HexColor('#e1e7ed')
    for j, (_, name, color) in enumerate(shown):
        plot.lines[j].strokeColor = colors.HexColor(color)
        plot.lines[j].strokeWidth = 1.6
        fig.add(Line(965, 490-32*j, 990, 490-32*j, strokeColor=colors.HexColor(color), strokeWidth=2))
        label(fig, 998, 486-32*j, name, 10)
    plot.lines[2].strokeDashArray = [3, 2]
    fig.add(plot)
    label(fig, 85, 536, 'Cumulative net profit (units)', 11)
    label(fig, 510, 76, 'Test match order (including skipped matches)', 11, textAnchor='middle')
    label(fig, 40, 40, 'Home bets pay B365H - 1 on a home win, lose 1 on a draw or away win. Skipped matches pay 0.', 11)
    label(fig, 40, 20, 'Away-or-draw odds are a synthetic comparison price only; no hedge is placed. Exploratory backtest.', 11)
    save(fig, 'home_win_or_skip_strategy')
    notes = ('Compare p_home * sqrt(home_odds - 1) with (p_away + p_draw) * '
             'sqrt(max(D - 1, 0)), where D = 1 / (1/away_odds + 1/draw_odds). '
             'Bet one unit on home if its score is at least as high; otherwise do not bet. '
             'Actual draws lose home bets. Skips have zero stake and profit. ROI uses placed bets only. '
             'Weighting changes selection, not actual payouts. No added expected-utility or value threshold.')
    report_path = OUT / 'report.html'
    report = report_path.read_text(encoding='utf-8')
    report = re.sub(r'<!-- home-or-skip:start -->.*?<!-- home-or-skip:end -->', '', report, flags=re.S)
    section = ('<!-- home-or-skip:start --><h2>Home win or skip: square-root-profit weighting</h2><p>'
               + notes + '</p>' + summary.to_html(index=False, float_format=lambda v: f'{v:.4f}')
               + '<img src="home_win_or_skip_strategy.svg" alt="Home win or skip cumulative profit" '
               'style="width:100%;height:auto"><!-- home-or-skip:end -->')
    report_path.write_text(report.replace('<h2>Method and limitations</h2>', section + '<h2>Method and limitations</h2>'), encoding='utf-8')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
