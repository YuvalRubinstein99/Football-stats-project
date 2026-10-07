"""Plot saved results without refitting; uses bundled ReportLab and PDFium."""
from pathlib import Path
import math
import re
import pandas as pd
from reportlab.graphics.shapes import Drawing, String, Line, Rect
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics import renderSVG, renderPDF
from reportlab.lib import colors
import pypdfium2 as pdfium

from project_paths import ROOT, DATA, OUT

MODELS = [('RF_independent', 'Original independent', '#2864b4'),
          ('RF_bivariate_exact', 'Joint 2D: exact', '#098574'),
          ('RF_bivariate_simulated', 'Joint 2D: simulated', '#ce7821'),
          ('RF_vector', 'One RF: goal vector', '#9453b2')]
TITLES = ['Most likely outcome', 'Expected return > 1', 'Log odds weighting',
          'Square-root odds', 'Square-root profit', 'Legacy draw-to-home',
          'Legacy draw-to-home return', 'Legacy market power', 'Legacy variance score']


def label(d, x, y, text, size=11, color='#263548', **kwargs):
    d.add(String(x, y, text, fontName='Helvetica', fontSize=size,
                 fillColor=colors.HexColor(color), **kwargs))


def legend(d, y):
    for i, (_, name, color) in enumerate(MODELS):
        x = 40 + 290*i
        d.add(Line(x, y+4, x+26, y+4, strokeColor=colors.HexColor(color), strokeWidth=2))
        label(d, x+34, y, name)


def save(d, name):
    renderSVG.drawToFile(d, str(OUT / (name + '.svg')))
    document = pdfium.PdfDocument(renderPDF.drawToString(d))
    page = document[0]
    bitmap = page.render(scale=1.5)
    bitmap.to_pil().save(OUT / (name + '.png'))
    bitmap.close()
    page.close()
    document.close()


def main():
    ledger = pd.read_csv(OUT / 'bet_ledger.csv')
    summary = pd.read_csv(OUT / 'strategy_results.csv')
    # The original nine rules belong to this grid; binary rules have dedicated plots.
    strategies = summary.strategy.drop_duplicates().tolist()[:len(TITLES)]
    curves = {}
    for strategy in strategies:
        for model, _, _ in MODELS:
            frame = ledger[(ledger.strategy == strategy) & (ledger.model == model)].sort_values('test_position')
            cumulative = frame.net_profit_units.cumsum().tolist()
            expected = summary[(summary.strategy == strategy) & (summary.model == model)].net_profit_units.iloc[0]
            assert abs(cumulative[-1] - expected) < 1e-8
            curves[strategy, model] = [(0, 0)] + list(enumerate(cumulative, start=1))
    values = [p[1] for curve in curves.values() for p in curve]
    ymin = math.floor(min(values)/25)*25
    ymax = math.ceil(max(values)/25)*25
    matches = int(ledger.test_position.max()) + 1
    d = Drawing(1200, 1040)
    d.add(Rect(0, 0, 1200, 1040, fillColor=colors.white, strokeColor=None))
    label(d, 40, 1005, 'Cumulative profit by betting strategy', 25)
    label(d, 40, 978, f'{matches:,} held-out matches | One unit per bet | Shared y-axis: net profit in units', 12)
    legend(d, 945)
    for index, (strategy, title) in enumerate(zip(strategies, TITLES)):
        col, row = index % 3, index // 3
        x, y = 60 + 395*col, 680 - 290*row
        label(d, x, y+222, title, 13)
        plot = LinePlot()
        plot.x, plot.y, plot.width, plot.height = x, y, 325, 205
        plot.data = [curves[strategy, model] for model, _, _ in MODELS]
        plot.xValueAxis.valueMin, plot.xValueAxis.valueMax = 0, matches
        plot.xValueAxis.valueSteps = [0, 500, 1000, matches]
        plot.yValueAxis.valueMin, plot.yValueAxis.valueMax = ymin, ymax
        plot.yValueAxis.valueSteps = list(range(ymin, ymax+1, 50))
        plot.yValueAxis.visibleGrid = True
        plot.yValueAxis.gridStrokeColor = colors.HexColor('#e1e7ed')
        plot.yValueAxis.labels.fontSize = 9
        plot.xValueAxis.labels.fontSize = 9
        for j, (_, _, color) in enumerate(MODELS):
            plot.lines[j].strokeColor = colors.HexColor(color)
            plot.lines[j].strokeWidth = 1.2
        plot.lines[2].strokeDashArray = [3, 2]
        d.add(plot)
        label(d, x+160, y-33, 'Test match order', 10, textAnchor='middle')
    label(d, 40, 43, 'Order follows test.csv; dates are unavailable. Dashed simulation curves include Monte Carlo noise.', 11)
    label(d, 40, 23, 'Legacy rules reproduce notebook heuristics. A draw still loses a home-win bet.', 11)
    save(d, 'cumulative_profit')

    d = Drawing(1200, 940)
    d.add(Rect(0, 0, 1200, 940, fillColor=colors.white, strokeColor=None))
    label(d, 40, 900, 'Winning bets by strategy', 25)
    label(d, 40, 874, 'Labels show wins / bets. Most rules bet every match; expected-return rules can skip matches.', 12)
    legend(d, 839)
    left, width, maximum = 285, 740, 850
    for tick in range(0, 851, 100):
        xx = left + width*tick/maximum
        d.add(Line(xx, 89, xx, 805, strokeColor=colors.HexColor('#e1e7ed')))
        label(d, xx, 70, str(tick), 10, textAnchor='middle')
    for i, (strategy, title) in enumerate(zip(strategies, TITLES)):
        y = 780 - i*78
        label(d, left-15, y-10, title, 11, textAnchor='end')
        for j, (model, _, color) in enumerate(MODELS):
            record = summary[(summary.model == model) & (summary.strategy == strategy)].iloc[0]
            bar_width = width*record.wins/maximum
            yy = y-j*16
            d.add(Rect(left, yy, bar_width, 12, fillColor=colors.HexColor(color), strokeColor=None))
            label(d, left+bar_width+7, yy+2, f'{int(record.wins)} / {int(record.bets):,}', 10)
    label(d, 650, 47, 'Winning bets (count)', 11, textAnchor='middle')
    label(d, 40, 20, 'Win counts do not measure profitability: payouts differ. Read alongside the cumulative-profit chart.', 11)
    save(d, 'wins_by_strategy')

    report_path = OUT / 'report.html'
    report = report_path.read_text(encoding='utf-8')
    section = ('<!-- comparison-plots:start --><h2>Strategy comparison plots</h2>'
               '<p>Original independent model, exact joint probabilities, 50,000 joint simulations per match, and one multi-output forest predicting a goal vector. The vector model uses independent Poisson outcome conversion.</p>'
               '<img src="cumulative_profit.svg" alt="Cumulative profit by betting strategy" style="width:100%;height:auto">'
               '<img src="wins_by_strategy.svg" alt="Winning bets by strategy" style="width:100%;height:auto">'
               '<!-- comparison-plots:end -->')
    report = re.sub(r'<!-- comparison-plots:start -->.*?<!-- comparison-plots:end -->', '', report, flags=re.S)
    report = report.replace('<h2>Predictive performance</h2>', section + '<h2>Predictive performance</h2>')
    report_path.write_text(report, encoding='utf-8')
    print('Saved both comparison charts as PNG and SVG; updated report.html.')


if __name__ == '__main__':
    main()
