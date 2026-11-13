"""Compact money report and cumulative profit chart from the verified replay."""
import base64
from backtest_match_form import OUT, MODES, SEARCH_CANDIDATE, pd
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics import renderPDF, renderSVG
from reportlab.lib.colors import HexColor
import pypdfium2 as pdfium


def main():
    summary = pd.read_csv(OUT / 'summary.csv')
    fixed = pd.read_csv(OUT / 'fixed_filtered_strategy.csv')
    bankroll = pd.read_csv(OUT / 'bankroll_summary.csv')
    curves = pd.read_csv(OUT / 'daily_curves.csv.gz')
    chosen = summary[(summary.split == 'test_2022') & (summary.league == 'ALL') &
                     (summary.variant == 'minimum_cv_form_allowed')]
    selected = curves[(curves.split == 'test_2022') &
                      (curves.variant == 'minimum_cv_form_allowed') &
                      (curves.strategy == SEARCH_CANDIDATE) & (curves.staking == MODES[2])]
    bc = pd.read_csv(OUT / 'bankroll_curves.csv')
    bc = bc[(bc.split == 'test_2022') & (bc.variant == 'minimum_cv_form_allowed') &
            (bc.method == 'Robust quarter Kelly') & (bc.date != 'Start')]
    bc['bankroll'] = bc.net + 1000
    drawing = Drawing(1100, 850)
    drawing.add(Rect(0, 0, 1100, 850, fillColor=HexColor('#ffffff'), strokeColor=None))
    colors = ['#2563eb', '#d97706', '#16a34a', '#db2777', '#7c3aed', '#0891b2', '#475569']
    origin = pd.Timestamp('2022-08-01')
    for data, value, title, y, baseline in [
        (selected, 'cumulative_net', 'Filtered legacy rule: cumulative net profit ($), fixed $1-$5 tiers', 480, 0),
        (bc, 'bankroll', 'Robust quarter Kelly: bankroll ($), starting with $1,000', 80, 1000)]:
        drawing.add(String(65, y+310, title, fontName='Helvetica-Bold', fontSize=17))
        chart = LinePlot()
        chart.x, chart.y, chart.width, chart.height = 75, y, 970, 250
        series = list(data.groupby('family'))
        chart.data = [[(float((pd.Timestamp(row.date)-origin).days), float(getattr(row, value)))
                       for row in group.itertuples()] for _, group in series]
        chart.data.append([(0, baseline), (310, baseline)])
        values = [point[1] for line in chart.data for point in line]
        padding = max(10., (max(values)-min(values)) * .08)
        chart.yValueAxis.valueMin = min(values)-padding
        chart.yValueAxis.valueMax = max(values)+padding
        chart.xValueAxis.valueMin, chart.xValueAxis.valueMax = 0, 310
        chart.xValueAxis.valueSteps = [0, 61, 122, 184, 243, 304]
        chart.xValueAxis.labelTextFormat = lambda x: (origin+pd.Timedelta(days=x)).strftime('%b %Y')
        chart.yValueAxis.labelTextFormat = '$%0.0f'
        chart.yValueAxis.visibleGrid = True
        chart.yValueAxis.gridStrokeColor = HexColor('#e2e8f0')
        for i, (family, _) in enumerate(series):
            chart.lines[i].strokeColor = HexColor(colors[i])
            chart.lines[i].strokeWidth = 1.3
            x = 75 + (i % 4)*245
            ly = y+285-(i//4)*18
            drawing.add(Line(x, ly, x+18, ly, strokeColor=HexColor(colors[i]), strokeWidth=2))
            drawing.add(String(x+24, ly-3, family, fontSize=10))
        chart.lines[len(series)].strokeColor = HexColor('#111827')
        chart.lines[len(series)].strokeWidth = .5
        drawing.add(chart)
    renderSVG.drawToFile(drawing, str(OUT / 'profit_curves.svg'))
    with pdfium.PdfDocument(renderPDF.drawToString(drawing)) as doc:
        page = doc[0]
        bitmap = page.render(scale=1.5)
        bitmap.to_pil().save(OUT / 'profit_curves.png')
        bitmap.close()
        page.close()

    def table(data, index=False):
        return data.to_html(index=index, float_format=lambda x: f'{x:,.2f}', escape=True)

    page = """<!doctype html><meta charset="utf-8"><title>Betting rerun: money results</title>
<style>body{font:15px system-ui;color:#172536;max-width:1400px;margin:36px auto;padding:0 20px}
table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd}
th{background:#eef2f5}img{max-width:100%}.scroll{overflow:auto}h2{margin-top:32px}</style>
<h1>Historical betting rerun — 2022–23</h1>
<p>1,445 fixtures across England, Spain, Germany and Italy. Seven model families, 15 unchanged
selection rules and three fixed staking modes. Feature sets were chosen using earlier chronological
cross-validation; the main tables use those frozen minimum-CV choices with form allowed.</p>
<p><b>These are hypothetical historical returns, not money earned.</b> Each row/model is a separate
alternative. Do not add their profits together. All amounts below are dollars; ROI is net profit
divided by total stakes. Bet365 recorded odds, without fees, taxes, stake rounding or execution costs.
The audit/test seasons were previously inspected; rankings are exploratory, not evidence of future profit.</p>
<p><a href="report.html">Full results, all feature variants and the 2021–22 audit</a> ·
<a href="summary.csv">All fixed-stake results (CSV)</a> ·
<a href="form_profit_comparison.csv">Matched form comparisons (CSV)</a> ·
<a href="bankroll_summary.csv">Bankroll results (CSV)</a></p>
<h2>Previously used filtered legacy rule: did form add money?</h2>
<p>Same rule and fixed $1–$5 tiers on both sides. Odds 4–1000 and expected return above 2%.
This rule was previously explored with RF; its unchanged transfer to other models is exploratory.</p>"""
    display = fixed[['family', 'net_profit_without', 'net_profit_form', 'net_change_with_form',
                     'bets_form', 'total_staked_form', 'roi_form', 'max_daily_drawdown_dollars_form']].copy()
    display['roi_form'] *= 100
    display.columns = ['Model', 'Net without form ($)', 'Net form allowed ($)', 'Change ($)',
                       'Bets', 'Total staked ($)', 'ROI (%)', 'Max daily drawdown ($)']
    page += '<div class="scroll">' + table(display) + '</div>'
    page += '<h2>All 15 rules: net profits by model</h2><p>Flat original-rule stakes bet $1 whenever the rule selects a bet. Positive-edge flat stakes and the fixed tiers skip nonpositive model edge. Tiers use $1–$5 per selected bet; no compounding or cash cap.</p>'
    for mode in MODES:
        grid = chosen[chosen.staking == mode].pivot(index='strategy', columns='family', values='net_profit')
        page += '<h3>' + mode + '</h3><div class="scroll">' + table(grid, True) + '</div>'
    page += '<h2>What happens to a $1,000 starting bankroll?</h2><p>These three existing simulations select the outcome with the highest expected return and skip nonpositive edge. Quarter Kelly caps each match at 2% of opening cash and each calendar day at 10%. Robust quarter Kelly first subtracts two percentage points from the selected outcome probability. Same-day stakes are reserved before settlement. These are separate experiments from the 15-rule tables.</p>'
    b = bankroll[(bankroll.split == 'test_2022') & bankroll.variant.isin(
        ['minimum_cv_without_form', 'minimum_cv_form_allowed'])]
    no = b[b.variant == 'minimum_cv_without_form'].set_index(['family', 'method'])
    yes = b[b.variant == 'minimum_cv_form_allowed'].set_index(['family', 'method'])
    display = yes[['bets', 'staked', 'net', 'ending_bankroll', 'roi', 'max_drawdown']].join(
        no[['net']], rsuffix='_without_form').reset_index()
    display['change_with_form'] = display.net - display.net_without_form
    display['roi'] *= 100
    display['max_drawdown'] *= 100
    display.columns = ['Model', 'Staking', 'Bets', 'Total staked ($)', 'Net ($)',
                       'Ending bankroll ($)', 'ROI (%)', 'Max drawdown (%)',
                       'Net without form ($)', 'Change with form ($)']
    page += '<div class="scroll">' + table(display) + '</div>'
    page += '<h2>Prior-season bankroll check: 2021–22 audit</h2><p>Each simulation again starts at $1,000. This separate season helps show whether the 2022–23 profits repeat.</p>'
    audit = bankroll[(bankroll.split == 'audit_2021') & (bankroll.variant == 'minimum_cv_form_allowed')]
    page += '<div class="scroll">' + table(audit.pivot(index='family', columns='method', values='net'), True) + '</div>'
    image = base64.b64encode((OUT / 'profit_curves.png').read_bytes()).decode()
    page += '<h2>How returns developed through the season</h2><img alt="Cumulative profits and bankroll by model" src="data:image/png;base64,' + image + '">'
    (OUT / 'overview.html').write_text(page, encoding='utf-8')
    print('Saved', OUT / 'overview.html')


if __name__ == '__main__':
    main()
