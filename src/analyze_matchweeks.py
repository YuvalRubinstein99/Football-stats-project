"""Rank individual matchweeks using saved held-out error summaries."""
import re
import numpy as np
import pandas as pd
from reportlab.graphics.shapes import Drawing, Rect, Line, Circle
from reportlab.lib import colors
from project_paths import OUT
from result_files import write_csv
from plot_bivariate_results import label, save


def main():
    slices = pd.read_csv(OUT / 'error_slices.csv')
    weeks = slices[slices.area == 'Individual matchweek'].copy()
    weeks['matchweek'] = weeks['group'].astype(int)
    weeks['errors'] = weeks.matches - weeks.correct
    for metric in ['error_rate', 'mean_log_loss', 'goal_mse']:
        weeks[metric + '_rank'] = weeks.groupby('model')[metric].rank(method='min', ascending=False)
    columns = ['model', 'matchweek', 'matches', 'small_sample', 'errors', 'error_rate',
               'mean_log_loss', 'goal_mae', 'goal_mse', 'error_rate_rank', 'mean_log_loss_rank', 'goal_mse_rank']
    weeks = weeks[columns].sort_values(['model', 'error_rate', 'mean_log_loss'], ascending=[True, False, False])
    write_csv(weeks, OUT / 'matchweek_errors.csv')
    vector = weeks[weeks.model == 'RF_vector'].sort_values('matchweek')
    fig = Drawing(1200, 910)
    fig.add(Rect(0, 0, 1200, 910, fillColor=colors.white, strokeColor=None))
    label(fig, 40, 873, 'Which matchweeks have the highest errors?', 25)
    label(fig, 40, 849, 'Vector RF (2,000 trees) | All test matches pooled by matchweek number | Higher values = worse', 12)
    label(fig, 40, 828, 'Labels above points show match counts. Hollow points: fewer than 30 matches. Dashed line: overall test error.', 11)
    overall = slices[(slices.model == 'RF_vector') & (slices.area == 'Overall')].iloc[0]
    xmin, xmax = vector.matchweek.min(), vector.matchweek.max()
    for i, (metric, title) in enumerate([('error_rate', 'Wrong home/draw/away predictions (%)'),
                                       ('mean_log_loss', 'Mean log loss'), ('goal_mse', 'Goal MSE (home and away averaged)')]):
        x0, y0, width, height = 85, 610 - 255*i, 1050, 170
        factor = 100 if metric == 'error_rate' else 1
        maximum = float(vector[metric].max() * factor) * 1.18
        label(fig, x0, y0+height+20, title, 14)
        for value in np.linspace(0, maximum, 5):
            yy = y0 + height * value / maximum
            fig.add(Line(x0, yy, x0+width, yy, strokeColor=colors.HexColor('#e1e7ed')))
            label(fig, x0-10, yy-4, f'{value:.1f}', 10, textAnchor='end')
        baseline = y0 + height * overall[metric] * factor / maximum
        fig.add(Line(x0, baseline, x0+width, baseline, strokeColor=colors.HexColor('#888888'), strokeDashArray=[4, 3]))
        previous = None
        for row in vector.itertuples():
            xx = x0 + width * (row.matchweek-xmin) / max(1, xmax-xmin)
            yy = y0 + height * getattr(row, metric) * factor / maximum
            if previous:
                fig.add(Line(*previous, xx, yy, strokeColor=colors.HexColor('#9453b2')))
            fig.add(Circle(xx, yy, 3, fillColor=colors.white if row.small_sample else colors.HexColor('#9453b2'),
                           strokeColor=colors.HexColor('#9453b2')))
            label(fig, xx, yy+9, str(row.matches), 8, textAnchor='middle')
            label(fig, xx, y0-15, str(row.matchweek), 8, textAnchor='middle')
            previous = (xx, yy)
    label(fig, 610, 61, 'Matchweek', 12, textAnchor='middle')
    label(fig, 40, 32, 'Descriptive rankings across the existing test set; league/season composition can differ between matchweeks.', 11)
    label(fig, 40, 14, 'Outcome error, probability error, and goal error measure different failures; their rankings can differ.', 11)
    save(fig, 'matchweek_errors')
    report_path = OUT / 'report.html'
    report = report_path.read_text(encoding='utf-8')
    report = re.sub(r'<!-- matchweek-errors:start -->.*?<!-- matchweek-errors:end -->', '', report, flags=re.S)
    section = ('<!-- matchweek-errors:start --><h2>Highest-error matchweeks</h2>'
               '<p>Ranked by outcome error rate, with log loss breaking ties. All test matches are pooled by '
               'matchweek number; groups below 30 are flagged. Rankings are descriptive.</p>'
               '<img src="matchweek_errors.svg" alt="Errors by individual matchweek" style="width:100%;height:auto">'
               + weeks[weeks.model == 'RF_vector'].head(10).to_html(index=False, float_format=lambda v: f'{v:.4f}')
               + '<!-- matchweek-errors:end -->')
    report_path.write_text(report.replace('<h2>Method and limitations</h2>', section+'<h2>Method and limitations</h2>'), encoding='utf-8')
    print(weeks[weeks.model == 'RF_vector'].head(10).to_string(index=False))


if __name__ == '__main__':
    main()
