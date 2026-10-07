"""Binned probability spread versus realized net winnings on most-likely bets."""
from pathlib import Path
import re
import math
import numpy as np
import pandas as pd
from reportlab.graphics.shapes import Drawing, Rect, Line, Circle
from reportlab.lib import colors
from plot_bivariate_results import MODELS, label, save

from project_paths import ROOT, DATA, OUT



def main():
    predictions = pd.read_csv(OUT / 'match_predictions.csv')
    ledger = pd.read_csv(OUT / 'bet_ledger.csv')
    rows = []
    for model, _, _ in MODELS:
        bets = ledger[(ledger.model == model) & (ledger.strategy == 'Most likely outcome')].sort_values('test_position')
        assert np.array_equal(bets.test_position.to_numpy(), np.arange(len(predictions)))
        assert bets.bet.all()
        prob = predictions[[model+'_'+s for s in ['Away','Draw','Home']]].to_numpy()
        gap = prob.max(axis=1)-prob.min(axis=1)
        f = pd.DataFrame({'gap':gap,'profit':bets.net_profit_units.to_numpy(),'win':bets.win.to_numpy()})
        f['bin'] = pd.cut(f.gap, np.linspace(0,1,11), include_lowest=True)
        for bucket, part in f.groupby('bin',observed=True):
            rows.append({'model':model,'gap_bin':str(bucket),'matches':len(part),
                         'mean_probability_gap':part.gap.mean(),
                         'mean_net_winnings':part.profit.mean(),
                         'total_net_winnings':part.profit.sum(),
                         'win_rate':part.win.mean(),'small_sample':len(part)<30})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'probability_gap_winnings.csv', index=False)
    for model,_,_ in MODELS:
        part = table[table.model==model]
        expected = ledger[(ledger.model==model)&(ledger.strategy=='Most likely outcome')].net_profit_units.sum()
        assert part.matches.sum()==len(predictions)
        assert np.isclose((part.mean_net_winnings*part.matches).sum(),expected)
    ymin = min(-.1, math.floor(table.mean_net_winnings.min()*10)/10-.1)
    ymax = max(.1, math.ceil(table.mean_net_winnings.max()*10)/10+.1)
    d = Drawing(1200,850)
    d.add(Rect(0,0,1200,850,fillColor=colors.white,strokeColor=None))
    label(d,40,810,'Probability gap versus average winnings',25)
    label(d,40,783,'Gap = max(P away, P draw, P home) minus min(P away, P draw, P home).',12)
    label(d,40,762,'One unit on the most likely outcome; net winnings = odds - 1 if correct, otherwise -1.',12)
    for index,(model,name,color) in enumerate(MODELS):
        x,y = 95+(index%2)*575, 440-(index//2)*345
        w,h=445,230
        label(d,x,y+h+35,name,16)
        px=lambda value:x+w*value
        py=lambda value:y+h*(value-ymin)/(ymax-ymin)
        for tick in np.linspace(0,1,6):
            d.add(Line(px(tick),y,px(tick),y+h,strokeColor=colors.HexColor('#e8edf1')))
            label(d,px(tick),y-19,f'{tick:.1f}',10,textAnchor='middle')
        for tick in np.arange(math.ceil(ymin/.2)*.2, ymax+.01, .2):
            tick = round(float(tick), 8)
            d.add(Line(x,py(tick),x+w,py(tick),strokeColor=colors.HexColor('#e8edf1')))
            label(d,x-10,py(tick)-3,f'{tick:+.2f}',10,textAnchor='end')
        d.add(Line(x,py(0),x+w,py(0),strokeColor=colors.HexColor('#6b7280'),strokeWidth=1.3,strokeDashArray=[4,3]))
        d.add(Line(x,y,x+w,y,strokeColor=colors.HexColor('#526171')))
        d.add(Line(x,y,x,y+h,strokeColor=colors.HexColor('#526171')))
        label(d,x,y+h+12,'Average net winnings (units / bet)',10)
        label(d,x+w/2,y-40,'Mean max - min probability gap within bin',11,textAnchor='middle')
        for row in table[table.model==model].itertuples():
            xx,yy=px(row.mean_probability_gap),py(row.mean_net_winnings)
            d.add(Circle(xx,yy,5,fillColor=colors.white if row.small_sample else colors.HexColor(color),
                         strokeColor=colors.HexColor(color),strokeWidth=1.8))
            label(d,xx,yy+11,f'n={row.matches}',9,textAnchor='middle')
    label(d,40,31,'Fixed bins of width 0.10; empty bins omitted. Labels show bets per bin; hollow points have fewer than 30 bets.',11)
    label(d,40,12,'Dashed line = break-even. These are retrospective test results, not validated betting thresholds.',11)
    save(d,'probability_gap_winnings')
    report_path=OUT/'report.html'
    report=report_path.read_text(encoding='utf-8')
    report=re.sub(r'<!-- gap-plot:start -->.*?<!-- gap-plot:end -->','',report,flags=re.S)
    section='<!-- gap-plot:start --><h2>Probability gap and average winnings</h2><p>Most-likely outcome, one-unit stake. Fixed 0.10-wide bins; max-minus-min probability gap.</p><img src="probability_gap_winnings.svg" alt="Binned probability gap versus net winnings" style="width:100%;height:auto"><!-- gap-plot:end -->'
    report=report.replace('<h2>Predictive performance</h2>',section+'<h2>Predictive performance</h2>')
    report_path.write_text(report,encoding='utf-8')
    print('Saved probability_gap_winnings.png/.svg and CSV; binned profit totals verified.')


if __name__=='__main__':
    main()
