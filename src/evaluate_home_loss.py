"""Binary home-loss decision with a stake-balanced home/draw hedge."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from reportlab.graphics.shapes import Drawing, Rect, Line
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.lib import colors
from plot_bivariate_results import label, save, MODELS
from result_files import write_csv

from project_paths import ROOT, DATA, OUT

STRATEGY = 'Home loses vs not (synthetic double chance)'


def main(weighting="probability"):
    if weighting not in {"probability", "sqrt_profit"}:
        raise ValueError("Unknown weighting")
    weighted = weighting == "sqrt_profit"
    strategy = 'Home loses vs not (sqrt profit)' if weighted else STRATEGY
    stem = 'home_loss_sqrt_profit' if weighted else 'home_loss'
    marker = 'binary-home-loss-sqrt' if weighted else 'binary-home'
    title = 'Home loses vs not: square-root-profit weighting' if weighted else 'Home loses or does not lose: cumulative winnings'

    p = pd.read_csv(OUT/'match_predictions.csv')
    models = pd.read_csv(OUT/'model_metrics.csv').model.tolist()
    a, d, h = p.B365A.to_numpy(), p.B365D.to_numpy(), p.B365H.to_numpy()
    # Dutch two mutually exclusive bets to receive the same gross payout for H or D.
    no_loss_odds = 1 / (1/h + 1/d)
    home_weight, draw_weight = no_loss_odds/h, no_loss_odds/d
    assert np.allclose(home_weight+draw_weight,1)
    assert np.allclose(home_weight*h,draw_weight*d)
    actual_loss = (p.home_score < p.away_score).to_numpy()
    rows, ledgers, common = [], [], []
    for model in models:
        loss_probability = p[model+'_Away'].to_numpy()
        no_loss_probability = p[model+'_Home'].to_numpy()+p[model+'_Draw'].to_numpy()
        assert np.allclose(loss_probability+no_loss_probability,1)
        if weighted:
            if np.any(a <= 1):
                raise ValueError("Away-win decimal odds must exceed 1")
            loss_value = loss_probability * np.sqrt(a - 1)
            # A synthetic payout <= 1 cannot yield positive net profit even if correct.
            # Give that dominated side zero weight (avoid sqrt of a negative profit).
            no_loss_value = no_loss_probability * np.sqrt(np.maximum(no_loss_odds - 1, 0))
        else:
            loss_value, no_loss_value = loss_probability, no_loss_probability
        choose_loss = loss_value >= no_loss_value
        win = choose_loss == actual_loss
        odds = np.where(choose_loss,a,no_loss_odds)
        profit = np.where(win,odds-1,-1.)
        away_stake = choose_loss.astype(float)
        home_stake = np.where(choose_loss,0,home_weight)
        draw_stake = np.where(choose_loss,0,draw_weight)
        realized = (away_stake*a*actual_loss + home_stake*h*(p.home_score>p.away_score).to_numpy()
                    + draw_stake*d*(p.home_score==p.away_score).to_numpy()-1)
        assert np.allclose(realized,profit)
        assert np.allclose(away_stake+home_stake+draw_stake,1)
        rows.append({'model':model,'strategy':strategy,'bets':len(p),'wins':int(win.sum()),
                     'losses':int((~win).sum()),'win_rate':win.mean(),'net_profit_units':profit.sum(),
                     'roi':profit.mean(),'home_loss_bets':int(choose_loss.sum()),
                     'home_not_loss_bets':int((~choose_loss).sum()),
                     'loss_side_profit':profit[choose_loss].sum(),
                     'not_loss_side_profit':profit[~choose_loss].sum()})
        ledger = pd.DataFrame({'model':model,'strategy':strategy,'test_position':np.arange(len(p)),
                               'source_test_row':p.source_test_row,'p_home_loses':loss_probability,
                               'p_home_not_lose':no_loss_probability,
                               'loss_selection_score':loss_value,'no_loss_selection_score':no_loss_value,
                               'chosen_side':np.where(choose_loss,'Home loses','Home not lose'),
                               'away_stake':away_stake,'home_stake':home_stake,'draw_stake':draw_stake,
                               'effective_decimal_odds':odds,'win':win,'net_profit_units':profit})
        ledgers.append(ledger)
        common.append(pd.DataFrame({'model':model,'strategy':strategy,'test_position':np.arange(len(p)),
                                    'pick':np.where(choose_loss,0,3),'bet':True,'win':win,
                                    'decimal_odds':odds,'net_profit_units':profit}))
    summary = pd.DataFrame(rows)
    binary_ledger = pd.concat(ledgers,ignore_index=True)
    summary.to_csv(OUT/(stem+'_strategy.csv'),index=False)
    binary_ledger.to_csv(OUT/(stem+'_bet_ledger.csv'),index=False)
    existing = pd.read_csv(OUT/'strategy_results.csv')
    columns = ['model','strategy','bets','wins','losses','win_rate','net_profit_units','roi']
    write_csv(pd.concat([existing[existing.strategy!=strategy],summary[columns]],ignore_index=True), OUT/'strategy_results.csv')
    existing_ledger = pd.read_csv(OUT/'bet_ledger.csv')
    write_csv(pd.concat([existing_ledger[existing_ledger.strategy!=strategy],*common],ignore_index=True), OUT/'bet_ledger.csv')
    shown = MODELS+[('Market','Market probabilities','#657386')]
    curves=[]
    for model,_,_ in shown:
        values=binary_ledger[binary_ledger.model==model].net_profit_units.cumsum().to_numpy()
        curves.append([(0,0.)]+list(enumerate(values,1)))
    all_values=[value for curve in curves for _,value in curve]
    ymin=25*np.floor(min(all_values)/25)
    ymax=max(25,25*np.ceil(max(all_values)/25))
    figure=Drawing(1200,680)
    figure.add(Rect(0,0,1200,680,fillColor=colors.white,strokeColor=None))
    label(figure,40,637,title,24)
    label(figure,40,610,'Choose max probability x sqrt(odds - 1). One unit total per match.' if weighted else 'Choose the more likely binary outcome. One unit total per match; draws win the home-not-lose bet.',12)
    label(figure,40,589,'Home-not-lose odds are synthesized by splitting the stake between recorded home-win and draw odds.',11)
    plot=LinePlot()
    plot.x,plot.y,plot.width,plot.height=85,115,850,410
    plot.data=curves
    plot.xValueAxis.valueMin,plot.xValueAxis.valueMax=0,len(p)
    plot.xValueAxis.valueSteps=[0,500,1000,len(p)]
    plot.yValueAxis.valueMin,plot.yValueAxis.valueMax=ymin,ymax
    plot.yValueAxis.visibleGrid=True
    plot.yValueAxis.gridStrokeColor=colors.HexColor('#e1e7ed')
    for j,(_,name,color) in enumerate(shown):
        plot.lines[j].strokeColor=colors.HexColor(color)
        plot.lines[j].strokeWidth=1.6
        figure.add(Line(970,500-32*j,994,500-32*j,strokeColor=colors.HexColor(color),strokeWidth=2))
        label(figure,1002,496-32*j,name,10)
    plot.lines[2].strokeDashArray=[3,2]
    figure.add(plot)
    label(figure,85,542,'Cumulative net profit (units)',11)
    label(figure,510,81,'Test match order',11,textAnchor='middle')
    label(figure,40,40,'Synthetic payout = 1 / (1 / home odds + 1 / draw odds). These are not quoted double-chance prices.',11)
    label(figure,40,20,'Assumes both component bets can be placed at recorded prices with fractional stakes; excludes fees and rounding.',11)
    save(figure,stem+'_strategy')
    notes=('This is an outcome conversion of the existing score models, not a new binary-trained model. '
           'P(home loses)=P(away win); P(home not lose)=P(home win)+P(draw). '
           'Bet on the more likely side, with ties assigned to home loses. There are no double-chance '
           'prices in the CSV. The no-loss side is constructed by splitting a one-unit total stake '
           'between home and draw: D=1/(1/home_odds+1/draw_odds), home stake=D/home_odds, '
           'draw stake=D/draw_odds. Either home win or draw returns D units; away win returns zero. '
           'An away-win bet uses B365A. Unlike the legacy draw-to-home heuristic, an actual draw '
           'is a winning outcome for this no-loss bet, with its own lower synthetic payout. '
           'Assumes fractional stakes and simultaneous availability of recorded odds, without fees or rounding. '
           'In bet_ledger.csv, pick=3 denotes home-or-draw; pick=0 denotes away. '
           'The dedicated ledger records both component stakes. Wins counts correct binary selections; '
           'net winnings are reported separately using the effective odds.')
    if weighted:
        notes = notes.replace('Bet on the more likely side', 'Bet on the side with the greater probability times sqrt(effective decimal odds minus one)')
        notes += (' Synthetic no-loss payouts at or below 1 get zero selection weight, since they cannot '
                  'earn positive net profit even when correct. This occurs in '
                  + str(int((no_loss_odds <= 1).sum())) + ' matches. All matches still stake one unit.')
    (OUT/(stem+'_methodology.txt')).write_text(notes,encoding='utf-8')
    report_path=OUT/'report.html'
    report=report_path.read_text(encoding='utf-8')
    report=re.sub(rf'<!-- {marker}:start -->.*?<!-- {marker}:end -->','',report,flags=re.S)
    section=f'<!-- {marker}:start --><h2>{title}</h2><p>'+notes+'</p>'
    section+=summary.to_html(index=False,float_format=lambda x:f'{x:.4f}')
    section+=f'<img src="{stem}_strategy.svg" alt="Home loss binary strategy winnings" style="width:100%;height:auto"><!-- {marker}:end -->'
    report=report.replace('<h2>Method and limitations</h2>',section+'<h2>Method and limitations</h2>')
    report_path.write_text(report,encoding='utf-8')
    print(summary.to_string(index=False))
    print('Verified component stake totals, payouts on all actual outcomes, and saved binary strategy.')


if __name__=='__main__':
    main()
