"""Descriptive held-out failure analysis; does not tune or train any model."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from reportlab.graphics.shapes import Drawing, Rect, Line
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.lib import colors
from plot_bivariate_results import label, save

from project_paths import ROOT, DATA, OUT

OUTCOMES = np.array(['Away', 'Draw', 'Home'])
NAMES = {'RF_independent': 'Separate RFs', 'RF_independent_calibrated': 'Calibrated separate RFs',
         'RF_bivariate_exact': 'Joint Poisson', 'RF_bivariate_simulated': 'Joint simulation',
         'RF_vector': 'Vector RF (2,000 trees)', 'Market': 'Market'}
MIN_GROUP = 30


def summarize(frame, model, area, group):
    n = len(frame)
    accuracy = frame.correct.mean()
    return {'model': model, 'area': area, 'group': str(group), 'matches': n,
            'small_sample': n < MIN_GROUP, 'correct': int(frame.correct.sum()),
            'accuracy': accuracy, 'error_rate': 1-accuracy,
            'mean_log_loss': frame.loss.mean(), 'mean_brier': frame.brier.mean(),
            'mean_confidence': frame.confidence.mean(),
            'confidence_minus_accuracy': frame.confidence.mean()-accuracy,
            'home_goal_bias': frame.home_residual.mean(), 'away_goal_bias': frame.away_residual.mean(),
            'goal_mae': frame.goal_mae.mean(), 'goal_mse': frame.goal_mse.mean()}


def main():
    predictions = pd.read_csv(OUT / 'match_predictions.csv')
    raw = pd.read_csv(DATA / 'test.csv').iloc[predictions.source_test_row.to_numpy()].reset_index(drop=True)
    assert len(raw) == len(predictions)
    assert (raw.home_score.to_numpy() == predictions.home_score.to_numpy()).all()
    actual = np.sign(predictions.home_score-predictions.away_score).astype(int).to_numpy()+1
    totals = predictions.home_score + predictions.away_score
    gap = raw.HomePlayer_Overall_mean - raw.AwayPlayer_Overall_mean
    weeks = raw.Matchweek.astype(str).str.split().str[-1].astype(int)
    # Fixed, interpretable bins; not selected after looking at error rates.
    cohorts = {
        'Actual outcome (retrospective)': pd.Series(OUTCOMES[actual]),
        'Actual total goals (retrospective)': pd.cut(totals, [-1,1,3,5,np.inf], labels=['0–1','2–3','4–5','6+']),
        'Player rating gap (absolute)': pd.cut(gap.abs(), [-np.inf,2,5,np.inf], labels=['0–2','2–5','5+']),
        'Matchweek': pd.cut(weeks, [0,10,20,30,np.inf], labels=['1–10','11–20','21–30','31+']),
        'Individual matchweek': weeks,
        'Market favorite strength': pd.cut(predictions[['Market_Away','Market_Draw','Market_Home']].max(axis=1),
                                          [0,.4,.55,.7,1], labels=['≤40%','40–55%','55–70%','70%+'])}
    mean_prefix = {'RF_independent':'independent', 'RF_independent_calibrated':'calibrated',
                   'RF_bivariate_exact':'joint', 'RF_bivariate_simulated':'joint', 'RF_vector':'vector'}
    rows, team_rows, examples, calibrations = [], [], [], []
    confusions = {}
    for model in NAMES:
        prob = predictions[[model+'_'+s for s in OUTCOMES]].to_numpy()
        assert np.allclose(prob.sum(axis=1), 1)
        pick = prob.argmax(axis=1)
        f = pd.DataFrame({'correct': pick==actual, 'confidence': prob.max(axis=1),
                          'loss': -np.log(np.clip(prob[np.arange(len(prob)), actual], 1e-15, 1)),
                          'brier': ((prob-np.eye(3)[actual])**2).sum(axis=1)})
        for c in ['home_residual','away_residual','goal_mae','goal_mse']:
            f[c] = np.nan
        if model in mean_prefix:
            prefix = mean_prefix[model]
            rh = predictions[prefix+'_expected_home_goals'] - predictions.home_score
            ra = predictions[prefix+'_expected_away_goals'] - predictions.away_score
            f['home_residual'], f['away_residual'] = rh, ra
            f['goal_mae'], f['goal_mse'] = (rh.abs()+ra.abs())/2, (rh**2+ra**2)/2
        model_cohorts = dict(cohorts)
        model_cohorts['Prediction confidence'] = pd.cut(f.confidence, [0,.4,.5,.6,.7,.8,1],
                                                       labels=['≤40%','40–50%','50–60%','60–70%','70–80%','80%+'])
        rows.append(summarize(f, model, 'Overall', 'All matches'))
        for area, categories in model_cohorts.items():
            for group, indices in categories.groupby(categories, observed=True).groups.items():
                rows.append(summarize(f.loc[indices], model, area, group))
        # A match appears once for each participating team; team rows are not disjoint cohorts.
        for team in sorted(set(predictions.home_team_name) | set(predictions.away_team_name)):
            mask = (predictions.home_team_name == team) | (predictions.away_team_name == team)
            team_rows.append(summarize(f[mask], model, 'Team (home or away)', team))
        matrix = np.zeros((3,3), dtype=int)
        np.add.at(matrix, (actual,pick), 1)
        confusions[model] = matrix
        for outcome in range(3):
            bins = pd.cut(prob[:,outcome], [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1], include_lowest=True)
            for bucket in bins.categories:
                mask = bins == bucket
                if mask.any():
                    calibrations.append({'model':model,'outcome':OUTCOMES[outcome], 'bin':str(bucket),
                                         'matches':int(mask.sum()), 'mean_probability':prob[mask,outcome].mean(),
                                         'observed_frequency':float((actual[mask]==outcome).mean())})
        wrong = f[~f.correct].sort_values('loss', ascending=False).head(15)
        ex = predictions.loc[wrong.index, ['source_test_row','home_team_name','away_team_name','home_score','away_score']].copy()
        ex.insert(0,'model',model)
        ex['predicted_outcome'] = OUTCOMES[pick[wrong.index]]
        ex['confidence'] = f.loc[wrong.index,'confidence']
        ex['actual_outcome_probability'] = prob[wrong.index,actual[wrong.index]]
        ex['log_loss'] = f.loc[wrong.index,'loss']
        for j, outcome in enumerate(OUTCOMES):
            ex['p_'+outcome] = prob[wrong.index,j]
        examples.append(ex)
    slices = pd.DataFrame(rows)
    teams = pd.DataFrame(team_rows)
    bad_matches = pd.concat(examples, ignore_index=True)
    calibration = pd.DataFrame(calibrations)
    slices.to_csv(OUT/'error_slices.csv', index=False)
    teams.to_csv(OUT/'team_errors.csv', index=False)
    bad_matches.to_csv(OUT/'worst_predictions.csv', index=False)
    calibration.to_csv(OUT/'outcome_calibration.csv', index=False)
    assert (slices[slices.area=='Overall'].matches == len(predictions)).all()

    d = Drawing(1200, 880)
    d.add(Rect(0,0,1200,880,fillColor=colors.white,strokeColor=None))
    label(d,40,842,'Where predictions fail',25)
    label(d,40,816,'Confusion matrices: rows = actual outcome; columns = predicted outcome. Cells show match counts.',12)
    for j, model in enumerate(['RF_independent','RF_bivariate_exact','RF_vector']):
        matrix = confusions[model]
        x, top, w = 80+j*390, 745, 80
        label(d,x,778,NAMES[model],15)
        for k, outcome in enumerate(OUTCOMES):
            label(d,x+k*w+w/2,top+10,outcome,11,textAnchor='middle')
            label(d,x-8,top-k*w-w/2,outcome,11,textAnchor='end')
            for col in range(3):
                fraction = matrix[k,col]/max(1,matrix[k].sum())
                shade = colors.Color(1-.80*fraction,1-.55*fraction,1-.20*fraction)
                d.add(Rect(x+col*w,top-(k+1)*w,w,w,fillColor=shade,strokeColor=colors.white))
                label(d,x+col*w+w/2,top-k*w-w/2-4,str(matrix[k,col]),15,textAnchor='middle')
    label(d,45,454,'Confidence versus actual accuracy',18)
    label(d,45,432,'Each point is a confidence bin with at least 30 matches. Below the dashed line = overconfident.',11)
    plot = LinePlot()
    plot.x,plot.y,plot.width,plot.height = 80,115,660,280
    lines, legend_rows = [[(.3,.3),(1,1)]], [('Perfect calibration','#9aa3af')]
    for model,color in [('RF_independent','#2864b4'),('RF_bivariate_exact','#098574'),('RF_vector','#9453b2'),('Market','#ce7821')]:
        subset = slices[(slices.model==model)&(slices.area=='Prediction confidence')&(slices.matches>=MIN_GROUP)]
        pairs = sorted(zip(subset.mean_confidence,subset.accuracy))
        if pairs:
            lines.append(pairs)
            legend_rows.append((NAMES[model],color))
    plot.data = lines
    for axis in [plot.xValueAxis,plot.yValueAxis]:
        axis.valueMin,axis.valueMax = .3,1
        axis.valueSteps = [.3,.4,.5,.6,.7,.8,.9,1]
    plot.yValueAxis.visibleGrid = True
    plot.yValueAxis.gridStrokeColor = colors.HexColor('#e1e7ed')
    for j,(_,color) in enumerate(legend_rows):
        plot.lines[j].strokeColor = colors.HexColor(color)
        plot.lines[j].strokeWidth = 2
    plot.lines[0].strokeDashArray = [4,3]
    d.add(plot)
    label(d,400,82,'Mean predicted confidence',11,textAnchor='middle')
    label(d,80,403,'Observed accuracy',11)
    for j,(name,color) in enumerate(legend_rows):
        y = 364-30*j
        d.add(Line(805,y+4,835,y+4,strokeColor=colors.HexColor(color),strokeWidth=2))
        label(d,845,y,name,11)
    label(d,40,40,'Descriptive test-set diagnostics, not new training features. Small groups are flagged in the tables.',11)
    label(d,40,20,'Actual outcome and goal-total slices are retrospective; confidence bins can contain different matches for each model.',11)
    save(d,'failure_diagnostics')
    main_models = ['RF_independent','RF_bivariate_exact','RF_vector','Market']
    selected = slices[slices.model.isin(main_models)]
    columns = ['model','area','group','matches','accuracy','mean_log_loss','mean_confidence','goal_mae','home_goal_bias','away_goal_bias']
    difficult_teams = teams[(teams.matches>=MIN_GROUP)&teams.model.isin(main_models)].sort_values(['model','mean_log_loss'],ascending=[True,False]).groupby('model',sort=False).head(5)
    notes = ('Positive goal bias means overprediction; negative means underprediction. '
             'Outcome accuracy within an actual-outcome slice is that outcome\'s recall. '
             'Team groups count matches with the team at either venue and overlap. '
             'Only teams with at least 30 matches are ranked. Tables show descriptive results, not proven causes of failure. '
             'Confidence calibration assesses the most likely outcome; outcome_calibration.csv also assesses each of away/draw/home separately.')
    section = '<!-- failure-analysis:start --><h2>Where do the models fail?</h2><p>'+notes+'</p>'
    section += '<img src="failure_diagnostics.svg" alt="Confusion matrices and confidence calibration" style="width:100%;height:auto">'
    section += '<h3>Performance by match type</h3>'+selected[columns].to_html(index=False,float_format=lambda v:f'{v:.4f}')
    section += '<h3>Most difficult teams (at least 30 matches)</h3>'+difficult_teams[columns].to_html(index=False,float_format=lambda v:f'{v:.4f}')
    section += '<h3>Most costly prediction errors: vector forest</h3>'+bad_matches[bad_matches.model=='RF_vector'].to_html(index=False,float_format=lambda v:f'{v:.4f}')
    section += '<!-- failure-analysis:end -->'
    report_path = OUT/'report.html'
    report = report_path.read_text(encoding='utf-8')
    report = re.sub(r'<!-- failure-analysis:start -->.*?<!-- failure-analysis:end -->','',report,flags=re.S)
    report = report.replace('<h2>Method and limitations</h2>',section+'<h2>Method and limitations</h2>')
    report_path.write_text(report,encoding='utf-8')
    print('Error analysis saved: outcome/goal/rating/week/confidence slices, teams, calibration, worst matches.',flush=True)


if __name__ == '__main__':
    main()
