"""Create a focused copy of the original notebook with saved experiment results."""
from pathlib import Path
import copy
import json
import base64
import ast
import pandas as pd
from plot_bivariate_results import main as make_plots
from analyze_prediction_errors import main as analyze_errors
from plot_probability_gap import main as plot_gap
from evaluate_home_loss import main as evaluate_binary
from evaluate_home_win import main as evaluate_binary_win
from rebuild_strategy_results import main as rebuild_strategies
from plot_home_or_skip import main as plot_home_or_skip
from analyze_matchweeks import main as analyze_matchweeks

from project_paths import ROOT as root
rebuild_strategies()
make_plots()
analyze_errors()
analyze_matchweeks()
plot_gap()
evaluate_binary()
evaluate_binary('sqrt_profit')
evaluate_binary_win()
evaluate_binary_win('sqrt_profit')
plot_home_or_skip()
original = json.loads((root / 'notebooks/legacy/prediction.ipynb').read_text(encoding='utf-8'))
notebook = copy.deepcopy(original)


def markdown(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(True)}


def code(text, table=None):
    outputs = []
    if table is not None:
        outputs = [{'output_type': 'display_data', 'metadata': {},
                    'data': {'text/html': [table.to_html(index=False, float_format=lambda v: f'{v:.5f}')],
                             'text/plain': [table.to_string(index=False)]}}]
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {},
            'source': text.splitlines(True), 'outputs': outputs}


metrics = pd.read_csv(root / 'results/model_metrics.csv')
strategies = pd.read_csv(root / 'results/strategy_results.csv')
home_skip_summary = pd.read_csv(root / 'results/home_win_or_skip_strategy.csv')
home_skip_plot = code('from plot_home_or_skip import main as plot_home_or_skip\nplot_home_or_skip()\n'
                      'display(Image(filename=str(experiment.OUT / "home_win_or_skip_strategy.png")))\n')
home_skip_plot['outputs'] = [{'output_type':'display_data','metadata':{},'data':{
    'image/png':base64.b64encode((root/'results/home_win_or_skip_strategy.png').read_bytes()).decode('ascii'),
    'text/plain':['Home win or skip cumulative profit']}}]
implementation = (root / 'src/prediction_bivariate.py').read_text(encoding='utf-8')
vector_function = next(node for node in ast.parse(implementation).body
                       if isinstance(node, ast.FunctionDef) and node.name == 'fit_vector_forest')
vector_source = ast.get_source_segment(implementation, vector_function)
error_slices = pd.read_csv(root / 'results/error_slices.csv')
matchweeks = pd.read_csv(root / 'results/matchweek_errors.csv')
matchweek_plot = code('from analyze_matchweeks import main as analyze_matchweeks\nanalyze_matchweeks()\n'
                      'display(Image(filename=str(experiment.OUT / "matchweek_errors.png")))\n')
matchweek_plot['outputs'] = [{'output_type':'display_data','metadata':{},'data':{
    'image/png':base64.b64encode((root/'results/matchweek_errors.png').read_bytes()).decode('ascii'),
    'text/plain':['Individual matchweek errors: vector RF']}}]
team_errors = pd.read_csv(root / 'results/team_errors.csv')
worst_predictions = pd.read_csv(root / 'results/worst_predictions.csv')
error_columns = ['model','area','group','matches','small_sample','accuracy','mean_log_loss',
                 'mean_confidence','goal_mae','home_goal_bias','away_goal_bias']
failure_plot = code('from analyze_prediction_errors import main as analyze_errors\n'
                    'analyze_errors()\n'
                    'display(Image(filename=str(experiment.OUT / "failure_diagnostics.png")))\n')
failure_plot['outputs'] = [{'output_type':'display_data','metadata':{},'data':{
    'image/png':base64.b64encode((root/'results/failure_diagnostics.png').read_bytes()).decode('ascii'),
    'text/plain':['Confusion matrices and confidence calibration']}}]
gap_plot = code('from plot_probability_gap import main as plot_gap\nplot_gap()\n'
                'display(Image(filename=str(experiment.OUT / "probability_gap_winnings.png")))\n')
gap_plot['outputs'] = [{'output_type':'display_data','metadata':{},'data':{
    'image/png':base64.b64encode((root/'results/probability_gap_winnings.png').read_bytes()).decode('ascii'),
    'text/plain':['Probability gap versus average net winnings']}}]
binary_summary = pd.read_csv(root / 'results/home_loss_strategy.csv')
binary_plot = code('from evaluate_home_loss import main as evaluate_binary\nevaluate_binary()\n'
                   'display(Image(filename=str(experiment.OUT / "home_loss_strategy.png")))\n')
binary_plot['outputs'] = [{'output_type':'display_data','metadata':{},'data':{
    'image/png':base64.b64encode((root/'results/home_loss_strategy.png').read_bytes()).decode('ascii'),
    'text/plain':['Home loses versus home not lose: cumulative winnings']}}]
sqrt_loss_summary = pd.read_csv(root / 'results/home_loss_sqrt_profit_strategy.csv')
sqrt_loss_plot = code('evaluate_binary("sqrt_profit")\n'
                     'display(Image(filename=str(experiment.OUT / "home_loss_sqrt_profit_strategy.png")))\n')
sqrt_loss_plot['outputs'] = [{'output_type':'display_data','metadata':{},'data':{
    'image/png':base64.b64encode((root/'results/home_loss_sqrt_profit_strategy.png').read_bytes()).decode('ascii'),
    'text/plain':['Home loses versus not: square-root-profit weighting']}}]
binary_win_summary = pd.read_csv(root / 'results/home_win_strategy.csv')
binary_win_plot = code('from evaluate_home_win import main as evaluate_binary_win\nevaluate_binary_win()\n'
                       'display(Image(filename=str(experiment.OUT / "home_win_strategy.png")))\n')
binary_win_plot['outputs'] = [{'output_type':'display_data','metadata':{},'data':{
    'image/png':base64.b64encode((root/'results/home_win_strategy.png').read_bytes()).decode('ascii'),
    'text/plain':['Home wins versus home not win: cumulative winnings']}}]
sqrt_win_summary = pd.read_csv(root / 'results/home_win_sqrt_profit_strategy.csv')
sqrt_win_plot = code('evaluate_binary_win("sqrt_profit")\n'
                    'display(Image(filename=str(experiment.OUT / "home_win_sqrt_profit_strategy.png")))\n')
sqrt_win_plot['outputs'] = [{'output_type':'display_data','metadata':{},'data':{
    'image/png':base64.b64encode((root/'results/home_win_sqrt_profit_strategy.png').read_bytes()).decode('ascii'),
    'text/plain':['Home wins versus not: square-root-profit weighting']}}]
plot_cell = code('from IPython.display import Image, display\nfrom plot_bivariate_results import main as make_plots\nmake_plots()\nfor name in ["cumulative_profit", "wins_by_strategy"]:\n    display(Image(filename=str(experiment.OUT / (name + ".png"))))\n')
plot_cell['outputs'] = [{'output_type': 'display_data', 'metadata': {}, 'data': {
    'image/png': base64.b64encode((root / 'results' / (name + '.png')).read_bytes()).decode('ascii'),
    'text/plain': [name.replace('_', ' ')]}} for name in ['cumulative_profit', 'wins_by_strategy']]
notebook['cells'] = [
    markdown('# Bivariate Poisson goal prediction\n\n'
             'Current experiment based on the [historical workflow](legacy/prediction.ipynb). '
             'Historical code and outputs are preserved in `legacy/`, with portable setup/model paths. '
             'Implementation is in `../src/prediction_bivariate.py`; results below are embedded from its completed run. '
             'Notebook cells have not themselves been executed by a Jupyter kernel.\n\n'
             'The same RF grid and train/test files are used. A five-coefficient joint likelihood layer '
             'learns match-specific unshared home, unshared away, and shared rates from out-of-fold '
             'training predictions. Both exact probabilities and 50,000 joint simulations per match '
             'are evaluated. A calibrated independent control helps isolate the dependence contribution.'),
    code('from pathlib import Path\nimport sys\n'
         'PROJECT_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src/prediction_bivariate.py").is_file())\n'
         'sys.path.insert(0, str(PROJECT_ROOT / "src"))\n'
         'import pandas as pd\nimport prediction_bivariate as experiment\n\n'
         '# Set True to rerun training and simulation (several minutes).\n'
         'RERUN = False\n'
         'if RERUN or not (experiment.OUT / "model_metrics.csv").exists():\n'
         '    experiment.main()\n'),
    markdown('## One random forest predicting a goal vector\n'
             'The new `RF_vector` model trains one native multi-output forest with **2,000 trees** on '
             '`Y_train = [home_score, away_score]`. Both targets influence the same tree splits. '
             'Training uses `forest.fit(X_train, Y_train)` and predictions have shape `(n_matches, 2)`. '
             'It uses `criterion="squared_error"` and out-of-bag MSE evaluation, holding prior settings fixed. Predictions average leaf means. '
             'These point predictions use the original independent Poisson conversion as a heuristic for betting evaluation; '
             'shared trees alone do not define a dependent score distribution. '
             'The cell below loads the already-trained forest; set `REFIT_VECTOR=True` to refit just this model.'),
    code('import numpy as np\nimport joblib\n'
         'from sklearn.ensemble import RandomForestRegressor\n'
         'from sklearn.model_selection import GridSearchCV, KFold\n'
         'from sklearn.metrics import mean_squared_error\n'
         'SEED, OUT = experiment.SEED, experiment.OUT\n\n' + vector_source + '\n'),
    code('import joblib\n'
         'X_train, X_test, Y_train, Y_test, *_ = experiment.prepare()\n'
         'REFIT_VECTOR = False\n'
         'if REFIT_VECTOR:\n'
         '    vector_predictions, vector_params = fit_vector_forest(X_train, X_test, Y_train)\n'
         'if (experiment.OUT / "vector_forest.joblib").exists():\n'
         '    vector_rf = joblib.load(experiment.OUT / "vector_forest.joblib")\n'
         '    assert vector_rf.n_outputs_ == 2\n'
         '    vector_predictions = vector_rf.predict(X_test)\n'
         'else:\n'
         '    saved = pd.read_csv(experiment.OUT / "match_predictions.csv")\n'
         '    vector_predictions = saved[["vector_expected_home_goals", "vector_expected_away_goals"]].to_numpy()\n'
         'assert vector_predictions.shape == (len(X_test), 2)\n'
         'display(pd.DataFrame(vector_predictions, columns=["Predicted home goals", "Predicted away goals"]).head())\n'),
    markdown('## Strategy comparison plots\nOriginal independent, joint exact, joint simulated, and single-forest vector results. '
             'Profit curves follow the saved test-row order, which is not verified chronological order.'),
    plot_cell,
    markdown('## Probability gap versus average winnings\n'
             'Gap = maximum minus minimum of the three outcome probabilities. '
             'Each dot shows the mean gap and mean realized **net profit per one-unit bet**, '
             'betting on the most likely outcome. Fixed bins have width 0.10. '
             'Counts appear beside each dot; hollow dots have fewer than 30 bets.'),
    gap_plot,
    code('gap_results = pd.read_csv(experiment.OUT / "probability_gap_winnings.csv")\n'
         'display(gap_results[gap_results.model == "RF_vector"])\n'),
    markdown('## Predictive performance\nLower log loss, Brier score, and joint score NLL are better. '
             'Correct predictions means selecting the most likely of away/draw/home.'),
    code('metrics = pd.read_csv(experiment.OUT / "model_metrics.csv")\ndisplay(metrics)\n', metrics),
    markdown('## Wins per betting strategy\nOne unit staked per bet. Net profit excludes the original '
             'notebook\'s initial one-unit balance; ROI is net profit divided by bets. '
             'Legacy rules are reproduced as heuristics, not presented as valid double-chance bets or Sharpe ratios.'),
    code('strategies = pd.read_csv(experiment.OUT / "strategy_results.csv")\ndisplay(strategies)\n', strategies),
    markdown('## New strategy: does the home team lose?\n'
             'Choose between **home loses** and **home does not lose**, using the existing score probabilities. '
             'The second includes both a home win and a draw. Every match stakes one unit in total.\n\n'
             'No quoted double-chance odds are available, so home-not-to-lose is constructed by splitting '
             'the stake between home win and draw for an equal payout. Its effective decimal odds are '
             '`1 / (1 / home_odds + 1 / draw_odds)`. This pays on a draw, unlike the legacy heuristic. '
             'It assumes both recorded odds are available with fractional stakes and no fees or rounding.'),
    code('binary_summary = pd.read_csv(experiment.OUT / "home_loss_strategy.csv")\ndisplay(binary_summary)\n',binary_summary),
    binary_plot,
    markdown('## Home loses or not: square-root-profit strategy\n'
             'Choose the larger **probability × √(decimal odds − 1)**. '
             'Home loses uses the away-win probability and odds; home does not lose uses '
             'the home-plus-draw probability and synthetic home-or-draw odds. '
             'Stake one unit total per match, with the same split-stake payout assumptions. '
             'Synthetic payouts at or below 1 receive zero selection weight: that side cannot '
             'produce a positive profit even when correct.'),
    code('sqrt_loss_summary = pd.read_csv(experiment.OUT / "home_loss_sqrt_profit_strategy.csv")\n'
         'display(sqrt_loss_summary)\n',sqrt_loss_summary),
    sqrt_loss_plot,
    markdown('## New strategy: does the home team win?\n'
             'Choose the more likely of **home wins** and **home does not win**. '
             'Home does not win includes an away win or a draw. Stake one unit in total on each match. '
             'The no-win side splits the stake between away and draw, giving effective decimal odds '
             '`1 / (1 / away_odds + 1 / draw_odds)`. These are synthetic payouts, not quoted '
             'double-chance odds. The same fractional-stake and no-fee assumptions apply.'),
    code('binary_win_summary = pd.read_csv(experiment.OUT / "home_win_strategy.csv")\ndisplay(binary_win_summary)\n',binary_win_summary),
    binary_win_plot,
    markdown('## Home wins or not: square-root-profit strategy\n'
             'Choose the side with the larger **probability × √(decimal odds − 1)**. '
             'Use the synthetic away-or-draw odds for home not winning. '
             'The stake remains one unit per match; this weighting changes the selected side, not the stake or payout.'),
    code('sqrt_win_summary = pd.read_csv(experiment.OUT / "home_win_sqrt_profit_strategy.csv")\n'
         'display(sqrt_win_summary)\n',sqrt_win_summary),
    sqrt_win_plot,
    markdown('## Home win or skip: square-root-profit weighting\n'
             'Use the same binary comparison above: **p(home) × √(home odds − 1)** versus '
             '**(p(away) + p(draw)) × √(synthetic away-or-draw odds − 1)**. '
             'Bet one unit on home when its score is at least as high; otherwise **do not bet**. '
             'There is no additional expected-utility or positive-return filter. '
             'A draw or away win loses a placed bet. Skips have zero stake and profit; ROI divides by placed bets. '
             'Square-root weighting affects selection only; reported profit uses actual decimal payouts.'),
    code('home_skip_summary = pd.read_csv(experiment.OUT / "home_win_or_skip_strategy.csv")\n'
         'display(home_skip_summary)\n', home_skip_summary),
    home_skip_plot,
    markdown('## Where do the models fail?\n'
             'Compare actual versus predicted outcomes and confidence versus observed accuracy below. '
             'The CSV diagnostics cover **all models**. Group sizes below 30 are flagged; '
             'small differences in small groups should not be treated as established weaknesses. '
             'Actual-outcome accuracy is recall for that outcome. Actual outcomes and total goals are '
             'retrospective slices, not pre-match predictors. Positive goal bias means overprediction.'),
    failure_plot,
    code('error_slices = pd.read_csv(experiment.OUT / "error_slices.csv")\n'
         'ERROR_MODEL = "RF_vector"  # Change to RF_independent, RF_bivariate_exact, Market, etc.\n'
         'error_columns = '+repr(error_columns)+'\n'
         'display(error_slices.loc[error_slices.model == ERROR_MODEL, error_columns])\n',
         error_slices.loc[error_slices.model=='RF_vector',error_columns]),
    markdown('### Which matchweeks have the highest errors?\n'
             'Individual matchweeks are ranked by **outcome error rate** (wrong home/draw/away predictions), '
             'breaking ties by log loss. The table also shows mean log loss and goal MSE/MAE. '
             'Rank 1 means highest error for that metric. Counts below 30 are flagged; all weeks remain visible. '
             'Matches are pooled across the test data by matchweek number, so league and season composition '
             'may differ. This is descriptive, not evidence that matchweek causes errors. '
             'The plot shows the vector RF; change ERROR_MODEL below to inspect another model.'),
    code('matchweeks = pd.read_csv(experiment.OUT / "matchweek_errors.csv")\n'
         'display(matchweeks[matchweeks.model == ERROR_MODEL].sort_values(["error_rate", "mean_log_loss"], '
         'ascending=False).head(10))\n', matchweeks[matchweeks.model=='RF_vector'].head(10)),
    matchweek_plot,
    markdown('### Compare outcome recall across models'),
    code('outcome_errors = error_slices[error_slices.area == "Actual outcome (retrospective)"]\n'
         'display(outcome_errors.pivot(index="model", columns="group", values="accuracy").reset_index())\n',
         error_slices[error_slices.area=='Actual outcome (retrospective)'].pivot(index='model',columns='group',values='accuracy').reset_index()),
    markdown('### Most difficult teams\nTeams are ranked by average outcome log loss, with at least 30 matches. '
             'Each team group includes both home and away games; these groups overlap.'),
    code('team_errors = pd.read_csv(experiment.OUT / "team_errors.csv")\n'
         'display(team_errors[(team_errors.model == ERROR_MODEL) & (team_errors.matches >= 30)]'
         '.sort_values("mean_log_loss", ascending=False).head(10)[error_columns])\n',
         team_errors[(team_errors.model=='RF_vector')&(team_errors.matches>=30)].sort_values('mean_log_loss',ascending=False).head(10)[error_columns]),
    markdown('### Inspect individual failures\nThe wrong predictions with the largest log loss: '
             'matches where the actual result received particularly low probability.'),
    code('worst_predictions = pd.read_csv(experiment.OUT / "worst_predictions.csv")\n'
         'display(worst_predictions[worst_predictions.model == ERROR_MODEL])\n',
         worst_predictions[worst_predictions.model=='RF_vector']),
    markdown('### Calibration for each outcome\nCompare predicted probability with actual frequency '
             'separately for away wins, draws, and home wins. The plot above instead checks confidence in '
             'the most likely result. Confidence bins can contain different matches across models.'),
    code('calibration = pd.read_csv(experiment.OUT / "outcome_calibration.csv")\n'
         'display(calibration[(calibration.model == ERROR_MODEL) & (calibration.matches >= 30)])\n'),
    markdown('## Per-match probabilities and fitted rates'),
    code('predictions = pd.read_csv(experiment.OUT / "match_predictions.csv")\ndisplay(predictions.head())\n'),
    markdown('## Method and limitations\n\n' + (root / 'results/methodology.txt').read_text(encoding='utf-8')),
]
notebook['metadata']['bivariate_experiment'] = {
    'source_notebook': 'notebooks/legacy/prediction.ipynb', 'implementation': 'src/prediction_bivariate.py',
    'outputs': 'Embedded from completed script run; notebook kernel execution counts are unset.'}
notebook['nbformat_minor'] = 4
(root / 'notebooks/prediction_bivariate.ipynb').write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding='utf-8')
print('Created prediction_bivariate.ipynb with saved result tables.')
