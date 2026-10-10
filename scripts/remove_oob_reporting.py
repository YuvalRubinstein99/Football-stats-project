"""One-time migration of active betting reports to held-out test rows only."""
from pathlib import Path
import csv,json,io
ROOT=Path(__file__).resolve().parents[1]
def edit(name,fn):
    p=ROOT/name;p.write_text(fn(p.read_text(encoding='utf-8')),encoding='utf-8')
def builder(s):
    s=s.replace('vector RF OOB/train and held-out test predictions','held-out vector RF test predictions only')
    for line in ['import numpy as np\n','from lineup_model import LineupForest, clean\n','from live_predictions import probabilities\n','    model = LineupForest()  # Checks training hash, feature order and sklearn version.\n']:
        s=s.replace(line,'')
    a=s.index('    train = clean(');b=s.index('    tests = pd.read_csv',a);s=s[:a]+s[b:]
    s=s.replace("[('Train (OOB)',train),('Test',tests)]","[('Test',tests)]")
    a=s.index("                if split == 'Train (OOB)':");b=s.index('                seasons =',a)
    block=s[s.index('                else:\n',a)+len('                else:\n'):b]
    s=s[:a]+''.join(line[4:] if line.startswith('    ') else line for line in block.splitlines(True))+s[b:]
    a=s.index("    methodology = (");b=s.index('    report = dict',a)
    s=s[:a]+"    methodology = 'Held-out test predictions only. Training profit is unavailable pending a chronological backtest. Historical actual lineups differ from live projected lineups. Gross returns include stakes; net profit subtracts stakes. No compounding, bankroll limit or fees. The test set has already been inspected; filtered presets are exploratory, not independently validated.'\n"+s[b:]
    return s.replace('train_matches=len(train)','train_matches=0')
edit('scripts/backtest_stake_levels.py',builder)
def ui(s):
    a=s.index('const searchLink=');b=s.index('const cumulativeChart=',a);s=s[:a]+s[b:]
    s=s.replace("['Train (OOB)','Test']","['Test']").replace('Train vs test ROI by strategy','Test ROI by strategy')
    s=s.replace('Blue: training OOB; green: held-out test.','Held-out test only.').replace('Blue: Train (OOB) · Green: Test · ','Held-out Test · ')
    a=s.index(' candidateNotice.textContent=');b=s.index('\n',a)
    s=s[:a]+" candidateNotice.textContent='Training-period results are disabled pending a chronological backtest. The filtered search preset is retained as an exploratory rule; its earlier training-based optimization is not active validation.';"+s[b:]
    return s
edit('src/stake-ui.js',ui)
def app(s):
    s=s.replace('Training OOB and held-out test results are shown separately.','Only held-out test results are shown. Training-period results require a chronological backtest.')
    anchor="            if path == '/api/status':"
    s=s.replace("            if path in stake_files:","            if path in ('/strategy-search','/strategy-search.csv'):\n                return self.send(410, {'error': 'The earlier OOB-based strategy search is retired. Use manual filters with held-out test results.'})\n            if path in stake_files:")
    return s
edit('src/prediction_app.py',app)
def manual(s):
    s=s.replace("if r['strategy'] not in STRATEGIES:continue","if r['strategy'] not in STRATEGIES or r['split']!='Test':continue")
    return s.replace("('Train (OOB)','Test')","('Test',)").replace('Training uses OOB predictions, not chronological model refits.','Training-period results are unavailable pending a chronological backtest.')
edit('src/manual_strategies.py',manual)
edit('scripts/search_betting_strategies.py',lambda s:s.replace('def main():','def main():\n    raise RuntimeError("OOB strategy search has been retired. Build chronological predictions before running a new search.")'))
def cumulative(s):
    s=s.replace("[('Train (OOB)','train.csv'),('Test','test.csv')]","[('Test','test.csv')]")
    s=s.replace("            week=weeks[row['split']]", "            if row['split']!='Test':continue\n            week=weeks[row['split']]")
    s=s.replace('Drawing(1150,760)','Drawing(1150,460)').replace('1150,760,fillColor','1150,460,fillColor').replace('35,725,','35,425,').replace('35,698,','35,398,')
    s=s.replace("[('Train (OOB)',641,TRAIN),('Test',351,TEST)]","[('Test',351,TEST)]")
    s=s.replace('Curves use separate vertical scales.','Held-out test only.').replace('OOB excludes each predicted row from its trees, but may use later-season matches. It is not walk-forward testing.','Training-period profit is unavailable pending a chronological backtest.')
    return s
edit('scripts/plot_stake_cumulative.py',cumulative)
def figures(s):
    s=s.replace("['Train (OOB)','Test']","['Test']").replace('Training vs test ROI across all 14 strategies','Held-out test ROI across strategies')
    s=s.replace('Blue: training OOB, 2014-22. Green: test, 2022-23. Different exposure and sample sizes.','Held-out test, 2022-23. Training-period reporting is disabled.')
    s=s.replace('Blue: training OOB (2014-15 to 2021-22). Green: held-out test (2022-23).','Held-out test only (2022-23). Training-period reporting is disabled.')
    s=s.replace('OOB is not walk-forward testing.','Training results require a chronological backtest.').replace('OOB trees exclude the predicted match but may include later seasons; this is not a chronological simulation.','Training-period profit is unavailable pending a chronological backtest.')
    return s
edit('scripts/plot_stake_history.py',figures)
out=ROOT/'results'
p=out/'stake_level_results.json';r=json.loads(p.read_text(encoding='utf-8'));r['rows']=[s for s in r['rows'] if s['split']=='Test'];r['train_matches']=0
r['methodology']='Held-out test predictions only. Training profit is unavailable pending a chronological backtest. No compounding, bankroll constraints or fees. The test set has already been inspected; filtered presets are exploratory, not independently validated.'
p.write_text(json.dumps(r,indent=2),encoding='utf-8')
for name in ['stake_level_results.csv','stake_level_bet_ledger.csv']:
    path=out/name
    with path.open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=[row for row in reader if row['split']=='Test']
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
for name in ['test_filtered_strategy.py','test_manual_filters.py']:
    edit('scripts/'+name,lambda s:s.replace("[('search',{2018,2019}),('validation',{2020,2021}),('test',{2022})]","[('test',{2022})]"))
edit('scripts/test_stake_levels.py',lambda s:s.replace("{'Train (OOB)':11561,'Test':1447}","{'Train (OOB)':0,'Test':1447}"))
print('Active historical reports now contain held-out test rows only.')
