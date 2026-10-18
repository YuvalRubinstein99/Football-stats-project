"""Cross-check strategy replay against the independently computed bankroll run."""
from select_expanded_features import pd,np,ROOT
from backtest_transfermarkt_strategies import OUT,simulate,SIZING,selections,NAMES

f=pd.DataFrame([dict(fixture_date='2022-01-01')])
a=np.array([[1,3,.12,.08,2.]])
assert simulate(f,a,SIZING[0])['net']==2
assert simulate(f,a,SIZING[1])['net']==6
assert simulate(f,a,SIZING[2])['net']==40 # match cap at $20
assert simulate(f,a,SIZING[3])['net']==40
new=pd.read_csv(OUT/'selection_strategies.csv');old=pd.read_csv(OUT/'bankroll.csv')
keys=['split','model','variant','league','method']
expected=new[new.strategy=='Expected return > 1']
matched=expected.merge(old,on=keys,suffixes=('_new','_old'))
assert len(matched)==540,len(matched)
for col in ['matches','bets','staked','net','max_drawdown']:
    assert np.allclose(matched[col+'_new'],matched[col+'_old'],rtol=1e-9,atol=1e-7),col
assert len(new)==10800 and not new.duplicated(keys+['strategy']).any()
for _,g in new.groupby(['split','model','league','method','strategy']):
    assert set(g.variant)=={'Baseline','With values'} and g.matches.nunique()==1
print('Passed: 10,800 strategy/sizing/league results; paired row counts; stake-cap examples; 540 independent bankroll comparisons.')
