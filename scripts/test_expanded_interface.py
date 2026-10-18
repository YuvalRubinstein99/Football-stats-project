import sys,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.lineup_deps'));sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
import numpy as np
from expanded_model import ExpandedForest
from manual_strategies import historical,settings

class ExpandedInterface(unittest.TestCase):
    def test_inference_parity(self):
        model=ExpandedForest();test=pd.read_csv(ROOT/'data/expanded_2008/test.csv');pred=pd.read_csv(ROOT/'results/expanded_comparison/match_predictions.csv')
        for i in (0,173,1446):
            actual=model.predict(test.iloc[i].to_dict())
            expected=[pred.iloc[i]['expanded__RF_vector__goals_'+s] for s in ('home','away')]
            np.testing.assert_allclose(actual,expected,atol=1e-10)

    def test_league_and_manual_reconciliation(self):
        root=ROOT/'results/expanded_interface'
        report=json.loads((root/'stake_level_results.json').read_text())
        self.assertEqual((report['train_matches'],report['test_matches']),(19263,1447))
        for amounts in ([1,2,3,4,5],[2,2,2,2,2]):
            h=historical(settings(dict(strategy='Legacy variance score',amounts=amounts)),root/'stake_level_bet_ledger.csv')
            self.assertEqual(len(h['league_curves']),8)
            for split in ('Train (OOB)','Test'):
                actual=sum(c['points'][-1]['net'] for c in h['league_curves'] if c['split']==split)
                expected=next(r['net'] for r in h['rows'] if r['split']==split and r['season']=='All seasons')
                self.assertAlmostEqual(actual,expected)
            if amounts==[1,2,3,4,5]:self.assertAlmostEqual(next(r['net'] for r in h['rows'] if r['split']=='Test' and r['season']=='All seasons'),163.13)
        curves=json.loads((root/'stake_cumulative.json').read_text())
        for c in curves['curves']:
            profit=lambda pts:sum(sum((i+1)*v[1] for i,v in enumerate(p['levels'])) for p in pts)
            total=sum(profit(lg['points']) for lg in curves['league_curves'] if lg['strategy']==c['strategy'] and lg['split']==c['split'])
            self.assertAlmostEqual(total,profit(c['points']))

if __name__=='__main__':unittest.main()
