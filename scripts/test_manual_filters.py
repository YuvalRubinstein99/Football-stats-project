import json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from manual_strategies import settings,gate,historical,evaluate_match


class ManualFilters(unittest.TestCase):
    def config(self,**kw):return settings(dict(strategy='Legacy variance score',**kw))

    def test_boundaries_and_outcome(self):
        c=self.config()
        self.assertEqual(gate(4,.02,'away',c)[0],0)
        self.assertGreater(gate(4,.021,'away',c)[0],0)
        self.assertEqual(gate(3.99,.3,'away',c)[0],0)
        self.assertEqual(gate(5,.3,'combined',self.config(outcome='home'))[0],0)

    def test_invalid_settings(self):
        for args in [dict(min_odds=10,max_odds=2),dict(width=0),dict(min_edge=float('nan')),dict(amounts=[1,2,3,4,.5]),dict(amounts=[1,2,3,4,5.001])]:
            with self.assertRaises(ValueError):self.config(**args)

    def test_live_amount_and_choice(self):
        m=dict(home='H',away='A',p_away=.3,p_draw=.3,p_home=.4,B365A=5,B365D=4,B365H=2)
        b=evaluate_match(m,self.config(sizing='flat',amounts=[2,3,4,5,6]))
        self.assertEqual(b['decision'],'Bet');self.assertEqual(b['amount'],2)
        self.assertEqual(b['selection'],'A wins')
        b=evaluate_match(m,self.config(outcome='home'))
        self.assertEqual(b['decision'],'Skip');self.assertEqual(b['amount'],0)

    def test_history_reproduces_search_and_curves(self):
        h=historical(self.config());search=json.loads((ROOT/'results/strategy_search.json').read_text(encoding='utf-8'))['chosen']['profit']
        for period,years in [('test',{2022})]:
            rows=[s for s in h['rows'] if s['season']!='All seasons' and int(s['season'][:4]) in years]
            self.assertAlmostEqual(sum(s['net'] for s in rows),search[period]['net'])
            self.assertEqual(sum(s['bets'] for s in rows),search[period]['bets'])
        for curve in h['curves']:
            total=next(s for s in h['rows'] if s['split']==curve['split'] and s['season']=='All seasons')
            self.assertAlmostEqual(curve['points'][-1]['net'],total['net'])
        flat=historical(self.config(sizing='flat'))
        test=next(s for s in flat['rows'] if s['split']=='Test' and s['season']=='All seasons')
        self.assertAlmostEqual(test['net'],-11.04)


if __name__=='__main__':unittest.main()
