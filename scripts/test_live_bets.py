"""Check recommendation parity, true probabilities, and split-stake accounting."""
from pathlib import Path
import sys
import unittest
import math
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from betting_rules import strategy_scores
from live_bets import recommendations, STRATEGIES

def match(p,o):
    return dict(home='H',away='A',p_away=p[0],p_draw=p[1],p_home=p[2],B365A=o[0],B365D=o[1],B365H=o[2])

class StrategyTests(unittest.TestCase):
    def test_existing_rules_parity(self):
        rng=np.random.default_rng(42)
        for _ in range(100):
            p=rng.dirichlet([2,2,2]); o=rng.uniform(1.05,8,3)
            market=1/o; market/=market.sum()
            ref=strategy_scores(p[None,:],o[None,:],market[None,:])
            actual={r['strategy']:r for r in recommendations(match(p,o))}
            self.assertEqual(set(actual),set(STRATEGIES))
            for name,(scores,threshold) in ref.items():
                pick=int(scores[0].argmax()); bet=scores[0,pick]>threshold; r=actual[name]
                self.assertEqual(r['action'],'Bet' if bet else 'Skip',name)
                if bet:
                    self.assertEqual(r['selection'],['A wins','Draw','H wins'][pick],name)
                    self.assertAlmostEqual(r['expected_net_per_unit'],p[pick]*o[pick]-1)

    def test_binary_selection_and_accounting(self):
        p=np.array([.3,.4,.3]); o=np.array([3.2,3.4,3.1])
        for r in recommendations(match(p,o))[10:]:
            stakes=np.array([r['away_stake'],r['draw_stake'],r['home_stake']])
            self.assertAlmostEqual(stakes.sum(),1.)
            self.assertAlmostEqual(r['expected_net_per_unit'],sum(stakes*o*p)-1)
            if r['synthetic']:
                payouts=(stakes*o)[stakes>0]
                self.assertAlmostEqual(payouts[0],payouts[1])
                self.assertAlmostEqual(payouts[0],r['decimal_odds'])
            single=0 if r['strategy'].startswith('Home loses') else 2
            others=[i for i in range(3) if i!=single]; hedge=1/sum(1/o[i] for i in others)
            left,right=p[single],p[others].sum()
            if 'sqrt profit' in r['strategy']:
                left*=math.sqrt(o[single]-1); right*=math.sqrt(max(hedge-1,0))
            self.assertEqual(r['synthetic'],left<right)

    def test_skips_invalid_odds_and_ties(self):
        rows=recommendations(match([.3,.4,.3],[2,2,2]))
        self.assertEqual(rows[1]['action'],'Skip')
        self.assertEqual(rows[1]['home_stake'],0)
        self.assertTrue(all(r['action']=='Unavailable' for r in recommendations(match([.3,.4,.3],[None,3,2]))))
        self.assertEqual(recommendations(match([1/3]*3,[3]*3))[0]['selection'],'A wins')

    def test_legacy_shift_keeps_actual_probability(self):
        r=recommendations(match([.2,.5,.3],[4,3,2]))[5]
        self.assertEqual(r['selection'],'H wins')
        self.assertEqual(r['probability'],.3)
        self.assertAlmostEqual(r['expected_net_per_unit'],-.4)

if __name__=='__main__':
    unittest.main()
