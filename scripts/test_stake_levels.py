"""Stake boundaries, skips, split accounting and exported backtest reconciliation."""
from pathlib import Path
import sys
import csv
import json
import unittest
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from stake_levels import size_bet
from live_bets import recommendations, enrich, export_csv


class StakesTests(unittest.TestCase):
    def test_boundaries_and_skip(self):
        bet=dict(action='Bet',decimal_odds=3,expected_net_per_unit=0,
                 home_stake=1,draw_stake=0,away_stake=0,note='')
        for score,expected in [(0,0),(-.01,0),(.001,1),(.01,1),(.01001,2),(.02,2),(.03,3),(.04,4),(.04001,5),(.9,5)]:
            bet['expected_net_per_unit']=score*2
            result=size_bet(bet)
            self.assertEqual(result['stake_level'],expected)
            self.assertEqual(result['stake_dollars'],expected)
        bet.update(action='Skip',expected_net_per_unit=.5)
        self.assertEqual(size_bet(bet)['stake_dollars'],0)

    def test_synthetic_dollars(self):
        match=dict(home='H',away='A',p_home=.45,p_draw=.35,p_away=.2,B365H=3,B365D=4,B365A=5)
        bet=recommendations(match)[10]
        self.assertTrue(bet['synthetic'])
        result=size_bet(bet)
        self.assertGreater(result['stake_level'],0)
        self.assertAlmostEqual(sum(result[k+'_dollars'] for k in ('home','draw','away')),result['stake_dollars'])
        self.assertAlmostEqual(result['home_dollars']*3,result['draw_dollars']*4)
        match.update(date='2026-10-10',league='E0')
        report=enrich(dict(predictions=[match]))
        self.assertIn('stake_dollars',export_csv(report).splitlines()[0])

    def test_historical_reconciliation(self):
        report=json.loads((ROOT/'results/stake_level_results.json').read_text(encoding='utf-8'))
        totals=defaultdict(lambda:[0,0.,0.])
        handle=(ROOT/'results/stake_level_bet_ledger.csv').open(encoding='utf-8')
        self.addCleanup(handle.close)
        for row in csv.DictReader(handle):
            stake=float(row['stake_dollars']);gross=float(row['gross_return_dollars'])
            self.assertAlmostEqual(gross-stake,float(row['net_profit_dollars']))
            self.assertEqual(stake,int(row['level']))
            for season in [row['season'],'All seasons']:
                t=totals[(row['split'],season,row['strategy'],int(row['level']))]
                t[0]+=1;t[1]+=stake;t[2]+=gross
        for s in report['rows']:
            t=totals[(s['split'],s['season'],s['strategy'],s['level'])]
            self.assertEqual(t[0],s['matches']);self.assertAlmostEqual(t[1],s['stake_dollars'])
            self.assertAlmostEqual(t[2],s['gross_return_dollars'])
            self.assertAlmostEqual(t[2]-t[1],s['net_profit_dollars'])
        counts={split:sum(s['matches'] for s in report['rows'] if s['split']==split and s['season']=='All seasons' and s['strategy']=='Most likely outcome') for split in ['Train (OOB)','Test']}
        self.assertEqual(counts,{'Train (OOB)':11561,'Test':1447})


if __name__=='__main__':unittest.main()
