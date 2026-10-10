"""Audit search selection and exported accounting without rerunning the model."""
import json
import unittest
from pathlib import Path


class SearchAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=json.loads((Path(__file__).resolve().parents[1]/'results/strategy_search.json').read_text(encoding='utf-8'))

    def test_candidates_and_probability_families(self):
        r=self.report
        self.assertEqual(r['grid_count'],2240)
        self.assertGreater(r['eligible_count'],0)
        self.assertEqual({d['model'] for d in r['diagnostics']},{'Raw RF','Calibrated RF','Market'})
        for d in r['diagnostics']:
            self.assertTrue(0<=d['auc']<=1)
            self.assertGreaterEqual(d['log_loss'],0)

    def test_winners_use_search_only(self):
        finalists=self.report['finalists']
        for objective,chosen in self.report['chosen'].items():
            def rank(c):
                s=c['search']
                if objective=='profit':return s['net']
                if objective=='drawdown':return -s['drawdown'] if s['net']>0 else -1e12+s['net']
                return s['balanced']
            expected=max(finalists,key=rank)
            self.assertEqual(chosen['config'],expected['config'])
            # Arbitrary changes to all holdout metrics cannot change selection.
            altered=[{**c,'validation':{'net':1e12},'test':{'net':-1e12}} for c in finalists]
            self.assertEqual(chosen['config'],max(altered,key=rank)['config'])

    def test_period_and_stake_accounting(self):
        for c in self.report['finalists']:
            for period,years in [('search',{2018,2019}),('validation',{2020,2021}),('test',{2022})]:
                s=c[period];annual=[a for a in c['seasons'] if int(a['season'][:4]) in years]
                self.assertAlmostEqual(s['net'],sum(a['net'] for a in annual))
                self.assertAlmostEqual(s['staked'],sum(a['staked'] for a in annual))
                self.assertEqual(s['bets'],sum(a['bets'] for a in annual))
                self.assertTrue(s['bets']<=s['staked']<=5*s['bets'])
                self.assertAlmostEqual(s['roi'],s['net']/s['staked'] if s['staked'] else 0)
                self.assertGreaterEqual(s['drawdown'],0)
            self.assertGreaterEqual(c['search']['bets'],100)


if __name__=='__main__':unittest.main()
