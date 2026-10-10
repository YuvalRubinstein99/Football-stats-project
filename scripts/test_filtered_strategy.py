"""Verify the deployed filter matches the frozen search candidate and its returns."""
import sys,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from live_bets import integrated_recommendations,SEARCH_CANDIDATE,enrich,export_csv
from stake_levels import size_bet


class FilteredTests(unittest.TestCase):
    def test_filter_and_model_gate(self):
        match=dict(home='H',away='A',p_away=.3,p_draw=.3,p_home=.4,B365A=5,B365D=4,B365H=2)
        original=integrated_recommendations(match)
        b=original[-1]
        self.assertEqual(b['strategy'],SEARCH_CANDIDATE)
        self.assertEqual(b['action'],'Bet')
        self.assertEqual(b['selection'],original[8]['selection'])
        self.assertGreater(size_bet(b)['stake_dollars'],0)
        unavailable=integrated_recommendations(match,'poisson')[-1]
        self.assertEqual(unavailable['action'],'Unavailable')
        self.assertEqual(size_bet(unavailable)['stake_dollars'],0)
        match.update(B365A=2,B365D=2,B365H=2)
        self.assertEqual(integrated_recommendations(match)[-1]['action'],'Skip')

    def test_report_and_csv(self):
        report=enrich(dict(model_kind='lineup_rf',predictions=[dict(date='2026-10-10',league='E0',home='H',away='A',p_away=.3,p_draw=.3,p_home=.4,B365A=5,B365D=4,B365H=2)]))
        self.assertEqual(len(report['strategy_names']),15)
        self.assertIn(SEARCH_CANDIDATE,export_csv(report))

    def test_search_accounting_parity(self):
        search=json.loads((ROOT/'results/strategy_search.json').read_text(encoding='utf-8'))['chosen']['profit']
        self.assertEqual(search['config'],dict(probabilities='raw',strategy='Legacy variance score',min_edge=.02,min_odds=4,max_odds=1000,outcome='all',sizing=.01))
        rows=json.loads((ROOT/'results/stake_level_results.json').read_text(encoding='utf-8'))['rows']
        for period,years in [('test',{2022})]:
            selected=[s for s in rows if s['strategy']==SEARCH_CANDIDATE and s['season']!='All seasons' and int(s['season'][:4]) in years]
            self.assertEqual(sum(s['bets'] for s in selected),search[period]['bets'])
            self.assertAlmostEqual(sum(s['stake_dollars'] for s in selected),search[period]['staked'])
            self.assertAlmostEqual(sum(s['net_profit_dollars'] for s in selected),search[period]['net'])


if __name__=='__main__':unittest.main()
