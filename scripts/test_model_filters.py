import sys,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.lineup_deps'));sys.path.insert(0,str(ROOT/'src'))
from model_filters import config,apply,historical
from live_bets import STRATEGIES,SEARCH_CANDIDATE

class ModelFilters(unittest.TestCase):
    def test_leagues_reconcile_and_filter(self):
        from model_filters import LEAGUES
        from model_strategy_app import view
        cfg=config(SEARCH_CANDIDATE)
        _,totals,curves,groups=historical('Linear__no_teams',cfg,by_league=True)
        for total in totals:
            subset=[r for r in groups if r['split']==total['split']]
            for key in ['matches','bets','staked','net','flat_net']:
                self.assertAlmostEqual(sum(r[key] for r in subset),total[key],places=7)
        for code in LEAGUES:
            _,selected=historical('Linear__no_teams',config(SEARCH_CANDIDATE,dict(league=code)))
            for row in selected:
                group=next(r for r in groups if r['league']==code and r['split']==row['split'])
                self.assertEqual(row['matches'],group['matches'])
                self.assertAlmostEqual(row['net'],group['net'])
        for curve in curves:
            group=next(r for r in groups if r['league']==curve['league'] and r['split']==curve['split'])
            self.assertAlmostEqual(curve['points'][-1]['net'],group['net'])
        result=view('linear',False,SEARCH_CANDIDATE,dict(league='E0'))
        self.assertTrue(all(r['league']=='E0' for r in result['matches']))
        self.assertTrue(all(r['league']=='E0' for r in result['league_summary']))
        with self.assertRaises(ValueError):config(SEARCH_CANDIDATE,dict(league='invalid'))

    def test_historical_parity_and_scaling(self):
        saved=json.loads((ROOT/'results/team_identity/train_test_curves.json').read_text())
        for strategy in STRATEGIES+[SEARCH_CANDIDATE]:
            curves,summary=historical('Linear__no_teams',config(strategy))
            for row in summary:
                old=next(r for r in saved['summary'] if r['model']=='Linear__no_teams' and r['strategy']==strategy and r['split']==row['split'])
                self.assertEqual(row['bets'],old['bets'])
                self.assertAlmostEqual(row['net'],old['net'],places=7)
                self.assertAlmostEqual(row['staked'],old['staked'])
        _,base=historical('Linear__no_teams',config(SEARCH_CANDIDATE))
        curves,doubled=historical('Linear__no_teams',config(SEARCH_CANDIDATE,dict(amounts=[2,4,6,8,10])))
        for a,b,c in zip(base,doubled,curves):
            self.assertAlmostEqual(b['net'],2*a['net'])
            self.assertAlmostEqual(c['points'][-1]['net'],b['net'])
        _,empty=historical('Linear__no_teams',config(SEARCH_CANDIDATE,dict(min_odds=999,max_odds=1000)))
        self.assertTrue(all(r['bets']==0 and r['net']==0 for r in empty))
    def test_live_outcome_gate(self):
        bet=dict(action='Bet',synthetic=False,decimal_odds=5.,expected_net_per_unit=.1,home_stake=0.,draw_stake=0.,away_stake=1.)
        blocked=apply(bet,config(SEARCH_CANDIDATE,dict(outcome='home')))
        self.assertEqual(blocked['stake_dollars'],0)
        self.assertIn('Outcome',blocked['staking_reason'])
        flat=apply(bet,config(SEARCH_CANDIDATE,dict(outcome='away',sizing='flat',amounts=[7,7,7,7,7])))
        self.assertEqual(flat['away_dollars'],7)
        with self.assertRaises(ValueError):config(SEARCH_CANDIDATE,dict(min_odds=10,max_odds=2))

if __name__=='__main__':unittest.main()
