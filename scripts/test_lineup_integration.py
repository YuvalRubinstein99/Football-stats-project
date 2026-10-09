"""Source parsing, identity matching, feature semantics, and saved-model parity."""
import csv
from datetime import date, datetime, timedelta, timezone
import math
from pathlib import Path
import statistics
import sys
import unittest
import tempfile
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if (ROOT / '.lineup_deps').exists():
    sys.path.insert(0, str(ROOT / '.lineup_deps'))
from lineup_sources import parse_lineups, match_player, team_key
from lineup_model import rating_features, form_features, LineupForest


class LineupTests(unittest.TestCase):
    def test_starters_exclude_injury_list_and_preserve_status(self):
        players = ''.join(f'<li class="lineup__player"><a title="Player {i}" href="/soccer/player/player-{i}">P. {i}</a></li>' for i in range(11))
        lists = ''.join(f'<ul class="lineup__list {side}"><li class="lineup__status">{status} Lineup</li>{players}'
                        '<li class="lineup__title">Injuries</li><li class="lineup__player"><a title="Injured" href="/soccer/player/x-99">Injured</a></li></ul>'
                        for side,status in [('is-home','Predicted'),('is-visit','Confirmed')])
        html = '<div class="lineup is-soccer"><div class="lineup__time"><b>January 2</b></div><div class="lineup__mteam is-home">Arsenal</div><div class="lineup__mteam is-visit">Leeds</div>'+lists+'</div>'
        m = parse_lineups(html, 'E0', date(2026,12,30))[0]
        self.assertEqual(m['date'], '2027-01-02')
        self.assertEqual(len(m['home_players']), 11)
        self.assertEqual(m['home_status'], 'predicted')
        self.assertEqual(m['away_status'], 'confirmed')
        self.assertNotIn('Injured', [p['name'] for p in m['home_players']])

    def test_no_fuzzy_identity_guessing(self):
        rows=[{'ea_id':'1','name':'David Raya Martín','common_name':'','overall':85}]
        p={'source_id':'100','name':'David Raya'}
        self.assertEqual(match_player(p,rows)['ea_id'],'1')
        with self.assertRaises(ValueError):
            match_player({**p,'name':'David Rayo'},rows)
        with self.assertRaises(ValueError):
            match_player(p,rows + [{**rows[0],'ea_id':'2'}])
        self.assertEqual(match_player(p,rows,{'100':'1'})['match_method'],'explicit_id')
        with self.assertRaises(ValueError):
            match_player(p,rows,{'100':'wrong'})

    def test_rating_statistics_match_legacy_definitions(self):
        ratings=list(range(70,81))
        players=[{'ea_id':str(i),'overall':r} for i,r in enumerate(ratings)]
        f=rating_features(players,'HomePlayer')
        self.assertAlmostEqual(f['HomePlayer_Overall_sd'],statistics.stdev(ratings))
        self.assertAlmostEqual(f['HomePlayer_Overall_mean_ln'],sum(map(math.log,ratings))/11)
        self.assertNotAlmostEqual(f['HomePlayer_Overall_mean_ln'],math.log(statistics.mean(ratings)))
        self.assertEqual(f['HomePlayer_3rd_best'],78)
        with self.assertRaises(ValueError):
            rating_features(players[:10],'HomePlayer')
        with self.assertRaises(ValueError):
            rating_features(players[:10]+players[:1],'HomePlayer')

    def test_form_is_pre_match_current_season_last_five(self):
        history=[{'league':'E0','date':date(2026,8,i+1),'home':'A','away':'B','hg':i,'ag':0} for i in range(7)]
        history += [{'league':'E0','date':d,'home':'A','away':'B','hg':99,'ag':0}
                    for d in [date(2025,8,1),date(2026,8,8),date(2026,8,9)]]
        f,n=form_features(history,{'league':'E0','home':'A','away':'B'},date(2026,8,8))
        self.assertEqual(n,{'home':7,'away':7})
        self.assertEqual(f['home_GD_prior'],21)
        self.assertEqual(f['home_GD_form'],20)
        self.assertEqual(f['away_GD_form'],-20)
        self.assertEqual(f['home_Points_form'],15)

    def test_club_aliases(self):
        self.assertEqual(team_key('Manchester United'),team_key('Man Utd'))
        self.assertEqual(team_key('Bayern Munich'),team_key('FC Bayern München'))
        rows=[{'ea_id':'1','name':'Đorđe Petrović','common_name':'','overall':80}]
        self.assertEqual(match_player({'source_id':'2','name':'Djordje Petrovic'},rows)['ea_id'],'1')

    def test_missing_rating_withholds_fixture_without_model_fallback(self):
        import lineup_pipeline
        now = datetime.now(timezone.utc)
        fixture = {'league':'E0','date':now.date()+timedelta(days=1),'time':'15:00',
                   'home':'A','away':'B','B365H':2.,'B365D':3.,'B365A':4.}
        history = [{**fixture,'date':now.date()-timedelta(days=4),'hg':1,'ag':0}]
        players=[{'source_id':str(i),'name':f'Player {i}'} for i in range(11)]
        lineup={**fixture,'date':fixture['date'].isoformat(),'home_players':players,'away_players':players,
                'home_status':'predicted','away_status':'predicted'}
        rating_source=Mock()
        rating_source.edition='test-edition'
        rating_source.sources=[]
        rating_source.squad.return_value=[{'ea_id':str(i),'name':f'Player {i}','common_name':'','overall':80} for i in range(10)]
        rating_source.search_player.side_effect=ValueError('Missing test player rating')
        forest=Mock()
        forest.metadata={'name':'test forest'}
        source={'url':'test','fetched_at':now.isoformat()}
        def downloaded(relative,*args):
            return ([fixture] if relative=='fixtures.csv' else history),source
        with tempfile.TemporaryDirectory() as directory, \
             patch('lineup_model.LineupForest',return_value=forest), \
             patch('lineup_sources.Ratings',return_value=rating_source), \
             patch('lineup_sources.fetch',return_value=('',source)), \
             patch('lineup_sources.parse_lineups',return_value=[lineup]), \
             patch.object(lineup_pipeline,'download',side_effect=downloaded):
            report=lineup_pipeline.run(['E0'],output=Path(directory),progress=lambda _:None)
        self.assertEqual(report['predictions'],[])
        self.assertEqual(len(report['skipped']),1)
        self.assertIn('Missing test player rating',report['skipped'][0]['reason'])
        forest.predict.assert_not_called()

    @unittest.skipUnless((ROOT/'results/vector_forest.joblib').exists(), 'Local saved forest not present')
    def test_saved_forest_reproduces_committed_predictions(self):
        forest=LineupForest()
        import pandas as pd
        test=pd.read_csv(ROOT/'data/test.csv')
        expected=pd.read_csv(ROOT/'results/match_predictions.csv')
        for _,saved in expected.head(8).iterrows():
            row=test.loc[int(saved['source_test_row'])].to_dict()
            home,away=forest.predict(row)
            self.assertAlmostEqual(home,saved['vector_expected_home_goals'],places=10)
            self.assertAlmostEqual(away,saved['vector_expected_away_goals'],places=10)


if __name__ == '__main__':
    unittest.main()
