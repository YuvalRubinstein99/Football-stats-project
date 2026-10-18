import sys,unittest,copy
from pathlib import Path
from datetime import datetime,timezone,timedelta
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import model_strategy_app
from winner_source import parse,quote

class WinnerDownload(unittest.TestCase):
    def setUp(self):
        self.now=datetime.now(timezone.utc)
        self.date=(self.now+timedelta(days=2)).strftime('%y%m%d')
        self.market=dict(mId=123,eId=456,sId=240,league='פרמייר ליג',mp='\u202e1X2\u202c - תוצאת סיום (ללא הארכות)',desc='ליברפול - אברטון',isLive=False,score='',periodLive='',e_date=int(self.date),m_hour='2000',outcomes=[dict(desc='ליברפול',price='1.8',spread=''),dict(desc='\u202bX\u202c',price='3.2',spread=''),dict(desc='אברטון',price='4.1',spread='')])
    def test_market_and_matching(self):
        feed=parse(dict(markets=[self.market]),self.now);m=feed['markets'][0]
        odds,meta=quote(feed,m);self.assertEqual(odds,dict(home=1.8,draw=3.2,away=4.1));self.assertEqual(meta['currency'],'ILS')
        for changes in [dict(home='Everton',away='Liverpool'),dict(league='D1'),dict(date='2000-01-01')]:
            with self.assertRaises(ValueError):quote(feed,{**m,**changes})
        with self.assertRaises(ValueError):quote({**feed,'markets':[m,m]},m)
        with self.assertRaises(ValueError):quote({**feed,'fetched_at':(self.now-timedelta(minutes=6)).isoformat()},m)
    def test_reject_incorrect_markets(self):
        for change in [dict(isLive=True),dict(mp='1X2 - מחצית ראשונה'),dict(sId=227)]:
            with self.assertRaises(ValueError):parse(dict(markets=[{**self.market,**change}]),self.now)
        bad=copy.deepcopy(self.market);bad['outcomes'][1]['desc']='Liverpool'
        feed=parse(dict(markets=[self.market,bad]),self.now)
        self.assertEqual(len(feed['markets']),1);self.assertEqual(len(feed['rejected']),1)
    def test_model_uses_downloaded_execution_prices(self):
        feed=parse(dict(markets=[self.market]),self.now);m=feed['markets'][0]
        report=dict(generated_at=self.now.isoformat(),predictions=[{**m,'B365H':2.2,'B365D':3.3,'B365A':4.4}],feature_audit=[dict(fixture='|'.join(m[k] for k in ['league','date','home','away']),features={'B365H':2.2})])
        with patch('model_strategy_app.source',return_value=report),patch('winner_source.download',return_value=feed),patch('model_strategy_app.infer',return_value=([.6,.25,.15],None)) as infer,patch('model_filters.historical',side_effect=AssertionError('No Winner historical odds')):
            result=model_strategy_app.view('poisson',True,'Expected return > 1',dict(source='winner_auto',min_odds=1.01,min_edge=0))
        bet=result['matches'][0]['bet']
        self.assertEqual(bet['decimal_odds'],1.8)
        self.assertAlmostEqual(bet['expected_net_per_unit'],.08)
        self.assertEqual(infer.call_args.args[1]['B365H'],2.2)

if __name__=='__main__':unittest.main()
