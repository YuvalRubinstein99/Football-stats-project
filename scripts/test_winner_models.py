import sys,unittest
from pathlib import Path
from datetime import datetime,timezone,timedelta
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from model_strategy_app import view,source
from odds_sources import quote_for,fixture_key

class WinnerModels(unittest.TestCase):
    def test_quote_validation(self):
        entry=dict(home=2,draw=3,away=4,confirmed=True,observed_at=datetime.now(timezone.utc).isoformat())
        odds,meta=quote_for('winner',entry,{})
        self.assertEqual(meta['currency'],'ILS')
        self.assertEqual(odds['draw'],3)
        for changes in [dict(confirmed=False),dict(observed_at=None),dict(home=float('nan')),dict(observed_at=(datetime.now(timezone.utc)-timedelta(minutes=16)).isoformat())]:
            with self.assertRaises(ValueError):quote_for('winner',{**entry,**changes},{})

    def test_model_inputs_and_history_separated(self):
        report=source();match=report['predictions'][0]
        quote=dict(home=2,draw=3,away=4,confirmed=True,observed_at=datetime.now(timezone.utc).isoformat())
        filters=dict(source='winner',quotes={fixture_key(match):quote},min_odds=1.01,min_edge=0)
        with patch('model_filters.historical',side_effect=AssertionError('Winner cannot reuse Bet365 history')),patch('model_strategy_app.infer',return_value=([.6,.25,.15],None)) as infer:
            result=view('poisson',True,'Expected return > 1',filters)
        item=result['matches'][0]
        self.assertEqual(item['bet']['decimal_odds'],2)
        self.assertAlmostEqual(item['bet']['expected_net_per_unit'],.2)
        self.assertEqual(item['probabilities'],[.6,.25,.15])
        self.assertEqual(result['curves'],[])
        key='|'.join(str(match[k]) for k in ['league','date','home','away'])
        original=next(a['features'] for a in report['feature_audit'] if a['fixture']==key)
        self.assertEqual(infer.call_args_list[0].args[1]['B365H'],original['B365H'])
        for other in result['matches'][1:]:self.assertNotIn('bet',other)

if __name__=='__main__':unittest.main()
