import sys,unittest
from pathlib import Path
from datetime import date
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.lineup_deps'));sys.path.insert(0,str(ROOT/'src'))
from classifier_features import form_history,lineup_features,club,FORM,RICH
import numpy as np

class Classifiers(unittest.TestCase):
    def test_form_no_current_or_future_outcomes(self):
        game=dict(league='E0',date=date(2020,6,28),season=2019,home='Arsenal',away='Chelsea',hg=2,ag=0,B365H=2,B365D=4,B365A=4)
        query=dict(league='E0',date=date(2020,7,3),season=2019,home='Arsenal',away='Chelsea')
        same={**game,'date':query['date'],'hg':9}
        future={**game,'date':date(2020,7,4),'hg':20}
        key=lambda r:(r['league'],r['date'],club(r['home']),club(r['away']))
        a=form_history([game],[query]);b=form_history([game,same,future],[query])
        self.assertEqual(a[key(query)],b[key(query)])
        self.assertEqual(a[key(game)]['home_adjusted_form_count'],0)
        self.assertAlmostEqual(a[key(query)]['home_adjusted_points_last5'],3-(3*.5+.25))
        self.assertEqual(a[key(query)]['home_adjusted_form_count'],1) # COVID season crosses July.
        new={**query,'date':date(2020,9,3),'season':2020}
        self.assertEqual(form_history([game],[new])[key(new)]['home_adjusted_form_count'],0)
    def test_lineup_order_invariance(self):
        a=np.arange(70,81);b=np.arange(60,71)
        self.assertEqual(lineup_features(a,b),lineup_features(a[::-1],b[::-1]))
        self.assertEqual(lineup_features(a,b)['lineup_top3_advantage'],10)
        incomplete=a.astype(float);incomplete[0]=np.nan
        self.assertEqual(lineup_features(incomplete,b)['home_starter_rating_count'],10)
        with self.assertRaises(ValueError):lineup_features(a[:10],b)
    def test_live_feature_provenance(self):
        from model_strategy_app import source
        from classifier_model import live_features
        report=source();audits={a['fixture']:a['features'] for a in report['feature_audit']}
        for match in report['predictions']:
            key='|'.join(str(match[k]) for k in ['league','date','home','away'])
            enriched=live_features(audits[key],match,report,FORM+RICH)
            self.assertTrue(all(np.isfinite(enriched[c]) for c in FORM+RICH))
            self.assertEqual(enriched['home_starter_rating_count'],11)
        bad={**report,'sources':[]}
        with self.assertRaisesRegex(ValueError,'refresh'):live_features({},report['predictions'][0],bad,FORM)
    def test_integration(self):
        from model_strategy_app import pd,infer,view
        from train_classifier_models import VARIANTS
        from probability_calibration import calibrate
        from live_bets import SEARCH_CANDIDATE
        saved=pd.read_csv(ROOT/'results/classifier/match_predictions.csv')
        for name in VARIANTS:
            for i in [0,173,1446]:
                p,g=infer(name,saved.iloc[i].to_dict());self.assertIsNone(g)
                np.testing.assert_allclose(p,saved.iloc[i][[name+'__p_'+s for s in ['home','draw','away']]].to_numpy(float),atol=1e-8)
                changed=saved.iloc[i].to_dict();changed.update(home_score=99,away_score=88)
                np.testing.assert_allclose(infer(name,changed)[0],p,atol=1e-12)
                pc,_=infer(name+'__calibrated',saved.iloc[i].to_dict());np.testing.assert_allclose(pc,calibrate(name,p))
        result=view('clf_both',True,SEARCH_CANDIDATE,dict(league='D1'),calibrated=True)
        self.assertTrue(set(FORM+RICH)<=set(result['features']))
        self.assertEqual(len(result['curves']),2)
        for c,s in zip(result['curves'],result['summary']):self.assertAlmostEqual(c['points'][-1]['net'],s['net'])
        for m in result['matches']:
            if 'probabilities' in m:self.assertAlmostEqual(sum(m['probabilities']),1)
            else:self.assertIn('refresh',m['error'].lower())

if __name__=='__main__':unittest.main()
