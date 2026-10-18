import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from model_strategy_app import infer,resolve,view,FAMILIES,pd,np,SEARCH_CANDIDATE

class ModelApp(unittest.TestCase):
    def test_all_model_prediction_parity(self):
        test=pd.read_csv(ROOT/'data/expanded_2008/test.csv')
        expected=pd.read_csv(ROOT/'results/team_identity/match_predictions.csv')
        for family in ['rf','linear','poisson','logit','probit']:
            for teams in ([True] if family=='rf' else [True,False]):
                name=resolve(family,teams)
                for i in [0,173,1446]:
                    p,g=infer(name,test.iloc[i].to_dict())
                    np.testing.assert_allclose(p,[expected.iloc[i][name+'__p_'+s] for s in ['home','draw','away']],atol=1e-8)
                    self.assertEqual(g is None,family in ['logit','probit'])
    def test_views_and_histories(self):
        for family in ['rf','linear','poisson','logit','probit']:
            result=view(family,True,SEARCH_CANDIDATE)
            self.assertTrue(result['matches'])
            self.assertTrue(all('bet' in r for r in result['matches']))
            self.assertIn('home_team_name',result['features'])
            self.assertIn('home_GD_form',result['features'])
            self.assertIn('Matchweek',result['features'])
            self.assertEqual(len(result['curves']),3 if family=='rf' else 2)
            for c in result['curves']:
                summary=next(s for s in result['summary'] if s['split']==c['split'])
                self.assertAlmostEqual(c['points'][-1]['net'],summary['net'])

if __name__=='__main__':unittest.main()
