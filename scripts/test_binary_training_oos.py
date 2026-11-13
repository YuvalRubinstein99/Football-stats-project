"""Check temporal feature selection and independently reconcile saved returns."""
import json
import unittest
from backtest_binary_training_oos import OUT,np,pd,choose_candidate


class ForwardTests(unittest.TestCase):
    def test_future_validation_cannot_change_choice(self):
        records=[dict(season=year,candidate=name,log_loss=loss)
                 for year in [2015,2016] for name,loss in [('a',.6),('b',.7)]]
        self.assertEqual(choose_candidate(records,2017,['a','b']),('a',[2015,2016]))
        records.extend([dict(season=2017,candidate='a',log_loss=10),
                        dict(season=2017,candidate='b',log_loss=0)])
        self.assertEqual(choose_candidate(records,2017,['a','b'])[0],'a')

    def test_incomplete_history_rejected(self):
        with self.assertRaises(ValueError):
            choose_candidate([dict(season=2016,candidate='a',log_loss=.6)],2017,['a'])

    def test_saved_prediction_chronology_and_selection(self):
        p=pd.read_csv(OUT/'predictions.csv')
        self.assertFalse(p.duplicated(['protocol','source_row']).any())
        self.assertTrue((pd.to_datetime(p.trained_through)<pd.to_datetime(p.fixture_date)).all())
        validation=pd.read_csv(OUT/'validation.csv')
        plan=json.loads((OUT/'plan.json').read_text())
        folds=json.loads((OUT/'folds.json').read_text())
        for fold in folds:
            self.assertLess(pd.Timestamp(fold['trained_through']),pd.Timestamp(fold['forecast_start']))
            if fold['season']>=2017:
                chosen,years=choose_candidate(validation.to_dict('records'),fold['season'],list(plan['candidates']))
                self.assertEqual(chosen,fold['nested_candidate'])
                self.assertEqual(years,fold['selection_seasons'])
                actual=p[(p.protocol=='nested')&(p.season==fold['season'])]
                self.assertEqual(set(actual.candidate),{chosen})
                self.assertEqual(len(actual),fold['forecast_matches'])

    def test_independent_bet_settlement(self):
        ledger=pd.read_csv(OUT/'ledger.csv.gz')
        edge=ledger.p_home_win*ledger.B365H-1
        cutoff=np.where(ledger.strategy.str.endswith('>2%'),.02,0.)
        active=edge>cutoff
        tier=np.clip(np.ceil(np.round(edge/(ledger.B365H-1)*100,10)),1,5)
        stakes=np.where(active,np.where(ledger.staking.eq('flat'),1,tier),0)
        np.testing.assert_array_equal(stakes,ledger.stake)
        wins=ledger.home_score>ledger.away_score
        net=stakes*np.where(wins,ledger.B365H-1,-1)
        np.testing.assert_allclose(net,ledger.net,atol=1e-12)
        totals=pd.read_csv(OUT/'summary.csv')
        for row in totals.itertuples():
            mask=(ledger.protocol==row.protocol)&(ledger.strategy==row.strategy)&(ledger.staking==row.staking)
            self.assertAlmostEqual(float(net[mask].sum()),row.net)
            self.assertAlmostEqual(float(stakes[mask].sum()),row.staked)
            self.assertEqual(int((stakes[mask]>0).sum()),row.bets)
            self.assertAlmostEqual(row.net/row.staked,row.roi)


if __name__=='__main__':
    unittest.main()
