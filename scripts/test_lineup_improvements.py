"""Identity, chronology, player-MAP gradient and preprocessing invariants."""
import unittest
from select_expanded_features import np, pd, threadpool_limits
from scipy.sparse import csr_matrix
from scipy.optimize import check_grad
from player_identity import IdentityIndex, stable_id, canonical_club
from lineup_history import (STRENGTHS, lineup_changes, poisson_objective,
                            fit_player_effects, player_features, lineup_key)
from compare_lineup_improvements import candidates, fit, predict


def person(pid, full, short, team):
    return dict(sofifa_id=pid,long_name=full,short_name=short,club_name=team,overall=75)


def historical_frame():
    rows,lineups = [],{}
    groups = [[dict(player_id=str(10*t+i),overall=75.) for i in range(8)] for t in range(3)]
    for i in range(120):
        home,away = [(0,1),(1,0),(0,2),(2,0),(1,2),(2,1)][i%6]
        rows.append(dict(data_split='train',source_row=i,fixture_date=str(pd.Timestamp('2015-01-01')+pd.Timedelta(days=i))[:10],
            league='L',home_score=[3,1,0][home],away_score=[3,1,0][away]))
        lineups[f'train:{i}'] = dict(home=groups[home],away=groups[away])
    return pd.DataFrame(rows),lineups


class ImprovementsTests(unittest.TestCase):
    def test_club_aliases_and_ambiguity(self):
        players=[person(1,'Iker Muniain Goni','Muniain','Athletic Club de Bilbao'),
                 person(2,'John Smith','J. Smith','A'),person(3,'James Smith','J. Smith','B')]
        index=IdentityIndex(players)
        self.assertEqual(index.match('Iker Muniain','Athletic Club')[0],0)
        self.assertIsNone(index.match('J. Smith','Unknown')[0])
        self.assertEqual(index.match('J. Smith','B')[0],2)
        self.assertIsNone(index.match('Jon Smith','A')[0])
        self.assertIsNone(index.match('John Smithson','A')[0])
        self.assertEqual(canonical_club('Levante Unión Deportiva'),canonical_club('Levante'))

    def test_prior_names_and_stable_ids(self):
        players=[person(123,'New Full Name','New Name','A')]
        index=IdentityIndex(players,{'old name':{'123'}})
        self.assertEqual(index.match('Old Name','A')[0],0)
        self.assertEqual(stable_id({'Image Link':'https://cdn.sofifa.net/players/158/023/23_60.png'}),'158023')
        self.assertIsNone(stable_id({'Image Link':'invalid'}))
        with self.assertRaises(ValueError):
            IdentityIndex(players+players)

    def test_initials_and_full_surname_required(self):
        index=IdentityIndex([person(1,'Daniel Carvajal Ramos','Carvajal','Real Madrid CF')])
        self.assertEqual(index.match('Dani Carvajal','Real Madrid')[0],0)
        self.assertEqual(index.match('D Carvajal','Real Madrid')[0],0)
        self.assertIsNone(index.match('David Carvajal','Real Madrid')[0])

    def test_lineup_history_past_only_and_same_day(self):
        rows=[]
        for i,day in enumerate(['2019-01-01','2019-01-02','2019-01-03','2019-01-04','2019-01-04','2021-01-01']):
            r=dict(source_row=i,fixture_date=day,league='L',home_team_name='A',away_team_name=f'B{i}')
            r.update({f'{s}_{v}_mean':float(70+i) for s in ['home','away'] for v in STRENGTHS})
            rows.append(r)
        frame=pd.DataFrame(rows)
        result,_=lineup_changes(frame)
        self.assertTrue(np.isnan(result.loc[2,'home_attack_shooting_change']))
        self.assertEqual(result.loc[3,'home_attack_shooting_change'],2.)
        self.assertEqual(result.loc[4,'home_attack_shooting_change'],3.)
        self.assertEqual(result.loc[4,'home_attack_shooting_history_count'],3)
        self.assertTrue(np.isnan(result.loc[5,'home_attack_shooting_change']))
        altered=frame.copy()
        altered.loc[3,'home_attack_shooting_mean']=99.
        changed,_=lineup_changes(altered)
        self.assertEqual(changed.loc[4,'home_attack_shooting_change'],result.loc[4,'home_attack_shooting_change'])
        pd.testing.assert_frame_equal(lineup_changes(frame.iloc[:3])[0],result.iloc[:3])

    def test_poisson_map_gradient(self):
        x=csr_matrix([[1.,.5,-.5],[1.,-.5,.5],[1.,1.,0.]])
        offset=np.array([.1,.2,-.1]);y=np.array([2.,0.,1.]);w=np.array([1.,.7,.9]);penalty=np.array([.001,1.,1.])
        fun=lambda t:poisson_objective(t,x,offset,y,w,penalty)[0]
        jac=lambda t:poisson_objective(t,x,offset,y,w,penalty)[1]
        self.assertLess(check_grad(fun,jac,np.array([.2,.3,-.1])),1e-5)

    def test_player_update_shrinkage_and_unseen_player_prior(self):
        frame,lineups=historical_frame()
        weak=fit_player_effects(frame,lineups,'2015-06-01',alpha=1.)
        strong=fit_player_effects(frame,lineups,'2015-06-01',alpha=100.)
        n=2*len(weak['ids'])
        self.assertGreater(np.linalg.norm(weak['theta'][:n]),.01)
        self.assertLess(np.linalg.norm(strong['theta'][:n]),np.linalg.norm(weak['theta'][:n]))
        self.assertLess(weak['max_gradient'],.02)
        unknown={s:[dict(player_id=f'{s}{i}',overall=80.) for i in range(8)] for s in ['home','away']}
        f=player_features(weak,unknown)
        self.assertEqual(f['home_fifa_prior'],.2)
        self.assertEqual(f['home_learned_attack'],0.)
        self.assertEqual(f['away_player_history_mean'],0.)
        with self.assertRaises(ValueError):
            fit_player_effects(frame,lineups,frame.fixture_date.max())

    def test_independent_weights_and_no_outcome_features(self):
        sets=candidates(['home_team_name','away_team_name','B365H'])
        self.assertIn('home_attack_shooting_mean',sets['learned_weights'])
        self.assertIn('away_defense_defending_mean',sets['learned_weights'])
        self.assertNotIn('compact_attack_defense',sets['learned_weights'])
        for cols in sets.values():
            self.assertEqual(len(cols),len(set(cols)))
            self.assertFalse(set(cols)&{'home_score','away_score','fixture_date','source_row','season'})

    def test_fold_only_imputation_and_probabilities(self):
        frame=pd.DataFrame(dict(home_team_name=['A']*30,away_team_name=['B']*30,
            home_score=[0,1,2]*10,away_score=[2,1,0]*10,gap=[1.,np.nan,3.]*10))
        bundle=fit(frame,['home_team_name','away_team_name','gap'],'Poisson',5,.1)
        prep=bundle['fitted']['preprocessing']
        self.assertEqual(prep['imputer'].statistics_[0],2.)
        p,g=predict(bundle,frame.assign(gap=5.))
        self.assertTrue(np.allclose(p.sum(axis=1),1))
        self.assertTrue(np.isfinite(g).all() and (g>0).all())
        self.assertEqual(prep['imputer'].statistics_[0],2.)


if __name__=='__main__':
    with threadpool_limits(limits=4):
        unittest.main(verbosity=2)
