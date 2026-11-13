"""Evaluate XGBoost with the same fixtures and positional features as the compact forest."""
import json
from itertools import product
from select_expanded_features import ROOT, np, pd, threadpool_limits
import xgboost as xgb
from regularized_home_win import POSITIONS, features, market_probability
from home_win_model import target
from train_regularized_home_win import binary_metrics
from train_binary_home_win import paired
from binary_betting import selections, settle, summary
from compare_matched_betting_strategies import chart

DATA = ROOT/'results/lineup_improvements'
BASE = DATA/'binary_retrained'
MATCHED = BASE/'matched_comparison'
LATER = BASE/'regularized/holdout_2025'
OUT = MATCHED/'xgboost'
COLORS = {'Forest': '#087f8c', 'XGBoost': '#8052aa'}
SETTINGS = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist',
                learning_rate=.04, min_child_weight=25, reg_lambda=10, subsample=.8,
                colsample_bytree=.85, random_state=20261020, n_jobs=4)


def model(depth, rounds):
    return xgb.XGBClassifier(**SETTINGS, max_depth=depth, n_estimators=rounds)


def select_settings(train):
    matrix = features(train, POSITIONS)
    labels = target(train)
    scores = []
    for depth, rounds in product([1, 2, 3], [80, 160]):
        for year in range(2018, 2022):
            earlier = train.season < year
            following = train.season == year
            assert train.loc[earlier,'fixture_date'].max() < train.loc[following,'fixture_date'].min()
            estimator = model(depth, rounds).fit(matrix[earlier], labels[earlier])
            p = estimator.predict_proba(matrix[following])[:,1].astype(float)
            scores.append(dict(depth=depth, rounds=rounds, season=year,
                               loss=binary_metrics(p, train[following])['log_loss']))
    validation = pd.DataFrame(scores)
    validation.to_csv(OUT/'validation.csv', index=False)
    losses = validation.groupby(['depth','rounds']).loss.mean()
    return tuple(map(int, losses.idxmin()))


def aligned_probabilities(frame, reference):
    key = ['season','source_row','fixture_date','league','home_team_name','away_team_name',
           'home_score','away_score','B365H','B365D','B365A']
    merged = frame[key].merge(reference[key+['p_home_win']], how='left', on=key,
                              validate='one_to_one', sort=False)
    assert merged.p_home_win.notna().all() and len(frame) == len(reference)
    return merged.p_home_win.to_numpy()


def evaluate_bets(frame, p, name, split, curves, results):
    odds = frame[['B365H','B365D','B365A']].to_numpy()
    outcomes = (1-np.sign(frame.home_score-frame.away_score)).to_numpy(int)
    for strategy, choice in selections(p, odds).items():
        for staking in ['flat','tiers']:
            bet = settle(p, odds, outcomes, choice, staking)
            net = bet['stakes']*(np.where(bet['pick']==outcomes, odds[np.arange(len(p)), bet['pick']], 0)-1)
            np.testing.assert_allclose(net, bet['profit'])
            meta = dict(model=name, split=split, staking=staking, strategy=strategy)
            results.append(dict(**meta, **summary(bet,outcomes), uses_away_proxy=choice['proxy']))
            daily = pd.Series(net,index=pd.to_datetime(frame.fixture_date)).groupby(level=0).sum()
            start = pd.Series([0.],index=[daily.index[0]-pd.Timedelta(days=1)])
            curve = pd.concat([start,daily]).cumsum().rename('profit').rename_axis('date').reset_index()
            curves.append(curve.assign(**meta))


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    train = pd.read_csv(DATA/'train_features.csv')
    depth, rounds = select_settings(train)
    config = dict(**SETTINGS, max_depth=depth,n_estimators=rounds,features=POSITIONS,
                  xgboost_version=xgb.__version__,selection='Minimum mean seasonal log loss, 2018–21 forward folds')
    (OUT/'selection.json').write_text(json.dumps(config,indent=2))
    print('Selected depth', depth, 'and rounds', rounds, flush=True)
    test = pd.read_csv(DATA/'test_features.csv')
    later = pd.read_csv(LATER/'matches.csv')
    past = pd.concat([train,test],ignore_index=True)
    matched = pd.read_csv(MATCHED/'predictions.csv')
    holdout = pd.read_csv(LATER/'predictions.csv')
    scores, forecasts, differences, curves, returns = [], [], [], [], []
    for training, samples, tag in [(train,[('train',train),('test',test)],'through_2021'),
                                   (past,[('test_2025',later)],'through_2022')]:
        estimator = model(depth,rounds).fit(features(training,POSITIONS),target(training))
        estimator.save_model(OUT/f'{tag}.ubj')
        for split, sample in samples:
            frame = sample.sort_values(['fixture_date','league','source_row']).reset_index(drop=True)
            if split!='train':assert training.fixture_date.max()<frame.fixture_date.min()
            ref = holdout if split=='test_2025' else matched[(matched.model=='new')&(matched.split==split)]
            forest_p = aligned_probabilities(frame,ref)
            p = estimator.predict_proba(features(frame.drop(columns=['home_score','away_score']),POSITIONS))[:,1].astype(float)
            probabilities = {'XGBoost':p,'Forest':forest_p,'Market':market_probability(frame)}
            if split!='test_2025':
                probabilities['Old forest']=aligned_probabilities(frame,matched[(matched.model=='old')&(matched.split==split)])
            for name, values in probabilities.items():
                scores.append(dict(split=split,model=name,**binary_metrics(values,frame)))
                if name in COLORS:evaluate_bets(frame,values,name,split,curves,returns)
            forecasts.append(frame[['season','source_row','fixture_date','home_team_name','away_team_name']].assign(
                split=split, xgboost=p, forest=forest_p, trained_through=training.fixture_date.max()))
            if split!='train':
                differences.append(dict(split=split,**paired(np.column_stack([1-p,p]),
                    np.column_stack([1-forest_p,forest_p]),frame)))
    quality = pd.DataFrame(scores)
    quality.to_csv(OUT/'metrics.csv',index=False)
    pd.concat(forecasts).to_csv(OUT/'predictions.csv',index=False)
    pd.DataFrame(differences).to_csv(OUT/'paired_log_loss.csv',index=False)
    table = pd.DataFrame(returns)
    table.to_csv(OUT/'strategies.csv',index=False)
    daily = pd.concat(curves,ignore_index=True)
    indexed = table.set_index(['split','staking','strategy','model'])
    for split in ['train','test','test_2025']:
        for staking in ['flat','tiers']:
            period = '2025–26 test, 286 eligible fixtures' if split=='test_2025' else None
            chart(split,staking,daily,indexed,colors=COLORS,out=OUT,period=period)
    print(quality[['split','model','matches','accuracy','log_loss','brier']].to_string(index=False))
    print(pd.DataFrame(differences).to_string(index=False))


if __name__=='__main__':
    with threadpool_limits(limits=4):run()
