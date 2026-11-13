"""XGBoost vector leaves against the original goal-vector RF and its 14 betting rules."""
import hashlib
import json
from itertools import product

from select_expanded_features import ROOT, np, pd, joblib, threadpool_limits
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss
from lineup_model import clean
from compare_expanded_models import outcome_probabilities
from live_bets import recommendations, STRATEGIES
from stake_levels import size_bet
from plot_binary_home_filters import Drawing, Rect, Line, PolyLine, HexColor, text, renderPDF, pdfium

RESULTS = ROOT/'results'
OUT = RESULTS/'xgboost_goal_vector'
TARGETS = ['home_score','away_score']
TEAMS = ['home_team_name','away_team_name']
ODDS = ['B365H','B365D','B365A']
SETTINGS = dict(objective='reg:squarederror', tree_method='hist', multi_strategy='multi_output_tree',
                learning_rate=.04, min_child_weight=30, reg_lambda=10, subsample=.8,
                colsample_bytree=.85, random_state=20261007, n_jobs=4)
COLORS = {'RF_vector':'#087f8c', 'XGBoost':'#8052aa'}


def load_sources():
    details = json.loads((RESULTS/'run_details.json').read_text())
    frames = []
    for name in ['train','test']:
        path = ROOT/'data'/f'{name}.csv'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == details['input_sha256'][path.name]
        frame = clean(pd.read_csv(path)).replace([np.inf,-np.inf],np.nan).dropna()
        assert len(frame) == details[name+'_rows']
        frames.append(frame)
    source = pd.read_csv(RESULTS/'stake_level_bet_ledger.csv',usecols=['split','season','source_row','strategy','home','away','week'])
    for frame, split in zip(frames,['Train (OOB)','Test']):
        labels = source[(source.split==split)&(source.strategy==STRATEGIES[0])].set_index('source_row')
        assert labels.index.is_unique and len(labels)==len(frame)
        labels = labels.loc[frame.index]
        np.testing.assert_array_equal(labels.home,frame.home_team_name)
        np.testing.assert_array_equal(labels.away,frame.away_team_name)
        np.testing.assert_array_equal(labels.week,frame.Matchweek)
        frame['season'] = labels.season.str[:4].astype(int)
    assert frames[0].season.max() < frames[1].season.min()
    return frames, details


def matrices(training, evaluation):
    numeric = [c for c in training if c not in [*TARGETS,*TEAMS,'season']]
    encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore').fit(training[TEAMS])
    def transform(frame):
        return np.column_stack([encoder.transform(frame[TEAMS]),frame[numeric].to_numpy(float)])
    return transform(training), transform(evaluation), encoder, numeric


def outcome(frame):
    return (1-np.sign(frame.home_score-frame.away_score)).to_numpy(int)


def train_vector(train,test,details):
    rows = []
    for year in range(2018,2022):
        past, valid = train[train.season<year],train[train.season==year]
        x, xv, _, _ = matrices(past,valid)
        for depth, rounds in product([1,2,3],[80,160]):
            estimator = xgb.XGBRegressor(**SETTINGS,max_depth=depth,n_estimators=rounds)
            estimator.fit(x,past[TARGETS].to_numpy())
            p = outcome_probabilities(estimator.predict(xv).astype(float))
            rows.append(dict(season=year,depth=depth,rounds=rounds,
                             log_loss=log_loss(outcome(valid),p,labels=[0,1,2])))
    validation = pd.DataFrame(rows)
    validation.to_csv(OUT/'validation.csv',index=False)
    depth,rounds = map(int,validation.groupby(['depth','rounds']).log_loss.mean().idxmin())
    estimator = xgb.XGBRegressor(**SETTINGS,max_depth=depth,n_estimators=rounds)
    x,xt,encoder,numeric = matrices(train,test)
    assert list(encoder.get_feature_names_out(TEAMS))+numeric == details['features']
    estimator.fit(x,train[TARGETS].to_numpy())
    estimator.save_model(OUT/'goal_vector.ubj')
    joblib.dump((encoder,numeric),OUT/'preprocessing.joblib')
    config = dict(**SETTINGS,max_depth=depth,n_estimators=rounds,xgboost_version=xgb.__version__,
                  selection='Mean three-way outcome log loss on 2018–21 expanding season folds',
                  targets=TARGETS,probability_order=['Home','Draw','Away'],data_hashes=details['input_sha256'])
    (OUT/'selection.json').write_text(json.dumps(config,indent=2))
    forest = joblib.load(RESULTS/'vector_forest.joblib')
    forest.verbose=0
    print('Selected depth',depth,'rounds',rounds,flush=True)
    return {name: [est.predict(arr).astype(float) for arr in [x,xt]] for name,est in [('RF_vector',forest),('XGBoost',estimator)]}


def backtest(frame, p, model, split):
    prices=frame[ODDS].to_numpy()
    actual=outcome(frame)
    # Rules consume named H/D/A probabilities and split one unit across any covered outcomes.
    n=len(frame)
    net=np.zeros((n,len(STRATEGIES),2))
    stake=np.zeros_like(net)
    for i,(probs,odds,y) in enumerate(zip(p,prices,actual)):
        fixture=dict(home=frame.home_team_name.iat[i],away=frame.away_team_name.iat[i],
                     **dict(zip(['p_home','p_draw','p_away'],probs)),**dict(zip(ODDS,odds)))
        for j,bet in enumerate(recommendations(fixture)):
            assert bet['strategy']==STRATEGIES[j]
            if bet['action']!='Bet':continue
            fractions=np.array([bet['home_stake'],bet['draw_stake'],bet['away_stake']])
            assert abs(fractions.sum()-1)<1e-9
            unit_profit=fractions[y]*odds[y]-1
            stake[i,j]=[1,size_bet(bet)['stake_dollars']]
            net[i,j]=stake[i,j]*unit_profit
    curves,stats=[],[]
    for j,rule in enumerate(STRATEGIES):
        for k,staking in enumerate(['flat','tiers']):
            units=stake[:,j,k].sum()
            meta=dict(model=model,split=split,strategy=rule,staking=staking)
            profit=net[:,j,k].sum()
            stats.append(dict(**meta,bets=int((stake[:,j,k]>0).sum()),staked=units,net=profit,
                              roi=profit/units if units else np.nan))
            weekly=pd.DataFrame(dict(season=frame.season,week=frame.Matchweek,net=net[:,j,k])).groupby(['season','week']).net.sum().reset_index()
            weekly['cumulative_net']=weekly.net.cumsum()
            assert np.isclose(weekly.cumulative_net.iat[-1],profit)
            curves.append(weekly.assign(**meta))
    return stats,curves


def plot(curves,stats,split,staking):
    drawing=Drawing(1440,1290)
    drawing.add(Rect(0,0,1440,1290,fillColor=HexColor('#ffffff'),strokeColor=None))
    mode='2022–23 test' if split=='test' else '2014–22 in-sample fit'
    size='1 unit per selection' if staking=='flat' else '1–5-unit stakes, positive model edge required'
    text(drawing,40,1248,f'Goal vectors  |  {mode}  |  {size}',23,bold=True)
    text(drawing,40,1214,'Original forest',14,color=COLORS['RF_vector'],bold=True)
    text(drawing,210,1214,'XGBoost',14,color=COLORS['XGBoost'],bold=True)
    text(drawing,365,1214,'Cumulative net profit in units, grouped by season and scheduled matchweek. Each panel has its own scale.',12)
    for idx,rule in enumerate(STRATEGIES):
        row,col=divmod(idx,3)
        left=60+col*473
        right=left+386
        top=1137-row*226
        bottom=top-141
        data=curves[(curves.split==split)&(curves.staking==staking)&(curves.strategy==rule)]
        lo,hi=min(0.,data.cumulative_net.min()),max(0.,data.cumulative_net.max())
        if hi==lo:hi+=1
        text(drawing,left-22,top+39,rule,13,bold=True)
        y=lambda v:bottom+(v-lo)/(hi-lo)*(top-bottom)
        for tick in sorted({lo,0.,hi}):
            drawing.add(Line(left,y(tick),right,y(tick),strokeWidth=.6,strokeColor=HexColor('#b4bdc7')))
            text(drawing,left-5,y(tick)-3,f'{tick:.0f}',9,textAnchor='end')
        for j,(name,color) in enumerate(COLORS.items()):
            series=data[data.model==name].sort_values(['season','week']).cumulative_net.to_numpy()
            points=[left,y(0)]
            for i,v in enumerate(series,1):
                points.extend([left+(right-left)*i/len(series),y(v)])
            drawing.add(PolyLine(points,strokeWidth=1.6,strokeColor=HexColor(color),fillColor=None))
            entry=stats.loc[(name,split,rule,staking)]
            text(drawing,left-20+j*202,top+17,f'{entry.net:+,.2f}  |  {int(entry.bets):,} bets',11,color=color)
        text(drawing,left,bottom-16,'Week 1' if split=='test' else '2014–15',10)
        text(drawing,right,bottom-16,'Week 38' if split=='test' else '2021–22',10,textAnchor='end')
    with pdfium.PdfDocument(renderPDF.drawToString(drawing)) as doc:
        page=doc[0]
        bitmap=page.render(scale=1.35)
        bitmap.to_pil().save(OUT/f'{split}_{staking}.png')
        bitmap.close()
        page.close()


def main():
    OUT.mkdir(exist_ok=True)
    (train,test),details=load_sources()
    means=train_vector(train,test,details)
    reference=pd.read_csv(RESULTS/'match_predictions.csv').set_index('source_test_row').loc[test.index]
    np.testing.assert_allclose(means['RF_vector'][1],reference[['vector_expected_home_goals','vector_expected_away_goals']],atol=1e-9)
    stats,curves,metrics,predictions=[],[],[],[]
    for j,(split,frame) in enumerate([('train',train),('test',test)]):
        goals=frame[TARGETS].to_numpy()
        for name in ['RF_vector','XGBoost','Market']:
            if name=='Market':
                probs=1/frame[ODDS].to_numpy()
                p=probs/probs.sum(axis=1,keepdims=True)
                errors={}
            else:
                values=means[name][j]
                assert values.shape==goals.shape and np.isfinite(values).all()
                p=outcome_probabilities(values)
                errors=dict(goal_mae=np.mean(np.abs(values-goals)),goal_rmse=np.sqrt(np.mean((values-goals)**2)))
                if split=='test' and name=='RF_vector':
                    np.testing.assert_allclose(p,reference[['RF_vector_Home','RF_vector_Draw','RF_vector_Away']],atol=1e-9)
                record=frame[['season','Matchweek','home_team_name','away_team_name',*TARGETS,*ODDS]].copy()
                record=record.assign(source_row=frame.index,model=name,split=split,mean_home=values[:,0],mean_away=values[:,1],
                                     p_home=p[:,0],p_draw=p[:,1],p_away=p[:,2])
                predictions.append(record)
                sums,paths=backtest(frame,p,name,split)
                stats.extend(sums)
                curves.extend(paths)
            labels=outcome(frame)
            metrics.append(dict(model=name,split=split,matches=len(frame),correct=int((p.argmax(axis=1)==labels).sum()),
                                accuracy=np.mean(p.argmax(axis=1)==labels),log_loss=log_loss(labels,p,labels=[0,1,2]),
                                predicted_draws=int((p.argmax(axis=1)==1).sum()),**errors))
    table=pd.DataFrame(stats)
    original=pd.concat([pd.read_csv(RESULTS/f'{file}.csv') for file in [
        'strategy_results','home_win_strategy','home_loss_strategy','home_win_sqrt_profit_strategy','home_loss_sqrt_profit_strategy']])
    saved=original[original.model=='RF_vector'].set_index('strategy')
    actual=table.query("split=='test' and model=='RF_vector' and staking=='flat'").set_index('strategy')
    assert set(saved.index)==set(STRATEGIES)
    np.testing.assert_allclose(actual.loc[saved.index,'net'],saved.net_profit_units,atol=1e-8)
    np.testing.assert_array_equal(actual.loc[saved.index,'bets'],saved.bets)
    table.to_csv(OUT/'strategies.csv',index=False)
    quality=pd.DataFrame(metrics)
    quality.to_csv(OUT/'metrics.csv',index=False)
    pd.concat(predictions).to_csv(OUT/'predictions.csv.gz',index=False)
    weekly=pd.concat(curves,ignore_index=True)
    weekly.to_csv(OUT/'weekly_curves.csv.gz',index=False)
    stats_index=table.set_index(['model','split','strategy','staking'])
    for split,staking in product(['train','test'],['flat','tiers']):plot(weekly,stats_index,split,staking)
    print(quality.to_string(index=False))
    print(table.query("split=='test' and staking=='flat'").pivot(index='strategy',columns='model',values=['bets','net','roi']).round(4).to_string())


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
