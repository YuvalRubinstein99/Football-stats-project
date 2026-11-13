"""Direct binary home-win classifiers; output columns are [no home win, home win]."""
import warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.exceptions import ConvergenceWarning

FORBIDDEN = {'home_score','away_score','outcome','target','source_row','fixture_date','data_split','season'}


def target(frame):
    scores = frame[['home_score','away_score']].to_numpy(float)
    if not np.isfinite(scores).all() or (scores<0).any() or not np.equal(scores,np.floor(scores)).all():
        raise ValueError('Finite non-negative integer final scores required')
    return (scores[:,0]>scores[:,1]).astype(int)


def transform(bundle, frame):
    numeric = bundle['imputer'].transform(frame[bundle['numeric']])
    if bundle['scaler'] is not None:
        numeric = bundle['scaler'].transform(numeric)
    categorical = (bundle['encoder'].transform(frame[bundle['categorical']])
                   if bundle['categorical'] else np.empty((len(frame),0)))
    return np.column_stack([categorical,numeric])


def fit(frame, columns, family, trees=500, regularization_c=1.):
    if set(columns)&FORBIDDEN or len(columns)!=len(set(columns)):
        raise ValueError('Forbidden or duplicate predictor columns')
    y = target(frame)
    if not np.array_equal(np.unique(y),[0,1]):
        raise ValueError('Training data must contain home wins and non-wins')
    cats = [c for c in columns if c.endswith('_team_name')]
    nums = [c for c in columns if c not in cats]
    encoder = OneHotEncoder(handle_unknown='ignore',sparse_output=False) if cats else None
    categorical = encoder.fit_transform(frame[cats]) if cats else np.empty((len(frame),0))
    imputer = SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)
    numeric = imputer.fit_transform(frame[nums])
    scaler = StandardScaler() if family=='Logistic' else None
    if scaler is not None:
        numeric = scaler.fit_transform(numeric)
    if family=='Logistic':
        model = LogisticRegression(C=regularization_c,max_iter=4000,tol=1e-7)
    elif family=='RF_binary':
        model = RandomForestClassifier(n_estimators=trees,criterion='log_loss',max_depth=5,
            max_features=.3,max_samples=.5,random_state=20261020,n_jobs=4)
    else:
        raise ValueError(f'Unknown binary family: {family}')
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(np.column_stack([categorical,numeric]),y)
    if not np.array_equal(model.classes_,[0,1]):
        raise ValueError('Unexpected classifier class order')
    feature_names = (list(encoder.get_feature_names_out(cats)) if cats else [])
    feature_names += list(imputer.get_feature_names_out(nums))
    return dict(family=family,columns=columns,categorical=cats,numeric=nums,encoder=encoder,
        imputer=imputer,scaler=scaler,model=model,regularization_c=regularization_c,
        feature_names=feature_names,target='1 = home_score > away_score; 0 = draw or away win',
        probability_columns=['p_no_home_win','p_home_win'],decision_threshold=.5)


def predict(bundle, frame):
    if not np.array_equal(bundle['model'].classes_,[0,1]):
        raise ValueError('Invalid saved class order')
    p = bundle['model'].predict_proba(transform(bundle,frame))
    if p.shape!=(len(frame),2) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(axis=1),1):
        raise ValueError('Invalid binary probabilities')
    return p
