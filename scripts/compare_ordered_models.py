"""Matched expanded-data ordered logit/probit experiment (unpenalized MLE).

P(Y <= k | x) = F(cutpoint[k] - x @ beta), Y=away/draw/home.
Train-only centering/SVD removes constants and exact collinearity without
changing the identifiable predictor span. No tuning uses test outcomes.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import sys, json, hashlib, html
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.lineup_deps'))
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize, check_grad
from scipy.stats import norm, logistic
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import log_loss, roc_auc_score
from threadpoolctl import threadpool_limits
from live_bets import integrated_recommendations, SEARCH_CANDIDATE
from stake_levels import size_bet

OUT = ROOT / 'results/ordered_comparison'
BASE = ROOT / 'results/expanded_comparison'


def log_difference(big, small):
    return big + np.log(-np.expm1(np.minimum(small-big, -1e-16)))


def distribution(theta, x, link):
    dist = logistic if link == 'logit' else norm
    eta = x @ theta[:-2]
    gap = np.exp(theta[-1])
    a = theta[-2] - eta
    b = a + gap
    # Switch to survival probabilities in the positive tail to avoid cancellation.
    mid = np.where(a > 0, log_difference(dist.logsf(a), dist.logsf(b)),
                   log_difference(dist.logcdf(b), dist.logcdf(a)))
    logs = np.column_stack([dist.logcdf(a), mid, dist.logsf(b)])
    return logs, a, b, gap, dist


def objective(theta, x, y, link, sample_weight=None):
    logs, a, b, gap, dist = distribution(theta, x, link)
    chosen = logs[np.arange(len(y)), y]
    fa = np.exp(dist.logpdf(a)-chosen)
    fb = np.exp(dist.logpdf(b)-chosen)
    eta_grad = np.where(y == 0, fa, np.where(y == 1, fb-fa, -fb))
    b_grad = np.where(y == 1, -fb, np.where(y == 2, fb, 0.))
    weights=np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight)
    weights=weights/weights.sum()
    grad = np.r_[x.T @ (eta_grad*weights), -(eta_grad*weights).sum(), gap*(b_grad*weights).sum()]
    return -(chosen*weights).sum(), grad


def probability(theta, x, link):
    p = np.exp(distribution(theta, x, link)[0])[:, ::-1]  # H,D,A
    assert np.isfinite(p).all() and (p >= 0).all()
    assert np.allclose(p.sum(axis=1), 1., atol=1e-10)
    return p


def metrics(name, p, y):
    return dict(model=name, log_loss=log_loss(y, p, labels=[0,1,2]),
                auc_macro=roc_auc_score(y, p, multi_class='ovr', average='macro'),
                brier=float(np.mean(np.sum((p-np.eye(3)[y])**2, axis=1))),
                accuracy=float(np.mean(p.argmax(axis=1) == y)))


def bets(name, p, test, y):
    totals = {}
    ledger = []
    for i, r in test.iterrows():
        odds = np.array([r.B365H,r.B365D,r.B365A])
        match = dict(home=r.home_team_name, away=r.away_team_name,
                     p_home=p[i,0], p_draw=p[i,1], p_away=p[i,2],
                     B365H=odds[0], B365D=odds[1], B365A=odds[2])
        for bet in integrated_recommendations(match):
            stake = size_bet(bet)['stake_dollars']
            ret = bet[('home','draw','away')[y[i]]+'_stake'] * odds[y[i]]
            a = totals.setdefault(bet['strategy'], dict(bets=0,staked=0.,net=0.,flat_net=0.))
            net = stake*(ret-1)
            if stake:
                a['bets'] += 1
                a['staked'] += stake
                a['net'] += net
                a['flat_net'] += ret-1
            ledger.append(dict(model=name, row=i, week=int(r.Matchweek),
                               strategy=bet['strategy'], stake=stake, net=net))
    rows = [dict(model=name,strategy=k,**a,roi=a['net']/a['staked'] if a['staked'] else 0.)
            for k,a in totals.items()]
    return rows, ledger


def render(m, s, ci, ledger, status):
    def table(df):
        return df.to_html(index=False, float_format=lambda v:f'{v:,.5f}')
    body = '<h1>Expanded data: ordered logit and ordered probit</h1>'
    body += '<p>19,263 training matches (2008–22); identical 1,447 test matches (2022–23). Same FIFA, team, form, matchweek and Bet365 predictors.</p>'
    body += '<h2>Test prediction quality</h2><p>Lower log loss/Brier is better; higher AUC/accuracy is better.</p>'+table(m)
    body += '<h2>Fixed filtered Legacy variance strategy</h2><p>Odds ≥ 4, expected return &gt; 2%, existing $1–$5 stake levels.</p>'+table(s[s.strategy == SEARCH_CANDIDATE])
    body += '<h2>Cumulative test profit</h2><p>Fixed filtered Legacy variance, by matchweek; all curves start at $0.</p>'
    weekly = ledger[ledger.strategy == SEARCH_CANDIDATE].groupby(['week','model']).net.sum().unstack(fill_value=0).sort_index()
    weekly.to_csv(OUT/'weekly_profit.csv')
    curves = {name:np.r_[0,weekly[name].cumsum().to_numpy()] for name in weekly}
    low = min(0.,min(v.min() for v in curves.values()))-20
    high = max(0.,max(v.max() for v in curves.values()))+20
    xp = lambda i:80+i/len(weekly)*870
    yp = lambda v:300-(v-low)/(high-low)*260
    svg = '<svg viewBox="0 0 1000 360" role="img" aria-label="Cumulative test profit by model">'
    for v in np.linspace(low,high,5):
        svg += f'<line x1="80" x2="950" y1="{yp(v)}" y2="{yp(v)}" stroke="#ccd5df"/><text x="5" y="{yp(v)}">${v:.0f}</text>'
    for j,(name,values) in enumerate(curves.items()):
        color = ['#287b63','#4267c5','#b34c47','#8b599c','#a86f20'][j]
        pts = ' '.join(f'{xp(i):.2f},{yp(v):.2f}' for i,v in enumerate(values))
        svg += f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5"><title>{html.escape(name)}: ${values[-1]:.2f}</title></polyline>'
        svg += f'<text x="{15+j*195}" y="345" fill="{color}" font-size="12">{html.escape(name.replace("expanded__", ""))}</text>'
    svg += '<text x="80" y="320">Start</text><text x="850" y="320">Matchweek 38</text></svg>'
    body += svg
    body += '<h2>All 15 fixed strategies: net dollars</h2>'+table(s.pivot(index='strategy',columns='model',values='net').reset_index())
    body += '<h2>Full strategy accounting</h2><p>Flat net uses $1 on exactly the same selected bets. ROI is net / total staked.</p>'+table(s)
    body += '<h2>Paired log-loss differences versus expanded RF</h2><p>Negative favors the ordered model. Approximate 95% intervals from 2,000 paired matchweek resamples; these do not cover all team/time dependence or selection uncertainty.</p>'+table(ci)
    body += '<h2>Method and limitations</h2><p>Unpenalized maximum likelihood, trained on away win &lt; draw &lt; home win. Logistic or normal cumulative link with two ordered cutpoints and shared coefficients across cutpoints. Probabilities are direct; no Poisson goal conversion. These models do not predict goal counts. Training-only scaling and rank reduction remove constants and exact collinearity; no feature selection or tuning on test results. Unknown teams have zero dummy features. Train in-sample results are not OOB and are omitted. No strategy optimization; the test season has already been inspected, so these are exploratory comparisons. Historical odds and actual lineups, no fees, compounding or bankroll constraints.</p>'
    body += '<p><a href="https://www.statsmodels.org/stable/examples/notebooks/generated/ordinal_regression.html">Ordered model reference and parameterization</a>. The local implementation uses an analytic likelihood gradient, finite-difference checks, two starts, convergence checks and prediction/export reconciliation.</p>'
    body += '<details><summary>Fit diagnostics</summary><pre>'+html.escape(json.dumps(status,indent=2))+'</pre></details>'
    page = '<!doctype html><meta charset="utf-8"><title>Ordered models vs expanded RF</title><style>body{font:15px system-ui;margin:30px;color:#182532}table{border-collapse:collapse;font-size:12px}td,th{padding:8px;border-bottom:1px solid #ddd}svg{width:100%;max-width:1000px}pre{white-space:pre-wrap}</style>'+body
    (OUT/'report.html').write_text(page,encoding='utf-8')
    # Publish through the already-running comparison route without replacing its existing report.
    path = BASE/'report.html'
    existing = path.read_text(encoding='utf-8').split('<!-- ordered-comparison -->')[0]
    path.write_text(existing+'<!-- ordered-comparison --><hr><section id="ordered-models">'+body+'</section>',encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=True)
    status = json.loads((BASE/'status.json').read_text())
    for name,digest in status['input_hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
    train = pd.read_csv(ROOT/'data/expanded_2008/train.csv')
    test = pd.read_csv(ROOT/'data/expanded_2008/test.csv')
    saved = pd.read_csv(BASE/'match_predictions.csv')
    for c in saved.columns[:7]:
        assert np.array_equal(saved[c].to_numpy(),test[c].to_numpy()), c
    cats = ['home_team_name','away_team_name']
    nums = [c for c in train if c not in cats+['home_score','away_score']]
    enc = OneHotEncoder(sparse_output=False,handle_unknown='ignore')
    scale = StandardScaler()
    x = np.column_stack([enc.fit_transform(train[cats]),scale.fit_transform(train[nums])])
    xt = np.column_stack([enc.transform(test[cats]),scale.transform(test[nums])])
    center = x.mean(axis=0)
    x -= center; xt -= center
    _,singular,vh = np.linalg.svd(x,full_matrices=False)
    keep = singular > singular[0]*1e-10
    basis = vh[keep].T / (singular[keep]/np.sqrt(len(train)))
    x = x @ basis; xt = xt @ basis
    y = np.sign(train.home_score-train.away_score).to_numpy(int)+1
    labels = (1-np.sign(test.home_score-test.away_score)).to_numpy(int)
    fit_status = dict(train_matches=len(train),test_matches=len(test),rank=x.shape[1],
                      raw_features=len(center),input_hashes=status['input_hashes'],fits={})
    probs = {}
    rng = np.random.default_rng(20261012)
    for link in ['logit','probit']:
        print('Fitting ordered '+link,flush=True)
        dist = logistic if link == 'logit' else norm
        cuts = dist.ppf(np.cumsum(np.bincount(y)/len(y))[:2])
        init = np.r_[np.zeros(x.shape[1]),cuts[0],np.log(cuts[1]-cuts[0])]
        # Check all parameter derivatives on a deterministic small subset.
        probe = init+rng.normal(0,.015,len(init))
        err = check_grad(lambda t:objective(t,x[:80],y[:80],link)[0],
                         lambda t:objective(t,x[:80],y[:80],link)[1],probe)
        assert err < 1e-4, (link,err)
        fits = []
        for start in [init,init+rng.normal(0,.02,len(init))]:
            result = minimize(objective,start,args=(x,y,link),jac=True,method='L-BFGS-B',
                              bounds=[(None,None)]*(len(init)-1)+[(-12,8)],
                              options=dict(maxiter=2000,ftol=1e-13,gtol=1e-7,maxls=50))
            assert result.success, result.message
            assert np.max(abs(result.jac)) < 2e-5, np.max(abs(result.jac))
            fits.append(result)
        assert abs(fits[0].fun-fits[1].fun) < 1e-7
        result = min(fits,key=lambda r:r.fun)
        name = 'expanded__Ordered_'+link
        p = probability(result.x,xt,link); probs[name] = p
        fit_status['fits'][name] = dict(converged=True,iterations=result.nit,
            train_nll=result.fun,gradient_max=float(np.max(abs(result.jac))),
            finite_difference_error=err,two_start_difference=abs(fits[0].fun-fits[1].fun))
        artifact = dict(theta=result.x,link=link,encoder=enc,scaler=scale,center=center,
                        basis=basis,numeric=nums,categorical=cats)
        joblib.dump(artifact,OUT/(name+'.joblib'))
        loaded = joblib.load(OUT/(name+'.joblib'))
        np.testing.assert_allclose(probability(loaded['theta'],xt,link),p)
        print(json.dumps(metrics(name,p,labels)),flush=True)
    for family in ['RF_vector','Linear_vector','Poisson_alpha001']:
        name = 'expanded__'+family
        probs[name] = saved[[name+'__p_'+side for side in ['home','draw','away']]].to_numpy()
    m = pd.DataFrame([metrics(name,p,labels) for name,p in probs.items()])
    market = 1/test[['B365H','B365D','B365A']].to_numpy(); market /= market.sum(axis=1,keepdims=True)
    m = pd.concat([m,pd.DataFrame([metrics('Market',market,labels)])],ignore_index=True)
    rows=[]; ledger=[]
    for name,p in probs.items():
        a,b = bets(name,p,test,labels);rows.extend(a);ledger.extend(b)
    s = pd.DataFrame(rows);ledger = pd.DataFrame(ledger)
    old = pd.read_csv(BASE/'strategy_results.csv')
    for name in probs:
        if 'Ordered' in name:continue
        merged = s[s.model==name].merge(old[old.model==name],on='strategy',suffixes=('_new','_old'))
        assert len(merged)==15
        for col in ['bets','staked','net','flat_net','roi']:
            np.testing.assert_allclose(merged[col+'_new'],merged[col+'_old'],atol=1e-7)
    weeks = test.Matchweek.to_numpy(int); unique=np.unique(weeks)
    samples = rng.integers(0,len(unique),size=(2000,len(unique)))
    ref = -np.log(probs['expanded__RF_vector'][np.arange(len(test)),labels])
    ci=[]
    for name,p in probs.items():
        if 'Ordered' not in name:continue
        delta = -np.log(p[np.arange(len(test)),labels])-ref
        sums=np.array([delta[weeks==w].sum() for w in unique])
        counts=np.array([(weeks==w).sum() for w in unique])
        draws=sums[samples].sum(axis=1)/counts[samples].sum(axis=1)
        ci.append(dict(model=name,difference=delta.mean(),ci_low=np.quantile(draws,.025),ci_high=np.quantile(draws,.975)))
    pred = test[['home_team_name','away_team_name','home_score','away_score','B365H','B365D','B365A']].copy()
    for name,p in probs.items():
        for j,side in enumerate(['home','draw','away']):pred[name+'__p_'+side]=p[:,j]
    pred.to_csv(OUT/'match_predictions.csv',index=False)
    for name,p in probs.items():
        recovered=pd.read_csv(OUT/'match_predictions.csv')[[name+'__p_'+side for side in ['home','draw','away']]].to_numpy()
        np.testing.assert_allclose(recovered,p,atol=1e-14)
    calibration=[]
    for name,p in probs.items():
        for j,side in enumerate(['home','draw','away']):
            for b in range(10):
                selected=(p[:,j]>=b/10)&(p[:,j]<(b+1)/10)
                if selected.any():calibration.append(dict(model=name,outcome=side,bin=b,count=int(selected.sum()),predicted=p[selected,j].mean(),observed=(labels[selected]==j).mean()))
    pd.DataFrame(calibration).to_csv(OUT/'calibration.csv',index=False)
    m.to_csv(OUT/'model_metrics.csv',index=False);s.to_csv(OUT/'strategy_results.csv',index=False)
    pd.DataFrame(ci).to_csv(OUT/'paired_differences.csv',index=False)
    ledger.to_csv(OUT/'test_bet_ledger.csv',index=False)
    fit_status['verification']='Gradients, convergence, two starts, test alignment, probability normalization, saved predictions and baseline strategy reconciliation passed.'
    (OUT/'status.json').write_text(json.dumps(fit_status,indent=2),encoding='utf-8')
    render(m,s,pd.DataFrame(ci),ledger,fit_status)
    print(m.to_string(index=False));print(s[s.strategy==SEARCH_CANDIDATE].to_string(index=False))
    print(pd.DataFrame(ci).to_string(index=False))


if __name__ == '__main__':
    with threadpool_limits(limits=4):main()
