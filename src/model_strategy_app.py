"""Unified expanded-model predictions and matched strategy histories."""
import sys,json,csv,io,secrets,threading,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.lineup_deps'));sys.path.insert(0,str(ROOT/'scripts'))
from functools import lru_cache
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse,parse_qs
from select_expanded_features import transform,predict,np,pd,joblib
from expanded_model import ExpandedForest
from lineup_sources import team_key
from live_bets import integrated_recommendations,STRATEGIES,SEARCH_CANDIDATE
from stake_levels import size_bet
from live_predictions import probabilities
from probability_calibration import SUFFIX,calibrate,artifacts

FAMILIES={'rf':'Expanded RF','linear':'Linear','poisson':'Poisson','logit':'Ordered_logit','probit':'Ordered_probit'}
FAMILIES.update(clf='RF classifier',clf_form='RF classifier + adjusted form',clf_lineup='RF classifier + lineup',clf_both='RF classifier + both')
SOURCE=ROOT/'results/model_strategy/live/predictions.json'
HISTORY=ROOT/'results/team_identity'
BASE=ROOT/'results/expanded_comparison'
RECENCY=ROOT/'results/recency'
CLASSIFIER=ROOT/'results/classifier'

def model_path(name):
    base=name.removesuffix(SUFFIX)
    if base.startswith('RF classifier'):return CLASSIFIER/(base+'.joblib')
    if base.endswith('__recent'):return RECENCY/(base+'.joblib')
    return BASE/'expanded__RF_vector.joblib' if base=='Expanded RF' else HISTORY/(base+'.joblib')


@lru_cache(maxsize=9)
def model(name):
    if name.endswith(SUFFIX):return model(name.removesuffix(SUFFIX))
    if name.startswith('RF classifier'):
        from classifier_model import ClassifierForest
        artifacts(classifier=True)
        return ClassifierForest(model_path(name))
    if name.endswith('__recent'):
        artifacts(True) # Verify weighted model and dataset fingerprints.
        return ExpandedForest(model_path(name)) if name.startswith('Expanded RF') else joblib.load(model_path(name))
    if name=='Expanded RF':return ExpandedForest()
    return joblib.load(HISTORY/(name+'.joblib'))


@lru_cache(maxsize=1)
def catalog():
    history=json.loads((HISTORY/'train_test_curves.json').read_text())
    train=pd.read_csv(ROOT/'data/expanded_2008/train.csv')
    names={team_key(n):n for c in ['home_team_name','away_team_name'] for n in train[c].unique()}
    return history,names


def source(lineup_only=False):
    paths=[SOURCE,ROOT/'results/expanded_interface/live/predictions.json',ROOT/'results/live/predictions.json']
    available=[p for p in paths if p.exists()]
    if not available:raise ValueError('No saved inputs yet. Refresh inputs first.')
    for path in sorted(available,key=lambda p:p.stat().st_mtime_ns,reverse=True):
        report=json.loads(path.read_text(encoding='utf-8'))
        if not lineup_only or report.get('model_kind')=='lineup_rf':
            return {**report,'_source_path':str(path)}
    raise ValueError('Refresh match inputs to load upcoming lineups.')


def resolve(family,teams):
    if family not in FAMILIES:raise ValueError('Unknown model')
    if family.startswith('clf'):return FAMILIES[family]
    return 'Expanded RF' if family=='rf' else FAMILIES[family]+('__teams' if teams else '__no_teams')


def infer(name,row):
    if name.endswith(SUFFIX):
        p,g=infer(name.removesuffix(SUFFIX),row)
        return calibrate(name,p).tolist(),g
    fitted=model(name)
    if name.startswith('RF classifier'):return fitted.predict(row),None
    if name.startswith('Expanded RF'):
        h,a=fitted.predict(row);p,_=probabilities(max(h,1e-8),max(a,1e-8))
        return list(p),[h,a]
    row=dict(row);names=catalog()[1]
    for col in ['home_team_name','away_team_name']:
        row[col]=names.get(team_key(row[col]),row[col])
    frame=pd.DataFrame([row]);x,xo=transform(fitted,frame)
    p,g=predict(fitted['fit'],x,xo)
    return p[0].tolist(),g[0].tolist() if g is not None else None


def view(family='rf',teams=True,strategy=SEARCH_CANDIDATE,filters=None,calibrated=False,recent=False):
    if strategy not in STRATEGIES+[SEARCH_CANDIDATE]:raise ValueError('Unknown strategy')
    is_classifier=family.startswith('clf')
    if is_classifier and recent:raise ValueError('Classifier ablations use equal weights; select original equal weights')
    original=resolve(family,teams);base=original+('__recent' if recent else '');name=base+SUFFIX if calibrated else base
    report=source();history,_=catalog()
    from model_filters import config,apply,historical
    cfg=config(strategy,filters)
    if cfg['source'] not in ('bet365','winner','winner_auto'):raise ValueError('Choose Bet365 or Winner')
    winner_feed=None;winner_error=None
    if cfg['source']=='winner_auto':
        from winner_source import download
        try:winner_feed=download()
        except ValueError as exc:winner_error=str(exc)
    audits={r['fixture']:r['features'] for r in report.get('feature_audit',[])}
    matches=[]
    for match in report['predictions']:
        if cfg['league']!='all' and match['league']!=cfg['league']:continue
        key='|'.join(str(match[k]) for k in ['league','date','home','away'])
        item={k:match.get(k) for k in ['league','date','home','away','B365H','B365D','B365A','lineup_status','notes']}
        try:
            if key not in audits:raise ValueError('Saved feature inputs are unavailable; refresh required')
            features_row=audits[key]
            if is_classifier:
                from classifier_model import live_features
                features_row=live_features(features_row,match,report,model(name).numeric)
            p,g=infer(name,features_row)
            row={**match,'p_home':p[0],'p_draw':p[1],'p_away':p[2]}
            item.update(probabilities=p,goals=g)
            if cfg['source']=='winner_auto':
                from winner_source import quote
                if winner_error:raise ValueError(winner_error)
                odds,meta=quote(winner_feed,match)
                row.update(B365H=odds['home'],B365D=odds['draw'],B365A=odds['away'])
                item['quote']=meta
            if cfg['source']=='winner':
                from odds_sources import quote_for,fixture_key
                odds,meta=quote_for('winner',(filters or {}).get('quotes',{}).get(fixture_key(match)),match)
                row.update(B365H=odds['home'],B365D=odds['draw'],B365A=odds['away'])
                item['quote']=meta
            bet=next(b for b in integrated_recommendations(row) if b['strategy']==cfg['strategy'])
            item.update(probabilities=p,goals=g,bet=apply(bet,cfg))
        except (ValueError,KeyError,TypeError) as exc:item['error']=str(exc)
        matches.append(item)
    fitted=model(name)
    features=fitted['columns'] if isinstance(fitted,dict) else fitted.numeric+fitted.categorical
    diagnostic=pd.read_csv(CLASSIFIER/'diagnostics.csv' if is_classifier else RECENCY/'diagnostics.csv' if recent else HISTORY/'overfitting_diagnostics.csv').set_index('model').loc[base].to_dict()
    diagnostic={k:float(v) if np.isfinite(v) else None for k,v in diagnostic.items()}
    curves,summary,league_curves,league_summary=historical(name,cfg,by_league=True) if cfg['source']=='bet365' else ([],[],[],[])
    calibration=artifacts(recent,is_classifier)['models'][base] if (ROOT/'results/calibration/calibrators.json').exists() else None
    weighting=json.loads((RECENCY/'summary.json').read_text()) if not is_classifier and (RECENCY/'summary.json').exists() else None
    if weighting:weighting=dict(**weighting['models'][original],metrics=[r for r in weighting['metrics'] if r['model']==original])
    experiment=json.loads((CLASSIFIER/'summary.json').read_text()) if (CLASSIFIER/'summary.json').exists() else None
    return dict(model=name,strategy=strategy,features=features,diagnostics=diagnostic,filters=cfg,calibrated=calibrated,calibration=calibration,recent=recent,weighting=weighting,classifier_experiment=experiment,
        winner_feed=winner_feed,winner_error=winner_error,
        generated_at=report['generated_at'],matches=matches,warnings=report.get('warnings',[])+(['Winner stakes are ILS. Model input odds remain Bet365. No historical Winner backtest is available. Single-bet eligibility and minimum stakes must be checked on Winner.'] if cfg['source'].startswith('winner') else []),skipped_count=len(report.get('skipped',[])),
        curves=curves,summary=summary,league_curves=league_curves,league_summary=league_summary)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=8767);args=parser.parse_args(argv)
    token=secrets.token_urlsafe(32);lock=threading.Lock();state=dict(running=False,message='Saved inputs available. Select a model and strategy.',error=None)
    def refresh(days,offline):
        def progress(message):
            with lock:state['message']=message
        try:
            from lineup_pipeline import run
            run(leagues=['E0','SP1','D1','I1'],days=days,offline=offline,expanded=True,output=SOURCE.parent,progress=progress)
        except Exception as exc:
            with lock:state['error']=str(exc)
        finally:
            with lock:state['running']=False
    class Handler(BaseHTTPRequestHandler):
        def send(self,code,data,kind='application/json'):
            raw=data if isinstance(data,bytes) else json.dumps(data,allow_nan=False).encode()
            self.send_response(code);self.send_header('Content-Type',kind+'; charset=utf-8');self.send_header('Content-Length',str(len(raw)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(raw)
        def valid(self):return self.headers.get('Host') in {f'127.0.0.1:{args.port}',f'localhost:{args.port}'}
        def do_GET(self):
            if not self.valid():return self.send(403,dict(error='Invalid host'))
            parsed=urlparse(self.path)
            try:
                if parsed.path in ('/', '/lineups/'):
                    return self.send(200,Path(__file__).with_name('lineups.html').read_text(encoding='utf-8').replace('__TOKEN__',token).encode(),'text/html')
                if parsed.path=='/api/lineups':
                    from upcoming_lineups import board
                    return self.send(200,board(source(lineup_only=True)))
                if parsed.path=='/transfermarkt-comparison/':return self.send(200,(ROOT/'results/transfermarkt_models/report.html').read_bytes(),'text/html')
                if parsed.path=='/bankroll-backtest/':return self.send(200,Path(__file__).with_name('bankroll-backtest.html').read_bytes(),'text/html')
                if parsed.path=='/bankroll-mixture.json':
                    import hashlib
                    name=parse_qs(parsed.query).get('model',['Expanded RF'])[0]
                    filename=hashlib.sha256(name.encode()).hexdigest()+'.json'
                    return self.send(200,(ROOT/'results/bankroll_backtest/daily'/filename).read_bytes())
                bankroll_files={'/bankroll-results.json':('results.json','application/json'),'/bankroll-summary.csv':('summary.csv','text/csv'),'/bankroll-ledger.csv.gz':('ledger.csv.gz','application/gzip')}
                if parsed.path in bankroll_files:
                    filename,kind=bankroll_files[parsed.path]
                    return self.send(200,(ROOT/'results/bankroll_backtest'/filename).read_bytes(),kind)
                if parsed.path=='/lab/':return self.send(200,Path(__file__).with_name('model-strategy.html').read_text(encoding='utf-8').replace('__TOKEN__',token).encode(),'text/html')
                if parsed.path=='/api/catalog':return self.send(200,dict(strategies=STRATEGIES+[SEARCH_CANDIDATE],default_strategy=SEARCH_CANDIDATE))
                if parsed.path=='/api/status':
                    with lock:return self.send(200,dict(state))
                if parsed.path in ['/api/view','/selections.csv']:
                    q=parse_qs(parsed.query);result=view(q.get('model',['rf'])[0],q.get('teams',['true'])[0]=='true',q.get('strategy',[SEARCH_CANDIDATE])[0],json.loads(q['filters'][0]) if 'filters' in q else None,q.get('calibrated',['false'])[0]=='true',q.get('recent',['false'])[0]=='true')
                    if parsed.path=='/api/view':return self.send(200,result)
                    buf=io.StringIO();fields=['model','strategy','source','currency','observed_at','decimal_odds','league','date','home','away','p_home','p_draw','p_away','selection','staking_action','stake_level','stake_amount','home_amount','draw_amount','away_amount','error']
                    writer=csv.DictWriter(buf,fieldnames=fields,extrasaction='ignore');writer.writeheader()
                    for r in result['matches']:
                        p=r.get('probabilities',[None]*3)
                        b=r.get('bet',{})
                        writer.writerow({**{k:r.get(k) for k in ['league','date','home','away','error']},'model':result['model'],'strategy':result['strategy'],'source':result['filters']['source'],'currency':'ILS' if result['filters']['source'].startswith('winner') else 'USD','observed_at':r.get('quote',{}).get('fetched_at',result['generated_at']),'p_home':p[0],'p_draw':p[1],'p_away':p[2],**b,'stake_amount':b.get('stake_dollars'),**{s+'_amount':b.get(s+'_dollars') for s in ['home','draw','away']}})
                    return self.send(200,buf.getvalue().encode(),'text/csv')
                return self.send(404,dict(error='Not found'))
            except Exception as exc:return self.send(400,dict(error=str(exc)))
        def do_POST(self):
            if not self.valid() or self.headers.get('X-App-Token')!=token:return self.send(403,dict(error='Invalid request'))
            if self.path!='/api/refresh':return self.send(404,dict(error='Not found'))
            try:
                length=int(self.headers.get('Content-Length',0))
                if not 0<length<2048:raise ValueError('Invalid request size')
                data=json.loads(self.rfile.read(length));days=data['days'];offline=data['offline']
                if type(days)is not int or not 1<=days<=30 or type(offline)is not bool:raise ValueError('Invalid options')
            except (KeyError,ValueError,TypeError) as exc:return self.send(400,dict(error=str(exc)))
            with lock:
                if state['running']:return self.send(409,dict(error='Refresh already running'))
                state.update(running=True,message='Refreshing lineup, FIFA and odds inputs…',error=None)
            threading.Thread(target=refresh,args=(days,offline),daemon=True).start();self.send(202,dict(started=True))
        def log_message(self,*args):pass
    catalog()
    print(f'Model & Strategy Lab: http://127.0.0.1:{args.port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()

if __name__=='__main__':main()
