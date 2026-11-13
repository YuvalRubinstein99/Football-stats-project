"""Resumable public historical collection; never modifies the active model/data.

Sources: Football-Data CSV archive and Hugo Mathien's European Soccer Database.
Run: python scripts/scrape_older_seasons.py
"""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import sqlite3
import statistics
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/older_seasons'
LEAGUES={'England':'E0','Spain':'SP1','Germany':'D1','Italy':'I1','France':'F1'}
ARCHIVE='https://www.kaggle.com/api/v1/datasets/download/hugomathien/soccer'

def stamp():return datetime.now(timezone.utc).isoformat()

def atomic_json(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)

def write_csv(path,rows):
    if not rows:return
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tmp.replace(path)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def fetch(url,path,validate,delay,*,offline=False,refresh=False):
    meta=path.with_suffix(path.suffix+'.json')
    if offline and refresh:raise ValueError('Cannot refresh in offline mode')
    if not refresh and path.exists() and meta.exists():
        record=json.loads(meta.read_text())
        if record.get('url')==url and record['sha256']==digest(path):validate(path);return record
    if offline:raise FileNotFoundError('Verified cached source unavailable: '+str(path))
    for attempt in range(3):
        try:
            time.sleep(delay)
            with urlopen(Request(url,headers={'User-Agent':'FootballStatsProject historical research/1.0'}),timeout=60) as response:
                tmp=path.with_suffix('.download')
                total=0
                with tmp.open('wb') as f:
                    while block:=response.read(1024*1024):
                        total+=len(block)
                        if total>500_000_000:raise ValueError('Download exceeds 500 MB limit')
                        f.write(block)
            validate(tmp)
            tmp.replace(path)
            record=dict(url=url,fetched_at=stamp(),bytes=total,sha256=digest(path))
            atomic_json(meta,record);return record
        except HTTPError as exc:
            if exc.code in (401,403,404):raise
            if attempt==2:raise
            time.sleep(min(60,max(10, int(exc.headers.get('Retry-After','10')) if exc.headers.get('Retry-After','10').isdigit() else 10)*(attempt+1)))
        except (OSError,ValueError):
            if attempt==2:raise
            time.sleep(10*(attempt+1))

def csv_rows(path):
    text=path.read_bytes().decode('utf-8-sig',errors='replace')
    rows=[r for r in csv.DictReader(io.StringIO(text)) if any(v and str(v).strip() for v in r.values())]
    if not rows or not {'Date','HomeTeam','AwayTeam','FTHG','FTAG'}<=set(rows[0]):raise ValueError('Not a valid results CSV')
    return rows

def valid_zip(path):
    with zipfile.ZipFile(path) as z:
        if 'database.sqlite' not in z.namelist():raise ValueError('Expected database.sqlite missing')
        if z.testzip():raise ValueError('Archive checksum failed')

def date_value(value):
    for fmt in ('%d/%m/%Y','%d/%m/%y','%Y-%m-%d'):
        try:return datetime.strptime(value,fmt).date().isoformat()
        except ValueError:pass
    raise ValueError('Unrecognized date: '+value)

def odds_ok(r):
    try:return all(math.isfinite(float(r[k])) and float(r[k])>1 for k in ('B365H','B365D','B365A'))
    except (ValueError,TypeError,KeyError):return False

def rating_before(history,day):
    """Strictly prior calendar day, never the nearest future rating."""
    pos=bisect_left([r[0] for r in history],day)-1
    return history[pos] if pos>=0 else None

def rating_features(values,prefix):
    return {prefix+'_Overall_max':max(values),prefix+'_Overall_min':min(values),
        prefix+'_Overall_sd':statistics.stdev(values),prefix+'_Overall_mean':statistics.mean(values),
        prefix+'_Overall_mean_ln':statistics.mean(map(math.log,values)),
        prefix+'_Overall_mean_sqrt':statistics.mean(map(math.sqrt,values)),
        prefix+'_3rd_best':sorted(values,reverse=True)[2],prefix+'_3rd_worst':sorted(values)[2]}

def import_lineups(delay):
    archive=OUT/'raw/european_soccer.zip'
    fetch(ARCHIVE,archive,valid_zip,delay)
    db=OUT/'raw/database.sqlite'
    if not db.exists():
        with zipfile.ZipFile(archive) as z,z.open('database.sqlite') as src,db.with_suffix('.tmp').open('wb') as dst:shutil.copyfileobj(src,dst)
        db.with_suffix('.tmp').replace(db)
    con=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True);con.row_factory=sqlite3.Row
    if con.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise ValueError('Database integrity failure')
    countries={r['id']:LEAGUES[r['name']] for r in con.execute('SELECT * FROM Country') if r['name'] in LEAGUES}
    teams={r['team_api_id']:r['team_long_name'] for r in con.execute('SELECT * FROM Team')}
    players={r['player_api_id']:r['player_name'] for r in con.execute('SELECT * FROM Player')}
    ratings=defaultdict(list)
    for r in con.execute('SELECT player_api_id,date,overall_rating FROM Player_Attributes WHERE overall_rating BETWEEN 1 AND 99 ORDER BY date,id'):
        ratings[r[0]].append((r[1][:10],float(r[2])))
    matches=[dict(r) for r in con.execute('SELECT * FROM Match ORDER BY date,match_api_id') if r['country_id'] in countries and int(r['season'][:4])<2014]
    counts=defaultdict(Counter);history=defaultdict(list);features=[];audit=[];excluded=[]
    days=defaultdict(list)
    for r in matches:days[r['date'][:10]].append(r)
    for day,group in sorted(days.items()):
        for r in group:
            league=countries[r['country_id']];season=r['season'];count=counts[league,season];count['matches']+=1
            base=dict(match_id=r['match_api_id'],date=day,league=league,season=season,home_team_name=teams[r['home_team_api_id']],away_team_name=teams[r['away_team_api_id']],home_score=r['home_team_goal'],away_score=r['away_team_goal'],Matchweek=r['stage'],B365H=r['B365H'],B365D=r['B365D'],B365A=r['B365A'])
            reasons=[];values={};latest=[]
            if not odds_ok(r):reasons.append('missing_bet365_odds')
            ids=[r[f'{side}_player_{i}'] for side in ('home','away') for i in range(1,12)]
            if None in ids or len(set(ids))!=22:reasons.append('incomplete_or_duplicate_lineup')
            for side in ('home','away'):
                vals=[]
                for i in range(1,12):
                    pid=r[f'{side}_player_{i}'];rating=rating_before(ratings.get(pid,[]),day)
                    if rating is None:reasons.append('missing_prior_rating')
                    else:
                        vals.append(rating[1]);latest.append(rating[0])
                        audit.append(dict(match_id=r['match_api_id'],date=day,side=side,position=i,player_id=pid,player_name=players.get(pid,''),rating=rating[1],rating_date=rating[0]))
                if len(vals)==11:values.update(rating_features(vals,'HomePlayer' if side=='home' else 'AwayPlayer'))
                prev=history[league,season,r[side+'_team_api_id']]
                values.update({side+'_GD_prior':sum(x[0] for x in prev),side+'_Points_prior':sum(x[1] for x in prev),side+'_GD_form':sum(x[0] for x in prev[-5:]),side+'_Points_form':sum(x[1] for x in prev[-5:])})
            if reasons:
                for reason in set(reasons):count[reason]+=1
                excluded.append(dict(**base,reasons=';'.join(sorted(set(reasons)))))
            else:
                count['eligible']+=1
                features.append(dict(**base,**values,latest_rating_date=max(latest),oldest_rating_date=min(latest)))
        # Settle only after every prediction on this calendar date has been built.
        for r in group:
            for side,sign in [('home',1),('away',-1)]:
                gd=sign*(r['home_team_goal']-r['away_team_goal'])
                history[countries[r['country_id']],r['season'],r[side+'_team_api_id']].append((gd,3 if gd>0 else 1 if gd==0 else 0))
    con.close()
    if len({r['match_id'] for r in features})!=len(features):raise ValueError('Duplicate match IDs')
    assert all(r['latest_rating_date']<r['date'] for r in features)
    write_csv(OUT/'lineup_features.csv',features);write_csv(OUT/'player_rating_audit.csv',audit);write_csv(OUT/'excluded_lineup_matches.csv',excluded)
    coverage=[dict(league=k[0],season=k[1],**{field:v.get(field,0) for field in ['matches','eligible','missing_bet365_odds','incomplete_or_duplicate_lineup','missing_prior_rating']}) for k,v in sorted(counts.items())]
    write_csv(OUT/'lineup_coverage.csv',coverage)
    return dict(eligible=len(features),earliest_eligible=min((r['date'] for r in features),default=None),coverage=coverage,
        limitations='Actual historical starting lineups, not expected XIs. Ratings use strictly earlier dates, may be stale, and source timestamps are not independently verified publication times. Existing model not retrained. Form resets each season. No future-rating backfill.')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--earliest',type=int,default=2008);parser.add_argument('--latest',type=int,default=2013);parser.add_argument('--delay',type=float,default=3)
    args=parser.parse_args()
    if not 2008<=args.earliest<=args.latest<=2013 or args.delay<1:parser.error('Use years 2008–2013 and delay >=1 second')
    (OUT/'raw').mkdir(parents=True,exist_ok=True)
    lock=OUT/'run.lock'
    try:lock.mkdir()
    except FileExistsError:raise SystemExit('Another run or stale run.lock exists; verify process before removing lock')
    status=dict(started_at=stamp(),state='running',sources=[],errors=[])
    try:
        print('Importing dated lineups and ratings',flush=True)
        try:status['lineups']=import_lineups(args.delay)
        except Exception as exc:status['errors'].append(dict(stage='lineups',error=str(exc)))
        atomic_json(OUT/'status.json',status)
        normalized=[];coverage=[];blocked=set()
        for year in range(args.latest,args.earliest-1,-1):
            code=f'{year%100:02d}{(year+1)%100:02d}'
            for league in LEAGUES.values():
                if 'football-data' in blocked:continue
                url=f'https://www.football-data.co.uk/mmz4281/{code}/{league}.csv'
                print(f'Collecting {league} {year}-{year+1}',flush=True)
                try:
                    path=OUT/f'raw/{league}_{code}.csv';source=fetch(url,path,csv_rows,args.delay);status['sources'].append(source)
                    valid=[];seen=set();invalid=0
                    for r in csv_rows(path):
                        try:
                            day=date_value(r['Date']);hg=int(r['FTHG']);ag=int(r['FTAG'])
                            if hg<0 or ag<0 or not r['HomeTeam'] or not r['AwayTeam']:raise ValueError('Invalid match')
                            key=(day,r['HomeTeam'],r['AwayTeam'])
                            if key in seen:raise ValueError('Duplicate match')
                            seen.add(key)
                            row=dict(league=league,season=f'{year}/{year+1}',date=day,home=r['HomeTeam'],away=r['AwayTeam'],home_goals=hg,away_goals=ag,B365H=r.get('B365H',''),B365D=r.get('B365D',''),B365A=r.get('B365A',''),bet365_complete=int(odds_ok(r)),source_url=url)
                            valid.append(row)
                        except (ValueError,TypeError,KeyError):invalid+=1
                    normalized.extend(valid);coverage.append(dict(league=league,season=f'{year}/{year+1}',matches=len(valid),with_bet365=sum(r['bet365_complete'] for r in valid),invalid_rows=invalid))
                    write_csv(OUT/'results_and_odds.csv',normalized);write_csv(OUT/'results_coverage.csv',coverage)
                except Exception as exc:
                    status['errors'].append(dict(url=url,error=str(exc)))
                    if isinstance(exc,HTTPError) and exc.code in (401,403,429):blocked.add('football-data')
                status['updated_at']=stamp();atomic_json(OUT/'status.json',status)
        status.update(state='completed_with_gaps' if status['errors'] else 'completed',finished_at=stamp(),results_matches=len(normalized),result_seasons=len(coverage))
        atomic_json(OUT/'status.json',status)
        print(json.dumps({k:v for k,v in status.items() if k not in ('sources','lineups')},indent=2),flush=True)
    finally:lock.rmdir()

if __name__=='__main__':main()
