"""Resumable 2023–26 source collection, isolated from active training data."""
import csv,gzip,json,zipfile,io
from collections import Counter
from pathlib import Path
from scrape_older_seasons import fetch,csv_rows,date_value,odds_ok,write_csv,atomic_json,stamp

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/recent_seasons'
TM='https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/'
LEAGUES={'GB1':'E0','ES1':'SP1','L1':'D1','IT1':'I1'}


def valid_gzip(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        if not next(csv.DictReader(stream),None):raise ValueError('Empty compressed CSV')


def valid_ratings(path):
    with zipfile.ZipFile(path) as z:
        if not any(n.endswith('.csv') for n in z.namelist()):raise ValueError('Missing ratings CSV')
        if z.testzip():raise ValueError('Invalid ratings archive')


def main():
    raw=OUT/'raw';raw.mkdir(parents=True,exist_ok=True)
    status=dict(started_at=stamp(),scope='2023/24, 2024/25, 2025/26; E0 SP1 D1 I1',sources=[],errors=[],model_ready=False)
    rows=[];coverage=[]
    for year in [2023,2024,2025]:
        code=f'{year%100:02d}{(year+1)%100:02d}'
        for league in LEAGUES.values():
            label=f'{code}/{league}';path=raw/f'{code}_{league}.csv'
            try:
                print('Download results '+label,flush=True)
                status['sources'].append(fetch(f'https://www.football-data.co.uk/mmz4281/{label}.csv',path,csv_rows,3))
                matches=[];seen=set()
                for r in csv_rows(path):
                    if not str(r.get('FTHG','')).isdigit() or not str(r.get('FTAG','')).isdigit():continue
                    day=date_value(r['Date']);key=(day,r['HomeTeam'],r['AwayTeam'])
                    if key in seen:raise ValueError('Duplicate fixture '+str(key))
                    seen.add(key)
                    matches.append(dict(season=year,league=league,fixture_date=day,home_team_name=r['HomeTeam'],away_team_name=r['AwayTeam'],home_score=int(r['FTHG']),away_score=int(r['FTAG']),B365H=r.get('B365H',''),B365D=r.get('B365D',''),B365A=r.get('B365A',''),valid_bet365=odds_ok(r),source_file=path.name))
                rows.extend(matches)
                expected=306 if league=='D1' else 380
                coverage.append(dict(season=year,league=league,matches=len(matches),expected=expected,complete=len(matches)==expected,with_bet365=sum(r['valid_bet365'] for r in matches)))
            except Exception as exc:status['errors'].append(dict(source=label,error=str(exc)))
            atomic_json(OUT/'status.json',status)
    write_csv(OUT/'results_odds.csv',rows);write_csv(OUT/'coverage.csv',coverage)
    games={}
    for table in ['games','game_lineups','players']:
        path=raw/(table+'.csv.gz')
        try:
            print('Download historical '+table,flush=True)
            status['sources'].append(fetch(TM+path.name,path,valid_gzip,3))
            with gzip.open(path,'rt',encoding='utf-8') as stream:
                reader=csv.DictReader(stream)
                if table=='games':
                    selected=[r for r in reader if r['competition_id'] in LEAGUES and r['season'] in ['2023','2024','2025']]
                    games={r['game_id']:r for r in selected};write_csv(OUT/'games.csv',selected)
                elif table=='game_lineups':
                    selected=[r for r in reader if r['game_id'] in games]
                    write_csv(OUT/'lineups.csv',selected)
                    counts=Counter((r['game_id'],r['club_id']) for r in selected if r.get('type')=='starting_lineup')
                    status['lineups']=dict(rows=len(selected),games=len(games),games_with_22_starters=sum(counts[g,k['home_club_id']]==11 and counts[g,k['away_club_id']]==11 for g,k in games.items()))
        except Exception as exc:status['errors'].append(dict(source=table,error=str(exc)))
        atomic_json(OUT/'status.json',status)
    try:
        print('Download EA FC24 historical rating archive',flush=True)
        path=raw/'fc24.zip'
        status['sources'].append(fetch('https://www.kaggle.com/api/v1/datasets/download/stefanoleone992/ea-sports-fc-24-complete-player-dataset',path,valid_ratings,3))
        with zipfile.ZipFile(path) as z:
            names=[n for n in z.namelist() if n.endswith('male_players.csv') and not n.endswith('female_players.csv')]
            if len(names)!=1:raise ValueError('Expected male_players.csv not uniquely identified')
            with z.open(names[0]) as stream:
                ratings=[r for r in csv.DictReader(io.TextIOWrapper(stream,encoding='utf-8-sig')) if float(r.get('fifa_version','0'))>=23]
        write_csv(OUT/'ratings_fc23_fc24.csv',ratings)
        status['ratings']=dict(rows=len(ratings),versions=dict(Counter(r['fifa_version'] for r in ratings)),dates=sorted(set(r.get('fifa_update_date') or r.get('update_as_of','') for r in ratings)))
    except Exception as exc:status['errors'].append(dict(source='FC24 ratings',error=str(exc)))
    status.update(finished_at=stamp(),results_matches=len(rows),coverage=coverage,note='Raw sources collected only. Lineup identities and strictly pre-match rating dates must be matched/audited before model use. See recent_ratings_audit.json for separate FC25/26 collection. Existing models and train/test untouched.')
    atomic_json(OUT/'status.json',status)
    print(json.dumps(status,indent=2),flush=True)


if __name__=='__main__':main()
