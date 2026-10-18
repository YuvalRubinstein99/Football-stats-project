"""Match starter identities and build strictly prior-day rating treatments."""
import csv,gzip,sqlite3,json,hashlib
from collections import defaultdict
from bisect import bisect_left
from select_expanded_features import ROOT,pd,np
from lineup_sources import normalize
from classifier_features import club,lineup_features
from scrape_older_seasons import rating_features

OUT=ROOT/'data/fifa_updates'

def main():
    db=sqlite3.connect(OUT/'history.sqlite')
    if not (OUT/'index_complete.json').exists():
        db.executescript('DROP TABLE IF EXISTS ratings; DROP TABLE IF EXISTS names; CREATE TABLE ratings (id INTEGER, day TEXT, version INTEGER, rating INTEGER, PRIMARY KEY(id,day)); CREATE TABLE names (name TEXT,id INTEGER,PRIMARY KEY(name,id));')
        batch=[];names=set();n=0
        with gzip.open(OUT/'history.csv.gz','rt',encoding='utf-8',newline='') as f:
            for r in csv.DictReader(f):
                pid=int(r['player_id']);batch.append((pid,r['fifa_update_date'][:10],int(r['fifa_version']),int(r['overall'])))
                for key in ['short_name','long_name']:
                    if r[key]:names.add((normalize(r[key]),pid))
                n+=1
                if len(batch)==50000:
                    db.executemany('INSERT INTO ratings VALUES (?,?,?,?) ON CONFLICT(id,day) DO UPDATE SET version=excluded.version,rating=excluded.rating WHERE excluded.version>=ratings.version',batch);db.commit();batch=[]
                if n%1000000==0:print('Indexed',n,flush=True)
        db.executemany('INSERT INTO ratings VALUES (?,?,?,?) ON CONFLICT(id,day) DO UPDATE SET version=excluded.version,rating=excluded.rating WHERE excluded.version>=ratings.version',batch)
        db.executemany('INSERT OR IGNORE INTO names VALUES (?,?)',names);db.commit()
        (OUT/'index_complete.json').write_text(json.dumps(dict(rows=n)))
    aliases=defaultdict(set)
    for name,pid in db.execute('SELECT name,id FROM names'):aliases[name].add(pid)
    # Exact, globally unique aliases only; no guessed identities or future rating values.
    raw={}
    for year in range(2014,2023):
        path=ROOT/f'data/season_data/season{year%100:02d}{(year+1)%100:02d}.csv'
        if not path.exists():continue
        for r in pd.read_csv(path).to_dict('records'):
            key=(year,club(r['home_team_name']),club(r['away_team_name']))
            names=tuple(str(r.get(f'{side}Player{i}','')) for side in ['Home','Away'] for i in range(1,12))
            raw.setdefault(key,set()).add(names)
    frames={s:pd.read_csv(ROOT/f'results/classifier/{s}.csv') for s in ['train','test']}
    combined=pd.concat(frames.values())
    cutoffs=combined.groupby(['season','league']).fixture_date.min().to_dict()
    histories={};audit=[];missing=[];coverage=[]
    for split,frame in frames.items():
        variants={v:[] for v in ['original','season_frozen','updated']}
        for idx,r in frame.iterrows():
            key=(int(r.season),club(r.home_team_name),club(r.away_team_name))
            candidates=raw.get(key,set())
            if len(candidates)!=1:
                missing.append(dict(split=split,row=idx,reason='lineup missing/ambiguous',name=''));continue
            names=next(iter(candidates));ratings={v:[] for v in ['season_frozen','updated']};pending=[];reason=None
            for position,name in enumerate(names):
                ids=aliases.get(normalize(name),set())
                if len(ids)!=1:
                    reason='identity missing/ambiguous';missing.append(dict(split=split,row=idx,reason=reason,name=name));break
                pid=next(iter(ids))
                if pid not in histories:
                    values=list(db.execute('SELECT day,rating FROM ratings WHERE id=? ORDER BY day',(pid,)))
                    histories[pid]=([x[0] for x in values],[x[1] for x in values])
                dates,values=histories[pid]
                for variant,cutoff in [('season_frozen',cutoffs[r.season,r.league]),('updated',r.fixture_date)]:
                    loc=bisect_left(dates,cutoff)-1
                    if loc<0 or (pd.Timestamp(r.fixture_date)-pd.Timestamp(dates[loc])).days>550:
                        reason='no eligible rating';break
                    day=dates[loc];assert day<cutoff<=r.fixture_date
                    ratings[variant].append(values[loc])
                    pending.append(dict(split=split,row=idx,position=position+1,player=name,player_id=pid,variant=variant,rating=values[loc],rating_date=day,cutoff=cutoff,fixture_date=r.fixture_date))
                if reason:break
            if reason:
                if reason=='no eligible rating':missing.append(dict(split=split,row=idx,reason=reason,name=name))
                continue
            original=r.to_dict();original['source_row']=idx
            variants['original'].append(original)
            for variant,pair in ratings.items():
                assert len(pair)==22
                home,away=pair[:11],pair[11:];record=original.copy()
                record.update(rating_features(home,'HomePlayer'));record.update(rating_features(away,'AwayPlayer'))
                record.update(lineup_features(home,away));record['Home_min_max']=min(home)*max(home);record['Away_min_max']=min(away)*max(away)
                variants[variant].append(record)
            audit.extend(pending)
        for variant,rows in variants.items():
            output=pd.DataFrame(rows)
            # The earlier sample contains one duplicated test fixture; exclude repeats equally.
            output=output.drop_duplicates(['fixture_date','league','home_team_name','away_team_name'])
            output.to_csv(OUT/f'{split}_{variant}.csv',index=False)
            if variant=='updated':
                coverage.extend(dict(split=split,season=int(y),league=l,matches=len(g)) for (y,l),g in output.groupby(['season','league']))
        print(split,len(variants['updated']),'eligible before deduplication',flush=True)
    pd.DataFrame(audit).to_csv(OUT/'rating_join_audit.csv.gz',index=False,compression='gzip')
    pd.DataFrame(missing).to_csv(OUT/'excluded.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
    (OUT/'join_policy.json').write_text(json.dumps(dict(identity='Exact unique normalized name across archive; ambiguous names excluded, never fuzzy guessed',date_rule='Snapshot effective date strictly before fixture calendar date; same-day excluded',baseline='Latest rating strictly before first league fixture of each season; held fixed within season',max_age_days=550,sample='Identical complete 22-player fixtures in all variants; duplicate fixture excluded',caveat='Archive effective dates are not independent publication timestamps. Live Form unavailable. Older pre-archive seasons excluded from paired experiment.'),indent=2))

if __name__=='__main__':main()
