"""Collect dated Transfermarkt valuations and strict pre-match starter joins.

Uses the published transfermarkt-datasets archive; does not substitute today's
player values for historical records. Does not alter active models.
"""
import csv,gzip,json,math,statistics
from collections import defaultdict,Counter
from bisect import bisect_left
from datetime import date
from scrape_recent_seasons import ROOT,TM,LEAGUES,valid_gzip
from scrape_older_seasons import fetch,write_csv,atomic_json,stamp

OUT=ROOT/'data/transfermarkt_values'
RAW=ROOT/'data/recent_seasons/raw'

def rows(path):
    with gzip.open(path,'rt',encoding='utf-8',newline='') as f:
        yield from csv.DictReader(f)

def before(history,day):
    i=bisect_left(history[0],day)-1
    return (history[0][i],history[1][i]) if i>=0 else None

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    path=RAW/'player_valuations.csv.gz'
    source=fetch(TM+path.name,path,valid_gzip,1)
    print('Downloaded valuations',flush=True)
    histories=defaultdict(dict);conflicts=set();count=0;years=Counter()
    for r in rows(path):
        pid=r['player_id'];day=r['date'][:10];date.fromisoformat(day)
        value=float(r['market_value_in_eur']);assert math.isfinite(value) and value>=0
        if day in histories[pid] and histories[pid][day]!=value:conflicts.add((pid,day))
        histories[pid][day]=value;count+=1;years[day[:4]]+=1
    for pid,day in conflicts:del histories[pid][day]
    histories={pid:(sorted(h),[h[d] for d in sorted(h)]) for pid,h in histories.items()}
    games={r['game_id']:r for r in rows(RAW/'games.csv.gz') if r['competition_id'] in LEAGUES}
    lineups=defaultdict(dict)
    for r in rows(RAW/'game_lineups.csv.gz'):
        if r['game_id'] in games and r['type']=='starting_lineup':
            lineups[r['game_id'],r['club_id']][r['player_id']]=r['player_name']
    audit=[];features=[];coverage=defaultdict(Counter)
    for gid,g in games.items():
        day=g['date'][:10];season=int(g['season']);league=LEAGUES[g['competition_id']]
        record=dict(game_id=gid,fixture_date=day,season=season,league=league,home_club_id=g['home_club_id'],away_club_id=g['away_club_id'],home_team_name=g['home_club_name'],away_team_name=g['away_club_name'])
        complete=True;coverage[season,league]['matches']+=1
        for side in ['home','away']:
            starters=lineups.get((gid,g[side+'_club_id']),{});values=[];ages=[]
            for pid,name in starters.items():
                hit=before(histories.get(pid,([],[])),day)
                if hit:
                    valued,value=hit;assert valued<day
                    age=(date.fromisoformat(day)-date.fromisoformat(valued)).days
                    values.append(value);ages.append(age)
                else:valued='';value='';age=''
                audit.append(dict(game_id=gid,league=league,season=season,fixture_date=day,side=side,player_id=pid,player_name=name,valuation_date=valued,market_value_eur=value,age_days=age))
            record[side+'_starters']=len(starters);record[side+'_valued_starters']=len(values)
            full=len(starters)==11 and len(values)==11;complete &= full
            # Incomplete lineups remain missing, never silently zero-filled.
            record[side+'_xi_value_eur']=sum(values) if full else ''
            record[side+'_xi_log1p_mean']=statistics.mean(math.log1p(v) for v in values) if full else ''
            record[side+'_valuation_max_age_days']=max(ages) if ages else ''
        record['complete_22']=complete
        if complete:
            record['log_xi_value_ratio']=math.log1p(record['home_xi_value_eur'])-math.log1p(record['away_xi_value_eur'])
            coverage[season,league]['complete_22']+=1
        else:record['log_xi_value_ratio']=''
        features.append(record)
    write_csv(OUT/'match_features.csv',features)
    with gzip.open(OUT/'starter_valuation_audit.csv.gz','wt',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(audit[0]));w.writeheader();w.writerows(audit)
    write_csv(OUT/'coverage.csv',[dict(season=y,league=l,matches=c['matches'],complete_22=c['complete_22']) for (y,l),c in sorted(coverage.items())])
    status=dict(finished_at=stamp(),source=source,valuation_rows=count,players=len(histories),valuation_year_counts=dict(sorted(years.items())),first_valuation=min(d for h in histories.values() for d in h[0]),last_valuation=max(d for h in histories.values() for d in h[0]),conflicting_player_dates_excluded=len(conflicts),league_matches=len(features),complete_22_matches=sum(r['complete_22'] for r in features),date_rule='Latest historical valuation strictly before fixture calendar date; no same-day or future values. Ages retained; no interpolation.',limitations='Archive historical effective dates, not independently timestamped publication receipts. Actual starters known near kickoff; do not imply early pre-lineup availability. Dataset has survivor/coverage limitations. Values include age and contract effects, not just playing ability. No retraining performed.')
    atomic_json(OUT/'status.json',status);print(json.dumps(status,indent=2),flush=True)

if __name__=='__main__':main()
