"""Check source coverage without promoting unjoined data to model inputs."""
import csv,json
from collections import Counter,defaultdict
from scrape_recent_seasons import OUT,LEAGUES,write_csv,atomic_json

def read(name):
    with (OUT/name).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))

def main():
    results=read('results_odds.csv');games=read('games.csv');lineups=read('lineups.csv')
    assert len({(r['season'],r['league'],r['fixture_date'],r['home_team_name'],r['away_team_name']) for r in results})==len(results)
    starters=defaultdict(list)
    for r in lineups:
        if r['type']=='starting_lineup':starters[r['game_id'],r['club_id']].append(r['player_id'])
    audits=[]
    for year in ['2023','2024','2025']:
        for competition,league in LEAGUES.items():
            matches=[r for r in results if r['season']==year and r['league']==league]
            archive=[r for r in games if r['season']==year and r['competition_id']==competition]
            full=0
            for r in archive:
                home=starters[r['game_id'],r['home_club_id']];away=starters[r['game_id'],r['away_club_id']]
                full+=len(home)==11 and len(away)==11 and len(set(home+away))==22
            audits.append(dict(season=year,league=league,results_matches=len(matches),bet365_matches=sum(r['valid_bet365']=='True' for r in matches),lineup_source_matches=len(archive),complete_unique_starter_sets=full,model_ready_matches=0))
    write_csv(OUT/'collection_audit.csv',audits)
    status=json.loads((OUT/'status.json').read_text())
    ratings=read('ratings_fc23_fc24.csv')
    status['ratings']['dates']=sorted(set(r.get('fifa_update_date') or r.get('update_as_of','') for r in ratings))
    status['note']='Sources collected separately; not yet joined or promoted to models. See collection_audit.csv and recent_ratings_audit.json. Ratings must predate each match; verify identities and snapshot publication dates.'
    status['recent_ratings_audit']='recent_ratings_audit.json'
    atomic_json(OUT/'status.json',status)
    print(json.dumps(audits,indent=2))

if __name__=='__main__':main()
