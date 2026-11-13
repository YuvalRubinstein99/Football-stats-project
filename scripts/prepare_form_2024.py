"""Build 2024-25 inputs offline, using only dated pre-season EA snapshots."""
import csv
import hashlib
import json
import re
from collections import defaultdict
from itertools import groupby
from select_expanded_features import ROOT, pd, np
from live_predictions import parse_csv
from classifier_features import club
from lineup_sources import normalize
from match_form import build_form, fixture_key
from prepare_team_attributes import PlayerIndex

DEST=ROOT/'results/match_form/season_2024'
SOURCE=ROOT/'data/recent_seasons'
LEAGUES={'GB1':'E0','L1':'D1','ES1':'SP1','IT1':'I1'}
# Reviewed club spellings only, never inferred from scores or fixture outcomes.
ALIASES={**{x+' FC':x for x in ['Arsenal','Brentford','Chelsea','Everton','Fulham','Liverpool','Southampton']},
 '1. Fußballclub Heidenheim 1846':'Heidenheim','CD Leganés':'Leganes',
 'Celta de Vigo':'Celta','Girona FC':'Girona','RCD Espanyol Barcelona':'Espanol',
 'Real Valladolid CF':'Valladolid','UD Las Palmas':'Las Palmas','AC Monza':'Monza',
 'ACF Fiorentina':'Fiorentina','Associazione Sportiva Roma':'Roma','Atalanta BC':'Atalanta',
 'Bologna Football Club 1909':'Bologna','Cagliari Calcio':'Cagliari','Como 1907':'Como',
 'FC Empoli':'Empoli','Genoa CFC':'Genoa','Juventus FC':'Juventus','Parma Calcio 1913':'Parma',
 'Società Sportiva Lazio S.p.A.':'Lazio','Torino FC':'Torino','US Lecce':'Lecce',
 'Udinese Calcio':'Udinese','Venezia FC':'Venezia'}


def team(value):
    return club(ALIASES.get(value,value))


def legacy_form(games):
    state=defaultdict(list)
    output={}
    for day,batch in groupby(sorted(games,key=lambda g:g['date']),key=lambda g:g['date']):
        batch=list(batch)
        for g in batch:
            row={}
            for side in ['home','away']:
                past=state[g['league'],club(g[side])]
                gd=[a for a,b in past];points=[b for a,b in past]
                row.update({side+'_GD_prior':sum(gd),side+'_Points_prior':sum(points),
                            side+'_GD_form':sum(gd[-5:]),side+'_Points_form':sum(points[-5:])})
            output[fixture_key(g)]=row
        for g in batch:
            for side,gd in [('home',g['hg']-g['ag']),('away',g['ag']-g['hg'])]:
                state[g['league'],club(g[side])].append((gd,3 if gd>0 else 1 if gd==0 else 0))
    return output


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    inputs=[SOURCE/'games.csv',SOURCE/'lineups.csv',SOURCE/'ratings_fc23_fc24.csv']
    games=[]
    for league in LEAGUES.values():
        path=SOURCE/f'raw/2425_{league}.csv';inputs.append(path)
        raw=list(csv.DictReader(path.open(encoding='utf-8-sig')))
        parsed=parse_csv(path.read_bytes(),results=True)
        assert len(raw)==len(parsed)==(306 if league=='D1' else 380)
        for g,r in zip(parsed,raw):
            assert g['league']==league
            g.update(season=2024,hst=float(r['HST']) if r['HST'] else np.nan,
                     ast=float(r['AST']) if r['AST'] else np.nan)
            games.append(g)
    games.sort(key=lambda g:(g['date'],g['league'],g['home']))
    assert len({fixture_key(g) for g in games})==len(games)
    forms=build_form(games,games);old=legacy_form(games)
    tm=pd.read_csv(SOURCE/'games.csv',dtype=str)
    tm=tm[(tm.season=='2024')&tm.competition_id.isin(LEAGUES)]
    lookup=defaultdict(list)
    for r in tm.to_dict('records'):
        lookup[LEAGUES[r['competition_id']],pd.Timestamp(r['date']).date(),
               team(r['home_club_name']),team(r['away_club_name'])].append(r)
    lineup=pd.read_csv(SOURCE/'lineups.csv',dtype=str)
    lineup=lineup[(lineup.type=='starting_lineup')&lineup.game_id.isin(tm.game_id)]
    starters={key:g.to_dict('records') for key,g in lineup.groupby(['game_id','club_id'])}
    ratings=pd.read_csv(SOURCE/'ratings_fc23_fc24.csv')
    first=min(g['date'] for g in games).isoformat()
    ratings=ratings[ratings.update_as_of<first].copy()
    assert len(ratings) and ratings.update_as_of.notna().all()
    latest=ratings.sort_values(['update_as_of','fifa_version']).drop_duplicates('player_id',keep='last')
    index=PlayerIndex(latest)
    by_id={int(r['player_id']):i for i,r in enumerate(index.records)}
    token_index=defaultdict(set);name_tokens=defaultdict(list)
    # Older spellings may identify the same EA ID; only pre-season snapshots enter.
    for r in ratings.to_dict('records'):
        for name in [r['short_name'],r['long_name']]:
            if pd.notna(name):
                i=by_id[int(r['player_id'])];key=normalize(str(name))
                index.names[key].add(i)
                tokens=set(key.split());name_tokens[i].append(tokens)
                for token in tokens:token_index[token].add(i)
    rows=[];excluded=[];audit=[];all_rows=[]
    for source_row,g in enumerate(games):
        key=fixture_key(g);matches=lookup.get(key,[])
        reason=''
        if len(matches)!=1:reason='missing_or_ambiguous_fixture_identity'
        elif (int(matches[0]['home_club_goals']),int(matches[0]['away_club_goals']))!=(g['hg'],g['ag']):
            reason='score_disagreement_between_sources'
        r=dict(source_row=source_row,fixture_date=g['date'].isoformat(),season=2024,league=g['league'],
               home_team_name=g['home'],away_team_name=g['away'],home_score=g['hg'],away_score=g['ag'],
               **{k:g[k] for k in ['B365H','B365D','B365A']},**old[key],**forms[key])
        all_rows.append(r.copy())
        if not reason:
            match=matches[0];r['game_id']=match['game_id']
            r['Matchweek']=int(re.match(r'\d+',match['round'])[0])
            for side,prefix in [('home','Home'),('away','Away')]:
                people=starters.get((match['game_id'],match[side+'_club_id']),[])
                if len(people)!=11 or len({p['player_id'] for p in people})!=11:
                    reason='incomplete_or_duplicate_starting_lineup';break
                matched=[]
                for player in people:
                    i,method=index.match(player['player_name'],g[side])
                    # Transfers can invalidate the old club field. Require all supplied
                    # full tokens and one unique EA identity across dated catalogues.
                    tokens=set(normalize(player['player_name']).split())
                    if i is None and method=='ambiguous_or_unmatched_name' and len(tokens)>=2 and all(len(t)>1 for t in tokens):
                        candidates=set.intersection(*(token_index[t] for t in tokens))
                        candidates={c for c in candidates if any(tokens<=n for n in name_tokens[c])}
                        if len(candidates)==1:i,method=next(iter(candidates)),'unique_global_full_name_tokens'
                    found=index.records[i] if i is not None else None
                    audit.append(dict(source_row=source_row,date=r['fixture_date'],league=g['league'],
                        side=side,player=player['player_name'],tm_player_id=player['player_id'],
                        ea_id=found['player_id'] if found else '',method=method,
                        rating=found['overall'] if found else '',rating_date=found['update_as_of'] if found else ''))
                    if found:
                        assert found['update_as_of']<first<=r['fixture_date']
                        matched.append(found)
                if len({m['player_id'] for m in matched})!=len(matched):
                    reason='duplicate_matched_ea_identity';break
                r[side+'_rated_starters']=len(matched)
                if len(matched)<10:
                    reason='fewer_than_10_dated_starter_ratings';break
                values=[float(m['overall']) for m in matched]
                assert all(1<=v<=99 for v in values)
                r[prefix+'Player_Overall_mean']=float(np.mean(values))
        if any(g[k] is None for k in ['B365H','B365D','B365A']):reason='missing_valid_odds'
        if reason:excluded.append(dict(source_row=source_row,date=r['fixture_date'],league=g['league'],
                                       home=g['home'],away=g['away'],reason=reason))
        else:rows.append(r)
    frame=pd.DataFrame(rows)
    assert len(frame)>0
    for side in ['home','away']:
        dates=pd.to_datetime(frame[side+'_recent_latest_date'])
        assert (dates.isna()|(dates<pd.to_datetime(frame.fixture_date))).all()
    frame.to_csv(DEST/'matches.csv',index=False)
    pd.DataFrame(all_rows).to_csv(DEST/'all_matches.csv',index=False)
    pd.DataFrame(audit).to_csv(DEST/'rating_audit.csv',index=False)
    pd.DataFrame(excluded).to_csv(DEST/'excluded.csv',index=False)
    coverage=pd.DataFrame([dict(league=l,history_matches=sum(g['league']==l for g in games),
                              tested_matches=int((frame.league==l).sum())) for l in LEAGUES.values()])
    coverage.to_csv(DEST/'coverage.csv',index=False)
    receipt=dict(history_matches=len(games),tested_matches=len(frame),rating_dates=sorted(ratings.update_as_of.unique()),
        min_rated_starters_per_side=10,first_fixture=first,last_fixture=max(g['date'] for g in games).isoformat(),
        exclusions=dict(pd.Series([r['reason'] for r in excluded]).value_counts().items()),
        policy='Frozen pre-season dated EA snapshots; unique name/club matches; at least ten of eleven '
               'starters rated per side, like the existing historical data minimum. All 1446 results feed '
               'within-season form before eligibility filtering. Same-day results excluded.',
        limitations='FC24/FC23 ratings are stale, not contemporaneous FC25 ratings. Supplied snapshot dates '
                    'are used; undated FC25 archive excluded. Actual historical lineups, not forecast lineups. '
                    'Eligible subset can differ systematically from excluded matches; not a full-season sample.',
        input_hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [*inputs,ROOT/'scripts/prepare_form_2024.py',ROOT/'src/match_form.py']})
    (DEST/'data_audit.json').write_text(json.dumps(receipt,indent=2,default=int),encoding='utf-8')
    print(coverage.to_string(index=False));print(json.dumps({k:v for k,v in receipt.items() if k!='input_hashes'},indent=2,default=int))


if __name__=='__main__':main()
