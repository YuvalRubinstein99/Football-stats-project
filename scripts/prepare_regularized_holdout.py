"""Prepare the reserved 2025-26 sample only after the model and thresholds are frozen."""
import csv
import hashlib
import json
import re
from collections import defaultdict
from select_expanded_features import ROOT,np,pd
from train_regularized_home_win import OUT
from prepare_form_2024 import team,legacy_form
from live_predictions import parse_csv
from match_form import fixture_key
from lineup_sources import normalize
from prepare_team_attributes import PlayerIndex
from positional_features import aggregate_units

SOURCE=ROOT/'data/recent_seasons'
DEST=OUT/'holdout_2025'


def main():
    frozen=json.loads((OUT/'frozen_selection.json').read_text())
    assert hashlib.sha256((OUT/'frozen_model.joblib').read_bytes()).hexdigest()==frozen['model_sha256']
    assert hashlib.sha256((OUT/'plan.json').read_bytes()).hexdigest()==frozen['plan_sha256']
    DEST.mkdir(exist_ok=True)
    policy=json.loads((OUT/'plan.json').read_text())['holdout']
    assert policy['season']==2025 and policy['minimum_rated_starters']==10
    paths=[SOURCE/'games.csv',SOURCE/'lineups.csv',SOURCE/'ratings_fc23_fc24.csv']
    games=[]
    for league in policy['leagues']:
        path=SOURCE/f'raw/2526_{league}.csv';paths.append(path)
        parsed=parse_csv(path.read_bytes(),results=True)
        assert len(parsed)==(306 if league=='D1' else 380)
        for g in parsed:g.update(season=2025)
        games.extend(parsed)
    games.sort(key=lambda g:(g['date'],g['league'],g['home']))
    assert len({fixture_key(g) for g in games})==len(games)
    first=min(g['date'] for g in games).isoformat()
    assert frozen['training_last_date']<first
    forms=legacy_form(games)
    leagues={'L1':'D1','IT1':'I1','ES1':'SP1'}
    tm=pd.read_csv(SOURCE/'games.csv',dtype=str)
    tm=tm[(tm.season=='2025')&tm.competition_id.isin(leagues)]
    lookup=defaultdict(list)
    for row in tm.to_dict('records'):
        lookup[leagues[row['competition_id']],pd.Timestamp(row['date']).date(),
               team(row['home_club_name']),team(row['away_club_name'])].append(row)
    lineup=pd.read_csv(SOURCE/'lineups.csv',dtype=str)
    lineup=lineup[(lineup.type=='starting_lineup')&lineup.game_id.isin(tm.game_id)]
    starters={key:g.to_dict('records') for key,g in lineup.groupby(['game_id','club_id'])}
    ratings=pd.read_csv(SOURCE/'ratings_fc23_fc24.csv')
    ratings=ratings[ratings.update_as_of<first].copy()
    assert len(ratings) and ratings.update_as_of.notna().all()
    latest=ratings.sort_values(['update_as_of','fifa_version']).drop_duplicates('player_id',keep='last')
    index=PlayerIndex(latest)
    by_id={int(row['player_id']):i for i,row in enumerate(index.records)}
    tokens_index=defaultdict(set);names=defaultdict(list)
    for row in ratings.to_dict('records'):
        for name in [row['short_name'],row['long_name']]:
            if pd.notna(name):
                i=by_id[int(row['player_id'])];key=normalize(str(name));index.names[key].add(i)
                tokens=set(key.split());names[i].append(tokens)
                for token in tokens:tokens_index[token].add(i)
    rows=[];all_rows=[];excluded=[];audit=[]
    for source_row,g in enumerate(games):
        key=fixture_key(g);matches=lookup.get(key,[])
        row=dict(source_row=source_row,season=2025,fixture_date=g['date'].isoformat(),league=g['league'],
                 home_team_name=g['home'],away_team_name=g['away'],home_score=g['hg'],away_score=g['ag'],
                 **{k:g[k] for k in ['B365H','B365D','B365A']},**forms[key])
        all_rows.append(row.copy())
        reason=''
        if len(matches)!=1:reason='missing_or_ambiguous_fixture'
        elif (int(matches[0]['home_club_goals']),int(matches[0]['away_club_goals']))!=(g['hg'],g['ag']):
            reason='source_score_disagreement'
        if not reason:
            match=matches[0]
            week=re.match(r'\d+',str(match['round']))
            if week is None:reason='missing_matchweek'
            else:row['Matchweek']=int(week[0])
            for side,prefix in [('home','Home'),('away','Away')]:
                people=starters.get((match['game_id'],match[side+'_club_id']),[])
                if len(people)!=11 or len({p['player_id'] for p in people})!=11:
                    reason='incomplete_or_duplicate_lineup';break
                matched=[]
                for player in people:
                    i,method=index.match(player['player_name'],g[side])
                    tokens=set(normalize(player['player_name']).split())
                    if i is None and method=='ambiguous_or_unmatched_name' and len(tokens)>=2 and all(len(t)>1 for t in tokens):
                        candidates=set.intersection(*(tokens_index[t] for t in tokens))
                        candidates={c for c in candidates if any(tokens<=n for n in names[c])}
                        if len(candidates)==1:i,method=next(iter(candidates)),'unique_global_full_name_tokens'
                    found=index.records[i] if i is not None else None
                    audit.append(dict(source_row=source_row,side=side,player=player['player_name'],method=method,
                        ea_id=found['player_id'] if found else '',rating_date=found['update_as_of'] if found else ''))
                    if found:
                        assert found['update_as_of']<first<=row['fixture_date']
                        matched.append(found)
                if len({p['player_id'] for p in matched})!=len(matched):
                    reason='duplicate_matched_identity';break
                if len(matched)<10:reason='fewer_than_10_dated_ratings';break
                row[prefix+'Player_Overall_mean']=float(np.mean([float(p['overall']) for p in matched]))
                row.update(aggregate_units(matched,side))
        if any(g[k] is None or not np.isfinite(g[k]) or g[k]<=1 for k in ['B365H','B365D','B365A']):
            reason='invalid_odds'
        if reason:excluded.append(dict(source_row=source_row,league=g['league'],reason=reason))
        else:rows.append(row)
    frame=pd.DataFrame(rows)
    if frame.empty:raise ValueError('No eligible holdout matches')
    frame.to_csv(DEST/'matches.csv',index=False)
    pd.DataFrame(all_rows).to_csv(DEST/'all_matches.csv',index=False)
    pd.DataFrame(excluded).to_csv(DEST/'excluded.csv',index=False)
    pd.DataFrame(audit).to_csv(DEST/'rating_audit.csv',index=False)
    coverage=pd.DataFrame([dict(league=league,all_matches=sum(g['league']==league for g in games),
                                eligible=int(frame.league.eq(league).sum())) for league in policy['leagues']])
    coverage.to_csv(DEST/'coverage.csv',index=False)
    receipt=dict(season=2025,matches=len(games),eligible=len(frame),rating_dates=sorted(ratings.update_as_of.unique()),
        first_fixture=first,last_fixture=max(g['date'] for g in games).isoformat(),
        frozen_model_sha256=frozen['model_sha256'],policy=policy,
        exclusions={str(k):int(v) for k,v in pd.Series([r['reason'] for r in excluded]).value_counts().items()},
        limitation='2025-26 held out from this experiment. Earlier training seasons had prior researcher exposure. '
                   'FC23/FC24 ratings are stale; actual starting lineups and an eligible subset are used. '
                   'Odds capture time relative to lineup announcement is not verified.',
        input_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        output_sha256=hashlib.sha256((DEST/'matches.csv').read_bytes()).hexdigest())
    (DEST/'data_audit.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(coverage.to_string(index=False));print(receipt['exclusions'])


if __name__=='__main__':main()
