"""Independently reconcile the 2024 inputs against cached results and ratings."""
import json
from select_expanded_features import pd, np
from prepare_form_2024 import DEST, SOURCE, LEAGUES
from live_predictions import parse_csv
from lineup_model import form_features


def main():
    all_rows=pd.read_csv(DEST/'all_matches.csv')
    eligible=pd.read_csv(DEST/'matches.csv')
    audit=pd.read_csv(DEST/'rating_audit.csv')
    games=[]
    for league in LEAGUES.values():
        games.extend(parse_csv((SOURCE/f'raw/2425_{league}.csv').read_bytes(),results=True))
    assert len(all_rows)==len(games)==1446
    # This existing inference helper independently filters dates and sums results.
    lookup={(g['league'],g['date'].isoformat(),g['home'],g['away']):g for g in games}
    for row in all_rows.to_dict('records'):
        game=lookup[row['league'],row['fixture_date'],row['home_team_name'],row['away_team_name']]
        assert (row['home_score'],row['away_score'])==(game['hg'],game['ag'])
        expected,counts=form_features(games,game,game['date'])
        for key,value in expected.items():assert row[key]==value,(key,row['source_row'])
        for side in ['home','away']:
            assert row[side+'_recent_matches_10']==min(counts[side],10)
    # Direct raw-row averages check attack, defence and shots across home/away venues.
    for league in LEAGUES.values():
        raw=pd.read_csv(SOURCE/f'raw/2425_{league}.csv')
        raw['day']=pd.to_datetime(raw.Date,dayfirst=True).dt.strftime('%Y-%m-%d')
        for row in all_rows[all_rows.league==league].to_dict('records'):
            for side in ['home','away']:
                name=row[side+'_team_name']
                past=raw[(raw.day<row['fixture_date'])&((raw.HomeTeam==name)|(raw.AwayTeam==name))].sort_values('day')
                at_home=(past.HomeTeam==name).to_numpy()
                for metric,h,a in [('goals_for','FTHG','FTAG'),('goals_against','FTAG','FTHG'),
                                   ('sot_for','HST','AST'),('sot_against','AST','HST')]:
                    values=np.where(at_home,past[h],past[a]).astype(float)
                    for window in [3,5,10]:
                        part=values[-window:]
                        expected=np.nanmean(part) if len(part) else np.nan
                        np.testing.assert_allclose(row[f'{side}_recent_{metric}_{window}'],expected,equal_nan=True)
                    weights=2.**(-np.arange(len(values)-1,-1,-1)/5.)
                    valid=np.isfinite(values)
                    expected=np.average(values[valid],weights=weights[valid]) if valid.any() else np.nan
                    np.testing.assert_allclose(row[f'{side}_recent_{metric}_ewm5'],expected,equal_nan=True)
    used=audit[audit.source_row.isin(eligible.source_row)]
    for source_row,group in used.groupby('source_row'):
        assert len(group)==22 and group.tm_player_id.nunique()==22
        matched=group[group.ea_id.notna()]
        assert matched.ea_id.nunique()==len(matched),source_row
        assert (matched.rating_date<'2024-08-15').all()
        row=eligible[eligible.source_row==source_row].iloc[0]
        for side,prefix in [('home','Home'),('away','Away')]:
            ratings=matched[matched.side==side].rating
            assert 10<=len(ratings)<=11
            np.testing.assert_allclose(ratings.mean(),row[prefix+'Player_Overall_mean'])
    result=dict(fixtures=1446,eligible=len(eligible),legacy_form='passed',
                attack_defence_shots_all_windows='passed',dated_distinct_player_ratings='passed')
    (DEST/'input_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
