"""Recover verified XI ratings and real fixture dates without changing sample rows."""
import json,hashlib
from collections import defaultdict
from select_expanded_features import ROOT,pd,np,season_years
from model_filters import league_rows
from classifier_features import club,lineup_features,form_history
from live_predictions import download,parse_csv
OUT=ROOT/'results/classifier'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(ROOT/'data/expanded_2008/train.csv');test=pd.read_csv(ROOT/'data/expanded_2008/test.csv')
    old=pd.read_csv(ROOT/'data/expanded_2008/added_row_provenance.csv');audit=pd.read_csv(ROOT/'data/older_seasons/player_rating_audit.csv')
    old_ratings={(int(mid),side):g.sort_values('position').rating.to_numpy() for (mid,side),g in audit.groupby(['match_id','side'])}
    arrays={};inputs=[ROOT/'data/expanded_2008/train.csv',ROOT/'data/expanded_2008/test.csv',ROOT/'data/expanded_2008/added_row_provenance.csv',ROOT/'data/older_seasons/player_rating_audit.csv']
    for year in range(2014,2023):
        path=ROOT/f'data/season_data/season{year%100:02d}{(year+1)%100:02d}_proccessed.csv';inputs.append(path)
        data=pd.read_csv(path)
        for _,r in data.iterrows():
            key=(year,club(r.home_team_name),club(r.away_team_name))
            value=tuple(np.array([r[f'{side}Player{i}_Overall'] for i in range(1,12)],float) for side in ['Home','Away'])
            arrays.setdefault(key,[]).append(value)
    games=[];sources=[]
    for year in range(2008,2023):
        for league in ['E0','SP1','D1','I1']:
            code=f'{year%100:02d}{(year+1)%100:02d}';oldcache=ROOT/f'data/older_seasons/raw/{league}_{code}.csv'
            if oldcache.exists():rows=parse_csv(oldcache.read_bytes(),results=True);inputs.append(oldcache)
            else:
                relative=f'mmz4281/{code}/{league}.csv';cache=OUT/'cache';exists=(cache/relative.replace('/','_')).exists()
                print(f'Reading fixture dates: {league} {year}',flush=True)
                rows,source=download(relative,cache,offline=exists,results=True);sources.append(source);inputs.append(cache/relative.replace('/','_'))
            games.extend({**r,'season':year} for r in rows)
    dates=defaultdict(list)
    for g in games:dates[g['season'],g['league'],club(g['home']),club(g['away'])].append(g)
    forms=form_history(games);mapping=league_rows();errors=[];warnings=[];outputs={}
    for split,frame,years in [('train',train,season_years(train)),('test',test,np.full(len(test),2022))]:
        codes=mapping['Train (in-sample)' if split=='train' else 'Test'];extra=[]
        for i,r in frame.iterrows():
            year=int(years[i]);key=(year,codes[i],club(r.home_team_name),club(r.away_team_name))
            found=dates[key]
            if len(found)>1:found=[g for g in found if (g['hg'],g['ag'])==(r.home_score,r.away_score)]
            if len(found)!=1:errors.append(dict(split=split,row=i,reason='date',key=str(key),found=len(found)));continue
            g=found[0]
            if (g['hg'],g['ag'])!=(r.home_score,r.away_score):warnings.append(dict(split=split,row=i,reason='Score differs across sources; original target retained, historical form uses dated source result',key=str(key),original=[int(r.home_score),int(r.away_score)],dated_source=[g['hg'],g['ag']]))
            if split=='train' and i<len(old):pair=[old_ratings[int(old.iloc[i].match_id),side] for side in ['home','away']]
            else:
                pairs=arrays.get((year,club(r.home_team_name),club(r.away_team_name)),[])
                pairs=[p for p in pairs if all(np.isclose(np.nanmean(a),r[prefix+'_Overall_mean'],atol=1e-6) and np.isclose(np.nanstd(a,ddof=1),r[prefix+'_Overall_sd'],atol=1e-6) for a,prefix in zip(p,['HomePlayer','AwayPlayer']))]
                pair=pairs[0] if len(pairs)==1 else None
            if pair is None:errors.append(dict(split=split,row=i,reason='ratings missing',key=str(key)));continue
            if any(not np.isclose(np.nanmean(a),r[p+'_Overall_mean'],atol=1e-6) for a,p in zip(pair,['HomePlayer','AwayPlayer'])):
                errors.append(dict(split=split,row=i,reason='rating mean differs',key=str(key)));continue
            extra.append(dict(**lineup_features(*pair),**forms[codes[i],g['date'],club(r.home_team_name),club(r.away_team_name)],fixture_date=g['date'].isoformat(),league=codes[i],season=year))
        if len(extra)==len(frame):outputs[split]=pd.concat([frame,pd.DataFrame(extra)],axis=1)
    if errors:
        pd.DataFrame(errors).to_csv(OUT/'alignment_errors.csv',index=False)
        raise ValueError(f'{len(errors)} fixture/rating alignment failures; see alignment_errors.csv')
    pd.DataFrame(columns=['split','row','reason','key','found']).to_csv(OUT/'alignment_errors.csv',index=False)
    for split,frame in outputs.items():frame.to_csv(OUT/(split+'.csv'),index=False)
    (OUT/'feature_audit.json').write_text(json.dumps(dict(train=len(train),test=len(test),new_features=20,warnings=warnings,incomplete_starter_rows=int(((outputs['train'].home_starter_rating_count<11)|(outputs['train'].away_starter_rating_count<11)).sum()),duplicate_test_fixtures=int(outputs['test'].duplicated(['fixture_date','league','home_team_name','away_team_name']).sum()),chronology='Strictly earlier actual fixture dates; same-day results excluded; source season labels preserve COVID seasons',lineup='Top/bottom three means, median, IQR, rating coverage and home-away advantages; three historical rows contain ten rated starters on one side. No positional/absence claims.',input_hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},sources=sources),indent=2))
    print('Verified fixture alignment and XI rating aggregates; source discrepancies recorded.',flush=True)

if __name__=='__main__':main()
