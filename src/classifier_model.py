"""Direct RF outcome inference and verified cached live feature enrichment."""
from pathlib import Path
from datetime import date
import hashlib,re
import numpy as np
import pandas as pd
import joblib
from classifier_features import FORM,RICH,lineup_features,form_history,club,season
from lineup_model import clean
from lineup_sources import team_key

class ClassifierForest:
    def __init__(self,path):
        b=joblib.load(path);self.forest=b['forest'];self.encoder=b['encoder'];self.numeric=b['numeric'];self.categorical=b['categorical']
        self.teams={team_key(n):n for group in self.encoder.categories_ for n in group}
        if not np.array_equal(self.forest.classes_,[0,1,2]):raise ValueError('Classifier outcome order mismatch')
    def predict_frame(self,frame):
        frame=clean(frame)
        for col in self.categorical:frame[col]=frame[col].map(lambda n:self.teams.get(team_key(n),n))
        x=np.column_stack([self.encoder.transform(frame[self.categorical]),frame[self.numeric].to_numpy(float)])
        if not np.isfinite(x).all():raise ValueError('Missing classifier features; refresh inputs')
        return self.forest.predict_proba(x)
    def predict(self,row):return self.predict_frame(pd.DataFrame([row]))[0].tolist()

def live_features(row,match,report,numeric):
    row=dict(row)
    if set(numeric)&set(RICH):
        row.update(lineup_features(*[[p['overall'] for p in match[side+'_lineup']] for side in ['home','away']]))
    if set(numeric)&set(FORM):
        from live_predictions import ROOT,parse_csv,deduplicate
        root=Path(report['_source_path']).parent;games=[]
        fixture_date=date.fromisoformat(match['date']);year=season(fixture_date)
        code=f'{year%100:02d}{(year+1)%100:02d}'
        for source in report.get('sources',[]):
            url=source.get('url','');found=re.search(r'/((?:mmz4281)/\d{4}/'+re.escape(match['league'])+r'\.csv)$',url)
            if not found:continue
            if found[1].split('/')[1]!=code:continue
            candidates=[parent/'cache'/found[1].replace('/','_') for parent in [root,ROOT/'results/live',ROOT/'results/model_strategy/live',ROOT/'results/expanded_interface/live']]
            path=next((p for p in candidates if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==source.get('sha256')),None)
            if path is None:
                raise ValueError('Cached results provenance unavailable for adjusted form; refresh inputs')
            games.extend(parse_csv(path.read_bytes(),results=True))
        if not games:raise ValueError('Current-season results unavailable for adjusted form; refresh inputs')
        query=dict(league=match['league'],date=date.fromisoformat(match['date']),home=match['home'],away=match['away'])
        cutoff=min(query['date'],date.fromisoformat(report.get('as_of',report['generated_at'][:10])))
        games=[g for g in deduplicate(games) if date(year,7,1)<=g['date']<cutoff]
        values=form_history(games,[query])[query['league'],query['date'],club(query['home']),club(query['away'])]
        row.update(values)
    return row
