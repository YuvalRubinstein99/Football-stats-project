"""Resolve historical leagues from original source fixtures, not name guesses."""
import csv
from pathlib import Path
from functools import lru_cache
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]

@lru_cache(maxsize=1)
def row_leagues():
    mapping=defaultdict(set);teams=defaultdict(set)
    sources={'bl_data':'Bundesliga','laliga_data':'La Liga','seriea_data':'Serie A','epl_data(fully_proccessed)':'Premier League'}
    for folder,league in sources.items():
        pattern='*.csv'
        for path in (ROOT/'data'/folder).glob(pattern):
            with path.open(encoding='utf-8-sig',newline='') as f:
                for r in csv.DictReader(f):
                    if 'home_team_name' not in r or 'away_team_name' not in r:continue
                    mapping[r['home_team_name'],r['away_team_name']].add(league)
                    teams[r['home_team_name']].add(league);teams[r['away_team_name']].add(league)
    # Additional archived fixtures retain league codes and cover promoted teams.
    archive=ROOT/'data/older_seasons/lineup_features.csv'
    names={'E0':'Premier League','SP1':'La Liga','D1':'Bundesliga','I1':'Serie A'}
    if archive.exists():
        with archive.open(encoding='utf-8',newline='') as f:
            for r in csv.DictReader(f):
                if r['league'] in names:
                    for col in ('home_team_name','away_team_name'):teams[r[col]].add(names[r['league']])
    aliases={'Mallorca':'RCD Mallorca','Cádiz':'Cadiz','Almería':'UD Almería'}
    for alias,source in aliases.items():teams[alias].update(teams[source])
    output={}
    for split,file in [('Train (OOB)','train.csv'),('Test','test.csv')]:
        with (ROOT/'data'/file).open(encoding='utf-8-sig',newline='') as f:
            for i,r in enumerate(csv.DictReader(f)):
                leagues=mapping[r['home_team_name'],r['away_team_name']]
                if not leagues:leagues=teams[r['home_team_name']]|teams[r['away_team_name']]
                if len(leagues)!=1:raise ValueError(f'Unresolved league for {split} row {i}')
                output[split,i]=next(iter(leagues))
    return output
