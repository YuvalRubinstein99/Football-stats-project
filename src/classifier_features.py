"""Pregame market-adjusted form and starting-XI rating distribution features."""
from collections import defaultdict,deque
from itertools import groupby
import numpy as np
from lineup_sources import team_key

# Explicit alternate club spellings; no fuzzy or score-based matching.
ALIASES={'West Brom':'West Bromwich Albion','Blackburn':'Blackburn Rovers','Birmingham':'Birmingham City','Bolton':'Bolton Wanderers','Wigan':'Wigan Athletic','Stoke':'Stoke City','Swansea':'Swansea City','Cardiff':'Cardiff City','QPR':'Queens Park Rangers','Reading':'Reading','Hertha':'Hertha BSC','Nurnberg':'Nürnberg','Bielefeld':'Arminia','Dusseldorf':'Düsseldorf','Sp Gijon':'Sporting Gijón','La Coruna':'La Coruña','Almeria':'Almería','FC Bayern Munich':'Bayern Munich','1. FC Nürnberg':'Nürnberg','DSC Arminia Bielefeld':'Arminia','Chievo Verona':'Chievo','Fortuna Düsseldorf':'Düsseldorf','Granada CF':'Granada','Hertha BSC Berlin':'Hertha BSC','RC Deportivo de La Coruña':'La Coruña','RCD Mallorca':'Mallorca','Real Betis Balompié':'Betis','Real Madrid CF':'Real Madrid','Real Sporting de Gijón':'Sporting Gijón','Real Valladolid':'Valladolid','SpVgg Greuther Fürth':'Greuther Fürth','VfL Bochum':'Bochum','VfL Wolfsburg':'Wolfsburg','UD Almería':'Almería'}
ALIASES.update({'Hannover 96':'Hannover','Real Zaragoza':'Zaragoza','Huddersfield Town':'Huddersfield','Darmstadt 98':'Darmstadt','Ingolstadt 04':'Ingolstadt','1. FC Kaiserslautern':'Kaiserslautern','FC St. Pauli':'St Pauli','Eintracht Braunschweig':'Braunschweig','FC Energie Cottbus':'Cottbus','Karlsruher SC':'Karlsruhe','Reggio Calabria':'Reggina','CD Tenerife':'Tenerife','RC Recreativo':'Recreativo','Hércules Club de Fútbol':'Hercules','Xerez Club Deportivo':'Xerez','CD Numancia':'Numancia'})
KEYS={team_key(k):team_key(v) for k,v in ALIASES.items()}
def club(name):return KEYS.get(team_key(name),team_key(name))

FORM=[side+'_'+feature for side in ['home','away'] for feature in ['adjusted_points_last5','expected_points_last5','adjusted_points_season','adjusted_form_count']]
RICH=[side+'_'+feature for side in ['home','away'] for feature in ['rating_top3_mean','rating_bottom3_mean','rating_median','rating_iqr','starter_rating_count']]+['lineup_top3_advantage','lineup_bottom3_advantage']

def lineup_features(home,away):
    result={}
    for side,ratings in [('home',home),('away',away)]:
        a=np.asarray(ratings,dtype=float)
        if a.shape!=(11,) or np.isinf(a).any() or (a<1).any() or (a>99).any() or np.isfinite(a).sum()<10:raise ValueError('Eleven starter slots with at least ten valid ratings required')
        a=np.sort(a[np.isfinite(a)])
        result[side+'_starter_rating_count']=len(a)
        result.update({side+'_rating_top3_mean':float(a[-3:].mean()),side+'_rating_bottom3_mean':float(a[:3].mean()),side+'_rating_median':float(np.median(a)),side+'_rating_iqr':float(np.quantile(a,.75)-np.quantile(a,.25))})
    result['lineup_top3_advantage']=result['home_rating_top3_mean']-result['away_rating_top3_mean']
    result['lineup_bottom3_advantage']=result['home_rating_bottom3_mean']-result['away_rating_bottom3_mean']
    return result

def season(date):return date.year if date.month>=7 else date.year-1

def form_history(games,queries=()):
    """Queries and matches on a date all read state before that date's results."""
    state=defaultdict(lambda:dict(last=deque(maxlen=5),total=0.,count=0));output={}
    entries=[{**r,'query':False} for r in games]+[{**r,'query':True} for r in queries]
    for date,group in groupby(sorted(entries,key=lambda r:r['date']),key=lambda r:r['date']):
        batch=list(group)
        for r in batch:
            values={}
            for side in ['home','away']:
                s=state[r['league'],r.get('season',season(date)),club(r[side])];last=list(s['last'])
                values.update({side+'_adjusted_points_last5':float(np.mean([v[0] for v in last])) if last else 0.,side+'_expected_points_last5':float(np.mean([v[1] for v in last])) if last else 0.,side+'_adjusted_points_season':s['total']/s['count'] if s['count'] else 0.,side+'_adjusted_form_count':len(last)})
            output[r['league'],date,club(r['home']),club(r['away'])]=values
        for r in batch:
            if r['query'] or any(r.get(o) is None for o in ['B365H','B365D','B365A']):continue
            odds=np.array([r[o] for o in ['B365H','B365D','B365A']],float)
            if not np.isfinite(odds).all() or (odds<=1).any():continue
            p=1/odds;p/=p.sum()
            for side,j in [('home',0),('away',2)]:
                gd=(r['hg']-r['ag'])*(1 if side=='home' else -1);pts=3 if gd>0 else 1 if gd==0 else 0
                expected=3*p[j]+p[1];s=state[r['league'],r.get('season',season(date)),club(r[side])]
                s['last'].append((pts-expected,expected));s['total']+=pts-expected;s['count']+=1
    return output
