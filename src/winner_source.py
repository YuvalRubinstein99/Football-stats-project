"""Winner public pre-match football feed; no login or betting actions."""
import json,re,threading,uuid
from datetime import datetime,timezone
from pathlib import Path
from urllib.request import Request,urlopen
from zoneinfo import ZoneInfo
from lineup_sources import team_key

URL='https://api.winner.co.il/v2/publicapi/GetCMobileLine'
ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'results/winner'
LEAGUES={'פרמייר ליג':'E0','ספרדית ראשונה':'SP1','גרמנית ראשונה':'D1','איטלקית ראשונה':'I1'}
TEAMS=dict(line.split('|') for line in '''אברטון|Everton
איפסוויץ'|Ipswich
אסטון וילה|Aston Villa
ארסנל|Arsenal
בורנמות'|Bournemouth
ברייטון|Brighton
ברנטפורד|Brentford
האל|Hull
טוטנהאם|Tottenham
ליברפול|Liverpool
לידס|Leeds
מנצ'סטר יונייטד|Man United
מנצ'סטר סיטי|Man City
נוטינגהאם פורסט|Nott'm Forest
ניוקאסל|Newcastle
סנדרלנד|Sunderland
פולהאם|Fulham
צ'לסי|Chelsea
קובנטרי|Coventry
קריסטל פאלאס|Crystal Palace
אוסאסונה|Osasuna
אלאבס|Alaves
אלצ'ה|Elche
אספניול|Espanol
אתלטיק בילבאו|Ath Bilbao
אתלטיקו מדריד|Ath Madrid
ברצלונה|Barcelona
דפורטיבו לה קורוניה|La Coruna
ויאריאל|Villarreal
ולנסיה|Valencia
חטאפה|Getafe
לבאנטה|Levante
מלאגה|Malaga
סביליה|Sevilla
סלטה ויגו|Celta
ראיו וייקאנו|Vallecano
ראסינג דה סנטנדר|Santander
ריאל בטיס|Betis
ריאל מדריד|Real Madrid
ריאל סוסיאדד|Sociedad
אוגסבורג|Augsburg
אוניון ברלין|Union Berlin
איינטרכט פרנקפורט|Ein Frankfurt
אלברסברג|Elversberg
באייר לברקוזן|Leverkusen
בורוסיה דורטמונד|Dortmund
בורוסיה מינשנגלדבאך|M'gladbach
ביירן מינכן|Bayern Munich
הופנהיים|Hoffenheim
המבורג|Hamburg
ורדר ברמן|Werder Bremen
לייפציג|RB Leipzig
מיינץ|Mainz
פ.צ. קלן|FC Koln
פאדרבורן|Paderborn
פרייבורג|Freiburg
שאלקה 04|Schalke 04
שטוטגרט|Stuttgart
אודינזה|Udinese
אטאלנטה|Atalanta
אינטר|Inter
בולוניה|Bologna
גנואה|Genoa
ונציה|Venezia
טורינו|Torino
יובנטוס|Juventus
לאציו|Lazio
לצ'ה|Lecce
מונזה|Monza
מילאן|Milan
נאפולי|Napoli
ססואולו|Sassuolo
פארמה|Parma
פיורנטינה|Fiorentina
פרוסינונה|Frosinone
קומו|Como
קליארי|Cagliari
רומא|Roma'''.splitlines())
_lock=threading.Lock();_cache=None;_device=uuid.uuid4().hex

def clean(s):return re.sub('[\u202a-\u202e\u2066-\u2069]','',str(s)).strip()

def parse(payload,observed):
    from manual_strategies import number
    if not isinstance(payload,dict) or not isinstance(payload.get('markets'),list):raise ValueError('Winner feed schema changed: markets missing')
    rows=[];rejected=[]
    for m in payload['markets']:
        if m.get('sId')!=240 or clean(m.get('mp'))!='1X2 - תוצאת סיום (ללא הארכות)' or m.get('league') not in LEAGUES:continue
        try:
            if m.get('isLive') is not False or m.get('score') or m.get('periodLive'):continue
            outcomes=m['outcomes']
            if len(outcomes)!=3 or clean(outcomes[1]['desc']).upper()!='X':raise ValueError('Unrecognized 1/X/2 outcome order')
            home,away=clean(outcomes[0]['desc']),clean(outcomes[2]['desc'])
            if clean(m['desc'])!=home+' - '+away:raise ValueError('Fixture and outcome names disagree')
            if any(o.get('spread') not in ('',None) for o in outcomes):raise ValueError('Handicap outcomes excluded')
            date=datetime.strptime(str(m['e_date']),'%y%m%d').date();hour=str(m['m_hour']).zfill(4)
            kickoff=datetime.combine(date,datetime.strptime(hour,'%H%M').time(),ZoneInfo('Asia/Jerusalem'))
            if kickoff<=observed:continue
            if home not in TEAMS or away not in TEAMS:raise ValueError('Unknown team translation: '+home+' / '+away)
            odds={s:number(o['price'],1.000001,1000) for s,o in zip(['home','draw','away'],outcomes)}
            rows.append(dict(market_id=str(m['mId']),event_id=str(m['eId']),league=LEAGUES[m['league']],date=date.isoformat(),kickoff=kickoff.isoformat(),home=TEAMS[home],away=TEAMS[away],home_he=home,away_he=away,odds=odds))
        except (KeyError,ValueError,TypeError) as exc:rejected.append(dict(market_id=m.get('mId'),reason=str(exc)))
    if not rows:raise ValueError('No supported upcoming full-time 1/X/2 markets in Winner feed')
    return dict(source='Winner public API',url=URL,fetched_at=observed.isoformat(),currency='ILS',markets=rows,rejected=rejected)

def download():
    global _cache
    with _lock:
        now=datetime.now(timezone.utc)
        if _cache and (now-datetime.fromisoformat(_cache['fetched_at'])).total_seconds()<60:return _cache
        headers={'Accept':'application/json','User-Agent':'FootballForecast/1.0','Origin':'https://www.winner.co.il','Referer':'https://www.winner.co.il/','RequestId':uuid.uuid4().hex,'DeviceId':_device,'HashesMessage':'{"reason":"Initiated"}','appVersion':'2.7.0','UserAgentData':json.dumps(dict(devicemodel='desktop',deviceos='windows',deviceosversion='10',appversion='2.7.0',apptype='desktop',originId='3',isAccessibility=False))}
        try:
            with urlopen(Request(URL,headers=headers),timeout=25) as response:raw=response.read(16_000_001)
            if len(raw)>16_000_000:raise ValueError('Winner feed exceeds size limit')
            result=parse(json.loads(raw),datetime.now(timezone.utc))
        except Exception as exc:raise ValueError('Winner download failed; no stale odds substituted: '+str(exc)) from exc
        FOLDER.mkdir(parents=True,exist_ok=True);archive=FOLDER/'snapshots';archive.mkdir(exist_ok=True)
        (archive/(now.strftime('%Y%m%dT%H%M%S%fZ')+'.json')).write_bytes(raw)
        target=FOLDER/'latest.json';temp=FOLDER/'latest.tmp';temp.write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8');temp.replace(target)
        _cache=result
        return result

def quote(feed,match):
    now=datetime.now(timezone.utc)
    if not -30<=(now-datetime.fromisoformat(feed['fetched_at'])).total_seconds()<=300:raise ValueError('Winner automatic quote expired; refresh prices')
    hits=[m for m in feed['markets'] if m['league']==match['league'] and m['date']==match['date'] and team_key(m['home'])==team_key(match['home']) and team_key(m['away'])==team_key(match['away'])]
    if len(hits)!=1:raise ValueError('No unique Winner fixture match for this league/date/home/away; refresh shared match inputs if dated in the past')
    m=hits[0]
    if datetime.fromisoformat(m['kickoff'])<=now:raise ValueError('Winner fixture has started; pre-match quote withheld')
    return m['odds'],dict(source='Winner automatic',fetched_at=feed['fetched_at'],currency='ILS',market_id=m['market_id'],kickoff=m['kickoff'])

if __name__=='__main__':
    result=download();print(json.dumps(dict(markets=len(result['markets']),rejected=len(result['rejected']),fetched_at=result['fetched_at'])))
