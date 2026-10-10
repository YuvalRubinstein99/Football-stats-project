"""Read-only alternate prices. No accounts, wallets, orders or access-control workarounds."""
import json,math,re,secrets,threading
from datetime import datetime,timezone
from urllib.request import Request,urlopen
from urllib.parse import urlparse,quote
from manual_strategies import number

_snapshots={}
_lock=threading.Lock()
SIDES=('home','draw','away')


def now():return datetime.now(timezone.utc)


def get_json(url):
    request=Request(url,headers={'User-Agent':'FootballForecast/1.0 (public read-only market data)','Accept':'application/json'})
    with urlopen(request,timeout=15) as response:
        payload=response.read(4_000_001)
    if len(payload)>4_000_000:raise ValueError('Provider response is too large')
    return json.loads(payload)


def slug_from(value):
    value=str(value).strip()
    if '://' in value:
        u=urlparse(value)
        if u.scheme!='https' or u.hostname not in ('polymarket.com','www.polymarket.com') or not u.path.startswith('/event/'):
            raise ValueError('Use a Polymarket event URL or slug')
        value=u.path.split('/')[2]
    if not re.fullmatch(r'[a-zA-Z0-9-]{1,200}',value):raise ValueError('Invalid Polymarket event slug')
    return value


def store(value):
    with _lock:
        for key in list(_snapshots):
            if (now()-_snapshots[key]['created']).total_seconds()>300:del _snapshots[key]
        if len(_snapshots)>=100:raise ValueError('Too many snapshots; retry after older snapshots expire')
        key=secrets.token_urlsafe(20);_snapshots[key]=dict(value,created=now());return key


def snapshot(key):
    with _lock:value=_snapshots.get(key)
    if not value or (now()-value['created']).total_seconds()>300:raise ValueError('Polymarket snapshot expired; fetch prices again')
    return value


def discover(reference):
    slug=slug_from(reference)
    event=get_json('https://gamma-api.polymarket.com/events/slug/'+quote(slug))
    markets=[m for m in event.get('markets',[]) if m.get('active') is True and m.get('closed') is False]
    if not markets:raise ValueError('No active, open markets in this event')
    key=store(dict(kind='event',slug=slug,markets=markets,title=event.get('title','')))
    return dict(snapshot=key,title=event.get('title',''),url='https://polymarket.com/event/'+slug,
        markets=[dict(id=str(m['id']),question=m.get('question',''),description=m.get('description',''),
                      game_start=m.get('gameStartTime'),market_type=m.get('sportsMarketType')) for m in markets])


def parse_array(value):return json.loads(value) if isinstance(value,str) else value


def price_from_book(market,book):
    if market.get('acceptingOrders') is not True:raise ValueError('Market is not accepting orders')
    asks=[(number(a['price'],.000001,.999999),number(a['size'],.000001,1e12)) for a in book.get('asks',[])]
    if not asks:raise ValueError('No available asks')
    ask=min(p for p,s in asks);shares=sum(s for p,s in asks if p==ask)
    if market.get('feesEnabled') is False:rate=0.
    elif market.get('feesEnabled') is True:
        fee=market.get('feeSchedule') or {}
        if fee.get('exponent')!=1:raise ValueError('Unsupported or missing fee schedule; cannot calculate net price')
        rate=number(fee.get('rate'),0,1)
    else:raise ValueError('Provider did not report whether fees are enabled')
    cost=ask+rate*ask*(1-ask)
    if not 0<cost<1:raise ValueError('Invalid fee-adjusted share cost')
    return dict(ask=ask,fee_rate=rate,cost_per_share=cost,decimal_odds=1/cost,
        max_spend=shares*cost,min_spend=number(market.get('orderMinSize'),0,1e6),
        min_shares=number(book.get('min_order_size'),0,1e6),book_timestamp=book.get('timestamp'))


def fetch_prices(event_key,market_ids,fixture,confirmed):
    if confirmed is not True:raise ValueError('Confirm the teams, date, Yes outcomes, and 90-minute settlement rules first')
    event=snapshot(event_key)
    if event['kind']!='event':raise ValueError('Invalid event snapshot')
    if set(market_ids)!=set(SIDES) or len(set(market_ids.values()))!=3:raise ValueError('Choose three different markets: home win, draw, away win')
    selected={str(m['id']):m for m in event['markets']};prices={}
    for side in SIDES:
        m=selected.get(str(market_ids[side]))
        if not m:raise ValueError('Selected market is not in this event')
        if m.get('sportsMarketType')!='moneyline':raise ValueError('Only full-match moneyline markets are supported')
        if str(m.get('gameStartTime',''))[:10]!=fixture['date']:raise ValueError('Game date differs from the selected fixture or is unavailable')
        outcomes=parse_array(m.get('outcomes',[]))
        if sorted(str(x).lower() for x in outcomes)!=['no','yes']:raise ValueError('Select binary Yes/No outcome markets')
        yes=next(i for i,x in enumerate(outcomes) if str(x).lower()=='yes')
        ids=parse_array(m.get('positionIds') if m.get('version')=='v2' else m.get('clobTokenIds',[]))
        if not ids or len(ids)!=2 or not str(ids[yes]).isdigit():raise ValueError('Missing outcome asset ID')
        book=get_json('https://clob.polymarket.com/book?token_id='+quote(str(ids[yes])))
        prices[side]=price_from_book(m,book)
        prices[side]['question']=m.get('question','')
    value=dict(kind='quotes',fixture=fixture,prices=prices,source_url='https://polymarket.com/event/'+event['slug'])
    return dict(snapshot=store(value),prices=prices,fetched_at=now().isoformat(),url=value['source_url'])


def fixture_key(m):return '|'.join(str(m[k]) for k in ('date','home','away'))


def quote_for(source,entry,match):
    if not isinstance(entry,dict):raise ValueError('Enter or fetch all three source prices for this match')
    if source=='winner':
        if entry.get('confirmed') is not True:raise ValueError('Confirm Winner full-time 1/X/2 rules')
        observed=datetime.fromisoformat(entry['observed_at'].replace('Z','+00:00'))
        if observed.tzinfo is None:raise ValueError('Quote time must include a time zone')
        if not -30<=(now()-observed).total_seconds()<=900:raise ValueError('Winner quote expired; enter current prices')
        return {s:number(entry.get(s),1.000001,1000) for s in SIDES},dict(source='Winner manual',fetched_at=entry['observed_at'],currency='ILS')
    snap=snapshot(entry.get('snapshot'))
    if snap['kind']!='quotes' or fixture_key(snap['fixture'])!=fixture_key(match):raise ValueError('Polymarket snapshot belongs to a different fixture')
    return {s:snap['prices'][s]['decimal_odds'] for s in SIDES},dict(source='Polymarket ask + fee',fetched_at=snap['created'].isoformat(),currency='USD',prices=snap['prices'],url=snap['source_url'])


def check_execution(bet,meta):
    if bet['decision']!='Bet' or 'prices' not in meta:return bet
    for side in SIDES:
        amount=bet[side+'_amount']
        if amount<=0:continue
        price=meta['prices'][side]
        if amount>price['max_spend'] or amount<price['min_spend'] or amount/price['cost_per_share']<price['min_shares']:
            return dict(bet,decision='Unavailable',reason='Stake does not satisfy order minimums or available depth at the quoted ask.',amount=0,level=0,home_amount=0,draw_amount=0,away_amount=0)
    return bet
