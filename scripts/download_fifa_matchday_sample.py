"""Retrieve a specifically labelled historical Match Day sample; no execution."""
import hashlib,json,io,zipfile
from pathlib import Path
from datetime import datetime,timezone
from urllib.request import urlopen,Request
from urllib.parse import urljoin
from html.parser import HTMLParser

URL='https://digiex.net/threads/fifa-13-squads-matchday-and-assets.12490/'
OUT=Path(__file__).resolve().parents[1]/'data/fifa_live_form'
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='a' and '12819' in a.get('href',''):self.links.append(a['href'])

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    status=dict(source=URL,checked_at=datetime.now(timezone.utc).isoformat(),claimed_snapshot='2013-04-07',eligible_for_models=False,
                note='Forum-labelled Xbox 360 FIFA 13 squads, Matchday and assets sample. Contents and actual live-form fields must be verified; not a multi-season history.')
    try:
        with urlopen(Request(URL,headers={'User-Agent':'FootballStatsProject research'}),timeout=30) as r:page=r.read(2_000_000)
        (OUT/'sample_source.html').write_bytes(page)
        parser=Links();parser.feed(page.decode('utf-8',errors='replace'))
        links=list(dict.fromkeys(urljoin(URL,u) for u in parser.links));status['download_links']=links
        if not links:raise ValueError('No expected attachment found in source page')
        with urlopen(links[0],timeout=45) as r:data=r.read(20_000_001)
        if len(data)>20_000_000:raise ValueError('Sample exceeds 20 MB bound')
        if data.startswith(b'PK'):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                status['members']=[dict(name=i.filename,bytes=i.file_size) for i in z.infolist()]
                assert not z.testzip(),'Archive checksum failed'
            suffix='zip'
        elif data.startswith(b'Rar!'):suffix='rar'
        else:raise ValueError('Attachment is not a ZIP or RAR archive')
        (OUT/('fifa13_matchday_20130407.'+suffix)).write_bytes(data)
        status.update(downloaded=True,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    except Exception as exc:status.update(downloaded=False,error=str(exc))
    (OUT/'sample_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(json.dumps(status,indent=2))

if __name__=='__main__':main()
