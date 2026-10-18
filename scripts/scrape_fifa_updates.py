"""Stream the public SoFIFA history mirror into a compact, dated archive.

Career-mode ratings, NOT EA Live Form. Never treats collection time as release time.
"""
import csv, gzip, io, json, hashlib
from pathlib import Path
from urllib.request import urlopen
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/fifa_updates'
URL='https://huggingface.co/datasets/jsulz/FIFA23/resolve/main/male_players.csv'
COLS=['player_id','fifa_version','fifa_update','fifa_update_date','short_name','long_name','dob','overall','club_name']

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    dest=OUT/'history.csv.gz'
    if dest.exists() and (OUT/'source.json').exists():
        print('Verified archive already collected',flush=True);return
    counts={};n=0
    with urlopen(URL,timeout=90) as response, gzip.open(OUT/'history.partial.gz','wt',encoding='utf-8',newline='') as target:
        reader=csv.DictReader(io.TextIOWrapper(response,encoding='utf-8-sig',newline=''))
        assert set(COLS)<=set(reader.fieldnames),reader.fieldnames
        writer=csv.DictWriter(target,fieldnames=COLS);writer.writeheader()
        for row in reader:
            if not row.get('player_id'):continue
            day=row['fifa_update_date'][:10]
            datetime.strptime(day,'%Y-%m-%d')
            assert 1<=int(row['overall'])<=99
            writer.writerow({c:row[c] for c in COLS})
            counts[day]=counts.get(day,0)+1;n+=1
            if n%250000==0:print(f'{n:,} player snapshots, {len(counts)} dates',flush=True)
    (OUT/'history.partial.gz').replace(dest)
    meta=dict(url=URL,collected_at=datetime.now(timezone.utc).isoformat(),rows=n,dates=counts,
              sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),kind='Career-mode base ratings; not Live Form',
              timestamp_caveat='Historical effective date supplied by archive; not independently timestamped publication evidence')
    (OUT/'source.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps({k:v for k,v in meta.items() if k!='dates'}),flush=True)

if __name__=='__main__':main()
