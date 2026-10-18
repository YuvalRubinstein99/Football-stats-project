"""Collect FC25/26 archives without assuming their snapshots predate matches."""
import csv,io,json,zipfile
from scrape_recent_seasons import OUT,fetch,valid_ratings,atomic_json,stamp

def main():
    raw=OUT/'raw';raw.mkdir(parents=True,exist_ok=True)
    audit=[]
    for edition,slug in [('25','nyagami/ea-sports-fc-25-database-ratings-and-stats'),('26','rovnez/fc-26-fifa-26-player-data')]:
        row=dict(edition=edition,source=slug,eligible_for_historical_model=False)
        try:
            path=raw/f'fc{edition}.zip'
            print('Download FC'+edition,flush=True)
            row['download']=fetch('https://www.kaggle.com/api/v1/datasets/download/'+slug,path,valid_ratings,3)
            tables=[]
            with zipfile.ZipFile(path) as z:
                for member in z.namelist():
                    if not member.endswith('.csv'):continue
                    with z.open(member) as stream:
                        reader=csv.DictReader(io.TextIOWrapper(stream,encoding='utf-8-sig'))
                        columns=reader.fieldnames
                        datecols=[c for c in columns if 'date' in c.lower() or 'as_of' in c.lower()]
                        dates={c:set() for c in datecols};count=0
                        for record in reader:
                            count+=1
                            for c in datecols:dates[c].add(record[c])
                    tables.append(dict(file=member,rows=count,columns=columns,date_values={c:sorted(v)[:30] for c,v in dates.items()}))
            row['tables']=tables
            row['note']='Snapshot publication dates must be verified before joining ratings to historical matches; archive retrieval time is not the rating date.'
        except Exception as exc:row['error']=str(exc)
        audit.append(row)
        atomic_json(OUT/'recent_ratings_audit.json',dict(fetched_at=stamp(),archives=audit))
    print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':main()
