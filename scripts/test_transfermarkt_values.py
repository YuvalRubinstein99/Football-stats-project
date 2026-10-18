"""Audit historical valuation chronology and complete-lineup aggregation."""
import csv,gzip,math
from collections import defaultdict
from scrape_transfermarkt_values import OUT,before

h=(['2020-01-01','2020-02-01'],[10,20])
assert before(h,'2020-01-01') is None
assert before(h,'2020-02-01')==('2020-01-01',10)
assert before(h,'2020-02-02')==('2020-02-01',20)
totals=defaultdict(list);n=0
with gzip.open(OUT/'starter_valuation_audit.csv.gz','rt',encoding='utf-8') as f:
    for r in csv.DictReader(f):
        if r['valuation_date']:
            assert r['valuation_date']<r['fixture_date']
            assert int(r['age_days'])>0
            totals[r['game_id'],r['side']].append(float(r['market_value_eur']));n+=1
complete=0
with (OUT/'match_features.csv').open(encoding='utf-8',newline='') as f:
    for r in csv.DictReader(f):
        if r['complete_22']=='True':
            for side in ['home','away']:
                vals=totals[r['game_id'],side]
                assert len(vals)==11 and int(r[side+'_starters'])==11
                assert math.isclose(sum(vals),float(r[side+'_xi_value_eur']))
            complete+=1
print(f'Passed: {n:,} strictly prior valuations; {complete:,} complete XI-pair totals; same-day boundary excluded.')
