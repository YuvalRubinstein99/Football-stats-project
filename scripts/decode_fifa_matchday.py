"""Read-only forensic extraction of the FIFA13 sample. No model ingestion.

Format references: Free60 STFS documentation and Celtian/fifa-t3db docs/format.md.
FIFA15 XML is a provisional shortname dictionary, NOT an authoritative FIFA13 schema.
"""
import hashlib
import json
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'data/fifa_live_form'

def uint(b, p, n=4, endian='big'):
    return int.from_bytes(b[p:p+n], endian)

def extract(b):
    assert b[:4] == b'CON '
    header = (uint(b, 0x340)+4095)//4096*4096
    assert header == 0xa000 and b[0x37b] == 0, 'Unsupported STFS layout'
    def block(k):
        physical = k + 2*(k//170+1) + (2*(k//28900+1) if k >= 170 else 0)
        return b[header+physical*4096:header+(physical+1)*4096]
    entry = block(uint(b, 0x37e, 3, 'little'))[:64]
    assert entry[40] & 64, 'Requires contiguous file'
    start = uint(entry,47,3,'little')
    size = uint(entry,52)
    chunks = []
    for k in range(start, start+(size+4095)//4096):
        chunk = block(k)
        group = k//170
        hashblock = group*172 + (2 if group else 0)
        hashes = [b[header+(hashblock+c)*4096+(k%170)*24:header+(hashblock+c)*4096+(k%170)*24+20] for c in (0,1)]
        assert hashlib.sha1(chunk).digest() in hashes, f'Invalid block hash {k}'
        chunks.append(chunk)
    return b''.join(chunks)[:size]

def tables(payload):
    base = payload.index(b'DB\0\x08')
    b = payload[base:]
    assert uint(b,8) == len(b), (uint(b,8),len(b))
    count = uint(b,16)
    start = 28+count*8
    result = {}
    for i in range(count):
        p = 24+i*8
        name = b[p:p+4][::-1].decode('ascii')
        h = start+uint(b,p+4)
        size, rows, fields = uint(b,h+4),uint(b,h+18,2),b[h+24]
        desc = []
        for f in range(fields):
            d = h+36+f*16
            desc.append(dict(type=uint(b,d),offset=uint(b,d+4),shortname=b[d+8:d+12][::-1].decode('ascii'),depth=uint(b,d+12)))
        records = h+36+16*fields
        assert records+size*rows <= len(b)
        result[name] = dict(size=size,rows=rows,fields=desc,raw=b[records:records+size*rows])
    return result

def main():
    out = ROOT/'decoded'
    out.mkdir(exist_ok=True)
    meta = ET.parse(ROOT/'fifa15-reference-meta.xml').getroot()
    names = {t.attrib['shortname']:t.attrib['name'] for t in meta.findall('table')}
    fields = {f.attrib['shortname']:f.attrib for f in meta.findall('.//field')}
    dbs = {}
    with zipfile.ZipFile(ROOT/'fifa13_matchday_20130407.zip') as z:
        for kind in ('MatchDay','Squads'):
            member = next(n for n in z.namelist() if '/'+kind+' ' in n)
            payload = extract(z.read(member))
            (out/(kind+'.bin')).write_bytes(payload)
            dbs[kind] = tables(payload)
    report = []
    for key,a in dbs['MatchDay'].items():
        b = dbs['Squads'].get(key)
        row = dict(shortname=key,reference_name=names.get(key),rows=a['rows'],record_bytes=a['size'],identical=bool(b and a==b),fields=[dict(f,reference_name=fields.get(f['shortname'],{}).get('name')) for f in a['fields']])
        if b and a['size']==b['size'] and a['rows']==b['rows']:
            row['changed_rows_by_position'] = sum(a['raw'][i:i+a['size']]!=b['raw'][i:i+a['size']] for i in range(0,len(a['raw']),a['size']))
        report.append(row)
    (out/'table_comparison.json').write_text(json.dumps(report,indent=2))
    decoded = {}
    for key in ('UlaA','OIcD','CZUM','lyxL','rBig'):
        table = dbs['MatchDay'][key]
        schema = meta.find(f"table[@shortname='{key}']")
        fm = {f.attrib['shortname']:f.attrib for f in schema.findall('./fields/field')} if schema is not None else {
            'aQrQ':dict(name='leagueid',rangelow='1'), 'mCXg':dict(name='teamid',rangelow='1'),
            'ykFq':dict(name='playerid',rangelow='0'), 'UERs':dict(name='overallrating',rangelow='1'),
            'rLZx':dict(name='form',rangelow='0'), 'yJXl':dict(name='prevform',rangelow='0'),
            'lwui':dict(name='unknown_lwui_raw',rangelow='0'), 'vjla':dict(name='position_raw',rangelow='0')}
        rows = []
        for i in range(table['rows']):
            raw = table['raw'][i*table['size']:(i+1)*table['size']]
            row = {}
            for f in table['fields']:
                if f['shortname'] not in fm:continue
                m = fm[f['shortname']]
                if f['type']==3:
                    value = (int.from_bytes(raw,'big') >> (8*len(raw)-f['offset']-f['depth'])) & ((1<<f['depth'])-1)
                    row[m['name']] = value+int(m.get('rangelow',0))
                elif f['type']==0:
                    row[m['name']] = raw[f['offset']//8:(f['offset']+f['depth'])//8].split(b'\0')[0].decode('utf-8',errors='replace')
            rows.append(row)
        decoded[names.get(key,'candidate_live_form')] = rows
    for kind in ('playerformdiff','teamformdiff'):
        rows = decoded[kind]
        assert all(r['newoverallrating']-r['oldoverallrating']==r['overallratingdiff'] for r in rows), 'Form arithmetic inconsistent'
        idname, parent = ('playerid','players') if kind=='playerformdiff' else ('teamid','teams')
        ids = {r[idname] for r in decoded[parent]}
        assert all(r[idname] in ids for r in rows), 'Orphan form IDs'
        (out/(kind+'.json')).write_text(json.dumps(rows,indent=2))
    (out/'reference_decoded_players.json').write_text(json.dumps(decoded['players']))
    (out/'reference_decoded_teams.json').write_text(json.dumps(decoded['teams']))
    candidates = decoded['candidate_live_form']
    players = {r['playerid']:r for r in decoded['players']}
    teams = {r['teamid']:r for r in decoded['teams']}
    for row in candidates:
        row['base_overallrating'] = players.get(row['playerid'],{}).get('overallrating')
        row['team_name'] = teams.get(row['teamid'],{}).get('teamname')
    (out/'candidate_live_form.json').write_text(json.dumps(candidates,indent=2))
    from collections import Counter
    evidence = dict(eligible_for_models=False,extraction='All STFS payload block SHA1 hashes verified',
        schema_caveat='FIFA15 shortname reference; rBig table identity and numeric offsets provisional. Not a verified FIFA13 schema.',
        player_form_changes=len(decoded['playerformdiff']),team_form_changes=len(decoded['teamformdiff']),
        candidate_live_form_rows=len(candidates),candidate_matched_player_ids=sum(r['playerid'] in players for r in candidates),
        candidate_matched_team_ids=sum(r['teamid'] in teams for r in candidates),
        form_counts=dict(Counter(r['form'] for r in candidates)),previous_form_counts=dict(Counter(r['prevform'] for r in candidates)),
        candidate_rating_differences=dict(Counter(r['overallrating']-r['base_overallrating'] for r in candidates if r['base_overallrating'] is not None)),
        internal_filename_date='2013-03-30',publication_date='2013-04-07',historical_join_not_before='2013-04-08')
    (out/'evidence.json').write_text(json.dumps(evidence,indent=2))
    print('Validated form rating arithmetic and parent IDs:', {k:len(decoded[k]) for k in ('playerformdiff','teamformdiff')})
    print(json.dumps(evidence,indent=2))

if __name__=='__main__':main()
