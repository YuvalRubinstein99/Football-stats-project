"""Cache public fantasy history and record checksums; never execute source code."""
import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/fantasy/raw'


def fetch(url, offline=False):
    RAW.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode()).hexdigest()
    path, receipt = RAW / (key + '.bin'), RAW / (key + '.json')
    if path.exists() and receipt.exists():
        data = path.read_bytes()
        meta = json.loads(receipt.read_text())
        if meta['url'] == url and meta['sha256'] == hashlib.sha256(data).hexdigest():
            return data
        raise ValueError('Fantasy cache checksum mismatch: ' + url)
    if offline:
        raise FileNotFoundError('Uncached fantasy source: ' + url)
    request = urllib.request.Request(url, headers={'User-Agent': 'FootballStatsResearch/1.0'})
    with urllib.request.urlopen(request, timeout=45) as response:
        data = response.read()
    temp = path.with_suffix('.tmp')
    temp.write_bytes(data)
    temp.replace(path)
    receipt.write_text(json.dumps(dict(url=url, sha256=hashlib.sha256(data).hexdigest(),
        retrieved_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), bytes=len(data)), indent=2))
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect', nargs='*', default=[])
    parser.add_argument('--fpl', action='store_true')
    parser.add_argument('--start', type=int, default=2016)
    parser.add_argument('--end', type=int, default=2022)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    for url in args.inspect:
        data = fetch(url, args.offline)
        print(url, 'bytes=', len(data), 'cache=', hashlib.sha256(url.encode()).hexdigest(), flush=True)
    if args.fpl:
        for year in range(args.start, args.end + 1):
            season = f'{year}-{str(year+1)[-2:]}'
            for file in ['gws/merged_gw.csv', 'teams.csv']:
                url = f'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/{file}'
                fetch(url, args.offline)
                print('Cached', season, file, flush=True)
                if not args.offline:
                    time.sleep(.2)


if __name__ == '__main__':
    main()
