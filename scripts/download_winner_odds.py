import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.lineup_deps'));sys.path.insert(0,str(ROOT/'src'))
from winner_source import download
if __name__=='__main__':
    data=download()
    print(json.dumps(dict(markets=len(data['markets']),rejected=len(data['rejected']),fetched_at=data['fetched_at'],saved_to=str(ROOT/'results/winner/latest.json'))))
