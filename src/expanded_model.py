"""Verified inference adapter for the saved 2008–22 vector forest bundle."""
import hashlib,json
from pathlib import Path
import joblib
import pandas as pd
from lineup_model import LineupForest
from lineup_sources import team_key
from live_predictions import ROOT

class ExpandedForest(LineupForest):
    def __init__(self,bundle_path=None):
        status=json.loads((ROOT/'results/expanded_comparison/status.json').read_text())
        if status['state']!='completed':raise ValueError('Expanded experiment incomplete')
        for path,expected in status['input_hashes'].items():
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=expected:raise ValueError('Expanded training input changed: '+path)
        import sklearn
        if status['verification']['versions']['sklearn']!=sklearn.__version__:raise ValueError('Expanded model sklearn version mismatch')
        path=Path(bundle_path) if bundle_path is not None else ROOT/'results/expanded_comparison/expanded__RF_vector.joblib'
        bundle=joblib.load(path)
        if bundle['scaler'] is not None:raise ValueError('Expanded inference expects unscaled forest features')
        self.forest=bundle['model'];self.encoder=bundle['encoder'];self.numeric=bundle['numeric'];self.categorical=bundle['categorical']
        if self.forest.n_outputs_!=2 or self.forest.n_features_in_!=sum(len(c) for c in self.encoder.categories_)+len(self.numeric):raise ValueError('Expanded feature schema mismatch')
        frame=pd.read_csv(ROOT/'data/expanded_2008/train.csv')
        self.teams={team_key(n):n for c in self.categorical for n in frame[c].unique()}
        self.metadata=dict(name='Expanded 2008–22 FIFA lineup RF · 19,263 training matches',path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),train_matches=len(frame),variant='expanded_2008')
