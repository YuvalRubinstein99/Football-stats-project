"""Temperature scaling of H/D/A probabilities; fitted on forward predictions only."""
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax

ROOT=Path(__file__).resolve().parents[1]
SUFFIX='__calibrated'

def scale(probabilities,temperature):
    p=np.asarray(probabilities,dtype=float)
    if not np.isfinite(temperature) or temperature<=0:raise ValueError('Invalid temperature')
    if p.shape[-1]!=3 or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(axis=-1),1):
        raise ValueError('Expected normalized H/D/A probabilities')
    return softmax(np.log(np.clip(p,1e-15,1))/temperature,axis=-1)

def fit_temperature(p,y):
    p=np.asarray(p);y=np.asarray(y,dtype=int)
    def loss(log_t):return -np.log(scale(p,np.exp(log_t))[np.arange(len(y)),y]).mean()
    result=minimize_scalar(loss,bounds=(np.log(.25),np.log(4)),method='bounded',options={'xatol':1e-10})
    if not result.success:raise ValueError('Calibration failed')
    return float(np.exp(result.x))

@lru_cache(maxsize=3)
def artifacts(recent=False,classifier=False):
    folder='classifier' if classifier else 'recency' if recent else 'calibration'
    result=json.loads((ROOT/f'results/{folder}/calibrators.json').read_text())
    import hashlib
    for path,digest in result['input_hashes'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:raise ValueError('Calibration input changed; rerun calibration')
    return result

def calibrate(name,p):
    base=name.removesuffix(SUFFIX)
    return scale(p,artifacts(base.endswith('__recent'),base.startswith('RF classifier'))['models'][base]['temperature'])
