"""Audit chronological boundaries, frozen selection and reported curve accounting."""
import json
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]/'results/chronological_selection'
audit=json.loads((BASE/'fold_audit.json').read_text())
assert len(audit)==66
for r in audit:
    assert r['train_last_date']<r['validation_first_date']
    assert all(y<r['season'] for y in r['calibration_seasons'])
frozen=json.loads((BASE/'frozen_choices.json').read_text())
report=json.loads((BASE/'report.json').read_text())
assert len(frozen['candidates'])==220 and len(report['results'])==5
for r in report['results']:
    candidates=[v for v in frozen['candidates'] if v['league']==r['league']]
    chosen=next(v for v in frozen['choices'] if v['league']==r['league'])
    assert r['model']==chosen['model'] and r['weights']==chosen['weights']
    assert chosen['oof']['score']==max(v['oof']['score'] for v in candidates)
    robust=next(v for v in candidates if v['model']==r['robust_selected_model'])
    assert robust['robust_oof']['score']==max(v['robust_oof']['score'] for v in candidates)
    for key in ['oof_mixture','oof_robust','test_mixture','test_robust','test_robust_selected']:
        result=r[key];points=result['points']
        assert abs(points[-1]['net']-result['net'])<1e-6
        peak=1000.;dd=0.
        for p in points:
            balance=1000+p['net'];peak=max(peak,balance);dd=max(dd,1-balance/peak)
        assert abs(dd-result['max_drawdown'])<1e-9
        dates=[p['date'] for p in points[1:]]
        assert dates==sorted(dates)
    assert r['oof_mixture']['points'][-1]['date']<r['test_mixture']['points'][1]['date']
print('Passed: 66 forward-fit/calibration boundaries, 220 OOF candidates, five frozen selections, and all cumulative profit/drawdown curves.')
