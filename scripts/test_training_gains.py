"""Reconcile all chart endpoints, seasons, and independent RF OOB accounting."""
import csv,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

class TrainingGains(unittest.TestCase):
    def test_all_curves(self):
        payload=json.loads((ROOT/'results/team_identity/train_test_curves.json').read_text())
        totals={(r['model'],r['split'],r['strategy']):r for r in payload['summary']}
        self.assertEqual(len(payload['curves']),285)
        for c in payload['curves']:
            r=totals[c['model'],c['split'],c['strategy']]
            self.assertEqual(c['points'][0]['net'],0)
            self.assertAlmostEqual(c['points'][-1]['net'],r['net'],places=7)
            self.assertAlmostEqual(c['points'][-1]['staked'],r['staked'],places=7)
            self.assertEqual(r['matches'],1447 if c['split']=='Test' else 19263)
            labels=[p['label'] for p in c['points'][1:]]
            order=[(int(v[:4]),int(v.split(' W')[1])) for v in labels]
            self.assertEqual(order,sorted(set(order)))
            self.assertEqual({y for y,w in order},{2022} if c['split']=='Test' else set(range(2008,2022)))
        baseline=json.loads((ROOT/'results/expanded_interface/stake_cumulative.json').read_text())
        for c in baseline['curves']:
            if c['split']!='Train (OOB)':continue
            net=sum((i+1)*v[1] for p in c['points'] for i,v in enumerate(p['levels']))
            self.assertAlmostEqual(net,totals['Expanded RF','Train (OOB)',c['strategy']]['net'],places=7)

if __name__=='__main__':unittest.main()
