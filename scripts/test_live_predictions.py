"""Focused checks for the live data and prediction pipeline (no network)."""
from datetime import date, timedelta
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import live_predictions as live


class LiveTests(unittest.TestCase):
    def test_csv_schema_and_bad_scores(self):
        rows = live.parse_csv(b'Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\nE0,01/09/26,A,B,0,2\n', True)
        self.assertEqual(rows[0]['hg'], 0)
        self.assertEqual(rows[0]['date'], date(2026, 9, 1))
        for data in [b'<html>Error</html>', b'Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\nE0,01/09/26,A,B,-1,2\n']:
            with self.assertRaises(ValueError):
                live.parse_csv(data, True)

    def test_no_same_day_or_future_leakage(self):
        history = [{'date': date(2026, 9, 1), 'home': 'A', 'away': 'B', 'hg': 2, 'ag': 1}]
        cutoff = date(2026, 9, 2)
        original = live.goal_rates(live.fit(history, cutoff), 'A', 'B')
        for day in (cutoff, cutoff + timedelta(days=1)):
            history.append({'date': day, 'home': 'A', 'away': 'B', 'hg': 50, 'ag': 0})
        self.assertEqual(original, live.goal_rates(live.fit(history, cutoff), 'A', 'B'))

    def test_unknown_teams_use_league_prior(self):
        model = live.fit([{'date': date(2026, 1, 1), 'home': 'A', 'away': 'B', 'hg': 2, 'ag': 1}], date(2026, 1, 2))
        self.assertEqual(live.goal_rates(model, 'New home', 'New away'), (2., 1., 0.))

    def test_probability_mass_and_symmetry(self):
        for h, a in [(.05, 8), (8, .05), (1.2, 1.2), (8, 8)]:
            p, _ = live.probabilities(h, a)
            reverse, _ = live.probabilities(a, h)
            self.assertAlmostEqual(sum(p), 1.)
            self.assertTrue(all(0 <= x <= 1 for x in p))
            self.assertAlmostEqual(p[0], reverse[2])
            self.assertAlmostEqual(p[1], reverse[1])

    def test_conflicting_duplicates_fail(self):
        m = {'date': date(2026, 1, 1), 'league': 'E0', 'home': 'A', 'away': 'B', 'hg': 1, 'ag': 0}
        self.assertEqual(len(live.deduplicate([m, m])), 1)
        with self.assertRaises(ValueError):
            live.deduplicate([m, {**m, 'hg': 2}])

    def test_pipeline_exports_and_filters(self):
        today = live.datetime.now(live.timezone.utc).date()
        history = [{'league': 'E0', 'date': today - timedelta(days=200-i),
                    'home': 'A', 'away': 'B', 'hg': i % 4, 'ag': i % 3, 'time': ''} for i in range(180)]
        fixture = {'league': 'E0', 'date': today + timedelta(days=1), 'home': 'A', 'away': 'B', 'time': '15:00'}
        fixtures = [fixture, {**fixture, 'date': today-timedelta(days=1)},
                    {**fixture, 'league': 'F1'}, {**fixture, 'date': today+timedelta(days=50)}]
        def fake_download(relative, *args):
            return (fixtures if relative == 'fixtures.csv' else history), {'url': relative}
        with tempfile.TemporaryDirectory() as directory, patch.object(live, 'download', side_effect=fake_download):
            out = Path(directory)
            report = live.pipeline(['E0'], seasons=1, output=out, progress=lambda _: None)
            self.assertEqual(len(report['predictions']), 1)
            self.assertEqual(report['validation']['E0']['matches'], 60)
            self.assertTrue((out / 'predictions.csv').read_text().startswith('league,date'))
            self.assertEqual(json.loads((out / 'predictions.json').read_text())['as_of'], today.isoformat())


if __name__ == '__main__':
    unittest.main()
