"""Offline regressions for season selection, cache integrity and safe publication."""
import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scrape_older_seasons import digest, fetch
from scrape_recent_seasons import extract_history, main, result_rows, stream_csv
from audit_recent_seasons import audit


class RecentScraperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_results_exclude_future_and_unplayed(self):
        path = self.root / 'results.csv'
        fields = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'B365H', 'B365D', 'B365A']
        rows = [
            ['01/08/2026', 'A', 'B', '1', '0', '2', '3', '4'],
            ['02/08/2026', 'A', 'C', '', '', '', '', ''],
            ['01/12/2026', 'B', 'C', '2', '1', '2', '3', '4'],
        ]
        stream_csv(path, (dict(zip(fields, r)) for r in rows), fields)
        result = result_rows(path, 2026, 'E0', '2026-10-20')
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0]['valid_bet365'])
        with path.open('a') as stream:
            stream.write('01/08/2026,A,B,1,0,2,3,4\n')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            result_rows(path, 2026, 'E0', '2026-10-20')

    def test_stream_failure_preserves_old_output_and_empty_clears_it(self):
        path = self.root / 'output.csv'
        path.write_text('old')
        def broken():
            yield {'id': '1'}
            raise ValueError('broken source')
        with self.assertRaises(ValueError):
            stream_csv(path, broken(), ['id'])
        self.assertEqual(path.read_text(), 'old')
        stream_csv(path, [], ['id'])
        self.assertEqual(path.read_text(), 'id\n')

    def test_offline_cache_checks_hash_and_source_without_network(self):
        path = self.root / 'source.csv'
        path.write_text('cached')
        meta = path.with_suffix('.csv.json')
        meta.write_text(json.dumps(dict(url='https://example.test/data', sha256=digest(path))))
        with patch('scrape_older_seasons.urlopen', side_effect=AssertionError('network used')):
            fetch('https://example.test/data', path, lambda p: None, 0, offline=True)
            with self.assertRaises(FileNotFoundError):
                fetch('https://example.test/wrong', path, lambda p: None, 0, offline=True)
            path.write_text('corrupt')
            with self.assertRaises(FileNotFoundError):
                fetch('https://example.test/data', path, lambda p: None, 0, offline=True)

    def make_history(self):
        raw = self.root / 'raw'
        raw.mkdir()
        games = [
            dict(game_id=str(year), competition_id='GB1', season=str(year), date=f'{year}-08-01',
                 home_club_id='1', away_club_id='2', home_club_goals='1', away_club_goals='0')
            for year in (2023, 2024)
        ]
        lineups = [
            dict(game_id=str(year), club_id=str(club), type='starting_lineup',
                 player_id=str(club * 100 + i))
            for year in (2023, 2024) for club in (1, 2) for i in range(11)
        ]
        sources = {}
        for name, rows in [('games', games), ('game_lineups', lineups)]:
            path = raw / (name + '.csv.gz')
            with gzip.open(path, 'wt', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            sources[name] = dict(sha256=digest(path))
        return raw, sources

    def test_scope_and_cached_extraction_and_empty_season(self):
        raw, sources = self.make_history()
        args = raw, self.root, [2024], ['E0'], '2026-10-20', sources
        result = extract_history(*args)
        self.assertEqual((result['games'], result['games_with_22_starters']), (1, 1))
        self.assertTrue(extract_history(*args)['reused_extraction'])
        (self.root / 'lineups.csv').write_text('corrupt')
        self.assertFalse(extract_history(*args)['reused_extraction'])
        result = extract_history(raw, self.root, [2026], ['E0'], '2026-10-20', sources)
        self.assertEqual(result['games'], 0)
        self.assertEqual(len(list(csv.DictReader((self.root / 'games.csv').open()))), 0)
        rows = audit(self.root, dict(seasons=[2026], leagues=['E0'], results_published=True))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['lineup_source_matches'], 0)

    def test_failed_partition_does_not_replace_existing_results(self):
        target = self.root / 'results_odds.csv'
        target.write_text('prior successful output')
        with patch('scrape_recent_seasons.fetch', side_effect=FileNotFoundError('uncached')), \
             patch('audit_recent_seasons.audit', return_value=[]):
            code = main(['--earliest', '2023', '--latest', '2024', '--leagues', 'E0',
                         '--output', str(self.root), '--offline'])
        self.assertEqual(code, 1)
        self.assertEqual(target.read_text(), 'prior successful output')
        self.assertFalse((self.root / 'run.lock').exists())
        status = json.loads((self.root / 'status.json').read_text())
        self.assertFalse(status['results_published'])


if __name__ == '__main__':
    unittest.main()
