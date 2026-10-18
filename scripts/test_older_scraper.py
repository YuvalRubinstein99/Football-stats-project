import tempfile
import unittest
from pathlib import Path
from scrape_older_seasons import rating_before, rating_features, odds_ok, date_value, csv_rows

class HistoricalSafety(unittest.TestCase):
    def test_no_future_or_same_day_rating(self):
        history=[('2008-08-01',70),('2008-08-15',90),('2008-08-20',95)]
        self.assertEqual(rating_before(history,'2008-08-15'),('2008-08-01',70))
        self.assertIsNone(rating_before(history,'2008-08-01'))
        self.assertEqual(rating_before(history,'2008-08-16'),('2008-08-15',90))

    def test_feature_summary(self):
        features=rating_features(list(range(70,81)),'HomePlayer')
        self.assertEqual(features['HomePlayer_Overall_mean'],75)
        self.assertEqual(features['HomePlayer_3rd_best'],78)
        self.assertEqual(features['HomePlayer_3rd_worst'],72)

    def test_odds_and_dates(self):
        for value in ('',None,'nan','inf','1','0'):
            self.assertFalse(odds_ok(dict(B365H=value,B365D=3,B365A=4)))
        self.assertTrue(odds_ok(dict(B365H=2,B365D=3,B365A=4)))
        self.assertEqual(date_value('14/08/93'),'1993-08-14')
        self.assertEqual(date_value('14/08/2008'),'2008-08-14')

    def test_reject_block_page(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'source.csv';p.write_text('<html>Access denied</html>')
            with self.assertRaises(ValueError):csv_rows(p)

if __name__=='__main__':unittest.main()
