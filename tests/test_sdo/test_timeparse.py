"""Characterization tests for the FITS-datetime parsers (egghouse.sdo.timeparse).

Expected values were captured from the original solaris-data/core/parse.py
implementation before promotion, locking behavior across the move. The pure
string/filename parsers need no astropy; parse_fits_datetime is exercised on
its filename-fallback path (nonexistent FITS path -> fits.open fails -> the
filename pattern is used), which needs no real FITS file.
"""
from __future__ import annotations

import unittest
from datetime import datetime

from egghouse.sdo.timeparse import (
    _parse_tai_datetime,
    _parse_datetime_string,
    _parse_datetime_from_filename,
    parse_fits_datetime,
)


class TestParseTaiDatetime(unittest.TestCase):
    def test_cases(self):
        cases = {
            "2024.01.01_00:00:00_TAI": datetime(2024, 1, 1, 0, 0),
            "2024.01.01_00:00:00.500_TAI": datetime(2024, 1, 1, 0, 0, 0, 500000),
            "2024-01-01T00:00:00Z": None,   # no _TAI
            "": None,
            "garbage_TAI": None,
        }
        for s, expect in cases.items():
            self.assertEqual(_parse_tai_datetime(s), expect, msg=s)


class TestParseDatetimeString(unittest.TestCase):
    def test_cases(self):
        cases = [
            ("2024.01.01_00:00:00_TAI", datetime(2024, 1, 1, 0, 0)),
            ("2024-01-01T12:34:56.789", datetime(2024, 1, 1, 12, 34, 56, 789000)),
            ("2024-01-01T12:34:56Z", datetime(2024, 1, 1, 12, 34, 56)),
            ("2024/01/01T12:34:56", datetime(2024, 1, 1, 12, 34, 56)),
            ("2024-01-01 12:34:56", datetime(2024, 1, 1, 12, 34, 56)),
            ("2024/01/01 12:34:56.250", datetime(2024, 1, 1, 12, 34, 56, 250000)),
            ("01/02/24 12:34:56", datetime(2024, 2, 1, 12, 34, 56)),   # dd/mm/yy
            ("01/02/2024 12:34:56", datetime(2024, 2, 1, 12, 34, 56)),
            ("2024-01-01", datetime(2024, 1, 1, 0, 0)),
            ("2024/01/01", datetime(2024, 1, 1, 0, 0)),
            ("01/02/24", datetime(2024, 2, 1, 0, 0)),
            ("01/02/2024", datetime(2024, 2, 1, 0, 0)),
            ("", None),
            (None, None),
            ("hello", None),
        ]
        for s, expect in cases:
            self.assertEqual(_parse_datetime_string(s), expect, msg=repr(s))


class TestParseDatetimeFromFilename(unittest.TestCase):
    def test_cases(self):
        aia = "aia.lev1_euv_12s.2010-09-01T000008Z.193.image_lev1.fits"
        hmi = "hmi.m_45s.20100901_000000_TAI.2.magnetogram.fits"
        self.assertEqual(_parse_datetime_from_filename(aia),
                         datetime(2010, 9, 1, 0, 0, 8))
        self.assertEqual(_parse_datetime_from_filename(hmi),
                         datetime(2010, 9, 1, 0, 0))
        self.assertEqual(_parse_datetime_from_filename("/data/sdo/" + aia),
                         datetime(2010, 9, 1, 0, 0, 8))
        self.assertIsNone(_parse_datetime_from_filename("random_file.fits"))


class TestParseFitsDatetimeFallback(unittest.TestCase):
    def test_filename_fallback(self):
        # Nonexistent path -> fits.open raises -> filename fallback.
        aia = "some/path/aia.lev1_euv_12s.2010-09-01T000008Z.193.image_lev1.fits"
        hmi = "some/path/hmi.m_45s.20100901_000000_TAI.2.magnetogram.fits"
        self.assertEqual(parse_fits_datetime(aia), datetime(2010, 9, 1, 0, 0, 8))
        self.assertEqual(parse_fits_datetime(hmi), datetime(2010, 9, 1, 0, 0))
        self.assertIsNone(parse_fits_datetime("some/path/nomatch.fits"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
