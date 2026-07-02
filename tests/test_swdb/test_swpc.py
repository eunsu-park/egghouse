"""Tests for the SWPC real-time parsers (egghouse.swdb.swpc).

Pure-function tests over representative payloads matching the live SWPC schemas
(verified 2026-06-26). Promoted with the parsers from solaris-data.
"""
from __future__ import annotations

import unittest

import pandas as pd

from egghouse.swdb.swpc import (
    _to_utc_naive, parse_xray, parse_proton, parse_solar_wind, parse_kp_1m,
    parse_kp_forecast, parse_solar_probabilities, parse_alerts, parse_3day_forecast,
)


class TestTimeParsing(unittest.TestCase):
    def test_z_and_bare_tags_both_naive_utc(self):
        # Regression: across feeds SWPC uses 'Z' (tz-aware, e.g. X-ray) and bare
        # tags (e.g. solar wind); each is UTC and must land tz-naive at the same
        # clock time — the KST-shift bug. (Each feed is homogeneous, so tested
        # one format per call, as in production.)
        z = _to_utc_naive(pd.Series(["2026-06-26T13:20:00Z"]))
        bare = _to_utc_naive(pd.Series(["2026-06-26 13:20:00.000"]))
        self.assertIsNone(z.dt.tz)
        self.assertIsNone(bare.dt.tz)
        self.assertEqual(z.iloc[0], pd.Timestamp("2026-06-26 13:20:00"))
        self.assertEqual(bare.iloc[0], pd.Timestamp("2026-06-26 13:20:00"))


class TestXray(unittest.TestCase):
    DATA = [
        {"time_tag": "2026-06-26T13:20:00Z", "satellite": 18, "flux": 2.4e-8, "energy": "0.05-0.4nm"},
        {"time_tag": "2026-06-26T13:20:00Z", "satellite": 18, "flux": 1.3e-6, "energy": "0.1-0.8nm"},
    ]

    def test_pivots_short_long(self):
        df = parse_xray(self.DATA)
        self.assertEqual(len(df), 1)  # two bands → one row
        row = df.iloc[0]
        self.assertEqual(row["satellite"], 18)
        self.assertAlmostEqual(row["xrs_short_w_m2"], 2.4e-8)
        self.assertAlmostEqual(row["xrs_long_w_m2"], 1.3e-6)
        self.assertIsNone(df["datetime"].dt.tz)
        self.assertEqual(row["datetime"], pd.Timestamp("2026-06-26 13:20:00"))

    def test_empty(self):
        self.assertTrue(parse_xray([]).empty)


class TestProton(unittest.TestCase):
    def test_long_format(self):
        data = [{"time_tag": "2026-06-26T13:20:00Z", "satellite": 18,
                 "flux": 0.22, "energy": ">=10 MeV"}]
        df = parse_proton(data)
        self.assertEqual(list(df.columns), ["satellite", "datetime", "energy", "flux"])
        self.assertEqual(df.iloc[0]["energy"], ">=10 MeV")
        self.assertAlmostEqual(df.iloc[0]["flux"], 0.22)


class TestSolarWind(unittest.TestCase):
    def test_plasma(self):
        data = [["time_tag", "density", "speed", "temperature"],
                ["2026-06-26 13:20:00.000", "5.2", "400.0", "100000"]]
        df = parse_solar_wind(data, "plasma")
        r = df.iloc[0]
        self.assertAlmostEqual(r["density_p_cc"], 5.2)
        self.assertAlmostEqual(r["speed_km_s"], 400.0)
        self.assertAlmostEqual(r["temperature_k"], 100000.0)
        self.assertEqual(r["source"], "DSCOVR")
        self.assertEqual(r["datetime"], pd.Timestamp("2026-06-26 13:20:00"))

    def test_mag(self):
        data = [["time_tag", "bx_gsm", "by_gsm", "bz_gsm", "lon_gsm", "lat_gsm", "bt"],
                ["2026-06-26 13:20:00.000", "1.0", "2.0", "-3.0", "339.9", "55.8", "4.0"]]
        df = parse_solar_wind(data, "mag")
        r = df.iloc[0]
        self.assertAlmostEqual(r["bz_gsm_nt"], -3.0)
        self.assertAlmostEqual(r["bt_nt"], 4.0)

    def test_short_payload_empty(self):
        self.assertTrue(parse_solar_wind([["time_tag", "density"]], "plasma").empty)


class TestKp(unittest.TestCase):
    def test_kp_1m(self):
        data = [{"time_tag": "2026-06-26T13:20:00", "kp_index": 1,
                 "estimated_kp": 1.33, "kp": "1P"}]
        df = parse_kp_1m(data)
        self.assertAlmostEqual(df.iloc[0]["estimated_kp"], 1.33)
        self.assertEqual(df.iloc[0]["datetime"], pd.Timestamp("2026-06-26 13:20:00"))

    def test_kp_forecast_null_scale(self):
        data = [["time_tag", "kp", "observed", "noaa_scale"],
                ["2026-06-26 12:00:00", "3.33", "observed", "null"],
                ["2026-06-29 00:00:00", "5.00", "predicted", "G1"]]
        df = parse_kp_forecast(data)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["observed_flag"], "observed")
        self.assertTrue(pd.isna(df.iloc[0]["noaa_scale"]))  # 'null' → NaN (→ NULL on upsert)
        self.assertEqual(df.iloc[1]["noaa_scale"], "G1")


class TestProbabilities(unittest.TestCase):
    def test_mapping(self):
        data = [{"date": "2026-06-26", "c_class_1_day": 99, "m_class_1_day": 50,
                 "x_class_1_day": 10, "c_class_2_day": 95, "m_class_2_day": 40,
                 "x_class_2_day": 5, "c_class_3_day": 90, "m_class_3_day": 30,
                 "x_class_3_day": 5, "10mev_protons_1_day": 5, "10mev_protons_2_day": 5,
                 "10mev_protons_3_day": 5, "polar_cap_absorption": "green"}]
        df = parse_solar_probabilities(data)
        r = df.iloc[0]
        self.assertEqual(r["c_class_1_day"], 99)
        self.assertEqual(r["proton_10mev_1_day"], 5)      # 10mev_protons → proton_10mev
        self.assertEqual(r["polar_cap_absorption"], "green")
        self.assertEqual(str(r["valid_date"]), "2026-06-26")


class TestAlerts(unittest.TestCase):
    def test_parse_and_dropna(self):
        data = [
            {"product_id": "K04W", "issue_datetime": "2026-06-25 23:43:25.917", "message": "WATCH ..."},
            {"product_id": None, "issue_datetime": None, "message": "bad"},
        ]
        df = parse_alerts(data)
        self.assertEqual(len(df), 1)  # the null row dropped
        self.assertEqual(df.iloc[0]["product_id"], "K04W")


class TestThreeDayForecast(unittest.TestCase):
    def test_issue_stamp(self):
        text = ":Product: 3-Day Forecast\n:Issued: 2026 Jun 26 1230 UTC\n# ...\nbody"
        df = parse_3day_forecast(text)
        self.assertEqual(df.iloc[0]["issued_at"], pd.Timestamp("2026-06-26 12:30:00"))
        self.assertEqual(df.iloc[0]["raw_text"], text)

    def test_no_stamp_empty(self):
        self.assertTrue(parse_3day_forecast("no issue line here").empty)


if __name__ == "__main__":
    unittest.main(verbosity=2)
