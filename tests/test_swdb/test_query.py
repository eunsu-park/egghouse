"""Tests for the `sdo` read queries, with `PostgresManager` mocked out."""

from __future__ import annotations

from datetime import datetime
from unittest import mock

import pandas as pd

from egghouse.swdb import get_sdo_best_match, get_sdo_best_matches


class _FakeDB:
    """Stand-in for PostgresManager used as a context manager.

    Records the last execute() call and returns a scripted result.
    """

    def __init__(self, result):
        self._result = result
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params, fetch):
        self.calls.append({"sql": sql, "params": params, "fetch": fetch})
        return self._result


def _patch_pm(fake):
    """Patch the PostgresManager symbol in the query module."""
    return mock.patch("egghouse.swdb.query.PostgresManager", return_value=fake)


DB_CONFIG = {"host": "x", "user": "u", "password": "p", "database": "d"}
ROW = {"file_path": "/arch/aia.193.fits", "channel": "193", "quality": 0}


class TestGetSdoBestMatch:
    def test_returns_first_row(self):
        fake = _FakeDB([ROW])
        with _patch_pm(fake):
            out = get_sdo_best_match(DB_CONFIG, "aia", "193", datetime(2014, 1, 1, 12))
        assert out == ROW
        call = fake.calls[0]
        # telescope, channel, start, end, target — five bound params.
        assert call["params"][0] == "aia"
        assert call["params"][1] == "193"
        assert len(call["params"]) == 5
        assert call["fetch"] is True

    def test_none_when_empty(self):
        fake = _FakeDB([])
        with _patch_pm(fake):
            out = get_sdo_best_match(DB_CONFIG, "aia", "193", datetime(2014, 1, 1, 12))
        assert out is None

    def test_quality_condition_toggles(self):
        fake = _FakeDB([ROW])
        with _patch_pm(fake):
            get_sdo_best_match(DB_CONFIG, "aia", "193", datetime(2014, 1, 1, 12))
        assert "quality = 0" in fake.calls[0]["sql"]

        fake2 = _FakeDB([ROW])
        with _patch_pm(fake2):
            get_sdo_best_match(
                DB_CONFIG, "aia", "193", datetime(2014, 1, 1, 12),
                require_quality_zero=False,
            )
        assert "quality = 0" not in fake2.calls[0]["sql"]

    def test_window_brackets_target(self):
        fake = _FakeDB([ROW])
        target = datetime(2014, 1, 1, 12, 0, 0)
        with _patch_pm(fake):
            get_sdo_best_match(DB_CONFIG, "aia", "193", target, time_range_minutes=30)
        _, _, start, end, t = fake.calls[0]["params"]
        assert start < target < end
        assert (end - start).total_seconds() == 60 * 60  # ±30 min
        assert t == target


class TestGetSdoBestMatches:
    def test_drops_misses_and_tags_target(self):
        target_hit = datetime(2014, 1, 1, 12)
        target_miss = datetime(2014, 1, 2, 12)

        # First target hits, second misses.
        def fake_single(db_config, telescope, channel, target_time, *a, **k):
            return dict(ROW) if target_time == target_hit else None

        with mock.patch("egghouse.swdb.query.get_sdo_best_match", side_effect=fake_single):
            df = get_sdo_best_matches(DB_CONFIG, "aia", "193", [target_hit, target_miss])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["target_time"] == target_hit

    def test_empty_dataframe_when_no_hits(self):
        with mock.patch("egghouse.swdb.query.get_sdo_best_match", return_value=None):
            df = get_sdo_best_matches(DB_CONFIG, "aia", "193", [datetime(2014, 1, 1)])
        assert isinstance(df, pd.DataFrame)
        assert df.empty
