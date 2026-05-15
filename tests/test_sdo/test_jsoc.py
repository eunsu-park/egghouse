"""Tests for egghouse.sdo.jsoc.

Network-bound paths (`jsoc_export`, the aiapy cache miss path) are not
exercised here — those require live JSOC and aiapy network access. The
local pieces — query composition and cache-hit reads — are tested
fully.
"""

from __future__ import annotations

import pickle
from datetime import datetime, timedelta
from pathlib import Path

import pytest


class TestAiaEuvQuery:
    def test_single_timestamp_default_channels(self):
        from egghouse.sdo.jsoc import aia_euv_query

        q = aia_euv_query([datetime(2014, 1, 1, 12, 0, 0)])
        assert q.startswith("aia.lev1_euv_12s[2014.01.01_12:00:00_TAI/12s]")
        # All six DEM channels must appear in the predicate.
        for wave in (94, 131, 171, 193, 211, 335):
            assert f"WAVELNTH={wave}" in q
        # Predicate must be wrapped in [? ... ?].
        assert "[? " in q and " ?]" in q

    def test_multiple_timestamps_are_concatenated(self):
        from egghouse.sdo.jsoc import aia_euv_query

        times = [
            datetime(2014, 1, 1, 0, 0, 0),
            datetime(2014, 1, 1, 6, 0, 0),
            datetime(2014, 1, 1, 12, 0, 0),
        ]
        q = aia_euv_query(times)
        for t in times:
            assert f"[{t.strftime('%Y.%m.%d_%H:%M:%S')}_TAI/12s]" in q

    def test_custom_wavelengths_and_tolerance(self):
        from egghouse.sdo.jsoc import aia_euv_query

        q = aia_euv_query(
            [datetime(2014, 1, 1, 0, 0, 0)],
            wavelengths=[171, 193],
            tolerance=timedelta(seconds=60),
        )
        assert "/60s]" in q
        # 171 and 193 in predicate; other channels absent.
        assert "WAVELNTH=171" in q
        assert "WAVELNTH=193" in q
        assert "WAVELNTH=94" not in q
        assert "WAVELNTH=335" not in q

    def test_custom_series(self):
        from egghouse.sdo.jsoc import aia_euv_query

        q = aia_euv_query(
            [datetime(2014, 1, 1, 0, 0, 0)],
            series="aia.lev1_uv_24s",
        )
        assert q.startswith("aia.lev1_uv_24s[")

    def test_empty_times_raises(self):
        from egghouse.sdo.jsoc import aia_euv_query

        with pytest.raises(ValueError, match="non-empty"):
            aia_euv_query([])

    def test_empty_wavelengths_raises(self):
        from egghouse.sdo.jsoc import aia_euv_query

        with pytest.raises(ValueError, match="non-empty"):
            aia_euv_query([datetime(2014, 1, 1)], wavelengths=[])

    def test_non_positive_tolerance_raises(self):
        from egghouse.sdo.jsoc import aia_euv_query

        with pytest.raises(ValueError, match="positive"):
            aia_euv_query([datetime(2014, 1, 1)], tolerance=timedelta(0))


class TestCachedTables:
    """Cache-hit paths for the aiapy table caches.

    The cache-miss path requires a live JSOC connection plus aiapy and is
    not exercised here; we verify only that an existing pickle is
    deserialized as-is and that the network path is not touched.
    """

    def test_correction_table_loads_from_existing_pickle(self, tmp_path: Path):
        from egghouse.sdo.jsoc import cached_correction_table

        sentinel = {"_test": "correction-sentinel"}
        cache = tmp_path / "correction_table.pkl"
        with open(cache, "wb") as f:
            pickle.dump(sentinel, f)

        loaded = cached_correction_table(cache)
        assert loaded == sentinel

    def test_pointing_table_loads_from_existing_pickle(self, tmp_path: Path):
        from egghouse.sdo.jsoc import cached_pointing_table

        sentinel = {"_test": "pointing-sentinel"}
        cache = tmp_path / "pointing_table.pkl"
        with open(cache, "wb") as f:
            pickle.dump(sentinel, f)

        # start / end are ignored on cache hit; passing arbitrary values.
        loaded = cached_pointing_table(
            cache,
            start=datetime(2014, 1, 1),
            end=datetime(2014, 1, 2),
        )
        assert loaded == sentinel


class TestJsocExportWrapper:
    """jsoc_export() error handling — drives drms via a stub client so no
    network access is needed."""

    def test_failed_export_raises_runtime_error(self):
        from egghouse.sdo import jsoc

        class _FailedRequest:
            status = "ERROR"

            def wait(self):
                return None

            def has_succeeded(self):
                return False

        class _StubClient:
            def export(self, ds, **kwargs):
                return _FailedRequest()

        with pytest.raises(RuntimeError, match="did not succeed"):
            jsoc.jsoc_export(
                "aia.lev1_euv_12s[2014.01.01_12:00:00_TAI/12s][? WAVELNTH=171 ?]",
                email="dummy@example.com",
                client=_StubClient(),
            )

    def test_successful_export_returns_url_list(self):
        from egghouse.sdo import jsoc

        class _Urls:
            url = ["https://jsoc.example/a.fits", "https://jsoc.example/b.fits"]

        class _OkRequest:
            status = "DONE"
            urls = _Urls()

            def wait(self):
                return None

            def has_succeeded(self):
                return True

        class _StubClient:
            def __init__(self):
                self.calls = []

            def export(self, ds, **kwargs):
                self.calls.append((ds, kwargs))
                return _OkRequest()

        stub = _StubClient()
        out = jsoc.jsoc_export(
            "aia.lev1_euv_12s[2014.01.01_12:00:00_TAI/12s][? WAVELNTH=171 ?]",
            email="dummy@example.com",
            client=stub,
        )
        assert out == ["https://jsoc.example/a.fits", "https://jsoc.example/b.fits"]
        assert stub.calls and stub.calls[0][1]["method"] == "url"
        assert stub.calls[0][1]["protocol"] == "fits"
