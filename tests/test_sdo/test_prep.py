"""Tests for egghouse.sdo.prep.

The aiapy-wrapping prep stages (`aia_update_pointing`, `aia_respike`,
`aia_correct_degradation`, `aia_deconvolve`) require live aiapy calls
and real AIA data to run end-to-end, so this file does not exercise
their happy path. Instead we cover:

- mask_out_of_disk on a hand-built sunpy.Map (no network, no aiapy);
- cached_aia_psfs cache-hit deserialization (the cache-miss path
  requires aiapy.psf.psf, which is ~minutes per channel);
- aia_correct_degradation early-return on a non-AIA wavelength;
- aia_deconvolve early-return on a non-AIA wavelength.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# mask_out_of_disk
# ---------------------------------------------------------------------------


def _synthetic_aia_map(*, size: int = 64, radius_px: int = 16, value: float = 7.0):
    """Build a minimal sunpy.Map with the keywords mask_out_of_disk needs."""
    sunpy_map = pytest.importorskip("sunpy.map")
    Map = sunpy_map.Map
    data = np.full((size, size), value, dtype=np.float64)
    meta = {
        "naxis1": size,
        "naxis2": size,
        "crpix1": size / 2 + 0.5,
        "crpix2": size / 2 + 0.5,
        "cdelt1": 0.6,
        "cdelt2": 0.6,
        "cunit1": "arcsec",
        "cunit2": "arcsec",
        "ctype1": "HPLN-TAN",
        "ctype2": "HPLT-TAN",
        "crval1": 0.0,
        "crval2": 0.0,
        "wavelnth": 171,
        "date-obs": "2014-01-01T12:00:00",
        # Provide R_SUN directly so we exercise the R_SUN branch.
        "r_sun": float(radius_px),
    }
    return Map(data, meta)


class TestMaskOutOfDisk:
    def test_off_disk_pixels_filled_with_sentinel(self):
        from egghouse.sdo import mask_out_of_disk

        smap = _synthetic_aia_map(size=64, radius_px=16, value=7.0)
        masked = mask_out_of_disk(smap, fill_value=-5000.0)

        d = masked.data
        cx = cy = 32.5  # crpix
        yy, xx = np.ogrid[: d.shape[0], : d.shape[1]]
        off_disk = (xx - cx) ** 2 + (yy - cy) ** 2 > 16.0 ** 2

        assert (d[off_disk] == -5000.0).all()
        assert (d[~off_disk] == 7.0).all()

    def test_falls_back_to_rsun_obs_over_cdelt1(self):
        """When R_SUN is missing, the function derives radius from
        RSUN_OBS / CDELT1."""
        from egghouse.sdo import mask_out_of_disk

        sunpy_map = pytest.importorskip("sunpy.map")
        Map = sunpy_map.Map
        size = 64
        meta = {
            "naxis1": size,
            "naxis2": size,
            "crpix1": size / 2 + 0.5,
            "crpix2": size / 2 + 0.5,
            "cdelt1": 0.5,  # arcsec/pix
            "cdelt2": 0.5,
            "cunit1": "arcsec",
            "cunit2": "arcsec",
            "ctype1": "HPLN-TAN",
            "ctype2": "HPLT-TAN",
            "crval1": 0.0,
            "crval2": 0.0,
            "wavelnth": 171,
            "date-obs": "2014-01-01T12:00:00",
            # No R_SUN; RSUN_OBS / CDELT1 = 8 -> 16-pixel radius.
            "rsun_obs": 8.0,
        }
        smap = Map(np.ones((size, size), dtype=np.float64), meta)
        masked = mask_out_of_disk(smap, fill_value=0.0)
        # Center pixel must remain on-disk (=1), far corner must be off (=0).
        assert masked.data[size // 2, size // 2] == 1.0
        assert masked.data[0, 0] == 0.0

    def test_missing_radius_keys_raises(self):
        from egghouse.sdo import mask_out_of_disk

        sunpy_map = pytest.importorskip("sunpy.map")
        Map = sunpy_map.Map
        meta = {
            "naxis1": 8,
            "naxis2": 8,
            "crpix1": 4.5,
            "crpix2": 4.5,
            "cdelt1": 0.6,
            "cdelt2": 0.6,
            "cunit1": "arcsec",
            "cunit2": "arcsec",
            "ctype1": "HPLN-TAN",
            "ctype2": "HPLT-TAN",
            "crval1": 0.0,
            "crval2": 0.0,
            "wavelnth": 171,
            "date-obs": "2014-01-01T12:00:00",
        }
        smap = Map(np.zeros((8, 8), dtype=np.float64), meta)
        with pytest.raises(KeyError, match="R_SUN"):
            mask_out_of_disk(smap)


# ---------------------------------------------------------------------------
# cached_aia_psfs — cache-hit only (cache-miss invokes aiapy.psf.psf)
# ---------------------------------------------------------------------------


class TestCachedAiaPsfs:
    def test_cache_hit_returns_pickled_dict(self, tmp_path: Path):
        from egghouse.sdo import cached_aia_psfs

        sentinel = {w: np.full((4, 4), float(w)) for w in (94, 131, 171, 193, 211, 304, 335)}
        cache = tmp_path / "psfs.pkl"
        with open(cache, "wb") as f:
            pickle.dump(sentinel, f)

        loaded = cached_aia_psfs(cache)
        assert set(loaded.keys()) == set(sentinel.keys())
        for w in sentinel:
            np.testing.assert_array_equal(loaded[w], sentinel[w])

    def test_cache_hit_only_returns_requested_wavelengths(self, tmp_path: Path):
        from egghouse.sdo import cached_aia_psfs

        sentinel = {w: np.zeros((2, 2)) + w for w in (94, 131, 171, 193, 211, 304, 335)}
        cache = tmp_path / "psfs.pkl"
        with open(cache, "wb") as f:
            pickle.dump(sentinel, f)

        loaded = cached_aia_psfs(cache, wavelengths=[171, 193])
        assert set(loaded.keys()) == {171, 193}


# ---------------------------------------------------------------------------
# Wavelength-guarded prep stages (early-return paths)
# ---------------------------------------------------------------------------


class TestWavelengthGuards:
    """For wavelengths outside the standard AIA EUV+304 set, the
    degradation and PSF deconvolution prep stages must return the input
    map unchanged — no aiapy call is attempted."""

    def test_correct_degradation_skips_non_aia_wavelength(self):
        from egghouse.sdo import aia_correct_degradation

        class _SentinelMap:
            meta = {"wavelnth": 9999}

        smap = _SentinelMap()
        assert aia_correct_degradation(smap) is smap

    def test_deconvolve_skips_non_aia_wavelength(self):
        from egghouse.sdo import aia_deconvolve

        class _SentinelMap:
            meta = {"wavelnth": 9999}

        smap = _SentinelMap()
        assert aia_deconvolve(smap) is smap
