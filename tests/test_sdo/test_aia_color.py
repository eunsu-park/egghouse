"""Tests for egghouse.sdo.aia_color module (domain-standard AIA color tables)."""

import numpy as np
import pytest

from egghouse.sdo import (
    aia_color_lut,
    aia_colorize,
    AIA_COLOR_WAVELENGTHS,
)


class TestAiaColorLut:
    """Tests for aia_color_lut (pure-numpy path)."""

    def test_all_wavelengths_shape_dtype(self):
        for wl in AIA_COLOR_WAVELENGTHS:
            lut = aia_color_lut(wl)
            assert lut.shape == (256, 3)
            assert lut.dtype == np.uint8

    def test_endpoints_black_to_bright(self):
        # Every official AIA table starts at black.
        for wl in AIA_COLOR_WAVELENGTHS:
            lut = aia_color_lut(wl)
            assert lut[0].tolist() == [0, 0, 0]

    def test_known_midpoints(self):
        # Spot-check reference values (index 128) for a few channels.
        assert aia_color_lut(171)[128].tolist() == [185, 128, 0]
        assert aia_color_lut(193)[128].tolist() == [181, 128, 64]
        assert aia_color_lut(304)[128].tolist() == [185, 15, 0]

    def test_invalid_wavelength_raises(self):
        with pytest.raises(ValueError):
            aia_color_lut(500)

    def test_invalid_source_raises(self):
        with pytest.raises(ValueError):
            aia_color_lut(171, source="bogus")

    def test_cache_returns_copy(self):
        lut = aia_color_lut(171)
        lut[0] = (9, 9, 9)
        # Mutating the returned array must not corrupt the cache.
        assert aia_color_lut(171)[0].tolist() == [0, 0, 0]


class TestNumpySunpyAgreement:
    """The numpy and sunpy sources must be bit-identical."""

    def test_bit_identical_all_wavelengths(self):
        pytest.importorskip("sunpy")
        pytest.importorskip("matplotlib")
        for wl in AIA_COLOR_WAVELENGTHS:
            a = aia_color_lut(wl, source="numpy")
            b = aia_color_lut(wl, source="sunpy")
            assert np.array_equal(a, b), f"mismatch at {wl}"


class TestAiaColorize:
    """Tests for the high-level aia_colorize."""

    def test_raw_with_exptime(self):
        raw = np.linspace(0, 4000, 32 * 32).reshape(32, 32)
        rgb = aia_colorize(raw, 171, exptime=2.9)
        assert rgb.shape == (32, 32, 3)
        assert rgb.dtype == np.uint8

    def test_prescaled_uint8_matches_lut(self):
        gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
        rgb = aia_colorize(gray, 304)
        # Colorizing a 0..255 ramp reproduces the LUT row-for-row.
        assert np.array_equal(rgb.reshape(-1, 3)[:256], aia_color_lut(304))

    def test_non_uint8_without_exptime_is_bytescaled(self):
        img = np.linspace(0, 1000, 8 * 8).reshape(8, 8)
        rgb = aia_colorize(img, 193)
        assert rgb.shape == (8, 8, 3)
        assert rgb.dtype == np.uint8

    def test_invalid_wavelength_raises(self):
        with pytest.raises(ValueError):
            aia_colorize(np.zeros((4, 4), dtype=np.uint8), 500)
