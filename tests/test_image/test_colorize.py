"""Tests for egghouse.image.colorize module."""

import numpy as np
import pytest

from egghouse.image import apply_colormap, lut_from_matplotlib


def gray_ramp_lut():
    """A (256, 3) grayscale-ramp LUT: row i -> (i, i, i)."""
    return np.stack([np.arange(256)] * 3, axis=1).astype(np.uint8)


class TestApplyColormap:
    """Tests for apply_colormap."""

    def test_output_shape_and_dtype(self):
        gray = np.zeros((8, 5), dtype=np.uint8)
        rgb = apply_colormap(gray, gray_ramp_lut())
        assert rgb.shape == (8, 5, 3)
        assert rgb.dtype == np.uint8

    def test_lookup_is_exact(self):
        gray = np.array([[0, 128], [255, 64]], dtype=np.uint8)
        lut = gray_ramp_lut()
        rgb = apply_colormap(gray, lut)
        # ramp LUT reproduces the value in every channel
        assert np.array_equal(rgb[..., 0], gray)
        assert np.array_equal(rgb[..., 1], gray)
        assert np.array_equal(rgb[..., 2], gray)

    def test_arbitrary_lut(self):
        gray = np.array([[0, 1, 2]], dtype=np.uint8)
        lut = np.zeros((256, 3), dtype=np.uint8)
        lut[0] = (10, 20, 30)
        lut[1] = (40, 50, 60)
        lut[2] = (70, 80, 90)
        rgb = apply_colormap(gray, lut)
        assert rgb[0, 0].tolist() == [10, 20, 30]
        assert rgb[0, 1].tolist() == [40, 50, 60]
        assert rgb[0, 2].tolist() == [70, 80, 90]

    def test_non_uint8_input_is_clipped(self):
        gray = np.array([[-5.0, 128.0, 300.0]])  # out-of-range floats
        rgb = apply_colormap(gray, gray_ramp_lut())
        # -5 -> 0, 300 -> 255
        assert rgb[0, 0].tolist() == [0, 0, 0]
        assert rgb[0, 2].tolist() == [255, 255, 255]

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            apply_colormap(np.zeros((4, 4, 3), dtype=np.uint8), gray_ramp_lut())

    def test_rejects_bad_lut_shape(self):
        with pytest.raises(ValueError):
            apply_colormap(np.zeros((4, 4), dtype=np.uint8), np.zeros((128, 3)))


class TestLutFromMatplotlib:
    """Tests for lut_from_matplotlib (requires matplotlib)."""

    def test_named_colormap(self):
        pytest.importorskip("matplotlib")
        lut = lut_from_matplotlib("viridis")
        assert lut.shape == (256, 3)
        assert lut.dtype == np.uint8

    def test_feeds_apply_colormap(self):
        pytest.importorskip("matplotlib")
        lut = lut_from_matplotlib("inferno")
        gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
        rgb = apply_colormap(gray, lut)
        assert rgb.shape == (16, 16, 3)
