"""Tests for egghouse.image.transforms (pure functions, no I/O)."""

import numpy as np
import pytest

from egghouse.image import transforms as T


def test_to_float32_changes_dtype_preserves_shape():
    x = np.arange(12, dtype=">i2").reshape(3, 4)
    y = T.to_float32(x)
    assert y.dtype == np.float32
    assert y.shape == (3, 4)
    assert np.array_equal(y, x.astype(np.float32))


def test_compose_applies_in_order():
    add1 = lambda a: a + 1
    times2 = lambda a: a * 2
    f = T.compose([add1, times2])
    assert f(np.array([0, 1, 2]))[0] == 2  # (0+1)*2
    assert f(np.array([3]))[0] == 8  # (3+1)*2


def test_nan_to_value_replaces_non_finite():
    x = np.array([1.0, np.nan, np.inf, -np.inf, 2.0], dtype=np.float32)
    y = T.nan_to_value(0.0)(x)
    assert np.array_equal(y, np.array([1.0, 0.0, 0.0, 0.0, 2.0], dtype=np.float32))


def test_nan_to_value_integer_input_passthrough():
    x = np.arange(5, dtype=np.int16)
    y = T.nan_to_value(0.0)(x)
    assert np.array_equal(y, x)


def test_percentile_clip_clips_outliers():
    x = np.arange(100, dtype=np.float32)
    y = T.percentile_clip(low=5, high=95)(x)
    # float32 percentile interpolation differs by ~1e-6 between the clip
    # bound and a recomputed reference; compare with a small tolerance.
    tol = 1e-4
    assert float(y.min()) >= float(np.percentile(x, 5)) - tol
    assert float(y.max()) <= float(np.percentile(x, 95)) + tol


def test_percentile_clip_rejects_invalid_bounds():
    with pytest.raises(ValueError):
        T.percentile_clip(low=10, high=5)
    with pytest.raises(ValueError):
        T.percentile_clip(low=-1, high=50)
    with pytest.raises(ValueError):
        T.percentile_clip(low=0, high=101)


def test_normalize_minmax_produces_zero_to_one():
    x = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    y = T.normalize_minmax()(x)
    assert float(y.min()) == 0.0
    assert float(y.max()) == 1.0


def test_normalize_minmax_handles_constant_image():
    x = np.full((4, 4), 7.0, dtype=np.float32)
    y = T.normalize_minmax()(x)
    assert np.all(np.isfinite(y))  # no NaN even when range is zero.


def test_normalize_log1p_is_monotonic_and_non_negative():
    x = np.array([0.0, 10.0, 100.0, 1000.0], dtype=np.float32)
    y = T.normalize_log1p()(x)
    assert np.all(np.diff(y) >= 0)
    assert float(y.min()) >= 0.0


def test_circular_mask_zeroes_central_disk():
    x = np.ones((10, 10), dtype=np.float32)
    y = T.circular_mask(radius_frac=0.4)(x)
    assert y[5, 5] == 0.0  # center pixel masked
    assert y[0, 0] == 1.0  # corner preserved


def test_circular_mask_inverse_zeroes_corners():
    x = np.ones((10, 10), dtype=np.float32)
    y = T.circular_mask(radius_frac=0.4, inverse=True)(x)
    assert y[5, 5] == 1.0  # center preserved
    assert y[0, 0] == 0.0  # corner masked


def test_resize_changes_dimensions():
    x = np.arange(100, dtype=np.float32).reshape(10, 10)
    y = T.resize(target_size=(20, 30))(x)
    assert y.shape == (20, 30)


def test_resize_rejects_unknown_interpolation():
    with pytest.raises(ValueError):
        T.resize(target_size=(4, 4), interpolation="warp")
