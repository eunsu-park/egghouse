"""Tests for egghouse.image.spatial (padding, cropping, rolling, binning)."""

from __future__ import annotations

import numpy as np
import pytest

from egghouse.image import bin_ndarray


class TestBinNdarray:
    def test_2d_sum_matches_doctest(self):
        m = np.arange(100).reshape((10, 10))
        binned = bin_ndarray(m, (5, 5), operation="sum")
        # Each output cell is the sum of a 2x2 block of the source.
        expected = np.array(
            [
                [22, 30, 38, 46, 54],
                [102, 110, 118, 126, 134],
                [182, 190, 198, 206, 214],
                [262, 270, 278, 286, 294],
                [342, 350, 358, 366, 374],
            ]
        )
        assert binned.shape == (5, 5)
        np.testing.assert_array_equal(binned, expected)

    def test_2d_mean_is_sum_over_block_size(self):
        m = np.arange(100, dtype=np.float64).reshape((10, 10))
        mean = bin_ndarray(m, (5, 5), operation="mean")
        sum_ = bin_ndarray(m, (5, 5), operation="sum")
        # block_size = (10/5) * (10/5) = 4
        np.testing.assert_allclose(mean * 4, sum_)

    def test_3d_array(self):
        m = np.ones((4, 6, 8), dtype=np.int64)
        binned = bin_ndarray(m, (2, 3, 4), operation="sum")
        # block size = 2 * 2 * 2 = 8
        assert binned.shape == (2, 3, 4)
        assert (binned == 8).all()

    def test_unsupported_operation_raises(self):
        with pytest.raises(ValueError, match="operation"):
            bin_ndarray(np.zeros((4, 4)), (2, 2), operation="max")

    def test_ndim_mismatch_raises(self):
        with pytest.raises(ValueError, match="match array.ndim"):
            bin_ndarray(np.zeros((4, 4)), (2,), operation="sum")

    def test_indivisible_dimension_raises(self):
        with pytest.raises(ValueError, match="divide"):
            bin_ndarray(np.zeros((10, 10)), (3, 5), operation="sum")  # 10/3 non-integer

    def test_zero_target_dimension_raises(self):
        with pytest.raises(ValueError, match="divide"):
            bin_ndarray(np.zeros((4, 4)), (0, 2), operation="sum")
