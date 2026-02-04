"""Tests for egghouse.image.stats module."""

import numpy as np
import pytest

from egghouse.image import (
    normalize_image,
    get_image_stats,
    histogram_equalization,
    percentile_scale,
    find_disk_center,
)


class TestNormalizeImage:
    """Tests for normalize_image function."""

    def test_normalize_basic(self, sample_image_float):
        """Test basic normalization."""
        normalized = normalize_image(sample_image_float)
        assert np.isclose(np.mean(normalized), 0, atol=1e-10)
        assert np.isclose(np.std(normalized), 1, atol=1e-10)

    def test_normalize_output_dtype(self, sample_image_2d):
        """Test that output is float64."""
        normalized = normalize_image(sample_image_2d)
        assert normalized.dtype == np.float64

    def test_normalize_custom_stats(self):
        """Test normalization with custom mean/std."""
        data = np.array([100, 200, 300], dtype=np.float32)
        normalized = normalize_image(data, mean=200, std=100)
        expected = np.array([-1, 0, 1])
        np.testing.assert_array_almost_equal(normalized, expected)

    def test_normalize_preserves_shape(self, sample_image_float):
        """Test that shape is preserved."""
        normalized = normalize_image(sample_image_float)
        assert normalized.shape == sample_image_float.shape

    def test_normalize_constant_image(self):
        """Test normalization of constant image (std=0)."""
        data = np.ones((10, 10)) * 100
        # Should not raise an error, std is replaced with small value
        normalized = normalize_image(data)
        assert not np.any(np.isnan(normalized))


class TestGetImageStats:
    """Tests for get_image_stats function."""

    def test_stats_basic(self, sample_image_float):
        """Test basic statistics computation."""
        stats = get_image_stats(sample_image_float)
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'median' in stats
        assert 'count' in stats

    def test_stats_values(self):
        """Test that statistics are computed correctly."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        stats = get_image_stats(data)
        assert stats['mean'] == 5.0
        assert stats['min'] == 1.0
        assert stats['max'] == 9.0
        assert stats['median'] == 5.0
        assert stats['count'] == 9

    def test_stats_with_mask(self, sample_image_float):
        """Test statistics with mask."""
        mask = np.zeros(sample_image_float.shape, dtype=bool)
        mask[10:20, 10:20] = True
        stats = get_image_stats(sample_image_float, mask=mask)
        assert stats['count'] == 100  # 10x10 region

    def test_stats_percentiles_default(self, sample_image_float):
        """Test default percentile computation."""
        stats = get_image_stats(sample_image_float)
        assert 'p1' in stats
        assert 'p5' in stats
        assert 'p25' in stats
        assert 'p50' in stats
        assert 'p75' in stats
        assert 'p95' in stats
        assert 'p99' in stats

    def test_stats_custom_percentiles(self, sample_image_float):
        """Test custom percentiles."""
        stats = get_image_stats(sample_image_float, percentiles=(10, 90))
        assert 'p10' in stats
        assert 'p90' in stats
        assert 'p50' not in stats

    def test_stats_p50_equals_median(self, sample_image_float):
        """Test that p50 equals median."""
        stats = get_image_stats(sample_image_float)
        assert np.isclose(stats['p50'], stats['median'])


class TestHistogramEqualization:
    """Tests for histogram_equalization function."""

    def test_histogram_equalization_basic(self, sample_image_2d):
        """Test basic histogram equalization."""
        enhanced = histogram_equalization(sample_image_2d)
        assert enhanced.shape == sample_image_2d.shape
        assert enhanced.dtype == np.uint8

    def test_histogram_equalization_output_range(self, sample_image_2d):
        """Test that output is in valid range."""
        enhanced = histogram_equalization(sample_image_2d)
        assert enhanced.min() >= 0
        assert enhanced.max() <= 255

    def test_histogram_equalization_improves_contrast(self):
        """Test that equalization improves contrast."""
        # Create low-contrast image
        low_contrast = np.random.randint(100, 150, size=(64, 64), dtype=np.uint8)
        enhanced = histogram_equalization(low_contrast)
        # Enhanced should have larger range
        assert enhanced.max() - enhanced.min() > low_contrast.max() - low_contrast.min()

    def test_histogram_equalization_invalid_ndim(self):
        """Test error on 3D image."""
        img_3d = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="2D"):
            histogram_equalization(img_3d)


class TestPercentileScale:
    """Tests for percentile_scale function."""

    def test_percentile_scale_basic(self, sample_image_float):
        """Test basic percentile scaling."""
        scaled = percentile_scale(sample_image_float)
        assert scaled.dtype == np.uint8
        assert scaled.min() >= 0
        assert scaled.max() <= 255

    def test_percentile_scale_custom_percentiles(self, sample_image_float):
        """Test custom percentile values."""
        scaled = percentile_scale(
            sample_image_float, low_percentile=5, high_percentile=95
        )
        assert scaled.dtype == np.uint8

    def test_percentile_scale_custom_output_range(self, sample_image_float):
        """Test custom output range."""
        scaled = percentile_scale(sample_image_float, omin=10, omax=200)
        assert scaled.min() >= 10
        assert scaled.max() <= 200

    def test_percentile_scale_handles_outliers(self):
        """Test that percentile scaling handles outliers."""
        # Create data with outliers
        data = np.random.randn(100, 100) * 10 + 100
        data[0, 0] = 10000  # Extreme outlier
        data[99, 99] = -10000

        scaled = percentile_scale(data, low_percentile=1, high_percentile=99)
        # Most values should use full range despite outliers
        assert scaled[50, 50] > 10
        assert scaled[50, 50] < 245


class TestFindDiskCenter:
    """Tests for find_disk_center function."""

    def test_find_center_centered_disk(self):
        """Test finding center of a centered disk."""
        # Create image with centered bright disk
        size = 100
        img = np.zeros((size, size), dtype=np.float32)
        y, x = np.ogrid[:size, :size]
        center = size // 2
        mask = (x - center) ** 2 + (y - center) ** 2 < 30**2
        img[mask] = 1000

        cy, cx = find_disk_center(img)
        assert np.isclose(cy, 50, atol=1)
        assert np.isclose(cx, 50, atol=1)

    def test_find_center_off_center_disk(self):
        """Test finding center of an off-center disk."""
        size = 100
        img = np.zeros((size, size), dtype=np.float32)
        y, x = np.ogrid[:size, :size]
        target_cy, target_cx = 30, 70
        mask = (x - target_cx) ** 2 + (y - target_cy) ** 2 < 20**2
        img[mask] = 1000

        cy, cx = find_disk_center(img)
        assert np.isclose(cy, target_cy, atol=2)
        assert np.isclose(cx, target_cx, atol=2)

    def test_find_center_with_threshold(self):
        """Test with custom threshold."""
        # Create solar disk image
        size = 128
        center = size // 2
        radius = size // 3
        yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        disk = np.sqrt((xx - center) ** 2 + (yy - center) ** 2) < radius
        image = np.zeros((size, size), dtype=np.float32)
        image[disk] = 1000

        cy, cx = find_disk_center(image, threshold=500)
        # Should find center of the disk region
        assert 50 < cy < 80
        assert 50 < cx < 80

    def test_find_center_geometric_method(self):
        """Test geometric center method."""
        size = 100
        img = np.zeros((size, size), dtype=np.float32)
        y, x = np.ogrid[:size, :size]
        mask = (x - 50) ** 2 + (y - 50) ** 2 < 30**2
        img[mask] = 1000

        cy, cx = find_disk_center(img, method='geometric')
        assert np.isclose(cy, 50, atol=1)
        assert np.isclose(cx, 50, atol=1)

    def test_find_center_invalid_method(self):
        """Test error on invalid method."""
        img = np.zeros((100, 100), dtype=np.float32)
        img[40:60, 40:60] = 1000
        with pytest.raises(ValueError, match="method"):
            find_disk_center(img, method='invalid')

    def test_find_center_invalid_ndim(self):
        """Test error on non-2D image."""
        img_3d = np.zeros((10, 10, 3))
        with pytest.raises(ValueError, match="2D"):
            find_disk_center(img_3d)
