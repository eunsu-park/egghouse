"""Tests for egghouse.image.core module."""

import numpy as np
import pytest

from egghouse.image import resize_image, rotate_image, bytescale_image


class TestResizeImage:
    """Tests for resize_image function."""

    def test_resize_basic(self, sample_image_2d):
        """Test basic resize operation."""
        resized = resize_image(sample_image_2d, (32, 32))
        assert resized.shape == (32, 32)
        assert resized.dtype == sample_image_2d.dtype

    def test_resize_upscale(self, sample_image_2d):
        """Test upscaling."""
        resized = resize_image(sample_image_2d, (128, 128))
        assert resized.shape == (128, 128)

    def test_resize_preserve_dtype_uint8(self, sample_image_2d):
        """Test dtype preservation for uint8."""
        resized = resize_image(sample_image_2d, (32, 32))
        assert resized.dtype == np.uint8

    def test_resize_preserve_dtype_float32(self, sample_image_float):
        """Test dtype preservation for float32."""
        resized = resize_image(sample_image_float, (32, 32))
        assert resized.dtype == np.float32

    def test_resize_3d_image(self, sample_image_3d):
        """Test 3D (RGB) image resize."""
        resized = resize_image(sample_image_3d, (32, 32))
        assert resized.shape == (32, 32, 3)
        assert resized.dtype == sample_image_3d.dtype

    def test_resize_different_orders(self, sample_image_float):
        """Test different interpolation orders."""
        for order in [0, 1, 2, 3]:
            resized = resize_image(sample_image_float, (32, 32), order=order)
            assert resized.shape == (32, 32)

    def test_resize_invalid_ndim(self):
        """Test error on invalid dimensions."""
        img_4d = np.zeros((2, 10, 10, 3))
        with pytest.raises(ValueError, match="2D or 3D"):
            resize_image(img_4d, (5, 5))

    def test_resize_1d_error(self):
        """Test error on 1D array."""
        img_1d = np.zeros((100,))
        with pytest.raises(ValueError, match="2D or 3D"):
            resize_image(img_1d, (50, 50))


class TestRotateImage:
    """Tests for rotate_image function."""

    def test_rotate_basic(self, sample_image_2d):
        """Test basic rotation."""
        rotated = rotate_image(sample_image_2d, 45)
        assert rotated.shape == sample_image_2d.shape
        assert rotated.dtype == sample_image_2d.dtype

    def test_rotate_90_degrees(self, sample_image_2d):
        """Test 90 degree rotation."""
        rotated = rotate_image(sample_image_2d, 90)
        assert rotated.shape == sample_image_2d.shape

    def test_rotate_360_degrees(self, sample_image_float):
        """Test 360 degree rotation (should be similar to original)."""
        rotated = rotate_image(sample_image_float, 360)
        assert rotated.shape == sample_image_float.shape
        # Values should be close to original (small interpolation differences)
        np.testing.assert_allclose(rotated, sample_image_float, rtol=0.1, atol=1)

    def test_rotate_with_reshape(self, sample_image_2d):
        """Test rotation with reshape enabled."""
        rotated = rotate_image(sample_image_2d, 45, reshape=True)
        # With reshape=True, output will be larger to contain full rotated image
        assert rotated.ndim == 2
        assert rotated.shape[0] >= sample_image_2d.shape[0]

    def test_rotate_3d_image(self, sample_image_3d):
        """Test 3D image rotation."""
        rotated = rotate_image(sample_image_3d, 30)
        assert rotated.shape == sample_image_3d.shape
        assert rotated.dtype == sample_image_3d.dtype

    def test_rotate_preserve_dtype(self, sample_image_float):
        """Test dtype preservation after rotation."""
        rotated = rotate_image(sample_image_float, 45)
        assert rotated.dtype == np.float32

    def test_rotate_invalid_ndim(self):
        """Test error on invalid dimensions."""
        img_4d = np.zeros((2, 10, 10, 3))
        with pytest.raises(ValueError, match="2D or 3D"):
            rotate_image(img_4d, 45)


class TestBytescaleImage:
    """Tests for bytescale_image function."""

    def test_bytescale_basic(self, sample_image_float):
        """Test basic bytescaling."""
        scaled = bytescale_image(sample_image_float)
        assert scaled.dtype == np.uint8
        assert scaled.min() >= 0
        assert scaled.max() <= 255

    def test_bytescale_covers_full_range(self, sample_image_float):
        """Test that bytescaling uses full output range."""
        scaled = bytescale_image(sample_image_float)
        # Should cover most of the range
        assert scaled.min() < 10
        assert scaled.max() > 245

    def test_bytescale_custom_input_range(self):
        """Test custom input range."""
        data = np.array([0, 50, 100], dtype=np.float32)
        scaled = bytescale_image(data, imin=0, imax=100)
        assert scaled[0] == 0
        assert scaled[2] == 255
        assert 100 < scaled[1] < 150  # Should be around 127

    def test_bytescale_custom_output_range(self):
        """Test custom output range."""
        data = np.array([0, 50, 100], dtype=np.float32)
        scaled = bytescale_image(data, omin=10, omax=200)
        assert scaled.min() >= 10
        assert scaled.max() <= 200

    def test_bytescale_negative_values(self):
        """Test with negative input values."""
        data = np.array([-100, 0, 100], dtype=np.float32)
        scaled = bytescale_image(data)
        assert scaled.dtype == np.uint8
        assert scaled[0] == 0
        assert scaled[2] == 255

    def test_bytescale_uint16_input(self):
        """Test with uint16 input (common for FITS images)."""
        data = np.array([0, 32767, 65535], dtype=np.uint16)
        scaled = bytescale_image(data)
        assert scaled.dtype == np.uint8
        assert scaled[0] == 0
        assert scaled[2] == 255

    def test_bytescale_invalid_range(self):
        """Test error on invalid input range."""
        data = np.ones((10, 10))
        with pytest.raises(ValueError, match="imin.*less than.*imax"):
            bytescale_image(data, imin=100, imax=50)

    def test_bytescale_equal_imin_imax(self):
        """Test error when imin equals imax."""
        data = np.ones((10, 10))
        with pytest.raises(ValueError, match="imin.*less than.*imax"):
            bytescale_image(data, imin=50, imax=50)
