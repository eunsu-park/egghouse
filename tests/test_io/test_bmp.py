"""Tests for egghouse.io.bmp module."""

import os

import numpy as np
import pytest

from egghouse.io import read_bmp, write_bmp, read_bmp_header


class TestWriteBmp:
    """Tests for write_bmp function."""

    def test_write_rgb(self, temp_dir):
        """Test writing RGB BMP."""
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:, :, 0] = 255  # Red channel

        path = os.path.join(temp_dir, "test_rgb.bmp")
        write_bmp(path, img)

        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_write_grayscale(self, temp_dir):
        """Test writing grayscale BMP."""
        img = np.arange(256, dtype=np.uint8).reshape(16, 16)

        path = os.path.join(temp_dir, "test_gray.bmp")
        write_bmp(path, img)

        assert os.path.exists(path)

    def test_write_creates_directory(self, temp_dir):
        """Test that write_bmp creates parent directories."""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        path = os.path.join(temp_dir, "subdir", "nested", "test.bmp")
        write_bmp(path, img)

        assert os.path.exists(path)

    def test_write_overwrite_false(self, temp_dir):
        """Test that overwrite=False prevents overwriting."""
        img = np.zeros((10, 10), dtype=np.uint8)
        path = os.path.join(temp_dir, "test.bmp")
        write_bmp(path, img)

        with pytest.raises(FileExistsError):
            write_bmp(path, img, overwrite=False)

    def test_write_overwrite_true(self, temp_dir):
        """Test that overwrite=True allows overwriting."""
        img = np.zeros((10, 10), dtype=np.uint8)
        path = os.path.join(temp_dir, "test.bmp")
        write_bmp(path, img)

        # Should not raise
        img[5, 5] = 100
        write_bmp(path, img, overwrite=True)

    def test_write_invalid_dtype(self, temp_dir):
        """Test error on non-uint8 data."""
        img = np.zeros((10, 10), dtype=np.float32)
        path = os.path.join(temp_dir, "test.bmp")

        with pytest.raises(ValueError, match="uint8"):
            write_bmp(path, img)

    def test_write_invalid_shape(self, temp_dir):
        """Test error on invalid shape."""
        img = np.zeros((10, 10, 4), dtype=np.uint8)  # RGBA not supported
        path = os.path.join(temp_dir, "test.bmp")

        with pytest.raises(ValueError, match="shape"):
            write_bmp(path, img)


class TestReadBmp:
    """Tests for read_bmp function."""

    def test_read_rgb_roundtrip(self, temp_dir):
        """Test writing and reading RGB BMP."""
        original = np.zeros((64, 64, 3), dtype=np.uint8)
        original[:, :, 0] = 255  # Red
        original[32:, :, 1] = 128  # Green in bottom half

        path = os.path.join(temp_dir, "test_rgb.bmp")
        write_bmp(path, original)

        data, info = read_bmp(path)

        assert data.shape == (64, 64, 3)
        assert data.dtype == np.uint8
        np.testing.assert_array_equal(data, original)

    def test_read_grayscale_returns_rgb(self, temp_dir):
        """Test that grayscale BMP is returned as RGB."""
        original = np.arange(256, dtype=np.uint8).reshape(16, 16)

        path = os.path.join(temp_dir, "test_gray.bmp")
        write_bmp(path, original)

        data, info = read_bmp(path)

        # Grayscale BMP returns as RGB (3 channels)
        assert data.shape == (16, 16, 3)
        assert info['bpp'] == 8

    def test_read_header_info(self, temp_dir):
        """Test that header info is correct."""
        original = np.zeros((100, 200, 3), dtype=np.uint8)
        path = os.path.join(temp_dir, "test.bmp")
        write_bmp(path, original)

        data, info = read_bmp(path)

        assert info['width'] == 200
        assert info['height'] == 100
        assert info['bpp'] == 24

    def test_read_nonexistent_file(self):
        """Test error on nonexistent file."""
        with pytest.raises(FileNotFoundError):
            read_bmp("/nonexistent/path/file.bmp")


class TestReadBmpHeader:
    """Tests for read_bmp_header function."""

    def test_read_header_rgb(self, temp_dir):
        """Test reading header of RGB BMP."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        path = os.path.join(temp_dir, "test.bmp")
        write_bmp(path, img)

        info = read_bmp_header(path)

        assert info['width'] == 200
        assert info['height'] == 100
        assert info['bpp'] == 24
        assert info['compression'] == 0

    def test_read_header_grayscale(self, temp_dir):
        """Test reading header of grayscale BMP."""
        img = np.zeros((50, 75), dtype=np.uint8)
        path = os.path.join(temp_dir, "test.bmp")
        write_bmp(path, img)

        info = read_bmp_header(path)

        assert info['width'] == 75
        assert info['height'] == 50
        assert info['bpp'] == 8

    def test_read_header_nonexistent_file(self):
        """Test error on nonexistent file."""
        with pytest.raises(FileNotFoundError):
            read_bmp_header("/nonexistent/path/file.bmp")


class TestBmpRoundtrip:
    """Integration tests for BMP read/write cycle."""

    def test_various_sizes(self, temp_dir):
        """Test various image sizes (including non-4-byte-aligned widths)."""
        sizes = [(10, 10), (13, 17), (100, 100), (256, 256), (7, 11)]

        for height, width in sizes:
            img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            path = os.path.join(temp_dir, f"test_{width}x{height}.bmp")
            write_bmp(path, img)

            data, _ = read_bmp(path)
            np.testing.assert_array_equal(data, img)

    def test_gradient_image(self, temp_dir):
        """Test gradient image to verify pixel order."""
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        # Create gradients
        img[:, :, 0] = np.arange(256)  # Red gradient left-right
        img[:, :, 1] = np.arange(256)[:, np.newaxis]  # Green gradient top-bottom

        path = os.path.join(temp_dir, "gradient.bmp")
        write_bmp(path, img)

        data, _ = read_bmp(path)
        np.testing.assert_array_equal(data, img)
