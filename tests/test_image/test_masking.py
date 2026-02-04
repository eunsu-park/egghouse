"""Tests for egghouse.image.masking module."""

import numpy as np
import pytest

from egghouse.image import circle_mask, annulus_mask


class TestCircleMask:
    """Tests for circle_mask function."""

    def test_circle_mask_basic(self):
        """Test basic circle mask creation."""
        mask = circle_mask(100, radius=30)
        assert mask.shape == (100, 100)
        assert mask.dtype == bool
        assert mask.sum() > 0

    def test_circle_mask_square_size(self):
        """Test with integer size (square mask)."""
        mask = circle_mask(64, radius=20)
        assert mask.shape == (64, 64)

    def test_circle_mask_rectangular(self):
        """Test rectangular image size."""
        mask = circle_mask((50, 100), radius=20)
        assert mask.shape == (50, 100)

    def test_circle_mask_center_default(self):
        """Test that default center is image center."""
        size = 100
        mask = circle_mask(size, radius=10)
        # Center point should be True
        center = size // 2
        assert mask[center, center] == True

    def test_circle_mask_custom_center(self):
        """Test mask with custom center."""
        mask = circle_mask(100, radius=20, center=(25, 25))
        # Custom center should be True
        assert mask[25, 25] == True
        # Default center should be False (outside radius)
        assert mask[50, 50] == False

    def test_circle_mask_inner_type(self):
        """Test inner mask type."""
        mask = circle_mask(100, radius=30, mask_type='inner')
        # Center should be True, corners should be False
        assert mask[50, 50] == True
        assert mask[0, 0] == False

    def test_circle_mask_outer_type(self):
        """Test outer mask type."""
        mask = circle_mask(100, radius=30, mask_type='outer')
        # Center should be False, corners should be True
        assert mask[50, 50] == False
        assert mask[0, 0] == True

    def test_circle_mask_inner_outer_complementary(self):
        """Test that inner and outer masks are complementary."""
        mask_inner = circle_mask(100, radius=30, mask_type='inner')
        mask_outer = circle_mask(100, radius=30, mask_type='outer')
        # Every pixel should be in exactly one mask
        assert np.all(mask_inner != mask_outer)

    def test_circle_mask_invalid_type(self):
        """Test error on invalid mask_type."""
        with pytest.raises(ValueError, match="mask_type"):
            circle_mask(100, radius=30, mask_type='invalid')

    def test_circle_mask_large_radius(self):
        """Test with radius larger than image."""
        mask = circle_mask(100, radius=200)
        # All pixels should be inside the circle
        assert mask.all()

    def test_circle_mask_zero_radius(self):
        """Test with zero radius."""
        mask = circle_mask(100, radius=0)
        # No pixels should be inside
        assert not mask.any()


class TestAnnulusMask:
    """Tests for annulus_mask function."""

    def test_annulus_mask_basic(self):
        """Test basic annulus mask creation."""
        mask = annulus_mask(100, inner_radius=20, outer_radius=40)
        assert mask.shape == (100, 100)
        assert mask.dtype == bool
        assert mask.sum() > 0

    def test_annulus_mask_center_is_hole(self):
        """Test that center is masked out (hole)."""
        mask = annulus_mask(100, inner_radius=20, outer_radius=40)
        # Center should be False (inside inner radius)
        assert mask[50, 50] == False

    def test_annulus_mask_edge_is_masked_out(self):
        """Test that corners are outside annulus."""
        mask = annulus_mask(100, inner_radius=20, outer_radius=40)
        # Corners should be False (outside outer radius)
        assert mask[0, 0] == False

    def test_annulus_mask_ring_region(self):
        """Test that ring region is True."""
        mask = annulus_mask(100, inner_radius=20, outer_radius=40)
        # A point in the ring should be True
        # Point at distance ~30 from center (50, 80) is 30 pixels away
        assert mask[50, 80] == True

    def test_annulus_mask_custom_center(self):
        """Test annulus with custom center."""
        mask = annulus_mask(100, inner_radius=10, outer_radius=30, center=(25, 25))
        # Custom center should be False (inside inner radius)
        assert mask[25, 25] == False
        # Point in ring around custom center
        assert mask[25, 45] == True  # 20 pixels from center

    def test_annulus_mask_invalid_radii(self):
        """Test error when inner >= outer radius."""
        with pytest.raises(ValueError, match="inner_radius"):
            annulus_mask(100, inner_radius=40, outer_radius=20)

    def test_annulus_mask_equal_radii(self):
        """Test error when inner equals outer radius."""
        with pytest.raises(ValueError, match="inner_radius"):
            annulus_mask(100, inner_radius=30, outer_radius=30)

    def test_annulus_mask_rectangular(self):
        """Test annulus on rectangular image."""
        mask = annulus_mask((50, 100), inner_radius=10, outer_radius=20)
        assert mask.shape == (50, 100)
