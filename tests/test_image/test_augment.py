"""Tests for egghouse.image.augment: paired_random_crop, paired_flip_rot."""

from __future__ import annotations

import numpy as np
import pytest

from egghouse.image import augment


def test_paired_random_crop_shape_and_same_offset():
    a = np.arange(64, dtype=np.float64).reshape(8, 8)
    b = a + 100.0
    rng = np.random.default_rng(0)
    a_c, b_c = augment.paired_random_crop(a, b, 4, rng)
    assert a_c.shape == (4, 4) and b_c.shape == (4, 4)
    # b is a constant offset of a, so the same crop offset ⇒ b_c == a_c + 100.
    assert np.array_equal(b_c, a_c + 100.0)


def test_paired_random_crop_full_size_is_identity():
    a = np.arange(16, dtype=np.float64).reshape(4, 4)
    b = a.copy()
    rng = np.random.default_rng(1)
    a_c, b_c = augment.paired_random_crop(a, b, 4, rng)
    assert np.array_equal(a_c, a) and np.array_equal(b_c, b)


def test_paired_random_crop_too_small_raises():
    a = np.zeros((3, 3))
    with pytest.raises(ValueError, match="smaller than patch"):
        augment.paired_random_crop(a, a, 4, np.random.default_rng(0))


def test_paired_flip_rot_applies_same_transform_to_both():
    a = np.arange(16, dtype=np.float64).reshape(4, 4)
    b = a + 100.0
    rng = np.random.default_rng(3)
    a_t, b_t = augment.paired_flip_rot(a, b, rng)
    assert a_t.shape == a.shape
    # Same transform ⇒ the constant offset is preserved element-wise.
    assert np.array_equal(b_t, a_t + 100.0)


def test_paired_flip_rot_is_deterministic_for_seed():
    a = np.arange(16, dtype=np.float64).reshape(4, 4)
    b = a.copy()
    t1 = augment.paired_flip_rot(a, b, np.random.default_rng(7))
    t2 = augment.paired_flip_rot(a, b, np.random.default_rng(7))
    assert np.array_equal(t1[0], t2[0]) and np.array_equal(t1[1], t2[1])


def test_paired_flip_rot_result_is_a_permutation_of_values():
    a = np.arange(16, dtype=np.float64).reshape(4, 4)
    a_t, _ = augment.paired_flip_rot(a, a.copy(), np.random.default_rng(5))
    assert sorted(a_t.ravel().tolist()) == sorted(a.ravel().tolist())
