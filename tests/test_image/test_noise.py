"""Tests for egghouse.image.noise: MAD, robust sigma, Gaussian-core sigma."""

from __future__ import annotations

import numpy as np

from egghouse.image import gaussian_core_sigma, mad, robust_sigma


def test_mad_constant_array_is_zero():
    assert mad(np.full((8, 8), 3.0)) == 0.0


def test_mad_matches_definition():
    x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    # median = 3; |x-3| = [2,1,0,1,97]; median = 1.
    assert mad(x) == 1.0


def test_robust_sigma_is_outlier_robust():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 2.0, size=200_000)
    base = robust_sigma(x)
    # Inject extreme outliers: robust sigma barely moves, std blows up.
    y = x.copy()
    y[:1000] = 1e6
    assert abs(robust_sigma(y) - base) / base < 0.05
    assert float(np.std(y)) > 10 * base


def test_robust_sigma_recovers_gaussian_scale():
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 5.0, size=200_000)
    assert abs(robust_sigma(x) - 5.0) / 5.0 < 0.02


def test_mad_center_override():
    x = np.array([0.0, 0.0, 10.0])
    # About 0: |x| = [0,0,10] -> median 0.
    assert mad(x, center=0.0) == 0.0


def test_gaussian_core_sigma_recovers_gaussian_scale():
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 5.0, size=400_000)
    assert abs(gaussian_core_sigma(x) - 5.0) / 5.0 < 0.05


def test_gaussian_core_sigma_isolates_noise_from_field_wings():
    # Liu (2012): noise core + a real-field wing. The fit should recover the
    # noise core (~5), while the plain std is inflated by the wing.
    rng = np.random.default_rng(4)
    noise = rng.normal(0.0, 5.0, size=400_000)
    field = rng.normal(0.0, 80.0, size=80_000)  # strong-field tail
    x = np.concatenate([noise, field])
    s = gaussian_core_sigma(x)
    assert abs(s - 5.0) / 5.0 < 0.10
    assert float(np.std(x)) > 2.0 * s


def test_gaussian_core_sigma_constant_array_is_zero():
    assert gaussian_core_sigma(np.full((16, 16), 2.0)) == 0.0


def test_gaussian_core_sigma_handles_nans():
    rng = np.random.default_rng(5)
    x = rng.normal(0.0, 3.0, size=200_000)
    x[:500] = np.nan
    assert abs(gaussian_core_sigma(x) - 3.0) / 3.0 < 0.05
