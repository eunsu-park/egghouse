"""Tests for egghouse.image.noise: MAD, robust sigma, Gaussian-core sigma."""

from __future__ import annotations

import numpy as np

from egghouse.image import (
    gaussian_core_sigma,
    mad,
    photon_transfer_fit,
    poisson_gaussian_noise,
    robust_sigma,
)


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


def test_photon_transfer_fit_exact_line():
    intensity = np.array([10.0, 100.0, 1000.0, 5000.0])
    g_true, r2_true = 0.4, 12.0
    g, r2, r_sq = photon_transfer_fit(intensity, g_true * intensity + r2_true)
    assert abs(g - g_true) < 1e-9
    assert abs(r2 - r2_true) < 1e-6
    assert r_sq > 0.999999


def test_photon_transfer_fit_too_few_points_is_nan():
    g, r2, r_sq = photon_transfer_fit(np.array([1.0]), np.array([2.0]))
    assert np.isnan(g) and np.isnan(r2) and np.isnan(r_sq)


def test_poisson_gaussian_noise_recovers_parameters():
    # Synthetic Poisson-Gaussian pair: sigma^2(I) = g*I + r2.
    rng = np.random.default_rng(6)
    i0 = rng.uniform(1.0, 8000.0, size=(2400, 2400))
    g_true, r2_true = 0.45, 25.0
    sig = np.sqrt(g_true * i0 + r2_true)
    a = i0 + rng.normal(0.0, 1.0, i0.shape) * sig
    b = i0 + rng.normal(0.0, 1.0, i0.shape) * sig
    res = poisson_gaussian_noise(a, b, bins=24, min_count=500)
    assert abs(res.g - g_true) / g_true < 0.10
    assert abs(res.r2 - r2_true) / r2_true < 0.40   # intercept is harder
    assert res.r_squared > 0.95


def test_poisson_gaussian_noise_shape_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        poisson_gaussian_noise(np.zeros((4, 4)), np.zeros((4, 5)))
