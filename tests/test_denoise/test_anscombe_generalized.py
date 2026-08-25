"""Tests for the generalized Anscombe transform and its exact unbiased inverse."""

import numpy as np
import pytest

from egghouse.denoise.anscombe import generalized_forward, generalized_inverse


def _pg_samples(rng, lam, gain, sigma, offset, n=400_000):
    """Draws ``n`` Poisson-Gaussian samples ``a*P(lam) + N(offset, sigma^2)``."""
    return gain * rng.poisson(lam, n) + rng.normal(offset, sigma, n)


@pytest.mark.parametrize("lam", [0.5, 2.0, 10.0, 100.0])
def test_generalized_inverse_is_mean_unbiased_at_low_and_high_counts(lam):
    # Gain 0.5 DN/e- (2 e-/DN), read noise 0.4 DN, offset 3 DN.
    gain, sigma, offset = 0.5, 0.4, 3.0
    rng = np.random.default_rng(0)
    y = _pg_samples(rng, lam, gain, sigma, offset)
    z_mean = float(np.mean(generalized_forward(y, gain, sigma, offset)))
    est = float(generalized_inverse(z_mean, gain, sigma, offset))
    truth = gain * lam + offset
    # Tolerance: 2 % of the signal plus a small absolute floor (MC noise).
    assert abs(est - truth) < 0.02 * gain * lam + 0.02


def test_generalized_inverse_beats_algebraic_inverse_at_low_counts():
    gain, sigma, offset, lam = 0.5, 0.4, 0.0, 0.5
    rng = np.random.default_rng(1)
    y = _pg_samples(rng, lam, gain, sigma, offset)
    z_mean = float(np.mean(generalized_forward(y, gain, sigma, offset)))
    exact = float(generalized_inverse(z_mean, gain, sigma, offset))
    # Algebraic inverse of the forward formula, i.e. solving z for y.
    algebraic = (gain * (z_mean / 2.0) ** 2 - 0.375 * gain ** 2 - sigma ** 2) / gain + offset
    truth = gain * lam + offset
    assert abs(exact - truth) < abs(algebraic - truth)


def test_generalized_forward_has_unit_variance_where_counts_are_not_tiny():
    gain, sigma, offset = 0.07, 1.0, 700.0  # SECCHI-like: ~14 e-/DN, bias 700 DN
    rng = np.random.default_rng(2)
    for lam in (200.0, 5_000.0, 50_000.0):
        y = _pg_samples(rng, lam, gain, sigma, offset, n=200_000)
        z = generalized_forward(y, gain, sigma, offset)
        assert abs(float(np.var(z)) - 1.0) < 0.05


def test_generalized_pair_round_trips_without_noise():
    gain, sigma, offset = 0.07, 1.0, 700.0
    y = np.linspace(800.0, 20_000.0, 50)
    z = generalized_forward(y, gain, sigma, offset)
    back = generalized_inverse(z, gain, sigma, offset)
    # Noise-free round trip is the asymptotic identity to well under 1 %.
    assert np.allclose(back, y, rtol=5e-3)
