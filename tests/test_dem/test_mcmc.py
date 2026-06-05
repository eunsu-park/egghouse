"""Tests for the Metropolis-Hastings MCMC DEM inversion (egghouse.dem.mcmc).

Synthetic data only (no CHIANTI/fiasco needed). The chain length is kept small
so the suite runs in a few seconds; reconstruction quality is therefore checked
with loose tolerances.
"""

import numpy as np
import pytest

# Imported from the module path directly because __init__.py is intentionally
# not modified to export the experimental MCMC solver.
from egghouse.dem.mcmc import dem_mcmc


def _synthetic(n_bins=21):
    """Build the 6-Gaussian-response / single-Gaussian-DEM test problem."""
    T = np.logspace(5.5, 7.5, n_bins)
    logt = np.log10(T)
    peaks = [6.8, 5.6, 5.9, 6.2, 6.3, 6.4]
    R = np.stack(
        [np.exp(-0.5 * ((logt - p) / 0.2) ** 2) * 1e-24 for p in peaks], axis=1
    )  # (n_T, 6)
    dt = T * np.log(10) * np.gradient(logt)
    true = np.exp(-0.5 * ((logt - 6.2) / 0.15) ** 2) * 1e22
    I = (R * dt[:, None] * true[:, None]).sum(axis=0)  # (6,)
    return T, R, dt, I


def test_shapes_and_nonnegative():
    T, R, dt, I = _synthetic()
    dem, info = dem_mcmc(
        I, 0.05 * I, R, T, n_steps=1500, n_burn=300, seed=1
    )
    assert dem.shape == (T.size,)
    assert np.all(dem >= 0)
    assert info["dem_std"].shape == (T.size,)
    assert np.all(info["dem_std"] >= 0)
    assert "chi2" in info and "chi2_map" in info


def test_reconstructs_intensities():
    T, R, dt, I = _synthetic()
    dem, info = dem_mcmc(
        I, 0.05 * I, R, T, n_steps=2000, n_burn=400, seed=2
    )
    Isyn = (R * dt[:, None] * dem[:, None]).sum(axis=0)
    # Loose: short chain, ill-posed inversion.
    np.testing.assert_allclose(Isyn / I, 1.0, rtol=0.3)


def test_peak_near_truth():
    T, R, dt, I = _synthetic()
    dem, _ = dem_mcmc(I, 0.05 * I, R, T, n_steps=2000, n_burn=400, seed=3)
    logt = np.log10(T)
    assert abs(logt[np.argmax(dem)] - 6.2) <= 0.3


def test_reproducible_with_seed():
    T, R, dt, I = _synthetic()
    d1, _ = dem_mcmc(I, 0.05 * I, R, T, n_steps=1000, n_burn=200, seed=7)
    d2, _ = dem_mcmc(I, 0.05 * I, R, T, n_steps=1000, n_burn=200, seed=7)
    np.testing.assert_array_equal(d1, d2)


def test_batch_shape():
    T, R, dt, I = _synthetic()
    batch = np.repeat(I[None, :], 2, axis=0)
    dem, info = dem_mcmc(
        batch, 0.05 * batch, R, T, n_steps=800, n_burn=200, seed=0
    )
    assert dem.shape == (2, T.size)
    assert info["dem_std"].shape == (2, T.size)
    assert info["chi2_map"].shape == (2,)


def test_response_shape_validation():
    T, R, dt, I = _synthetic()
    bad_R = R[:, :3]  # wrong number of channels
    with pytest.raises(ValueError, match="Response shape"):
        dem_mcmc(I, 0.05 * I, bad_R, T, n_steps=500, n_burn=100, seed=0)


def test_burn_in_validation():
    T, R, dt, I = _synthetic()
    with pytest.raises(ValueError, match="n_burn"):
        dem_mcmc(I, 0.05 * I, R, T, n_steps=500, n_burn=500, seed=0)
