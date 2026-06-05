"""Tests for the fast linear (Plowman-style) DEM solver.

All synthetic (no CHIANTI/fiasco/aiapy needed). The forward model mirrors the
solver's own convention: I = A @ DEM with A[c,t] = response[t,c] * dt[t].
"""

import time

import numpy as np
import pytest


def _synthetic(n_bins=21):
    """Known-DEM synthetic AIA-like problem (6 Gaussian channels)."""
    T = np.logspace(5.5, 7.5, n_bins)
    logt = np.log10(T)
    peaks = [6.8, 5.6, 5.9, 6.2, 6.3, 6.4]
    R = np.stack(
        [np.exp(-0.5 * ((logt - p) / 0.2) ** 2) * 1e-24 for p in peaks], axis=1
    )  # (n_T, 6)
    dt = T * np.log(10) * np.gradient(logt)
    true = np.exp(-0.5 * ((logt - 6.2) / 0.15) ** 2) * 1e22
    A = (R * dt[:, None]).T  # (6, n_T)
    I = A @ true  # (6,)
    return T, R, A, true, I


class TestDemPlowman:
    def test_reconstructs_intensities_and_nonnegative(self):
        from egghouse.dem.plowman import dem_plowman

        T, R, A, true, I = _synthetic()
        dem, info = dem_plowman(I, 0.05 * I, R, T)
        assert dem.shape == (T.size,)
        assert np.all(dem >= 0)
        I_syn = A @ dem
        # Channels probing the DEM peak (the bright, well-constrained ones)
        # reconstruct to a few percent. Channels 0/1 peak at log T 6.8/5.6 where
        # the true DEM is ~0, so their tiny intensities are dominated by the
        # positivity floor and intentionally excluded.
        bright = I > 0.2 * I.max()
        np.testing.assert_allclose(I_syn[bright] / I[bright], 1.0, rtol=0.05)
        assert "chi2_map" in info and "reg_lambda" in info

    def test_peak_temperature_recovered(self):
        from egghouse.dem.plowman import dem_plowman

        T, R, A, true, I = _synthetic()
        dem, _ = dem_plowman(I, 0.05 * I, R, T)
        peak_logt = np.log10(T[np.argmax(dem)])
        # True DEM peaks at log T = 6.2; regularized peak within one decade-fifth.
        assert abs(peak_logt - 6.2) < 0.2

    def test_batch_shape_and_chi2_map(self):
        from egghouse.dem.plowman import dem_plowman

        T, R, A, true, I = _synthetic()
        batch = np.repeat(I[None, :], 4, axis=0)
        dem, info = dem_plowman(batch, 0.05 * batch, R, T)
        assert dem.shape == (4, T.size)
        assert info["chi2_map"].shape == (4,)

    def test_explicit_lambda(self):
        from egghouse.dem.plowman import dem_plowman

        T, R, A, true, I = _synthetic()
        dem, info = dem_plowman(I, 0.05 * I, R, T, reg_lambda=1.0)
        assert info["reg_lambda"] == 1.0
        assert np.all(dem >= 0)

    def test_shape_mismatch_raises(self):
        from egghouse.dem.plowman import dem_plowman

        T, R, A, true, I = _synthetic()
        wrong = np.zeros((T.size, 4))  # wrong channel count
        with pytest.raises(ValueError, match="doesn't match"):
            dem_plowman(I, 0.05 * I, wrong, T)

    def test_calibrate_targets_chi2(self):
        from egghouse.dem.plowman import calibrate_lambda, dem_plowman

        T, R, A, true, I = _synthetic()
        batch = np.repeat(I[None, :], 20, axis=0)
        lam = calibrate_lambda(batch, 0.1 * batch, R, T, target_chi2=6.0)
        assert lam > 0
        _, info = dem_plowman(batch, 0.1 * batch, R, T, reg_lambda=lam)
        assert np.isfinite(info["chi2"])

    def test_batch_speed(self):
        """5000-pixel batch should be fast (single vectorized solve)."""
        from egghouse.dem.plowman import dem_plowman

        T, R, A, true, I = _synthetic()
        n = 5000
        batch = np.repeat(I[None, :], n, axis=0)
        t0 = time.perf_counter()
        dem, _ = dem_plowman(batch, 0.05 * batch, R, T)
        elapsed = time.perf_counter() - t0
        assert dem.shape == (n, T.size)
        # Generous bound; vectorized apply is typically << 0.5 s.
        assert elapsed < 2.0
