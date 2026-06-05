"""Tests for the spline forward-fit DEM inversion (egghouse.sdo.dem.spline).

Synthetic only; no CHIANTI/fiasco needed. Mirrors the synthetic setup used by
the NNLS tests so the two solvers are exercised on the same problem.
"""

import numpy as np
import pytest


def _synthetic(n_bins=21):
    """Build the validation problem: 6 Gaussian channels, one Gaussian DEM."""
    T = np.logspace(5.5, 7.5, n_bins)
    logt = np.log10(T)
    peaks = [6.8, 5.6, 5.9, 6.2, 6.3, 6.4]
    R = np.stack(
        [np.exp(-0.5 * ((logt - p) / 0.2) ** 2) * 1e-24 for p in peaks], axis=1
    )  # (n_T, 6)
    dt = T * np.log(10) * np.gradient(logt)
    true = np.exp(-0.5 * ((logt - 6.2) / 0.15) ** 2) * 1e22
    A = (R * dt[:, None]).T  # (6, n_T)
    I = A @ true
    return T, R, dt, A, true, I


class TestDemSpline:

    def test_reconstructs_and_nonnegative(self):
        from egghouse.sdo.dem.spline import dem_spline

        T, R, dt, A, true, I = _synthetic()
        dem, info = dem_spline(I, 0.05 * I, R, T, n_knots=5)

        assert dem.shape == (T.size,)
        assert np.all(dem >= 0)
        # Predicted intensities reproduce the observations.
        Isyn = A @ dem
        np.testing.assert_allclose(Isyn / I, 1.0, rtol=0.15)
        # Peak temperature recovered near the true log T = 6.2.
        logt = np.log10(T)
        assert abs(logt[np.argmax(dem)] - 6.2) < 0.3
        assert "chi2_map" in info
        assert info["knot_logt"].size == 5

    def test_squeeze_1d(self):
        from egghouse.sdo.dem.spline import dem_spline

        T, R, dt, A, true, I = _synthetic()
        dem, _ = dem_spline(I, 0.05 * I, R, T)
        assert dem.ndim == 1

    def test_batch_shape(self):
        from egghouse.sdo.dem.spline import dem_spline

        T, R, dt, A, true, I = _synthetic()
        batch = np.repeat(I[None, :], 3, axis=0)
        dem, info = dem_spline(batch, 0.05 * batch, R, T)
        assert dem.shape == (3, T.size)
        assert info["chi2_map"].shape == (3,)

    def test_shape_mismatch_raises(self):
        from egghouse.sdo.dem.spline import dem_spline

        T, R, dt, A, true, I = _synthetic()
        wrong = np.zeros((T.size, 4))  # wrong channel count
        with pytest.raises(ValueError, match="doesn't match"):
            dem_spline(I, 0.05 * I, wrong, T)

    def test_invalid_n_knots_raises(self):
        from egghouse.sdo.dem.spline import dem_spline

        T, R, dt, A, true, I = _synthetic()
        with pytest.raises(ValueError, match="n_knots"):
            dem_spline(I, 0.05 * I, R, T, n_knots=1)

    def test_zero_pixel_returns_zero_dem(self):
        from egghouse.sdo.dem.spline import dem_spline

        T, R, dt, A, true, I = _synthetic()
        zero = np.zeros_like(I)
        dem, info = dem_spline(zero, np.ones_like(I), R, T)
        assert np.all(dem == 0)
        assert info["chi2"] == 0.0
