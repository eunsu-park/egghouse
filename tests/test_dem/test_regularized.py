"""Tests for the linear regularized (Tikhonov/GSVD) DEM inversion.

All synthetic: no CHIANTI/fiasco/aiapy needed. Builds a 6-channel response
of Gaussians and a single-Gaussian true DEM, then checks reconstruction,
non-negativity, peak temperature, batch handling, and input validation.
"""

import numpy as np
import pytest

from egghouse.dem.regularized import dem_regularized


def _synthetic(n_bins=21):
    """T, response (n_T,6), dt, true DEM, observed I, errors."""
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
    errors = 0.05 * I
    return T, R, dt, true, I, errors


class TestDemRegularized:
    def test_reconstructs_and_nonnegative(self):
        T, R, dt, true, I, errors = _synthetic()
        dem, info = dem_regularized(I, errors, R, T, reg_order=2, reg_tweak=1.0)
        assert dem.shape == (T.size,)
        assert np.all(dem >= 0)
        A = (R * dt[:, None]).T
        Isyn = A @ dem
        # The (positivity-clipped) reconstruction tracks the observations in the
        # aggregate. Individual faint channels far from the DEM peak can be off
        # after clipping; the median ratio is the robust check.
        assert 0.7 < np.median(Isyn / I) < 1.3
        assert "chi2" in info and "chi2_map" in info

    def test_discrepancy_principle_hits_target(self):
        # The reported chi^2 is that of the linear solution, which the
        # discrepancy principle drives to reg_tweak * n_channels.
        T, R, dt, true, I, errors = _synthetic()
        _, info = dem_regularized(I, errors, R, T, reg_order=2, reg_tweak=1.0)
        # n_channels = 6; allow a generous band around the target.
        assert 3.0 <= info["chi2"] <= 9.0

    def test_peak_temperature(self):
        T, R, dt, true, I, errors = _synthetic()
        dem, _ = dem_regularized(I, errors, R, T, reg_order=2)
        logt = np.log10(T)
        peak_logt = logt[int(np.argmax(dem))]
        # True DEM peaks at logT = 6.2; allow one-bin (~0.1 dex) tolerance.
        assert abs(peak_logt - 6.2) <= 0.12

    def test_batch_shape(self):
        T, R, dt, true, I, errors = _synthetic()
        batch = np.repeat(I[None, :], 4, axis=0)
        ebatch = np.repeat(errors[None, :], 4, axis=0)
        dem, info = dem_regularized(batch, ebatch, R, T)
        assert dem.shape == (4, T.size)
        assert info["chi2_map"].shape == (4,)
        assert info["lambda_map"].shape == (4,)

    def test_reg_order_zero(self):
        T, R, dt, true, I, errors = _synthetic()
        dem, info = dem_regularized(I, errors, R, T, reg_order=0)
        assert dem.shape == (T.size,)
        assert np.all(dem >= 0)

    def test_shape_mismatch_raises(self):
        T, R, dt, true, I, errors = _synthetic()
        bad_R = R[:, :3]  # wrong channel count vs intensities
        with pytest.raises(ValueError):
            dem_regularized(I, errors, bad_R, T)

    def test_invalid_reg_order_raises(self):
        T, R, dt, true, I, errors = _synthetic()
        with pytest.raises(ValueError):
            dem_regularized(I, errors, R, T, reg_order=1)

    def test_reg_tweak_monotone_smoothing(self):
        # Larger reg_tweak -> larger lambda -> higher data chi^2.
        T, R, dt, true, I, errors = _synthetic()
        _, info_lo = dem_regularized(I, errors, R, T, reg_tweak=0.5)
        _, info_hi = dem_regularized(I, errors, R, T, reg_tweak=4.0)
        assert info_hi["lambda_map"][0] >= info_lo["lambda_map"][0]
