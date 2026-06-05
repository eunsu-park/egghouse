"""Tests for the single-Gaussian forward-fit DEM inversion (synthetic).

No CHIANTI / fiasco / aiapy needed: a synthetic response and a true single-
Gaussian DEM are constructed so the forward model can recover the truth.
"""

import numpy as np


def _synthetic(n_bins=21):
    """T, response, dt, intensities for a true single-Gaussian DEM.

    response: 6 Gaussians in logT (~1e-24); true DEM: Gaussian at logT 6.2,
    sigma 0.15, amplitude ~1e22. The single-Gaussian model can fit this well.
    """
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


class TestDemGaussian:
    """Single-Gaussian forward-fit DEM inversion."""

    def test_reconstructs_and_nonnegative(self):
        from egghouse.sdo.dem.gaussian import dem_gaussian

        T, R, dt, I = _synthetic()
        dem, info = dem_gaussian(I, I * 0.05, R, T)
        assert dem.shape == (T.size,)
        assert np.all(dem >= 0)
        Isyn = (R * dt[:, None] * dem[:, None]).sum(axis=0)
        np.testing.assert_allclose(Isyn / I, 1.0, rtol=0.1)
        assert "chi2_map" in info

    def test_recovers_fitted_params(self):
        from egghouse.sdo.dem.gaussian import dem_gaussian

        T, R, dt, I = _synthetic()
        _, info = dem_gaussian(I, I * 0.05, R, T)
        # True peak logT 6.2, sigma 0.15.
        assert abs(float(info["logt_peak"]) - 6.2) < 0.1
        assert abs(float(info["sigma"]) - 0.15) < 0.1
        assert float(info["em_peak"]) > 0

    def test_batch_shape(self):
        from egghouse.sdo.dem.gaussian import dem_gaussian

        T, R, dt, I = _synthetic()
        batch = np.repeat(I[None, :], 3, axis=0)
        dem, info = dem_gaussian(batch, batch * 0.05, R, T)
        assert dem.shape == (3, T.size)
        assert info["chi2_map"].shape == (3,)
        assert info["logt_peak"].shape == (3,)
        assert info["em_peak"].shape == (3,)
        assert info["sigma"].shape == (3,)

    def test_shape_mismatch_raises(self):
        from egghouse.sdo.dem.gaussian import dem_gaussian

        T, R, dt, I = _synthetic()
        bad_R = R[:, :3]  # wrong number of channels vs intensities
        try:
            dem_gaussian(I, I * 0.05, bad_R, T)
        except ValueError as exc:
            assert "shape" in str(exc).lower()
        else:
            raise AssertionError("expected ValueError on response shape mismatch")

    def test_invalid_sigma_bound_raises(self):
        from egghouse.sdo.dem.gaussian import dem_gaussian

        T, R, dt, I = _synthetic()
        try:
            dem_gaussian(I, I * 0.05, R, T, sigma_bounds=(0.0, 1.0))
        except ValueError as exc:
            assert "sigma" in str(exc).lower()
        else:
            raise AssertionError("expected ValueError on non-positive sigma bound")

    def test_dead_pixel_returns_zero(self):
        from egghouse.sdo.dem.gaussian import dem_gaussian

        T, R, dt, I = _synthetic()
        dead = np.zeros_like(I)
        dem, info = dem_gaussian(dead, dead + 1.0, R, T)
        assert np.all(dem == 0)
        assert float(info["chi2"]) == 0.0
