"""Tests for sparse / basis-pursuit DEM inversion (egghouse.sdo.dem.sparse)."""

import numpy as np


class TestDemSparse:
    """Basis-pursuit DEM inversion (synthetic; no CHIANTI/fiasco needed)."""

    def _synthetic(self, n_bins=21):
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

    def test_reconstructs_and_nonnegative(self):
        from egghouse.sdo.dem.sparse import dem_sparse

        T, R, dt, I = self._synthetic()
        dem, info = dem_sparse(I, I * 0.05, R, T)
        assert dem.shape == (T.size,)
        assert np.all(dem >= 0)
        Isyn = (R * dt[:, None] * dem[:, None]).sum(axis=0)
        np.testing.assert_allclose(Isyn / I, 1.0, rtol=0.2)
        assert "chi2_map" in info
        assert "chi2" in info

    def test_peak_near_truth(self):
        from egghouse.sdo.dem.sparse import dem_sparse

        T, R, dt, I = self._synthetic()
        dem, _ = dem_sparse(I, I * 0.05, R, T)
        peak_logt = np.log10(T[np.argmax(dem)])
        assert abs(peak_logt - 6.2) < 0.25

    def test_sparsity(self):
        """L1 objective should populate only a few temperature bins."""
        from egghouse.sdo.dem.sparse import dem_sparse

        T, R, dt, I = self._synthetic()
        dem, _ = dem_sparse(I, I * 0.05, R, T)
        nnz = np.sum(dem > dem.max() * 1e-6)
        # 6 channels -> sparse LP basis solution has at most ~n_channels bins.
        assert nnz <= 6

    def test_batch_shape(self):
        from egghouse.sdo.dem.sparse import dem_sparse

        T, R, dt, I = self._synthetic()
        batch = np.repeat(I[None, :], 3, axis=0)
        dem, info = dem_sparse(batch, batch * 0.05, R, T)
        assert dem.shape == (3, T.size)
        assert info["chi2_map"].shape == (3,)
        assert info["feasible_map"].shape == (3,)

    def test_shape_mismatch_raises(self):
        from egghouse.sdo.dem.sparse import dem_sparse
        import pytest

        T, R, dt, I = self._synthetic()
        with pytest.raises(ValueError):
            dem_sparse(I, I * 0.05, R[:, :3], T)

    def test_zero_intensity_pixel(self):
        from egghouse.sdo.dem.sparse import dem_sparse

        T, R, dt, I = self._synthetic()
        zero = np.zeros_like(I)
        dem, info = dem_sparse(zero, np.ones_like(I), R, T)
        assert dem.shape == (T.size,)
        assert np.all(dem == 0)

    def test_feasibility_info(self):
        from egghouse.sdo.dem.sparse import dem_sparse

        T, R, dt, I = self._synthetic()
        _, info = dem_sparse(I, I * 0.05, R, T)
        assert "n_infeasible" in info
        assert info["n_infeasible"] == 0
