"""Tests for egghouse.dem module."""

import numpy as np
import pytest


class TestGetDefaultTemperatures:
    """Tests for get_default_temperatures function."""

    def test_default_values(self):
        """Test default temperature grid."""
        from egghouse.dem import get_default_temperatures

        temps = get_default_temperatures()
        assert temps.shape == (100,)
        assert np.isclose(np.log10(temps[0]), 5.5, rtol=0.01)
        assert np.isclose(np.log10(temps[-1]), 7.5, rtol=0.01)

    def test_custom_range(self):
        """Test custom temperature range."""
        from egghouse.dem import get_default_temperatures

        temps = get_default_temperatures(logt_min=6.0, logt_max=7.0, n_bins=50)
        assert temps.shape == (50,)
        assert np.isclose(np.log10(temps[0]), 6.0, rtol=0.01)
        assert np.isclose(np.log10(temps[-1]), 7.0, rtol=0.01)

    def test_temperature_ordering(self):
        """Test that temperatures are in ascending order."""
        from egghouse.dem import get_default_temperatures

        temps = get_default_temperatures()
        assert np.all(np.diff(temps) > 0)


class TestGetTemperatureResponse:
    """Tests for get_temperature_response function.

    These tests exercise the built-in Gaussian *fallback* path, which is only
    reached when aiapy is unavailable. The fallback is force-enabled via
    monkeypatch so the suite stays meaningful regardless of whether aiapy is
    installed in the test environment.
    """

    @pytest.fixture
    def force_fallback(self, monkeypatch):
        from egghouse.sdo import dem_response as _resp

        monkeypatch.setattr(_resp, "HAS_AIAPY", False)
        return monkeypatch

    def test_fallback_response_shape(self, force_fallback):
        """Test fallback response has correct shape."""
        from egghouse.sdo import get_temperature_response; from egghouse.dem import get_default_temperatures

        temps = get_default_temperatures(n_bins=50)
        with pytest.warns(UserWarning, match="approximate"):
            response = get_temperature_response(temperatures=temps)

        assert response.shape == (50, 6)

    def test_fallback_response_positive(self, force_fallback):
        """Test fallback response values are positive."""
        from egghouse.sdo import get_temperature_response; from egghouse.dem import get_default_temperatures

        temps = get_default_temperatures(n_bins=50)
        with pytest.warns(UserWarning, match="approximate"):
            response = get_temperature_response(temperatures=temps)

        assert np.all(response >= 0)

    def test_custom_wavelengths(self, force_fallback):
        """Test response with custom wavelengths."""
        from egghouse.sdo import get_temperature_response; from egghouse.dem import get_default_temperatures

        temps = get_default_temperatures(n_bins=30)
        wavelengths = [171, 193, 211]
        with pytest.warns(UserWarning, match="approximate"):
            response = get_temperature_response(
                wavelengths=wavelengths, temperatures=temps
            )

        assert response.shape == (30, 3)

    def test_invalid_wavelength(self, force_fallback):
        """Test error on invalid wavelength."""
        from egghouse.sdo import get_temperature_response; from egghouse.dem import get_default_temperatures

        temps = get_default_temperatures(n_bins=20)
        with pytest.raises(ValueError, match="Unknown wavelength"):
            with pytest.warns(UserWarning):
                get_temperature_response(wavelengths=[999], temperatures=temps)

    def test_aiapy_path_raises_notimplemented(self, monkeypatch):
        """With aiapy installed but fiasco unavailable and no SSW table, the
        disabled aiapy-only path must raise a clear NotImplementedError
        rather than silently failing on the removed
        ``Channel.temperature_response`` upstream API."""
        from egghouse.sdo import dem_response as _resp
        from egghouse.dem import get_default_temperatures; from egghouse.sdo import get_temperature_response

        if not _resp.HAS_AIAPY:
            pytest.skip("aiapy not installed; aiapy path is not exercised here")
        # Force the aiapy-only path (skip the fiasco source).
        monkeypatch.setattr(_resp, "HAS_FIASCO", False)
        temps = get_default_temperatures(n_bins=10)
        with pytest.raises(NotImplementedError, match="ssw_table_path"):
            get_temperature_response(temperatures=temps)  # no ssw_table_path -> aiapy path


class TestFoldLinesIntoChannel:
    """Unit-chain test for the fiasco response folding (no CHIANTI/fiasco needed).

    Validates ``K(T) = (Omega/4pi) * sum_lines g_photon * R(lambda)`` — the
    arithmetic core of ``_get_fiasco_response`` — in isolation.
    """

    def test_single_line_unit_chain(self):
        from egghouse.dem.response import _fold_lines_into_channel

        # One line at 171.1 A; flat channel response 2.0 cm2 DN/ph; Omega=4pi
        # so the prefactor is exactly 1 -> K(T) = g_photon * R.
        g_photon = np.array([[3.0], [5.0]])  # (n_T=2, n_lines=1), cm^3 ph/s
        out = _fold_lines_into_channel(
            g_photon,
            np.array([171.1]),
            np.array([170.0, 172.0]),
            np.array([2.0, 2.0]),
            4.0 * np.pi,
        )
        np.testing.assert_allclose(out, [6.0, 10.0])

    def test_line_outside_grid_contributes_zero(self):
        from egghouse.dem.response import _fold_lines_into_channel

        out = _fold_lines_into_channel(
            np.array([[3.0]]),
            np.array([500.0]),  # outside the 170-172 grid -> R interpolates to 0
            np.array([170.0, 172.0]),
            np.array([2.0, 2.0]),
            4.0 * np.pi,
        )
        np.testing.assert_allclose(out, [0.0])


def _make_synthetic_ssw_npz(path, *, n_t=11, response_key="response_v10_en"):
    """Build a tiny SSW-shaped .npz fixture for the SSW loader tests.

    Layout matches what ``aia_get_response.pro`` (IDL SSW) produces and what
    demregpy consumes: ``log_temperature`` (n_T,), ``channels`` (n_lambda,),
    one or more response arrays of shape (n_lambda, n_T).
    """
    log_t = np.linspace(5.5, 7.5, n_t)
    channels = np.array([94, 131, 171, 193, 211, 335], dtype=np.int64)
    # Distinguishable per-channel Gaussian peaks so reordering can be verified.
    peaks = {94: 6.8, 131: 5.6, 171: 5.9, 193: 6.2, 211: 6.3, 335: 6.4}
    response = np.zeros((channels.size, log_t.size), dtype=np.float64)
    for i, ch in enumerate(channels):
        response[i] = np.exp(-0.5 * ((log_t - peaks[int(ch)]) / 0.25) ** 2)
    np.savez(
        path,
        log_temperature=log_t,
        channels=channels,
        **{response_key: response},
    )
    return log_t, channels, response


class TestLoadSswTemperatureResponse:
    """Tests for load_ssw_temperature_response and the SSW dispatch in
    get_temperature_response."""

    def test_returns_source_grid_when_no_interpolation(self, tmp_path):
        from egghouse.dem import load_ssw_temperature_response

        path = tmp_path / "ssw.npz"
        log_t, channels, response = _make_synthetic_ssw_npz(path)

        out = load_ssw_temperature_response(path)
        # (n_T, n_lambda), channels in AIA_DEM_WAVELENGTHS order
        assert out.shape == (log_t.size, channels.size)
        np.testing.assert_allclose(out, response.T)

    def test_interpolates_to_target_log_t(self, tmp_path):
        from egghouse.dem import load_ssw_temperature_response

        path = tmp_path / "ssw.npz"
        log_t, _, response = _make_synthetic_ssw_npz(path, n_t=21)

        # Pick a strict subset of the source grid; interp at a known grid
        # point must reproduce the source value exactly.
        target_log_t = log_t[[3, 7, 12, 18]]
        out = load_ssw_temperature_response(path, log_temperatures=target_log_t)
        assert out.shape == (4, 6)
        np.testing.assert_allclose(out, response[:, [3, 7, 12, 18]].T)

    def test_reorders_channels(self, tmp_path):
        from egghouse.dem import load_ssw_temperature_response

        path = tmp_path / "ssw.npz"
        _, _, response = _make_synthetic_ssw_npz(path)

        out = load_ssw_temperature_response(path, wavelengths=[171, 94])
        # Column 0 of the SSW table is 94; column 2 is 171. The output must
        # therefore reorder to (171, 94).
        np.testing.assert_allclose(out[:, 0], response[2])  # 171 row
        np.testing.assert_allclose(out[:, 1], response[0])  # 94  row

    def test_missing_wavelength_raises(self, tmp_path):
        from egghouse.dem import load_ssw_temperature_response

        path = tmp_path / "ssw.npz"
        _make_synthetic_ssw_npz(path)
        with pytest.raises(KeyError, match="not found"):
            load_ssw_temperature_response(path, wavelengths=[999])

    def test_missing_response_key_raises(self, tmp_path):
        from egghouse.dem import load_ssw_temperature_response

        path = tmp_path / "ssw.npz"
        _make_synthetic_ssw_npz(path, response_key="response_v10_en")
        with pytest.raises(KeyError, match="response_v9_en"):
            load_ssw_temperature_response(path, response_key="response_v9_en")

    def test_out_of_range_log_t_raises(self, tmp_path):
        from egghouse.dem import load_ssw_temperature_response

        path = tmp_path / "ssw.npz"
        _make_synthetic_ssw_npz(path)
        with pytest.raises(ValueError, match="outside the source grid"):
            load_ssw_temperature_response(
                path, log_temperatures=np.array([5.0, 6.0])  # 5.0 < min(5.5)
            )

    def test_get_temperature_response_dispatches_to_ssw(self, tmp_path):
        from egghouse.dem import (
            get_default_temperatures,
            load_ssw_temperature_response,
        )
        from egghouse.sdo import get_temperature_response

        path = tmp_path / "ssw.npz"
        _make_synthetic_ssw_npz(path, n_t=21)
        temps = get_default_temperatures(n_bins=11)  # log T 5.5..7.5 linear

        via_get = get_temperature_response(temperatures=temps, ssw_table_path=path)
        via_load = load_ssw_temperature_response(
            path, log_temperatures=np.log10(temps)
        )
        np.testing.assert_allclose(via_get, via_load)


class TestDemSites:
    """Tests for dem_sites function."""

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic test data with known DEM."""
        from egghouse.dem import get_default_temperatures

        # Simple temperature grid
        temps = get_default_temperatures(logt_min=5.8, logt_max=6.8, n_bins=30)
        n_temps = len(temps)
        n_channels = 6

        # Create simple Gaussian response (mock)
        response = np.zeros((n_temps, n_channels), dtype=np.float64)
        peak_temps = [6.8, 5.6, 5.9, 6.2, 6.3, 6.4]  # log T
        for c, peak in enumerate(peak_temps):
            logt = np.log10(temps)
            response[:, c] = 1e-26 * np.exp(-0.5 * ((logt - peak) / 0.3) ** 2)

        # Create known DEM (single Gaussian)
        dem_true = 1e22 * np.exp(-0.5 * ((np.log10(temps) - 6.2) / 0.2) ** 2)

        # Compute synthetic intensities
        logt = np.log10(temps)
        dlogt = np.gradient(logt)
        dt = temps * np.log(10) * dlogt
        intensities = np.sum(response * dem_true[:, np.newaxis] * dt[:, np.newaxis], axis=0)
        errors = intensities * 0.1

        return {
            "temps": temps,
            "response": response,
            "dem_true": dem_true,
            "intensities": intensities,
            "errors": errors,
        }

    def test_dem_sites_single_pixel(self, synthetic_data):
        """Test DEM inversion for single pixel."""
        from egghouse.dem import dem_sites

        dem, info = dem_sites(
            synthetic_data["intensities"],
            synthetic_data["errors"],
            synthetic_data["response"],
            synthetic_data["temps"],
            max_iter=50,
        )

        assert dem.shape == (30,)
        assert info["iterations"] <= 50
        assert "chi2" in info
        assert "converged" in info

    def test_dem_sites_positivity(self, synthetic_data):
        """Test DEM positivity constraint."""
        from egghouse.dem import dem_sites

        dem, _ = dem_sites(
            synthetic_data["intensities"],
            synthetic_data["errors"],
            synthetic_data["response"],
            synthetic_data["temps"],
            positivity=True,
        )

        assert np.all(dem >= 0)

    def test_dem_sites_batch(self, synthetic_data):
        """Test batch DEM inversion."""
        from egghouse.dem import dem_sites

        # Create batch of 10 pixels
        n_pixels = 10
        intensities = np.tile(synthetic_data["intensities"], (n_pixels, 1))
        errors = np.tile(synthetic_data["errors"], (n_pixels, 1))

        dem, info = dem_sites(
            intensities,
            errors,
            synthetic_data["response"],
            synthetic_data["temps"],
        )

        assert dem.shape == (n_pixels, 30)

    def test_dem_sites_shape_mismatch(self, synthetic_data):
        """Test error on shape mismatch."""
        from egghouse.dem import dem_sites

        wrong_response = np.zeros((30, 4))  # Wrong number of channels

        with pytest.raises(ValueError, match="doesn't match"):
            dem_sites(
                synthetic_data["intensities"],
                synthetic_data["errors"],
                wrong_response,
                synthetic_data["temps"],
            )


class TestDemSitesPixel:
    """Tests for dem_sites_pixel function."""

    def test_single_pixel_interface(self):
        """Test simplified single-pixel interface."""
        from egghouse.dem import dem_sites_pixel, get_default_temperatures

        temps = get_default_temperatures(n_bins=25)
        response = np.random.rand(25, 6) * 1e-26
        intensities = np.array([10.0, 50.0, 200.0, 150.0, 80.0, 20.0])
        errors = intensities * 0.1

        dem, info = dem_sites_pixel(intensities, errors, response, temps)

        assert dem.shape == (25,)
        assert "iterations" in info


class TestDemMap:
    """Tests for dem_map function."""

    def test_dem_map_small(self):
        """Test DEM map on small image."""
        from egghouse.dem import dem_map, get_default_temperatures

        temps = get_default_temperatures(n_bins=20)
        response = np.random.rand(20, 6) * 1e-26

        # Small test image
        images = np.random.rand(16, 16, 6).astype(np.float32) * 100
        errors = images * 0.1

        dem_cube, info = dem_map(
            images, errors, response, temps, chunk_size=8, max_iter=10
        )

        assert dem_cube.shape == (16, 16, 20)
        assert "n_pixels" in info
        assert "chi2_map" in info
        assert info["chi2_map"].shape == (16, 16)

    def test_dem_map_with_mask(self):
        """Test DEM map with mask."""
        from egghouse.dem import dem_map, get_default_temperatures

        temps = get_default_temperatures(n_bins=15)
        response = np.random.rand(15, 6) * 1e-26

        images = np.random.rand(8, 8, 6).astype(np.float32) * 100
        errors = images * 0.1

        # Mask only center pixels
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 2:6] = True

        dem_cube, info = dem_map(
            images, errors, response, temps, mask=mask, max_iter=5
        )

        assert dem_cube.shape == (8, 8, 15)
        assert info["n_pixels"] == 16  # 4x4 center region

    def test_dem_map_invalid_shape(self):
        """Test error on invalid input shape."""
        from egghouse.dem import dem_map, get_default_temperatures

        temps = get_default_temperatures(n_bins=10)
        response = np.random.rand(10, 6) * 1e-26

        # 2D image instead of 3D
        images = np.random.rand(32, 32).astype(np.float32)
        errors = images * 0.1

        with pytest.raises(ValueError, match="Expected 3D"):
            dem_map(images, errors, response, temps)


class TestEmissionMeasure:
    """Tests for get_emission_measure function."""

    def test_em_calculation(self):
        """Test emission measure calculation."""
        from egghouse.dem import get_emission_measure, get_default_temperatures

        temps = get_default_temperatures(n_bins=50)
        # Simple DEM
        dem = 1e22 * np.exp(-0.5 * ((np.log10(temps) - 6.2) / 0.3) ** 2)

        em = get_emission_measure(dem, temps)

        assert em > 0
        assert np.isfinite(em)

    def test_em_temperature_range(self):
        """Test emission measure with temperature limits."""
        from egghouse.dem import get_emission_measure, get_default_temperatures

        temps = get_default_temperatures(n_bins=50)
        dem = np.ones(50) * 1e22

        em_full = get_emission_measure(dem, temps)
        em_limited = get_emission_measure(dem, temps, t_min=1e6, t_max=5e6)

        assert em_limited < em_full

    def test_em_batch(self):
        """Test emission measure for batch."""
        from egghouse.dem import get_emission_measure, get_default_temperatures

        temps = get_default_temperatures(n_bins=30)
        dem = np.random.rand(10, 30) * 1e22

        em = get_emission_measure(dem, temps)

        assert em.shape == (10,)


class TestMeanTemperature:
    """Tests for get_mean_temperature function."""

    def test_mean_temperature(self):
        """Test mean temperature calculation."""
        from egghouse.dem import get_mean_temperature, get_default_temperatures

        temps = get_default_temperatures(n_bins=50)
        # Gaussian DEM peaked at log T = 6.2
        dem = 1e22 * np.exp(-0.5 * ((np.log10(temps) - 6.2) / 0.1) ** 2)

        t_mean = get_mean_temperature(dem, temps)

        # Should be close to 10^6.2 K
        assert np.isclose(np.log10(t_mean), 6.2, atol=0.1)

    def test_mean_temperature_batch(self):
        """Test mean temperature for batch."""
        from egghouse.dem import get_mean_temperature, get_default_temperatures

        temps = get_default_temperatures(n_bins=30)
        dem = np.random.rand(5, 30) * 1e22

        t_mean = get_mean_temperature(dem, temps)

        assert t_mean.shape == (5,)
        assert np.all(t_mean > 0)


class TestDemToLoci:
    """Tests for dem_to_loci function."""

    def test_loci_shape(self):
        """Test EM loci curves shape."""
        from egghouse.dem.utils import dem_to_loci

        response = np.random.rand(50, 6) * 1e-26
        intensities = np.array([10.0, 50.0, 200.0, 150.0, 80.0, 20.0])
        temps = np.logspace(5.5, 7.5, 50)

        loci = dem_to_loci(intensities, response, temps)

        assert loci.shape == (50, 6)

    def test_loci_positive(self):
        """Test EM loci values are positive."""
        from egghouse.dem.utils import dem_to_loci

        response = np.random.rand(30, 6) * 1e-26 + 1e-30
        intensities = np.array([10.0, 50.0, 200.0, 150.0, 80.0, 20.0])
        temps = np.logspace(5.5, 7.5, 30)

        loci = dem_to_loci(intensities, response, temps)

        assert np.all(loci >= 0)


class TestComputeDemErrors:
    """Tests for compute_dem_errors function."""

    def test_error_shape(self):
        """Test DEM error shape."""
        from egghouse.dem import get_default_temperatures
        from egghouse.dem.utils import compute_dem_errors

        temps = get_default_temperatures(n_bins=20)
        response = np.random.rand(20, 6) * 1e-26
        intensities = np.array([10.0, 50.0, 200.0, 150.0, 80.0, 20.0])
        errors = intensities * 0.1
        dem = np.random.rand(20) * 1e22

        dem_err = compute_dem_errors(
            dem, intensities, errors, response, temps, n_monte_carlo=10
        )

        assert dem_err.shape == (20,)

    def test_error_positive(self):
        """Test DEM errors are positive."""
        from egghouse.dem import get_default_temperatures
        from egghouse.dem.utils import compute_dem_errors

        temps = get_default_temperatures(n_bins=15)
        response = np.random.rand(15, 6) * 1e-26
        intensities = np.array([10.0, 50.0, 200.0, 150.0, 80.0, 20.0])
        errors = intensities * 0.1
        dem = np.random.rand(15) * 1e22

        dem_err = compute_dem_errors(
            dem, intensities, errors, response, temps, n_monte_carlo=5
        )

        assert np.all(dem_err >= 0)


class TestDemNNLS:
    """Tikhonov-NNLS DEM inversion (synthetic; no CHIANTI/fiasco needed)."""

    def _synthetic(self, n_bins=21):
        from egghouse.dem import get_default_temperatures
        T = get_default_temperatures(n_bins=n_bins)
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
        from egghouse.dem import dem_nnls
        T, R, dt, I = self._synthetic()
        dem, info = dem_nnls(I, I * 0.05, R, T, reg_order=2, reg_scale=1e-3)
        assert dem.shape == (T.size,)
        assert np.all(dem >= 0)
        Isyn = (R * dt[:, None] * dem[:, None]).sum(axis=0)
        np.testing.assert_allclose(Isyn / I, 1.0, rtol=0.1)
        assert "chi2_map" in info

    def test_batch_shape(self):
        from egghouse.dem import dem_nnls
        T, R, dt, I = self._synthetic()
        batch = np.repeat(I[None, :], 3, axis=0)
        dem, info = dem_nnls(batch, batch * 0.05, R, T, reg_scale=1e-3)
        assert dem.shape == (3, T.size)
        assert info["chi2_map"].shape == (3,)

    def test_calibrate_targets_chi2(self):
        from egghouse.dem import calibrate_reg_scale, dem_nnls
        T, R, dt, I = self._synthetic()
        batch = np.repeat(I[None, :], 20, axis=0)
        rs = calibrate_reg_scale(batch, batch * 0.1, R, T, target_chi2=6.0, reg_order=2)
        assert rs > 0
        _, info = dem_nnls(batch, batch * 0.1, R, T, reg_scale=rs)
        assert np.isfinite(info["chi2"])


class TestDemNNLSAdaptiveLambda:
    """Per-pixel discrepancy-principle reg_scale (target_chi2)."""

    def test_target_chi2_is_approached(self):
        from egghouse.dem import dem_nnls, get_default_temperatures
        T = get_default_temperatures(n_bins=21)
        logt = np.log10(T)
        peaks = [6.8, 5.6, 5.9, 6.2, 6.3, 6.4]
        R = np.stack(
            [np.exp(-0.5 * ((logt - p) / 0.2) ** 2) * 1e-24 for p in peaks], axis=1
        )
        dt = T * np.log(10) * np.gradient(logt)
        true = np.exp(-0.5 * ((logt - 6.2) / 0.2) ** 2) * 1e22
        I = (R * dt[:, None] * true[:, None]).sum(axis=0)
        err = I * 0.1
        dem, info = dem_nnls(I, err, R, T, reg_order=2, target_chi2=6.0)
        assert np.all(dem >= 0)
        # adaptive bisection should bring chi^2 close to (or below) the target
        assert info["chi2"] <= 6.0 * 2.5
        assert info["target_chi2"] == 6.0
