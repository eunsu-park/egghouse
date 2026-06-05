"""
Single-Gaussian forward-fit DEM inversion.

Models the differential emission measure as a single Gaussian in log-temperature

    DEM(logT) = EM_peak * exp(-0.5 * ((logT - logT_peak) / sigma)^2)

with ``EM_peak >= 0``, ``logT_peak`` bounded to the coronal range, and
``sigma > 0``. The three parameters are fitted, per pixel, by minimizing the
error-weighted residual ``(A @ DEM - I) / sigma_I`` with
``A_{c,t} = K_t(c) * dT_t`` the response folded with the temperature bin widths
so that ``I = A @ DEM`` (DEM in cm^-5 K^-1). Because the forward model is
strictly positive and low-dimensional (3 parameters), the inversion is
well-posed and robust even for the 6-channel SDO/AIA problem.

This single-peak parameterization is deliberately restrictive: it *cannot*
represent multi-thermal or dual-peak plasma (e.g. a cool loop plus a hot flare
component along the same line of sight). For such structure use a free-form
inversion (``dem_nnls``) or a sparse / basis-pursuit inversion (``dem_sparse``).

References
----------
- Aschwanden, M. J., Boerner, P., Schrijver, C. J. & Malanushenko, A. 2013,
  Sol. Phys. 283, 5. DOI 10.1007/s11207-011-9876-5 — Gaussian forward-fit DEM
  for SDO/AIA.
"""

from typing import Dict, Tuple

import numpy as np
from scipy.optimize import least_squares


def _dt(temperatures: np.ndarray) -> np.ndarray:
    """Temperature bin widths dT = T ln(10) dlogT (matches dem_nnls/dem_sites)."""
    logt = np.log10(temperatures)
    dlogt = np.gradient(logt)
    return temperatures * np.log(10.0) * dlogt


def _design_matrix(response: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """A with I = A @ DEM. response (n_temps, n_channels) -> A (n_channels, n_temps)."""
    return (response * dt[:, np.newaxis]).T


def _gaussian_dem(params: np.ndarray, logt: np.ndarray) -> np.ndarray:
    """Single Gaussian DEM(logT) from [em_peak, logt_peak, sigma]."""
    em_peak, logt_peak, sigma = params
    return em_peak * np.exp(-0.5 * ((logt - logt_peak) / sigma) ** 2)


def _initial_guess(
    intensity: np.ndarray,
    A: np.ndarray,
    logt: np.ndarray,
    logt_bounds: Tuple[float, float],
) -> np.ndarray:
    """Sensible [em_peak, logt_peak, sigma] start from the observed intensities.

    ``logt_peak`` is the response-weighted mean temperature of the observed
    signal: each channel votes for the temperature where its response peaks,
    weighted by how bright that channel is. ``em_peak`` is set so the modeled
    total signal roughly matches the observed total. ``sigma`` starts at a
    moderate width spanning a few temperature bins.
    """
    # Temperature at which each channel's response peaks (n_channels,).
    peak_logt = logt[np.argmax(A, axis=1)]
    weights = np.clip(intensity, 0.0, None)
    if weights.sum() > 0:
        logt_peak = float(np.average(peak_logt, weights=weights))
    else:
        logt_peak = float(np.mean(logt_bounds))
    logt_peak = float(np.clip(logt_peak, logt_bounds[0], logt_bounds[1]))

    sigma0 = 0.2
    # Choose em_peak so sum(A @ DEM) ~ sum(I) for the trial Gaussian shape.
    trial = _gaussian_dem(np.array([1.0, logt_peak, sigma0]), logt)
    model_total = float((A @ trial).sum())
    obs_total = float(np.clip(intensity, 0.0, None).sum())
    em_peak = obs_total / model_total if model_total > 0 else 1.0
    em_peak = max(em_peak, 1e-30)
    return np.array([em_peak, logt_peak, sigma0], dtype=np.float64)


def _solve_pixel(
    intensity: np.ndarray,
    error: np.ndarray,
    A: np.ndarray,
    logt: np.ndarray,
    logt_bounds: Tuple[float, float],
    sigma_bounds: Tuple[float, float],
    max_nfev: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """One-pixel Gaussian forward-fit. Returns (dem, params, data_chi2).

    ``params`` is ``[em_peak, logt_peak, sigma]``. A dead pixel (non-finite or
    all-nonpositive intensities) returns a zero DEM and ``chi2 = 0``.
    """
    n_temps = A.shape[1]
    if not np.all(np.isfinite(intensity)) or np.all(intensity <= 0):
        return (
            np.zeros(n_temps, dtype=np.float64),
            np.array([0.0, float(np.mean(logt_bounds)), sigma_bounds[0]]),
            0.0,
        )

    w = 1.0 / np.maximum(error, 1e-30)

    def residuals(params: np.ndarray) -> np.ndarray:
        dem = _gaussian_dem(params, logt)
        return (A @ dem - intensity) * w

    p0 = _initial_guess(intensity, A, logt, logt_bounds)
    # em_peak scale sets a sane upper bound for the amplitude search.
    em_upper = max(p0[0] * 1e6, 1e-30)
    lower = np.array([0.0, logt_bounds[0], sigma_bounds[0]])
    upper = np.array([em_upper, logt_bounds[1], sigma_bounds[1]])
    p0 = np.clip(p0, lower, upper)

    res = least_squares(
        residuals,
        p0,
        bounds=(lower, upper),
        max_nfev=max_nfev,
        method="trf",
    )
    params = np.asarray(res.x, dtype=np.float64)
    dem = _gaussian_dem(params, logt)
    chi2 = float(np.sum(res.fun ** 2))
    return dem, params, chi2


def dem_gaussian(
    intensities: np.ndarray,
    errors: np.ndarray,
    response: np.ndarray,
    temperatures: np.ndarray,
    *,
    logt_bounds: Tuple[float, float] = (5.5, 7.5),
    sigma_bounds: Tuple[float, float] = (0.05, 1.0),
    max_nfev: int = 200,
) -> Tuple[np.ndarray, Dict]:
    """Single-Gaussian forward-fit DEM inversion (single pixel or batch).

    Per pixel, fits a single Gaussian in log-temperature

        DEM(logT) = EM_peak * exp(-0.5 * ((logT - logT_peak) / sigma)^2)

    to the observed intensities by minimizing the error-weighted residual
    ``(A @ DEM - I) / sigma_I`` over the three parameters ``EM_peak >= 0``,
    ``logT_peak`` (bounded) and ``sigma > 0`` (bounded), using
    :func:`scipy.optimize.least_squares` (Aschwanden et al. 2013).

    Parameters
    ----------
    intensities : np.ndarray
        Observed intensities (DN/s/pixel), shape ``(n_channels,)`` or
        ``(n_pixels, n_channels)``.
    errors : np.ndarray
        1-sigma uncertainties, same shape as ``intensities``.
    response : np.ndarray
        Temperature response, shape ``(n_temps, n_channels)`` (same convention
        as :func:`egghouse.sdo.dem.dem_sites` / :func:`dem_nnls`).
    temperatures : np.ndarray
        Temperatures in Kelvin, shape ``(n_temps,)``.
    logt_bounds : (float, float)
        Inclusive bounds on the fitted peak log-temperature ``logT_peak``.
    sigma_bounds : (float, float)
        Bounds on the fitted Gaussian width ``sigma`` (in log-temperature).
        The lower bound must be > 0.
    max_nfev : int
        Maximum function evaluations per pixel for the least-squares solver.

    Returns
    -------
    dem : np.ndarray
        DEM in cm^-5 K^-1, shape ``(n_temps,)`` or ``(n_pixels, n_temps)``.
    info : dict
        ``chi2`` (mean data chi^2), ``chi2_map`` (per pixel), and the fitted
        parameter maps ``em_peak``, ``logt_peak``, ``sigma`` (each shape
        ``()`` for a single pixel or ``(n_pixels,)`` for a batch).

    Notes
    -----
    The single-Gaussian model assumes an *isothermal-ish*, single-peaked DEM.
    It cannot represent multi-thermal or dual-peak plasma; for such structure
    prefer :func:`dem_nnls` (free-form) or :func:`dem_sparse` (sparse).

    References
    ----------
    Aschwanden, M. J., et al. 2013, Sol. Phys. 283, 5.
    """
    if sigma_bounds[0] <= 0.0:
        raise ValueError(f"sigma lower bound must be > 0; got {sigma_bounds[0]}")

    squeeze = intensities.ndim == 1
    intensities = np.atleast_2d(intensities).astype(np.float64)
    errors = np.atleast_2d(errors).astype(np.float64)
    n_pixels, n_channels = intensities.shape
    n_temps = len(temperatures)
    if response.shape != (n_temps, n_channels):
        raise ValueError(
            f"Response shape {response.shape} doesn't match "
            f"expected ({n_temps}, {n_channels})"
        )

    dt = _dt(temperatures)
    A = _design_matrix(response, dt)
    logt = np.log10(temperatures)

    dem = np.zeros((n_pixels, n_temps), dtype=np.float64)
    chi2_map = np.zeros(n_pixels, dtype=np.float64)
    em_peak = np.zeros(n_pixels, dtype=np.float64)
    logt_peak = np.zeros(n_pixels, dtype=np.float64)
    sigma = np.zeros(n_pixels, dtype=np.float64)
    for p in range(n_pixels):
        dem[p], params, chi2_map[p] = _solve_pixel(
            intensities[p], errors[p], A, logt, logt_bounds, sigma_bounds, max_nfev
        )
        em_peak[p], logt_peak[p], sigma[p] = params

    info = {
        "chi2": float(np.mean(chi2_map)),
        "chi2_map": chi2_map,
        "em_peak": em_peak,
        "logt_peak": logt_peak,
        "sigma": sigma,
    }
    if squeeze:
        dem = dem.squeeze()
        info["em_peak"] = em_peak.squeeze()
        info["logt_peak"] = logt_peak.squeeze()
        info["sigma"] = sigma.squeeze()
    return dem, info
