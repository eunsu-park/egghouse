"""
Spline forward-fit DEM inversion (xrt_dem_iterative2 style).

Parameterizes the DEM as a *positive* function by modeling ``log10(DEM)``
as a cubic spline through a small number of knots evenly spaced in
``log10(T)``. Because ``DEM = 10 ** spline``, positivity is automatic and no
inequality constraints are needed. The knot *values* (the spline heights) are
the free parameters; they are forward-fit per pixel with
:func:`scipy.optimize.least_squares` so that the predicted intensities

    I_pred = A @ DEM,   A_{c,t} = K_t(c) * dT_t

match the observed intensities ``I`` in the chi-squared sense
``(I_pred - I) / sigma``.

This mirrors the SolarSoft ``xrt_dem_iterative2.pro`` strategy: a smooth,
low-parameter DEM forward-folded through the temperature response and fit to
the band intensities, with the spline guaranteeing a smooth, strictly
positive solution.

References
----------
- Weber, M. A., DeLuca, E. E., Golub, L. & Sette, A. L. 2004, in IAU Symp.
  223, Multi-Wavelength Investigations of Solar Activity, ed. A. V.
  Stepanov, E. E. Benevolenskaya & A. G. Kosovichev (Cambridge: Cambridge
  Univ. Press), 321 — spline forward-fit DEM (``xrt_dem_iterative2``).
"""

from typing import Dict, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares


def _dt(temperatures: np.ndarray) -> np.ndarray:
    """Temperature bin widths dT = T ln(10) dlogT (matches dem_nnls/dem_sites)."""
    logt = np.log10(temperatures)
    dlogt = np.gradient(logt)
    return temperatures * np.log(10.0) * dlogt


def _design_matrix(response: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """A with I = A @ DEM. response (n_temps, n_channels) -> A (n_channels, n_temps)."""
    return (response * dt[:, np.newaxis]).T


def _dem_from_knots(
    knot_logdem: np.ndarray, knot_logt: np.ndarray, logt: np.ndarray
) -> np.ndarray:
    """Evaluate DEM(logT) from spline knot heights.

    The spline interpolates ``log10(DEM)`` through ``(knot_logt, knot_logdem)``
    and is evaluated on the full ``logt`` grid; ``DEM = 10 ** spline``. The
    exponent is clipped to a sane range to keep the forward model finite while
    the optimizer explores parameter space.
    """
    spline = CubicSpline(knot_logt, knot_logdem, extrapolate=True)
    log_dem = np.clip(spline(logt), -100.0, 100.0)
    return np.power(10.0, log_dem)


def _solve_pixel(
    intensity: np.ndarray,
    error: np.ndarray,
    A: np.ndarray,
    logt: np.ndarray,
    knot_logt: np.ndarray,
    floor_logdem: float,
    max_nfev: int,
) -> Tuple[np.ndarray, float]:
    """One-pixel spline forward-fit. Returns (dem, data_chi2).

    The initial guess is a *flat* DEM whose level reproduces the total
    observed signal (sum of intensities) given the total response, expressed
    as a constant ``log10(DEM)`` at every knot. ``least_squares`` then adjusts
    the knot heights to minimize the error-weighted residuals.
    """
    n_temps = A.shape[1]
    if not np.all(np.isfinite(intensity)) or np.all(intensity <= 0):
        return np.zeros(n_temps, dtype=np.float64), 0.0

    w = 1.0 / np.maximum(error, 1e-30)

    # Flat-DEM initial guess: pick the constant DEM level d0 minimizing the
    # weighted residual of A @ (d0 * ones) vs intensity, i.e. project the data
    # onto the all-ones DEM direction. Fall back to a small positive level.
    ones_pred = A.sum(axis=1)  # I from a flat DEM of unit height
    denom = float(np.sum((ones_pred * w) ** 2))
    if denom > 0:
        d0 = float(np.sum(intensity * w * w * ones_pred) / denom)
    else:
        d0 = 0.0
    log_d0 = np.log10(d0) if d0 > 0 else floor_logdem
    x0 = np.full(knot_logt.size, log_d0, dtype=np.float64)

    def residuals(knot_logdem: np.ndarray) -> np.ndarray:
        dem = _dem_from_knots(knot_logdem, knot_logt, logt)
        return (A @ dem - intensity) * w

    result = least_squares(residuals, x0, method="trf", max_nfev=max_nfev)
    dem = _dem_from_knots(result.x, knot_logt, logt)
    chi2 = float(np.sum(result.fun ** 2))
    return dem, chi2


def dem_spline(
    intensities: np.ndarray,
    errors: np.ndarray,
    response: np.ndarray,
    temperatures: np.ndarray,
    *,
    n_knots: int = 5,
    floor_logdem: float = 15.0,
    max_nfev: int = 200,
) -> Tuple[np.ndarray, Dict]:
    """Spline forward-fit DEM inversion (single pixel or batch).

    Models ``log10(DEM)`` as a cubic spline through ``n_knots`` evenly spaced
    in ``log10(T)``; ``DEM = 10 ** spline`` so the solution is strictly
    positive and smooth by construction. The knot heights are forward-fit per
    pixel with :func:`scipy.optimize.least_squares` minimizing the
    error-weighted intensity residuals ``(A @ DEM - I) / sigma``.

    Parameters
    ----------
    intensities : np.ndarray
        Observed intensities (DN/s/pixel), shape ``(n_channels,)`` or
        ``(n_pixels, n_channels)``.
    errors : np.ndarray
        1-sigma uncertainties, same shape as ``intensities``.
    response : np.ndarray
        Temperature response, shape ``(n_temps, n_channels)`` (same
        convention as :func:`egghouse.dem.dem_nnls`).
    temperatures : np.ndarray
        Temperatures in Kelvin, shape ``(n_temps,)``.
    n_knots : int
        Number of spline knots across the ``log10(T)`` range (default 5).
        Must be >= 2; >= 4 recommended for a cubic spline.
    floor_logdem : float
        Fallback ``log10(DEM)`` for the initial guess when the flat-DEM
        projection is non-positive (default 15.0).
    max_nfev : int
        Maximum residual evaluations per pixel for ``least_squares``.

    Returns
    -------
    dem : np.ndarray
        DEM in cm^-5 K^-1, shape ``(n_temps,)`` or ``(n_pixels, n_temps)``.
    info : dict
        ``chi2`` (mean data chi^2), ``chi2_map`` (per pixel), ``n_knots``,
        ``knot_logt`` (knot positions in log10 T).

    References
    ----------
    Weber et al. (2004, IAU Symp. 223, 321) — ``xrt_dem_iterative2``.
    """
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
    if n_knots < 2:
        raise ValueError(f"n_knots must be >= 2; got {n_knots}")

    logt = np.log10(temperatures)
    dt = _dt(temperatures)
    A = _design_matrix(response, dt)
    knot_logt = np.linspace(logt[0], logt[-1], n_knots)

    dem = np.zeros((n_pixels, n_temps), dtype=np.float64)
    chi2_map = np.zeros(n_pixels, dtype=np.float64)
    for p in range(n_pixels):
        dem[p], chi2_map[p] = _solve_pixel(
            intensities[p], errors[p], A, logt, knot_logt, floor_logdem, max_nfev
        )

    info = {
        "chi2": float(np.mean(chi2_map)),
        "chi2_map": chi2_map,
        "n_knots": int(n_knots),
        "knot_logt": knot_logt,
    }
    if squeeze:
        dem = dem.squeeze()
    return dem, info
