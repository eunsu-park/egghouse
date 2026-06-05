"""
Tikhonov-regularized non-negative least squares (NNLS) DEM inversion.

Solves, per pixel, the regularized non-negative least-squares problem

    min_{x >= 0}  || (A x - I) / sigma ||^2 + lambda^2 || L x ||^2

with ``A_{c,t} = K_t(c) * dT_t`` the error-folded response so that
``I = A @ DEM`` (DEM in cm^-5 K^-1), ``L`` a finite-difference smoothness
operator, and ``lambda`` chosen (discrepancy principle) so the data
chi-squared is near the number of channels.

The non-negativity is both physical (DEM >= 0) and a stabiliser for this
ill-posed 6-channel -> many-temperature inversion.

References
----------
- Lawson, C. L. & Hanson, R. J. 1995, *Solving Least Squares Problems*,
  SIAM Classics in Applied Mathematics 15 (orig. 1974) — NNLS active-set.
- Hannah, I. G. & Kontar, E. P. 2012, A&A 539, A146 — regularized DEM
  inversion and chi-squared-based regularization for SDO/AIA.
- Morozov, V. A. 1966, Soviet Math. Dokl. 7, 414 — discrepancy principle
  (choose lambda so chi^2 ~ number of data points).
"""

from typing import Dict, Tuple

import numpy as np
from scipy.optimize import nnls


def _dt(temperatures: np.ndarray) -> np.ndarray:
    """Temperature bin widths dT = T ln(10) dlogT (matches dem_sites)."""
    logt = np.log10(temperatures)
    dlogt = np.gradient(logt)
    return temperatures * np.log(10.0) * dlogt


def _regularization_operator(n_temps: int, order: int) -> np.ndarray:
    """Finite-difference smoothness operator L.

    order=0 -> identity (zeroth-order Tikhonov; penalizes magnitude).
    order=2 -> second difference (penalizes curvature -> smooth DEM).
    """
    if order == 0:
        return np.eye(n_temps, dtype=np.float64)
    if order == 2:
        L = np.zeros((n_temps - 2, n_temps), dtype=np.float64)
        for i in range(n_temps - 2):
            L[i, i], L[i, i + 1], L[i, i + 2] = 1.0, -2.0, 1.0
        return L
    raise ValueError(f"reg_order must be 0 or 2; got {order}")


def _design_matrix(response: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """A with I = A @ DEM. response is (n_temps, n_channels) -> A is (n_channels, n_temps)."""
    return (response * dt[:, np.newaxis]).T


def _solve_pixel(
    intensity: np.ndarray,
    error: np.ndarray,
    A: np.ndarray,
    L: np.ndarray,
    reg_scale: float,
) -> Tuple[np.ndarray, float]:
    """One-pixel Tikhonov-NNLS. Returns (dem, data_chi2).

    ``reg_scale`` is a dimensionless knob; the absolute penalty is
    ``lambda = reg_scale * RMS(A_weighted)`` so it auto-scales with the
    per-pixel problem (brightness/error).
    """
    if not np.all(np.isfinite(intensity)) or np.all(intensity <= 0):
        return np.zeros(A.shape[1], dtype=np.float64), 0.0
    w = 1.0 / np.maximum(error, 1e-30)
    Aw = A * w[:, np.newaxis]
    Iw = intensity * w
    a_scale = float(np.sqrt(np.mean(Aw ** 2))) or 1.0
    lam = reg_scale * a_scale
    A_aug = np.vstack([Aw, lam * L])
    I_aug = np.concatenate([Iw, np.zeros(L.shape[0])])
    dem, _ = nnls(A_aug, I_aug)
    resid = (A @ dem - intensity) * w
    return dem, float(np.sum(resid ** 2))


def _solve_pixel_adaptive(
    intensity: np.ndarray,
    error: np.ndarray,
    A: np.ndarray,
    L: np.ndarray,
    target_chi2: float,
    bounds: Tuple[float, float] = (1e-4, 1e3),
    n_iter: int = 22,
) -> Tuple[np.ndarray, float]:
    """Per-pixel Tikhonov-NNLS with a discrepancy-principle reg_scale.

    Bisects ``reg_scale`` (log space) so the data chi^2 approaches
    ``target_chi2`` (typically n_channels). Data chi^2 increases
    monotonically with ``reg_scale``, so a clean bisection applies. If the
    target lies outside the bracket, returns the nearest-bracket solution.
    """
    if not np.all(np.isfinite(intensity)) or np.all(intensity <= 0):
        return np.zeros(A.shape[1], dtype=np.float64), 0.0
    lo, hi = bounds
    d_lo, c_lo = _solve_pixel(intensity, error, A, L, lo)
    if c_lo >= target_chi2:
        return d_lo, c_lo  # already over-fit floor; cannot regularize less
    d_hi, c_hi = _solve_pixel(intensity, error, A, L, hi)
    if c_hi <= target_chi2:
        return d_hi, c_hi  # cannot fit worse than this; positivity/L floor
    best = (d_hi, c_hi)
    for _ in range(n_iter):
        mid = np.sqrt(lo * hi)
        d_m, c_m = _solve_pixel(intensity, error, A, L, mid)
        best = (d_m, c_m)
        if c_m < target_chi2:
            lo = mid
        else:
            hi = mid
    return best


def calibrate_reg_scale(
    intensities: np.ndarray,
    errors: np.ndarray,
    response: np.ndarray,
    temperatures: np.ndarray,
    *,
    target_chi2: float,
    reg_order: int = 2,
    grid: Tuple[float, float, int] = (-4.0, 2.0, 25),
) -> float:
    """Pick a global ``reg_scale`` so the *median* data chi^2 ~ target_chi2.

    Discrepancy principle (Morozov 1966): scan a log-spaced grid of
    ``reg_scale`` over a sample of pixels and return the value whose median
    chi^2 is closest to ``target_chi2`` (typically the number of channels).
    """
    intensities = np.atleast_2d(intensities)
    errors = np.atleast_2d(errors)
    dt = _dt(temperatures)
    A = _design_matrix(response, dt)
    L = _regularization_operator(len(temperatures), reg_order)
    scales = np.logspace(grid[0], grid[1], int(grid[2]))
    best, best_gap = scales[0], np.inf
    for s in scales:
        chi2s = [_solve_pixel(intensities[p], errors[p], A, L, s)[1]
                 for p in range(intensities.shape[0])]
        med = float(np.median(chi2s))
        gap = abs(med - target_chi2)
        if gap < best_gap:
            best, best_gap = s, gap
    return float(best)


def dem_nnls(
    intensities: np.ndarray,
    errors: np.ndarray,
    response: np.ndarray,
    temperatures: np.ndarray,
    *,
    reg_order: int = 2,
    reg_scale: float = 1e-2,
    target_chi2: float = None,
) -> Tuple[np.ndarray, Dict]:
    """Tikhonov-regularized NNLS DEM inversion (single pixel or batch).

    Parameters
    ----------
    intensities : np.ndarray
        Observed intensities (DN/s/pixel), shape ``(n_channels,)`` or
        ``(n_pixels, n_channels)``.
    errors : np.ndarray
        1-sigma uncertainties, same shape as ``intensities``.
    response : np.ndarray
        Temperature response, shape ``(n_temps, n_channels)`` (same
        convention as :func:`egghouse.sdo.dem.dem_sites`).
    temperatures : np.ndarray
        Temperatures in Kelvin, shape ``(n_temps,)``.
    reg_order : {0, 2}
        Tikhonov order: 0 = magnitude, 2 = second-difference smoothness.
    reg_scale : float
        Fixed dimensionless regularization knob (used when ``target_chi2``
        is None; see :func:`calibrate_reg_scale` to pick a global value).
    target_chi2 : float, optional
        If given, each pixel's ``reg_scale`` is found by a per-pixel
        discrepancy-principle bisection so its data chi^2 approaches this
        value (typically n_channels). Slower but avoids a global reg_scale
        over-fitting bright pixels / under-fitting faint ones. Overrides
        ``reg_scale``.

    Returns
    -------
    dem : np.ndarray
        DEM in cm^-5 K^-1, shape ``(n_temps,)`` or ``(n_pixels, n_temps)``.
    info : dict
        ``chi2`` (mean data chi^2), ``chi2_map`` (per pixel), ``reg_scale``.

    References
    ----------
    Lawson & Hanson (1995); Hannah & Kontar (2012, A&A 539, A146).
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

    dt = _dt(temperatures)
    A = _design_matrix(response, dt)
    L = _regularization_operator(n_temps, reg_order)

    dem = np.zeros((n_pixels, n_temps), dtype=np.float64)
    chi2_map = np.zeros(n_pixels, dtype=np.float64)
    for p in range(n_pixels):
        if target_chi2 is not None:
            dem[p], chi2_map[p] = _solve_pixel_adaptive(
                intensities[p], errors[p], A, L, target_chi2)
        else:
            dem[p], chi2_map[p] = _solve_pixel(intensities[p], errors[p], A, L, reg_scale)

    info = {
        "chi2": float(np.mean(chi2_map)),
        "chi2_map": chi2_map,
        "reg_scale": None if target_chi2 is not None else float(reg_scale),
        "target_chi2": target_chi2,
    }
    if squeeze:
        dem = dem.squeeze()
    return dem, info
