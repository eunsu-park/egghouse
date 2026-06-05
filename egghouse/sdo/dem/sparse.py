"""
Sparse / Basis-Pursuit DEM inversion via linear programming.

Solves, per pixel, the linear program

    min_{x >= 0}  sum_t x_t      subject to   |A x - I| <= n_sigma * sigma

with ``A_{c,t} = K_t(c) * dT_t`` the response folded with the temperature bin
widths so that ``I = A @ DEM`` (DEM in cm^-5 K^-1, x_t = DEM(T_t)). The L1
objective (sum of non-negative coefficients) together with the positivity
constraint and the per-channel data tolerance drives a *sparse* DEM: a
basis-pursuit solution that explains the n_channels intensities with as few
populated temperature bins as possible. The temperature bins are used directly
as the basis.

References
----------
- Cheung, M. C. M., Boerner, P., Schrijver, C. J., et al. 2015, ApJ 807, 143.
  DOI 10.1088/0004-637X/807/2/143 — Basis-pursuit / sparse DEM inversion for
  SDO/AIA.
"""

from typing import Dict, Tuple

import numpy as np
from scipy.optimize import linprog


def _dt(temperatures: np.ndarray) -> np.ndarray:
    """Temperature bin widths dT = T ln(10) dlogT (matches dem_nnls/dem_sites)."""
    logt = np.log10(temperatures)
    dlogt = np.gradient(logt)
    return temperatures * np.log(10.0) * dlogt


def _design_matrix(response: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """A with I = A @ DEM. response (n_temps, n_channels) -> A (n_channels, n_temps)."""
    return (response * dt[:, np.newaxis]).T


def _solve_pixel(
    intensity: np.ndarray,
    error: np.ndarray,
    A: np.ndarray,
    col_scale: np.ndarray,
    n_sigma: float,
    relax_factor: float,
    max_relax: int,
) -> Tuple[np.ndarray, float, bool]:
    """One-pixel basis-pursuit DEM. Returns (dem, data_chi2, feasible_flag).

    The LP solved (in scaled variables y, with x = DEM = y * iscale/col_scale,
    ``As = A / col_scale`` and intensities divided by a single ``iscale``) is

        min sum(y)   s.t.   As y <=  Is + tol
                           -As y <= -Is + tol
                           y >= 0

    The column scaling (each basis column normalized by its peak channel
    response) and the single intensity scale keep the design matrix
    well-conditioned, which is essential here because the raw response spans
    ~20 orders of magnitude and an unscaled LP is reported infeasible by HiGHS.

    If the LP is infeasible (the positive cone cannot reach the data box), the
    tolerance ``n_sigma`` is multiplicatively relaxed up to ``max_relax`` times.
    A pixel that never becomes feasible returns zeros with ``feasible=False``.
    """
    n_temps = A.shape[1]
    if not np.all(np.isfinite(intensity)) or np.all(intensity <= 0):
        return np.zeros(n_temps, dtype=np.float64), 0.0, True

    w = 1.0 / np.maximum(error, 1e-30)
    iscale = float(np.max(intensity))
    if iscale <= 0:
        return np.zeros(n_temps, dtype=np.float64), 0.0, True

    # Well-conditioned scaled system.
    As = A / col_scale[np.newaxis, :]
    Is = intensity / iscale
    errs = error / iscale
    y_to_x = iscale / col_scale  # maps scaled coefficients y back to DEM x

    # L1 objective: minimize sum of (scaled) DEM coefficients -> sparse solution.
    c = np.ones(n_temps, dtype=np.float64)
    # Box constraint |As y - Is| <= tol  ->  two stacked inequality blocks.
    A_ub_base = np.vstack([As, -As])

    sigma = n_sigma
    for _ in range(max_relax + 1):
        tol = sigma * errs
        b_ub = np.concatenate([Is + tol, -Is + tol])
        res = linprog(
            c,
            A_ub=A_ub_base,
            b_ub=b_ub,
            bounds=[(0.0, None)] * n_temps,
            method="highs",
        )
        if res.success:
            dem = np.clip(np.asarray(res.x, dtype=np.float64) * y_to_x, 0.0, None)
            resid = (A @ dem - intensity) * w
            feasible = sigma <= n_sigma * (1.0 + 1e-9)
            return dem, float(np.sum(resid ** 2)), feasible
        sigma *= relax_factor

    return np.zeros(n_temps, dtype=np.float64), np.inf, False


def dem_sparse(
    intensities: np.ndarray,
    errors: np.ndarray,
    response: np.ndarray,
    temperatures: np.ndarray,
    *,
    n_sigma: float = 1.0,
    relax_factor: float = 1.5,
    max_relax: int = 6,
) -> Tuple[np.ndarray, Dict]:
    """Sparse / basis-pursuit DEM inversion (single pixel or batch).

    Per pixel, finds the non-negative DEM that minimizes the L1 norm of the
    coefficients (``sum_t DEM_t``) while fitting the observed intensities to
    within ``n_sigma`` standard deviations. The positivity + L1 objective
    yields a sparse DEM (Cheung et al. 2015).

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
    n_sigma : float
        Data-fit tolerance: the fit must satisfy ``|A@DEM - I| <= n_sigma*sigma``
        for every channel. Larger values give sparser, looser fits.
    relax_factor : float
        Multiplicative factor by which ``n_sigma`` is enlarged when a pixel's
        LP is infeasible (must be > 1).
    max_relax : int
        Maximum number of relaxation steps before giving up on a pixel (then
        returns zeros for that pixel and flags it infeasible).

    Returns
    -------
    dem : np.ndarray
        DEM in cm^-5 K^-1, shape ``(n_temps,)`` or ``(n_pixels, n_temps)``.
    info : dict
        ``chi2`` (mean data chi^2 over feasible pixels), ``chi2_map``
        (per-pixel data chi^2), ``n_sigma``, ``feasible_map`` (bool per pixel),
        ``n_infeasible`` (count of pixels needing relaxation or returning zeros).

    References
    ----------
    Cheung, M. C. M., et al. 2015, ApJ 807, 143.
    """
    if relax_factor <= 1.0:
        raise ValueError(f"relax_factor must be > 1; got {relax_factor}")

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
    # Per-column normalization (peak channel response of each basis column).
    # Shared across pixels since it depends only on A; keeps the LP conditioned.
    col_scale = np.max(A, axis=0)
    col_scale[col_scale <= 0] = 1.0

    dem = np.zeros((n_pixels, n_temps), dtype=np.float64)
    chi2_map = np.zeros(n_pixels, dtype=np.float64)
    feasible_map = np.ones(n_pixels, dtype=bool)
    for p in range(n_pixels):
        dem[p], chi2_map[p], feasible_map[p] = _solve_pixel(
            intensities[p], errors[p], A, col_scale, n_sigma, relax_factor, max_relax
        )

    finite = np.isfinite(chi2_map)
    mean_chi2 = float(np.mean(chi2_map[finite])) if np.any(finite) else float("inf")
    info = {
        "chi2": mean_chi2,
        "chi2_map": chi2_map,
        "n_sigma": float(n_sigma),
        "feasible_map": feasible_map,
        "n_infeasible": int(np.sum(~feasible_map)),
    }
    if squeeze:
        dem = dem.squeeze()
    return dem, info
