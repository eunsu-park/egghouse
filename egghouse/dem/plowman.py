"""
Fast (essentially non-iterative) linear DEM inversion (Plowman-style).

Implements a single error-weighted, Tikhonov-regularized *linear operator*
``M`` that maps observed intensities to a DEM in one matrix multiply and is
applied to every pixel at once (vectorized, not per-pixel iteration):

    DEM = clip( M @ I_w , 0 ),     M = (A_w^T A_w + lambda^2 L^T L)^{-1} A_w^T

with ``A_{c,t} = K_t(c) * dT_t`` the response so that ``I = A @ DEM``
(DEM in cm^-5 K^-1), ``A_w = A / sigma`` and ``I_w = I / sigma`` the
error-weighted design and data, ``L`` a finite-difference smoothness
operator, and ``lambda`` a single global regularization weight picked
(discrepancy principle) so the typical data chi^2 is near the number of
channels. Positivity (DEM >= 0) is enforced by clipping negative bins.

Relation to Plowman, Kankelborg & Martens (2013):
That paper's headline result is a fast, non-iterative pixel-by-pixel linear
solve that is orders of magnitude faster than iterative DEM codes. This module
keeps that core idea -- precompute one linear inverse operator and apply it to
the whole image with a single ``einsum`` -- and stabilizes the ill-posed
6-channel -> many-temperature problem with Tikhonov regularization plus a
positivity clip. It is *not* a line-by-line reproduction of the paper's basis /
chi^2-bookkeeping; it is a fast linear inversion in the same spirit.

Because the operator is built from one error weighting, batches whose pixels
share the same *relative* errors (e.g. ``errors = f * intensities``, the AIA
photon-noise regime to first order) get an exact per-pixel solve; otherwise the
weighting is the batch-median relative error and per-pixel chi^2 is reported so
deviations are visible.

References:
- Plowman, J., Kankelborg, C. & Martens, P. 2013, ApJ 771, 2.
  DOI 10.1088/0004-637X/771/1/2.
- Hannah, I. G. & Kontar, E. P. 2012, A&A 539, A146 -- chi^2-based
  regularization weight for SDO/AIA DEM inversion.
- Morozov, V. A. 1966, Soviet Math. Dokl. 7, 414 -- discrepancy principle.
"""

from typing import Dict, Tuple

import numpy as np


def _dt(temperatures: np.ndarray) -> np.ndarray:
    """Temperature bin widths dT = T ln(10) dlogT (matches dem_sites/nnls)."""
    logt = np.log10(temperatures)
    dlogt = np.gradient(logt)
    return temperatures * np.log(10.0) * dlogt


def _regularization_operator(n_temps: int, order: int) -> np.ndarray:
    """Finite-difference smoothness operator L.

    order=0 -> identity (penalizes magnitude).
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
    """A with I = A @ DEM. response (n_temps, n_channels) -> A (n_channels, n_temps)."""
    return (response * dt[:, np.newaxis]).T


def _build_operator(
    A: np.ndarray, sigma: np.ndarray, L: np.ndarray, lam: float
) -> np.ndarray:
    """Regularized pseudo-inverse M acting on *error-weighted* intensities.

    Returns ``M`` such that ``DEM ~= M @ (I / sigma)``, the minimizer of
    ``|| A_w x - I_w ||^2 + lam^2 || L x ||^2`` with ``A_w = A/sigma``. Built
    once and reused for every pixel.

    Solved via a stacked least-squares system ``[A_w; lam*L] x = [I_w; 0]``
    rather than the normal equations, so it stays well-conditioned even when
    ``lam`` is small and ``A_w^T A_w + lam^2 L^T L`` would be singular (the
    6-channel -> many-temperature problem is rank-deficient on its own).
    """
    Aw = A / sigma[:, np.newaxis]
    # L (order ~1) and A_w (here ~1e-20 for AIA) live on wildly different
    # scales; scale the penalty by RMS(A_w) so ``lam`` is dimensionless and a
    # value near 1 balances data fit against smoothness regardless of units.
    a_scale = float(np.sqrt(np.mean(Aw ** 2))) or 1.0
    stacked = np.vstack([Aw, lam * a_scale * L])  # (n_channels + n_L, n_temps)
    # M maps a weighted-intensity vector y -> argmin || stacked x - [y; 0] ||.
    # The zero bottom RHS means only the first n_channels columns of the
    # pseudo-inverse act on the data, so M = pinv(stacked)[:, :n_channels].
    return np.linalg.pinv(stacked)[:, : A.shape[0]]


def _typical_chi2(
    intensities: np.ndarray,
    sigma: np.ndarray,
    A: np.ndarray,
    L: np.ndarray,
    lam: float,
) -> float:
    """Median per-pixel data chi^2 for a trial lambda (with positivity clip)."""
    M = _build_operator(A, sigma, L, lam)
    Iw = intensities / sigma[np.newaxis, :]
    dem = np.clip(Iw @ M.T, 0.0, None)
    resid = (dem @ A.T - intensities) / sigma[np.newaxis, :]
    chi2 = np.sum(resid ** 2, axis=1)
    return float(np.median(chi2))


def calibrate_lambda(
    intensities: np.ndarray,
    errors: np.ndarray,
    response: np.ndarray,
    temperatures: np.ndarray,
    *,
    target_chi2: float,
    reg_order: int = 2,
    grid: Tuple[float, float, int] = (-3.0, 3.0, 25),
) -> float:
    """Pick a global ``lambda`` so the *median* data chi^2 ~ target_chi2.

    Discrepancy principle (Morozov 1966): scan a log-spaced grid of lambda over
    a sample of pixels using the batch-median absolute error for the operator,
    and return the lambda whose median chi^2 is closest to ``target_chi2``
    (typically the number of channels).
    """
    intensities = np.atleast_2d(intensities).astype(np.float64)
    errors = np.atleast_2d(errors).astype(np.float64)
    dt = _dt(temperatures)
    A = _design_matrix(response, dt)
    L = _regularization_operator(len(temperatures), reg_order)
    sigma = np.maximum(np.median(errors, axis=0), 1e-30)
    lams = np.logspace(grid[0], grid[1], int(grid[2]))
    best, best_gap = lams[0], np.inf
    for lam in lams:
        med = _typical_chi2(intensities, sigma, A, L, lam)
        gap = abs(med - target_chi2)
        if gap < best_gap:
            best, best_gap = lam, gap
    return float(best)


def dem_plowman(
    intensities: np.ndarray,
    errors: np.ndarray,
    response: np.ndarray,
    temperatures: np.ndarray,
    *,
    reg_order: int = 2,
    reg_lambda: float = None,
    target_chi2: float = None,
) -> Tuple[np.ndarray, Dict]:
    """Fast linear (Plowman-style) DEM inversion (single pixel or batch).

    Builds one error-weighted, Tikhonov-regularized inverse operator and
    applies it to every pixel with a single vectorized matrix multiply, then
    clips negative DEM bins to enforce positivity. This trades the per-pixel
    optimality of NNLS for a large speedup (the selling point of the linear
    method); see the module docstring for the relation to Plowman et al. 2013.

    Args:
        intensities: Observed intensities (DN/s/pixel), shape ``(n_channels,)`` or
            ``(n_pixels, n_channels)``.
        errors: 1-sigma uncertainties, same shape as ``intensities``.
        response: Temperature response, shape ``(n_temps, n_channels)`` (same convention
            as :func:`egghouse.dem.dem_sites`).
        temperatures: Temperatures in Kelvin, shape ``(n_temps,)``.
        reg_order: Tikhonov order: 0 = magnitude, 2 = second-difference smoothness.
        reg_lambda: Global regularization weight. If ``None`` it is calibrated once via
            :func:`calibrate_lambda` so the median chi^2 ~ ``target_chi2``.
        target_chi2: Target chi^2 for auto-calibration. Defaults to the number of channels.

    Returns:
        dem: DEM in cm^-5 K^-1, shape ``(n_temps,)`` or ``(n_pixels, n_temps)``.
            Non-negative.
        info: ``chi2`` (mean data chi^2), ``chi2_map`` (per pixel), ``reg_lambda``.

    References:
        Plowman, Kankelborg & Martens (2013, ApJ 771, 2); Hannah & Kontar (2012).
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

    if reg_lambda is None:
        if target_chi2 is None:
            target_chi2 = float(n_channels)
        reg_lambda = calibrate_lambda(
            intensities, errors, response, temperatures,
            target_chi2=target_chi2, reg_order=reg_order,
        )

    # One global error weighting (batch-median absolute error). For the common
    # AIA regime errors ~ f * intensities the *relative* weighting is identical
    # across pixels, so the precomputed operator is the exact per-pixel solve.
    sigma = np.maximum(np.median(errors, axis=0), 1e-30)
    M = _build_operator(A, sigma, L, reg_lambda)  # (n_temps, n_channels)

    # Vectorized apply across all pixels: DEM = clip(M @ (I / sigma)).
    Iw = intensities / sigma[np.newaxis, :]
    dem = np.clip(Iw @ M.T, 0.0, None)  # (n_pixels, n_temps)

    # Per-pixel data chi^2 using each pixel's own errors.
    resid = (dem @ A.T - intensities) / np.maximum(errors, 1e-30)
    chi2_map = np.sum(resid ** 2, axis=1)

    info = {
        "chi2": float(np.mean(chi2_map)),
        "chi2_map": chi2_map,
        "reg_lambda": float(reg_lambda),
    }
    if squeeze:
        dem = dem.squeeze()
    return dem, info
