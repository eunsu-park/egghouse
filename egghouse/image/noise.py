"""Robust noise-scale estimation.

Small, dependency-light helpers for estimating the noise scale of an image
from its own pixels using the median absolute deviation (MAD). MAD-based
estimates are robust to bright outliers (hot pixels, transients), unlike the
plain standard deviation.

For a wavelet-detail noise estimate that is robust to smooth scene structure,
use ``skimage.restoration.estimate_sigma`` instead; this module covers the
single-array MAD case used for per-frame normalisation.
"""

from __future__ import annotations

import numpy as np

# Consistency constant: for Gaussian data, std == 1.4826 * MAD.
_MAD_TO_SIGMA = 1.4826


def mad(x: np.ndarray, *, center: float | None = None) -> float:
    """Median absolute deviation about the median.

    Parameters
    ----------
    x : np.ndarray
        Input array (any shape; flattened).
    center : float, optional
        Value to deviate about. Defaults to ``median(x)``.

    Returns
    -------
    float
        ``median(|x - center|)``. Zero for a constant array.
    """
    a = np.asarray(x, dtype=np.float64)
    c = float(np.median(a)) if center is None else float(center)
    return float(np.median(np.abs(a - c)))


def robust_sigma(x: np.ndarray, *, center: float | None = None) -> float:
    """Robust estimate of the noise standard deviation via MAD.

    Returns ``1.4826 * MAD(x)``, the MAD scaled to match the standard
    deviation for Gaussian-distributed data. Robust to outliers.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    center : float, optional
        Value to deviate about. Defaults to ``median(x)``.

    Returns
    -------
    float
        The robust sigma estimate. Zero for a constant array.
    """
    return _MAD_TO_SIGMA * mad(x, center=center)


def gaussian_core_sigma(
    x: np.ndarray,
    *,
    bins: int = 201,
    n_sigma: float = 2.5,
    center: float | None = None,
) -> float:
    """Noise sigma from a Gaussian fit to the core of the value histogram.

    Estimates the noise level of a near-zero-centred image — e.g. a
    line-of-sight magnetogram — by fitting a Gaussian to the low-amplitude
    core of its pixel-value histogram and returning the fitted standard
    deviation. This follows the magnetogram-noise method of Liu et al.
    (2012, Sol. Phys. 279, 295): the distribution of low-field pixels is
    Gaussian (noise), while real magnetic field populates the wings, so the
    width of the central Gaussian is the noise scale. Unlike
    :func:`robust_sigma`, which estimates a robust scale of *all* the data,
    this fits only the noise core and so is the appropriate estimator when a
    single frame's histogram mixes a noise peak with a real-signal tail.

    The fit is done without SciPy: within a window of ``+/- n_sigma`` initial
    robust sigmas about the centre, ``log(counts)`` of the histogram is a
    parabola for Gaussian data, so a count-weighted quadratic least-squares
    fit recovers ``sigma = sqrt(-1 / (2 a2))`` from its leading coefficient.

    Parameters
    ----------
    x : np.ndarray
        Input array (any shape; flattened). NaNs are ignored.
    bins : int, optional
        Number of histogram bins across the fit window. Default 201.
    n_sigma : float, optional
        Half-width of the fit window in units of the initial robust sigma.
        Default 2.5 — wide enough for a stable fit, narrow enough to exclude
        the real-field wings.
    center : float, optional
        Histogram centre. Defaults to ``median(x)``.

    Returns
    -------
    float
        The fitted Gaussian sigma. Falls back to :func:`robust_sigma` when the
        core is degenerate (constant array, too few pixels, or a non-concave
        log-histogram). Zero for a constant array.
    """
    a = np.asarray(x, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0
    c = float(np.median(a)) if center is None else float(center)
    sigma0 = robust_sigma(a, center=c)
    if sigma0 == 0.0:
        return 0.0

    half = n_sigma * sigma0
    core = a[(a >= c - half) & (a <= c + half)]
    if core.size < 10:
        return sigma0

    counts, edges = np.histogram(core, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    keep = counts > 0
    if int(keep.sum()) < 3:
        return sigma0

    # log(Gaussian) is a parabola; weight by sqrt(counts) (Poisson) for the fit.
    a2 = np.polyfit(centers[keep] - c, np.log(counts[keep].astype(np.float64)),
                    2, w=np.sqrt(counts[keep].astype(np.float64)))[0]
    if a2 >= 0:  # not concave -> no Gaussian core; fall back.
        return sigma0
    return float(np.sqrt(-1.0 / (2.0 * a2)))
