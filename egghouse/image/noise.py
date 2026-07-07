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

from typing import NamedTuple

import numpy as np

# Consistency constant: for Gaussian data, std == 1.4826 * MAD.
_MAD_TO_SIGMA = 1.4826


def mad(x: np.ndarray, *, center: float | None = None) -> float:
    """Median absolute deviation about the median.

    Args:
        x: Input array (any shape; flattened).
        center: Value to deviate about. Defaults to ``median(x)``.

    Returns:
        ``median(|x - center|)``. Zero for a constant array.
    """
    a = np.asarray(x, dtype=np.float64)
    c = float(np.median(a)) if center is None else float(center)
    return float(np.median(np.abs(a - c)))


def robust_sigma(x: np.ndarray, *, center: float | None = None) -> float:
    """Robust estimate of the noise standard deviation via MAD.

    Returns ``1.4826 * MAD(x)``, the MAD scaled to match the standard
    deviation for Gaussian-distributed data. Robust to outliers.

    Args:
        x: Input array.
        center: Value to deviate about. Defaults to ``median(x)``.

    Returns:
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

    Args:
        x: Input array (any shape; flattened). NaNs are ignored.
        bins: Number of histogram bins across the fit window. Default 201.
        n_sigma: Half-width of the fit window in units of the initial robust
            sigma. Default 2.5 — wide enough for a stable fit, narrow enough to
            exclude the real-field wings.
        center: Histogram centre. Defaults to ``median(x)``.

    Returns:
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


def photon_transfer_fit(
    intensity: np.ndarray,
    variance: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Linear photon-transfer fit ``variance = g * intensity + r2``.

    Fits the Poisson-Gaussian noise model in which the per-pixel noise variance
    grows linearly with intensity: the slope ``g`` is the gain (variance per
    unit intensity, i.e. DN per photon) and the intercept ``r2`` is the
    read-noise variance. Used to recover the per-channel parameters a
    generalized Anscombe variance-stabilizing transform needs.

    Args:
        intensity, variance: Matched samples of intensity and noise variance
            (e.g. one point per intensity bin). Flattened; non-finite pairs are
            dropped.
        weights: Per-sample fit weights (e.g. pixel counts per bin). Defaults
            to equal.

    Returns:
        (g, r2, r_squared)
        Slope, intercept, and coefficient of determination. ``(nan, nan, nan)``
        if fewer than two valid points remain.
    """
    x = np.asarray(intensity, dtype=np.float64).ravel()
    y = np.asarray(variance, dtype=np.float64).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    w = None
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).ravel()
        m &= np.isfinite(w) & (w > 0)
    if int(m.sum()) < 2:
        return float("nan"), float("nan"), float("nan")
    g, r2 = np.polyfit(x[m], y[m], 1, w=(w[m] if w is not None else None))
    pred = g * x[m] + r2
    ss_res = float(np.sum((y[m] - pred) ** 2))
    ss_tot = float(np.sum((y[m] - y[m].mean()) ** 2)) + 1e-12
    return float(g), float(r2), 1.0 - ss_res / ss_tot


class PoissonGaussianNoise(NamedTuple):
    """Result of :func:`poisson_gaussian_noise`.

    Attributes:
        g: Variance slope (gain; variance per unit intensity).
        r2: Read-noise variance (intercept at zero intensity).
        r_squared: Goodness of fit of ``variance = g * intensity + r2``.
        intensity, variance, count: Per-bin mean intensity, estimated noise
            variance, and pixel count.
    """

    g: float
    r2: float
    r_squared: float
    intensity: np.ndarray
    variance: np.ndarray
    count: np.ndarray


def poisson_gaussian_noise(
    a: np.ndarray,
    b: np.ndarray,
    *,
    bins: int = 30,
    intensity_range: tuple[float, float] | None = None,
    clip: float | None = 6.0,
    min_count: int = 1,
) -> PoissonGaussianNoise:
    """Estimate signal-dependent noise ``sigma^2(I) = g*I + r2`` from a frame pair.

    Given two independent noisy observations ``a`` and ``b`` of the *same*
    scene (e.g. two short-cadence frames), the per-pixel difference
    ``D = a - b`` has ``Var(D) = 2 sigma^2(I)`` at intensity ``I = (a+b)/2``.
    Binning by intensity and fitting ``sigma^2(I) = g*I + r2`` recovers the
    Poisson-Gaussian gain ``g`` and read-noise variance ``r2`` — the
    signal-dependent counterpart of :func:`gaussian_core_sigma` (which assumes
    a single intensity-independent noise scale). This is the photon-transfer /
    variance-vs-mean method; for EUV intensity images (e.g. SDO/AIA) the noise
    is Poisson-dominated and this, not a single-frame histogram width, is the
    appropriate estimator.

    Args:
        a, b: Two noisy frames of the same scene, identical shape and exposure.
        bins: Number of log-spaced intensity bins. Default 30.
        intensity_range: ``(lo, hi)`` intensity range for the bins. Defaults to
            the 1st-99.9th percentiles of the positive intensities.
        clip: Drop pixels whose ``|D|`` exceeds ``clip`` robust sigmas
            (transients / cosmic rays). ``None`` disables clipping. Default 6.0.
        min_count: Minimum pixels per bin for a bin to enter the fit. Default 1.

    Returns:
        ``(g, r2, r_squared, intensity, variance, count)``.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    inten = 0.5 * (a + b)
    diff = a - b
    m = np.isfinite(inten) & np.isfinite(diff) & (inten > 0)
    if clip is not None:
        d = diff[m]
        scale = _MAD_TO_SIGMA * float(np.median(np.abs(d - np.median(d)))) + 1e-12
        m &= np.abs(diff) < clip * scale
    inten, diff = inten[m], diff[m]
    if inten.size < 2:
        nan = float("nan")
        empty = np.empty(0)
        return PoissonGaussianNoise(nan, nan, nan, empty, empty, empty)

    if intensity_range is None:
        lo = max(float(np.percentile(inten, 1.0)), np.finfo(np.float64).tiny)
        hi = float(np.percentile(inten, 99.9))
    else:
        lo, hi = intensity_range
    lo = max(lo, np.finfo(np.float64).tiny)
    edges = np.logspace(np.log10(lo), np.log10(max(hi, lo * 1.0001)), bins + 1)

    idx = np.digitize(inten, edges) - 1
    v = (idx >= 0) & (idx < bins)
    idx, ii, dd = idx[v], inten[v], diff[v]
    count = np.bincount(idx, minlength=bins).astype(np.float64)
    sum_i = np.bincount(idx, weights=ii, minlength=bins)
    sum_d2 = np.bincount(idx, weights=dd * dd, minlength=bins)
    good = count > 0
    i_bin = np.full(bins, np.nan)
    var = np.full(bins, np.nan)
    i_bin[good] = sum_i[good] / count[good]
    var[good] = 0.5 * sum_d2[good] / count[good]  # Var(D)/2 ~ per-frame variance

    fit_mask = good & (count >= min_count)
    g, r2, r_sq = photon_transfer_fit(
        i_bin[fit_mask], var[fit_mask], weights=count[fit_mask])
    return PoissonGaussianNoise(g, r2, r_sq, i_bin, var, count)
