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
