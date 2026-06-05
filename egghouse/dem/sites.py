"""
SITES (Simple Iterative Temperature Emission Solver) algorithm.

Implementation of the SITES DEM inversion method described in:
    Morgan & Pickering (2019), Solar Physics 294, 135
    DOI: 10.1007/s11207-019-1525-4

The algorithm iteratively adjusts DEM values to minimize the difference
between observed and synthetic intensities, with positivity constraint.
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np


def dem_sites(
    intensities: np.ndarray,
    errors: np.ndarray,
    response: np.ndarray,
    temperatures: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-3,
    positivity: bool = True,
    regularization: float = 0.0,
) -> Tuple[np.ndarray, Dict]:
    """
    Compute DEM using SITES algorithm.

    SITES (Simple Iterative Temperature Emission Solver) inverts
    multi-wavelength observations to obtain the Differential Emission
    Measure as a function of temperature.

    Parameters
    ----------
    intensities : np.ndarray
        Observed intensities (DN/s/pixel).
        Shape: (n_channels,) for single pixel, or
               (n_pixels, n_channels) for batch processing.
    errors : np.ndarray
        Intensity uncertainties, same shape as intensities.
    response : np.ndarray
        Temperature response matrix, shape (n_temps, n_channels).
    temperatures : np.ndarray
        Temperature array in Kelvin, shape (n_temps,).
    max_iter : int, optional
        Maximum number of iterations. Default: 100.
    tol : float, optional
        Convergence tolerance (relative change in chi-squared).
        Default: 1e-3.
    positivity : bool, optional
        Enforce positivity constraint on DEM. Default: True.
    regularization : float, optional
        Regularization parameter (smoothness constraint). Default: 0.0.

    Returns
    -------
    dem : np.ndarray
        DEM solution, shape (n_temps,) or (n_pixels, n_temps).
        Units: cm^-5 K^-1.
    info : dict
        Convergence information:
        - "iterations": number of iterations performed
        - "converged": whether convergence was achieved
        - "chi2": final chi-squared value
        - "chi2_history": chi-squared per iteration
        - "residuals": final intensity residuals

    Notes
    -----
    The solver uses a multiplicative (MART / EM-style) iteration, which
    preserves positivity and self-scales the DEM magnitude:

    1. Initialize a strictly-positive flat DEM reproducing the total signal.
    2. Compute synthetic intensities: I_syn = R @ DEM * dT.
    3. Scale each temperature bin by the error-weighted, response-weighted
       ratio of observed to synthetic intensity:
       DEM(T) *= [sum_c w_c R_c(T) (I_obs/I_syn)_c] / [sum_c w_c R_c(T)].
    4. Check chi-squared convergence.

    For an exact, deterministic alternative see ``dem_nnls``
    (Tikhonov-regularized non-negative least squares).

    References
    ----------
    Morgan & Pickering (2019), Solar Physics 294, 135
    DOI: 10.1007/s11207-019-1525-4

    Examples
    --------
    >>> import numpy as np
    >>> from egghouse.dem import get_temperature_response, get_default_temperatures
    >>> temps = get_default_temperatures(n_bins=50)
    >>> response = get_temperature_response(temperatures=temps)
    >>> intensities = np.array([10.0, 50.0, 200.0, 150.0, 80.0, 20.0])
    >>> errors = intensities * 0.1
    >>> dem, info = dem_sites(intensities, errors, response, temps)
    >>> dem.shape
    (50,)
    """
    # Validate inputs - check original shape before converting to 2D
    squeeze_output = intensities.ndim == 1

    intensities = np.atleast_2d(intensities)
    errors = np.atleast_2d(errors)

    n_pixels, n_channels = intensities.shape
    n_temps = len(temperatures)

    if response.shape != (n_temps, n_channels):
        raise ValueError(
            f"Response shape {response.shape} doesn't match "
            f"expected ({n_temps}, {n_channels})"
        )

    # Compute temperature bin widths (dT)
    logt = np.log10(temperatures)
    dlogt = np.gradient(logt)
    dt = temperatures * np.log(10) * dlogt  # dT = T * ln(10) * d(log T)

    # Error weights (1/sigma^2) and the design matrix A[c,t] = R[t,c]*dT[t],
    # so synthetic intensities are I_syn = DEM @ A.T.
    eps = 1e-300
    weights = 1.0 / np.maximum(errors, 1e-30) ** 2           # (n_pixels, n_channels)
    A = (response * dt[:, np.newaxis]).T                     # (n_channels, n_temps)

    # Strictly-positive initialization (a flat DEM reproducing the total
    # signal) — required because the update is multiplicative.
    a_total = max(float(A.sum()), eps)
    dem = np.maximum(intensities.sum(axis=1, keepdims=True), eps) / a_total
    dem = np.repeat(dem, n_temps, axis=1)                    # (n_pixels, n_temps)

    chi2_history = []
    converged = False
    residuals = intensities - dem @ A.T
    chi2_mean = float(np.mean(np.sum(weights * residuals ** 2, axis=1)))

    for iteration in range(max_iter):
        synthetic = np.maximum(dem @ A.T, eps)              # (n_pixels, n_channels)
        residuals = intensities - synthetic
        chi2_mean = float(np.mean(np.sum(weights * residuals ** 2, axis=1)))
        chi2_history.append(chi2_mean)

        if iteration > 0:
            rel_change = abs(chi2_history[-1] - chi2_history[-2]) / (
                chi2_history[-2] + 1e-10
            )
            if rel_change < tol:
                converged = True
                break

        # MART / EM multiplicative update (Morgan & Pickering 2019): scale
        # each temperature bin by the response-weighted ratio of observed to
        # synthetic intensity. Multiplicative => preserves positivity and
        # self-scales the DEM magnitude.
        weighted_ratio = weights * (intensities / synthetic)   # (n_pixels, n_channels)
        numer = weighted_ratio @ A                             # (n_pixels, n_temps)
        denom = np.maximum(weights @ A, eps)
        dem = dem * (numer / denom)

        if positivity:
            dem = np.maximum(dem, 0.0)
        if regularization > 0:
            dem = np.maximum(_apply_smoothing(dem, regularization), eps)

    info = {
        "iterations": iteration + 1,
        "converged": converged,
        "chi2": chi2_mean,
        "chi2_history": np.array(chi2_history),
        "residuals": residuals,
    }

    if squeeze_output:
        dem = dem.squeeze()

    return dem, info


def dem_sites_pixel(
    intensities: np.ndarray,
    errors: np.ndarray,
    response: np.ndarray,
    temperatures: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-3,
) -> Tuple[np.ndarray, Dict]:
    """
    Compute DEM for a single pixel using SITES algorithm.

    This is a simplified interface for single-pixel DEM inversion.

    Parameters
    ----------
    intensities : np.ndarray
        Observed intensities (DN/s/pixel), shape (n_channels,).
    errors : np.ndarray
        Intensity uncertainties, shape (n_channels,).
    response : np.ndarray
        Temperature response matrix, shape (n_temps, n_channels).
    temperatures : np.ndarray
        Temperature array in Kelvin, shape (n_temps,).
    max_iter : int, optional
        Maximum number of iterations. Default: 100.
    tol : float, optional
        Convergence tolerance. Default: 1e-3.

    Returns
    -------
    dem : np.ndarray
        DEM solution, shape (n_temps,).
    info : dict
        Convergence information.

    Examples
    --------
    >>> import numpy as np
    >>> temps = np.logspace(5.5, 7.5, 50)
    >>> response = np.random.rand(50, 6) * 1e-26
    >>> intensities = np.array([10.0, 50.0, 200.0, 150.0, 80.0, 20.0])
    >>> errors = intensities * 0.1
    >>> dem, info = dem_sites_pixel(intensities, errors, response, temps)
    >>> dem.shape
    (50,)
    """
    return dem_sites(
        intensities,
        errors,
        response,
        temperatures,
        max_iter=max_iter,
        tol=tol,
    )


def _apply_smoothing(
    dem: np.ndarray,
    strength: float,
) -> np.ndarray:
    """
    Apply smoothing regularization to DEM.

    Parameters
    ----------
    dem : np.ndarray
        DEM array, shape (n_pixels, n_temps).
    strength : float
        Smoothing strength (0-1).

    Returns
    -------
    np.ndarray
        Smoothed DEM.
    """
    from scipy.ndimage import gaussian_filter1d

    smoothed = gaussian_filter1d(dem, sigma=strength, axis=1)
    return smoothed


def compute_synthetic_intensities(
    dem: np.ndarray,
    response: np.ndarray,
    temperatures: np.ndarray,
) -> np.ndarray:
    """
    Compute synthetic intensities from DEM.

    I = integral(K(T) * DEM(T) * dT)

    Parameters
    ----------
    dem : np.ndarray
        DEM solution, shape (n_temps,) or (n_pixels, n_temps).
    response : np.ndarray
        Temperature response, shape (n_temps, n_channels).
    temperatures : np.ndarray
        Temperature array.

    Returns
    -------
    np.ndarray
        Synthetic intensities, shape (n_channels,) or (n_pixels, n_channels).
    """
    dem = np.atleast_2d(dem)
    squeeze = dem.shape[0] == 1

    # Compute dT
    logt = np.log10(temperatures)
    dlogt = np.gradient(logt)
    dt = temperatures * np.log(10) * dlogt

    # Compute intensities
    synthetic = np.einsum("tc,pt,t->pc", response, dem, dt)

    if squeeze:
        synthetic = synthetic.squeeze()

    return synthetic
