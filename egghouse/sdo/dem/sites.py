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
    The SITES algorithm works by iteratively updating the DEM:

    1. Initialize DEM from observations
    2. Compute synthetic intensities: I_syn = R @ DEM * dT
    3. Compute residuals: delta_I = I_obs - I_syn
    4. Update DEM: DEM += alpha * delta_I / R_weight
    5. Apply positivity constraint
    6. Check convergence

    References
    ----------
    Morgan & Pickering (2019), Solar Physics 294, 135
    DOI: 10.1007/s11207-019-1525-4

    Examples
    --------
    >>> import numpy as np
    >>> from egghouse.sdo.dem import get_temperature_response, get_default_temperatures
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

    # Response weights for update step
    response_sum = np.sum(response, axis=1)
    response_sum = np.maximum(response_sum, 1e-30)  # Avoid division by zero

    # Initialize DEM
    dem = _initialize_dem(intensities, response, temperatures, dt)

    # Iterative solver
    chi2_history = []
    converged = False

    for iteration in range(max_iter):
        # Compute synthetic intensities: I_syn = sum_T(R * DEM * dT)
        # Shape: (n_pixels, n_channels)
        synthetic = np.einsum("tc,pt,t->pc", response, dem, dt)

        # Compute chi-squared
        residuals = intensities - synthetic
        chi2 = np.sum((residuals / errors) ** 2, axis=1)
        chi2_mean = np.mean(chi2)
        chi2_history.append(chi2_mean)

        # Check convergence
        if iteration > 0:
            rel_change = abs(chi2_history[-1] - chi2_history[-2]) / (
                chi2_history[-2] + 1e-10
            )
            if rel_change < tol:
                converged = True
                break

        # Update DEM
        # delta_DEM = alpha * (I_obs - I_syn) / R_weight
        alpha = 0.5  # Relaxation factor
        for c in range(n_channels):
            update = alpha * residuals[:, c:c+1] * response[:, c] / response_sum
            dem += update

        # Apply positivity constraint
        if positivity:
            dem = np.maximum(dem, 0.0)

        # Apply regularization (smoothness)
        if regularization > 0:
            dem = _apply_smoothing(dem, regularization)

    # Prepare output
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


def _initialize_dem(
    intensities: np.ndarray,
    response: np.ndarray,
    temperatures: np.ndarray,
    dt: np.ndarray,
) -> np.ndarray:
    """
    Initialize DEM estimate from observations.

    Uses a simple back-projection method:
        DEM_init = I / (sum_c R_c * dT)

    Parameters
    ----------
    intensities : np.ndarray
        Observed intensities, shape (n_pixels, n_channels).
    response : np.ndarray
        Response matrix, shape (n_temps, n_channels).
    temperatures : np.ndarray
        Temperature array.
    dt : np.ndarray
        Temperature bin widths.

    Returns
    -------
    np.ndarray
        Initial DEM estimate, shape (n_pixels, n_temps).
    """
    n_pixels, n_channels = intensities.shape
    n_temps = len(temperatures)

    # Sum of response over channels, weighted by dT
    response_integral = np.sum(response * dt[:, np.newaxis], axis=0)
    response_integral = np.maximum(response_integral, 1e-30)

    # Back-projection
    dem_init = np.zeros((n_pixels, n_temps), dtype=np.float64)

    # Weighted average across channels
    for c in range(n_channels):
        weight = response[:, c] / response_integral[c]
        dem_init += intensities[:, c:c+1] * weight

    # Normalize by temperature bin width
    dem_init /= dt

    # Ensure positivity
    dem_init = np.maximum(dem_init, 0.0)

    return dem_init


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
