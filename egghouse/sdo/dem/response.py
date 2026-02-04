"""
Temperature response functions for AIA channels.

This module provides functions to obtain temperature response functions
for SDO/AIA EUV channels, either via aiapy (if available) or using
pre-computed fallback tables.

References:
    - Boerner et al. (2012), Solar Physics 275, 41-66
    - Boerner et al. (2014), Solar Physics 289, 2377-2397
"""

from datetime import datetime
from typing import List, Optional, Union

import numpy as np

# Check for aiapy availability
try:
    import aiapy.response
    from aiapy.calibrate import degradation
    HAS_AIAPY = True
except ImportError:
    HAS_AIAPY = False

# Standard AIA EUV wavelengths for DEM analysis
AIA_DEM_WAVELENGTHS = [94, 131, 171, 193, 211, 335]

# Default temperature grid (log10 K)
DEFAULT_LOGT_MIN = 5.5
DEFAULT_LOGT_MAX = 7.5
DEFAULT_LOGT_BINS = 100


def get_default_temperatures(
    logt_min: float = DEFAULT_LOGT_MIN,
    logt_max: float = DEFAULT_LOGT_MAX,
    n_bins: int = DEFAULT_LOGT_BINS,
) -> np.ndarray:
    """
    Get default temperature array for DEM analysis.

    Parameters
    ----------
    logt_min : float, optional
        Minimum log10(T/K), default 5.5.
    logt_max : float, optional
        Maximum log10(T/K), default 7.5.
    n_bins : int, optional
        Number of temperature bins, default 100.

    Returns
    -------
    np.ndarray
        Temperature array in Kelvin.

    Examples
    --------
    >>> temps = get_default_temperatures()
    >>> temps.shape
    (100,)
    >>> np.log10(temps[0])  # doctest: +ELLIPSIS
    5.5...
    """
    logt = np.linspace(logt_min, logt_max, n_bins)
    return 10.0 ** logt


def get_temperature_response(
    wavelengths: Optional[List[int]] = None,
    temperatures: Optional[np.ndarray] = None,
    time: Optional[datetime] = None,
    include_degradation: bool = True,
) -> np.ndarray:
    """
    Get AIA temperature response functions.

    Computes the temperature response K(T) for each AIA channel,
    which relates the observed intensity to the DEM:
        I = integral(K(T) * DEM(T) * dT)

    Parameters
    ----------
    wavelengths : list of int, optional
        AIA wavelengths in Angstroms. Default: [94, 131, 171, 193, 211, 335].
    temperatures : np.ndarray, optional
        Temperature array in Kelvin. Default: logarithmic grid from 10^5.5 to 10^7.5 K.
    time : datetime, optional
        Observation time for degradation correction. Default: None (no degradation).
    include_degradation : bool, optional
        Whether to include time-dependent degradation. Default: True.

    Returns
    -------
    np.ndarray
        Temperature response matrix, shape (n_temperatures, n_wavelengths).
        Units: DN cm^5 s^-1 pixel^-1.

    Raises
    ------
    ImportError
        If aiapy is not installed and no fallback data is available.

    Notes
    -----
    This function wraps aiapy.response for convenience. If aiapy is not
    installed, it falls back to pre-computed response tables (less accurate
    for time-dependent degradation).

    References
    ----------
    - Boerner et al. (2012), Solar Physics 275, 41-66
    - Boerner et al. (2014), Solar Physics 289, 2377-2397

    Examples
    --------
    >>> temps = get_default_temperatures()
    >>> response = get_temperature_response(temperatures=temps)
    >>> response.shape
    (100, 6)
    """
    if wavelengths is None:
        wavelengths = AIA_DEM_WAVELENGTHS

    if temperatures is None:
        temperatures = get_default_temperatures()

    if not HAS_AIAPY:
        return _get_fallback_response(wavelengths, temperatures)

    return _get_aiapy_response(
        wavelengths, temperatures, time, include_degradation
    )


def _get_aiapy_response(
    wavelengths: List[int],
    temperatures: np.ndarray,
    time: Optional[datetime],
    include_degradation: bool,
) -> np.ndarray:
    """
    Get temperature response using aiapy.

    Parameters
    ----------
    wavelengths : list of int
        AIA wavelengths.
    temperatures : np.ndarray
        Temperature array in Kelvin.
    time : datetime, optional
        Observation time.
    include_degradation : bool
        Whether to include degradation correction.

    Returns
    -------
    np.ndarray
        Response matrix (n_temps, n_wavelengths).
    """
    import astropy.units as u
    from sunpy.time import parse_time

    n_temps = len(temperatures)
    n_waves = len(wavelengths)
    response = np.zeros((n_temps, n_waves), dtype=np.float64)

    # Convert temperature to astropy units
    temps_k = temperatures * u.K

    for i, wave in enumerate(wavelengths):
        # Create channel response object
        channel = aiapy.response.Channel(wave * u.angstrom)

        # Get temperature response
        tresp = channel.temperature_response(temps_k)

        # Apply degradation correction if requested
        if include_degradation and time is not None:
            obs_time = parse_time(time)
            deg = degradation(channel.channel, obs_time)
            tresp = tresp * deg

        # Store (convert from astropy units)
        response[:, i] = tresp.to(u.cm**5 * u.ct / u.s / u.pix).value

    return response


def _get_fallback_response(
    wavelengths: List[int],
    temperatures: np.ndarray,
) -> np.ndarray:
    """
    Get approximate temperature response without aiapy.

    This provides rough estimates based on characteristic temperatures
    of each AIA channel. For accurate DEM analysis, install aiapy.

    Parameters
    ----------
    wavelengths : list of int
        AIA wavelengths.
    temperatures : np.ndarray
        Temperature array in Kelvin.

    Returns
    -------
    np.ndarray
        Approximate response matrix (n_temps, n_wavelengths).

    Warnings
    --------
    This is a simplified approximation. For research-quality DEM analysis,
    please install aiapy for accurate temperature response functions.
    """
    import warnings

    warnings.warn(
        "Using approximate temperature response. "
        "Install aiapy for accurate DEM analysis: pip install aiapy",
        UserWarning,
    )

    # Characteristic peak temperatures (log10 K) and widths for each channel
    # Based on Lemen et al. (2012)
    channel_params = {
        94: {"logt_peak": 6.8, "width": 0.3, "amplitude": 1e-26},  # Fe XVIII
        131: {"logt_peak": 5.6, "width": 0.4, "amplitude": 5e-27},  # Fe VIII, XXI
        171: {"logt_peak": 5.9, "width": 0.3, "amplitude": 1e-25},  # Fe IX
        193: {"logt_peak": 6.2, "width": 0.4, "amplitude": 5e-26},  # Fe XII, XXIV
        211: {"logt_peak": 6.3, "width": 0.3, "amplitude": 3e-26},  # Fe XIV
        335: {"logt_peak": 6.4, "width": 0.4, "amplitude": 5e-27},  # Fe XVI
    }

    n_temps = len(temperatures)
    n_waves = len(wavelengths)
    response = np.zeros((n_temps, n_waves), dtype=np.float64)

    logt = np.log10(temperatures)

    for i, wave in enumerate(wavelengths):
        if wave in channel_params:
            params = channel_params[wave]
            # Simple Gaussian approximation
            response[:, i] = params["amplitude"] * np.exp(
                -0.5 * ((logt - params["logt_peak"]) / params["width"]) ** 2
            )
        else:
            raise ValueError(f"Unknown wavelength: {wave} Angstrom")

    return response


def compute_response_derivative(
    response: np.ndarray,
    temperatures: np.ndarray,
) -> np.ndarray:
    """
    Compute temperature derivative of response functions.

    Used in SITES algorithm for DEM inversion.

    Parameters
    ----------
    response : np.ndarray
        Temperature response matrix (n_temps, n_wavelengths).
    temperatures : np.ndarray
        Temperature array in Kelvin.

    Returns
    -------
    np.ndarray
        Response derivative dK/dT, same shape as response.
    """
    # Use log temperature for numerical stability
    logt = np.log10(temperatures)
    dlogt = np.gradient(logt)

    # Compute gradient along temperature axis
    dresponse = np.gradient(response, axis=0) / dlogt[:, np.newaxis]

    return dresponse


def get_response_weights(
    response: np.ndarray,
    method: str = "sum",
) -> np.ndarray:
    """
    Compute weights from temperature response for DEM initialization.

    Parameters
    ----------
    response : np.ndarray
        Temperature response matrix (n_temps, n_wavelengths).
    method : str, optional
        Weighting method: "sum", "max", or "mean". Default: "sum".

    Returns
    -------
    np.ndarray
        Weight array of shape (n_temps,).
    """
    if method == "sum":
        return np.sum(response, axis=1)
    elif method == "max":
        return np.max(response, axis=1)
    elif method == "mean":
        return np.mean(response, axis=1)
    else:
        raise ValueError(f"Unknown method: {method}")
