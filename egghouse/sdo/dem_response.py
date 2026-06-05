"""
SDO/AIA temperature response — instrument-specific wiring for DEM.

The DEM inversion machinery and the generic CHIANTI fold live in
``egghouse.dem``; this module supplies the **AIA** wavelength response
(via aiapy) and assembles the AIA temperature response ``K(T)`` by calling
``egghouse.dem.temperature_response_from_chianti``.

References:
    - Boerner et al. (2012), Solar Physics 275, 41-66
    - Boerner et al. (2014), Solar Physics 289, 2377-2397
"""

import os
from datetime import datetime
from typing import List, Optional, Union

import numpy as np

from egghouse.dem.response import (
    DEFAULT_DEM_ABUNDANCE,
    DEFAULT_DEM_DENSITY_CM3,
    DEFAULT_RESPONSE_BAND,
    HAS_FIASCO,
    get_default_temperatures,
    load_ssw_temperature_response,
    temperature_response_from_chianti,
)

# Check for aiapy availability (supplies the AIA wavelength response).
try:
    import aiapy.response  # noqa: F401
    HAS_AIAPY = True
except ImportError:
    HAS_AIAPY = False

# Standard AIA EUV wavelengths for DEM analysis.
AIA_DEM_WAVELENGTHS = [94, 131, 171, 193, 211, 335]


def _aia_channel_responses(wavelengths: List[int]):
    """Per-AIA-channel ``(wavelength_grid, R(lambda), plate_scale)`` from aiapy.

    ``R(lambda)`` is the aiapy ``Channel.wavelength_response`` in
    ``cm^2 DN/photon``; ``plate_scale`` is in steradian/pixel.
    """
    import astropy.units as u
    from aiapy.response import Channel

    out = []
    for w in wavelengths:
        ch = Channel(int(w) * u.angstrom)
        out.append((
            ch.wavelength.to_value("angstrom"),
            ch.wavelength_response().to_value("cm2 DN / ph"),
            ch.plate_scale.to_value("sr / pix"),
        ))
    return out


def get_temperature_response(
    wavelengths: Optional[List[int]] = None,
    temperatures: Optional[np.ndarray] = None,
    time: Optional[datetime] = None,
    include_degradation: bool = True,
    ssw_table_path: Optional[Union[str, os.PathLike]] = None,
    ssw_response_key: str = "response_v10_en",
    density_cm3: float = DEFAULT_DEM_DENSITY_CM3,
    abundance: str = DEFAULT_DEM_ABUNDANCE,
    band: tuple = DEFAULT_RESPONSE_BAND,
) -> np.ndarray:
    """
    Get the SDO/AIA temperature response K(T), shape (n_temperatures, n_wavelengths).

    Sources, in priority order:

    1. ``ssw_table_path`` — an SSW ``aia_get_response.pro`` ``.npz`` table
       (canonical; via :func:`egghouse.dem.load_ssw_temperature_response`).
    2. ``fiasco`` + ``aiapy`` — recompute from CHIANTI contribution functions
       folded with the aiapy AIA wavelength response (the live replacement
       for aiapy's removed ``Channel.temperature_response``; slow, one-time).
    3. ``aiapy`` alone — disabled (aiapy 0.12 removed
       ``Channel.temperature_response``); raises with a pointer.
    4. Built-in Gaussian fallback — only when aiapy is unavailable; emits a
       ``UserWarning`` and is **not** research-grade.

    Units: DN cm^5 s^-1 pixel^-1. Time-dependent degradation is not folded in
    here; apply it at the image level (``aiapy.calibrate.correct_degradation``).
    """
    if wavelengths is None:
        wavelengths = AIA_DEM_WAVELENGTHS
    if temperatures is None:
        temperatures = get_default_temperatures()

    if ssw_table_path is not None:
        log_temperatures = np.log10(np.asarray(temperatures, dtype=np.float64))
        return load_ssw_temperature_response(
            ssw_table_path,
            log_temperatures=log_temperatures,
            wavelengths=wavelengths,
            response_key=ssw_response_key,
        )

    if HAS_FIASCO and HAS_AIAPY:
        channel_responses = _aia_channel_responses(wavelengths)
        return temperature_response_from_chianti(
            channel_responses, temperatures,
            density_cm3=density_cm3, abundance=abundance, band=band,
        )

    if not HAS_AIAPY:
        return _get_fallback_response(wavelengths, temperatures)

    return _get_aiapy_response(wavelengths, temperatures, time, include_degradation)


def _get_aiapy_response(wavelengths, temperatures, time, include_degradation):
    """Disabled: aiapy 0.12 removed ``Channel.temperature_response``."""
    raise NotImplementedError(
        "egghouse.sdo.get_temperature_response's aiapy-only path is disabled "
        "for aiapy >= 0.12 (Channel.temperature_response was removed). Install "
        "fiasco for the CHIANTI path, or supply ssw_table_path."
    )


def _get_fallback_response(
    wavelengths: List[int],
    temperatures: np.ndarray,
) -> np.ndarray:
    """Approximate Gaussian AIA response (no aiapy). Not research-grade."""
    import warnings

    warnings.warn(
        "Using approximate temperature response. "
        "Install aiapy + fiasco for accurate DEM analysis.",
        UserWarning,
    )
    # Characteristic peak temperatures (log10 K) and widths (Lemen et al. 2012).
    channel_params = {
        94: {"logt_peak": 6.8, "width": 0.3, "amplitude": 1e-26},
        131: {"logt_peak": 5.6, "width": 0.4, "amplitude": 5e-27},
        171: {"logt_peak": 5.9, "width": 0.3, "amplitude": 1e-25},
        193: {"logt_peak": 6.2, "width": 0.4, "amplitude": 5e-26},
        211: {"logt_peak": 6.3, "width": 0.3, "amplitude": 3e-26},
        335: {"logt_peak": 6.4, "width": 0.4, "amplitude": 5e-27},
    }
    n_temps = len(temperatures)
    response = np.zeros((n_temps, len(wavelengths)), dtype=np.float64)
    logt = np.log10(temperatures)
    for i, wave in enumerate(wavelengths):
        if wave not in channel_params:
            raise ValueError(f"Unknown wavelength: {wave} Angstrom")
        p = channel_params[wave]
        response[:, i] = p["amplitude"] * np.exp(
            -0.5 * ((logt - p["logt_peak"]) / p["width"]) ** 2
        )
    return response
