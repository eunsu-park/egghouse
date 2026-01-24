"""
AIA (Atmospheric Imaging Assembly) utilities for SDO data.

This module provides intensity scaling and normalization functions
for SDO/AIA images across different wavelength channels.
"""

from typing import Dict, Any
import numpy as np
from ..image import bytescale_image


# AIA wavelength calibration factors
# Reference: Boerner et al. 2012, Solar Physics
AIA_CALIBRATION: Dict[int, Dict[str, Any]] = {
    94:   {'norm_exptime': 4.99803, 'vmin': 1.5/1.06,   'vmax': 50/1.06,    'scale': 'sqrt'},
    131:  {'norm_exptime': 6.99685, 'vmin': 7.0/1.49,   'vmax': 1200/1.49,  'scale': 'log'},
    171:  {'norm_exptime': 4.99803, 'vmin': 10.0/1.49,  'vmax': 6000/1.49,  'scale': 'sqrt'},
    193:  {'norm_exptime': 2.9995,  'vmin': 120.0/2.2,  'vmax': 6000.0/2.2, 'scale': 'log'},
    211:  {'norm_exptime': 4.99801, 'vmin': 30.0/1.10,  'vmax': 13000/1.10, 'scale': 'log'},
    304:  {'norm_exptime': 4.99941, 'vmin': 50.0/12.11, 'vmax': 2000/12.11, 'scale': 'log'},
    335:  {'norm_exptime': 6.99734, 'vmin': 3.5/2.97,   'vmax': 1000/2.97,  'scale': 'log'},
    1600: {'norm_exptime': 2.99911, 'vmin': -8,         'vmax': 200,        'scale': 'linear'},
    1700: {'norm_exptime': 1.00026, 'vmin': 0,          'vmax': 2500,       'scale': 'linear'},
    4500: {'norm_exptime': 1.00026, 'vmin': 0,          'vmax': 26000,      'scale': 'linear'},
}


def aia_intscale(
    image: np.ndarray,
    exptime: float,
    wavelnth: int,
    to_bytescale: bool = True
) -> np.ndarray:
    """
    Apply AIA intensity scaling for visualization.

    Normalizes the image by exposure time and applies wavelength-specific
    scaling (linear, sqrt, or log) for optimal visualization.

    Args:
        image: Input AIA image array.
        exptime: Exposure time in seconds (from FITS header EXPTIME).
        wavelnth: Wavelength in Angstroms (94, 131, 171, 193, 211, 304, 335, 1600, 1700, 4500).
        to_bytescale: If True, return uint8 [0-255]. If False, return scaled float.

    Returns:
        Scaled image array (uint8 if to_bytescale=True, float64 otherwise).

    Raises:
        ValueError: If wavelength is not supported.

    Example:
        >>> from astropy.io import fits
        >>> hdu = fits.open('aia_171.fits')[0]
        >>> scaled = aia_intscale(hdu.data, hdu.header['EXPTIME'], 171)
    """
    wavelnth = int(np.rint(wavelnth))

    if wavelnth not in AIA_CALIBRATION:
        raise ValueError(
            f"Unsupported wavelength: {wavelnth}. "
            f"Supported wavelengths: {list(AIA_CALIBRATION.keys())}"
        )

    cal = AIA_CALIBRATION[wavelnth]
    vmin = cal['vmin']
    vmax = cal['vmax']
    norm_exptime = cal['norm_exptime']
    scale_method = cal['scale']

    # Handle NaN values
    image = np.asarray(image, dtype=np.float64)
    image = np.nan_to_num(image, nan=0.0)

    # Normalize by exposure time
    normalized = image * (norm_exptime / exptime)

    # Clip to valid range
    clipped = np.clip(normalized, vmin, vmax)

    # Apply scaling transformation
    if scale_method == 'sqrt':
        transformed = np.sqrt(clipped)
        t_vmin, t_vmax = np.sqrt(vmin), np.sqrt(vmax)
    elif scale_method == 'log':
        # Ensure positive values for log
        clipped = np.clip(clipped, max(vmin, 1e-10), vmax)
        transformed = np.log10(clipped)
        t_vmin, t_vmax = np.log10(max(vmin, 1e-10)), np.log10(vmax)
    else:  # linear
        transformed = clipped
        t_vmin, t_vmax = vmin, vmax

    if to_bytescale:
        return bytescale_image(transformed, t_vmin, t_vmax)
    else:
        return transformed


def get_aia_calibration(wavelnth: int) -> Dict[str, Any]:
    """
    Get calibration parameters for a specific AIA wavelength.

    Args:
        wavelnth: Wavelength in Angstroms.

    Returns:
        Dictionary with calibration parameters (norm_exptime, vmin, vmax, scale).

    Raises:
        ValueError: If wavelength is not supported.

    Example:
        >>> cal = get_aia_calibration(171)
        >>> print(f"Scale method: {cal['scale']}")
    """
    wavelnth = int(np.rint(wavelnth))

    if wavelnth not in AIA_CALIBRATION:
        raise ValueError(
            f"Unsupported wavelength: {wavelnth}. "
            f"Supported wavelengths: {list(AIA_CALIBRATION.keys())}"
        )

    return AIA_CALIBRATION[wavelnth].copy()
