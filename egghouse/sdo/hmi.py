"""
HMI (Helioseismic and Magnetic Imager) utilities for SDO data.

This module provides intensity scaling functions for SDO/HMI magnetogram data.
"""

import numpy as np


def hmi_intscale(
    data: np.ndarray,
    vmin: float = -100.0,
    vmax: float = 100.0
) -> np.ndarray:
    """
    Scale HMI magnetogram data to uint8 for visualization.

    Maps the magnetic field strength from [vmin, vmax] Gauss to [0, 255].
    Default range of [-100, 100] Gauss is suitable for most quiet sun
    and weak active region visualization.

    Args:
        data: HMI magnetogram data in Gauss.
        vmin: Minimum magnetic field value in Gauss. Defaults to -100.
        vmax: Maximum magnetic field value in Gauss. Defaults to 100.

    Returns:
        Scaled uint8 image array.

    Example:
        >>> from astropy.io import fits
        >>> hdu = fits.open('hmi_m.fits')[0]
        >>> scaled = hmi_intscale(hdu.data)
        >>> # For active regions with stronger fields:
        >>> scaled_ar = hmi_intscale(hdu.data, vmin=-500, vmax=500)
    """
    data = np.asarray(data, dtype=np.float64)

    # Handle NaN values
    data = np.nan_to_num(data, nan=0.0)

    # Linear scaling from [vmin, vmax] to [0, 255]
    scaled = (data - vmin) * (255.0 / (vmax - vmin))

    return np.clip(scaled, 0, 255).astype(np.uint8)


def hmi_field_strength(
    bx: np.ndarray,
    by: np.ndarray,
    bz: np.ndarray
) -> np.ndarray:
    """
    Calculate total magnetic field strength from vector components.

    Computes |B| = sqrt(Bx^2 + By^2 + Bz^2) for vector magnetogram data.

    Args:
        bx: Magnetic field x-component in Gauss.
        by: Magnetic field y-component in Gauss.
        bz: Magnetic field z-component in Gauss.

    Returns:
        Total magnetic field strength array in Gauss.

    Example:
        >>> from astropy.io import fits
        >>> bx = fits.getdata('hmi_bx.fits')
        >>> by = fits.getdata('hmi_by.fits')
        >>> bz = fits.getdata('hmi_bz.fits')
        >>> b_total = hmi_field_strength(bx, by, bz)
    """
    bx = np.asarray(bx, dtype=np.float64)
    by = np.asarray(by, dtype=np.float64)
    bz = np.asarray(bz, dtype=np.float64)

    return np.sqrt(bx**2 + by**2 + bz**2)
