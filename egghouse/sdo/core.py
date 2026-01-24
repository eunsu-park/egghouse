"""
Core utilities for SDO data processing.

Provides common functions used across AIA and HMI modules, including
FITS header parsing and image validation.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np


# Check for astropy availability
try:
    from astropy.io import fits
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False


def parse_fits_header(filepath: str) -> Dict[str, Any]:
    """
    Extract commonly used header keywords from SDO FITS files.

    Extracts observation metadata including time, wavelength, exposure,
    and coordinate system information.

    Args:
        filepath: Path to FITS file.

    Returns:
        Dictionary containing extracted header values:
            - date_obs: Observation datetime string
            - wavelnth: Wavelength in Angstroms (AIA only)
            - exptime: Exposure time in seconds
            - crpix1, crpix2: Reference pixel coordinates
            - cdelt1, cdelt2: Plate scale in arcsec/pixel
            - crota2: Rotation angle in degrees
            - rsun_obs: Solar radius in arcsec
            - dsun_obs: Sun-observer distance in meters
            - instrume: Instrument name (AIA or HMI)

    Raises:
        ImportError: If astropy is not installed.

    Example:
        >>> header = parse_fits_header('aia_171.fits')
        >>> print(f"Wavelength: {header['wavelnth']} A")
        >>> print(f"Exposure: {header['exptime']} s")
    """
    if not HAS_ASTROPY:
        raise ImportError(
            "astropy is required for FITS operations. "
            "Install it with: pip install astropy"
        )

    with fits.open(filepath) as hdul:
        header = hdul[0].header
        return {
            'date_obs': header.get('DATE-OBS'),
            'wavelnth': header.get('WAVELNTH'),
            'exptime': header.get('EXPTIME'),
            'crpix1': header.get('CRPIX1'),
            'crpix2': header.get('CRPIX2'),
            'cdelt1': header.get('CDELT1'),
            'cdelt2': header.get('CDELT2'),
            'crota2': header.get('CROTA2'),
            'rsun_obs': header.get('RSUN_OBS'),
            'dsun_obs': header.get('DSUN_OBS'),
            'instrume': header.get('INSTRUME'),
        }


def validate_sdo_image(
    image: np.ndarray,
    expected_shape: Optional[Tuple[int, int]] = (4096, 4096)
) -> bool:
    """
    Validate SDO image dimensions and basic data quality.

    Checks that the image is 2D and optionally matches the expected shape.
    Standard SDO images are 4096x4096 pixels.

    Args:
        image: Input image array.
        expected_shape: Expected (height, width). Pass None to skip shape check.

    Returns:
        True if valid.

    Raises:
        ValueError: If image dimensions or shape are invalid.

    Example:
        >>> from astropy.io import fits
        >>> data = fits.getdata('aia_171.fits')
        >>> validate_sdo_image(data)  # Returns True for valid 4096x4096 image
        True
        >>> validate_sdo_image(data, expected_shape=None)  # Skip shape check
        True
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got {image.ndim}D")

    if expected_shape is not None and image.shape != expected_shape:
        raise ValueError(
            f"Expected shape {expected_shape}, got {image.shape}"
        )

    return True


def get_solar_disk_params(header: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate solar disk parameters from FITS header metadata.

    Extracts and computes useful parameters for masking and coordinate
    transformations.

    Args:
        header: Dictionary from parse_fits_header() or similar.

    Returns:
        Dictionary containing:
            - center_x, center_y: Solar disk center in pixels
            - radius_pixels: Solar radius in pixels
            - plate_scale: Arcsec per pixel

    Example:
        >>> header = parse_fits_header('aia_171.fits')
        >>> params = get_solar_disk_params(header)
        >>> print(f"Solar radius: {params['radius_pixels']:.1f} pixels")
    """
    crpix1 = header.get('crpix1', 2048.5)
    crpix2 = header.get('crpix2', 2048.5)
    cdelt1 = header.get('cdelt1', 0.6)
    rsun_obs = header.get('rsun_obs', 960.0)

    plate_scale = abs(cdelt1)
    radius_pixels = rsun_obs / plate_scale

    return {
        'center_x': crpix1 - 1,  # Convert to 0-indexed
        'center_y': crpix2 - 1,
        'radius_pixels': radius_pixels,
        'plate_scale': plate_scale,
    }
