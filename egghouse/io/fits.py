"""
FITS (Flexible Image Transport System) file I/O utilities.

FITS Format Overview:
    FITS is the standard data format in astronomy, defined by the IAU FITS
    Working Group. A FITS file consists of one or more Header Data Units (HDUs):

    - Primary HDU (index 0): Contains the main header and optional image data.
      The header stores metadata as 80-character keyword=value cards.
      Each card has the format: KEYWORD = value / comment
    - Extension HDUs (index 1+): Additional data units, which can be:
        - ImageHDU: Additional image arrays
        - BinTableHDU: Binary table data
        - TableHDU: ASCII table data

    Common header keywords:
        SIMPLE   - Standard FITS file (T/F)
        BITPIX   - Bits per pixel (-64=float64, -32=float32, 16=int16, etc.)
        NAXIS    - Number of data axes
        NAXIS1/2 - Size of each axis (NAXIS1=columns, NAXIS2=rows)
        BSCALE   - Linear scaling factor (physical = BZERO + BSCALE * array)
        BZERO    - Zero offset for scaling
        BLANK    - Value representing undefined pixels (integer data only)

    Data is stored in big-endian byte order. Header blocks are padded to
    multiples of 2880 bytes. Astropy handles byte-swapping and scaling
    automatically when reading.

Example:
    >>> from egghouse.io import read_fits, write_fits
    >>> data, header = read_fits('image.fits')
    >>> print(data.shape, data.dtype)
    >>> write_fits('output.fits', data, header)
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np

# Optional dependency
try:
    from astropy.io import fits
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False


def _require_astropy() -> None:
    """Raise ImportError if astropy is not available."""
    if not HAS_ASTROPY:
        raise ImportError(
            "astropy is required for FITS I/O. "
            "Install with: pip install astropy"
        )


def read_fits(
    filepath: str,
    hdu_index: int = 0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read data and header from a FITS file.

    Args:
        filepath: Path to FITS file.
        hdu_index: HDU index to read. Defaults to 0 (Primary HDU).

    Returns:
        Tuple of (data, header) where data is a numpy array and
        header is a dictionary of keyword-value pairs.

    Raises:
        ImportError: If astropy is not installed.
        FileNotFoundError: If file does not exist.
        IndexError: If hdu_index is out of range.

    Example:
        >>> data, header = read_fits('image.fits')
        >>> print(data.shape)
        (4096, 4096)
        >>> print(header['BITPIX'])
        -32
    """
    _require_astropy()

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"FITS file not found: {filepath}")

    with fits.open(filepath) as hdul:
        data = hdul[hdu_index].data
        header = dict(hdul[hdu_index].header)

    return data, header


def write_fits(
    filepath: str,
    data: np.ndarray,
    header: Optional[Dict[str, Any]] = None,
    overwrite: bool = False
) -> None:
    """
    Write data and optional header to a FITS file.

    Creates a Primary HDU with the given data. If a header dictionary is
    provided, its key-value pairs are added to the FITS header.

    Args:
        filepath: Output file path.
        data: Image data array (2D or 3D).
        header: Optional dictionary of header keyword-value pairs.
        overwrite: If True, overwrite existing file. Defaults to False.

    Raises:
        ImportError: If astropy is not installed.
        FileExistsError: If file exists and overwrite is False.

    Example:
        >>> import numpy as np
        >>> data = np.zeros((512, 512), dtype=np.float32)
        >>> write_fits('output.fits', data, header={'OBJECT': 'Sun'})
    """
    _require_astropy()

    if not overwrite and os.path.exists(filepath):
        raise FileExistsError(f"File already exists: {filepath}")

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    hdr = fits.Header()
    if header is not None:
        for key, value in header.items():
            if key in ('SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2',
                       'EXTEND', ''):
                continue
            try:
                hdr[key] = value
            except (ValueError, TypeError):
                continue

    hdu = fits.PrimaryHDU(data=data, header=hdr)
    hdu.writeto(filepath, overwrite=overwrite)


def read_fits_header(
    filepath: str,
    hdu_index: int = 0
) -> Dict[str, Any]:
    """
    Read only the header from a FITS file without loading data.

    More memory-efficient than read_fits() when only metadata is needed.

    Args:
        filepath: Path to FITS file.
        hdu_index: HDU index to read. Defaults to 0.

    Returns:
        Header as a dictionary of keyword-value pairs.

    Raises:
        ImportError: If astropy is not installed.
        FileNotFoundError: If file does not exist.

    Example:
        >>> header = read_fits_header('image.fits')
        >>> print(header.get('DATE-OBS'))
        2024-01-01T00:00:00.00
    """
    _require_astropy()

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"FITS file not found: {filepath}")

    header = fits.getheader(filepath, ext=hdu_index)
    return dict(header)


def append_fits(
    filepath: str,
    data: np.ndarray,
    header: Optional[Dict[str, Any]] = None
) -> None:
    """
    Append a new Image HDU extension to an existing FITS file.

    The new data is added as an ImageHDU extension after existing HDUs.

    Args:
        filepath: Path to existing FITS file.
        data: Data array for the new extension.
        header: Optional dictionary of header keyword-value pairs.

    Raises:
        ImportError: If astropy is not installed.
        FileNotFoundError: If file does not exist.

    Example:
        >>> write_fits('multi.fits', primary_data)
        >>> append_fits('multi.fits', extension_data, header={'EXTNAME': 'EXT1'})
    """
    _require_astropy()

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"FITS file not found: {filepath}")

    hdr = fits.Header()
    if header is not None:
        for key, value in header.items():
            if key in ('SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2',
                       'EXTEND', 'XTENSION', 'PCOUNT', 'GCOUNT', ''):
                continue
            try:
                hdr[key] = value
            except (ValueError, TypeError):
                continue

    with fits.open(filepath, mode='append') as hdul:
        hdul.append(fits.ImageHDU(data=data, header=hdr))
