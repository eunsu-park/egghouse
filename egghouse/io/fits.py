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


# =============================================================================
# Pure NumPy FITS reading (no astropy dependency)
# =============================================================================

# BITPIX to numpy dtype mapping (big-endian)
_BITPIX_DTYPE = {
    8: '>u1',     # unsigned 8-bit integer
    16: '>i2',    # signed 16-bit integer
    32: '>i4',    # signed 32-bit integer
    64: '>i8',    # signed 64-bit integer
    -32: '>f4',   # 32-bit floating point
    -64: '>f8',   # 64-bit floating point
}

_FITS_BLOCK_SIZE = 2880  # FITS standard block size


def _parse_header_card(card: bytes) -> Tuple[Optional[str], Any]:
    """
    Parse a single 80-byte FITS header card.

    Args:
        card: 80-byte header card.

    Returns:
        Tuple of (keyword, value). Returns (None, None) for blank/comment cards.
    """
    card_str = card.decode('ascii', errors='replace')

    # Check for END card
    if card_str.startswith('END'):
        return 'END', None

    # Check for COMMENT, HISTORY, or blank cards
    keyword = card_str[:8].strip()
    if not keyword or keyword in ('COMMENT', 'HISTORY', ''):
        return None, None

    # Check for value indicator
    if card_str[8:10] != '= ':
        return None, None

    # Parse value (starts at position 10)
    value_str = card_str[10:].split('/')[0].strip()  # Remove comment

    if not value_str:
        return keyword, None

    # Parse value type
    if value_str.startswith("'"):
        # String value
        end_quote = value_str.find("'", 1)
        if end_quote > 0:
            value = value_str[1:end_quote].rstrip()
        else:
            value = value_str[1:].rstrip()
    elif value_str in ('T', 'F'):
        # Boolean
        value = value_str == 'T'
    elif '.' in value_str or 'E' in value_str.upper():
        # Float
        try:
            value = float(value_str.replace('D', 'E'))  # FITS uses D for exponent
        except ValueError:
            value = value_str
    else:
        # Integer
        try:
            value = int(value_str)
        except ValueError:
            value = value_str

    return keyword, value


def _parse_fits_header(f) -> Tuple[Dict[str, Any], int]:
    """
    Parse FITS header from current file position.

    Args:
        f: File object positioned at start of header.

    Returns:
        Tuple of (header_dict, data_size_bytes).
    """
    header = {}
    header_complete = False

    while not header_complete:
        block = f.read(_FITS_BLOCK_SIZE)
        if len(block) < _FITS_BLOCK_SIZE:
            raise ValueError("Unexpected end of file while reading header")

        # Parse 36 cards per block (36 * 80 = 2880)
        for i in range(36):
            card = block[i * 80:(i + 1) * 80]
            keyword, value = _parse_header_card(card)

            if keyword == 'END':
                header_complete = True
                break
            elif keyword is not None:
                header[keyword] = value

    # Calculate data size
    naxis = header.get('NAXIS', 0)
    if naxis == 0:
        data_size = 0
    else:
        bitpix = header.get('BITPIX', 8)
        data_size = abs(bitpix) // 8
        for i in range(1, naxis + 1):
            data_size *= header.get(f'NAXIS{i}', 0)

    # Add PCOUNT and GCOUNT for extensions
    pcount = header.get('PCOUNT', 0)
    gcount = header.get('GCOUNT', 1)
    data_size = (data_size + pcount) * gcount

    return header, data_size


def _skip_hdu_data(f, data_size: int) -> None:
    """Skip over HDU data block (padded to 2880 bytes)."""
    if data_size > 0:
        # Calculate padded size
        padded_size = ((data_size + _FITS_BLOCK_SIZE - 1)
                       // _FITS_BLOCK_SIZE) * _FITS_BLOCK_SIZE
        f.seek(padded_size, 1)  # Seek relative to current position


def read_fits_simple(
    filepath: str,
    hdu_index: int = 0,
    apply_scaling: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read FITS image HDU without astropy dependency.

    Supports Primary HDU and Extension ImageHDUs. Uses only numpy and
    standard library. For binary tables, compressed FITS, or advanced
    features, use read_fits() which requires astropy.

    Args:
        filepath: Path to FITS file.
        hdu_index: HDU index to read (0=Primary, 1+=Extensions).
        apply_scaling: If True, apply BSCALE/BZERO scaling to get
                      physical values. Defaults to True.

    Returns:
        Tuple of (data, header) where data is a numpy array and
        header is a dictionary of keyword-value pairs.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If HDU is not an image (e.g., binary table).
        ValueError: If hdu_index is out of range.

    Example:
        >>> data, header = read_fits_simple('image.fits')
        >>> print(data.shape, data.dtype)
        (4096, 4096) float32

        >>> # Read extension HDU
        >>> ext_data, ext_header = read_fits_simple('multi.fits', hdu_index=1)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"FITS file not found: {filepath}")

    with open(filepath, 'rb') as f:
        # Navigate to requested HDU
        current_hdu = 0
        header = None

        while current_hdu <= hdu_index:
            header, data_size = _parse_fits_header(f)

            if current_hdu < hdu_index:
                _skip_hdu_data(f, data_size)
                current_hdu += 1

                # Check if we've reached end of file
                next_byte = f.read(1)
                if not next_byte:
                    raise ValueError(
                        f"HDU index {hdu_index} out of range "
                        f"(file has {current_hdu} HDUs)"
                    )
                f.seek(-1, 1)  # Go back one byte
            else:
                break

        # Validate that this is an image HDU
        if hdu_index > 0:
            xtension = header.get('XTENSION', '').strip()
            if xtension and xtension != 'IMAGE':
                raise ValueError(
                    f"HDU {hdu_index} is not an image (XTENSION='{xtension}'). "
                    "Use read_fits() with astropy for binary tables."
                )

        # Read data
        naxis = header.get('NAXIS', 0)
        if naxis == 0 or data_size == 0:
            return None, header

        bitpix = header.get('BITPIX')
        if bitpix not in _BITPIX_DTYPE:
            raise ValueError(f"Unsupported BITPIX value: {bitpix}")

        dtype = np.dtype(_BITPIX_DTYPE[bitpix])

        # Get shape (FITS uses column-major, numpy uses row-major)
        shape = []
        for i in range(naxis, 0, -1):
            shape.append(header.get(f'NAXIS{i}', 0))
        shape = tuple(shape)

        # Read raw data
        raw_data = f.read(data_size)
        if len(raw_data) < data_size:
            raise ValueError("Unexpected end of file while reading data")

        data = np.frombuffer(raw_data, dtype=dtype).reshape(shape)

        # Make a copy to ensure array is writable
        data = data.copy()

        # Apply BSCALE/BZERO scaling
        if apply_scaling:
            bscale = header.get('BSCALE', 1.0)
            bzero = header.get('BZERO', 0.0)

            if bscale != 1.0 or bzero != 0.0:
                # Convert to float for scaling
                data = bzero + bscale * data.astype(np.float64)

    return data, header


def read_fits_header_simple(
    filepath: str,
    hdu_index: int = 0
) -> Dict[str, Any]:
    """
    Read FITS header without astropy dependency.

    More memory-efficient than read_fits_simple() when only metadata is needed.

    Args:
        filepath: Path to FITS file.
        hdu_index: HDU index to read (0=Primary, 1+=Extensions).

    Returns:
        Header as a dictionary of keyword-value pairs.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If hdu_index is out of range.

    Example:
        >>> header = read_fits_header_simple('image.fits')
        >>> print(header.get('DATE-OBS'))
        2024-01-01T00:00:00.00
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"FITS file not found: {filepath}")

    with open(filepath, 'rb') as f:
        current_hdu = 0

        while current_hdu <= hdu_index:
            header, data_size = _parse_fits_header(f)

            if current_hdu < hdu_index:
                _skip_hdu_data(f, data_size)
                current_hdu += 1

                # Check if we've reached end of file
                next_byte = f.read(1)
                if not next_byte:
                    raise ValueError(
                        f"HDU index {hdu_index} out of range "
                        f"(file has {current_hdu} HDUs)"
                    )
                f.seek(-1, 1)
            else:
                break

    return header
