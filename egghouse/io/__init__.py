"""
File I/O utilities for various scientific data formats.

Supported formats:
    - FITS: Flexible Image Transport System (astronomy standard).
      Read/write image data and headers via astropy.
      See egghouse.io.fits for format details and usage.
    - BMP: Windows Bitmap format.
      Read/write 8-bit grayscale and 24-bit RGB images.
      No external dependencies (numpy + struct only).
      See egghouse.io.bmp for format details and usage.

Future formats (planned):
    - NetCDF, HDF5, etc.

Example:
    >>> from egghouse.io import read_fits, write_fits
    >>> data, header = read_fits('image.fits')
    >>> write_fits('output.fits', data, header, overwrite=True)
    >>>
    >>> from egghouse.io import read_bmp, write_bmp
    >>> data, info = read_bmp('image.bmp')
    >>> write_bmp('output.bmp', data)
"""

from .fits import (
    read_fits,
    write_fits,
    read_fits_header,
    append_fits,
    HAS_ASTROPY,
)
from .bmp import (
    read_bmp,
    write_bmp,
    read_bmp_header,
)

__all__ = [
    # FITS
    'read_fits',
    'write_fits',
    'read_fits_header',
    'append_fits',
    'HAS_ASTROPY',
    # BMP
    'read_bmp',
    'write_bmp',
    'read_bmp_header',
]
