"""
File I/O utilities for various scientific data formats.

Supported formats:
    - FITS: Flexible Image Transport System (astronomy standard).
      Two implementations available:
        - read_fits(), write_fits(): Full-featured via astropy.
        - read_fits_simple(): Pure numpy, no dependencies.
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
    >>> # Without astropy
    >>> from egghouse.io import read_fits_simple
    >>> data, header = read_fits_simple('image.fits')
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
    read_fits_simple,
    read_fits_header_simple,
    HAS_ASTROPY,
)
from .bmp import (
    read_bmp,
    write_bmp,
    read_bmp_header,
)

__all__ = [
    # FITS (astropy)
    'read_fits',
    'write_fits',
    'read_fits_header',
    'append_fits',
    'HAS_ASTROPY',
    # FITS (pure numpy)
    'read_fits_simple',
    'read_fits_header_simple',
    # BMP
    'read_bmp',
    'write_bmp',
    'read_bmp_header',
]
