"""
Domain-standard SDO/AIA color tables and colorization.

Reproduces the *official* AIA color tables (SolarSoft ``aia_lct.pro`` by Karel
Schrijver, 2010, as adopted by sunpy) for the ten AIA channels:
94, 131, 171, 193, 211, 304, 335, 1600, 1700, 4500 Angstrom.

Each color table is exposed as a 256-entry RGB lookup table (LUT), suitable for
:func:`egghouse.image.apply_colormap`. Two independent, interchangeable sources
are provided and always agree bit-for-bit:

- ``source="numpy"`` (default): a **pure-NumPy** reproduction that needs no
  optional dependency. It rebuilds each channel's ``(R, G, B)`` curves from the
  same analytic functions and the tabulated IDL "Red Temperature" base
  (color table 3) that the reference implementation uses. Verified
  bit-identical to sunpy for all ten channels.
- ``source="sunpy"``: derives the LUT from sunpy's ``aia_color_table`` (the
  reference implementation; requires the optional ``sunpy`` dependency).

The ``source`` argument only selects which dependency is used, not the result.

Example:
    >>> from egghouse.sdo import aia_colorize, aia_color_lut
    >>> from astropy.io import fits                       # doctest: +SKIP
    >>> hdu = fits.open('aia_171.fits')[0]                # doctest: +SKIP
    >>> rgb = aia_colorize(hdu.data, 171,                 # doctest: +SKIP
    ...                    exptime=hdu.header['EXPTIME'])  # -> (H, W, 3) uint8
    >>> lut = aia_color_lut(171)                          # (256, 3) uint8
"""

from typing import Dict, Optional, Tuple

import numpy as np

from .aia import aia_intscale

# Wavelengths (Angstrom) that have an official AIA color table.
AIA_COLOR_WAVELENGTHS: Tuple[int, ...] = (94, 131, 171, 193, 211, 304, 335, 1600, 1700, 4500)

# Cache of built LUTs, keyed by (wavelength, source).
_LUT_CACHE: Dict[Tuple[int, str], np.ndarray] = {}


# ---------------------------------------------------------------------------
# Pure-NumPy reproduction of the official AIA color tables
# ---------------------------------------------------------------------------
# Base curves and the wavelength -> (R, G, B) mapping below mirror sunpy's
# ``create_aia_wave_dict`` exactly, which in turn mirrors SolarSoft
# ``aia_lct.pro``. ``_IDL3_RED_TEMPERATURE`` is the tabulated IDL color table 3
# ("Red Temperature") that sunpy ships as ``idl_3.csv``; it is embedded here so
# the numpy path carries no data-file or sunpy dependency.


def _base_curves() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                            np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(c0, c1, c2, c3, r0, g0, b0)`` as float32 (r0/g0/b0 float64)."""
    idl3 = np.asarray(_IDL3_RED_TEMPERATURE, dtype=np.float64)  # (256, 3)
    r0, g0, b0 = idl3[:, 0], idl3[:, 1], idl3[:, 2]

    c0 = np.arange(256, dtype=np.float32)
    c1 = (np.sqrt(c0) * np.sqrt(255.0)).astype(np.float32)
    c2 = (np.arange(256) ** 2 / 255.0).astype(np.float32)
    c3 = ((c1 + c2 / 2.0) * 255.0 / (c1.max() + c2.max() / 2.0)).astype(np.float32)
    return c0, c1, c2, c3, r0, g0, b0


def _aia_channels(wavelength: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the ``(R, G, B)`` curves (each 256 values, 0..255) for a channel."""
    c0, c1, c2, c3, r0, g0, b0 = _base_curves()
    wave_dict: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        1600: (c3, c3, c2),
        1700: (c1, c0, c0),
        4500: (c0, c0, b0 / 2.0),
        94:   (c2, c3, c0),
        131:  (g0, r0, r0),
        171:  (r0, c0, b0),
        193:  (c1, c0, c2),
        211:  (c1, c0, c3),
        304:  (r0, g0, b0),
        335:  (c2, c0, c1),
    }
    if wavelength not in wave_dict:
        raise ValueError(
            f"No AIA color table for wavelength {wavelength}. "
            f"Valid values: {list(AIA_COLOR_WAVELENGTHS)}"
        )
    return wave_dict[wavelength]


def _lut_numpy(wavelength: int) -> np.ndarray:
    """Build the (256, 3) uint8 LUT with the pure-NumPy path."""
    r, g, b = _aia_channels(wavelength)
    lut = np.stack([r, g, b], axis=1)
    return np.rint(lut).astype(np.uint8)


def _lut_sunpy(wavelength: int) -> np.ndarray:
    """Build the (256, 3) uint8 LUT from sunpy's reference color table."""
    try:
        import astropy.units as u
        from sunpy.visualization.colormaps.color_tables import aia_color_table
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "source='sunpy' requires sunpy (and astropy). Install with "
            "`pip install sunpy` or use source='numpy' (no dependency)."
        ) from exc

    if wavelength not in AIA_COLOR_WAVELENGTHS:
        raise ValueError(
            f"No AIA color table for wavelength {wavelength}. "
            f"Valid values: {list(AIA_COLOR_WAVELENGTHS)}"
        )
    cmap = aia_color_table(wavelength * u.angstrom)
    samples = cmap(np.linspace(0.0, 1.0, 256))[:, :3]
    return np.rint(samples * 255.0).astype(np.uint8)


def aia_color_lut(wavelength: int, source: str = "numpy") -> np.ndarray:
    """
    Return the official AIA color table for a channel as a 256-entry RGB LUT.

    Args:
        wavelength: AIA channel in Angstrom. One of
            ``94, 131, 171, 193, 211, 304, 335, 1600, 1700, 4500``.
        source: Which implementation builds the LUT (default ``'numpy'``):

            - ``'numpy'``: pure-NumPy reproduction, no optional dependency.
            - ``'sunpy'``: sunpy's reference ``aia_color_table`` (requires
              sunpy).

            Both return the same values.

    Returns:
        LUT of shape (256, 3), dtype uint8. Row ``i`` is the ``(R, G, B)``
        color for 8-bit intensity ``i``. Cached per ``(wavelength, source)``.

    Raises:
        ValueError: If ``wavelength`` has no AIA color table or ``source`` is
            unknown.
    """
    wavelength = int(np.rint(wavelength))
    if source not in ("numpy", "sunpy"):
        raise ValueError(f"source must be 'numpy' or 'sunpy', got {source!r}")

    key = (wavelength, source)
    if key not in _LUT_CACHE:
        _LUT_CACHE[key] = (_lut_numpy if source == "numpy" else _lut_sunpy)(wavelength)
    return _LUT_CACHE[key].copy()


def aia_colormap(wavelength: int) -> "object":
    """
    Return sunpy's official AIA color table as a matplotlib ``Colormap``.

    Convenience accessor for plotting with matplotlib (e.g.
    ``ax.imshow(data, cmap=aia_colormap(171))``). Requires the optional
    ``sunpy`` dependency. For raw pixel-level colorization prefer
    :func:`aia_color_lut` / :func:`aia_colorize`, which have a pure-NumPy path.

    Args:
        wavelength: AIA channel in Angstrom.

    Returns:
        The ``SDO AIA <wavelength>`` matplotlib ``Colormap``.
    """
    try:
        import astropy.units as u
        from sunpy.visualization.colormaps.color_tables import aia_color_table
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "aia_colormap requires sunpy (and astropy). Install with "
            "`pip install sunpy`."
        ) from exc

    wavelength = int(np.rint(wavelength))
    if wavelength not in AIA_COLOR_WAVELENGTHS:
        raise ValueError(
            f"No AIA color table for wavelength {wavelength}. "
            f"Valid values: {list(AIA_COLOR_WAVELENGTHS)}"
        )
    return aia_color_table(wavelength * u.angstrom)


def aia_colorize(
    image: np.ndarray,
    wavelnth: int,
    exptime: Optional[float] = None,
    source: str = "numpy",
) -> np.ndarray:
    """
    Colorize an AIA image with its official per-wavelength color table.

    Two input modes:

    - Raw image + ``exptime``: the image is intensity-scaled to 8 bit via
      :func:`egghouse.sdo.aia_intscale` (exposure normalization + the
      wavelength's linear/sqrt/log stretch), then colorized.
    - Already 8-bit grayscale (``exptime=None``): the image is colorized
      directly. Non-uint8 input is min/max byte-scaled first.

    Args:
        image: 2D AIA image: raw counts (with ``exptime``) or an 8-bit
            grayscale display image (without ``exptime``).
        wavelnth: AIA channel in Angstrom (selects both the intensity scaling
            and the color table).
        exptime: Exposure time in seconds (FITS ``EXPTIME``). If given,
            ``image`` is treated as raw and intensity-scaled first.
        source: Color-table source, see :func:`aia_color_lut`.

    Returns:
        Colorized RGB image of shape ``(*image.shape, 3)``, dtype uint8.

    Example:
        >>> import numpy as np
        >>> raw = np.linspace(0, 4000, 64 * 64).reshape(64, 64)
        >>> rgb = aia_colorize(raw, 171, exptime=2.9)
        >>> rgb.shape, rgb.dtype
        ((64, 64, 3), dtype('uint8'))
    """
    from ..image import bytescale_image
    from ..image.colorize import apply_colormap

    if exptime is not None:
        gray = aia_intscale(image, exptime, wavelnth, to_bytescale=True)
    else:
        gray = np.asarray(image)
        if gray.dtype != np.uint8:
            gray = bytescale_image(gray)

    lut = aia_color_lut(wavelnth, source=source)
    return apply_colormap(gray, lut)


# ---------------------------------------------------------------------------
# IDL "Red Temperature" (color table 3) — tabulated base for the numpy path.
# Values are the exact 256x3 entries sunpy ships as idl_3.csv (0..255).
# ---------------------------------------------------------------------------
_IDL3_RED_TEMPERATURE = (
    (  0,  0,  0), (  1,  0,  0), (  2,  0,  0), (  4,  0,  0), (  5,  0,  0), (  7,  0,  0),
    (  8,  0,  0), ( 10,  0,  0), ( 11,  0,  0), ( 13,  0,  0), ( 14,  0,  0), ( 15,  0,  0),
    ( 17,  0,  0), ( 18,  0,  0), ( 20,  0,  0), ( 21,  0,  0), ( 23,  0,  0), ( 24,  0,  0),
    ( 26,  0,  0), ( 27,  0,  0), ( 28,  0,  0), ( 30,  0,  0), ( 31,  0,  0), ( 33,  0,  0),
    ( 34,  0,  0), ( 36,  0,  0), ( 37,  0,  0), ( 39,  0,  0), ( 40,  0,  0), ( 42,  0,  0),
    ( 43,  0,  0), ( 44,  0,  0), ( 46,  0,  0), ( 47,  0,  0), ( 49,  0,  0), ( 50,  0,  0),
    ( 52,  0,  0), ( 53,  0,  0), ( 55,  0,  0), ( 56,  0,  0), ( 57,  0,  0), ( 59,  0,  0),
    ( 60,  0,  0), ( 62,  0,  0), ( 63,  0,  0), ( 65,  0,  0), ( 66,  0,  0), ( 68,  0,  0),
    ( 69,  0,  0), ( 70,  0,  0), ( 72,  0,  0), ( 73,  0,  0), ( 75,  0,  0), ( 76,  0,  0),
    ( 78,  0,  0), ( 79,  0,  0), ( 81,  0,  0), ( 82,  0,  0), ( 84,  0,  0), ( 85,  0,  0),
    ( 86,  0,  0), ( 88,  0,  0), ( 89,  0,  0), ( 91,  0,  0), ( 92,  0,  0), ( 94,  0,  0),
    ( 95,  0,  0), ( 97,  0,  0), ( 98,  0,  0), ( 99,  0,  0), (101,  0,  0), (102,  0,  0),
    (104,  0,  0), (105,  0,  0), (107,  0,  0), (108,  0,  0), (110,  0,  0), (111,  0,  0),
    (113,  0,  0), (114,  0,  0), (115,  0,  0), (117,  0,  0), (118,  0,  0), (120,  0,  0),
    (121,  0,  0), (123,  0,  0), (124,  0,  0), (126,  0,  0), (127,  0,  0), (128,  0,  0),
    (130,  0,  0), (131,  0,  0), (133,  0,  0), (134,  0,  0), (136,  0,  0), (137,  0,  0),
    (139,  0,  0), (140,  0,  0), (141,  0,  0), (143,  0,  0), (144,  0,  0), (146,  0,  0),
    (147,  0,  0), (149,  0,  0), (150,  0,  0), (152,  0,  0), (153,  0,  0), (155,  0,  0),
    (156,  0,  0), (157,  0,  0), (159,  0,  0), (160,  0,  0), (162,  0,  0), (163,  0,  0),
    (165,  0,  0), (166,  0,  0), (168,  0,  0), (169,  0,  0), (170,  0,  0), (172,  0,  0),
    (173,  0,  0), (175,  1,  0), (176,  3,  0), (178,  5,  0), (179,  7,  0), (181,  9,  0),
    (182, 11,  0), (184, 13,  0), (185, 15,  0), (186, 17,  0), (188, 18,  0), (189, 20,  0),
    (191, 22,  0), (192, 24,  0), (194, 26,  0), (195, 28,  0), (197, 30,  0), (198, 32,  0),
    (199, 34,  0), (201, 35,  0), (202, 37,  0), (204, 39,  0), (205, 41,  0), (207, 43,  0),
    (208, 45,  0), (210, 47,  0), (211, 49,  0), (212, 51,  0), (214, 52,  0), (215, 54,  0),
    (217, 56,  0), (218, 58,  0), (220, 60,  0), (221, 62,  0), (223, 64,  0), (224, 66,  0),
    (226, 68,  0), (227, 69,  0), (228, 71,  0), (230, 73,  0), (231, 75,  0), (233, 77,  0),
    (234, 79,  0), (236, 81,  0), (237, 83,  0), (239, 85,  0), (240, 86,  0), (241, 88,  0),
    (243, 90,  0), (244, 92,  0), (246, 94,  0), (247, 96,  0), (249, 98,  0), (250,100,  0),
    (252,102,  0), (253,103,  0), (255,105,  0), (255,107,  0), (255,109,  0), (255,111,  0),
    (255,113,  0), (255,115,  0), (255,117,  0), (255,119,  0), (255,120,  0), (255,122,  0),
    (255,124,  0), (255,126,  0), (255,128,  0), (255,130,  0), (255,132,  0), (255,134,  3),
    (255,136,  7), (255,137, 11), (255,139, 15), (255,141, 19), (255,143, 23), (255,145, 27),
    (255,147, 31), (255,149, 35), (255,151, 39), (255,153, 43), (255,154, 47), (255,156, 51),
    (255,158, 54), (255,160, 58), (255,162, 62), (255,164, 66), (255,166, 70), (255,168, 74),
    (255,170, 78), (255,171, 82), (255,173, 86), (255,175, 90), (255,177, 94), (255,179, 98),
    (255,181,102), (255,183,105), (255,185,109), (255,187,113), (255,188,117), (255,190,121),
    (255,192,125), (255,194,129), (255,196,133), (255,198,137), (255,200,141), (255,202,145),
    (255,204,149), (255,205,153), (255,207,156), (255,209,160), (255,211,164), (255,213,168),
    (255,215,172), (255,217,176), (255,219,180), (255,221,184), (255,222,188), (255,224,192),
    (255,226,196), (255,228,200), (255,230,204), (255,232,207), (255,234,211), (255,236,215),
    (255,238,219), (255,239,223), (255,241,227), (255,243,231), (255,245,235), (255,247,239),
    (255,249,243), (255,251,247), (255,253,251), (255,255,255),
)
