"""
Generic colorization primitives.

Turns an 8-bit grayscale image into an RGB image through a 256-entry color
lookup table (LUT). This module is instrument-agnostic: the domain-standard
SDO/AIA color tables live in :mod:`egghouse.sdo.aia_color`, which builds LUTs
that are consumed here.

Example:
    >>> import numpy as np
    >>> from egghouse.image import apply_colormap, lut_from_matplotlib
    >>> gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
    >>> lut = lut_from_matplotlib('inferno')      # (256, 3) uint8
    >>> rgb = apply_colormap(gray, lut)           # (16, 16, 3) uint8
"""

from typing import Optional, Union

import numpy as np


def apply_colormap(
    gray: np.ndarray,
    lut: np.ndarray,
) -> np.ndarray:
    """
    Colorize an 8-bit grayscale image through a 256-entry RGB lookup table.

    Each pixel value ``v`` in ``gray`` is replaced by ``lut[v]``, producing an
    RGB image. This is a pure index operation, so it is exact and fast (no
    interpolation).

    Parameters
    ----------
    gray : np.ndarray
        2D grayscale image. If dtype is not uint8, values are clipped to
        [0, 255] and cast to uint8 before lookup (they are assumed to already
        be on an 8-bit scale, e.g. the output of ``bytescale_image`` or
        ``aia_intscale``).
    lut : np.ndarray
        Color lookup table of shape (256, 3), dtype uint8. Row ``i`` is the
        ``(R, G, B)`` color for grayscale value ``i``.

    Returns
    -------
    np.ndarray
        RGB image of shape ``(*gray.shape, 3)``, dtype uint8.

    Raises
    ------
    ValueError
        If ``gray`` is not 2D or ``lut`` does not have shape (256, 3).

    Examples
    --------
    >>> gray = np.array([[0, 128], [255, 64]], dtype=np.uint8)
    >>> lut = np.stack([np.arange(256)] * 3, axis=1).astype(np.uint8)  # gray ramp
    >>> apply_colormap(gray, lut).shape
    (2, 2, 3)
    """
    gray = np.asarray(gray)
    if gray.ndim != 2:
        raise ValueError(f"gray must be a 2D image, got {gray.ndim}D")

    lut = np.asarray(lut)
    if lut.shape != (256, 3):
        raise ValueError(f"lut must have shape (256, 3), got {lut.shape}")
    if lut.dtype != np.uint8:
        lut = np.clip(lut, 0, 255).astype(np.uint8)

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    # Fancy indexing: (H, W) uint8 indices into (256, 3) -> (H, W, 3)
    return lut[gray]


def lut_from_matplotlib(
    cmap: Union[str, "object"],
    n: int = 256,
) -> np.ndarray:
    """
    Build a 256-entry uint8 RGB LUT from a matplotlib colormap.

    Convenience bridge for using any matplotlib colormap (by name or as a
    ``Colormap`` object) with :func:`apply_colormap`. Requires matplotlib
    (imported lazily).

    Parameters
    ----------
    cmap : str or matplotlib.colors.Colormap
        Colormap name (e.g. ``'inferno'``) or a Colormap instance.
    n : int, optional
        Number of LUT entries (default 256). Must be 256 to feed
        :func:`apply_colormap`.

    Returns
    -------
    np.ndarray
        LUT of shape ``(n, 3)``, dtype uint8.
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt  # noqa: F401  (ensures colormaps registered)
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "lut_from_matplotlib requires matplotlib. "
            "Install it with `pip install matplotlib`."
        ) from exc

    if isinstance(cmap, str):
        cmap = matplotlib.colormaps[cmap]

    samples = cmap(np.linspace(0.0, 1.0, n))[:, :3]
    return np.rint(samples * 255.0).astype(np.uint8)
