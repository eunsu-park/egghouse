"""
Basic image transformation utilities.

Provides scipy.ndimage-based functions for resizing, rotating, and scaling
images while preserving original data types and ranges.
"""

from typing import Tuple, Union, Optional

import numpy as np
from scipy import ndimage


def resize_image(
    image: np.ndarray,
    size: Tuple[int, int],
    order: int = 1,
    preserve_range: bool = True
) -> np.ndarray:
    """
    Resize an image to the specified size.

    Args:
        image: Input image. Can be 2D (grayscale) or 3D (color, channel last).
            Supports any numpy dtype (uint8, uint16, float32, float64, etc.)
        size: Target size as (height, width).
        order: Interpolation order (default=1):
            - 0: nearest-neighbor
            - 1: bilinear
            - 2: bi-quadratic
            - 3: bi-cubic
        preserve_range: If True, preserve the original data range and dtype (default=True).

    Returns:
        Resized image with the same dtype as input.

    Example:
    >>> img = np.random.rand(100, 100).astype(np.float32)
    >>> resized = resize_image(img, (50, 50))
    >>> resized.shape
    (50, 50)
    >>> resized.dtype
    dtype('float32')
    """
    if image.ndim not in (2, 3):
        raise ValueError(f"Image must be 2D or 3D, got {image.ndim}D")

    original_dtype = image.dtype
    target_height, target_width = size

    # Calculate zoom factors
    if image.ndim == 2:
        zoom_factors = (target_height / image.shape[0], target_width / image.shape[1])
    else:
        # For 3D images (H, W, C), don't zoom the channel dimension
        zoom_factors = (
            target_height / image.shape[0],
            target_width / image.shape[1],
            1.0
        )

    # Perform resize using scipy.ndimage.zoom
    resized = ndimage.zoom(image.astype(np.float64), zoom_factors, order=order)

    if preserve_range:
        # Clip to original dtype range if integer type
        if np.issubdtype(original_dtype, np.integer):
            info = np.iinfo(original_dtype)
            resized = np.clip(resized, info.min, info.max)
        resized = resized.astype(original_dtype)

    return resized


def rotate_image(
    image: np.ndarray,
    angle: float,
    order: int = 1,
    reshape: bool = False,
    cval: Union[float, int] = 0,
    preserve_range: bool = True
) -> np.ndarray:
    """
    Rotate an image by the specified angle (in degrees).

    Args:
        image: Input image. Can be 2D (grayscale) or 3D (color, channel last).
            Supports any numpy dtype (uint8, uint16, float32, float64, etc.)
        angle: Rotation angle in degrees. Positive values rotate counter-clockwise.
        order: Interpolation order (default=1):
            - 0: nearest-neighbor
            - 1: bilinear
            - 2: bi-quadratic
            - 3: bi-cubic
        reshape: If True, output shape is adapted to contain the entire rotated image.
            If False, output shape is the same as input (default=False).
        cval: Value used for points outside the boundaries (default=0).
        preserve_range: If True, preserve the original data range and dtype (default=True).

    Returns:
        Rotated image with the same dtype as input.

    Example:
    >>> img = np.random.rand(100, 100).astype(np.float32)
    >>> rotated = rotate_image(img, 45)
    >>> rotated.shape
    (100, 100)
    >>> rotated.dtype
    dtype('float32')

    >>> # Rotate with expanded canvas
    >>> rotated_full = rotate_image(img, 45, reshape=True)
    """
    if image.ndim not in (2, 3):
        raise ValueError(f"Image must be 2D or 3D, got {image.ndim}D")

    original_dtype = image.dtype

    if image.ndim == 2:
        # 2D grayscale image
        rotated = ndimage.rotate(
            image.astype(np.float64),
            angle,
            order=order,
            reshape=reshape,
            cval=float(cval)
        )
    else:
        # 3D image (H, W, C) - rotate only spatial dimensions
        rotated = ndimage.rotate(
            image.astype(np.float64),
            angle,
            axes=(0, 1),
            order=order,
            reshape=reshape,
            cval=float(cval)
        )

    if preserve_range:
        # Clip to original dtype range if integer type
        if np.issubdtype(original_dtype, np.integer):
            info = np.iinfo(original_dtype)
            rotated = np.clip(rotated, info.min, info.max)
        rotated = rotated.astype(original_dtype)

    return rotated


def bytescale_image(
    data: np.ndarray,
    imin: Optional[Union[float, int]] = None,
    imax: Optional[Union[float, int]] = None,
    omin: int = 0,
    omax: int = 255
) -> np.ndarray:
    """
    Scale input data to byte range [omin, omax].

    Linearly maps input values from [imin, imax] to output range [omin, omax],
    then converts to uint8. Useful for visualizing scientific data (e.g., FITS images)
    or preparing data for display/saving.

    Args:
        data: Input array of any numeric dtype.
        imin: Input minimum value for scaling. If None, uses data.min().
        imax: Input maximum value for scaling. If None, uses data.max().
        omin: Output minimum value (default=0).
        omax: Output maximum value (default=255).

    Returns:
        Scaled array with dtype=uint8.

    Raises:
        ValueError: If imin >= imax (invalid input range).

    Example:
    >>> # Scale 16-bit solar image to 8-bit for display
    >>> img_16bit = np.random.randint(0, 65535, (512, 512), dtype=np.uint16)
    >>> img_8bit = bytescale_image(img_16bit, imin=0, imax=65535)

    >>> # Auto-scale float data with contrast stretch
    >>> data = np.random.rand(100, 100) * 1000 - 500  # range: [-500, 500]
    >>> scaled = bytescale_image(data)  # auto imin/imax from data

    >>> # Apply percentile-based scaling for better contrast
    >>> imin, imax = np.percentile(data, [1, 99])
    >>> scaled = bytescale_image(data, imin=imin, imax=imax)
    """
    # Convert to float64 for precision during calculations
    data = np.asarray(data, dtype=np.float64)

    # Auto-detect input range if not specified
    if imin is None:
        imin = np.nanmin(data)
    if imax is None:
        imax = np.nanmax(data)

    # Validate input range
    if imin >= imax:
        raise ValueError(f"imin ({imin}) must be less than imax ({imax})")

    # Linear scaling: [imin, imax] -> [0, 1] -> [omin, omax]
    scaled = (data - imin) / (imax - imin)
    scaled = scaled * (omax - omin) + omin

    # Clip to output range and convert to uint8
    scaled = np.clip(scaled, omin, omax).astype(np.uint8)

    return scaled


# Convenience aliases
resize = resize_image
rotate = rotate_image
bytescale = bytescale_image
