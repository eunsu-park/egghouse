"""
Mask generation utilities for circular and annular regions.

Commonly used for solar disk analysis and feature isolation in solar physics.
"""

from typing import Tuple, Union, Optional, Literal

import numpy as np


def circle_mask(
    image_size: Union[int, Tuple[int, int]],
    radius: float,
    center: Optional[Tuple[float, float]] = None,
    mask_type: Literal['inner', 'outer'] = 'inner'
) -> np.ndarray:
    """
    Generate a circular boolean mask.

    Creates a 2D boolean array with True values inside (or outside) a circle.
    Commonly used for solar disk masking in SDO/AIA or H-alpha imagery.

    Parameters
    ----------
    image_size : int or tuple of int
        Size of the output mask. If int, creates a square mask (size, size).
        If tuple, creates a mask with shape (height, width).
    radius : float
        Radius of the circle in pixels.
    center : tuple of float, optional
        Center coordinates as (cy, cx). If None, defaults to image center.
    mask_type : {'inner', 'outer'}, optional
        - 'inner': True inside the circle (default)
        - 'outer': True outside the circle

    Returns
    -------
    np.ndarray
        Boolean mask with shape (height, width).

    Raises
    ------
    ValueError
        If mask_type is not 'inner' or 'outer'.

    Examples
    --------
    >>> # Mask for 4096x4096 SDO image with solar disk radius ~1600 pixels
    >>> disk_mask = circle_mask(4096, radius=1600, mask_type='inner')
    >>> masked_image = np.where(disk_mask, image, 0)

    >>> # Rectangular image with off-center circle
    >>> mask = circle_mask((512, 1024), radius=200, center=(256, 600))

    >>> # Mask out the solar disk to analyze corona
    >>> corona_mask = circle_mask(4096, radius=1600, mask_type='outer')
    """
    # Handle image_size as int or tuple
    if isinstance(image_size, (int, np.integer)):
        height, width = image_size, image_size
    else:
        height, width = image_size

    # Default center: image center
    if center is None:
        cy, cx = height / 2.0, width / 2.0
    else:
        cy, cx = center

    # Create coordinate grids (memory-efficient with ogrid)
    y, x = np.ogrid[:height, :width]

    # Calculate distance from center
    distance_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)

    # Generate mask based on type
    if mask_type == 'inner':
        mask = distance_from_center < radius
    elif mask_type == 'outer':
        mask = distance_from_center >= radius
    else:
        raise ValueError(f"mask_type must be 'inner' or 'outer', got '{mask_type}'")

    return mask


def annulus_mask(
    image_size: Union[int, Tuple[int, int]],
    inner_radius: float,
    outer_radius: float,
    center: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Generate an annular (ring-shaped) boolean mask.

    Useful for analyzing solar features at specific radial distances,
    such as the chromosphere or inner corona.

    Parameters
    ----------
    image_size : int or tuple of int
        Size of the output mask.
    inner_radius : float
        Inner radius of the annulus in pixels.
    outer_radius : float
        Outer radius of the annulus in pixels.
    center : tuple of float, optional
        Center coordinates as (cy, cx). If None, defaults to image center.

    Returns
    -------
    np.ndarray
        Boolean mask with True values in the annular region.

    Raises
    ------
    ValueError
        If inner_radius >= outer_radius.

    Examples
    --------
    >>> # Analyze region between 1.0 and 1.3 solar radii
    >>> solar_radius = 1600  # pixels
    >>> corona_ring = annulus_mask(4096, inner_radius=solar_radius,
    ...                            outer_radius=solar_radius * 1.3)

    >>> # Extract data from annular region
    >>> annular_data = image[corona_ring]
    >>> mean_intensity = annular_data.mean()
    """
    if inner_radius >= outer_radius:
        raise ValueError(
            f"inner_radius ({inner_radius}) must be less than "
            f"outer_radius ({outer_radius})"
        )

    inner = circle_mask(image_size, inner_radius, center=center, mask_type='inner')
    outer = circle_mask(image_size, outer_radius, center=center, mask_type='inner')
    return outer & ~inner
