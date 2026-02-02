"""
Spatial transformation utilities for image padding, cropping, and flipping.

Provides functions for size normalization and geometric transformations.
"""

from typing import Tuple, Union, Literal

import numpy as np


def pad_image(
    data: np.ndarray,
    target_size: Tuple[int, int],
    pad_value: Union[int, float] = 0,
    center: bool = True
) -> np.ndarray:
    """
    Pad an image to a target size with a constant value.

    Adds padding symmetrically around the image to reach the target dimensions.
    For odd padding amounts, the extra pixel is added to the bottom/right by default
    when center=True.

    Parameters
    ----------
    data : np.ndarray
        Input image of shape (height, width) or (height, width, channels).
    target_size : tuple of int
        Target dimensions as (target_height, target_width).
    pad_value : int or float, optional
        Value to use for padding (default=0). For multi-channel images,
        the same value is used for all channels.
    center : bool, optional
        If True, center the image in the padded canvas (default=True).
        If False, align to top-left corner.

    Returns
    -------
    np.ndarray
        Padded image with shape matching target_size (plus channels if 3D).

    Raises
    ------
    ValueError
        If target size is smaller than input image size in any dimension.

    Examples
    --------
    >>> # Pad 512x512 image to 1024x1024 for alignment
    >>> img = np.random.rand(512, 512)
    >>> padded = pad_image(img, target_size=(1024, 1024), pad_value=0)
    >>> padded.shape
    (1024, 1024)

    >>> # Pad SDO/AIA image with NaN for off-disk regions
    >>> padded = pad_image(aia_img, (5000, 5000), pad_value=np.nan)

    >>> # Pad multi-channel image
    >>> rgb_img = np.random.rand(100, 100, 3)
    >>> padded_rgb = pad_image(rgb_img, (200, 200))
    >>> padded_rgb.shape
    (200, 200, 3)
    """
    if data.ndim not in (2, 3):
        raise ValueError(f"Input must be 2D or 3D, got {data.ndim}D")

    target_height, target_width = target_size
    input_height, input_width = data.shape[:2]

    # Validate target size
    if target_height < input_height:
        raise ValueError(
            f"target_height ({target_height}) must be >= input height ({input_height})"
        )
    if target_width < input_width:
        raise ValueError(
            f"target_width ({target_width}) must be >= input width ({input_width})"
        )

    # Calculate padding amounts
    if center:
        # Center the image: distribute padding symmetrically
        pad_top = (target_height - input_height) // 2
        pad_bottom = target_height - input_height - pad_top

        pad_left = (target_width - input_width) // 2
        pad_right = target_width - input_width - pad_left
    else:
        # Align to top-left: all padding goes to bottom-right
        pad_top = 0
        pad_bottom = target_height - input_height
        pad_left = 0
        pad_right = target_width - input_width

    # Construct pad_width for np.pad
    if data.ndim == 2:
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))
    else:  # 3D image
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))

    # Apply padding
    padded = np.pad(
        data,
        pad_width=pad_width,
        mode='constant',
        constant_values=pad_value
    )

    return padded


def crop_or_pad(
    data: np.ndarray,
    target_size: Tuple[int, int],
    pad_value: Union[int, float] = 0,
    center: bool = True
) -> np.ndarray:
    """
    Crop or pad an image to exactly match the target size.

    If the input is larger than target, crops from the center (or top-left).
    If the input is smaller, pads with the specified value.

    Parameters
    ----------
    data : np.ndarray
        Input image of shape (height, width) or (height, width, channels).
    target_size : tuple of int
        Target dimensions as (target_height, target_width).
    pad_value : int or float, optional
        Value to use for padding if needed (default=0).
    center : bool, optional
        If True, center-crop/center-pad (default=True).
        If False, use top-left alignment.

    Returns
    -------
    np.ndarray
        Image resized to target_size (plus channels if 3D).

    Examples
    --------
    >>> # Ensure consistent size for batching
    >>> img1 = np.random.rand(400, 600)
    >>> img2 = np.random.rand(800, 500)
    >>> normalized1 = crop_or_pad(img1, (512, 512))
    >>> normalized2 = crop_or_pad(img2, (512, 512))
    >>> assert normalized1.shape == normalized2.shape == (512, 512)
    """
    if data.ndim not in (2, 3):
        raise ValueError(f"Input must be 2D or 3D, got {data.ndim}D")

    target_height, target_width = target_size
    input_height, input_width = data.shape[:2]

    # Determine crop/pad for height
    if input_height > target_height:
        # Need to crop
        if center:
            crop_top = (input_height - target_height) // 2
            crop_bottom = crop_top + target_height
        else:
            crop_top = 0
            crop_bottom = target_height
        data = data[crop_top:crop_bottom, :]
    elif input_height < target_height:
        # Need to pad
        data = pad_image(
            data,
            target_size=(target_height, data.shape[1]),
            pad_value=pad_value,
            center=center
        )

    # Determine crop/pad for width
    if input_width > target_width:
        # Need to crop
        if center:
            crop_left = (input_width - target_width) // 2
            crop_right = crop_left + target_width
        else:
            crop_left = 0
            crop_right = target_width
        data = data[:, crop_left:crop_right]
    elif input_width < target_width:
        # Need to pad
        data = pad_image(
            data,
            target_size=(data.shape[0], target_width),
            pad_value=pad_value,
            center=center
        )

    return data


def flip_image(
    image: np.ndarray,
    axis: Literal['vertical', 'horizontal', 'both'] = 'vertical'
) -> np.ndarray:
    """
    Flip an image vertically, horizontally, or both.

    Parameters
    ----------
    image : np.ndarray
        Input image (2D or 3D).
    axis : {'vertical', 'horizontal', 'both'}, optional
        - 'vertical': Flip top to bottom (default)
        - 'horizontal': Flip left to right
        - 'both': Flip both axes (180 degree rotation)

    Returns
    -------
    np.ndarray
        Flipped image with same shape and dtype.

    Examples
    --------
    >>> img = np.array([[1, 2], [3, 4]])
    >>> flip_image(img, 'vertical')
    array([[3, 4],
           [1, 2]])

    >>> flip_image(img, 'horizontal')
    array([[2, 1],
           [4, 3]])

    >>> # Useful for SDO coordinate transformations
    >>> flipped = flip_image(sdo_image, 'vertical')
    """
    if axis == 'vertical':
        return np.flipud(image)
    elif axis == 'horizontal':
        return np.fliplr(image)
    elif axis == 'both':
        return np.flipud(np.fliplr(image))
    else:
        raise ValueError(f"axis must be 'vertical', 'horizontal', or 'both', got '{axis}'")


def roll_image(
    image: np.ndarray,
    shift_y: int,
    shift_x: int
) -> np.ndarray:
    """
    Cyclically roll an image by specified amounts.

    Pixels that roll beyond the edge reappear on the opposite side.
    Useful for aligning time-series images or periodic boundary conditions.

    Parameters
    ----------
    image : np.ndarray
        Input image (2D or 3D).
    shift_y : int
        Shift amount in y-direction (positive = down).
    shift_x : int
        Shift amount in x-direction (positive = right).

    Returns
    -------
    np.ndarray
        Rolled image with same shape and dtype.

    Examples
    --------
    >>> img = np.array([[1, 2, 3],
    ...                 [4, 5, 6],
    ...                 [7, 8, 9]])
    >>> roll_image(img, shift_y=1, shift_x=0)
    array([[7, 8, 9],
           [1, 2, 3],
           [4, 5, 6]])

    >>> # Align image stack for solar rotation
    >>> for i, img in enumerate(images):
    ...     aligned = roll_image(img, shift_y=0, shift_x=shift_x[i])
    """
    return np.roll(image, (shift_y, shift_x), axis=(0, 1))


# Convenience alias
pad = pad_image
