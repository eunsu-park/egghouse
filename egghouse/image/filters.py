"""
Image filtering utilities for noise reduction and edge detection.

Provides scipy.ndimage-based filters commonly used in image preprocessing.
"""

from typing import Union, Tuple, Optional

import numpy as np
from scipy import ndimage


def gaussian_smooth(
    image: np.ndarray,
    sigma: Union[float, Tuple[float, ...]] = 1.0,
    preserve_range: bool = True
) -> np.ndarray:
    """
    Apply Gaussian smoothing filter for noise reduction.

    Parameters
    ----------
    image : np.ndarray
        Input image (2D or 3D). For 3D images, filter is applied to
        spatial dimensions only.
    sigma : float or tuple of float, optional
        Standard deviation for Gaussian kernel. If float, same sigma
        is used for all spatial dimensions. Default is 1.0.
    preserve_range : bool, optional
        If True, preserve original dtype (default=True).

    Returns
    -------
    np.ndarray
        Smoothed image.

    Examples
    --------
    >>> # Basic smoothing
    >>> smoothed = gaussian_smooth(image, sigma=1.5)

    >>> # Different sigma for each axis
    >>> smoothed = gaussian_smooth(image, sigma=(2.0, 1.0))

    >>> # Preprocessing for AIA data
    >>> aia_smooth = gaussian_smooth(aia_data, sigma=1.0)
    """
    original_dtype = image.dtype

    # Handle sigma for different dimensions
    if image.ndim == 3 and isinstance(sigma, (int, float)):
        # For 3D images, don't smooth channel dimension
        sigma = (sigma, sigma, 0)
    elif image.ndim == 3 and len(sigma) == 2:
        sigma = (sigma[0], sigma[1], 0)

    smoothed = ndimage.gaussian_filter(image.astype(np.float64), sigma=sigma)

    if preserve_range:
        if np.issubdtype(original_dtype, np.integer):
            info = np.iinfo(original_dtype)
            smoothed = np.clip(smoothed, info.min, info.max)
        smoothed = smoothed.astype(original_dtype)

    return smoothed


def median_denoise(
    image: np.ndarray,
    size: Union[int, Tuple[int, ...]] = 3,
    preserve_range: bool = True
) -> np.ndarray:
    """
    Apply median filter for noise removal.

    Effective for removing salt-and-pepper noise and cosmic rays
    while preserving edges better than Gaussian smoothing.

    Parameters
    ----------
    image : np.ndarray
        Input image (2D or 3D).
    size : int or tuple of int, optional
        Size of the median filter window. If int, same size is used
        for all spatial dimensions. Default is 3.
    preserve_range : bool, optional
        If True, preserve original dtype (default=True).

    Returns
    -------
    np.ndarray
        Denoised image.

    Examples
    --------
    >>> # Remove cosmic rays from CCD image
    >>> clean = median_denoise(raw_image, size=3)

    >>> # Larger window for stronger noise
    >>> clean = median_denoise(noisy_image, size=5)

    >>> # Non-square window
    >>> clean = median_denoise(image, size=(3, 5))
    """
    original_dtype = image.dtype

    # Handle size for different dimensions
    if image.ndim == 3 and isinstance(size, int):
        # For 3D images, don't filter channel dimension
        size = (size, size, 1)
    elif image.ndim == 3 and len(size) == 2:
        size = (size[0], size[1], 1)

    denoised = ndimage.median_filter(image.astype(np.float64), size=size)

    if preserve_range:
        if np.issubdtype(original_dtype, np.integer):
            info = np.iinfo(original_dtype)
            denoised = np.clip(denoised, info.min, info.max)
        denoised = denoised.astype(original_dtype)

    return denoised


def laplacian_edge(
    image: np.ndarray,
    mode: str = 'reflect'
) -> np.ndarray:
    """
    Detect edges using Laplacian operator.

    Computes the Laplacian (second derivative) of the image,
    highlighting regions of rapid intensity change.

    Parameters
    ----------
    image : np.ndarray
        Input image (2D).
    mode : str, optional
        How to handle boundaries. Options: 'reflect', 'constant',
        'nearest', 'mirror', 'wrap'. Default is 'reflect'.

    Returns
    -------
    np.ndarray
        Edge-detected image (float64).

    Examples
    --------
    >>> # Basic edge detection
    >>> edges = laplacian_edge(image)

    >>> # Find zero crossings for precise edges
    >>> edges = laplacian_edge(gaussian_smooth(image, sigma=1.0))
    """
    if image.ndim != 2:
        raise ValueError(f"laplacian_edge requires 2D image, got {image.ndim}D")

    return ndimage.laplace(image.astype(np.float64), mode=mode)


def sobel_edge(
    image: np.ndarray,
    axis: Optional[int] = None,
    mode: str = 'reflect'
) -> np.ndarray:
    """
    Detect edges using Sobel operator.

    Computes the gradient magnitude or directional gradient of the image.

    Parameters
    ----------
    image : np.ndarray
        Input image (2D).
    axis : int, optional
        Axis along which to compute gradient.
        - None: Compute gradient magnitude (default)
        - 0: Gradient in y-direction (vertical edges)
        - 1: Gradient in x-direction (horizontal edges)
    mode : str, optional
        How to handle boundaries. Default is 'reflect'.

    Returns
    -------
    np.ndarray
        Edge-detected image (float64).

    Examples
    --------
    >>> # Gradient magnitude (all edges)
    >>> edges = sobel_edge(image)

    >>> # Vertical edges only
    >>> edges_y = sobel_edge(image, axis=0)

    >>> # Horizontal edges only
    >>> edges_x = sobel_edge(image, axis=1)
    """
    if image.ndim != 2:
        raise ValueError(f"sobel_edge requires 2D image, got {image.ndim}D")

    img = image.astype(np.float64)

    if axis is None:
        # Compute gradient magnitude
        sobel_y = ndimage.sobel(img, axis=0, mode=mode)
        sobel_x = ndimage.sobel(img, axis=1, mode=mode)
        return np.hypot(sobel_x, sobel_y)
    else:
        return ndimage.sobel(img, axis=axis, mode=mode)


def unsharp_mask(
    image: np.ndarray,
    sigma: float = 1.0,
    amount: float = 1.0,
    preserve_range: bool = True
) -> np.ndarray:
    """
    Sharpen an image using unsharp masking.

    Enhances edges by subtracting a blurred version of the image.

    Parameters
    ----------
    image : np.ndarray
        Input image (2D or 3D).
    sigma : float, optional
        Gaussian blur sigma. Default is 1.0.
    amount : float, optional
        Sharpening strength. Values > 1.0 increase sharpening.
        Default is 1.0.
    preserve_range : bool, optional
        If True, preserve original dtype (default=True).

    Returns
    -------
    np.ndarray
        Sharpened image.

    Examples
    --------
    >>> # Basic sharpening
    >>> sharp = unsharp_mask(image, sigma=1.0, amount=1.5)

    >>> # Stronger sharpening
    >>> sharp = unsharp_mask(image, sigma=2.0, amount=2.0)
    """
    original_dtype = image.dtype
    img = image.astype(np.float64)

    blurred = gaussian_smooth(img, sigma=sigma, preserve_range=False)
    sharpened = img + amount * (img - blurred)

    if preserve_range:
        if np.issubdtype(original_dtype, np.integer):
            info = np.iinfo(original_dtype)
            sharpened = np.clip(sharpened, info.min, info.max)
        sharpened = sharpened.astype(original_dtype)

    return sharpened
