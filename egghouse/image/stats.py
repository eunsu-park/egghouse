"""
Image statistics and normalization utilities.

Provides functions for image analysis, normalization, and histogram operations.
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy import ndimage


def normalize_image(
    image: np.ndarray,
    mean: Optional[float] = None,
    std: Optional[float] = None
) -> np.ndarray:
    """
    Normalize image to zero mean and unit variance.

    Performs z-score normalization: (image - mean) / std.
    Commonly used as preprocessing for neural networks.

    Parameters
    ----------
    image : np.ndarray
        Input image of any shape.
    mean : float, optional
        Mean value for normalization. If None, computed from image.
    std : float, optional
        Standard deviation for normalization. If None, computed from image.

    Returns
    -------
    np.ndarray
        Normalized image (float64).

    Examples
    --------
    >>> # Auto-compute mean and std
    >>> normalized = normalize_image(image)

    >>> # Use pre-computed statistics (e.g., from training set)
    >>> normalized = normalize_image(image, mean=127.5, std=64.0)

    >>> # Normalize each image in a batch
    >>> for img in batch:
    ...     norm_img = normalize_image(img)
    """
    img = image.astype(np.float64)

    if mean is None:
        mean = np.mean(img)
    if std is None:
        std = np.std(img)

    # Avoid division by zero
    if std == 0:
        std = 1e-10

    return (img - mean) / std


def get_image_stats(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    percentiles: Tuple[float, ...] = (1, 5, 25, 50, 75, 95, 99)
) -> Dict[str, float]:
    """
    Compute comprehensive statistics for an image.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    mask : np.ndarray, optional
        Boolean mask. If provided, statistics are computed only
        for pixels where mask is True.
    percentiles : tuple of float, optional
        Percentiles to compute. Default is (1, 5, 25, 50, 75, 95, 99).

    Returns
    -------
    dict
        Dictionary containing:
        - mean, std, min, max, median
        - p1, p5, p25, p50, p75, p95, p99 (or custom percentiles)
        - count: number of pixels used

    Examples
    --------
    >>> stats = get_image_stats(image)
    >>> print(f"Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")

    >>> # Stats only within solar disk
    >>> disk_mask = circle_mask(4096, radius=1600)
    >>> stats = get_image_stats(image, mask=disk_mask)

    >>> # Check for saturated pixels
    >>> if stats['max'] >= 65535:
    ...     print("Warning: Image may be saturated")
    """
    # Apply mask if provided
    if mask is not None:
        data = image[mask].flatten()
    else:
        data = image.flatten()

    # Remove NaN values for statistics
    data = data[~np.isnan(data)]

    result = {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'median': float(np.median(data)),
        'count': len(data),
    }

    # Add percentiles
    for p in percentiles:
        result[f'p{int(p)}'] = float(np.percentile(data, p))

    return result


def histogram_equalization(
    image: np.ndarray,
    nbins: int = 256
) -> np.ndarray:
    """
    Enhance contrast using histogram equalization.

    Redistributes pixel intensities to achieve a more uniform histogram,
    improving contrast in images with narrow intensity distributions.

    Parameters
    ----------
    image : np.ndarray
        Input image (2D). Should be integer type or will be converted.
    nbins : int, optional
        Number of histogram bins. Default is 256.

    Returns
    -------
    np.ndarray
        Contrast-enhanced image (uint8).

    Examples
    --------
    >>> # Enhance low-contrast image
    >>> enhanced = histogram_equalization(image)

    >>> # Display comparison
    >>> fig, (ax1, ax2) = plt.subplots(1, 2)
    >>> ax1.imshow(image, cmap='gray')
    >>> ax2.imshow(enhanced, cmap='gray')
    """
    if image.ndim != 2:
        raise ValueError(f"histogram_equalization requires 2D image, got {image.ndim}D")

    # Flatten image
    flat = image.flatten().astype(np.float64)

    # Compute histogram
    hist, bin_edges = np.histogram(flat, bins=nbins, range=(np.min(flat), np.max(flat)))

    # Compute cumulative distribution function (CDF)
    cdf = np.cumsum(hist).astype(np.float64)
    cdf = cdf / cdf[-1]  # Normalize to [0, 1]

    # Map original values to equalized values
    bin_indices = np.digitize(flat, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, nbins - 1)

    equalized = (cdf[bin_indices] * 255).reshape(image.shape)

    return equalized.astype(np.uint8)


def percentile_scale(
    image: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    omin: int = 0,
    omax: int = 255
) -> np.ndarray:
    """
    Scale image using percentile-based clipping.

    Clips values at specified percentiles before scaling, which is
    more robust to outliers than min/max scaling.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    low_percentile : float, optional
        Lower percentile for clipping. Default is 1.0.
    high_percentile : float, optional
        Upper percentile for clipping. Default is 99.0.
    omin : int, optional
        Output minimum value. Default is 0.
    omax : int, optional
        Output maximum value. Default is 255.

    Returns
    -------
    np.ndarray
        Scaled image (uint8).

    Examples
    --------
    >>> # Standard percentile scaling
    >>> scaled = percentile_scale(image)

    >>> # More aggressive clipping for high dynamic range
    >>> scaled = percentile_scale(hdr_image, low_percentile=5, high_percentile=95)

    >>> # Custom output range
    >>> scaled = percentile_scale(image, omin=10, omax=245)
    """
    data = image.astype(np.float64)

    # Compute percentile values
    vmin = np.percentile(data, low_percentile)
    vmax = np.percentile(data, high_percentile)

    # Handle edge case where vmin == vmax
    if vmin >= vmax:
        vmax = vmin + 1

    # Scale to output range
    scaled = (data - vmin) / (vmax - vmin)
    scaled = scaled * (omax - omin) + omin

    return np.clip(scaled, omin, omax).astype(np.uint8)


def find_disk_center(
    image: np.ndarray,
    threshold: Optional[float] = None,
    method: str = 'centroid'
) -> Tuple[float, float]:
    """
    Find the center of a bright disk (e.g., solar disk) in an image.

    Parameters
    ----------
    image : np.ndarray
        Input image (2D) with a bright disk.
    threshold : float, optional
        Intensity threshold for disk detection. Pixels below this
        value are ignored. If None, uses mean intensity.
    method : str, optional
        Method for center detection:
        - 'centroid': Center of mass of thresholded image (default)
        - 'geometric': Geometric center of thresholded region

    Returns
    -------
    tuple of float
        Center coordinates as (cy, cx).

    Examples
    --------
    >>> # Find solar disk center
    >>> cy, cx = find_disk_center(aia_image)
    >>> print(f"Disk center: ({cy:.1f}, {cx:.1f})")

    >>> # Use custom threshold for noisy image
    >>> cy, cx = find_disk_center(image, threshold=100)

    >>> # Create centered mask
    >>> mask = circle_mask(image.shape, radius=1600, center=(cy, cx))
    """
    if image.ndim != 2:
        raise ValueError(f"find_disk_center requires 2D image, got {image.ndim}D")

    img = image.astype(np.float64)

    # Apply threshold
    if threshold is None:
        threshold = np.mean(img)

    binary = img > threshold

    if method == 'centroid':
        # Weight by intensity
        weighted = img * binary.astype(float)
        cy, cx = ndimage.center_of_mass(weighted)
    elif method == 'geometric':
        # Pure geometric center of thresholded region
        cy, cx = ndimage.center_of_mass(binary.astype(float))
    else:
        raise ValueError(f"method must be 'centroid' or 'geometric', got '{method}'")

    return float(cy), float(cx)


def adaptive_threshold(
    image: np.ndarray,
    block_size: int = 35,
    offset: float = 0.0
) -> np.ndarray:
    """
    Apply adaptive thresholding for binarization.

    Uses local mean intensity to determine threshold, handling
    images with varying illumination.

    Parameters
    ----------
    image : np.ndarray
        Input image (2D).
    block_size : int, optional
        Size of local neighborhood for threshold computation.
        Must be odd. Default is 35.
    offset : float, optional
        Constant subtracted from local mean. Positive values
        result in more foreground. Default is 0.

    Returns
    -------
    np.ndarray
        Binary image (bool).

    Examples
    --------
    >>> # Basic adaptive threshold
    >>> binary = adaptive_threshold(image)

    >>> # More sensitive to foreground
    >>> binary = adaptive_threshold(image, offset=-5)
    """
    if image.ndim != 2:
        raise ValueError(f"adaptive_threshold requires 2D image, got {image.ndim}D")

    # Ensure odd block size
    if block_size % 2 == 0:
        block_size += 1

    img = image.astype(np.float64)

    # Compute local mean using uniform filter
    local_mean = ndimage.uniform_filter(img, size=block_size)

    # Apply threshold
    binary = img > (local_mean - offset)

    return binary
