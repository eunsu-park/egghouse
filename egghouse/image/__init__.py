"""
Image manipulation and processing utilities.

This package provides functions for resizing, rotating, scaling, filtering,
and masking images. Designed for scientific image data (FITS, solar observations,
etc.) while preserving data types and ranges.

Modules:
    core: Basic image transformations (resize, rotate, bytescale).
    masking: Circular and annular mask generation for solar disk analysis.
    spatial: Image padding, cropping, flipping, and rolling utilities.
    augment: Paired random crop / flip / rotation for N2N-style training.
    filters: Gaussian, median, and edge detection filters.
    stats: Normalization, histogram operations, and image statistics.
    colorize: Apply a 256-entry RGB LUT to an 8-bit grayscale image.

Example:
    >>> from egghouse.image import resize_image, circle_mask, bytescale_image
    >>> # Resize image to 512x512
    >>> resized = resize_image(img, (512, 512))
    >>> # Create solar disk mask
    >>> disk = circle_mask(4096, radius=1600)
    >>> # Scale for visualization
    >>> scaled = bytescale_image(data)

    >>> # Apply filters
    >>> from egghouse.image import gaussian_smooth, median_denoise
    >>> smoothed = gaussian_smooth(image, sigma=1.5)
    >>> denoised = median_denoise(image, size=3)

    >>> # Get statistics
    >>> from egghouse.image import get_image_stats, normalize_image
    >>> stats = get_image_stats(image)
    >>> normalized = normalize_image(image)
"""

from .core import (
    resize_image,
    rotate_image,
    bytescale_image,
    # Aliases
    resize,
    rotate,
    bytescale,
)

from .masking import (
    circle_mask,
    annulus_mask,
)

from .spatial import (
    pad_image,
    crop_or_pad,
    flip_image,
    roll_image,
    bin_ndarray,
    # Alias
    pad,
)

from .filters import (
    gaussian_smooth,
    median_denoise,
    laplacian_edge,
    sobel_edge,
    unsharp_mask,
)

from .stats import (
    normalize_image,
    get_image_stats,
    histogram_equalization,
    percentile_scale,
    find_disk_center,
    adaptive_threshold,
)

from .noise import (
    mad,
    robust_sigma,
    gaussian_core_sigma,
    photon_transfer_fit,
    theil_sen_fit,
    poisson_gaussian_noise,
    PoissonGaussianNoise,
)

from .metrics import (
    psnr,
    ssim,
    ms_ssim,
    weak_signal_contrast,
    pearson_corr,
    db_ratio,
)

from .augment import (
    paired_random_crop,
    paired_flip_rot,
)

from .transforms import (
    compose,
    to_float32,
    nan_to_value,
    percentile_clip,
    normalize_minmax,
    normalize_log1p,
    circular_mask,
)

from .colorize import (
    apply_colormap,
    lut_from_matplotlib,
)

__all__ = [
    # Core transformations
    'resize_image',
    'rotate_image',
    'bytescale_image',
    # Masking
    'circle_mask',
    'annulus_mask',
    # Spatial utilities
    'pad_image',
    'crop_or_pad',
    'flip_image',
    'roll_image',
    'bin_ndarray',
    # Filters
    'gaussian_smooth',
    'median_denoise',
    'laplacian_edge',
    'sobel_edge',
    'unsharp_mask',
    # Statistics
    'normalize_image',
    'get_image_stats',
    'histogram_equalization',
    'percentile_scale',
    'find_disk_center',
    'adaptive_threshold',
    # Noise-scale estimation
    'mad',
    'robust_sigma',
    'gaussian_core_sigma',
    'photon_transfer_fit',
    'theil_sen_fit',
    'poisson_gaussian_noise',
    'PoissonGaussianNoise',
    # Image-quality metrics
    'psnr',
    'ssim',
    'ms_ssim',
    'weak_signal_contrast',
    'pearson_corr',
    'db_ratio',
    # Paired augmentation
    'paired_random_crop',
    'paired_flip_rot',
    # Composable transforms
    'compose',
    'to_float32',
    'nan_to_value',
    'percentile_clip',
    'normalize_minmax',
    'normalize_log1p',
    'circular_mask',
    # Colorization
    'apply_colormap',
    'lut_from_matplotlib',
    # Convenience aliases
    'resize',
    'rotate',
    'bytescale',
    'pad',
]
