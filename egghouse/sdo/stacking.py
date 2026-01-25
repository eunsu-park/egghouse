"""
Image stacking utilities for SDO/HMI data.

Provides solar rotation-corrected image stacking for time-series analysis
of HMI magnetograms and continuum intensity images.
"""

import warnings
from typing import Callable, List, Literal, Optional, Tuple, Union

import numpy as np
from numpy.fft import fft2, fftshift, ifft2
from scipy.ndimage import shift as ndimage_shift

# Optional dependencies
try:
    from sunpy.map import Map
    from astropy.time import Time
    HAS_SUNPY = True
except ImportError:
    HAS_SUNPY = False


# =============================================================================
# Constants
# =============================================================================

# Default solar rotation period in days (Carrington rotation at ~26° latitude)
SOLAR_ROTATION_PERIOD = 25.38

# Snodgrass (1983) differential rotation coefficients (deg/day)
# omega(B) = A + B*sin^2(B) + C*sin^4(B), where B is heliographic latitude
SNODGRASS_A = 14.71  # Equatorial rotation rate
SNODGRASS_B = -2.39  # First latitude term
SNODGRASS_C = -1.78  # Second latitude term

# HMI standard cadences (seconds)
HMI_CADENCE_45S = 45.0    # Line-of-sight magnetograms
HMI_CADENCE_720S = 720.0  # Vector magnetograms, continuum

# Type aliases
StackingMethod = Literal['list', 'mean', 'median', 'sigma_clipped']
ProgressCallback = Callable[[int, int, str], None]


# =============================================================================
# Helper Functions
# =============================================================================

def _get_hmi_header_value(
    meta: dict,
    keys: List[str],
    default: float
) -> float:
    """
    Get header value from HMI metadata with fallback keys.

    Args:
        meta: FITS header metadata dictionary.
        keys: List of keyword names to try in order.
        default: Default value if no keys found.

    Returns:
        Header value or default.
    """
    for key in keys:
        value = meta.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return default


def _combine_stack(
    stack: np.ndarray,
    method: StackingMethod,
    sigma_lower: float = 3.0,
    sigma_upper: float = 3.0
) -> np.ndarray:
    """
    Combine a stack of images using the specified method.

    Args:
        stack: 3D array of shape (n_images, height, width).
        method: Combining method ('mean', 'median', 'sigma_clipped').
        sigma_lower: Lower sigma threshold for sigma clipping.
        sigma_upper: Upper sigma threshold for sigma clipping.

    Returns:
        Combined 2D image.

    Raises:
        ValueError: If method is 'list' (should not be passed here).
    """
    if method == 'list':
        raise ValueError("Use 'list' method at caller level, not in _combine_stack")

    if method == 'mean':
        return np.nanmean(stack, axis=0)

    elif method == 'median':
        return np.nanmedian(stack, axis=0)

    elif method == 'sigma_clipped':
        # Iterative sigma clipping (3 iterations typically sufficient)
        result = stack.astype(np.float64)
        for _ in range(3):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mean = np.nanmean(result, axis=0)
                std = np.nanstd(result, axis=0)
            lower = mean - sigma_lower * std
            upper = mean + sigma_upper * std
            mask = (result < lower[np.newaxis, :, :]) | (result > upper[np.newaxis, :, :])
            result = np.where(mask, np.nan, result)
        return np.nanmean(result, axis=0)

    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# Differential Rotation
# =============================================================================

def snodgrass_rotation_rate(latitude_deg: float) -> float:
    """
    Calculate solar surface rotation rate at given latitude.

    Uses the Snodgrass (1983) differential rotation formula:
        omega(B) = A + B*sin^2(B) + C*sin^4(B)

    where B is heliographic latitude.

    Args:
        latitude_deg: Heliographic latitude in degrees.

    Returns:
        Angular rotation rate in degrees per day.

    Example:
        >>> # Equator rotates faster than poles
        >>> snodgrass_rotation_rate(0)   # ~14.71 deg/day
        14.71
        >>> snodgrass_rotation_rate(60)  # ~12.0 deg/day (slower)
        12.0075

    Reference:
        Snodgrass, H.B. 1983, ApJ, 270, 288
    """
    sin_lat = np.sin(np.radians(latitude_deg))
    sin2 = sin_lat ** 2
    sin4 = sin2 ** 2
    return SNODGRASS_A + SNODGRASS_B * sin2 + SNODGRASS_C * sin4


def solar_rotation_shift(
    rsun_pixels: float,
    time_offset_hours: float,
    latitude_deg: float = 0.0,
    rotation_period_days: Optional[float] = None,
    use_differential: bool = True
) -> float:
    """
    Calculate pixel shift due to solar rotation.

    Supports both fixed rotation period (backward compatible) and
    latitude-dependent differential rotation using Snodgrass model.

    Args:
        rsun_pixels: Solar radius in pixels.
        time_offset_hours: Time offset from reference in hours.
        latitude_deg: Heliographic latitude in degrees. Defaults to 0 (equator).
        rotation_period_days: Fixed rotation period in days. If provided,
            overrides differential rotation calculation. Defaults to None.
        use_differential: If True, use Snodgrass differential rotation.
            Ignored if rotation_period_days is provided. Defaults to True.

    Returns:
        Pixel shift in x-direction (negative = westward correction).

    Example:
        >>> # Basic usage (equatorial, backward compatible)
        >>> shift = solar_rotation_shift(1600, 1.0)
        >>> print(f"Shift: {shift:.2f} pixels")
        Shift: -2.74 pixels

        >>> # Latitude-dependent (high latitude rotates slower)
        >>> shift_eq = solar_rotation_shift(1600, 1.0, latitude_deg=0)
        >>> shift_hi = solar_rotation_shift(1600, 1.0, latitude_deg=60)
        >>> abs(shift_eq) > abs(shift_hi)
        True

        >>> # Fixed rotation period (Carrington)
        >>> shift_carr = solar_rotation_shift(1600, 1.0, rotation_period_days=25.38)
    """
    if rotation_period_days is not None:
        # Use fixed rotation period (backward compatible)
        rotation_hours = rotation_period_days * 24.0
        omega_rad_per_hour = (2 * np.pi) / rotation_hours
    elif use_differential:
        # Use Snodgrass differential rotation
        omega_deg_per_day = snodgrass_rotation_rate(latitude_deg)
        omega_rad_per_hour = np.radians(omega_deg_per_day) / 24.0
    else:
        # Use Carrington rotation (backward compatible default)
        omega_rad_per_hour = (2 * np.pi) / (SOLAR_ROTATION_PERIOD * 24.0)

    angle_rad = time_offset_hours * omega_rad_per_hour

    # Shift at latitude: rsun * cos(lat) * sin(angle)
    # The cos(lat) factor accounts for reduced linear velocity at higher latitudes
    cos_lat = np.cos(np.radians(latitude_deg))
    return -rsun_pixels * cos_lat * np.sin(angle_rad)


# =============================================================================
# Cadence Detection
# =============================================================================

def detect_cadence_from_maps(
    maps: list,
    fallback_seconds: float = HMI_CADENCE_720S
) -> float:
    """
    Detect observation cadence from a sequence of SunPy Maps.

    Calculates the median time difference between consecutive observations.

    Args:
        maps: List of SunPy Map objects with DATE-OBS or T_REC metadata.
        fallback_seconds: Cadence to use if detection fails. Defaults to 720s.

    Returns:
        Median cadence in seconds.

    Raises:
        ValueError: If maps list has fewer than 2 elements.

    Example:
        >>> from sunpy.map import Map
        >>> maps = Map(sorted(fits_files))
        >>> cadence = detect_cadence_from_maps(maps)
        >>> print(f"Cadence: {cadence} seconds")
    """
    if len(maps) < 2:
        raise ValueError("Need at least 2 maps to detect cadence")

    times = []
    for m in maps:
        meta = m.meta
        # Try common time keywords in order of preference
        time_str = meta.get('T_REC') or meta.get('DATE-OBS') or meta.get('DATE_OBS')
        if time_str:
            try:
                times.append(Time(time_str))
            except Exception:
                continue

    if len(times) < 2:
        return fallback_seconds

    # Calculate time differences
    times = sorted(times)
    deltas = [(times[i + 1] - times[i]).sec for i in range(len(times) - 1)]

    # Use median to be robust against gaps
    return float(np.median(deltas))


# =============================================================================
# Cross-Correlation Alignment
# =============================================================================

def cross_correlate_shift(
    reference: np.ndarray,
    target: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate sub-pixel shift between two images using phase correlation.

    Uses FFT-based cross-correlation for robust shift detection.
    Useful when rotation-based shifts are insufficient (e.g., pointing jitter).
    Sub-pixel precision is achieved via parabolic fitting around the peak.

    Args:
        reference: Reference image (2D array).
        target: Image to align to reference (2D array, same shape).

    Returns:
        Tuple of (y_shift, x_shift) in pixels needed to align target to reference.

    Example:
        >>> dy, dx = cross_correlate_shift(ref_image, target_image)
        >>> aligned = ndimage_shift(target_image, (dy, dx))
    """
    if reference.shape != target.shape:
        raise ValueError("Reference and target must have the same shape")

    # Compute cross-power spectrum
    f_ref = fft2(reference.astype(np.float64))
    f_target = fft2(target.astype(np.float64))

    # Normalized cross-power spectrum
    cross_power = (f_ref * np.conj(f_target))
    magnitude = np.abs(cross_power)
    magnitude[magnitude == 0] = 1e-10  # Avoid division by zero
    cross_power = cross_power / magnitude

    # Inverse FFT to get correlation
    correlation = np.abs(ifft2(cross_power))
    correlation = fftshift(correlation)

    # Find peak (integer pixel)
    peak_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
    center = np.array(correlation.shape) // 2
    shift_int = np.array(peak_idx) - center

    # Sub-pixel refinement using parabolic fit around peak
    y_peak, x_peak = peak_idx
    h, w = correlation.shape

    # Ensure we have neighbors for fitting
    if 1 <= y_peak < h - 1 and 1 <= x_peak < w - 1:
        # Parabolic fit in y
        y_vals = correlation[y_peak - 1:y_peak + 2, x_peak]
        dy_sub = 0.5 * (y_vals[0] - y_vals[2]) / (y_vals[0] - 2 * y_vals[1] + y_vals[2] + 1e-10)

        # Parabolic fit in x
        x_vals = correlation[y_peak, x_peak - 1:x_peak + 2]
        dx_sub = 0.5 * (x_vals[0] - x_vals[2]) / (x_vals[0] - 2 * x_vals[1] + x_vals[2] + 1e-10)

        return float(shift_int[0] + dy_sub), float(shift_int[1] + dx_sub)

    return float(shift_int[0]), float(shift_int[1])


# =============================================================================
# Streaming Accumulator
# =============================================================================

class StreamingStackAccumulator:
    """
    Memory-efficient accumulator for large stacking operations.

    Uses Welford's online algorithm for mean/variance calculation,
    allowing processing of arbitrarily large image sequences without
    loading all images into memory.

    Args:
        shape: Expected image shape (height, width).

    Example:
        >>> acc = StreamingStackAccumulator((512, 512))
        >>> for fits_file in large_file_list:
        ...     img = load_and_process(fits_file)
        ...     acc.add(img)
        >>> mean_image = acc.get_mean()
        >>> std_image = acc.get_std()
    """

    def __init__(self, shape: Tuple[int, int]) -> None:
        """
        Initialize the accumulator.

        Args:
            shape: Expected image shape (height, width).
        """
        self.shape = shape
        self.count = 0
        self._mean = np.zeros(shape, dtype=np.float64)
        self._m2 = np.zeros(shape, dtype=np.float64)  # For variance (Welford's)

    def add(self, image: np.ndarray) -> None:
        """
        Add an image to the accumulator using Welford's algorithm.

        Args:
            image: 2D image array with shape matching self.shape.

        Raises:
            ValueError: If image shape doesn't match expected shape.
        """
        if image.shape != self.shape:
            raise ValueError(f"Expected shape {self.shape}, got {image.shape}")

        img = image.astype(np.float64)
        self.count += 1
        delta = img - self._mean
        self._mean += delta / self.count
        delta2 = img - self._mean
        self._m2 += delta * delta2

    def get_mean(self) -> np.ndarray:
        """
        Get the current running mean.

        Returns:
            Mean image as 2D array.
        """
        return self._mean.copy()

    def get_variance(self) -> np.ndarray:
        """
        Get the current sample variance.

        Returns:
            Variance image as 2D array.
        """
        if self.count < 2:
            return np.zeros(self.shape, dtype=np.float64)
        return self._m2 / (self.count - 1)

    def get_std(self) -> np.ndarray:
        """
        Get the current sample standard deviation.

        Returns:
            Standard deviation image as 2D array.
        """
        return np.sqrt(self.get_variance())

    def reset(self) -> None:
        """Reset the accumulator to initial state."""
        self.count = 0
        self._mean.fill(0)
        self._m2.fill(0)


# =============================================================================
# Main Stacking Functions
# =============================================================================

def stack_with_rotation_correction(
    images: List[np.ndarray],
    rsun_pixels: float,
    cadence_hours: float,
    crop_center: Optional[Tuple[int, int]] = None,
    crop_size: int = 512,
    interpolation_order: int = 3,
    latitude_deg: float = 0.0,
    method: StackingMethod = 'list',
    sigma_lower: float = 3.0,
    sigma_upper: float = 3.0,
    refine_alignment: bool = False,
    progress_callback: Optional[ProgressCallback] = None
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Stack images with solar rotation correction.

    Applies sub-pixel shifts to correct for solar rotation, allowing
    co-alignment of features across time.

    Args:
        images: List of 2D image arrays (same shape).
        rsun_pixels: Solar radius in pixels.
        cadence_hours: Time between consecutive images in hours.
        crop_center: Center (y, x) for cropping. None for image center.
        crop_size: Size of square crop region. Defaults to 512.
        interpolation_order: Spline interpolation order (0-5). Defaults to 3.
        latitude_deg: Heliographic latitude for differential rotation.
            Defaults to 0 (equator).
        method: Stacking method. Options:
            - 'list': Return 3D array of aligned images (default, backward compatible)
            - 'mean': Return mean-combined 2D image
            - 'median': Return median-combined 2D image
            - 'sigma_clipped': Return sigma-clipped mean 2D image
        sigma_lower: Lower sigma threshold for clipping. Defaults to 3.0.
        sigma_upper: Upper sigma threshold for clipping. Defaults to 3.0.
        refine_alignment: If True, use cross-correlation to refine shifts.
            Defaults to False.
        progress_callback: Optional callback(current, total, message) for progress.

    Returns:
        If method='list': 3D array with shape (n_images, crop_size, crop_size).
        Otherwise: Combined 2D array with shape (crop_size, crop_size).

    Example:
        >>> images = [fits.getdata(f) for f in fits_files]
        >>> # Basic stacking (returns aligned stack)
        >>> stacked = stack_with_rotation_correction(
        ...     images, rsun_pixels=1600, cadence_hours=0.2
        ... )
        >>> # Mean combined image
        >>> combined = stack_with_rotation_correction(
        ...     images, rsun_pixels=1600, cadence_hours=0.2, method='mean'
        ... )
    """
    if not images:
        raise ValueError("Empty image list")

    n_images = len(images)
    ref_idx = n_images // 2  # Use middle image as reference

    # Determine crop region
    h, w = images[0].shape
    if crop_center is None:
        cy, cx = h // 2, w // 2
    else:
        cy, cx = crop_center

    half = crop_size // 2
    y_slice = slice(cy - half, cy + half)
    x_slice = slice(cx - half, cx + half)

    # Stack with rotation correction
    result = np.zeros((n_images, crop_size, crop_size), dtype=np.float64)
    reference_cropped = None

    for idx, img in enumerate(images):
        if progress_callback:
            progress_callback(idx + 1, n_images, f"Processing image {idx + 1}/{n_images}")

        # Calculate time offset from reference
        time_offset = (idx - ref_idx) * cadence_hours

        # Calculate rotation shift
        pixel_shift = solar_rotation_shift(
            rsun_pixels, time_offset,
            latitude_deg=latitude_deg,
            use_differential=True
        )

        # Crop first, then shift (more efficient)
        cropped = img[y_slice, x_slice].astype(np.float64)

        # Apply sub-pixel shift in x-direction only
        shifted = ndimage_shift(cropped, (0.0, pixel_shift), order=interpolation_order)

        # Optional: refine with cross-correlation
        if refine_alignment and reference_cropped is not None:
            dy, dx = cross_correlate_shift(reference_cropped, shifted)
            if abs(dy) < 5 and abs(dx) < 5:  # Sanity check
                shifted = ndimage_shift(shifted, (dy, dx), order=interpolation_order)

        result[idx] = shifted

        # Store reference for cross-correlation
        if idx == ref_idx:
            reference_cropped = shifted.copy()

    # Return based on method
    if method == 'list':
        return result
    else:
        return _combine_stack(result, method, sigma_lower, sigma_upper)


# =============================================================================
# Stacking Class
# =============================================================================

class Stacking:
    """
    Solar rotation-corrected image stacking for HMI data.

    This class provides a convenient interface for stacking multiple HMI images
    while compensating for solar rotation. Useful for noise reduction and
    feature tracking in magnetograms.

    Args:
        nb_stack: Number of images to stack. Defaults to 21.
        solar_rot_period: Solar rotation period in days. If None, uses
            differential rotation. Defaults to None.
        crop_size: Size of cropped region. Defaults to 512.
        cadence_seconds: Observation cadence in seconds. None for auto-detect.
            Defaults to None.
        latitude_deg: Heliographic latitude for differential rotation.
            Defaults to 0 (equator).
        method: Stacking method ('list', 'mean', 'median', 'sigma_clipped').
            Defaults to 'list'.
        sigma_lower: Lower sigma threshold for clipping. Defaults to 3.0.
        sigma_upper: Upper sigma threshold for clipping. Defaults to 3.0.

    Example:
        >>> # Basic usage
        >>> stacker = Stacking(nb_stack=21)
        >>> result = stacker.run(fits_files)

        >>> # With mean combining
        >>> stacker = Stacking(nb_stack=21, method='mean')
        >>> mean_image = stacker.run(fits_files)

        >>> # High-latitude region with specific cadence
        >>> stacker = Stacking(
        ...     nb_stack=21,
        ...     latitude_deg=45,
        ...     cadence_seconds=720
        ... )
    """

    def __init__(
        self,
        nb_stack: int = 21,
        solar_rot_period: Optional[float] = None,
        crop_size: int = 512,
        cadence_seconds: Optional[float] = None,
        latitude_deg: float = 0.0,
        method: StackingMethod = 'list',
        sigma_lower: float = 3.0,
        sigma_upper: float = 3.0
    ) -> None:
        """Initialize the Stacking processor."""
        self.nb_stack = nb_stack
        self.solar_rot_period = solar_rot_period
        self.crop_size = crop_size
        self.cadence_seconds = cadence_seconds
        self.latitude_deg = latitude_deg
        self.method = method
        self.sigma_lower = sigma_lower
        self.sigma_upper = sigma_upper

    def run(
        self,
        fits_files: List[str],
        progress_callback: Optional[ProgressCallback] = None
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Run stacking on a list of FITS files.

        Args:
            fits_files: List of paths to HMI FITS files.
            progress_callback: Optional callback(current, total, message) for progress.

        Returns:
            If method='list': List of rotation-corrected, cropped image arrays.
            Otherwise: Combined 2D array.

        Raises:
            ImportError: If sunpy is not installed.
        """
        if not HAS_SUNPY:
            raise ImportError(
                "sunpy is required for Stacking.run(). "
                "Install with: pip install sunpy"
            )

        # Suppress warnings during processing
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')

            # Load and sort maps by time
            maps = Map(sorted(fits_files))
            if not isinstance(maps, list):
                maps = [maps]

            # Sort by observation time
            times = []
            for m in maps:
                time_str = m.meta.get('T_REC') or m.meta.get('DATE-OBS')
                if time_str:
                    times.append(Time(time_str))
                else:
                    times.append(Time.now())  # Fallback
            sorted_pairs = sorted(zip(times, maps))
            _, maps = zip(*sorted_pairs)
            maps = list(maps)

            # Detect cadence if not specified
            if self.cadence_seconds is None:
                cadence_sec = detect_cadence_from_maps(maps, HMI_CADENCE_720S)
            else:
                cadence_sec = self.cadence_seconds
            cadence_hours = cadence_sec / 3600.0

            # Process images
            results = []
            half = self.crop_size // 2
            center_idx = self.nb_stack // 2
            n_to_process = min(self.nb_stack, len(maps))

            for idx in range(n_to_process):
                if progress_callback:
                    progress_callback(idx + 1, n_to_process, f"Processing {idx + 1}/{n_to_process}")

                m = maps[idx]
                meta = m.meta
                data = m.data

                # Crop center region
                h, w = data.shape
                cy, cx = h // 2, w // 2
                cropped = data[cy - half:cy + half, cx - half:cx + half]

                # Calculate solar radius in pixels
                rsun_arcsec = _get_hmi_header_value(
                    meta, ['RSUN_OBS', 'R_SUN'], default=960.0
                )
                cdelt = _get_hmi_header_value(
                    meta, ['CDELT1', 'CDELT2'], default=0.5
                )
                rsun = rsun_arcsec / cdelt

                # Time offset from center
                shift_idx = idx - center_idx
                time_offset = shift_idx * cadence_hours

                # Calculate rotation shift
                pixel_shift = solar_rotation_shift(
                    rsun, time_offset,
                    latitude_deg=self.latitude_deg,
                    rotation_period_days=self.solar_rot_period,
                    use_differential=(self.solar_rot_period is None)
                )

                # Apply shift
                shifted = ndimage_shift(
                    cropped.astype(np.float64),
                    (0.0, pixel_shift),
                    order=3
                )
                results.append(shifted)

            # Convert to array and combine if needed
            result_array = np.array(results)

            if self.method == 'list':
                return results
            else:
                return _combine_stack(result_array, self.method, self.sigma_lower, self.sigma_upper)

    def __call__(
        self,
        fits_files: List[str],
        progress_callback: Optional[ProgressCallback] = None
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Alias for run()."""
        return self.run(fits_files, progress_callback)
