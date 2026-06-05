"""
DEM analysis utilities.

Provides functions for full-map DEM processing, error computation,
and derived quantities from DEM solutions.
"""

from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np

from .sites import dem_sites
from .nnls import calibrate_reg_scale, dem_nnls
from .sparse import dem_sparse
from .regularized import dem_regularized
from .plowman import dem_plowman
from .mcmc import dem_mcmc
from .spline import dem_spline
from .gaussian import dem_gaussian

# Solvers that take the common (intensities, errors, response, temperatures)
# signature with their own internal defaults (sites/nnls are handled
# separately because they take extra map-level parameters).
_EXTRA_SOLVERS = {
    "sparse": dem_sparse,
    "regularized": dem_regularized,
    "plowman": dem_plowman,
    "mcmc": dem_mcmc,
    "spline": dem_spline,
    "gaussian": dem_gaussian,
}


def dem_map(
    image_cube: np.ndarray,
    error_cube: np.ndarray,
    response: np.ndarray,
    temperatures: np.ndarray,
    mask: Optional[np.ndarray] = None,
    chunk_size: int = 512,
    max_iter: int = 100,
    tol: float = 1e-3,
    method: str = "sites",
    reg_order: int = 2,
    reg_scale: Optional[float] = None,
    target_chi2: Optional[float] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Compute DEM map for full image cube.

    Processes a multi-wavelength image cube pixel-by-pixel (with chunking
    for memory efficiency) to produce a DEM map.

    Parameters
    ----------
    image_cube : np.ndarray
        Multi-wavelength image cube, shape (height, width, n_channels).
        Values should be in DN/s/pixel.
    error_cube : np.ndarray
        Uncertainty cube, same shape as image_cube.
    response : np.ndarray
        Temperature response matrix, shape (n_temps, n_channels).
    temperatures : np.ndarray
        Temperature array in Kelvin.
    mask : np.ndarray, optional
        Boolean mask, shape (height, width). True = process pixel.
        Default: process all pixels.
    chunk_size : int, optional
        Chunk size for batch processing. Default: 512.
    max_iter : int, optional
        Maximum iterations for SITES. Default: 100.
    tol : float, optional
        Convergence tolerance. Default: 1e-3.
    progress_callback : callable, optional
        Progress callback function(current, total).

    Returns
    -------
    dem_cube : np.ndarray
        DEM map, shape (height, width, n_temps).
        Units: cm^-5 K^-1.
    info : dict
        Processing information:
        - "n_pixels": total pixels processed
        - "n_converged": pixels that converged
        - "mean_iterations": average iterations per pixel
        - "chi2_map": chi-squared map, shape (height, width)

    Notes
    -----
    Memory estimation for 4096x4096 image:
    - Input: ~400 MB (6 channels, float32)
    - Output: ~6.5 GB (100 temperatures, float64)
    - Peak during processing: ~2 GB (with chunk_size=512)

    Examples
    --------
    >>> import numpy as np
    >>> from egghouse.sdo.dem import get_temperature_response, get_default_temperatures
    >>> temps = get_default_temperatures(n_bins=50)
    >>> response = get_temperature_response(temperatures=temps)
    >>> images = np.random.rand(256, 256, 6).astype(np.float32) * 100
    >>> errors = images * 0.1
    >>> dem_cube, info = dem_map(images, errors, response, temps)
    >>> dem_cube.shape
    (256, 256, 50)
    """
    # Input validation
    if image_cube.ndim != 3:
        raise ValueError(f"Expected 3D image cube, got shape {image_cube.shape}")

    height, width, n_channels = image_cube.shape
    n_temps = len(temperatures)

    if error_cube.shape != image_cube.shape:
        raise ValueError(
            f"Error cube shape {error_cube.shape} doesn't match "
            f"image cube shape {image_cube.shape}"
        )

    if response.shape[1] != n_channels:
        raise ValueError(
            f"Response has {response.shape[1]} channels, "
            f"image cube has {n_channels} channels"
        )

    # Create output arrays
    dem_cube = np.zeros((height, width, n_temps), dtype=np.float64)
    chi2_map = np.zeros((height, width), dtype=np.float64)
    iter_map = np.zeros((height, width), dtype=np.int32)

    # Default mask: process all pixels
    if mask is None:
        mask = np.ones((height, width), dtype=bool)

    # Count total pixels to process
    total_pixels = np.sum(mask)
    processed = 0
    converged_count = 0

    # For NNLS: pick one global regularization scale (discrepancy principle)
    # targeting chi^2 ~ n_channels, from a sample of signal-bearing pixels.
    if method == "nnls" and reg_scale is None:
        if target_chi2 is None:
            target_chi2 = float(n_channels)
        flat_img = image_cube.reshape(-1, n_channels)
        flat_err = error_cube.reshape(-1, n_channels)
        flat_m = mask.flatten() & (flat_img.sum(axis=1) > 0)
        valid_idx = np.where(flat_m)[0]
        if valid_idx.size:
            step = max(1, valid_idx.size // 400)
            sample = valid_idx[::step][:400]
            reg_scale = calibrate_reg_scale(
                flat_img[sample], flat_err[sample], response, temperatures,
                target_chi2=target_chi2, reg_order=reg_order,
            )
        else:
            reg_scale = 1e-2

    # Process in chunks
    for y_start in range(0, height, chunk_size):
        y_end = min(y_start + chunk_size, height)

        for x_start in range(0, width, chunk_size):
            x_end = min(x_start + chunk_size, width)

            # Get chunk mask
            chunk_mask = mask[y_start:y_end, x_start:x_end]
            if not np.any(chunk_mask):
                continue

            # Extract chunk data
            chunk_images = image_cube[y_start:y_end, x_start:x_end]
            chunk_errors = error_cube[y_start:y_end, x_start:x_end]

            # Flatten to (n_chunk_pixels, n_channels)
            chunk_h, chunk_w = chunk_mask.shape
            flat_images = chunk_images.reshape(-1, n_channels)
            flat_errors = chunk_errors.reshape(-1, n_channels)
            flat_mask = chunk_mask.flatten()

            # Process only masked pixels
            valid_images = flat_images[flat_mask]
            valid_errors = flat_errors[flat_mask]

            if len(valid_images) == 0:
                continue

            # Run the selected solver on the batch
            if method == "nnls":
                dem_batch, batch_info = dem_nnls(
                    valid_images, valid_errors, response, temperatures,
                    reg_order=reg_order, reg_scale=reg_scale,
                )
            elif method == "sites":
                dem_batch, batch_info = dem_sites(
                    valid_images, valid_errors, response, temperatures,
                    max_iter=max_iter, tol=tol,
                )
            elif method in _EXTRA_SOLVERS:
                dem_batch, batch_info = _EXTRA_SOLVERS[method](
                    valid_images, valid_errors, response, temperatures,
                )
            else:
                valid = ["sites", "nnls", *sorted(_EXTRA_SOLVERS)]
                raise ValueError(f"method must be one of {valid}; got {method!r}")

            # Store results
            dem_flat = np.zeros((chunk_h * chunk_w, n_temps), dtype=np.float64)
            chi2_flat = np.zeros(chunk_h * chunk_w, dtype=np.float64)
            iter_flat = np.zeros(chunk_h * chunk_w, dtype=np.int32)

            dem_flat[flat_mask] = dem_batch
            # per-pixel chi^2 when the solver provides it (nnls), else scalar
            chi2_per = batch_info.get("chi2_map")
            if chi2_per is None:
                chi2_per = np.full(int(np.sum(flat_mask)), batch_info["chi2"])
            chi2_flat[flat_mask] = chi2_per
            iter_flat[flat_mask] = batch_info.get("iterations", 1)

            dem_cube[y_start:y_end, x_start:x_end] = dem_flat.reshape(
                chunk_h, chunk_w, n_temps
            )
            chi2_map[y_start:y_end, x_start:x_end] = chi2_flat.reshape(
                chunk_h, chunk_w
            )
            iter_map[y_start:y_end, x_start:x_end] = iter_flat.reshape(
                chunk_h, chunk_w
            )

            # Update progress
            processed += np.sum(flat_mask)
            if batch_info.get("converged", True):
                converged_count += np.sum(flat_mask)

            if progress_callback is not None:
                progress_callback(processed, total_pixels)

    # Compile info
    info = {
        "n_pixels": int(total_pixels),
        "n_converged": int(converged_count),
        "mean_iterations": float(np.mean(iter_map[mask])) if total_pixels > 0 else 0,
        "chi2_map": chi2_map,
        "iter_map": iter_map,
        "method": method,
        "reg_scale": reg_scale if method == "nnls" else None,
    }

    return dem_cube, info


def compute_dem_errors(
    dem: np.ndarray,
    intensities: np.ndarray,
    errors: np.ndarray,
    response: np.ndarray,
    temperatures: np.ndarray,
    n_monte_carlo: int = 100,
) -> np.ndarray:
    """
    Compute DEM uncertainties using Monte Carlo sampling.

    Perturbs input intensities by their uncertainties and re-runs
    the inversion to estimate DEM errors.

    Parameters
    ----------
    dem : np.ndarray
        DEM solution, shape (n_temps,) or (n_pixels, n_temps).
    intensities : np.ndarray
        Original intensities.
    errors : np.ndarray
        Intensity uncertainties.
    response : np.ndarray
        Temperature response matrix.
    temperatures : np.ndarray
        Temperature array.
    n_monte_carlo : int, optional
        Number of Monte Carlo samples. Default: 100.

    Returns
    -------
    dem_errors : np.ndarray
        DEM 1-sigma uncertainties, same shape as dem.
    """
    dem = np.atleast_2d(dem)
    intensities = np.atleast_2d(intensities)
    errors = np.atleast_2d(errors)

    n_pixels, n_temps = dem.shape
    n_channels = intensities.shape[1]

    # Storage for MC samples
    dem_samples = np.zeros((n_monte_carlo, n_pixels, n_temps), dtype=np.float64)

    for i in range(n_monte_carlo):
        # Perturb intensities
        perturbed = intensities + np.random.randn(*intensities.shape) * errors
        perturbed = np.maximum(perturbed, 0)  # Ensure positivity

        # Run inversion
        dem_mc, _ = dem_sites(perturbed, errors, response, temperatures)
        dem_samples[i] = np.atleast_2d(dem_mc)

    # Compute standard deviation
    dem_errors = np.std(dem_samples, axis=0)

    if n_pixels == 1:
        dem_errors = dem_errors.squeeze()

    return dem_errors


def get_emission_measure(
    dem: np.ndarray,
    temperatures: np.ndarray,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
) -> Union[float, np.ndarray]:
    """
    Compute total Emission Measure from DEM.

    EM = integral(DEM * dT)

    Parameters
    ----------
    dem : np.ndarray
        DEM solution, shape (n_temps,) or (..., n_temps).
    temperatures : np.ndarray
        Temperature array in Kelvin.
    t_min : float, optional
        Minimum temperature for integration. Default: all.
    t_max : float, optional
        Maximum temperature for integration. Default: all.

    Returns
    -------
    em : float or np.ndarray
        Total emission measure in cm^-5.

    Examples
    --------
    >>> import numpy as np
    >>> temps = np.logspace(5.5, 7.5, 50)
    >>> dem = np.exp(-((np.log10(temps) - 6.2) / 0.3) ** 2) * 1e22
    >>> em = get_emission_measure(dem, temps)
    >>> em > 0
    True
    """
    # Temperature bin widths
    logt = np.log10(temperatures)
    dlogt = np.gradient(logt)
    dt = temperatures * np.log(10) * dlogt

    # Select temperature range
    mask = np.ones(len(temperatures), dtype=bool)
    if t_min is not None:
        mask &= temperatures >= t_min
    if t_max is not None:
        mask &= temperatures <= t_max

    # Integrate
    if dem.ndim == 1:
        em = np.sum(dem[mask] * dt[mask])
    else:
        em = np.sum(dem[..., mask] * dt[mask], axis=-1)

    return em


def get_mean_temperature(
    dem: np.ndarray,
    temperatures: np.ndarray,
    weight: str = "dem",
) -> Union[float, np.ndarray]:
    """
    Compute DEM-weighted mean temperature.

    Parameters
    ----------
    dem : np.ndarray
        DEM solution, shape (n_temps,) or (..., n_temps).
    temperatures : np.ndarray
        Temperature array in Kelvin.
    weight : str, optional
        Weighting scheme: "dem" (DEM-weighted) or "em" (EM-weighted).
        Default: "dem".

    Returns
    -------
    t_mean : float or np.ndarray
        Mean temperature in Kelvin.

    Examples
    --------
    >>> import numpy as np
    >>> temps = np.logspace(5.5, 7.5, 50)
    >>> dem = np.exp(-((np.log10(temps) - 6.2) / 0.3) ** 2) * 1e22
    >>> t_mean = get_mean_temperature(dem, temps)
    >>> 1e6 < t_mean < 2e6
    True
    """
    # Temperature bin widths
    logt = np.log10(temperatures)
    dlogt = np.gradient(logt)
    dt = temperatures * np.log(10) * dlogt

    if weight == "dem":
        weights = dem
    elif weight == "em":
        weights = dem * dt
    else:
        raise ValueError(f"Unknown weight: {weight}")

    # Compute weighted mean
    if dem.ndim == 1:
        total_weight = np.sum(weights)
        if total_weight > 0:
            t_mean = np.sum(temperatures * weights) / total_weight
        else:
            t_mean = 0.0
    else:
        total_weight = np.sum(weights, axis=-1, keepdims=True)
        total_weight = np.maximum(total_weight, 1e-30)
        t_mean = np.sum(temperatures * weights, axis=-1) / total_weight.squeeze()

    return t_mean


def get_peak_temperature(
    dem: np.ndarray,
    temperatures: np.ndarray,
) -> Union[float, np.ndarray]:
    """
    Get temperature at DEM peak.

    Parameters
    ----------
    dem : np.ndarray
        DEM solution, shape (n_temps,) or (..., n_temps).
    temperatures : np.ndarray
        Temperature array in Kelvin.

    Returns
    -------
    t_peak : float or np.ndarray
        Peak temperature in Kelvin.
    """
    if dem.ndim == 1:
        idx = np.argmax(dem)
        return temperatures[idx]
    else:
        idx = np.argmax(dem, axis=-1)
        return temperatures[idx]


def dem_to_loci(
    intensities: np.ndarray,
    response: np.ndarray,
    temperatures: np.ndarray,
) -> np.ndarray:
    """
    Compute EM loci curves from intensities.

    The EM loci method provides upper limits on DEM by computing
    EM_loci(T) = I / K(T) for each channel.

    Parameters
    ----------
    intensities : np.ndarray
        Observed intensities, shape (n_channels,).
    response : np.ndarray
        Temperature response, shape (n_temps, n_channels).
    temperatures : np.ndarray
        Temperature array.

    Returns
    -------
    loci : np.ndarray
        EM loci curves, shape (n_temps, n_channels).

    Notes
    -----
    The intersection of EM loci curves indicates the approximate
    DEM distribution. This is useful for quick visualization.
    """
    # Avoid division by zero
    response_safe = np.maximum(response, 1e-30)

    # Compute loci: EM = I / K(T) for each channel
    loci = intensities[np.newaxis, :] / response_safe

    return loci
