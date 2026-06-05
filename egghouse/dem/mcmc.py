"""Metropolis-Hastings MCMC DEM inversion.

Samples, per pixel, the posterior of the Differential Emission Measure by
walking the log10(DEM) per temperature bin with a random-walk
Metropolis-Hastings sampler.  Because the free parameters are ``log10(DEM)``,
the DEM is strictly positive by construction.

The likelihood is Gaussian in the error-folded residuals

    chi2 = sum_c ( (A @ DEM - I)_c / sigma_c )^2 ,

with ``A_{c,t} = K_t(c) * dT_t`` so that ``I = A @ DEM`` (DEM in cm^-5 K^-1).
An optional mild second-difference smoothness prior on ``log10(DEM)`` keeps the
posterior from oscillating in this ill-posed 6-channel -> many-temperature
inversion.  The posterior-mean DEM is returned, and the posterior standard
deviation per temperature is reported in ``info["dem_std"]`` -- quantifying the
uncertainty is the whole point of the MCMC approach.

References
----------
- Kashyap, V. & Drake, J. J. 1998, ApJ 503, 450, DOI 10.1086/305964 --
  Markov-Chain Monte Carlo reconstruction of the emission measure
  distribution with uncertainties.
"""

from typing import Dict, Tuple

import numpy as np


def _dt(temperatures: np.ndarray) -> np.ndarray:
    """Temperature bin widths dT = T ln(10) dlogT (matches dem_nnls)."""
    logt = np.log10(temperatures)
    dlogt = np.gradient(logt)
    return temperatures * np.log(10.0) * dlogt


def _design_matrix(response: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """A with I = A @ DEM. response (n_temps, n_channels) -> A (n_channels, n_temps)."""
    return (response * dt[:, np.newaxis]).T


def _neg_log_posterior(
    log_dem: np.ndarray,
    A: np.ndarray,
    intensity: np.ndarray,
    inv_var: np.ndarray,
    smooth: float,
) -> float:
    """Return chi2/2 + smoothness penalty (i.e. -log posterior up to a const).

    ``inv_var`` is ``1 / sigma^2`` per channel.  The smoothness term penalizes
    the second difference of ``log_dem`` with weight ``smooth``.
    """
    dem = np.power(10.0, log_dem)
    resid = A @ dem - intensity
    chi2 = float(np.sum(resid * resid * inv_var))
    nlp = 0.5 * chi2
    if smooth > 0.0:
        d2 = log_dem[2:] - 2.0 * log_dem[1:-1] + log_dem[:-2]
        nlp += 0.5 * smooth * float(np.sum(d2 * d2))
    return nlp


def _sample_pixel(
    intensity: np.ndarray,
    error: np.ndarray,
    A: np.ndarray,
    *,
    n_steps: int,
    n_burn: int,
    proposal_scale: float,
    n_block: int,
    smooth: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Random-walk Metropolis on log10(DEM) for one pixel.

    Returns ``(dem_mean, dem_std, chi2)`` where the statistics are over the
    post-burn-in DEM samples (in linear space) and ``chi2`` is the data
    chi-squared of the posterior-mean DEM.
    """
    n_temps = A.shape[1]
    inv_var = 1.0 / np.maximum(error, 1e-30) ** 2

    # Initialize log-DEM from a crude flat guess scaled to the data level so
    # the walker starts in a sensible region rather than at zero.
    typ_dem = float(np.mean(intensity) / max(np.mean(A) * n_temps, 1e-300))
    typ_dem = max(typ_dem, 1e-300)
    log_dem = np.full(n_temps, np.log10(typ_dem), dtype=np.float64)
    cur_nlp = _neg_log_posterior(log_dem, A, intensity, inv_var, smooth)

    samples = np.empty((n_steps - n_burn, n_temps), dtype=np.float64)
    out = 0
    for step in range(n_steps):
        # Block update: perturb a random contiguous block of bins each step.
        start = int(rng.integers(0, n_temps))
        size = min(n_block, n_temps - start)
        prop = log_dem.copy()
        prop[start:start + size] += rng.normal(0.0, proposal_scale, size=size)
        prop_nlp = _neg_log_posterior(prop, A, intensity, inv_var, smooth)
        # Metropolis acceptance on -log posterior.
        if prop_nlp < cur_nlp or rng.random() < np.exp(cur_nlp - prop_nlp):
            log_dem = prop
            cur_nlp = prop_nlp
        if step >= n_burn:
            samples[out] = log_dem
            out += 1

    dem_samples = np.power(10.0, samples)
    dem_mean = dem_samples.mean(axis=0)
    dem_std = dem_samples.std(axis=0)
    resid = (A @ dem_mean - intensity) / np.maximum(error, 1e-30)
    chi2 = float(np.sum(resid * resid))
    return dem_mean, dem_std, chi2


def dem_mcmc(
    intensities: np.ndarray,
    errors: np.ndarray,
    response: np.ndarray,
    temperatures: np.ndarray,
    *,
    n_steps: int = 5000,
    n_burn: int = 1000,
    seed: int = 0,
    proposal_scale: float = 0.15,
    n_block: int = 3,
    smooth: float = 1.0,
) -> Tuple[np.ndarray, Dict]:
    """Metropolis-Hastings MCMC DEM inversion (single pixel or batch).

    Parameters
    ----------
    intensities : np.ndarray
        Observed intensities (DN/s/pixel), shape ``(n_channels,)`` or
        ``(n_pixels, n_channels)``.
    errors : np.ndarray
        1-sigma uncertainties, same shape as ``intensities``.
    response : np.ndarray
        Temperature response, shape ``(n_temps, n_channels)`` (same
        convention as :func:`egghouse.dem.dem_nnls`).
    temperatures : np.ndarray
        Temperatures in Kelvin, shape ``(n_temps,)``.
    n_steps : int
        Total Metropolis steps per pixel (including burn-in).
    n_burn : int
        Burn-in steps discarded before collecting samples.
    seed : int
        Base RNG seed for reproducibility (each pixel offset by its index).
    proposal_scale : float
        Std of the Gaussian random-walk proposal in dex (log10 DEM units).
    n_block : int
        Number of contiguous temperature bins perturbed per step.
    smooth : float
        Weight of the second-difference smoothness prior on log10(DEM).
        Set to 0 to disable.

    Returns
    -------
    dem : np.ndarray
        Posterior-mean DEM in cm^-5 K^-1, shape ``(n_temps,)`` or
        ``(n_pixels, n_temps)``.
    info : dict
        ``chi2`` (mean data chi^2), ``chi2_map`` (per pixel), and ``dem_std``
        (posterior std per temperature; same shape as ``dem``).

    References
    ----------
    Kashyap, V. & Drake, J. J. 1998, ApJ 503, 450, DOI 10.1086/305964.

    Notes
    -----
    This is a per-pixel sampler and is slow; batch input is handled by simply
    looping over pixels.  It is intended for small regions or single-pixel
    uncertainty studies, not full-disk maps.
    """
    if n_burn >= n_steps:
        raise ValueError(f"n_burn ({n_burn}) must be < n_steps ({n_steps})")

    squeeze = intensities.ndim == 1
    intensities = np.atleast_2d(intensities).astype(np.float64)
    errors = np.atleast_2d(errors).astype(np.float64)
    n_pixels, n_channels = intensities.shape
    n_temps = len(temperatures)
    if response.shape != (n_temps, n_channels):
        raise ValueError(
            f"Response shape {response.shape} doesn't match "
            f"expected ({n_temps}, {n_channels})"
        )

    dt = _dt(temperatures)
    A = _design_matrix(response, dt)

    dem = np.zeros((n_pixels, n_temps), dtype=np.float64)
    dem_std = np.zeros((n_pixels, n_temps), dtype=np.float64)
    chi2_map = np.zeros(n_pixels, dtype=np.float64)
    for p in range(n_pixels):
        if not np.all(np.isfinite(intensities[p])) or np.all(intensities[p] <= 0):
            continue
        rng = np.random.default_rng(seed + p)
        dem[p], dem_std[p], chi2_map[p] = _sample_pixel(
            intensities[p],
            errors[p],
            A,
            n_steps=n_steps,
            n_burn=n_burn,
            proposal_scale=proposal_scale,
            n_block=n_block,
            smooth=smooth,
            rng=rng,
        )

    info = {
        "chi2": float(np.mean(chi2_map)),
        "chi2_map": chi2_map,
        "dem_std": dem_std,
    }
    if squeeze:
        dem = dem.squeeze()
        info["dem_std"] = dem_std.squeeze()
    return dem, info
