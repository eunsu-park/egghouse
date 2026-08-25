"""Tests for the per-bin outlier clip of ``poisson_gaussian_noise``."""

import numpy as np

from egghouse.image.noise import poisson_gaussian_noise


def _pair_with_cosmic_rays(seed=0, gain=0.07, sigma=1.0, n_hits=300):
    """Two Poisson-Gaussian frames of one smooth scene plus sparse bright hits."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:256, 0:256]
    electrons = 2_000.0 + 60_000.0 * np.exp(-((xx - 128) ** 2 + (yy - 128) ** 2) / 4_000.0)

    def frame():
        f = gain * rng.poisson(electrons) + rng.normal(0.0, sigma, electrons.shape)
        idx = rng.integers(0, f.size, n_hits)
        f.ravel()[idx] += rng.uniform(200.0, 2_000.0, n_hits)  # cosmic-ray hits in DN
        return f

    return frame(), frame()


def test_per_bin_clip_recovers_gain_with_cosmic_rays():
    a, b = _pair_with_cosmic_rays()
    fit = poisson_gaussian_noise(a, b, bins=20, clip=6.0, clip_per_bin=True)
    assert abs(fit.g - 0.07) / 0.07 < 0.05
    assert fit.r_squared > 0.99


def test_per_bin_clip_keeps_high_intensity_noise_that_global_clip_prunes():
    a, b = _pair_with_cosmic_rays(n_hits=0)
    glob = poisson_gaussian_noise(a, b, bins=20, clip=3.0, clip_per_bin=False)
    per = poisson_gaussian_noise(a, b, bins=20, clip=3.0, clip_per_bin=True)
    # A tight global clip truncates the wide high-intensity difference
    # distribution, biasing the slope low; the per-bin clip does not.
    assert per.g > glob.g
    assert abs(per.g - 0.07) / 0.07 < 0.05
