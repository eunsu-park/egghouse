"""Anscombe variance-stabilising transform with exact unbiased inverse.

Wraps a Gaussian-noise denoiser so it can be applied to Poisson-noised
imagery, which is the relevant noise regime for photon-limited coronal
imaging (LASCO/SECCHI at high cadence or in faint regions).

Pipeline:
  forward:  z = 2 * sqrt(x + 3/8)         (Anscombe 1948)
  denoise:  z_hat = inner(z)               (any AWGN-targeting denoiser)
  inverse:  x_hat = anscombe_inv(z_hat)    (Makitalo & Foi 2011, exact unbiased)

The Anscombe transform is only meaningful for non-negative inputs
(Poisson counts). On zero-centred data (e.g. post background- or
running-difference outputs, or NRGF/FNRGF z-scores), the
`sqrt(x + 3/8)` operator is undefined on a substantial fraction of
pixels and silently returns garbage. To avoid this trap, the
`AnscombeDenoiser` wrapper detects that regime and bypasses the
Anscombe transform — applying the inner denoiser directly. The guard
uses the fraction of *negative* pixels (~50% on zero-centred data,
~0% on Poisson counts), which catches both wide-tailed zero-centred
outputs (background-subtracted, temporal-difference) and tight-tailed
ones (radially normalised z-scores).

Algorithm references:
  Anscombe, F.J. (1948). "The transformation of Poisson, binomial and
  negative-binomial data," Biometrika. — forward VST.
  Makitalo, M., Foi, A. (2011). "Optimal inversion of the Anscombe
  transformation in low-count Poisson image denoising," IEEE TIP. — closed
  form for the exact unbiased inverse used here.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np

Denoiser = Callable[[np.ndarray], np.ndarray]

# Above this fraction of *negative* pixels (any value < 0), the input
# is treated as zero-centred / not Poisson-count data and the Anscombe
# transform is bypassed. Poisson-count data is ~all non-negative, so
# 10% catches background-subtracted outputs (pixel-min, running-diff,
# NRGF, FNRGF — ~50% negative) without false positives on real
# Poisson data. The earlier "1% below -3/8" threshold was too narrow
# and missed tight-tailed zero-centred outputs (FNRGF).
_NEGATIVE_BYPASS_FRACTION = 0.10


def forward(x: np.ndarray) -> np.ndarray:
    """Applies the Anscombe forward transform, mapping Poisson to ~N(.,1)."""
    return 2.0 * np.sqrt(np.maximum(x, 0.0) + 3.0 / 8.0)


def inverse(z: np.ndarray) -> np.ndarray:
    """Applies the Makitalo-Foi 2011 exact unbiased inverse Anscombe.

    Valid for z above the asymptotic floor 2*sqrt(3/8); below the floor the
    Poisson mean is indistinguishable from zero and we clamp to 0 rather
    than evaluate the rational expansion in an unstable region.
    """
    floor = 2.0 * np.sqrt(3.0 / 8.0)
    z_safe = np.where(z > floor, z, floor)
    inv = (
        0.25 * z_safe ** 2
        + 0.25 * np.sqrt(1.5) / z_safe
        - 11.0 / 8.0 / z_safe ** 2
        + (5.0 / 8.0) * np.sqrt(1.5) / z_safe ** 3
        - 1.0 / 8.0
    )
    return np.where(z > floor, inv, 0.0)


def _has_substantial_negative_mass(image: np.ndarray) -> bool:
    """Returns True if more than `_NEGATIVE_BYPASS_FRACTION` of pixels
    are negative — the signal that this is zero-centred (not Poisson)
    data, regardless of how tight the distribution is."""
    n_neg = float((image < 0).mean())
    return n_neg > _NEGATIVE_BYPASS_FRACTION


def denoise(image: np.ndarray, inner: Denoiser,
            bypass_on_negative: bool = True) -> np.ndarray:
    """Denoises a Poisson-noised image by stabilising → inner → inverting.

    Args:
      image: 2-D float array. Anscombe is meaningful only for non-
        negative inputs; on zero-centred (e.g. background-subtracted)
        inputs the transform produces garbage and this function
        bypasses it (returns the inner denoiser's output directly)
        when more than ~10% of pixels are negative.
      inner: Any Gaussian-noise denoiser conforming to the `Denoiser`
        protocol (e.g., `bm3d.denoise`, `wavelet.denoise`).
      bypass_on_negative: If True (default), bypass the Anscombe
        transform on inputs with substantial negative mass. Set to
        False to force the transform (useful only for diagnostics).

    Returns:
      Denoised 2-D float64 array of the same shape as `image`. Same
      intensity domain as the input (un-stabilised) unless bypassed,
      in which case the output is `inner(image)` directly.
    """
    image = np.asarray(image, dtype=np.float64)
    if bypass_on_negative and _has_substantial_negative_mass(image):
        warnings.warn(
            "AnscombeDenoiser: input has substantial negative mass — "
            "bypassing the Anscombe transform and applying the inner "
            "denoiser directly (Anscombe is only valid for Poisson counts).",
            stacklevel=2,
        )
        return np.asarray(inner(image), dtype=np.float64)
    z = forward(image)
    z_hat = inner(z)
    return inverse(np.asarray(z_hat, dtype=np.float64))


class AnscombeDenoiser:
    """Parametric form of `denoise()` matching the `Denoiser` protocol.

    Composes any Gaussian-target denoiser into a Poisson-target denoiser:
        poisson_bm3d = AnscombeDenoiser(BM3DDenoiser(sigma=1.0))
        out = poisson_bm3d(noisy_counts)

    The wrapped inner denoiser sees data with approximate unit variance, so
    its sigma argument (if applicable) should be ~1.0. On zero-centred
    inputs the wrapper bypasses the Anscombe transform and applies the
    inner denoiser directly — see `denoise()` for the rule.
    """

    def __init__(self, inner: Denoiser,
                 bypass_on_negative: bool = True) -> None:
        self.inner = inner
        self.bypass_on_negative = bypass_on_negative

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return denoise(image, self.inner,
                       bypass_on_negative=self.bypass_on_negative)
