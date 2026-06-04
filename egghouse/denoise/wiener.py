"""Local Wiener denoising via `scipy.signal.wiener`.

Applies an adaptive local Wiener filter: each pixel is shrunk toward the
local mean by a factor determined by the local variance vs. the noise
variance. Fast, deterministic, and a useful weak baseline when the noise
is approximately additive Gaussian.

Algorithm reference:
  Lim, J.S. (1990). "Two-Dimensional Signal and Image Processing,"
  Prentice Hall. — local Wiener filter, §9.5.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import wiener as _scipy_wiener


def denoise(
    image: np.ndarray,
    *,
    mysize: int = 5,
    noise: float | None = None,
) -> np.ndarray:
    """Denoises a 2-D image via a local Wiener filter.

    Args:
      image: 2-D float array.
      mysize: Side length of the square local window used to estimate the
        local mean and variance. 5 is the scipy default; larger values
        smooth more aggressively.
      noise: Noise variance. When None, scipy estimates it as the mean of
        the local variances across the image.

    Returns:
      Denoised 2-D float64 array of the same shape as `image`.
    """
    arr = image.astype(np.float64)
    return np.asarray(_scipy_wiener(arr, mysize=mysize, noise=noise))


class WienerDenoiser:
    """Parametric form of `denoise()` matching the `Denoiser` protocol."""

    def __init__(self, mysize: int = 5, noise: float | None = None) -> None:
        self.mysize = mysize
        self.noise = noise

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return denoise(image, mysize=self.mysize, noise=self.noise)
