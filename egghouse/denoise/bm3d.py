"""BM3D (Block-Matching 3D) denoising via the `bm3d` PyPI package.

BM3D groups similar patches across the image, applies a 3-D collaborative
transform, hard-thresholds, and inverts. Considered the standard-of-care
classical denoiser; common comparison floor in the restoration literature.

Algorithm reference:
  Dabov, K., Foi, A., Katkovnik, V., Egiazarian, K. (2007). "Image
  denoising by sparse 3-D transform-domain collaborative filtering,"
  IEEE TIP. — original BM3D.

Note: the `bm3d` package shadows this module's name when imported under
the same alias. We import the third-party library as `_bm3d_lib` to keep
both available within the project.
"""

from __future__ import annotations

import numpy as np
import bm3d as _bm3d_lib
from skimage.restoration import estimate_sigma


def denoise(
    image: np.ndarray,
    *,
    sigma: float | None = None,
    profile: str = "np",
) -> np.ndarray:
    """Denoises a 2-D image with BM3D.

    Args:
      image: 2-D float array.
      sigma: Noise standard deviation (sigma_psd in BM3D terminology). When
        None, estimated from the input via skimage.
      profile: BM3D profile name. 'np' (default) is the noise-profile
        suited to natural images; 'refilter' applies a refilter pass
        (slower, marginally better PSNR).

    Returns:
      Denoised 2-D float64 array of the same shape as `image`.
    """
    arr = image.astype(np.float64)
    if sigma is None:
        sigma = float(estimate_sigma(arr, average_sigmas=True))
    return np.asarray(_bm3d_lib.bm3d(arr, sigma_psd=sigma, profile=profile))


class BM3DDenoiser:
    """Parametric form of `denoise()` matching the `Denoiser` protocol."""

    def __init__(self, sigma: float | None = None, profile: str = "np") -> None:
        self.sigma = sigma
        self.profile = profile

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return denoise(image, sigma=self.sigma, profile=self.profile)
