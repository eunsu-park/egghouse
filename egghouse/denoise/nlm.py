"""Non-Local Means denoising via scikit-image.

NLM averages each pixel with a weighted mean of similar patches found
elsewhere in the image. Patch similarity is the noise-robust metric.

Algorithm reference:
  Buades, A., Coll, B., Morel, J.-M. (2005). "A non-local algorithm for
  image denoising," CVPR. — original NLM.
"""

from __future__ import annotations

import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma


def denoise(
    image: np.ndarray,
    *,
    sigma: float | None = None,
    patch_size: int = 7,
    patch_distance: int = 11,
    h_factor: float = 0.8,
    fast_mode: bool = True,
) -> np.ndarray:
    """Denoises a 2-D image with non-local means.

    Args:
      image: 2-D float array.
      sigma: Noise standard deviation. When None, estimated from the input.
      patch_size: Side length of square patches compared for similarity.
        Defaults to 7 (skimage default).
      patch_distance: Maximum search distance for similar patches, in
        pixels. Defaults to 11.
      h_factor: Cut-off in patch-similarity weighting, expressed as a
        fraction of `sigma`. The skimage convention is `h ≈ 0.8 * sigma`
        for fast_mode=True; we surface it as `h_factor` so callers can
        sweep without having to handle sigma manually.
      fast_mode: Use the (much faster) approximate NLM. Defaults to True;
        set False only for paper-quality runs on small images.

    Returns:
      Denoised 2-D float64 array of the same shape as `image`.
    """
    arr = image.astype(np.float64)
    if sigma is None:
        sigma = float(estimate_sigma(arr, average_sigmas=True))
    h = h_factor * sigma
    return denoise_nl_means(
        arr,
        patch_size=patch_size,
        patch_distance=patch_distance,
        h=h,
        sigma=sigma,
        fast_mode=fast_mode,
        preserve_range=True,
    )


class NLMDenoiser:
    """Parametric form of `denoise()` matching the `Denoiser` protocol."""

    def __init__(
        self,
        sigma: float | None = None,
        patch_size: int = 7,
        patch_distance: int = 11,
        h_factor: float = 0.8,
        fast_mode: bool = True,
    ) -> None:
        self.sigma = sigma
        self.patch_size = patch_size
        self.patch_distance = patch_distance
        self.h_factor = h_factor
        self.fast_mode = fast_mode

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return denoise(
            image,
            sigma=self.sigma,
            patch_size=self.patch_size,
            patch_distance=self.patch_distance,
            h_factor=self.h_factor,
            fast_mode=self.fast_mode,
        )
