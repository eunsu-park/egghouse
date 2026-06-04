"""Wavelet denoising (BayesShrink / VisuShrink) via scikit-image.

Wraps `skimage.restoration.denoise_wavelet`, which in turn rests on
`PyWavelets`. Inputs are 2-D float arrays; outputs match input shape and
dtype where possible (float64 internally).

Algorithm reference:
  Chang, S.G., Yu, B., Vetterli, M. (2000). "Adaptive Wavelet Thresholding
  for Image Denoising and Compression," IEEE TIP. — BayesShrink.
  Donoho, D., Johnstone, I. (1994). "Ideal spatial adaptation by wavelet
  shrinkage," Biometrika. — VisuShrink / universal threshold.
"""

from __future__ import annotations

import numpy as np
from skimage.restoration import denoise_wavelet, estimate_sigma


def denoise(
    image: np.ndarray,
    *,
    wavelet: str = "db4",
    method: str = "BayesShrink",
    mode: str = "soft",
    sigma: float | None = None,
) -> np.ndarray:
    """Denoises a 2-D image via wavelet thresholding.

    Args:
      image: 2-D float array.
      wavelet: PyWavelets wavelet name. 'db4' (Daubechies-4) is a robust
        default for natural and astronomical imagery.
      method: 'BayesShrink' (adaptive per-subband threshold; default) or
        'VisuShrink' (universal threshold).
      mode: 'soft' (default; better PSNR) or 'hard' (preserves edges
        sharper, can leave artifacts).
      sigma: Noise standard deviation. When None, estimated from the input
        via `skimage.restoration.estimate_sigma`.

    Returns:
      Denoised 2-D float64 array of the same shape as `image`.
    """
    arr = image.astype(np.float64)
    if sigma is None:
        sigma = float(estimate_sigma(arr, average_sigmas=True))
    return denoise_wavelet(
        arr,
        sigma=sigma,
        wavelet=wavelet,
        method=method,
        mode=mode,
        rescale_sigma=True,
    )


class WaveletDenoiser:
    """Parametric form of `denoise()` matching the `Denoiser` protocol.

    Useful when sweeping hyperparameters via `evaluation.harness.compare()`:
        denoisers = {
            "wavelet-db4": WaveletDenoiser(wavelet="db4"),
            "wavelet-sym8": WaveletDenoiser(wavelet="sym8"),
        }
    """

    def __init__(
        self,
        wavelet: str = "db4",
        method: str = "BayesShrink",
        mode: str = "soft",
        sigma: float | None = None,
    ) -> None:
        self.wavelet = wavelet
        self.method = method
        self.mode = mode
        self.sigma = sigma

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return denoise(
            image,
            wavelet=self.wavelet,
            method=self.method,
            mode=self.mode,
            sigma=self.sigma,
        )
