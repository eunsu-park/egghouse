"""Total Variation denoising (Chambolle algorithm) via scikit-image.

TV regularisation minimises the L1 norm of the image gradient subject to a
data-fidelity term. Preserves piecewise-constant regions and sharp edges,
which makes it a useful comparison floor for cartoon-like structure;
however, smooth gradients can develop staircase artefacts.

Algorithm reference:
  Chambolle, A. (2004). "An algorithm for total variation minimization and
  applications," J. Math. Imaging Vis. — projected-gradient TV.
"""

from __future__ import annotations

import numpy as np
from skimage.restoration import denoise_tv_chambolle


def denoise(
    image: np.ndarray,
    *,
    weight: float = 0.1,
    max_num_iter: int = 200,
    eps: float = 2e-4,
) -> np.ndarray:
    """Denoises a 2-D image via Chambolle TV minimisation.

    Args:
      image: 2-D float array.
      weight: Regularisation strength. Larger values smooth more aggressively
        at the cost of detail. The scikit-image default 0.1 is appropriate
        for inputs roughly in [0, 1]; rescale `weight` proportionally for
        arrays in other ranges.
      max_num_iter: Maximum number of Chambolle iterations.
      eps: Convergence tolerance on the relative change in the dual variable.

    Returns:
      Denoised 2-D float64 array of the same shape as `image`.
    """
    arr = image.astype(np.float64)
    return denoise_tv_chambolle(
        arr,
        weight=weight,
        max_num_iter=max_num_iter,
        eps=eps,
    )


class TVDenoiser:
    """Parametric form of `denoise()` matching the `Denoiser` protocol."""

    def __init__(
        self,
        weight: float = 0.1,
        max_num_iter: int = 200,
        eps: float = 2e-4,
    ) -> None:
        self.weight = weight
        self.max_num_iter = max_num_iter
        self.eps = eps

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return denoise(
            image,
            weight=self.weight,
            max_num_iter=self.max_num_iter,
            eps=self.eps,
        )
