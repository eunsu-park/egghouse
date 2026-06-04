"""Image-quality metrics for denoiser evaluation.

Pure numpy + scipy. No torch, no skimage dependency, so the same scalars
are computable from notebooks, classical baselines, and (later) DL eval
loops without dragging heavy stacks.

All metrics take 2-D float arrays. FITS frames in `lolipop` are
single-channel by convention; multi-channel callers should compute per
channel and aggregate themselves.

References:
  - Wang, Bovik, Sheikh, Simoncelli (2004). "Image Quality Assessment:
    From Error Visibility to Structural Similarity," IEEE TIP. — single-
    scale SSIM.
  - Wang, Simoncelli, Bovik (2003). "Multi-Scale Structural Similarity
    for Image Quality Assessment," Asilomar. — MS-SSIM weights and
    luminance-only-at-finest-scale convention.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import sobel
from scipy.signal import convolve2d


# Wang et al. 2003 5-scale weights. Sum = 1.0 (canonical).
_MS_SSIM_WEIGHTS = np.array(
    [0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=np.float64
)


def _resolve_data_range(reference: np.ndarray, data_range: float | None) -> float:
    """Returns the dynamic range L. Defaults to reference.max() - reference.min()."""
    if data_range is not None:
        return float(data_range)
    return float(reference.max() - reference.min())


def _gaussian_window(size: int, sigma: float) -> np.ndarray:
    """Returns a normalized 2-D Gaussian window."""
    coords = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    g = np.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g /= g.sum()
    return np.outer(g, g)


def _ssim_components(
    x: np.ndarray,
    y: np.ndarray,
    data_range: float,
    win_size: int = 11,
    sigma: float = 1.5,
    K1: float = 0.01,
    K2: float = 0.03,
) -> tuple[float, float]:
    """Returns (luminance, contrast_structure) — the L and CS pieces of SSIM.

    Implementation follows Wang et al. 2004: Gaussian-weighted local stats
    with K1=0.01, K2=0.03, 11x11 window, sigma=1.5. 'valid' convolution
    so boundary pixels do not bias the result.
    """
    if min(x.shape) < win_size or min(y.shape) < win_size:
        raise ValueError(
            f"input too small for win_size={win_size}: "
            f"shapes are {x.shape}, {y.shape}"
        )
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    win = _gaussian_window(win_size, sigma)

    mu_x = convolve2d(x, win, mode="valid")
    mu_y = convolve2d(y, win, mode="valid")
    mu_xx = mu_x * mu_x
    mu_yy = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_xx = convolve2d(x * x, win, mode="valid") - mu_xx
    sigma_yy = convolve2d(y * y, win, mode="valid") - mu_yy
    sigma_xy = convolve2d(x * y, win, mode="valid") - mu_xy

    luminance_map = (2.0 * mu_xy + C1) / (mu_xx + mu_yy + C1)
    cs_map = (2.0 * sigma_xy + C2) / (sigma_xx + sigma_yy + C2)
    return float(luminance_map.mean()), float(cs_map.mean())


def _downsample_2x(arr: np.ndarray) -> np.ndarray:
    """2x average pooling. Drops odd boundary pixel(s) if any."""
    h, w = arr.shape
    h2, w2 = h - (h % 2), w - (w % 2)
    return arr[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2).mean(axis=(1, 3))


def psnr(
    image: np.ndarray,
    reference: np.ndarray,
    *,
    data_range: float | None = None,
) -> float:
    """Peak signal-to-noise ratio in dB.

    Args:
      image: Denoised / reconstructed image (2-D float).
      reference: Ground-truth or reference image (2-D float).
      data_range: Dynamic range L of the ideal signal. Defaults to
        `reference.max() - reference.min()`. FITS float images have no
        fixed L, so passing an explicit, frame-independent value is
        recommended for cross-frame comparability.

    Returns:
      PSNR in dB. Returns +inf when image == reference exactly.

    Raises:
      ValueError: When the resolved data_range is non-positive.
    """
    dr = _resolve_data_range(reference, data_range)
    if dr <= 0:
        raise ValueError(f"non-positive data_range: {dr}")
    diff = image.astype(np.float64) - reference.astype(np.float64)
    mse = float((diff * diff).mean())
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(dr * dr / mse))


def ssim(
    image: np.ndarray,
    reference: np.ndarray,
    *,
    data_range: float | None = None,
    win_size: int = 11,
) -> float:
    """Single-scale structural similarity index.

    Returns:
      Mean SSIM in [-1, 1] (typically [0, 1] for natural images).
      1.0 when image == reference.
    """
    dr = _resolve_data_range(reference, data_range)
    L, CS = _ssim_components(
        image.astype(np.float64),
        reference.astype(np.float64),
        data_range=dr,
        win_size=win_size,
    )
    return float(L * CS)


def ms_ssim(
    image: np.ndarray,
    reference: np.ndarray,
    *,
    data_range: float | None = None,
    weights: np.ndarray | None = None,
    win_size: int = 11,
) -> float:
    """Multi-scale SSIM (Wang et al. 2003).

    Computes contrast*structure (CS) at each scale and luminance (L)
    only at the coarsest scale, combined as
        L_final ** w_M * prod_j (CS_j ** w_j)
    with the canonical 5-scale weights by default.

    The smallest scale must satisfy the win_size requirement; with the
    default 5 scales and 11x11 window, the input must be at least
    11 * 16 = 176 pixels per side. Smaller inputs raise ValueError.

    Returns:
      MS-SSIM. Typically in [0, 1] for natural images; 1.0 when
      image == reference.
    """
    if weights is None:
        weights = _MS_SSIM_WEIGHTS
    weights = np.asarray(weights, dtype=np.float64)
    n_scales = len(weights)

    dr = _resolve_data_range(reference, data_range)
    required = win_size * (2 ** (n_scales - 1))
    if min(image.shape) < required:
        raise ValueError(
            f"input too small for {n_scales}-scale MS-SSIM with "
            f"win_size={win_size}: min dim is {min(image.shape)}, "
            f"need >= {required}"
        )

    img = image.astype(np.float64)
    ref = reference.astype(np.float64)
    cs_scores = np.empty(n_scales, dtype=np.float64)
    luminance_final = 1.0
    for j in range(n_scales):
        L, cs = _ssim_components(img, ref, data_range=dr, win_size=win_size)
        cs_scores[j] = max(cs, 1e-8)
        if j == n_scales - 1:
            luminance_final = max(L, 1e-8)
        else:
            img = _downsample_2x(img)
            ref = _downsample_2x(ref)

    return float(
        luminance_final ** weights[-1] * np.prod(cs_scores ** weights)
    )


def _gradient_magnitude(arr: np.ndarray) -> np.ndarray:
    """Returns the Sobel gradient magnitude (2-D float64)."""
    a = arr.astype(np.float64)
    gx = sobel(a, axis=1, mode="reflect")
    gy = sobel(a, axis=0, mode="reflect")
    return np.hypot(gx, gy)


def weak_signal_contrast(
    image: np.ndarray,
    reference: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> float:
    """Placeholder weak-signal / edge-preservation score.

    Pearson correlation between the Sobel gradient-magnitude maps of
    `image` and `reference`. The intuition: classical denoisers (and
    naive DL ones) over-smooth weak edges; this metric drops when faint
    structure is washed out, even if PSNR/SSIM look fine. Range [-1, 1];
    higher is better; 1.0 when image == reference.

    Status: PLACEHOLDER. The project's actual weak-signal goal is task-
    specific (e.g., CME front detection AUC, F-corona contrast preserved
    after K/F separation). This first-pass score gives the harness a
    valid column to populate from day one; it will be replaced once the
    noise characterization (C1) and pair/target strategy (C4) decisions
    fix the right downstream task.

    Args:
      image: Denoised / reconstructed image.
      reference: Ground-truth or reference image.
      mask: Optional boolean mask of pixels to include (True = keep).
        Useful for excluding the occulter disc on coronagraph frames.
        When None, all pixels are used.

    Returns:
      Pearson correlation coefficient of gradient magnitudes, as float.
      Returns 1.0 when both gradient maps are constant (degenerate, e.g.
      uniform-intensity images).
    """
    g_img = _gradient_magnitude(image)
    g_ref = _gradient_magnitude(reference)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        g_img = g_img[mask]
        g_ref = g_ref[mask]
    else:
        g_img = g_img.ravel()
        g_ref = g_ref.ravel()

    std_img = float(g_img.std())
    std_ref = float(g_ref.std())
    if std_img == 0.0 and std_ref == 0.0:
        return 1.0
    if std_img == 0.0 or std_ref == 0.0:
        return 0.0
    return float(np.corrcoef(g_img, g_ref)[0, 1])
