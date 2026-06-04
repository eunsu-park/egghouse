"""Composable numpy-based transforms for low-light FITS preprocessing.

All transforms operate on numpy arrays so they can be used both by classical
CV pipelines and by PyTorch `Dataset` classes. Each transform is a free
function (or a factory returning one); combine via `compose([t1, t2, ...])`.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


Array = np.ndarray
Transform = Callable[[Array], Array]


def compose(transforms: Sequence[Transform]) -> Transform:
    """Composes a sequence of transforms into a single callable.

    Args:
      transforms: Ordered list of transforms applied left to right.

    Returns:
      A new callable that applies each transform in sequence.
    """

    def _apply(image: Array) -> Array:
        for t in transforms:
            image = t(image)
        return image

    return _apply


def to_float32(image: Array) -> Array:
    """Casts an image to native-endian float32 without rescaling values.

    FITS arrays are commonly big-endian int16 (`'>i2'`); converting here
    ensures downstream torch/numpy operations get a clean native dtype.
    """
    return image.astype(np.float32, copy=False)


def nan_to_value(value: float = 0.0) -> Transform:
    """Returns a transform that replaces NaN and +/-Inf with `value`."""

    def _apply(image: Array) -> Array:
        if not np.issubdtype(image.dtype, np.floating):
            return image  # integers cannot hold NaN; nothing to do.
        return np.where(np.isfinite(image), image, value).astype(image.dtype, copy=False)

    return _apply


def percentile_clip(low: float = 0.5, high: float = 99.5) -> Transform:
    """Returns a transform that clips an image to the [low, high] percentile range.

    Useful for suppressing saturated/dead pixels before normalization. Both
    bounds are computed on the input image (per-frame, not dataset-wide).

    Args:
      low: Lower percentile in [0, 100).
      high: Upper percentile in (low, 100].

    Raises:
      ValueError: If `low`/`high` are not in `0 <= low < high <= 100`.
    """
    if not 0.0 <= low < high <= 100.0:
        raise ValueError(f"need 0 <= low < high <= 100; got ({low}, {high})")

    def _apply(image: Array) -> Array:
        lo, hi = np.percentile(image, [low, high])
        return np.clip(image, lo, hi)

    return _apply


def normalize_minmax(eps: float = 1e-8) -> Transform:
    """Returns a transform that scales to [0, 1] via (x - min) / (max - min).

    Operates per frame. `eps` guards against division by zero on constant
    images.
    """

    def _apply(image: Array) -> Array:
        x = image.astype(np.float32, copy=False)
        mn = float(x.min())
        mx = float(x.max())
        return ((x - mn) / max(mx - mn, eps)).astype(np.float32, copy=False)

    return _apply


def normalize_log1p(scale: float = 1.0) -> Transform:
    """Returns a transform applying log1p(scale * (x - min(x))).

    Compresses high dynamic range — a classical pre-step for visualizing the
    faint corona alongside bright streamers and useful as DL model input on
    coronagraph frames.
    """

    def _apply(image: Array) -> Array:
        x = image.astype(np.float32, copy=False)
        shifted = x - x.min()
        return np.log1p(scale * shifted).astype(np.float32, copy=False)

    return _apply


def circular_mask(
    radius_frac: float, fill: float = 0.0, inverse: bool = False
) -> Transform:
    """Returns a transform that fills pixels inside (or outside) a centered circle.

    Args:
      radius_frac: Radius as a fraction of `min(H, W) / 2`. E.g., `0.3` masks
        the inner 30% radius region — useful for blocking the occulter disk.
      fill: Value assigned to masked pixels.
      inverse: If True, fill OUTSIDE the circle (keep the disk, mask the rest).
    """

    def _apply(image: Array) -> Array:
        h, w = image.shape[-2], image.shape[-1]
        cy, cx = h / 2.0, w / 2.0
        r_px = radius_frac * min(h, w) / 2.0
        yy, xx = np.ogrid[:h, :w]
        inside = (yy - cy) ** 2 + (xx - cx) ** 2 <= r_px**2
        mask_to_fill = inside if not inverse else ~inside
        out = image.copy()
        out[..., mask_to_fill] = fill
        return out

    return _apply


def resize(target_size: tuple[int, int], interpolation: str = "bilinear") -> Transform:
    """Returns a transform that resizes to `(H, W)` using OpenCV.

    Args:
      target_size: Tuple `(H, W)` in pixels.
      interpolation: One of `'nearest'`, `'bilinear'`, `'bicubic'`, `'lanczos'`.

    Raises:
      ValueError: If `interpolation` is unknown.
    """
    import cv2  # local import: only this transform pulls OpenCV.

    interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    if interpolation not in interp_map:
        raise ValueError(
            f"unknown interpolation: {interpolation!r}; "
            f"valid: {sorted(interp_map)}"
        )

    h, w = target_size

    def _apply(image: Array) -> Array:
        # cv2.resize takes (W, H) order.
        return cv2.resize(image, (w, h), interpolation=interp_map[interpolation])

    return _apply
