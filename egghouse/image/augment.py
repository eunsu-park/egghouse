"""Paired image augmentations for self-supervised / Noise2Noise training.

Apply the *same* random spatial transform to two arrays at once — an
``(input, target)`` pair — so the pixel correspondence between them is
preserved. Randomness is driven by a caller-supplied
``numpy.random.Generator`` so results are reproducible.
"""

from __future__ import annotations

import numpy as np


def paired_random_crop(
    a: np.ndarray,
    b: np.ndarray,
    patch: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop a `patch` x `patch` window at the same location from `a` and `b`.

    Args:
      a: First 2-D array.
      b: Second 2-D array, same shape as `a`.
      patch: Side length of the square crop.
      rng: NumPy random generator supplying the crop offset.

    Returns:
      `(a_crop, b_crop)` — views cropped at the same random offset.

    Raises:
      ValueError: If either spatial dimension is smaller than `patch`.
    """
    h, w = a.shape
    if h < patch or w < patch:
        raise ValueError(f"frame {a.shape} smaller than patch {patch}")
    y = int(rng.integers(0, h - patch + 1))
    x = int(rng.integers(0, w - patch + 1))
    return a[y:y + patch, x:x + patch], b[y:y + patch, x:x + patch]


def paired_flip_rot(
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same random flip + 90-degree rotation to `a` and `b`.

    Independently draws a horizontal flip (p=0.5), a vertical flip
    (p=0.5), and a rotation by ``k * 90`` degrees (``k`` uniform in
    0..3), applying each identically to both arrays.

    Args:
      a: First 2-D array.
      b: Second 2-D array, same shape as `a`.
      rng: NumPy random generator supplying the flip/rotation choices.

    Returns:
      `(a_aug, b_aug)` — contiguous copies with the same transform applied.
    """
    if rng.random() < 0.5:
        a = a[:, ::-1].copy()
        b = b[:, ::-1].copy()
    if rng.random() < 0.5:
        a = a[::-1, :].copy()
        b = b[::-1, :].copy()
    k = int(rng.integers(0, 4))
    if k:
        a = np.rot90(a, k).copy()
        b = np.rot90(b, k).copy()
    return a, b
