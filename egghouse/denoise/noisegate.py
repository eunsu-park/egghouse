"""DeForest (2017) noise-gating — coherence-preserving shot-noise removal.

Separates *coherent* image structure from *incoherent* noise with locally
adaptive filters ("noise gates") in the Fourier domain. Overlapping
neighbourhoods are Fourier-transformed; the per-coefficient noise level is
estimated from a low percentile of the coefficient amplitudes over many
neighbourhoods (DeForest 2017, Eq. 7); coefficients below ``gamma`` times that
level are zeroed (a hard gate); the inverse transform is apodised and
overlap-added (weighted overlap-add).

Unlike per-pixel / per-patch spatial denoisers, noise-gating **preserves
coherent structure** (point sources, fronts, fine background detail) and
removes only the incoherent noise floor. It is strongest on image
*sequences*, where the time axis adds coherence to discriminate against; it
works on 2-D single images and 3-D ``(t, y, x)`` sequences alike (the
transform spans whichever axes the input has).

The estimate is data-driven: for each Fourier coefficient the noise amplitude
is the ``beta_percentile``-th percentile of ``|F|`` over the sampled tiles
(noise-dominated tiles set the floor), so colored noise and the instrument MTF
are handled without an explicit noise model.

Algorithm reference:
  DeForest, C. E. (2017). "Noise-gating to Clean Astrophysical Image Data,"
  ApJ 838, 155 (arXiv:1703.06228).
"""

from __future__ import annotations

import itertools

import numpy as np


def _hann_nd(shape: tuple[int, ...]) -> np.ndarray:
    """Separable, strictly-positive Hann apodisation window of `shape`.

    Uses ``hanning(n + 2)[1:-1]`` so the window never hits zero at a tile
    edge (every pixel keeps a positive overlap-add weight).
    """
    w = np.ones(shape, dtype=np.float64)
    for ax, n in enumerate(shape):
        h = np.hanning(n + 2)[1:-1] if n > 1 else np.ones(1)
        sh = [1] * len(shape)
        sh[ax] = n
        w = w * h.reshape(sh)
    return w


def _starts(length: int, width: int, stride: int) -> list[int]:
    """Tile start indices along one axis, always covering the final edge."""
    if length <= width:
        return [0]
    s = list(range(0, length - width + 1, stride))
    if s[-1] != length - width:
        s.append(length - width)
    return s


def noise_gate(
    data: np.ndarray,
    *,
    width: int = 12,
    stride: int | None = None,
    gamma: float = 2.0,
    beta_percentile: float = 50.0,
    max_beta_tiles: int = 4000,
    apodize: bool = True,
    keep_dc: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """Noise-gates a 2-D image or a 3-D ``(t, y, x)`` sequence.

    Args:
      data: 2-D image or 3-D sequence (float).
      width: Fourier tile side length. A scalar makes a square/cubic tile;
        a per-axis sequence (len == ``data.ndim``) sizes each axis, e.g.
        ``(5, 12, 12)`` for a short time window with a larger spatial one.
      stride: Tile step (scalar or per-axis); defaults to ``width // 2``
        per axis (50 % overlap).
      gamma: Gate level. A coefficient is kept iff ``|F| > gamma * beta``
        for its frequency; larger removes more.
      beta_percentile: Percentile of ``|F|`` over tiles used as the
        per-coefficient noise level ``beta`` (DeForest Eq. 7).
      max_beta_tiles: Cap on tiles sampled to estimate ``beta`` (all tiles
        are still gated; only the noise estimate is subsampled).
      apodize: Apply a Hann window (analysis + synthesis); recommended.
      keep_dc: Never gate the zero-frequency (local-mean) coefficient.
      seed: RNG seed for the tile subsample (reproducible ``beta``).

    Returns:
      Denoised array, same shape and dtype-float64 as `data`.
    """
    arr0 = np.asarray(data, dtype=np.float64)
    ndim = arr0.ndim
    if ndim not in (2, 3):
        raise ValueError("noise_gate supports 2-D images or 3-D (t, y, x) sequences")
    # Per-axis tile widths / strides (a scalar broadcasts to every axis),
    # each clamped to its axis length. A per-axis width lets a sequence use a
    # short time window with a larger spatial one, e.g. width=(5, 12, 12).
    if np.isscalar(width):
        wts = [min(int(width), arr0.shape[ax]) for ax in range(ndim)]
    else:
        width = list(width)
        if len(width) != ndim:
            raise ValueError("width sequence must match data.ndim")
        wts = [min(int(width[ax]), arr0.shape[ax]) for ax in range(ndim)]
    if min(wts) < 2:
        return arr0.copy()
    if stride is None:
        sts = [max(1, w // 2) for w in wts]
    elif np.isscalar(stride):
        sts = [max(1, int(stride))] * ndim
    else:
        stride = list(stride)
        if len(stride) != ndim:
            raise ValueError("stride sequence must match data.ndim")
        sts = [max(1, int(s)) for s in stride]
    tshape = tuple(wts)
    win = _hann_nd(tshape) if apodize else np.ones(tshape)
    win2 = win * win

    # Symmetric-pad by one tile (per axis) so every real pixel is interior
    # with full overlap coverage; otherwise edge pixels seen only by a
    # tapered tile corner divide by a near-zero weight and amplify ringing.
    arr = np.pad(arr0, [(wts[ax], wts[ax]) for ax in range(ndim)], mode="symmetric")

    starts = [_starts(arr.shape[ax], wts[ax], sts[ax]) for ax in range(ndim)]
    all_starts = list(itertools.product(*starts))

    def _slice(st_idx):
        return tuple(slice(s, s + wts[ax]) for ax, s in enumerate(st_idx))

    # Pass 1 — per-coefficient noise level beta from a tile subsample.
    rng = np.random.default_rng(seed)
    n = len(all_starts)
    if n > max_beta_tiles:
        sample = rng.choice(n, size=max_beta_tiles, replace=False)
    else:
        sample = np.arange(n)
    amps = np.empty((len(sample),) + tshape, dtype=np.float64)
    for i, si in enumerate(sample):
        amps[i] = np.abs(np.fft.fftn(arr[_slice(all_starts[si])] * win))
    beta = np.percentile(amps, beta_percentile, axis=0)
    thresh = gamma * beta
    if keep_dc:
        thresh[(0,) * ndim] = 0.0
    del amps

    # Pass 2 — gate every tile, weighted overlap-add.
    out = np.zeros_like(arr)
    wsum = np.zeros_like(arr)
    for st_idx in all_starts:
        sl = _slice(st_idx)
        f = np.fft.fftn(arr[sl] * win)
        f[np.abs(f) < thresh] = 0.0
        rec = np.fft.ifftn(f).real * win
        out[sl] += rec
        wsum[sl] += win2
    nz = wsum > 0
    out[nz] /= wsum[nz]
    out[~nz] = arr[~nz]
    crop = tuple(slice(wts[ax], wts[ax] + arr0.shape[ax]) for ax in range(ndim))
    return out[crop]


def denoise(
    image: np.ndarray,
    *,
    width: int = 12,
    stride: int | None = None,
    gamma: float = 2.0,
    beta_percentile: float = 50.0,
    max_beta_tiles: int = 4000,
    apodize: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """Noise-gates a single 2-D image (the `Denoiser` protocol entry point).

    For an image *sequence* use `noise_gate_sequence`, which exploits the
    time axis and is where noise-gating is strongest.
    """
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("denoise expects a 2-D image; use noise_gate_sequence for 3-D")
    return noise_gate(
        arr, width=width, stride=stride, gamma=gamma,
        beta_percentile=beta_percentile, max_beta_tiles=max_beta_tiles,
        apodize=apodize, seed=seed,
    )


def noise_gate_sequence(cube: np.ndarray, **kwargs) -> np.ndarray:
    """Noise-gates a 3-D ``(t, y, x)`` image sequence (spatiotemporal gate)."""
    arr = np.asarray(cube, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError("noise_gate_sequence expects a 3-D (t, y, x) cube")
    return noise_gate(arr, **kwargs)


class NoiseGateDenoiser:
    """Parametric 2-D form of `denoise()` matching the `Denoiser` protocol."""

    def __init__(
        self,
        width: int = 12,
        stride: int | None = None,
        gamma: float = 2.0,
        beta_percentile: float = 50.0,
        max_beta_tiles: int = 4000,
        apodize: bool = True,
        seed: int = 0,
    ) -> None:
        self.width = width
        self.stride = stride
        self.gamma = gamma
        self.beta_percentile = beta_percentile
        self.max_beta_tiles = max_beta_tiles
        self.apodize = apodize
        self.seed = seed

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return denoise(
            image,
            width=self.width,
            stride=self.stride,
            gamma=self.gamma,
            beta_percentile=self.beta_percentile,
            max_beta_tiles=self.max_beta_tiles,
            apodize=self.apodize,
            seed=self.seed,
        )
