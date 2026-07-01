# egghouse.denoise Usage Guide

A collection of channel-agnostic classical (non-DL) image denoisers. Each module
provides both a functional `denoise(image, ...)` and a parametric `*Denoiser`
class; both satisfy the `Callable[[np.ndarray], np.ndarray]` protocol (input is a
2-D float array, output has the same shape). v0.9+.

> **Installation:** The heavy dependencies are separated into the `denoise`
> extra. `pip install egghouse[denoise]` (scikit-image, PyWavelets, bm3d).
> Submodules are imported only when needed, so if you use only
> `anscombe`/`wiener` it works even without scikit-image/bm3d.

---

## Module Structure

```
egghouse/denoise/
├── __init__.py   # lightweight (does not eager-import submodules)
├── wavelet.py    # WaveletDenoiser  (skimage BayesShrink)
├── bm3d.py       # BM3DDenoiser     (Block-Matching 3D)
├── nlm.py        # NLMDenoiser      (skimage non-local means)
├── tv.py         # TVDenoiser       (Total Variation, Chambolle)
├── wiener.py     # WienerDenoiser   (scipy.signal.wiener)
└── anscombe.py   # AnscombeDenoiser (Poisson variance-stabilizing wrapper)
```

---

## Function List

| Module | Functional | Class | Notes |
|------|--------|--------|------|
| `wavelet` | `denoise(image, sigma=None, ...)` | `WaveletDenoiser` | BayesShrink; sigma auto-estimated |
| `bm3d` | `denoise(image, sigma=None)` | `BM3DDenoiser` | Powerful, but beware signal damage in scenes with few self-similar patches (e.g. K-corona) |
| `nlm` | `denoise(image, sigma=None, ...)` | `NLMDenoiser` | non-local means |
| `tv` | `denoise(image, weight=0.1)` | `TVDenoiser` | piecewise-flat prior |
| `wiener` | `denoise(image, mysize=...)` | `WienerDenoiser` | aggressive high-frequency suppression |
| `anscombe` | `forward(x)` / `inverse(z)` / `denoise(image, inner, ...)` | `AnscombeDenoiser` | wraps an inner Gaussian denoiser |

---

## Basic Usage

```python
import numpy as np
from egghouse.denoise.wavelet import WaveletDenoiser

noisy = ...                       # 2-D float
denoise = WaveletDenoiser()       # sigma auto-estimated if omitted
clean = denoise(noisy)            # np.ndarray -> np.ndarray

# functional form is identical
from egghouse.denoise import wavelet
clean = wavelet.denoise(noisy, sigma=0.3)
```

Evaluate with `egghouse.image.metrics`:

```python
from egghouse.image import psnr, ssim
print(psnr(clean, reference, data_range=4.0), ssim(clean, reference, data_range=4.0))
```

---

## Anscombe (Poisson data)

For data dominated by Poisson noise, such as photon counts, it works well to
apply a Gaussian denoiser after the Anscombe variance-stabilizing transform. The
wrapper wraps an arbitrary inner denoiser.

```python
from egghouse.denoise import anscombe
from egghouse.denoise.bm3d import BM3DDenoiser

clean = anscombe.denoise(noisy_counts, BM3DDenoiser(sigma=1.0))
```

> **Guard:** `sqrt(x + 3/8)` is valid only for Poisson counts (nearly always
> positive). On **zero-centered** inputs (with a large fraction of negatives),
> such as background-subtracted/differenced data, the transform breaks, so the
> wrapper detects this, bypasses the transform, applies the inner denoiser
> directly, and raises a `UserWarning`. To force the transform for diagnostic
> purposes, set `bypass_on_negative=False`.

---

## Notes

- The functional `denoise()` and the class-form `*Denoiser()(x)` give identical
  results (BM3D may differ at the ~1e-3 level due to internal non-determinism).
- Full signatures: [API_REFERENCE.md](../API_REFERENCE.md), `egghouse.denoise`
  section.
</content>
</invoke>
