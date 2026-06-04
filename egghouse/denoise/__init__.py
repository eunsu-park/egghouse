"""Classical, channel-agnostic image denoisers.

Each module exposes a top-level ``denoise(image, ...) -> np.ndarray`` callable
(matching the ``Callable[[np.ndarray], np.ndarray]`` denoiser protocol) plus a
parametric ``*Denoiser`` class when configuration is needed:

    anscombe : Anscombe variance-stabilising transform + unbiased inverse,
               wrapping any inner Gaussian denoiser (Makitalo & Foi 2011).
    bm3d     : Block-Matching 3D collaborative filtering.
    nlm      : Non-local means (scikit-image).
    tv       : Total-variation (Chambolle) denoising.
    wiener   : Wiener filter (scipy.signal.wiener).
    wavelet  : Wavelet (BayesShrink) denoising (scikit-image).

Submodules are imported on demand so that the heavier optional dependencies
(``bm3d``, ``scikit-image``, ``PyWavelets``) are only required by the modules
that actually use them. Install them with ``pip install egghouse[denoise]``.

    >>> from egghouse.denoise.wavelet import WaveletDenoiser
    >>> denoise = WaveletDenoiser()
    >>> clean = denoise(noisy)
"""
