"""Tests for egghouse.denoise (channel-agnostic classical denoisers).

Validates each denoiser improves PSNR/SSIM over Gaussian-noised synthetic
structured images. Does not test against any real FITS data.
"""

from __future__ import annotations

import numpy as np
import pytest

from egghouse.denoise import anscombe, bm3d, nlm, tv, wavelet, wiener
from egghouse.image import metrics


@pytest.fixture
def synthetic_pair():
    """Returns (noisy, clean) where noisy = clean + Gaussian noise."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:128, 0:128].astype(np.float64)
    clean = np.sin(xx / 16.0) + np.cos(yy / 24.0)
    clean[40:80, 40:80] += 2.0
    clean[90:110, 60:90] -= 1.5
    noisy = clean + 0.3 * rng.standard_normal(clean.shape)
    return noisy, clean


def _psnr(out, ref):
    return metrics.psnr(out, ref, data_range=float(ref.max() - ref.min()))


def _ssim(out, ref):
    return metrics.ssim(out, ref, data_range=float(ref.max() - ref.min()))


# --- wavelet ---


def test_wavelet_improves_psnr(synthetic_pair):
    noisy, clean = synthetic_pair
    out = wavelet.denoise(noisy)
    assert _psnr(out, clean) > _psnr(noisy, clean) + 2.0


def test_wavelet_improves_ssim(synthetic_pair):
    noisy, clean = synthetic_pair
    out = wavelet.denoise(noisy)
    assert _ssim(out, clean) > _ssim(noisy, clean) + 0.1


def test_wavelet_preserves_shape(synthetic_pair):
    noisy, _ = synthetic_pair
    out = wavelet.denoise(noisy)
    assert out.shape == noisy.shape


def test_wavelet_class_form_matches_function(synthetic_pair):
    noisy, _ = synthetic_pair
    fn_out = wavelet.denoise(noisy, sigma=0.3)
    cls_out = wavelet.WaveletDenoiser(sigma=0.3)(noisy)
    np.testing.assert_array_equal(fn_out, cls_out)


# --- bm3d ---


def test_bm3d_improves_psnr(synthetic_pair):
    noisy, clean = synthetic_pair
    out = bm3d.denoise(noisy)
    assert _psnr(out, clean) > _psnr(noisy, clean) + 5.0


def test_bm3d_improves_ssim_substantially(synthetic_pair):
    noisy, clean = synthetic_pair
    out = bm3d.denoise(noisy)
    assert _ssim(out, clean) > 0.85


def test_bm3d_preserves_shape(synthetic_pair):
    noisy, _ = synthetic_pair
    out = bm3d.denoise(noisy)
    assert out.shape == noisy.shape


def test_bm3d_class_form_matches_function(synthetic_pair):
    noisy, _ = synthetic_pair
    fn_out = bm3d.denoise(noisy, sigma=0.3)
    cls_out = bm3d.BM3DDenoiser(sigma=0.3)(noisy)
    # BM3D is non-deterministic at the ~1e-4 level (likely internal
    # threading / SIMD reductions). Use a tolerance well above that floor
    # but still many orders of magnitude tighter than any signal we care
    # about, so the equivalence claim is meaningful.
    np.testing.assert_allclose(fn_out, cls_out, atol=1e-3)


# --- nlm ---


def test_nlm_improves_psnr(synthetic_pair):
    noisy, clean = synthetic_pair
    out = nlm.denoise(noisy)
    assert _psnr(out, clean) > _psnr(noisy, clean) + 3.0


def test_nlm_improves_ssim(synthetic_pair):
    noisy, clean = synthetic_pair
    out = nlm.denoise(noisy)
    assert _ssim(out, clean) > 0.75


def test_nlm_preserves_shape(synthetic_pair):
    noisy, _ = synthetic_pair
    out = nlm.denoise(noisy)
    assert out.shape == noisy.shape


def test_nlm_class_form_matches_function(synthetic_pair):
    noisy, _ = synthetic_pair
    fn_out = nlm.denoise(noisy, sigma=0.3)
    cls_out = nlm.NLMDenoiser(sigma=0.3)(noisy)
    np.testing.assert_allclose(fn_out, cls_out, atol=1e-7)


# --- tv ---


def test_tv_improves_psnr(synthetic_pair):
    noisy, clean = synthetic_pair
    out = tv.denoise(noisy, weight=0.2)
    assert _psnr(out, clean) > _psnr(noisy, clean) + 2.0


def test_tv_improves_ssim(synthetic_pair):
    noisy, clean = synthetic_pair
    out = tv.denoise(noisy, weight=0.2)
    assert _ssim(out, clean) > _ssim(noisy, clean) + 0.1


def test_tv_preserves_shape(synthetic_pair):
    noisy, _ = synthetic_pair
    out = tv.denoise(noisy)
    assert out.shape == noisy.shape


def test_tv_class_form_matches_function(synthetic_pair):
    noisy, _ = synthetic_pair
    fn_out = tv.denoise(noisy, weight=0.15)
    cls_out = tv.TVDenoiser(weight=0.15)(noisy)
    np.testing.assert_array_equal(fn_out, cls_out)


# --- wiener ---


def test_wiener_improves_psnr(synthetic_pair):
    noisy, clean = synthetic_pair
    out = wiener.denoise(noisy)
    assert _psnr(out, clean) > _psnr(noisy, clean) + 2.0


def test_wiener_improves_ssim(synthetic_pair):
    noisy, clean = synthetic_pair
    out = wiener.denoise(noisy)
    assert _ssim(out, clean) > _ssim(noisy, clean) + 0.1


def test_wiener_preserves_shape(synthetic_pair):
    noisy, _ = synthetic_pair
    out = wiener.denoise(noisy)
    assert out.shape == noisy.shape


def test_wiener_class_form_matches_function(synthetic_pair):
    noisy, _ = synthetic_pair
    fn_out = wiener.denoise(noisy, mysize=7)
    cls_out = wiener.WienerDenoiser(mysize=7)(noisy)
    np.testing.assert_array_equal(fn_out, cls_out)


# --- anscombe (Poisson-stabilised wrapper) ---


@pytest.fixture
def poisson_pair():
    """Returns (noisy, clean) where noisy ~ Poisson(clean) on a high-rate field."""
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:128, 0:128].astype(np.float64)
    clean = 30.0 + 20.0 * np.sin(xx / 16.0) + 15.0 * np.cos(yy / 24.0)
    clean[40:80, 40:80] += 40.0
    clean[90:110, 60:90] = np.maximum(clean[90:110, 60:90] - 10.0, 1.0)
    clean = np.maximum(clean, 1.0)
    noisy = rng.poisson(clean).astype(np.float64)
    return noisy, clean


def test_anscombe_inverse_is_unbiased_in_expectation_asymptotic():
    # Makitalo-Foi inverse is unbiased *in expectation over Poisson
    # realisations* asymptotically. The closed-form Taylor approximation
    # used here retains a small residual bias at low rates (paper Table III,
    # ~5% at rate=5, ~1% at rate=20); only the high-rate regime is a clean
    # test of correctness. Cross-check that the residual bias *shrinks*
    # as the rate increases.
    rng = np.random.default_rng(2)
    residuals = []
    for rate in (100.0, 1000.0):
        samples = rng.poisson(rate, size=20000).astype(np.float64)
        mean_recovered = anscombe.inverse(anscombe.forward(samples)).mean()
        residuals.append(abs(mean_recovered - rate) / rate)
    assert residuals[0] < 0.005, f"rate=100 rel err {residuals[0]:.4f}"
    assert residuals[1] < 0.001, f"rate=1000 rel err {residuals[1]:.4f}"


def test_anscombe_wrapper_improves_psnr_over_inner_alone(poisson_pair):
    noisy, clean = poisson_pair
    bare = bm3d.denoise(noisy)
    wrapped = anscombe.denoise(noisy, bm3d.denoise)
    # Anscombe stabilisation should help BM3D on Poisson data; require a
    # modest improvement (not a tight optimum, just sign of the effect).
    assert _psnr(wrapped, clean) > _psnr(bare, clean) + 0.5


def test_anscombe_preserves_shape(poisson_pair):
    noisy, _ = poisson_pair
    out = anscombe.denoise(noisy, bm3d.denoise)
    assert out.shape == noisy.shape


def test_anscombe_class_form_matches_function(poisson_pair):
    noisy, _ = poisson_pair
    inner = bm3d.BM3DDenoiser(sigma=1.0)
    fn_out = anscombe.denoise(noisy, inner)
    cls_out = anscombe.AnscombeDenoiser(inner)(noisy)
    # BM3D is non-deterministic at the ~1e-3 level; the inverse Anscombe
    # term z**2 / 4 then amplifies that floor by ~z/2 ~ 8x at the rates in
    # this fixture. Compare with rtol that brackets the propagated floor
    # while still being many orders tighter than any signal we care about.
    np.testing.assert_allclose(fn_out, cls_out, rtol=2e-3)


def test_anscombe_bypasses_on_zero_centred_input():
    """On background-subtracted (zero-centred) data, the Anscombe
    sqrt(x + 3/8) is undefined on the negative tail. Wrapper must
    bypass + warn, returning the inner denoiser's output directly."""
    rng = np.random.default_rng(0)
    img = rng.standard_normal((64, 64)) * 5.0  # zero-centred, ~50/50 negative

    def inner_identity(x: np.ndarray) -> np.ndarray:
        return x

    with pytest.warns(UserWarning, match="negative mass"):
        out = anscombe.denoise(img, inner_identity)
    # Bypass path: output == inner_identity(input) exactly.
    np.testing.assert_array_equal(out, img.astype(np.float64))


def test_anscombe_no_bypass_when_explicitly_disabled():
    """`bypass_on_negative=False` forces the broken transform — used
    only for diagnostics. Must NOT emit the bypass warning."""
    rng = np.random.default_rng(0)
    img = rng.standard_normal((32, 32)) * 5.0

    def inner_identity(x: np.ndarray) -> np.ndarray:
        return x

    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")  # promote any warning to an exception
        # No bypass warning expected; numerical garbage is fine.
        anscombe.denoise(img, inner_identity, bypass_on_negative=False)


def test_anscombe_does_not_bypass_on_positive_input(poisson_pair):
    """The bypass must NOT trigger on legitimate Poisson-count data."""
    noisy, _ = poisson_pair  # all positive (Poisson counts)
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")
        anscombe.denoise(noisy, bm3d.denoise)  # should not warn


def test_anscombe_bypasses_on_tight_zero_centred_input():
    """Regression test for the FNRGF edge case (p1-fnrgf-cor2.md):
    a tight zero-centred distribution where few pixels fall below
    -3/8 must STILL trigger the bypass. The original "1% below -3/8"
    threshold missed this case; the new "10% < 0" threshold catches it."""
    rng = np.random.default_rng(0)
    img = rng.standard_normal((64, 64)) * 0.1  # tight, MAD ~ 0.07
    n_below = float((img < -3.0 / 8.0).mean())
    n_neg = float((img < 0).mean())
    # Fixture defeats the old (-3/8) guard but trips the new (<0) one.
    assert n_below < 0.01, f"fixture has {n_below*100:.1f}% < -3/8"
    assert n_neg > 0.40

    def inner_identity(x: np.ndarray) -> np.ndarray:
        return x

    with pytest.warns(UserWarning, match="negative mass"):
        out = anscombe.denoise(img, inner_identity)
    np.testing.assert_array_equal(out, img.astype(np.float64))
