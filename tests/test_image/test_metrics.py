"""Tests for egghouse.image.metrics: PSNR, SSIM, MS-SSIM, weak_signal_contrast.

Uses synthetic structured + noisy images so the contract is testable
without any FITS dependency.
"""

from __future__ import annotations

import numpy as np
import pytest

from egghouse.image import metrics


def _structured_image(shape=(256, 256), seed=0):
    """Returns a deterministic image with edges + gradients (not pure noise)."""
    rng = np.random.default_rng(seed)
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    image = np.sin(xx / 16.0) + np.cos(yy / 24.0)
    # A few square features at different intensities.
    image[40:80, 40:80] += 2.0
    image[150:200, 100:150] -= 1.5
    image += 0.05 * rng.standard_normal(shape)
    return image


# --- psnr ---


def test_psnr_identity_returns_inf():
    img = _structured_image()
    assert metrics.psnr(img, img, data_range=4.0) == float("inf")


def test_psnr_monotonic_in_noise_level():
    rng = np.random.default_rng(1)
    ref = _structured_image()
    low = ref + 0.1 * rng.standard_normal(ref.shape)
    high = ref + 0.5 * rng.standard_normal(ref.shape)
    assert metrics.psnr(low, ref, data_range=4.0) > metrics.psnr(
        high, ref, data_range=4.0
    )


def test_psnr_derives_data_range_from_reference():
    ref = _structured_image()
    img = ref + 0.1
    auto = metrics.psnr(img, ref)
    manual = metrics.psnr(img, ref, data_range=float(ref.max() - ref.min()))
    assert auto == pytest.approx(manual)


def test_psnr_rejects_non_positive_data_range():
    img = np.zeros((16, 16))
    with pytest.raises(ValueError):
        metrics.psnr(img, img, data_range=0.0)


# --- ssim ---


def test_ssim_identity_is_one():
    img = _structured_image()
    assert metrics.ssim(img, img, data_range=4.0) == pytest.approx(1.0, abs=1e-6)


def test_ssim_monotonic_in_noise_level():
    rng = np.random.default_rng(2)
    ref = _structured_image()
    low = ref + 0.1 * rng.standard_normal(ref.shape)
    high = ref + 0.5 * rng.standard_normal(ref.shape)
    assert metrics.ssim(low, ref, data_range=4.0) > metrics.ssim(
        high, ref, data_range=4.0
    )


def test_ssim_in_unit_range_for_natural_image():
    rng = np.random.default_rng(3)
    ref = _structured_image()
    noisy = ref + 0.2 * rng.standard_normal(ref.shape)
    score = metrics.ssim(noisy, ref, data_range=4.0)
    assert 0.0 < score < 1.0


# --- ms_ssim ---


def test_ms_ssim_identity_is_one():
    img = _structured_image()
    assert metrics.ms_ssim(img, img, data_range=4.0) == pytest.approx(
        1.0, abs=1e-6
    )


def test_ms_ssim_monotonic_in_noise_level():
    rng = np.random.default_rng(4)
    ref = _structured_image()
    low = ref + 0.1 * rng.standard_normal(ref.shape)
    high = ref + 0.5 * rng.standard_normal(ref.shape)
    assert metrics.ms_ssim(low, ref, data_range=4.0) > metrics.ms_ssim(
        high, ref, data_range=4.0
    )


def test_ms_ssim_small_input_raises():
    small = np.zeros((50, 50))
    with pytest.raises(ValueError, match="too small"):
        metrics.ms_ssim(small, small, data_range=1.0)


def test_ms_ssim_custom_weights_3_scales():
    img = _structured_image()
    weights = np.array([0.3, 0.3, 0.4])
    score = metrics.ms_ssim(img, img, data_range=4.0, weights=weights)
    assert score == pytest.approx(1.0, abs=1e-6)


# --- weak_signal_contrast ---


def test_weak_signal_contrast_identity_is_one():
    img = _structured_image()
    assert metrics.weak_signal_contrast(img, img) == pytest.approx(
        1.0, abs=1e-6
    )


def test_weak_signal_contrast_oversmoothed_is_zero():
    ref = _structured_image()
    flat = np.full_like(ref, ref.mean())
    assert metrics.weak_signal_contrast(flat, ref) == 0.0


def test_weak_signal_contrast_drops_with_noise():
    rng = np.random.default_rng(5)
    ref = _structured_image()
    low = ref + 0.1 * rng.standard_normal(ref.shape)
    high = ref + 0.8 * rng.standard_normal(ref.shape)
    assert metrics.weak_signal_contrast(low, ref) > metrics.weak_signal_contrast(
        high, ref
    )


def test_weak_signal_contrast_applies_mask():
    ref = _structured_image()
    img = ref.copy()
    # Corrupt one quadrant. Masking it (plus a small Sobel-boundary margin)
    # should largely restore the score; not exactly 1.0 because the 3x3
    # Sobel kernel still touches boundary pixels just outside the mask.
    img[:128, :128] = 0
    full_score = metrics.weak_signal_contrast(img, ref)
    mask = np.ones_like(ref, dtype=bool)
    mask[:130, :130] = False  # 2-pixel margin past the corruption boundary
    masked_score = metrics.weak_signal_contrast(img, ref, mask=mask)
    assert masked_score > full_score
    assert masked_score > 0.99  # close to perfect after excluding the region


def test_weak_signal_contrast_degenerate_constant_inputs():
    # Both flat → gradient maps both zero → defined as 1.0 (perfectly equal).
    flat = np.zeros((32, 32))
    assert metrics.weak_signal_contrast(flat, flat) == 1.0


# --- pearson_corr ---


def test_pearson_corr_identical_is_one():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((16, 16))
    assert metrics.pearson_corr(a, a) == pytest.approx(1.0)


def test_pearson_corr_anticorrelated_is_minus_one():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((16, 16))
    assert metrics.pearson_corr(a, -a) == pytest.approx(-1.0)


def test_pearson_corr_constant_input_is_nan():
    a = np.arange(64.0).reshape(8, 8)
    const = np.full((8, 8), 3.0)
    assert np.isnan(metrics.pearson_corr(a, const))


def test_pearson_corr_matches_manual_formula():
    rng = np.random.default_rng(2)
    a = rng.standard_normal(50)
    b = 0.7 * a + 0.3 * rng.standard_normal(50)
    sa, sb = np.std(a), np.std(b)
    expected = float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))
    assert metrics.pearson_corr(a, b) == pytest.approx(expected)


# --- db_ratio ---


def test_db_ratio_factor_of_ten_is_twenty_db():
    assert metrics.db_ratio(10.0, 1.0) == pytest.approx(20.0)


def test_db_ratio_equal_scales_is_zero_db():
    assert metrics.db_ratio(5.0, 5.0) == pytest.approx(0.0)


def test_db_ratio_zero_numerator_is_nan():
    assert np.isnan(metrics.db_ratio(0.0, 1.0))


def test_db_ratio_zero_denominator_is_inf():
    assert metrics.db_ratio(1.0, 0.0) == float("inf")
