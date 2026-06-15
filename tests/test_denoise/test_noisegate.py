"""Tests for egghouse.denoise.noisegate (DeForest 2017 Fourier noise-gating).

Noise-gating is a coherence-preserving denoiser: it removes the incoherent
noise floor while keeping coherent structure. The tests therefore use a
*sparse-structure* field (smooth bumps on a flat, noisy background) — the
regime the gate is built for — rather than the global-sinusoid fixture used
for the per-pixel spatial denoisers.
"""

from __future__ import annotations

import numpy as np
import pytest

from egghouse.denoise import noisegate


def _bumps(shape, specs):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float64)
    img = np.zeros(shape, dtype=np.float64)
    for cy, cx, amp, sig in specs:
        img += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sig**2))
    return img


@pytest.fixture
def sparse_pair():
    """(noisy, clean): a few smooth coherent bumps on a flat noisy field."""
    rng = np.random.default_rng(0)
    clean = _bumps((96, 96), [(30, 30, 5.0, 4.0), (60, 70, 4.0, 5.0), (72, 22, 3.0, 3.0)])
    noisy = clean + 0.5 * rng.standard_normal(clean.shape)
    return noisy, clean


def test_noisegate_reduces_background_noise(sparse_pair):
    noisy, clean = sparse_pair
    out = noisegate.denoise(noisy, width=12, gamma=2.0)
    bg = clean < 0.02 * clean.max()  # signal-free background pixels
    assert out[bg].std() < 0.75 * noisy[bg].std()


def test_noisegate_preserves_coherent_amplitude(sparse_pair):
    noisy, clean = sparse_pair
    out = noisegate.denoise(noisy, width=12, gamma=2.0)
    cy, cx = np.unravel_index(int(clean.argmax()), clean.shape)
    # The brightest coherent bump must survive (not be gated as noise).
    local = out[cy - 3 : cy + 4, cx - 3 : cx + 4].max()
    assert local > 0.6 * clean.max()


def test_noisegate_identity_when_gamma_zero(sparse_pair):
    noisy, _ = sparse_pair
    # gamma=0 gates nothing → weighted overlap-add reconstructs the input.
    out = noisegate.denoise(noisy, width=12, gamma=0.0)
    np.testing.assert_allclose(out, noisy, atol=1e-7)


def test_noisegate_preserves_shape(sparse_pair):
    noisy, _ = sparse_pair
    out = noisegate.denoise(noisy)
    assert out.shape == noisy.shape


def test_noisegate_class_form_matches_function(sparse_pair):
    noisy, _ = sparse_pair
    fn_out = noisegate.denoise(noisy, gamma=1.5)
    cls_out = noisegate.NoiseGateDenoiser(gamma=1.5)(noisy)
    np.testing.assert_array_equal(fn_out, cls_out)


def test_noisegate_sequence_reduces_noise_and_keeps_shape():
    rng = np.random.default_rng(1)
    frame = _bumps((48, 48), [(20, 20, 5.0, 4.0), (32, 34, 4.0, 4.0)])
    clean = np.broadcast_to(frame, (8, 48, 48))
    noisy = clean + 0.5 * rng.standard_normal(clean.shape)
    out = noisegate.noise_gate_sequence(noisy, width=8, gamma=2.0)
    assert out.shape == noisy.shape
    bg = clean < 0.02 * clean.max()
    assert out[bg].std() < 0.75 * noisy[bg].std()


def test_noisegate_per_axis_width():
    rng = np.random.default_rng(2)
    frame = _bumps((40, 40), [(18, 18, 5.0, 4.0)])
    clean = np.broadcast_to(frame, (10, 40, 40))
    noisy = clean + 0.5 * rng.standard_normal(clean.shape)
    # Short time window, larger spatial tile.
    out = noisegate.noise_gate_sequence(noisy, width=(4, 12, 12), gamma=2.0)
    assert out.shape == noisy.shape
    bg = clean < 0.02 * clean.max()
    assert out[bg].std() < noisy[bg].std()
    # A per-axis width whose length != ndim is an error.
    with pytest.raises(ValueError):
        noisegate.noise_gate(np.zeros((8, 8)), width=(4, 4, 4))


def test_noisegate_rejects_bad_ndim():
    with pytest.raises(ValueError):
        noisegate.noise_gate(np.zeros((4, 4, 4, 4)))
    with pytest.raises(ValueError):
        noisegate.denoise(np.zeros((4, 4, 4)))
    with pytest.raises(ValueError):
        noisegate.noise_gate_sequence(np.zeros((4, 4)))
