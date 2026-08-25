"""Tests for the Theil-Sen photon-transfer line fit."""

import numpy as np

from egghouse.image.noise import photon_transfer_fit, theil_sen_fit


def test_theil_sen_recovers_line_with_outlier_bins():
    x = np.linspace(100.0, 4000.0, 30)
    y = 0.07 * x + 1.0
    y[-6:] *= np.array([2.0, 5.0, 20.0, 100.0, 300.0, 800.0])  # scene-change bins
    g, r2 = theil_sen_fit(x, y)
    assert abs(g - 0.07) / 0.07 < 0.02
    assert abs(r2 - 1.0) < 5.0
    g_ols, _, _ = photon_transfer_fit(x, y)
    assert abs(g_ols - 0.07) > abs(g - 0.07)  # OLS is pulled by the outliers


def test_theil_sen_degenerate_inputs():
    assert all(np.isnan(v) for v in theil_sen_fit([1.0], [2.0]))
    assert all(np.isnan(v) for v in theil_sen_fit([1.0, 1.0], [2.0, 3.0]))
