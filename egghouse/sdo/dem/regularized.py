"""
Linear regularized (Tikhonov) DEM inversion via the GSVD reg-tweak method.

Reimplements the Hannah & Kontar (2012) regularized DEM algorithm using only
``numpy``/``scipy`` linear algebra (it deliberately avoids ``demregpy``, whose
SVD path can fail to converge under NumPy 2).

Per pixel the inversion solves the error-weighted Tikhonov problem

    min_x  || A_w x - I_w ||^2 + lambda^2 || L x ||^2 ,

with ``A_{c,t} = K_t(c) * dT_t`` so that ``I = A @ DEM`` (DEM in cm^-5 K^-1),
``A_w = A / sigma[:, None]``, ``I_w = I / sigma`` the error-folded data, and
``L`` a regularization operator (order 0 = identity, order 2 = second
difference). The DEM amplitude (~1e22) and the error-folded response (~1e-21)
differ by ~40 orders of magnitude, so the unknown is first column-scaled,
``x = diag(d) z`` with ``d = 1 / colnorm(A_w)``, giving a well-conditioned
design matrix with unit-norm columns. The closed-form Tikhonov filtered
solution is then obtained from the Generalized SVD (GSVD) of the scaled pair
``(B, L diag(d))`` (Hansen 1998); writing the standard-form singular values as
``c_i``, the solution is

    z(lambda) = z_null + sum_i  [ c_i / (c_i^2 + lambda^2) ]
                                * (u_i . I_w) * z_i ,

where ``z_null`` is the unpenalised least-squares fit in ``null(L)`` (carries
the constant/ramp the second-difference operator does not see). The data
residual chi^2(lambda) of this linear solution is monotone increasing in
lambda, so the discrepancy principle (Morozov 1966) — pick lambda so the data
chi^2 equals ``reg_tweak * n_channels`` — is solved by a 1-D bisection per
pixel. Negative DEM bins (the linear method is not constrained to be positive)
are clipped to zero in the *returned* DEM, but the reported chi^2 is that of
the unclipped linear solution (the quantity the discrepancy principle targets;
clipping would otherwise inflate it above target).

What is implemented vs. the full Hannah & Kontar (2012) method
--------------------------------------------------------------
Implemented: the error-weighted Tikhonov inversion via the GSVD of the scaled
pair (B, L diag(d)) for reg_order 0 and 2; column preconditioning for the
40-decade scale gap; per-pixel selection of lambda by the discrepancy principle
(the "reg_tweak" chi^2 target); the GSVD filter-factor form of the solution;
positivity by clipping the returned DEM.

NOT implemented (differences from demregpy / the full paper): demregpy's
two-step scheme that builds a *data-derived* guess DEM and folds it into a
constraint matrix ``L`` (the ``gloci`` / weighted-constraint iteration); the
DEM temperature-resolution / horizontal-error products; and the
positive-constraint reformulation. This is a single-pass linear Tikhonov
inversion with clipping, not the iterative reweighted scheme. Reconstructed
intensities and the peak DEM temperature agree well on smooth inputs, but the
DEM shape can differ from demregpy where the iterative reweighting matters.

References
----------
- Hannah, I. G. & Kontar, E. P. 2012, A&A 539, A146.
  DOI: 10.1051/0004-6361/201117576.
- Hansen, P. C. 1998, *Rank-Deficient and Discrete Ill-Posed Problems*, SIAM
  (GSVD form of Tikhonov regularization, filter factors).
- Morozov, V. A. 1966, Soviet Math. Dokl. 7, 414 (discrepancy principle).
"""

from typing import Dict, Tuple

import numpy as np
from scipy.linalg import svd


def _dt(temperatures: np.ndarray) -> np.ndarray:
    """Temperature bin widths dT = T ln(10) dlogT (matches dem_sites/dem_nnls)."""
    logt = np.log10(temperatures)
    dlogt = np.gradient(logt)
    return temperatures * np.log(10.0) * dlogt


def _regularization_operator(n_temps: int, order: int) -> np.ndarray:
    """Regularization operator L (same convention as nnls.py).

    order=0 -> identity (penalizes magnitude).
    order=2 -> second difference (penalizes curvature -> smooth DEM).
    """
    if order == 0:
        return np.eye(n_temps, dtype=np.float64)
    if order == 2:
        L = np.zeros((n_temps - 2, n_temps), dtype=np.float64)
        for i in range(n_temps - 2):
            L[i, i], L[i, i + 1], L[i, i + 2] = 1.0, -2.0, 1.0
        return L
    raise ValueError(f"reg_order must be 0 or 2; got {order}")


def _design_matrix(response: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """A with I = A @ DEM. response (n_temps, n_channels) -> A (n_channels, n_temps)."""
    return (response * dt[:, np.newaxis]).T


def _gsvd_pair(B: np.ndarray, Lz: np.ndarray):
    """Generalized SVD of the pair ``(B, Lz)`` for Tikhonov filtering.

    Operates in the *column-scaled* variable ``z`` (see :func:`_solve_pixel_reg`):
    ``B`` has unit-norm columns and ``Lz = L @ diag(d)``. The filtered Tikhonov
    solution in ``z`` is, for any ``lambda``,

        z(lam) = z_null + sum_i [c_i / (c_i^2 + lam^2)] * (u_i . Iw) * zcol_i ,

    where the sum runs over the penalised subspace (row space of ``Lz``) and
    ``z_null`` is the unpenalised least-squares fit in ``null(Lz)`` of the
    leftover residual. Only plain SVDs are used (no LAPACK GGSVD), avoiding the
    NumPy-2 convergence issue in demregpy.

    Reduction to standard form follows Hansen (1998, sec. 2.3): an SVD of ``Lz``
    splits its row space (penalised, whitened by the singular values) from its
    null space (unpenalised). The null-space response is removed by orthogonal
    projection so the two parts decouple, then an SVD of the whitened response
    gives the filter factors ``c_i / (c_i^2 + lam^2)``.

    Parameters
    ----------
    B  : (m, n) error-weighted, column-scaled design matrix (unit columns).
    Lz : (p, n) regularization operator in the scaled variable.

    Returns
    -------
    gamma : (k,) standard-form singular values c_i (descending).
    U     : (m, k) left singular vectors of the whitened response.
    zcols : (n, k) solution basis columns in z-space.
    V_null: (n, q) orthonormal basis of null(Lz), q = n - rank(Lz).
    BN    : (m, q) = B @ V_null (response of the null-space basis).
    """
    _Ul, sl, Vlt = svd(Lz, full_matrices=True)
    Vl = Vlt.T
    tol = max(Lz.shape) * np.finfo(np.float64).eps * (sl[0] if sl.size else 1.0)
    rank_l = int(np.sum(sl > tol))

    V_range = Vl[:, :rank_l]              # (n, rank_l) spans row(Lz)
    V_null = Vl[:, rank_l:]               # (n, q) spans null(Lz)
    s_range = sl[:rank_l]

    # Whitening map: Lz @ W has orthonormal rows.
    W = V_range / s_range[np.newaxis, :]  # (n, rank_l)

    # Response of the unpenalised (null) directions.
    BN = B @ V_null                       # (m, q)

    # Decouple: project the penalised response off range(BN) (Hansen 1998).
    A_bar = B @ W                          # (m, rank_l)
    if BN.shape[1] > 0:
        Q, _ = np.linalg.qr(BN)
        A_bar = A_bar - Q @ (Q.T @ A_bar)

    U, c, Zt = svd(A_bar, full_matrices=False)  # A_bar = U diag(c) Zt
    zcols = W @ Zt.T                       # (n, k) map whitened soln to z space
    return c, U, zcols, V_null, BN


def _solve_pixel_reg(
    intensity: np.ndarray,
    error: np.ndarray,
    A: np.ndarray,
    L: np.ndarray,
    reg_tweak: float,
    lam_bounds: Tuple[float, float],
) -> Tuple[np.ndarray, float, float]:
    """One-pixel linear regularized DEM. Returns (dem, data_chi2, lambda).

    The DEM amplitude (~1e22) and the error-folded response (~1e-21) differ by
    ~40 orders of magnitude, which wrecks the conditioning of a naive solve. We
    therefore substitute ``x = diag(d) z`` with ``d = 1 / colnorm(A_w)`` so the
    scaled design matrix ``B = A_w diag(d)`` has unit-norm columns and ``z`` is
    O(1). The Tikhonov problem becomes, in ``z``,

        min ||B z - I_w||^2 + lam^2 || L diag(d) z ||^2 ,

    solved in closed form for any ``lambda`` from the GSVD of ``(B, L diag(d))``.
    ``lambda`` is chosen by bisection so the data chi^2 equals
    ``reg_tweak * n_channels`` (discrepancy principle), then negatives are
    clipped.
    """
    n_t = A.shape[1]
    if not np.all(np.isfinite(intensity)) or np.all(intensity <= 0):
        return np.zeros(n_t, dtype=np.float64), 0.0, 0.0

    w = 1.0 / np.maximum(error, 1e-30)
    Aw = A * w[:, np.newaxis]          # (m, n)
    Iw = intensity * w                 # (m,)
    n_ch = intensity.size
    target = reg_tweak * n_ch

    # Column scaling so z is O(1) and B is well-conditioned.
    cn = np.linalg.norm(Aw, axis=0)
    cn[cn == 0.0] = 1.0
    d = 1.0 / cn
    B = Aw * d[np.newaxis, :]          # unit-norm columns
    Lz = L * d[np.newaxis, :]          # penalty in z-space

    gamma, U, zcols, V_null, BN = _gsvd_pair(B, Lz)
    beta = U.T @ Iw                    # data projection onto penalised modes

    # Truncate numerically-zero singular modes: with only ~6 channels most of
    # the (rank_l) penalised modes are noise (gamma ~ 1e-15 * gamma_max) and
    # would explode the filter at small lambda. Keep only well-resolved modes.
    g_max = float(gamma.max()) if gamma.size else 0.0
    keep = gamma > (1e-10 * g_max)
    gamma = gamma[keep]
    U = U[:, keep]
    zcols = zcols[:, keep]
    beta = beta[keep]
    if gamma.size == 0:
        return np.zeros(n_t, dtype=np.float64), 0.0, 0.0

    def solve_lambda(lam: float) -> Tuple[np.ndarray, float]:
        """Filtered z-solution -> x, and data chi^2, for a given lambda."""
        filt = gamma / (gamma ** 2 + lam ** 2)
        z = zcols @ (filt * beta)
        if V_null.shape[1] > 0:
            # Unpenalised null part: LS fit of the leftover residual.
            r = Iw - B @ z
            a, *_ = np.linalg.lstsq(BN, r, rcond=None)
            z = z + V_null @ a
        x = d * z
        resid = Aw @ x - Iw
        return x, float(resid @ resid)

    # The data chi^2 only responds to lambda when lambda is comparable to the
    # generalized singular values gamma. Those values absorb the (large) column
    # scaling, so we anchor the search range to the gamma spectrum rather than
    # to fixed absolute numbers. ``lam_bounds`` then acts as relative padding.
    lo = float(gamma.min()) * lam_bounds[0]
    hi = float(gamma.max()) * lam_bounds[1]
    x_lo, chi_lo = solve_lambda(lo)
    x_hi, chi_hi = solve_lambda(hi)

    if chi_lo >= target:
        x, chosen, chi2_lin = x_lo, lo, chi_lo
    elif chi_hi <= target:
        x, chosen, chi2_lin = x_hi, hi, chi_hi
    else:
        # Bisection in log(lambda): data chi^2 is monotone increasing in lambda.
        log_lo, log_hi = np.log(lo), np.log(hi)
        x, chosen, chi2_lin = x_hi, hi, chi_hi
        for _ in range(60):
            log_mid = 0.5 * (log_lo + log_hi)
            lam = np.exp(log_mid)
            x, chi2_lin = solve_lambda(lam)
            chosen = lam
            if chi2_lin > target:
                log_hi = log_mid
            else:
                log_lo = log_mid
            if abs(chi2_lin - target) < 1e-3 * target:
                break

    # Positivity: the linear method is unconstrained and can produce small
    # negative DEM bins; clip them for the returned (physical) DEM. The reported
    # chi^2 is that of the *linear* solution -- the quantity the discrepancy
    # principle actually targeted -- since clipping breaks data consistency and
    # would otherwise inflate chi^2 above the target.
    x = np.clip(x, 0.0, None)
    return x, float(chi2_lin), float(chosen)


def dem_regularized(
    intensities: np.ndarray,
    errors: np.ndarray,
    response: np.ndarray,
    temperatures: np.ndarray,
    *,
    reg_order: int = 2,
    reg_tweak: float = 1.0,
    lam_bounds: Tuple[float, float] = (1e-3, 1e3),
) -> Tuple[np.ndarray, Dict]:
    """Linear regularized (Tikhonov / Hannah-Kontar-style) DEM inversion.

    Parameters
    ----------
    intensities : np.ndarray
        Observed intensities (DN/s/pixel), shape ``(n_channels,)`` or
        ``(n_pixels, n_channels)``.
    errors : np.ndarray
        1-sigma uncertainties, same shape as ``intensities``.
    response : np.ndarray
        Temperature response, shape ``(n_temps, n_channels)`` (same convention
        as :func:`egghouse.sdo.dem.dem_sites` / :func:`dem_nnls`).
    temperatures : np.ndarray
        Temperatures in Kelvin, shape ``(n_temps,)``.
    reg_order : {0, 2}
        Regularization operator order: 0 = identity (magnitude),
        2 = second difference (smoothness). Default 2.
    reg_tweak : float
        Discrepancy-principle target multiplier; lambda is chosen per pixel so
        the data chi^2 ~ ``reg_tweak * n_channels``. Larger -> smoother /
        more regularized. Default 1.0.
    lam_bounds : (float, float)
        Search range for lambda, given *relative* to the generalized singular
        value spectrum: the bisection scans
        ``[gamma_min * lam_bounds[0], gamma_max * lam_bounds[1]]``. Anchoring to
        gamma makes the search scale-invariant. Default ``(1e-3, 1e3)``.

    Returns
    -------
    dem : np.ndarray
        DEM in cm^-5 K^-1, shape ``(n_temps,)`` or ``(n_pixels, n_temps)``.
        Non-negative (negatives clipped to zero).
    info : dict
        ``chi2`` (mean data chi^2), ``chi2_map`` (per pixel),
        ``lambda_map`` (per-pixel chosen lambda), ``reg_tweak``.

    Notes
    -----
    This is a single-pass GSVD Tikhonov inversion with discrepancy-principle
    lambda selection and positivity clipping. See the module docstring for what
    parts of the full Hannah & Kontar (2012) / demregpy scheme are and are not
    reproduced (notably the iterative data-derived constraint reweighting).

    References
    ----------
    Hannah & Kontar (2012, A&A 539, A146); Hansen (1998); Morozov (1966).
    """
    squeeze = intensities.ndim == 1
    intensities = np.atleast_2d(intensities).astype(np.float64)
    errors = np.atleast_2d(errors).astype(np.float64)
    n_pixels, n_channels = intensities.shape
    n_temps = len(temperatures)
    if response.shape != (n_temps, n_channels):
        raise ValueError(
            f"Response shape {response.shape} doesn't match "
            f"expected ({n_temps}, {n_channels})"
        )

    dt = _dt(temperatures)
    A = _design_matrix(response, dt)
    L = _regularization_operator(n_temps, reg_order)

    dem = np.zeros((n_pixels, n_temps), dtype=np.float64)
    chi2_map = np.zeros(n_pixels, dtype=np.float64)
    lambda_map = np.zeros(n_pixels, dtype=np.float64)
    for p in range(n_pixels):
        dem[p], chi2_map[p], lambda_map[p] = _solve_pixel_reg(
            intensities[p], errors[p], A, L, reg_tweak, lam_bounds
        )

    info = {
        "chi2": float(np.mean(chi2_map)),
        "chi2_map": chi2_map,
        "lambda_map": lambda_map,
        "reg_tweak": float(reg_tweak),
    }
    if squeeze:
        dem = dem.squeeze()
    return dem, info
