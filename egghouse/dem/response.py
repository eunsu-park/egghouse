"""
Generic temperature-response tools for DEM analysis.

Instrument-agnostic helpers shared by all DEM solvers:

- ``get_default_temperatures`` — a log-spaced temperature grid.
- ``temperature_response_from_chianti`` — fold CHIANTI (fiasco) contribution
  functions with *any* instrument's per-channel wavelength response to build
  the temperature response ``K(T)``. Instrument-specific wiring (e.g. the
  AIA wavelength response via aiapy) lives in the instrument package
  (``egghouse.sdo`` for SDO/AIA), which calls this function.
- ``load_ssw_temperature_response`` — read an SSW ``aia_get_response``-style
  ``.npz`` table (still generic: any channel list).
- ``compute_response_derivative`` / ``get_response_weights`` — operate on a
  response matrix.

References:
    - Boerner et al. (2012), Solar Physics 275, 41-66
    - Boerner et al. (2014), Solar Physics 289, 2377-2397
    - Dere et al. (1997), A&AS 125, 149 (CHIANTI)
"""

import os
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

# Check for fiasco (CHIANTI) availability — provides the contribution
# function G(lambda, T) for temperature_response_from_chianti.
try:
    import fiasco  # noqa: F401
    HAS_FIASCO = True
except ImportError:
    HAS_FIASCO = False

# Default electron density and abundance for the CHIANTI synthesis.
# Coronal (Feldman) abundances match the SSW aia_get_response convention.
DEFAULT_DEM_DENSITY_CM3 = 1e9
DEFAULT_DEM_ABUNDANCE = "sun_coronal_1992_feldman_ext"
# Wavelength band (Angstrom) over which EUV channels have response;
# transitions outside are skipped (and ions with none are not solved).
DEFAULT_RESPONSE_BAND = (80.0, 360.0)

# Default temperature grid (log10 K).
DEFAULT_LOGT_MIN = 5.5
DEFAULT_LOGT_MAX = 7.5
DEFAULT_LOGT_BINS = 100


def get_default_temperatures(
    logt_min: float = DEFAULT_LOGT_MIN,
    logt_max: float = DEFAULT_LOGT_MAX,
    n_bins: int = DEFAULT_LOGT_BINS,
) -> np.ndarray:
    """Get a default temperature array (Kelvin) for DEM analysis.

    Parameters
    ----------
    logt_min, logt_max : float
        log10(T/K) range (default 5.5 .. 7.5).
    n_bins : int
        Number of temperature points (default 100).

    Examples
    --------
    >>> get_default_temperatures().shape
    (100,)
    """
    logt = np.linspace(logt_min, logt_max, n_bins)
    return 10.0 ** logt


def _fold_lines_into_channel(
    g_photon: np.ndarray,
    line_wavelength: np.ndarray,
    channel_wavelength: np.ndarray,
    channel_response: np.ndarray,
    plate_scale_sr: float,
) -> np.ndarray:
    """Fold per-line contribution functions into one channel's K(T).

    ``K(T) = (Omega_pix / 4pi) * sum_lines g_photon(T, line) * R(lambda_line)``

    Parameters
    ----------
    g_photon : np.ndarray
        Per-line contribution function in photon units, shape
        ``(n_T, n_lines)`` (``cm^3 photon / s``).
    line_wavelength : np.ndarray
        Wavelength of each line, ``(n_lines,)`` Angstrom.
    channel_wavelength, channel_response : np.ndarray
        Channel wavelength grid (Angstrom) and wavelength response
        (``cm^2 DN / photon``); interpolated onto the lines (0 out of grid).
    plate_scale_sr : float
        Pixel solid angle (steradian).

    Returns
    -------
    np.ndarray
        ``(n_T,)`` in ``DN cm^5 s^-1 pixel^-1``.
    """
    r_at_line = np.interp(
        line_wavelength, channel_wavelength, channel_response, left=0.0, right=0.0
    )
    return (plate_scale_sr / (4.0 * np.pi)) * (g_photon * r_at_line).sum(axis=1)


def temperature_response_from_chianti(
    channel_responses: Sequence[Tuple[np.ndarray, np.ndarray, float]],
    temperatures: np.ndarray,
    *,
    density_cm3: float = DEFAULT_DEM_DENSITY_CM3,
    abundance: str = DEFAULT_DEM_ABUNDANCE,
    band: tuple = DEFAULT_RESPONSE_BAND,
) -> np.ndarray:
    """Temperature response K(T) from CHIANTI (fiasco), instrument-agnostic.

    For each channel,

        K_i(T) = (Omega_pix / 4pi)
                 * sum_transitions G_ij(T) / (hc/lambda_ij) * R_i(lambda_ij)

    where ``G_ij`` is the CHIANTI contribution function
    (`fiasco.Ion.contribution_function`, ``cm^3 erg/s``), ``hc/lambda``
    converts energy to photons, ``R_i`` is the channel's wavelength response
    (``cm^2 DN/photon``), and ``Omega_pix`` is the pixel solid angle. Result
    in ``DN cm^5 s^-1 pixel^-1``.

    Parameters
    ----------
    channel_responses : sequence of (wavelength_grid, response, plate_scale)
        One tuple per channel: ``wavelength_grid`` (Angstrom),
        ``response`` (``cm^2 DN/photon`` on that grid), ``plate_scale``
        (steradian/pixel). The caller (instrument package) supplies these —
        e.g. ``egghouse.sdo`` builds them from aiapy for AIA.
    temperatures : np.ndarray
        Temperatures in Kelvin, shape ``(n_T,)``.
    density_cm3, abundance, band :
        CHIANTI electron density, abundance set, and the wavelength band
        (Angstrom) over which transitions are folded (ions with no in-band
        lines are skipped — avoids their expensive level-population solve).

    Returns
    -------
    np.ndarray
        ``(n_temperatures, n_channels)``.

    Notes
    -----
    One-time, slow (solves level populations for every contributing CHIANTI
    ion). Persist the result. Requires the CHIANTI database (fiasco downloads
    it on first use). Time-dependent instrument degradation is not folded in
    here; apply it at the image level.

    References
    ----------
    Dere et al. (1997), A&AS 125, 149 (CHIANTI); Boerner et al. (2012).
    """
    import astropy.constants as const
    import astropy.units as u

    temperatures = np.asarray(temperatures, dtype=np.float64)
    T = temperatures * u.K
    n_e = density_cm3 * u.cm**-3
    n_ch = len(channel_responses)

    K = np.zeros((n_ch, temperatures.size), dtype=np.float64)
    hc = (const.h * const.c).to_value("erg angstrom")  # erg * Angstrom

    for ion_name in fiasco.list_ions():
        try:
            ion = fiasco.Ion(ion_name, T, abundance=abundance)
            line_wl = ion.transitions.wavelength.to_value("angstrom")
            in_band = (line_wl >= band[0]) & (line_wl <= band[1])
            if not in_band.any():
                continue
            g = ion.contribution_function(n_e).to_value("cm3 erg / s")[:, 0, :]
            line_wl = line_wl[in_band]
            g_photon = g[:, in_band] / (hc / line_wl)  # cm^3 photon / s
            for i, (ch_wl, ch_r, omega) in enumerate(channel_responses):
                K[i] += _fold_lines_into_channel(g_photon, line_wl, ch_wl, ch_r, omega)
        except Exception:
            # Ions with incomplete CHIANTI data are skipped.
            continue

    return K.T  # (n_temperatures, n_channels)


def load_ssw_temperature_response(
    path: Union[str, os.PathLike],
    *,
    log_temperatures: Optional[np.ndarray] = None,
    wavelengths: Optional[List[int]] = None,
    response_key: str = "response_v10_en",
) -> np.ndarray:
    """
    Load an SSW-derived temperature response from a ``.npz`` archive.

    The archive follows the convention of ``aia_get_response.pro`` (IDL
    SolarSoftWare) as exposed by demregpy:

    - ``log_temperature``: 1D ``log10(T/K)`` values, length ``n_T``.
    - ``channels``: 1D integer channel wavelengths (Angstroms), ``n_lambda``.
    - response arrays of shape ``(n_lambda, n_T)`` under keys like
      ``response_v9_en`` / ``response_v10_en``.

    Parameters
    ----------
    path : path-like
        Path to the ``.npz`` archive.
    log_temperatures : np.ndarray, optional
        Target ``log10(T/K)`` grid (linear interp along T). If None, the
        source grid is used.
    wavelengths : list of int, optional
        Channels to extract, in output order. If None, all channels in the
        file (file order) are returned.
    response_key : str
        Which response variant to read.

    Returns
    -------
    np.ndarray
        ``(n_temperatures, n_wavelengths)``. Units inherit from the file
        (typically DN cm^5 s^-1 pixel^-1).
    """
    with np.load(path, allow_pickle=False) as data:
        if response_key not in data.files:
            available = sorted(k for k in data.files if k.startswith("response"))
            raise KeyError(
                f"response_key {response_key!r} not in {os.fspath(path)!r}; "
                f"available response keys: {available}"
            )
        raw = np.asarray(data[response_key], dtype=np.float64)
        src_channels = np.asarray(data["channels"]).astype(int).tolist()
        src_log_t = np.asarray(data["log_temperature"], dtype=np.float64)

    target_waves = list(wavelengths) if wavelengths is not None else list(src_channels)

    if raw.shape != (len(src_channels), src_log_t.size):
        raise ValueError(
            f"response array shape {raw.shape} does not match "
            f"(n_channels={len(src_channels)}, n_log_t={src_log_t.size})"
        )

    missing = [w for w in target_waves if w not in src_channels]
    if missing:
        raise KeyError(
            f"wavelengths {missing} not found in SSW table; "
            f"available: {src_channels}"
        )

    channel_indices = [src_channels.index(w) for w in target_waves]
    raw_selected = raw[channel_indices]  # (n_waves, n_T_src)

    if log_temperatures is None:
        return raw_selected.T  # (n_T_src, n_waves)

    target_log_t = np.asarray(log_temperatures, dtype=np.float64)
    src_min, src_max = float(src_log_t.min()), float(src_log_t.max())
    tol = 1e-9
    if target_log_t.min() < src_min - tol or target_log_t.max() > src_max + tol:
        raise ValueError(
            f"target log T range [{target_log_t.min()}, {target_log_t.max()}] "
            f"falls outside the source grid [{src_min}, {src_max}]; "
            "extrapolation is not allowed"
        )

    interpolated = np.empty((len(target_waves), target_log_t.size), dtype=np.float64)
    if not np.all(np.diff(src_log_t) > 0):
        order = np.argsort(src_log_t)
        src_log_t = src_log_t[order]
        raw_selected = raw_selected[:, order]
    for i in range(len(target_waves)):
        interpolated[i] = np.interp(target_log_t, src_log_t, raw_selected[i])
    return interpolated.T  # (n_T_target, n_waves)


def compute_response_derivative(
    response: np.ndarray,
    temperatures: np.ndarray,
) -> np.ndarray:
    """Temperature derivative dK/d(logT) of a response matrix.

    Parameters
    ----------
    response : np.ndarray
        Response matrix ``(n_temps, n_channels)``.
    temperatures : np.ndarray
        Temperatures in Kelvin.

    Returns
    -------
    np.ndarray
        Same shape as ``response``.
    """
    logt = np.log10(temperatures)
    dlogt = np.gradient(logt)
    return np.gradient(response, axis=0) / dlogt[:, np.newaxis]


def get_response_weights(
    response: np.ndarray,
    method: str = "sum",
) -> np.ndarray:
    """Per-temperature weights from a response matrix (sum/max/mean)."""
    if method == "sum":
        return np.sum(response, axis=1)
    elif method == "max":
        return np.max(response, axis=1)
    elif method == "mean":
        return np.mean(response, axis=1)
    else:
        raise ValueError(f"Unknown method: {method}")
