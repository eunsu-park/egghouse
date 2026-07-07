"""AIA Level 1.0 → 1.5 preparation stages and a sunpy.Map disk mask.

`egghouse.sdo.level15.to_level15` already covers rotation, resampling,
and padding for both AIA and HMI. This module adds the *aiapy-backed*
prep stages that `to_level15` does not perform on its own:

- :func:`aia_update_pointing` — refreshes outdated WCS keywords with
  the JSOC master pointing table.
- :func:`aia_respike` — re-injects the spike pixels that the Level 1
  pipeline removed.
- :func:`aia_correct_degradation` — applies the time-dependent
  effective-area correction.
- :func:`aia_deconvolve` — PSF deconvolution. Pair with
  :func:`cached_aia_psfs` so the PSFs (slow to compute, ~minutes per
  channel) are amortized across a batch.

Plus :func:`mask_out_of_disk`, a sunpy.Map utility that flags pixels
outside the solar limb with a sentinel value — handy when downstream
code needs to ignore off-disk regions (e.g. the DEM model's training
loop).

All third-party heavy imports (`sunpy`, `aiapy`, `astropy`) happen
inside the function bodies, so importing this module is cheap and
does not pull aiapy unless one of the prep functions is actually
called.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from astropy.table import Table
    from sunpy.map import Map


# Channels for which aiapy provides degradation correction and PSFs.
# Order matches snsw, which in turn matches `aia_get_response`.
_AIA_WAVELENGTHS_WITH_CALIBRATION: tuple[int, ...] = (94, 131, 171, 193, 211, 304, 335)


# ---------------------------------------------------------------------------
# Prep stage wrappers
# ---------------------------------------------------------------------------


def aia_update_pointing(
    sdo_map: "Map",
    pointing_table: Optional["Table"] = None,
) -> "Map":
    """Refresh AIA WCS keywords with the JSOC master pointing table.

    Wraps `aiapy.calibrate.update_pointing`. Supplying a pre-fetched
    `pointing_table` (e.g. from :func:`cached_pointing_table`) avoids
    refetching the table for every record in a batch.
    """
    from aiapy.calibrate import update_pointing

    if pointing_table is None:
        return update_pointing(sdo_map)
    return update_pointing(sdo_map, pointing_table=pointing_table)


def aia_respike(
    sdo_map: "Map",
    spikes: Optional[object] = None,
) -> "Map":
    """Re-inject removed spike pixels into a Level 1 AIA map.

    Wraps `aiapy.calibrate.respike`. If `spikes` is not provided it is
    fetched via `aiapy.calibrate.fetch_spikes`, which is a JSOC round-trip
    per record. Callers running a batch should pre-fetch and pass the
    result in.
    """
    from aiapy.calibrate import fetch_spikes, respike

    if spikes is None:
        spikes = fetch_spikes(sdo_map)
    return respike(sdo_map, spikes=spikes)


def aia_correct_degradation(
    sdo_map: "Map",
    correction_table: Optional["Table"] = None,
) -> "Map":
    """Apply the AIA time-dependent effective-area correction.

    Wraps `aiapy.calibrate.correct_degradation`. The correction is only
    defined for the EUV passbands plus 304 Å; for any other wavelength
    the map is returned unchanged.

    Pass a `correction_table` (e.g. from :func:`cached_correction_table`)
    to amortize the JSOC fetch across a batch.
    """
    from aiapy.calibrate import correct_degradation

    wavelength = int(sdo_map.meta.get("wavelnth", 0))
    if wavelength not in _AIA_WAVELENGTHS_WITH_CALIBRATION:
        return sdo_map
    if correction_table is None:
        return correct_degradation(sdo_map)
    return correct_degradation(sdo_map, correction_table=correction_table)


def aia_deconvolve(
    sdo_map: "Map",
    psfs: Optional[dict[int, np.ndarray]] = None,
) -> "Map":
    """PSF-deconvolve an AIA map.

    Wraps `aiapy.psf.deconvolve`. PSF computation is the expensive part
    (`aiapy.psf.psf(...)` takes minutes per channel), so callers should
    pre-compute a `{wavelength: psf}` dict — `cached_aia_psfs` returns
    one for the standard seven channels — and pass it in.

    The map is returned unchanged when its wavelength is outside the
    seven channels for which aiapy provides a PSF.
    """
    import aiapy.psf
    import astropy.units as u

    wavelength = int(sdo_map.meta.get("wavelnth", 0))
    if wavelength not in _AIA_WAVELENGTHS_WITH_CALIBRATION:
        return sdo_map

    if psfs is None:
        psf = aiapy.psf.psf(wavelength * u.angstrom)
    else:
        psf = psfs[wavelength]
    return aiapy.psf.deconvolve(sdo_map, psf=psf)


# ---------------------------------------------------------------------------
# PSF cache (paired with aia_deconvolve)
# ---------------------------------------------------------------------------


def cached_aia_psfs(
    path: Union[str, os.PathLike],
    *,
    wavelengths: Iterable[int] = _AIA_WAVELENGTHS_WITH_CALIBRATION,
) -> dict[int, np.ndarray]:
    """Load (or compute and cache) AIA PSFs for the requested channels.

    Computing the AIA PSF via `aiapy.psf.psf(...)` is slow (~minutes per
    channel), so we pickle a `{wavelength_int: psf_array}` dict on first
    call and deserialize on subsequent calls. The cache is keyed by
    `wavelengths`; if the file is missing or does not cover *all* of the
    requested wavelengths, it is rebuilt for the full set.

    Args:
        path: Pickle cache location.
        wavelengths: Wavelengths in Angstroms to include in the cache. (default
            the seven calibrated AIA channels)

    Returns:
        Mapping wavelength (Å, int) -> PSF array.
    """
    path = Path(path)
    requested = tuple(int(w) for w in wavelengths)

    if path.is_file():
        with open(path, "rb") as f:
            cached = pickle.load(f)
        if all(int(w) in cached for w in requested):
            # Return a fresh dict with int keys, regardless of how it was pickled.
            return {int(w): cached[int(w)] for w in requested}

    import aiapy.psf
    import astropy.units as u

    psfs: dict[int, np.ndarray] = {}
    for wave in requested:
        psfs[wave] = aiapy.psf.psf(wave * u.angstrom)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(psfs, f)
    return psfs


# ---------------------------------------------------------------------------
# Disk masking
# ---------------------------------------------------------------------------


def mask_out_of_disk(
    sdo_map: "Map",
    *,
    fill_value: float = -5000.0,
) -> "Map":
    """Return a copy of `sdo_map` with off-disk pixels set to `fill_value`.

    The solar limb radius is read from the header: `R_SUN` if present
    (already in pixels) else `RSUN_OBS / CDELT1`. The disk center is
    `(CRPIX1, CRPIX2)`. Useful for marking pixels that should be ignored
    by downstream pipelines (e.g. the DEM model's training loop).

    Args:
        sdo_map: Input map. Not mutated.
        fill_value: Sentinel written into off-disk pixels. The snsw default
            matches what undine's pre-research used. (default -5000.0)

    Returns:
        New Map with the masked data array and the original metadata.
    """
    from sunpy.map import Map

    data = sdo_map.data.copy()
    meta = sdo_map.meta.copy()
    if "R_SUN" in meta:
        radius_px = float(meta["R_SUN"])
    elif "RSUN_OBS" in meta and "CDELT1" in meta:
        radius_px = float(meta["RSUN_OBS"]) / float(meta["CDELT1"])
    else:
        raise KeyError(
            "sdo_map.meta lacks both 'R_SUN' and the ('RSUN_OBS', 'CDELT1') "
            "pair required to derive the disk radius in pixels"
        )
    center_x = float(meta["CRPIX1"])
    center_y = float(meta["CRPIX2"])
    yy, xx = np.ogrid[: data.shape[0], : data.shape[1]]
    off_disk = (xx - center_x) ** 2 + (yy - center_y) ** 2 > radius_px ** 2
    data[off_disk] = fill_value
    return Map(data, meta)
