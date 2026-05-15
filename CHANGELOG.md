# Changelog

All notable changes to **egghouse** are recorded here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/) and the project follows
[Semantic Versioning](https://semver.org/).

## [0.5.0] — 2026-05-15

### Added — AIA Level 1.0 → 1.5 prep stages

New module **`egghouse.sdo.prep`**, migrated from the soon-to-be-retired
solar-and-space-weather (snsw) repo. All functions are re-exported from
`egghouse.sdo`.

- `aia_update_pointing(sdo_map, pointing_table=None)` — refresh WCS
  keywords against the JSOC master pointing table.
- `aia_respike(sdo_map, spikes=None)` — re-inject spike pixels removed
  by the Level 1 pipeline.
- `aia_correct_degradation(sdo_map, correction_table=None)` —
  time-dependent effective-area correction; returns input unchanged
  for non-AIA wavelengths.
- `aia_deconvolve(sdo_map, psfs=None)` — PSF deconvolution; pair with
  `cached_aia_psfs` so the slow PSF computation is amortized.
- `cached_aia_psfs(path, *, wavelengths=...)` — pickle-cached AIA PSF
  dict for the standard seven channels.
- `mask_out_of_disk(sdo_map, *, fill_value=-5000.0)` — sunpy.Map utility
  that flags off-disk pixels with a sentinel; derives the disk radius
  from `R_SUN` or `RSUN_OBS / CDELT1`.

### Added — image utility

- `egghouse.image.bin_ndarray(array, new_shape, operation)` — n-D
  block-sum / block-mean down-binner. Each output dimension must
  evenly divide the corresponding source dimension; raises on
  unsupported operations, dim mismatches, and non-divisible shapes.

### Notes

- These additions complete the practically useful subset of the snsw
  retirement. Items deliberately *not* migrated and the rationale:
    * `aia_register` — trivial wrapper over `aiapy.calibrate.register`,
      callers should use aiapy directly.
    * `preparation` / `define_preparation_func` — small composer +
      `functools.partial` helper; compose at the call site.
    * `pad_to_target_shape` / `resize(Map, ...)` — `to_level15` already
      pads to 4096 and `bin_ndarray` plus a 4-line meta update covers
      the snsw `resize` use case.
    * `transfer.{get_file_list, download_http, download_wget}` —
      egghouse already has more robust requests-based equivalents in
      `egghouse.transfer`.
    * snsw `io.{png,jpeg,gif,tiff}` — separate, larger scope; deferred.
- Tests: 14 new cases (7 for `bin_ndarray`, 7 for prep). Full suite:
  204 passed.

---

## [0.4.0] — 2026-05-15

### Added

- **New module `egghouse.sdo.jsoc`** consolidating JSOC export utilities
  (partial migration from the soon-to-be-retired
  `solar-and-space-weather` package).
    - `jsoc_export(query, *, email, method='url', protocol='fits', client=None)`
      submits a DRMS export request, blocks until staging is done, and
      returns the resulting URL list. Network-bound; pair it with
      `egghouse.transfer.download_parallel` for retries.
    - `aia_euv_query(times, *, wavelengths, series, tolerance)` composes
      a DRMS record-set string selecting AIA EUV records near each
      timestamp, optionally filtered to specific channels. Multiple
      timestamps are concatenated into a single export request.
    - `cached_correction_table(path)` and
      `cached_pointing_table(path, *, start, end)` memoize the
      slow-to-fetch aiapy calibration tables to disk so batch jobs do
      not re-hit JSOC on every record.
    - All four are re-exported from `egghouse.sdo`.
- The `drms` package is now a soft dependency: `jsoc.py` imports it
  inside the functions that need it, so simply importing the module
  does not require `drms` to be installed.

### Notes

- Tests cover the local query composer, cache-hit reads, and the
  `jsoc_export` failure / success paths via a stub client; the
  network-bound code paths are deliberately not exercised.

---

## [0.3.0] — 2026-05-15

### Added

- **`egghouse.sdo.dem.load_ssw_temperature_response(path, *, log_temperatures=None, wavelengths=None, response_key='response_v10_en')`**
  loads the canonical CHIANTI-based AIA temperature response from a
  SolarSoftWare `aia_get_response.pro` `.npz` archive. Supports linear
  interpolation in log T to a target grid; refuses extrapolation
  explicitly. The function is re-exported from `egghouse.sdo` and
  `egghouse.sdo.dem`.
- **`get_temperature_response`** gains two keyword arguments,
  `ssw_table_path` and `ssw_response_key`, that dispatch through the new
  SSW loader. When supplied, the aiapy / Gaussian-fallback paths are
  bypassed entirely.

### Changed

- **Breaking (effective):** `_get_aiapy_response` (the internal aiapy
  path of `get_temperature_response`) now raises a clear
  `NotImplementedError` pointing at the SSW loader. Previously it
  silently died with `AttributeError` on aiapy ≥ 0.12 because
  `Channel.temperature_response` was removed upstream. Callers that
  relied on the aiapy path must either:
    1. supply `ssw_table_path=...`, or
    2. call `load_ssw_temperature_response(...)` directly.
- Existing fallback (Gaussian) tests in `tests/test_sdo/test_dem.py` now
  force `HAS_AIAPY = False` via `monkeypatch`, so the suite exercises
  the documented fallback regardless of aiapy availability.

### Why

aiapy 0.12 deliberately removed `Channel.temperature_response` because
computing K(T) correctly requires CHIANTI atomic data that aiapy does
not ship. Silently falling back to a Gaussian approximation has caused
real harm in downstream research (an undine pre-research project lost
four experiments to a similar silent-synthetic fallback), so the SSW
loader is now the recommended path for research-quality DEM analysis.

### Migration

```python
# Before (v0.2.x, will now raise NotImplementedError on aiapy >= 0.12):
response = get_temperature_response(temperatures=temps)

# After (v0.3+):
response = get_temperature_response(
    temperatures=temps, ssw_table_path='response_matrix.npz'
)
# or, equivalently:
import numpy as np
response = load_ssw_temperature_response(
    'response_matrix.npz', log_temperatures=np.log10(temps)
)
```

---

## [0.2.0] and earlier

See git history; this is the first formal changelog entry.
