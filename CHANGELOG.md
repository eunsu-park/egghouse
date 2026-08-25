# Changelog

All notable changes to **egghouse** are recorded here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/) and the project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added — Poisson-Gaussian primitives for the lolipop evaluation rail

- `egghouse.denoise.anscombe.generalized_forward` / `generalized_inverse`: the
  generalized Anscombe transform for Poisson-Gaussian data (gain, read noise,
  offset) and its exact unbiased inverse via the Poisson closed form
  (Makitalo & Foi 2013). Mean-unbiasedness at 0.5–100 counts is covered by tests.
- `egghouse.image.noise.poisson_gaussian_noise(..., clip_per_bin=True)`: robust
  outlier clip estimated inside each intensity bin (photon-transfer-curve
  convention) instead of one global sigma, so cosmic rays are rejected without
  truncating the wide high-intensity noise. Default unchanged (global clip).

### Docs

- Condensed `README.MD` (517 → 132 lines) into an orientation/index: per-module
  cookbooks now live solely in `docs/*_guide.md`, and the dependency section was
  corrected to reflect the solar stacks being core (not optional) and the new
  `image` extra. Added the previously-unlinked `docs/aia_pipeline_guide.md`.

### Changed — solar stacks promoted to core dependencies

- `install_requires` now includes the SDO / DEM / transfer / FITS stacks
  (`aiapy`, `drms`, `fiasco`, `sunpy`, `astropy`, `matplotlib`, `requests`,
  `beautifulsoup4`) that were previously optional extras, so a plain
  `pip install -e .` (or `pip install egghouse`) yields a fully working install
  — JSOC download, Level-1.5 prep, CHIANTI temperature response, and HTTP
  download all work with no extras. Only niche stacks stay optional.
- The `sdo`, `dem`, `transfer`, and `fits` extras are retained as **empty no-op
  aliases** so existing `egghouse[sdo]`-style references keep resolving.
- Fixed a packaging gap: `drms` (JSOC) and `aiapy` (prep/response) were used by
  `egghouse.sdo` but declared in no installable extra.
- Added an `image` extra (`opencv-python`) for the OpenCV-backed transform in
  `egghouse.image.transforms`, which was an undeclared lazy import.

### Added — domain-standard AIA color tables & colorization

- `egghouse.sdo.aia_color` — the official SDO/AIA color tables (SolarSoft
  `aia_lct.pro` by K. Schrijver, as adopted by sunpy) for the ten channels
  94, 131, 171, 193, 211, 304, 335, 1600, 1700, 4500 Å:
  - `aia_color_lut(wavelength, source="numpy")` → `(256, 3)` uint8 RGB LUT.
    The default `source="numpy"` is a **pure-NumPy** reproduction (embeds the
    IDL "Red Temperature" base table; no optional dependency) and is verified
    **bit-identical** to `source="sunpy"` for all ten channels.
  - `aia_colorize(image, wavelnth, exptime=None, source="numpy")` → `(H, W, 3)`
    uint8. With `exptime`, the raw image is intensity-scaled via `aia_intscale`
    first; otherwise an 8-bit grayscale input is colorized directly.
  - `aia_colormap(wavelength)` → matplotlib `Colormap` via sunpy (optional).
  - `AIA_COLOR_WAVELENGTHS` constant.
- `egghouse.image.colorize` — instrument-agnostic primitives backing the above:
  `apply_colormap(gray, lut)` (exact 256-entry LUT lookup, `(H, W)` uint8 →
  `(H, W, 3)` uint8) and `lut_from_matplotlib(cmap, n=256)`.
- `setup.py`: `sdo`/`all` extras now include `matplotlib>=3.5` (only the
  optional sunpy-sourced tables / `aia_colormap` need it; the default numpy
  path needs none of it).

### Changed — `egghouse.swdb` lazy-loads its DB submodules

- `egghouse.swdb.__init__` no longer eagerly imports `register` / `query`
  (which pull in `egghouse.database` → psycopg2). They are now resolved
  lazily via module `__getattr__` on first attribute access. This lets the
  pure NOAA SWPC parsers (`egghouse.swdb.swpc`) and the package schemas /
  handlers import **without** the `[database]` extra, matching egghouse's
  guard-heavy-optional-deps convention (as `sdo` / `transfer` already do).
- Backward compatible: `from egghouse.swdb import get_sdo_best_match`,
  `register_fits_dir`, `scan_fits`, `RegisterReport`, `get_sdo_best_matches`
  resolve unchanged when psycopg2 is installed; only then is it required.
  Verified: consumers are `undine` (`get_sdo_best_match`) and `solaris-data`
  (`core/query.py`) — both use lazy symbols and keep working.

### Added — `egghouse.sdo` FITS-datetime parser

- `parse_fits_datetime(file_path)` (new module `egghouse.sdo.timeparse`) —
  extracts the observation datetime from a FITS header (T_REC / T_OBS /
  DATE-OBS, incl. SDO TAI and legacy DD/MM/YY formats) and falls back to
  parsing the SDO AIA/HMI filename. Astropy is lazy-imported (`HAS_ASTROPY`);
  the header path is skipped and filename parsing is used when astropy is
  absent. Re-exported from `egghouse.sdo`.
- Promoted verbatim from `solaris-data`'s `core/parse.py` so SOLARIS
  sub-projects reading SDO FITS share one tested implementation. The
  overlapping SDO-specific string parsers (`_parse_t_rec` / `_parse_obs_time`
  in `core/sdo.py`) were intentionally **not** merged — their format lists
  differ and merging would change outputs.
- Migration: `from core.parse import parse_fits_datetime` still works
  (`solaris-data` keeps a re-export shim); new code should use
  `from egghouse.sdo import parse_fits_datetime`.

### Added — `egghouse.swdb` SWPC real-time parsers

- `parse_xray`, `parse_proton`, `parse_solar_wind`, `parse_kp_1m`,
  `parse_kp_forecast`, `parse_solar_probabilities`, `parse_alerts`,
  `parse_3day_forecast` (new module `egghouse.swdb.swpc`) — pure pandas parsers
  for NOAA SWPC real-time JSON products into `rt_*` / `swpc_*` table rows.
  Promoted from `solaris-data`'s `core/swpc.py` so every SOLARIS sub-project
  shares one tested implementation. Re-exported from `egghouse.swdb`.
- Parsers are pure (pandas only, no DB). Importing them via `egghouse.swdb`
  no longer forces psycopg2 — the DB-backed `register`/`query` submodules are
  lazy-loaded (see the `### Changed` entry above).
- Migration: `from core.swpc import parse_xray` still works (`solaris-data`
  keeps a re-export shim); new code should use
  `from egghouse.swdb import parse_xray` (or `egghouse.swdb.swpc`).

### Added — generic correlation / dB metrics (`egghouse.image.metrics`)

Promoted from `lolipop` (duplicated across its evaluation metrics):

- `pearson_corr(a, b)` — Pearson correlation between two arrays, returning
  NaN when either input is constant.
- `db_ratio(numerator, denominator)` — amplitude ratio in dB
  (`20*log10(num/den)`), guarding zero scales (NaN / +inf).

### Added — paired augmentation (`egghouse.image.augment`)

Promoted from `lolipop` (duplicated across its two Noise2Noise datasets):

- `paired_random_crop(a, b, patch, rng)` — same random square crop applied
  to an `(input, target)` pair.
- `paired_flip_rot(a, b, rng)` — same random flip + 90-degree rotation
  applied to both arrays, using a caller-supplied `numpy.random.Generator`.

## [0.10.0] — 2026-06-05

### Changed — DEM is now a top-level, instrument-agnostic package

**Breaking:** moved `egghouse.sdo.dem` → **`egghouse.dem`**. The DEM
inversion solvers and generic temperature-response tools are
instrument-agnostic (they take `(intensities, errors, response,
temperatures)`), so they no longer live under `sdo`.

- `egghouse.dem` — solvers (`dem_sites`, `dem_nnls`, `dem_regularized`,
  `dem_sparse`, `dem_plowman`, `dem_mcmc`, `dem_spline`, `dem_gaussian`),
  `dem_map`, generic `temperature_response_from_chianti`,
  `load_ssw_temperature_response`, `get_default_temperatures`, utils.
- `egghouse.sdo` — keeps the **AIA-specific** temperature response:
  `get_temperature_response` and `AIA_DEM_WAVELENGTHS` (new module
  `egghouse.sdo.dem_response`), which builds the AIA wavelength response via
  aiapy and calls `egghouse.dem.temperature_response_from_chianti`.
- Migration: `from egghouse.sdo.dem import dem_sites` →
  `from egghouse.dem import dem_sites`; `get_temperature_response` /
  `AIA_DEM_WAVELENGTHS` stay at `from egghouse.sdo import ...`.

### Added — DEM solvers + CHIANTI response (under `egghouse.dem`)

- Seven solvers beyond SITES: Tikhonov-NNLS (Lawson & Hanson 1995; Hannah &
  Kontar 2012), regularized GSVD (Hannah & Kontar 2012), sparse basis-pursuit
  (Cheung et al. 2015), fast linear (Plowman et al. 2013), MCMC (Kashyap &
  Drake 1998), spline forward-fit (Weber et al. 2004), single-Gaussian
  (Aschwanden et al. 2013). `dem_map(method=...)` dispatches.
- `temperature_response_from_chianti` — recompute K(T) from CHIANTI (fiasco)
  contribution functions × a wavelength response (replaces aiapy's removed
  `Channel.temperature_response`). Also fixed the SITES iteration
  (multiplicative MART); the previous additive update was non-physical.

### Added — `egghouse.swdb` sdo best-match read queries

- `get_sdo_best_match` / `get_sdo_best_matches` — promoted from
  `solaris-data`'s `core/query.py` so they live with the rest of the swdb
  domain layer and other projects import them instead of duplicating.
  Re-exported from `egghouse.swdb`.

### Fixed

- `egghouse.sdo`: `get_correction_table` import path updated for **aiapy
  0.12** (moved to `aiapy.calibrate.utils`).

## [0.9.0] — 2026-06-04

### Added — `egghouse.denoise` (classical image denoisers)

New subpackage of channel-agnostic classical denoisers, migrated from the
`lolipop` project so both `lolipop` and future projects import them instead
of duplicating. Each module exposes `denoise(image, ...)` + a parametric
`*Denoiser` class.

- `anscombe` — Anscombe variance-stabilising transform + unbiased inverse
  (Makitalo & Foi 2011), wrapping any inner Gaussian denoiser; guards
  against zero-centred input.
- `bm3d` — Block-Matching 3D collaborative filtering.
- `nlm` — Non-local means (scikit-image).
- `tv` — Total-variation (Chambolle) denoising.
- `wiener` — Wiener filter (scipy.signal.wiener).
- `wavelet` — Wavelet BayesShrink denoising (scikit-image).
- New `extras_require['denoise']` = scikit-image, PyWavelets, bm3d.

### Added — `egghouse.image` metrics / transforms / noise

- `egghouse.image.metrics` — `psnr`, `ssim`, `ms_ssim` (Wang 2003/2004),
  `weak_signal_contrast`. Pure numpy/scipy, no torch.
- `egghouse.image.transforms` — composable numpy transforms (`compose`,
  `to_float32`, `nan_to_value`, `percentile_clip`, `normalize_minmax`,
  `normalize_log1p`, `circular_mask`).
- `egghouse.image.noise` — `mad` and `robust_sigma` (1.4826·MAD) for
  outlier-robust per-frame noise-scale estimation.

All exported from `egghouse.image`. 136 tests added under
`tests/test_denoise` and `tests/test_image`.

## [0.8.0] — 2026-05-17

### Added — `egghouse.swdb` (solar / space-weather DB domain layer)

New subpackage on top of the generic `egghouse.database` (0.7.0)
infrastructure. Domain layer; AIA-only (what undine needs). Other
instruments subclass the ABC in their own projects.

- `ValidationResult` — type-safe ok/fail result for FITS validation
  (ported from setup-sw-db `core/result.py`).
- `SDO_SCHEMA` / `LASCO_SCHEMA` / `SECCHI_SCHEMA` — reference
  declarative table specs. **Verified byte-identical** to the
  setup-sw-db `schema_config` blocks, so a project can build a
  setup-sw-db-compatible `solar_images` DB straight from these
  constants via `egghouse.database.create_tables_from_schema`.
- `FitsHandler` ABC + `AiaFitsHandler` — turn a FITS file into a
  validated `ValidationResult`, a flat DB row (`to_db_record`), and an
  archive path (`target_dir`). astropy is imported lazily.
    - **Timestamp policy divergence (documented):** `AiaFitsHandler`
      keys on `T_OBS` (UTC), consistent with undine's acquisition
      grouping, whereas setup-sw-db's SDO validator uses `T_REC` (the
      JSOC slotted record time). The `sdo` *table shape* stays
      setup-sw-db compatible; only the column's semantic source differs
      per project. For AIA EUV `T_OBS` needs no TAI conversion.
- `register_fits_dir(scan_dir, *, handler, table, db_config,
  conflict_columns, move_root=None, error_dirs=None, …)` +
  `RegisterReport` — generalized scan → validate → idempotent upsert →
  optional archive-move flow (from setup-sw-db
  `scripts/register_sdo.py`). Header-only validation is I/O-bound, so
  parallelism uses a thread pool (no pickling constraints; handler may
  hold state). DB write delegated to
  `egghouse.database.upsert_dataframe` (idempotent). `RegisterReport`
  counts reconcile: scanned == valid + sum(errors) + skipped_existing.
- `scan_fits` — recursive FITS listing with substring exclusion
  (default excludes AIA `spike` artifacts).
- `swdb` added to `egghouse.__all__`; `find_packages` picks up the
  subpackage automatically.

### Notes

- Purely additive; no existing behavior changed. setup-sw-db (git-pins
  egghouse, imports only `PostgresManager`) is unaffected — verified by
  byte-identical DDL on its real lasco/sdo/secchi configs.
- Tests: 21 new cases (schemas DDL round-trip, AiaFitsHandler against
  synthetic AIA FITS incl. error categories, register_fits_dir with a
  stubbed upsert covering valid/invalid/move/parallel/empty). Full
  suite: 235 passed.

---

## [0.7.0] — 2026-05-17

### Added — declarative schema + bulk record helpers in `egghouse.database`

Generic, instrument-blind DB infrastructure lifted (in spirit) from the
setup-sw-db `core/database.py` so any project can build a schema and
register records with just a config dict + an egghouse import.

`egghouse.database.schema`:
- `build_create_table_sql(table, table_spec)` / `build_index_sql(table, indexes)`
  / `split_schema_meta(table_spec)` — pure SQL/metadata builders
  (no DB connection; unit-testable without PostgreSQL).
- `create_tables_from_schema(db_config, schema_config, *, drop, verbose)`
  — declarative table creation supporting composite `_primary_key`,
  `_unique`, and `_indexes`. Skips existing tables unless `drop=True`.
- `create_database(db_config)` — idempotent CREATE DATABASE via an
  admin connection (template1 → postgres).
- `initialize_database(db_config, schema_config)` — create_database +
  create_tables_from_schema.

`egghouse.database.records`:
- `normalize_records(df)` — DataFrame → list[dict], lowercased columns,
  NaN→None (handles the empty-DataFrame edge that the original
  setup-sw-db code crashed on).
- `build_upsert_sql(table, columns, conflict_columns)` — pure
  `INSERT ... ON CONFLICT (...) DO NOTHING`, composite-key aware.
- `upsert_dataframe(df, table, db_config, *, conflict_columns, batch)`
  — idempotent bulk upsert; tolerates secondary UNIQUE violations as
  skips.
- `find_orphans(file_paths)` / `delete_orphans(table, db_config, *,
  file_column)` — drop rows whose referenced file is gone.

All public names re-exported from `egghouse.database`. The schema
format is **instrument-blind** — feeding any declarative config (solar
images, space-weather, anything) through `initialize_database` creates
exactly those tables; no domain code required.

### Notes

- Identifier safety: table/column/index identifiers are validated
  against `^[A-Za-z_][A-Za-z0-9_]*$` and raise `ValueError` on
  violation (schema configs are developer-authored, but a typo that
  produced injectable DDL would otherwise be silent). Column *types*
  are free-form text.
- This is purely additive; existing `PostgresManager` / `config`
  behavior is unchanged. setup-sw-db (which git-pins egghouse and uses
  only `PostgresManager`) is unaffected.
- Tests: 26 new cases — pure builders driven directly, table-creation
  logic driven through a fake db that records SQL. The
  connection-opening wrappers are integration-level and not run in the
  unit suite. Full suite: 214 passed.

---

## [0.6.0] — 2026-05-16

### Removed (breaking) — `egghouse.io` retired

The `egghouse.io` subpackage has been removed in full. An audit found
that:

- The submodule was effectively unused inside egghouse itself.
- Downstream consumers (the undine project) touched it in exactly one
  line (`acquisition/core.py:211`'s `read_fits_header`), which is now
  three lines of `astropy.io.fits.getheader(...)`.
- The originally stated purposes ("study" + "dep-light") were only
  genuinely served by the pure-Python FITS / BMP implementations.
  Adding the snsw Pillow-backed PNG / JPEG / GIF / TIFF wrappers would
  not have served either purpose — they were just thin `PIL.Image`
  adapters.
- Maintaining wrappers that re-export the canonical libraries (astropy,
  Pillow) under a different name adds an indirection layer without
  buying anything; the API churn risk (e.g. the aiapy 0.12 break that
  we just patched in 0.3.0) is duplicated.

What this means for callers:

```python
# Before (egghouse 0.5.0 and earlier):
from egghouse.io import read_fits, write_fits, read_fits_header, read_bmp, write_bmp

# After (egghouse 0.6.0+):
from astropy.io import fits        # for FITS
from PIL import Image              # for BMP / PNG / JPEG / GIF / TIFF / ...
header = fits.getheader(path)      # was read_fits_header(path)
data = fits.getdata(path)          # was part of read_fits(path)
```

The pure-Python FITS / BMP implementations that lived under
`egghouse.io` are not migrated elsewhere; format-internals study
materials, if ever needed again, belong in a dedicated repo rather
than a production utility library.

`egghouse.__all__` no longer lists `"io"`. The `tests/test_io/`
directory is removed.

---

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
