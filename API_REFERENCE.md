# API Reference

Complete list of all public functions, classes, and constants in egghouse.

**Detailed Usage Guides:** See [docs/](docs/) folder for module-specific guides.

- [image_guide.md](docs/image_guide.md) - Image processing utilities (incl. metrics, transforms, noise)
- [denoise_guide.md](docs/denoise_guide.md) - Classical image denoisers
- [sdo_guide.md](docs/sdo_guide.md) - SDO/AIA/HMI data processing
- [config_guide.md](docs/config_guide.md) - Configuration management
- [database_guide.md](docs/database_guide.md) - PostgreSQL utilities
- [transfer_guide.md](docs/transfer_guide.md) - File transfer utilities (HTTP, FTP, SFTP)

---

> **Note (v0.6.0):** `egghouse.io` was retired. Use `astropy.io.fits` for
> FITS and `PIL.Image` (Pillow) for other image formats directly.

---

## egghouse.image

Generic image processing utilities. Organized into submodules:
- `core` - Basic transformations
- `masking` - Circle and annulus masks
- `spatial` - Padding, cropping, flipping
- `filters` - Gaussian, median, edge detection
- `stats` - Normalization, histogram, statistics
- `metrics` - PSNR / SSIM / MS-SSIM / weak-signal
- `transforms` - composable numpy transforms
- `noise` - robust noise-scale estimation (MAD)

### Core

| Function | Signature | Description |
|----------|-----------|-------------|
| `resize_image` | `(image, size, order=1, preserve_range=True) -> ndarray` | Resize image to (height, width), preserves dtype |
| `rotate_image` | `(image, angle, order=1, reshape=False, cval=0, preserve_range=True) -> ndarray` | Rotate image by angle (degrees) |
| `bytescale_image` | `(data, imin=None, imax=None, omin=0, omax=255) -> ndarray` | Scale to uint8 [omin, omax] |

### Masking

| Function | Signature | Description |
|----------|-----------|-------------|
| `circle_mask` | `(image_size, radius, center=None, mask_type='inner') -> ndarray` | Circular boolean mask |
| `annulus_mask` | `(image_size, inner_radius, outer_radius, center=None) -> ndarray` | Ring-shaped boolean mask |

### Binning (v0.5+)

| Function | Signature | Description |
|----------|-----------|-------------|
| `bin_ndarray` | `(array, new_shape, operation='sum') -> ndarray` | Block-wise sum / mean n-D down-binner. Each `new_shape[i]` must evenly divide `array.shape[i]`. |

### Spatial

| Function | Signature | Description |
|----------|-----------|-------------|
| `pad_image` | `(data, target_size, pad_value=0, center=True) -> ndarray` | Pad image to target size |
| `crop_or_pad` | `(data, target_size, pad_value=0, center=True) -> ndarray` | Crop or pad to exact target size |
| `flip_image` | `(image, axis='vertical') -> ndarray` | Flip image ('vertical', 'horizontal', 'both') |
| `roll_image` | `(image, shift_y, shift_x) -> ndarray` | Cyclic roll image by (shift_y, shift_x) |

### Filters

| Function | Signature | Description |
|----------|-----------|-------------|
| `gaussian_smooth` | `(image, sigma=1.0, preserve_range=True) -> ndarray` | Gaussian smoothing filter |
| `median_denoise` | `(image, size=3, preserve_range=True) -> ndarray` | Median filter for noise removal |
| `laplacian_edge` | `(image, mode='reflect') -> ndarray` | Laplacian edge detection (2nd derivative) |
| `sobel_edge` | `(image, axis=None, mode='reflect') -> ndarray` | Sobel edge detection (gradient) |
| `unsharp_mask` | `(image, sigma=1.0, amount=1.0, preserve_range=True) -> ndarray` | Sharpen image via unsharp masking |

### Stats

| Function | Signature | Description |
|----------|-----------|-------------|
| `normalize_image` | `(image, mean=None, std=None) -> ndarray` | Z-score normalization (mean=0, std=1) |
| `get_image_stats` | `(image, mask=None, percentiles=...) -> dict` | Compute image statistics (mean, std, min, max, percentiles) |
| `histogram_equalization` | `(image, nbins=256) -> ndarray` | Enhance contrast via histogram equalization |
| `percentile_scale` | `(image, low=1.0, high=99.0, omin=0, omax=255) -> ndarray` | Scale using percentile clipping |
| `find_disk_center` | `(image, threshold=None, method='centroid') -> Tuple[float, float]` | Find bright disk center (cy, cx) |
| `adaptive_threshold` | `(image, block_size=35, offset=0.0) -> ndarray` | Adaptive binarization for uneven illumination |

### Metrics (v0.9+)

| Function | Signature | Description |
|----------|-----------|-------------|
| `psnr` | `(image, reference, *, data_range=None) -> float` | Peak signal-to-noise ratio (dB); `+inf` on exact match |
| `ssim` | `(image, reference, *, data_range=None, win_size=11) -> float` | Single-scale structural similarity (Wang 2004) |
| `ms_ssim` | `(image, reference, *, data_range=None, weights=None, win_size=11) -> float` | Multi-scale SSIM (Wang 2003, 5-scale) |
| `weak_signal_contrast` | `(image, reference, *, mask=None) -> float` | Sobel gradient-magnitude correlation (weak-edge preservation, placeholder) |
| `pearson_corr` | `(a, b) -> float` | Pearson correlation of two arrays; NaN if either is constant |
| `db_ratio` | `(numerator, denominator) -> float` | Amplitude ratio in dB (`20*log10`); NaN/`+inf` guards for zero scales |

### Augment (paired)

| Function | Signature | Description |
|----------|-----------|-------------|
| `paired_random_crop` | `(a, b, patch, rng) -> (np.ndarray, np.ndarray)` | Same random square crop applied to an (input, target) pair |
| `paired_flip_rot` | `(a, b, rng) -> (np.ndarray, np.ndarray)` | Same random flip + 90° rotation applied to both arrays |

### Transforms (v0.9+)

Composable numpy transform factories; combine with `compose([...])`.

| Function | Signature | Description |
|----------|-----------|-------------|
| `compose` | `(transforms) -> Transform` | Chain transforms left-to-right |
| `to_float32` | `(image) -> ndarray` | Cast to native float32, no rescale |
| `nan_to_value` | `(value=0.0) -> Transform` | Replace NaN/Inf with `value` |
| `percentile_clip` | `(low=0.5, high=99.5) -> Transform` | Clip to per-frame percentile range |
| `normalize_minmax` | `(eps=1e-8) -> Transform` | Scale to [0, 1] per frame |
| `normalize_log1p` | `(scale=1.0) -> Transform` | `log1p(scale*(x-min))` dynamic-range compression |
| `circular_mask` | `(radius_frac, fill=0.0, inverse=False) -> Transform` | Fill inside/outside a centered circle |

### Noise (v0.9+)

| Function | Signature | Description |
|----------|-----------|-------------|
| `mad` | `(x, *, center=None) -> float` | Median absolute deviation about the median |
| `robust_sigma` | `(x, *, center=None) -> float` | Robust noise sigma `1.4826 * MAD(x)` |

### Aliases

| Alias | Function |
|-------|----------|
| `resize` | `resize_image` |
| `rotate` | `rotate_image` |
| `bytescale` | `bytescale_image` |
| `pad` | `pad_image` |

---

## egghouse.denoise (v0.9+)

Classical, channel-agnostic image denoisers. Each module exposes a
`denoise(image, ...)` callable plus a parametric `*Denoiser` class (both
satisfy `Callable[[np.ndarray], np.ndarray]`). Requires
`pip install egghouse[denoise]` (scikit-image, PyWavelets, bm3d).

| Module | Function | Class | Notes |
|--------|----------|-------|-------|
| `wavelet` | `denoise(image, sigma=None, ...)` | `WaveletDenoiser` | BayesShrink (scikit-image); sigma auto-estimated |
| `bm3d` | `denoise(image, sigma=None)` | `BM3DDenoiser` | Block-Matching 3D |
| `nlm` | `denoise(image, sigma=None, ...)` | `NLMDenoiser` | Non-local means (scikit-image) |
| `tv` | `denoise(image, weight=0.1)` | `TVDenoiser` | Total variation (Chambolle) |
| `wiener` | `denoise(image, mysize=...)` | `WienerDenoiser` | Wiener filter (scipy) |
| `anscombe` | `forward(x)` / `inverse(z)` / `denoise(image, inner, ...)` | `AnscombeDenoiser` | Poisson variance stabilisation around an inner Gaussian denoiser |

```python
from egghouse.denoise.wavelet import WaveletDenoiser
clean = WaveletDenoiser()(noisy)
```

---

## egghouse.sdo

SDO/AIA and SDO/HMI data processing utilities.

### AIA

| Function | Signature | Description |
|----------|-----------|-------------|
| `aia_intscale` | `(data, wavelength, ...) -> ndarray` | Wavelength-specific intensity scaling (94-6173 Å) |
| `get_aia_calibration` | `(wavelength) -> dict` | Get calibration parameters for wavelength |

### HMI

| Function | Signature | Description |
|----------|-----------|-------------|
| `hmi_intscale` | `(data, vmin=-1000, vmax=1000, ...) -> ndarray` | Magnetogram scaling to uint8 |
| `hmi_field_strength` | `(bx, by, bz) -> ndarray` | Calculate vector field strength |

### Core

| Function | Signature | Description |
|----------|-----------|-------------|
| `parse_fits_header` | `(filepath) -> dict` | Extract SDO FITS header keywords |
| `validate_sdo_image` | `(image, expected_shape=(4096,4096)) -> bool` | Validate SDO image dimensions |
| `get_solar_disk_params` | `(header) -> dict` | Calculate solar disk center/radius from header |

### Level 1.5

| Function | Signature | Description |
|----------|-----------|-------------|
| `to_level15` | `(fits_file, instrument=None, target_plate_scale=None, target_size=4096, order=3, missing=0.0) -> Map` | Convert Level 1.0 to 1.5 (north-up, centered) |
| `batch_to_level15` | `(fits_files, output_dir, instrument=None, overwrite=False, progress_callback=None, **kwargs) -> List[str]` | Batch Level 1.5 conversion |
| `get_level_info` | `(fits_file) -> dict` | Get processing level info from FITS |

### AIA Level 1.0 → 1.5 prep stages (v0.5+)

aiapy-backed wrappers for the prep stages not handled by `to_level15`.
All are no-ops for wavelengths outside the AIA EUV + 304 Å set so they
are safe to apply across heterogeneous batches. `aiapy` and `sunpy`
are imported only inside the bodies; merely importing
`egghouse.sdo.prep` does not require either.

| Function | Signature | Description |
|----------|-----------|-------------|
| `aia_update_pointing` | `(sdo_map, pointing_table=None) -> Map` | Refresh WCS keywords against the JSOC master pointing table. |
| `aia_respike` | `(sdo_map, spikes=None) -> Map` | Re-inject spike pixels removed by the Level 1 pipeline. When `spikes` is `None`, `aiapy.calibrate.fetch_spikes` is called (one JSOC round-trip per record). |
| `aia_correct_degradation` | `(sdo_map, correction_table=None) -> Map` | Time-dependent effective-area correction. Returns input unchanged for non-AIA wavelengths. |
| `aia_deconvolve` | `(sdo_map, psfs=None) -> Map` | PSF deconvolution. Pre-compute PSFs via `cached_aia_psfs` for batch jobs (PSF computation is ~minutes per channel). Returns input unchanged for non-AIA wavelengths. |
| `cached_aia_psfs` | `(path, *, wavelengths=(94,131,171,193,211,304,335)) -> dict[int, ndarray]` | Pickle-cached `{wavelength: PSF}` dict. Rebuilds if the cache does not cover all requested wavelengths. |
| `mask_out_of_disk` | `(sdo_map, *, fill_value=-5000.0) -> Map` | Off-disk pixels set to `fill_value`. Reads disk radius from `R_SUN` or `RSUN_OBS / CDELT1`; raises `KeyError` if neither is available. |

### JSOC (drms-based data acquisition; v0.4+)

Soft dependency on the `drms` package — imported inside the functions
that need it.

| Function | Signature | Description |
|----------|-----------|-------------|
| `jsoc_export` | `(query, *, email, method='url', protocol='fits', client=None) -> list[str]` | Submit a JSOC DRMS export request; block until staging completes; return staged URLs. Raises `RuntimeError` if the request did not succeed. |
| `aia_euv_query` | `(times, *, wavelengths=AIA_DEM_WAVELENGTHS, series=AIA_LEV1_EUV_SERIES, tolerance=timedelta(seconds=12)) -> str` | Compose a DRMS record-set string selecting AIA EUV records near each `datetime` in `times`, filtered to the given wavelengths. Multiple times are concatenated into one export. |
| `cached_correction_table` | `(path) -> astropy.table.Table` | Pickle-cached `aiapy.calibrate.util.get_correction_table()`. First call fetches and writes; subsequent calls deserialize. |
| `cached_pointing_table` | `(path, *, start, end) -> astropy.table.Table` | Pickle-cached `aiapy.calibrate.util.get_pointing_table(start, end)`. Cache file is reused as-is; delete to refresh. |

**Constants:** `AIA_LEV1_EUV_SERIES = "aia.lev1_euv_12s"`.

### Stacking

| Class/Function | Signature | Description |
|----------------|-----------|-------------|
| `Stacking` | `(nb_stack, method='mean', latitude_deg=0, ...)` | Solar rotation-corrected stacking |
| `StreamingStackAccumulator` | `(method, ...)` | Memory-efficient streaming accumulator |
| `stack_with_rotation_correction` | `(maps, ...) -> ndarray` | Stack with rotation correction |
| `solar_rotation_shift` | `(delta_t, latitude_deg, plate_scale) -> float` | Calculate pixel shift for solar rotation |
| `snodgrass_rotation_rate` | `(latitude_deg) -> float` | Snodgrass (1983) differential rotation rate |
| `detect_cadence_from_maps` | `(maps) -> float` | Detect time cadence from Map sequence |
| `cross_correlate_shift` | `(ref, target, ...) -> Tuple[float, float]` | Sub-pixel shift via phase correlation |

### Quality

| Function | Signature | Description |
|----------|-----------|-------------|
| `decode_quality` | `(quality, instrument='AIA') -> List[dict]` | Decode QUALITY flag into list of issues |
| `format_quality` | `(quality, instrument='AIA', verbose=True) -> str` | Format QUALITY flag as readable string |
| `is_quality_ok` | `(quality, strict=False, ignore_bits=None) -> bool` | Check if data quality is acceptable |
| `get_quality_summary` | `(quality, instrument='AIA') -> dict` | Get structured quality summary |
| `print_all_quality_bits` | `(instrument='AIA') -> None` | Print all quality bit definitions |

### DEM (Differential Emission Measure)

SITES algorithm implementation for DEM inversion from multi-wavelength AIA observations.

**Temperature response source (v0.3+):** aiapy 0.12 removed
`Channel.temperature_response`, so the canonical CHIANTI-based response is now
loaded from an SSW `aia_get_response.pro` `.npz` archive (e.g. the one shipped
with demregpy). See `load_ssw_temperature_response` below. The built-in
Gaussian fallback in `get_temperature_response` is **not** suitable for
research-quality DEM analysis.

**Response Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `get_temperature_response` | `(wavelengths=None, temperatures=None, time=None, include_degradation=True, ssw_table_path=None, ssw_response_key='response_v10_en') -> ndarray` | Get AIA temperature response K(T). Pass `ssw_table_path` to load from an SSW `.npz`. Without it, raises `NotImplementedError` when aiapy is installed (aiapy 0.12+ no longer supplies K(T) directly), or falls back to a Gaussian approximation with a `UserWarning` when aiapy is not installed. |
| `load_ssw_temperature_response` | `(path, *, log_temperatures=None, wavelengths=None, response_key='response_v10_en') -> ndarray` | Load and interpolate K(T) from an SSW `aia_get_response` `.npz` archive. Archive keys: `log_temperature`, `channels`, and one or more response arrays (`response_v9_en`, `response_v10_en`, `response_v10_en_nb`). Returns shape `(n_temperatures, n_wavelengths)`. Linear interpolation in log T; refuses extrapolation. |
| `get_default_temperatures` | `(logt_min=5.5, logt_max=7.5, n_bins=100) -> ndarray` | Get default log-spaced temperature grid |

**SITES Algorithm:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `dem_sites` | `(intensities, errors, response, temperatures, max_iter=100, tol=1e-3, ...) -> Tuple[ndarray, dict]` | Multi-wavelength DEM inversion using SITES |
| `dem_sites_pixel` | `(intensities, errors, response, temperatures, ...) -> Tuple[ndarray, dict]` | Single-pixel DEM inversion interface |

**Map Processing:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `dem_map` | `(image_cube, error_cube, response, temperatures, mask=None, chunk_size=512, ...) -> Tuple[ndarray, dict]` | Full-map DEM with chunked processing |
| `compute_dem_errors` | `(dem, intensities, errors, response, temperatures, n_monte_carlo=100) -> ndarray` | Monte Carlo DEM error estimation |

**Derived Quantities:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `get_emission_measure` | `(dem, temperatures, t_min=None, t_max=None) -> float or ndarray` | Compute total emission measure from DEM |
| `get_mean_temperature` | `(dem, temperatures, weight='dem') -> float or ndarray` | Compute DEM-weighted mean temperature |

**Constants:**

| Name | Value | Description |
|------|-------|-------------|
| `HAS_AIAPY` | bool | True if aiapy is available |

### Constants

| Name | Value | Description |
|------|-------|-------------|
| `AIA_PLATE_SCALE` | 0.6 | AIA plate scale (arcsec/px) |
| `HMI_PLATE_SCALE` | 0.6 | HMI plate scale (arcsec/px) |
| `SDO_IMAGE_SIZE` | 4096 | Standard SDO image size |
| `SOLAR_ROTATION_PERIOD` | 25.38 | Sidereal rotation period (days) |
| `SNODGRASS_A` | 14.713 | Snodgrass coefficient A |
| `SNODGRASS_B` | -2.396 | Snodgrass coefficient B |
| `SNODGRASS_C` | -1.787 | Snodgrass coefficient C |
| `HMI_CADENCE_45S` | 45.0 | HMI 45-second cadence |
| `HMI_CADENCE_720S` | 720.0 | HMI 720-second cadence |
| `AIA_CALIBRATION` | dict | AIA wavelength calibration data |
| `AIA_QUALITY_BITS` | dict | AIA QUALITY bit definitions |
| `HMI_QUALITY_BITS` | dict | HMI QUALITY bit definitions |
| `QUALLEV0_BITS` | dict | Level 0 quality bit definitions |
| `HAS_ASTROPY` | bool | True if astropy available |
| `HAS_SUNPY` | bool | True if sunpy available |

---

## egghouse.config

ML/DL configuration management.

### Classes

| Class | Description |
|-------|-------------|
| `BaseConfig` | Dataclass-based configuration with multiple loading methods |

### BaseConfig Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `from_yaml` | `(path) -> T` | Load from YAML file |
| `from_json` | `(path) -> T` | Load from JSON file |
| `from_env` | `(prefix="") -> T` | Load from environment variables |
| `from_args` | `(args=None) -> T` | Load from CLI arguments |
| `to_yaml` | `(path) -> None` | Save to YAML file |
| `to_json` | `(path, indent=2) -> None` | Save to JSON file |

---

## egghouse.database

PostgreSQL database utilities.

### Classes

| Class | Description |
|-------|-------------|
| `PostgresManager` | Simplified PostgreSQL operations |

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `load_config` | `(path) -> dict` | Load database config from YAML/JSON |
| `from_dict` | `(d) -> dict` | Create config from dictionary |
| `create_example_config` | `(path) -> None` | Create example config file |
| `to_dataframe` | `(results, columns) -> DataFrame` | Convert query results to pandas DataFrame |

### Declarative schema (v0.7+)

Instrument-blind: any declarative `schema_config` builds exactly those
tables. Pure builders need no DB connection.

| Function | Signature | Description |
|----------|-----------|-------------|
| `build_create_table_sql` | `(table, table_spec) -> str` | Pure: spec → CREATE TABLE (composite `_primary_key` / `_unique`) |
| `build_index_sql` | `(table, indexes) -> list[str]` | Pure: `_indexes` → CREATE INDEX statements |
| `split_schema_meta` | `(table_spec) -> (columns, pk, unique, indexes)` | Pure, non-mutating metadata split |
| `create_tables_from_schema` | `(db_config, schema_config, *, drop=False, verbose=False) -> dict` | Create tables; returns `{table: created\|recreated\|skipped}` |
| `create_database` | `(db_config, *, verbose=False) -> bool` | Idempotent CREATE DATABASE via admin connection |
| `initialize_database` | `(db_config, schema_config, *, verbose=False) -> dict` | create_database + create_tables_from_schema |

### Bulk records (v0.7+)

| Function | Signature | Description |
|----------|-----------|-------------|
| `normalize_records` | `(df) -> list[dict]` | DataFrame → rows, lowercased cols, NaN→None |
| `build_upsert_sql` | `(table, columns, conflict_columns) -> str` | Pure INSERT … ON CONFLICT DO NOTHING (composite-aware) |
| `upsert_dataframe` | `(df, table, db_config, *, conflict_columns='datetime', batch=1000) -> int` | Idempotent bulk upsert; returns inserted count |
| `find_orphans` | `(file_paths) -> list[str]` | Paths no longer on disk |
| `delete_orphans` | `(table, db_config, *, file_column='file_path') -> int` | Delete rows whose file is gone |

### PostgresManager Methods

| Method | Description |
|--------|-------------|
| `insert(table, data)` | Insert row(s) into table |
| `select(table, columns=None, where=None)` | Query table with optional conditions |
| `select_date_range(table, date_column, start, end)` | Query by date range |
| `update(table, data, where)` | Update rows matching condition |
| `delete(table, where)` | Delete rows matching condition |
| `upsert(table, data, conflict_columns, update_columns=None)` | Insert or update on conflict |
| `execute(query, params=None)` | Execute raw SQL |
| `to_dataframe(results, columns)` | Convert results to DataFrame |

---

## egghouse.swdb (v0.8+)

Solar / space-weather DB domain layer on top of `egghouse.database`.
AIA-only; other instruments subclass `FitsHandler` per project.

### Reference schemas

| Constant | Description |
|----------|-------------|
| `SDO_SCHEMA` | `(telescope, channel, datetime)` PK, `UNIQUE(file_path)`. Byte-identical to setup-sw-db's `sdo`. |
| `LASCO_SCHEMA` | `(camera, datetime)` PK. |
| `SECCHI_SCHEMA` | `(datatype, spacecraft, instrument, channel, datetime)` PK. |

### Classes / Functions

| Name | Signature | Description |
|------|-----------|-------------|
| `ValidationResult` | dataclass; `.ok(metadata, file_path)` / `.fail(error, file_path)` | Type-safe FITS validation result |
| `FitsHandler` | ABC: `extract_metadata`, `to_db_record`, `target_dir` | Instrument FITS handler interface |
| `AiaFitsHandler` | `(*, check_data=False, require_quality_zero=False)` | SDO/AIA handler; keys on `T_OBS` (UTC) |
| `scan_fits` | `(scan_dir, *, pattern='*.fits', exclude_substrings=('spike',)) -> list[Path]` | Recursive FITS listing |
| `register_fits_dir` | `(scan_dir, *, handler, table, db_config, conflict_columns, move_root=None, error_dirs=None, pattern='*.fits', exclude_substrings=('spike',), parallel=1, batch_size=1000, verbose=False) -> RegisterReport` | Scan → validate → idempotent upsert → optional archive |
| `RegisterReport` | dataclass; `.summary()` | Counts: scanned/valid/inserted/skipped_existing/errors (reconcile) |

### SWPC real-time parsers (v0.11+)

Pure pandas parsers (no DB) for NOAA SWPC real-time JSON products, in
`egghouse.swdb.swpc`; re-exported from `egghouse.swdb`.

| Name | Signature | Description |
|------|-----------|-------------|
| `parse_xray` | `(data: list) -> pd.DataFrame` | GOES X-ray flux → `rt_goes_xray` (short/long bands pivoted per satellite/time) |
| `parse_proton` | `(data: list) -> pd.DataFrame` | GOES integral proton flux → `rt_goes_proton` (long format per energy threshold) |
| `parse_solar_wind` | `(data: list, kind: str, source='DSCOVR') -> pd.DataFrame` | L1 solar wind 'plasma'/'mag' header-row product → `rt_*` |
| `parse_kp_1m` | `(data: list) -> pd.DataFrame` | Estimated 1-min planetary K index → `rt_kp` |
| `parse_kp_forecast` | `(data: list) -> pd.DataFrame` | 3-hourly Kp forecast → `rt_kp_forecast` |
| `parse_solar_probabilities` | `(data: list) -> pd.DataFrame` | C/M/X flare + 10 MeV proton probabilities → `swpc_solar_probabilities` |
| `parse_alerts` | `(data: list) -> pd.DataFrame` | Alerts/watches/warnings → `swpc_alerts` |
| `parse_3day_forecast` | `(text: str) -> pd.DataFrame` | 3-day forecast text → `swpc_3day_forecast` (raw + issue time) |

---

## egghouse.transfer

File transfer utilities for HTTP, FTP, and SFTP protocols.

### HTTP

| Function | Signature | Description |
|----------|-----------|-------------|
| `download_single_file` | `(source_url, destination, overwrite=False, max_retries=3, timeout=30, verify_ssl=True) -> bool` | Download single file with retry |
| `get_file_list` | `(base_url, extensions, timeout=30, verify_ssl=True) -> List[str]` | Scrape file links from directory listing |
| `download_parallel` | `(download_tasks, overwrite=False, max_retries=3, parallel=1, timeout=30, verify_ssl=True) -> Dict[str, int]` | Parallel download with ThreadPoolExecutor |

### FTP (no external dependencies)

| Function | Signature | Description |
|----------|-----------|-------------|
| `ftp_connection` | `(host, port=21, user='anonymous', password='', timeout=30, passive=True) -> ContextManager[FTP]` | Context manager for FTP connection |
| `ftp_download_file` | `(ftp, remote_path, local_path, overwrite=False) -> bool` | Download single file via FTP |
| `ftp_upload_file` | `(ftp, local_path, remote_path, overwrite=False) -> bool` | Upload single file via FTP |
| `ftp_list_files` | `(ftp, remote_dir='.', extensions=None) -> List[str]` | List files in remote directory |
| `ftp_download_parallel` | `(host, download_tasks, port=21, user='anonymous', password='', overwrite=False, max_retries=3, parallel=1, timeout=30, passive=True) -> Dict[str, int]` | Parallel FTP download |
| `ftp_upload_parallel` | `(host, upload_tasks, port=21, user='anonymous', password='', overwrite=False, max_retries=3, parallel=1, timeout=30, passive=True) -> Dict[str, int]` | Parallel FTP upload |

### SFTP (requires paramiko)

| Function | Signature | Description |
|----------|-----------|-------------|
| `sftp_connection` | `(host, port=22, user=None, password=None, key_file=None, timeout=30) -> ContextManager[SFTPClient]` | Context manager for SFTP connection |
| `sftp_download_file` | `(sftp, remote_path, local_path, overwrite=False) -> bool` | Download single file via SFTP |
| `sftp_upload_file` | `(sftp, local_path, remote_path, overwrite=False) -> bool` | Upload single file via SFTP |
| `sftp_list_files` | `(sftp, remote_dir='.', extensions=None) -> List[str]` | List files in remote directory |
| `sftp_download_parallel` | `(host, download_tasks, port=22, user=None, password=None, key_file=None, overwrite=False, max_retries=3, parallel=1, timeout=30) -> Dict[str, int]` | Parallel SFTP download |
| `sftp_upload_parallel` | `(host, upload_tasks, port=22, user=None, password=None, key_file=None, overwrite=False, max_retries=3, parallel=1, timeout=30) -> Dict[str, int]` | Parallel SFTP upload |

### Constants

| Name | Description |
|------|-------------|
| `HAS_HTTP` | `True` if requests/beautifulsoup4 are available for HTTP |
| `HAS_PARAMIKO` | `True` if paramiko is available for SFTP |
