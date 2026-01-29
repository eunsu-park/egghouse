# API Reference

Complete list of all public functions, classes, and constants in egghouse.

---

## egghouse.io

File I/O utilities for scientific data formats.

### FITS (requires astropy)

| Function | Signature | Description |
|----------|-----------|-------------|
| `read_fits` | `(filepath, hdu_index=0) -> Tuple[ndarray, dict]` | Read FITS data and header |
| `write_fits` | `(filepath, data, header=None, overwrite=False) -> None` | Write numpy array to FITS |
| `read_fits_header` | `(filepath, hdu_index=0) -> dict` | Read header only (no data loading) |
| `append_fits` | `(filepath, data, header=None) -> None` | Append HDU extension to existing FITS |

### BMP (no external dependencies)

| Function | Signature | Description |
|----------|-----------|-------------|
| `read_bmp` | `(filepath) -> Tuple[ndarray, dict]` | Read BMP as (H,W,3) uint8 RGB + info |
| `write_bmp` | `(filepath, data, overwrite=False) -> None` | Write (H,W) or (H,W,3) uint8 to BMP |
| `read_bmp_header` | `(filepath) -> dict` | Read BMP header only |

### Constants

| Name | Description |
|------|-------------|
| `HAS_ASTROPY` | `True` if astropy is available |

---

## egghouse.image

Generic image processing utilities.

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `resize_image` | `(image, size, order=1, preserve_range=True) -> ndarray` | Resize image to (height, width), preserves dtype |
| `rotate_image` | `(image, angle, order=1, reshape=False, cval=0, preserve_range=True) -> ndarray` | Rotate image by angle (degrees) |
| `bytescale_image` | `(data, imin=None, imax=None, omin=0, omax=255) -> ndarray` | Scale to uint8 [omin, omax] |
| `circle_mask` | `(image_size, radius, center=None, mask_type='inner') -> ndarray` | Circular boolean mask |
| `annulus_mask` | `(image_size, inner_radius, outer_radius, center=None) -> ndarray` | Ring-shaped boolean mask |
| `pad_image` | `(data, target_size, pad_value=0, center=True) -> ndarray` | Pad image to target size |
| `crop_or_pad` | `(data, target_size, pad_value=0, center=True) -> ndarray` | Crop or pad to exact target size |

### Aliases

| Alias | Function |
|-------|----------|
| `resize` | `resize_image` |
| `rotate` | `rotate_image` |
| `bytescale` | `bytescale_image` |
| `pad` | `pad_image` |

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

## egghouse.transfer

HTTP file download utilities.

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `download_single_file` | `(url, output_path, ...) -> bool` | Download single file with retry |
| `get_file_list` | `(url, extension=None, ...) -> List[str]` | Scrape file links from directory listing |
| `download_parallel` | `(urls, output_dir, max_workers=4, ...) -> List[str]` | Parallel download with ThreadPoolExecutor |
