# egghouse

A personal arsenal of reusable code for solar physics research.

이 레포지터리는 개인 프로젝트들에서 공통으로 사용하는 유틸리티를 모아둔 라이브러리입니다.

> **Shared common layer.** `egghouse` is the place where functionality common
> to the sibling projects — the **SOLARIS** program (`solaris-data` acquisition + DB,
> `solaris-denoise` noise-removal models), **`lolipop`** (low-light denoising research),
> and **`undine`** (DEM) — is implemented once and consumed via `import`. New generic primitives (image metrics, denoisers, transforms,
> noise-scale estimation, config, DB/transfer helpers) belong here, not
> duplicated in the consumers. `egghouse.config.BaseConfig` is the standard
> mechanism those projects use to read paths/params from env/YAML/CLI.

---

## Installation

```bash
pip install git+https://github.com/eunsu-park/egghouse.git
```

For development:
```bash
git clone https://github.com/eunsu-park/egghouse.git
cd egghouse
pip install -e .
```

### Dependencies

| Module | Required | Optional |
|--------|----------|----------|
| Core | numpy, scipy, pandas | - |
| image | numpy, scipy | - |
| denoise | numpy, scipy | scikit-image, PyWavelets, bm3d |
| sdo | numpy, scipy | astropy, sunpy, aiapy |
| dem | numpy, scipy | fiasco (CHIANTI temperature response) |
| config | pyyaml | - |
| database | psycopg2-binary | pyyaml |
| transfer | requests, beautifulsoup4 | paramiko (SFTP) |

> `egghouse.io` was removed in v0.6.0. Use `astropy.io.fits` for FITS
> and `PIL.Image` (Pillow) for other image formats directly.

---

## Quick Start

```python
# 1. 이미지 전처리 (resize, mask, scaling)
from egghouse.image import resize_image, circle_mask, bytescale_image
resized = resize_image(data, (1024, 1024))
mask = circle_mask(4096, radius=1600)
display = bytescale_image(data, imin=0, imax=5000)

# 2. SDO Level 1.5 변환
from egghouse.sdo import to_level15
level15_map = to_level15('aia_lev1.fits')  # auto-detect AIA/HMI

# 2b. JSOC export (drms-based AIA data acquisition; v0.4+)
from datetime import datetime
from egghouse.sdo import jsoc_export, aia_euv_query
q = aia_euv_query([datetime(2014, 1, 1, 12, 0, 0)])
urls = jsoc_export(q, email='you@example.com')

# 2c. AIA Level 1 -> 1.5 prep stages (v0.5+)
from egghouse.sdo import (
    aia_update_pointing, aia_respike,
    aia_correct_degradation, aia_deconvolve,
    cached_aia_psfs, mask_out_of_disk,
)
# Compose at the call site; pass cached tables/PSFs for batch jobs.

# 3. DEM (Differential Emission Measure) 분석
# DEM solvers are instrument-agnostic and live in `egghouse.dem`; the AIA
# temperature response (instrument-specific) is in `egghouse.sdo`.
from egghouse.dem import dem_sites, get_default_temperatures
from egghouse.sdo import get_temperature_response
temps = get_default_temperatures(n_bins=50)
# AIA response: CHIANTI (fiasco) + aiapy wavelength response, or an SSW
# aia_get_response .npz table (e.g. one shipped with demregpy).
response = get_temperature_response(temperatures=temps, ssw_table_path='response_matrix.npz')
dem, info = dem_sites(intensities, errors, response, temps)
```

---

## Modules

### `egghouse.image`
Generic image processing utilities. Submodules: `core` (resize/rotate/bytescale),
`masking`, `spatial`, `filters`, `stats`, plus `metrics`, `transforms`, `noise`.

```python
from egghouse.image import resize_image, rotate_image, bytescale_image
from egghouse.image import circle_mask, annulus_mask, pad_image, crop_or_pad

# Resize/Rotate (preserves dtype)
resized = resize_image(image, (1024, 1024), order=1)
rotated = rotate_image(image, angle=45, reshape=False)

# Scale to uint8 for display
display = bytescale_image(data, imin=0, imax=5000)

# Masks for solar disk analysis
disk_mask = circle_mask(4096, radius=1600)
corona_mask = annulus_mask(4096, inner_radius=1600, outer_radius=2000)

# Size normalization
padded = pad_image(image, (4096, 4096), pad_value=0)
normalized = crop_or_pad(image, (512, 512))

# Image-quality metrics (no torch dependency)
from egghouse.image import psnr, ssim, ms_ssim, weak_signal_contrast
score = psnr(denoised, reference, data_range=4.0)

# Noise-scale estimation
from egghouse.image import mad, robust_sigma, gaussian_core_sigma
sigma = robust_sigma(frame)               # 1.4826 * MAD (outlier-robust)
sigma = gaussian_core_sigma(magnetogram)  # Gaussian fit to the low-field histogram
                                          # core (Liu 2012 magnetogram-noise method)

# Signal-dependent (Poisson–Gaussian) noise from two same-scene frames, e.g. AIA:
from egghouse.image import poisson_gaussian_noise, photon_transfer_fit
fit = poisson_gaussian_noise(frame_a, frame_b)   # σ²(I) = g·I + r²
g, r2, r_squared = fit.g, fit.r2, fit.r_squared  # VST (generalized Anscombe) params

# Composable numpy transforms
from egghouse.image import transforms as T
pipe = T.compose([T.to_float32, T.percentile_clip(0.5, 99.5), T.normalize_minmax()])
out = pipe(frame)
```

### `egghouse.denoise`
Classical, channel-agnostic image denoisers. Each module exposes a
`denoise(image, ...)` callable plus a parametric `*Denoiser` class.
Install the extra deps with `pip install egghouse[denoise]`.

```python
from egghouse.denoise.wavelet import WaveletDenoiser   # also: bm3d, nlm, tv, wiener
from egghouse.denoise import anscombe

denoise = WaveletDenoiser()        # BayesShrink; sigma auto-estimated
clean = denoise(noisy)

# Anscombe variance-stabilisation wrapping any inner Gaussian denoiser
clean = anscombe.denoise(noisy_counts, WaveletDenoiser())
```

### `egghouse.sdo`
SDO/AIA and SDO/HMI data processing utilities.

```python
from egghouse.sdo import aia_intscale, hmi_intscale, to_level15, Stacking

# AIA intensity scaling
scaled = aia_intscale(aia_data, wavelength=171)

# HMI magnetogram scaling
mag_scaled = hmi_intscale(hmi_data, vmin=-500, vmax=500)

# Level 1.0 to Level 1.5 conversion (north-up, centered, 4096x4096)
level15_map = to_level15('aia_lev1.fits')  # auto-detect instrument
level15_map = to_level15('hmi_lev1.fits', instrument='HMI')

# Solar rotation-corrected stacking (with differential rotation)
stacker = Stacking(nb_stack=21, method='mean', latitude_deg=30)
mean_image = stacker.run(fits_files)

# Snodgrass differential rotation model
from egghouse.sdo import snodgrass_rotation_rate
rate_eq = snodgrass_rotation_rate(0)    # 14.71 deg/day at equator
rate_60 = snodgrass_rotation_rate(60)   # ~12.0 deg/day at 60° latitude

# Snodgrass differential rotation is shown above; DEM lives in egghouse.dem.
```

### `egghouse.dem`
Instrument-agnostic DEM (Differential Emission Measure) inversion: eight
solvers + generic temperature-response tools. Solvers take
`(intensities, errors, response, temperatures)` and are not tied to any
instrument — the AIA temperature response is built in `egghouse.sdo`
(`get_temperature_response`) and passed in.

```python
from egghouse.dem import (
    dem_sites,        # Morgan & Pickering 2019 (multiplicative MART)
    dem_nnls,         # Tikhonov-NNLS (Lawson & Hanson; Hannah & Kontar 2012)
    dem_regularized,  # Hannah & Kontar 2012 (GSVD-Tikhonov)
    dem_sparse,       # Cheung et al. 2015 (basis pursuit)
    dem_plowman,      # Plowman et al. 2013 (fast linear)
    dem_mcmc,         # Kashyap & Drake 1998 (MCMC + uncertainties)
    dem_spline,       # Weber et al. 2004 (spline forward-fit)
    dem_gaussian,     # Aschwanden et al. 2013 (single-Gaussian)
    dem_map,          # chunked full-image driver (method="sites"|"nnls"|...)
    get_default_temperatures, load_ssw_temperature_response,
    get_emission_measure, get_mean_temperature,
)
from egghouse.sdo import get_temperature_response  # AIA K(T) (instrument-specific)

temps = get_default_temperatures(logt_min=5.5, logt_max=7.5, n_bins=50)

# AIA temperature response K(T): CHIANTI (fiasco) + aiapy wavelength response,
# or an SSW aia_get_response .npz table (e.g. the one shipped with demregpy).
response = get_temperature_response(temperatures=temps, ssw_table_path='response_matrix.npz')

# Single-pixel inversion (every solver shares this signature)
intensities = [10.0, 50.0, 200.0, 150.0, 80.0, 20.0]  # 6 AIA channels (94-335 Å)
errors = [i * 0.1 for i in intensities]
dem, info = dem_nnls(intensities, errors, response, temps)

# Full-map DEM with chunked processing + solver choice
dem_cube, map_info = dem_map(image_cube, error_cube, response, temps, method="nnls")

# Derived quantities
em = get_emission_measure(dem, temps)        # total emission measure
t_mean = get_mean_temperature(dem, temps)    # DEM-weighted mean temperature
```

### `egghouse.sdo` — AIA Level 1 → 1.5 prep stages (v0.5+)

`to_level15` (above) handles rotation, resampling, and padding for both
AIA and HMI. For full AIA preparation you typically also want some of
these aiapy-backed stages, exposed individually so callers can compose
exactly what they need:

```python
from egghouse.sdo import (
    to_level15,
    aia_update_pointing, aia_respike,
    aia_correct_degradation, aia_deconvolve,
    cached_correction_table, cached_pointing_table, cached_aia_psfs,
    mask_out_of_disk,
)
import sunpy.map
import datetime as dt

# Pre-fetch slow tables once per batch.
correction = cached_correction_table('/cache/correction_table.pkl')
pointing = cached_pointing_table(
    '/cache/pointing_table.pkl',
    start=dt.datetime(2014, 1, 1),
    end=dt.datetime(2014, 1, 2),
)
psfs = cached_aia_psfs('/cache/aia_psfs.pkl')  # ~minutes on first call

m = sunpy.map.Map('aia.lev1.fits')
m = aia_update_pointing(m, pointing_table=pointing)
m = aia_respike(m)                                    # JSOC fetch per record
m = aia_deconvolve(m, psfs=psfs)
m = aia_correct_degradation(m, correction_table=correction)
m = to_level15(m)                                     # rotation + resample + pad
m = mask_out_of_disk(m, fill_value=-5000.0)
```

### `egghouse.image` — n-D binning (v0.5+)

```python
import numpy as np
from egghouse.image import bin_ndarray
m = np.arange(100).reshape((10, 10))
bin_ndarray(m, (5, 5), operation='sum')    # 2x2 block sum
bin_ndarray(m, (5, 5), operation='mean')   # 2x2 block mean
```

### `egghouse.sdo` — JSOC export (drms-based)

For data acquisition from the Joint Science Operations Center (JSOC).
The `drms` package is a soft dependency: imported only when the JSOC
functions are actually called.

```python
from datetime import datetime, timedelta
from egghouse.sdo import (
    jsoc_export, aia_euv_query,
    cached_correction_table, cached_pointing_table,
)

# Compose a record-set selecting AIA EUV records near specific timestamps,
# filtered to a given set of channels. Multiple timestamps are batched
# into a single export request.
times = [datetime(2014, 1, 1, h, 0, 0) for h in range(0, 24, 6)]
query = aia_euv_query(
    times,
    wavelengths=[94, 131, 171, 193, 211, 335],
    tolerance=timedelta(seconds=12),
)
urls = jsoc_export(query, email='you@example.com')

# Download the URLs with retry / atomic-write semantics:
from egghouse.transfer import download_parallel
tasks = [(u, f'/data/{u.rsplit("/", 1)[-1]}') for u in urls]
download_parallel(tasks, parallel=4)

# Memoize the slow aiapy calibration tables so batch Level 1.5 jobs do
# not re-hit JSOC on every record.
correction = cached_correction_table('/cache/correction_table.pkl')
pointing = cached_pointing_table(
    '/cache/pointing_table.pkl',
    start=datetime(2014, 1, 1),
    end=datetime(2014, 1, 2),
)
```

### `egghouse.config`
ML/DL configuration management with dataclass support.

```python
from dataclasses import dataclass
from egghouse.config import BaseConfig

@dataclass
class TrainConfig(BaseConfig):
    lr: float = 0.001
    epochs: int = 100
    batch_size: int = 32

# Load from various sources
config = TrainConfig.from_yaml('train_config.yaml')
config = TrainConfig.from_json('train_config.json')
config = TrainConfig.from_env(prefix="TRAIN_")
config = TrainConfig.from_args()  # CLI: --config base.yaml --lr 0.0001

# Save
config.to_yaml('output_config.yaml')
config.to_json('output_config.json')
```

### `egghouse.database`
PostgreSQL database utilities with simplified CRUD operations.

```python
from egghouse.database import PostgresManager, load_config

config = load_config('database.yaml')
db = PostgresManager(**config)

# Insert
db.insert('observations', {'date': '2024-01-01', 'wavelength': 171})

# Upsert (insert or update on conflict)
db.upsert('observations',
          {'filepath': '/data/aia.fits', 'wavelength': 171, 'processed': True},
          conflict_columns='filepath',
          update_columns=['processed'])

# Query
results = db.select('observations', where={'wavelength': 171})
df = db.to_dataframe(results, columns=['id', 'date', 'wavelength'])
```

Declarative schema + bulk records (v0.7+): build tables from a config
dict and idempotently upsert, no DB-specific code required.

```python
from egghouse.database import initialize_database, upsert_dataframe

schema = {
    "sdo": {
        "telescope": "VARCHAR(10) NOT NULL",
        "channel":   "VARCHAR(20) NOT NULL",
        "datetime":  "TIMESTAMP NOT NULL",
        "file_path": "VARCHAR(512) NOT NULL",
        "_primary_key": ["telescope", "channel", "datetime"],
        "_unique":      ["file_path"],
        "_indexes":     [["datetime"]],
    },
}
initialize_database(db_config, schema)            # CREATE DATABASE + tables
upsert_dataframe(df, "sdo", db_config,
                 conflict_columns=["telescope", "channel", "datetime"])
```

### `egghouse.swdb` — solar / space-weather DB (v0.8+)

Domain layer on top of `egghouse.database`. Reference schemas + FITS
handlers + a generalized directory-registration flow, plus read queries
(best-match resolution against the `sdo` table). AIA ships here; other
instruments subclass `FitsHandler` in their own projects.

```python
from datetime import datetime
from egghouse.database import initialize_database
from egghouse.swdb import (
    SDO_SCHEMA, AiaFitsHandler, register_fits_dir,
    get_sdo_best_match, get_sdo_best_matches,
)

# Build a setup-sw-db-compatible solar_images.sdo table from the
# reference schema (byte-identical DDL):
initialize_database(db_config, {"sdo": SDO_SCHEMA})

# Scan a directory of AIA FITS, validate, idempotently upsert, and
# (optionally) archive into {root}/aia/YYYY/YYYYMMDD/:
report = register_fits_dir(
    "/data/incoming",
    handler=AiaFitsHandler(),          # T_OBS (UTC); quality stored, not gated
    table="sdo",
    db_config=db_config,
    conflict_columns=["telescope", "channel", "datetime"],
    move_root="/archive",              # omit to register files in place
    parallel=4,
)
print(report.summary())                # scanned/valid/inserted/errors reconcile

# Read side: resolve the stored record closest to a target time
# (one channel; all sdo columns incl. file_path are returned):
row = get_sdo_best_match(
    db_config, telescope="aia", channel="193",
    target_time=datetime(2014, 1, 1, 12), time_range_minutes=1,
)
print(row["file_path"] if row else "no match")
```

### `egghouse.transfer`
File transfer utilities with HTTP, FTP, and SFTP support.

```python
from egghouse.transfer import download_single_file, download_parallel, get_file_list

# HTTP: Single file download
download_single_file('http://example.com/data.fits', 'data.fits')

# HTTP: Get file list from directory listing
files = get_file_list('http://example.com/data/', extensions=['fits'])

# HTTP: Parallel download
tasks = [('http://example.com/f1.fits', 'f1.fits'), ('http://example.com/f2.fits', 'f2.fits')]
download_parallel(tasks, parallel=4)

# FTP: Download files (no external dependencies)
from egghouse.transfer import ftp_connection, ftp_download_file, ftp_list_files

with ftp_connection('ftp.example.com', user='anonymous') as ftp:
    files = ftp_list_files(ftp, '/data/', extensions=['fits'])
    ftp_download_file(ftp, '/data/file.fits', 'local_file.fits')

# FTP: Parallel download
from egghouse.transfer import ftp_download_parallel
tasks = [('/remote/f1.fits', 'f1.fits'), ('/remote/f2.fits', 'f2.fits')]
result = ftp_download_parallel('ftp.example.com', tasks, parallel=4)

# SFTP: Secure file transfer (requires paramiko)
from egghouse.transfer import sftp_connection, sftp_download_file, sftp_upload_file

with sftp_connection('sftp.example.com', user='admin', key_file='~/.ssh/id_rsa') as sftp:
    sftp_download_file(sftp, '/data/file.fits', 'local_file.fits')
    sftp_upload_file(sftp, 'local_file.fits', '/upload/file.fits')
```

---

## Configuration

### Database (Environment Variables)

```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=solar_data
export DB_USER=username
export DB_PASSWORD=password
export DB_LOG_QUERIES=true  # Optional: enable query logging
```

### Database (YAML Config)

```yaml
host: localhost
port: 5432
database: solar_data
user: username
password: password
```

---

## Documentation

**Detailed Guides** (in [docs/](docs/) folder):
- [image_guide.md](docs/image_guide.md) - Image processing utilities
- [sdo_guide.md](docs/sdo_guide.md) - SDO/AIA/HMI data processing
- [config_guide.md](docs/config_guide.md) - Configuration management
- [database_guide.md](docs/database_guide.md) - PostgreSQL utilities (CRUD + declarative schema + bulk records)
- [swdb_guide.md](docs/swdb_guide.md) - Solar / space-weather DB domain layer (schemas + FITS handlers + registration)
- [transfer_guide.md](docs/transfer_guide.md) - File transfer utilities (HTTP, FTP, SFTP)

**API Reference:** See [API_REFERENCE.md](API_REFERENCE.md) for complete function signatures.

---

## License

MIT License

## Author

**Eunsu Park**
- [ORCID: 0000-0003-0969-286X](https://orcid.org/0000-0003-0969-286X)
