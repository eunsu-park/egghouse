# egghouse

A personal arsenal of reusable code for solar physics research.

> **Shared common layer.** `egghouse` implements generic primitives once —
> image metrics, denoisers, transforms, noise-scale estimation, DEM inversion,
> SDO data prep, config, and DB/transfer helpers — so they can be consumed via
> `import` rather than duplicated. `egghouse.config.BaseConfig` is the standard
> mechanism for reading paths/params from env/YAML/CLI.

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

A plain install is **fully working**: the SDO / DEM / transfer / FITS stacks
(`astropy`, `sunpy`, `aiapy`, `drms`, `fiasco`, `matplotlib`, `requests`,
`beautifulsoup4`) ship in core `install_requires` alongside `numpy` / `scipy`
/ `pandas`. Only niche stacks stay optional:

| Extra | Adds | For |
|-------|------|-----|
| `database` | psycopg2-binary, pyyaml | `egghouse.database` (PostgreSQL) |
| `sftp` | paramiko | SFTP transfers in `egghouse.transfer` |
| `denoise` | scikit-image, PyWavelets, bm3d | `egghouse.denoise` |
| `image` | opencv-python | the OpenCV-backed transform in `egghouse.image` |
| `config` | pyyaml | YAML support in `egghouse.config` |
| `all` | all of the above | — |

`egghouse[sdo]`, `[dem]`, `[transfer]`, `[fits]` remain as **empty no-op
aliases** (their deps are now core) so existing references keep resolving.

> `egghouse.io` was removed in v0.6.0. Use `astropy.io.fits` for FITS
> and `PIL.Image` (Pillow) for other image formats directly.

---

## Quick Start

```python
# Image preprocessing (resize, mask, scaling)
from egghouse.image import resize_image, circle_mask, bytescale_image
resized = resize_image(data, (1024, 1024))
mask = circle_mask(4096, radius=1600)
display = bytescale_image(data, imin=0, imax=5000)

# SDO Level 1.5 conversion (auto-detect AIA/HMI)
from egghouse.sdo import to_level15
level15_map = to_level15('aia_lev1.fits')

# JSOC export (drms-based AIA acquisition)
from datetime import datetime
from egghouse.sdo import jsoc_export, aia_euv_query
urls = jsoc_export(aia_euv_query([datetime(2014, 1, 1, 12)]), email='you@example.com')

# DEM inversion — solvers are instrument-agnostic (egghouse.dem); the AIA
# temperature response (instrument-specific) comes from egghouse.sdo.
from egghouse.dem import dem_sites, get_default_temperatures
from egghouse.sdo import get_temperature_response
temps = get_default_temperatures(n_bins=50)
response = get_temperature_response(temperatures=temps, ssw_table_path='response_matrix.npz')
dem, info = dem_sites(intensities, errors, response, temps)
```

---

## Modules

Eight modules; each has a detailed guide under [docs/](docs/) (or the
[API reference](API_REFERENCE.md) for full signatures). Heavy deps are
lazy-imported, so importing one module never eagerly pulls another's.

| Module | What it provides | Guide |
|--------|------------------|-------|
| `image` | metrics (PSNR/SSIM/robust_sigma), transforms, noise, augment, filters, stats, colorize | [image_guide](docs/image_guide.md) |
| `denoise` | wavelet, BM3D, NLM, TV, Wiener, Anscombe, NoiseGate | [denoise_guide](docs/denoise_guide.md) |
| `sdo` | AIA/HMI Level-1.5, degradation correction, JSOC export, temperature response, stacking, AIA color tables, FITS timeparse | [sdo_guide](docs/sdo_guide.md), [aia_pipeline_guide](docs/aia_pipeline_guide.md) |
| `dem` | instrument-agnostic DEM solvers (SITES, NNLS, regularized, sparse, Plowman, MCMC, spline, Gaussian) + `dem_map` | [API reference](API_REFERENCE.md) |
| `database` | `PostgresManager`, declarative schema + bulk records | [database_guide](docs/database_guide.md) |
| `swdb` | solar/space-weather DB layer + NOAA SWPC parsers | [swdb_guide](docs/swdb_guide.md) |
| `transfer` | HTTP / FTP / SFTP downloads | [transfer_guide](docs/transfer_guide.md) |
| `config` | `BaseConfig` — env / YAML / JSON / CLI config | [config_guide](docs/config_guide.md) |

---

## Configuration

`egghouse.database` reads its connection from `DB_*` environment variables or a
YAML file (consumers supply all data paths; the library bundles no data):

```bash
export DB_HOST=localhost DB_PORT=5432 DB_NAME=solar_data
export DB_USER=username DB_PASSWORD=password
export DB_LOG_QUERIES=true   # optional: enable query logging
```

```yaml
host: localhost
port: 5432
database: solar_data
user: username
password: password
```

---

## Documentation

- **Guides** — per-module cookbooks in [docs/](docs/) (linked in the table above).
- **[API_REFERENCE.md](API_REFERENCE.md)** — complete function signatures.
- **[CHANGELOG.md](CHANGELOG.md)** — version history (SemVer, Keep-a-Changelog).

---

## License

MIT License

## Author

**Eunsu Park**
- [ORCID: 0000-0003-0969-286X](https://orcid.org/0000-0003-0969-286X)
