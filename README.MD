# egghouse

A personal arsenal of reusable code for solar physics research.

이 레포지터리는 개인 프로젝트들에서 공통으로 사용하는 유틸리티를 모아둔 라이브러리입니다.
일부 내용은 직접 작성했으며, 일부는 AI의 도움을 받아 작성되었습니다.

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
| io | - | astropy |
| image | numpy, scipy | - |
| sdo | numpy, scipy | astropy, sunpy |
| config | pyyaml | - |
| database | psycopg2-binary | pyyaml |
| transfer | requests, beautifulsoup4 | - |

---

## Quick Start

```python
# 1. FITS 파일 읽기/쓰기
from egghouse.io import read_fits, write_fits
data, header = read_fits('image.fits')
write_fits('output.fits', data, overwrite=True)

# 2. 이미지 전처리 (resize, mask, scaling)
from egghouse.image import resize_image, circle_mask, bytescale_image
resized = resize_image(data, (1024, 1024))
mask = circle_mask(4096, radius=1600)
display = bytescale_image(data, imin=0, imax=5000)

# 3. SDO Level 1.5 변환
from egghouse.sdo import to_level15
level15_map = to_level15('aia_lev1.fits')  # auto-detect AIA/HMI
```

---

## Modules

### `egghouse.io`
File I/O for scientific data formats (FITS, BMP).

```python
from egghouse.io import read_fits, write_fits, read_fits_header, append_fits

# FITS read/write
data, header = read_fits('image.fits')
header = read_fits_header('image.fits')  # header only (memory-efficient)
write_fits('output.fits', data, header={'OBJECT': 'Sun'}, overwrite=True)
append_fits('output.fits', ext_data, header={'EXTNAME': 'EXT1'})

# BMP read/write (no external dependencies)
from egghouse.io import read_bmp, write_bmp
data, info = read_bmp('image.bmp')       # (H, W, 3) uint8 RGB
write_bmp('output.bmp', data)            # 24-bit RGB
write_bmp('gray.bmp', gray_uint8)        # 8-bit grayscale
```

### `egghouse.image`
Generic image processing utilities.

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

### `egghouse.transfer`
HTTP file download utilities with parallel support.

```python
from egghouse.transfer import download_single_file, download_parallel, get_file_list

# Single file download
download_single_file('http://example.com/data.fits', 'data.fits')

# Get file list from directory listing
files = get_file_list('http://example.com/data/', extension='.fits')

# Parallel download
download_parallel(url_list, output_dir, max_workers=4)
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

## API Reference

See [API_REFERENCE.md](API_REFERENCE.md) for complete function signatures and descriptions.

---

## License

This project is for personal use. Feel free to reference or adapt for your own projects.

---

## Author

Eunsu Park
