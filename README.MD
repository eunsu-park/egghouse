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
| config | pyyaml | - |
| database | psycopg2-binary | pyyaml |
| transfer | requests, beautifulsoup4 | - |
| sdo | numpy, scipy | astropy, sunpy |

---

## Modules

### `egghouse.database`
PostgreSQL database utilities with simplified CRUD operations.

```python
from egghouse.database import PostgresManager, load_config

config = load_config('database.yaml')
db = PostgresManager(**config)

# Insert
db.insert('observations', {'date': '2024-01-01', 'wavelength': 171})

# Query
results = db.select('observations', where={'wavelength': 171})
df = db.to_dataframe(results, columns=['id', 'date', 'wavelength'])
```

### `egghouse.sdo`
SDO/AIA and SDO/HMI data processing utilities.

```python
from egghouse.sdo import aia_intscale, hmi_intscale, Stacking, to_level15

# AIA intensity scaling
scaled = aia_intscale(aia_data, wavelength=171)

# HMI magnetogram scaling
mag_scaled = hmi_intscale(hmi_data, vmin=-500, vmax=500)

# Level 1.0 to Level 1.5 conversion (north-up, centered, 4096x4096)
level15_map = to_level15('aia_lev1.fits', instrument='AIA')
level15_map = to_level15('hmi_lev1.fits', instrument='HMI')

# Solar rotation-corrected stacking (with differential rotation)
stacker = Stacking(nb_stack=21, method='mean', latitude_deg=30)
mean_image = stacker.run(fits_files)

# Snodgrass differential rotation model
from egghouse.sdo import snodgrass_rotation_rate
rate_eq = snodgrass_rotation_rate(0)    # 14.71 deg/day at equator
rate_60 = snodgrass_rotation_rate(60)   # ~12.0 deg/day at 60° latitude
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

# Load from YAML
config = TrainConfig.from_yaml('train_config.yaml')

# Load from JSON
config = TrainConfig.from_json('train_config.json')

# Load from environment variables (TRAIN_LR, TRAIN_EPOCHS, ...)
config = TrainConfig.from_env(prefix="TRAIN_")

# Load from CLI with YAML base
# python train.py --config base.yaml --lr 0.0001
config = TrainConfig.from_args()

# Save to YAML/JSON
config.to_yaml('output_config.yaml')
config.to_json('output_config.json')
```

### `egghouse.image`
Generic image processing utilities.

```python
from egghouse.image import resize_image, circle_mask, bytescale_image

# Resize preserving dtype
resized = resize_image(image, target_shape=(1024, 1024))

# Solar disk mask
mask = circle_mask(image.shape, center=(2048, 2048), radius=1600)

# Scale to uint8
display = bytescale_image(image, vmin=0, vmax=1000)
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

## License

This project is for personal use. Feel free to reference or adapt for your own projects.

---

## Author

Eunsu Park
