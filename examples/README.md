# egghouse Examples

Quick examples to get started with egghouse.

## Examples

| File | Description | Dependencies |
|------|-------------|--------------|
| `01_basic_image.py` | Basic image processing operations | Core only |
| `02_sdo_quality.py` | SDO QUALITY flag interpretation | Core only |
| `03_config_usage.py` | Configuration management | `pyyaml` |

## Running Examples

```bash
# Core examples (no extra dependencies)
python examples/01_basic_image.py
python examples/02_sdo_quality.py

# Config example (requires pyyaml)
pip install pyyaml
python examples/03_config_usage.py
```

## Quick Start

```python
# Image processing
from egghouse.image import resize_image, circle_mask
import numpy as np

img = np.random.rand(256, 256).astype(np.float32)
small = resize_image(img, (64, 64))
mask = circle_mask(256, radius=100)

# SDO quality check
from egghouse.sdo import is_quality_ok

quality = 0x20000  # From FITS header
if is_quality_ok(quality):
    print("Data is usable")
```
