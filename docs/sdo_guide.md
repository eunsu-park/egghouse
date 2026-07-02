# egghouse.sdo User Guide

SDO (Solar Dynamics Observatory) AIA and HMI data processing utilities.

> **New in v0.4+**: JSOC drms export (`jsoc_export`, `aia_euv_query`,
> v0.4) and AIA Level 1 → 1.5 prep steps (`aia_update_pointing`, `aia_respike`,
> `aia_correct_degradation`, `aia_deconvolve`, `mask_out_of_disk`,
> cache helpers, v0.5) are covered in the **JSOC export** and **AIA Level 1 → 1.5 prep
> steps** sections below. Function signatures are documented in `API_REFERENCE.md`, and the
> change history is summarized in `CHANGELOG.md`.

---

## Overview

The SDO module provides specialized tools for processing solar observation data:
- **AIA**: multi-wavelength EUV/UV image intensity scaling
- **HMI**: magnetic field data scaling and vector field processing
- **Level 1.5**: Level 1.0 → 1.5 preprocessing (north-up, centered)
- **Stacking**: solar-rotation-corrected image stacking
- **Quality**: QUALITY keyword interpretation and data quality validation
- **DEM**: temperature distribution inversion from multi-wavelength AIA observations (SITES algorithm)

---

## AIA Processing

### aia_intscale

Wavelength-optimized intensity scaling. For visualization.

```python
from egghouse.sdo import aia_intscale
from astropy.io import fits

# Read FITS file (egghouse.io was removed in v0.6.0 — use astropy directly)
data, header = fits.getdata('aia_171.fits', header=True)
exptime = header['EXPTIME']
wavelnth = header['WAVELNTH']

# Apply scaling (returns uint8)
scaled = aia_intscale(data, exptime, wavelnth)

# Return float (for further processing)
scaled_float = aia_intscale(data, exptime, wavelnth, to_bytescale=False)
```

### Supported Wavelengths and Scaling Methods

| Wavelength (Å) | Scale | Description |
|---------|-------|------|
| 94 | sqrt | Fe XVIII (flares) |
| 131 | log | Fe VIII/XXI |
| 171 | sqrt | Fe IX (corona) |
| 193 | log | Fe XII/XXIV |
| 211 | log | Fe XIV |
| 304 | log | He II (transition region) |
| 335 | log | Fe XVI |
| 1600 | linear | C IV + continuum |
| 1700 | linear | Continuum |
| 4500 | linear | Photosphere |

### get_aia_calibration

Look up per-wavelength calibration parameters.

```python
from egghouse.sdo import get_aia_calibration

cal = get_aia_calibration(171)
print(f"Normalization exposure time: {cal['norm_exptime']}")
print(f"Min/Max: {cal['vmin']}, {cal['vmax']}")
print(f"Scale method: {cal['scale']}")
```

---

## HMI Processing

### hmi_intscale

Scale magnetic field data to uint8.

```python
from egghouse.sdo import hmi_intscale
from astropy.io import fits

data, header = fits.getdata('hmi_m.fits', header=True)

# Default range [-100, 100] Gauss (quiet sun)
scaled = hmi_intscale(data)

# Wide range for active regions
scaled_ar = hmi_intscale(data, vmin=-500, vmax=500)

# Strong magnetic field regions
scaled_strong = hmi_intscale(data, vmin=-2000, vmax=2000)
```

### hmi_field_strength

Compute total magnetic field strength from vector magnetic field components.

```python
from egghouse.sdo import hmi_field_strength
from astropy.io import fits

# Read vector magnetic field components
bx, _ = fits.getdata('hmi_bx.fits', header=True)
by, _ = fits.getdata('hmi_by.fits', header=True)
bz, _ = fits.getdata('hmi_bz.fits', header=True)

# Total magnetic field strength: |B| = sqrt(Bx² + By² + Bz²)
b_total = hmi_field_strength(bx, by, bz)
print(f"Max magnetic field: {b_total.max():.1f} Gauss")
```

---

## Level 1.5 Conversion

Convert Level 1.0 data to the standardized Level 1.5 format:
- North-up alignment (CROTA2 = 0)
- Solar center alignment (CRPIX at center)
- Standard plate scale (0.6 arcsec/px)
- Fixed 4096×4096 size

### to_level15

Single-file conversion.

```python
from egghouse.sdo import to_level15

# Automatic instrument detection
m = to_level15('aia_171_lev1.fits')

# Explicit instrument specification
m_hmi = to_level15('hmi_m_lev1.fits', instrument='HMI')

# Check result
print(f"Shape: {m.data.shape}")        # (4096, 4096)
print(f"CROTA2: {m.meta['CROTA2']}")   # 0.0
print(f"LVL_NUM: {m.meta['LVL_NUM']}") # 1.5

# Save FITS
m.save('aia_171_lev15.fits', overwrite=True)
```

### Parameters

```python
m = to_level15(
    'aia_171.fits',
    instrument=None,           # 'AIA' or 'HMI', None=auto
    target_plate_scale=None,   # arcsec/px (default 0.6)
    target_size=4096,          # output size
    order=3,                   # interpolation order (0-5)
    missing=0.0                # padding value
)
```

### batch_to_level15

Batch-convert multiple files.

```python
from egghouse.sdo import batch_to_level15
import glob

fits_files = glob.glob('/data/aia_171_*.fits')

# Progress callback
def progress(current, total, msg):
    print(f"[{current}/{total}] {msg}")

output_files = batch_to_level15(
    fits_files,
    output_dir='/output/level15/',
    instrument='AIA',
    overwrite=False,
    progress_callback=progress
)

print(f"Conversion complete: {len(output_files)} files")
```

### get_level_info

Look up the processing level information of a file.

```python
from egghouse.sdo import get_level_info

info = get_level_info('aia_171.fits')

print(f"Level: {info['level']}")
print(f"CROTA2: {info['crota2']}")
print(f"Is Level 1.5: {info['is_level15']}")

if not info['is_level15']:
    m = to_level15('aia_171.fits')
```

---

## JSOC export (drms-based, v0.4+)

Uses the DRMS export system of JSOC (jsoc.stanford.edu) to directly stage
SDO/AIA data and retrieve URLs. The `egghouse.sdo.jsoc` module uses the `drms`
package as a **soft dependency** — it is lazily imported inside function bodies,
so simply importing the module does not require `drms`
(the query-composition logic can be tested without network access). The
recommended flow is to pass the returned URLs to
`egghouse.transfer.download_parallel` and download them using its retry/atomic-write
features.

### aia_euv_query

Composes a DRMS record-set string that selects AIA EUV records for multiple
timestamps. Each time scans a `tolerance` interval, and multiple times are
combined into a single export request.

```python
from datetime import datetime, timedelta
from egghouse.sdo import aia_euv_query

times = [
    datetime(2014, 1, 1, 12, 0, 0),
    datetime(2014, 1, 1, 13, 0, 0),
    datetime(2014, 1, 1, 14, 0, 0),
]

# Default 6 DEM channels, aia.lev1_euv_12s series
query = aia_euv_query(times)
print(query)
# aia.lev1_euv_12s[2014.01.01_12:00:00_TAI/12s][...]/12s][? WAVELNTH=94 OR ... ?]

# Specific channels only, extended tolerance
query_171 = aia_euv_query(
    times,
    wavelengths=[171, 193],
    tolerance=timedelta(seconds=24),
)
```

#### Parameters

```python
query = aia_euv_query(
    times,                              # datetime sequence (non-empty)
    wavelengths=AIA_DEM_WAVELENGTHS,    # default 6 DEM channels (Å)
    series=AIA_LEV1_EUV_SERIES,         # 'aia.lev1_euv_12s'
    tolerance=timedelta(seconds=12),    # scan width after each time (>0)
)
```

- Naive `datetime` values are interpreted as the TAI that JSOC expects.
- If `times` or `wavelengths` is empty, or `tolerance` is zero or less,
  a `ValueError` is raised.

### jsoc_export

Submits a DRMS export request, blocks until staging finishes, and then
returns a list of staged file URLs.

```python
from egghouse.sdo import jsoc_export

urls = jsoc_export(
    query,
    email='you@example.com',   # email registered at jsoc.stanford.edu
)
print(f"Staged files: {len(urls)}")
```

#### Parameters

```python
urls = jsoc_export(
    query,                  # record-set string produced by aia_euv_query, etc.
    email='you@example.com',# JSOC export registration email (required, keyword)
    method='url',           # 'url' blocks on the server until staging completes
    protocol='fits',        # staged file protocol
    client=None,            # drms.Client to reuse (None creates a new one)
)
```

- `method='url'` waits on the server side until the dataset is staged, then
  returns the concrete URLs.
- If the export does not succeed, a `RuntimeError` is raised. If there are no
  matching records, an empty list may be returned.

### cached_correction_table / cached_pointing_table

Pickle-caches to disk the aiapy correction tables that would be slow to refetch
from JSOC for every record in a batch job. On the first call it fetches and
records them; subsequent calls deserialize from disk and skip the network round trip.

```python
from datetime import datetime
from egghouse.sdo import cached_correction_table, cached_pointing_table

# Degradation correction table (aiapy.calibrate.util.get_correction_table)
corr = cached_correction_table('cache/aia_correction.pkl')

# Time-window pointing table (end is exclusive by aiapy convention)
pointing = cached_pointing_table(
    'cache/aia_pointing.pkl',
    start=datetime(2014, 1, 1),
    end=datetime(2014, 1, 2),
)
```

- The `start`/`end` of `cached_pointing_table` are used **only when
  populating the cache anew**. An existing cache file is reused as-is, so if you
  need a different time window, delete the file first.
- Both functions create parent directories as needed.

### AIA_LEV1_EUV_SERIES constant

```python
from egghouse.sdo import AIA_LEV1_EUV_SERIES

print(AIA_LEV1_EUV_SERIES)  # 'aia.lev1_euv_12s'
```

### End-to-end: query → export → parallel download

```python
from datetime import datetime
from egghouse.sdo import aia_euv_query, jsoc_export
from egghouse.transfer import download_parallel

# 1. Compose a record-set for multiple timestamps
times = [
    datetime(2014, 1, 1, 12, 0, 0),
    datetime(2014, 1, 1, 13, 0, 0),
    datetime(2014, 1, 1, 14, 0, 0),
]
query = aia_euv_query(times, wavelengths=[94, 131, 171, 193, 211, 335])

# 2. JSOC export → list of staged URLs
urls = jsoc_export(query, email='you@example.com')

# 3. Convert to (url, dest) tasks, then download in parallel
import os
os.makedirs('data', exist_ok=True)
tasks = [(u, os.path.join('data', u.rsplit('/', 1)[-1])) for u in urls]

result = download_parallel(tasks, parallel=4, max_retries=3)
print(f"Downloaded: {result['downloaded']}, Failed: {result['failed']}")
```

---

## AIA Level 1 → 1.5 prep steps (v0.5+)

`to_level15` (the section above) performs rotation, resampling, and padding
common to both AIA and HMI, but does not handle the additional aiapy-based prep
steps. The `egghouse.sdo.prep` module provides aiapy wrappers that fill that
gap. `sunpy`/`aiapy`/`astropy` are all lazily imported inside function bodies,
so importing the module itself is lightweight and does not force aiapy.

**Canonical aiapy processing order.** The canonical order for the
AIA Level 1 → 1.5 conversion is as follows:

```
update_pointing → respike → correct_degradation → deconvolve
                → register / to_level15 (rotation·resample)
```

`register`/`to_level15` (L1.5 registration) is applied **last**.
`aia_correct_degradation` and `aia_deconvolve` **return the input map
unchanged** for wavelengths outside the AIA channels for which corrections/PSFs
are defined (EUV + 304 Å: 94, 131, 171, 193, 211, 304, 335 Å), so they can be
safely applied uniformly to batches containing mixed channels.

### aia_update_pointing

Updates the AIA WCS keywords with the JSOC master pointing table
(`aiapy.calibrate.update_pointing` wrapper). In batches, pass a table prefetched
with `cached_pointing_table` to avoid re-requesting for every record.

```python
from sunpy.map import Map
from egghouse.sdo import aia_update_pointing, cached_pointing_table
from datetime import datetime

pointing = cached_pointing_table(
    'cache/pointing.pkl',
    start=datetime(2014, 1, 1), end=datetime(2014, 1, 2),
)

m = Map('aia_171_lev1.fits')
m = aia_update_pointing(m, pointing_table=pointing)
# If pointing_table=None, aiapy fetches from JSOC every time
```

### aia_respike

Reinjects the spike pixels that the Level 1 pipeline removed
(`aiapy.calibrate.respike` wrapper). If `spikes` is not provided, they are
fetched with `aiapy.calibrate.fetch_spikes`, which incurs a JSOC round trip per
record, so in batches it is best to prefetch and pass them in.

```python
from egghouse.sdo import aia_respike

m = aia_respike(m)                # spikes=None → fetch per record
# m = aia_respike(m, spikes=prefetched_spikes)
```

### aia_correct_degradation

Applies the time-dependent effective-area correction
(`aiapy.calibrate.correct_degradation` wrapper). For wavelengths where no
correction is defined, it returns the map unchanged. Pass the
`cached_correction_table` result to amortize the JSOC fetch across the whole batch.

```python
from egghouse.sdo import aia_correct_degradation, cached_correction_table

corr = cached_correction_table('cache/correction.pkl')
m = aia_correct_degradation(m, correction_table=corr)
```

### aia_deconvolve

PSF-deconvolves an AIA map (`aiapy.psf.deconvolve` wrapper).
The PSF computation (`aiapy.psf.psf(...)`) is the most expensive part, taking
several minutes per channel, so prebuild a `{wavelength: PSF}` dict with
`cached_aia_psfs` and pass it in. For wavelengths where no PSF is provided, it
returns the map unchanged.

```python
from egghouse.sdo import aia_deconvolve, cached_aia_psfs

psfs = cached_aia_psfs('cache/aia_psfs.pkl')   # standard 7 channels
m = aia_deconvolve(m, psfs=psfs)
# If psfs=None, the PSF for that channel is computed directly at call time (several minutes)
```

### cached_aia_psfs

Computes AIA PSFs per channel and pickle-caches them as a
`{wavelength(int): PSF array}` dict. If the cache file does not exist or does
**not** contain **all** requested wavelengths, it regenerates the full set.

```python
from egghouse.sdo import cached_aia_psfs

# Default: 7 correction channels (94, 131, 171, 193, 211, 304, 335)
psfs = cached_aia_psfs('cache/aia_psfs.pkl')

# A subset of channels only
psfs_subset = cached_aia_psfs('cache/aia_psfs.pkl', wavelengths=[171, 193])
```

### mask_out_of_disk

Returns a copy of the `sunpy.Map` with pixels beyond the solar limb filled with
a sentinel value (the input is not modified). For the limb radius it uses the
header's `R_SUN` (in pixels) if present, otherwise `RSUN_OBS / CDELT1`, and if
neither is available it raises a `KeyError`. This is useful for marking off-disk
regions so downstream code (e.g. a DEM model training loop) can ignore them.

```python
from egghouse.sdo import mask_out_of_disk

masked = mask_out_of_disk(m, fill_value=-5000.0)  # default sentinel -5000.0
```

### Batch workflow: prefetch caches/PSFs once, then prep in bulk

```python
from datetime import datetime
import glob
from sunpy.map import Map
from egghouse.sdo import (
    cached_pointing_table,
    cached_correction_table,
    cached_aia_psfs,
    aia_update_pointing,
    aia_respike,
    aia_correct_degradation,
    aia_deconvolve,
    to_level15,
    mask_out_of_disk,
)

# 1. Prefetch the slow tables/PSFs only once before the batch (disk cache)
pointing = cached_pointing_table(
    'cache/pointing.pkl',
    start=datetime(2014, 1, 1), end=datetime(2014, 1, 2),
)
corr = cached_correction_table('cache/correction.pkl')
psfs = cached_aia_psfs('cache/aia_psfs.pkl')

# 2. Apply prep in the canonical aiapy order to each file
for path in sorted(glob.glob('/data/aia_*_lev1.fits')):
    m = Map(path)
    m = aia_update_pointing(m, pointing_table=pointing)
    m = aia_respike(m)
    m = aia_correct_degradation(m, correction_table=corr)
    m = aia_deconvolve(m, psfs=psfs)
    # 3. L1.5 registration last (rotation·resample·padding)
    m = to_level15(m)
    # 4. (Optional) off-disk masking
    m = mask_out_of_disk(m)
    m.save(path.replace('_lev1', '_lev15'), overwrite=True)
```

Even if non-AIA or correction-undefined wavelengths are mixed in, you can use
the loop above as-is because `aia_correct_degradation`/`aia_deconvolve` pass
those maps through unchanged.

---

## Solar-Rotation-Corrected Stacking

Stacks time-series images while correcting for the shift caused by solar rotation.

### Stacking class

```python
from egghouse.sdo import Stacking
import glob

fits_files = sorted(glob.glob('/data/hmi_m_*.fits'))

# Basic usage (returns a list)
stacker = Stacking(nb_stack=21)
aligned_list = stacker.run(fits_files)

# mean combine
stacker = Stacking(nb_stack=21, method='mean')
mean_image = stacker.run(fits_files)

# median combine
stacker = Stacking(nb_stack=21, method='median')
median_image = stacker.run(fits_files)

# sigma-clipped mean
stacker = Stacking(nb_stack=21, method='sigma_clipped', sigma_lower=3, sigma_upper=3)
clipped_image = stacker.run(fits_files)
```

### Stacking parameters

```python
stacker = Stacking(
    nb_stack=21,                    # number of images to stack
    solar_rot_period=None,          # rotation period (days), None=Snodgrass differential rotation
    crop_size=512,                  # crop region size
    cadence_seconds=None,           # imaging cadence (seconds), None=auto-detect
    latitude_deg=0.0,               # latitude at which differential rotation is applied
    method='list',                  # combine method
    sigma_lower=3.0,                # sigma clipping lower bound
    sigma_upper=3.0                 # sigma clipping upper bound
)
```

### Differential Rotation (Snodgrass Model)

The Sun rotates at different speeds depending on latitude (the equator is fastest).

```python
from egghouse.sdo import snodgrass_rotation_rate

# Equator (0°): 14.71 deg/day
rate_eq = snodgrass_rotation_rate(0)

# Latitude 30°: ~13.8 deg/day
rate_30 = snodgrass_rotation_rate(30)

# Latitude 60°: ~12.0 deg/day
rate_60 = snodgrass_rotation_rate(60)

print(f"Equator: {rate_eq:.2f} deg/day")
print(f"30°: {rate_30:.2f} deg/day")
print(f"60°: {rate_60:.2f} deg/day")
```

**Snodgrass formula:**
```
ω(B) = A + B·sin²(B) + C·sin⁴(B)
```
- A = 14.71 (equatorial rotation rate)
- B = -2.39
- C = -1.78

### solar_rotation_shift

Compute the pixel shift over elapsed time.

```python
from egghouse.sdo import solar_rotation_shift

rsun_pixels = 1600  # solar radius (pixels)
time_hours = 1.0    # time offset

# Shift at the equator
shift_eq = solar_rotation_shift(rsun_pixels, time_hours, latitude_deg=0)
print(f"Equator 1-hour shift: {shift_eq:.2f} pixels")

# Shift at high latitude (slower)
shift_60 = solar_rotation_shift(rsun_pixels, time_hours, latitude_deg=60)
print(f"60° 1-hour shift: {shift_60:.2f} pixels")

# Use a fixed rotation period (Carrington)
shift_carr = solar_rotation_shift(rsun_pixels, time_hours, rotation_period_days=25.38)
```

### cross_correlate_shift

Subpixel alignment via FFT-based phase correlation.

```python
from egghouse.sdo import cross_correlate_shift
from scipy.ndimage import shift as ndimage_shift

# Compute the shift between two images
dy, dx = cross_correlate_shift(reference_image, target_image)
print(f"Shift: dy={dy:.2f}, dx={dx:.2f} pixels")

# Apply the shift correction
aligned = ndimage_shift(target_image, (dy, dx), order=3)
```

### StreamingStackAccumulator

Memory-efficient streaming stacking (Welford algorithm).

```python
from egghouse.sdo import StreamingStackAccumulator
from astropy.io import fits

# Process large files
acc = StreamingStackAccumulator(shape=(4096, 4096))

for fits_file in large_file_list:
    data, _ = fits.getdata(fits_file, header=True)
    acc.add(data)

# Extract results
mean_image = acc.get_mean()
std_image = acc.get_std()
variance = acc.get_variance()

print(f"Number of processed images: {acc.count}")
```

---

## Core Utilities

### parse_fits_header

Extract key keywords from an SDO FITS header.

```python
from egghouse.sdo import parse_fits_header

header = parse_fits_header('aia_171.fits')

print(f"Observation time: {header['date_obs']}")
print(f"Wavelength: {header['wavelnth']} Å")
print(f"Exposure time: {header['exptime']} s")
print(f"Plate scale: {header['cdelt1']} arcsec/px")
print(f"CROTA2: {header['crota2']}°")
print(f"Solar radius: {header['rsun_obs']} arcsec")
```

### validate_sdo_image

Validate image validity.

```python
from egghouse.sdo import validate_sdo_image

# Validate standard SDO size
validate_sdo_image(data)  # 4096x4096

# Skip size validation
validate_sdo_image(data, expected_shape=None)

# Custom size validation
validate_sdo_image(resized_data, expected_shape=(1024, 1024))
```

### get_solar_disk_params

Compute solar disk parameters from the header.

```python
from egghouse.sdo import parse_fits_header, get_solar_disk_params

header = parse_fits_header('aia_171.fits')
params = get_solar_disk_params(header)

print(f"Solar center: ({params['center_x']:.1f}, {params['center_y']:.1f})")
print(f"Solar radius: {params['radius_pixels']:.1f} pixels")
print(f"Plate scale: {params['plate_scale']:.3f} arcsec/px")
```

### parse_fits_datetime

Extract the observation datetime from a FITS file. It reads the header
(`T_REC` → `T_OBS` → `DATE-OBS`, handling the SDO TAI form
`YYYY.MM.DD_HH:MM:SS_TAI`, ISO `T`-separated strings, and legacy
`DD/MM/YY` dates) and, if the header is missing or unreadable, falls back to
parsing the SDO AIA/HMI **filename**. Returns a naive `datetime`, or `None`
when nothing parses. TAI is not converted to UTC (~35 s difference).

```python
from egghouse.sdo import parse_fits_datetime

dt = parse_fits_datetime('/data/aia.lev1_euv_12s.2010-09-01T000008Z.193.image_lev1.fits')
# datetime.datetime(2010, 9, 1, 0, 0, 8)  (from filename if header absent)
```

`astropy` is lazy-imported (`HAS_ASTROPY`): when it is not installed the header
path is skipped and only the filename is parsed. Promoted from `solaris-data`
(`core/parse.py`), which keeps a re-export shim for backward compatibility.

---

## Constants

```python
from egghouse.sdo import (
    AIA_PLATE_SCALE,       # 0.6 arcsec/px
    HMI_PLATE_SCALE,       # 0.6 arcsec/px
    SDO_IMAGE_SIZE,        # 4096 pixels
    SOLAR_ROTATION_PERIOD, # 25.38 days (Carrington)
    SNODGRASS_A,           # 14.71 deg/day
    SNODGRASS_B,           # -2.39
    SNODGRASS_C,           # -1.78
    HMI_CADENCE_45S,       # 45.0 seconds
    HMI_CADENCE_720S,      # 720.0 seconds
    AIA_CALIBRATION,       # per-wavelength calibration table
)
```

---

## Full Workflow Examples

### AIA data processing

```python
from astropy.io import fits, write_fits
from egghouse.sdo import to_level15, aia_intscale
from egghouse.image import circle_mask, resize_image

# 1. Level 1.5 conversion
m = to_level15('aia_171_lev1.fits')

# 2. Intensity scaling
scaled = aia_intscale(m.data, m.meta['EXPTIME'], 171)

# 3. Solar disk masking
mask = circle_mask(4096, radius=1600)
masked = np.where(mask, scaled, 0)

# 4. Resize
resized = resize_image(masked, (512, 512))
```

### HMI stacking workflow

```python
from egghouse.sdo import Stacking, to_level15, hmi_intscale
import glob

# 1. Level 1.5 conversion
fits_files = sorted(glob.glob('/data/hmi_m_*.fits'))

# 2. Stacking (21 frames, mean)
stacker = Stacking(
    nb_stack=21,
    method='mean',
    latitude_deg=15,  # target latitude
    crop_size=512
)
stacked = stacker.run(fits_files)

# 3. Scaling for visualization
display = hmi_intscale(stacked, vmin=-200, vmax=200)
```

---

## Dependencies

| Feature | Required | Optional |
|------|------|------|
| AIA/HMI scaling | numpy | - |
| Level 1.5 conversion | - | sunpy, astropy |
| Stacking | numpy, scipy | sunpy, astropy |
| DEM analysis | numpy | aiapy (accurate response functions) |
| FITS header parsing | - | astropy |

Installation:
```bash
# Basic (scaling only)
pip install numpy scipy

# Full SDO functionality
pip install "egghouse[sdo]"

# Including DEM analysis
pip install "egghouse[dem]"

# All features
pip install "egghouse[all]"
```

---

## QUALITY Keyword Interpretation

The QUALITY keyword in SDO FITS headers is a 32-bit integer indicating data quality issues.
Each bit represents a specific quality issue.

### decode_quality

Decode a QUALITY value into a human-readable form.

```python
from egghouse.sdo import decode_quality

# Nominal data
result = decode_quality(0)
print(result)
# [{'bit': -1, 'hex': '0x0', 'description': 'nominal', 'severity': 'ok'}]

# ISS loop open (bit 17)
result = decode_quality(0x20000)
print(result)
# [{'bit': 17, 'hex': '0x20000', 'description': 'ISS loop open', 'severity': 'caution'}]

# Multiple issues (HMI data)
result = decode_quality(0x30000, instrument="HMI")
for issue in result:
    print(f"[Bit {issue['bit']}] {issue['description']} ({issue['severity']})")
```

### format_quality

Output as a formatted string.

```python
from egghouse.sdo import format_quality

# Verbose output
print(format_quality(0x30000, instrument="HMI"))
# QUALITY = 0x30000 (196608)
# Status: 2 issue(s) detected
#   [Bit 16] 0x10000: Dark image (info)
#   [Bit 17] 0x20000: ISS loop open (caution)

# Brief output
print(format_quality(0x30000, verbose=False))
```

### is_quality_ok

Check whether the data is usable for analysis.

```python
from egghouse.sdo import is_quality_ok

# Default mode: ignore minor issues
is_quality_ok(0)         # True - nominal
is_quality_ok(0x1)       # True - Flatfield not applied (minor)
is_quality_ok(0x2000)    # False - during eclipse (severe)

# strict mode: check all issues
is_quality_ok(0x1, strict=True)  # False

# Ignore specific bits
is_quality_ok(0x2000, ignore_bits=[13])  # True - ignore eclipse bit
```

### get_quality_summary

Quality information summary dictionary.

```python
from egghouse.sdo import get_quality_summary

summary = get_quality_summary(0x30000)
print(f"Is nominal: {summary['is_nominal']}")      # False
print(f"Is usable: {summary['is_usable']}")       # False
print(f"Severity counts: {summary['severity_counts']}")  # {'info': 1, 'caution': 1}
print(f"Active bits: {summary['active_bits']}")     # [16, 17]
```

### print_all_quality_bits

Print all bit definitions (for reference).

```python
from egghouse.sdo import print_all_quality_bits

# Print AIA bit definitions
print_all_quality_bits("AIA")

# Print HMI bit definitions
print_all_quality_bits("HMI")
```

### Major QUALITY bits

| Bit | Hex | Description | Severity |
|-----|-----|------|--------|
| 0 | 0x1 | No flatfield data | minor |
| 8 | 0x100 | Some pixels missing | warning |
| 9 | 0x200 | More than 1% of pixels missing | warning |
| 10 | 0x400 | More than 5% of pixels missing | caution |
| 11 | 0x800 | More than 25% of pixels missing | severe |
| 13 | 0x2000 | During eclipse | severe |
| 16 | 0x10000 | Dark image | info |
| 17 | 0x20000 | ISS loop open | caution |
| 18 | 0x40000 | Calibration image | info |
| 30 | 0x40000000 | Quicklook image | info |
| 31 | 0x80000000 | No image | severe |

### Quality filtering workflow

```python
from egghouse.sdo import is_quality_ok, get_quality_summary
from astropy.io import fits  # egghouse.io was removed in v0.6.0
import glob

fits_files = glob.glob('/data/aia_171_*.fits')

# Filter for only good-quality files
good_files = []
for f in fits_files:
    header = fits.getheader(f)
    quality = header.get('QUALITY', 0)

    if is_quality_ok(quality):
        good_files.append(f)
    else:
        summary = get_quality_summary(quality)
        print(f"Excluded: {f} - {summary['severity_counts']}")

print(f"Usable: {len(good_files)}/{len(fits_files)} files")
```

---

## DEM (Differential Emission Measure) Analysis

Inverts the temperature-resolved emission measure (DEM) from multi-wavelength AIA observations.
Uses the SITES (Simple Iterative Temperature Emission Solver) algorithm.

### Basic concepts

The DEM represents the temperature distribution of the corona:
```
I(λ) = ∫ K(T,λ) × DEM(T) × dT
```
- `I(λ)`: observed intensity (DN/s)
- `K(T,λ)`: temperature response function
- `DEM(T)`: Differential Emission Measure (cm⁻⁵ K⁻¹)

Uses the 6 AIA EUV channels (94, 131, 171, 193, 211, 335 Å) to solve the
inversion problem and estimate the temperature distribution.

### Temperature response function

```python
from egghouse.dem import get_default_temperatures
from egghouse.sdo import get_temperature_response

# Default temperature grid (10^5.5 ~ 10^7.5 K)
temps = get_default_temperatures(n_bins=100)

# Obtain the temperature response function (v0.3.0+)
# Channel.temperature_response was removed in aiapy 0.12, so
# the canonical CHIANTI-based response is loaded from the SSW aia_get_response .npz table
# (e.g. response_matrix.npz bundled with demregpy). Calling without ssw_table_path
# raises NotImplementedError in an environment where aiapy is installed.
response = get_temperature_response(
    temperatures=temps,
    ssw_table_path='response_matrix.npz',
)
print(f"Response shape: {response.shape}")  # (100, 6)

# Or call the SSW loader directly:
import numpy as np
from egghouse.dem import load_ssw_temperature_response
response = load_ssw_temperature_response(
    'response_matrix.npz',
    log_temperatures=np.log10(temps),
)
```

### Single-pixel DEM inversion

```python
from egghouse.dem import dem_sites_pixel
import numpy as np

# 6-channel intensities (DN/s)
intensities = np.array([10.0, 50.0, 200.0, 150.0, 80.0, 20.0])
errors = intensities * 0.1  # 10% error

# DEM inversion with the SITES algorithm
dem, info = dem_sites_pixel(
    intensities,
    errors,
    response,
    temps,
    max_iter=100,
    tol=1e-4
)

print(f"Converged: {info['converged']}")
print(f"Iterations: {info['iterations']}")
print(f"Chi-squared: {info['chi2']:.2f}")
print(f"DEM peak: {dem.max():.2e} cm^-5 K^-1")
```

### Full-map DEM processing

```python
from egghouse.dem import dem_map

# image_cube: shape (height, width, 6)
# error_cube: same shape

dem_cube, info = dem_map(
    image_cube,
    error_cube,
    response,
    temps,
    chunk_size=512,  # chunk size for memory efficiency
    max_iter=100,
)

print(f"DEM cube shape: {dem_cube.shape}")  # (height, width, n_temps)
print(f"Processed pixels: {info['n_pixels']}")
```

### Derived quantity calculation

```python
from egghouse.dem import get_emission_measure, get_mean_temperature

# Total emission measure: EM = ∫ DEM × dT
em = get_emission_measure(dem, temps)
print(f"Emission Measure: {em:.2e} cm^-5")

# Integrate only over a specific temperature range
em_hot = get_emission_measure(dem, temps, t_min=1e6, t_max=1e7)

# DEM-weighted mean temperature
t_mean = get_mean_temperature(dem, temps)
print(f"Mean Temperature: {t_mean/1e6:.2f} MK")

# Apply to the entire map
em_map = get_emission_measure(dem_cube, temps)
t_map = get_mean_temperature(dem_cube, temps)
```

### Error estimation (Monte Carlo)

```python
from egghouse.dem.utils import compute_dem_errors

# Estimate DEM uncertainty via the Monte Carlo method
dem_errors = compute_dem_errors(
    dem,
    intensities,
    errors,
    response,
    temps,
    n_monte_carlo=100
)
print(f"DEM error range: {dem_errors.min():.2e} - {dem_errors.max():.2e}")
```

### Real-data workflow

```python
from datetime import datetime
import numpy as np
from sunpy.net import Fido, attrs as a
from sunpy.map import Map
import astropy.units as u
from egghouse.dem import get_default_temperatures, dem_map, get_emission_measure, get_mean_temperature
from egghouse.sdo import get_temperature_response

# 1. Download AIA data
obs_time = datetime(2024, 1, 15, 12, 0, 0)
wavelengths = [94, 131, 171, 193, 211, 335]

files = {}
for wave in wavelengths:
    result = Fido.search(
        a.Time(obs_time, obs_time),
        a.Instrument("AIA"),
        a.Wavelength(wave * u.angstrom),
    )
    downloaded = Fido.fetch(result, path='./data/')
    files[wave] = downloaded[0]

# 2. Build the image cube
maps = [Map(files[w]) for w in wavelengths]
image_cube = np.stack([
    m.data / m.exposure_time.to(u.s).value
    for m in maps
], axis=-1)
error_cube = image_cube * 0.1

# 3. Temperature response function (v0.3.0+: SSW table path required — see the note above)
temps = get_default_temperatures(n_bins=100)
response = get_temperature_response(
    wavelengths=wavelengths,
    temperatures=temps,
    ssw_table_path='response_matrix.npz',
)

# 4. Compute DEM
dem_cube, info = dem_map(
    image_cube,
    error_cube,
    response,
    temps,
    chunk_size=512,
)

# 5. Derived quantities
em = get_emission_measure(dem_cube, temps)
t_mean = get_mean_temperature(dem_cube, temps)

print(f"EM range: {em.min():.2e} - {em.max():.2e} cm^-5")
print(f"T_mean range: {t_mean.min()/1e6:.2f} - {t_mean.max()/1e6:.2f} MK")
```

### DEM constants

```python
from egghouse.sdo import HAS_AIAPY
from egghouse.sdo import AIA_DEM_WAVELENGTHS

print(f"aiapy installed: {HAS_AIAPY}")
print(f"DEM wavelengths: {AIA_DEM_WAVELENGTHS} Å")  # [94, 131, 171, 193, 211, 335]
```

### Notes

1. **aiapy installation recommended**: aiapy is required for accurate temperature response functions
   ```bash
   pip install aiapy
   ```

2. **Memory usage**: for a 4096×4096 map, the output DEM cube is ~6.5 GB
   - Adjustable via the `chunk_size` parameter
   - Process only the needed region via the `mask` parameter

3. **Positivity constraint**: the SITES algorithm applies the DEM ≥ 0 constraint

4. **Convergence check**: check convergence via `info['converged']`

---

## References

- Boerner, P. et al. 2012, Solar Physics, 275, 41 (AIA calibration)
- Snodgrass, H.B. 1983, ApJ, 270, 288 (Differential rotation)
- Lemen, J.R. et al. 2012, Solar Physics, 275, 17 (AIA instrument)
- Schou, J. et al. 2012, Solar Physics, 275, 229 (HMI instrument)
- Morgan, H. & Pickering, J. 2019, Solar Physics, 294, 135 (SITES algorithm)
- Hannah, I.G. & Kontar, E.P. 2012, A&A, 539, A146 (DEM regularization)
- JSOC QUALITY documentation: http://jsoc.stanford.edu/jsocwiki/Lev1qualBits
