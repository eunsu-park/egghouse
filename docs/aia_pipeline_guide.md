# SDO/AIA Pipeline Guide: Preprocessing → 8-bit → Colorization

This guide walks through the three-stage pipeline that turns a raw SDO/AIA
Level 1.0 FITS file into a publication-ready, colorized RGB image:

1. **Preprocessing / calibration** — Level 1.0 → 1.5 geometry plus the
   aiapy-backed radiometric corrections.
2. **8-bit conversion** — exposure normalization, a per-channel intensity
   stretch, and byte scaling to `uint8`.
3. **Colorization** — mapping the 8-bit grayscale image through the official
   AIA color table for that wavelength.

Each stage points at the exact module and function that implements it.

---

## Stage 0 — Loading the FITS file

A raw AIA record is a single-HDU FITS file: a 2D image plus a rich header
(`WAVELNTH`, `EXPTIME`, `CROTA2`, `RSUN_OBS`, `CDELT1`, …). You can read it
either as a bare array via `astropy.io.fits` or, preferably, as a
`sunpy.map.Map` so the WCS metadata travels with the data.

```python
from astropy.io import fits

hdu = fits.open("aia_171.fits")[0]
image = hdu.data                     # 2D float array (raw counts)
exptime = hdu.header["EXPTIME"]      # seconds
wavelnth = int(hdu.header["WAVELNTH"])  # 171
```

Retrieval of AIA data from JSOC (export requests, EUV queries, and the cached
calibration/pointing tables) lives in **`egghouse/sdo/jsoc.py`**
(`jsoc_export`, `aia_euv_query`, `cached_correction_table`,
`cached_pointing_table`).

---

## Stage 1 — Preprocessing / calibration

Two independent layers of preprocessing are available; use either or both
depending on how rigorous you need to be.

### 1a. Geometric standardization (Level 1.0 → 1.5)

**Code:** `egghouse/sdo/level15.py` → `to_level15(fits_file, instrument=None, ...)`

Produces a standardized `sunpy.map.Map`:

1. **Rotation** — rotate by `-CROTA2` so solar north points up (`CROTA2 → 0`).
2. **Resampling** — resample to a fixed plate scale of **0.6 arcsec/px**
   (both AIA and HMI).
3. **Padding** — pad with zeros to a fixed **4096×4096** output.

```python
from egghouse.sdo import to_level15

m = to_level15("aia_171.fits")   # instrument auto-detected from the header
assert m.meta["CROTA2"] == 0.0
assert m.data.shape == (4096, 4096)
```

This handles the geometry that the raw Level 1.0 product does not
(north-up alignment, common plate scale, common frame size).

### 1b. Radiometric calibration (aiapy-backed)

**Code:** `egghouse/sdo/prep.py`

`to_level15` does *not* touch the photometry. The prep stages that do are thin,
lazy-importing wrappers around `aiapy`:

| Function | Purpose | aiapy call |
| --- | --- | --- |
| `aia_update_pointing(map, pointing_table=None)` | Refresh outdated WCS keywords from the JSOC master pointing table | `aiapy.calibrate.update_pointing` |
| `aia_respike(map, spikes=None)` | Re-inject spike pixels the L1 pipeline removed | `aiapy.calibrate.respike` |
| `aia_correct_degradation(map, correction_table=None)` | Time-dependent effective-area correction | `aiapy.calibrate.correct_degradation` |
| `aia_deconvolve(map, psfs=None)` | PSF deconvolution | `aiapy.psf.deconvolve` |

Notes:

- Radiometric correction is defined only for the seven calibrated EUV channels
  `(94, 131, 171, 193, 211, 304, 335)`
  (`_AIA_WAVELENGTHS_WITH_CALIBRATION`). For any other wavelength the map is
  returned unchanged.
- PSF computation (`aiapy.psf.psf`) is slow (~minutes per channel). Precompute
  and reuse them with `cached_aia_psfs(path, wavelengths=...)`, and pass the
  cached correction/pointing tables from `jsoc.py` to avoid a JSOC round-trip
  per record in a batch.
- `mask_out_of_disk(map, fill_value=-5000.0)` flags off-disk pixels with a
  sentinel — useful when downstream code (e.g. the DEM model) must ignore
  off-limb regions.

A typical full-calibration order for one EUV record:

```python
from egghouse.sdo import to_level15
from egghouse.sdo.prep import (
    aia_update_pointing, aia_respike, aia_correct_degradation, aia_deconvolve,
    cached_aia_psfs,
)
from egghouse.sdo.jsoc import cached_correction_table, cached_pointing_table

pointing = cached_pointing_table("cache/pointing.pkl")
corr = cached_correction_table("cache/correction.pkl")
psfs = cached_aia_psfs("cache/psfs.pkl")

m = to_level15("aia_171.fits")                       # geometry
m = aia_update_pointing(m, pointing_table=pointing)  # WCS
m = aia_respike(m, spikes=None)                      # de-spike undo (optional)
m = aia_deconvolve(m, psfs=psfs)                     # PSF
m = aia_correct_degradation(m, correction_table=corr)  # effective area
data = m.data                                        # calibrated float array
```

For visualization-only workflows you can skip Stage 1b entirely and feed the
raw counts straight into Stage 2 — the intensity scaling already normalizes by
exposure time.

---

## Stage 2 — 8-bit conversion

**Code:** `egghouse/sdo/aia.py` → `aia_intscale(image, exptime, wavelnth, to_bytescale=True)`
(built on `egghouse/image/core.py` → `bytescale_image`)

This is the visualization stretch. Per-wavelength parameters live in the
`AIA_CALIBRATION` table (`egghouse/sdo/aia.py`, from Boerner et al. 2012):
`norm_exptime`, `vmin`, `vmax`, and a `scale` of `linear` / `sqrt` / `log`.

The steps inside `aia_intscale`:

1. **NaN cleanup** — `np.nan_to_num(image, nan=0.0)`.
2. **Exposure normalization** — `image * (norm_exptime / exptime)`, putting
   every frame on a common exposure basis.
3. **Clip** to the channel's `[vmin, vmax]`.
4. **Stretch** with the channel's method:
   - `sqrt` (94, 171),
   - `log10` (131, 193, 211, 304, 335),
   - `linear` (1600, 1700, 4500).
5. **Byte scale** the stretched values from their transformed range
   `[t_vmin, t_vmax]` to `[0, 255]` via `bytescale_image`, returning `uint8`.

```python
from egghouse.sdo import aia_intscale

gray = aia_intscale(image, exptime, 171)   # (H, W) uint8, ready to display
```

Set `to_bytescale=False` to get the stretched float array instead (e.g. to
apply your own normalization).

`bytescale_image(data, imin, imax, omin=0, omax=255)` is the generic, reusable
primitive: a linear `[imin, imax] → [0, 255]` map with clipping and a `uint8`
cast. It works on any scientific array, not just AIA.

---

## Stage 3 — Colorization

**Code:**
- `egghouse/sdo/aia_color.py` — the AIA-specific color tables and the one-call
  `aia_colorize`.
- `egghouse/image/colorize.py` — the generic LUT machinery.

### The AIA color tables

`egghouse/sdo/aia_color.py` reproduces the official SolarSoft `aia_lct.pro`
(K. Schrijver, 2010) color tables — the same ones sunpy ships — for all ten
channels `(94, 131, 171, 193, 211, 304, 335, 1600, 1700, 4500)`.

Two independent sources produce a byte-identical `(256, 3)` uint8 LUT:

- `source="numpy"` — a **pure-numpy** reconstruction from the embedded IDL
  "Red Temperature" (color table 3) base and the analytic base curves. No
  sunpy/matplotlib import needed.
- `source="sunpy"` — samples sunpy's `LinearSegmentedColormap` (lazy import).

```python
from egghouse.sdo.aia_color import aia_color_lut, aia_colormap

lut = aia_color_lut(171)                 # (256, 3) uint8  (numpy path)
lut_sp = aia_color_lut(171, "sunpy")     # bit-identical to `lut`
cmap = aia_colormap(171)                 # matplotlib Colormap, for plt.imshow
```

### One-call colorization

`aia_colorize(image, wavelnth, exptime=None, source="numpy")` ties Stages 2 and
3 together and returns an `(H, W, 3)` uint8 RGB image:

- **Raw image + `exptime`** — runs `aia_intscale` (Stage 2) first, then applies
  the color table.
- **Already 8-bit (`exptime=None`)** — colorizes directly; non-uint8 input is
  min/max byte-scaled first.

```python
from egghouse.sdo.aia_color import aia_colorize

# From raw counts (does the 8-bit stretch internally):
rgb = aia_colorize(image, 171, exptime=exptime)   # (H, W, 3) uint8

# From an already-scaled 8-bit grayscale image:
rgb = aia_colorize(gray, 171)
```

### Generic LUT primitives

The wavelength-agnostic building blocks live in `egghouse/image/colorize.py`:

- `apply_colormap(gray, lut)` — pure index lookup, `(H, W) uint8 → lut[gray] →
  (H, W, 3) uint8`. Exact and fast, no interpolation.
- `lut_from_matplotlib(cmap, n=256)` — sample any matplotlib colormap into a
  `(256, 3)` uint8 LUT, so you can colorize with `inferno`, `viridis`, etc.

```python
from egghouse.image import apply_colormap, lut_from_matplotlib

rgb = apply_colormap(gray, lut_from_matplotlib("inferno"))
```

---

## End-to-end example

```python
from astropy.io import fits
from egghouse.sdo.aia_color import aia_colorize

hdu = fits.open("aia_171.fits")[0]

# Stage 2 (intensity stretch → 8-bit) + Stage 3 (official color table),
# both handled by aia_colorize when exptime is supplied:
rgb = aia_colorize(
    hdu.data,
    wavelnth=int(hdu.header["WAVELNTH"]),
    exptime=hdu.header["EXPTIME"],
)   # (H, W, 3) uint8

# For full radiometric calibration first, run Stage 1b (egghouse.sdo.prep)
# on a sunpy Map, then pass m.data + m.meta["EXPTIME"] into aia_colorize.
```

---

## Code map

| Stage | Module | Key functions |
| --- | --- | --- |
| Load / retrieve | `egghouse/sdo/jsoc.py` | `jsoc_export`, `aia_euv_query`, `cached_correction_table`, `cached_pointing_table` |
| 1a. Geometry (L1.0→1.5) | `egghouse/sdo/level15.py` | `to_level15` |
| 1b. Radiometric calibration | `egghouse/sdo/prep.py` | `aia_update_pointing`, `aia_respike`, `aia_correct_degradation`, `aia_deconvolve`, `cached_aia_psfs`, `mask_out_of_disk` |
| 2. 8-bit conversion | `egghouse/sdo/aia.py`, `egghouse/image/core.py` | `aia_intscale`, `get_aia_calibration`, `bytescale_image` |
| 3. Colorization | `egghouse/sdo/aia_color.py`, `egghouse/image/colorize.py` | `aia_colorize`, `aia_color_lut`, `aia_colormap`, `apply_colormap`, `lut_from_matplotlib` |

See also: [`docs/sdo_guide.md`](sdo_guide.md) and [`docs/image_guide.md`](image_guide.md).
